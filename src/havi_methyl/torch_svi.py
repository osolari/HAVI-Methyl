"""End-to-end torch SVI loop for the full HAVI-Methyl architecture (Phase 1).

Composes the Set Transformer fragment-bag encoder
(``encoders.SetTransformerEncoder``) with a per-(s, l) variational
posterior head and the population / sample-shift variational layers.

Two posterior heads are supported:

  - ``gaussian`` (default, stable) — encoder predicts ``(mu, log_sigma)``
    of a Gaussian on logit-beta. ``eta = mu + sigma * eps``,
    ``log q(eta|c) = -log sigma - 0.5 * eps^2 - 0.5 log(2 pi)``. Closed
    form everywhere.
  - ``flow`` (experimental) — uses ``flow.ConditionalNSFStack``. The
    rational-quadratic spline parameterisation is documented in App. C
    but its current torch implementation has indexing edge cases that
    produce NaN in early training; a clean rewrite with explicit
    ``num_bins + 1`` knots is tracked under IMPL-04 follow-up.

The loop:
  1. Sub-samples a (B_s, B_l) mini-batch.
  2. Encodes each fragment bag with the Set Transformer.
  3. Builds the encoder context per (s, l): ``[c_frag || mu_pop_l ||
     mu_delta_s || log(1+n_frag)]``.
  4. Samples ``eta`` from the chosen posterior head and ``beta = sigm(eta)``.
  5. Computes the Beta-Binomial reconstruction term (Sec. 5.2) and the
     KL terms (population, sample-shift, conditional flow KL via
     Monte-Carlo eq. \\ref{eq:kl-mc}).
  6. Updates the encoder + posterior head with AdamW; updates the
     population / sample-shift natural parameters with Robbins-Monro.
  7. Logs the surrogate ELBO and the global recentering residual.

This is the IMPL-02..05 integration target. Canonical Sec. 11 numbers
stay with ``fit_svi_simplified``; this routine produces real numbers
that can be reported as the "full torch" measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


if HAS_TORCH:
    from havi_methyl.encoders import SetTransformerEncoder

    @dataclass
    class TorchSVIConfig:
        in_dim: int = 5  # fragment feature dim
        hidden: int = 32
        num_inducing: int = 16
        num_layers: int = 2
        kappa: float = 20.0  # Beta-Binomial concentration
        sigma_eta: float = 0.6
        sigma_pop: float = 2.0
        sigma_delta: float = 0.5
        lr: float = 5e-3
        rho_exponent: float = 0.6
        batch_samples: int = 4
        batch_loci: int = 32
        posterior: str = "gaussian"  # "gaussian" or "flow"
        # Phase 2 ablation toggles (Sec. 12.3 / IMPL-06).
        # All default to 0 so the existing Phase 1 numbers are unchanged.
        vib_weight: float = 0.0
        counterfactual_weight: float = 0.0
        adversarial_weight: float = 0.0
        # mQTL anchor data: tuple(genotype, intercept, effect, anchor_idx) or None.
        mqtl_anchors: tuple | None = None
        mqtl_weight: float = 0.0
        # IWAE-tightened bound (Sec. 5.3 / Phase 1 IMPL-04 finetune).
        # k_iwae=1 reproduces the standard reparam ELBO (Phase 1 default).
        # k_iwae>1 with iwae_dreg=False uses the standard K-sample IWAE
        # objective, which gives a tighter bound (Jensen gap shrinks).
        # iwae_dreg=True applies a simplified Tucker-2019 doubly-reparam
        # variance reduction; the full DReG estimator requires detaching
        # the encoder parameters in log q_phi (PyTorch functional_call),
        # which is not implemented here. The simplified surrogate is
        # available but is a research-grade flag, not benefit-positive at
        # the small synthetic scales used in this repo.
        k_iwae: int = 1
        iwae_dreg: bool = False
        # Compute device. "auto" picks cuda > mps > cpu. Any explicit string
        # accepted by torch.device is also valid (e.g. "cuda:0", "cpu", "mps").
        device: str = "auto"

    class _GaussianPosteriorHead(nn.Module):
        """Encoder context -> (mu, log_sigma) of the q(eta|c) Gaussian head."""

        def __init__(self, ctx_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(ctx_dim, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )

        def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.net(ctx)
            mu = out[..., 0]
            log_sigma = torch.clamp(out[..., 1], -3.0, 3.0)
            return mu, log_sigma

    @dataclass
    class TorchSVIState:
        config: TorchSVIConfig
        encoder: nn.Module
        head: nn.Module
        pop_mean: torch.Tensor
        pop_var: torch.Tensor
        delta_mean: torch.Tensor
        delta_var: torch.Tensor
        elbo_history: list[float] = field(default_factory=list)
        recentering_history: list[float] = field(default_factory=list)
        # Optional snapshots of (iter, pred_mean) saved by fit_svi_torch
        # when snapshot_every > 0. Lets the training-curve figure plot
        # actual per-iter Pearson r without re-running the model.
        snapshots: list[tuple] = field(default_factory=list)

    def _encode_bag(
        encoder: SetTransformerEncoder, bag: np.ndarray, device: torch.device, hidden: int
    ) -> torch.Tensor:
        """Wrap a single (n, in_dim) numpy bag in a (1, n, in_dim) torch batch.

        Returns a zero vector of size ``hidden`` for empty bags (matches the
        behaviour of the numpy reference).
        """
        if bag.shape[0] == 0:
            return torch.zeros(hidden, device=device)
        x = torch.tensor(bag, dtype=torch.float32, device=device).unsqueeze(0)
        return encoder(x).squeeze(0)

    def fit_svi_torch(
        bags: list[list[np.ndarray]],
        n_frag: np.ndarray,
        n_meth: np.ndarray,
        n_iter: int = 30,
        config: TorchSVIConfig | None = None,
        seed: int = 0,
        n_obs: np.ndarray | None = None,
        snapshot_every: int = 0,
    ) -> TorchSVIState:
        """End-to-end torch SVI training loop.

        ``bags[s][l]`` is a numpy array ``(n_{s,l}, in_dim)``; ``n_frag`` is the
        WGS fragment-count matrix used by the encoder's
        ``log(1+n_frag)`` context feature; ``n_meth`` is the Beta-Binomial
        success count (methylated reads). ``n_obs`` is the Beta-Binomial
        trial count -- for the synthetic simulator this equals ``n_frag``
        (each fragment is one trial), but for real cfDNA paired data
        ``n_obs`` is the WGBS read coverage at the CpG, which is a
        DIFFERENT observation stream from the WGS fragments. Defaults to
        ``n_frag`` so existing callers keep their behaviour; real-data
        callers should pass ``ds.n_total`` from ``load_finaleme_dataset``.

        Returns the final state with ``elbo_history``,
        ``recentering_history``, and the trained encoder + head.
        """
        cfg = config or TorchSVIConfig()
        if n_obs is None:
            n_obs_mat = n_frag
        else:
            n_obs_mat = n_obs
        torch.manual_seed(seed)
        if cfg.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(cfg.device)
        S = len(bags)
        L = n_frag.shape[1]
        encoder = SetTransformerEncoder(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            num_inducing=cfg.num_inducing,
        ).to(device)
        ctx_dim = cfg.hidden + 3
        if cfg.posterior == "gaussian":
            head: nn.Module = _GaussianPosteriorHead(ctx_dim).to(device)
        elif cfg.posterior == "flow":  # pragma: no cover
            from havi_methyl.flow import ConditionalNSFStack

            head = ConditionalNSFStack(context_dim=ctx_dim).to(device)
        else:
            raise ValueError(f"Unknown posterior {cfg.posterior!r}")

        # Warm-start population logit from the empirical beta per locus.
        # Without this, training on shallow real-data WGBS can drift to a
        # degenerate near-zero solution because the Beta-Binomial
        # likelihood is too weak to pull pop_mean back to the right
        # global location. The init uses logit(beta_hat) where
        # beta_hat = (sum_s n_meth) / (sum_s max(n_obs, 1)), clamped
        # away from {0, 1} so the logit stays finite. Uses the BB
        # trials matrix (n_obs_mat), which is the WGBS coverage for
        # real data and equals n_frag for synthetic.
        n_meth_sum_loc = n_meth.sum(axis=0).astype(np.float64)
        n_obs_sum_loc = n_obs_mat.sum(axis=0).astype(np.float64)
        beta_hat = np.where(n_obs_sum_loc > 0, n_meth_sum_loc / np.maximum(n_obs_sum_loc, 1.0), 0.5)
        beta_hat = np.clip(beta_hat, 0.05, 0.95)
        logit_init = np.log(beta_hat / (1.0 - beta_hat)).astype(np.float32)
        pop_mean = torch.tensor(logit_init, device=device)
        pop_var = torch.full((L,), cfg.sigma_pop**2, device=device)
        delta_mean = torch.zeros(S, device=device)
        delta_var = torch.full((S,), cfg.sigma_delta**2, device=device)

        opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr)
        rng = np.random.default_rng(seed)
        state = TorchSVIState(
            config=cfg,
            encoder=encoder,
            head=head,
            pop_mean=pop_mean,
            pop_var=pop_var,
            delta_mean=delta_mean,
            delta_var=delta_var,
        )

        for t in range(n_iter):
            sample_batch = rng.choice(S, size=min(cfg.batch_samples, S), replace=False)
            loci_batch = rng.choice(L, size=min(cfg.batch_loci, L), replace=False)

            ctx_list = []
            n_obs_list = []
            n_meth_list = []
            mu_prior_list = []
            for s in sample_batch:
                for ell in loci_batch:
                    bag = bags[s][ell]
                    c_frag = _encode_bag(encoder, bag, device, cfg.hidden)
                    n_l = float(n_frag[s, ell])
                    ctx = torch.cat(
                        [
                            c_frag,
                            torch.tensor(
                                [
                                    pop_mean[ell].item(),
                                    delta_mean[s].item(),
                                    np.log1p(n_l),
                                ],
                                dtype=torch.float32,
                                device=device,
                            ),
                        ]
                    )
                    ctx_list.append(ctx)
                    n_obs_list.append(float(n_obs_mat[s, ell]))
                    n_meth_list.append(float(n_meth[s, ell]))
                    mu_prior_list.append(pop_mean[ell].item() + delta_mean[s].item())
            if not ctx_list:
                continue
            ctx_batch = torch.stack(ctx_list, dim=0)
            mu_prior = torch.tensor(mu_prior_list, dtype=torch.float32, device=device)

            n_obs = torch.tensor(n_obs_list, dtype=torch.float32, device=device)
            n_m = torch.tensor(n_meth_list, dtype=torch.float32, device=device)
            kappa = cfg.kappa
            K = max(1, int(cfg.k_iwae))

            # Sample K independent eta per (s, l). For K>1 we evaluate the
            # IWAE bound; the K=1 path is identical to the previous Phase 1
            # ELBO (so existing numbers are unchanged when k_iwae=1).
            log_ws: list[torch.Tensor] = []  # (K, batch_size)
            etas: list[torch.Tensor] = []
            for _k in range(K):
                if cfg.posterior == "gaussian":
                    mu_q, log_sigma = head(ctx_batch)
                    epsilon = torch.randn_like(mu_q)
                    eta = mu_q + torch.exp(log_sigma) * epsilon
                    log_q_k = -log_sigma - 0.5 * epsilon**2 - 0.5 * np.log(2 * np.pi)
                else:  # pragma: no cover
                    epsilon = torch.randn(ctx_batch.shape[0], device=device)
                    eta, log_jac = head(epsilon, ctx_batch)
                    log_q_k = -0.5 * epsilon**2 - 0.5 * np.log(2 * np.pi) - log_jac
                beta_k = torch.sigmoid(eta)
                alpha = kappa * beta_k + 1e-3
                gamma = kappa * (1.0 - beta_k) + 1e-3
                log_recon_k = (
                    torch.lgamma(n_m + alpha)
                    + torch.lgamma(n_obs - n_m + gamma)
                    - torch.lgamma(n_obs + alpha + gamma)
                    - torch.lgamma(alpha)
                    - torch.lgamma(gamma)
                    + torch.lgamma(alpha + gamma)
                )
                log_p_k = -0.5 * ((eta - mu_prior) / cfg.sigma_eta) ** 2 - 0.5 * np.log(
                    2 * np.pi * cfg.sigma_eta**2
                )
                log_ws.append(log_recon_k + log_p_k - log_q_k)
                etas.append(eta)
            log_w = torch.stack(log_ws, dim=0)  # (K, B)
            eta_stack = torch.stack(etas, dim=0)  # (K, B)
            # For the population update use the IWAE-weighted posterior mean
            # so K>1 does not increase the Robbins-Monro variance.
            with torch.no_grad():
                norm_w_for_update = torch.softmax(log_w, dim=0)
            eta = (norm_w_for_update * eta_stack).sum(dim=0)  # (B,)

            if K == 1:
                elbo_surrogate = log_w[0].mean()
                loss = -elbo_surrogate
            elif cfg.iwae_dreg:
                # DReG: encoder gradient uses squared importance weights as
                # mixing coefficients (Tucker et al. 2019). The detached
                # ``norm_w`` acts as a stop-grad multiplier so the score-
                # function term contribution cancels in expectation. The
                # K factor rescales the surrogate so it sits at the same
                # magnitude as the standard IWAE bound; without it the
                # optimisation signal is K-times smaller and training
                # under-progresses.
                with torch.no_grad():
                    norm_w = torch.softmax(log_w, dim=0)
                dreg = K * (norm_w**2 * log_w).sum(dim=0).mean()
                elbo_surrogate = (torch.logsumexp(log_w, dim=0) - np.log(K)).mean()
                loss = -dreg
            else:
                # Standard K-sample IWAE bound.
                iwae = (torch.logsumexp(log_w, dim=0) - np.log(K)).mean()
                elbo_surrogate = iwae
                loss = -iwae

            # Keep mu_q, log_sigma, eta references valid for the Phase 2
            # add-ons below (they expect a single sample's tensors).
            if cfg.posterior == "gaussian":
                mu_q, log_sigma = head(ctx_batch)

            # Phase 2 IMPL-06 add-ons. All zero-weight by default so the
            # Phase 1 ELBO and recovery numbers are unchanged.
            if cfg.vib_weight > 0 and cfg.posterior == "gaussian":
                # KL(N(mu_q, sigma_q^2) || N(0, 1)) closed form.
                sigma_q2 = torch.exp(2 * log_sigma)
                kl_vib = 0.5 * (sigma_q2 + mu_q**2 - 1.0 - 2 * log_sigma)
                loss = loss + cfg.vib_weight * kl_vib.mean()
            if cfg.counterfactual_weight > 0:
                # Swap the (pop, delta) inputs in the context vector and ask
                # the encoder to give the same posterior mean. Penalises
                # invariance breaks of the prior-input swap.
                ctx_swap = ctx_batch.clone()
                # Zero out the (mu_pop_l, mu_delta_s) channels (positions
                # cfg.hidden, cfg.hidden+1).
                ctx_swap[:, cfg.hidden] = 0.0
                ctx_swap[:, cfg.hidden + 1] = 0.0
                if cfg.posterior == "gaussian":
                    mu_q_swap, _ = head(ctx_swap)
                    cf_loss = ((mu_q - mu_q_swap) ** 2).mean()
                else:  # pragma: no cover
                    eta_swap, _ = head(epsilon, ctx_swap)
                    cf_loss = ((eta - eta_swap) ** 2).mean()
                loss = loss + cfg.counterfactual_weight * cf_loss
            if cfg.adversarial_weight > 0:
                # Gradient-reversal placeholder: penalise variance of the
                # encoder context across samples (simple invariance proxy).
                # A true gradient-reversal head with a per-sample classifier
                # is tracked as an IMPL-06 follow-up.
                ctx_means = ctx_batch[:, : cfg.hidden].mean(dim=0, keepdim=True)
                adv_loss = ((ctx_batch[:, : cfg.hidden] - ctx_means) ** 2).mean()
                loss = loss + cfg.adversarial_weight * adv_loss
            if cfg.mqtl_anchors is not None and cfg.mqtl_weight > 0:
                geno, intercept, effect, anchor_idx = cfg.mqtl_anchors
                anchor_mask = np.isin(
                    np.array([ell for s in sample_batch for ell in loci_batch]), anchor_idx
                )
                if anchor_mask.any():
                    a_mu = mu_q[torch.tensor(anchor_mask, device=device)]
                    a_pred = torch.tensor(
                        [
                            float(intercept[ell] + effect[ell] * geno[s, ell])
                            for s in sample_batch
                            for ell in loci_batch
                            if ell in anchor_idx
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    if a_pred.numel() > 0 and a_mu.numel() == a_pred.numel():
                        mqtl_resid = ((a_mu - a_pred) ** 2).mean()
                        loss = loss + cfg.mqtl_weight * mqtl_resid

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), max_norm=5.0
            )
            opt.step()

            with torch.no_grad():
                rho = (t + 1) ** (-cfg.rho_exponent)
                eta_np = eta.detach().cpu().numpy()
                idx = 0
                pop_update = np.zeros(L)
                pop_count = np.zeros(L)
                delta_update = np.zeros(S)
                delta_count = np.zeros(S)
                for s in sample_batch:
                    for ell in loci_batch:
                        e = eta_np[idx]
                        pop_update[ell] += e - delta_mean[s].item()
                        pop_count[ell] += 1
                        delta_update[s] += e - pop_mean[ell].item()
                        delta_count[s] += 1
                        idx += 1
                for ell in loci_batch:
                    if pop_count[ell] > 0:
                        new_pop = pop_update[ell] / pop_count[ell]
                        pop_mean[ell] = (1 - rho) * pop_mean[ell].item() + rho * new_pop
                for s in sample_batch:
                    if delta_count[s] > 0:
                        new_delta = delta_update[s] / delta_count[s]
                        delta_mean[s] = (1 - rho) * delta_mean[s].item() + rho * new_delta
                shift = float(delta_mean.mean().item())
                delta_mean -= shift
                pop_mean += shift
                state.recentering_history.append(float(delta_mean.sum().item()))

            state.elbo_history.append(float(elbo_surrogate.item()))

            if snapshot_every > 0 and ((t + 1) % snapshot_every == 0 or t == n_iter - 1):
                with torch.no_grad():
                    pred_mean_t, _ = predict_with_torch_state(state, bags, n_frag, n_samples=4)
                state.snapshots.append((t + 1, pred_mean_t.copy()))

        return state

    def predict_with_torch_state(
        state: TorchSVIState,
        bags: list[list[np.ndarray]],
        n_frag: np.ndarray,
        n_samples: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior beta mean / std under the trained encoder + head."""
        cfg = state.config
        device = next(state.encoder.parameters()).device
        S = len(bags)
        L = n_frag.shape[1]
        out_mean = np.zeros((S, L))
        out_std = np.zeros((S, L))
        with torch.no_grad():
            for s in range(S):
                for ell in range(L):
                    bag = bags[s][ell]
                    c_frag = _encode_bag(state.encoder, bag, device, cfg.hidden)
                    ctx = torch.cat(
                        [
                            c_frag,
                            torch.tensor(
                                [
                                    state.pop_mean[ell].item(),
                                    state.delta_mean[s].item(),
                                    np.log1p(float(n_frag[s, ell])),
                                ],
                                dtype=torch.float32,
                                device=device,
                            ),
                        ]
                    ).unsqueeze(0)
                    if cfg.posterior == "gaussian":
                        mu_q, log_sigma = state.head(ctx)
                        eps = torch.randn(n_samples, device=device)
                        eta = mu_q + torch.exp(log_sigma) * eps
                    else:  # pragma: no cover
                        eps = torch.randn(n_samples, device=device)
                        ctx_repeat = ctx.expand(n_samples, -1)
                        eta, _ = state.head(eps, ctx_repeat)
                    beta = torch.sigmoid(eta).cpu().numpy()
                    out_mean[s, ell] = float(beta.mean())
                    out_std[s, ell] = float(beta.std())
        return out_mean, out_std

else:  # pragma: no cover

    class TorchSVIConfig:  # type: ignore[no-redef]
        pass

    class TorchSVIState:  # type: ignore[no-redef]
        pass

    def fit_svi_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required for fit_svi_torch")

    def predict_with_torch_state(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required for predict_with_torch_state")
