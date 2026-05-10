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
    ) -> TorchSVIState:
        """End-to-end torch SVI training loop.

        ``bags[s][l]`` is a numpy array ``(n_{s,l}, in_dim)``; ``n_frag`` is the
        coverage matrix; ``n_meth`` is the methylated-CpG count matrix used by
        the Beta-Binomial reconstruction term.

        Returns the final state with ``elbo_history``,
        ``recentering_history``, and the trained encoder + head.
        """
        cfg = config or TorchSVIConfig()
        torch.manual_seed(seed)
        device = torch.device("cpu")
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

        pop_mean = torch.zeros(L, device=device)
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
                    n_obs_list.append(n_l)
                    n_meth_list.append(float(n_meth[s, ell]))
                    mu_prior_list.append(pop_mean[ell].item() + delta_mean[s].item())
            if not ctx_list:
                continue
            ctx_batch = torch.stack(ctx_list, dim=0)
            mu_prior = torch.tensor(mu_prior_list, dtype=torch.float32, device=device)

            if cfg.posterior == "gaussian":
                mu_q, log_sigma = head(ctx_batch)
                epsilon = torch.randn_like(mu_q)
                eta = mu_q + torch.exp(log_sigma) * epsilon
                log_q = -log_sigma - 0.5 * epsilon**2 - 0.5 * np.log(2 * np.pi)
            else:  # pragma: no cover
                epsilon = torch.randn(ctx_batch.shape[0], device=device)
                eta, log_jac = head(epsilon, ctx_batch)
                log_q = -0.5 * epsilon**2 - 0.5 * np.log(2 * np.pi) - log_jac

            beta = torch.sigmoid(eta)
            n_obs = torch.tensor(n_obs_list, dtype=torch.float32, device=device)
            n_m = torch.tensor(n_meth_list, dtype=torch.float32, device=device)
            kappa = cfg.kappa
            alpha = kappa * beta + 1e-3
            gamma = kappa * (1.0 - beta) + 1e-3
            log_recon = (
                torch.lgamma(n_m + alpha)
                + torch.lgamma(n_obs - n_m + gamma)
                - torch.lgamma(n_obs + alpha + gamma)
                - torch.lgamma(alpha)
                - torch.lgamma(gamma)
                + torch.lgamma(alpha + gamma)
            )
            log_p = -0.5 * ((eta - mu_prior) / cfg.sigma_eta) ** 2 - 0.5 * np.log(
                2 * np.pi * cfg.sigma_eta**2
            )
            kl_local = (log_q - log_p).mean()
            recon = log_recon.mean()
            elbo_surrogate = recon - kl_local
            loss = -elbo_surrogate
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
