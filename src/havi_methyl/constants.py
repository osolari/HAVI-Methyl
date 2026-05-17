"""Default hyperparameters and constants from Appendix F (App.~\\ref{app:hparams}).

All values match Table in Appendix F and the operating regime described in
Section 6.4-6.5 (\\ref{sec:algo}).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------- Model priors (Appendix F) ----------
MU_0: float = 0.0  # Population logit prior mean
TAU_0: float = 2.0  # Population logit prior std (broad)
SIGMA_DELTA: float = 0.5  # Per-sample shift std
SIGMA_ETA: float = 0.8  # Per-(s,l) latent std
KAPPA: float = 20.0  # Beta-Binomial concentration

# ---------- Regularization weights ----------
BETA_VIB: float = 1.0  # VIB weight
LAMBDA_CF: float = 0.5  # Counterfactual invariance weight
LAMBDA_MQTL: float = 0.2  # mQTL anchor weight
LAMBDA_TOO: float = 1.0  # Tissue-of-origin loss weight

# ---------- Architecture ----------
HIDDEN_DIM: int = 128  # d_c
ISAB_LAYERS: int = 2  # L_e
INDUCING_POINTS: int = 64  # m
FLOW_BLOCKS: int = 6  # K
NSF_BINS: int = 8  # spline knots per block
T_MAX_HDP: int = 64  # HDP truncation level

# ---------- Optimization ----------
LR_ENCODER: float = 1e-3
WEIGHT_DECAY: float = 1e-4
RHO_EXPONENT: float = 0.6  # Robbins-Monro: rho_t = (t+1)^{-RHO_EXPONENT}
BATCH_SAMPLES: int = 8  # |B_s|
BATCH_LOCI: int = 4096  # |B_l|
GRAD_CLIP: float = 5.0
T_ANNEAL_FRAC: float = 1.0  # one epoch
LR_WARMUP_FRAC: float = 0.05  # 5% of total steps

# ---------- IWAE / fine-tuning ----------
K_IWAE_TRAIN: int = 1
K_IWAE_FINETUNE: int = 8

# ---------- Data ----------
TRAIN_VAL_TEST_SPLIT: tuple[float, float, float] = (0.70, 0.15, 0.15)
HELDOUT_CHROMOSOMES: tuple[str, ...] = ("chr1", "chr22")

# ---------- Simulator (Appendix E) ----------
NUC_SPACING_MEAN: float = 187.0
NUC_SPACING_STD: float = 25.0
STRAUSS_GAMMA: float = 0.1
SIGMA_NUC_BP: float = 20.0
PERIODICITY_AMP: float = 0.3
PERIODICITY_PERIOD: float = 10.4
# Fragment-length mixture re-fit on real Liu 2024 cfDNA fragments
# (5M fragments from 8 patients on chr1+chr19-22; 3-mode Gaussian EM).
# The published 0.005 target for the 320-350 bp peak is wrong: real
# cfDNA shows 0.001 per bp at that interval.
LENGTH_MIXTURE_WEIGHTS: tuple[float, float, float] = (0.874, 0.117, 0.009)
LENGTH_MIXTURE_MEANS: tuple[float, float, float] = (161.0, 313.0, 455.0)
LENGTH_MIXTURE_STDS: tuple[float, float, float] = (21.0, 38.0, 27.0)
SEQ_ERROR_RATE: float = 0.01
READ_LENGTH: int = 150
NB_DISPERSION_R: float = 2.0

# ---------- Numerical guards ----------
EPS: float = 1e-9
LOGIT_CLIP: float = 1 - 1e-6  # avoid logit(0)/logit(1) overflow

# ---------- Public dataset reference (Sec. 12) ----------
N_FINALEME_SAMPLES: int = 80
N_LOYFER_TYPES: int = 39
N_LOYFER_SAMPLES: int = 205


@dataclass(frozen=True)
class Hyperparams:
    """Bundle of all hyperparameters; defaults match Appendix F."""

    mu_0: float = MU_0
    tau_0: float = TAU_0
    sigma_delta: float = SIGMA_DELTA
    sigma_eta: float = SIGMA_ETA
    kappa: float = KAPPA
    beta_vib: float = BETA_VIB
    lambda_cf: float = LAMBDA_CF
    lambda_mqtl: float = LAMBDA_MQTL
    lambda_too: float = LAMBDA_TOO
    hidden_dim: int = HIDDEN_DIM
    isab_layers: int = ISAB_LAYERS
    inducing_points: int = INDUCING_POINTS
    flow_blocks: int = FLOW_BLOCKS
    nsf_bins: int = NSF_BINS
    t_max_hdp: int = T_MAX_HDP
    lr_encoder: float = LR_ENCODER
    weight_decay: float = WEIGHT_DECAY
    rho_exponent: float = RHO_EXPONENT
    batch_samples: int = BATCH_SAMPLES
    batch_loci: int = BATCH_LOCI
    grad_clip: float = GRAD_CLIP
    k_iwae_train: int = K_IWAE_TRAIN
    k_iwae_finetune: int = K_IWAE_FINETUNE


DEFAULT_HPARAMS = Hyperparams()
