"""Concrete SPADE configuration dataclasses.

Names follow the codebase's functional module layout (representation /
generative / synthesis), not the paper's stage numbers. Defaults follow the
paper's experimental setup (d=64, alpha=beta=2, gamma=5, WGAN-GP n_critic=5,
gradient-penalty weight 10, five random seeds). All values are overridable from
the CLI via :mod:`spade.config.base`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DataConfig",
    "RepresentationConfig",
    "GenerativeConfig",
    "SynthesisConfig",
    "EvalConfig",
    "BaselineConfig",
    "ExperimentConfig",
]


@dataclass
class DataConfig:
    """Dataset selection, splitting, and sampling."""

    dataset: str = "ml-100k"            # ml-100k | ml-1m | amazon
    data_dir: str = "data"
    val_frac: float = 0.1
    test_frac: float = 0.1
    min_user_interactions: int = 5      # k-core style filtering
    min_item_interactions: int = 5
    n_neg: int = 5                      # negatives per positive for gate training


@dataclass
class RepresentationConfig:
    """Representation learning: encoders, interaction gate, rating decoder."""

    latent_dim: int = 64
    encoder_hidden: list[int] = field(default_factory=lambda: [128])
    gate_hidden: list[int] = field(default_factory=lambda: [128, 64])
    decoder_hidden: list[int] = field(default_factory=lambda: [128, 64])
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 100
    l2_lambda: float = 1e-5             # embedding-norm regularization
    early_stop_patience: int = 10


@dataclass
class GenerativeConfig:
    """WGAN-GP latent generators (shared config for the user and item nets)."""

    noise_dim: int = 64
    generator_hidden: list[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [128, 128])
    n_critic: int = 5
    gp_lambda: float = 10.0             # gradient-penalty weight
    moment_lambda: float = 1.0         # mean/covariance moment-matching weight
    lr: float = 1e-4
    adam_b1: float = 0.0                # WGAN-GP Adam betas (Gulrajani et al.)
    adam_b2: float = 0.9
    batch_size: int = 256
    epochs: int = 500


@dataclass
class SynthesisConfig:
    """Inference-only synthesis."""

    alpha: float = 2.0                  # user expansion ratio  U' = alpha * U
    beta: float = 2.0                   # item expansion ratio  I' = beta * I
    gamma: float = 5.0                  # ANN oversampling buffer
    ann_metric: str = "cosine"
    score_batch_size: int = 100_000     # pairs per gate/decoder scoring chunk


@dataclass
class EvalConfig:
    """Evaluation protocol.

    Covers the shared reference space (a fixed MF/NCF trained on real data), the
    transductive map that places synthetic entities into it, and the metric
    knobs. ``ref_*`` size and train the reference models; ``map_ridge`` is the
    L2 strength of the least-squares map from Stage I latents to the reference
    space; ``neighbor_k`` is the neighborhood size for PGPS/NDI; ``tstr_model``
    picks the downstream recommender for the TS-TR utility check.
    """

    topk: list[int] = field(default_factory=lambda: [10, 20])
    reference_models: list[str] = field(default_factory=lambda: ["mf", "ncf"])
    n_seeds: int = 5

    # Reference models (MF + NCF), trained on the real train split.
    ref_dim: int = 32                   # reference embedding dimension
    ref_hidden: list[int] = field(default_factory=lambda: [64, 32])  # NCF tower
    ref_epochs: int = 50
    ref_lr: float = 1e-3
    ref_l2: float = 1e-5                # embedding-norm regularization
    ref_batch_size: int = 1024

    # Transductive linear map: Z_real -> reference space (ridge least squares).
    map_ridge: float = 1e-2

    # PGPS / NDI neighborhood size (the k in top-k preference geometry).
    neighbor_k: int = 10

    # TS-TR downstream recommender + its (synthetic) holdout fraction.
    tstr_model: str = "mf"              # mf | ncf
    tstr_test_frac: float = 0.1


@dataclass
class BaselineConfig:
    """Hyperparameters for the comparison generators (Phase 6).

    Synthetic universe sizes and target sparsity are taken from
    :class:`SynthesisConfig`/the data so every baseline expands to the same
    ``U'``/``I'`` and density as SPADE — only the *generation mechanism* differs.
    """

    # Noise-Perturbed MF: MF source + isotropic embedding noise.
    noise_mf_dim: int = 32
    noise_mf_epochs: int = 50
    noise_mf_lr: float = 1e-3
    noise_std: float = 0.1

    # GANRS: DeepMF source -> vanilla GAN over interaction tuples -> K-Means.
    deepmf_dim: int = 32
    deepmf_epochs: int = 50
    gan_noise_dim: int = 32
    gan_hidden: list[int] = field(default_factory=lambda: [128, 128])
    gan_epochs: int = 100
    gan_lr: float = 1e-4
    gan_batch_size: int = 256
    gan_n_generate: int = 50_000

    # VAE over interaction tuples.
    vae_dim: int = 32
    vae_latent: int = 16
    vae_hidden: list[int] = field(default_factory=lambda: [128])
    vae_epochs: int = 50
    vae_lr: float = 1e-3
    vae_beta: float = 1.0
    vae_n_generate: int = 50_000

    # K-Means identifier recovery (GANRS, VAE).
    kmeans_iters: int = 25


@dataclass
class ExperimentConfig:
    """Top-level config composing every stage plus run/logging settings."""

    name: str = "spade-default"
    seed: int = 42
    output_dir: str = "results"
    wandb_project: str = "spade"
    wandb_mode: str = "online"          # online | offline | disabled
    data: DataConfig = field(default_factory=DataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    generative: GenerativeConfig = field(default_factory=GenerativeConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
