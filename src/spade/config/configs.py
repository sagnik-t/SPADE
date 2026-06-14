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


@dataclass
class EvalConfig:
    """Evaluation protocol."""

    topk: list[int] = field(default_factory=lambda: [10, 20])
    reference_models: list[str] = field(default_factory=lambda: ["mf", "ncf"])
    n_seeds: int = 5


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
