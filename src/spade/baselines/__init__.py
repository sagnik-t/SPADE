"""Baseline dataset generators and the common generator interface.

Every generator — SPADE (via :class:`SpadeGenerator`) and the five baselines —
implements :class:`Generator.generate`, returning a :class:`GeneratorOutput` that
bundles a discrete :class:`~spade.synthesis.SyntheticDataset` with an optional
:class:`LatentBundle` for the geometry metrics. The baselines span the spectrum
from structure-free lower bounds (Random, Marginal) through embedding perturbation
(Noise-Perturbed MF) to learned tuple generators (GANRS, VAE).

``BASELINE_REGISTRY`` maps each baseline's name to its class; all baseline
constructors share the ``(train_store, cfg)`` signature so the harness can build
any of them uniformly. SPADE is excluded from the registry because it is
constructed from already-trained stage models.
"""

from spade.baselines.base import (
    BaselineGenerator,
    Generator,
    GeneratorOutput,
    LatentBundle,
    RatingSampler,
    assemble_dataset,
    rng_from_key,
    synthetic_sizes,
    target_nnz,
)
from spade.baselines.ganrs import GANRSGenerator
from spade.baselines.marginal import MarginalMatchingGenerator
from spade.baselines.noise_mf import NoisePerturbedMFGenerator
from spade.baselines.random_gen import RandomGenerator
from spade.baselines.spade_adapter import SpadeGenerator
from spade.baselines.vae import VAEGenerator

BASELINE_REGISTRY: dict[str, type[BaselineGenerator]] = {
    RandomGenerator.name: RandomGenerator,
    MarginalMatchingGenerator.name: MarginalMatchingGenerator,
    NoisePerturbedMFGenerator.name: NoisePerturbedMFGenerator,
    GANRSGenerator.name: GANRSGenerator,
    VAEGenerator.name: VAEGenerator,
}

__all__ = [
    "Generator",
    "BaselineGenerator",
    "GeneratorOutput",
    "LatentBundle",
    "RatingSampler",
    "assemble_dataset",
    "rng_from_key",
    "synthetic_sizes",
    "target_nnz",
    "RandomGenerator",
    "MarginalMatchingGenerator",
    "NoisePerturbedMFGenerator",
    "GANRSGenerator",
    "VAEGenerator",
    "SpadeGenerator",
    "BASELINE_REGISTRY",
]
