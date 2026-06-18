"""Common generator interface for SPADE and its baselines.

The experiment harness must treat every dataset generator uniformly, so
each one — SPADE's synthesis pipeline and all five baselines — implements the
same :class:`Generator` contract: ``generate(key)`` returns a
:class:`GeneratorOutput` bundling a discrete :class:`SyntheticDataset` with an
*optional* :class:`LatentBundle`.

The latent bundle is what lets the geometry metrics (PGPS, NDI, latent W₂) treat
baselines uniformly without forcing structure onto generators that have none. A
generator that owns a latent space (SPADE, Noise-MF, VAE, GANRS) returns, for
both axes, its latent coordinates for the *real* entities (aligned to the real
index space) and for the *synthetic* entities; the evaluation fits a transductive
map from the real latents to the reference embeddings and carries the synthetic
ones across. Structure-free generators (Random, Marginal) return ``None`` and are
compared on the distribution/utility metrics only.

This module also holds the helpers every baseline shares: synthetic universe
sizing, target-sparsity bookkeeping, empirical rating sampling, and de-duplicated
assembly of a :class:`SyntheticDataset`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import numpy as np

from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.synthesis.dataset import SyntheticDataset

__all__ = [
    "LatentBundle",
    "GeneratorOutput",
    "Generator",
    "BaselineGenerator",
    "synthetic_sizes",
    "target_nnz",
    "RatingSampler",
    "assemble_dataset",
    "rng_from_key",
]


def rng_from_key(key: jax.Array) -> np.random.Generator:
    """Seed a NumPy generator from a JAX key (baselines sample in NumPy)."""
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class LatentBundle:
    """Real and synthetic entity latents in one generator's own space.

    ``real_users[k]`` is that generator's latent for real user ``k`` (aligned to
    the train index space), so a map to the reference space can be fit on the
    real pairs and applied to the synthetic latents. All four arrays are ``(n, d)``.
    """

    real_users: np.ndarray
    real_items: np.ndarray
    synth_users: np.ndarray
    synth_items: np.ndarray


@dataclass(frozen=True)
class GeneratorOutput:
    """A generated dataset plus optional latents for the geometry metrics."""

    name: str
    dataset: SyntheticDataset
    latents: LatentBundle | None = None


class Generator(ABC):
    """A dataset generator producing a synthetic dataset from a single PRNG key."""

    name: str = "generator"

    @abstractmethod
    def generate(self, key: jax.Array) -> GeneratorOutput:
        """Generate one synthetic dataset (and optional latents) under ``key``."""


class BaselineGenerator(Generator):
    """A baseline built from the real train store and the experiment config.

    Fixes the shared ``(train, cfg)`` construction signature so the registry and
    harness can instantiate any baseline uniformly; concrete baselines override
    this constructor (and implement :meth:`generate`). SPADE is *not* a baseline
    — it is assembled from already-trained stage models — so it subclasses
    :class:`Generator` directly rather than this.
    """

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None: ...


def synthetic_sizes(
    n_users: int, n_items: int, alpha: float, beta: float
) -> tuple[int, int]:
    """``(U', I') = (ceil(alpha * U), ceil(beta * I))`` — SPADE's expansion rule."""
    return math.ceil(alpha * n_users), math.ceil(beta * n_items)


def target_nnz(rho: float, n_users: int, n_items: int) -> int:
    """Interactions needed to hit density ``rho`` over a ``U' x I'`` universe."""
    return int(round(rho * n_users * n_items))


class RatingSampler:
    """Samples ratings from the empirical training marginal (no new values)."""

    def __init__(self, ratings: np.ndarray) -> None:
        values, counts = np.unique(np.asarray(ratings), return_counts=True)
        self.values = values.astype(np.float32)
        self.probs = counts / counts.sum()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` ratings i.i.d. from the empirical distribution."""
        if n == 0:
            return np.empty(0, dtype=np.float32)
        return rng.choice(self.values, size=n, p=self.probs).astype(np.float32)

    def nearest(self, values: np.ndarray) -> np.ndarray:
        """Snap continuous predictions to the nearest legal rating value."""
        idx = np.abs(values[:, None] - self.values[None, :]).argmin(axis=1)
        return self.values[idx].astype(np.float32)


def assemble_dataset(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
) -> SyntheticDataset:
    """De-duplicate ``(user, item)`` pairs (keeping first) and pack a dataset.

    Several baselines can emit the same pair twice (independent sampling, cluster
    collisions); the canonical synthetic dataset has at most one rating per pair,
    matching the SPADE invariant the evaluation asserts.
    """
    user_idx = np.asarray(user_idx, dtype=np.int64)
    item_idx = np.asarray(item_idx, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float32)
    if user_idx.size:
        flat = user_idx * n_items + item_idx
        _, first = np.unique(flat, return_index=True)
        first.sort()
        user_idx, item_idx, ratings = user_idx[first], item_idx[first], ratings[first]
    return SyntheticDataset(
        user_idx=user_idx,
        item_idx=item_idx,
        ratings=ratings,
        n_users=n_users,
        n_items=n_items,
    )
