"""Rating decoder: a categorical distribution over discrete rating levels.

Ratings are discrete by construction. The decoder consumes ``[z_u; z_i]`` and
emits one logit per rating level; a softmax gives a categorical distribution and
Stage III samples it (or takes the argmax) to assign a rating. There is no
post-hoc rounding or clipping — the support is exactly the observed rating
vocabulary.

:class:`RatingVocab` captures that vocabulary, inferred from the *train* split's
ratings, and maps between raw rating values (e.g. ``1.0..5.0``) and contiguous
class indices ``0..R-1``. It is persisted with the exported embeddings so Stage
III decodes class indices back to the correct rating values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax import nnx

from spade.models.mlp import MLP

__all__ = ["RatingVocab", "RatingDecoder", "ContinuousRatingDecoder"]


@dataclass(frozen=True)
class RatingVocab:
    """Bidirectional map between raw rating values and class indices.

    Built from the sorted unique ratings in the train split, so the decoder's
    softmax has exactly one class per observed rating level. Robust to any scale
    (integer 1-5, half-stars, etc.) rather than assuming a fixed range.
    """

    values: np.ndarray  # class index -> raw rating value, shape (n_levels,)

    @classmethod
    def from_ratings(cls, ratings: np.ndarray) -> RatingVocab:
        return cls(values=np.unique(np.asarray(ratings)))

    @property
    def n_levels(self) -> int:
        return len(self.values)

    def to_index(self, ratings: np.ndarray) -> np.ndarray:
        """Map raw rating values to class indices ``0..R-1`` (exact match)."""
        idx = np.searchsorted(self.values, ratings)
        idx = np.clip(idx, 0, self.n_levels - 1)
        if not np.allclose(self.values[idx], ratings):
            raise ValueError("rating value not present in the vocabulary")
        return idx.astype(np.int64)

    def to_value(self, indices: np.ndarray) -> np.ndarray:
        """Map class indices back to raw rating values."""
        return self.values[np.asarray(indices)]

    def snap(self, predictions: np.ndarray) -> np.ndarray:
        """Snap continuous predictions to the nearest valid rating value.

        Used only by the continuous-decoder ablation, where a regressor emits an
        off-grid real number that must be projected back onto the discrete rating
        support. This is the post-hoc rounding/clipping step the categorical
        decoder avoids by construction; isolating it here keeps that contrast
        explicit. Out-of-range predictions clamp to the nearest endpoint.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        nearest = np.abs(self.values[None, :] - preds[:, None]).argmin(axis=1)
        return self.values[nearest]


class RatingDecoder(nnx.Module):
    """MLP scoring ``[z_u; z_i] -> logits`` over ``n_levels`` rating classes."""

    def __init__(
        self,
        latent_dim: int,
        hidden: Sequence[int],
        n_levels: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_levels = n_levels
        self.mlp = MLP(2 * latent_dim, hidden, n_levels, rngs=rngs)

    def __call__(self, z_u: jnp.ndarray, z_i: jnp.ndarray) -> jnp.ndarray:
        """Return rating logits ``(batch, n_levels)`` for paired latents."""
        x = jnp.concatenate([z_u, z_i], axis=-1)
        return self.mlp(x)

    def distribution(self, z_u: jnp.ndarray, z_i: jnp.ndarray) -> jnp.ndarray:
        """Return categorical rating probabilities ``(batch, n_levels)``."""
        return nnx.softmax(self(z_u, z_i), axis=-1)


class ContinuousRatingDecoder(nnx.Module):
    """Regress a single continuous rating from ``[z_u; z_i]`` (ablation variant).

    The discrete-vs-continuous ablation replaces the categorical decoder with a
    scalar regressor trained under MSE; at synthesis the prediction is snapped to
    the nearest valid rating via :meth:`RatingVocab.snap`. This deliberately
    reintroduces the post-hoc rounding the categorical decoder removes, so the
    ablation measures the cost of leaving rating discreteness unconstrained.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.mlp = MLP(2 * latent_dim, hidden, 1, rngs=rngs)

    def __call__(self, z_u: jnp.ndarray, z_i: jnp.ndarray) -> jnp.ndarray:
        """Return predicted continuous ratings ``(batch,)`` for paired latents."""
        x = jnp.concatenate([z_u, z_i], axis=-1)
        return self.mlp(x).squeeze(-1)
