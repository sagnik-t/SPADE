"""Shared tuple encoding and identifier recovery for GANRS and the VAE baseline.

Both generators model the real data as a cloud of interaction *tuples* — each a
concatenation ``[user_embedding, item_embedding, rating_one_hot]`` — and generate
new tuples (by a GAN or a VAE). Turning those continuous tuples back into a
discrete dataset over a fixed synthetic universe is identical for both: K-Means
the user fragments into ``U'`` clusters and the item fragments into ``I'``
clusters, read each generated tuple's rating off its one-hot block, and assemble
the de-duplicated triples. The cluster centers double as the synthetic entity
latents for the geometry metrics.
"""

from __future__ import annotations

import numpy as np

from spade.baselines.base import assemble_dataset
from spade.baselines.clustering import kmeans
from spade.synthesis.dataset import SyntheticDataset

__all__ = ["TupleCodec", "recover_universe"]


class TupleCodec:
    """Encodes interactions as ``[emb_u, emb_i, rating_one_hot]`` and back.

    Holds the rating vocabulary so a one-hot block decodes to a legal value.
    ``dim`` is the per-entity embedding width; ``tuple_dim = 2*dim + n_levels``.
    """

    def __init__(self, dim: int, rating_values: np.ndarray) -> None:
        self.dim = dim
        self.rating_values = np.asarray(rating_values, dtype=np.float32)
        self.n_levels = len(self.rating_values)
        self.tuple_dim = 2 * dim + self.n_levels

    def rating_index(self, ratings: np.ndarray) -> np.ndarray:
        """Map rating values to class indices via exact nearest-value match."""
        idx = np.abs(ratings[:, None] - self.rating_values[None, :]).argmin(axis=1)
        return idx.astype(np.int64)

    def one_hot(self, idx: np.ndarray) -> np.ndarray:
        oh = np.zeros((idx.shape[0], self.n_levels), dtype=np.float32)
        oh[np.arange(idx.shape[0]), idx] = 1.0
        return oh

    def encode(self, emb_u: np.ndarray, emb_i: np.ndarray, rating_idx: np.ndarray) -> np.ndarray:
        """Build the real tuple matrix ``(n, tuple_dim)``."""
        return np.concatenate([emb_u, emb_i, self.one_hot(rating_idx)], axis=1)

    def split(self, tuples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split generated tuples into user / item fragments and rating values."""
        p = tuples[:, : self.dim]
        q = tuples[:, self.dim : 2 * self.dim]
        rating_logits = tuples[:, 2 * self.dim :]
        ratings = self.rating_values[rating_logits.argmax(axis=1)]
        return p, q, ratings


def recover_universe(
    p_fake: np.ndarray,
    q_fake: np.ndarray,
    ratings: np.ndarray,
    n_synth_users: int,
    n_synth_items: int,
    kmeans_iters: int,
    rng: np.random.Generator,
) -> tuple[SyntheticDataset, np.ndarray, np.ndarray]:
    """Cluster generated fragments into a discrete universe + synthetic latents.

    Returns the de-duplicated :class:`SyntheticDataset` plus the user and item
    cluster centers (the synthetic entity coordinates for the geometry metrics).
    """
    user_labels, user_centers = kmeans(p_fake, n_synth_users, kmeans_iters, rng)
    item_labels, item_centers = kmeans(q_fake, n_synth_items, kmeans_iters, rng)
    dataset = assemble_dataset(
        user_labels, item_labels, ratings, n_synth_users, n_synth_items
    )
    return dataset, user_centers, item_centers
