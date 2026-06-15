"""Shared nearest-neighbor helpers for the geometry metrics (PGPS, NDI).

Both metrics reduce to top-k neighborhoods in the reference space, so the
neighbor search lives here once. Cosine is the default because the reference
embeddings encode *preference direction*, where magnitude is a less meaningful
axis of comparison; Euclidean is available for ablations. The searches are exact
(brute-force NumPy), which keeps the metrics deterministic at the entity counts
involved (thousands), mirroring the exact-flat choice made for synthesis ANN.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pairwise_scores", "top_k", "top_k_against"]


def _prepare(x: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if metric == "cosine":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-12, None)
    if metric == "euclidean":
        return x
    raise ValueError(f"unknown metric {metric!r}; expected 'cosine' or 'euclidean'")


def pairwise_scores(query: np.ndarray, base: np.ndarray, metric: str) -> np.ndarray:
    """Similarity ``(n_query, n_base)`` — higher means nearer, for either metric."""
    q = _prepare(query, metric)
    b = _prepare(base, metric)
    if metric == "cosine":
        return q @ b.T
    # Negated squared Euclidean so that "higher = nearer" holds uniformly.
    sq = (q**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * (q @ b.T)
    return -sq


def top_k(latents: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Top-``k`` neighbors of each row among the *other* rows, shape ``(n, k)``.

    Self is excluded. ``k`` is clamped to ``n-1``.
    """
    n = latents.shape[0]
    k = max(1, min(k, n - 1))
    scores = pairwise_scores(latents, latents, metric)
    np.fill_diagonal(scores, -np.inf)
    return _argtopk(scores, k)


def top_k_against(
    query: np.ndarray, base: np.ndarray, k: int, metric: str = "cosine"
) -> np.ndarray:
    """Top-``k`` ``base`` indices for each ``query`` row, shape ``(n_query, k)``."""
    k = max(1, min(k, base.shape[0]))
    scores = pairwise_scores(query, base, metric)
    return _argtopk(scores, k)


def _argtopk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest entries per row, sorted descending."""
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1)
    return part[row, order]
