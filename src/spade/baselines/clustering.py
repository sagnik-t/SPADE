"""Minimal Lloyd K-Means for identifier recovery (GANRS, VAE baselines).

The tuple-based generators emit continuous user/item embedding fragments; to turn
a cloud of generated fragments into a *discrete* synthetic universe we cluster
them and treat each cluster as one synthetic entity. A small dependency-free
K-Means suffices and keeps the baselines deterministic given a seed (no sklearn).
Empty clusters are re-seeded to a random point so exactly ``k`` centers survive.

The assignment step is computed in row chunks rather than as a single
``(n, k)`` distance matrix: with the density-matched tuple budget ``n`` can reach
~10^6 (ml-100k) to ~10^7 (ml-1m), so a full ``n*k`` matrix would allocate tens of
GB and OOM. Chunking keeps the working set at ``ASSIGN_CHUNK * k`` while producing
labels bit-identical to the full-matrix computation; the center update is
vectorized with ``bincount`` instead of a per-cluster Python loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kmeans"]

# Rows processed per assignment chunk. Caps the transient distance matrix at
# roughly ``ASSIGN_CHUNK * k`` floats (~0.2 GB at k=3364), independent of ``n``.
ASSIGN_CHUNK = 8192


def kmeans(
    x: np.ndarray, k: int, iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster ``x`` ``(n, d)`` into ``k`` groups; return ``(labels, centers)``.

    ``labels`` is ``(n,)`` in ``[0, k)`` and ``centers`` is ``(k, d)``. ``k`` is
    clamped to ``n``. Initialization picks ``k`` distinct points at random.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # Assign: nearest center by squared Euclidean distance, in row chunks so
        # the transient distance matrix never exceeds ``ASSIGN_CHUNK * k``.
        centers_sq = (centers**2).sum(1)
        new_labels = np.empty(n, dtype=np.int64)
        for start in range(0, n, ASSIGN_CHUNK):
            xb = x[start : start + ASSIGN_CHUNK]
            d2 = (
                (xb**2).sum(1)[:, None]
                - 2.0 * (xb @ centers.T)
                + centers_sq[None, :]
            )
            new_labels[start : start + ASSIGN_CHUNK] = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update: mean of assigned points (vectorized); re-seed empty clusters.
        counts = np.bincount(labels, minlength=k)
        sums = np.empty((k, d), dtype=np.float64)
        for j in range(d):
            sums[:, j] = np.bincount(labels, weights=x[:, j], minlength=k)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = np.where(~nonempty)[0]
        if empty.size:
            centers[empty] = x[rng.integers(0, n, size=empty.size)]
    return labels, centers
