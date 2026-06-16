"""Minimal Lloyd K-Means for identifier recovery (GANRS, VAE baselines).

The tuple-based generators emit continuous user/item embedding fragments; to turn
a cloud of generated fragments into a *discrete* synthetic universe we cluster
them and treat each cluster as one synthetic entity. A small dependency-free
K-Means suffices and keeps the baselines deterministic given a seed (no sklearn).
Empty clusters are re-seeded to a random point so exactly ``k`` centers survive.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kmeans"]


def kmeans(
    x: np.ndarray, k: int, iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster ``x`` ``(n, d)`` into ``k`` groups; return ``(labels, centers)``.

    ``labels`` is ``(n,)`` in ``[0, k)`` and ``centers`` is ``(k, d)``. ``k`` is
    clamped to ``n``. Initialization picks ``k`` distinct points at random.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # Assign: nearest center by squared Euclidean distance.
        d2 = (
            (x**2).sum(1)[:, None]
            - 2.0 * (x @ centers.T)
            + (centers**2).sum(1)[None, :]
        )
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update: mean of assigned points; re-seed empty clusters.
        for c in range(k):
            members = x[labels == c]
            if members.shape[0] > 0:
                centers[c] = members.mean(axis=0)
            else:
                centers[c] = x[rng.integers(0, n)]
    return labels, centers
