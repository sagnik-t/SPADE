"""Uniform negative sampling over unobserved user-item pairs.

The interaction gate in Stage I is trained with negative sampling (BPR-style)
because treating every unobserved pair as a true negative bakes in MNAR bias.
This module draws, for each positive ``(u, i)``, ``n_neg`` items the user has not
interacted with, sampling uniformly from the unobserved set by rejection. The
same utility backs any BPR-style baseline, so it lives in the data layer with an
explicit RNG (no global state) for reproducibility.
"""

from __future__ import annotations

import numpy as np

from spade.data.interactions import InteractionStore

__all__ = ["sample_negatives", "sample_negatives_for_users"]


def _observed_sets(store: InteractionStore) -> list[set[int]]:
    """Per-user sets of observed item indices, for O(1) membership tests."""
    seen: list[set[int]] = [set() for _ in range(store.n_users)]
    for u, i in zip(store.user_idx.tolist(), store.item_idx.tolist(), strict=True):
        seen[u].add(i)
    return seen


def sample_negatives_for_users(
    users: np.ndarray,
    store: InteractionStore,
    n_neg: int,
    rng: np.random.Generator,
    observed: list[set[int]] | None = None,
) -> np.ndarray:
    """Sample ``n_neg`` unobserved items for each user id in ``users``.

    Returns an array of shape ``(len(users), n_neg)`` of item indices, each not
    present in that user's observed set in ``store``. Sampling is uniform over
    items via rejection; ``observed`` may be precomputed (see
    :func:`_observed_sets`) and reused across calls to avoid rebuilding it.

    A user who has interacted with every item has no valid negative; this raises
    rather than silently returning a positive, since that signals a degenerate
    (post-filter) dataset the caller should handle explicitly.
    """
    if n_neg < 1:
        raise ValueError("n_neg must be >= 1")
    if observed is None:
        observed = _observed_sets(store)

    n_items = store.n_items
    out = np.empty((len(users), n_neg), dtype=np.int64)
    for row, u in enumerate(users.tolist()):
        seen = observed[u]
        if len(seen) >= n_items:
            raise ValueError(
                f"user {u} has interacted with all {n_items} items; no negatives"
            )
        drawn: set[int] = set()
        # Rejection sampling: cheap while the observed+drawn set stays small
        # relative to the catalogue, which holds under realistic CF sparsity.
        while len(drawn) < n_neg:
            cand = int(rng.integers(0, n_items))
            if cand not in seen and cand not in drawn:
                drawn.add(cand)
        out[row] = list(drawn)
    return out


def sample_negatives(
    store: InteractionStore,
    n_neg: int,
    rng: np.random.Generator,
    observed: list[set[int]] | None = None,
) -> np.ndarray:
    """Sample negatives aligned with every observed interaction in ``store``.

    Returns shape ``(store.nnz, n_neg)``: row ``k`` holds ``n_neg`` items not
    observed for ``store.user_idx[k]``, pairing each positive with its negatives
    for a BPR-style objective. The positive item itself is excluded by virtue of
    being in the user's observed set.
    """
    return sample_negatives_for_users(
        store.user_idx, store, n_neg, rng, observed=observed
    )
