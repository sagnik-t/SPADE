"""Top-k ranking metrics: Recall, NDCG, and MAP.

These are the standard implicit-feedback ranking measures used by the TS-TR
downstream check. Each operates on per-user *ranked item lists* (already masked
of train-seen items) and per-user *relevant sets* (the held-out positives), with
binary relevance. Metrics are averaged over the evaluated users.

Kept dependency-free (NumPy only) and separate from the protocol in
:mod:`spade.eval.downstream` so they can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["recall_at_k", "ndcg_at_k", "average_precision_at_k", "ranking_metrics"]


def recall_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Fraction of a user's relevant items recovered in the top ``k``."""
    if not relevant:
        return 0.0
    hits = sum(1 for it in ranked[:k] if it in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Normalized DCG at ``k`` with binary relevance (ideal = front-loaded hits)."""
    if not relevant:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.array([1.0 if it in relevant else 0.0 for it in ranked[:k]])
    dcg = float((gains * discounts).sum())
    ideal_hits = min(len(relevant), k)
    idcg = float(discounts[:ideal_hits].sum())
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Average precision at ``k`` (the per-user term of MAP)."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, it in enumerate(ranked[:k], start=1):
        if it in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ranking_metrics(
    ranked_by_user: dict[int, np.ndarray],
    relevant_by_user: dict[int, set[int]],
    ks: list[int],
) -> dict[str, float]:
    """Average Recall@k, NDCG@k for each ``k`` plus MAP over evaluated users.

    Only users present in ``relevant_by_user`` with a non-empty relevant set are
    scored. MAP uses the largest ``k`` as its cutoff.
    """
    users = [u for u, rel in relevant_by_user.items() if rel and u in ranked_by_user]
    out: dict[str, float] = {}
    if not users:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["map"] = 0.0
        return out

    for k in ks:
        out[f"recall@{k}"] = float(
            np.mean([recall_at_k(ranked_by_user[u], relevant_by_user[u], k) for u in users])
        )
        out[f"ndcg@{k}"] = float(
            np.mean([ndcg_at_k(ranked_by_user[u], relevant_by_user[u], k) for u in users])
        )
    kmax = max(ks)
    out["map"] = float(
        np.mean(
            [average_precision_at_k(ranked_by_user[u], relevant_by_user[u], kmax) for u in users]
        )
    )
    return out
