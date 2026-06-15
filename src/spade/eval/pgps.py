"""PGPS — Preference Geometry Preservation Score.

PGPS asks whether synthetic items occupy *geometrically faithful* positions in
the shared reference space: does a synthetic item sit where a real item with the
same local preference structure would? It is the primary development metric and a
novel contribution, so the computation is deliberately explicit.

For each synthetic item ``s`` (embedded in the reference space transductively):

1. Find its **anchor**, the nearest real item ``a = argmin dist(s, real)``.
2. Take the anchor's ``k`` nearest *real* items ``N(a)`` — the local
   preference neighborhood the synthetic item ought to reproduce.
3. Take the synthetic item's own ``k`` nearest *real* items ``N(s)``.
4. Score the overlap ``|N(s) ∩ N(a)| / k``.

PGPS is the mean overlap over all synthetic items, in ``[0, 1]``. A score near 1
means synthetic items preserve the real local geometry; a degenerate generator
that scatters items randomly scores around the **random baseline ``k / |I_real|``**
(the expected overlap of two independent size-``k`` subsets). The reported
``lift`` is ``pgps - random`` — meaningfully positive is the target, while a
score approaching 1.0 everywhere can also signal trivial copying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spade.eval.neighbors import top_k, top_k_against

__all__ = ["PGPSResult", "pgps"]


@dataclass(frozen=True)
class PGPSResult:
    """PGPS score with its random reference and the lift over it."""

    pgps: float
    random_baseline: float
    lift: float
    k: int

    def as_dict(self) -> dict[str, float]:
        return {
            "pgps": self.pgps,
            "pgps_random": self.random_baseline,
            "pgps_lift": self.lift,
        }


def pgps(
    real_item_emb: np.ndarray,
    synth_item_emb: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> PGPSResult:
    """Preference-geometry overlap of synthetic items against real items.

    ``real_item_emb`` / ``synth_item_emb`` are reference-space coordinates
    ``(n, d)`` (synthetic items already mapped in transductively). ``k`` is the
    neighborhood size; it is clamped to ``n_real - 1``.
    """
    n_real = real_item_emb.shape[0]
    if n_real < 2 or synth_item_emb.shape[0] == 0:
        return PGPSResult(pgps=0.0, random_baseline=0.0, lift=0.0, k=k)

    k_eff = max(1, min(k, n_real - 1))

    # Anchor = nearest real item to each synthetic item.
    anchors = top_k_against(synth_item_emb, real_item_emb, 1, metric)[:, 0]  # (n_syn,)
    # Each real item's k nearest real neighbors (self excluded).
    real_neighbors = top_k(real_item_emb, k_eff, metric)                     # (n_real, k)
    # Each synthetic item's k nearest real items.
    synth_neighbors = top_k_against(synth_item_emb, real_item_emb, k_eff, metric)

    overlaps = np.empty(synth_item_emb.shape[0], dtype=np.float64)
    for s in range(synth_item_emb.shape[0]):
        anchor_set = set(real_neighbors[anchors[s]].tolist())
        synth_set = set(synth_neighbors[s].tolist())
        overlaps[s] = len(anchor_set & synth_set) / k_eff

    score = float(overlaps.mean())
    random = k_eff / n_real
    return PGPSResult(pgps=score, random_baseline=random, lift=score - random, k=k_eff)
