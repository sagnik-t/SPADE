"""NDI — Neighborhood Distortion Index.

Where PGPS audits synthetic *items*, NDI audits synthetic *users*: do they crowd
into and disrupt the neighborhoods of real users in the shared reference space?
A good synthetic population augments the space without collapsing onto real users
or shredding their local structure, so **lower is better** and NDI is independent
of PGPS.

For each real user ``u``:

1. ``N_real(u)`` — its ``k`` nearest *real* users (the clean neighborhood).
2. ``N_mixed(u)`` — its ``k`` nearest users once synthetic users are pooled in.
3. distortion ``= 1 - |N_mixed(u) ∩ N_real(u)| / k`` — the fraction of the clean
   neighborhood displaced by intruding synthetic users.

NDI is the mean distortion over real users, in ``[0, 1]``. The companion
``intrusion`` rate (mean fraction of ``N_mixed(u)`` that are synthetic) is
reported alongside for interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spade.eval.neighbors import pairwise_scores, top_k

__all__ = ["NDIResult", "ndi"]


@dataclass(frozen=True)
class NDIResult:
    """NDI distortion plus the synthetic-intrusion rate."""

    ndi: float
    intrusion: float
    k: int

    def as_dict(self) -> dict[str, float]:
        return {"ndi": self.ndi, "ndi_intrusion": self.intrusion}


def _argtopk(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1)
    return part[row, order]


def ndi(
    real_user_emb: np.ndarray,
    synth_user_emb: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> NDIResult:
    """Distortion of real-user neighborhoods caused by synthetic users.

    Both arrays are reference-space coordinates ``(n, d)`` (synthetic users
    already mapped in transductively). ``k`` is clamped to ``n_real - 1``.
    """
    n_real = real_user_emb.shape[0]
    if n_real < 2:
        return NDIResult(ndi=0.0, intrusion=0.0, k=k)

    k_eff = max(1, min(k, n_real - 1))

    # Clean real-only neighborhoods.
    clean = top_k(real_user_emb, k_eff, metric)                # (n_real, k)

    if synth_user_emb.shape[0] == 0:
        return NDIResult(ndi=0.0, intrusion=0.0, k=k_eff)

    # Mixed neighborhoods: real users queried against real ∪ synthetic.
    combined = np.concatenate([real_user_emb, synth_user_emb], axis=0)
    scores = pairwise_scores(real_user_emb, combined, metric)  # (n_real, n_total)
    diag = np.arange(n_real)
    scores[diag, diag] = -np.inf                              # exclude self
    mixed = _argtopk(scores, k_eff)                            # (n_real, k)

    distortions = np.empty(n_real, dtype=np.float64)
    intrusions = np.empty(n_real, dtype=np.float64)
    for u in range(n_real):
        mixed_set = set(mixed[u].tolist())
        clean_set = set(clean[u].tolist())                    # real ids only
        retained = len(mixed_set & clean_set)
        distortions[u] = 1.0 - retained / k_eff
        intrusions[u] = sum(1 for m in mixed[u] if m >= n_real) / k_eff

    return NDIResult(
        ndi=float(distortions.mean()),
        intrusion=float(intrusions.mean()),
        k=k_eff,
    )
