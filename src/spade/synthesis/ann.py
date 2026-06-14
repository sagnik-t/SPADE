"""Approximate-nearest-neighbour candidate retrieval for Stage III synthesis.

Synthesis must decide, for each synthetic user, which synthetic items it could
plausibly interact with. Scoring every user-item pair through the gate would be
``O(U' x I')`` and dominate runtime, so we instead retrieve a sparse candidate
set: the top-``C`` items per user by similarity in the shared latent space, and
only those pairs are passed to the frozen gate and decoder.

``faiss`` does the retrieval on NumPy arrays (framework-agnostic). Cosine
similarity is implemented as inner product on L2-normalized vectors
(``IndexFlatIP``); ``l2`` uses ``IndexFlatL2``. The flat indexes are exact, which
keeps synthesis deterministic — "ANN" here denotes the sparse-candidate strategy,
not an approximate index; swapping in an IVF/HNSW index later is a drop-in change.
"""

from __future__ import annotations

import faiss
import numpy as np

__all__ = ["top_c_candidates"]


def top_c_candidates(
    user_latents: np.ndarray,
    item_latents: np.ndarray,
    c: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Return the top-``c`` item indices per user, shape ``(n_users, c)``.

    ``user_latents``/``item_latents`` are ``(n, d)`` float arrays in the same
    space. ``c`` is clamped to the number of items. For ``cosine`` the inputs are
    L2-normalized (copies, inputs untouched) and compared by inner product.
    """
    users = np.ascontiguousarray(user_latents, dtype=np.float32)
    items = np.ascontiguousarray(item_latents, dtype=np.float32)
    n_items, d = items.shape
    c = max(1, min(c, n_items))

    if metric == "cosine":
        users, items = users.copy(), items.copy()
        faiss.normalize_L2(users)
        faiss.normalize_L2(items)
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"unknown ann metric {metric!r}; expected 'cosine' or 'l2'")

    index.add(items)
    _, candidates = index.search(users, c)  # (n_users, c), int64
    return candidates
