"""Sparse interaction store with contiguous index maps, degrees, and sparsity.

The store is the canonical in-memory representation consumed by every later
stage: a CSR matrix of explicit ratings plus the bookkeeping needed to map
between raw dataset ids and contiguous ``[0, n)`` indices. It also exposes the
user/item degree distributions and the observed sparsity ``rho`` (= density),
which Stage III uses to size its ANN candidate set ``C = ceil(I' * rho * gamma)``.

Building a store from :class:`~spade.data.datasets.RawInteractions` applies
iterative k-core filtering first, so the resulting index space contains only
users and items that survive the minimum-interaction thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

__all__ = ["IndexMap", "InteractionStore", "apply_kcore", "build_store"]


@dataclass(frozen=True)
class IndexMap:
    """Bidirectional map between raw ids and contiguous indices for one axis.

    ``raw_ids[i]`` is the original id of contiguous index ``i``; ``to_index`` is
    the inverse lookup. Kept as a small object so user and item maps travel with
    the store and can be persisted alongside splits.
    """

    raw_ids: np.ndarray              # contiguous index -> raw id, shape (n,)
    to_index: dict[int, int]         # raw id -> contiguous index

    @classmethod
    def from_raw(cls, raw_ids: np.ndarray) -> IndexMap:
        """Build a map from the *unique* raw ids (sorted for determinism)."""
        uniq = np.unique(raw_ids)
        return cls(raw_ids=uniq, to_index={int(r): i for i, r in enumerate(uniq)})

    @property
    def size(self) -> int:
        return len(self.raw_ids)

    def encode(self, raw: np.ndarray) -> np.ndarray:
        """Vectorized raw-id -> contiguous-index lookup."""
        lookup = np.vectorize(self.to_index.__getitem__, otypes=[np.int64])
        return lookup(raw)


def apply_kcore(
    users: np.ndarray,
    items: np.ndarray,
    min_user: int,
    min_item: int,
) -> np.ndarray:
    """Return a boolean mask of interactions surviving iterative k-core filtering.

    Repeatedly drops users with fewer than ``min_user`` interactions and items
    with fewer than ``min_item`` interactions until the set is stable, because
    removing one axis can push entities on the other below threshold. A mask is
    returned (rather than filtered arrays) so callers can apply it to ratings and
    timestamps in lockstep.
    """
    if min_user <= 1 and min_item <= 1:
        return np.ones(len(users), dtype=bool)

    keep = np.ones(len(users), dtype=bool)
    while True:
        u, i = users[keep], items[keep]
        u_ids, u_counts = np.unique(u, return_counts=True)
        i_ids, i_counts = np.unique(i, return_counts=True)
        bad_users = set(u_ids[u_counts < min_user].tolist())
        bad_items = set(i_ids[i_counts < min_item].tolist())
        if not bad_users and not bad_items:
            break
        idx = np.flatnonzero(keep)
        drop = np.array(
            [(uu in bad_users) or (ii in bad_items) for uu, ii in zip(u, i, strict=True)],
            dtype=bool,
        )
        keep[idx[drop]] = False
        if not keep.any():
            break
    return keep


@dataclass
class InteractionStore:
    """Indexed sparse interactions plus degree and sparsity bookkeeping.

    Holds the rating matrix as CSR (``n_users x n_items``) along with the parallel
    COO-style arrays (``user_idx``, ``item_idx``, ``ratings``) used for sampling
    and splitting. Index maps carry the raw-id correspondence. Construct via
    :func:`build_store`, or directly when sub-setting (e.g. a train split).
    """

    user_idx: np.ndarray            # contiguous user indices, shape (nnz,)
    item_idx: np.ndarray            # contiguous item indices, shape (nnz,)
    ratings: np.ndarray             # ratings aligned with the index arrays
    n_users: int
    n_items: int
    user_map: IndexMap
    item_map: IndexMap

    def __post_init__(self) -> None:
        self._csr: sparse.csr_matrix | None = None

    @property
    def nnz(self) -> int:
        """Number of observed interactions."""
        return len(self.ratings)

    @property
    def matrix(self) -> sparse.csr_matrix:
        """Lazily built CSR rating matrix (cached after first access)."""
        if self._csr is None:
            self._csr = sparse.csr_matrix(
                (self.ratings, (self.user_idx, self.item_idx)),
                shape=(self.n_users, self.n_items),
            )
        return self._csr

    @property
    def user_degree(self) -> np.ndarray:
        """Per-user interaction counts, shape ``(n_users,)``."""
        return np.bincount(self.user_idx, minlength=self.n_users)

    @property
    def item_degree(self) -> np.ndarray:
        """Per-item interaction counts, shape ``(n_items,)``."""
        return np.bincount(self.item_idx, minlength=self.n_items)

    @property
    def rho(self) -> float:
        """Observed sparsity / density = nnz / (n_users * n_items).

        This is the rho fed to Stage III's candidate count
        ``C = ceil(I' * rho * gamma)``; reported on whichever interaction set the
        store holds (use the *train* store for leakage-safe figures).
        """
        denom = self.n_users * self.n_items
        return float(self.nnz / denom) if denom else 0.0

    def degree_stats(self) -> dict[str, float]:
        """Summary of degree distributions and sparsity for logging."""
        ud, idg = self.user_degree, self.item_degree
        return {
            "n_users": float(self.n_users),
            "n_items": float(self.n_items),
            "nnz": float(self.nnz),
            "rho": self.rho,
            "user_degree_mean": float(ud.mean()) if self.n_users else 0.0,
            "user_degree_min": float(ud.min()) if self.n_users else 0.0,
            "user_degree_max": float(ud.max()) if self.n_users else 0.0,
            "item_degree_mean": float(idg.mean()) if self.n_items else 0.0,
            "item_degree_min": float(idg.min()) if self.n_items else 0.0,
            "item_degree_max": float(idg.max()) if self.n_items else 0.0,
        }


def build_store(
    raw,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> InteractionStore:
    """Filter, index, and pack :class:`RawInteractions` into an InteractionStore.

    Applies iterative k-core filtering, then builds contiguous user/item index
    maps from the *surviving* ids so the index space has no gaps. The full
    (post-filter) interaction set defines the user/item universe; leakage-safe
    train-only statistics come from splitting this store afterwards.
    """
    keep = apply_kcore(
        raw.users, raw.items, min_user_interactions, min_item_interactions
    )
    users, items = raw.users[keep], raw.items[keep]
    ratings = raw.ratings[keep]

    user_map = IndexMap.from_raw(users)
    item_map = IndexMap.from_raw(items)
    return InteractionStore(
        user_idx=user_map.encode(users),
        item_idx=item_map.encode(items),
        ratings=np.asarray(ratings, dtype=np.float32),
        n_users=user_map.size,
        n_items=item_map.size,
        user_map=user_map,
        item_map=item_map,
    )
