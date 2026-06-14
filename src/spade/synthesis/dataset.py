"""Container for a synthesized interaction dataset (Stage III output).

:class:`SyntheticDataset` holds the discrete result of synthesis — the sparse
``(user, item, rating)`` triples plus the synthetic universe sizes ``U'``/``I'``.
Ratings are exact values drawn from the training rating vocabulary (no post-hoc
rounding). The dataset can be materialized into an :class:`InteractionStore` for
the evaluation stage; unlike :func:`spade.data.build_store`, this conversion
applies **no** k-core filtering, so the synthetic universe is preserved exactly
(including any entity that drew zero interactions).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spade.data.interactions import IndexMap, InteractionStore

__all__ = ["SyntheticDataset"]


@dataclass
class SyntheticDataset:
    """Sparse synthetic interactions over a synthetic user/item universe."""

    user_idx: np.ndarray            # (nnz,) synthetic user indices [0, n_users)
    item_idx: np.ndarray            # (nnz,) synthetic item indices [0, n_items)
    ratings: np.ndarray             # (nnz,) rating values from the vocabulary
    n_users: int
    n_items: int

    @property
    def nnz(self) -> int:
        return len(self.ratings)

    @property
    def density(self) -> float:
        denom = self.n_users * self.n_items
        return float(self.nnz / denom) if denom else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "n_users": float(self.n_users),
            "n_items": float(self.n_items),
            "nnz": float(self.nnz),
            "density": self.density,
        }

    def as_store(self) -> InteractionStore:
        """Materialize an :class:`InteractionStore` (identity maps, no filtering)."""
        user_map = IndexMap.from_raw(np.arange(self.n_users))
        item_map = IndexMap.from_raw(np.arange(self.n_items))
        return InteractionStore(
            user_idx=np.asarray(self.user_idx, dtype=np.int64),
            item_idx=np.asarray(self.item_idx, dtype=np.int64),
            ratings=np.asarray(self.ratings, dtype=np.float32),
            n_users=self.n_users,
            n_items=self.n_items,
            user_map=user_map,
            item_map=item_map,
        )

    def save(self, path: str | Path) -> Path:
        """Persist the triples and universe sizes to a ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            user_idx=np.asarray(self.user_idx, dtype=np.int64),
            item_idx=np.asarray(self.item_idx, dtype=np.int64),
            ratings=np.asarray(self.ratings, dtype=np.float32),
            n_users=self.n_users,
            n_items=self.n_items,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SyntheticDataset:
        loaded = np.load(path)
        return cls(
            user_idx=loaded["user_idx"],
            item_idx=loaded["item_idx"],
            ratings=loaded["ratings"],
            n_users=int(loaded["n_users"]),
            n_items=int(loaded["n_items"]),
        )
