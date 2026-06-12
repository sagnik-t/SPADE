"""Leakage-safe train/val/test splitting and on-disk persistence.

Splitting uses a *per-user random holdout*: for each user, a fraction of their
interactions is assigned to validation and test, the rest to train. This is the
standard collaborative-filtering protocol and keeps every user represented in
train, so encoders never have to embed a user seen only at evaluation time.

Leakage safety is structural: all representation learning and preprocessing
statistics (degrees, rho, normalization) must be derived from the *train* store
only. Validation and test stores exist solely to score held-out interactions.
Splits are deterministic in the seed and persisted per seed so every stage and
baseline reads identical partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spade.data.interactions import IndexMap, InteractionStore

__all__ = ["Splits", "split_store", "save_splits", "load_splits"]


@dataclass
class Splits:
    """A train/val/test partition over a shared user/item index space.

    All three stores reuse the parent's index maps, so a user or item index means
    the same entity across splits. ``train`` is the only partition any model is
    permitted to fit on.
    """

    train: InteractionStore
    val: InteractionStore
    test: InteractionStore
    seed: int

    def summary(self) -> dict[str, int]:
        return {
            "train": self.train.nnz,
            "val": self.val.nnz,
            "test": self.test.nnz,
            "n_users": self.train.n_users,
            "n_items": self.train.n_items,
        }


def _substore(parent: InteractionStore, mask: np.ndarray) -> InteractionStore:
    """Build a child store over the same index space from a row mask."""
    return InteractionStore(
        user_idx=parent.user_idx[mask],
        item_idx=parent.item_idx[mask],
        ratings=parent.ratings[mask],
        n_users=parent.n_users,
        n_items=parent.n_items,
        user_map=parent.user_map,
        item_map=parent.item_map,
    )


def split_store(
    store: InteractionStore,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Per-user random holdout split of ``store`` into train/val/test.

    For each user the interactions are shuffled (deterministically in ``seed``)
    and the last ``round(d * test_frac)`` go to test, the next
    ``round(d * val_frac)`` to val, and the remainder to train. Users with too
    few interactions to spare a holdout keep all of theirs in train, which
    guarantees no user is absent from training. Returns three stores sharing the
    parent's index maps.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError("require val_frac, test_frac >= 0 and their sum < 1")

    rng = np.random.default_rng(seed)
    n = store.nnz
    assign = np.zeros(n, dtype=np.int8)  # 0=train, 1=val, 2=test

    order = np.argsort(store.user_idx, kind="stable")
    sorted_users = store.user_idx[order]
    # Boundaries between per-user contiguous runs in the sorted order.
    boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
    groups = np.split(order, boundaries)

    for positions in groups:
        d = len(positions)
        n_test = int(round(d * test_frac))
        n_val = int(round(d * val_frac))
        # Never starve train: leave at least one interaction in train.
        while n_test + n_val >= d and (n_test + n_val) > 0:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
        shuffled = rng.permutation(positions)
        assign[shuffled[:n_test]] = 2
        assign[shuffled[n_test : n_test + n_val]] = 1

    return Splits(
        train=_substore(store, assign == 0),
        val=_substore(store, assign == 1),
        test=_substore(store, assign == 2),
        seed=seed,
    )


def _split_path(data_dir: str | Path, dataset: str, seed: int) -> Path:
    return Path(data_dir) / dataset / "splits" / f"seed_{seed}.npz"


def save_splits(
    splits: Splits,
    data_dir: str | Path,
    dataset: str,
) -> Path:
    """Persist a partition to ``<data_dir>/<dataset>/splits/seed_<seed>.npz``.

    Stores the index arrays and ratings for each split plus the shared raw-id
    maps, so :func:`load_splits` reconstructs identical stores without re-reading
    the source dataset.
    """
    path = _split_path(data_dir, dataset, splits.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seed=splits.seed,
        n_users=splits.train.n_users,
        n_items=splits.train.n_items,
        user_raw_ids=splits.train.user_map.raw_ids,
        item_raw_ids=splits.train.item_map.raw_ids,
        **{
            f"{name}_{field}": getattr(getattr(splits, name), field)
            for name in ("train", "val", "test")
            for field in ("user_idx", "item_idx", "ratings")
        },
    )
    return path


def load_splits(
    data_dir: str | Path,
    dataset: str,
    seed: int,
) -> Splits:
    """Load a partition previously written by :func:`save_splits`."""
    path = _split_path(data_dir, dataset, seed)
    if not path.exists():
        raise FileNotFoundError(f"no saved splits at {path}")
    z = np.load(path, allow_pickle=False)
    user_map = IndexMap.from_raw(z["user_raw_ids"])
    item_map = IndexMap.from_raw(z["item_raw_ids"])
    n_users, n_items = int(z["n_users"]), int(z["n_items"])

    def store(name: str) -> InteractionStore:
        return InteractionStore(
            user_idx=z[f"{name}_user_idx"],
            item_idx=z[f"{name}_item_idx"],
            ratings=z[f"{name}_ratings"],
            n_users=n_users,
            n_items=n_items,
            user_map=user_map,
            item_map=item_map,
        )

    return Splits(
        train=store("train"),
        val=store("val"),
        test=store("test"),
        seed=int(z["seed"]),
    )
