"""Data layer: dataset loaders, sparse interaction store, splitting, sampling.

Typical flow::

    raw = load_dataset("ml-100k", data_dir)
    store = build_store(raw, min_user_interactions=5, min_item_interactions=5)
    splits = split_store(store, val_frac=0.1, test_frac=0.1, seed=42)
    negatives = sample_negatives(splits.train, n_neg=5, rng=np.random.default_rng(42))

All representation learning uses ``splits.train`` only; ``rho`` and degrees are
read from the train store for leakage-safe statistics.
"""

from spade.data.datasets import (
    DATASET_REGISTRY,
    RawInteractions,
    available_datasets,
    load_dataset,
)
from spade.data.interactions import (
    IndexMap,
    InteractionStore,
    apply_kcore,
    build_store,
)
from spade.data.sampling import sample_negatives, sample_negatives_for_users
from spade.data.split import Splits, load_splits, save_splits, split_store

__all__ = [
    "RawInteractions",
    "load_dataset",
    "available_datasets",
    "DATASET_REGISTRY",
    "IndexMap",
    "InteractionStore",
    "apply_kcore",
    "build_store",
    "Splits",
    "split_store",
    "save_splits",
    "load_splits",
    "sample_negatives",
    "sample_negatives_for_users",
]
