"""Download a dataset, build the interaction store, split it, and persist splits.

Run for each seed to materialize leakage-safe partitions before training::

    poetry run python scripts/prepare_data.py --data.dataset ml-100k --seed 42

Reuses the project's dataclass/CLI config so flags match the rest of the pipeline
(``--data.dataset``, ``--data.val-frac``, ``--data.min-user-interactions``, ...).
Prints train/val/test sizes and the train-only degree/sparsity summary.
"""

from __future__ import annotations

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.data import build_store, load_dataset, save_splits, split_store
from spade.utils import get_logger, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()  # pick up .env (W&B / CUDA / JAX vars) before anything reads them
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)
    d = cfg.data

    logger.info("loading dataset %s from %s", d.dataset, d.data_dir)
    raw = load_dataset(d.dataset, d.data_dir)
    store = build_store(raw, d.min_user_interactions, d.min_item_interactions)
    logger.info("post-filter store: %s", store.degree_stats())

    splits = split_store(store, d.val_frac, d.test_frac, cfg.seed)
    logger.info("split sizes (seed=%d): %s", cfg.seed, splits.summary())
    logger.info("train-only degree/sparsity: %s", splits.train.degree_stats())

    path = save_splits(splits, d.data_dir, d.dataset)
    logger.info("wrote splits to %s", path)


if __name__ == "__main__":
    main()
