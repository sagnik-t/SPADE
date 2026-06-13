"""Train Stage I on a prepared dataset split and export the real embeddings.

Loads (or builds) the leakage-safe splits for the configured dataset/seed, trains
the representation model jointly, and writes ``Z_u``/``Z_i`` plus the rating
vocabulary under ``output_dir``. Reuses the project's dataclass/CLI config::

    poetry run python scripts/train_stage1.py --data.dataset ml-100k --seed 42 \
        --representation.epochs 100
"""

from __future__ import annotations

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.data import build_store, load_dataset, load_splits, save_splits, split_store
from spade.models import export_stage1, train_stage1
from spade.utils import get_logger, init_wandb, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)
    d = cfg.data

    try:
        splits = load_splits(d.data_dir, d.dataset, cfg.seed)
        logger.info("loaded cached splits for %s seed=%d", d.dataset, cfg.seed)
    except FileNotFoundError:
        logger.info("no cached splits; building from raw %s", d.dataset)
        raw = load_dataset(d.dataset, d.data_dir)
        store = build_store(raw, d.min_user_interactions, d.min_item_interactions)
        splits = split_store(store, d.val_frac, d.test_frac, cfg.seed)
        save_splits(splits, d.data_dir, d.dataset)

    logger.info("train/val/test = %s", splits.summary())

    run = init_wandb(cfg, project=cfg.wandb_project, name=cfg.name, mode=cfg.wandb_mode)
    state = train_stage1(cfg, splits.train, splits.val)
    for row in state.history:
        run.log(row, step=int(row["epoch"]))

    path = export_stage1(
        state,
        splits.train.n_users,
        splits.train.n_items,
        cfg.output_dir,
        d.dataset,
        cfg.seed,
    )
    logger.info("Stage I complete (best epoch %d) -> %s", state.best_epoch, path)
    run.finish()


if __name__ == "__main__":
    main()
