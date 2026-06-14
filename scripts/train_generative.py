"""Train the generative stage from a representation-stage export.

Loads the frozen ``Z_u``/``Z_i`` written by ``scripts/train_representation.py``
for the configured dataset/seed, fits two independent WGAN-GP pairs via
:class:`GenerativeTrainer`, logs the Wasserstein / gradient-penalty /
moment-matching curves to W&B, and exports the trained generative model for the
synthesis stage::

    poetry run python scripts/train_generative.py --data.dataset ml-100k \
        --seed 42 --generative.epochs 500
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.training import GenerativeTrainer
from spade.utils import get_logger, init_wandb, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)

    rep_path = (
        Path(cfg.output_dir)
        / cfg.data.dataset
        / f"representation_seed_{cfg.seed}.npz"
    )
    if not rep_path.exists():
        raise FileNotFoundError(
            f"Representation export not found: {rep_path}. "
            "Run scripts/train_representation.py first."
        )
    data = np.load(rep_path)
    logger.info(
        "generative stage on Z_u%s, Z_i%s",
        data["z_users"].shape,
        data["z_items"].shape,
    )

    run = init_wandb(
        cfg, project=cfg.wandb_project, name=f"{cfg.name}-generative", mode=cfg.wandb_mode
    )
    trainer = GenerativeTrainer(
        cfg, data["z_users"], data["z_items"], run=run
    ).fit()
    path = trainer.export(cfg.output_dir, cfg.data.dataset)
    logger.info("generative stage complete -> %s", path)
    run.finish()


if __name__ == "__main__":
    main()
