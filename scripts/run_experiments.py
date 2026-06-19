"""Run (or resume) the SPADE experiment matrix and write aggregate tables.

Trains SPADE once per (dataset, seed), runs SPADE plus the baselines through the
full metric suite across the requested seeds and ablations, and writes mean±std
tables under ``<output_dir>/tables``. The matrix is resumable: finished cells are
cached as JSON and skipped on re-run.

    poetry run python scripts/run_experiments.py \
        --datasets ml-100k --generators spade random marginal noise_mf ganrs vae \
        --seeds 0 1 2 3 4 \
        --ablations base alpha_1.5 alpha_3.0 latent_reg_off gating_off \
                    joint_generator continuous_decoder

All other flags are the usual nested config flags (e.g. ``--representation.epochs``).
Training curves and per-cell metrics are logged to W&B (one run per trained stage
and per evaluated cell); pass ``--wandb-mode disabled`` to turn that off, or
``offline`` to log locally. Amazon stays deferred until its subset is pinned;
full-scale runs belong on GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.experiments import run_matrix, write_tables
from spade.utils import get_logger, load_env, set_global_seed

logger = get_logger(__name__)

_DEFAULT_GENERATORS = ["spade", "random", "marginal", "noise_mf", "ganrs", "vae"]


def main() -> None:
    load_env()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--datasets", nargs="+", default=["ml-100k"])
    pre.add_argument("--generators", nargs="+", default=_DEFAULT_GENERATORS)
    pre.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    pre.add_argument("--ablations", nargs="+", default=["base"])
    known, rest = pre.parse_known_args()

    cfg: ExperimentConfig = parse_args(ExperimentConfig, rest)
    set_global_seed(cfg.seed)

    results_dir = Path(cfg.output_dir) / "matrix"
    records = run_matrix(
        cfg,
        datasets=known.datasets,
        generators=known.generators,
        seeds=known.seeds,
        ablations=known.ablations,
        results_dir=results_dir,
        track=True,
    )

    tables_dir = Path(cfg.output_dir) / "tables"
    written = write_tables(records, tables_dir)
    logger.info(
        "matrix complete: %d cells over %d datasets -> tables in %s (%d files)",
        len(records), len(known.datasets), tables_dir, len(written),
    )


if __name__ == "__main__":
    main()
