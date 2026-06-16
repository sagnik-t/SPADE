"""Generate a synthetic dataset from a baseline generator and save it.

Selects a baseline by name, sizes it to the configured dataset/seed's train
universe, generates, and writes the discrete dataset under ``output_dir``::

    poetry run python scripts/run_baseline.py --baseline ganrs \
        --data.dataset ml-100k --seed 42

``--baseline`` is one of: random, marginal, noise_mf, ganrs, vae. All other flags
are the usual nested config flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spade.baselines import BASELINE_REGISTRY
from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.data import load_splits
from spade.utils import get_logger, jax_key, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--baseline", required=True, choices=sorted(BASELINE_REGISTRY))
    known, rest = pre.parse_known_args()

    cfg: ExperimentConfig = parse_args(ExperimentConfig, rest)
    set_global_seed(cfg.seed)

    splits = load_splits(cfg.data.data_dir, cfg.data.dataset, cfg.seed)
    generator = BASELINE_REGISTRY[known.baseline](splits.train, cfg)
    output = generator.generate(jax_key(cfg.seed))

    base = Path(cfg.output_dir) / cfg.data.dataset
    out = base / f"baseline_{known.baseline}_seed_{cfg.seed}.npz"
    output.dataset.save(out)
    logger.info("baseline %s -> %s | %s", known.baseline, out, output.dataset.summary())


if __name__ == "__main__":
    main()
