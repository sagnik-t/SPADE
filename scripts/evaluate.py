"""Evaluate a synthesized SPADE run against the full metric suite.

Loads the representation/generative exports and the saved synthetic dataset for
the configured dataset/seed, computes PGPS + NDI (per reference model), latent
W₂, KS degree distance, and the TS-TR downstream comparison, then writes the
results to JSON under ``output_dir``::

    poetry run python scripts/evaluate.py --data.dataset ml-100k --seed 42

Run the representation, generative, and synthesis scripts first.
"""

from __future__ import annotations

import json
from pathlib import Path

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.eval import run_evaluation
from spade.utils import get_logger, init_wandb, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)

    results = run_evaluation(cfg)

    base = Path(cfg.output_dir) / cfg.data.dataset
    out = base / f"evaluation_seed_{cfg.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    logger.info("evaluation complete -> %s", out)

    # Forward the flat scalar metrics to W&B if configured.
    with init_wandb(cfg, project=cfg.wandb_project, name=f"{cfg.name}-eval",
                    mode=cfg.wandb_mode) as run:
        flat: dict[str, float] = {}
        for kind, metrics in results["geometry"].items():
            flat.update({f"{kind}/{k}": v for k, v in metrics.items()})
        flat.update(results["latent"])
        flat.update(results["degree"])
        flat.update(results["tstr"])
        run.log(flat)


if __name__ == "__main__":
    main()
