"""Run-matrix orchestrator: datasets × generators × seeds × ablations, resumable.

Drives the whole experiment grid in-process. For each (dataset, ablation, seed)
it loads/prepares the leakage-safe splits once, lazily trains the SPADE stages
only if the ``spade`` generator still needs them, then runs each generator's
cell. Every cell's metrics are written to its own JSON file under
``results_dir/<dataset>/<ablation>/<generator>_seed<seed>.json``; a re-run reads
finished cells from disk and skips them, so a long matrix can be resumed after an
interruption (or extended with a new generator/seed) without recomputation.

The splits loader is injectable so tests can feed tiny in-memory data; the default
loads cached splits or builds them from the raw dataset.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

from spade.config.configs import ExperimentConfig
from spade.data import (
    build_store,
    load_dataset,
    load_splits,
    save_splits,
    split_store,
)
from spade.data.split import Splits
from spade.experiments.ablations import get_ablation
from spade.experiments.pipeline import run_cell, train_spade_stages
from spade.utils import get_logger

__all__ = ["default_splits_loader", "cell_config", "run_matrix"]

logger = get_logger(__name__)

SplitsLoader = Callable[[ExperimentConfig], Splits]


def default_splits_loader(cfg: ExperimentConfig) -> Splits:
    """Load cached splits for the cell's dataset/seed, building them if absent."""
    d = cfg.data
    try:
        return load_splits(d.data_dir, d.dataset, cfg.seed)
    except FileNotFoundError:
        logger.info("building splits for %s seed=%d", d.dataset, cfg.seed)
        raw = load_dataset(d.dataset, d.data_dir)
        store = build_store(raw, d.min_user_interactions, d.min_item_interactions)
        splits = split_store(store, d.val_frac, d.test_frac, cfg.seed)
        save_splits(splits, d.data_dir, d.dataset)
        return splits


def cell_config(
    base_cfg: ExperimentConfig, dataset: str, seed: int, ablation_name: str
) -> ExperimentConfig:
    """Specialize ``base_cfg`` to one cell's dataset/seed and apply its ablation."""
    cfg = copy.deepcopy(base_cfg)
    cfg.seed = seed
    cfg.data.dataset = dataset
    return get_ablation(ablation_name).apply(cfg)


def run_matrix(
    base_cfg: ExperimentConfig,
    *,
    datasets: list[str],
    generators: list[str],
    seeds: list[int],
    ablations: list[str],
    results_dir: str | Path,
    cache_dir: str | Path | None = None,
    splits_loader: SplitsLoader = default_splits_loader,
    track: bool = False,
) -> list[dict]:
    """Execute (or resume) the full grid; return one flat record per cell.

    Each record is ``{dataset, ablation, seed, generator, cell}`` where ``cell``
    is the metric dict from :func:`evaluate_output`. Stage models are trained at
    most once per (dataset, ablation, seed) and only when the ``spade`` generator
    is among the cells still to compute.

    With ``track=True``, training curves and per-cell metrics are logged to W&B
    (honoring ``base_cfg.wandb_mode``): one run per trained stage and one per
    evaluated cell, named ``<name>-<dataset>-<ablation>-seed<seed>[-<stage|gen>]``.
    Tracking is off by default so test/offline runs never touch W&B.
    """
    results_dir = Path(results_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else results_dir / "_stage_cache"

    records: list[dict] = []
    for dataset in datasets:
        for ablation_name in ablations:
            for seed in seeds:
                cfg = cell_config(base_cfg, dataset, seed, ablation_name)
                cell_dir = results_dir / dataset / ablation_name
                cell_dir.mkdir(parents=True, exist_ok=True)
                paths = {g: cell_dir / f"{g}_seed{seed}.json" for g in generators}
                missing = [g for g, p in paths.items() if not p.exists()]

                prefix = (
                    f"{base_cfg.name}-{dataset}-{ablation_name}-seed{seed}"
                    if track else None
                )

                if missing:
                    splits = splits_loader(cfg)
                    models = None
                    if "spade" in missing:
                        models = train_spade_stages(
                            cfg, splits, cache_dir=cache_dir / dataset,
                            wandb_prefix=prefix,
                        )
                    for g in missing:
                        logger.info(
                            "cell %s/%s/%s seed=%d", dataset, ablation_name, g, seed
                        )
                        cell = run_cell(
                            cfg, g, splits, models=models,
                            wandb_name=f"{prefix}-{g}" if prefix else None,
                        )
                        paths[g].write_text(json.dumps(cell, indent=2, sort_keys=True))

                for g in generators:
                    cell = json.loads(paths[g].read_text())
                    records.append({
                        "dataset": dataset,
                        "ablation": ablation_name,
                        "seed": seed,
                        "generator": g,
                        "cell": cell,
                    })
    return records
