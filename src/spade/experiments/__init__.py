"""Experiment orchestration (Phase 7).

Ties the whole project together: trains SPADE once per (dataset, seed), runs every
generator (SPADE + baselines) through the metric suite, sweeps the registered
ablations, and aggregates per-cell results into mean±std tables. The matrix is
resumable — finished cells are cached on disk and skipped on re-run.
"""

from spade.experiments.ablations import (
    ABLATIONS,
    DEFERRED_ABLATIONS,
    Ablation,
    get_ablation,
)
from spade.experiments.aggregate import build_summary, flatten_cell, write_tables
from spade.experiments.matrix import cell_config, default_splits_loader, run_matrix
from spade.experiments.pipeline import (
    SpadeModels,
    evaluate_output,
    run_cell,
    train_spade_stages,
)

__all__ = [
    "Ablation",
    "ABLATIONS",
    "DEFERRED_ABLATIONS",
    "get_ablation",
    "SpadeModels",
    "train_spade_stages",
    "evaluate_output",
    "run_cell",
    "cell_config",
    "default_splits_loader",
    "run_matrix",
    "flatten_cell",
    "build_summary",
    "write_tables",
]
