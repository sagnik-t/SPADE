"""End-to-end pipeline reproduction run, launched from the command line.

This is the integration check the roadmap calls for: it runs the *entire* SPADE
pipeline for one ``(dataset, seed)`` — representation training, generative
training, Stage III synthesis, and the full evaluation suite — as a single
command, then verifies the properties that must hold on a real run (it checks
wiring and invariants, not model quality):

* the stage artifacts are written under ``output_dir`` with the exact names the
  synthesis/eval scripts depend on;
* Stage III domain constraints hold — entity counts, rating support, and no
  duplicate ``(user, item)`` pairs;
* **determinism** — re-running synthesis under the same seed reproduces the
  dataset bit-for-bit;
* **PGPS sanity** — PGPS sits above its random baseline (geometry preserved) and
  below a degenerate trivial-copy score, for every reference model.

Unlike the unit tests, this trains real models, so it is a script you run by hand
(CPU is fine for ml-100k; use the GPU box for ml-1m)::

    poetry run python scripts/smoke_pipeline.py --data.dataset ml-100k --seed 42

``--skip-train`` reuses existing stage exports and only re-checks synthesis and
evaluation. The script prints a PASS/FAIL line per check and exits non-zero if any
check fails, so it doubles as a reproducibility gate before a full matrix run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.data import build_store, load_dataset, load_splits, save_splits, split_store
from spade.data.split import Splits
from spade.eval import run_evaluation
from spade.synthesis import SynthesisModel, SyntheticDataset
from spade.training import (
    GenerativeTrainer,
    RepresentationTrainer,
    load_generative_model,
    load_representation_model,
)
from spade.utils import get_logger, jax_key, load_env, set_global_seed

logger = get_logger(__name__)


class Checks:
    """Collects pass/fail results and prints them as they happen."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, condition: bool, message: str) -> None:
        status = "PASS" if condition else "FAIL"
        logger.info("[%s] %s", status, message)
        if not condition:
            self.failures += 1


def _prepare_splits(cfg: ExperimentConfig) -> Splits:
    """Load cached leakage-safe splits for the dataset/seed, building if absent."""
    d = cfg.data
    try:
        splits = load_splits(d.data_dir, d.dataset, cfg.seed)
        logger.info("loaded cached splits for %s seed=%d", d.dataset, cfg.seed)
    except FileNotFoundError:
        logger.info("building splits from raw %s", d.dataset)
        raw = load_dataset(d.dataset, d.data_dir)
        store = build_store(raw, d.min_user_interactions, d.min_item_interactions)
        splits = split_store(store, d.val_frac, d.test_frac, cfg.seed)
        save_splits(splits, d.data_dir, d.dataset)
    logger.info("train/val/test = %s", splits.summary())
    return splits


def _train_stages(cfg: ExperimentConfig, splits: Splits) -> None:
    """Train and export the representation and generative stages."""
    rep = RepresentationTrainer(cfg, splits.train, splits.val).fit()
    rep.export(cfg.output_dir, cfg.data.dataset)
    rep.export_model(cfg.output_dir, cfg.data.dataset)
    z_u, z_i = rep.model.export_embeddings(splits.train.n_users, splits.train.n_items)
    gen = GenerativeTrainer(cfg, np.asarray(z_u), np.asarray(z_i)).fit()
    gen.export(cfg.output_dir, cfg.data.dataset)


def main() -> None:
    load_env()
    skip_train = "--skip-train" in sys.argv
    if skip_train:
        sys.argv.remove("--skip-train")
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)

    base = Path(cfg.output_dir) / cfg.data.dataset
    rep_model_path = base / f"representation_model_seed_{cfg.seed}.npz"
    gen_path = base / f"generative_seed_{cfg.seed}.npz"
    synth_path = base / f"synthetic_seed_{cfg.seed}.npz"

    splits = _prepare_splits(cfg)
    train = splits.train

    if not skip_train:
        logger.info("=== training stages ===")
        _train_stages(cfg, splits)

    checks = Checks()
    checks.check(rep_model_path.exists(), f"representation export written: {rep_model_path.name}")
    checks.check(gen_path.exists(), f"generative export written: {gen_path.name}")

    # === synthesis (+ determinism) ===
    logger.info("=== synthesis ===")
    representation, vocab = load_representation_model(rep_model_path, cfg, seed=cfg.seed)
    generative = load_generative_model(gen_path, cfg, seed=cfg.seed)
    synthesizer = SynthesisModel(
        representation, generative, vocab,
        source_n_users=train.n_users, source_n_items=train.n_items,
        source_rho=train.rho, cfg=cfg.synthesis,
    )
    synth = synthesizer.synthesize(jax_key(cfg.seed))
    synth.save(synth_path)
    checks.check(synth_path.exists(), f"synthetic dataset written: {synth_path.name}")

    # Determinism: a second synthesis under the same seed must match exactly.
    again = synthesizer.synthesize(jax_key(cfg.seed))
    deterministic = (
        np.array_equal(synth.user_idx, again.user_idx)
        and np.array_equal(synth.item_idx, again.item_idx)
        and np.array_equal(synth.ratings, again.ratings)
    )
    checks.check(deterministic, "synthesis is deterministic under a fixed seed")

    # Stage III domain constraints on the saved artifact.
    reloaded = SyntheticDataset.load(synth_path)
    checks.check(
        reloaded.n_users == int(np.ceil(cfg.synthesis.alpha * train.n_users))
        and reloaded.n_items == int(np.ceil(cfg.synthesis.beta * train.n_items)),
        "synthetic entity counts match alpha/beta expansion",
    )
    allowed = set(vocab.values.tolist())
    checks.check(
        set(np.unique(reloaded.ratings).tolist()) <= allowed,
        "synthetic ratings stay within the learned rating support",
    )
    flat = reloaded.user_idx.astype(np.int64) * reloaded.n_items + reloaded.item_idx
    checks.check(
        np.unique(flat).shape[0] == reloaded.nnz,
        "no duplicate (user, item) pairs in the synthetic dataset",
    )

    # === evaluation (+ PGPS-vs-random sanity) ===
    logger.info("=== evaluation ===")
    results = run_evaluation(cfg)
    for ref_model, metrics in results["geometry"].items():
        checks.check(
            metrics["pgps_lift"] > 0.0,
            f"[{ref_model}] PGPS above random baseline "
            f"(pgps={metrics['pgps']:.4f}, random={metrics['pgps_random']:.4f})",
        )
        checks.check(
            metrics["pgps"] < 0.999,
            f"[{ref_model}] PGPS below trivial-copy ceiling (pgps={metrics['pgps']:.4f})",
        )

    logger.info("=== %d check(s) failed ===", checks.failures)
    if checks.failures:
        sys.exit(1)
    logger.info("smoke pipeline PASSED for %s seed=%d", cfg.data.dataset, cfg.seed)


if __name__ == "__main__":
    main()
