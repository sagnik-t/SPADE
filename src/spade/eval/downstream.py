"""TS-TR — the train-on-synthetic / test-on-real utility check.

A synthetic dataset is *useful* if a recommender trained on it is about as good
as one trained on real data. TS-TR quantifies that: train the downstream
recommender on the synthetic data and on the real data separately, score each on
its own held-out test split, and report the ratio (**RelPerf**). RelPerf ≈ 1.0
means the synthetic data is as learnable as the real data.

Because synthetic and real entities share no index axis, "test on real" can't
mean scoring a synth-trained model on real test *pairs* — those embeddings don't
exist. The implemented protocol instead compares each model on a held-out split
of its *own* universe (synthetic→synthetic-holdout, real→real-test) and forms
RelPerf from the two, which is the standard resolution and isolates whether the
synthetic data carries the same recommendable structure as the real data.

Ranking uses the reference recommenders from :mod:`spade.eval.reference`; the
metric helpers live in :mod:`spade.eval.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from spade.config.configs import EvalConfig
from spade.data.interactions import InteractionStore
from spade.data.split import split_store
from spade.eval.metrics import ranking_metrics
from spade.eval.reference import train_recommender
from spade.utils import get_logger

__all__ = ["TSTRResult", "evaluate_ranking", "ts_tr"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TSTRResult:
    """Downstream metrics for synthetic- and real-trained recommenders + RelPerf."""

    synthetic: dict[str, float]
    real: dict[str, float]
    rel_perf: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, d in (("synth", self.synthetic), ("real", self.real)):
            out.update({f"tstr_{name}_{k}": v for k, v in d.items()})
        out.update({f"relperf_{k}": v for k, v in self.rel_perf.items()})
        return out


def _seen_and_relevant(
    train: InteractionStore, test: InteractionStore
) -> tuple[list[set[int]], dict[int, set[int]]]:
    """Per-user train-seen item sets (to mask) and test relevant sets (to score)."""
    seen: list[set[int]] = [set() for _ in range(train.n_users)]
    for u, i in zip(train.user_idx.tolist(), train.item_idx.tolist(), strict=True):
        seen[u].add(i)
    relevant: dict[int, set[int]] = {}
    for u, i in zip(test.user_idx.tolist(), test.item_idx.tolist(), strict=True):
        relevant.setdefault(u, set()).add(i)
    return seen, relevant


def evaluate_ranking(
    model,
    train: InteractionStore,
    test: InteractionStore,
    ks: list[int],
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    """Rank all items per test user (masking train-seen) and score the metrics.

    Items observed in ``train`` are masked to ``-inf`` so the model is judged on
    held-out discovery, not memorization. Scoring is chunked over users.
    """
    seen, relevant = _seen_and_relevant(train, test)
    eval_users = sorted(relevant)
    if not eval_users:
        return ranking_metrics({}, {}, ks)

    kmax = max(ks)
    n_items = train.n_items
    ranked_by_user: dict[int, np.ndarray] = {}
    for start in range(0, len(eval_users), batch_size):
        batch = np.array(eval_users[start : start + batch_size], dtype=np.int64)
        scores = np.array(model.all_item_scores(jnp.asarray(batch)))  # (b, n_items), writable
        for row, u in enumerate(batch.tolist()):
            if seen[u]:
                scores[row, list(seen[u])] = -np.inf
            top = min(kmax, n_items)
            part = np.argpartition(-scores[row], kth=top - 1)[:top]
            ranked_by_user[u] = part[np.argsort(-scores[row][part])]
    return ranking_metrics(ranked_by_user, relevant, ks)


def ts_tr(
    synth_store: InteractionStore,
    real_train: InteractionStore,
    real_test: InteractionStore,
    cfg: EvalConfig,
    *,
    seed: int = 0,
) -> TSTRResult:
    """Run the TS-TR comparison and return synthetic/real metrics + RelPerf.

    The synthetic store is split per-user (no validation) into its own train/test
    holdout; the real splits are passed in directly. RelPerf is the per-metric
    ratio synthetic/real, guarding against a zero real denominator.
    """
    synth_splits = split_store(
        synth_store, val_frac=0.0, test_frac=cfg.tstr_test_frac, seed=seed
    )
    kind = cfg.tstr_model

    logger.info("TS-TR: training downstream %s on synthetic data", kind)
    synth_model = train_recommender(synth_splits.train, cfg, kind=kind, seed=seed)
    synth_metrics = evaluate_ranking(
        synth_model, synth_splits.train, synth_splits.test, cfg.topk
    )

    logger.info("TS-TR: training downstream %s on real data", kind)
    real_model = train_recommender(real_train, cfg, kind=kind, seed=seed)
    real_metrics = evaluate_ranking(real_model, real_train, real_test, cfg.topk)

    rel_perf = {
        k: (synth_metrics[k] / real_metrics[k]) if real_metrics[k] > 0 else 0.0
        for k in synth_metrics
    }
    return TSTRResult(synthetic=synth_metrics, real=real_metrics, rel_perf=rel_perf)
