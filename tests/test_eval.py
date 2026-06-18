"""Evaluation-suite tests: reference models, transductive map, metrics.

Runs on CPU JAX with NumPy/SciPy. Reference recommenders are trained for only a
couple of epochs on tiny synthetic stores — the tests check shapes, ranges,
determinism, and known closed-form values rather than recommendation quality.
"""

import numpy as np
import pytest

from spade.config.configs import ExperimentConfig
from spade.data.interactions import IndexMap, InteractionStore
from spade.eval import (
    average_precision_at_k,
    build_reference_space,
    degree_ks,
    evaluate_ranking,
    fit_linear_map,
    fit_transductive,
    gaussian_w2,
    ks_distance,
    ndcg_at_k,
    ndi,
    pgps,
    ranking_metrics,
    recall_at_k,
    train_recommender,
    ts_tr,
)


def _store(user_idx, item_idx, ratings, n_users, n_items) -> InteractionStore:
    return InteractionStore(
        user_idx=np.asarray(user_idx, dtype=np.int64),
        item_idx=np.asarray(item_idx, dtype=np.int64),
        ratings=np.asarray(ratings, dtype=np.float32),
        n_users=n_users,
        n_items=n_items,
        user_map=IndexMap.from_raw(np.arange(n_users)),
        item_map=IndexMap.from_raw(np.arange(n_items)),
    )


def _dense_store(n_users=15, n_items=12, per_user=6, seed=0) -> InteractionStore:
    rng = np.random.default_rng(seed)
    u, i, r = [], [], []
    for user in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        for it in items:
            u.append(user)
            i.append(int(it))
            r.append(rng.integers(1, 6))
    return _store(u, i, r, n_users, n_items)


def _eval_cfg(seed=0):
    cfg = ExperimentConfig()
    cfg.seed = seed
    cfg.eval.ref_dim = 8
    cfg.eval.ref_hidden = [16]
    cfg.eval.ref_epochs = 2
    cfg.eval.topk = [3, 5]
    return cfg


# --------------------------------------------------------------------------- #
# Ranking metrics                                                             #
# --------------------------------------------------------------------------- #
def test_recall_ndcg_map_known_values():
    ranked = np.array([0, 1, 2, 3])
    relevant = {1, 3}
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(0.63092 / 1.63092, rel=1e-3)
    assert average_precision_at_k(ranked, relevant, 2) == pytest.approx(0.25)


def test_recall_empty_relevant_is_zero():
    ranked = np.array([0, 1, 2])
    assert recall_at_k(ranked, set(), 3) == 0.0
    assert ndcg_at_k(ranked, set(), 3) == 0.0


def test_ranking_metrics_aggregates_users():
    ranked = {0: np.array([1, 0, 2]), 1: np.array([2, 1, 0])}
    relevant = {0: {1}, 1: {0}}
    out = ranking_metrics(ranked, relevant, [1, 3])
    assert set(out) == {"recall@1", "ndcg@1", "recall@3", "ndcg@3", "map"}
    assert out["recall@1"] == pytest.approx(0.5)  # user 0 hits at rank 1, user 1 misses


# --------------------------------------------------------------------------- #
# Transductive linear map                                                     #
# --------------------------------------------------------------------------- #
def test_linear_map_recovers_affine_relation():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((60, 5))
    true_w = rng.standard_normal((6, 4))  # augmented (5 + bias) -> 4
    target = np.concatenate([source, np.ones((60, 1))], axis=1) @ true_w
    mapped = fit_linear_map(source, target, ridge=1e-9).apply(source)
    assert mapped.shape == (60, 4)
    assert np.allclose(mapped, target, atol=1e-3)


def test_fit_transductive_shapes():
    rng = np.random.default_rng(1)
    zu, zi = rng.standard_normal((20, 6)), rng.standard_normal((16, 6))
    ru, ri = rng.standard_normal((20, 4)), rng.standard_normal((16, 4))
    embed = fit_transductive(zu, zi, ru, ri, ridge=1e-2)
    assert embed.embed_users(rng.standard_normal((7, 6))).shape == (7, 4)
    assert embed.embed_items(rng.standard_normal((9, 6))).shape == (9, 4)


# --------------------------------------------------------------------------- #
# PGPS                                                                         #
# --------------------------------------------------------------------------- #
def test_pgps_high_when_synth_matches_real():
    rng = np.random.default_rng(2)
    real = rng.standard_normal((40, 8)).astype(np.float32)
    res = pgps(real, real.copy(), k=5)
    assert res.random_baseline == pytest.approx(5 / 40)
    assert res.pgps > 0.5 and res.lift > 0.0


def test_pgps_near_random_when_synth_unrelated():
    rng = np.random.default_rng(3)
    real = rng.standard_normal((60, 8)).astype(np.float32)
    synth = rng.standard_normal((60, 8)).astype(np.float32)
    res = pgps(real, synth, k=5)
    assert 0.0 <= res.pgps <= 1.0
    assert res.pgps < 0.5  # unrelated cloud should not preserve geometry


# --------------------------------------------------------------------------- #
# NDI                                                                          #
# --------------------------------------------------------------------------- #
def test_ndi_low_when_synth_users_far_away():
    rng = np.random.default_rng(4)
    real = rng.standard_normal((50, 8)).astype(np.float32)
    synth = real + 1000.0  # displaced far from every real user
    res = ndi(real, synth, k=5, metric="euclidean")
    assert res.ndi == pytest.approx(0.0, abs=1e-6)
    assert res.intrusion == pytest.approx(0.0, abs=1e-6)


def test_ndi_high_when_synth_users_overlap_real():
    rng = np.random.default_rng(5)
    real = rng.standard_normal((50, 8)).astype(np.float32)
    synth = real.copy()  # synthetic users sit on top of real users
    res = ndi(real, synth, k=5)
    assert res.ndi > 0.5 and res.intrusion > 0.5


# --------------------------------------------------------------------------- #
# Distribution metrics                                                        #
# --------------------------------------------------------------------------- #
def test_gaussian_w2_zero_for_identical_clouds():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 5))
    assert gaussian_w2(x, x.copy()) == pytest.approx(0.0, abs=1e-4)


def test_gaussian_w2_grows_with_mean_shift():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 5))
    near = gaussian_w2(x, x + 0.5)
    far = gaussian_w2(x, x + 3.0)
    assert far > near > 0.0


def test_ks_distance_and_degree_ks():
    a = np.zeros(100)
    b = np.ones(100)
    assert ks_distance(a, b) == pytest.approx(1.0)
    real = _dense_store(seed=8)
    out = degree_ks(real, real)
    assert out["ks_user_degree"] == pytest.approx(0.0)
    assert set(out) == {"ks_user_degree", "ks_item_degree"}


# --------------------------------------------------------------------------- #
# Reference models                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["mf", "ncf"])
def test_reference_space_shapes_and_scoring(kind):
    cfg = _eval_cfg()
    store = _dense_store(n_users=15, n_items=12, seed=9)
    ref = build_reference_space(store, cfg.eval, kind=kind, seed=cfg.seed)
    assert ref.user_emb.shape == (15, cfg.eval.ref_dim)
    assert ref.item_emb.shape == (12, cfg.eval.ref_dim)
    import jax.numpy as jnp

    scores = ref.model.all_item_scores(jnp.arange(4))
    assert scores.shape == (4, 12)


# --------------------------------------------------------------------------- #
# Downstream ranking + TS-TR                                                   #
# --------------------------------------------------------------------------- #
def test_evaluate_ranking_masks_and_scores():
    cfg = _eval_cfg()
    train = _dense_store(n_users=15, n_items=12, per_user=6, seed=10)
    test = _dense_store(n_users=15, n_items=12, per_user=2, seed=11)
    model = train_recommender(train, cfg.eval, kind="mf", seed=cfg.seed)
    out = evaluate_ranking(model, train, test, cfg.eval.topk)
    assert "map" in out
    for k in cfg.eval.topk:
        assert 0.0 <= out[f"recall@{k}"] <= 1.0
        assert 0.0 <= out[f"ndcg@{k}"] <= 1.0


def test_ts_tr_runs_and_reports_relperf():
    cfg = _eval_cfg()
    synth = _dense_store(n_users=18, n_items=14, per_user=6, seed=12)
    real_train = _dense_store(n_users=15, n_items=12, per_user=6, seed=13)
    real_test = _dense_store(n_users=15, n_items=12, per_user=2, seed=14)
    res = ts_tr(synth, real_train, real_test, cfg.eval, seed=cfg.seed)
    assert "map" in res.synthetic and "map" in res.real
    flat = res.as_dict()
    assert any(k.startswith("relperf_") for k in flat)
    assert any(k.startswith("tstr_synth_") for k in flat)


def test_bpr_recommender_runs_and_scores():
    cfg = _eval_cfg()
    cfg.eval.bpr_neg_samples = 2  # exercise the multi-negative repeat path
    train = _dense_store(n_users=15, n_items=12, per_user=6, seed=10)
    test = _dense_store(n_users=15, n_items=12, per_user=2, seed=11)
    model = train_recommender(train, cfg.eval, kind="bpr", seed=cfg.seed)
    out = evaluate_ranking(model, train, test, cfg.eval.topk)
    assert "map" in out
    for k in cfg.eval.topk:
        assert 0.0 <= out[f"recall@{k}"] <= 1.0
        assert 0.0 <= out[f"ndcg@{k}"] <= 1.0


def test_ts_tr_supports_bpr_downstream():
    cfg = _eval_cfg()
    cfg.eval.tstr_model = "bpr"
    synth = _dense_store(n_users=18, n_items=14, per_user=6, seed=12)
    real_train = _dense_store(n_users=15, n_items=12, per_user=6, seed=13)
    real_test = _dense_store(n_users=15, n_items=12, per_user=2, seed=14)
    res = ts_tr(synth, real_train, real_test, cfg.eval, seed=cfg.seed)
    flat = res.as_dict()
    assert any(k.startswith("tstr_synth_") for k in flat)
    assert any(k.startswith("relperf_") for k in flat)
