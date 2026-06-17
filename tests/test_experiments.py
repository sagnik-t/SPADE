"""Phase 7 orchestration tests: pipeline, ablations, resumable matrix, aggregation.

Runs the full grid end-to-end on tiny in-memory splits (injected via a custom
splits loader) with 1–2 epoch stages — checking the orchestration contract
(stage caching, cell evaluation, resume, mean±std tables, the gating ablation),
not model quality.
"""

import jax
import numpy as np
import pytest
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.data.interactions import IndexMap, InteractionStore
from spade.data.split import split_store
from spade.experiments import (
    ABLATIONS,
    DEFERRED_ABLATIONS,
    build_summary,
    flatten_cell,
    get_ablation,
    run_cell,
    run_matrix,
    train_spade_stages,
    write_tables,
)
from spade.experiments.pipeline import SpadeModels
from spade.models import GenerativeModel, RatingVocab, RepresentationModel
from spade.synthesis.synthesizer import SynthesisModel


def _store(n_users=24, n_items=18, per_user=6, seed=0) -> InteractionStore:
    rng = np.random.default_rng(seed)
    u, i, r = [], [], []
    for user in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        for it in items:
            u.append(user)
            i.append(int(it))
            r.append(int(rng.integers(1, 6)))
    return InteractionStore(
        user_idx=np.array(u, dtype=np.int64),
        item_idx=np.array(i, dtype=np.int64),
        ratings=np.array(r, dtype=np.float32),
        n_users=n_users,
        n_items=n_items,
        user_map=IndexMap.from_raw(np.arange(n_users)),
        item_map=IndexMap.from_raw(np.arange(n_items)),
    )


def _splits(seed=0):
    return split_store(_store(seed=seed), val_frac=0.15, test_frac=0.15, seed=seed)


def _cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.dataset = "tiny"
    r = cfg.representation
    r.latent_dim, r.encoder_hidden, r.gate_hidden, r.decoder_hidden = 8, [16], [16], [16]
    r.epochs, r.batch_size, r.early_stop_patience = 2, 64, 5
    g = cfg.generative
    g.noise_dim, g.generator_hidden, g.critic_hidden = 8, [16], [16]
    g.epochs, g.n_critic, g.batch_size = 2, 1, 32
    cfg.synthesis.alpha = cfg.synthesis.beta = 2.0
    cfg.synthesis.gamma = 1.0
    e = cfg.eval
    e.ref_dim, e.ref_hidden, e.ref_epochs, e.topk, e.neighbor_k = 8, [16], 2, [3, 5], 3
    b = cfg.baselines
    b.noise_mf_dim = b.deepmf_dim = b.vae_dim = 8
    b.noise_mf_epochs = b.deepmf_epochs = b.vae_epochs = 1
    b.gan_noise_dim = b.vae_latent = 8
    b.gan_hidden = b.vae_hidden = [16]
    b.gan_epochs = 1
    b.gan_n_generate = b.vae_n_generate = 1500
    b.kmeans_iters = 5
    return cfg


def _loader(splits):
    return lambda cfg: splits


# --------------------------------------------------------------------------- #
# Ablations                                                                   #
# --------------------------------------------------------------------------- #
def test_ablation_registry_and_transforms():
    assert {"base", "alpha_1.5", "alpha_3.0", "latent_reg_off", "gating_off"} <= set(ABLATIONS)
    cfg = _cfg()
    assert get_ablation("gating_off").apply(cfg).synthesis.gating is False
    assert get_ablation("alpha_3.0").apply(cfg).synthesis.alpha == 3.0
    assert get_ablation("latent_reg_off").apply(cfg).generative.moment_lambda == 0.0
    # original config is untouched (transforms return copies)
    assert cfg.synthesis.gating is True


def test_deferred_ablations_raise():
    assert set(DEFERRED_ABLATIONS) == {"joint_generator", "continuous_decoder"}
    with pytest.raises(NotImplementedError):
        get_ablation("joint_generator").apply(_cfg())


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def test_train_spade_stages_caches(tmp_path):
    cfg, splits = _cfg(), _splits()
    models = train_spade_stages(cfg, splits, cache_dir=tmp_path)
    assert isinstance(models, SpadeModels)
    files = list(tmp_path.glob("*.npz"))
    assert any(f.name.startswith("rep_") for f in files)
    assert any(f.name.startswith("gen_") for f in files)
    # second call loads from cache and still returns usable models
    again = train_spade_stages(cfg, splits, cache_dir=tmp_path)
    assert isinstance(again, SpadeModels)


def test_run_cell_spade_has_geometry_and_baseline_does_not(tmp_path):
    cfg, splits = _cfg(), _splits()
    models = train_spade_stages(cfg, splits, cache_dir=tmp_path)

    spade_cell = run_cell(cfg, "spade", splits, models=models)
    assert spade_cell["geometry"] is not None
    assert set(spade_cell["geometry"]) == {"mf", "ncf"}
    assert "map" in spade_cell["tstr"] or any("relperf" in k for k in spade_cell["tstr"])

    rand_cell = run_cell(cfg, "random", splits)
    assert rand_cell["geometry"] is None
    assert rand_cell["latent"] is None
    assert rand_cell["degree"]


def test_run_cell_spade_requires_models():
    with pytest.raises(ValueError):
        run_cell(_cfg(), "spade", _splits(), models=None)


# --------------------------------------------------------------------------- #
# Matrix + resume                                                             #
# --------------------------------------------------------------------------- #
def test_run_matrix_and_resume(tmp_path):
    cfg = _cfg()
    splits = _splits()
    kwargs = dict(
        datasets=["tiny"], generators=["spade", "random"], seeds=[0, 1],
        ablations=["base"], results_dir=tmp_path, splits_loader=_loader(splits),
    )
    records = run_matrix(cfg, **kwargs)
    assert len(records) == 4  # 2 generators x 2 seeds x 1 ablation
    cell_files = list((tmp_path / "tiny" / "base").glob("*.json"))
    assert len(cell_files) == 4

    # Resume: cells are cached, re-run returns the same records without recompute.
    again = run_matrix(cfg, **kwargs)
    assert len(again) == 4
    assert {(r["generator"], r["seed"]) for r in again} == {
        ("spade", 0), ("spade", 1), ("random", 0), ("random", 1)
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def test_aggregate_writes_tables(tmp_path):
    cfg = _cfg()
    splits = _splits()
    records = run_matrix(
        cfg, datasets=["tiny"], generators=["random", "marginal"], seeds=[0, 1],
        ablations=["base"], results_dir=tmp_path / "m", splits_loader=_loader(splits),
    )
    summary = build_summary(records)
    assert not summary.empty
    expected_cols = {"dataset", "ablation", "generator", "metric", "mean", "std", "n_seeds"}
    assert expected_cols <= set(summary.columns)
    assert (summary["n_seeds"] == 2).all()

    written = write_tables(records, tmp_path / "tables")
    assert written["summary"].exists()
    assert (tmp_path / "tables" / "tiny__base.md").exists()


def test_flatten_cell_includes_degree_and_tstr():
    cell = {
        "synthetic": {"density": 0.1, "nnz": 50},
        "degree": {"ks_user_degree": 0.2, "ks_item_degree": 0.3},
        "tstr": {"relperf_map": 0.8},
        "geometry": {"mf": {"pgps": 0.5, "ndi": 0.1}},
        "latent": {"w2_user_latent": 1.2},
    }
    flat = flatten_cell(cell)
    assert flat["density"] == 0.1
    assert flat["ks_user_degree"] == 0.2
    assert flat["relperf_map"] == 0.8
    assert flat["mf/pgps"] == 0.5
    assert flat["w2_user_latent"] == 1.2


# --------------------------------------------------------------------------- #
# Gating ablation effect                                                       #
# --------------------------------------------------------------------------- #
def test_gating_off_keeps_more_interactions():
    cfg = _cfg()
    n_users, n_items, n_levels = 24, 18, 5
    rep = RepresentationModel(n_users, n_items, n_levels, cfg.representation, rngs=nnx.Rngs(0))
    gen = GenerativeModel(cfg.representation.latent_dim, cfg.generative, rngs=nnx.Rngs(1))
    vocab = RatingVocab(values=np.arange(1, n_levels + 1, dtype=np.float32))

    def synth(gating: bool):
        c = _cfg()
        c.synthesis.gating = gating
        model = SynthesisModel(
            rep, gen, vocab, source_n_users=n_users, source_n_items=n_items,
            source_rho=0.3, cfg=c.synthesis,
        )
        return model.synthesize(jax.random.key(0)).nnz

    # Bypassing the gate keeps every candidate, so density can only go up.
    assert synth(gating=False) >= synth(gating=True)
