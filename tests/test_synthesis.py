"""Stage III synthesis tests: ANN retrieval, pipeline, constraints, determinism.

Runs on CPU JAX with faiss-cpu. Frozen (untrained) stage models are fine here —
synthesis is pure inference, so the tests exercise shapes, intrinsic constraints,
and determinism rather than sample quality.
"""

import jax
import numpy as np
import pytest
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.models import GenerativeModel, RatingVocab, RepresentationModel
from spade.synthesis import SynthesisModel, SyntheticDataset, top_c_candidates


def _rngs(seed=0):
    return nnx.Rngs(seed)


def _build(latent_dim=8, n_users=20, n_items=16, n_levels=5, seed=0):
    cfg = ExperimentConfig()
    cfg.representation.latent_dim = latent_dim
    cfg.generative.noise_dim = 8
    cfg.generative.generator_hidden = [16]
    cfg.generative.critic_hidden = [16]
    rep = RepresentationModel(n_users, n_items, n_levels, cfg.representation, rngs=_rngs(seed))
    gen = GenerativeModel(latent_dim, cfg.generative, rngs=_rngs(seed + 1))
    vocab = RatingVocab(values=np.arange(1, n_levels + 1, dtype=np.float32))
    return cfg, rep, gen, vocab


def _synth_model(cfg, rep, gen, vocab, *, n_users=20, n_items=16, rho=0.1):
    return SynthesisModel(
        rep,
        gen,
        vocab,
        source_n_users=n_users,
        source_n_items=n_items,
        source_rho=rho,
        cfg=cfg.synthesis,
    )


# --------------------------------------------------------------------------- #
# ANN retrieval                                                               #
# --------------------------------------------------------------------------- #
def test_top_c_candidates_shape_and_range():
    rng = np.random.default_rng(0)
    users = rng.standard_normal((12, 8)).astype(np.float32)
    items = rng.standard_normal((30, 8)).astype(np.float32)
    cand = top_c_candidates(users, items, 5, metric="cosine")
    assert cand.shape == (12, 5)
    assert cand.min() >= 0 and cand.max() < 30
    # within a row candidates are distinct (top-k over distinct items)
    assert all(len(set(row.tolist())) == 5 for row in cand)


def test_top_c_candidates_clamps_to_item_count():
    rng = np.random.default_rng(1)
    cand = top_c_candidates(
        rng.standard_normal((4, 8)).astype(np.float32),
        rng.standard_normal((3, 8)).astype(np.float32),
        10,
    )
    assert cand.shape == (4, 3)  # c clamped to n_items


def test_top_c_candidates_rejects_unknown_metric():
    x = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        top_c_candidates(x, x, 1, metric="manhattan")


# --------------------------------------------------------------------------- #
# Sizing                                                                      #
# --------------------------------------------------------------------------- #
def test_expansion_and_candidate_count():
    cfg, rep, gen, vocab = _build()
    cfg.synthesis.alpha = 2.0
    cfg.synthesis.beta = 3.0
    cfg.synthesis.gamma = 5.0
    model = _synth_model(cfg, rep, gen, vocab, n_users=20, n_items=16, rho=0.1)
    assert model.n_synth_users == 40
    assert model.n_synth_items == 48
    # C = ceil(48 * 0.1 * 5) = 24
    assert model.candidate_count() == 24


def test_candidate_count_clamped_to_items():
    cfg, rep, gen, vocab = _build()
    cfg.synthesis.gamma = 1000.0
    model = _synth_model(cfg, rep, gen, vocab, n_items=16, rho=0.5)
    assert model.candidate_count() == model.n_synth_items


# --------------------------------------------------------------------------- #
# End-to-end synthesis + constraints                                          #
# --------------------------------------------------------------------------- #
def test_synthesize_produces_valid_dataset():
    cfg, rep, gen, vocab = _build(n_users=20, n_items=16, n_levels=5)
    model = _synth_model(cfg, rep, gen, vocab, n_users=20, n_items=16, rho=0.1)
    synth = model.synthesize(jax.random.key(0))

    assert isinstance(synth, SyntheticDataset)
    assert synth.n_users == model.n_synth_users
    assert synth.n_items == model.n_synth_items
    assert synth.user_idx.shape == synth.item_idx.shape == synth.ratings.shape
    if synth.nnz:
        assert synth.user_idx.max() < synth.n_users
        assert synth.item_idx.max() < synth.n_items
        assert set(np.unique(synth.ratings).tolist()) <= set(vocab.values.tolist())
        # density bounded by the candidate set
        assert synth.density <= model.candidate_count() / synth.n_items + 1e-9
    # uniqueness of (user, item)
    flat = synth.user_idx.astype(np.int64) * synth.n_items + synth.item_idx
    assert np.unique(flat).shape[0] == synth.nnz


def test_synthesis_is_deterministic_in_key():
    cfg, rep, gen, vocab = _build()
    model = _synth_model(cfg, rep, gen, vocab)
    a = model.synthesize(jax.random.key(7))
    b = model.synthesize(jax.random.key(7))
    assert np.array_equal(a.user_idx, b.user_idx)
    assert np.array_equal(a.item_idx, b.item_idx)
    assert np.array_equal(a.ratings, b.ratings)


def test_synthetic_dataset_as_store_preserves_universe(tmp_path):
    cfg, rep, gen, vocab = _build()
    model = _synth_model(cfg, rep, gen, vocab)
    synth = model.synthesize(jax.random.key(3))

    store = synth.as_store()
    assert store.n_users == synth.n_users
    assert store.n_items == synth.n_items
    assert store.nnz == synth.nnz

    path = synth.save(tmp_path / "synth.npz")
    reloaded = SyntheticDataset.load(path)
    assert np.array_equal(reloaded.user_idx, synth.user_idx)
    assert reloaded.n_users == synth.n_users
