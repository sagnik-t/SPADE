"""Representation-stage model tests: shapes, determinism, distributions, learning.

Runs on CPU JAX. A tiny synthetic dataset keeps everything fast while still
exercising the joint training loop, early stopping, and embedding export.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.data import RawInteractions, build_store, split_store
from spade.models import (
    InteractionGate,
    RatingDecoder,
    RatingVocab,
    RepresentationModel,
    UserEncoder,
    representation_loss,
)
from spade.training import RepresentationTrainer


def _store(n_users=30, n_items=24, density=0.35, seed=0):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < int(n_users * n_items * density):
        pairs.add((int(rng.integers(0, n_users)), int(rng.integers(0, n_items))))
    u, i = zip(*pairs, strict=True)
    u, i = np.array(u), np.array(i)
    r = rng.integers(1, 6, size=len(u)).astype(np.float32)
    t = np.zeros(len(u), dtype=np.int64)
    raw = RawInteractions(u, i, r, t, "synthetic")
    return build_store(raw, 1, 1)


def _rngs():
    return nnx.Rngs(0)


# --------------------------------------------------------------------------- #
# Component shapes and determinism                                            #
# --------------------------------------------------------------------------- #
def test_encoder_shape_and_determinism():
    enc = UserEncoder(50, latent_dim=8, hidden=[16], rngs=_rngs())
    idx = jnp.array([0, 1, 2, 3])
    z1, z2 = enc(idx), enc(idx)
    assert z1.shape == (4, 8)
    assert jnp.allclose(z1, z2)  # deterministic for fixed params


def test_gate_returns_probabilities_in_unit_interval():
    gate = InteractionGate(latent_dim=8, hidden=[16], rngs=_rngs())
    z_u = jnp.ones((5, 8))
    z_i = jnp.ones((5, 8))
    logits = gate(z_u, z_i)
    probs = gate.probability(z_u, z_i)
    assert logits.shape == (5,)
    assert jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)


def test_decoder_distribution_is_a_simplex():
    dec = RatingDecoder(latent_dim=8, hidden=[16], n_levels=5, rngs=_rngs())
    z = jnp.ones((7, 8))
    logits = dec(z, z)
    dist = dec.distribution(z, z)
    assert logits.shape == (7, 5)
    assert jnp.allclose(dist.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.all(dist >= 0.0)


# --------------------------------------------------------------------------- #
# Rating vocabulary                                                           #
# --------------------------------------------------------------------------- #
def test_rating_vocab_roundtrip():
    vocab = RatingVocab.from_ratings(np.array([3.0, 1.0, 5.0, 1.0, 3.0]))
    assert vocab.n_levels == 3
    assert vocab.values.tolist() == [1.0, 3.0, 5.0]
    idx = vocab.to_index(np.array([5.0, 1.0, 3.0]))
    assert idx.tolist() == [2, 0, 1]
    assert vocab.to_value(idx).tolist() == [5.0, 1.0, 3.0]


def test_rating_vocab_rejects_unknown_value():
    vocab = RatingVocab.from_ratings(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        vocab.to_index(np.array([2.5]))


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #
def test_representation_loss_components_present_and_finite():
    store = _store()
    cfg = ExperimentConfig()
    vocab = RatingVocab.from_ratings(store.ratings)
    model = RepresentationModel(
        store.n_users, store.n_items, vocab.n_levels, cfg.representation, rngs=_rngs()
    )
    u = jnp.asarray(store.user_idx[:16])
    i_pos = jnp.asarray(store.item_idx[:16])
    i_neg = jnp.asarray(np.zeros((16, 5), dtype=np.int64))
    ridx = jnp.asarray(vocab.to_index(store.ratings[:16]))
    total, parts = representation_loss(model, u, i_pos, i_neg, ridx, l2_lambda=1e-5)
    assert set(parts) == {"gate", "rating", "l2", "total"}
    assert jnp.isfinite(total)
    assert parts["gate"] > 0 and parts["rating"] > 0


# --------------------------------------------------------------------------- #
# End-to-end training                                                         #
# --------------------------------------------------------------------------- #
def _tiny_cfg(epochs=15):
    cfg = ExperimentConfig()
    cfg.representation.latent_dim = 8
    cfg.representation.encoder_hidden = [16]
    cfg.representation.gate_hidden = [16]
    cfg.representation.decoder_hidden = [16]
    cfg.representation.batch_size = 32
    cfg.representation.epochs = epochs
    cfg.representation.early_stop_patience = 50  # don't stop early in the short run
    cfg.data.n_neg = 4
    return cfg


def test_training_reduces_loss_and_exports(tmp_path):
    store = _store(seed=1)
    cfg = _tiny_cfg(epochs=20)
    splits = split_store(store, 0.15, 0.15, seed=1)

    trainer = RepresentationTrainer(cfg, splits.train, splits.val).fit()
    assert len(trainer.history) >= 1
    first, last = trainer.history[0]["total"], trainer.history[-1]["total"]
    assert last < first  # joint objective decreases on validation

    z_u, z_i = trainer.model.export_embeddings(store.n_users, store.n_items)
    assert z_u.shape == (store.n_users, cfg.representation.latent_dim)
    assert z_i.shape == (store.n_items, cfg.representation.latent_dim)
    assert np.isfinite(z_u).all() and np.isfinite(z_i).all()

    path = trainer.export(tmp_path, "synthetic")
    loaded = np.load(path)
    assert loaded["z_users"].shape == z_u.shape
    assert loaded["rating_values"].tolist() == trainer.vocab.values.tolist()


def test_early_stopping_triggers():
    store = _store(seed=2)
    cfg = _tiny_cfg(epochs=200)
    cfg.representation.early_stop_patience = 3
    splits = split_store(store, 0.15, 0.15, seed=2)
    trainer = RepresentationTrainer(cfg, splits.train, splits.val).fit()
    # With patience=3 it must stop well before the 200-epoch cap.
    assert len(trainer.history) < 200
