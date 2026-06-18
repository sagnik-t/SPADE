"""Reproducibility + PGPS sanity checks (no training).

These are fast, training-free unit checks of two properties the paper leans on.
The *real* end-to-end determinism and PGPS-vs-random sanity on a trained model are
verified by ``scripts/smoke_pipeline.py``, which is run from the command line;
here we only pin the cheap, deterministic-function guarantees:

* **Determinism.** With PRNG keys threaded explicitly, Stage III synthesis must be
  bit-identical across calls under the same key — a guard against hidden
  global-RNG reliance creeping into model code. (Uses a randomly *initialised*
  model; nothing is trained.)
* **PGPS validity.** PGPS must behave like a geometry-preservation score: a
  trivial *copy* of the real items scores near 1 and well above the random
  baseline, while an independent generator scores materially lower and never
  masquerades as a copy. The random reference itself is fixed at ``k / |I_real|``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.eval.pgps import pgps
from spade.models import GenerativeModel, RatingVocab, RepresentationModel
from spade.synthesis import SynthesisModel


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #
def _cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.seed = 3
    r = cfg.representation
    r.latent_dim, r.encoder_hidden, r.gate_hidden, r.decoder_hidden = 8, [16], [16], [16]
    r.epochs, r.batch_size, r.early_stop_patience = 2, 64, 5
    g = cfg.generative
    g.noise_dim, g.generator_hidden, g.critic_hidden = 8, [16], [16]
    g.epochs, g.n_critic, g.batch_size = 2, 1, 32
    cfg.synthesis.alpha = cfg.synthesis.beta = 2.0
    cfg.synthesis.gamma = 1.0
    return cfg


def _synth_model(cfg: ExperimentConfig, n_users: int, n_items: int) -> SynthesisModel:
    rep = RepresentationModel(n_users, n_items, 5, cfg.representation, rngs=nnx.Rngs(0))
    gen = GenerativeModel(cfg.representation.latent_dim, cfg.generative, rngs=nnx.Rngs(1))
    vocab = RatingVocab(values=np.arange(1, 6, dtype=np.float32))
    return SynthesisModel(
        rep, gen, vocab,
        source_n_users=n_users, source_n_items=n_items,
        source_rho=0.3, cfg=cfg.synthesis,
    )


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #
def test_synthesis_is_deterministic_under_fixed_key():
    cfg = _cfg()
    model = _synth_model(cfg, 32, 20)
    key = jax.random.key(123)

    a = model.synthesize(key)
    b = model.synthesize(key)

    np.testing.assert_array_equal(a.user_idx, b.user_idx)
    np.testing.assert_array_equal(a.item_idx, b.item_idx)
    np.testing.assert_array_equal(a.ratings, b.ratings)


def test_sample_latents_matches_synthesis_key_derivation():
    # The clouds eval re-derives must be reproducible across calls with one key.
    cfg = _cfg()
    model = _synth_model(cfg, 32, 20)
    key = jax.random.key(7)
    zu1, zi1 = model.sample_latents(key)
    zu2, zi2 = model.sample_latents(key)
    np.testing.assert_array_equal(np.asarray(zu1), np.asarray(zu2))
    np.testing.assert_array_equal(np.asarray(zi1), np.asarray(zi2))


# --------------------------------------------------------------------------- #
# PGPS sanity: above random, below/at trivial-copy                            #
# --------------------------------------------------------------------------- #
def test_pgps_trivial_copy_scores_high_and_lifts_over_random():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(80, 8))
    # Synthetic items are an exact copy of the real items.
    res = pgps(real, real.copy(), k=10, metric="cosine")
    assert res.pgps > 0.9          # near-perfect geometry preservation
    assert res.lift > 0.5          # far above the random baseline


def test_pgps_random_baseline_is_k_over_n_and_below_copy():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(120, 8))
    synth = rng.normal(size=(120, 8))  # independent of the real geometry
    res = pgps(real, synth, k=10, metric="cosine")

    # The random reference is the expected overlap of two independent size-k
    # subsets, k / n_real — and it is what `lift` is measured against.
    assert res.random_baseline == pytest.approx(10 / 120)
    assert res.lift == pytest.approx(res.pgps - res.random_baseline)
    # A random-but-in-space generator does not masquerade as a trivial copy.
    assert res.pgps < 0.9
    # Trivial copy on the same reals still clears it comfortably.
    assert pgps(real, real.copy(), k=10).pgps > res.pgps


def test_pgps_copy_beats_random():
    rng = np.random.default_rng(2)
    real = rng.normal(size=(100, 8))
    copy_lift = pgps(real, real.copy(), k=8).lift
    rand_lift = pgps(real, rng.normal(size=(100, 8)), k=8).lift
    assert copy_lift > rand_lift


def test_pgps_clamps_k_and_handles_degenerate_input():
    rng = np.random.default_rng(3)
    real = rng.normal(size=(5, 4))
    res = pgps(real, real.copy(), k=100)  # k clamped to n_real - 1
    assert res.k == 4
    # Empty synthetic set is handled, not crashed.
    assert pgps(real, np.empty((0, 4)), k=3).pgps == 0.0
