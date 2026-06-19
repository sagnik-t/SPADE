"""Generative-stage tests: components, losses, composite models, training.

Runs on CPU JAX. Generators learn a small, well-separated Gaussian target so the
adversarial loop measurably reduces the moment-matching distance within a short
budget while staying fast.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spade.config.configs import ExperimentConfig, GenerativeConfig
from spade.models import (
    SPADE,
    AdversarialPair,
    Critic,
    GenerativeModel,
    JointGenerativeModel,
    LatentGenerator,
    RepresentationModel,
    critic_loss,
    generator_loss,
    gradient_penalty,
    moment_matching_loss,
)
from spade.training import (
    GenerativeTrainer,
    JointGenerativeTrainer,
    load_generative_model,
)


def _rngs(seed=0):
    return nnx.Rngs(seed)


def _target_cloud(n=512, d=6, seed=0):
    """A shifted, anisotropic Gaussian cloud to model."""
    rng = np.random.default_rng(seed)
    mean = np.linspace(1.0, 2.0, d)
    scale = np.linspace(0.3, 1.0, d)
    return (mean + scale * rng.standard_normal((n, d))).astype(np.float32)


def _exp_cfg(epochs=40):
    cfg = ExperimentConfig()
    g = cfg.generative
    g.noise_dim = 8
    g.generator_hidden = [32, 32]
    g.critic_hidden = [32, 32]
    g.n_critic = 3
    g.batch_size = 128
    g.epochs = epochs
    return cfg


# --------------------------------------------------------------------------- #
# Component shapes                                                            #
# --------------------------------------------------------------------------- #
def test_generator_shapes_and_sampling():
    gen = LatentGenerator(8, [16], latent_dim=6, rngs=_rngs())
    noise = gen.sample_noise(jax.random.key(0), 5)
    assert noise.shape == (5, 8)
    assert gen(noise).shape == (5, 6)
    assert gen.sample(jax.random.key(1), 7).shape == (7, 6)


def test_critic_returns_scalar_per_sample():
    critic = Critic(latent_dim=6, hidden=[16], rngs=_rngs())
    assert critic(jnp.ones((9, 6))).shape == (9,)


# --------------------------------------------------------------------------- #
# Losses                                                                      #
# --------------------------------------------------------------------------- #
def test_gradient_penalty_is_nonnegative_and_finite():
    critic = Critic(latent_dim=6, hidden=[16], rngs=_rngs())
    real = jnp.asarray(_target_cloud(32, 6))
    fake = jax.random.normal(jax.random.key(3), (32, 6))
    gp = gradient_penalty(critic, real, fake, jax.random.key(4))
    assert gp.shape == ()
    assert float(gp) >= 0.0 and jnp.isfinite(gp)


def test_moment_matching_zero_for_identical_clouds():
    x = jnp.asarray(_target_cloud(256, 5))
    assert float(moment_matching_loss(x, x)) < 1e-6
    assert float(moment_matching_loss(x + 3.0, x)) > 1.0  # mean term dominates


def test_critic_and_generator_losses_have_parts():
    critic = Critic(latent_dim=6, hidden=[16], rngs=_rngs())
    gen = LatentGenerator(8, [16], latent_dim=6, rngs=_rngs(1))
    real = jnp.asarray(_target_cloud(32, 6))
    fake = gen(gen.sample_noise(jax.random.key(5), 32))
    closs, cparts = critic_loss(critic, real, fake, jax.random.key(6), gp_lambda=10.0)
    gloss, gparts = generator_loss(critic, fake, real, moment_lambda=1.0)
    assert set(cparts) == {"critic_loss", "wasserstein", "gp"}
    assert set(gparts) == {"gen_loss", "gen_adversarial", "moment"}
    assert jnp.isfinite(closs) and jnp.isfinite(gloss)


# --------------------------------------------------------------------------- #
# Composite models                                                            #
# --------------------------------------------------------------------------- #
def test_adversarial_pair_has_generator_and_critic():
    pair = AdversarialPair(6, GenerativeConfig(), rngs=_rngs())
    assert isinstance(pair.generator, LatentGenerator)
    assert isinstance(pair.critic, Critic)
    assert pair.sample(jax.random.key(0), 4).shape == (4, 6)


def test_generative_model_composition_and_sampling():
    model = GenerativeModel(6, GenerativeConfig(), rngs=_rngs())
    assert model.sample_users(jax.random.key(0), 5).shape == (5, 6)
    assert model.sample_items(jax.random.key(1), 7).shape == (7, 6)
    assert model.pair("user") is model.user and model.pair("item") is model.item
    with pytest.raises(ValueError):
        model.pair("nope")


def test_joint_generative_model_samples_halves():
    model = JointGenerativeModel(6, GenerativeConfig(), rngs=_rngs())
    users = model.sample_users(jax.random.key(0), 5)
    items = model.sample_items(jax.random.key(1), 7)
    assert users.shape == (5, 6) and items.shape == (7, 6)
    assert model.joint.generator.latent_dim == 12  # operates over [z_u; z_i]


def test_joint_generative_trainer_reduces_moment_and_roundtrips(tmp_path):
    # Concatenated interaction-level pairs: width is 2 * latent_dim.
    z_pairs = _target_cloud(512, 12, seed=4)
    cfg = _exp_cfg(epochs=40)
    trainer = JointGenerativeTrainer(cfg, z_pairs).fit()

    early = np.mean([row["joint/moment"] for row in trainer.history[:5]])
    late = np.mean([row["joint/moment"] for row in trainer.history[-5:]])
    assert late < early

    cfg.generative.joint = True  # load path must rebuild the joint variant
    path = trainer.export(tmp_path, "synthetic")
    reloaded = load_generative_model(path, cfg, seed=123)
    assert isinstance(reloaded, JointGenerativeModel)
    key = jax.random.key(0)
    a = np.asarray(trainer.model.sample_users(key, 16))
    b = np.asarray(reloaded.sample_users(key, 16))
    assert np.allclose(a, b, atol=1e-6)


def test_spade_umbrella_composes_stage_models():
    cfg = ExperimentConfig()
    cfg.representation.latent_dim = 6
    rep = RepresentationModel(10, 8, 5, cfg.representation, rngs=_rngs())
    gen = GenerativeModel(6, GenerativeConfig(), rngs=_rngs(1))
    spade = SPADE(rep, gen)
    assert spade.representation is rep and spade.generative is gen
    assert spade.latent_dim == 6
    assert spade.synthesis is None  # Stage III not attached yet


# --------------------------------------------------------------------------- #
# End-to-end training                                                         #
# --------------------------------------------------------------------------- #
def test_generative_trainer_reduces_moment_distance():
    z_users = _target_cloud(512, 6, seed=2)
    z_items = _target_cloud(400, 6, seed=3)
    cfg = _exp_cfg(epochs=60)
    trainer = GenerativeTrainer(cfg, z_users, z_items).fit()

    early = np.mean([row["user/moment"] for row in trainer.history[:5]])
    late = np.mean([row["user/moment"] for row in trainer.history[-5:]])
    assert late < early

    gen = np.asarray(trainer.model.sample_users(jax.random.key(99), 512))
    assert np.linalg.norm(gen.mean(0) - z_users.mean(0)) < np.linalg.norm(
        z_users.mean(0)
    )


def test_generative_model_save_load_roundtrip(tmp_path):
    z_users = _target_cloud(256, 6, seed=7)
    z_items = _target_cloud(200, 6, seed=8)
    cfg = _exp_cfg(epochs=5)
    trainer = GenerativeTrainer(cfg, z_users, z_items).fit()

    path = trainer.export(tmp_path, "synthetic")
    reloaded = load_generative_model(path, cfg, seed=123)

    key = jax.random.key(0)
    a = np.asarray(trainer.model.sample_items(key, 16))
    b = np.asarray(reloaded.sample_items(key, 16))
    assert np.allclose(a, b, atol=1e-6)  # identical params -> identical outputs
