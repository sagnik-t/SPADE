"""Baseline tests: every generator yields a valid synthetic dataset.

Runs on CPU JAX. Networks train for 1–2 epochs on tiny stores — the tests check
the common-interface contract (universe sizes, index ranges, rating support,
pair uniqueness, latent-bundle shapes), not sample quality. The geometry path is
exercised end-to-end on a baseline's latent bundle.
"""

import jax
import numpy as np
import pytest
from flax import nnx

from spade.baselines import (
    BASELINE_REGISTRY,
    GeneratorOutput,
    SpadeGenerator,
    synthetic_sizes,
)
from spade.config.configs import ExperimentConfig
from spade.data.interactions import IndexMap, InteractionStore
from spade.eval import geometry_metrics
from spade.models import GenerativeModel, RatingVocab, RepresentationModel
from spade.synthesis import SyntheticDataset


def _store(n_users=15, n_items=12, per_user=6, seed=0) -> InteractionStore:
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


def _cfg(seed=0) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.seed = seed
    cfg.synthesis.alpha = 2.0
    cfg.synthesis.beta = 2.0
    b = cfg.baselines
    b.noise_mf_dim = 8
    b.noise_mf_epochs = 1
    b.deepmf_dim = 8
    b.deepmf_epochs = 1
    b.gan_noise_dim = 8
    b.gan_hidden = [16]
    b.gan_epochs = 1
    b.vae_dim = 8
    b.vae_latent = 4
    b.vae_hidden = [16]
    b.vae_epochs = 1
    b.kmeans_iters = 5
    b.gen_oversample = 1.0
    return cfg


def _assert_valid(output: GeneratorOutput, train, cfg, *, expect_latents: bool):
    exp_u, exp_i = synthetic_sizes(
        train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
    )
    ds = output.dataset
    assert isinstance(ds, SyntheticDataset)
    assert ds.n_users == exp_u and ds.n_items == exp_i
    assert ds.user_idx.shape == ds.item_idx.shape == ds.ratings.shape
    vocab = set(np.unique(train.ratings).tolist())
    if ds.nnz:
        assert ds.user_idx.min() >= 0 and ds.user_idx.max() < exp_u
        assert ds.item_idx.min() >= 0 and ds.item_idx.max() < exp_i
        assert set(np.unique(ds.ratings).tolist()) <= vocab
        flat = ds.user_idx.astype(np.int64) * exp_i + ds.item_idx
        assert np.unique(flat).shape[0] == ds.nnz  # no duplicate pairs
        assert ds.density <= 1.0

    if expect_latents:
        lb = output.latents
        assert lb is not None
        assert lb.real_users.shape[0] == train.n_users
        assert lb.real_items.shape[0] == train.n_items
        assert lb.synth_users.shape[0] == exp_u
        assert lb.synth_items.shape[0] == exp_i
    else:
        assert output.latents is None


@pytest.mark.parametrize("name", ["random", "marginal"])
def test_structure_free_baselines(name):
    cfg = _cfg()
    train = _store(seed=1)
    gen = BASELINE_REGISTRY[name](train, cfg)
    out = gen.generate(jax.random.key(0))
    _assert_valid(out, train, cfg, expect_latents=False)


@pytest.mark.parametrize("name", ["noise_mf", "ganrs", "vae"])
def test_latent_baselines(name):
    cfg = _cfg()
    train = _store(seed=2)
    gen = BASELINE_REGISTRY[name](train, cfg)
    out = gen.generate(jax.random.key(0))
    _assert_valid(out, train, cfg, expect_latents=True)


def test_registry_has_five_baselines():
    assert set(BASELINE_REGISTRY) == {
        "random", "marginal", "noise_mf", "ganrs", "vae"
    }


def test_random_density_matches_rho():
    cfg = _cfg()
    train = _store(seed=3)
    out = BASELINE_REGISTRY["random"](train, cfg).generate(jax.random.key(1))
    # uniform sampling targets rho exactly (subject to integer rounding)
    assert out.dataset.density == pytest.approx(train.rho, rel=0.15)


def test_baseline_latents_feed_geometry_metrics():
    cfg = _cfg()
    cfg.eval.ref_dim = 8
    cfg.eval.ref_hidden = [16]
    cfg.eval.ref_epochs = 2
    cfg.eval.neighbor_k = 3
    train = _store(seed=4)
    out = BASELINE_REGISTRY["noise_mf"](train, cfg).generate(jax.random.key(2))
    lb = out.latents
    assert lb is not None
    geo = geometry_metrics(
        train, cfg.eval, lb.real_users, lb.real_items,
        lb.synth_users, lb.synth_items, seed=cfg.seed,
    )
    assert set(geo) == {"mf", "ncf"}
    for metrics in geo.values():
        assert "pgps" in metrics and "ndi" in metrics


def test_spade_adapter_conforms_to_interface():
    cfg = _cfg()
    n_users, n_items, n_levels = 15, 12, 5
    rep = RepresentationModel(n_users, n_items, n_levels, cfg.representation, rngs=nnx.Rngs(0))
    gen = GenerativeModel(cfg.representation.latent_dim, cfg.generative, rngs=nnx.Rngs(1))
    vocab = RatingVocab(values=np.arange(1, n_levels + 1, dtype=np.float32))
    train = _store(n_users=n_users, n_items=n_items, seed=5)
    spade = SpadeGenerator(rep, gen, vocab, train, cfg)
    out = spade.generate(jax.random.key(0))
    _assert_valid(out, train, cfg, expect_latents=True)
