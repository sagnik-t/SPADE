"""GANRS baseline — the direct predecessor, reimplemented faithfully.

Pipeline (Fernández et al.'s GANRS): a DeepMF model embeds real users and items;
each observed interaction becomes a tuple ``[p_u, q_i, one_hot(rating)]``; a
*vanilla* GAN (binary cross-entropy, not Wasserstein) learns to generate such
tuples; finally K-Means recovers discrete identifiers by clustering the generated
user fragments into ``U'`` synthetic users and the item fragments into ``I'``
synthetic items, with each generated tuple contributing one (user, item, rating)
triple.

The generator/discriminator reuse SPADE's :class:`LatentGenerator` and
:class:`Critic` MLP backbones (the critic emits an unbounded logit, exactly what
BCE-with-logits needs). The DeepMF embeddings and recovered cluster centers form
the latent bundle for the geometry metrics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spade.baselines.base import (
    BaselineGenerator,
    GeneratorOutput,
    LatentBundle,
    rng_from_key,
    synthetic_sizes,
)
from spade.baselines.tuples import TupleCodec, recover_universe
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.eval.reference import train_mf
from spade.models.critics import Critic
from spade.models.generators import LatentGenerator
from spade.utils import get_logger

__all__ = ["GANRSGenerator"]

logger = get_logger(__name__)


@nnx.jit
def _disc_step(disc, gen, d_opt, real, noise):
    fake = gen(noise)  # constant w.r.t. the discriminator update

    def loss_fn(d):
        real_l = d(real)
        fake_l = d(fake)
        return (
            optax.sigmoid_binary_cross_entropy(real_l, jnp.ones_like(real_l)).mean()
            + optax.sigmoid_binary_cross_entropy(fake_l, jnp.zeros_like(fake_l)).mean()
        )

    loss, grads = nnx.value_and_grad(loss_fn)(disc)
    d_opt.update(disc, grads)
    return loss


@nnx.jit
def _gen_step(gen, disc, g_opt, noise):
    def loss_fn(g):
        logits = disc(g(noise))
        return optax.sigmoid_binary_cross_entropy(logits, jnp.ones_like(logits)).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(gen)
    g_opt.update(gen, grads)
    return loss


class GANRSGenerator(BaselineGenerator):
    """DeepMF + vanilla GAN over interaction tuples + K-Means recovery."""

    name = "ganrs"

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        bcfg = cfg.baselines
        self.n_users, self.n_items = synthetic_sizes(
            train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
        )
        self.kmeans_iters = bcfg.kmeans_iters
        self.n_generate = bcfg.gan_n_generate

        mf = train_mf(
            train, dim=bcfg.deepmf_dim, epochs=bcfg.deepmf_epochs, seed=cfg.seed
        )
        self.p = mf.user_table()
        self.q = mf.item_table()
        rating_values = np.unique(train.ratings)
        self.codec = TupleCodec(bcfg.deepmf_dim, rating_values)

        rating_idx = self.codec.rating_index(train.ratings.astype(np.float32))
        self.real_tuples = self.codec.encode(
            self.p[train.user_idx], self.q[train.item_idx], rating_idx
        )

        self.gen = LatentGenerator(
            bcfg.gan_noise_dim, bcfg.gan_hidden, self.codec.tuple_dim,
            rngs=nnx.Rngs(cfg.seed),
        )
        self.disc = Critic(self.codec.tuple_dim, bcfg.gan_hidden, rngs=nnx.Rngs(cfg.seed + 1))
        self.g_opt = nnx.Optimizer(self.gen, optax.adam(bcfg.gan_lr, b1=0.5), wrt=nnx.Param)
        self.d_opt = nnx.Optimizer(self.disc, optax.adam(bcfg.gan_lr, b1=0.5), wrt=nnx.Param)
        self.batch_size = bcfg.gan_batch_size
        self.epochs = bcfg.gan_epochs

    def _train(self, key: jax.Array) -> None:
        real = jnp.asarray(self.real_tuples)
        n = real.shape[0]
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        noise_dim = self.gen.noise_dim
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            last = 0.0
            for start in range(0, n, self.batch_size):
                sl = perm[start : start + self.batch_size]
                key, kd, kg = jax.random.split(key, 3)
                batch = real[sl]
                noise_d = jax.random.normal(kd, (batch.shape[0], noise_dim))
                _disc_step(self.disc, self.gen, self.d_opt, batch, noise_d)
                noise_g = jax.random.normal(kg, (batch.shape[0], noise_dim))
                last = float(_gen_step(self.gen, self.disc, self.g_opt, noise_g))
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("ganrs gan | epoch %d | g_loss %.4f", epoch, last)

    def _sample_tuples(self, key: jax.Array) -> np.ndarray:
        out = np.empty((self.n_generate, self.codec.tuple_dim), dtype=np.float32)
        noise_dim = self.gen.noise_dim
        bs = 4096
        for start in range(0, self.n_generate, bs):
            k = min(bs, self.n_generate - start)
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, (k, noise_dim))
            out[start : start + k] = np.asarray(self.gen(noise))
        return out

    def generate(self, key: jax.Array) -> GeneratorOutput:
        k_train, k_gen, k_cluster = jax.random.split(key, 3)
        self._train(k_train)
        tuples = self._sample_tuples(k_gen)
        p_fake, q_fake, ratings = self.codec.split(tuples)

        rng = rng_from_key(k_cluster)
        dataset, user_centers, item_centers = recover_universe(
            p_fake, q_fake, ratings, self.n_users, self.n_items, self.kmeans_iters, rng
        )
        latents = LatentBundle(
            real_users=self.p, real_items=self.q,
            synth_users=user_centers, synth_items=item_centers,
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=latents)
