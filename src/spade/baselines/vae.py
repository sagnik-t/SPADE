"""VAE baseline — a variational autoencoder trained directly on interaction tuples.

Unlike GANRS (which embeds with a separate DeepMF), this baseline learns its own
user/item embeddings jointly with the autoencoder: each interaction is encoded as
``[emb_u, emb_i, one_hot(rating)]``, pushed through a Gaussian VAE, and
reconstructed (squared error on the embedding blocks, cross-entropy on the rating
block, plus a ``beta``-weighted KL to the standard normal prior). New interactions
are produced by sampling the latent prior, decoding to tuples, and recovering a
discrete universe with the same K-Means step as GANRS.

The learned embedding tables are the real coordinates and the recovered cluster
centers the synthetic ones for the geometry metrics.
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
    target_nnz,
)
from spade.baselines.tuples import TupleCodec, recover_universe
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.models.mlp import MLP
from spade.utils import get_logger

__all__ = ["VAEModel", "VAEGenerator"]

logger = get_logger(__name__)


class VAEModel(nnx.Module):
    """Tuple VAE with jointly learned user/item embeddings."""

    def __init__(
        self, n_users, n_items, dim, latent, hidden, n_levels, *, rngs: nnx.Rngs
    ) -> None:
        self.dim = dim
        self.latent = latent
        self.n_levels = n_levels
        self.user_emb = nnx.Embed(n_users, dim, rngs=rngs)
        self.item_emb = nnx.Embed(n_items, dim, rngs=rngs)
        tuple_dim = 2 * dim + n_levels
        self.encoder = MLP(tuple_dim, hidden, 2 * latent, rngs=rngs)
        self.decoder = MLP(latent, list(reversed(list(hidden))), tuple_dim, rngs=rngs)

    def tuple_of(self, u, i, rating_oh) -> jnp.ndarray:
        return jnp.concatenate([self.user_emb(u), self.item_emb(i), rating_oh], axis=1)

    def encode(self, x) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = self.encoder(x)
        return h[:, : self.latent], h[:, self.latent :]

    def decode(self, z) -> jnp.ndarray:
        return self.decoder(z)


@nnx.jit
def _vae_step(model, optimizer, u, i, rating_oh, rating_idx, eps, beta):
    def loss_fn(m):
        x = m.tuple_of(u, i, rating_oh)
        mu, logvar = m.encode(x)
        z = mu + jnp.exp(0.5 * logvar) * eps
        xhat = m.decode(z)
        d2 = 2 * m.dim
        recon = jnp.mean((xhat[:, :d2] - x[:, :d2]) ** 2)
        rating = optax.softmax_cross_entropy_with_integer_labels(
            xhat[:, d2:], rating_idx
        ).mean()
        kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1))
        return recon + rating + beta * kl

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


class VAEGenerator(BaselineGenerator):
    """VAE over interaction tuples with K-Means identifier recovery."""

    name = "vae"

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None:
        bcfg = cfg.baselines
        self.n_users, self.n_items = synthetic_sizes(
            train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
        )
        self.kmeans_iters = bcfg.kmeans_iters
        # Density-scaled tuple budget (see GANRS): oversample target_nnz and
        # truncate after recovery so the VAE hits the target density on any dataset.
        self.target_nnz = target_nnz(train.rho, self.n_users, self.n_items)
        self.n_generate = max(1, int(np.ceil(bcfg.gen_oversample * self.target_nnz)))
        self.beta = bcfg.vae_beta
        self.batch_size = 1024
        self.epochs = bcfg.vae_epochs

        rating_values = np.unique(train.ratings)
        self.codec = TupleCodec(bcfg.vae_dim, rating_values)
        self._u = train.user_idx
        self._i = train.item_idx
        self._rating_idx = self.codec.rating_index(train.ratings.astype(np.float32))

        self.model = VAEModel(
            train.n_users, train.n_items, bcfg.vae_dim, bcfg.vae_latent,
            bcfg.vae_hidden, self.codec.n_levels, rngs=nnx.Rngs(cfg.seed),
        )
        self.optimizer = nnx.Optimizer(self.model, optax.adam(bcfg.vae_lr), wrt=nnx.Param)
        self.latent_dim = bcfg.vae_latent

    def _train(self, key: jax.Array) -> None:
        n = self._u.shape[0]
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            last = 0.0
            for start in range(0, n, self.batch_size):
                sl = perm[start : start + self.batch_size]
                idx = self._rating_idx[sl]
                rating_oh = self.codec.one_hot(idx)
                key, sub = jax.random.split(key)
                eps = jax.random.normal(sub, (sl.shape[0], self.latent_dim))
                last = float(
                    _vae_step(
                        self.model, self.optimizer,
                        jnp.asarray(self._u[sl]), jnp.asarray(self._i[sl]),
                        jnp.asarray(rating_oh), jnp.asarray(idx),
                        eps, self.beta,
                    )
                )
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info("vae | epoch %d | loss %.4f", epoch, last)

    def _sample_tuples(self, key: jax.Array) -> np.ndarray:
        out = np.empty((self.n_generate, self.codec.tuple_dim), dtype=np.float32)
        bs = 4096
        for start in range(0, self.n_generate, bs):
            k = min(bs, self.n_generate - start)
            key, sub = jax.random.split(key)
            z = jax.random.normal(sub, (k, self.latent_dim))
            out[start : start + k] = np.asarray(self.model.decode(z))
        return out

    def generate(self, key: jax.Array) -> GeneratorOutput:
        k_train, k_gen, k_cluster = jax.random.split(key, 3)
        self._train(k_train)
        tuples = self._sample_tuples(k_gen)
        p_fake, q_fake, ratings = self.codec.split(tuples)

        rng = rng_from_key(k_cluster)
        dataset, user_centers, item_centers = recover_universe(
            p_fake, q_fake, ratings, self.n_users, self.n_items, self.kmeans_iters,
            rng, max_nnz=self.target_nnz,
        )
        latents = LatentBundle(
            real_users=np.asarray(self.model.user_emb.embedding[...]),
            real_items=np.asarray(self.model.item_emb.embedding[...]),
            synth_users=user_centers, synth_items=item_centers,
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=latents)
