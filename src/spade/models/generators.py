"""Latent generator for Stage II distribution modeling (Flax nnx module).

A :class:`LatentGenerator` maps a noise vector ``z ~ N(0, I)`` of size
``noise_dim`` through an MLP to a point in the same ``latent_dim`` space the
frozen Stage I encoders produced. One generator is trained per entity type, so
the two clouds ``Z_u`` and ``Z_i`` are modelled by *independent* generators
``G_u`` and ``G_i`` — a deliberate product-of-experts factorization that trades
joint fidelity for training stability and is recoupled at synthesis time by the
shared gate and decoder.

The module is a plain feed-forward net (no batch-norm) which keeps the WGAN-GP
gradient penalty well defined: the penalty differentiates the critic through its
input, and per-sample statistics would otherwise leak across the batch.

A generator pairs with a :class:`spade.models.Critic` in an
:class:`spade.models.AdversarialPair`; two such pairs compose the Stage II
:class:`spade.models.GenerativeModel`, which is fit by
:class:`spade.training.GenerativeTrainer`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from spade.models.mlp import MLP

__all__ = ["LatentGenerator"]


class LatentGenerator(nnx.Module):
    """MLP mapping noise ``(batch, noise_dim) -> latent (batch, latent_dim)``."""

    def __init__(
        self,
        noise_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.mlp = MLP(noise_dim, hidden, latent_dim, rngs=rngs)

    def __call__(self, noise: jnp.ndarray) -> jnp.ndarray:
        """Map a batch of noise vectors to generated latent vectors."""
        return self.mlp(noise)

    def sample_noise(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` standard-normal noise vectors ``(n, noise_dim)``."""
        return jax.random.normal(key, (n, self.noise_dim))

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Generate ``n`` synthetic latent vectors ``(n, latent_dim)``.

        Convenience for Stage III: splits a single key into noise then maps it
        through the trained generator. Threads the PRNG key explicitly.
        """
        return self(self.sample_noise(key, n))
