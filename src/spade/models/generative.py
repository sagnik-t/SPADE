"""Generative model for latent distribution modeling (Stage II composite).

This is the second of SPADE's composite stage models: a full model assembled
from smaller components, mirroring :class:`spade.models.RepresentationModel`. It
learns to reproduce the frozen Stage I latent clouds so Stage III can draw novel
synthetic entities.

* :class:`AdversarialPair` bundles one :class:`LatentGenerator` with its
  :class:`Critic` — a single WGAN-GP for one entity type.
* :class:`GenerativeModel` composes two *independent* pairs (users and items).
  The independence is deliberate (a product-of-experts factorization that trades
  joint fidelity for training stability); the two clouds are recoupled only at
  synthesis time by the shared Stage I gate and decoder.

The module owns the critics as well as the generators so the whole adversarial
system is one addressable object during training; synthesis uses only the
generator halves via :meth:`sample_users` / :meth:`sample_items`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from spade.config.configs import GenerativeConfig
from spade.models.critics import Critic
from spade.models.generators import LatentGenerator

__all__ = ["AdversarialPair", "GenerativeModel"]


class AdversarialPair(nnx.Module):
    """One WGAN-GP: a latent generator paired with its critic."""

    def __init__(
        self,
        latent_dim: int,
        cfg: GenerativeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.generator = LatentGenerator(
            cfg.noise_dim, cfg.generator_hidden, latent_dim, rngs=rngs
        )
        self.critic = Critic(latent_dim, cfg.critic_hidden, rngs=rngs)

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` synthetic latent vectors from the generator."""
        return self.generator.sample(key, n)


class GenerativeModel(nnx.Module):
    """Two independent WGAN-GP pairs modeling the user and item latent clouds."""

    def __init__(
        self,
        latent_dim: int,
        cfg: GenerativeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.user = AdversarialPair(latent_dim, cfg, rngs=rngs)
        self.item = AdversarialPair(latent_dim, cfg, rngs=rngs)

    def pair(self, entity: str) -> AdversarialPair:
        """Return the ``"user"`` or ``"item"`` adversarial pair."""
        if entity not in ("user", "item"):
            raise ValueError(f"unknown entity {entity!r}; expected 'user' or 'item'")
        return self.user if entity == "user" else self.item

    def sample_users(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Generate ``n`` synthetic user latent vectors ``(n, latent_dim)``."""
        return self.user.sample(key, n)

    def sample_items(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Generate ``n`` synthetic item latent vectors ``(n, latent_dim)``."""
        return self.item.sample(key, n)
