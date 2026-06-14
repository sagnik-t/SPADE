"""WGAN critic (discriminator) for Stage II distribution modeling.

A :class:`Critic` is an MLP scoring a latent vector with a single unbounded real
value. Under the Wasserstein GAN objective the critic is not a probabilistic
classifier but a 1-Lipschitz potential whose expected score gap between the real
and generated clouds estimates their Wasserstein-1 distance. Lipschitzness is
enforced softly by the gradient penalty in :mod:`spade.models.gan_losses` rather
than by weight clipping, so the critic is a plain unconstrained MLP with no
output activation and no normalization layers (which would break the per-sample
gradient penalty).

One critic is paired with a :class:`spade.models.LatentGenerator` in an
:class:`spade.models.AdversarialPair`; the user and item pairs that compose the
:class:`spade.models.GenerativeModel` are entirely independent. The critic is
used only during training (by :class:`spade.training.GenerativeTrainer`) and
discarded at synthesis, which needs only the generators.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from spade.models.mlp import MLP

__all__ = ["Critic"]


class Critic(nnx.Module):
    """MLP scoring a latent vector ``(batch, latent_dim) -> score (batch,)``."""

    def __init__(
        self,
        latent_dim: int,
        hidden: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.mlp = MLP(latent_dim, hidden, 1, rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Return scalar critic scores ``(batch,)`` for a batch of latents."""
        return self.mlp(z).squeeze(-1)
