"""Interaction gate: probability that a (user, item) pair is observed.

The gate models sparsity explicitly. It consumes the concatenation
``[z_u; z_i]`` and emits a single logit; ``sigmoid`` of that logit is the
Bernoulli probability that the pair interacts. Stage III samples this Bernoulli
directly to decide which synthetic pairs exist, so the gate must produce a
calibrated probability — hence it is trained with binary cross-entropy over
observed positives and uniformly sampled unobserved negatives (negative sampling
avoids the MNAR bias of treating every unobserved pair as a true negative).

``__call__`` returns the raw logit for numerically stable BCE-with-logits;
:meth:`probability` wraps it in a sigmoid for inference and Stage III sampling.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from spade.models.mlp import MLP

__all__ = ["InteractionGate"]


class InteractionGate(nnx.Module):
    """MLP scoring ``[z_u; z_i] -> logit`` for P(interaction)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.mlp = MLP(2 * latent_dim, hidden, 1, rngs=rngs)

    def __call__(self, z_u: jnp.ndarray, z_i: jnp.ndarray) -> jnp.ndarray:
        """Return interaction logits ``(batch,)`` for paired latents."""
        x = jnp.concatenate([z_u, z_i], axis=-1)
        return self.mlp(x).squeeze(-1)

    def probability(self, z_u: jnp.ndarray, z_i: jnp.ndarray) -> jnp.ndarray:
        """Return the Bernoulli interaction probability ``(batch,)`` in [0, 1]."""
        return nnx.sigmoid(self(z_u, z_i))
