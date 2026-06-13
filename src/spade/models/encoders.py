"""Deterministic user and item encoders for Stage I representation learning.

Each encoder is an embedding table followed by an MLP that projects to the
shared ``latent_dim``-dimensional space (``embedding -> MLP -> z``). The mapping
is deterministic: given a fixed set of trained parameters, an id always yields
the same latent vector, which is what lets Stage I freeze and export the real
embedding clouds ``Z_u`` and ``Z_i`` for Stage II to model.

``UserEncoder`` and ``ItemEncoder`` are distinct classes (not one parameterized
module) so the two halves of the model stay independently addressable, matching
the modular-components constraint and the separate generators in Stage II.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

from spade.models.mlp import MLP

__all__ = ["UserEncoder", "ItemEncoder"]


class _Encoder(nnx.Module):
    """Shared implementation: ``nnx.Embed`` lookup then an MLP projection."""

    def __init__(
        self,
        num_entities: int,
        latent_dim: int,
        hidden: Sequence[int],
        *,
        embedding_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        embedding_dim = embedding_dim or latent_dim
        self.embedding = nnx.Embed(num_entities, embedding_dim, rngs=rngs)
        self.project = MLP(embedding_dim, hidden, latent_dim, rngs=rngs)

    def __call__(self, idx: jnp.ndarray) -> jnp.ndarray:
        """Map integer ids ``(batch,)`` to latent vectors ``(batch, latent_dim)``."""
        return self.project(self.embedding(idx))


class UserEncoder(_Encoder):
    """Encode user ids into deterministic ``latent_dim`` vectors."""


class ItemEncoder(_Encoder):
    """Encode item ids into deterministic ``latent_dim`` vectors."""
