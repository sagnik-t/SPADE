"""Composed Stage I representation model.

:class:`RepresentationModel` bundles the four Stage I components — user encoder,
item encoder, interaction gate, rating decoder — into one ``nnx`` module so they
train jointly under a single optimizer. This is composition, not a monolith: each
submodule remains an independent, separately addressable network, and after
training the encoders are frozen and their outputs exported for Stage II.

Convenience methods encode ids and score pairs; the actual loss assembly lives in
:mod:`spade.models.losses` and the training loop in
:class:`spade.training.RepresentationTrainer`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from spade.config.configs import RepresentationConfig
from spade.models.decoder import RatingDecoder
from spade.models.encoders import ItemEncoder, UserEncoder
from spade.models.gate import InteractionGate

__all__ = ["RepresentationModel"]


class RepresentationModel(nnx.Module):
    """User/item encoders + interaction gate + rating decoder, trained jointly."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_rating_levels: int,
        cfg: RepresentationConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.user_encoder = UserEncoder(
            n_users, cfg.latent_dim, cfg.encoder_hidden, rngs=rngs
        )
        self.item_encoder = ItemEncoder(
            n_items, cfg.latent_dim, cfg.encoder_hidden, rngs=rngs
        )
        self.gate = InteractionGate(cfg.latent_dim, cfg.gate_hidden, rngs=rngs)
        self.decoder = RatingDecoder(
            cfg.latent_dim, cfg.decoder_hidden, n_rating_levels, rngs=rngs
        )

    def encode_users(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.user_encoder(u)

    def encode_items(self, i: jnp.ndarray) -> jnp.ndarray:
        return self.item_encoder(i)

    def gate_logits(self, u: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """Interaction logits for paired ids ``(batch,)``."""
        return self.gate(self.encode_users(u), self.encode_items(i))

    def rating_logits(self, u: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """Rating logits ``(batch, n_levels)`` for paired ids."""
        return self.decoder(self.encode_users(u), self.encode_items(i))

    def export_embeddings(
        self, n_users: int, n_items: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run every id through the (frozen) encoders to get ``Z_u`` and ``Z_i``.

        Returns the real latent clouds as NumPy arrays ``(n_users, d)`` and
        ``(n_items, d)`` for Stage II to model and Stage III to draw from.
        """
        z_u = self.encode_users(jnp.arange(n_users))
        z_i = self.encode_items(jnp.arange(n_items))
        return np.asarray(z_u), np.asarray(z_i)
