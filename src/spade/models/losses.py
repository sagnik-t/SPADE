"""Stage I loss terms: gate BCE, rating NLL, and embedding L2 regularization.

The joint objective is

    L = L_gate + L_rating + l2_lambda * ||embeddings||^2

where the gate term is binary cross-entropy over observed positives (label 1)
and uniformly sampled unobserved negatives (label 0); the rating term is the
categorical NLL of the observed ratings only (unobserved pairs have no rating to
predict); and the L2 term regularizes the encoder embedding tables.

Functions take logits and return scalar means so they compose directly inside a
jitted train step. Negatives are sampled outside (in the data layer), keeping
these functions pure and deterministic.
"""

from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx

__all__ = [
    "gate_bce_loss",
    "rating_nll_loss",
    "embedding_l2",
    "stage1_loss",
]


def gate_bce_loss(
    pos_logits: jnp.ndarray,
    neg_logits: jnp.ndarray,
) -> jnp.ndarray:
    """Binary cross-entropy for the gate over positives and sampled negatives.

    ``pos_logits`` are gate logits for observed pairs (target 1); ``neg_logits``
    are logits for sampled unobserved pairs (target 0), shape ``(n_pos, n_neg)``
    or flattened. Returns the mean BCE across all positive and negative scores.
    """
    pos = optax.sigmoid_binary_cross_entropy(pos_logits, jnp.ones_like(pos_logits))
    neg = optax.sigmoid_binary_cross_entropy(neg_logits, jnp.zeros_like(neg_logits))
    # Weight positives and negatives equally per-example by pooling all scores.
    return (pos.sum() + neg.sum()) / (pos.size + neg.size)


def rating_nll_loss(
    rating_logits: jnp.ndarray,
    rating_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Categorical NLL of observed ratings (class indices) under the decoder."""
    return optax.softmax_cross_entropy_with_integer_labels(
        rating_logits, rating_idx
    ).mean()


def embedding_l2(model: nnx.Module) -> jnp.ndarray:
    """Sum of squared L2 norms of the two encoder embedding tables.

    Regularizes representation magnitude (paper's embedding-norm penalty) without
    touching the gate/decoder MLP weights, which are shaped by their own losses.
    """
    tables = [
        model.user_encoder.embedding.embedding[...],
        model.item_encoder.embedding.embedding[...],
    ]
    return sum(jnp.sum(jnp.square(t)) for t in tables)


def stage1_loss(
    model: nnx.Module,
    u: jnp.ndarray,
    i_pos: jnp.ndarray,
    i_neg: jnp.ndarray,
    rating_idx: jnp.ndarray,
    l2_lambda: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Assemble the joint Stage I loss and a dict of its components.

    ``u``/``i_pos`` are aligned observed pairs ``(B,)``; ``i_neg`` is ``(B, n_neg)``
    sampled negative items for the same users; ``rating_idx`` are the observed
    ratings as class indices ``(B,)``. Returns ``(total, parts)`` so the caller
    can log each term.
    """
    z_u = model.encode_users(u)                       # (B, d)
    z_i_pos = model.encode_items(i_pos)               # (B, d)

    pos_logits = model.gate(z_u, z_i_pos)             # (B,)
    # Negatives: broadcast each user's latent across its n_neg sampled items.
    z_i_neg = model.encode_items(i_neg.reshape(-1))   # (B*n_neg, d)
    z_u_rep = jnp.repeat(z_u, i_neg.shape[1], axis=0)  # (B*n_neg, d)
    neg_logits = model.gate(z_u_rep, z_i_neg)         # (B*n_neg,)

    gate = gate_bce_loss(pos_logits, neg_logits)
    rating = rating_nll_loss(model.decoder(z_u, z_i_pos), rating_idx)
    l2 = embedding_l2(model)
    total = gate + rating + l2_lambda * l2
    parts = {"gate": gate, "rating": rating, "l2": l2, "total": total}
    return total, parts
