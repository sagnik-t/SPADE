"""Shared multilayer perceptron used by the encoders, gate, and decoder.

A single small MLP keeps the Stage I components consistent (same activation,
initialization path, and PRNG threading) while remaining a proper Flax ``nnx``
module rather than a monolithic network. Hidden layers use ReLU; the final layer
is linear so callers can apply their own head (identity latent, sigmoid logit, or
softmax logits).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx

__all__ = ["MLP"]


class MLP(nnx.Module):
    """A ReLU MLP mapping ``in_features -> hidden... -> out_features``.

    With no hidden layers this is a single linear map. Activations are applied
    between layers only, never on the output, so the module is reusable as a
    latent projector or a logit head.
    """

    def __init__(
        self,
        in_features: int,
        hidden: Sequence[int],
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        dims = [in_features, *hidden, out_features]
        # nnx requires submodule containers to be explicit data nodes.
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = nnx.relu(x)
        return x
