"""The top-level SPADE model tying the stage models together.

SPADE is built in stages that train sequentially, so this umbrella is assembled
by *composition of already-trained submodels* rather than constructed from
scratch: train the :class:`RepresentationModel`, freeze it and fit the
:class:`GenerativeModel` on its exported latents, then attach the synthesis
model. Holding them on one object gives a single handle for inference
and serialization once every stage exists.

The synthesis stage is not implemented yet; ``synthesis`` is reserved and stays
``None`` until Stage III lands, at which point this class composes all three.
"""

from __future__ import annotations

from typing import Any

from flax import nnx

from spade.models.generative import GenerativeModel
from spade.models.representation import RepresentationModel

__all__ = ["SPADE"]


class SPADE(nnx.Module):
    """Composition of the trained stage models (representation + generative).

    Submodels are injected after their own training rather than initialized
    here, which keeps the staged freeze-then-fit workflow intact. ``synthesis``
    is a placeholder for the Stage III synthesis model.
    """

    def __init__(
        self,
        representation: RepresentationModel,
        generative: GenerativeModel,
        synthesis: Any | None = None,
    ) -> None:
        self.representation = representation
        self.generative = generative
        self.synthesis = synthesis

    @property
    def latent_dim(self) -> int:
        return self.generative.latent_dim
