"""Stage I representation-learning components (Flax nnx modules) and training.

Components are modular and independently addressable: shared :class:`MLP`,
:class:`UserEncoder`/:class:`ItemEncoder`, :class:`InteractionGate`,
:class:`RatingDecoder`. :class:`RepresentationModel` composes them for joint
training (see :func:`train_stage1`), after which encoders are frozen and their
outputs exported via :func:`export_stage1`.
"""

from spade.models.decoder import RatingDecoder, RatingVocab
from spade.models.encoders import ItemEncoder, UserEncoder
from spade.models.gate import InteractionGate
from spade.models.losses import (
    embedding_l2,
    gate_bce_loss,
    rating_nll_loss,
    stage1_loss,
)
from spade.models.mlp import MLP
from spade.models.representation import RepresentationModel
from spade.models.train_stage1 import TrainState, export_stage1, train_stage1

__all__ = [
    "MLP",
    "UserEncoder",
    "ItemEncoder",
    "InteractionGate",
    "RatingDecoder",
    "RatingVocab",
    "RepresentationModel",
    "gate_bce_loss",
    "rating_nll_loss",
    "embedding_l2",
    "stage1_loss",
    "TrainState",
    "train_stage1",
    "export_stage1",
]
