"""SPADE model components and composite stage models (Flax nnx modules).

Components are modular and independently addressable: the shared :class:`MLP`;
:class:`UserEncoder`/:class:`ItemEncoder`, :class:`InteractionGate`,
:class:`RatingDecoder`; and :class:`LatentGenerator`/:class:`Critic`. These
assemble into the composite stage models — :class:`RepresentationModel` (Stage
I) and :class:`GenerativeModel` (Stage II, via :class:`AdversarialPair`) — which
the :class:`SPADE` umbrella ties together. Training lives in
:mod:`spade.training`; loss functions stay here next to the modules they score.
"""

from spade.models.critics import Critic
from spade.models.decoder import ContinuousRatingDecoder, RatingDecoder, RatingVocab
from spade.models.encoders import ItemEncoder, UserEncoder
from spade.models.gan_losses import (
    critic_loss,
    generator_loss,
    gradient_penalty,
    moment_matching_loss,
)
from spade.models.gate import InteractionGate
from spade.models.generative import (
    AdversarialPair,
    GenerativeModel,
    JointGenerativeModel,
)
from spade.models.generators import LatentGenerator
from spade.models.losses import (
    embedding_l2,
    gate_bce_loss,
    rating_mse_loss,
    rating_nll_loss,
    representation_loss,
)
from spade.models.mlp import MLP
from spade.models.representation import RepresentationModel
from spade.models.spade import SPADE

__all__ = [
    "MLP",
    "UserEncoder",
    "ItemEncoder",
    "InteractionGate",
    "RatingDecoder",
    "ContinuousRatingDecoder",
    "RatingVocab",
    "RepresentationModel",
    "gate_bce_loss",
    "rating_nll_loss",
    "rating_mse_loss",
    "embedding_l2",
    "representation_loss",
    "LatentGenerator",
    "Critic",
    "AdversarialPair",
    "GenerativeModel",
    "JointGenerativeModel",
    "SPADE",
    "gradient_penalty",
    "critic_loss",
    "moment_matching_loss",
    "generator_loss",
]
