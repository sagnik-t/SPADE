"""Stage trainers for SPADE.

Each composite stage model is fit by a :class:`Trainer` subclass sharing one
template loop (epochs, history, W&B logging, early-stop/checkpoint hooks):
:class:`RepresentationTrainer` for Stage I and :class:`GenerativeTrainer` for
Stage II. Generic parameter checkpointing for any ``nnx`` module lives in
:mod:`spade.training.checkpoint`.
"""

from spade.training.base import Trainer
from spade.training.checkpoint import load_params_into, save_params
from spade.training.generative import GenerativeTrainer, load_generative_model
from spade.training.representation import RepresentationTrainer

__all__ = [
    "Trainer",
    "RepresentationTrainer",
    "GenerativeTrainer",
    "load_generative_model",
    "save_params",
    "load_params_into",
]
