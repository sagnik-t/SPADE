"""Abstract base for SPADE's stage trainers.

Each composite stage model (representation, generative, ...) is fit by its own
:class:`Trainer` subclass. The base owns everything the stages share — PRNG-key
threading, per-epoch history, optional W&B logging, and a template ``fit`` loop
with hooks — so subclasses only implement the parts that genuinely differ:
:meth:`train_epoch` (one pass of optimization returning a metrics dict) and
:meth:`export` (persisting the trained artifacts).

The ``fit`` template runs ``num_epochs`` epochs, recording and logging the
metrics from each, and stops early when :meth:`should_stop` says so (default:
never). :meth:`on_fit_end` lets a subclass finalize — e.g. restore the best
validation parameters. Trainers are plain classes, not ``nnx`` modules: they
orchestrate training but hold no differentiable state themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np

from spade.utils import get_logger

__all__ = ["Trainer"]


class Trainer(ABC):
    """Template-method base implementing the shared epoch/logging machinery."""

    def __init__(self, *, seed: int, run: Any | None = None) -> None:
        self.seed = seed
        self.run = run
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.key(seed)
        self.history: list[dict[str, float]] = []
        self.logger = get_logger(type(self).__name__)

    # -- key threading ----------------------------------------------------- #
    def next_keys(self, n: int) -> tuple[jax.Array, ...]:
        """Split and advance the trainer's PRNG key, returning ``n`` fresh keys."""
        self.key, *keys = jax.random.split(self.key, n + 1)
        return tuple(keys)

    # -- logging ----------------------------------------------------------- #
    def record(self, row: dict[str, float]) -> None:
        """Append a metrics row to history and forward it to W&B if present."""
        self.history.append(row)
        if self.run is not None:
            self.run.log(row, step=int(row.get("epoch", len(self.history) - 1)))

    # -- template fit loop ------------------------------------------------- #
    def fit(self) -> Self:
        """Run the training loop: ``num_epochs`` epochs with early-stop hook."""
        self.on_fit_start()
        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)
            row = {"epoch": float(epoch), **metrics}
            self.record(row)
            if self.should_stop(epoch, metrics):
                self.logger.info("stopping early at epoch %d", epoch)
                break
        self.on_fit_end()
        return self

    # -- hooks (overridable) ----------------------------------------------- #
    def on_fit_start(self) -> None:  # noqa: B027 - intentional no-op hook
        """Called once before the first epoch (default: no-op)."""

    def on_fit_end(self) -> None:  # noqa: B027 - intentional no-op hook
        """Called once after the loop (default: no-op)."""

    def should_stop(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Return ``True`` to halt training early (default: never)."""
        return False

    # -- required of every stage ------------------------------------------- #
    @property
    @abstractmethod
    def num_epochs(self) -> int:
        """Total number of epochs to run."""

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one epoch of optimization and return its scalar metrics."""

    @abstractmethod
    def export(self, output_dir: str | Path, dataset: str) -> Any:
        """Persist the trained artifacts for the next stage."""
