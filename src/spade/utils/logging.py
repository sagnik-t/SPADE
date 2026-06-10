"""Logging helpers: a stdlib console logger and a thin Weights & Biases wrapper.

The W&B wrapper degrades gracefully: if wandb is not installed or the run mode
is ``disabled``, calls become no-ops so training code stays unconditional.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, cast

__all__ = ["get_logger", "WandbRun", "init_wandb"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "spade", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def _to_dict(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        return config
    return {}


class WandbRun:
    """Minimal context-managed run handle that no-ops when W&B is unavailable."""

    def __init__(self, run: Any | None) -> None:
        self._run = run

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def __enter__(self) -> "WandbRun":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()


def init_wandb(
    config: Any,
    project: str = "spade",
    name: str | None = None,
    mode: str = "online",
) -> WandbRun:
    """Initialize a W&B run, returning a :class:`WandbRun` (possibly inactive)."""
    if mode == "disabled":
        return WandbRun(None)
    try:
        import wandb
    except ImportError:
        get_logger().warning("wandb not installed; metric logging disabled.")
        return WandbRun(None)

    run = wandb.init(
        project=project, name=name, mode=cast(Any, mode), config=_to_dict(config)
    )
    return WandbRun(run)
