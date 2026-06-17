"""Ablation registry — named config transforms for the diagnostic experiments.

Each :class:`Ablation` is a name plus a pure function that returns a *modified
copy* of an :class:`ExperimentConfig` (the original is never mutated, so the same
base config can spawn every ablation). The harness runs the full generator matrix
under each registered ablation and tags the results with its name.

Implemented here are the config/synthesis-level ablations:

* the **expansion-ratio sweep** ``alpha = beta in {1.5, 2, 3}`` (2 == base);
* **latent-reg off** — drop the Stage II moment-matching term (``moment_lambda=0``);
* **gating off** — bypass the interaction gate at synthesis.

Two further ablations from the paper — **joint-vs-factorized** Stage II and
**discrete-vs-continuous** decoding — require new *model architectures*, not config
toggles, and are deferred to dedicated work; they are registered as explicit
``NotImplementedError`` stubs so the gap is visible rather than silent.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from spade.config.configs import ExperimentConfig

__all__ = ["Ablation", "ABLATIONS", "DEFERRED_ABLATIONS", "get_ablation"]


@dataclass(frozen=True)
class Ablation:
    """A named transform producing a modified copy of an experiment config."""

    name: str
    apply: Callable[[ExperimentConfig], ExperimentConfig]


def _with(cfg: ExperimentConfig, mutate: Callable[[ExperimentConfig], None]) -> ExperimentConfig:
    clone = copy.deepcopy(cfg)
    mutate(clone)
    return clone


def _set_alpha(value: float) -> Ablation:
    def mutate(c: ExperimentConfig) -> None:
        c.synthesis.alpha = value
        c.synthesis.beta = value

    return Ablation(f"alpha_{value}", lambda cfg: _with(cfg, mutate))


def _latent_reg_off() -> Ablation:
    def mutate(c: ExperimentConfig) -> None:
        c.generative.moment_lambda = 0.0

    return Ablation("latent_reg_off", lambda cfg: _with(cfg, mutate))


def _gating_off() -> Ablation:
    def mutate(c: ExperimentConfig) -> None:
        c.synthesis.gating = False

    return Ablation("gating_off", lambda cfg: _with(cfg, mutate))


def _deferred(name: str, reason: str) -> Ablation:
    def apply(_: ExperimentConfig) -> ExperimentConfig:
        raise NotImplementedError(
            f"ablation {name!r} needs a new model variant ({reason}); deferred."
        )

    return Ablation(name, apply)


_REGISTERED = [
    Ablation("base", copy.deepcopy),
    _set_alpha(1.5),
    _set_alpha(3.0),
    _latent_reg_off(),
    _gating_off(),
]

ABLATIONS: dict[str, Ablation] = {a.name: a for a in _REGISTERED}

# Visible placeholders for the structural ablations that need model surgery.
DEFERRED_ABLATIONS: dict[str, Ablation] = {
    a.name: a
    for a in [
        _deferred("joint_generator", "joint Stage II generator over (z_u, z_i)"),
        _deferred("continuous_decoder", "continuous-rating regression decoder"),
    ]
}


def get_ablation(name: str) -> Ablation:
    """Look up a registered ablation, with a clear error for deferred ones."""
    if name in ABLATIONS:
        return ABLATIONS[name]
    if name in DEFERRED_ABLATIONS:
        return DEFERRED_ABLATIONS[name]
    raise KeyError(f"unknown ablation {name!r}; have {sorted(ABLATIONS)}")
