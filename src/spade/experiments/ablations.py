"""Ablation registry — named config transforms for the diagnostic experiments.

Each :class:`Ablation` is a name plus a pure function that returns a *modified
copy* of an :class:`ExperimentConfig` (the original is never mutated, so the same
base config can spawn every ablation). The harness runs the full generator matrix
under each registered ablation and tags the results with its name.

Every ablation is a single config toggle; the model variants they select live in
the model/training layers and are keyed into their own stage caches by the config
signature, so no ablation needs bespoke orchestration here:

* the **expansion-ratio sweep** ``alpha = beta in {1.5, 2, 3}`` (2 == base);
* **latent-reg off** — drop the Stage II moment-matching term (``moment_lambda=0``);
* **gating off** — bypass the interaction gate at synthesis;
* **joint generator** — replace the two factorized Stage II generators with a
  single joint WGAN-GP over ``[z_u; z_i]`` (``generative.joint``);
* **continuous decoder** — replace the categorical rating decoder with a
  regressor plus post-hoc snapping (``representation.continuous_decoder``).

``DEFERRED_ABLATIONS`` is retained (now empty) for backward compatibility.
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


def _joint_generator() -> Ablation:
    def mutate(c: ExperimentConfig) -> None:
        c.generative.joint = True

    return Ablation("joint_generator", lambda cfg: _with(cfg, mutate))


def _continuous_decoder() -> Ablation:
    def mutate(c: ExperimentConfig) -> None:
        c.representation.continuous_decoder = True

    return Ablation("continuous_decoder", lambda cfg: _with(cfg, mutate))


_REGISTERED = [
    Ablation("base", copy.deepcopy),
    _set_alpha(1.5),
    _set_alpha(3.0),
    _latent_reg_off(),
    _gating_off(),
    _joint_generator(),
    _continuous_decoder(),
]

ABLATIONS: dict[str, Ablation] = {a.name: a for a in _REGISTERED}

# Both structural ablations are now implemented as config toggles; the mapping is
# kept (empty) so existing imports of ``DEFERRED_ABLATIONS`` keep working.
DEFERRED_ABLATIONS: dict[str, Ablation] = {}


def get_ablation(name: str) -> Ablation:
    """Look up a registered ablation, with a clear error for deferred ones."""
    if name in ABLATIONS:
        return ABLATIONS[name]
    if name in DEFERRED_ABLATIONS:
        return DEFERRED_ABLATIONS[name]
    raise KeyError(f"unknown ablation {name!r}; have {sorted(ABLATIONS)}")
