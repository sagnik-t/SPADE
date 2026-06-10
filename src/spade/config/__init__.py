"""Configuration package: dataclass configs plus CLI binding helpers."""

from spade.config.base import build_parser, from_namespace, parse_args
from spade.config.configs import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    GenerativeConfig,
    RepresentationConfig,
    SynthesisConfig,
)

__all__ = [
    "build_parser",
    "from_namespace",
    "parse_args",
    "DataConfig",
    "RepresentationConfig",
    "GenerativeConfig",
    "SynthesisConfig",
    "EvalConfig",
    "ExperimentConfig",
]
