"""Tests for the dataclass <-> CLI configuration layer."""

from spade.config import ExperimentConfig, RepresentationConfig, parse_args
from spade.config.base import build_parser, from_namespace


def test_defaults_round_trip():
    cfg = parse_args(ExperimentConfig, [])
    assert cfg.name == "spade-default"
    assert cfg.seed == 42
    assert cfg.representation.latent_dim == 64
    assert cfg.synthesis.gamma == 5.0
    assert cfg.eval.n_seeds == 5


def test_nested_override():
    cfg = parse_args(
        ExperimentConfig,
        ["--seed", "7", "--data.dataset", "ml-1m", "--representation.latent-dim", "128"],
    )
    assert cfg.seed == 7
    assert cfg.data.dataset == "ml-1m"
    assert cfg.representation.latent_dim == 128
    # Untouched fields keep their defaults.
    assert cfg.generative.n_critic == 5


def test_list_and_bool_fields():
    cfg = parse_args(
        ExperimentConfig,
        ["--eval.topk", "5", "10", "20", "--wandb-mode", "offline"],
    )
    assert cfg.eval.topk == [5, 10, 20]
    assert cfg.wandb_mode == "offline"


def test_build_parser_is_reusable():
    parser = build_parser(RepresentationConfig)
    ns = parser.parse_args(["--epochs", "3"])
    cfg = from_namespace(RepresentationConfig, ns)
    assert cfg.epochs == 3
    assert cfg.latent_dim == 64
