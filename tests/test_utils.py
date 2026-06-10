"""Reproducibility tests for seeding and a smoke test for the W&B wrapper."""

import random

from spade.utils import init_wandb, set_global_seed


def test_python_rng_is_deterministic():
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_rng_is_deterministic():
    import numpy as np

    set_global_seed(7)
    a = np.random.rand(4).tolist()
    set_global_seed(7)
    b = np.random.rand(4).tolist()
    assert a == b


def test_set_global_seed_returns_seed():
    assert set_global_seed(99) == 99


def test_disabled_wandb_is_inactive_and_noops():
    run = init_wandb({"lr": 0.1}, mode="disabled")
    assert run.active is False
    run.log({"loss": 1.0})  # must not raise
    run.finish()
