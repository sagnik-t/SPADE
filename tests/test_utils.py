"""Reproducibility tests for seeding and a smoke test for the W&B wrapper."""

import os
import random

from spade.utils import init_wandb, load_env, set_global_seed


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


def test_load_env_reads_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SPADE_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SPADE_TEST_VAR=hello\n")
    assert load_env(env_file) is True
    assert os.environ["SPADE_TEST_VAR"] == "hello"


def test_load_env_does_not_override_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_TEST_VAR", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("SPADE_TEST_VAR=from-file\n")
    load_env(env_file)  # override=False
    assert os.environ["SPADE_TEST_VAR"] == "from-shell"
    load_env(env_file, override=True)
    assert os.environ["SPADE_TEST_VAR"] == "from-file"


def test_load_env_missing_file_is_falsey(tmp_path):
    assert load_env(tmp_path / "nope.env") is False
