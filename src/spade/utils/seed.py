"""Seeding utilities.

JAX threads explicit PRNG keys rather than relying on global RNG state, so
:func:`set_global_seed` covers the stateful libraries (Python, NumPy) used for
data shuffling, negative sampling, and faiss, while :func:`jax_key` mints an
explicit key for model code to split and thread through.
"""

from __future__ import annotations

import os
import random
from typing import Any

__all__ = ["set_global_seed", "jax_key"]


def set_global_seed(seed: int) -> int:
    """Seed the stateful RNGs (Python, NumPy). Returns ``seed`` for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass
    return seed


def jax_key(seed: int) -> Any:
    """Return a JAX PRNG key (typed-key API) to be split and threaded by callers."""
    import jax

    return jax.random.key(seed)
