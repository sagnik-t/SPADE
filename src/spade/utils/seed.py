"""Global seeding for reproducibility across the 5-seed experimental protocol."""

from __future__ import annotations

import os
import random

__all__ = ["set_global_seed"]


def set_global_seed(seed: int, deterministic_tf: bool = True) -> int:
    """Seed Python, NumPy, and TensorFlow RNGs.

    TensorFlow and NumPy are imported lazily so lightweight code paths (config
    parsing, tests) do not pay the TF import cost. Returns ``seed`` for logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic_tf:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:  # pragma: no cover - hardware/op dependent
                pass
    except ImportError:
        pass

    return seed
