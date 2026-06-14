"""Generic parameter checkpointing for Flax ``nnx`` modules.

Saves a module's :class:`nnx.Param` leaves to a path-keyed ``.npz`` and restores
them into a freshly built module of the same structure. This is intentionally
framework-light (NumPy archives, no orbax dependency) and shared by every stage
trainer so generators, critics, and composite models all round-trip the same
way.

Reload requires rebuilding the module first (same constructor args) and then
calling :func:`load_params_into`; the saved tree is matched to the live tree by
parameter path, so structures must agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = ["save_params", "load_params_into"]

_SEP = "/"


def save_params(module: nnx.Module, path: str | Path, **meta: Any) -> Path:
    """Save ``module``'s parameters to ``path`` (``.npz``), plus scalar ``meta``.

    Metadata values are stored alongside the arrays under ``_``-prefixed keys so
    a loader can recover construction hints (e.g. dimensions) if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = nnx.to_flat_state(nnx.state(module, nnx.Param))
    # dict[str, Any] so the kwargs spread does not collide with savez(allow_pickle=).
    payload: dict[str, Any] = {f"_{k}": v for k, v in meta.items()}
    payload.update({_SEP.join(map(str, p)): np.asarray(v[...]) for p, v in flat})
    np.savez(path, **payload)
    return path


def load_params_into(module: nnx.Module, path: str | Path) -> nnx.Module:
    """Restore parameters saved by :func:`save_params` into ``module`` in place.

    Returns the same ``module`` for convenience. Raises ``KeyError`` if the saved
    archive is missing a parameter present in the live module (structure drift).
    """
    loaded = np.load(path)
    flat = nnx.to_flat_state(nnx.state(module, nnx.Param))
    restored = [
        (p, v.replace(jnp.asarray(loaded[_SEP.join(map(str, p))]))) for p, v in flat
    ]
    nnx.update(module, nnx.from_flat_state(restored))
    return module
