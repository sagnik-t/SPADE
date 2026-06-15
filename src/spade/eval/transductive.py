"""Transductive placement of synthetic entities into the reference space.

PGPS and NDI compare real and synthetic entities inside the *fixed* reference
space of :mod:`spade.eval.reference`. Synthetic users and items, however, share
no index axis with real entities (a synthetic user interacts only with synthetic
items), so a synthetic entity cannot be folded into the reference space by its
interactions. Instead we exploit the one space both kinds of entity *do* share —
SPADE's Stage I latent space — and learn a linear bridge from it to the reference
space.

Concretely, for each axis we solve a ridge least-squares problem
``min_W ||Z_real W - R_ref||^2 + lambda ||W||^2`` on the real entities, where
``Z_real`` are the Stage I encoder latents and ``R_ref`` the reference embeddings
(both indexed identically over real entities). The closed-form solution
``W = (Z^T Z + lambda I)^-1 Z^T R`` (with an appended bias column) is deterministic.
Synthetic entities are then mapped in as ``Z_synth_aug @ W``. This is the
"transductive inference" the metrics rely on: the reference model stays fixed on
real data; only the cross-space alignment is fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LinearMap", "TransductiveEmbedder", "fit_linear_map", "fit_transductive"]


def _augment(z: np.ndarray) -> np.ndarray:
    """Append a constant column so the map can represent an affine offset."""
    z = np.asarray(z, dtype=np.float64)
    return np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)


@dataclass(frozen=True)
class LinearMap:
    """An affine map ``z -> [z, 1] @ W`` from a source space to a target space."""

    weight: np.ndarray              # (d_src + 1, d_tgt)

    def apply(self, z: np.ndarray) -> np.ndarray:
        """Map source coordinates ``(n, d_src)`` to target ``(n, d_tgt)``."""
        return (_augment(z) @ self.weight).astype(np.float32)


def fit_linear_map(
    source: np.ndarray, target: np.ndarray, ridge: float = 1e-2
) -> LinearMap:
    """Ridge least-squares map ``source -> target`` fit on aligned rows.

    ``source[k]`` and ``target[k]`` must describe the same entity. The bias
    column is not penalized. Solved via the normal equations on the augmented
    design matrix, which is small (``d_src+1`` square) and deterministic.
    """
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of rows")
    x = _augment(source)                                   # (n, d+1)
    y = np.asarray(target, dtype=np.float64)               # (n, d_tgt)
    d1 = x.shape[1]
    reg = ridge * np.eye(d1)
    reg[-1, -1] = 0.0                                       # don't penalize bias
    weight = np.linalg.solve(x.T @ x + reg, x.T @ y)       # (d+1, d_tgt)
    return LinearMap(weight=weight)


@dataclass(frozen=True)
class TransductiveEmbedder:
    """Paired user/item maps from Stage I latents to the reference space."""

    user_map: LinearMap
    item_map: LinearMap

    def embed_users(self, z_users: np.ndarray) -> np.ndarray:
        return self.user_map.apply(z_users)

    def embed_items(self, z_items: np.ndarray) -> np.ndarray:
        return self.item_map.apply(z_items)


def fit_transductive(
    z_users_real: np.ndarray,
    z_items_real: np.ndarray,
    ref_user_emb: np.ndarray,
    ref_item_emb: np.ndarray,
    ridge: float = 1e-2,
) -> TransductiveEmbedder:
    """Fit both axis maps from real Stage I latents to real reference embeddings.

    All four arrays are indexed over the *real* train universe: row ``k`` of
    ``z_users_real`` and ``ref_user_emb`` is the same user, likewise for items.
    """
    return TransductiveEmbedder(
        user_map=fit_linear_map(z_users_real, ref_user_emb, ridge),
        item_map=fit_linear_map(z_items_real, ref_item_emb, ridge),
    )
