"""Reference-space geometry metrics (PGPS + NDI), generator-agnostic.

Factored out of the SPADE-specific suite so any generator that supplies entity
latents — SPADE or a latent baseline — gets the same treatment: for each
reference model, build its space on the real train split, fit a transductive map
from the generator's *real* latents to the reference embeddings, carry the
*synthetic* latents across, and score PGPS (items) and NDI (users).
"""

from __future__ import annotations

import numpy as np

from spade.config.configs import EvalConfig
from spade.data.interactions import InteractionStore
from spade.eval.ndi import ndi
from spade.eval.pgps import pgps
from spade.eval.reference import build_reference_space
from spade.eval.transductive import fit_transductive
from spade.utils import get_logger

__all__ = ["geometry_metrics"]

logger = get_logger(__name__)


def geometry_metrics(
    train: InteractionStore,
    ecfg: EvalConfig,
    real_users: np.ndarray,
    real_items: np.ndarray,
    synth_users: np.ndarray,
    synth_items: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """PGPS + NDI per reference model from a generator's real/synth latents.

    ``real_*`` are the generator's latents for real entities (aligned to the
    train index space); ``synth_*`` are its latents for synthetic entities.
    Returns ``{reference_model: {metric: value}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for kind in ecfg.reference_models:
        logger.info("building %s reference space", kind)
        ref = build_reference_space(train, ecfg, kind=kind, seed=seed)
        embed = fit_transductive(
            real_users, real_items, ref.user_emb, ref.item_emb, ecfg.map_ridge
        )
        synth_user_ref = embed.embed_users(synth_users)
        synth_item_ref = embed.embed_items(synth_items)

        pgps_res = pgps(ref.item_emb, synth_item_ref, ecfg.neighbor_k)
        ndi_res = ndi(ref.user_emb, synth_user_ref, ecfg.neighbor_k)
        out[kind] = {**pgps_res.as_dict(), **ndi_res.as_dict()}
        logger.info("%s | %s", kind, out[kind])
    return out
