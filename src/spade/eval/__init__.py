"""Evaluation suite (Phase 5).

A fixed reference space (MF/NCF on real data) plus a transductive linear map from
SPADE's Stage I latents lets the novel geometry metrics — PGPS and NDI — compare
synthetic and real entities that share no index axis. Alongside them sit standard
distributional checks (latent W₂, KS degree distance) and the TS-TR downstream
utility comparison. :func:`run_evaluation` runs the whole battery for one seed.
"""

from spade.eval.distributions import degree_ks, gaussian_w2, ks_distance
from spade.eval.downstream import TSTRResult, evaluate_ranking, ts_tr
from spade.eval.geometry import geometry_metrics
from spade.eval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    ranking_metrics,
    recall_at_k,
)
from spade.eval.ndi import NDIResult, ndi
from spade.eval.pgps import PGPSResult, pgps
from spade.eval.reference import (
    MFModel,
    NCFModel,
    ReferenceSpace,
    build_reference_space,
    train_recommender,
)
from spade.eval.suite import run_evaluation
from spade.eval.transductive import (
    LinearMap,
    TransductiveEmbedder,
    fit_linear_map,
    fit_transductive,
)

__all__ = [
    "MFModel",
    "NCFModel",
    "ReferenceSpace",
    "build_reference_space",
    "train_recommender",
    "LinearMap",
    "TransductiveEmbedder",
    "fit_linear_map",
    "fit_transductive",
    "PGPSResult",
    "pgps",
    "NDIResult",
    "ndi",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision_at_k",
    "ranking_metrics",
    "TSTRResult",
    "evaluate_ranking",
    "ts_tr",
    "gaussian_w2",
    "ks_distance",
    "degree_ks",
    "geometry_metrics",
    "run_evaluation",
]
