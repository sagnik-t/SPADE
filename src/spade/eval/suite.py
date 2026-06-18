"""End-to-end evaluation suite: every metric over one synthesized run.

Given a trained SPADE run (representation + generative exports) and its saved
synthetic dataset, :func:`run_evaluation` assembles the shared reference space,
maps synthetic entities into it transductively, and computes the full metric
battery:

* **PGPS** (+ random baseline) and **NDI**, once per reference model (MF, NCF),
  so geometry findings can be checked for reference-model sensitivity;
* **latent W₂** between real and synthetic Stage I clouds (users and items);
* **KS** distance on user/item degree distributions;
* **TS-TR** downstream Recall/NDCG/MAP with RelPerf.

The synthetic latent clouds are re-derived from the generative model under the
synthesis seed (see :meth:`SynthesisModel.sample_latents`), so they match the
saved discrete dataset exactly without persisting extra arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spade.config.configs import ExperimentConfig
from spade.data import Splits, load_splits
from spade.eval.distributions import degree_ks, gaussian_w2
from spade.eval.downstream import ts_tr
from spade.eval.geometry import geometry_metrics
from spade.synthesis import SyntheticDataset
from spade.synthesis.synthesizer import SynthesisModel
from spade.training import load_generative_model, load_representation_model
from spade.utils import get_logger, jax_key

__all__ = ["run_evaluation"]

logger = get_logger(__name__)


def run_evaluation(cfg: ExperimentConfig) -> dict:
    """Compute the full metric suite for one run, returning a nested results dict."""
    base = Path(cfg.output_dir) / cfg.data.dataset
    rep_path = base / f"representation_model_seed_{cfg.seed}.npz"
    gen_path = base / f"generative_seed_{cfg.seed}.npz"
    synth_path = base / f"synthetic_seed_{cfg.seed}.npz"
    for p in (rep_path, gen_path, synth_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing artifact {p}; run training and synthesis for this seed first."
            )

    splits: Splits = load_splits(cfg.data.data_dir, cfg.data.dataset, cfg.seed)
    train, test = splits.train, splits.test

    representation, vocab = load_representation_model(rep_path, cfg, seed=cfg.seed)
    generative = load_generative_model(gen_path, cfg, seed=cfg.seed)
    z_users_real, z_items_real = representation.export_embeddings(
        train.n_users, train.n_items
    )

    synth = SyntheticDataset.load(synth_path)
    synth_store = synth.as_store()

    # Re-derive the synthetic latent clouds under the synthesis seed.
    synthesizer = SynthesisModel(
        representation,
        generative,
        vocab,
        source_n_users=train.n_users,
        source_n_items=train.n_items,
        source_rho=train.rho,
        cfg=cfg.synthesis,
    )
    z_u_synth, z_i_synth = synthesizer.sample_latents(jax_key(cfg.seed))
    z_u_synth = np.asarray(z_u_synth)
    z_i_synth = np.asarray(z_i_synth)

    geometry = geometry_metrics(
        train, cfg.eval, z_users_real, z_items_real, z_u_synth, z_i_synth, seed=cfg.seed
    )

    latent = {
        "w2_user_latent": gaussian_w2(z_users_real, z_u_synth),
        "w2_item_latent": gaussian_w2(z_items_real, z_i_synth),
    }
    degrees = degree_ks(train, synth_store)

    logger.info("running TS-TR downstream check")
    tstr = ts_tr(synth_store, train, test, cfg.eval, seed=cfg.seed)

    results = {
        "dataset": cfg.data.dataset,
        "seed": cfg.seed,
        "synthetic": synth.summary(),
        "geometry": geometry,
        "latent": latent,
        "degree": degrees,
        "tstr": tstr.as_dict(),
    }
    return results
