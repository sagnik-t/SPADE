"""Per-cell experiment pipeline: train SPADE once, generate, evaluate any generator.

One *cell* of the run matrix is a (dataset, seed, ablation, generator) point. The
expensive part — fitting SPADE's representation and generative stages — depends
only on (dataset, seed, stage configs), so it is trained once per such key and
cached on disk; every generator for that key reuses it (baselines ignore it
entirely, being self-contained). :func:`evaluate_output` then runs the full
metric battery on whatever :class:`GeneratorOutput` a generator produced, reusing
the generator-agnostic geometry path plus the distribution and TS-TR metrics.

Stage caches are keyed by a short signature of the relevant config section so an
ablation that changes a stage (e.g. latent-reg off → different generative config)
gets its own cache, while ablations that only touch synthesis/expansion reuse the
base stages.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spade.baselines import BASELINE_REGISTRY, GeneratorOutput, SpadeGenerator
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.data.split import Splits
from spade.eval.distributions import degree_ks, gaussian_w2
from spade.eval.downstream import ts_tr
from spade.eval.geometry import geometry_metrics
from spade.experiments.aggregate import flatten_cell
from spade.models.decoder import RatingVocab
from spade.models.generative import GenerativeModel, JointGenerativeModel
from spade.models.representation import RepresentationModel
from spade.training import (
    GenerativeTrainer,
    JointGenerativeTrainer,
    RepresentationTrainer,
    load_generative_model,
    load_representation_model,
)
from spade.training.checkpoint import save_params
from spade.utils import WandbRun, get_logger, init_wandb, jax_key

__all__ = ["SpadeModels", "train_spade_stages", "evaluate_output", "run_cell"]

logger = get_logger(__name__)


def _sig(section) -> str:
    """Short stable signature of a config dataclass section (for cache keys)."""
    return hashlib.md5(repr(dataclasses.asdict(section)).encode()).hexdigest()[:8]


def _run(cfg: ExperimentConfig, name: str | None, group: str | None = None) -> WandbRun | None:
    """Open a W&B run for a stage/cell, or ``None`` when tracking is off.

    ``name is None`` means the caller opted out of tracking entirely; otherwise
    :func:`init_wandb` still returns an inactive (no-op) handle when the configured
    ``wandb_mode`` is ``disabled`` or wandb is unavailable, so callers never have
    to special-case it.
    """
    if name is None:
        return None
    return init_wandb(cfg, project=cfg.wandb_project, name=name, mode=cfg.wandb_mode, group=group)


@dataclass
class SpadeModels:
    """The trained SPADE stage models needed to drive synthesis."""

    representation: RepresentationModel
    vocab: RatingVocab
    generative: GenerativeModel | JointGenerativeModel


def train_spade_stages(
    cfg: ExperimentConfig,
    splits: Splits,
    *,
    cache_dir: str | Path,
    wandb_prefix: str | None = None,
) -> SpadeModels:
    """Train (or load from cache) SPADE's representation and generative stages.

    When ``wandb_prefix`` is given, each stage that is actually trained (not loaded
    from cache) logs its per-epoch curves to its own W&B run named
    ``<wandb_prefix>-representation`` / ``<wandb_prefix>-generative``. Cached stages
    log nothing, since there is no training to track.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    train = splits.train
    rep_path = cache / f"rep_seed{cfg.seed}_{_sig(cfg.representation)}.npz"
    gen_path = cache / f"gen_seed{cfg.seed}_{_sig(cfg.generative)}.npz"

    if rep_path.exists():
        representation, vocab = load_representation_model(rep_path, cfg, seed=cfg.seed)
    else:
        logger.info("training representation stage -> %s", rep_path.name)
        run = _run(cfg, f"{wandb_prefix}-representation" if wandb_prefix else None, group=wandb_prefix)
        trainer = RepresentationTrainer(cfg, train, splits.val, run=run).fit()
        if run is not None:
            run.finish()
        save_params(
            trainer.model, rep_path,
            n_users=train.n_users, n_items=train.n_items,
            n_levels=trainer.vocab.n_levels, rating_values=trainer.vocab.values,
        )
        representation, vocab = trainer.model, trainer.vocab

    if gen_path.exists():
        generative = load_generative_model(gen_path, cfg, seed=cfg.seed)
    else:
        logger.info("training generative stage -> %s", gen_path.name)
        run = _run(cfg, f"{wandb_prefix}-generative" if wandb_prefix else None, group=wandb_prefix)
        z_u, z_i = representation.export_embeddings(train.n_users, train.n_items)
        if cfg.generative.joint:
            # Joint ablation: train one WGAN-GP over the concatenated latents of
            # observed interactions instead of two independent entity generators.
            z_pairs = np.concatenate(
                [z_u[train.user_idx], z_i[train.item_idx]], axis=1
            )
            gtrainer = JointGenerativeTrainer(cfg, z_pairs, run=run).fit()
        else:
            gtrainer = GenerativeTrainer(cfg, z_u, z_i, run=run).fit()
        if run is not None:
            run.finish()
        save_params(
            gtrainer.model, gen_path,
            latent_dim=gtrainer.model.latent_dim, noise_dim=cfg.generative.noise_dim,
        )
        generative = gtrainer.model

    return SpadeModels(representation, vocab, generative)


def evaluate_output(
    train: InteractionStore,
    test: InteractionStore,
    output: GeneratorOutput,
    cfg: ExperimentConfig,
    *,
    seed: int,
) -> dict:
    """Run the full metric suite on a generator's output.

    Geometry (PGPS/NDI) and latent W₂ are computed only when the generator
    exported a latent bundle; otherwise they are ``None`` and the generator is
    judged on degree-KS and TS-TR alone.
    """
    synth_store = output.dataset.as_store()
    result: dict = {
        "name": output.name,
        "synthetic": output.dataset.summary(),
        "degree": degree_ks(train, synth_store),
        "tstr": ts_tr(synth_store, train, test, cfg.eval, seed=seed).as_dict(),
    }
    if output.latents is not None:
        lb = output.latents
        result["geometry"] = geometry_metrics(
            train, cfg.eval, lb.real_users, lb.real_items,
            lb.synth_users, lb.synth_items, seed=seed,
        )
        result["latent"] = {
            "w2_user_latent": gaussian_w2(lb.real_users, lb.synth_users),
            "w2_item_latent": gaussian_w2(lb.real_items, lb.synth_items),
        }
    else:
        result["geometry"] = None
        result["latent"] = None
    return result


def run_cell(
    cfg: ExperimentConfig,
    generator_name: str,
    splits: Splits,
    *,
    models: SpadeModels | None = None,
    wandb_name: str | None = None,
) -> dict:
    """Generate with one generator and evaluate it.

    ``models`` are required only for the ``spade`` generator; baselines build
    themselves from the train store and config. When ``wandb_name`` is given, the
    cell's flat scalar metrics are logged to a W&B run of that name.
    """
    train = splits.train
    if generator_name == "spade":
        if models is None:
            raise ValueError("the spade generator requires trained stage models")
        generator = SpadeGenerator(
            models.representation, models.generative, models.vocab, train, cfg
        )
    else:
        generator = BASELINE_REGISTRY[generator_name](train, cfg)

    output = generator.generate(jax_key(cfg.seed))
    cell = evaluate_output(train, splits.test, output, cfg, seed=cfg.seed)

    group = wandb_name.rsplit("-", 1)[0] if wandb_name else None
    run = _run(cfg, wandb_name, group=group)
    if run is not None:
        run.log(flatten_cell(cell))
        run.finish()
    return cell
