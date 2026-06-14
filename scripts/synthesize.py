"""Synthesize a discrete dataset from trained representation + generative stages.

Loads the frozen representation model (gate/decoder + rating vocabulary) and the
trained generative model for the configured dataset/seed, sizes the synthesis to
the *train* split's universe and sparsity, runs the Stage III pipeline, and saves
the synthetic dataset under ``output_dir``::

    poetry run python scripts/synthesize.py --data.dataset ml-100k --seed 42 \
        --synthesis.alpha 2 --synthesis.beta 2
"""

from __future__ import annotations

from pathlib import Path

from spade.config import ExperimentConfig
from spade.config.base import parse_args
from spade.data import load_splits
from spade.synthesis import SynthesisModel
from spade.training import load_generative_model, load_representation_model
from spade.utils import get_logger, jax_key, load_env, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    load_env()
    cfg: ExperimentConfig = parse_args(ExperimentConfig)
    set_global_seed(cfg.seed)
    base = Path(cfg.output_dir) / cfg.data.dataset

    rep_path = base / f"representation_model_seed_{cfg.seed}.npz"
    gen_path = base / f"generative_seed_{cfg.seed}.npz"
    for p in (rep_path, gen_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing stage export: {p}. Run the representation and "
                "generative training scripts first."
            )

    # Train split defines the source universe and leakage-safe sparsity rho.
    splits = load_splits(cfg.data.data_dir, cfg.data.dataset, cfg.seed)
    train = splits.train

    representation, vocab = load_representation_model(rep_path, cfg, seed=cfg.seed)
    generative = load_generative_model(gen_path, cfg, seed=cfg.seed)

    synthesizer = SynthesisModel(
        representation,
        generative,
        vocab,
        source_n_users=train.n_users,
        source_n_items=train.n_items,
        source_rho=train.rho,
        cfg=cfg.synthesis,
    )
    synth = synthesizer.synthesize(jax_key(cfg.seed))

    out = base / f"synthetic_seed_{cfg.seed}.npz"
    synth.save(out)
    logger.info("synthesis complete %s -> %s", synth.summary(), out)


if __name__ == "__main__":
    main()
