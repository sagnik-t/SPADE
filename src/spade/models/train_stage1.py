"""Joint training of the Stage I representation model, with freeze + export.

Trains the four components together under one Adam optimizer (Optax), monitoring
a held-out validation loss for early stopping. Negative items are resampled every
batch in NumPy (uniform over unobserved pairs) and passed into the jitted step,
keeping the compiled graph pure. After training the best-validation parameters
are restored, the encoders are run over every id, and the resulting real latent
clouds ``Z_u``/``Z_i`` are exported with the rating vocabulary for Stage II/III.

The model is deterministic at inference, so "freezing" is simply: stop updating
and export. PRNG keys are threaded explicitly via ``nnx.Rngs`` for init and a
NumPy ``Generator`` for sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.data.sampling import sample_negatives_for_users
from spade.models.decoder import RatingVocab
from spade.models.losses import stage1_loss
from spade.models.representation import RepresentationModel
from spade.utils import get_logger

__all__ = ["TrainState", "train_stage1", "export_stage1"]

logger = get_logger(__name__)


@dataclass
class TrainState:
    """Result of Stage I training: the model, vocab, and per-epoch history."""

    model: RepresentationModel
    vocab: RatingVocab
    history: list[dict[str, float]]
    best_epoch: int


def _observed_sets(*stores: InteractionStore, n_users: int) -> list[set[int]]:
    """Per-user observed item sets across one or more stores (negative masking)."""
    seen: list[set[int]] = [set() for _ in range(n_users)]
    for store in stores:
        for u, i in zip(store.user_idx.tolist(), store.item_idx.tolist(), strict=True):
            seen[u].add(i)
    return seen


@nnx.jit
def _train_step(model, optimizer, u, i_pos, i_neg, rating_idx, l2_lambda):
    def loss_fn(m):
        total, parts = stage1_loss(m, u, i_pos, i_neg, rating_idx, l2_lambda)
        return total, parts

    (_, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return parts


def _eval_loss(model, store, rating_idx, observed, n_neg, l2_lambda, rng):
    """Validation loss: same objective, negatives sampled excluding observed."""
    if store.nnz == 0:
        return {"total": float("inf")}
    neg = sample_negatives_for_users(
        store.user_idx, store, n_neg, rng, observed=observed
    )
    _, parts = stage1_loss(
        model,
        jnp.asarray(store.user_idx),
        jnp.asarray(store.item_idx),
        jnp.asarray(neg),
        jnp.asarray(rating_idx),
        l2_lambda,
    )
    return {k: float(v) for k, v in parts.items()}


def train_stage1(
    cfg: ExperimentConfig,
    train: InteractionStore,
    val: InteractionStore,
) -> TrainState:
    """Train Stage I jointly and return the best-validation model + vocabulary."""
    rcfg = cfg.representation
    n_neg = cfg.data.n_neg
    n_users, n_items = train.n_users, train.n_items
    rng = np.random.default_rng(cfg.seed)

    vocab = RatingVocab.from_ratings(train.ratings)
    train_rating_idx = vocab.to_index(train.ratings)
    val_rating_idx = vocab.to_index(val.ratings) if val.nnz else np.empty(0, np.int64)
    logger.info("rating levels: %s", vocab.values.tolist())

    model = RepresentationModel(
        n_users, n_items, vocab.n_levels, rcfg, rngs=nnx.Rngs(cfg.seed)
    )
    optimizer = nnx.Optimizer(model, optax.adam(rcfg.lr), wrt=nnx.Param)

    train_obs = _observed_sets(train, n_users=n_users)
    val_obs = _observed_sets(train, val, n_users=n_users)  # exclude real val pairs

    n = train.nnz
    history: list[dict[str, float]] = []
    best_val, best_epoch, best_state = float("inf"), -1, None
    patience = 0

    for epoch in range(rcfg.epochs):
        perm = rng.permutation(n)
        for start in range(0, n, rcfg.batch_size):
            sl = perm[start : start + rcfg.batch_size]
            u = train.user_idx[sl]
            neg = sample_negatives_for_users(
                u, train, n_neg, rng, observed=train_obs
            )
            _train_step(
                model,
                optimizer,
                jnp.asarray(u),
                jnp.asarray(train.item_idx[sl]),
                jnp.asarray(neg),
                jnp.asarray(train_rating_idx[sl]),
                rcfg.l2_lambda,
            )

        val_parts = _eval_loss(
            model, val, val_rating_idx, val_obs, n_neg, rcfg.l2_lambda, rng
        )
        history.append({"epoch": epoch, **val_parts})
        logger.info("epoch %d | val %s", epoch, val_parts)

        if val_parts["total"] < best_val - 1e-5:
            best_val, best_epoch, patience = val_parts["total"], epoch, 0
            best_state = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        else:
            patience += 1
            if patience >= rcfg.early_stop_patience:
                logger.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    if best_state is not None:
        nnx.update(model, best_state)  # restore best-validation parameters
    return TrainState(model=model, vocab=vocab, history=history, best_epoch=best_epoch)


def export_stage1(
    state: TrainState,
    n_users: int,
    n_items: int,
    output_dir: str | Path,
    dataset: str,
    seed: int,
) -> Path:
    """Export frozen ``Z_u``/``Z_i`` and the rating vocabulary for Stage II/III."""
    z_u, z_i = state.model.export_embeddings(n_users, n_items)
    path = Path(output_dir) / dataset / f"stage1_seed_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z_users=z_u,
        z_items=z_i,
        rating_values=state.vocab.values,
        best_epoch=state.best_epoch,
    )
    logger.info("exported Z_u%s, Z_i%s to %s", z_u.shape, z_i.shape, path)
    return path
