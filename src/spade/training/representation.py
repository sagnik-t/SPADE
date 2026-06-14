"""Trainer for the representation stage (Stage I).

Fits the four :class:`RepresentationModel` components jointly under one Adam
optimizer, monitoring a held-out validation loss for early stopping. Negative
items are resampled every batch in NumPy (uniform over unobserved pairs) and
passed into a jitted step, keeping the compiled graph pure. The best-validation
parameters are restored at the end, and the frozen encoders are run over every
id to export the real latent clouds ``Z_u``/``Z_i`` (with the rating vocabulary)
for the generative stage.

Subclasses :class:`spade.training.base.Trainer`: the shared epoch loop and W&B
logging live in the base, while this class supplies the joint train step, the
validation objective used for early stopping, best-parameter restoration, and
the embedding export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.data.sampling import sample_negatives_for_users
from spade.models.decoder import RatingVocab
from spade.models.losses import representation_loss
from spade.models.representation import RepresentationModel
from spade.training.base import Trainer
from spade.training.checkpoint import load_params_into, save_params

__all__ = ["RepresentationTrainer", "load_representation_model"]


@nnx.jit
def _train_step(model, optimizer, u, i_pos, i_neg, rating_idx, l2_lambda):
    def loss_fn(m):
        return representation_loss(m, u, i_pos, i_neg, rating_idx, l2_lambda)

    (_, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return parts


def _observed_sets(*stores: InteractionStore, n_users: int) -> list[set[int]]:
    """Per-user observed item sets across one or more stores (negative masking)."""
    seen: list[set[int]] = [set() for _ in range(n_users)]
    for store in stores:
        for u, i in zip(store.user_idx.tolist(), store.item_idx.tolist(), strict=True):
            seen[u].add(i)
    return seen


class RepresentationTrainer(Trainer):
    """Jointly train the representation model with early stopping + export."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        train: InteractionStore,
        val: InteractionStore,
        *,
        run: Any | None = None,
    ) -> None:
        super().__init__(seed=cfg.seed, run=run)
        self.cfg = cfg
        self.train = train
        self.val = val
        self.n_neg = cfg.data.n_neg

        self.vocab = RatingVocab.from_ratings(train.ratings)
        self._train_rating_idx = self.vocab.to_index(train.ratings)
        self._val_rating_idx = (
            self.vocab.to_index(val.ratings) if val.nnz else np.empty(0, np.int64)
        )
        self.logger.info("rating levels: %s", self.vocab.values.tolist())

        self.model = RepresentationModel(
            train.n_users,
            train.n_items,
            self.vocab.n_levels,
            cfg.representation,
            rngs=nnx.Rngs(cfg.seed),
        )
        self.optimizer = nnx.Optimizer(
            self.model, optax.adam(cfg.representation.lr), wrt=nnx.Param
        )

        self._train_obs = _observed_sets(train, n_users=train.n_users)
        self._val_obs = _observed_sets(train, val, n_users=train.n_users)

        self.best_val = float("inf")
        self.best_epoch = -1
        self._best_state = None
        self._patience = 0

    @property
    def num_epochs(self) -> int:
        return self.cfg.representation.epochs

    def train_epoch(self, epoch: int) -> dict[str, float]:
        rcfg = self.cfg.representation
        n = self.train.nnz
        perm = self.rng.permutation(n)
        for start in range(0, n, rcfg.batch_size):
            sl = perm[start : start + rcfg.batch_size]
            u = self.train.user_idx[sl]
            neg = sample_negatives_for_users(
                u, self.train, self.n_neg, self.rng, observed=self._train_obs
            )
            _train_step(
                self.model,
                self.optimizer,
                jnp.asarray(u),
                jnp.asarray(self.train.item_idx[sl]),
                jnp.asarray(neg),
                jnp.asarray(self._train_rating_idx[sl]),
                rcfg.l2_lambda,
            )
        val_parts = self._eval_loss()
        self.logger.info("epoch %d | val %s", epoch, val_parts)
        return val_parts

    def _eval_loss(self) -> dict[str, float]:
        val = self.val
        if val.nnz == 0:
            return {"total": float("inf")}
        neg = sample_negatives_for_users(
            val.user_idx, val, self.n_neg, self.rng, observed=self._val_obs
        )
        _, parts = representation_loss(
            self.model,
            jnp.asarray(val.user_idx),
            jnp.asarray(val.item_idx),
            jnp.asarray(neg),
            jnp.asarray(self._val_rating_idx),
            self.cfg.representation.l2_lambda,
        )
        return {k: float(v) for k, v in parts.items()}

    def should_stop(self, epoch: int, metrics: dict[str, float]) -> bool:
        total = metrics["total"]
        if total < self.best_val - 1e-5:
            self.best_val = total
            self.best_epoch = epoch
            self._patience = 0
            self._best_state = jax.tree.map(jnp.copy, nnx.state(self.model, nnx.Param))
            return False
        self._patience += 1
        return self._patience >= self.cfg.representation.early_stop_patience

    def on_fit_end(self) -> None:
        if self._best_state is not None:
            nnx.update(self.model, self._best_state)  # restore best-validation params

    def export(self, output_dir: str | Path, dataset: str) -> Path:
        """Export frozen ``Z_u``/``Z_i`` and the rating vocabulary."""
        z_u, z_i = self.model.export_embeddings(
            self.train.n_users, self.train.n_items
        )
        path = Path(output_dir) / dataset / f"representation_seed_{self.seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_users=z_u,
            z_items=z_i,
            rating_values=self.vocab.values,
            best_epoch=self.best_epoch,
        )
        self.logger.info("exported Z_u%s, Z_i%s to %s", z_u.shape, z_i.shape, path)
        return path

    def export_model(self, output_dir: str | Path, dataset: str) -> Path:
        """Persist the full trained model (gate/decoder/encoders) for Stage III.

        The embedding export feeds Stage II, but Stage III reuses the *frozen
        gate and decoder*, so their parameters — plus the dims and rating
        vocabulary needed to rebuild the module — are checkpointed separately.
        """
        path = Path(output_dir) / dataset / f"representation_model_seed_{self.seed}.npz"
        save_params(
            self.model,
            path,
            n_users=self.train.n_users,
            n_items=self.train.n_items,
            n_levels=self.vocab.n_levels,
            rating_values=self.vocab.values,
        )
        self.logger.info("exported representation model -> %s", path)
        return path


def load_representation_model(
    path: str | Path,
    cfg: ExperimentConfig,
    *,
    seed: int = 0,
) -> tuple[RepresentationModel, RatingVocab]:
    """Rebuild a :class:`RepresentationModel` and its vocab from an export.

    Returns the model with restored parameters (frozen gate/decoder ready for
    synthesis) and the :class:`RatingVocab` for decoding class indices to ratings.
    """
    loaded = np.load(path)
    model = RepresentationModel(
        int(loaded["_n_users"]),
        int(loaded["_n_items"]),
        int(loaded["_n_levels"]),
        cfg.representation,
        rngs=nnx.Rngs(seed),
    )
    load_params_into(model, path)
    vocab = RatingVocab(values=np.asarray(loaded["_rating_values"]))
    return model, vocab
