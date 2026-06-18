"""Reference recommenders (MF, NCF) defining the shared evaluation space.

The novel geometry metrics (PGPS, NDI) need a *fixed* latent space in which to
compare real and synthetic entities. That space is given by a recommender trained
once on the real train split: matrix factorization (an inner-product model) or
neural collaborative filtering (an MLP over concatenated embeddings). Both expose
their learned user/item embedding tables as the reference coordinates, and both
can score a user against every item for the ranking metrics used by TS-TR — so a
single implementation serves the reference space *and* the downstream utility
check.

Models are Flax ``nnx`` modules trained with Optax on explicit ratings (MSE plus
embedding-L2), consistent with the rest of SPADE: modular components, explicit
PRNG threading, no global RNG reliance.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spade.config.configs import EvalConfig
from spade.data.interactions import InteractionStore
from spade.models.mlp import MLP
from spade.utils import get_logger

__all__ = [
    "MFModel",
    "NCFModel",
    "Recommender",
    "ReferenceSpace",
    "build_recommender",
    "train_recommender",
    "train_mf",
    "optimize_recommender",
    "optimize_bpr",
    "build_reference_space",
]

logger = get_logger(__name__)


class MFModel(nnx.Module):
    """Biased matrix factorization: ``score = <p_u, q_i> + b_u + b_i + mu``."""

    def __init__(self, n_users: int, n_items: int, dim: int, *, rngs: nnx.Rngs) -> None:
        self.user_emb = nnx.Embed(n_users, dim, rngs=rngs)
        self.item_emb = nnx.Embed(n_items, dim, rngs=rngs)
        self.user_bias = nnx.Embed(n_users, 1, rngs=rngs)
        self.item_bias = nnx.Embed(n_items, 1, rngs=rngs)
        self.global_bias = nnx.Param(jnp.zeros(()))

    def __call__(self, u: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """Predicted rating for paired ids ``(batch,)``."""
        dot = jnp.sum(self.user_emb(u) * self.item_emb(i), axis=-1)
        bias = self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1)
        return dot + bias + self.global_bias[...]

    def user_table(self) -> np.ndarray:
        return np.asarray(self.user_emb.embedding[...])

    def item_table(self) -> np.ndarray:
        return np.asarray(self.item_emb.embedding[...])

    def user_bias_table(self) -> np.ndarray:
        return np.asarray(self.user_bias.embedding[...]).squeeze(-1)

    def item_bias_table(self) -> np.ndarray:
        return np.asarray(self.item_bias.embedding[...]).squeeze(-1)

    def global_bias_value(self) -> float:
        return float(self.global_bias[...])

    def all_item_scores(self, u: jnp.ndarray) -> jnp.ndarray:
        """Scores of users ``u`` against every item, shape ``(len(u), n_items)``."""
        p = self.user_emb(u)                                  # (b, d)
        q = self.item_emb.embedding[...]                      # (n_items, d)
        bu = self.user_bias(u)                                # (b, 1)
        bi = self.item_bias.embedding[...].squeeze(-1)        # (n_items,)
        return p @ q.T + bu + bi[None, :] + self.global_bias[...]


class NCFModel(nnx.Module):
    """Neural CF: an MLP over ``[p_u; q_i]`` emitting a scalar rating."""

    def __init__(
        self, n_users: int, n_items: int, dim: int, hidden, *, rngs: nnx.Rngs
    ) -> None:
        self.user_emb = nnx.Embed(n_users, dim, rngs=rngs)
        self.item_emb = nnx.Embed(n_items, dim, rngs=rngs)
        self.mlp = MLP(2 * dim, hidden, 1, rngs=rngs)

    def _score(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(jnp.concatenate([p, q], axis=-1)).squeeze(-1)

    def __call__(self, u: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        return self._score(self.user_emb(u), self.item_emb(i))

    def user_table(self) -> np.ndarray:
        return np.asarray(self.user_emb.embedding[...])

    def item_table(self) -> np.ndarray:
        return np.asarray(self.item_emb.embedding[...])

    def all_item_scores(self, u: jnp.ndarray) -> jnp.ndarray:
        """Scores of users ``u`` against every item, shape ``(len(u), n_items)``."""
        p = self.user_emb(u)                                  # (b, d)
        q = self.item_emb.embedding[...]                      # (n_items, d)
        n_items = q.shape[0]
        # Broadcast each user against all items: (b, n_items, d) is materialized
        # per user-row to keep peak memory at one user at a time.
        def row(pu: jnp.ndarray) -> jnp.ndarray:
            pu_rep = jnp.broadcast_to(pu, (n_items, pu.shape[-1]))
            return self._score(pu_rep, q)
        return jax.vmap(row)(p)


Recommender = MFModel | NCFModel


@dataclass
class ReferenceSpace:
    """Fixed latent coordinates for real entities, plus the model that made them.

    ``user_emb``/``item_emb`` are the reference coordinates consumed by PGPS and
    NDI (after synthetic entities are mapped in transductively). ``model`` is
    retained so the same trained recommender can rank items when needed.
    """

    user_emb: np.ndarray            # (n_users, dim)
    item_emb: np.ndarray            # (n_items, dim)
    kind: str
    model: Recommender


def build_recommender(
    kind: str, n_users: int, n_items: int, cfg: EvalConfig, *, seed: int
) -> Recommender:
    """Construct an untrained reference recommender of the requested ``kind``."""
    rngs = nnx.Rngs(seed)
    # "bpr" shares the biased-MF scoring model; only its training objective
    # differs (pairwise ranking instead of rating MSE).
    if kind in ("mf", "bpr"):
        return MFModel(n_users, n_items, cfg.ref_dim, rngs=rngs)
    if kind == "ncf":
        return NCFModel(n_users, n_items, cfg.ref_dim, cfg.ref_hidden, rngs=rngs)
    raise ValueError(
        f"unknown reference model {kind!r}; expected 'mf', 'ncf', or 'bpr'"
    )


@nnx.jit
def _train_step(model, optimizer, u, i, r, l2_lambda):
    def loss_fn(m):
        pred = m(u, i)
        mse = jnp.mean((pred - r) ** 2)
        reg = jnp.mean(m.user_emb(u) ** 2) + jnp.mean(m.item_emb(i) ** 2)
        return mse + l2_lambda * reg

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def optimize_recommender(
    model: Recommender,
    store: InteractionStore,
    *,
    epochs: int,
    lr: float,
    l2: float,
    batch_size: int,
    seed: int = 0,
    tag: str = "recommender",
) -> Recommender:
    """Fit ``model`` on ``store`` in place (explicit-rating MSE + embedding L2).

    Mini-batch Adam with a per-epoch reshuffle. Shared by the reference space and
    the baselines so every MF/NCF is trained the same way.
    """
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    rng = np.random.default_rng(seed)
    u_all, i_all = store.user_idx, store.item_idx
    r_all = store.ratings.astype(np.float32)
    n = store.nnz
    for epoch in range(epochs):
        perm = rng.permutation(n)
        last = 0.0
        for start in range(0, n, batch_size):
            sl = perm[start : start + batch_size]
            last = float(
                _train_step(
                    model,
                    optimizer,
                    jnp.asarray(u_all[sl]),
                    jnp.asarray(i_all[sl]),
                    jnp.asarray(r_all[sl]),
                    l2,
                )
            )
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("%s | epoch %d | loss %.4f", tag, epoch, last)
    return model


@nnx.jit
def _bpr_step(model, optimizer, u, pos, neg, l2_lambda):
    def loss_fn(m):
        # BPR: maximize the margin between a positive and a sampled negative item.
        # The user bias and global bias cancel in the (pos - neg) difference; the
        # item biases (b_pos - b_neg) remain and are learned, as in BPR-MF.
        x_pos = m(u, pos)
        x_neg = m(u, neg)
        ranking = -jnp.mean(jax.nn.log_sigmoid(x_pos - x_neg))
        reg = (
            jnp.mean(m.user_emb(u) ** 2)
            + jnp.mean(m.item_emb(pos) ** 2)
            + jnp.mean(m.item_emb(neg) ** 2)
        )
        return ranking + l2_lambda * reg

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def optimize_bpr(
    model: MFModel,
    store: InteractionStore,
    *,
    epochs: int,
    lr: float,
    l2: float,
    batch_size: int,
    neg_samples: int = 1,
    seed: int = 0,
    tag: str = "bpr",
) -> MFModel:
    """Fit ``model`` in place with the BPR pairwise-ranking objective.

    Each observed interaction is a positive ``(u, i)``; ``neg_samples`` negative
    items per positive are drawn uniformly over the catalog (the standard BPR
    approximation — occasional false negatives are tolerated). Ratings are ignored:
    the objective is purely "observed ranks above unobserved", which is the right
    target for the top-k Recall/NDCG that TS-TR measures.
    """
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    rng = np.random.default_rng(seed)
    u_all, i_all = store.user_idx, store.item_idx
    n = store.nnz
    n_items = store.n_items
    for epoch in range(epochs):
        perm = rng.permutation(n)
        last = 0.0
        for start in range(0, n, batch_size):
            sl = perm[start : start + batch_size]
            u_b = u_all[sl]
            pos_b = i_all[sl]
            if neg_samples > 1:
                u_b = np.repeat(u_b, neg_samples)
                pos_b = np.repeat(pos_b, neg_samples)
            neg_b = rng.integers(0, n_items, size=u_b.shape[0])
            last = float(
                _bpr_step(
                    model,
                    optimizer,
                    jnp.asarray(u_b),
                    jnp.asarray(pos_b),
                    jnp.asarray(neg_b),
                    l2,
                )
            )
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("%s | epoch %d | loss %.4f", tag, epoch, last)
    return model


def train_recommender(
    store: InteractionStore,
    cfg: EvalConfig,
    *,
    kind: str = "mf",
    seed: int = 0,
) -> Recommender:
    """Fit a reference/downstream recommender on ``store`` using EvalConfig knobs.

    ``mf``/``ncf`` are trained with rating MSE; ``bpr`` trains the MF scoring model
    with the pairwise-ranking objective (the appropriate downstream model for the
    top-k ranking that TS-TR reports).
    """
    model = build_recommender(kind, store.n_users, store.n_items, cfg, seed=seed)
    if kind == "bpr":
        assert isinstance(model, MFModel)  # build_recommender guarantees this
        return optimize_bpr(
            model, store, epochs=cfg.ref_epochs, lr=cfg.ref_lr, l2=cfg.ref_l2,
            batch_size=cfg.ref_batch_size, neg_samples=cfg.bpr_neg_samples,
            seed=seed, tag="reference bpr",
        )
    return optimize_recommender(
        model, store, epochs=cfg.ref_epochs, lr=cfg.ref_lr, l2=cfg.ref_l2,
        batch_size=cfg.ref_batch_size, seed=seed, tag=f"reference {kind}",
    )


def train_mf(
    store: InteractionStore,
    *,
    dim: int,
    epochs: int,
    lr: float = 1e-3,
    l2: float = 1e-5,
    batch_size: int = 1024,
    seed: int = 0,
) -> MFModel:
    """Standalone biased-MF trainer for the baselines (returns the trained model)."""
    model = MFModel(store.n_users, store.n_items, dim, rngs=nnx.Rngs(seed))
    optimize_recommender(
        model, store, epochs=epochs, lr=lr, l2=l2, batch_size=batch_size,
        seed=seed, tag="baseline mf",
    )
    return model


def build_reference_space(
    store: InteractionStore,
    cfg: EvalConfig,
    *,
    kind: str = "mf",
    seed: int = 0,
) -> ReferenceSpace:
    """Train a reference recommender and package its embedding tables."""
    model = train_recommender(store, cfg, kind=kind, seed=seed)
    return ReferenceSpace(
        user_emb=model.user_table(),
        item_emb=model.item_table(),
        kind=kind,
        model=model,
    )
