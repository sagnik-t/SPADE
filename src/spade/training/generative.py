"""Trainer for the generative stage (Stage II).

Fits a :class:`GenerativeModel` — two independent WGAN-GP pairs — to the frozen
latent clouds ``Z_u``/``Z_i`` exported by the representation stage. Each epoch
advances both pairs by one pass; the user and item nets never share parameters,
batches, or optimizers, so their independence is preserved while their metrics
stay synchronized for logging.

Within a pair, every generator update is preceded by ``n_critic`` critic updates
(the standard WGAN-GP schedule); generated batches fed to the critic are
detached so the critic step never updates the generator. The Wasserstein-1
estimate, gradient penalty, and moment-matching distance are tracked per epoch.

Subclasses :class:`spade.training.base.Trainer`: the base supplies the epoch
loop, PRNG-key threading, and W&B logging; this class supplies the alternating
adversarial optimization and the composite-model export. PRNG keys come from the
base's threaded key; real-batch indices from the base's NumPy generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spade.config.configs import ExperimentConfig, GenerativeConfig
from spade.models.gan_losses import critic_loss, generator_loss
from spade.models.generative import (
    AdversarialPair,
    GenerativeModel,
    JointGenerativeModel,
)
from spade.training.base import Trainer
from spade.training.checkpoint import load_params_into, save_params

__all__ = [
    "GenerativeTrainer",
    "JointGenerativeTrainer",
    "load_generative_model",
]


@nnx.jit
def _critic_step(critic, optimizer, real, fake, key, gp_lambda):
    def loss_fn(c):
        return critic_loss(c, real, fake, key, gp_lambda)

    (_, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic)
    optimizer.update(critic, grads)
    return parts


@nnx.jit
def _generator_step(generator, critic, optimizer, noise, real, moment_lambda):
    def loss_fn(g):
        fake = g(noise)
        return generator_loss(critic, fake, real, moment_lambda)

    (_, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(generator)
    optimizer.update(generator, grads)
    return parts


def _adam(cfg: GenerativeConfig) -> optax.GradientTransformation:
    return optax.adam(cfg.lr, b1=cfg.adam_b1, b2=cfg.adam_b2)


def _pair_epoch(pair, gen_opt, critic_opt, real_all, gcfg, rng, next_keys):
    """Advance one :class:`AdversarialPair` by a full epoch over ``real_all``.

    Shared by the factorized and joint trainers: ``n_critic`` critic updates per
    generator update (standard WGAN-GP schedule), with generated batches detached
    in the critic step. Returns the per-epoch Wasserstein / gradient-penalty /
    moment-matching estimates.
    """
    n = real_all.shape[0]
    bs = min(gcfg.batch_size, n)
    steps = max(1, n // bs)

    w_est, gp_est, m_est = [], [], []
    for _ in range(steps):
        cparts: dict[str, jnp.ndarray] = {}
        for _ in range(max(1, gcfg.n_critic)):
            k_noise, k_gp = next_keys(2)
            real = real_all[rng.integers(0, n, size=bs)]
            fake = pair.generator(pair.generator.sample_noise(k_noise, bs))
            cparts = _critic_step(pair.critic, critic_opt, real, fake, k_gp, gcfg.gp_lambda)

        (k_gen,) = next_keys(1)
        real = real_all[rng.integers(0, n, size=bs)]
        noise = pair.generator.sample_noise(k_gen, bs)
        gparts = _generator_step(
            pair.generator, pair.critic, gen_opt, noise, real, gcfg.moment_lambda
        )
        w_est.append(float(cparts["wasserstein"]))
        gp_est.append(float(cparts["gp"]))
        m_est.append(float(gparts["moment"]))

    return {
        "wasserstein": float(np.mean(w_est)),
        "gp": float(np.mean(gp_est)),
        "moment": float(np.mean(m_est)),
    }


class GenerativeTrainer(Trainer):
    """Train two independent WGAN-GP pairs over the frozen latent clouds."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        z_users: np.ndarray,
        z_items: np.ndarray,
        *,
        run: Any | None = None,
    ) -> None:
        super().__init__(seed=cfg.seed, run=run)
        self.cfg = cfg
        latent_dim = z_users.shape[1]
        if z_items.shape[1] != latent_dim:
            raise ValueError("user/item latent dims must match")

        self.model = GenerativeModel(
            latent_dim, cfg.generative, rngs=nnx.Rngs(cfg.seed)
        )
        self.reals = {
            "user": jnp.asarray(z_users),
            "item": jnp.asarray(z_items),
        }
        self._opts = {
            entity: {
                "gen": nnx.Optimizer(
                    self.model.pair(entity).generator, _adam(cfg.generative), wrt=nnx.Param
                ),
                "critic": nnx.Optimizer(
                    self.model.pair(entity).critic, _adam(cfg.generative), wrt=nnx.Param
                ),
            }
            for entity in ("user", "item")
        }

    @property
    def num_epochs(self) -> int:
        return self.cfg.generative.epochs

    def train_epoch(self, epoch: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for entity in ("user", "item"):
            parts = self._train_pair_epoch(entity)
            metrics.update({f"{entity}/{k}": v for k, v in parts.items()})
        if epoch % max(1, self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
            self.logger.info("epoch %d | %s", epoch, metrics)
        return metrics

    def _train_pair_epoch(self, entity: str) -> dict[str, float]:
        return _pair_epoch(
            self.model.pair(entity),
            self._opts[entity]["gen"],
            self._opts[entity]["critic"],
            self.reals[entity],
            self.cfg.generative,
            self.rng,
            self.next_keys,
        )

    def export(self, output_dir: str | Path, dataset: str) -> Path:
        """Persist the trained generative model for the synthesis stage."""
        path = Path(output_dir) / dataset / f"generative_seed_{self.seed}.npz"
        save_params(
            self.model,
            path,
            latent_dim=self.model.latent_dim,
            noise_dim=self.cfg.generative.noise_dim,
        )
        self.logger.info("exported generative model -> %s", path)
        return path


class JointGenerativeTrainer(Trainer):
    """Train a single WGAN-GP over concatenated ``[z_u; z_i]`` pairs (ablation).

    Mirrors :class:`GenerativeTrainer` but fits one :class:`JointGenerativeModel`
    on the interaction-level concatenated latents instead of two independent
    clouds. ``z_pairs`` is ``(n_interactions, 2 * latent_dim)``; ``latent_dim`` is
    inferred as half its width.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        z_pairs: np.ndarray,
        *,
        run: Any | None = None,
    ) -> None:
        super().__init__(seed=cfg.seed, run=run)
        self.cfg = cfg
        if z_pairs.shape[1] % 2 != 0:
            raise ValueError("z_pairs width must be 2 * latent_dim")
        latent_dim = z_pairs.shape[1] // 2

        self.model = JointGenerativeModel(
            latent_dim, cfg.generative, rngs=nnx.Rngs(cfg.seed)
        )
        self.real = jnp.asarray(z_pairs)
        self._opts = {
            "gen": nnx.Optimizer(
                self.model.joint.generator, _adam(cfg.generative), wrt=nnx.Param
            ),
            "critic": nnx.Optimizer(
                self.model.joint.critic, _adam(cfg.generative), wrt=nnx.Param
            ),
        }

    @property
    def num_epochs(self) -> int:
        return self.cfg.generative.epochs

    def train_epoch(self, epoch: int) -> dict[str, float]:
        parts = _pair_epoch(
            self.model.joint,
            self._opts["gen"],
            self._opts["critic"],
            self.real,
            self.cfg.generative,
            self.rng,
            self.next_keys,
        )
        metrics = {f"joint/{k}": v for k, v in parts.items()}
        if epoch % max(1, self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
            self.logger.info("epoch %d | %s", epoch, metrics)
        return metrics

    def export(self, output_dir: str | Path, dataset: str) -> Path:
        """Persist the trained joint generative model for the synthesis stage."""
        path = Path(output_dir) / dataset / f"generative_seed_{self.seed}.npz"
        save_params(
            self.model,
            path,
            latent_dim=self.model.latent_dim,
            noise_dim=self.cfg.generative.noise_dim,
        )
        self.logger.info("exported joint generative model -> %s", path)
        return path


def load_generative_model(
    path: str | Path,
    cfg: ExperimentConfig,
    *,
    seed: int = 0,
) -> GenerativeModel | JointGenerativeModel:
    """Rebuild a generative model and restore its trained parameters.

    Returns a :class:`JointGenerativeModel` when ``cfg.generative.joint`` is set
    (the joint-vs-factorized ablation), otherwise the default factorized
    :class:`GenerativeModel`. The two have different parameter trees, so the cache
    signature over ``cfg.generative`` keeps their checkpoints separate.
    """
    loaded = np.load(path)
    latent_dim = int(loaded["_latent_dim"])
    model: GenerativeModel | JointGenerativeModel
    if cfg.generative.joint:
        model = JointGenerativeModel(latent_dim, cfg.generative, rngs=nnx.Rngs(seed))
    else:
        model = GenerativeModel(latent_dim, cfg.generative, rngs=nnx.Rngs(seed))
    load_params_into(model, path)  # restores params in place
    return model
