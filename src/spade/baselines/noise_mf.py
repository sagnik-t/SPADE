"""Noise-Perturbed MF generator.

Trains a matrix-factorization model on the real data, then manufactures synthetic
entities by injecting isotropic Gaussian noise into *real* embeddings: each
synthetic user (item) is a randomly chosen real user (item) embedding plus
``N(0, noise_std^2 I)``. Interactions are decoded directly from the perturbed
embeddings — for each synthetic user the top-``C`` items by predicted score
(``C = round(rho * I')`` to match density), with the MF's real rating prediction
snapped to the nearest legal rating.

It has a genuine latent space (the MF embeddings), so it exports a latent bundle:
the real MF embeddings as the real coordinates and the perturbed embeddings as the
synthetic ones, letting the geometry metrics map it into the reference space.
"""

from __future__ import annotations

import jax
import numpy as np

from spade.baselines.base import (
    BaselineGenerator,
    GeneratorOutput,
    LatentBundle,
    RatingSampler,
    assemble_dataset,
    rng_from_key,
    synthetic_sizes,
)
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.eval.reference import train_mf

__all__ = ["NoisePerturbedMFGenerator"]


class NoisePerturbedMFGenerator(BaselineGenerator):
    """MF embeddings perturbed by isotropic Gaussian noise, decoded to interactions."""

    name = "noise_mf"

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None:
        bcfg = cfg.baselines
        self.n_users, self.n_items = synthetic_sizes(
            train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
        )
        self.rho = train.rho
        self.noise_std = bcfg.noise_std
        self.ratings = RatingSampler(train.ratings)
        self.src_users = train.n_users
        self.src_items = train.n_items

        model = train_mf(
            train, dim=bcfg.noise_mf_dim, epochs=bcfg.noise_mf_epochs,
            lr=bcfg.noise_mf_lr, seed=cfg.seed,
        )
        self.p = model.user_table()                  # (U, d)
        self.q = model.item_table()                  # (I, d)
        self.bu = model.user_bias_table()            # (U,)
        self.bi = model.item_bias_table()            # (I,)
        self.g = model.global_bias_value()

    def generate(self, key: jax.Array) -> GeneratorOutput:
        rng = rng_from_key(key)
        d = self.p.shape[1]

        base_u = rng.integers(0, self.src_users, self.n_users)
        base_i = rng.integers(0, self.src_items, self.n_items)
        p_s = self.p[base_u] + rng.normal(0.0, self.noise_std, (self.n_users, d))
        q_s = self.q[base_i] + rng.normal(0.0, self.noise_std, (self.n_items, d))
        bu_s = self.bu[base_u]
        bi_s = self.bi[base_i]

        c = max(1, min(int(round(self.rho * self.n_items)), self.n_items))

        users, items, ratings = [], [], []
        chunk = 512
        for start in range(0, self.n_users, chunk):
            pu = p_s[start : start + chunk]
            scores = pu @ q_s.T + bu_s[start : start + chunk, None] + bi_s[None, :] + self.g
            top = np.argpartition(-scores, kth=c - 1, axis=1)[:, :c]
            rows = np.arange(pu.shape[0])[:, None]
            preds = scores[rows, top]
            for r in range(pu.shape[0]):
                u = start + r
                users.append(np.full(c, u, dtype=np.int64))
                items.append(top[r].astype(np.int64))
                ratings.append(self.ratings.nearest(preds[r]))

        user_idx = np.concatenate(users) if users else np.empty(0, np.int64)
        item_idx = np.concatenate(items) if items else np.empty(0, np.int64)
        rating_arr = np.concatenate(ratings) if ratings else np.empty(0, np.float32)

        dataset = assemble_dataset(
            user_idx, item_idx, rating_arr, self.n_users, self.n_items
        )
        latents = LatentBundle(
            real_users=self.p, real_items=self.q, synth_users=p_s, synth_items=q_s
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=latents)
