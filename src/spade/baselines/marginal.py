"""Marginal-matching generator — independent first-order structure only.

Reproduces the three one-dimensional marginals of the real data without any joint
or latent structure: user activity (degree distribution), item popularity, and
the rating distribution. Synthetic user degrees are drawn from the empirical
degree distribution scaled by the item-expansion ratio ``beta`` (so density lands
on ``rho``); each user's items are then drawn according to a popularity profile
sampled from the empirical item-degree distribution. Because the three axes are
sampled independently, this baseline isolates how much of a metric is explained by
marginals alone — its geometry latents are ``None``.
"""

from __future__ import annotations

import jax
import numpy as np

from spade.baselines.base import (
    BaselineGenerator,
    GeneratorOutput,
    RatingSampler,
    assemble_dataset,
    rng_from_key,
    synthetic_sizes,
)
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore

__all__ = ["MarginalMatchingGenerator"]


class MarginalMatchingGenerator(BaselineGenerator):
    """Match user-activity, item-popularity, and rating marginals independently."""

    name = "marginal"

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None:
        self.n_users, self.n_items = synthetic_sizes(
            train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
        )
        self.beta = cfg.synthesis.beta
        self.real_user_degrees = train.user_degree.astype(np.float64)
        self.real_item_degrees = train.item_degree.astype(np.float64)
        self.ratings = RatingSampler(train.ratings)

    def generate(self, key: jax.Array) -> GeneratorOutput:
        rng = rng_from_key(key)

        # Synthetic per-user degrees: resample real degrees, scale to the wider
        # item universe so the global density matches rho, clamp to [0, I'].
        sampled = rng.choice(self.real_user_degrees, size=self.n_users, replace=True)
        degrees = np.rint(sampled * self.beta).astype(np.int64)
        degrees = np.clip(degrees, 0, self.n_items)

        # Synthetic item popularity profile from the empirical item-degree law.
        # +1 smoothing keeps every item drawable, so a user's distinct-item draw
        # never starves even when sampled popularities include zeros.
        pop = rng.choice(self.real_item_degrees, size=self.n_items, replace=True) + 1.0
        item_probs = pop / pop.sum()

        users, items = [], []
        for u in range(self.n_users):
            d = int(degrees[u])
            if d == 0:
                continue
            chosen = rng.choice(self.n_items, size=d, replace=False, p=item_probs)
            users.append(np.full(d, u, dtype=np.int64))
            items.append(chosen.astype(np.int64))

        if users:
            user_idx = np.concatenate(users)
            item_idx = np.concatenate(items)
        else:
            user_idx = np.empty(0, dtype=np.int64)
            item_idx = np.empty(0, dtype=np.int64)
        ratings = self.ratings.sample(rng, user_idx.shape[0])

        dataset = assemble_dataset(
            user_idx, item_idx, ratings, self.n_users, self.n_items
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=None)
