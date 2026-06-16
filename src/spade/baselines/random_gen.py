"""Random interaction generator — the lower-bound baseline.

Places interactions uniformly at random over the synthetic ``U' x I'`` universe,
matched only on global sparsity ``rho`` (and rating marginal). It carries no
user/item structure whatsoever, so it is the floor every structured generator
should clear on every metric; its geometry latents are ``None``.
"""

from __future__ import annotations

import jax

from spade.baselines.base import (
    BaselineGenerator,
    GeneratorOutput,
    RatingSampler,
    assemble_dataset,
    rng_from_key,
    synthetic_sizes,
    target_nnz,
)
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore

__all__ = ["RandomGenerator"]


class RandomGenerator(BaselineGenerator):
    """Uniform user-item interactions with matched density."""

    name = "random"

    def __init__(self, train: InteractionStore, cfg: ExperimentConfig) -> None:
        self.n_users, self.n_items = synthetic_sizes(
            train.n_users, train.n_items, cfg.synthesis.alpha, cfg.synthesis.beta
        )
        self.rho = train.rho
        self.ratings = RatingSampler(train.ratings)

    def generate(self, key: jax.Array) -> GeneratorOutput:
        rng = rng_from_key(key)
        nnz = min(target_nnz(self.rho, self.n_users, self.n_items),
                  self.n_users * self.n_items)
        # Sample distinct flat (user, item) cells, then decode to 2-D indices.
        flat = rng.choice(self.n_users * self.n_items, size=nnz, replace=False)
        user_idx = flat // self.n_items
        item_idx = flat % self.n_items
        ratings = self.ratings.sample(rng, nnz)
        dataset = assemble_dataset(
            user_idx, item_idx, ratings, self.n_users, self.n_items
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=None)
