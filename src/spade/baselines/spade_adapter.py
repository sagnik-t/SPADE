"""Adapter exposing SPADE's synthesis pipeline through the common interface.

SPADE is the method under test, not a baseline, but the experiment harness should
treat it like any other generator. This thin wrapper drives a
:class:`SynthesisModel` and packages its output — plus the Stage I latent clouds
for real and synthetic entities — as a :class:`GeneratorOutput`, so SPADE flows
through the exact same geometry/utility evaluation as the baselines.
"""

from __future__ import annotations

import jax
import numpy as np

from spade.baselines.base import Generator, GeneratorOutput, LatentBundle
from spade.config.configs import ExperimentConfig
from spade.data.interactions import InteractionStore
from spade.models.decoder import RatingVocab
from spade.models.generative import GenerativeModel, JointGenerativeModel
from spade.models.representation import RepresentationModel
from spade.synthesis.synthesizer import SynthesisModel

__all__ = ["SpadeGenerator"]


class SpadeGenerator(Generator):
    """Wrap a trained SPADE synthesis pipeline as a :class:`Generator`."""

    name = "spade"

    def __init__(
        self,
        representation: RepresentationModel,
        generative: GenerativeModel | JointGenerativeModel,
        vocab: RatingVocab,
        train: InteractionStore,
        cfg: ExperimentConfig,
    ) -> None:
        self.representation = representation
        self.train_n_users = train.n_users
        self.train_n_items = train.n_items
        self.synth = SynthesisModel(
            representation, generative, vocab,
            source_n_users=train.n_users, source_n_items=train.n_items,
            source_rho=train.rho, cfg=cfg.synthesis,
        )

    def generate(self, key: jax.Array) -> GeneratorOutput:
        dataset = self.synth.synthesize(key)
        z_u_synth, z_i_synth = self.synth.sample_latents(key)
        z_u_real, z_i_real = self.representation.export_embeddings(
            self.train_n_users, self.train_n_items
        )
        latents = LatentBundle(
            real_users=np.asarray(z_u_real), real_items=np.asarray(z_i_real),
            synth_users=np.asarray(z_u_synth), synth_items=np.asarray(z_i_synth),
        )
        return GeneratorOutput(name=self.name, dataset=dataset, latents=latents)
