"""Stage III synthesis: assemble a discrete synthetic dataset (inference only).

:class:`SynthesisModel` is the third composite stage model. Unlike the
representation and generative stages it trains nothing — it *composes* the frozen
Stage I gate/decoder and the Stage II generators into an inference pipeline:

1. **Expand.** Draw ``U' = ceil(alpha * U)`` synthetic user latents and
   ``I' = ceil(beta * I)`` synthetic item latents from the trained generators.
2. **Retrieve.** For each synthetic user, take its top-``C`` candidate items by
   latent similarity (faiss), where ``C = ceil(I' * rho * gamma)`` sizes the
   candidate set to the observed sparsity ``rho`` with an oversampling buffer
   ``gamma``. This keeps scoring sparse instead of ``O(U' x I')``.
3. **Gate.** Score each candidate pair with the frozen interaction gate and keep
   it with its Bernoulli probability — this realizes the sparsity intrinsically.
4. **Decode.** For surviving pairs, sample a discrete rating from the frozen
   categorical decoder and map class indices back to real rating values.

All domain constraints (rating support, entity counts, sparsity bound) are
asserted, never corrected post-hoc. PRNG keys are threaded explicitly; faiss runs
on NumPy while gate/decoder scoring is chunked through JAX to bound memory.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from spade.config.configs import SynthesisConfig
from spade.models.decoder import RatingVocab
from spade.models.generative import GenerativeModel
from spade.models.representation import RepresentationModel
from spade.synthesis.ann import top_c_candidates
from spade.synthesis.dataset import SyntheticDataset
from spade.utils import get_logger

__all__ = ["SynthesisModel"]


class SynthesisModel:
    """Inference-only assembly of the frozen stage models into a synthesizer."""

    def __init__(
        self,
        representation: RepresentationModel,
        generative: GenerativeModel,
        vocab: RatingVocab,
        *,
        source_n_users: int,
        source_n_items: int,
        source_rho: float,
        cfg: SynthesisConfig,
    ) -> None:
        self.representation = representation
        self.generative = generative
        self.vocab = vocab
        self.source_n_users = source_n_users
        self.source_n_items = source_n_items
        self.source_rho = source_rho
        self.cfg = cfg
        self.logger = get_logger(type(self).__name__)

    @property
    def n_synth_users(self) -> int:
        return math.ceil(self.cfg.alpha * self.source_n_users)

    @property
    def n_synth_items(self) -> int:
        return math.ceil(self.cfg.beta * self.source_n_items)

    def candidate_count(self) -> int:
        """``C = ceil(I' * rho * gamma)``, clamped to ``[1, I']``.

        The product is rounded to 9 decimals before the ceil so floating-point
        dust (e.g. ``24.0000000004``) can't flip ``C`` by one across platforms,
        keeping synthesis reproducible.
        """
        product = round(self.n_synth_items * self.source_rho * self.cfg.gamma, 9)
        c = math.ceil(product)
        return max(1, min(c, self.n_synth_items))

    def sample_latents(self, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Re-derive the synthetic user/item latent clouds for a synthesis key.

        Uses the *same* key derivation as :meth:`synthesize` (the first two of
        four splits seed the user and item draws), so calling this with the key a
        dataset was synthesized under reproduces exactly the latents that produced
        it. The evaluation stage needs these clouds to place synthetic entities in
        the reference space and to measure latent Wasserstein distance.
        """
        k_users, k_items, _, _ = jax.random.split(key, 4)
        z_users = self.generative.sample_users(k_users, self.n_synth_users)
        z_items = self.generative.sample_items(k_items, self.n_synth_items)
        return z_users, z_items

    def synthesize(self, key: jax.Array) -> SyntheticDataset:
        """Run the full pipeline and return the discrete synthetic dataset."""
        k_users, k_items, k_gate, k_rating = jax.random.split(key, 4)
        n_u, n_i, c = self.n_synth_users, self.n_synth_items, self.candidate_count()
        self.logger.info("synthesizing U'=%d I'=%d C=%d", n_u, n_i, c)

        z_users = self.generative.sample_users(k_users, n_u)  # (U', d)
        z_items = self.generative.sample_items(k_items, n_i)  # (I', d)

        candidates = top_c_candidates(
            np.asarray(z_users), np.asarray(z_items), c, self.cfg.ann_metric
        )  # (U', c)
        users = np.repeat(np.arange(n_u, dtype=np.int64), candidates.shape[1])
        items = candidates.reshape(-1).astype(np.int64)

        probs = self._gate_probs(z_users, z_items, users, items)
        unif = np.asarray(jax.random.uniform(k_gate, (probs.shape[0],)))
        keep = unif < probs
        kept_users, kept_items = users[keep], items[keep]

        ratings = self._sample_ratings(z_users, z_items, kept_users, kept_items, k_rating)
        synth = SyntheticDataset(
            user_idx=kept_users,
            item_idx=kept_items,
            ratings=ratings,
            n_users=n_u,
            n_items=n_i,
        )
        self._assert_constraints(synth, c)
        self.logger.info("synthesized %s", synth.summary())
        return synth

    # -- frozen-component scoring (chunked) -------------------------------- #
    def _gate_probs(
        self,
        z_users: jnp.ndarray,
        z_items: jnp.ndarray,
        users: np.ndarray,
        items: np.ndarray,
    ) -> np.ndarray:
        gate = self.representation.gate
        out = np.empty(users.shape[0], dtype=np.float32)
        bs = self.cfg.score_batch_size
        for start in range(0, users.shape[0], bs):
            u = jnp.asarray(users[start : start + bs])
            i = jnp.asarray(items[start : start + bs])
            out[start : start + bs] = np.asarray(
                gate.probability(z_users[u], z_items[i])
            )
        return out

    def _sample_ratings(
        self,
        z_users: jnp.ndarray,
        z_items: jnp.ndarray,
        users: np.ndarray,
        items: np.ndarray,
        key: jax.Array,
    ) -> np.ndarray:
        n = users.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)
        decoder = self.representation.decoder
        bs = self.cfg.score_batch_size
        idx = np.empty(n, dtype=np.int64)
        chunk_keys = jax.random.split(key, math.ceil(n / bs))
        for j, start in enumerate(range(0, n, bs)):
            u = jnp.asarray(users[start : start + bs])
            i = jnp.asarray(items[start : start + bs])
            logits = decoder(z_users[u], z_items[i])  # (chunk, n_levels)
            sampled = jax.random.categorical(chunk_keys[j], logits)
            idx[start : start + bs] = np.asarray(sampled)
        return self.vocab.to_value(idx).astype(np.float32)

    # -- intrinsic constraints (asserted, never corrected) ----------------- #
    def _assert_constraints(self, synth: SyntheticDataset, c: int) -> None:
        n_u, n_i = synth.n_users, synth.n_items
        assert n_u == self.n_synth_users and n_i == self.n_synth_items, "entity counts"
        nnz = synth.nnz
        if nnz:
            assert synth.user_idx.min() >= 0 and synth.user_idx.max() < n_u, "user range"
            assert synth.item_idx.min() >= 0 and synth.item_idx.max() < n_i, "item range"
            allowed = set(self.vocab.values.tolist())
            assert set(np.unique(synth.ratings).tolist()) <= allowed, "rating support"
            flat = synth.user_idx.astype(np.int64) * n_i + synth.item_idx
            assert np.unique(flat).shape[0] == nnz, "duplicate (user, item) pairs"
        # Sparsity is bounded by construction: at most C kept per user.
        assert synth.density <= c / n_i + 1e-9, "density exceeds candidate bound"
