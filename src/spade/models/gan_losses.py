"""Stage II objectives: WGAN-GP critic/generator losses + moment matching.

The two latent generators are trained adversarially with the improved
Wasserstein GAN objective (WGAN-GP). For real samples ``x`` (a frozen Stage I
latent cloud) and generated samples ``g = G(z)``:

* **Critic** maximizes the score gap ``E[C(x)] - E[C(g)]`` (a Wasserstein-1
  estimate) subject to a soft 1-Lipschitz constraint. We minimize
  ``E[C(g)] - E[C(x)] + gp_lambda * GP``, where ``GP`` is the squared deviation
  of the critic's input-gradient norm from 1, evaluated at random interpolates
  between real and generated points (Gulrajani et al., 2017).
* **Generator** minimizes ``-E[C(g)]`` plus a **moment-matching** regularizer
  that aligns the first two moments (mean and covariance) of the generated and
  real clouds. The regularizer stabilizes the adversarial game and directly
  encodes the paper's requirement that synthetic latents reproduce the empirical
  distribution's location and spread.

The gradient penalty uses JAX's grad-of-grad: ``jax.grad`` over the critic input
inside a loss that is itself differentiated w.r.t. the critic parameters. All
functions are pure (sampling happens in the training loop) so they compose
inside a jitted step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "gradient_penalty",
    "critic_loss",
    "moment_matching_loss",
    "generator_loss",
]


def gradient_penalty(
    critic: nnx.Module,
    real: jnp.ndarray,
    fake: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Two-sided gradient penalty on critic interpolates ``(real <-> fake)``.

    Samples a per-example mixing coefficient ``eps ~ U[0, 1]``, forms the
    interpolate ``eps*real + (1-eps)*fake``, and penalizes the squared deviation
    of the critic's input-gradient L2 norm from 1. Returns a scalar mean.
    """
    eps = jax.random.uniform(key, (real.shape[0], 1))
    interp = eps * real + (1.0 - eps) * fake
    grads = jax.grad(lambda x: critic(x).sum())(interp)  # (batch, latent_dim)
    norms = jnp.sqrt(jnp.sum(grads**2, axis=-1) + 1e-12)
    return jnp.mean((norms - 1.0) ** 2)


def critic_loss(
    critic: nnx.Module,
    real: jnp.ndarray,
    fake: jnp.ndarray,
    key: jax.Array,
    gp_lambda: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """WGAN-GP critic loss and parts (loss, Wasserstein estimate, penalty).

    ``fake`` is a detached batch of generated samples (the generator is not a
    differentiation target here). Minimizing the returned loss maximizes the
    real-minus-fake score gap while keeping the critic ~1-Lipschitz.
    """
    real_score = critic(real).mean()
    fake_score = critic(fake).mean()
    wasserstein = real_score - fake_score
    gp = gradient_penalty(critic, real, fake, key)
    loss = -wasserstein + gp_lambda * gp
    parts = {"critic_loss": loss, "wasserstein": wasserstein, "gp": gp}
    return loss, parts


def _covariance(x: jnp.ndarray) -> jnp.ndarray:
    """Sample covariance ``(d, d)`` of rows in ``x`` ``(n, d)`` (n>=2)."""
    centered = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    return (centered.T @ centered) / jnp.maximum(n - 1, 1)


def moment_matching_loss(fake: jnp.ndarray, real: jnp.ndarray) -> jnp.ndarray:
    """Squared mean difference plus squared covariance difference.

    Both terms use summed squared (Frobenius) error so the regularizer scales
    with dimensionality consistently and is invariant to batch order.
    """
    mean_term = jnp.sum((fake.mean(axis=0) - real.mean(axis=0)) ** 2)
    cov_term = jnp.sum((_covariance(fake) - _covariance(real)) ** 2)
    return mean_term + cov_term


def generator_loss(
    critic: nnx.Module,
    fake: jnp.ndarray,
    real: jnp.ndarray,
    moment_lambda: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Generator loss: fool the critic plus a moment-matching regularizer.

    ``fake = G(noise)`` must be produced inside the differentiated function so
    gradients flow to the generator; ``real`` is a fixed batch from the target
    cloud used only for the moment statistics.
    """
    adversarial = -critic(fake).mean()
    moment = moment_matching_loss(fake, real)
    loss = adversarial + moment_lambda * moment
    parts = {"gen_loss": loss, "gen_adversarial": adversarial, "moment": moment}
    return loss, parts
