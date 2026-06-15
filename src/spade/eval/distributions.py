"""Distributional alignment: latent 2-Wasserstein and KS degree distance.

Two complementary sanity checks on synthesis fidelity:

* **Latent W₂.** How far the synthetic latent cloud sits from the real one,
  measured by the 2-Wasserstein distance under a Gaussian approximation (the
  closed-form Bures metric)::

      W₂² = ||μx - μy||² + Tr(Σx + Σy - 2 (Σx^{1/2} Σy Σx^{1/2})^{1/2})

  Exact for Gaussians and a stable, cheap surrogate otherwise; it captures both
  mean shift and covariance mismatch without density estimation. Computed in the
  SPADE Stage I latent space, where the generators are trained to match.

* **KS degree distance.** The two-sample Kolmogorov–Smirnov statistic between the
  real and synthetic degree distributions — a non-parametric check that synthesis
  reproduces the long-tailed popularity/activity structure, not just the global
  sparsity ρ. Reported separately for users and items.

Both return values in ``[0, ∞)`` (W₂) / ``[0, 1]`` (KS) where smaller is closer.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats

from spade.data.interactions import InteractionStore

__all__ = ["gaussian_w2", "ks_distance", "degree_ks"]


def gaussian_w2(x: np.ndarray, y: np.ndarray) -> float:
    """2-Wasserstein distance between two clouds under a Gaussian fit (Bures).

    ``x``/``y`` are ``(n, d)`` samples. Means and covariances are estimated
    empirically; the matrix square roots use ``scipy.linalg.sqrtm`` with the tiny
    imaginary part discarded. Returns ``W₂`` (not squared).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float(np.linalg.norm(x.mean(0) - y.mean(0))) if x.size and y.size else 0.0

    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    mean_term = float(np.sum((mu_x - mu_y) ** 2))

    sqrt_x = _sqrtm_psd(cov_x)
    cross = _sqrtm_psd(sqrt_x @ cov_y @ sqrt_x)
    cov_term = float(np.trace(cov_x + cov_y - 2.0 * cross))
    return float(np.sqrt(max(mean_term + cov_term, 0.0)))


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Real symmetric PSD matrix square root (imaginary dust discarded)."""
    root = linalg.sqrtm(mat)
    if np.iscomplexobj(root):
        root = root.real
    return np.asarray(root, dtype=np.float64)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic between samples ``a`` and ``b``."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(stats.ks_2samp(a, b).statistic)


def degree_ks(real: InteractionStore, synth: InteractionStore) -> dict[str, float]:
    """KS distance between real and synthetic user/item degree distributions."""
    return {
        "ks_user_degree": ks_distance(real.user_degree, synth.user_degree),
        "ks_item_degree": ks_distance(real.item_degree, synth.item_degree),
    }
