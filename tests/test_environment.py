"""Environment / hardware checks (CPU + GPU) for the JAX stack and faiss.

These are intentionally NOT part of the default test run. They verify that the
installed JAX / Flax / Optax build can place and execute computation on each
available device, and that faiss imports and runs. They depend on the full
(heavy) dependency set being installed, and the GPU cases require a CUDA box;
when no GPU is present those cases skip rather than fail.

Run them explicitly::

    pytest -m env             # all environment checks
    pytest -m env -k gpu      # just the GPU checks

The default ``pytest`` excludes them via ``addopts = -m 'not env'`` in pyproject.
All heavy imports are done inside the test bodies so the default suite can still
collect this file even when jax/flax/optax/faiss are not installed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.env


def _jax_gpus():
    """Return available JAX GPU devices, or [] if none/unsupported."""
    import jax

    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


# --- JAX ---------------------------------------------------------------------

def test_jax_cpu_compute():
    import jax
    import jax.numpy as jnp

    assert jax.devices("cpu"), "no CPU device reported by JAX"
    x = jax.device_put(jnp.arange(5.0), jax.devices("cpu")[0])
    assert float((x * 2).sum()) == 20.0


def test_jax_gpu_compute():
    import jax
    import jax.numpy as jnp

    gpus = _jax_gpus()
    if not gpus:
        pytest.skip("no JAX GPU device available")
    x = jax.device_put(jnp.arange(5.0), gpus[0])
    assert float((x * 2).sum()) == 20.0
    assert next(iter(x.devices())).platform == "gpu"


# --- Flax nnx ----------------------------------------------------------------

def _run_linear(device):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    with jax.default_device(device):
        model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
        out = model(jnp.ones((3, 4)))
        out.block_until_ready()
    return out


def test_flax_nnx_forward_cpu():
    import jax

    out = _run_linear(jax.devices("cpu")[0])
    assert out.shape == (3, 2)


def test_flax_nnx_forward_gpu():
    gpus = _jax_gpus()
    if not gpus:
        pytest.skip("no JAX GPU device available")
    out = _run_linear(gpus[0])
    assert out.shape == (3, 2)
    assert next(iter(out.devices())).platform == "gpu"


# --- Optax -------------------------------------------------------------------

def _optax_step(device):
    import jax
    import jax.numpy as jnp
    import optax

    with jax.default_device(device):
        params = {"w": jnp.ones((3,))}
        tx = optax.sgd(0.1)
        state = tx.init(params)
        updates, _ = tx.update({"w": jnp.ones((3,))}, state, params)
        params = optax.apply_updates(params, updates)
    return params


def test_optax_step_cpu():
    import jax

    params = _optax_step(jax.devices("cpu")[0])
    assert float(params["w"][0]) == pytest.approx(0.9)


def test_optax_step_gpu():
    gpus = _jax_gpus()
    if not gpus:
        pytest.skip("no JAX GPU device available")
    params = _optax_step(gpus[0])
    assert float(params["w"][0]) == pytest.approx(0.9)


# --- faiss (CPU-only by design: we install faiss-cpu) ------------------------

def test_faiss_cpu_index_search():
    import faiss
    import numpy as np

    d = 8
    xb = np.random.RandomState(0).rand(16, d).astype("float32")
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, ids = index.search(xb[:3], 4)
    assert ids.shape == (3, 4)
    # Under L2, the nearest neighbour of each vector is itself (distance 0).
    assert (ids[:, 0] == np.arange(3)).all()
