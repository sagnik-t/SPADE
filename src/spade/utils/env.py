"""Load environment variables from a ``.env`` file at process start.

Secrets and machine-specific knobs (W&B API key, CUDA device selection, JAX
memory fraction) live in an un-tracked ``.env`` rather than in code or the
dataclass configs. :func:`load_env` reads that file into ``os.environ`` so the
W&B client and JAX pick the values up automatically. It degrades gracefully when
python-dotenv is absent, mirroring the optional-dependency style of the W&B
wrapper, so importing the package never hard-fails on a bare environment.
"""

from __future__ import annotations

from pathlib import Path

from spade.utils.logging import get_logger

__all__ = ["load_env"]


def load_env(path: str | Path | None = None, override: bool = False) -> bool:
    """Load variables from a ``.env`` file into the process environment.

    Parameters
    ----------
    path:
        Explicit path to a ``.env`` file. When ``None``, search upward from the
        current working directory for the nearest ``.env``.
    override:
        If ``True``, values in the file replace existing environment variables;
        by default the existing environment wins (so an exported ``WANDB_API_KEY``
        or CI secret is never clobbered by a stale local file).

    Returns
    -------
    bool
        ``True`` if a file was found and loaded, ``False`` otherwise (missing
        python-dotenv or no ``.env`` present). Callers can treat it as advisory.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        get_logger().warning("python-dotenv not installed; skipping .env loading.")
        return False

    dotenv_path = str(path) if path is not None else find_dotenv(usecwd=True)
    if not dotenv_path:
        return False
    return load_dotenv(dotenv_path, override=override)
