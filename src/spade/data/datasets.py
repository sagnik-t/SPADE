"""Dataset loaders that parse raw sources into a common interaction format.

Every loader returns a :class:`RawInteractions` of raw (un-indexed) user ids,
item ids, ratings, and timestamps. Downstream code (:mod:`spade.data.interactions`)
is responsible for filtering and contiguous indexing, so loaders stay thin and
dataset-specific parsing lives in one place.

MovieLens is the primary benchmark family (as in GANRS and most synthetic-CF
work), so ML-100K and ML-1M are fully implemented with on-demand download.
``amazon`` is registered but deferred: the exact Reviews subset/core filter is
not yet pinned, and it sets the observed sparsity rho that drives Stage III's
candidate count, so it must be chosen deliberately rather than guessed.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "RawInteractions",
    "load_dataset",
    "available_datasets",
    "DATASET_REGISTRY",
]

_MOVIELENS_BASE = "https://files.grouplens.org/datasets/movielens"


@dataclass(frozen=True)
class RawInteractions:
    """Raw, un-indexed interactions in a common columnar layout.

    Ids are whatever the source uses (arbitrary integers); contiguous indexing
    happens later in :func:`spade.data.interactions.build_store`. All four arrays
    share the same length (one entry per observed interaction).
    """

    users: np.ndarray       # raw user ids, shape (n_obs,)
    items: np.ndarray       # raw item ids, shape (n_obs,)
    ratings: np.ndarray     # explicit ratings (float), shape (n_obs,)
    timestamps: np.ndarray  # unix timestamps (int64); zeros if unavailable
    name: str

    def __post_init__(self) -> None:
        n = len(self.users)
        if not (len(self.items) == len(self.ratings) == len(self.timestamps) == n):
            raise ValueError("users/items/ratings/timestamps must share a length")

    @property
    def n_obs(self) -> int:
        return len(self.users)

    def __len__(self) -> int:
        return self.n_obs


def _download_bytes(url: str) -> bytes:
    """Fetch ``url`` and return its body. Imported lazily to keep import cheap."""
    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _ensure_file(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` once; reuse the cached copy on later calls."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_download_bytes(url))
    return dest


def _load_ml_100k(data_dir: Path, download: bool) -> RawInteractions:
    """ML-100K: a single tab-separated ``u.data`` of user\titem\trating\ttimestamp."""
    raw_path = data_dir / "ml-100k" / "u.data"
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(
                f"{raw_path} missing and download=False; fetch ml-100k.zip manually."
            )
        archive = _download_bytes(f"{_MOVIELENS_BASE}/ml-100k.zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open("ml-100k/u.data") as src:
                raw_path.write_bytes(src.read())
    arr = np.loadtxt(raw_path, delimiter="\t", dtype=np.int64)
    return RawInteractions(
        users=arr[:, 0].astype(np.int64),
        items=arr[:, 1].astype(np.int64),
        ratings=arr[:, 2].astype(np.float32),
        timestamps=arr[:, 3].astype(np.int64),
        name="ml-100k",
    )


def _load_ml_1m(data_dir: Path, download: bool) -> RawInteractions:
    """ML-1M: ``ratings.dat`` with ``user::item::rating::timestamp`` rows."""
    raw_path = data_dir / "ml-1m" / "ratings.dat"
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(
                f"{raw_path} missing and download=False; fetch ml-1m.zip manually."
            )
        archive = _download_bytes(f"{_MOVIELENS_BASE}/ml-1m.zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open("ml-1m/ratings.dat") as src:
                raw_path.write_bytes(src.read())
    # "::" separated; np.loadtxt can't take multi-char delimiters, so split by hand.
    rows = np.array(
        [line.split("::") for line in raw_path.read_text().splitlines() if line],
        dtype=np.int64,
    )
    return RawInteractions(
        users=rows[:, 0].astype(np.int64),
        items=rows[:, 1].astype(np.int64),
        ratings=rows[:, 2].astype(np.float32),
        timestamps=rows[:, 3].astype(np.int64),
        name="ml-1m",
    )


def _load_amazon(data_dir: Path, download: bool) -> RawInteractions:
    """Deferred: the exact Amazon Reviews subset and k-core are not yet pinned."""
    raise NotImplementedError(
        "The 'amazon' loader is registered but not configured. MovieLens is the "
        "primary benchmark; the Amazon Reviews subset (e.g. Movies & TV 5-core) "
        "and its core filter must be pinned first because they set the observed "
        "sparsity rho that drives the Stage III candidate count. Once chosen, "
        "implement parsing here to return a RawInteractions."
    )


# name -> loader(data_dir, download) -> RawInteractions
DATASET_REGISTRY: dict[str, Callable[[Path, bool], RawInteractions]] = {
    "ml-100k": _load_ml_100k,
    "ml-1m": _load_ml_1m,
    "amazon": _load_amazon,
}


def available_datasets() -> list[str]:
    """Return the registered dataset names."""
    return sorted(DATASET_REGISTRY)


def load_dataset(
    name: str,
    data_dir: str | Path = "data",
    download: bool = True,
) -> RawInteractions:
    """Load a registered dataset into a :class:`RawInteractions`.

    Parameters
    ----------
    name:
        One of :func:`available_datasets` (``ml-100k``, ``ml-1m``, ``amazon``).
    data_dir:
        Root directory for cached raw files; created on demand.
    download:
        If ``True``, fetch the raw archive when it is not already cached.
    """
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(available_datasets())}"
        )
    return DATASET_REGISTRY[key](Path(data_dir), download)
