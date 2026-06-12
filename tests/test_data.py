"""Tests for the data layer: filtering, indexing, splitting, and sampling.

Uses a small synthetic :class:`RawInteractions` so the suite stays offline and
fast; the real dataset loaders are exercised separately (network-dependent).
"""

import numpy as np
import pytest

from spade.data import (
    RawInteractions,
    apply_kcore,
    available_datasets,
    build_store,
    load_dataset,
    load_splits,
    sample_negatives,
    save_splits,
    split_store,
)
from spade.data.interactions import IndexMap


def _synthetic(n_users=40, n_items=30, density=0.3, seed=0) -> RawInteractions:
    """Dense-ish synthetic interactions with non-contiguous raw ids."""
    rng = np.random.default_rng(seed)
    pairs = set()
    target = int(n_users * n_items * density)
    while len(pairs) < target:
        pairs.add((int(rng.integers(0, n_users)), int(rng.integers(0, n_items))))
    u, i = zip(*pairs, strict=True)
    u = np.array(u) * 10 + 1000        # raw user ids: gappy, offset
    i = np.array(i) * 7 + 500          # raw item ids: gappy, offset
    r = rng.integers(1, 6, size=len(u)).astype(np.float32)
    t = rng.integers(1_000, 9_999, size=len(u)).astype(np.int64)
    return RawInteractions(u, i, r, t, name="synthetic")


# --------------------------------------------------------------------------- #
# Index maps                                                                  #
# --------------------------------------------------------------------------- #
def test_index_map_is_contiguous_and_invertible():
    raw = np.array([1000, 30, 30, 7, 1000])
    m = IndexMap.from_raw(raw)
    assert m.size == 3
    assert sorted(m.raw_ids.tolist()) == [7, 30, 1000]
    enc = m.encode(raw)
    assert enc.min() == 0 and enc.max() == m.size - 1
    # round-trip raw -> index -> raw
    assert m.raw_ids[enc].tolist() == raw.tolist()


# --------------------------------------------------------------------------- #
# k-core filtering                                                            #
# --------------------------------------------------------------------------- #
def test_kcore_drops_low_degree_entities():
    # user 1 has 1 interaction; item 99 appears once -> both should be removed.
    users = np.array([1, 2, 2, 2, 3, 3, 3])
    items = np.array([99, 10, 11, 12, 10, 11, 12])
    keep = apply_kcore(users, items, min_user=2, min_item=2)
    assert keep[0] == False  # noqa: E712 - the (1, 99) interaction
    surviving_users = set(users[keep].tolist())
    assert 1 not in surviving_users and 99 not in items[keep].tolist()


def test_kcore_noop_when_thresholds_trivial():
    users = np.array([1, 1, 2])
    items = np.array([5, 6, 5])
    assert apply_kcore(users, items, 1, 1).all()


def test_build_store_respects_min_interactions():
    raw = _synthetic()
    store = build_store(raw, min_user_interactions=5, min_item_interactions=5)
    assert store.user_degree.min() >= 5
    assert store.item_degree.min() >= 5
    # index space is contiguous and matches degree array lengths
    assert store.user_idx.max() < store.n_users
    assert store.item_idx.max() < store.n_items


# --------------------------------------------------------------------------- #
# Degrees and sparsity                                                        #
# --------------------------------------------------------------------------- #
def test_rho_matches_density():
    raw = _synthetic()
    store = build_store(raw, 5, 5)
    expected = store.nnz / (store.n_users * store.n_items)
    assert store.rho == pytest.approx(expected)
    assert 0.0 < store.rho <= 1.0


def test_degree_sums_equal_nnz():
    store = build_store(_synthetic(), 5, 5)
    assert int(store.user_degree.sum()) == store.nnz
    assert int(store.item_degree.sum()) == store.nnz


# --------------------------------------------------------------------------- #
# Splitting                                                                   #
# --------------------------------------------------------------------------- #
def test_split_partitions_without_overlap():
    store = build_store(_synthetic(), 5, 5)
    s = split_store(store, val_frac=0.1, test_frac=0.1, seed=42)
    total = s.train.nnz + s.val.nnz + s.test.nnz
    assert total == store.nnz

    def pairset(st):
        return set(zip(st.user_idx.tolist(), st.item_idx.tolist(), strict=True))

    tr, va, te = pairset(s.train), pairset(s.val), pairset(s.test)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert tr | va | te == pairset(store)


def test_every_user_present_in_train():
    store = build_store(_synthetic(), 5, 5)
    s = split_store(store, 0.1, 0.1, seed=42)
    assert set(s.train.user_idx.tolist()) == set(range(store.n_users))


def test_split_is_deterministic_in_seed():
    store = build_store(_synthetic(), 5, 5)
    a = split_store(store, 0.1, 0.1, seed=7)
    b = split_store(store, 0.1, 0.1, seed=7)
    c = split_store(store, 0.1, 0.1, seed=8)
    assert np.array_equal(a.test.user_idx, b.test.user_idx)
    assert np.array_equal(a.test.item_idx, b.test.item_idx)
    # A different seed should generally produce a different test partition.
    assert not (
        a.test.nnz == c.test.nnz
        and np.array_equal(a.test.item_idx, c.test.item_idx)
    )


def test_split_rejects_bad_fractions():
    store = build_store(_synthetic(), 5, 5)
    with pytest.raises(ValueError):
        split_store(store, val_frac=0.6, test_frac=0.6)


def test_save_and_load_splits_roundtrip(tmp_path):
    store = build_store(_synthetic(), 5, 5)
    s = split_store(store, 0.1, 0.1, seed=11)
    save_splits(s, tmp_path, "synthetic")
    loaded = load_splits(tmp_path, "synthetic", seed=11)
    assert loaded.seed == 11
    assert loaded.train.n_users == s.train.n_users
    assert np.array_equal(loaded.train.user_idx, s.train.user_idx)
    assert np.array_equal(loaded.test.ratings, s.test.ratings)
    assert loaded.train.rho == pytest.approx(s.train.rho)


def test_load_splits_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path, "synthetic", seed=999)


# --------------------------------------------------------------------------- #
# Negative sampling                                                           #
# --------------------------------------------------------------------------- #
def test_negatives_are_unobserved_and_right_shape():
    store = build_store(_synthetic(), 5, 5)
    rng = np.random.default_rng(0)
    neg = sample_negatives(store, n_neg=4, rng=rng)
    assert neg.shape == (store.nnz, 4)
    observed = set(zip(store.user_idx.tolist(), store.item_idx.tolist(), strict=True))
    for k in range(store.nnz):
        u = int(store.user_idx[k])
        row = neg[k].tolist()
        assert len(set(row)) == 4               # no duplicates within a row
        for j in row:
            assert (u, j) not in observed       # genuinely unobserved
            assert 0 <= j < store.n_items


def test_negative_sampling_is_deterministic_in_rng():
    store = build_store(_synthetic(), 5, 5)
    a = sample_negatives(store, 5, np.random.default_rng(123))
    b = sample_negatives(store, 5, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_negatives_reject_invalid_n_neg():
    store = build_store(_synthetic(), 5, 5)
    with pytest.raises(ValueError):
        sample_negatives(store, 0, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_registry_lists_expected_datasets():
    assert {"ml-100k", "ml-1m", "amazon"} <= set(available_datasets())


def test_amazon_loader_is_deferred():
    with pytest.raises(NotImplementedError):
        load_dataset("amazon", "data", download=False)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("does-not-exist")
