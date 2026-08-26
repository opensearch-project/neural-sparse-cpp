# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the disk-resident DiskSeismic index, through SWIG only.

Pins the DiskSeismic-specific contracts: mmap-only reads, the top-k' block
budget, and bit-identical fresh-build vs mmap-reload results.
"""

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import (
    K,
    PAD_DIST,
    PAD_LABEL,
    make_index,
    roundtrip,
    search,
    search_each,
    slice_corpus,
)

SPEC = "disk_seismic,lambda=25|beta=4|alpha=0.4"

# Query knobs. CUT covers every query term (QUERY_NNZ=8); K_PRIME is generous so
# the block budget is not the recall bottleneck.
CUT, K_PRIME = 8, 200

# Calibrated against the session corpus; a floor, not a target.
RECALL_FLOOR = 0.80

# A seeded spec makes the build deterministic, for tests asserting an exact
# relationship (e.g. strict recall improvement across block budgets).
SEEDED = f"{SPEC}|seed=42"


def params(cut=CUT, k_prime=K_PRIME):
    return nsparse.DiskSeismicSearchParameters(cut, k_prime)


@pytest.fixture(scope="module")
def index(corpus):
    return make_index(SPEC, corpus)


@pytest.fixture(scope="module")
def mmap_index(corpus, tmp_path_factory):
    """Seeded + mmap-reloaded, so block-budget tests exercise the on-disk
    (fwd_) path deterministically."""
    index = make_index(SEEDED, corpus)
    path = tmp_path_factory.mktemp("disk_seismic") / "seeded.idx"
    return roundtrip(index, path, nsparse.kUseMmap)


@pytest.mark.parametrize("residency", ["memory", "mmap"])
def test_happy_case(residency, corpus, queries, oracle, tmp_path):
    """factory -> ingest -> build -> query -> accuracy, in both residencies."""
    index = make_index(SPEC, corpus)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    if residency == "mmap":
        # disk_seismic is mmap-only, so the copying flag (0) would raise; the
        # mmap flag is required. The persisted count must survive the reload.
        index = roundtrip(index, tmp_path / "disk_seismic.idx", nsparse.kUseMmap)
        assert index.num_vectors() == corpus.n

    dists, labels = search(index, queries, params=params())
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_fresh_build_matches_mmap_reload(index, queries, tmp_path):
    """The in-RAM build (vectors_) and the mmap reload (inline forward index)
    are two different code paths that must return identical results."""
    p = params()
    before_d, before_l = search(index, queries, params=p)
    reloaded = roundtrip(index, tmp_path / "disk_seismic.idx", nsparse.kUseMmap)
    after_d, after_l = search(reloaded, queries, params=p)
    np.testing.assert_array_equal(after_l, before_l)
    np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)


def test_copying_read_throws(index, tmp_path):
    """disk_seismic is mmap-only: reloading without kUseMmap must raise."""
    path = tmp_path / "disk_seismic.idx"
    nsparse.write_index(index, str(path))
    with pytest.raises(RuntimeError, match="mmap-only"):
        nsparse.read_index(str(path), 0)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_block_budget(index, queries, bad):
    """A non-positive k' (block budget) is rejected at search time."""
    with pytest.raises(ValueError, match="must be positive"):
        search(index, queries, params=params(k_prime=bad))


def test_block_budget_is_monotone(mmap_index, queries, oracle):
    """A larger block budget scores a superset of blocks, so recall never
    regresses and a big enough budget beats the smallest one. Runs on the
    seeded, mmap-reloaded index (deterministic, on-disk block-read path)."""
    want_labels, _ = oracle
    recalls = [
        recall_at_k(
            search(mmap_index, queries, params=params(k_prime=kp))[1], want_labels
        )
        for kp in [1, 4, 16, 64, 256]
    ]
    for lo, hi in zip(recalls, recalls[1:]):
        assert hi >= lo - 1e-9, f"recall regressed across k': {recalls}"
    assert recalls[-1] > recalls[0], "a larger budget must eventually help"


def test_block_budget_saturates(mmap_index, queries):
    """Past the candidate-block count, more budget changes nothing."""
    big_d, big_l = search(mmap_index, queries, params=params(k_prime=10**6))
    bigger_d, bigger_l = search(mmap_index, queries, params=params(k_prime=2 * 10**6))
    np.testing.assert_array_equal(bigger_l, big_l)
    np.testing.assert_array_equal(bigger_d, big_d)


def test_empty_index_roundtrip(queries, tmp_path):
    """An un-built, empty index writes, mmap-reloads, and returns all padding."""
    empty = nsparse.index_factory(queries.dim, SPEC)
    path = tmp_path / "disk_seismic_empty.idx"
    nsparse.write_index(empty, str(path))
    mapped = nsparse.read_index(str(path), nsparse.kUseMmap)
    assert mapped.num_vectors() == 0
    dists, labels = search(mapped, queries, params=params())
    assert (labels == PAD_LABEL).all()
    assert (dists == PAD_DIST).all()


def test_search_before_build(corpus, queries):
    """Searching an added-but-unbuilt index yields empty results, not an error.

    Pinned deliberately: it is a silent-empty footgun, so a change here should
    be a conscious one."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    index.add(corpus.n, corpus.indptr, corpus.indices, corpus.values)
    _, labels = search(index, queries, params=params())
    assert (labels == PAD_LABEL).all()


def test_filtered_search(index, queries, oracle):
    """An id selector larger than k filters results to its members.

    (DiskSeismic omits seismic's exact-match fast path, but still honors the
    selector per candidate doc.)"""
    want_labels, _ = oracle
    allowed = np.ascontiguousarray(
        np.unique(want_labels[want_labels >= 0])[: K * 5], dtype=np.int32
    )
    assert len(allowed) > K

    selector = nsparse.SetIDSelector(allowed)
    p = params()
    p.set_id_selector(selector)

    _, labels = search(index, queries, params=p)
    returned = labels[labels >= 0]
    assert np.isin(returned, allowed).all(), "filter must exclude non-members"


def test_with_id_map(corpus, queries, oracle, doc_ids, tmp_path):
    """idmap over disk_seismic returns the caller's ids, and reloads via mmap
    (the delegate's copying read is unsupported, so the whole idmap must be
    mmap-loaded)."""
    index = make_index(f"idmap,{SPEC}", corpus, ids=doc_ids)
    index = roundtrip(index, tmp_path / "idmap_disk_seismic.idx", nsparse.kUseMmap)

    _, labels = search(index, queries, params=params())
    returned = labels[labels >= 0]
    assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"

    want_labels, _ = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], -1)
    assert recall_at_k(labels, want_external) >= RECALL_FLOOR


def test_batch_matches_single_query(index, queries):
    """Batched (OpenMP-parallel over queries) results must equal the one-at-a-
    time path exactly, or the per-thread dense/visited scratch is leaking."""
    batch_d, batch_l = search(index, queries, params=params())
    single_d, single_l = search_each(index, queries, params=params())
    np.testing.assert_array_equal(batch_l, single_l)
    np.testing.assert_allclose(batch_d, single_d, rtol=0, atol=0)


def test_k_larger_than_corpus(corpus, queries):
    """Short result rows are padded with INVALID_IDX / -1.0, not truncated."""
    small = make_index(SPEC, slice_corpus(corpus, 0, 3))
    k = 10
    dists, labels = search(small, queries, k=k, params=params())
    assert labels.shape == (queries.n, k)
    assert (labels[:, 3:] == PAD_LABEL).all()
    assert (dists[:, 3:] == PAD_DIST).all()


def test_seeded_build_is_reproducible(corpus, queries):
    """seed= makes the build (and so the search results) reproducible."""
    seeded = f"{SPEC}|seed=42"
    first = search(make_index(seeded, corpus), queries, params=params())
    second = search(make_index(seeded, corpus), queries, params=params())
    np.testing.assert_array_equal(second[1], first[1])
    np.testing.assert_array_equal(second[0], first[0])
