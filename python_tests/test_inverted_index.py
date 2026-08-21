# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the inverted index.

Exact index, so accuracy is asserted against the oracle outright rather than
through a recall floor.
"""

import numpy as np
import pytest

import nsparse
from oracle import assert_exact
from support import (
    K,
    PAD_DIST,
    PAD_LABEL,
    Corpus,
    make_index,
    roundtrip,
    search,
    search_each,
    slice_corpus,
)

SPEC = "inverted"


@pytest.fixture(scope="module")
def index(corpus):
    return make_index(SPEC, corpus)


@pytest.mark.parametrize("residency", ["memory", "mmap"])
def test_happy_case(residency, corpus, queries, oracle, tmp_path):
    """factory -> ingest -> build -> query -> accuracy, in both residencies."""
    index = make_index(SPEC, corpus)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    if residency == "mmap":
        index = roundtrip(index, tmp_path / "inverted.idx", nsparse.kUseMmap)
        assert index.num_vectors() == corpus.n

    dists, labels = search(index, queries)
    assert labels.shape == (queries.n, K)
    want_labels, want_dists = oracle
    assert_exact(labels, dists, want_labels, want_dists)


def test_persistence_roundtrip(index, corpus, queries, tmp_path):
    """Reloading reproduces the results of the index that wrote the file."""
    before_d, before_l = search(index, queries)
    reloaded = roundtrip(index, tmp_path / "inverted.idx")
    assert reloaded.num_vectors() == corpus.n
    after_d, after_l = search(reloaded, queries)
    np.testing.assert_array_equal(after_l, before_l)
    np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)


def test_trailing_empty_document_survives_a_roundtrip(tmp_path):
    """A term-less document counts, and build() dropping it does not change that.

    It leaves no posting entry, so the count cannot be inferred from the lists;
    trailing, because an interior gap is bracketed by the ids around it.
    """
    empty_last = Corpus(
        dim=8,
        n=2,
        indptr=np.array([0, 2, 2], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.uint16),
        values=np.array([1.0, 0.5], dtype=np.float32),
    )
    index = make_index(SPEC, empty_last)
    assert index.num_vectors() == 2
    assert roundtrip(index, tmp_path / "empty_last.idx").num_vectors() == 2


def test_with_id_map(corpus, queries, oracle, doc_ids):
    """idmap returns caller ids in exactly the oracle's order."""
    index = make_index(f"idmap,{SPEC}", corpus, ids=doc_ids)
    dists, labels = search(index, queries)

    want_labels, want_dists = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], PAD_LABEL)
    np.testing.assert_array_equal(labels, want_external)
    np.testing.assert_allclose(dists, want_dists, rtol=1e-5, atol=1e-5)


def test_filtered_search(index, queries):
    """An id selector must restrict results to its members."""
    allowed = np.arange(0, 200, dtype=np.int32)
    selector = nsparse.SetIDSelector(allowed)
    params = nsparse.SearchParameters()
    params.set_id_selector(selector)

    _, labels = search(index, queries, params=params)
    returned = labels[labels >= 0]
    assert len(returned) > 0, "filter must not empty the result set"
    assert np.isin(returned, allowed).all(), "non-members must be excluded"


def test_filtered_search_is_exact(index, queries, corpus):
    """Filtering must return the exact top-k *within* the allowed set."""
    from oracle import brute_force_top_k

    allowed = np.arange(0, 200, dtype=np.int32)
    selector = nsparse.SetIDSelector(allowed)
    params = nsparse.SearchParameters()
    params.set_id_selector(selector)
    _, labels = search(index, queries, params=params)

    subset = slice_corpus(corpus, 0, 200)
    want_labels, _ = brute_force_top_k(subset.csr, queries.csr, corpus.dim, K)
    np.testing.assert_array_equal(labels, want_labels)


def test_excluded_ids(index, queries, oracle):
    """NotIDSelector removes ids. It is not enumerable, so it exercises the
    non-enumerable selector path."""
    want_labels, _ = oracle
    banned = np.ascontiguousarray(
        np.unique(want_labels[want_labels >= 0])[:20], dtype=np.int32
    )
    inner = nsparse.SetIDSelector(banned)
    selector = nsparse.NotIDSelector(inner)
    params = nsparse.SearchParameters()
    params.set_id_selector(selector)

    _, labels = search(index, queries, params=params)
    assert not np.isin(labels[labels >= 0], banned).any()


def test_incremental_add(corpus, queries, oracle):
    """Ingest split across add() calls matches a single-batch ingest."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    bounds = [0, corpus.n // 3, 2 * corpus.n // 3, corpus.n]
    for lo, hi in zip(bounds, bounds[1:]):
        part = slice_corpus(corpus, lo, hi)
        index.add(part.n, part.indptr, part.indices, part.values)
    index.build()

    assert index.num_vectors() == corpus.n
    dists, labels = search(index, queries)
    want_labels, want_dists = oracle
    assert_exact(labels, dists, want_labels, want_dists)


def test_k_larger_than_corpus(corpus, queries):
    """Short result rows are padded, not truncated."""
    small = make_index(SPEC, slice_corpus(corpus, 0, 3))
    dists, labels = search(small, queries, k=K)
    assert labels.shape == (queries.n, K)
    assert (labels[:, 3:] == PAD_LABEL).all()
    assert (dists[:, 3:] == PAD_DIST).all()


def test_batch_matches_single_query(index, queries):
    """Batched search is OpenMP-parallel over queries; it must agree with the
    one-at-a-time path exactly."""
    batch_d, batch_l = search(index, queries)
    single_d, single_l = search_each(index, queries)
    np.testing.assert_array_equal(batch_l, single_l)
    np.testing.assert_array_equal(batch_d, single_d)
