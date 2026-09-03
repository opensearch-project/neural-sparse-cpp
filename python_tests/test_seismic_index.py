# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the seismic index, driven only through the SWIG API."""

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import K, PAD_DIST, PAD_LABEL, make_index, roundtrip, search, search_each

SPEC = "seismic,lambda=25|beta=4|alpha=0.4"

# Calibrated against the session corpus; a floor, not a target.
RECALL_FLOOR = 0.80


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
        index = roundtrip(index, tmp_path / "seismic.idx", nsparse.kUseMmap)

    dists, labels = search(index, queries)
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_persistence_roundtrip(index, queries, tmp_path):
    """Reloading an index reproduces the results of the index that wrote it."""
    before_d, before_l = search(index, queries)
    reloaded = roundtrip(index, tmp_path / "seismic.idx")
    after_d, after_l = search(reloaded, queries)
    np.testing.assert_array_equal(after_l, before_l)
    np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)


def test_with_id_map(corpus, queries, oracle, doc_ids):
    """idmap returns the caller's ids, not internal ordinals."""
    index = make_index(f"idmap,{SPEC}", corpus, ids=doc_ids)
    _, labels = search(index, queries)

    returned = labels[labels >= 0]
    assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"

    # Same ranking as the plain index, expressed in caller ids.
    want_labels, _ = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], -1)
    assert recall_at_k(labels, want_external) >= RECALL_FLOOR


def _write_interchange_csr(path, corpus):
    """Corpus as an interchange CSR: int64 header {n, dim, nnz}, int64 indptr,
    int32 indices, float32 values -- the layout nsparse.convert consumes."""
    with open(path, "wb") as out:
        np.array(
            [corpus.n, corpus.dim, corpus.indices.size], dtype=np.int64
        ).tofile(out)
        corpus.indptr.astype(np.int64).tofile(out)
        corpus.indices.astype(np.int32).tofile(out)
        corpus.values.astype(np.float32).tofile(out)


def _write_id_map(path, external_ids):
    """The id-map file read_csr_and_ids reads: int64 count, then int32 ids,
    row-aligned with the CSR."""
    with open(path, "wb") as out:
        np.array([external_ids.size], dtype=np.int64).tofile(out)
        external_ids.astype(np.int32).tofile(out)


def test_id_map_from_csr_and_id_files(corpus, queries, oracle, doc_ids, tmp_path):
    """read_csr_and_ids builds an idmap from a native CSR (borrowed via mmap)
    plus a separate id file -- the memory-saving build path -- and must return
    the caller's external ids, matching the in-RAM add_with_ids path."""
    interchange = tmp_path / "corpus.csr"
    native = tmp_path / "corpus.mcsr"
    id_file = tmp_path / "ids.bin"
    _write_interchange_csr(interchange, corpus)
    nsparse.convert(str(interchange), str(native))
    _write_id_map(id_file, doc_ids)

    index = nsparse.index_factory(corpus.dim, f"idmap,{SPEC}")
    index.read_csr_and_ids(str(native), str(id_file), nsparse.Residency_kMmap)
    index.build()

    _, labels = search(index, queries)
    returned = labels[labels >= 0]
    assert returned.size > 0, "every query should return at least one hit"
    assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"

    want_labels, _ = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], -1)
    assert recall_at_k(labels, want_external) >= RECALL_FLOOR


def test_exact_match(index, queries, oracle):
    """An enumerable selector of size <= k switches search to the exact path.

    seismic_common.h::should_run_exact_match fires when the selector is
    enumerable and its size is <= k, so the result must equal the oracle
    outright rather than merely clearing the recall floor.
    """
    want_labels, want_dists = oracle
    ids = np.ascontiguousarray(want_labels[0][want_labels[0] >= 0], dtype=np.int32)
    assert len(ids) == K, "query 0 must have K positive-scoring docs"

    selector = nsparse.SetIDSelector(ids)
    params = nsparse.SeismicSearchParameters(K, 1.2)
    params.set_id_selector(selector)

    dists, labels = search_one(index, queries, 0, params)
    np.testing.assert_array_equal(labels, want_labels[0])
    np.testing.assert_allclose(dists, want_dists[0], rtol=1e-5, atol=1e-5)


def test_filtered_search(index, queries, oracle):
    """A selector larger than k filters but stays on the approximate path."""
    want_labels, _ = oracle
    allowed = np.ascontiguousarray(
        np.unique(want_labels[want_labels >= 0])[: K * 5], dtype=np.int32
    )
    assert len(allowed) > K

    selector = nsparse.SetIDSelector(allowed)
    params = nsparse.SeismicSearchParameters(K, 1.2)
    params.set_id_selector(selector)

    _, labels = search(index, queries, params=params)
    returned = labels[labels >= 0]
    assert np.isin(returned, allowed).all(), "filter must exclude non-members"


def test_excluded_ids(index, queries, oracle):
    """NotIDSelector removes ids; it is not enumerable, so no exact path."""
    want_labels, _ = oracle
    banned = np.ascontiguousarray(want_labels[0][:5][want_labels[0][:5] >= 0],
                                  dtype=np.int32)

    # Both selectors must outlive the search: NotIDSelector holds a raw pointer
    # to its delegate and does not keep the Python object alive.
    inner = nsparse.SetIDSelector(banned)
    selector = nsparse.NotIDSelector(inner)
    params = nsparse.SeismicSearchParameters(K, 1.2)
    params.set_id_selector(selector)

    _, labels = search_one(index, queries, 0, params)
    assert not np.isin(labels[labels >= 0], banned).any()


def test_incremental_add(corpus, queries, oracle):
    """Ingest split across several add() calls matches a single-batch ingest."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    bounds = [0, corpus.n // 3, 2 * corpus.n // 3, corpus.n]
    for lo, hi in zip(bounds, bounds[1:]):
        sl = slice(corpus.indptr[lo], corpus.indptr[hi])
        index.add(
            hi - lo,
            (corpus.indptr[lo : hi + 1] - corpus.indptr[lo]).astype(np.int32),
            np.ascontiguousarray(corpus.indices[sl]),
            np.ascontiguousarray(corpus.values[sl]),
        )
    index.build()

    assert index.num_vectors() == corpus.n
    _, labels = search(index, queries)
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_k_larger_than_corpus(corpus, queries):
    """Short result rows are padded with INVALID_IDX / -1.0, not truncated."""
    small = make_index(SPEC, _slice_corpus(corpus, 3))
    k = 10
    dists, labels = search(small, queries, k=k)
    assert labels.shape == (queries.n, k)
    assert (labels[:, 3:] == PAD_LABEL).all()
    assert (dists[:, 3:] == PAD_DIST).all()


def test_batch_matches_single_query(index, queries):
    """Batched search is OpenMP-parallel over queries; per-thread scratch is
    reused across the queries a thread handles. Batched results must equal the
    one-query-at-a-time path exactly, or that scratch is leaking between
    queries."""
    batch_d, batch_l = search(index, queries)
    single_d, single_l = search_each(index, queries)
    np.testing.assert_array_equal(batch_l, single_l)
    np.testing.assert_allclose(batch_d, single_d, rtol=0, atol=0)


@pytest.mark.parametrize("heap_factor", [0.7, 1.0, 1.2, 2.0])
def test_search_parameters(index, queries, oracle, heap_factor):
    """heap_factor trades recall for work but must never break the contract."""
    params = nsparse.SeismicSearchParameters(K, heap_factor)
    _, labels = search(index, queries, params=params)
    assert labels.shape == (queries.n, K)
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) > 0.0


@pytest.mark.parametrize(
    "spec",
    [
        "seismic,lambda=10|beta=2|alpha=0.2",
        "seismic,lambda=25|beta=4|alpha=0.4",
        "seismic,lambda=50|beta=8|alpha=0.6",
    ],
)
def test_pruning_parameters(corpus, queries, oracle, spec):
    """Every documented lambda/beta/alpha combination still returns k hits."""
    index = make_index(spec, corpus)
    _, labels = search(index, queries)
    assert (labels[:, 0] >= 0).all()
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) > 0.0


def test_seeded_build_is_reproducible(corpus, queries):
    """seed= makes a build reproducible; omitting it keeps it random."""
    seeded = f"{SPEC}|seed=42"
    first = search(make_index(seeded, corpus), queries)
    second = search(make_index(seeded, corpus), queries)
    np.testing.assert_array_equal(second[1], first[1])
    np.testing.assert_array_equal(second[0], first[0])

    other = search(make_index(f"{SPEC}|seed=43", corpus), queries)
    assert not np.array_equal(other[1], first[1]), "seed must change the build"

    unseeded = [search(make_index(SPEC, corpus), queries)[1] for _ in range(2)]
    assert not np.array_equal(*unseeded), "default must stay nondeterministic"


def test_search_before_build(corpus, queries):
    """Searching an unbuilt index yields empty results rather than an error.

    Pinned deliberately: it is a silent-empty footgun, so a change in this
    behaviour should be a conscious one.
    """
    index = nsparse.index_factory(corpus.dim, SPEC)
    index.add(corpus.n, corpus.indptr, corpus.indices, corpus.values)
    _, labels = search(index, queries)
    assert (labels == PAD_LABEL).all()


# --- helpers -------------------------------------------------------------


def search_one(index, queries, q, params=None, k=K):
    """Search with a single query row, returning 1-D (dists, labels)."""
    lo, hi = queries.indptr[q], queries.indptr[q + 1]
    d, l = index.search(
        1,
        np.array([0, hi - lo], dtype=np.int32),
        np.ascontiguousarray(queries.indices[lo:hi]),
        np.ascontiguousarray(queries.values[lo:hi]),
        k,
        params,
    )
    return d[0], l[0]


def _slice_corpus(corpus, n):
    from support import Corpus

    end = corpus.indptr[n]
    return Corpus(
        dim=corpus.dim,
        n=n,
        indptr=np.ascontiguousarray(corpus.indptr[: n + 1]),
        indices=np.ascontiguousarray(corpus.indices[:end]),
        values=np.ascontiguousarray(corpus.values[:end]),
    )
