# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Shared helpers for the black-box index tests.

These tests exercise only what SWIG exposes, so the dtypes here are the ones
the typemaps demand: int32 indptr, uint16 indices, float32 values, int32 ids.
Passing anything else to add() is not a graceful error -- see
test_error_paths.py.
"""

from dataclasses import dataclass

import numpy as np
import nsparse

K = 10

# Padding written into short result rows. Spelled literally because the C++
# INVALID_IDX sentinel is deliberately not exposed through SWIG.
PAD_LABEL = -1
PAD_DIST = -1.0


@dataclass(frozen=True)
class Corpus:
    dim: int
    n: int
    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray

    @property
    def csr(self):
        return (self.indptr, self.indices, self.values)


def make_corpus(n, dim, nnz, seed):
    """Random sparse rows with exactly `nnz` distinct terms each."""
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (n + 1) * nnz, nnz, dtype=np.int32)
    indices = np.empty(n * nnz, dtype=np.uint16)
    for i in range(n):
        indices[i * nnz : (i + 1) * nnz] = rng.choice(dim, size=nnz, replace=False)
    values = rng.uniform(0.1, 1.0, size=n * nnz).astype(np.float32)
    return Corpus(dim=dim, n=n, indptr=indptr, indices=indices, values=values)


def slice_corpus(corpus, lo, hi):
    """Rows [lo, hi) as a standalone corpus, with indptr rebased to 0."""
    start, end = corpus.indptr[lo], corpus.indptr[hi]
    return Corpus(
        dim=corpus.dim,
        n=hi - lo,
        indptr=np.ascontiguousarray(
            corpus.indptr[lo : hi + 1] - start, dtype=np.int32
        ),
        indices=np.ascontiguousarray(corpus.indices[start:end]),
        values=np.ascontiguousarray(corpus.values[start:end]),
    )


def add_corpus(index, corpus):
    index.add(corpus.n, corpus.indptr, corpus.indices, corpus.values)


def add_corpus_with_ids(index, corpus, ids):
    index.add_with_ids(
        corpus.n, corpus.indptr, corpus.indices, corpus.values, ids
    )


def make_index(spec, corpus, needs_build=True, ids=None):
    """factory -> ingest -> build, the first three lifecycle stages."""
    index = nsparse.index_factory(corpus.dim, spec)
    if ids is None:
        add_corpus(index, corpus)
    else:
        add_corpus_with_ids(index, corpus, ids)
    if needs_build:
        index.build()
    return index


def search(index, queries, k=K, params=None):
    return index.search(queries.n, *queries.csr, k, params)


def search_each(index, queries, k=K, params=None):
    """One search() call per query, so the internal parallel loop sees n=1.

    Used to prove batched (OpenMP-parallel over queries) results match the
    serial path; the per-thread scratch buffers in the search loop are reused
    across queries a thread handles, and this is what catches leakage.
    """
    labels = np.empty((queries.n, k), dtype=np.int64)
    dists = np.empty((queries.n, k), dtype=np.float32)
    for q in range(queries.n):
        lo, hi = queries.indptr[q], queries.indptr[q + 1]
        one_indptr = np.array([0, hi - lo], dtype=np.int32)
        d, l = index.search(
            1,
            one_indptr,
            np.ascontiguousarray(queries.indices[lo:hi]),
            np.ascontiguousarray(queries.values[lo:hi]),
            k,
            params,
        )
        labels[q], dists[q] = l[0], d[0]
    return dists, labels


def roundtrip(index, path, flag=0):
    """Persist and reload through the file-path API users actually have.

    The class-level write_index/read_index are %ignore'd in SWIG; the free
    functions plus kUseMmap are the supported surface, and read_index owns the
    mapping it hands back.
    """
    nsparse.write_index(index, str(path))
    return nsparse.read_index(str(path), flag)


def assert_same_ranking_modulo_ties(got, want):
    """Same scores rank for rank, and the same docs except where equal scores
    leave the order (or, in the last run of ties, the cut at k) up to the
    tie-break. Quantized scores are small integers, so ties are routine and
    which tied doc wins is not part of any format's contract."""
    got_d, got_l = got
    want_d, want_l = want
    np.testing.assert_allclose(got_d, want_d, rtol=1e-6, atol=1e-6)
    for q in range(want_d.shape[0]):
        row = want_d[q]
        start = 0
        for end in range(1, len(row) + 1):
            if end < len(row) and np.isclose(row[end], row[start]):
                continue
            if end < len(row):
                assert sorted(got_l[q, start:end]) == sorted(want_l[q, start:end]), (
                    f"query {q}: tie run at rank {start} holds different docs"
                )
            start = end
