# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the batched build, driven only through the SWIG API.

Batching is a build option rather than a separate entry point, so there is
nothing new to wrap: it is reached through the factory description, the same way
lambda and beta are. `inverted_list_batch_size` bounds the build's memory;
adding `batch_file_output_path` streams the index straight to that file instead
of retaining it, which is the path a corpus too large for RAM needs.
"""

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import K, add_corpus, make_index, search

LAMBDA = 25
BETA = 4
ALPHA = 0.4
SEED = 42
BASE = f"lambda={LAMBDA}|beta={BETA}|alpha={ALPHA}|seed={SEED}"

# Calibrated against the session corpus; a floor, not a target.
RECALL_FLOOR = 0.80


def streamed(corpus, out_path, batch_size, kind="seismic"):
    """Build straight to `out_path`; returns nothing, the file is the index."""
    spec = (
        f"{kind},{BASE}|inverted_list_batch_size={batch_size}"
        f"|batch_file_output_path={out_path}"
    )
    index = nsparse.index_factory(corpus.dim, spec)
    add_corpus(index, corpus)
    index.build()
    return str(out_path)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_happy_case(batch_size, corpus, queries, oracle, tmp_path):
    """build -> read back mapped -> query -> accuracy, at several splits."""
    path = streamed(corpus, tmp_path / "batched.idx", batch_size)
    index = nsparse.read_index(path, nsparse.kUseMmap)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    dists, labels = search(index, queries)
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


@pytest.mark.parametrize("kind", ["seismic", "seismic_sq"])
def test_matches_in_memory_build(kind, corpus, tmp_path):
    """At a fixed seed a streamed build is the in-memory build, byte for byte.

    Parametrized over a float and a quantizing index, because the shared build
    only needs the code width -- add() has already encoded the values.
    """
    in_memory = tmp_path / "memory.idx"
    nsparse.write_index(make_index(f"{kind},{BASE}", corpus), str(in_memory))
    batched = streamed(corpus, tmp_path / "batched.idx", 4, kind=kind)

    assert in_memory.read_bytes() == open(batched, "rb").read()


def test_batch_size_alone_leaves_the_index_in_memory(corpus, queries, tmp_path):
    """Without an output path, batching only bounds the build's intermediates.

    The index is still usable in memory and still the same index -- this is the
    path every index type gets from build(), including the disk-resident ones.
    """
    unbatched = make_index(f"seismic,{BASE}", corpus)
    batched = make_index(f"seismic,{BASE}|inverted_list_batch_size=8", corpus)
    assert batched.num_vectors() == corpus.n

    want_d, want_l = search(unbatched, queries)
    got_d, got_l = search(batched, queries)
    np.testing.assert_array_equal(got_l, want_l)
    np.testing.assert_allclose(got_d, want_d, rtol=1e-6, atol=1e-6)


def test_batch_count_is_not_observable(corpus, queries, tmp_path):
    """The split is a memory knob: at a fixed seed it cannot change the results."""
    one = streamed(corpus, tmp_path / "one.idx", 1)
    many = streamed(corpus, tmp_path / "many.idx", 16)

    want_d, want_l = search(nsparse.read_index(one), queries)
    got_d, got_l = search(nsparse.read_index(many), queries)
    np.testing.assert_array_equal(got_l, want_l)
    np.testing.assert_allclose(got_d, want_d, rtol=1e-6, atol=1e-6)


def test_rejects_dimension_smaller_than_corpus(corpus, tmp_path):
    """A term the declared dimension does not cover is an error, not a silent drop."""
    spec = (
        f"seismic,{BASE}|inverted_list_batch_size=4"
        f"|batch_file_output_path={tmp_path / 'bad.idx'}"
    )
    index = nsparse.index_factory(corpus.dim // 2, spec)
    add_corpus(index, corpus)
    with pytest.raises(ValueError):
        index.build()
