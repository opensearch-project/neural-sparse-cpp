# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the batched build, driven only through the SWIG API.

Batching is a build option rather than a separate entry point, so there is
nothing new to wrap: it is reached through the factory description, the same way
lambda and beta are. `inverted_list_batch_size` splits the term space, and
`batch_file_output_path` is a directory the build may spill windows into --
scratch, not output. build() writes no index and leaves nothing behind; the
index is serialized with write_index, exactly as an unbatched one is.
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


def scratch_dir(tmp_path, name="scratch"):
    """An existing directory for the build to spill into."""
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def batched_index(corpus, tmp_path, batch_size, kind="seismic", name="scratch"):
    """A build whose windows are spilled to scratch, ready to serve or write."""
    spec = (
        f"{kind},{BASE}|inverted_list_batch_size={batch_size}"
        f"|batch_file_output_path={scratch_dir(tmp_path, name)}"
    )
    index = nsparse.index_factory(corpus.dim, spec)
    add_corpus(index, corpus)
    index.build()
    return index


# Batching starts at 2: one window is an ordinary build with nothing to spill.
@pytest.mark.parametrize("batch_size", [2, 4, 32])
def test_happy_case(batch_size, corpus, queries, oracle, tmp_path):
    """build -> write_index -> read back mapped -> query -> accuracy."""
    path = str(tmp_path / "batched.idx")
    nsparse.write_index(batched_index(corpus, tmp_path, batch_size), path)

    index = nsparse.read_index(path, nsparse.kUseMmap)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    dists, labels = search(index, queries)
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


@pytest.mark.parametrize(
    "kind", ["seismic", "seismic_sq", "disk_seismic", "disk_seismic_sq"]
)
def test_matches_in_memory_build(kind, corpus, tmp_path):
    """At a fixed seed a batched build serializes to the in-memory build's file.

    Over all four types in the family: float and quantizing, since the shared
    build only needs the code width (add() has already encoded the values), and
    in-memory and disk-resident, which have different payloads to write from the
    same lists.
    """
    in_memory = tmp_path / "memory.idx"
    nsparse.write_index(make_index(f"{kind},{BASE}", corpus), str(in_memory))

    spilled = tmp_path / "batched.idx"
    nsparse.write_index(batched_index(corpus, tmp_path, 4, kind=kind), str(spilled))

    assert in_memory.read_bytes() == spilled.read_bytes()


def test_scratch_directory_is_left_empty(corpus, tmp_path):
    """The spill is scratch: whatever is left of it goes with the index.

    Its lists stay readable while the index lives, so it is servable and
    writable throughout; when the spill is removed (immediately where a mapped
    file can be unlinked, on release where it cannot) is the platform's business.
    """
    scratch = scratch_dir(tmp_path)
    index = batched_index(corpus, tmp_path, 8)

    assert index.num_vectors() == corpus.n
    assert len(list(scratch.iterdir())) <= 1
    nsparse.write_index(index, str(tmp_path / "out.idx"))
    assert (tmp_path / "out.idx").stat().st_size > 0

    del index
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("kind", ["seismic", "disk_seismic"])
def test_batched_index_is_searchable_after_build(
    kind, corpus, queries, oracle, tmp_path
):
    """build() leaves an index that serves, not an empty object.

    Its posting lists are borrowed from the spill's mapping, so there is no
    reopening by path and they are never copied onto the heap.
    """
    index = batched_index(corpus, tmp_path, 8, kind=kind)
    assert index.num_vectors() == corpus.n

    params = (
        nsparse.DiskSeismicSearchParameters(8, 200)
        if kind == "disk_seismic"
        else None
    )
    _, labels = search(index, queries, params=params)
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_batch_size_alone_leaves_the_index_in_memory(corpus, queries, tmp_path):
    """Without a scratch directory the window count is ignored, not half-applied.

    Splitting the term space with nowhere to spill the windows would leave the
    clustered lists accumulating anyway, for a corpus pass per window, so the
    build runs as one window and produces the same index it always did.
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
    plain = tmp_path / "plain.idx"
    nsparse.write_index(make_index(f"seismic,{BASE}", corpus), str(plain))
    many = tmp_path / "many.idx"
    nsparse.write_index(batched_index(corpus, tmp_path, 16), str(many))

    want_d, want_l = search(nsparse.read_index(str(plain)), queries)
    got_d, got_l = search(nsparse.read_index(str(many)), queries)
    np.testing.assert_array_equal(got_l, want_l)
    np.testing.assert_allclose(got_d, want_d, rtol=1e-6, atol=1e-6)


def test_rejects_dimension_smaller_than_corpus(corpus, tmp_path):
    """A term the declared dimension does not cover is an error, not a silent drop."""
    spec = (
        f"seismic,{BASE}|inverted_list_batch_size=4"
        f"|batch_file_output_path={scratch_dir(tmp_path)}"
    )
    index = nsparse.index_factory(corpus.dim // 2, spec)
    add_corpus(index, corpus)
    with pytest.raises(ValueError):
        index.build()


def test_rejects_a_scratch_path_that_is_not_a_directory(corpus, tmp_path):
    """Somewhere to spill is the caller's to provide."""
    not_a_dir = tmp_path / "regular-file"
    not_a_dir.write_bytes(b"")
    spec = (
        f"seismic,{BASE}|inverted_list_batch_size=4"
        f"|batch_file_output_path={not_a_dir}"
    )
    index = nsparse.index_factory(corpus.dim, spec)
    add_corpus(index, corpus)
    with pytest.raises(ValueError):
        index.build()
