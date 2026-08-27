# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the scalar-quantized DiskSeismic index, through SWIG only.

Pins the DiskSeismicSQ-specific contracts on top of DiskSeismic's: mmap-only
reads, the top-k' block budget, bit-identical fresh-build vs mmap-reload
results, both quantizer widths, and the on-disk size win from storing codes
instead of float.
"""

import os

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

VMIN, VMAX = 0.0, 1.0

# The corpus values span [0.1, 1.0]; the [0, 1] quantizer range covers them.
SPEC = f"disk_seismic_sq,quantizer=8bit|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4"

# Query knobs. CUT covers every query term (QUERY_NNZ=8); K_PRIME is generous so
# the block budget is not the recall bottleneck.
CUT, K_PRIME = 8, 200

# A floor with headroom, not a target: the build RNG is unseeded and 8-bit
# quantization adds noise, so a tight bound would flake.
RECALL_FLOOR = 0.75

# A seeded spec makes the build deterministic, for tests asserting an exact
# relationship (e.g. strict recall improvement across block budgets).
SEEDED = f"{SPEC}|seed=42"


def params(cut=CUT, k_prime=K_PRIME):
    return nsparse.DiskSeismicSQSearchParameters(VMIN, VMAX, cut, k_prime)


@pytest.fixture(scope="module")
def index(corpus):
    return make_index(SPEC, corpus)


@pytest.fixture(scope="module")
def mmap_index(corpus, tmp_path_factory):
    """Seeded + mmap-reloaded, so block-budget tests exercise the on-disk
    (fwd_) path deterministically."""
    index = make_index(SEEDED, corpus)
    path = tmp_path_factory.mktemp("disk_seismic_sq") / "seeded.idx"
    return roundtrip(index, path, nsparse.kUseMmap)


@pytest.mark.parametrize("residency", ["memory", "mmap"])
def test_happy_case(residency, corpus, queries, oracle, tmp_path):
    """factory -> ingest -> build -> query -> accuracy, in both residencies."""
    index = make_index(SPEC, corpus)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    if residency == "mmap":
        # disk_seismic_sq is mmap-only, so the copying flag (0) would raise; the
        # mmap flag is required. The persisted count must survive the reload.
        index = roundtrip(index, tmp_path / "disk_seismic_sq.idx", nsparse.kUseMmap)
        assert index.num_vectors() == corpus.n

    dists, labels = search(index, queries, params=params())
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_fresh_build_matches_mmap_reload(index, queries, tmp_path):
    """The in-RAM build (vectors_ codes) and the mmap reload (inline forward
    index codes) are two different code paths that must return identical
    results -- same clusters, same blocks, same integer dots, same decode."""
    p = params()
    before_d, before_l = search(index, queries, params=p)
    reloaded = roundtrip(index, tmp_path / "disk_seismic_sq.idx", nsparse.kUseMmap)
    after_d, after_l = search(reloaded, queries, params=p)
    np.testing.assert_array_equal(after_l, before_l)
    np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)


def test_copying_read_throws(index, tmp_path):
    """disk_seismic_sq is mmap-only: reloading without kUseMmap must raise."""
    path = tmp_path / "disk_seismic_sq.idx"
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


def test_empty_index_roundtrip(queries, tmp_path):
    """An un-built, empty index writes (quantizer header included), mmap-reloads,
    and returns all padding."""
    empty = nsparse.index_factory(queries.dim, SPEC)
    path = tmp_path / "disk_seismic_sq_empty.idx"
    nsparse.write_index(empty, str(path))
    mapped = nsparse.read_index(str(path), nsparse.kUseMmap)
    assert mapped.num_vectors() == 0
    dists, labels = search(mapped, queries, params=params())
    assert (labels == PAD_LABEL).all()
    assert (dists == PAD_DIST).all()


def test_with_id_map(corpus, queries, oracle, doc_ids, tmp_path):
    """idmap over disk_seismic_sq returns the caller's ids, and reloads via mmap
    (the delegate's copying read is unsupported, so the whole idmap must be
    mmap-loaded)."""
    index = make_index(f"idmap,{SPEC}", corpus, ids=doc_ids)
    index = roundtrip(index, tmp_path / "idmap_disk_seismic_sq.idx", nsparse.kUseMmap)

    _, labels = search(index, queries, params=params())
    returned = labels[labels >= 0]
    assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"

    want_labels, _ = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], -1)
    assert recall_at_k(labels, want_external) >= RECALL_FLOOR


@pytest.mark.parametrize("width", ["8bit", "16bit"])
def test_quantizer_widths(corpus, queries, oracle, width):
    """Both quantizer widths clear the recall floor."""
    spec = f"disk_seismic_sq,quantizer={width}|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4"
    idx = make_index(spec, corpus)
    _, labels = search(idx, queries, params=params())
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


@pytest.mark.parametrize("width", ["8bit", "16bit"])
def test_smaller_than_plain_disk_seismic(corpus, tmp_path, width):
    """The whole point of quantization: codes shrink the on-disk index versus
    the float disk_seismic, for the same corpus and cluster params. Both widths
    must win, so a 16-bit-specific layout regression can't slip through."""
    float_index = make_index("disk_seismic,lambda=25|beta=4|alpha=0.4|seed=42", corpus)
    float_path = tmp_path / "disk_seismic_float.idx"
    nsparse.write_index(float_index, str(float_path))

    sq_spec = f"disk_seismic_sq,quantizer={width}|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4|seed=42"
    sq_index = make_index(sq_spec, corpus)
    sq_path = tmp_path / f"disk_seismic_sq_{width}.idx"
    nsparse.write_index(sq_index, str(sq_path))

    assert os.path.getsize(sq_path) < os.path.getsize(float_path)


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


def test_query_range_override(index, queries):
    """A DiskSeismicSQSearchParameters with a non-index range re-encodes the
    query at that range (the query_quantizer override). The corpus values span
    [0.1, 1.0]; vmax=0.5 re-scales/clips the query, so it must actually change
    the decoded scores versus the default range -- if query_quantizer ignored
    the override this would silently no-op and still pass a shape-only check."""
    default_d, _ = search(index, queries, params=params())

    clipped = nsparse.DiskSeismicSQSearchParameters(0.0, 0.5, CUT, K_PRIME)
    clipped_d, clipped_l = search(index, queries, params=clipped)

    # Contract still holds.
    assert clipped_l.shape == (queries.n, K)
    assert clipped_l[clipped_l >= 0].size > 0, "clipping must not empty every result"
    # The override must take effect: a different query range changes the decode
    # scale (and the clipped encoding), so the decoded scores must differ.
    assert not np.allclose(clipped_d, default_d), "query-range override had no effect"


def test_search_before_build(corpus, queries):
    """Searching an added-but-unbuilt index yields empty results, not an error.

    Pinned deliberately (like the disk_seismic sibling): it is a silent-empty
    footgun, so a change here should be a conscious one."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    index.add(corpus.n, corpus.indptr, corpus.indices, corpus.values)
    _, labels = search(index, queries, params=params())
    assert (labels == PAD_LABEL).all()


@pytest.mark.parametrize("residency", ["memory", "mmap"])
def test_filtered_search(corpus, queries, oracle, residency, tmp_path):
    """An id selector larger than k filters results to its members, on both the
    in-RAM (vectors_) and mmap (fwd_) scoring paths.

    (DiskSeismic omits seismic's exact-match fast path, but still honors the
    selector per candidate doc.)"""
    idx = make_index(SPEC, corpus)
    if residency == "mmap":
        idx = roundtrip(idx, tmp_path / "disk_seismic_sq_filtered.idx", nsparse.kUseMmap)

    want_labels, _ = oracle
    allowed = np.ascontiguousarray(
        np.unique(want_labels[want_labels >= 0])[: K * 5], dtype=np.int32
    )
    assert len(allowed) > K

    selector = nsparse.SetIDSelector(allowed)
    p = params()
    p.set_id_selector(selector)

    _, labels = search(idx, queries, params=p)
    returned = labels[labels >= 0]
    assert np.isin(returned, allowed).all(), "filter must exclude non-members"


def test_factory_downcast_exposes_quantizer(corpus):
    """DSSQ is in the %factory downcast lists, so index_factory returns the
    concrete type and the DSSQ-only get_scalar_quantizer() is reachable (it is
    not a base Index method). If the downcast were missing, idx would be a base
    Index proxy and the attribute access below would raise."""

    def qtype(width):
        spec = f"disk_seismic_sq,quantizer={width}|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4"
        idx = nsparse.index_factory(corpus.dim, spec)
        return idx.get_scalar_quantizer().get_quantizer_type()

    assert qtype("8bit") != qtype("16bit"), "widths must report distinct types"


def test_seeded_build_is_reproducible(corpus, queries):
    """seed= makes the build (and so the search results) reproducible."""
    first = search(make_index(SEEDED, corpus), queries, params=params())
    second = search(make_index(SEEDED, corpus), queries, params=params())
    np.testing.assert_array_equal(second[1], first[1])
    np.testing.assert_array_equal(second[0], first[0])
