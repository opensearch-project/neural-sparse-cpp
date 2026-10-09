# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Shared black-box contract for the disk-resident SEISMIC indexes.

DiskSeismicIndex (float) and DiskSeismicScalarQuantizedIndex (codes) present the
same public contract -- mmap-only reads, the top-k' block budget, bit-identical
fresh-build vs mmap-reload results -- so those tests live here once. A suite
subclasses `DiskSeismicContract`, sets SPEC / SEEDED / RECALL_FLOOR, and
implements `params()`; the quantized suite adds its own quantization tests.
"""

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import (
    K,
    PAD_DIST,
    PAD_LABEL,
    assert_same_ranking_modulo_ties,
    make_index,
    roundtrip,
    search,
    search_each,
    slice_corpus,
)


class DiskSeismicContract:
    # Subclasses set these.
    SPEC: str
    SEEDED: str  # SPEC + a fixed seed, for the determinism-sensitive tests
    RECALL_FLOOR: float

    # CUT covers every query term (QUERY_NNZ=8); K_PRIME is generous so the
    # block budget is not the recall bottleneck.
    CUT = 8
    K_PRIME = 200

    # Components a truncated index keeps in each doc's extra inline copies. The
    # session corpus has 30 per doc, so this cuts most copies to under a third.
    INLINE_MAX_NNZ = 8

    def params(self, cut=None, k_prime=None, rescore=None):
        """The index's SearchParameters. Subclass provides the concrete type;
        None means "use the default", so k_prime=0 / rescore=-1 are passed
        through (not coalesced) for the rejection tests, and rescore=None
        leaves the library default in place."""
        raise NotImplementedError

    def truncated(self, corpus, path, limit=None):
        """A seeded index built with inline_max_nnz and mmap-reloaded:
        truncation exists only in the serialized form, so a fresh in-RAM build
        never exercises it."""
        limit = self.INLINE_MAX_NNZ if limit is None else limit
        spec = f"{self.SEEDED}|inline_max_nnz={limit}"
        return roundtrip(make_index(spec, corpus), path, nsparse.kUseMmap)

    @pytest.mark.parametrize("residency", ["memory", "mmap"])
    def test_happy_case(self, residency, corpus, queries, oracle, tmp_path):
        """factory -> ingest -> build -> query -> accuracy, in both residencies."""
        index = make_index(self.SPEC, corpus)
        assert index.num_vectors() == corpus.n
        assert index.get_dimension() == corpus.dim

        if residency == "mmap":
            # disk-resident indexes are mmap-only, so the copying flag (0) would
            # raise; the mmap flag is required, and the count must survive.
            index = roundtrip(index, tmp_path / "index.idx", nsparse.kUseMmap)
            assert index.num_vectors() == corpus.n

        dists, labels = search(index, queries, params=self.params())
        assert labels.shape == (queries.n, K)
        assert dists.shape == (queries.n, K)
        assert (labels[:, 0] >= 0).all(), "every query must return at least one hit"

        want_labels, _ = oracle
        assert recall_at_k(labels, want_labels) >= self.RECALL_FLOOR

    def test_fresh_build_matches_mmap_reload(self, corpus, queries, tmp_path):
        """The in-RAM build (vectors_) and the mmap reload (inline forward index)
        are two different code paths that must return identical results."""
        index = make_index(self.SPEC, corpus)
        p = self.params()
        before_d, before_l = search(index, queries, params=p)
        reloaded = roundtrip(index, tmp_path / "index.idx", nsparse.kUseMmap)
        after_d, after_l = search(reloaded, queries, params=p)
        np.testing.assert_array_equal(after_l, before_l)
        np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)

    def test_copying_read_throws(self, corpus, tmp_path):
        """mmap-only: reloading without kUseMmap must raise."""
        index = make_index(self.SPEC, corpus)
        path = tmp_path / "index.idx"
        nsparse.write_index(index, str(path))
        with pytest.raises(RuntimeError, match="mmap-only"):
            nsparse.read_index(str(path), 0)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_block_budget(self, corpus, queries, bad):
        """A non-positive k' (block budget) is rejected at search time."""
        index = make_index(self.SPEC, corpus)
        with pytest.raises(ValueError, match="must be positive"):
            search(index, queries, params=self.params(k_prime=bad))

    def test_block_budget_is_monotone(self, corpus, queries, oracle, tmp_path):
        """A larger block budget scores a superset of blocks, so recall never
        regresses and a big enough budget beats the smallest one. Runs on a
        seeded, mmap-reloaded index (deterministic on-disk block-read path)."""
        seeded = roundtrip(
            make_index(self.SEEDED, corpus), tmp_path / "seeded.idx", nsparse.kUseMmap
        )
        want_labels, _ = oracle
        recalls = [
            recall_at_k(
                search(seeded, queries, params=self.params(k_prime=kp))[1], want_labels
            )
            for kp in [1, 4, 16, 64, 256]
        ]
        for lo, hi in zip(recalls, recalls[1:]):
            assert hi >= lo - 1e-9, f"recall regressed across k': {recalls}"
        assert recalls[-1] > recalls[0], "a larger budget must eventually help"

    def test_block_budget_saturates(self, corpus, queries, tmp_path):
        """Past the candidate-block count, more budget changes nothing."""
        seeded = roundtrip(
            make_index(self.SEEDED, corpus), tmp_path / "seeded.idx", nsparse.kUseMmap
        )
        big_d, big_l = search(seeded, queries, params=self.params(k_prime=10**6))
        bigger_d, bigger_l = search(seeded, queries, params=self.params(k_prime=2 * 10**6))
        np.testing.assert_array_equal(bigger_l, big_l)
        np.testing.assert_array_equal(bigger_d, big_d)

    def test_empty_index_roundtrip(self, queries, tmp_path):
        """An un-built, empty index writes, mmap-reloads, and returns all padding."""
        empty = nsparse.index_factory(queries.dim, self.SPEC)
        path = tmp_path / "empty.idx"
        nsparse.write_index(empty, str(path))
        mapped = nsparse.read_index(str(path), nsparse.kUseMmap)
        assert mapped.num_vectors() == 0
        dists, labels = search(mapped, queries, params=self.params())
        assert (labels == PAD_LABEL).all()
        assert (dists == PAD_DIST).all()

    def test_search_before_build(self, corpus, queries):
        """Searching an added-but-unbuilt index yields empty results, not an
        error. Pinned deliberately: it is a silent-empty footgun."""
        index = nsparse.index_factory(corpus.dim, self.SPEC)
        index.add(corpus.n, corpus.indptr, corpus.indices, corpus.values)
        _, labels = search(index, queries, params=self.params())
        assert (labels == PAD_LABEL).all()

    @pytest.mark.parametrize("residency", ["memory", "mmap"])
    def test_filtered_search(self, residency, corpus, queries, oracle, tmp_path):
        """An id selector larger than k filters results to its members, on both
        the in-RAM (vectors_) and mmap (fwd_) scoring paths.

        (The disk-resident indexes omit seismic's exact-match fast path, but
        still honor the selector per candidate doc.)"""
        index = make_index(self.SPEC, corpus)
        if residency == "mmap":
            index = roundtrip(index, tmp_path / "filtered.idx", nsparse.kUseMmap)

        want_labels, _ = oracle
        allowed = np.ascontiguousarray(
            np.unique(want_labels[want_labels >= 0])[: K * 5], dtype=np.int32
        )
        assert len(allowed) > K

        selector = nsparse.SetIDSelector(allowed)
        p = self.params()
        p.set_id_selector(selector)

        _, labels = search(index, queries, params=p)
        returned = labels[labels >= 0]
        assert np.isin(returned, allowed).all(), "filter must exclude non-members"

    def test_with_id_map(self, corpus, queries, oracle, doc_ids, tmp_path):
        """idmap over the index returns the caller's ids, and reloads via mmap
        (the delegate's copying read is unsupported, so the whole idmap must be
        mmap-loaded)."""
        index = make_index(f"idmap,{self.SPEC}", corpus, ids=doc_ids)
        index = roundtrip(index, tmp_path / "idmap.idx", nsparse.kUseMmap)

        _, labels = search(index, queries, params=self.params())
        returned = labels[labels >= 0]
        assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"

        want_labels, _ = oracle
        want_external = np.where(want_labels >= 0, doc_ids[want_labels], -1)
        assert recall_at_k(labels, want_external) >= self.RECALL_FLOOR

    def test_batch_matches_single_query(self, corpus, queries):
        """Batched (OpenMP-parallel over queries) results must equal the one-at-
        a-time path exactly, or the per-thread scratch is leaking."""
        index = make_index(self.SPEC, corpus)
        batch_d, batch_l = search(index, queries, params=self.params())
        single_d, single_l = search_each(index, queries, params=self.params())
        np.testing.assert_array_equal(batch_l, single_l)
        np.testing.assert_allclose(batch_d, single_d, rtol=0, atol=0)

    def test_k_larger_than_corpus(self, corpus, queries):
        """Short result rows are padded with INVALID_IDX / -1.0, not truncated."""
        small = make_index(self.SPEC, slice_corpus(corpus, 0, 3))
        k = 10
        dists, labels = search(small, queries, k=k, params=self.params())
        assert labels.shape == (queries.n, k)
        assert (labels[:, 3:] == PAD_LABEL).all()
        assert (dists[:, 3:] == PAD_DIST).all()

    def test_seeded_build_is_reproducible(self, corpus, queries):
        """seed= makes the build (and so the search results) reproducible."""
        first = search(make_index(self.SEEDED, corpus), queries, params=self.params())
        second = search(make_index(self.SEEDED, corpus), queries, params=self.params())
        np.testing.assert_array_equal(second[1], first[1])
        np.testing.assert_array_equal(second[0], first[0])

    # --- inline_max_nnz (build) and rescore (query) ---

    def test_rescore_defaults_to_the_library_depth(self):
        """rescore is a plain per-query field: defaulted, overridable at
        construction, and settable afterwards."""
        assert self.params().rescore == nsparse.kDefaultRescoreDepth
        p = self.params(rescore=7)
        assert p.rescore == 7
        p.rescore = 11
        assert p.rescore == 11

    def test_truncation_shrinks_the_file(self, corpus, tmp_path):
        """The point of inline_max_nnz: each tighter limit is a smaller file."""
        sizes = []
        for limit in [0, 16, self.INLINE_MAX_NNZ]:
            path = tmp_path / f"limit{limit}.idx"
            nsparse.write_index(
                make_index(f"{self.SEEDED}|inline_max_nnz={limit}", corpus), str(path)
            )
            sizes.append(path.stat().st_size)
        assert sizes[0] > sizes[1] > sizes[2], sizes

    def test_truncated_matches_whole_when_everything_is_rescored(
        self, corpus, queries, tmp_path
    ):
        """Truncation only changes how candidates are ranked before the second
        pass, so with every block selected and every candidate re-scored the
        result is the untruncated index's."""
        whole = self.truncated(corpus, tmp_path / "whole.idx", limit=0)
        cut = self.truncated(corpus, tmp_path / "cut.idx")
        everything = self.params(k_prime=10**6, rescore=10**6)
        assert_same_ranking_modulo_ties(
            search(cut, queries, params=everything),
            search(whole, queries, params=self.params(k_prime=10**6)),
        )

    def test_truncated_index_clears_the_recall_floor(
        self, corpus, queries, oracle, tmp_path
    ):
        """At the default depth a truncated index is still a usable index."""
        cut = self.truncated(corpus, tmp_path / "cut.idx")
        _, labels = search(cut, queries, params=self.params())
        want_labels, _ = oracle
        assert recall_at_k(labels, want_labels) >= self.RECALL_FLOOR

    def test_rescore_is_ignored_without_truncation(self, corpus, queries, tmp_path):
        """With every copy whole there is nothing to re-score."""
        whole = self.truncated(corpus, tmp_path / "whole.idx", limit=0)
        shallow = search(whole, queries, params=self.params(rescore=K))
        deep = search(whole, queries, params=self.params(rescore=10**6))
        np.testing.assert_array_equal(shallow[1], deep[1])
        np.testing.assert_array_equal(shallow[0], deep[0])

    def test_rescored_scores_are_exact(self, corpus, queries, tmp_path):
        """A returned score is the doc's true score, not its truncated copy's
        lower bound -- even at depth k, where the second pass cannot change which
        docs come back. The oracle is the untruncated index scoring exactly the
        returned docs: a selector of size <= k takes the exact-match path."""
        whole = self.truncated(corpus, tmp_path / "whole.idx", limit=0)
        cut = self.truncated(corpus, tmp_path / "cut.idx")
        dists, labels = search(cut, queries, params=self.params(rescore=K))
        for q in range(queries.n):
            members = np.ascontiguousarray(labels[q][labels[q] >= 0], dtype=np.int32)
            exact = self.params()
            selector = nsparse.SetIDSelector(members)
            exact.set_id_selector(selector)
            want_d, want_l = search(whole, slice_corpus(queries, q, q + 1), params=exact)
            want = dict(zip(want_l[0].tolist(), want_d[0].tolist()))
            for doc, score in zip(labels[q], dists[q]):
                if doc >= 0:
                    assert score == pytest.approx(want[int(doc)], rel=1e-6, abs=1e-6), (
                        f"query {q} doc {doc} returned a lower bound"
                    )

    @pytest.mark.parametrize("below", [0, 1, K - 1])
    def test_rescore_below_k_behaves_like_k(self, below, corpus, queries, tmp_path):
        """Fewer than k re-scored candidates could not fill the result, so the
        depth is raised to k."""
        cut = self.truncated(corpus, tmp_path / "cut.idx")
        at_k = search(cut, queries, params=self.params(rescore=K))
        under = search(cut, queries, params=self.params(rescore=below))
        np.testing.assert_array_equal(under[1], at_k[1])
        np.testing.assert_array_equal(under[0], at_k[0])

    def test_deeper_rescore_never_lowers_the_result(self, corpus, queries, tmp_path):
        """A deeper pass re-scores a superset of candidates, and the k best exact
        scores of a superset are never worse: each query's summed top-k score is
        non-decreasing in the depth."""
        cut = self.truncated(corpus, tmp_path / "cut.idx")
        previous = None
        for depth in [K, 2 * K, 50, 200, 10**6]:
            dists, _ = search(cut, queries, params=self.params(k_prime=10**6, rescore=depth))
            total = np.where(dists > 0, dists, 0).astype(np.float64).sum(axis=1)
            if previous is not None:
                assert (total >= previous - 1e-4).all(), f"depth {depth} lowered a query"
            previous = total

    @pytest.mark.parametrize("limit", [0, 8])
    def test_rejects_negative_rescore(self, limit, corpus, queries, tmp_path):
        """A negative depth is rejected on any disk index, used or not."""
        index = self.truncated(corpus, tmp_path / "index.idx", limit=limit)
        with pytest.raises(ValueError, match="non-negative"):
            search(index, queries, params=self.params(rescore=-1))

    @pytest.mark.parametrize("bad", ["-1", "4294967296", "abc"])
    def test_factory_rejects_a_bad_inline_max_nnz(self, corpus, bad):
        """A malformed limit fails the spec instead of becoming "no truncation"."""
        with pytest.raises(ValueError):
            nsparse.index_factory(corpus.dim, f"{self.SPEC}|inline_max_nnz={bad}")

    def test_truncated_with_id_map(self, corpus, queries, doc_ids, tmp_path):
        """idmap forwards the search parameters, rescore included, to a
        truncated delegate and maps the re-scored ids back."""
        spec = f"idmap,{self.SEEDED}|inline_max_nnz={self.INLINE_MAX_NNZ}"
        index = roundtrip(
            make_index(spec, corpus, ids=doc_ids), tmp_path / "idmap.idx", nsparse.kUseMmap
        )
        _, labels = search(index, queries, params=self.params(rescore=50))
        returned = labels[labels >= 0]
        assert returned.size > 0
        assert np.isin(returned, doc_ids).all(), "returned ids must be caller ids"
