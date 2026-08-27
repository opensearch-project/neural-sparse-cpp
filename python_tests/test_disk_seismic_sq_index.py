# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the scalar-quantized DiskSeismic index, through SWIG only.

The shared disk-resident contract comes from disk_seismic_contract.py (bound
here to the quantized spec); this file adds only what is quantization-specific:
both quantizer widths, the on-disk size win, the query-range override, and the
factory's quantizer handling.
"""

import os

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import make_index, search
from disk_seismic_contract import DiskSeismicContract

VMIN, VMAX = 0.0, 1.0

# The corpus values span [0.1, 1.0]; the [0, 1] quantizer range covers them.
SPEC = f"disk_seismic_sq,quantizer=8bit|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4"


def sq_spec(width):
    return f"disk_seismic_sq,quantizer={width}|vmin={VMIN}|vmax={VMAX}|lambda=25|beta=4|alpha=0.4"


class TestDiskSeismicSQ(DiskSeismicContract):
    SPEC = SPEC
    SEEDED = f"{SPEC}|seed=42"
    # A floor with headroom, not a target: the build RNG is unseeded and 8-bit
    # quantization adds noise, so a tight bound would flake.
    RECALL_FLOOR = 0.75

    def params(self, cut=None, k_prime=None):
        cut = self.CUT if cut is None else cut
        k_prime = self.K_PRIME if k_prime is None else k_prime
        return nsparse.DiskSeismicSQSearchParameters(VMIN, VMAX, cut, k_prime)

    # --- quantization-specific ---

    @pytest.mark.parametrize("width", ["8bit", "16bit"])
    def test_quantizer_widths(self, corpus, queries, oracle, width):
        """Both quantizer widths clear the recall floor."""
        idx = make_index(sq_spec(width), corpus)
        _, labels = search(idx, queries, params=self.params())
        want_labels, _ = oracle
        assert recall_at_k(labels, want_labels) >= self.RECALL_FLOOR

    @pytest.mark.parametrize("width", ["8bit", "16bit"])
    def test_smaller_than_plain_disk_seismic(self, corpus, tmp_path, width):
        """The whole point of quantization: codes shrink the on-disk index
        versus the float disk_seismic, for the same corpus and cluster params.
        Both widths must win."""
        float_index = make_index("disk_seismic,lambda=25|beta=4|alpha=0.4|seed=42", corpus)
        float_path = tmp_path / "disk_seismic_float.idx"
        nsparse.write_index(float_index, str(float_path))

        sq_index = make_index(f"{sq_spec(width)}|seed=42", corpus)
        sq_path = tmp_path / f"disk_seismic_sq_{width}.idx"
        nsparse.write_index(sq_index, str(sq_path))

        assert os.path.getsize(sq_path) < os.path.getsize(float_path)

    def test_query_range_override(self, corpus, queries):
        """A DiskSeismicSQSearchParameters with a non-index range re-encodes the
        query at that range (the query_quantizer override). The corpus values
        span [0.1, 1.0]; vmax=0.5 re-scales/clips the query, so it must actually
        change the decoded scores versus the default range -- if query_quantizer
        ignored the override this would silently no-op."""
        index = make_index(self.SPEC, corpus)
        default_d, _ = search(index, queries, params=self.params())

        clipped = nsparse.DiskSeismicSQSearchParameters(
            0.0, 0.5, self.CUT, self.K_PRIME
        )
        clipped_d, clipped_l = search(index, queries, params=clipped)

        assert clipped_l.shape == default_d.shape
        assert clipped_l[clipped_l >= 0].size > 0, "clipping must not empty every result"
        assert not np.allclose(clipped_d, default_d), "query-range override had no effect"

    def test_factory_downcast_exposes_quantizer(self, corpus):
        """DSSQ is in the %factory downcast lists, so index_factory returns the
        concrete type and the DSSQ-only get_scalar_quantizer() is reachable (it
        is not a base Index method). The two widths must report distinct types."""

        def qtype(width):
            idx = nsparse.index_factory(corpus.dim, sq_spec(width))
            return idx.get_scalar_quantizer().get_quantizer_type()

        assert qtype("8bit") != qtype("16bit"), "widths must report distinct types"

    @pytest.mark.parametrize("bad", ["4bit", "int8", ""])
    def test_factory_rejects_unknown_quantizer(self, corpus, bad):
        """An unrecognized quantizer= is rejected, not silently treated as 8bit."""
        spec = f"disk_seismic_sq,quantizer={bad}|vmin={VMIN}|vmax={VMAX}|lambda=25"
        with pytest.raises(ValueError, match="quantizer"):
            nsparse.index_factory(corpus.dim, spec)
