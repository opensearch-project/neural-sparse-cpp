# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the disk-resident DiskSeismic index, through SWIG only.

The disk-resident contract (mmap-only reads, the top-k' block budget,
fresh-build vs mmap-reload parity) is shared with the quantized index and lives
in disk_seismic_contract.py; this suite just binds it to the float spec.
"""

import nsparse
from disk_seismic_contract import DiskSeismicContract

SPEC = "disk_seismic,lambda=25|beta=4|alpha=0.4"


class TestDiskSeismic(DiskSeismicContract):
    SPEC = SPEC
    SEEDED = f"{SPEC}|seed=42"
    # Calibrated against the session corpus; a floor, not a target.
    RECALL_FLOOR = 0.80

    def params(self, cut=None, k_prime=None):
        cut = self.CUT if cut is None else cut
        k_prime = self.K_PRIME if k_prime is None else k_prime
        return nsparse.DiskSeismicSearchParameters(cut, k_prime)
