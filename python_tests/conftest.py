# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Fixtures for the black-box index tests.

Everything here goes through the installed extension module and the SWIG
surface only. One harness guard runs before any test, because its failure mode
is silent: a shadowed import gives an incomplete module rather than an error.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import nsparse

from oracle import brute_force_top_k
from support import make_corpus

# Sized so the posting lists are long enough for lambda/beta pruning to do real
# work, and so every query has more than K positive-scoring docs -- otherwise
# top-K is degenerate and recall is trivially perfect.
N_DOCS = 2000
DIM = 1024
DOC_NNZ = 30
N_QUERIES = 50
QUERY_NNZ = 8
CORPUS_SEED = 0xC0FFEE
QUERY_SEED = 0xBEEF


def pytest_configure(config):
    """Fail loudly on the silent harness failure mode."""
    # Importing from the repo root resolves the source tree `nsparse/` as a
    # namespace package (__file__ is None) when the extension is not installed.
    # The result is a module without index_factory, which reads as a test bug.
    if getattr(nsparse, "__file__", None) is None or not hasattr(
        nsparse, "index_factory"
    ):
        raise pytest.UsageError(
            f"nsparse resolved to an incomplete module ({nsparse.__file__!r}). "
            "Install the built extension: pip install "
            "build/nsparse/python"
        )


@pytest.fixture(scope="session")
def corpus():
    return make_corpus(N_DOCS, DIM, DOC_NNZ, CORPUS_SEED)


@pytest.fixture(scope="session")
def queries():
    return make_corpus(N_QUERIES, DIM, QUERY_NNZ, QUERY_SEED)


@pytest.fixture(scope="session")
def oracle(corpus, queries):
    """Exact (labels, dists) for the session corpus at k=K."""
    from support import K

    return brute_force_top_k(corpus.csr, queries.csr, corpus.dim, K)


@pytest.fixture(scope="session")
def doc_ids():
    """Non-contiguous external ids, so idmap can't accidentally pass by
    returning ordinals."""
    return (np.arange(N_DOCS, dtype=np.int32) * 7 + 1000).astype(np.int32)


@pytest.fixture(scope="session")
def worker_script():
    return str(Path(__file__).parent / "_thread_worker.py")


def run_isolated(args, env=None, timeout=600):
    """Run a helper script in a fresh interpreter.

    Needed for two reasons: OMP_NUM_THREADS is read when the OpenMP runtime
    initialises, so it cannot be changed from inside a running test; and C++
    exceptions abort the process rather than raising, so error paths have to be
    observed from outside.
    """
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
