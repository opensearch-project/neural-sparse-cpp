# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

#!/usr/bin/env python3
"""
Example usage of the NSPARSE Python bindings with SeismicScalarQuantizedIndex.
"""

import numpy as np
import nsparse


def main():
    # Example: sparse vectors in CSR format
    # indptr: pointer array indicating where each vector starts/ends
    # indices: term indices for non-zero values
    # values: corresponding weight values

    # Example: 5 sparse vectors with dimension 100
    n_vectors = 5
    dim = 100

    # Vector 0: {0: 1.0, 10: 0.5, 50: 0.8}
    # Vector 1: {5: 2.0, 15: 1.5, 60: 0.3}
    # Vector 2: {0: 0.8, 5: 1.2, 20: 0.3}
    # Vector 3: {10: 1.0, 30: 0.7, 70: 0.9}
    # Vector 4: {0: 0.6, 15: 0.4, 50: 1.1}
    # term 0: 0, 2, 4
    # term 5: 1, 2
    # term 10: 0, 3
    # term 15: 1, 4
    # term 20: 2
    # term 30: 3,
    # term 50: 0, 4
    # term 60: 1
    # term 70: 3

    indptr = np.array([0, 3, 6, 9, 12, 15], dtype=np.int32)
    indices = np.array(
        [0, 10, 50, 5, 15, 60, 0, 5, 20, 10, 30, 70, 0, 15, 50], dtype=np.uint16
    )
    values = np.array(
        [1.0, 0.5, 0.8, 2.0, 1.5, 0.3, 0.8, 1.2, 0.3, 1.0, 0.7, 0.9, 0.6, 0.4, 1.1],
        dtype=np.float32,
    )

    # Create SeismicScalarQuantizedIndex using index_factory
    # Parameters:
    #   quantizer=8bit: 8-bit scalar quantization
    #   vmin=0.0: minimum value for quantization range
    #   vmax=3.0: maximum value for quantization range
    #   lambda=10: posting list length
    #   beta=2: number of clusters in a posting list
    #   alpha=0.4: summary vector prune alpha mass ratio
    index = nsparse.index_factory(
        dim,
        "idmap,seismic_sq,quantizer=8bit|vmin=0.0|vmax=3.0|lambda=10|beta=2|alpha=0.4",
    )
    print(index)

    # Add vectors to the index
    ids = np.array([3, 6, 9, 12, 15], dtype=np.int32)
    index.add_with_ids(n_vectors, indptr, indices, values, ids)
    print(f"Added {n_vectors} vectors to the index")

    # Build the index (required before searching)
    index.build()
    print("Index built successfully")

    # Query vectors
    n_queries = 2
    query_indptr = np.array([0, 2, 4], dtype=np.int32)
    query_indices = np.array([0, 50, 5, 15], dtype=np.uint16)
    query_values = np.array([1.0, 0.5, 1.5, 0.8], dtype=np.float32)

    k = 3  # Find top 3 nearest neighbors

    # Create search parameters
    params = nsparse.SeismicSearchParameters(k, 1.2)
    # Perform search
    distances, labels = index.search(
        n_queries, query_indptr, query_indices, query_values, k, params
    )

    print(f"\nTop {k} nearest neighbors:")
    print(f"Labels shape: {labels.shape}")
    for i in range(n_queries):
        neighbors = labels[i]
        scores = distances[i]
        print(f"Query {i}: {neighbors}, scores: {scores}")


if __name__ == "__main__":
    main()
