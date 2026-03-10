/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef RANDOM_KMEANS_H
#define RANDOM_KMEANS_H

#include <vector>

#include "nsparse/sparse_vectors.h"

namespace nsparse::detail {

class ClusterRepresentatives {
public:
    ClusterRepresentatives(size_t num_clusters, size_t sketch_size,
                           size_t alignmnt)
        : num_clusters_(num_clusters), sketch_size_(sketch_size) {
        // Align to 64-byte boundary for AVX-512

        data = static_cast<float*>(std::aligned_alloc(
            alignmnt, num_clusters * sketch_size * sizeof(float)));
    }

    ~ClusterRepresentatives() { std::free(data); }

    // Access element (i,j) where i is cluster index and j is dimension
    float& operator()(size_t i, size_t j) { return data[i * sketch_size_ + j]; }

    const float* getData() const { return data; }

private:
    float* data;
    size_t num_clusters_;
    size_t sketch_size_;
};

class RandomKMeans {
public:
    RandomKMeans();

    static std::vector<std::vector<idx_t>> train(
        const SparseVectors* vectors, const std::vector<idx_t>& doc_ids,
        size_t n_clusters);
};

}  // namespace nsparse::detail

#endif  // RANDOM_KMEANS_H