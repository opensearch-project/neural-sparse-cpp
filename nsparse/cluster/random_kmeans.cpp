/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/random_kmeans.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "nsparse/cluster/kmeans_utils.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

namespace nsparse::detail {

RandomKMeans::RandomKMeans() = default;

static std::vector<std::vector<idx_t>> random_select_initial_centroids(
    std::vector<idx_t> docs, size_t n_clusters) {
    size_t n_docs = docs.size();
    if (n_clusters > n_docs) {
        std::cerr << "n_clusters is larger than candidates size\n";
        return {};
    }
    if (n_clusters <= 0) {
        std::cerr << "n_clusters is smaller or equal to 0\n";
        return {};
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<idx_t>> clusters(n_clusters, std::vector<idx_t>());

    for (size_t i = 0; i < n_clusters; i++) {
        std::uniform_int_distribution<> distrib(0, n_docs - i - 1);
        size_t r = distrib(gen);
        idx_t doc_id = docs[r];
        clusters[i].push_back(doc_id);
        std::swap(docs[r], docs[n_docs - 1 - i]);
    }

    return clusters;
}

inline static size_t boundary_check_n_clusters(size_t n_docs,
                                               size_t n_clusters) {
    if (n_clusters <= 0) {
        n_clusters = static_cast<size_t>(std::sqrt(n_docs));
    }

    // Ensure at least one cluster
    n_clusters = n_clusters > n_docs ? n_docs : n_clusters;
    n_clusters = std::max(1UL, n_clusters);
    return n_clusters;
}

std::vector<std::vector<idx_t>> RandomKMeans::train(
    const SparseVectors* vectors, const std::vector<idx_t>& doc_ids,
    size_t n_clusters) {
    throw_if_any_null(vectors);
    size_t n_docs = doc_ids.size();
    if (n_docs == 0) {
        return {};
    }
    n_clusters = boundary_check_n_clusters(n_docs, n_clusters);

    auto clusters = random_select_initial_centroids(doc_ids, n_clusters);
    map_docs_to_clusters(vectors, doc_ids, clusters);
    return clusters;
}

}  // namespace nsparse::detail