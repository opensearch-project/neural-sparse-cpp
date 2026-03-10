/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
/**
 * @brief Map each document to its cluster based on the similarity between doc
 * and centroid.
 *
 * @param vectors: full dataset of sparse vectors
 * @param docs: list of documents to be mapped
 * @param clusters: output clusters, i.e., for each cluster, a list of docs. The
 * first element in each cluster is the centroid of the cluster.
 */
void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters);
}  // namespace nsparse::detail

#endif  // KMEANS_UTILS_H