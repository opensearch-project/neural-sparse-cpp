/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/gpu/gpu_cluster_assigner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "nsparse/cluster/kmeans_utils.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
namespace {

// CPU reference assignment, copied to mirror the scalar map_docs_to_clusters()
// semantics: strict-greater argmax (ties -> lowest cluster index) and skipping
// documents that are themselves a centroid.
std::vector<std::vector<idx_t>> cpu_reference_assign(
    const SparseVectors* vectors, const std::vector<idx_t>& docs,
    std::vector<std::vector<idx_t>> clusters) {
    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const float* values = vectors->values_data_float();
    const size_t n_clusters = clusters.size();
    for (size_t i = 0; i < docs.size(); ++i) {
        const auto dense = vectors->get_dense_vector_float(docs[i]);
        float best = std::numeric_limits<float>::lowest();
        size_t best_j = 0;
        bool is_centroid = false;
        for (size_t j = 0; j < n_clusters; ++j) {
            const idx_t c = clusters[j].front();
            if (docs[i] == c) {
                is_centroid = true;
                break;
            }
            const idx_t start = indptr[c];
            const size_t len = indptr[c + 1] - start;
            float score = 0.0F;
            for (size_t t = 0; t < len; ++t) {
                score += dense[indices[start + t]] * values[start + t];
            }
            if (score > best) {
                best = score;
                best_j = j;
            }
        }
        if (!is_centroid) {
            clusters[best_j].push_back(docs[i]);
        }
    }
    return clusters;
}

SparseVectors make_random_corpus(size_t n_docs, size_t dim, size_t nnz_per_doc,
                                 unsigned seed) {
    SparseVectors vectors({.element_size = U32, .dimension = dim});
    std::mt19937 gen(seed);
    std::uniform_int_distribution<term_t> term_dist(
        0, static_cast<term_t>(dim - 1));
    std::uniform_real_distribution<float> val_dist(0.1F, 1.0F);
    for (size_t d = 0; d < n_docs; ++d) {
        std::vector<term_t> terms;
        std::vector<float> vals;
        // Deduplicate terms so each column appears once per row.
        std::vector<term_t> chosen;
        for (size_t k = 0; k < nnz_per_doc; ++k) {
            chosen.push_back(term_dist(gen));
        }
        std::ranges::sort(chosen);
        chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
        for (term_t t : chosen) {
            terms.push_back(t);
            vals.push_back(val_dist(gen));
        }
        vectors.add_vector(
            terms.data(), terms.size(),
            reinterpret_cast<const uint8_t*>(vals.data()),
            vals.size() * sizeof(float));
    }
    return vectors;
}

// Build initial clusters: the first n_clusters docs are the centroids.
std::vector<std::vector<idx_t>> seed_clusters(const std::vector<idx_t>& docs,
                                              size_t n_clusters) {
    std::vector<std::vector<idx_t>> clusters(n_clusters);
    for (size_t j = 0; j < n_clusters; ++j) {
        clusters[j].push_back(docs[j]);
    }
    return clusters;
}

TEST(GpuClusterAssignerTest, MatchesCpuReference) {
    if (!GpuClusterAssigner::available()) {
        GTEST_SKIP() << "No CUDA-capable GPU available";
    }
    constexpr size_t kNumDocs = 6000;
    constexpr size_t kDim = 2000;
    constexpr size_t kNnz = 40;
    constexpr size_t kNumClusters = 32;

    SparseVectors vectors = make_random_corpus(kNumDocs, kDim, kNnz, 1234);

    std::vector<idx_t> docs(kNumDocs);
    std::iota(docs.begin(), docs.end(), 0);

    auto expected =
        cpu_reference_assign(&vectors, docs, seed_clusters(docs, kNumClusters));

    auto actual = seed_clusters(docs, kNumClusters);
    GpuClusterAssigner::instance().assign(&vectors, docs, actual);

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t j = 0; j < expected.size(); ++j) {
        // Assignment order within a cluster is preserved (docs iterated in
        // order on both paths), so compare directly.
        EXPECT_EQ(actual[j], expected[j]) << "cluster " << j << " differs";
    }
}

TEST(GpuClusterAssignerTest, AutoPathViaMapDocsMatchesReference) {
    if (!GpuClusterAssigner::available()) {
        GTEST_SKIP() << "No CUDA-capable GPU available";
    }
    // Force the auto-offload gate to fire for this size.
    ::setenv("NSPARSE_GPU_MIN_DOCS", "1", /*overwrite=*/1);
    constexpr size_t kNumDocs = 5000;
    constexpr size_t kDim = 1500;
    constexpr size_t kNnz = 30;
    constexpr size_t kNumClusters = 16;

    SparseVectors vectors = make_random_corpus(kNumDocs, kDim, kNnz, 99);
    std::vector<idx_t> docs(kNumDocs);
    std::iota(docs.begin(), docs.end(), 0);

    auto expected =
        cpu_reference_assign(&vectors, docs, seed_clusters(docs, kNumClusters));

    auto actual = seed_clusters(docs, kNumClusters);
    map_docs_to_clusters(&vectors, docs, actual);  // routes to GPU

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t j = 0; j < expected.size(); ++j) {
        EXPECT_EQ(actual[j], expected[j]) << "cluster " << j << " differs";
    }
}

}  // namespace
}  // namespace nsparse::detail
