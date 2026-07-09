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
#include <map>
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
    // With GPU support compiled in and a device present, map_docs_to_clusters
    // routes through the GPU unconditionally (no size gate).
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

// CPU reference for the summarize max-pool: per-term max weight over the docs
// of a cluster, plus the sum of maxes. Returns terms in ascending order.
void cpu_reference_maxpool(const SparseVectors* vectors,
                          const std::vector<idx_t>& doc_ids,
                          std::vector<term_t>& terms,
                          std::vector<float>& values, float& sum) {
    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const float* vals = vectors->values_data_float();
    std::map<term_t, float> m;
    for (idx_t d : doc_ids) {
        for (idx_t j = indptr[d]; j < indptr[d + 1]; ++j) {
            auto& v = m[indices[j]];
            v = std::max(v, vals[j]);
        }
    }
    sum = 0.0F;
    terms.clear();
    values.clear();
    for (auto& [t, v] : m) {
        terms.push_back(t);
        values.push_back(v);
        sum += v;
    }
}

// Compares GPU per-cluster max-pool against the CPU reference for a whole list
// of several clusters processed in one batched call.
TEST(GpuClusterAssignerTest, SummarizeListMaxpoolMatchesCpu) {
    if (!GpuClusterAssigner::available()) {
        GTEST_SKIP() << "No CUDA-capable GPU available";
    }
    constexpr size_t kNumDocs = 8000;
    constexpr size_t kDim = 3000;
    constexpr size_t kNnz = 50;
    SparseVectors vectors = make_random_corpus(kNumDocs, kDim, kNnz, 7);

    // Build a list of clusters: partition docs into varied-size groups.
    std::vector<idx_t> flat_docs;
    std::vector<idx_t> offsets = {0};
    const std::vector<size_t> sizes = {5, 137, 1, 900, 42, 2000};
    idx_t d = 0;
    for (size_t sz : sizes) {
        for (size_t k = 0; k < sz && d < static_cast<idx_t>(kNumDocs); ++k) {
            flat_docs.push_back(d++);
        }
        offsets.push_back(static_cast<idx_t>(flat_docs.size()));
    }
    const size_t n_clusters = offsets.size() - 1;

    std::vector<GpuClusterAssigner::ClusterSummary> gpu_out;
    ASSERT_TRUE(GpuClusterAssigner::instance().summarize_list_maxpool(
        &vectors, flat_docs.data(), offsets.data(), n_clusters, gpu_out));
    ASSERT_EQ(gpu_out.size(), n_clusters);

    for (size_t b = 0; b < n_clusters; ++b) {
        std::vector<idx_t> cluster(flat_docs.begin() + offsets[b],
                                   flat_docs.begin() + offsets[b + 1]);
        std::vector<term_t> exp_terms;
        std::vector<float> exp_vals;
        float exp_sum = 0.0F;
        cpu_reference_maxpool(&vectors, cluster, exp_terms, exp_vals, exp_sum);

        // GPU returns terms in touched-append order; sort by term to compare.
        const auto& gc = gpu_out[b];
        std::vector<size_t> order(gc.terms.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](size_t a, size_t c) {
            return gc.terms[a] < gc.terms[c];
        });
        ASSERT_EQ(gc.terms.size(), exp_terms.size()) << "cluster " << b;
        for (size_t i = 0; i < order.size(); ++i) {
            EXPECT_EQ(gc.terms[order[i]], exp_terms[i])
                << "cluster " << b << " term " << i;
            EXPECT_FLOAT_EQ(gc.values[order[i]], exp_vals[i])
                << "cluster " << b << " value " << i;
        }
        EXPECT_NEAR(gc.sum, exp_sum, exp_sum * 1e-5F + 1e-5F) << "cluster " << b;
    }
}

// A second list reusing the buffers must not leak accumulator state from the
// first — verifies the touched-slot reset path.
TEST(GpuClusterAssignerTest, SummarizeListMaxpoolReuseIsClean) {
    if (!GpuClusterAssigner::available()) {
        GTEST_SKIP() << "No CUDA-capable GPU available";
    }
    SparseVectors vectors = make_random_corpus(5000, 2500, 40, 21);
    auto& gpu = GpuClusterAssigner::instance();

    // First list.
    std::vector<idx_t> docs1;
    for (idx_t d = 0; d < 2000; ++d) docs1.push_back(d);
    std::vector<idx_t> off1 = {0, 800, 2000};
    std::vector<GpuClusterAssigner::ClusterSummary> out1;
    gpu.summarize_list_maxpool(&vectors, docs1.data(), off1.data(), 2, out1);

    // Second, different list must match its own CPU reference.
    std::vector<idx_t> docs2;
    for (idx_t d = 2000; d < 5000; ++d) docs2.push_back(d);
    std::vector<idx_t> off2 = {0, 1500, 3000};
    std::vector<GpuClusterAssigner::ClusterSummary> out2;
    ASSERT_TRUE(
        gpu.summarize_list_maxpool(&vectors, docs2.data(), off2.data(), 2,
                                   out2));
    for (size_t b = 0; b < 2; ++b) {
        std::vector<idx_t> cluster(docs2.begin() + off2[b],
                                   docs2.begin() + off2[b + 1]);
        std::vector<term_t> et;
        std::vector<float> ev;
        float es = 0.0F;
        cpu_reference_maxpool(&vectors, cluster, et, ev, es);
        EXPECT_EQ(out2[b].terms.size(), et.size()) << "cluster " << b;
        EXPECT_NEAR(out2[b].sum, es, es * 1e-5F + 1e-5F) << "cluster " << b;
    }
}

}  // namespace
}  // namespace nsparse::detail
