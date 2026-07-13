/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_COMMON_H
#define SEISMIC_COMMON_H

#include <atomic>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

#include <omp.h>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/id_selector.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

namespace nsparse {
struct SeismicClusterParameters {
    int lambda;
    int beta;
    float alpha;
    // Memory budget (bytes) used to size the inverted-list build batches.
    // 0 means "auto-detect" (see detail::resolve_build_mem_budget): on Linux
    // this reads MemAvailable from /proc/meminfo; on platforms without that
    // interface it falls back to a fixed default. Set explicitly (e.g. from
    // the native-engine index description, "mem_budget=<bytes>") to control
    // build-phase peak RSS deterministically.
    size_t mem_budget_bytes = 0;
    // When true, emit per-batch progress diagnostics to stderr during the
    // inverted-list build. Off by default to keep production logs quiet.
    bool verbose = false;
};

namespace detail {

constexpr int kDefaultLambda = -1;
constexpr float kDefaultPostingPruneRatio = 0.0005F;
constexpr int kDefaultPostingMinimumLength = 160;
constexpr float kDefaultBetaRatio = 0.1F;
constexpr int kDefaultBeta = -1;
constexpr float kDefaultAlpha = 0.4F;

// Reserve headroom for clustering temporaries and the OS when auto-detecting.
constexpr size_t kBuildMemReserveBytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
// Fixed fallback when the available memory cannot be detected (non-Linux) and
// no explicit budget was supplied.
constexpr size_t kBuildMemFallbackBytes = 16ULL * 1024 * 1024 * 1024;  // 16 GB

constexpr SeismicClusterParameters kDefaultSeismicClusterParams = {
    .lambda = kDefaultLambda, .beta = kDefaultBeta, .alpha = kDefaultAlpha};

/**
 * @brief Query the OS for currently-available physical memory, in bytes.
 * @return available bytes, or 0 if the platform provides no supported
 *         mechanism (e.g. non-Linux).
 */
inline size_t query_available_memory_bytes() {
#ifdef __linux__
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == nullptr) {
        return 0;
    }
    size_t available = 0;
    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
        size_t kb = 0;
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            available = kb * 1024;
            break;
        }
    }
    fclose(meminfo);
    return available;
#else
    return 0;
#endif
}

/**
 * @brief Resolve the memory budget for build batching.
 *
 * Priority:
 *   1. If @p requested_budget_bytes > 0, use it verbatim (caller override).
 *   2. Otherwise auto-detect available memory and subtract a reserve.
 *   3. If auto-detect is unavailable (returns 0) or too small, use a fixed
 *      documented fallback so behavior is well-defined on every platform.
 *
 * @param requested_budget_bytes explicit budget (0 = auto)
 * @return a positive budget in bytes, never 0.
 */
inline size_t resolve_build_mem_budget(size_t requested_budget_bytes) {
    if (requested_budget_bytes > 0) {
        return requested_budget_bytes;
    }
    size_t available = query_available_memory_bytes();
    if (available > kBuildMemReserveBytes) {
        return available - kBuildMemReserveBytes;
    }
    return kBuildMemFallbackBytes;
}

inline std::vector<float> calculate_summary_scores(
    const size_t element_size, const SparseVectors* summaries,
    const std::vector<uint8_t>& dense) {
    std::vector<float> summary_scores;
    if (element_size == U32) {
        summary_scores = dot_product_float_vectors_dense(
            summaries, reinterpret_cast<const float*>(dense.data()));
    } else if (element_size == U16) {
        summary_scores = dot_product_uint16_vectors_dense(
            summaries, reinterpret_cast<const uint16_t*>(dense.data()));
    } else {
        summary_scores =
            dot_product_uint8_vectors_dense(summaries, dense.data());
    }
    return summary_scores;
}

inline float compute_similarity(idx_t doc_id, const offset_t* indptr,
                                const term_t* indices, const uint8_t* values,
                                const uint8_t* dense, size_t element_size) {
    const offset_t start = indptr[doc_id];
    const size_t len = indptr[doc_id + 1] - start;
    float score = 0.0F;
    if (element_size == U32) {
        const auto* float_values =
            reinterpret_cast<const float*>(values + start * sizeof(float));
        const auto* float_dense = reinterpret_cast<const float*>(dense);
        score = dot_product_float_dense(indices + start, float_values, len,
                                        float_dense);
    } else if (element_size == U16) {
        // start is element index, need to convert to byte offset for
        // uint16_t access
        const auto* int16_values = reinterpret_cast<const uint16_t*>(
            values + start * sizeof(uint16_t));
        const auto* int16_dense = reinterpret_cast<const uint16_t*>(dense);
        score = dot_product_uint16_dense(indices + start, int16_values, len,
                                         int16_dense);
    } else {
        score = dot_product_uint8_dense(indices + start, values + start, len,
                                        dense);
    }
    return score;
}

inline std::vector<size_t> reorder_clusters(
    const std::vector<float>& summary_scores, bool first_list) {
    std::vector<size_t> cluster_order(summary_scores.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0);
    if (first_list) {
        std::ranges::sort(cluster_order, [&](size_t a, size_t b) {
            return summary_scores[a] > summary_scores[b];
        });
    }
    return cluster_order;
}

inline bool should_run_exact_match(const IDSelector* id_selector, int k,
                                   const SparseVectors* queries) {
    if (id_selector == nullptr) {
        return false;
    }
    const auto* id_selector_enumerable =
        dynamic_cast<const IDSelectorEnumerable*>(id_selector);
    if (id_selector_enumerable == nullptr) {
        return false;
    }
    return id_selector_enumerable->size() <= k;
}

inline int calculate_lambda(int lambda, size_t n_vectors) {
    if (lambda == kDefaultLambda) {
        return std::max(static_cast<int>(kDefaultPostingPruneRatio *
                                         static_cast<float>(n_vectors)),
                        kDefaultPostingMinimumLength);
    }
    return lambda;
}

inline int calculate_beta(int beta, int lambda) {
    if (beta == kDefaultBeta) {
        return static_cast<int>(static_cast<float>(lambda) * kDefaultBetaRatio);
    }
    return beta;
}

/**
 * @brief Build clustered inverted lists for all dimensions, in memory.
 *
 * Definition lives in seismic_common.cpp.
 */
std::vector<InvertedListClusters> build_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params);

// Streaming variant: builds inverted list clusters batch-by-batch and serializes
// each batch's clusters immediately to the IOWriter, freeing memory after each
// batch. This avoids accumulating all 65K InvertedListClusters in RAM.
//
// Definition lives in seismic_common.cpp.
void build_and_save_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params,
    IOWriter* io_writer);

}  // namespace detail
}  // namespace nsparse

#endif  // SEISMIC_COMMON_H