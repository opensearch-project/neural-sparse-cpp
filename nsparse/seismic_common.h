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

inline std::vector<InvertedListClusters> build_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params) {
    int lambda =
        calculate_lambda(seismic_cluster_params.lambda, vectors->num_vectors());
    int beta = calculate_beta(seismic_cluster_params.beta, lambda);
    size_t dimension = config.dimension;
    size_t element_size = config.element_size;
    std::vector<InvertedListClusters> clustered_inverted_lists(dimension);

    const size_t n_docs = vectors->num_vectors();
    const auto* indptr_data = vectors->indptr_data();
    const auto* indices_data = vectors->indices_data();
    const auto* values_data = vectors->values_data();

    // Estimate inverted list memory: NNZ * (sizeof(idx_t) + element_size)
    const size_t total_nnz = indptr_data[n_docs] - indptr_data[0];
    const size_t invlist_bytes_full =
        total_nnz * (sizeof(idx_t) + element_size);

    // Choose batch count to fit inverted lists within the memory budget.
    // Honors an explicit budget from cluster params; otherwise auto-detects
    // (Linux) with a documented fixed fallback on other platforms.
    const size_t mem_budget =
        resolve_build_mem_budget(seismic_cluster_params.mem_budget_bytes);

    size_t n_batches = (invlist_bytes_full + mem_budget - 1) / mem_budget;
    if (n_batches < 1) n_batches = 1;
    size_t batch_size = (dimension + n_batches - 1) / n_batches;

    fprintf(stderr, "[nsparse] build_inverted_lists: n_docs=%zu, dimension=%zu, "
            "element_size=%zu, lambda=%d, beta=%d, n_batches=%zu, batch_size=%zu\n",
            n_docs, dimension, element_size, lambda, beta, n_batches, batch_size);

    for (size_t batch_start = 0; batch_start < dimension;
         batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, dimension);
        size_t this_batch = batch_end - batch_start;

        // Parallel CSR → inverted list construction
        // Uses existing per-list spinlock in InvertedList for thread safety.
        // With 30K+ lists and 32 threads, lock contention is negligible.
        auto batch_invlists =
            std::make_unique<ArrayInvertedLists>(this_batch, element_size);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n_docs); ++i) {
            offset_t start = indptr_data[i];
            offset_t end = indptr_data[i + 1];
            for (offset_t j = start; j < end; ++j) {
                size_t term_id = indices_data[j];
                if (term_id >= batch_start && term_id < batch_end) {
                    batch_invlists->add_entry(
                        static_cast<term_t>(term_id - batch_start),
                        static_cast<idx_t>(i),
                        values_data + j * element_size);
                }
            }
        }

        // Count non-empty lists for diagnostics
        size_t non_empty = 0;
        size_t total_entries = 0;
        for (size_t i = 0; i < this_batch; ++i) {
            size_t sz = (*batch_invlists)[i].size();
            if (sz > 0) {
                non_empty++;
                total_entries += sz;
            }
        }
        fprintf(stderr, "[nsparse] batch [%zu, %zu): %zu/%zu non-empty lists, "
                "%zu total entries\n",
                batch_start, batch_end, non_empty, this_batch, total_entries);

        // Prune and cluster in parallel
        std::atomic<size_t> clustered_count{0};
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < static_cast<int64_t>(this_batch); ++i) {
            auto& invlist = (*batch_invlists)[i];
            if (invlist.size() == 0) {
                continue;
            }
            const auto& doc_ids = invlist.prune_and_keep_doc_ids(lambda);
            InvertedListClusters inverted_list_clusters(
                detail::RandomKMeans::train(vectors, doc_ids, beta));
            inverted_list_clusters.summarize(vectors,
                                             seismic_cluster_params.alpha);
            clustered_inverted_lists[batch_start + i] =
                std::move(inverted_list_clusters);
            invlist.clear();
            clustered_count.fetch_add(1, std::memory_order_relaxed);
        }
        fprintf(stderr, "[nsparse] batch done: %zu lists clustered\n",
                clustered_count.load());
    }

    return clustered_inverted_lists;
}

// Streaming variant: builds inverted list clusters batch-by-batch and serializes
// each batch's clusters immediately to the IOWriter, freeing memory after each
// batch. This avoids accumulating all 65K InvertedListClusters in RAM.
inline void build_and_save_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params,
    IOWriter* io_writer) {
    int lambda =
        calculate_lambda(seismic_cluster_params.lambda, vectors->num_vectors());
    int beta = calculate_beta(seismic_cluster_params.beta, lambda);
    size_t dimension = config.dimension;
    size_t element_size = config.element_size;

    const size_t n_docs = vectors->num_vectors();
    const auto* indptr_data = vectors->indptr_data();
    const auto* indices_data = vectors->indices_data();
    const auto* values_data = vectors->values_data();

    const size_t total_nnz = indptr_data[n_docs] - indptr_data[0];
    const size_t invlist_bytes_full =
        total_nnz * (sizeof(idx_t) + element_size);

    // Cap per-batch inverted list memory to keep peak RSS under physical RAM.
    // The CSR (~46 GB for float32 at 46M docs) remains throughout; each batch
    // adds inverted lists + k-means temporaries (~2x the raw invlist allocation).
    // An explicit budget from cluster params wins; otherwise cap at 8 GB (or a
    // third of the full invlist size, whichever is smaller).
    size_t mem_budget;
    if (seismic_cluster_params.mem_budget_bytes > 0) {
        mem_budget = seismic_cluster_params.mem_budget_bytes;
    } else {
        constexpr size_t kMaxBatchBytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB
        mem_budget = std::min(kMaxBatchBytes, invlist_bytes_full / 3);
    }
    if (mem_budget == 0) mem_budget = invlist_bytes_full;
    if (mem_budget == 0) mem_budget = 1;  // guard: empty index

    size_t n_batches = (invlist_bytes_full + mem_budget - 1) / mem_budget;
    if (n_batches < 1) n_batches = 1;
    size_t batch_size = (dimension + n_batches - 1) / n_batches;

    fprintf(stderr,
            "[nsparse] build_and_save_inverted_lists: n_docs=%zu, "
            "dimension=%zu, element_size=%zu, lambda=%d, beta=%d, "
            "n_batches=%zu, batch_size=%zu\n",
            n_docs, dimension, element_size, lambda, beta, n_batches,
            batch_size);

    // Write the total dimension count (number of InvertedListClusters)
    io_writer->write(&dimension, sizeof(dimension), 1);

    for (size_t batch_start = 0; batch_start < dimension;
         batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, dimension);
        size_t this_batch = batch_end - batch_start;

        auto batch_invlists =
            std::make_unique<ArrayInvertedLists>(this_batch, element_size);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n_docs); ++i) {
            offset_t start = indptr_data[i];
            offset_t end = indptr_data[i + 1];
            for (offset_t j = start; j < end; ++j) {
                size_t term_id = indices_data[j];
                if (term_id >= batch_start && term_id < batch_end) {
                    batch_invlists->add_entry(
                        static_cast<term_t>(term_id - batch_start),
                        static_cast<idx_t>(i),
                        values_data + j * element_size);
                }
            }
        }

        size_t non_empty = 0;
        size_t total_entries = 0;
        for (size_t i = 0; i < this_batch; ++i) {
            size_t sz = (*batch_invlists)[i].size();
            if (sz > 0) {
                non_empty++;
                total_entries += sz;
            }
        }
        fprintf(stderr,
                "[nsparse] batch [%zu, %zu): %zu/%zu non-empty lists, "
                "%zu total entries\n",
                batch_start, batch_end, non_empty, this_batch, total_entries);

        // Prune, cluster, serialize, and immediately free each list
        std::vector<InvertedListClusters> batch_clusters(this_batch);
        std::atomic<size_t> clustered_count{0};
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < static_cast<int64_t>(this_batch); ++i) {
            auto& invlist = (*batch_invlists)[i];
            if (invlist.size() == 0) {
                continue;
            }
            const auto& doc_ids = invlist.prune_and_keep_doc_ids(lambda);
            InvertedListClusters inverted_list_clusters(
                detail::RandomKMeans::train(vectors, doc_ids, beta));
            inverted_list_clusters.summarize(vectors,
                                             seismic_cluster_params.alpha);
            batch_clusters[i] = std::move(inverted_list_clusters);
            invlist.clear();
            clustered_count.fetch_add(1, std::memory_order_relaxed);
        }
        fprintf(stderr, "[nsparse] batch done: %zu lists clustered\n",
                clustered_count.load());

        // Free inverted lists immediately
        batch_invlists.reset();

        // Serialize this batch's clusters to disk and free
        for (size_t i = 0; i < this_batch; ++i) {
            batch_clusters[i].serialize(io_writer);
        }
        batch_clusters.clear();
        batch_clusters.shrink_to_fit();

        fprintf(stderr,
                "[nsparse] batch [%zu, %zu): serialized and freed\n",
                batch_start, batch_end);
    }
}

}  // namespace detail
}  // namespace nsparse

#endif  // SEISMIC_COMMON_H