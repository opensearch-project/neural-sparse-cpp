/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_common.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {
namespace detail {

namespace {

// Result of sizing the build into memory-bounded batches.
struct BatchPlan {
    size_t n_batches;
    size_t batch_size;
};

// Compute {n_batches, batch_size} so that each batch's inverted lists fit
// within mem_budget. Shared by both build functions.
BatchPlan compute_batch_plan(size_t dimension, size_t invlist_bytes_full,
                             size_t mem_budget) {
    size_t n_batches = (invlist_bytes_full + mem_budget - 1) / mem_budget;
    if (n_batches < 1) n_batches = 1;
    size_t batch_size = (dimension + n_batches - 1) / n_batches;
    return {n_batches, batch_size};
}

// Build one batch's ArrayInvertedLists via the parallel CSR -> invlist loop for
// the [batch_start, batch_end) term range. Uses the existing per-list spinlock
// in InvertedList for thread safety; with 30K+ lists and 32 threads, lock
// contention is negligible.
std::unique_ptr<ArrayInvertedLists> build_batch_invlists(
    const SparseVectors* vectors, size_t element_size, size_t n_docs,
    const offset_t* indptr_data, const term_t* indices_data,
    const uint8_t* values_data, size_t batch_start, size_t batch_end) {
    (void)vectors;
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
                    static_cast<idx_t>(i), values_data + j * element_size);
            }
        }
    }
    return batch_invlists;
}

// Diagnostic count: number of non-empty lists and the sum of their sizes.
struct ListStats {
    size_t non_empty;
    size_t total_entries;
};

ListStats count_non_empty_lists(const ArrayInvertedLists& batch_invlists,
                                size_t this_batch) {
    size_t non_empty = 0;
    size_t total_entries = 0;
    for (size_t i = 0; i < this_batch; ++i) {
        size_t sz = batch_invlists[i].size();
        if (sz > 0) {
            non_empty++;
            total_entries += sz;
        }
    }
    return {non_empty, total_entries};
}

}  // namespace

std::vector<InvertedListClusters> build_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params) {
    int lambda =
        calculate_lambda(seismic_cluster_params.lambda, vectors->num_vectors());
    int beta = calculate_beta(seismic_cluster_params.beta, lambda);
    size_t dimension = config.dimension;
    size_t element_size = config.element_size;
    if (dimension == 0) {
        throw std::runtime_error(
            "build_inverted_lists_clusters: dimension must be > 0");
    }
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

    BatchPlan plan =
        compute_batch_plan(dimension, invlist_bytes_full, mem_budget);
    size_t n_batches = plan.n_batches;
    size_t batch_size = plan.batch_size;

    if (seismic_cluster_params.verbose) {
        fprintf(stderr, "[nsparse] build_inverted_lists: n_docs=%zu, dimension=%zu, "
                "element_size=%zu, lambda=%d, beta=%d, n_batches=%zu, batch_size=%zu\n",
                n_docs, dimension, element_size, lambda, beta, n_batches, batch_size);
    }

    for (size_t batch_start = 0; batch_start < dimension;
         batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, dimension);
        size_t this_batch = batch_end - batch_start;

        // Parallel CSR → inverted list construction
        auto batch_invlists = build_batch_invlists(
            vectors, element_size, n_docs, indptr_data, indices_data,
            values_data, batch_start, batch_end);

        // Count non-empty lists for diagnostics
        ListStats stats = count_non_empty_lists(*batch_invlists, this_batch);
        if (seismic_cluster_params.verbose) {
            fprintf(stderr, "[nsparse] batch [%zu, %zu): %zu/%zu non-empty lists, "
                    "%zu total entries\n",
                    batch_start, batch_end, stats.non_empty, this_batch,
                    stats.total_entries);
        }

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
        if (seismic_cluster_params.verbose) {
            fprintf(stderr, "[nsparse] batch done: %zu lists clustered\n",
                    clustered_count.load());
        }
    }

    return clustered_inverted_lists;
}

void build_and_save_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params,
    IOWriter* io_writer) {
    int lambda =
        calculate_lambda(seismic_cluster_params.lambda, vectors->num_vectors());
    int beta = calculate_beta(seismic_cluster_params.beta, lambda);
    size_t dimension = config.dimension;
    size_t element_size = config.element_size;
    if (dimension == 0) {
        throw std::runtime_error(
            "build_and_save_inverted_lists_clusters: dimension must be > 0");
    }

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

    BatchPlan plan =
        compute_batch_plan(dimension, invlist_bytes_full, mem_budget);
    size_t n_batches = plan.n_batches;
    size_t batch_size = plan.batch_size;

    if (seismic_cluster_params.verbose) {
        fprintf(stderr,
                "[nsparse] build_and_save_inverted_lists: n_docs=%zu, "
                "dimension=%zu, element_size=%zu, lambda=%d, beta=%d, "
                "n_batches=%zu, batch_size=%zu\n",
                n_docs, dimension, element_size, lambda, beta, n_batches,
                batch_size);
    }

    // Write the total dimension count (number of InvertedListClusters)
    io_writer->write(&dimension, sizeof(dimension), 1);

    for (size_t batch_start = 0; batch_start < dimension;
         batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, dimension);
        size_t this_batch = batch_end - batch_start;

        auto batch_invlists = build_batch_invlists(
            vectors, element_size, n_docs, indptr_data, indices_data,
            values_data, batch_start, batch_end);

        ListStats stats = count_non_empty_lists(*batch_invlists, this_batch);
        if (seismic_cluster_params.verbose) {
            fprintf(stderr,
                    "[nsparse] batch [%zu, %zu): %zu/%zu non-empty lists, "
                    "%zu total entries\n",
                    batch_start, batch_end, stats.non_empty, this_batch,
                    stats.total_entries);
        }

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
        if (seismic_cluster_params.verbose) {
            fprintf(stderr, "[nsparse] batch done: %zu lists clustered\n",
                    clustered_count.load());
        }

        // Free inverted lists immediately
        batch_invlists.reset();

        // Serialize this batch's clusters to disk and free
        for (size_t i = 0; i < this_batch; ++i) {
            batch_clusters[i].serialize(io_writer);
        }
        batch_clusters.clear();
        batch_clusters.shrink_to_fit();

        if (seismic_cluster_params.verbose) {
            fprintf(stderr,
                    "[nsparse] batch [%zu, %zu): serialized and freed\n",
                    batch_start, batch_end);
        }
    }
}

}  // namespace detail
}  // namespace nsparse
