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

#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/id_selector.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

namespace nsparse {

// How a build bounds its own memory.
//
// Clustering the whole term space at once holds two intermediates for the whole
// corpus: the inverted lists (every posting) and then the clustered lists. Both
// scale with the corpus's non-zeros, which is what puts a ceiling on the corpus
// an index can be built from. Splitting the term space into `batch_size`
// contiguous windows and finishing one window before starting the next makes
// the first of those proportional to a window instead.
//
// `batch_file_output_path` bounds the second as well: with batch_size > 1, each
// window's clustered lists are spilled to a temporary file there and freed as
// they are produced, and the finished lists are then mapped back out of it, so
// the build never holds more than one window's worth. See
// spill_clustered_lists.
//
// The two knobs are therefore only useful together, and each is ignored without
// the other -- see effective_batch_size.
struct BatchClusteringOption {
    // Contiguous term windows. <= 1 means one window, i.e. no batching. Clamped
    // to the dimension, since a window cannot be narrower than one term.
    size_t batch_size = 1;
    // An existing directory the build may spill windows into. Scratch, not
    // output: build() writes no index there and leaves nothing behind, and
    // serializing an index remains write_index's job. Used only when batch_size
    // > 1, a single window being an ordinary build with nothing to spill.
    std::string batch_file_output_path;

    // Windows the build should actually run, which is 1 unless there is
    // somewhere to write them.
    //
    // Splitting the term space without a path to write to bounds only the first
    // intermediate: the clustered lists still accumulate for the whole corpus,
    // and they are the bulkier of the two by roughly kClusterCostRatio. So it
    // buys a fraction of the peak while costing a corpus pass per window, and
    // the peak stays where an unbatched build's is. Not worth doing quietly on
    // a caller's behalf, so `batch_size` alone is honoured as no batching at
    // all.
    [[nodiscard]] size_t effective_batch_size() const {
        return batch_file_output_path.empty() ? 1
                                              : std::max<size_t>(1, batch_size);
    }
};

// Draw fresh entropy at build time, which makes the build unreproducible. Any
// other value makes it reproducible.
constexpr int kRandomSeed = -1;

struct SeismicClusterParameters {
    int lambda;
    int beta;
    float alpha;
    BatchClusteringOption batch_clustering;
    // Fix this to make a build reproducible; two builds of the same corpus
    // differ by default.
    int seed = kRandomSeed;
};

namespace detail {

constexpr int kDefaultLambda = -1;
constexpr float kDefaultPostingPruneRatio = 0.0005F;
constexpr int kDefaultPostingMinimumLength = 160;
constexpr float kDefaultBetaRatio = 0.1F;
constexpr int kDefaultBeta = -1;
constexpr float kDefaultAlpha = 0.4F;

// const rather than constexpr: BatchClusteringOption holds a std::string for
// the output path, which is not a literal type.
inline const SeismicClusterParameters kDefaultSeismicClusterParams = {
    .lambda = kDefaultLambda, .beta = kDefaultBeta, .alpha = kDefaultAlpha};

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

inline float compute_similarity(idx_t doc_id, const idx_t* indptr,
                                const term_t* indices, const uint8_t* values,
                                const uint8_t* dense, size_t element_size) {
    const idx_t start = indptr[doc_id];
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

// Clusters and summarizes the posting lists of one term window at a time,
// handing each window to `sink` in ascending term order.
//
// This is the one place the seismic family's build work lives: every index type
// reaches it, either through build_inverted_lists_clusters below or through the
// spilling build in seismic_batched_build.h. The element width comes from
// `config`, so a quantizing index gets the same treatment as a float one -- the
// values in `vectors` are already encoded by the time they arrive here.
//
// `sink` receives the window's global first term and its lists, and must not
// hold on to them: they are freed as soon as it returns, which is what bounds
// the memory. Windows come from
// params.batch_clustering.effective_batch_size(), so a caller that set a batch
// size but no output path gets one window -- see BatchClusteringOption.
//
// Every window's lambda and beta are computed from the GLOBAL corpus, and every
// list's k-means seed from its own GLOBAL term id, so the window count cannot
// change what is produced -- see the batched-build tests, which assert file
// equality against an unbatched build.
using ClusteredWindowSink = std::function<void(
    size_t term_begin, std::vector<InvertedListClusters>&& clusters)>;

void for_each_clustered_window(const SparseVectors* vectors,
                               const SparseVectorsConfig& config,
                               const SeismicClusterParameters& params,
                               const ClusteredWindowSink& sink);

// Every term's clustered posting list, in term order. The whole-corpus form of
// for_each_clustered_window, and so the unbatched build: the result is retained
// in full, which is what bounds this by the clustered lists rather than by a
// window. Callers reach it without an output path, which is why that is also
// the case where the window count is dropped.
inline std::vector<InvertedListClusters> build_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params) {
    std::vector<InvertedListClusters> clustered(config.dimension);
    for_each_clustered_window(
        vectors, config, seismic_cluster_params,
        [&clustered](size_t term_begin,
                     std::vector<InvertedListClusters>&& window) {
            std::move(window.begin(), window.end(),
                      clustered.begin() + static_cast<ptrdiff_t>(term_begin));
        });
    return clustered;
}

}  // namespace detail
}  // namespace nsparse

#endif  // SEISMIC_COMMON_H