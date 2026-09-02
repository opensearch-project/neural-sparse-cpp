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
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
namespace {

// A window's [begin, end) slice of the term space.
struct TermWindow {
    size_t begin;
    size_t end;
    [[nodiscard]] size_t size() const { return end - begin; }
};

// Cuts [0, dimension) into `batches` near-equal windows.
std::vector<TermWindow> make_windows(size_t dim, size_t batches) {
    // Bounds are size_t, not term_t: dimension may be up to 65536 (term_t is
    // uint16), so a term_t window boundary would wrap and silently drop terms.
    const size_t per_batch = (dim + batches - 1) / batches;
    std::vector<TermWindow> windows;
    for (size_t begin = 0; begin < dim; begin += per_batch) {
        windows.push_back({begin, std::min(dim, begin + per_batch)});
    }
    return windows;
}

// Postings per term, over the whole corpus.
//
// Reads only the CSR's indices, not its values, so on a mapped corpus this
// faults in a third of the bytes a full pass would.
std::vector<size_t> count_postings_per_term(const SparseVectors& vectors,
                                            size_t dim) {
    std::vector<size_t> counts(dim, 0);
    const idx_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const idx_t nnz = indptr[vectors.num_vectors()];
    for (idx_t j = 0; j < nnz; ++j) {
        const size_t term = indices[j];
        if (term >= dim) {
            throw std::invalid_argument(
                "for_each_clustered_window: corpus has term " +
                std::to_string(term) + " outside dimension " +
                std::to_string(dim));
        }
        ++counts[term];
    }
    return counts;
}

// The inverted lists of one term window, sized exactly from the counting pass so
// no list ever grows.
//
// Doc ids arrive in ascending order, because fill_from_corpus walks documents
// ascending, which is what a single-window build produces. That matters beyond
// tidiness: pruning sorts by value with a non-stable sort, and k-means then
// consumes the kept ids in order, so a different posting order is a different
// index.
class WindowLists {
public:
    WindowLists(const std::vector<size_t>& term_counts,
                const TermWindow& window, size_t element_size)
        : element_size_(element_size),
          lists_(window.size(), element_size),
          ids_(window.size()),
          codes_(window.size()),
          fill_(window.size(), 0) {
        for (size_t local = 0; local < window.size(); ++local) {
            const size_t count = term_counts[window.begin + local];
            ids_[local].resize(count);
            codes_[local].resize(count * element_size);
        }
    }

    void add(size_t local_term, idx_t doc_id, const uint8_t* code) {
        const size_t slot = fill_[local_term]++;
        if (slot >= ids_[local_term].size()) {
            // The list was sized from the counting pass, so more postings than
            // that means the corpus changed under us. One compare on a path that
            // is bound by reading the corpus, and it is the difference between
            // an exception and a heap overflow.
            throw std::runtime_error(
                "for_each_clustered_window: corpus changed during the build");
        }
        ids_[local_term][slot] = doc_id;
        std::memcpy(codes_[local_term].data() + slot * element_size_, code,
                    element_size_);
    }

    // Hands the staged postings to the lists themselves. Separate from add()
    // because set_entries adopts whole buffers, which is the only way to fill a
    // list without the per-posting locking add_entry pays for.
    ArrayInvertedLists& seal() {
        for (size_t local = 0; local < ids_.size(); ++local) {
            lists_[local].set_entries(std::move(ids_[local]),
                                      std::move(codes_[local]));
        }
        ids_ = {};
        codes_ = {};
        return lists_;
    }

private:
    size_t element_size_;
    ArrayInvertedLists lists_;
    std::vector<std::vector<idx_t>> ids_;
    std::vector<std::vector<uint8_t>> codes_;
    std::vector<size_t> fill_;
};

// Finds the window's postings by scanning the corpus.
//
// Serial over documents, ascending: that ordering is what makes the output
// independent of the window count (see WindowLists), and the pass is bound by
// reading the corpus rather than by the work per posting, so threading it would
// cost the ordering and buy nothing.
void fill_from_corpus(const SparseVectors& vectors, const TermWindow& window,
                      size_t element_size, WindowLists* lists) {
    const idx_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const uint8_t* codes = vectors.values_data();
    const auto n_docs = static_cast<idx_t>(vectors.num_vectors());
    for (idx_t doc = 0; doc < n_docs; ++doc) {
        for (idx_t j = indptr[doc]; j < indptr[doc + 1]; ++j) {
            const size_t term = indices[j];
            if (term < window.begin || term >= window.end) {
                continue;
            }
            lists->add(term - window.begin, doc,
                       codes + static_cast<size_t>(j) * element_size);
        }
    }
}

// Lists per OpenMP chunk, at most. Posting lists are wildly uneven in length, so
// they are handed out dynamically rather than split up front.
constexpr size_t kMaxClusterChunk = 64;

// Chunks a window should break into, so the threads have something to steal. A
// window can be far narrower than the whole term space -- at 64 batches over
// 8192 terms it is 128 lists, which the flat chunk above would hand out as two
// chunks and leave every thread but two idle.
constexpr size_t kMinClusterChunks = 256;

// SeismicClusterParameters with the "compute me a default" values already
// resolved. Resolved once for the whole build rather than per window: lambda
// comes from the GLOBAL corpus size, and a window-local one would prune
// differently and make the window count visible in the output.
struct ResolvedParameters {
    int lambda;
    int beta;
    float alpha;
    uint32_t base_seed;
};

std::vector<InvertedListClusters> cluster_window(const SparseVectors& vectors,
                                                 ArrayInvertedLists& lists,
                                                 const TermWindow& window,
                                                 const ResolvedParameters& params) {
    std::vector<InvertedListClusters> clustered(window.size());
    const auto chunk = static_cast<int64_t>(std::clamp<size_t>(
        window.size() / kMinClusterChunks, 1, kMaxClusterChunk));
#pragma omp parallel for schedule(dynamic, chunk)
    for (int64_t local = 0; local < static_cast<int64_t>(window.size());
         ++local) {
        auto& invlist = lists[local];
        const auto& doc_ids = invlist.prune_and_keep_doc_ids(params.lambda);
        // Offset by the list's own GLOBAL term id, so neither the window it
        // landed in nor which thread picked it up can change the result.
        const auto seed =
            params.base_seed +
            static_cast<uint32_t>(window.begin + static_cast<size_t>(local));
        InvertedListClusters ilc(
            RandomKMeans::train(&vectors, doc_ids, params.beta, seed));
        ilc.summarize(&vectors, params.alpha);
        clustered[local] = std::move(ilc);
        invlist.clear();
    }
    return clustered;
}

}  // namespace

void for_each_clustered_window(const SparseVectors* vectors,
                               const SparseVectorsConfig& config,
                               const SeismicClusterParameters& params,
                               const ClusteredWindowSink& sink) {
    if (vectors == nullptr || vectors->num_vectors() == 0) {
        return;
    }
    if (vectors->get_element_size() != config.element_size) {
        throw std::invalid_argument(
            "for_each_clustered_window: corpus element width does not match the "
            "index's");
    }

    const size_t dim = config.dimension;
    const size_t batches =
        std::max<size_t>(1, std::min(params.batch_clustering.batch_size, dim));

    const int lambda = calculate_lambda(params.lambda, vectors->num_vectors());
    const ResolvedParameters resolved = {
        .lambda = lambda,
        .beta = calculate_beta(params.beta, lambda),
        .alpha = params.alpha,
        // Resolved once, outside the loop: std::random_device usually opens
        // /dev/urandom per construction, so drawing per posting list would put a
        // syscall on every iteration with every thread doing it. Once for the
        // whole build, not per window, or the window count would be observable.
        .base_seed = params.seed == kRandomSeed
                         ? std::random_device{}()
                         : static_cast<uint32_t>(params.seed)};

    // Exact per-term sizes, so no window's list ever grows and the bulk
    // set_entries path can be used instead of per-posting locking. Also the only
    // place a term outside the dimension is caught: the mapped read does not
    // range-check.
    const std::vector<size_t> term_counts =
        count_postings_per_term(*vectors, dim);

    for (const TermWindow& window : make_windows(dim, batches)) {
        WindowLists lists(term_counts, window, config.element_size);
        fill_from_corpus(*vectors, window, config.element_size, &lists);
        sink(window.begin,
             cluster_window(*vectors, lists.seal(), window, resolved));
        // The window's lists and clusters are freed here, before the next
        // window is built. That is what bounds the memory.
    }
}

}  // namespace nsparse::detail
