/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/inverted_index.h"

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
#endif

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/ranker.h"

namespace nsparse {
namespace {
constexpr int kScoreWindowSize = 1 << 12;
constexpr int kElementSize = U32;
constexpr idx_t kNoMoreDocs = std::numeric_limits<idx_t>::max();

// Lightweight non-virtual scorer that operates directly on raw arrays.
// Eliminates virtual dispatch, heap allocation, and branch-on-element-size
// from the inner loop.
struct DirectTermScorer {
    const idx_t* doc_ids;  // raw pointer into InvertedList::doc_ids_
    const float* values;   // raw pointer into InvertedList::codes_ (as float)
    int list_size;         // number of postings
    int current_index;     // current position (-1 = before first)
    float query_weight;    // query-side value for this term
    float max_score;       // query_weight * max(posting_value)

    // Current doc_id, or kNoMoreDocs if exhausted.
    idx_t doc_id() const {
        if (current_index < 0 || current_index >= list_size) {
            return kNoMoreDocs;
        }
        return doc_ids[current_index];
    }

    // Advance to next doc. Returns new doc_id.
    idx_t next_doc() {
        ++current_index;
        if (current_index >= list_size) {
            return kNoMoreDocs;
        }
        return doc_ids[current_index];
    }

    // Binary search to advance to >= target_doc.
    idx_t advance_to(idx_t target_doc) {
        if (current_index >= list_size) {
            return kNoMoreDocs;
        }
        if (current_index >= 0 && doc_ids[current_index] >= target_doc) {
            return doc_ids[current_index];
        }
        int start = (current_index < 0) ? 0 : current_index + 1;
        // Use raw pointer binary search — no vector overhead.
        const idx_t* begin = doc_ids + start;
        const idx_t* end = doc_ids + list_size;
        const idx_t* it = std::lower_bound(begin, end, target_doc);
        if (it == end) {
            current_index = list_size;
            return kNoMoreDocs;
        }
        current_index = static_cast<int>(it - doc_ids);
        return *it;
    }

    // Current posting value. Only valid when doc_id() != kNoMoreDocs.
    float value() const { return values[current_index]; }
};

void build_scorers(const term_t* indices, const float* values, int size,
                   const ArrayInvertedLists& inverted_lists,
                   const std::vector<float>& max_term_scores,
                   std::vector<DirectTermScorer>& scorers,
                   std::vector<float>& max_score_prefix) {
    // Build DirectTermScorers: no heap allocation, no virtual dispatch.
    // Operates directly on raw arrays from InvertedList.
    for (int i = 0; i < size; ++i) {
        term_t term_id = indices[i];
        const auto& list = inverted_lists[term_id];
        const auto& doc_ids_vec = list.get_doc_ids();
        const auto& codes = list.get_codes();
        scorers[i].doc_ids = doc_ids_vec.data();
        // element_size is always U32 (4 bytes = sizeof(float)) for
        // InvertedIndex, so codes can be reinterpreted as float* directly.
        scorers[i].values = reinterpret_cast<const float*>(codes.data());
        scorers[i].list_size = static_cast<int>(doc_ids_vec.size());
        scorers[i].current_index = -1;
        scorers[i].query_weight = values[i];
        scorers[i].max_score = values[i] * max_term_scores[term_id];
        // Advance to first doc.
        scorers[i].next_doc();
    }

    // Sort scorers by max_score ascending (lowest contribution first).
    std::sort(scorers.begin(), scorers.end(),
              [](const DirectTermScorer& lhs, const DirectTermScorer& rhs) {
                  return lhs.max_score < rhs.max_score;
              });

    // Precompute prefix sums of max_scores.
    for (int i = 0; i < size; ++i) {
        max_score_prefix[i + 1] = max_score_prefix[i] + scorers[i].max_score;
    }
}

void score_essential_terms(std::vector<DirectTermScorer>& scorers,
                           int first_essential, int window_base, int window_end,
                           std::vector<float>& window_scores,
                           uint64_t* bitmap) {
    // Score essential scorers into the window.
    // Set bitmap bits for touched slots — no branch overhead per posting.
    int size = static_cast<int>(scorers.size());
    for (int i = first_essential; i < size; ++i) {
        auto& scorer = scorers[i];
        const float query_weight = scorer.query_weight;
        while (scorer.current_index < scorer.list_size) {
            idx_t doc = scorer.doc_ids[scorer.current_index];
            if (doc >= window_end) {
                break;
            }
            int slot = doc - window_base;
            bitmap[slot >> 6] |= (1ULL << (slot & 63));
            window_scores[slot] +=
                query_weight * scorer.values[scorer.current_index];
            ++scorer.current_index;
        }
    }
}

void evaluate_window_candidates(std::vector<DirectTermScorer>& scorers,
                                int first_essential, int window_base,
                                const std::vector<float>& max_score_prefix,
                                std::vector<float>& window_scores,
                                const uint64_t* bitmap,
                                detail::TopKHolder<idx_t>& heap) {
    // Iterate only set bits in the bitmap.
    // Each word covers 64 slots; ctzll finds the next set bit.
    static constexpr int kBitmapWords = kScoreWindowSize / 64;
    float threshold = heap.full() ? heap.peek_score() : 0.0F;
    float non_essential_sum = max_score_prefix[first_essential];

    for (int word_idx = 0; word_idx < kBitmapWords; ++word_idx) {
        uint64_t word = bitmap[word_idx];
        while (word != 0) {
#ifdef _MSC_VER
            unsigned long bit_pos;
            _BitScanForward64(&bit_pos, word);
            int bit = static_cast<int>(bit_pos);
#else
            int bit = __builtin_ctzll(word);
#endif
            word &= word - 1;  // clear lowest set bit
            int slot = (word_idx << 6) | bit;

            float essential_score = window_scores[slot];
            window_scores[slot] = 0.0F;

            idx_t doc = window_base + slot;

            if (essential_score + non_essential_sum <= threshold) {
                continue;
            }

            float total_score = essential_score;
            for (int j = first_essential - 1; j >= 0; --j) {
                auto& scorer = scorers[j];
                scorer.advance_to(doc);
                if (scorer.doc_id() == doc) {
                    total_score += scorer.query_weight *
                                   scorer.values[scorer.current_index];
                }
                if (total_score + max_score_prefix[j] <= threshold) {
                    break;
                }
            }

            heap.add(total_score, doc);
            if (heap.full()) {
                threshold = heap.peek_score();
            }
        }
    }
}

}  // namespace

InvertedIndex::InvertedIndex(int dim) : Index(dim) {}

void InvertedIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = kElementSize,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indptr_size, indices, nnz,
                          reinterpret_cast<const uint8_t*>(values),
                          nnz * kElementSize);
}

void InvertedIndex::build() {
    inverted_lists_ = ArrayInvertedLists::build_inverted_lists(
        get_dimension(), kElementSize, vectors_.get());
    vectors_.reset();

    // Posting lists are already sorted by doc_id because
    // build_inverted_lists iterates documents in ascending order.
    // Precompute max values per term.
    size_t n_terms = inverted_lists_->size();
    max_term_scores_.resize(n_terms, 0.0F);
    for (size_t t = 0; t < n_terms; ++t) {
        auto& list = (*inverted_lists_)[t];
        assert(std::is_sorted(list.get_doc_ids().begin(),
                              list.get_doc_ids().end()));
        max_term_scores_[t] = list.max_value();
    }
}

auto InvertedIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                           const float* values, int k,
                           SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    if (inverted_lists_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];
    SparseVectors query_vectors({.element_size = kElementSize,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                              reinterpret_cast<const uint8_t*>(values),
                              nnz * kElementSize);
    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);

    const auto* query_indptr = query_vectors.indptr_data();
    const auto* query_indices = query_vectors.indices_data();
    const auto* query_values = query_vectors.values_data_float();

#pragma omp parallel for
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const idx_t start = query_indptr[query_idx];
        const size_t len = query_indptr[query_idx + 1] - start;

        auto [distances, labels] =
            single_query(query_indices + start, query_values + start, len, k);
        result_distances[query_idx] = std::move(distances);
        result_labels[query_idx] = std::move(labels);
    }

    return {result_distances, result_labels};
}

auto InvertedIndex::single_query(const term_t* indices, const float* values,
                                 int size, int k) -> pair_of_score_id_vector_t {
    if (size == 0) {
        std::vector<float> scores(k, -1.0F);
        std::vector<idx_t> ids(k, INVALID_IDX);
        return {scores, ids};
    }

    detail::TopKHolder<idx_t> heap(k);

    // Stack-local scorers — small and hot in L1 cache.
    std::vector<DirectTermScorer> scorers(size);
    std::vector<float> max_score_prefix(size + 1, 0.0F);

    // Thread-local window buffer — 16KB, reused across queries.
    thread_local std::vector<float> window_scores(kScoreWindowSize, 0.0F);

    build_scorers(indices, values, size, *inverted_lists_, max_term_scores_,
                  scorers, max_score_prefix);

    int first_essential = 0;

    // Bitmap to track which window slots have non-zero scores.
    // 4096 bits = 64 uint64_t words = 512 bytes (8 cache lines).
    static constexpr int kBitmapWords = kScoreWindowSize / 64;
    thread_local uint64_t bitmap[kBitmapWords];

    while (true) {
        // Re-partition based on current heap threshold.
        if (heap.full()) {
            float threshold = heap.peek_score();
            first_essential = 0;
            for (int i = 0; i < size; ++i) {
                if (max_score_prefix[i + 1] < threshold) {
                    first_essential = i + 1;
                } else {
                    break;
                }
            }
        }

        // Find the minimum doc_id among essential scorers.
        idx_t min_doc_id = kNoMoreDocs;
        for (int i = first_essential; i < size; ++i) {
            idx_t doc = scorers[i].doc_id();
            if (doc < min_doc_id) {
                min_doc_id = doc;
            }
        }
        if (min_doc_id == kNoMoreDocs) {
            break;
        }

        // Define the scoring window.
        int window_base = (min_doc_id / kScoreWindowSize) * kScoreWindowSize;
        int window_end = window_base + kScoreWindowSize;

        std::memset(bitmap, 0, sizeof(bitmap));
        score_essential_terms(scorers, first_essential, window_base, window_end,
                              window_scores, bitmap);
        evaluate_window_candidates(scorers, first_essential, window_base,
                                   max_score_prefix, window_scores, bitmap,
                                   heap);
    }

    auto [result_scores, ids] = heap.top_k_items_descending();
    result_scores.resize(k, -1.0F);
    ids.resize(k, INVALID_IDX);
    return {result_scores, ids};
}

void InvertedIndex::write_index(IOWriter* io_writer) {
    // Write inverted lists
    size_t n_terms = inverted_lists_ ? inverted_lists_->size() : 0;
    io_writer->write(&n_terms, sizeof(size_t), 1);
    size_t element_size = kElementSize;
    io_writer->write(&element_size, sizeof(size_t), 1);

    for (size_t i = 0; i < n_terms; ++i) {
        const auto& list = (*inverted_lists_)[i];
        const auto& doc_ids = list.get_doc_ids();
        const auto& codes = list.get_codes();

        size_t list_size = doc_ids.size();
        io_writer->write(&list_size, sizeof(size_t), 1);
        if (list_size > 0) {
            io_writer->write(const_cast<idx_t*>(doc_ids.data()), sizeof(idx_t),
                             list_size);
            size_t codes_size = codes.size();
            io_writer->write(const_cast<uint8_t*>(codes.data()),
                             sizeof(uint8_t), codes_size);
        }
    }

    // Write max_term_scores
    size_t scores_size = max_term_scores_.size();
    io_writer->write(&scores_size, sizeof(size_t), 1);
    if (scores_size > 0) {
        io_writer->write(max_term_scores_.data(), sizeof(float), scores_size);
    }
}

void InvertedIndex::read_index(IOReader* io_reader) {
    // Read inverted lists
    size_t n_terms = 0;
    io_reader->read(&n_terms, sizeof(size_t), 1);
    size_t element_size = 0;
    io_reader->read(&element_size, sizeof(size_t), 1);

    if (n_terms > 0) {
        inverted_lists_ =
            std::make_unique<ArrayInvertedLists>(n_terms, element_size);
        for (size_t i = 0; i < n_terms; ++i) {
            size_t list_size = 0;
            io_reader->read(&list_size, sizeof(size_t), 1);
            if (list_size > 0) {
                std::vector<idx_t> doc_ids(list_size);
                io_reader->read(doc_ids.data(), sizeof(idx_t), list_size);
                size_t codes_size = list_size * element_size;
                std::vector<uint8_t> codes(codes_size);
                io_reader->read(codes.data(), sizeof(uint8_t), codes_size);
                inverted_lists_->add_entries(static_cast<term_t>(i), list_size,
                                             doc_ids.data(), codes.data());
            }
        }
    }

    // Read max_term_scores
    size_t scores_size = 0;
    io_reader->read(&scores_size, sizeof(size_t), 1);
    if (scores_size > 0) {
        max_term_scores_.resize(scores_size);
        io_reader->read(max_term_scores_.data(), sizeof(float), scores_size);
    }
}

}  // namespace nsparse
