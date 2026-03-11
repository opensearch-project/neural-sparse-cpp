/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/invlists/inverted_lists.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/types.h"

namespace nsparse {

namespace {
// RAII lock guard for spinlock
class LockGuard {
public:
    explicit LockGuard(std::atomic<uint8_t>& lock) : lock_(lock) {
        while (lock_.exchange(1, std::memory_order_acquire) != 0) {
            // Busy-wait
        }
    }
    ~LockGuard() { lock_.store(0, std::memory_order_release); }
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    std::atomic<uint8_t>& lock_;
};

static std::vector<std::pair<float, idx_t>> create_value_doc_id_pair(
    const std::vector<uint8_t>& codes, const size_t element_size,
    const std::vector<idx_t>& doc_ids, const size_t n_docs) {
    std::vector<std::pair<float, idx_t>> value_doc_pairs;
    value_doc_pairs.reserve(n_docs);

    for (size_t i = 0; i < n_docs; ++i) {
        float value = 0.0F;
        const uint8_t* value_ptr = codes.data() + i * element_size;
        if (element_size == U32) {
            value = *reinterpret_cast<const float*>(value_ptr);
        } else if (element_size == U16) {
            value = static_cast<float>(
                *reinterpret_cast<const uint16_t*>(value_ptr));
        } else {
            value = static_cast<float>(*value_ptr);
        }
        value_doc_pairs.emplace_back(value, doc_ids[i]);
    }
    return value_doc_pairs;
}

}  // namespace

InvertedList::InvertedList(size_t element_size) : element_size_(element_size) {}

void InvertedList::add_entries(size_t n_entry, const idx_t* ids,
                               const uint8_t* codes) {
    if (n_entry == 0) {
        return;
    }

    LockGuard guard(lock_);

    // Critical section - modify data structures
    doc_ids_.insert(doc_ids_.end(), ids, ids + n_entry);
    codes_.insert(codes_.end(), codes, codes + (n_entry * element_size_));
}

void InvertedList::clear() {
    doc_ids_.clear();
    doc_ids_.shrink_to_fit();
    codes_.clear();
    codes_.shrink_to_fit();
}

float InvertedList::max_value() const {
    float max_val = 0.0F;
    size_t n = doc_ids_.size();
    for (size_t i = 0; i < n; ++i) {
        float v = get_value_float(i);
        if (v > max_val) max_val = v;
    }
    return max_val;
}

float InvertedList::get_value_float(size_t index) const {
    const uint8_t* value_ptr = codes_.data() + (index * element_size_);
    if (element_size_ == U32) {
        return *reinterpret_cast<const float*>(value_ptr);
    }
    if (element_size_ == U16) {
        return static_cast<float>(
            *reinterpret_cast<const uint16_t*>(value_ptr));
    }
    return static_cast<float>(*value_ptr);
}

std::vector<idx_t> InvertedList::prune_and_keep_doc_ids(size_t lambda) {
    LockGuard guard(lock_);

    size_t n_docs = doc_ids_.size();
    if (lambda <= 0 || n_docs == 0 || lambda >= n_docs) {
        return doc_ids_;
    }

    // Create pairs of (float_value, index) for sorting
    std::vector<std::pair<float, idx_t>> value_doc_pairs =
        create_value_doc_id_pair(codes_, element_size_, doc_ids_, n_docs);

    // Sort by float value in descending order (highest first)
    std::ranges::sort(value_doc_pairs, [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<idx_t> kept_doc_ids;
    kept_doc_ids.reserve(lambda);
    std::transform(value_doc_pairs.begin(), value_doc_pairs.begin() + lambda,
                   std::back_inserter(kept_doc_ids),
                   [](const auto& pair) { return pair.second; });
    return kept_doc_ids;
}

InvertedLists::InvertedLists(size_t n_term, size_t element_size)
    : n_term_(n_term), element_size_(element_size) {}

void InvertedLists::add_entry(term_t term_id, idx_t doc_id,
                              const uint8_t* code) {
    add_entries(term_id, 1, &doc_id, code);
}

ArrayInvertedLists::ArrayInvertedLists(size_t n_term, size_t element_size)
    : InvertedLists(n_term, element_size) {
    lists_.reserve(n_term);
    for (size_t i = 0; i < n_term; ++i) {
        lists_.emplace_back(element_size);
    }
}

void ArrayInvertedLists::add_entries(term_t term_id, size_t n_entry,
                                     idx_t* doc_ids, const uint8_t* code) {
    if (term_id >= get_n_term()) {
        throw std::invalid_argument("term_id out of range");
    }
    auto& inverted_list = lists_[term_id];
    inverted_list.add_entries(n_entry, doc_ids, code);
}

std::unique_ptr<ArrayInvertedLists> ArrayInvertedLists::build_inverted_lists(
    size_t n_term, size_t element_size, const SparseVectors* vectors) {
    std::unique_ptr<ArrayInvertedLists> inverted_lists =
        std::make_unique<ArrayInvertedLists>(n_term, element_size);
    size_t n_docs = vectors->num_vectors();

    const auto& indptr_data = vectors->indptr_data();
    const auto& indices_data = vectors->indices_data();
    const auto& values_data = vectors->values_data();

    // inverted_lists.add_entry is thread safe
    for (size_t i = 0; i < n_docs; ++i) {
        int start = indptr_data[i];
        int n_tokens = indptr_data[i + 1] - indptr_data[i];
        for (size_t j = start; j < start + n_tokens; ++j) {
            term_t term_id = indices_data[j];
            inverted_lists->add_entry(term_id, i,
                                      values_data + j * element_size);
        }
    }
    return inverted_lists;
}

}  // namespace nsparse