/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INVERTED_LIST_H
#define INVERTED_LIST_H

#include <atomic>
#include <memory>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {

class InvertedList {
public:
    InvertedList(size_t element_size);
    InvertedList(const InvertedList&) = delete;
    InvertedList& operator=(const InvertedList&) = delete;
    InvertedList(InvertedList&& other) noexcept
        : element_size_(other.element_size_),
          doc_ids_(std::move(other.doc_ids_)),
          codes_(std::move(other.codes_)) {}

    void add_entries(size_t n_entry, const idx_t* ids, const uint8_t* codes);
    const std::vector<idx_t>& get_doc_ids() const { return doc_ids_; };
    const std::vector<uint8_t>& get_codes() const { return codes_; };

    float get_value_float(size_t index) const;

    std::vector<idx_t> prune_and_keep_doc_ids(size_t lambda);
    void clear();
    float max_value() const;
    size_t size() const { return doc_ids_.size(); }

private:
    size_t element_size_;
    std::vector<idx_t> doc_ids_;
    std::vector<uint8_t> codes_;
    std::atomic<uint8_t> lock_{0};  // 0 = unlocked, 1 = locked
};

class InvertedLists {
public:
    InvertedLists(size_t n_term, size_t element_size);
    virtual ~InvertedLists() = default;
    virtual void add_entry(term_t term_id, idx_t doc_id, const uint8_t* code);
    virtual void add_entries(term_t term_id, size_t n_entry, idx_t* doc_ids,
                             const uint8_t* code) = 0;

    size_t get_n_term() const { return n_term_; }
    size_t get_element_size() const { return element_size_; }

private:
    size_t n_term_;  ///< number of possible key values
    size_t element_size_;
};

class ArrayInvertedLists : public InvertedLists {
public:
    ArrayInvertedLists(size_t n_term, size_t element_size);
    ~ArrayInvertedLists() = default;
    void add_entries(term_t term_id, size_t n_entry, idx_t* doc_ids,
                     const uint8_t* code) override;

    // Iterator support - delegate to lists_
    using iterator = std::vector<InvertedList>::iterator;
    using const_iterator = std::vector<InvertedList>::const_iterator;

    iterator begin() { return lists_.begin(); }
    iterator end() { return lists_.end(); }
    const_iterator begin() const { return lists_.begin(); }
    const_iterator end() const { return lists_.end(); }

    size_t size() const { return lists_.size(); };
    const InvertedList& operator[](size_t i) const { return lists_[i]; };
    InvertedList& operator[](size_t i) { return lists_[i]; };

    static std::unique_ptr<ArrayInvertedLists> build_inverted_lists(
        size_t n_term, size_t element_size, const SparseVectors* vectors);

private:
    std::vector<InvertedList> lists_;
};

}  // namespace nsparse

#endif  // INVERTED_LIST_H