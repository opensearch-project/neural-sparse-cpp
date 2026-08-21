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

#include "nsparse/io/mmap_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"

namespace nsparse {

class InvertedList : public MmapSerializable {
public:
    InvertedList(size_t element_size);
    // Move-only, because Buf is: a Buf may borrow memory it does not own, and
    // copying one would either alias the lender or silently deep-copy.
    InvertedList(const InvertedList&) = delete;
    InvertedList& operator=(const InvertedList&) = delete;
    InvertedList(InvertedList&& other) noexcept
        : element_size_(other.element_size_),
          doc_ids_(std::move(other.doc_ids_)),
          codes_(std::move(other.codes_)) {}

    void add_entries(size_t n_entry, const idx_t* ids, const uint8_t* codes);

    // Replaces the contents, adopting the caller's buffers. The bulk path
    // build_inverted_lists takes: an append has to round-trip both Bufs through
    // a vector, which is not something to pay per posting.
    void set_entries(std::vector<idx_t>&& doc_ids,
                     std::vector<uint8_t>&& codes);

    const Buf<idx_t>& get_doc_ids() const { return doc_ids_; };
    const Buf<uint8_t>& get_codes() const { return codes_; };

    float get_value_float(size_t index) const;

    std::vector<idx_t> prune_and_keep_doc_ids(size_t lambda);
    void clear();
    float max_value() const;
    size_t size() const { return doc_ids_.size(); }

    // Layout: the entry count, then the doc ids and the codes, each preceded by
    // padding to its element's alignment so a mapped reader can borrow it in
    // place (see io/align.h). The element width is not repeated per list; the
    // enclosing ArrayInvertedLists writes it once.
    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;
    void mmap_deserialize(MmapCursor* cursor) override;

private:
    size_t element_size_;
    Buf<idx_t> doc_ids_;
    Buf<uint8_t> codes_;
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

    // Layout: the term count and the element width, then each list in term
    // order.
    void serialize(IOWriter* writer) const;

    // Factories rather than a deserialize() on an existing object: the term
    // count comes from the file, and it is fixed at construction.
    //
    // `element_size` is the width the caller loads the codes at. A file
    // declaring another is rejected before any list is walked: the width sets
    // how many code bytes each list holds, so reading on would desynchronize
    // and the following list sizes would be garbage.
    static std::unique_ptr<ArrayInvertedLists> read(IOReader* reader,
                                                    size_t element_size);

    // Same layout read() walks, with every list borrowing from the mapping
    // instead of copying it. The mapping must outlive the result.
    static std::unique_ptr<ArrayInvertedLists> map(MmapCursor* cursor,
                                                   size_t element_size);

private:
    std::vector<InvertedList> lists_;
};

}  // namespace nsparse

#endif  // INVERTED_LIST_H
