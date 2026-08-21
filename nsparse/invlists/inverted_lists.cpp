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

#include "nsparse/io/align.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

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
    const Buf<uint8_t>& codes, const size_t element_size,
    const Buf<idx_t>& doc_ids, const size_t n_docs) {
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

// The codes are reinterpreted as element_size-wide words, so a stored width
// other than the one the caller reads at would stride halfway into each value.
// It also sets each list's length on the wire, so reading on would
// desynchronize everything after the first list.
void throw_if_foreign_element_size(size_t stored, size_t expected) {
    if (stored != expected) {
        throw std::runtime_error(
            "stored posting element width is not the one this index reads");
    }
}

}  // namespace

InvertedList::InvertedList(size_t element_size) : element_size_(element_size) {}

void InvertedList::add_entries(size_t n_entry, const idx_t* ids,
                               const uint8_t* codes) {
    if (n_entry == 0) {
        return;
    }

    LockGuard guard(lock_);

    // Critical section - modify data structures.
    // Taken out, extended and handed back: a Buf is fixed-length, and the
    // vectors' capacity survives the round trip so repeated appends stay
    // amortized. A mapped list throws here rather than write to the mapping.
    std::vector<idx_t> doc_ids = doc_ids_.take_vector();
    std::vector<uint8_t> code_bytes = codes_.take_vector();
    doc_ids.insert(doc_ids.end(), ids, ids + n_entry);
    code_bytes.insert(code_bytes.end(), codes,
                      codes + (n_entry * element_size_));
    doc_ids_ = Buf<idx_t>::own(std::move(doc_ids));
    codes_ = Buf<uint8_t>::own(std::move(code_bytes));
}

void InvertedList::set_entries(std::vector<idx_t>&& doc_ids,
                               std::vector<uint8_t>&& codes) {
    if (codes.size() != doc_ids.size() * element_size_) {
        throw std::invalid_argument(
            "codes must hold one element_size-wide value per doc id");
    }
    doc_ids_ = Buf<idx_t>::own(std::move(doc_ids));
    codes_ = Buf<uint8_t>::own(std::move(codes));
}

void InvertedList::clear() {
    doc_ids_ = {};
    codes_ = {};
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
        return {doc_ids_.begin(), doc_ids_.end()};
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

void InvertedList::serialize(IOWriter* writer) const {
    size_t list_size = doc_ids_.size();
    writer->write(&list_size, sizeof(size_t), 1);
    io_align::write_padded(writer, doc_ids_.data(), list_size);
    // The codes are bytes on the wire but reinterpreted as element_size-wide
    // words on read, so they are padded to that width, not to 1.
    io_align::write_padded(writer, codes_.data(), codes_.size(), element_size_);
}

void InvertedList::deserialize(IOReader* reader) {
    size_t list_size = 0;
    reader->read(&list_size, sizeof(size_t), 1);
    doc_ids_ = io_align::read_padded<idx_t>(reader, list_size);
    codes_ = io_align::read_padded<uint8_t>(
        reader, checked_mul(list_size, element_size_), element_size_);
}

void InvertedList::mmap_deserialize(MmapCursor* cursor) {
    throw_if_null(cursor, "cursor must not be null");
    const auto list_size = cursor->read_scalar<size_t>();
    doc_ids_ = io_align::borrow_padded<idx_t>(cursor, list_size);
    codes_ = io_align::borrow_padded<uint8_t>(
        cursor, checked_mul(list_size, element_size_), element_size_);
    // read_array only checks the alignment of the type it hands back, and that
    // is one byte here; the values are loaded at element_size_.
    if (list_size > 0 &&
        reinterpret_cast<uintptr_t>(codes_.data()) % element_size_ != 0) {
        throw std::runtime_error(
            "mmap: posting values are misaligned for the element size");
    }
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
    throw_if_null(vectors, "vectors must not be null");
    std::unique_ptr<ArrayInvertedLists> inverted_lists =
        std::make_unique<ArrayInvertedLists>(n_term, element_size);
    size_t n_docs = vectors->num_vectors();

    const auto* indptr_data = vectors->indptr_data();
    const auto* indices_data = vectors->indices_data();
    const auto* values_data = vectors->values_data();

    // Counted first, then filled into exactly sized buffers. Appending posting
    // by posting instead would round-trip a list's Buf through a vector on
    // every entry, and leave the geometric growth's slack behind.
    std::vector<size_t> counts(n_term, 0);
    for (size_t i = 0; i < n_docs; ++i) {
        for (idx_t j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
            const term_t term_id = indices_data[j];
            if (term_id >= n_term) {
                throw std::invalid_argument("term_id out of range");
            }
            ++counts[term_id];
        }
    }

    std::vector<std::vector<idx_t>> doc_ids(n_term);
    std::vector<std::vector<uint8_t>> codes(n_term);
    for (size_t term = 0; term < n_term; ++term) {
        doc_ids[term].reserve(counts[term]);
        codes[term].reserve(counts[term] * element_size);
    }
    // Documents in ascending order, so every posting list comes out sorted by
    // doc id -- which the search path relies on.
    for (size_t i = 0; i < n_docs; ++i) {
        for (idx_t j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
            const term_t term_id = indices_data[j];
            doc_ids[term_id].push_back(static_cast<idx_t>(i));
            const uint8_t* code = values_data + (j * element_size);
            codes[term_id].insert(codes[term_id].end(), code,
                                  code + element_size);
        }
    }
    for (size_t term = 0; term < n_term; ++term) {
        if (counts[term] == 0) {
            continue;  // leave the list empty rather than own an empty vector
        }
        (*inverted_lists)[term].set_entries(std::move(doc_ids[term]),
                                            std::move(codes[term]));
    }
    return inverted_lists;
}

void ArrayInvertedLists::serialize(IOWriter* writer) const {
    size_t n_term = get_n_term();
    writer->write(&n_term, sizeof(size_t), 1);
    size_t element_size = get_element_size();
    writer->write(&element_size, sizeof(size_t), 1);
    for (const auto& list : lists_) {
        list.serialize(writer);
    }
}

std::unique_ptr<ArrayInvertedLists> ArrayInvertedLists::read(
    IOReader* reader, size_t element_size) {
    throw_if_null(reader, "reader must not be null");
    size_t n_term = 0;
    reader->read(&n_term, sizeof(size_t), 1);
    size_t stored_element_size = 0;
    reader->read(&stored_element_size, sizeof(size_t), 1);
    throw_if_foreign_element_size(stored_element_size, element_size);

    auto lists = std::make_unique<ArrayInvertedLists>(n_term, element_size);
    for (auto& list : *lists) {
        list.deserialize(reader);
    }
    return lists;
}

std::unique_ptr<ArrayInvertedLists> ArrayInvertedLists::map(
    MmapCursor* cursor, size_t element_size) {
    throw_if_null(cursor, "cursor must not be null");
    const auto n_term = cursor->read_scalar<size_t>();
    throw_if_foreign_element_size(cursor->read_scalar<size_t>(), element_size);
    // Every list carries at least its entry count, so a term count the rest of
    // the file could not hold is rejected before it is allocated.
    if (n_term > cursor->remaining() / sizeof(size_t)) {
        throw std::runtime_error("mmap: implausible term count in index file");
    }

    auto lists = std::make_unique<ArrayInvertedLists>(n_term, element_size);
    for (auto& list : *lists) {
        list.mmap_deserialize(cursor);
    }
    return lists;
}

}  // namespace nsparse
