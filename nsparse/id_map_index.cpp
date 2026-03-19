/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/id_map_index.h"

#include <memory>

#include "nsparse/id_selector.h"
#include "nsparse/io/index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/utils/checks.h"

namespace nsparse {
IDMapIndex::IDMapIndex(Index* index) : delegate_(index) {}

void IDMapIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                     const float* values) {
    delegate_->add(n, indptr, indices, values);
}

void IDMapIndex::build() { delegate_->build(); }

void IDMapIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k, float* distances,
                        idx_t* labels, SearchParameters* search_parameters) {
    std::unique_ptr<IDSelector> id_selector_idmap = nullptr;
    if (search_parameters != nullptr) {
        const auto* id_selector = search_parameters->get_id_selector();
        if (id_selector != nullptr) {
            const auto* id_selector_enumerable =
                dynamic_cast<const IDSelectorEnumerable*>(id_selector);
            if (id_selector_enumerable != nullptr) {
                id_selector_idmap =
                    std::make_unique<detail::IDSelectorEnumerableWithIDMap>(
                        id_selector_enumerable, internal_to_external_,
                        external_to_internal_);
            } else {
                id_selector_idmap =
                    std::make_unique<detail::IDSelectorWithIDMap>(
                        id_selector, internal_to_external_);
            }
            search_parameters->set_id_selector(id_selector_idmap.get());
        }
    }
    delegate_->search(n, indptr, indices, values, k, distances, labels,
                      search_parameters);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            auto& result_id = labels[i * k + j];
            result_id =
                result_id < 0 ? result_id : internal_to_external_[result_id];
        }
    }
}

const SparseVectors* IDMapIndex::get_vectors() const {
    return delegate_ == nullptr ? nullptr : delegate_->get_vectors();
}

void IDMapIndex::add_with_ids(idx_t n, const idx_t* indptr,
                              const term_t* indices, const float* values,
                              const idx_t* ids) {
    size_t old_size = delegate_->num_vectors();
    delegate_->add(n, indptr, indices, values);
    internal_to_external_.resize(old_size + n);
    for (int i = 0; i < n; ++i) {
        internal_to_external_[old_size + i] = ids[i];
        external_to_internal_[ids[i]] = old_size + i;
    }
}
void IDMapIndex::write_index(IOWriter* io_writer) {
    nsparse::detail::write_index(delegate_, io_writer, true);

    // Write internal_to_external_ vector
    size_t map_size = internal_to_external_.size();
    io_writer->write(&map_size, sizeof(size_t), 1);
    if (map_size > 0) {
        io_writer->write(internal_to_external_.data(), sizeof(idx_t), map_size);
    }
}

void IDMapIndex::read_index(IOReader* io_reader) {
    delegate_ = nsparse::detail::read_index(io_reader, true);

    // Read internal_to_external_ vector
    size_t map_size = 0;
    io_reader->read(&map_size, sizeof(size_t), 1);
    internal_to_external_.resize(map_size);
    if (map_size > 0) {
        io_reader->read(internal_to_external_.data(), sizeof(idx_t), map_size);
    }

    // Rebuild external_to_internal_ from internal_to_external_
    external_to_internal_.clear();
    external_to_internal_.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i) {
        external_to_internal_[internal_to_external_[i]] = static_cast<idx_t>(i);
    }
}
}  // namespace nsparse