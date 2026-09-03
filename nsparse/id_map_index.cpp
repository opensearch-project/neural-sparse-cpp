/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/id_map_index.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

size_t IDMapIndex::num_vectors() const {
    return delegate_ == nullptr ? 0 : delegate_->num_vectors();
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

std::vector<idx_t> IDMapIndex::read_id_file(const char* id_path) {
    check_if_file_valid(id_path, "id map file");

    std::ifstream in(id_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("cannot open id map file: ") +
                                 id_path);
    }

    int64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
        throw std::runtime_error(std::string("truncated id map file: ") +
                                 id_path);
    }
    if (count < 0) {
        throw std::invalid_argument(std::string("negative id map count in: ") +
                                    id_path);
    }
    const auto map_size = static_cast<size_t>(count);

    // Guard the byte-size arithmetic against wraparound before relying on it.
    if (map_size > (std::numeric_limits<size_t>::max() - sizeof(int64_t)) /
                       sizeof(idx_t)) {
        throw std::invalid_argument(std::string("id map count is too large: ") +
                                    id_path);
    }

    // Reject a truncated or oversized file up front.
    const size_t expected_bytes = sizeof(int64_t) + map_size * sizeof(idx_t);
    if (std::filesystem::file_size(id_path) != expected_bytes) {
        throw std::invalid_argument(
            std::string(
                "id map file size does not match its count (expected ") +
            std::to_string(expected_bytes) + " bytes): " + id_path);
    }

    std::vector<idx_t> internal_to_external(map_size);
    if (map_size > 0) {
        in.read(reinterpret_cast<char*>(internal_to_external.data()),
                static_cast<std::streamsize>(map_size * sizeof(idx_t)));
        if (!in) {
            throw std::runtime_error(std::string("truncated id map file: ") +
                                     id_path);
        }
    }
    return internal_to_external;
}

void IDMapIndex::read_csr_and_ids(const char* csr_path, const char* id_path,
                                  Residency residency) {
    if (delegate_ == nullptr) {
        throw std::logic_error("IDMapIndex has no delegate index");
    }
    check_if_file_valid(csr_path, "csr file");

    // Fully validate and load the id file BEFORE ingesting the CSR, so a
    // missing/malformed/truncated id file leaves this index untouched. Only the
    // count-vs-CSR-row check below can fail once the delegate has ingested; on
    // ANY throw from this method the half-built index must be discarded.
    std::vector<idx_t> internal_to_external = read_id_file(id_path);

    // The id file is known-good; now ingest the vectors (borrowed from the
    // mapping when residency == kMmap). Afterward num_vectors() reflects the
    // CSR rows.
    delegate_->read_csr(csr_path, residency);

    // The map is row-aligned with the CSR, so its count must equal the vectors
    // the delegate just ingested.
    const size_t delegate_size = delegate_->num_vectors();
    if (internal_to_external.size() != delegate_size) {
        throw std::invalid_argument(
            "id map count (" + std::to_string(internal_to_external.size()) +
            ") does not match the CSR vector count (" +
            std::to_string(delegate_size) + "): " + id_path);
    }

    set_id_map(std::move(internal_to_external));
}

void IDMapIndex::set_id_map(std::vector<idx_t>&& internal_to_external) {
    internal_to_external_ = std::move(internal_to_external);
    external_to_internal_.clear();
    external_to_internal_.reserve(internal_to_external_.size());
    for (size_t i = 0; i < internal_to_external_.size(); ++i) {
        external_to_internal_[internal_to_external_[i]] = static_cast<idx_t>(i);
    }
}

void IDMapIndex::write_index(IOWriter* io_writer) {
    // Write internal_to_external_ vector
    size_t map_size = internal_to_external_.size();
    io_writer->write(&map_size, sizeof(size_t), 1);
    if (map_size > 0) {
        io_writer->write(internal_to_external_.data(), sizeof(idx_t), map_size);
    }
    // delegate should be written at the end
    nsparse::detail::write_index(delegate_.get(), io_writer, true);
}

void IDMapIndex::read_index(IOReader* io_reader, const IndexHeader& /*header*/,
                            int io_flags) {
    // Read the id map into a local vector, then load the delegate.
    size_t map_size = 0;
    io_reader->read(&map_size, sizeof(size_t), 1);
    std::vector<idx_t> internal_to_external(map_size);
    if (map_size > 0) {
        io_reader->read(internal_to_external.data(), sizeof(idx_t), map_size);
    }

    delegate_.reset(nsparse::detail::read_index(io_reader, true, io_flags));

    set_id_map(std::move(internal_to_external));
}
}  // namespace nsparse