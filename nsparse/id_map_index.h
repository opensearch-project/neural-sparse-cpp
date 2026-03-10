/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef ID_MAP_INDEX_H
#define ID_MAP_INDEX_H
#include <algorithm>
#include <array>
#include <ranges>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {
namespace detail {
class IDSelectorWithIDMap : public IDSelector {
public:
    IDSelectorWithIDMap(const IDSelector* id_selector,
                        const std::vector<idx_t>& id_maps)
        : delegate_(id_selector), id_map_(id_maps) {}

    bool is_member(idx_t id) const override {
        return delegate_->is_member(id_map_[id]);
    }

private:
    const IDSelector* delegate_;
    const std::vector<idx_t>& id_map_;
};

class IDSelectorEnumerableWithIDMap : public IDSelectorEnumerable {
public:
    IDSelectorEnumerableWithIDMap(
        const IDSelectorEnumerable* id_selector,
        const std::vector<idx_t>& id_maps,
        const absl::flat_hash_map<idx_t, idx_t>& external_id_map)
        : delegate_(id_selector),
          internal_id_map_(id_maps),
          external_id_map_(external_id_map) {}

    bool is_member(idx_t id) const override {
        return delegate_->is_member(internal_id_map_[id]);
    }

    std::vector<idx_t> ids() const override {
        auto vec = delegate_->ids();
        std::vector<idx_t> result;
        result.reserve(vec.size());
        for (const auto& id : vec) {
            auto it = external_id_map_.find(id);
            if (it != external_id_map_.end()) {
                result.push_back(it->second);
            }
        }
        return result;
    }

    size_t size() const override { return delegate_->size(); }

private:
    const IDSelectorEnumerable* delegate_;
    const std::vector<idx_t>& internal_id_map_;
    const absl::flat_hash_map<idx_t, idx_t>& external_id_map_;
};
}  // namespace detail

class IDMapIndex : public Index, public IndexIO {
public:
    IDMapIndex() = default;
    static constexpr std::array<char, 4> name = {'I', 'D', 'M', 'P'};
    explicit IDMapIndex(Index*);
    std::array<char, 4> id() const override { return name; }

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    void search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k, float* distances, idx_t* labels,
                SearchParameters* search_parameters = nullptr) override;
    const SparseVectors* get_vectors() const override;

    void add_with_ids(idx_t n, const idx_t* indptr, const term_t* indices,
                      const float* values, const idx_t* ids) override;
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

private:
    Index* delegate_ = nullptr;
    std::vector<idx_t> internal_to_external_;
    absl::flat_hash_map<idx_t, idx_t> external_to_internal_;
};
}  // namespace nsparse

#endif  // ID_MAP_INDEX_H