/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef ID_SELECTOR_H
#define ID_SELECTOR_H

#include <cstddef>
#include <ranges>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/types.h"

namespace nsparse {

class IDSelector {
public:
    virtual ~IDSelector() = default;
    virtual bool is_member(idx_t id) const = 0;
    bool operator()(idx_t id) const { return is_member(id); }
};

class IDSelectorEnumerable : public IDSelector {
public:
    ~IDSelectorEnumerable() = default;
    virtual std::vector<idx_t> ids() const = 0;
    virtual std::vector<idx_t> ordered_ids() const {
        auto vec = ids();
        std::ranges::sort(vec.begin(), vec.end());
        return vec;
    }
    virtual size_t size() const = 0;
};

class SetIDSelector : public IDSelectorEnumerable {
public:
    explicit SetIDSelector(size_t n, const idx_t* indices) {
        ids_ = absl::flat_hash_set<idx_t>(indices, indices + n);
    }

    bool is_member(idx_t id) const override { return ids_.contains(id); }

    std::vector<idx_t> ids() const override {
        return std::vector<idx_t>(ids_.begin(), ids_.end());
    }

    size_t size() const override { return ids_.size(); }

private:
    absl::flat_hash_set<idx_t> ids_;
};

class ArrayIDSelector : public IDSelectorEnumerable {
public:
    explicit ArrayIDSelector(size_t n, const idx_t* indices)
        : ids_(indices, indices + n) {}

    bool is_member(idx_t id) const override {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    std::vector<idx_t> ids() const override {
        return std::vector<idx_t>(ids_.begin(), ids_.end());
    }
    size_t size() const override { return ids_.size(); }

private:
    std::vector<idx_t> ids_;
};

class NotIDSelector : public IDSelector {
public:
    explicit NotIDSelector(IDSelector* selector) : delegate_(selector) {}
    bool is_member(idx_t id) const override {
        return !delegate_->is_member(id);
    }

private:
    IDSelector* delegate_;
};
}  // namespace nsparse

#endif  // ID_SELECTOR_H