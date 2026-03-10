/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef RANKER_H
#define RANKER_H

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

namespace nsparse::detail {
template <class T>
struct TopKItem {
    float score;
    T id;
    bool operator<(const TopKItem& other) const { return score < other.score; }
    bool operator>(const TopKItem& other) const { return score > other.score; }
};

template <typename T, typename Comparator = std::greater<float>>
class TopKHolder {
    using Item = TopKItem<T>;
    int k;
    struct CompareItem {
        Comparator comp;
        bool operator()(const Item& item1, const Item& item2) const {
            return comp(item1.score, item2.score);
        }
    };

    std::priority_queue<Item, std::vector<Item>, CompareItem> pq;
    float threshold_ = std::numeric_limits<float>::infinity();

public:
    TopKHolder(int k) : k(k) {
        throw_if_not_positive(k);
        std::vector<Item> vec;
        vec.reserve(k);
        pq = std::priority_queue<Item, std::vector<Item>, CompareItem>(
            CompareItem(), std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void add(const float score, const T& item) {
        if (pq.size() >= k && score <= threshold_) {
            return;  // Fast reject without touching heap
        }
        if (pq.size() < k) {
            pq.push({score, item});
            if (pq.size() == k) {
                threshold_ = pq.top().score;
            }
        } else {
            pq.pop();
            pq.push({score, item});
            threshold_ = pq.top().score;
        }
    }
    /**
     * get k data from pq in value ascending order, this is a disruptive
     * operation
     */
    std::vector<T> top_k() {
        if (k <= 0) {
            return {};
        }
        std::vector<T> ret(k);
        int idx = 0;
        while (!pq.empty() && idx < k) {
            ret[idx] = pq.top().id;
            pq.pop();
            ++idx;
        }
        return ret;
    }

    /**
     *  get data from pq in value descending order (highest scores first),
     *  this is a disruptive operation. Always returns exactly k elements,
     *  padding with default-constructed T if pq has fewer than k items.
     */
    std::vector<T> top_k_descending() {
        size_t size = pq.size();
        if (k <= 0 || size <= 0) {
            return {};
        }
        std::vector<T> ret(size);
        int idx = static_cast<int>(size) - 1;
        while (!pq.empty() && idx >= 0) {
            ret[idx--] = pq.top().id;
            pq.pop();
        }
        return ret;
    }

    pair_of_score_id_vector_t_t<T> top_k_items_descending() {
        size_t size = pq.size();
        if (k <= 0 || size <= 0) {
            return {};
        }
        std::vector<float> scores(size);
        std::vector<T> ids(size);
        for (int i = size - 1; i >= 0; --i) {
            auto top = pq.top();
            scores[i] = top.score;
            ids[i] = top.id;
            pq.pop();
        }
        return {scores, ids};
    }

    std::vector<T> top_k_descending_with_padding(T pad_with) {
        std::vector<T> ret = top_k_descending();
        ret.resize(k, pad_with);
        return ret;
    }

    [[nodiscard]] bool full() { return pq.size() == k; }
    [[nodiscard]] bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float peek_score() { return pq.top().score; }
};

template <typename T, typename ID_T = size_t>
struct DedupeTopKItem {
    float score;
    ID_T dedupe_id;
    T id;
};

template <typename T, typename ID_T = size_t,
          typename Comparator = std::greater<float>>
class DedupeTopKHolder {
    using Item = DedupeTopKItem<T, ID_T>;

private:
    int k;
    struct CompareItem {
        Comparator comp;
        bool operator()(const Item& item1, const Item& item2) const {
            return comp(item1.score, item2.score);
        }
    };

    std::priority_queue<Item, std::vector<Item>, CompareItem> pq;
    absl::flat_hash_set<ID_T> dedupe;

public:
    DedupeTopKHolder(int k) : k(k) {
        dedupe.reserve(k);
        std::vector<Item> vec;
        vec.reserve(k);
        pq = std::priority_queue<Item, std::vector<Item>, CompareItem>(
            CompareItem(), std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void add(const float score, ID_T id, const T& item) {
        if (pq.size() >= k && score <= pq.top().score) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.push({score, id, item});
            dedupe.insert(id);
        } else if (pq.top().score < score) {
            auto top = pq.top();
            dedupe.erase(top.dedupe_id);
            pq.pop();
            pq.push({score, id, item});
            dedupe.insert(id);
        }
    }

    void add(const float score, ID_T id) {
        if (pq.size() >= k && score <= pq.top().score) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.push({score, id, static_cast<T>(id)});
            dedupe.insert(id);
        } else if (pq.top().score < score) {
            auto top = pq.top();
            dedupe.erase(top.dedupe_id);
            pq.pop();
            pq.push({score, id, static_cast<T>(id)});
            dedupe.insert(id);
        }
    }

    [[nodiscard]] bool full() { return pq.size() == k; }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> top_k() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().id);
            pq.pop();
        }
        return ret;
    }

    /**
     *  get data from pq in descending order (highest scores first),
     *  this is a disruptive operation. Always returns exactly k elements,
     *  padding with default-constructed T if pq has fewer than k items.
     */
    std::vector<T> top_k_descending() {
        size_t size = pq.size();
        if (k <= 0 || size <= 0) {
            return {};
        }
        std::vector<T> ret(size);
        int idx = static_cast<int>(size) - 1;
        while (!pq.empty() && idx >= 0) {
            ret[idx--] = pq.top().id;
            pq.pop();
        }
        return ret;
    }

    std::vector<T> top_k_descending_with_padding(T pad_with) {
        std::vector<T> ret = top_k_descending();
        ret.resize(k, pad_with);
        return ret;
    }

    std::pair<std::vector<T>, std::vector<float>>
    top_k_descending_with_scores_and_padding(T pad_with, float score_pad) {
        size_t size = pq.size();
        std::vector<T> ids(k, pad_with);
        std::vector<float> scores(k, score_pad);
        if (k <= 0 || size <= 0) {
            return {ids, scores};
        }
        int idx = static_cast<int>(size) - 1;
        while (!pq.empty() && idx >= 0) {
            ids[idx] = pq.top().id;
            scores[idx] = pq.top().score;
            pq.pop();
            --idx;
        }
        return {ids, scores};
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float peek_score() { return pq.top().score; }
};

}  // namespace nsparse::detail
#endif