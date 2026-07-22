/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef NSPARSE_VISITED_SET_H
#define NSPARSE_VISITED_SET_H

#include <cstdint>
#include <vector>

namespace nsparse::detail {

// Membership set over a fixed doc-id domain [0, n), backed by a bit per doc.
//
// The seismic doc loop tests/inserts every candidate doc to dedupe across
// clusters. A hash set hashes and probes a random cache line per candidate and
// must be cleared each query. A full generation-stamped uint32 array (4 B/doc)
// avoids the hashing but at seismic scale (~35 MB) adds a random-access stream
// larger than L2, so it misses cache on every candidate. A bitset is 32x
// smaller (1 bit/doc, ~1.1 MB at 8.8M docs) so far fewer distinct cache lines
// are touched, while the touched-word list lets a new query clear only the
// words it actually dirtied (sparse O(visited) reset, not O(n)).
class VisitedSet {
public:
    VisitedSet() = default;
    explicit VisitedSet(size_t n) { resize(n); }

    void resize(size_t n) {
        bits_.assign((n + 63) / 64, 0);
        touched_.clear();
    }

    // Begin a new query: clear only the words dirtied by the previous query.
    void new_query() {
        for (const size_t w : touched_) {
            bits_[w] = 0;
        }
        touched_.clear();
    }

    // Mark `id` visited; return true if it was newly inserted this query.
    bool insert(size_t id) {
        const size_t w = id >> 6;
        const uint64_t mask = uint64_t{1} << (id & 63);
        const uint64_t prev = bits_[w];
        if (prev & mask) {
            return false;
        }
        if (prev == 0) {
            touched_.push_back(w);
        }
        bits_[w] = prev | mask;
        return true;
    }

private:
    std::vector<uint64_t> bits_;
    std::vector<size_t> touched_;
};

}  // namespace nsparse::detail

#endif  // NSPARSE_VISITED_SET_H
