/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef VECTOR_PROCESS_H
#define VECTOR_PROCESS_H
#include <vector>

#include "nsparse/types.h"
#include "nsparse/utils/ranker.h"

namespace nsparse::detail {
template <class T>
inline std::vector<term_t> top_k_tokens(const term_t* indices, const T* weights,
                                        int size, int k) {
    if (k >= size) {
        std::vector<term_t> result(indices, indices + size);
        return result;
    }
    TopKHolder<term_t> holder(k);
    for (int i = 0; i < size; ++i) {
        holder.add(weights[i], indices[i]);
    }
    return holder.top_k_descending();
}

}  // namespace nsparse::detail

#endif  // VECTOR_PROCESS_H