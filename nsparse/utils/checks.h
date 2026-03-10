/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdexcept>

namespace nsparse {

template <typename T>
T* throw_if_null(T* ptr, const char* msg = "unexpected nullptr") {
    if (ptr == nullptr) {
        throw std::invalid_argument(msg);
    }
    return ptr;
}

template <typename T>
T throw_if_not_positive(T value, const char* msg = "value must be positive") {
    if (value <= 0) {
        throw std::invalid_argument(msg);
    }
    return value;
}

template <typename... Args>
void throw_if_any_null(const char* msg, Args*... ptrs) {
    bool any_null = ((ptrs == nullptr) || ...);
    if (any_null) {
        throw std::invalid_argument(msg);
    }
}

template <typename... Args>
void throw_if_any_null(Args*... ptrs) {
    throw_if_any_null("unexpected nullptr", ptrs...);
}

[[noreturn]] inline void throw_not_implemented(
    const char* msg = "not implemented") {
    throw std::runtime_error(msg);
}

template <typename T, typename U>
[[noreturn]] void throw_if_not_equal(T&& t, U&& u,
                                     const char* msg = "values must be equal") {
    if (t != u) {
        throw std::invalid_argument(msg);
    }
}
}  // namespace nsparse
#endif  // COMMON_H