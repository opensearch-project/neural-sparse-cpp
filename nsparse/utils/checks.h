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

#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace nsparse {

template <typename T>
T* throw_if_null(T* ptr, const char* msg = "unexpected nullptr") {
    if (ptr == nullptr) {
        throw std::invalid_argument(msg);
    }
    return ptr;
}

// Rejects a file path that is null or does not exist, so a caller gets a clear
// error before trying to open it. `what` names the file in the message.
inline void check_if_file_valid(const char* path, const char* what = "file") {
    throw_if_null(path, "file path must not be null");
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument(std::string(what) +
                                    " does not exist: " + path);
    }
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

template <typename... Args>
void throw_if_any_non_null(const char* msg, Args*... ptrs) {
    bool any_non_null = ((ptrs != nullptr) || ...);
    if (any_non_null) {
        throw std::invalid_argument(msg);
    }
}

template <typename... Args>
void throw_if_any_non_null(Args*... ptrs) {
    throw_if_any_non_null("pointer cannot be reassigned", ptrs...);
}

// Rejects a product that would wrap: counts and widths read from an index file
// are multiplied to size an array, and a wrapped size passes bounds checks.
inline size_t checked_mul(size_t lhs, size_t rhs,
                          const char* msg = "size overflow in index file") {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        throw std::runtime_error(msg);
    }
    return lhs * rhs;
}

}  // namespace nsparse
#endif  // COMMON_H