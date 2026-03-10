/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DENSE_VECTOR_MATRIX_H
#define DENSE_VECTOR_MATRIX_H

#include <cstddef>
#include <cstdlib>

namespace nsparse::detail {

#if defined(__AVX512F__)
static constexpr int MATRIX_ALIGNMENT = 64;  // AVX512
#else
static constexpr int MATRIX_ALIGNMENT = 16;
#endif

template <typename T>
class DenseVectorMatrixT {
public:
    DenseVectorMatrixT(const DenseVectorMatrixT&) = delete;
    DenseVectorMatrixT& operator=(const DenseVectorMatrixT&) = delete;
    DenseVectorMatrixT(DenseVectorMatrixT&&) = delete;

    DenseVectorMatrixT(size_t row, size_t dimension)
        : rows_(row), dimension_(dimension) {
        data_ = static_cast<T*>(
            std::aligned_alloc(MATRIX_ALIGNMENT, row * dimension * sizeof(T)));
    }

    ~DenseVectorMatrixT() { std::free(data_); }

    T get(size_t row, size_t col) const {
        return data_[row * dimension_ + col];
    }

    void set(size_t row, size_t col, T value) {
        data_[row * dimension_ + col] = value;
    }

    T* data() const { return data_; }
    size_t get_rows() const { return rows_; }
    size_t get_dimension() const { return dimension_; }

private:
    T* data_;
    size_t rows_;
    size_t dimension_;
};

// Alias for backward compatibility
using DenseVectorMatrix = DenseVectorMatrixT<float>;

}  // namespace nsparse::detail

#endif  // DENSE_VECTOR_MATRIX_H