/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef NSPARSE_GPU_COMMON_CUH
#define NSPARSE_GPU_COMMON_CUH

#include <cusparse.h>

#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "cuda_runtime.h"
#include "nsparse/gpu/gpu_diagnostics.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {

inline void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error in " << what << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

inline void check_cusparse(cusparseStatus_t status, const char* what) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuSPARSE error in " << what << ": "
            << cusparseGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

#define NSPARSE_CUDA_CHECK(expr) check_cuda((expr), #expr)
#define NSPARSE_CUSPARSE_CHECK(expr) check_cusparse((expr), #expr)

// Grow *ptr only when it must; capacity tracked in bytes.
template <class T>
void ensure_capacity(T** ptr, size_t& cap_bytes, size_t need_bytes) {
    if (cap_bytes < need_bytes) {
        cudaFree(*ptr);
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, need_bytes), "cudaMalloc(ensure_capacity)");
        *ptr = static_cast<T*>(p);
        cap_bytes = need_bytes;
    }
}

// The full corpus uploaded to the GPU once and shared read-only by the
// assigner and the summarizer (a single ~corpus-sized allocation, never
// duplicated). Indices are widened to int32 to feed both the gather kernel and
// cuSPARSE's 32-bit CSR indices.
struct DeviceCorpus {
    int32_t* indptr = nullptr;   // n_vectors + 1
    int32_t* indices = nullptr;  // nnz
    // Raw code bytes at `element_size` width: float (U32) for the unquantized
    // indexes, int8 (U8) for 8-bit scalar-quantized. Consumers cast per width.
    void* values = nullptr;      // nnz * element_size bytes
    size_t element_size = 0;
    int64_t nnz = 0;
    size_t dim = 0;
    size_t n_vectors = 0;
    // Residency identity. A pointer alone is unsafe (a freed SparseVectors
    // address can be reused), so also compare the shape (n_vectors, nnz, dim)
    // and the code width (element_size). Width is part of the key because path
    // selection and byte interpretation now depend on element_size: a
    // same-address corpus of a different width (float vs 8-bit) with matching
    // counts must not be reused as-is.
    const SparseVectors* owner = nullptr;

    bool matches(const SparseVectors* v, size_t nv, int64_t nz, size_t es,
                 size_t d) const {
        return owner == v && n_vectors == nv && nnz == nz &&
               element_size == es && dim == d;
    }
};

// Process-wide resident corpus, shared by all GPU build helpers.
class GpuCorpus {
public:
    static GpuCorpus& instance() {
        static GpuCorpus c;
        return c;
    }

    // Upload the corpus on first use (or when it changes) and return it.
    // Thread-safe: the first caller uploads while the rest wait; later calls
    // are a cheap identity check.
    const DeviceCorpus& ensure_resident(const SparseVectors* vectors) {
        const size_t n_vectors = vectors->num_vectors();
        const offset_t* indptr = vectors->indptr_data();
        const term_t* indices = vectors->indices_data();
        const uint8_t* values = vectors->values_data();
        const size_t element_size = vectors->get_element_size();
        const size_t dim = vectors->get_dimension();
        const int64_t nnz = indptr[n_vectors];

        // The device stores CSR offsets as int32 (and cuSPARSE's CSR API is
        // 32-bit), so this path cannot represent nnz > INT32_MAX; rebuild such a
        // corpus with the CPU path.
        if (nnz > INT32_MAX) {
            throw std::runtime_error(
                "GPU build path requires nnz <= INT32_MAX; use the CPU path");
        }
        // Only float (U32) and 8-bit (U8) codes have GPU kernels; other widths
        // (e.g. 16-bit) must use the CPU path. The build-time callers guard on
        // element_size, but the invariant is re-checked here so a direct caller
        // cannot upload a width the kernels would misread.
        if (element_size != U32 && element_size != U8) {
            throw std::runtime_error(
                "GPU build path supports only float (U32) or 8-bit (U8) codes");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (corpus_.matches(vectors, n_vectors, nnz, element_size, dim)) {
            return corpus_;
        }
        free_locked();

        std::vector<int32_t> indptr32(n_vectors + 1);
        for (size_t i = 0; i <= n_vectors; ++i) {
            indptr32[i] = static_cast<int32_t>(indptr[i]);
        }
        std::vector<int32_t> indices32(static_cast<size_t>(nnz));
        for (int64_t i = 0; i < nnz; ++i) {
            indices32[i] = static_cast<int32_t>(indices[i]);
        }
        const size_t values_bytes = static_cast<size_t>(nnz) * element_size;
        check_cuda(cudaMalloc(&corpus_.indptr, (n_vectors + 1) * sizeof(int32_t)),
                   "cudaMalloc(corpus.indptr)");
        check_cuda(cudaMalloc(&corpus_.indices, nnz * sizeof(int32_t)),
                   "cudaMalloc(corpus.indices)");
        check_cuda(cudaMalloc(&corpus_.values, values_bytes),
                   "cudaMalloc(corpus.values)");
        check_cuda(cudaMemcpy(corpus_.indptr, indptr32.data(),
                              (n_vectors + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(corpus.indptr)");
        check_cuda(cudaMemcpy(corpus_.indices, indices32.data(),
                              nnz * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy(corpus.indices)");
        check_cuda(cudaMemcpy(corpus_.values, values, values_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(corpus.values)");
        corpus_.element_size = element_size;
        corpus_.nnz = nnz;
        corpus_.dim = dim;
        corpus_.n_vectors = n_vectors;
        corpus_.owner = vectors;
        return corpus_;
    }

    GpuCorpus(const GpuCorpus&) = delete;
    GpuCorpus& operator=(const GpuCorpus&) = delete;

private:
    GpuCorpus() = default;
    ~GpuCorpus() { free_locked(); }

    void free_locked() {
        // Drain kernels still reading the corpus before freeing it. Harmless in
        // the current single-corpus flow (no swap after first upload); guards a
        // future caller that switches corpora while another thread's kernels are
        // still queued against the old one.
        cudaDeviceSynchronize();
        cudaFree(corpus_.indptr);
        cudaFree(corpus_.indices);
        cudaFree(corpus_.values);
        corpus_ = DeviceCorpus{};
    }

    std::mutex mutex_;
    DeviceCorpus corpus_;
};

}  // namespace nsparse::detail

#endif  // NSPARSE_GPU_COMMON_CUH
