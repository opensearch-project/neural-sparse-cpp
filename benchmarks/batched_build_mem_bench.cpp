/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// Standalone driver measuring PEAK RESIDENT MEMORY (VmHWM) and wall time of a
// Seismic index build, for one configuration per process invocation. google-
// benchmark measures throughput, not peak RSS, and running each config in its
// own process gives a clean high-water mark uncontaminated by earlier builds.
//
// Usage:
//   batched_build_mem_bench convert <interchange_csr> <native_csr>
//   batched_build_mem_bench baseline <interchange_csr> <lambda> <beta> \
//                           <alpha> [out_index]
//   batched_build_mem_bench batched <inmem|mmap> <csr> <lambda> <beta> \
//                           <alpha> <num_batches> <out_dir>
//
// "convert" produces the native CSR the mapped read wants. "baseline" builds
// the in-memory SeismicIndex (streaming add of the interchange CSR, like the
// other benchmarks) -- the memory this feature exists to avoid. "batched" runs
// the term-batched build at the given batch count.
//
// Compare "batched inmem" against "baseline": both hold the corpus on the heap
// via the same streaming_add, so the difference between them is the batching
// and nothing else. "batched mmap" borrows a native CSR instead, which is
// cheaper by the whole size of the corpus -- a real saving, but one that comes
// from the residency rather than from batching, so it is not the baseline's
// counterpart.
//
// Point <out_dir> at a real disk: on a tmpfs such as /tmp the index is RAM, and
// the numbers are meaningless.

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"

namespace {

// One "Field: N kB" line from /proc/self/status, in KiB, or -1.
//
// Reads into a stack buffer rather than through iostreams because the sampler
// below calls this thousands of times while the thing being measured is memory:
// a per-sample allocation would show up in the number it is reporting.
long read_status_kib(const char* field) {
    const int fd = ::open("/proc/self/status", O_RDONLY);  // NOLINT
    if (fd < 0) {
        return -1;
    }
    std::array<char, 4096> buffer{};
    const ssize_t got = ::read(fd, buffer.data(), buffer.size() - 1);
    ::close(fd);
    if (got <= 0) {
        return -1;
    }
    buffer[static_cast<size_t>(got)] = '\0';
    const char* at = std::strstr(buffer.data(), field);
    if (at == nullptr) {
        return -1;
    }
    long kib = -1;
    std::sscanf(at + std::strlen(field), " %ld", &kib);
    return kib;
}

long read_vm_hwm_kib() { return read_status_kib("VmHWM:"); }

// Tracks the high-water mark of RssAnon and RssFile over its own lifetime.
//
// VmHWM is a kernel counter, so peak RSS needs no help. RssAnon and RssFile are
// reported only as current values, and there is no peak equivalent, so the only
// way to get their maxima is to sample. That makes them lower bounds: a spike
// shorter than the interval is missed. The interval is small next to a build
// that runs for minutes and allocates in per-window steps, so in practice they
// track the real peaks, but they are not the guarantee VmHWM is.
//
// The split is worth the trouble because it separates what the build allocates
// (RssAnon -- the inverted lists and clusters, which is what batching bounds)
// from what it merely touches (RssFile -- a mapped corpus, which is the
// kernel's to reclaim under pressure).
class PeakRssSampler {
public:
    explicit PeakRssSampler(
        std::chrono::milliseconds interval = std::chrono::milliseconds(50))
        : thread_([this, interval] {
              while (!stop_.load(std::memory_order_relaxed)) {
                  sample();
                  std::this_thread::sleep_for(interval);
              }
              // Once more after the stop, so the final state is never missed.
              sample();
          }) {}

    ~PeakRssSampler() {
        stop_.store(true, std::memory_order_relaxed);
        thread_.join();
    }

    PeakRssSampler(const PeakRssSampler&) = delete;
    PeakRssSampler& operator=(const PeakRssSampler&) = delete;

    [[nodiscard]] long peak_anon_kib() const {
        return peak_anon_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] long peak_file_kib() const {
        return peak_file_.load(std::memory_order_relaxed);
    }

private:
    void sample() {
        keep_max(&peak_anon_, read_status_kib("RssAnon:"));
        keep_max(&peak_file_, read_status_kib("RssFile:"));
    }

    static void keep_max(std::atomic<long>* peak, long value) {
        long seen = peak->load(std::memory_order_relaxed);
        while (value > seen && !peak->compare_exchange_weak(
                                   seen, value, std::memory_order_relaxed)) {
        }
    }

    std::atomic<bool> stop_{false};
    std::atomic<long> peak_anon_{-1};
    std::atomic<long> peak_file_{-1};
    std::thread thread_;
};

// Resets VmHWM to the current VmRSS, so a later read reports the peak since
// this call rather than since the process started.
//
// Necessary because loading the corpus costs more than holding it:
// streaming_add's staging buffers are a copy of the whole thing on top of the
// index's own. Without this the loader's high-water mark floors every reported
// number and a build whose true peak is below it is unmeasurable. The corpus
// stays resident across the reset, so it still counts toward the build's peak
// -- which is what should be compared.
void reset_vm_hwm() {
    std::ofstream clear_refs("/proc/self/clear_refs");
    // 5 == CLEAR_REFS_MM_HIWATER_RSS. Linux-only, and only an accounting hint:
    // if it is unavailable the numbers include the loader, so say so rather
    // than reporting them as if they did not.
    if (!clear_refs || !(clear_refs << "5\n")) {
        std::cerr << "warning: cannot reset VmHWM; peak_rss_mb includes corpus "
                     "loading\n";
        return;
    }
}

double now_seconds() {
    // CLOCK_MONOTONIC via clock() is process time; use wall clock instead.
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) +
           static_cast<double>(ts.tv_nsec) * 1e-9;
}

// Streaming add of an interchange CSR (int64 sizes[3], int64 indptr, int32
// indices, float values) into an index, in nnz-bounded batches (mirrors
// index_search_benchmark's streaming_add so the baseline peak is comparable).
void streaming_add(nsparse::Index* index, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }
    int64_t sizes[3];
    file.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    const int64_t nrow = sizes[0];
    const int64_t nnz = sizes[2];

    std::vector<int64_t> indptr64(nrow + 1);
    file.read(reinterpret_cast<char*>(indptr64.data()),
              static_cast<std::streamsize>((nrow + 1) * sizeof(int64_t)));
    const auto indices_off = file.tellg();
    const auto data_off = static_cast<std::streamoff>(indices_off) +
                          static_cast<std::streamoff>(nnz * sizeof(int32_t));

    constexpr int64_t kMaxBatchNnz = 1'500'000'000LL;
    int64_t row_start = 0;
    while (row_start < nrow) {
        int64_t row_end = row_start + 1;
        while (row_end < nrow &&
               (indptr64[row_end] - indptr64[row_start]) < kMaxBatchNnz) {
            ++row_end;
        }
        const int64_t brows = row_end - row_start;
        const int64_t bnnz = indptr64[row_end] - indptr64[row_start];
        const int64_t boff = indptr64[row_start];

        std::vector<nsparse::idx_t> bindptr(brows + 1);
        for (int64_t i = 0; i <= brows; ++i) {
            bindptr[i] =
                static_cast<nsparse::idx_t>(indptr64[row_start + i] - boff);
        }
        std::vector<nsparse::term_t> bindices(bnnz);
        {
            file.seekg(indices_off +
                       static_cast<std::streamoff>(boff * sizeof(int32_t)));
            constexpr int64_t kChunk = 1 << 22;
            std::vector<int32_t> tmp(std::min(kChunk, bnnz > 0 ? bnnz : 1));
            int64_t done = 0;
            while (done < bnnz) {
                int64_t take = std::min(kChunk, bnnz - done);
                file.read(reinterpret_cast<char*>(tmp.data()),
                          static_cast<std::streamsize>(take * sizeof(int32_t)));
                for (int64_t j = 0; j < take; ++j) {
                    bindices[done + j] = static_cast<nsparse::term_t>(tmp[j]);
                }
                done += take;
            }
        }
        std::vector<float> bdata(bnnz);
        {
            file.seekg(static_cast<std::streamoff>(data_off) +
                       static_cast<std::streamoff>(boff * sizeof(float)));
            file.read(reinterpret_cast<char*>(bdata.data()),
                      static_cast<std::streamsize>(bnnz * sizeof(float)));
        }
        index->add(static_cast<nsparse::idx_t>(brows), bindptr.data(),
                   bindices.data(), bdata.data());
        row_start = row_end;
    }
}

// Both layouts start with the same int64 (rows, cols, nnz) header.
int csr_dimension(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    int64_t sizes[3];
    file.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    return static_cast<int>(sizes[1]);
}

// `peak_rss_mb` is the build's own high-water mark (see reset_vm_hwm), split
// into what the build allocated (`peak_rss_anon_mb`) and what it touched of a
// mapping
// (`peak_rss_file_mb`). The anon figure is the one batching is meant to move;
// the file figure is a mapped corpus, which the kernel can reclaim.
// `load_peak_rss_mb` is what loading the corpus cost before the build, reported
// so a build peak that sits below the loader's is not mistaken for the whole
// story.
void report(const std::string& mode, const std::string& detail, double build_s,
            long load_hwm_kib, const PeakRssSampler& sampler) {
    const auto mb = [](long kib) { return static_cast<double>(kib) / 1024.0; };
    std::cout << "RESULT mode=" << mode << " " << detail
              << " build_s=" << build_s
              << " peak_rss_mb=" << mb(read_vm_hwm_kib())
              << " peak_rss_anon_mb=" << mb(sampler.peak_anon_kib())
              << " peak_rss_file_mb=" << mb(sampler.peak_file_kib())
              << " load_peak_rss_mb=" << mb(load_hwm_kib) << "\n";
}

int run_convert(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "convert <interchange_csr> <native_csr>\n";
        return 2;
    }
    nsparse::csr_layout::convert(argv[2], argv[3]);
    std::cout << "converted -> " << argv[3] << "\n";
    return 0;
}

int run_baseline(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "baseline <interchange_csr> <lambda> <beta> <alpha> "
                     "[out_index]\n";
        return 2;
    }
    const std::string csr = argv[2];
    const nsparse::SeismicClusterParameters params = {
        .lambda = std::atoi(argv[3]),
        .beta = std::atoi(argv[4]),
        .alpha = static_cast<float>(std::atof(argv[5]))};
    nsparse::SeismicIndex index(csr_dimension(csr), params);
    streaming_add(&index, csr);

    const long load_hwm = read_vm_hwm_kib();
    reset_vm_hwm();
    // Scoped to the build, so the sampled peaks exclude corpus loading exactly
    // as the reset makes VmHWM exclude it.
    PeakRssSampler sampler;
    const double started = now_seconds();
    index.build();
    const double build_s = now_seconds() - started;
    report("baseline", "batches=0", build_s, load_hwm, sampler);

    if (argc >= 7) {
        const std::string out = argv[6];
        nsparse::write_index(&index, const_cast<char*>(out.c_str()));
        std::ifstream file(out, std::ios::binary | std::ios::ate);
        std::cout << "index_bytes=" << file.tellg() << "\n";
    }
    return 0;
}

int run_batched(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "batched <inmem|mmap> <csr> <lambda> <beta> <alpha> "
                     "<num_batches> <out_dir>\n";
        return 2;
    }
    const std::string corpus_residency = argv[2];
    const std::string csr = argv[3];
    const nsparse::SeismicClusterParameters params = {
        .lambda = std::atoi(argv[4]),
        .beta = std::atoi(argv[5]),
        .alpha = static_cast<float>(std::atof(argv[6]))};
    const std::string out = std::string(argv[8]) + "/index.seismic.dat";
    nsparse::SeismicClusterParameters batched_params = params;
    batched_params.batch_clustering.batch_size =
        static_cast<size_t>(std::atoi(argv[7]));
    batched_params.batch_clustering.batch_file_output_path = out;

    // Which residency the corpus is held at is the point of the flag:
    //
    //   inmem -- streaming_add of an interchange CSR, byte for byte what the
    //            baseline above does. This is the comparison that isolates
    //            batching: same corpus, on the heap, in both arms.
    //   mmap  -- a native CSR, borrowed. Cheaper by the size of the corpus, but
    //            not comparable to the baseline, because the saving is the
    //            residency rather than the batching.
    nsparse::SeismicIndex index(csr_dimension(csr), batched_params);
    if (corpus_residency == "inmem") {
        streaming_add(&index, csr);
    } else if (corpus_residency == "mmap") {
        index.read_csr(csr.c_str(), nsparse::Residency::kMmap);
    } else {
        std::cerr << "corpus residency must be inmem or mmap\n";
        return 2;
    }

    const long load_hwm = read_vm_hwm_kib();
    reset_vm_hwm();
    PeakRssSampler sampler;
    const double started = now_seconds();
    // batch_file_output_path is set, so build() streams the index out rather
    // than retaining it -- the same call an ordinary build makes.
    index.build();
    const double build_s = now_seconds() - started;

    report("batched",
           "corpus=" + corpus_residency + " batches=" +
               std::to_string(batched_params.batch_clustering.batch_size),
           build_s, load_hwm, sampler);
    std::ifstream file(out, std::ios::binary | std::ios::ate);
    std::cout << "index_bytes=" << file.tellg() << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: batched_build_mem_bench "
                     "<convert|baseline|batched> ...\n";
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "convert") {
        return run_convert(argc, argv);
    }
    if (mode == "baseline") {
        return run_baseline(argc, argv);
    }
    if (mode == "batched") {
        return run_batched(argc, argv);
    }
    std::cerr << "unknown mode: " << mode << "\n";
    return 2;
}
