/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// Latency / QPS / memory-residency / recall driver for one serialized index.
//
// google-benchmark reports a whole-batch mean only, and nothing in the existing
// harnesses can select a residency or report the RssAnon/RssFile split, which
// is the whole point of comparing a mapped read against a copying one. Compiles
// against a tree with or without a mapped reader for the index type under test:
// `mmap` only sets IndexIoFlag::kUseMmap, and a tree that cannot honour it
// falls back to copying, which shows up in the residency split rather than
// silently.
//
//   build  <csr> <descriptor> <out.dat>
//   search <dat> <queries.csr> <k> <reps> <inmem|mmap> [truth.txt]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "nsparse/disk_seismic_index.h"
#include "nsparse/index.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"

namespace {

struct CSRMatrix {
    int64_t nrow = 0;
    int64_t ncol = 0;
    int64_t nnz = 0;
    std::vector<nsparse::idx_t> indptr;
    std::vector<nsparse::term_t> indices;
    std::vector<float> data;
};

CSRMatrix read_csr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open CSR file: " + path);
    }
    CSRMatrix m;
    int64_t sizes[3];
    f.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    m.nrow = sizes[0];
    m.ncol = sizes[1];
    m.nnz = sizes[2];

    std::vector<int64_t> indptr64(m.nrow + 1);
    f.read(reinterpret_cast<char*>(indptr64.data()),
           static_cast<std::streamsize>((m.nrow + 1) * sizeof(int64_t)));
    m.indptr.resize(m.nrow + 1);
    for (int64_t i = 0; i <= m.nrow; ++i) {
        m.indptr[i] = static_cast<nsparse::idx_t>(indptr64[i]);
    }

    std::vector<int32_t> indices32(m.nnz);
    f.read(reinterpret_cast<char*>(indices32.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(int32_t)));
    m.indices.resize(m.nnz);
    for (int64_t i = 0; i < m.nnz; ++i) {
        m.indices[i] = static_cast<nsparse::term_t>(indices32[i]);
    }

    m.data.resize(m.nnz);
    f.read(reinterpret_cast<char*>(m.data.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(float)));
    return m;
}

// Fields from /proc/self/status, in GiB. VmHWM is the peak, so it survives the
// index being freed and is the only one comparable across residencies.
void print_memory(const char* label) {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::cout << "  mem[" << label << "]";
    while (std::getline(status, line)) {
        for (const char* field :
             {"VmHWM:", "VmRSS:", "RssAnon:", "RssFile:", "RssShmem:"}) {
            if (line.rfind(field, 0) == 0) {
                std::istringstream iss(line);
                std::string name;
                double kib = 0;
                iss >> name >> kib;
                std::cout << " " << name << " " << (kib / (1024.0 * 1024.0))
                          << " GiB";
            }
        }
    }
    std::cout << "\n";
}

double percentile(std::vector<double>& sorted, double frac) {
    if (sorted.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(std::min<double>(
        sorted.size() - 1,
        std::floor(frac * static_cast<double>(sorted.size()))));
    return sorted[idx];
}

// recall@k against a comma-separated truth file, one query per line.
double recall_at_k(const std::string& truth_path,
                   const std::vector<nsparse::idx_t>& labels, int k,
                   int n_queries) {
    std::ifstream f(truth_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open truth file: " + truth_path);
    }
    std::string line;
    double hits = 0;
    int rows = 0;
    while (rows < n_queries && std::getline(f, line)) {
        std::unordered_set<int> truth;
        std::istringstream iss(line);
        std::string tok;
        while (std::getline(iss, tok, ',')) {
            // The ids are written in scientific notation ("7.187155e+06"), so
            // they have to be read as doubles: an integer parse stops at the
            // decimal point and silently yields 7.
            if (!tok.empty() &&
                tok.find_first_not_of(" \t\r") != std::string::npos) {
                truth.insert(static_cast<int>(std::llround(std::stod(tok))));
            }
        }
        for (int j = 0; j < k; ++j) {
            const auto label = labels[static_cast<size_t>(rows) * k + j];
            if (label >= 0 && truth.count(static_cast<int>(label)) > 0) {
                hits += 1;
            }
        }
        ++rows;
    }
    return rows == 0 ? 0.0 : hits / (static_cast<double>(rows) * k);
}

int do_build(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "build <csr> <descriptor> <out.dat>\n";
        return 2;
    }
    const std::string csr_path = argv[2];
    const std::string descriptor = argv[3];
    const std::string out_path = argv[4];

    std::cout << "Loading " << csr_path << "\n";
    CSRMatrix data = read_csr(csr_path);
    std::cout << "  rows=" << data.nrow << " cols=" << data.ncol
              << " nnz=" << data.nnz << "\n";

    std::unique_ptr<nsparse::Index> index(nsparse::index_factory(
        static_cast<int>(data.ncol), descriptor.c_str()));
    const auto t0 = std::chrono::steady_clock::now();
    index->add(static_cast<nsparse::idx_t>(data.nrow), data.indptr.data(),
               data.indices.data(), data.data.data());
    std::cout << "Added. Building...\n";
    index->build();
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "build_s " << std::chrono::duration<double>(t1 - t0).count()
              << "\n";
    print_memory("after_build");

    nsparse::write_index(index.get(), const_cast<char*>(out_path.c_str()));
    std::cout << "Wrote " << out_path << "\n";
    return 0;
}

int do_search(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "search <dat> <queries.csr> <k> <reps> <inmem|mmap> "
                     "[truth.txt|-] [cut] [k_prime] [heap_factor]\n";
        return 2;
    }
    std::string dat_path = argv[2];
    const std::string query_path = argv[3];
    const int k = std::atoi(argv[4]);
    const int reps = std::atoi(argv[5]);
    const std::string residency = argv[6];
    const std::string truth_path =
        (argc > 7 && std::strcmp(argv[7], "-") != 0) ? argv[7] : "";
    const int cut = argc > 8 ? std::atoi(argv[8]) : 3;
    // k_prime > 0 selects the DiskSeismic GroC path (score summaries, take the
    // global top-k' blocks); 0 uses the plain SEISMIC heap_factor traversal.
    const int k_prime = argc > 9 ? std::atoi(argv[9]) : 0;
    // SEISMIC pruning slack (used only when k_prime==0). Skip a cluster when
    // cluster_score*heap_factor < heap.peek, so LARGER = prune fewer = higher
    // recall + slower. Sweep this to raise SEISMIC recall for iso-recall points.
    const float heap_factor = argc > 10 ? static_cast<float>(std::atof(argv[10])) : 1.0F;

    const int io_flags =
        residency == "mmap" ? nsparse::IndexIoFlag::kUseMmap : 0;
    if (residency != "mmap" && residency != "inmem") {
        std::cerr << "residency must be inmem or mmap\n";
        return 2;
    }

    CSRMatrix query = read_csr(query_path);
    const int n_queries = static_cast<int>(query.nrow);
    std::cout << "queries " << n_queries << " nnz " << query.nnz << "\n";

    print_memory("before_load");
    const auto l0 = std::chrono::steady_clock::now();
    std::unique_ptr<nsparse::Index> index(
        nsparse::read_index(const_cast<char*>(dat_path.c_str()), io_flags));
    const auto l1 = std::chrono::steady_clock::now();
    std::cout << "residency " << residency << "\n";
    std::cout << "load_s " << std::chrono::duration<double>(l1 - l0).count()
              << "\n";
    std::cout << "num_vectors " << index->num_vectors() << "\n";
    const auto* vectors = index->get_vectors();
    std::cout << "element_size "
              << (vectors == nullptr ? 0 : vectors->get_element_size()) << "\n";
    print_memory("after_load");

    std::unique_ptr<nsparse::SeismicSearchParameters> params;
    if (k_prime > 0) {
        params = std::make_unique<nsparse::DiskSeismicSearchParameters>(cut, k_prime);
        std::cout << "search_params cut " << cut << " k_prime " << k_prime << "\n";
    } else {
        params = std::make_unique<nsparse::SeismicSearchParameters>(cut, heap_factor);
        std::cout << "search_params cut " << cut << " heap_factor " << heap_factor << "\n";
    }
    std::vector<float> distances(static_cast<size_t>(n_queries) * k);
    std::vector<nsparse::idx_t> labels(static_cast<size_t>(n_queries) * k);

    // Batched, one call for the whole query set: comparable to the protocol's
    // QPS. Rep -1 is an untimed warmup that faults the index in.
    for (int rep = -1; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        index->search(static_cast<nsparse::idx_t>(n_queries),
                      query.indptr.data(), query.indices.data(),
                      query.data.data(), k, distances.data(), labels.data(),
                      params.get());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (rep < 0) {
            std::cout << "warmup_batch_ms " << ms << "\n";
            continue;
        }
        std::cout << "batch_ms " << ms << " qps "
                  << (static_cast<double>(n_queries) * 1000.0 / ms) << "\n";
    }

    // Per-query calls for the tail. Slightly pessimistic in absolute terms --
    // each call re-allocates the dense scratch the batched path amortizes --
    // but identical across residencies, which is what is being compared.
    std::vector<double> per_query_ms;
    per_query_ms.reserve(static_cast<size_t>(n_queries));
    for (int qi = 0; qi < n_queries; ++qi) {
        const nsparse::idx_t start = query.indptr[qi];
        const nsparse::idx_t end = query.indptr[qi + 1];
        std::vector<nsparse::idx_t> q_indptr = {0, end - start};
        const auto t0 = std::chrono::steady_clock::now();
        index->search(1, q_indptr.data(), query.indices.data() + start,
                      query.data.data() + start, k, distances.data(),
                      labels.data(), params.get());
        const auto t1 = std::chrono::steady_clock::now();
        per_query_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(per_query_ms.begin(), per_query_ms.end());
    double sum = 0;
    for (const double ms : per_query_ms) sum += ms;
    std::cout << "single_query_ms_mean " << sum / per_query_ms.size() << " p50 "
              << percentile(per_query_ms, 0.50) << " p90 "
              << percentile(per_query_ms, 0.90) << " p99 "
              << percentile(per_query_ms, 0.99) << "\n";

    // Recall from a final batched pass, so `labels` holds every query again.
    index->search(static_cast<nsparse::idx_t>(n_queries), query.indptr.data(),
                  query.indices.data(), query.data.data(), k, distances.data(),
                  labels.data(), params.get());
    if (!truth_path.empty()) {
        std::cout << "recall@" << k << " "
                  << recall_at_k(truth_path, labels, k, n_queries) << "\n";
    }
    print_memory("after_search");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: sq_residency_bench build|search ...\n";
        return 2;
    }
    try {
        if (std::strcmp(argv[1], "build") == 0) return do_build(argc, argv);
        if (std::strcmp(argv[1], "search") == 0) return do_search(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown mode " << argv[1] << "\n";
    return 2;
}
