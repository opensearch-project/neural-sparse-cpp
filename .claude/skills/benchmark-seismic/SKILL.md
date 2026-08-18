---
name: benchmark-seismic
description: >-
  Benchmark the Seismic index in this repo — the fixed protocol (MS MARCO base_full.csr,
  lambda=6000/beta=400/alpha=0.4, cut=3/heap_factor=1.0, single-thread query) plus how to obtain the
  dataset, which harness to run (query_latency_bench, nsparse_benchmark, seismic_mmap.py,
  disk_build_mem_bench), and which metrics to report. Use whenever measuring, comparing, or validating
  query latency, QPS, recall, or memory (RSS/Anon) for a change to nsparse; when A/B-ing a branch
  against a baseline; or when deciding whether an optimization clears the accept bar. Skip for
  correctness-only work that makes no performance claim.
---

# Benchmarking the Seismic index

A benchmark result is only meaningful next to a baseline measured the same way. This fixes the
dataset, parameters, and threading so numbers from different runs, branches, and people are
comparable. Deviating is fine — say so explicitly in the report, because a run at a different
`lambda` or thread count is not a baseline comparison.

For the optimization methodology (perf triage, what counts as a win), see
`SEISMIC_QUERY_OPTIMIZATION.md` at the repo root.

## Dataset

MS MARCO v1 passage embeddings, in the interchange CSR layout this repo reads (`int64` rows/cols/nnz
header, then `int64` indptr, `int32` indices, `float32` values).

| File | Bytes | Contents |
|---|---|---|
| `base_full.csr` | 9,040,329,584 | corpus: 8,841,823 docs × 30,109 terms, 1,121,199,371 nnz |
| `queries.dev.csr` | 2,797,440 | 6,980 queries, 342,696 nnz |
| `iv_array.txt` | 1,745,000 | ground truth, top-10 ids per query, comma-separated |

Check for them before downloading — they are large and rarely change:

```bash
DATA=${NSPARSE_DATA_DIR:?set NSPARSE_DATA_DIR to your dataset directory}
ls -l "$DATA"/{base_full.csr,queries.dev.csr,iv_array.txt}
```

If missing:

```bash
mkdir -p "$DATA" && cd "$DATA"
curl -O https://do0ia2psryw9c.cloudfront.net/base_full.csr.gz    # 5,936,786,435 B compressed
curl -O https://do0ia2psryw9c.cloudfront.net/queries.dev.csr.gz  # 1,849,192 B
curl -O https://do0ia2psryw9c.cloudfront.net/iv_array.txt        # not compressed
gunzip base_full.csr.gz queries.dev.csr.gz
```

Budget ~24 GB of free disk for the corpus alone (9 GB decompressed + 6 GB archive, deletable after),
and another ~9–16 GB per serialized index you keep. Verify a download by reading the header — the
corpus must report 8,841,823 rows:

```bash
python3 -c "
import struct
print(struct.unpack('<3q', open('$DATA/base_full.csr','rb').read(24)))"
# (8841823, 30109, 1121199371)   rows, cols, nnz
```

`base_small.csr` (102,639,664 B) with `iv_small_array.txt` is a smoke test, not a benchmark: at that
size the index fits in cache and the memory-bound behavior that dominates the real workload
disappears. Never report a small-corpus number as a result.

## The protocol

| Knob | Value | Why |
|---|---|---|
| Corpus | `base_full.csr` | see above |
| Build params | `lambda=6000, beta=400, alpha=0.4` | the standard config |
| Queries | `queries.dev.csr`, all 6,980 | |
| Query params | `cut=3, heap_factor=1.0`, k=10 (add k=100 for throughput comparisons) | |
| Truth | `iv_array.txt`, recall@10 | |
| Index-build threads | all cores | build time is not the metric |
| **Query threads** | **exactly 1** (`OMP_NUM_THREADS=1`) | per-query cost is the metric; the workload is bandwidth-bound, so multi-thread runs compress real deltas |
| Host | a dedicated machine, ≥16 cores and ≥32 GB | the float index is ~15 GB resident; a shared or smaller box adds noise larger than most effects being measured |

Two choices must be stated with every result, because each changes latency, memory, and recall:

1. **Index type** — `seismic` (plain float) or `seismic_sq,quantizer=8bit|vmin=…|vmax=…` (~2× smaller,
   uint8 SIMD kernels, different scores). An `idmap,` prefix wraps either.
2. **Residency** — in-memory (heap copy, the default) or mmap (`IndexIoFlag::kUseMmap`, borrowed in
   place from the file). A mmap run is not comparable to an in-memory one.

## Metrics — report all four

- **Latency**, single-thread: mean plus p50/p90/p99. A batch mean cannot show a tail.
- **QPS**.
- **Memory**: peak RSS **and** `RssAnon` separately. Under mmap the index lands in
  `RssFile`/`RssShmem`, so total RSS alone cannot distinguish a heap copy from a mapping. Fields from
  `/proc/<pid>/status`: `VmHWM`, `VmRSS`, `RssAnon`, `RssFile`, `RssShmem`.
- **Recall@10** against `iv_array.txt`. A change that moves recall is not an optimization; report the
  baseline and new value even when unchanged.

Compare against a baseline measured on the same host, ideally interleaved in the same session rather
than recalled from a previous week.

## Build: confirm the binary is optimized

An unoptimized build silently invalidates every number.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNSPARSE_OPT_LEVEL=avx512 -DNSPARSE_ENABLE_BENCHMARKS=ON
cmake --build build -j$(nproc)
grep -E "CMAKE_BUILD_TYPE|CXX_FLAGS" build/CMakeCache.txt   # expect Release, -O3 -DNDEBUG
```

An **existing build directory keeps its old cache** — cmake will not upgrade `CMAKE_BUILD_TYPE` on a
reconfigure. Delete `build/` when in doubt. `NSPARSE_OPT_LEVEL=generic` compiles without the AVX
kernels and costs roughly 2×; that is a build mistake, not a measurement.

## Harnesses

All commands assume `DATA` points at the dataset directory.

### Latency percentiles — `query_latency_bench`

The google-benchmark harness reports whole-batch mean only (one batch call is one sample), so it
cannot produce a tail. This driver arms per-query timing inside the same batched `search()`.

```bash
./build/benchmarks/query_latency_bench build \
  "$DATA/base_full.csr" 6000 400 0.4 "$DATA/base_full.csr.seismic.dat"

OMP_NUM_THREADS=1 ./build/benchmarks/query_latency_bench search \
  "$DATA/base_full.csr.seismic.dat" "$DATA/queries.dev.csr" 10 5
```

Prints `batch_ms_mean/std`, QPS, and `lat_ms_p50/p90/p99`. Rep −1 is an untimed warmup that faults in
the index; never report a cold first rep.

### QPS and hardware counters — `nsparse_benchmark`

Driven by environment variables; caches the built index as `<data>.seismic.dat` next to the corpus.

```bash
export NSPARSE_DATA_CSR="$DATA/base_full.csr"
export NSPARSE_QUERY_CSR="$DATA/queries.dev.csr"
export NSPARSE_LAMBDA=6000 NSPARSE_BETA=400          # alpha is fixed at 0.4 in the fixture
OMP_NUM_THREADS=1 perf stat -e cycles,instructions,branch-misses,cache-misses,LLC-load-misses \
  ./build/benchmarks/nsparse_benchmark --benchmark_filter=BM_Seismic_Search
```

Filters: `BM_Seismic_Search`, `BM_SeismicSQ_Search` (honors `NSPARSE_SQ_BITS`), and
`BM_InvertedIndex_Search` for the exact baseline. Each registers k=10 and k=100 at 10 repetitions.

### Memory and residency — `demos/seismic_mmap.py`

The only harness that reports the `RssAnon` / `RssFile` split and `VmHWM` per residency mode.

```bash
OMP_NUM_THREADS=1 python demos/seismic_mmap.py \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" \
  --lambda- 6000 --beta 400 --alpha 0.4 --cut 3 --heap-factor 1.0 --k 10 \
  --reuse-index --keep
```

Needs `numpy>=2.1` installed *over* the nsparse package: `pyproject.toml` pins `numpy<2.0`, which on
Python 3.13+ resolves to a numpy older than the interpreter and silently corrupts arrays (`a - b`
overwrites `a`) while every check still passes. Pin Python 3.12 or force the newer numpy.

### Peak build memory — `disk_build_mem_bench`

Reports `VmHWM` and wall time for a build, comparing the batched `DiskSeismicIndex` path against the
in-memory baseline.

```bash
./build/benchmarks/disk_build_mem_bench baseline "$DATA/base_full.csr" 6000 400 0.4
```

### Recall

```python
import numpy as np
correct = np.loadtxt(f"{DATA}/iv_array.txt", delimiter=",").astype(int)
n = min(len(results), len(correct))
recall = np.mean([len(set(map(int, results[i])) & set(map(int, correct[i])))
                  for i in range(n)]) / k
```

## Gotchas that have invalidated real runs

- **The serialized `.dat` header carries no format version** — `write_header` writes only fourcc and
  dimension. A new binary reading an old file throws `std::length_error` (loud), but an **old binary
  reading a new file silently loads a garbage index** with the correct `num_vectors` and wrong
  results. Any change to on-disk layout (e.g. the alignment padding in `nsparse/io/align.h`)
  invalidates every cached `.dat`. Regenerate them whenever the tree changes, and distrust a
  suspiciously large speedup — one apparent 100× win was a binary misparsing a stale file.
- **mmap requires the unquantized write path.** `seismic_sq` is not mmap-able, so enabling
  `kUseMmap` means writing plain `seismic` and losing 8-bit quantization — the index file roughly
  doubles. Do not attribute a latency change to residency when quantization moved with it.
- **Change one variable per run.** A comparison that moved both `lambda` (2000→6000) and quantization
  produced 10.8 ms vs 1,126 ms and could not be attributed to either.
- **Warm up, then verify where the time goes.** Check `iowait` and page-cache residency before
  concluding "CPU-bound"; equally, a fully-cached index with 0% iowait and high latency is a genuine
  single-thread CPU cost, not I/O.
- **Keep the SQ variant in sync** — `seismic_scalar_quantized_index.cpp` shares `reorder_clusters`
  and the summary-scoring pattern, so a win here usually applies there and should be measured there.

## Bar for accepting an optimization

From `SEISMIC_QUERY_OPTIMIZATION.md`: **≥20% QPS improvement, recall unchanged, memory growth under
50%.** Measure in cycles with `perf`, classify the bottleneck (branch-misses → branching;
cache/LLC-misses → memory; otherwise ILP/SIMD), change one thing, re-measure, and record the measured
delta in that document's checklist.
