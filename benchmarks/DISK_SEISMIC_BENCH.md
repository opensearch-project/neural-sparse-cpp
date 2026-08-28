# DiskSeismic C++ benchmark (disk-vs-RAM residency)

Measures whether the disk-resident DiskSeismic index holds up on latency/recall as
RAM shrinks below the index size — the core question before investing in the full
OpenSearch plugin integration. Pure C++; no OpenSearch, no plugin.

## What it compares

Three residency regimes over the same MS MARCO corpus, single-thread query:

| # | Variant | Index / residency | Search |
|---|---------|-------------------|--------|
| V1 | in-RAM SEISMIC | `seismic`, in-memory (heap copy) | heap_factor traversal |
| V2 | scattered-disk SEISMIC | `seismic`, mmap (mapped forward index) | heap_factor traversal |
| V3 | **DiskSeismic** | `disk_seismic`, mmap (inline forward index) | **GroC top-k'** |

The result is the **RAM-cap sweep**: below the index size, V1 (heap) is OOM-killed
while the mmap variants keep serving by paging from disk. How V2/V3 latency degrades
as RAM shrinks — and where V1 dies — is the finding. DiskSeismic keeps only cluster
summaries resident (`RssAnon`), paging the forward index on demand (`RssFile`).

## Host prereqs (fresh g5.12xlarge)

- **g5.12xlarge = AMD Zen2 (EPYC 7R32): no AVX-512.** Build uses `OPT_LEVEL=avx2` (default). `avx512` would not run.
- Toolchain: `gcc10` (C++20) + `cmake3`. Build with `OPT_LEVEL=avx2` (the default); `OPT_LEVEL=generic` uses the scalar kernels and costs ~1.11–1.21× on the search path.
- **Run as root** — the cgroup cap (`systemd-run --scope`) and dropping the page cache between runs both need it. As a normal user the cap runs fail with "Interactive authentication required".
- ~24 GB free disk for the corpus + ~15 GB per serialized index (two here).

## Get the data

```bash
export NSPARSE_DATA_DIR=/data
mkdir -p "$NSPARSE_DATA_DIR" && cd "$NSPARSE_DATA_DIR"
curl -O https://do0ia2psryw9c.cloudfront.net/base_full.csr.gz
curl -O https://do0ia2psryw9c.cloudfront.net/queries.dev.csr.gz
curl -O https://do0ia2psryw9c.cloudfront.net/iv_array.txt
gunzip base_full.csr.gz queries.dev.csr.gz
# verify: (8841823, 30109, 1121199371)
python3 -c "import struct;print(struct.unpack('<3q',open('base_full.csr','rb').read(24)))"
```

## Run

```bash
sudo NSPARSE_DATA_DIR=/data \
  CAPS="unlimited 12G 8G 4G" \
  ./benchmarks/run_disk_seismic_bench.sh
```

It builds the binary, builds both indexes once (reused on re-runs), then sweeps
caps × variants and writes `summary.tsv` + per-run logs to
`$NSPARSE_DATA_DIR/disk_seismic_bench_out/`.

Knobs (env): `OPT_LEVEL=avx2 LAMBDA=6000 BETA=400 ALPHA=0.4 CUT=3 KPRIME=50 K=10 REPS=5 CAPS="unlimited 12G 8G 4G"`.
Set caps around and below the index size (~15 GB float; DiskSeismic's inline file is
larger on disk but only its summaries stay resident) to expose the disk benefit.

## Reading the output (`summary.tsv`)

`variant  cap  status  p50_ms  p90_ms  p99_ms  qps  recall  RssAnon/RssFile_GiB`

- `status=OOM/FAIL` at a cap = that variant could not run there. Expect V1 to hit this
  once the cap drops below its ~15 GB heap; V2/V3 should keep `status=ok`.
- Compare V3 (DiskSeismic) latency and recall to V1's uncapped baseline: the bet is
  that V3 stays acceptable at caps where V1 is dead.
- Memory split: V1 is almost all `RssAnon` (heap); V2/V3 are almost all `RssFile`
  (mapped, reclaimable), V3 with a small `RssAnon` for summaries.

## Caveats

- **`base_small` is a smoke test, not a benchmark** — it fits in cache, so the
  memory-bound behavior that is the whole point disappears. Only report `base_full`.
- Regenerate the `.dat` files whenever the source tree changes; the serialized format
  carries no version and an old binary silently misreads a new file.
- Query threads are pinned to 1 (`OMP_NUM_THREADS=1`) — per-query cost is the metric
  and the workload is bandwidth-bound.
