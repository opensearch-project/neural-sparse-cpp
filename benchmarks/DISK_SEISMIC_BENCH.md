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

## Where the on-disk bytes go — `index_size_stats`

`index_size_stats <index.dat>` re-parses a serialized `disk_seismic`/`disk_seismic_sq` file
and prints the section split, the inline forward index's per-array split, and what a narrower
encoding of each array would save. It only reads, so sizing a format change costs one pass
over the file instead of a rebuild.

That is how the block padding was found. A block holds about ten documents — a ~4.5 KB
payload — and `InlineForwardIndex` was aligning each to 4096, so **a third of the file was
padding nothing ever read**: 24.3 GiB of 74.2 on `base_full` at `lambda=6000 beta=400
alpha=0.4`, 8-bit `disk_seismic_sq`. Packed placement (`InlineLayout::kPacked`, now the
default) removes it. Same format either way — the header records the alignment and a reader
honours whatever the file declares — so a file written before the default changed still
loads, which is what let this A/B run one binary over two files:

| | page-aligned | packed | |
|---|---|---|---|
| index file | 74.205 GiB | **49.967 GiB** | −32.7% |
| inter-block padding | 24.273 GiB | 0.035 GiB | |
| bytes per non-zero | 4.73 | 3.09 | |
| first (cold) query pass | 982,994 ms | 888,829 ms | −9.6% |
| warm batch, 6,980 queries | 1492.6 ms | 1464.3 ms | −1.9% |
| warm p50 / p90 / p99 | 0.221 / 0.268 / 0.312 ms | 0.217 / 0.264 / 0.308 ms | −1.6% |
| `VmHWM` / `RssFile` | 6.567 / 6.236 GiB | 6.042 / 5.712 GiB | −8.0% |
| recall@10 | 0.9252 | 0.9261 | k-means seed noise |

Padding cost storage and page cache, not page faults — a page-aligned block already spanned
two pages and the padding sat in the tail of the second one. So removing it buys file size
and a smaller resident set, and touches query CPU only to the extent that blocks now share
pages. Warm latency came out marginally *better*, not worse.

The recall difference is seed noise: the two indexes were built at different times without a
fixed `seed=`. The layout change itself is exact, which `sq_residency_bench`'s trailing
`labels_out.txt` argument is there to show — dump one build's labels, score the other against
them, expect 1.0. With a fixed seed the two layouts' label files are byte-identical.

What is left, per the same tool, if the file needs to shrink further: component ids are 64.8%
of the inline forward index and delta coding would take 29.5 → 19.4 GiB, but decoding sits on
the critical path of scoring a block and measured a 1.5× warm-latency cost, so it was not
taken. 4-bit values would save 7.4 GiB and is lossy. `doc_id[]`/`off[]` are ~1% together.

## Caveats

- **`base_small` is a smoke test, not a benchmark** — it fits in cache, so the
  memory-bound behavior that is the whole point disappears. Only report `base_full`.
- Regenerate the `.dat` files whenever the on-disk layout changes. The header carries a
  per-type format version, so a mismatch is caught only if the change bumped
  `kFormatVersion`; one that did not is misread silently.
- Query threads are pinned to 1 (`OMP_NUM_THREADS=1`) — per-query cost is the metric
  and the workload is bandwidth-bound.
