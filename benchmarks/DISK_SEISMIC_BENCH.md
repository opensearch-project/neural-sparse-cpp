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

## Truncating the extra inline copies — `inline_max_nnz`

Packing the blocks removed bytes nothing read. What remains is bytes that *are* read but are
**duplicated**: `index_size_stats` puts the inline forward index at 45.6 of the 50.0 GiB, and
a doc's vector is stored **12.97 times** over — once per posting list that retains it. One
shared copy per doc would take `comps[] + vals[]` from 44.28 GiB to 3.42. That duplication is
not waste, it is the whole design: it makes a block one contiguous read.

`inline_max_nnz=T` splits the difference. One copy of each doc stays whole (the one the
`DocLocator` directory already points at); every other copy keeps only its T largest codes.
Block scores are then **lower bounds**, so search re-scores its `rescore` best candidates
against the whole copies and re-ranks. Two knobs, one at build time and one per query:

| | T=0 (baseline) | T=64 | T=48 | T=32 |
|---|---|---|---|---|
| index file | 49.967 GiB | 27.667 GiB | **22.971 GiB** | 18.246 GiB |
| vs baseline | | −44.6% | **−54.0%** | −63.5% |
| inline forward index | 45.570 GiB | 23.27 GiB | 18.574 GiB | 13.850 GiB |
| stored nnz | 15.85 e9 | 7.78 e9 | 6.20 e9 | 4.51 e9 |
| cluster summaries | 4.297 GiB (8.6%) | 4.297 (15.5%) | 4.297 (18.7%) | 4.297 (23.6%) |
| build wall / peak RSS | 9:01 / 20.9 GiB | 11:20 / 20.9 | 11:12 / 20.9 | 8:36 / 20.9 |

Latency and recall, `base_full` / λ=6000 β=400 α=0.4 / 8-bit / cut=3 k'=50 k=10 / 1 thread,
`seismic-ec2`, one binary per round over all the files, caches dropped before each arm's first
pass. `seed=42` on every build, so recall differences are the format's and not k-means noise:

| arm | rescore | warm p50 | vs baseline | recall@10 | Δ recall |
|---|---|---|---|---|---|
| T=0 | — | 0.2157 ms | 1.00× | 0.926246 | — |
| T=64 | 10 | 0.1698 ms | 0.79× | 0.844728 | −8.2 pp |
| T=64 | 50 | 0.1899 ms | 0.88× | 0.924900 | −0.13 pp |
| T=64 | 100 | **0.2117 ms** | **0.98×** | 0.925989 | −0.03 pp |
| T=64 | 200 | 0.2565 ms | 1.19× | 0.926232 | −0.001 pp |
| T=64 | 300 | 0.2986 ms | 1.38× | 0.926203 | −0.004 pp |
| T=48 | none | 0.1569 ms | 0.73× | 0.784585 | −14.2 pp |
| T=48 | 100 | 0.2033 ms | 0.94× | 0.924484 | −0.18 pp |
| T=48 | 200 | **0.2470 ms** | **1.15×** | 0.925888 | −0.04 pp |
| T=48 | 300 | 0.2877 ms | 1.33× | 0.926089 | −0.02 pp |
| T=48 | 600 | 0.4046 ms | 1.88× | 0.926203 | −0.004 pp |
| T=32 | 100 | 0.1936 ms | 0.90× | 0.915115 | −1.11 pp |
| T=32 | 200 | 0.2359 ms | 1.09× | 0.922908 | −0.33 pp |
| T=32 | 300 | 0.2760 ms | 1.28× | 0.924699 | −0.15 pp |

Using it, from C++ or Python (the SWIG bindings expose the same names):

```python
index = nsparse.index_factory(dim, "disk_seismic_sq,quantizer=8bit|vmin=0|vmax=3|"
                                   "lambda=6000|beta=400|alpha=0.4|inline_max_nnz=64")
# ... add, build, write_index, read_index(path, nsparse.kUseMmap) ...
params = nsparse.DiskSeismicSQSearchParameters(0.0, 3.0, 3, 50, 100)  # vmin vmax cut k' rescore
params.rescore = 200            # per query; default nsparse.kDefaultRescoreDepth (200)
```

`inline_max_nnz` is fixed at build time and recorded in the file. `rescore` is ignored on an
untruncated index, raised to k when smaller, and rejected when negative.

Two points worth naming. **T=64 with rescore=100 is free**: 44.6% off the file at 0.98× the
latency and a recall difference smaller than the ±0.1 pp a k-means seed causes on its own.
**T=48 with rescore=200** buys another 9 points of file size for 1.15× latency, still at
recall parity. Below T=48 recall starts to cost real ground, so T=32 is only worth it when disk
is the binding constraint.

Read the "none" row as the cost of the scan alone: with 39% of the nnz left inline, scoring the
candidate blocks is **0.73×** the baseline. Everything above that is the second pass, and it is
not arithmetic — at ~0.45 µs per re-scored doc it is ~1400 cycles to read 420 bytes, because
reaching one whole copy is a four-deep chase (locator → directory → block header → slice)
through tables far past any cache. That is why `rescore` is the latency knob and why
`resolve_docs` walks the candidates through each step together, in bounded chunks, rather than
one doc at a time: doing that took the per-doc cost from 0.68 µs to 0.45 µs (−34%) with
bit-identical results at every depth.

Two honest costs beyond latency:

- **Cold first pass** 0.897 → 1.153 Ms (+28.5%) at T=48. The file is half the size but the
  second pass touches ~300 additional scattered blocks per query, which is 6× the 50 the scan
  reads, so a cold run faults in more distinct pages even though there are fewer bytes overall.
- **Resident set** 6.04 → 7.56 GiB (+25%) at T=48, `RssFile` 5.71 → 7.23, for the same reason.
  The smaller file does *not* buy a smaller working set here; it buys disk.

What did **not** work, so nobody repeats it:

- **Choosing the kept components per block instead of per doc.** The hypothesis was that a
  query selecting a block correlates with that block's summary, so keeping the components with
  the largest `code × block-max` should rank better than the doc's own largest codes.
  Simulated over the real index it is a wash — T=48 at depth 100 gives 0.9249 against 0.9246,
  T=32 0.9172 against 0.9175. The block max is usually the doc's own value for the components
  that matter, so the ordering barely moves. More fundamentally: which components a query will
  hit is query information, and no static choice can carry it.
- **Bounding the second pass** so most candidates skip it. The dropped codes are all below the
  doc's boundary code, which gives `truncated + boundary × query_L1` as an upper bound — two
  orders of magnitude looser than the scores it would have to separate. It prunes nothing.

`prune_sim <index.dat> <queries.csr> <truth.txt> <k> <cut> <k'> <T,...> <R,...> [criterion]`
is what made this affordable: it reads an existing index and runs the real block-budget search
over a *simulated* truncated layout, so a whole (T, rescore) grid costs one pass instead of a
build per cell. Its T=0/R=0 row must reproduce the index's own measured recall — that is the
control, and it does (0.926103 against 0.926074 measured). Trust it for recall only; it
recomputes the truncation per query, so its timings mean nothing.

## Caveats

- **`base_small` is a smoke test, not a benchmark** — it fits in cache, so the
  memory-bound behavior that is the whole point disappears. Only report `base_full`.
- Regenerate the `.dat` files whenever the on-disk layout changes. The header carries a
  per-type format version, so a mismatch is caught only if the change bumped
  `kFormatVersion`; one that did not is misread silently.
- Query threads are pinned to 1 (`OMP_NUM_THREADS=1`) — per-query cost is the metric
  and the workload is bandwidth-bound.
