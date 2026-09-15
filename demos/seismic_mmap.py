# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

#!/usr/bin/env python3
"""
End-to-end check of SeismicIndex mmap mode through the NSPARSE Python bindings.

One index is built from a CSR corpus, written to disk, then read back both ways:

    read_index(path)                  -- deserialized onto the heap
    read_index(path, nsparse.kUseMmap) -- borrowed in place from the file

Both reads deserialize the same built index, so they must agree on every label
and score. That equality is the real invariant here; the timings and memory
figures alongside it are measurements, not assertions, except for the one check
that mapping keeps the index off the heap.

Then the other mmap entry point, read_csr with Residency.kMmap, which borrows the
raw CSR in place instead of copying it in:

    read_csr(path, Residency_kInMemory)   -- copied onto the heap
    convert(path, native); read_csr(native, Residency_kMmap)  -- borrowed

Clustering seeds from std::random_device, so two builds of the same corpus do not
produce the same search results. This phase therefore checks what is invariant --
vector counts, that the mapped reader rejects a non-native file rather than
silently copying, and that the mapped index is searchable -- plus the heap it
saves. Elementwise vector comparison stays in tests/mmap_index_test.cpp, since
the bindings do not hand out the raw CSR pointers.

Usage:
    python demos/seismic_mmap.py <data.csr> <queries.csr> [options]

Runs on whatever numpy the nsparse package is installed with, 1.26+ or 2.x: the
extension is built against numpy 2.x headers and loads on either.

Exits non-zero if any check fails.
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

import nsparse

# term_t is uint16 on the C++ side, so a corpus wider than this cannot be loaded.
MAX_TERMS = np.iinfo(np.uint16).max + 1

failures = 0

# Below this, a heap delta is allocator noise rather than a measurement, so the
# memory comparisons report but do not assert.
MIN_MEASURABLE_KB = 4 * 1024


def check(ok, what):
    global failures
    print(f"  {'[ok]  ' if ok else '[FAIL]'} {what}")
    if not ok:
        failures += 1


def banner(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Process metrics
# ---------------------------------------------------------------------------


def proc_status_kb(key):
    """A field of /proc/self/status in kB, or 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith(key):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def rss_anon_kb():
    """Anonymous (heap) pages.

    This is what separates the two reads: a copying read grows the heap by the
    size of the index, a mapping does not. Total RSS does not separate them,
    since faulted-in mapping pages count towards it just the same.
    """
    return proc_status_kb("RssAnon:")


def rss_mapped_kb():
    """Resident pages backed by a file.

    A mapping of a file on a real filesystem lands in RssFile, but one on tmpfs
    lands in RssShmem; summing them keeps the number meaningful either way.
    """
    return proc_status_kb("RssFile:") + proc_status_kb("RssShmem:")


def peak_rss_kb():
    return proc_status_kb("VmHWM:")


def mib(kb):
    return f"{kb / 1024.0:.1f} MiB"


def filesystem_type(path):
    """Filesystem backing `path`, via the longest matching mount point."""
    try:
        with open("/proc/mounts") as mounts:
            entries = [line.split()[1:3] for line in mounts]
    except OSError:
        return ""
    target = os.path.realpath(path)
    best = ""
    fstype = ""
    for mount_point, mount_type in entries:
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) >= len(best):
                best, fstype = mount_point, mount_type
    return fstype


# ---------------------------------------------------------------------------
# CSR reading
# ---------------------------------------------------------------------------


def read_csr_header(path):
    """(rows, cols, nnz) from the int64 header."""
    with open(path, "rb") as handle:
        rows, cols, nnz = np.fromfile(handle, dtype=np.int64, count=3)
    return int(rows), int(cols), int(nnz)


def read_csr(path):
    """Read an interchange-layout CSR file into the widths the bindings want.

    The layout is an int64 (rows, cols, nnz) header, then int64 indptr, int32
    indices and float32 values -- narrowed here to the int32/uint16/float32 the
    search typemaps expect.
    """
    with open(path, "rb") as handle:
        rows, cols, nnz = np.fromfile(handle, dtype=np.int64, count=3)
        rows, cols, nnz = int(rows), int(cols), int(nnz)
        if cols > MAX_TERMS:
            raise ValueError(
                f"{path}: {cols} columns exceeds the {MAX_TERMS} a uint16 term id holds"
            )
        indptr = np.fromfile(handle, dtype=np.int64, count=rows + 1)
        indices = np.fromfile(handle, dtype=np.int32, count=nnz)
        values = np.fromfile(handle, dtype=np.float32, count=nnz)
    if len(indptr) != rows + 1 or len(indices) != nnz or len(values) != nnz:
        raise ValueError(f"{path}: truncated CSR file")
    return {
        "rows": rows,
        "cols": cols,
        "nnz": nnz,
        "indptr": indptr.astype(np.int32),
        "indices": indices.astype(np.uint16),
        "values": values,
    }


def take_queries(csr, count):
    """The first `count` rows of a CSR, as its own set of arrays."""
    if count >= csr["rows"]:
        return csr["indptr"], csr["indices"], csr["values"]
    end = int(csr["indptr"][count])
    return csr["indptr"][: count + 1], csr["indices"][:end], csr["values"][:end]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def run_search(index, queries, n_queries, k, cut, heap_factor):
    indptr, indices, values = queries
    params = nsparse.SeismicSearchParameters(cut, heap_factor)
    started = time.perf_counter()
    distances, labels = index.search(n_queries, indptr, indices, values, k, params)
    return distances, labels, (time.perf_counter() - started) * 1000.0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="End-to-end check of SeismicIndex mmap mode."
    )
    parser.add_argument("data_csr", help="corpus in interchange CSR layout")
    parser.add_argument("query_csr", help="queries in interchange CSR layout")
    parser.add_argument(
        "--workdir",
        default=None,
        help="where the serialized index goes (default: the data file's directory)",
    )
    parser.add_argument("--k", type=int, default=10, help="top-k, default 10")
    parser.add_argument(
        "--queries",
        type=int,
        default=0,
        help="use only the first N queries, default all",
    )
    parser.add_argument("--lambda-", dest="lambda_", type=int, default=10)
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--cut", type=int, default=10)
    parser.add_argument("--heap-factor", type=float, default=1.0)
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="reuse an existing serialized index instead of rebuilding",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the serialized index on exit"
    )
    return parser.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)

    workdir = opts.workdir or (os.path.dirname(os.path.abspath(opts.data_csr)) or ".")
    index_path = os.path.join(
        workdir, os.path.basename(opts.data_csr) + ".mmap_demo.dat"
    )

    rows, cols, nnz = read_csr_header(opts.data_csr)
    print(f"data:    {opts.data_csr}  rows={rows} cols={cols} nnz={nnz}")
    print(f"queries: {opts.query_csr}")
    note = ""
    if filesystem_type(workdir) == "tmpfs":
        note = "  [tmpfs: the index lives in RAM, so the load times and memory split below are not a disk story]"
    print(f"workdir: {workdir}{note}")
    descriptor = f"seismic,lambda={opts.lambda_}|beta={opts.beta}|alpha={opts.alpha}"
    print(
        f"seismic: {descriptor}  cut={opts.cut} heap_factor={opts.heap_factor} k={opts.k}"
    )

    queries_csr = read_csr(opts.query_csr)
    n_queries = (
        min(opts.queries, queries_csr["rows"]) if opts.queries > 0 else queries_csr["rows"]
    )
    queries = take_queries(queries_csr, n_queries)
    print(f"         {n_queries} of {queries_csr['rows']} queries used")

    # -----------------------------------------------------------------------
    banner("Build and serialize")
    # -----------------------------------------------------------------------
    if opts.reuse_index and os.path.exists(index_path):
        print(f"  reusing {index_path}")
    else:
        index = nsparse.index_factory(cols, descriptor)

        started = time.perf_counter()
        index.read_csr(opts.data_csr)
        print(
            f"  read_csr      {(time.perf_counter() - started) * 1000:.0f} ms, "
            f"{index.num_vectors()} vectors"
        )

        started = time.perf_counter()
        index.build()
        print(f"  build         {(time.perf_counter() - started) * 1000:.0f} ms")

        started = time.perf_counter()
        nsparse.write_index(index, index_path)
        print(
            f"  write_index   {(time.perf_counter() - started) * 1000:.0f} ms, "
            f"{mib(os.path.getsize(index_path) / 1024)}"
        )

        # Freed before the loads below are measured, so its heap does not count
        # against them.
        del index
        gc.collect()

    # -----------------------------------------------------------------------
    banner("read_index(path, kUseMmap) vs read_index(path)")
    # -----------------------------------------------------------------------
    # Mapped first, on a clean heap. The copied load that follows reuses whatever
    # the allocator kept back, so its RssAnon delta is if anything understated --
    # which only makes the comparison conservative.
    anon_before_mapped = rss_anon_kb()
    mapped_before = rss_mapped_kb()
    started = time.perf_counter()
    mapped = nsparse.read_index(index_path, nsparse.kUseMmap)
    mapped_load_ms = (time.perf_counter() - started) * 1000.0
    anon_after_mapped = rss_anon_kb()
    mapped_after_load = rss_mapped_kb()

    mapped_scores, mapped_labels, mapped_search_ms = run_search(
        mapped, queries, n_queries, opts.k, opts.cut, opts.heap_factor
    )
    mapped_after_search = rss_mapped_kb()
    mapped_vectors = mapped.num_vectors()

    check(mapped is not None, "mapped read returned an index")
    check(
        isinstance(mapped, nsparse.SeismicIndex),
        "mapped read downcasts to SeismicIndex",
    )

    del mapped
    gc.collect()

    anon_before_copied = rss_anon_kb()
    started = time.perf_counter()
    copied = nsparse.read_index(index_path)
    copied_load_ms = (time.perf_counter() - started) * 1000.0
    anon_after_copied = rss_anon_kb()

    copied_scores, copied_labels, copied_search_ms = run_search(
        copied, queries, n_queries, opts.k, opts.cut, opts.heap_factor
    )

    check(
        copied.num_vectors() == mapped_vectors,
        f"both reads agree on num_vectors ({mapped_vectors})",
    )
    check(
        mapped_labels.shape == (n_queries, opts.k),
        f"labels have shape ({n_queries}, {opts.k}), got {mapped_labels.shape}",
    )
    mismatched = int(np.count_nonzero(mapped_labels != copied_labels))
    check(
        mismatched == 0,
        f"mapped and copied searches return identical labels "
        f"({mapped_labels.size} slots, {mismatched} mismatched)",
    )
    max_diff = (
        float(np.max(np.abs(mapped_scores - copied_scores)))
        if mapped_scores.size
        else 0.0
    )
    check(
        max_diff == 0.0,
        f"mapped and copied searches return identical scores (max diff {max_diff})",
    )

    # A read that silently returned an empty index would satisfy the comparisons
    # above, so confirm the searches actually hit.
    filled = int(np.count_nonzero(mapped_labels != -1))
    check(
        filled == mapped_labels.size,
        f"every top-k slot is filled ({filled}/{mapped_labels.size})",
    )
    check(
        bool(np.all((mapped_labels >= 0) & (mapped_labels < mapped_vectors))),
        "every returned label is a valid doc id",
    )

    print(f"\n  load    mapped {mapped_load_ms:.0f} ms vs copied {copied_load_ms:.0f} ms")
    print(
        f"  search  mapped {mapped_search_ms:.0f} ms vs copied {copied_search_ms:.0f} ms"
    )
    if anon_after_copied > 0:
        mapped_anon = anon_after_mapped - anon_before_mapped
        copied_anon = anon_after_copied - anon_before_copied
        print(
            f"  heap (RssAnon)   mapped +{mib(mapped_anon)} vs copied +{mib(copied_anon)}"
        )
        print(
            f"  mapped-in pages  +{mib(mapped_after_load - mapped_before)} at load, "
            f"+{mib(mapped_after_search - mapped_before)} after search "
            f"(they fault in as it touches them)"
        )
        if copied_anon >= MIN_MEASURABLE_KB:
            check(
                mapped_anon < copied_anon,
                f"the mapped load keeps the index off the heap: RssAnon grew "
                f"{mib(mapped_anon)} vs {mib(copied_anon)} for the copied load",
            )
        else:
            print(
                f"  index too small ({mib(copied_anon)} copied) for the heap "
                f"comparison to mean anything; reported, not checked"
            )
    else:
        print("  RssAnon unavailable on this platform; memory comparison skipped")

    del copied
    gc.collect()

    # -----------------------------------------------------------------------
    banner("read_csr(path, Residency_kMmap) vs read_csr(path, Residency_kInMemory)")
    # -----------------------------------------------------------------------
    native_csr = os.path.join(
        workdir, os.path.basename(opts.data_csr) + nsparse.kNativeSuffix
    )
    started = time.perf_counter()
    nsparse.convert(opts.data_csr, native_csr)
    print(
        f"  convert       {(time.perf_counter() - started) * 1000:.0f} ms, "
        f"{mib(os.path.getsize(native_csr) / 1024)}"
    )
    check(
        os.path.getsize(native_csr) == nsparse.native_file_size(rows + 1, nnz),
        "converted file matches the native layout size",
    )

    # Mapped first, on a clean heap, for the same reason as above.
    anon_before_map_csr = rss_anon_kb()
    mapped_csr = nsparse.index_factory(cols, descriptor)
    started = time.perf_counter()
    mapped_csr.read_csr(native_csr, nsparse.Residency_kMmap)
    map_csr_ms = (time.perf_counter() - started) * 1000.0
    anon_after_map_csr = rss_anon_kb()

    anon_before_mem_csr = rss_anon_kb()
    mem_csr = nsparse.index_factory(cols, descriptor)
    started = time.perf_counter()
    mem_csr.read_csr(opts.data_csr, nsparse.Residency_kInMemory)
    mem_csr_ms = (time.perf_counter() - started) * 1000.0
    anon_after_mem_csr = rss_anon_kb()

    check(
        mapped_csr.num_vectors() == mem_csr.num_vectors() == mapped_vectors,
        f"both residencies load {mapped_vectors} vectors, matching the "
        f"serialized index",
    )

    # Feeding the interchange file to the mapped reader would be the sharpest
    # check that the residency argument really reaches the mapped path -- it must
    # be rejected rather than silently copied. It is left out because the
    # bindings do not translate C++ exceptions: the throw escapes the wrapper and
    # aborts the interpreter instead of raising. The heap comparison below stands
    # in for it, since a silent fallback to copying would show in RssAnon.
    mem_anon = anon_after_mem_csr - anon_before_mem_csr
    map_anon = anon_after_map_csr - anon_before_map_csr
    if anon_after_mem_csr > 0:
        print(
            f"  heap (RssAnon)   mapped +{mib(map_anon)} vs in-memory +{mib(mem_anon)}"
        )
        if mem_anon >= MIN_MEASURABLE_KB:
            check(
                map_anon < mem_anon,
                f"the mapped read borrows the CSR instead of copying it: RssAnon "
                f"grew {mib(map_anon)} vs {mib(mem_anon)}",
            )
        else:
            print(
                f"  corpus too small ({mib(mem_anon)} copied) for the heap "
                f"comparison to mean anything; reported, not checked"
            )
    print(f"  read_csr  mapped {map_csr_ms:.0f} ms vs in-memory {mem_csr_ms:.0f} ms")

    del mem_csr
    gc.collect()

    # Clustering is randomly seeded, so this build's results cannot be compared
    # against phase one's. Build and search it to show a mapped CSR is usable.
    started = time.perf_counter()
    mapped_csr.build()
    print(f"  build (mapped CSR) {(time.perf_counter() - started) * 1000:.0f} ms")
    csr_scores, csr_labels, _ = run_search(
        mapped_csr, queries, n_queries, opts.k, opts.cut, opts.heap_factor
    )
    filled_csr = int(np.count_nonzero(csr_labels != -1))
    check(
        filled_csr == csr_labels.size,
        f"search over a mapped CSR fills every top-k slot "
        f"({filled_csr}/{csr_labels.size})",
    )
    check(
        bool(
            np.all((csr_labels >= 0) & (csr_labels < mapped_csr.num_vectors()))
            and np.all(csr_scores > 0)
        ),
        "search over a mapped CSR returns valid doc ids and non-zero scores",
    )
    del mapped_csr
    gc.collect()

    # -----------------------------------------------------------------------
    banner("Summary")
    # -----------------------------------------------------------------------
    top_k = min(3, opts.k)
    print(f"  query 0 top-{top_k}: labels={mapped_labels[0][:top_k]} "
          f"scores={np.round(mapped_scores[0][:top_k], 4)}")
    if peak_rss_kb() > 0:
        print(f"  peak RSS for the whole run: {mib(peak_rss_kb())}")

    if opts.keep:
        print(f"  kept {index_path}")
        print(f"  kept {native_csr}")
    else:
        os.remove(index_path)
        os.remove(native_csr)
        print("  removed the generated index and .mcsr (--keep to retain)")

    if failures == 0:
        print("\nPASS: all checks succeeded")
        return 0
    print(f"\nFAIL: {failures} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
