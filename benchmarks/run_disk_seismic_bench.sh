#!/usr/bin/env bash
#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained DiskSeismic C++ benchmark: compares three sparse-index residency
# regimes on MS MARCO under a sweep of RAM caps, single-thread query.
#
#   V1  in-RAM SEISMIC        seismic .dat, inmem   (heap copy, SEISMIC heap_factor)
#   V2  scattered-disk SEISMIC seismic .dat, mmap   (mapped forward index, SEISMIC heap_factor)
#   V3  DiskSeismic           disk_seismic .dat, mmap (inline forward index, GroC top-k')
#
# The point of the RAM-cap sweep: below the index size, V1 (heap) is OOM-killed
# while the mmap variants keep serving by paging from disk. That contrast — and
# how V2/V3 latency degrades as RAM shrinks — is the result.
#
# Prereqs on the host: gcc10 (C++20), cmake3, and MS MARCO CSR data. For the cap
# sweep, systemd (systemd-run --scope) and permission to set MemoryMax; dropping
# the page cache between runs needs root (falls back to sudo, else warm-cache).
#
# Usage:
#   NSPARSE_DATA_DIR=/data ./benchmarks/run_disk_seismic_bench.sh
# Env knobs (defaults): OPT_LEVEL=avx2  LAMBDA=6000 BETA=400 ALPHA=0.4
#   CUT=3 KPRIME=50 K=10 REPS=5  CAPS="unlimited 8G 4G 2G"
set -euo pipefail

: "${NSPARSE_DATA_DIR:?set NSPARSE_DATA_DIR to a dir with base_full.csr, queries.dev.csr, iv_array.txt}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$NSPARSE_DATA_DIR"
CORPUS="$DATA/base_full.csr"; Q="$DATA/queries.dev.csr"; TRUTH="$DATA/iv_array.txt"
for f in "$CORPUS" "$Q" "$TRUTH"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done

# g5.12xlarge is AMD Zen2 (EPYC 7R32): NO AVX-512. avx2 is correct; avx512 would not run.
OPT_LEVEL="${OPT_LEVEL:-avx2}"
CMAKE="${CMAKE:-cmake3}"; CXX="${CXX:-/usr/bin/gcc10-g++}"; CC="${CC:-/usr/bin/gcc10-gcc}"
LAMBDA="${LAMBDA:-6000}"; BETA="${BETA:-400}"; ALPHA="${ALPHA:-0.4}"
CUT="${CUT:-3}"; KPRIME="${KPRIME:-50}"; K="${K:-10}"; REPS="${REPS:-5}"
CAPS="${CAPS:-unlimited 8G 4G 2G}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-bench}"
OUT="${OUT:-$DATA/disk_seismic_bench_out}"; mkdir -p "$OUT"

echo "=== 1. build (Release, $OPT_LEVEL) ==="
"$CMAKE" -S "$REPO" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DNSPARSE_OPT_LEVEL="$OPT_LEVEL" \
  -DNSPARSE_ENABLE_BENCHMARKS=ON -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" >"$OUT/configure.log" 2>&1
"$CMAKE" --build "$BUILD_DIR" --target sq_residency_bench -j"$(nproc)" >"$OUT/build.log" 2>&1
BIN="$BUILD_DIR/benchmarks/sq_residency_bench"
[ -x "$BIN" ] || { echo "build failed; see $OUT/build.log"; exit 1; }
grep -q "CMAKE_BUILD_TYPE:STRING=Release" "$BUILD_DIR/CMakeCache.txt" || { echo "NOT a Release build"; exit 1; }

echo "=== 2. build indexes from base_full (all cores; reused if present) ==="
SEISMIC_DAT="$OUT/base_full.seismic.dat"; DISK_DAT="$OUT/base_full.disk_seismic.dat"
[ -f "$SEISMIC_DAT" ] || "$BIN" build "$CORPUS" "seismic,lambda=$LAMBDA|beta=$BETA|alpha=$ALPHA" "$SEISMIC_DAT"
[ -f "$DISK_DAT" ]    || "$BIN" build "$CORPUS" "disk_seismic,lambda=$LAMBDA|beta=$BETA|alpha=$ALPHA" "$DISK_DAT"

drop_caches() {
  sync
  if [ -w /proc/sys/vm/drop_caches ]; then echo 3 >/proc/sys/vm/drop_caches
  else sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches' 2>/dev/null || echo "  (warn: could not drop page cache; run as root for cold-cache numbers)"; fi
}

# systemd renamed the cgroup memory knob at v231 (MemoryMax) from v219 (MemoryLimit);
# MemorySwapMax also only exists on the newer line. EC2 hosts have no swap by default,
# so the limit is hard either way. Running as root avoids cgroup-delegation issues.
SYSTEMD_VER="$(systemctl --version 2>/dev/null | head -1 | awk '{print $2}')"
mem_props() {
  if [ "${SYSTEMD_VER:-0}" -ge 231 ] 2>/dev/null; then echo "-p MemoryMax=$1 -p MemorySwapMax=0"
  else echo "-p MemoryLimit=$1"; fi
}

# run one search, optionally inside a memory-capped transient scope
run_capped() {
  local cap="$1" logf="$2"; shift 2
  drop_caches
  if [ "$cap" = "unlimited" ]; then
    OMP_NUM_THREADS=1 "$@" >"$logf" 2>&1 || echo "RUN_FAILED rc=$?" >>"$logf"
  else
    # shellcheck disable=SC2046
    systemd-run --scope -q $(mem_props "$cap") \
      env OMP_NUM_THREADS=1 "$@" >"$logf" 2>&1 || echo "OOM_OR_FAIL rc=$? cap=$cap" >>"$logf"
  fi
}

echo "=== 3. sweep caps x variants (single-thread) ==="
for cap in $CAPS; do
  echo "  cap=$cap ..."
  run_capped "$cap" "$OUT/v1_inram.$cap.log"       "$BIN" search "$SEISMIC_DAT" "$Q" "$K" "$REPS" inmem "$TRUTH" "$CUT"
  run_capped "$cap" "$OUT/v2_scattered.$cap.log"   "$BIN" search "$SEISMIC_DAT" "$Q" "$K" "$REPS" mmap  "$TRUTH" "$CUT"
  run_capped "$cap" "$OUT/v3_diskseismic.$cap.log" "$BIN" search "$DISK_DAT"    "$Q" "$K" "$REPS" mmap  "$TRUTH" "$CUT" "$KPRIME"
done

echo "=== 4. summary (also written to $OUT/summary.tsv) ==="
python3 - "$OUT" <<'PY' | tee "$OUT/summary.tsv"
import os, re, sys, glob
outdir = sys.argv[1]
def grab(txt, pat):
    m = re.search(pat, txt); return m.group(1) if m else ""
rows = []
for f in sorted(glob.glob(os.path.join(outdir, "v*.log"))):
    base = os.path.basename(f)[:-4]              # v1_inram.8G
    variant, cap = base.split(".", 1)
    txt = open(f).read()
    if "OOM_OR_FAIL" in txt or "RUN_FAILED" in txt or "error:" in txt:
        rows.append((variant, cap, "OOM/FAIL", "", "", "", "", "", "")); continue
    p50 = grab(txt, r"p50 ([\d.]+)"); p90 = grab(txt, r"p90 ([\d.]+)"); p99 = grab(txt, r"p99 ([\d.]+)")
    qps = grab(txt, r"qps ([\d.]+)")             # last batch qps
    rec = grab(txt, r"recall@\d+ ([\d.]+)")
    # memory from the after_search line
    aft = txt.split("mem[after_search]")[-1] if "after_search" in txt else ""
    anon = grab(aft, r"RssAnon: ([\d.]+)"); rfile = grab(aft, r"RssFile: ([\d.]+)")
    rows.append((variant, cap, "ok", p50, p90, p99, qps, rec, f"{anon}/{rfile}"))
print("variant\tcap\tstatus\tp50_ms\tp90_ms\tp99_ms\tqps\trecall\tRssAnon/File_GiB")
for r in rows: print("\t".join(map(str, r)))
PY
echo "done. logs + summary in $OUT"
