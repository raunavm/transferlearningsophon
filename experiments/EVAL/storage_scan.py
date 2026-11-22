#!/usr/bin/env python3
"""Every file written during the storage faults of 17-19 Nov 2026, loaded in full with its own reader (PI 2026-11-21:
zero_fill_scan.py read the first 4 KB only and misses a file that lost its tail). Reads only; writes the report.

  load    every regular file under ROOTS modified 2026-11-17 00:00 to 2026-11-20 00:00 UTC:
            .json  json.load              .npz  zipfile testzip (every member's CRC), then every array loaded
            .npy   np.load in full        .pt/.pth/.ckpt  torch.load on the CPU
            .parquet  every row group     .root  every branch of every tree (uproot)
            .csv/.tsv  csv parse          anything else  read in full
          and, for every type, a file that is all NUL or whose last 4 KB are NUL is listed as zero-filled
  hashes  every (file, sha256) a manifest records: in a JSON (in the window, or named *manifest*) a key K_sha256 beside
          K naming an absolute path, "sha256" beside "file"/"path"/"from"/"src", "from_sha256" beside "from", or a
          {path: sha256} map; in a CSV/TSV named *manifest* (or MANIFEST.tsv) a sha256 column beside a path/file
          column. Each existing file is re-hashed once; a mismatch is listed with the manifest, and as "rewritten
          after its record" when the file is newer than the manifest

    python3 experiments/EVAL/storage_scan.py --out /data/results/eval/storage_scan_1117_1119
    python3 experiments/EVAL/storage_scan.py --since 2026-11-23T01:00:00Z --until 2026-11-23T02:30:00Z \
        --roots /data/results --no-records --out /data/results/eval/storage_scan_window/<since>_<until>
                                               # a storage failure's window (guard v7, PI 2026-11-22): the load only
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

ROOTS = ("/data",)
PREFIX = "/data/"                                  # a recorded path outside it is not ours to hash
SKIP = ("/data/transferlearningsophon",)             # a git checkout
LO, HI = calendar.timegm((2026, 11, 17, 0, 0, 0)), calendar.timegm((2026, 11, 20, 0, 0, 0))
TAIL = 4096
HEX = re.compile(r"^[0-9a-f]{64}$")
PATH_KEYS = ("file", "path", "from", "src")
BIG = 2 << 30          # an npz member above this is checked by its CRC and header only


def zero_filled(p: str, size: int) -> str | None:
    with open(p, "rb") as fh:
        fh.seek(max(0, size - TAIL))
        tail = fh.read()
    if tail and tail.count(0) == len(tail):
        with open(p, "rb") as fh:
            whole = all(b.count(0) == len(b) for b in iter(lambda: fh.read(1 << 24), b""))
        return "all NUL" if whole else f"last {len(tail)} bytes NUL"
    return None


def load(p: str) -> None:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".json":
        with open(p) as fh:
            json.load(fh)
    elif ext == ".npz":
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            if bad is not None:
                raise ValueError(f"CRC fails for member {bad}")
            for info in z.infolist():
                if info.filename.endswith(".npy") and info.file_size <= BIG:   # a bigger one: its CRC above
                    with z.open(info) as fh:
                        np.lib.format.read_array(fh, allow_pickle=False)
    elif ext == ".npy":
        a = np.load(p, mmap_mode="r", allow_pickle=False).reshape(-1)    # a short file fails the mapping
        step = max(1, (1 << 27) // max(1, a.itemsize))
        for i in range(0, a.size, step):
            np.asarray(a[i:i + step]).tobytes()
    elif ext in (".pt", ".pth", ".ckpt"):
        import torch
        torch.load(p, map_location="cpu", weights_only=False)
    elif ext == ".parquet":
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        for i in range(f.num_row_groups):
            f.read_row_group(i)
    elif ext == ".root":
        import uproot
        with uproot.open(p) as f:
            for k, cls in f.classnames().items():
                if cls == "TTree":
                    for _ in f[k].iterate(step_size="200 MB", library="np"):
                        pass
    elif ext in (".csv", ".tsv"):
        with open(p, newline="") as fh:
            for _ in csv.reader(fh, delimiter="\t" if ext == ".tsv" else ","):
                pass
    else:
        with open(p, "rb") as fh:
            while fh.read(1 << 24):
                pass


def json_records(x, out: list) -> None:
    """(path, sha256) pairs a JSON document records, by the rules of the module docstring."""
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, str) and HEX.match(v):
                if k.endswith("_sha256") and isinstance(x.get(k[:-7]), str):
                    out.append((x[k[:-7]], v))
                elif k == "sha256":
                    out += [(x[q], v) for q in PATH_KEYS if isinstance(x.get(q), str)][:1]
                elif k.startswith("/"):
                    out.append((k, v))
            else:
                json_records(v, out)
    elif isinstance(x, list):
        for v in x:
            json_records(v, out)


def table_records(p: str) -> list:
    with open(p, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t" if p.endswith(".tsv") else ","))
    if not rows:
        return []
    head = [h.strip().lower() for h in rows[0]]
    if "sha256" in head:
        col = next((head.index(c) for c in ("path", "file", "src") if c in head), None)
        if col is None:
            return []
        s = head.index("sha256")
        return [(r[col], r[s]) for r in rows[1:] if len(r) > max(col, s) and HEX.match(r[s])]
    if os.path.basename(p) == "MANIFEST.tsv":       # build_refit_v5_jobs.ASIDE: name, sha256, cpu, cpu, time
        d = os.path.dirname(p)
        return [(os.path.join(d, r[0]), r[1]) for r in rows if len(r) > 1 and HEX.match(r[1])]
    return []


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--records-only", action="store_true", help="skip the load; re-hash the recorded files and list the "
                    "absent ones (absent.tsv)")
    ap.add_argument("--since", help="window start, UTC YYYY-MM-DDTHH:MM:SSZ (default 2026-11-17T00:00:00Z)")
    ap.add_argument("--until", help="window end, UTC (default 2026-11-20T00:00:00Z)")
    ap.add_argument("--roots", help="comma-separated folders to walk (default /data)")
    ap.add_argument("--no-records", action="store_true", help="the load only: no manifest's recorded hashes")
    a = ap.parse_args(argv)
    global LO, HI, ROOTS
    utc = lambda t: calendar.timegm(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ"))      # noqa: E731
    LO, HI = (utc(a.since) if a.since else LO), (utc(a.until) if a.until else HI)
    ROOTS = tuple(a.roots.split(",")) if a.roots else ROOTS
    os.makedirs(a.out, exist_ok=True)
    fails, n, nbytes, records = [], 0, 0, []
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            if dp.startswith(SKIP):
                dn[:] = []
                continue
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                is_manifest = "manifest" in f.lower() or f == "MANIFEST.tsv"
                inwin = LO <= st.st_mtime < HI and st.st_size > 0
                if inwin and not a.records_only:
                    n, nbytes = n + 1, nbytes + st.st_size
                    try:
                        z = zero_filled(p, st.st_size)
                        if z:
                            fails.append((p, st.st_size, "zero-filled", z))
                        load(p)
                    except Exception as e:                     # noqa: BLE001 -- every failure is the result
                        fails.append((p, st.st_size, "load", f"{type(e).__name__}: {e}"[:300]))
                    if n % 2000 == 0:
                        print(f"{n} files, {nbytes / 1e9:.1f} GB loaded; {len(fails)} failures", flush=True)
                if (inwin or is_manifest) and st.st_size > 0 and not a.no_records:
                    try:
                        if f.endswith(".json"):
                            got = []
                            with open(p) as fh:
                                json_records(json.load(fh), got)
                            records += [(q, h, p, st.st_mtime) for q, h in got if q.startswith(PREFIX)]
                        elif f.endswith((".csv", ".tsv")) and is_manifest:
                            records += [(q, h, p, st.st_mtime) for q, h in table_records(p) if q.startswith(PREFIX)]
                    except Exception:                          # noqa: BLE001 -- a broken manifest is listed by load
                        pass
    print(f"loaded {n} files, {nbytes / 1e9:.1f} GB; {len(fails)} failures; {len(records)} recorded hashes", flush=True)
    seen, rehashed, missing, absent = {}, 0, 0, []
    for q, h, man, mt in records:
        if not os.path.isfile(q):
            missing += 1
            absent.append((q, h, man))
            continue
        if q not in seen:
            seen[q] = sha256(q)
            rehashed += 1
        if seen[q] != h:
            why = "rewritten after its record" if os.stat(q).st_mtime > mt else "hash differs"
            fails.append((q, os.stat(q).st_size, why, f"recorded {h[:16]} in {man}, now {seen[q][:16]}"))
    with open(os.path.join(a.out, "report.tsv"), "w") as fh:
        fh.write("path\tbytes\tkind\tdetail\n")
        fh.writelines(f"{p}\t{s}\t{k}\t{d}\n" for p, s, k, d in fails)
    summary = {"window_utc": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) for t in (LO, HI)], "roots": ROOTS,
               "files_loaded": n,
               "bytes_loaded": nbytes, "recorded_hashes": len(records), "files_rehashed": rehashed,
               "recorded_but_absent": missing, "failures": {k: sum(1 for f in fails if f[2] == k)
                                                            for k in sorted({f[2] for f in fails})}}
    with open(os.path.join(a.out, "absent.tsv"), "w") as fh:     # a deletion manifest's own rows are expected here
        fh.write("path\trecorded_sha256\tmanifest\n")
        fh.writelines(f"{q}\t{h}\t{m}\n" for q, h, m in sorted(set(absent)))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    for p, s, k, d in fails:
        print(f"{k}\t{p}\t{s}\t{d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
