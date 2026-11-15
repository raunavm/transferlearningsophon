#!/usr/bin/env python3
"""The one-time removal of superseded fine-tuning cells' features.npy (PI 2026-11-15, option A of the disk decision).

    manifest  every features.npy under the superseded trees: path, size, sha256, the checkpoint its
              extract_manifest.json says regenerates it, that checkpoint's recorded and actual sha256; writes the CSV
              and refuses to overwrite one
    delete    removes exactly the CSV's rows whose regenerating checkpoint exists with its recorded sha256, after
              re-hashing each file (a file that changed since the manifest is kept); appends what it did to a log

Nothing reads these files (every read-out takes logits, labels and observers; the fine-tuned embeddings re-extract
from the checkpoint). Checkpoints, logits, manifests and logs stay.

    python3 experiments/FT/features_manifest.py manifest --out /data/results/deleted_features_manifest.csv
    python3 experiments/FT/features_manifest.py delete --manifest /data/results/deleted_features_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

FT = pathlib.Path("/data/results/ft_v2")
SUPERSEDED = ("budget", "budget_long", "budget_fulldecay", "budget_long_fulldecay", "budget_prefull", "lrsweep",
              "lrsweep_v2")
THREADS = 8
FIELDS = ("path", "size_bytes", "sha256", "checkpoint", "checkpoint_sha256_recorded", "checkpoint_sha256",
          "regenerable", "extract_manifest")


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def manifest(out: pathlib.Path) -> int:
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists")
    files = [f for tree in SUPERSEDED for f in sorted((FT / tree).rglob("features.npy"))]
    ems = {f: f.parent / "extract_manifest.json" for f in files}
    meta = {f: json.loads(e.read_text()) if e.exists() else {} for f, e in ems.items()}
    cks = sorted({m.get("checkpoint", "") for m in meta.values()} - {""})
    with ThreadPoolExecutor(THREADS) as ex:    # the shared volume reads one stream slowly; hashlib frees the GIL
        ck_hash = dict(zip(cks, ex.map(lambda c: sha256(pathlib.Path(c)) if pathlib.Path(c).exists() else "", cks)))
        f_hash = dict(zip(files, ex.map(sha256, files)))
    rows = []
    for f in files:
        m, ck = meta[f], meta[f].get("checkpoint", "")
        actual = ck_hash.get(ck, "")
        rows.append({"path": str(f), "size_bytes": f.stat().st_size, "sha256": f_hash[f], "checkpoint": ck,
                     "checkpoint_sha256_recorded": m.get("checkpoint_sha256", ""), "checkpoint_sha256": actual,
                     "regenerable": int(bool(ck) and actual != "" and actual == m.get("checkpoint_sha256")),
                     "extract_manifest": str(ems[f]) if ems[f].exists() else ""})
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(out)
    ok = [r for r in rows if r["regenerable"]]
    print(f"{out}: {len(rows)} files, {sum(r['size_bytes'] for r in rows) / 1e9:.1f} GB; regenerable "
          f"{len(ok)} ({sum(r['size_bytes'] for r in ok) / 1e9:.1f} GB)")
    return 0


def delete(man: pathlib.Path) -> int:
    log = man.with_suffix(".deleted.log")
    rows = list(csv.DictReader(open(man)))
    freed, kept = 0, 0
    for r in rows:
        f = pathlib.Path(r["path"])
        if not f.is_relative_to(FT) or f.name != "features.npy" or f.relative_to(FT).parts[0] not in SUPERSEDED:
            raise SystemExit(f"FATAL: {f} is not a superseded cell's features.npy")
    with ThreadPoolExecutor(THREADS) as ex:
        now = list(ex.map(lambda r: sha256(pathlib.Path(r["path"])) if pathlib.Path(r["path"]).exists() else "", rows))
    with open(log, "a") as lg:
        for r, h in zip(rows, now):
            f = pathlib.Path(r["path"])
            if r["regenerable"] != "1" or h != r["sha256"]:
                kept += 1
                lg.write(f"kept {f}\n")
                continue
            f.unlink()
            freed += int(r["size_bytes"])
            lg.write(f"deleted {f} {r['sha256']}\n")
    print(f"deleted {len(rows) - kept} files, {freed / 1e9:.1f} GB; kept {kept}; log {log}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("manifest", "delete"))
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--manifest", type=pathlib.Path)
    a = ap.parse_args(argv)
    return manifest(a.out) if a.mode == "manifest" else delete(a.manifest)


if __name__ == "__main__":
    sys.exit(main())
