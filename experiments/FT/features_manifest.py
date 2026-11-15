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

FT = pathlib.Path("/data/results/ft_v2")
SUPERSEDED = ("budget", "budget_long", "budget_fulldecay", "budget_long_fulldecay", "budget_prefull", "lrsweep",
              "lrsweep_v2")
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
    rows, ck_cache = [], {}
    for tree in SUPERSEDED:
        for f in sorted((FT / tree).rglob("features.npy")):
            em = f.parent / "extract_manifest.json"
            m = json.loads(em.read_text()) if em.exists() else {}
            ck = m.get("checkpoint", "")
            if ck and ck not in ck_cache:
                ck_cache[ck] = sha256(pathlib.Path(ck)) if pathlib.Path(ck).exists() else ""
            actual = ck_cache.get(ck, "")
            rows.append({"path": str(f), "size_bytes": f.stat().st_size, "sha256": sha256(f), "checkpoint": ck,
                         "checkpoint_sha256_recorded": m.get("checkpoint_sha256", ""), "checkpoint_sha256": actual,
                         "regenerable": int(bool(ck) and actual != "" and actual == m.get("checkpoint_sha256")),
                         "extract_manifest": str(em) if em.exists() else ""})
            print(len(rows), f, rows[-1]["regenerable"], flush=True)
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
    with open(log, "a") as lg:
        for r in rows:
            f = pathlib.Path(r["path"])
            if not f.is_relative_to(FT) or f.name != "features.npy" or f.parts[4] not in SUPERSEDED:
                raise SystemExit(f"FATAL: {f} is not a superseded cell's features.npy")
            if r["regenerable"] != "1" or not f.exists() or sha256(f) != r["sha256"]:
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
