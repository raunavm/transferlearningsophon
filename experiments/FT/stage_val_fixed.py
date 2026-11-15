#!/usr/bin/env python3
"""val_fixed.parquet: the one validation set size of every fine-tuning (PI 2026-11-15).

The image's weaver validates in the training batch (512) and, with --in-memory, scores the same jets at every check
only if the set is a whole number of batches. So every task validates on N = 390 x 512 = 199,680 jets, the largest
multiple of the batch up to 200,000: this script takes a task's validation file (>= N jets) and removes the surplus
from the most populous classes first (one jet at a time from the largest class, ties broken at random), so the class
counts stay as even as the file allows, and keeps the rows' order. Seeded; refuses to overwrite.

    python3 experiments/FT/stage_val_fixed.py --src /data/finetune/top_sub/val.parquet --label label \
        --out /data/finetune/top_sub/val_fixed.parquet
    --label: a column of class indices (label, jet_label) or onehot:<col>,<col>,... (JetClass's label_* columns)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

N_FIXED, SEED = 390 * 512, 20261115


def target_counts(counts: np.ndarray, n: int, r: np.random.Generator) -> np.ndarray:
    """counts lowered to sum n, always taking from the largest class (ties at random)."""
    c = counts.astype(np.int64).copy()
    if c.sum() < n:
        raise SystemExit(f"FATAL: {c.sum()} jets for a validation set of {n}")
    while c.sum() > n:
        top = np.flatnonzero(c == c.max())
        take = min(len(top), int(c.sum() - n))
        c[r.choice(top, size=take, replace=False)] -= 1
    return c


def labels(table, spec: str) -> np.ndarray:
    if spec.startswith("onehot:"):
        cols = spec[len("onehot:"):].split(",")
        m = np.stack([table.column(k).to_numpy().astype(np.float64) for k in cols], axis=1)
        if not np.all(m.sum(axis=1) == 1):
            raise SystemExit("FATAL: the one-hot label columns are not one-hot")
        return m.argmax(axis=1)
    return table.column(spec).to_numpy().astype(np.int64)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--n", type=int, default=N_FIXED)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; staged data is never rebuilt")
    table = pq.read_table(a.src)
    lab = labels(table, a.label)
    classes, inv = np.unique(lab, return_inverse=True)
    r = np.random.default_rng(SEED)
    want = target_counts(np.bincount(inv), a.n, r)
    keep = np.zeros(len(lab), bool)
    for k, w in enumerate(want):
        rows = np.flatnonzero(inv == k)
        keep[np.sort(r.choice(rows, size=int(w), replace=False))] = True
    out = table.filter(pa.array(keep))
    tmp = a.out.with_suffix(".tmp")
    pq.write_table(out, tmp, row_group_size=50_000)
    tmp.rename(a.out)
    rec = {"script": "experiments/FT/stage_val_fixed.py", "seed": SEED, "rows": int(out.num_rows), "src": str(a.src),
           "src_rows": int(table.num_rows), "src_sha256": sha256(a.src), "sha256": sha256(a.out), "label": a.label,
           "counts": {str(c): int(w) for c, w in zip(classes, want)},
           "removed_rows": np.flatnonzero(~keep).tolist()}
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{a.out}: {out.num_rows:,} of {table.num_rows:,} jets, classes {len(classes)}, "
          f"min {want.min()} max {want.max()} per class", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
