#!/usr/bin/env python3
"""Shrink a v2 fine-tuning cell's read-out to what its reader opens (storage
budget, 2026-10-01; scripts/build_ft_jobs.py V2_CELL_BYTES).

    leg1 --dir FEATURES --stride K
        logits.npy (500,000 x 162 float32, 324 MB) becomes
          logits_auc.npy  every K-th row, float32 -- the rows leg1_metrics.py
                          computes the macro AUC on (--auc-stride K), unchanged
          argmax.npy      every row's argmax, int16 -- the full-sample accuracy
          slim.json       the stride, the row count and the sha256 of the
                          logits it was cut from
        81 MB + 1 MB. Float32 is kept: a float16 copy would move the AUC.
    pred --file pred.root
        weaver's pred.root (2,000,000 jets, 155 MB) keeps only the Events
        branches leg2_metrics.py and experiments/STATS/paired_errors.py read:
        label_<class> and score_label_<class>, bit for bit (~80 MB). The ten jet
        observers weaver copies in are dropped.

Each step checks what it wrote against what it read before it deletes anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np


def slim_leg1(fd: pathlib.Path, stride: int) -> dict:
    src = fd / "logits.npy"
    logits = np.load(src)
    rec = {"auc_stride": stride, "n_rows": int(logits.shape[0]), "n_classes": int(logits.shape[1]),
           "logits_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    auc, top = logits[::stride], logits.argmax(1).astype(np.int16)
    np.save(fd / "logits_auc.npy", auc)
    np.save(fd / "argmax.npy", top)
    if not (np.array_equal(np.load(fd / "logits_auc.npy"), auc)
            and np.array_equal(np.load(fd / "argmax.npy"), top) and int(top.max()) < logits.shape[1]):
        raise SystemExit(f"FATAL: {fd}: the slim arrays do not read back as written")
    (fd / "slim.json").write_text(json.dumps(rec, indent=1))
    src.unlink()
    print(f"{fd}: logits {logits.shape} -> logits_auc {auc.shape} + argmax {top.shape}")
    return rec


def read_leg1(fd: pathlib.Path, stride: int):
    """(argmax of every row, logits of every stride-th row) from a full or a slim cache."""
    if (fd / "logits.npy").exists():
        logits = np.load(fd / "logits.npy")
        return logits.argmax(1), logits[::stride]
    rec = json.loads((fd / "slim.json").read_text())
    if rec["auc_stride"] != stride:
        raise SystemExit(f"FATAL: {fd} keeps the logits of every {rec['auc_stride']}th row, "
                         f"not every {stride}th")
    return np.load(fd / "argmax.npy").astype(np.int64), np.load(fd / "logits_auc.npy")


def slim_pred(path: pathlib.Path) -> list[str]:
    import uproot
    with uproot.open(path) as f:
        t = f["Events"]
        keep = [k for k in t.keys() if k.startswith(("label_", "score_label_"))]
        names = [k[len("label_"):] for k in keep if k.startswith("label_")]
        if not names or sorted(keep) != sorted([f"label_{n}" for n in names] + [f"score_label_{n}" for n in names]):
            raise SystemExit(f"FATAL: {path} lacks a score_label_ branch for every label_ branch")
        arrays = t.arrays(keep, library="np")
    tmp = path.with_name(path.name + ".slim")
    with uproot.recreate(tmp) as f:
        f["Events"] = {k: arrays[k] for k in keep}
    with uproot.open(tmp) as f:
        back = f["Events"].arrays(keep, library="np")
    if any(not np.array_equal(back[k], arrays[k]) for k in keep):
        raise SystemExit(f"FATAL: {tmp} does not read back as written")
    tmp.replace(path)
    print(f"{path}: kept {len(keep)} branches ({len(names)} classes)")
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("leg1")
    a.add_argument("--dir", required=True, type=pathlib.Path)
    a.add_argument("--stride", required=True, type=int)
    b = sub.add_parser("pred")
    b.add_argument("--file", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.cmd == "leg1":
        slim_leg1(args.dir, args.stride)
    else:
        slim_pred(args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
