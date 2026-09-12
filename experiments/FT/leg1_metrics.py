#!/usr/bin/env python3
"""Leg 1 of the fine-tuning legs: the IN-DOMAIN 162-way recovery curve.

WHAT THIS IS AND WHY IT NEEDED WRITING. job-ft-legs produced two legs. Leg 2
fine-tunes on JetClass-I and weaver prints a "Test metric" per cell, so its
curve could be read straight out of predict.log. Leg 1 fine-tunes on
JetClass-II's own 162-way vocabulary and writes a feature cache instead --
features.npy, logits.npy, label188.npy per cell -- with no metric computed by
anything in this repo. 54 cells of finished inference sat unread. This is the
readout.

THE MEASUREMENT. For each (init, N, fine-tuning seed) cell, softmax the cached
162-way logits, map the native 188 labels down to the L162 rung, and report
accuracy and macro one-vs-rest AUC. Aggregated over fine-tuning seeds this is
the in-domain half of the transfer claim; leg 2 is the domain-shifted half.

METRICS ARE NOT REDEFINED HERE. accuracy / macro_auc_ovr / per_class_auc /
rejection-vs-QCD come from experiments/EVAL/eval_arm.py's metrics(), the same
function the E1 gate was measured with, and the rung mapping comes from
label_recovery.rung_maps(), which reads the committed
configs/labelmaps/rung_label_maps.v1.csv. Two metric definitions for one
quantity is how two papers' numbers stop being comparable.

WHY THE AUC IS COMPUTED ON A STRIDE AND THE ACCURACY IS NOT. Accuracy is an
argmax and costs nothing at 2 M x 162. Macro OvR AUC is 162 ROC curves over 2 M
points, and eval_arm.metrics() additionally computes per-class AUC, so a full
pass is ~2 x 162 sorts of 2 M rows per cell across 54 cells. The subsample is a
STRIDE, `arange(0, n, step)`, not a random draw:

  - it is deterministic, so the number is reproducible without recording a seed;
  - it takes the SAME rows in every cell, so cells stay paired -- the whole
    point of the comparison is between inits at fixed N, and a per-cell random
    draw would put sampling noise inside that contrast;
  - it preserves the class mix, which taking the first n rows would not. The
    test list interleaves Res2P / Res34P / QCD by file, so a head slice is
    biased toward whichever files come first.

Pass --auc-stride 1 to compute the AUC on every row.

ROW ALIGNMENT IS VERIFIED, NOT ASSUMED. Every cell must have read out the same
test jets in the same order or the between-init comparison is not paired. The
sha256 of label188.npy is checked across cells and a mismatch is fatal, which is
the same guard probe.py's check_alignment applies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
L162_QCD_GROUP = 161   # asserted against the committed map at runtime


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def discover(root: pathlib.Path) -> list[tuple[str, str, str, pathlib.Path]]:
    """(init, N, seed, features_dir) for every complete cell, sorted.

    A `.partial.<timestamp>` directory is an interrupted attempt that a later
    attempt superseded; ft-legs leaves them beside the finished cell on purpose.
    They are skipped by name rather than by checking for files, because a
    partial cell can contain a complete-looking features.npy from before the
    interruption.
    """
    out = []
    for init_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for n_dir in sorted(p for p in init_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in n_dir.iterdir() if p.is_dir()):
                if ".partial." in seed_dir.name:
                    continue
                fd = seed_dir / "features_v2"
                if (fd / "logits.npy").exists() and (fd / "label188.npy").exists():
                    out.append((init_dir.name, n_dir.name, seed_dir.name, fd))
    return out


def cell_metrics(fd: pathlib.Path, l162: dict[int, int], eval_arm, auc_stride: int):
    lab188 = np.load(fd / "label188.npy")
    sha = hashlib.sha256(lab188.tobytes()).hexdigest()
    logits = np.load(fd / "logits.npy")
    if logits.shape[0] != lab188.shape[0]:
        raise SystemExit(f"FATAL: {fd} has {logits.shape[0]} logits rows and "
                         f"{lab188.shape[0]} labels")
    if logits.shape[1] != 162:
        raise SystemExit(f"FATAL: {fd} head is {logits.shape[1]}-wide, not 162. "
                         "Leg 1 fine-tunes the 162-way vocabulary; a different "
                         "width means this cell is not what it claims to be.")
    # Size the table to cover BOTH the map and what the file actually holds.
    # Sizing it to the map alone makes an out-of-range label a raw IndexError
    # from numpy rather than the named failure below -- which is the same class
    # of defect as letting it through, since neither tells the reader that the
    # cache and the committed map disagree.
    if lab188.min() < 0:
        raise SystemExit(f"FATAL: {fd} holds negative native labels "
                         f"(min {int(lab188.min())})")
    lut = np.full(max(int(max(l162)), int(lab188.max())) + 1, -1, dtype=np.int64)
    for k, v in l162.items():
        lut[k] = v
    truth = lut[lab188]
    if (truth < 0).any():
        bad = np.unique(lab188[truth < 0])[:5]
        raise SystemExit(f"FATAL: {fd} holds native labels absent from the "
                         f"committed map: {bad.tolist()}")

    acc = float((logits.argmax(1) == truth).mean())      # full sample, cheap
    idx = np.arange(0, truth.shape[0], auc_stride)
    m = eval_arm.metrics(softmax(logits[idx]), truth[idx], 162, L162_QCD_GROUP)
    return {
        "accuracy": acc,
        "accuracy_on_auc_subsample": m["accuracy"],
        "macro_auc_ovr": m["macro_auc_ovr"],
        "n_classes_present": m["n_classes_present"],
        "n_jets": int(truth.shape[0]),
        "n_jets_auc": int(idx.size),
        "auc_stride": auc_stride,
        "label188_sha256": sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--auc-stride", type=int, default=4,
                    help="compute the AUC on every k-th row (1 = every row)")
    args = ap.parse_args()

    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    eval_arm = _load("eval_arm", "experiments/EVAL/eval_arm.py")
    l162 = lr.rung_maps()["L162"]
    groups = sorted(set(l162.values()))
    if groups != list(range(162)):
        raise SystemExit(f"FATAL: L162 has {len(groups)} groups, expected 162")
    qcd_members = [k for k, v in l162.items() if v == L162_QCD_GROUP]
    if len(qcd_members) != 27:
        raise SystemExit(f"FATAL: L162 group {L162_QCD_GROUP} has "
                         f"{len(qcd_members)} members, expected the 27 QCD "
                         "classes -- the QCD group id moved")

    cells = discover(args.root)
    if not cells:
        raise SystemExit(f"FATAL: no complete cells under {args.root}")
    print(f"{len(cells)} cells", flush=True)

    res, shas = {}, {}
    for init, n, seed, fd in cells:
        m = cell_metrics(fd, l162, eval_arm, args.auc_stride)
        shas.setdefault(m["label188_sha256"], []).append(f"{init}/{n}/{seed}")
        res.setdefault(init, {}).setdefault(n, {})[seed] = m
        print(f"  {init:16} {n:10} {seed:4} acc={m['accuracy']:.5f} "
              f"macroAUC={m['macro_auc_ovr']:.5f}", flush=True)

    if len(shas) != 1:
        for s, where in shas.items():
            print(f"  {s[:16]}  {len(where)} cells, e.g. {where[:3]}")
        raise SystemExit("FATAL: cells did not read out the same test jets in "
                         "the same order; the between-init contrast is not "
                         "paired and the table would be meaningless")

    summary = {}
    for init, per_n in res.items():
        for n, per_seed in per_n.items():
            a = np.array([v["accuracy"] for v in per_seed.values()])
            u = np.array([v["macro_auc_ovr"] for v in per_seed.values()])
            summary.setdefault(init, {})[n] = {
                "n_seeds": int(a.size),
                "accuracy_mean": float(a.mean()),
                "accuracy_sd": float(a.std(ddof=1)) if a.size > 1 else None,
                "macro_auc_mean": float(u.mean()),
                "macro_auc_sd": float(u.std(ddof=1)) if u.size > 1 else None,
            }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leg1_metrics.json").write_text(json.dumps(
        {"row_alignment_sha256": next(iter(shas)),
         "auc_stride": args.auc_stride,
         "cells": res, "summary": summary}, indent=1))
    print(f"\nwrote {args.out / 'leg1_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
