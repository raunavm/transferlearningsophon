#!/usr/bin/env python3
"""Read-out of the fixed-budget fine-tuning cells (scripts/build_ft_budget_jobs.py, plan 3.1/3.4).

Walks <root>/<tree>/<init>/N<N>/s<S>/ for every cell with DONE and scores it:
  leg_top, leg_qg   bench_metrics.cell_metrics on features/logits.npy (accuracy, AUC, log(1-AUC),
                    Rej50%, Rej30%), and on Herwig (logits_herwig.npy) for quark/gluon
  leg2 (JetClass)   from pred.root: accuracy, macro one-vs-rest AUC on every test jet, ParT's macro
                    one-vs-one AUC, and each class's AUC and rejection of QCD at ParT's working point
  leg_jc2b          from the 188 logits on every jet of the balanced test set: accuracy, the macro AUC
                    over all classes, the resonant and the QCD classes, each class's AUC and each
                    resonant class's rejection of summed QCD (linear_probe_v2.native_metrics)
Every cell also carries its early_stop.json (best epoch, validation metric, epochs run, stopped early)
and its manifest's recipe. A cell without DONE is listed as missing; the read-out never fills it.

    python3 experiments/FT/budget_metrics.py --root /data/results/ft_v2/budget --out budget_metrics.json
    python3 experiments/FT/budget_metrics.py --root /data/results/ft_v2/lrsweep --out lrsweep_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
JC1_CLASSES = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql",
               "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def jc1_metrics(cell: pathlib.Path, lp) -> dict:
    import uproot
    t = uproot.open(cell / "pred.root")["Events"]
    keys = set(t.keys())
    p = np.stack([t[f"score_{c}"].array(library="np") for c in JC1_CLASSES], 1).astype(np.float64)
    y = np.stack([t[c].array(library="np") for c in JC1_CLASSES], 1).argmax(1) \
        if all(c in keys for c in JC1_CLASSES) else None
    if y is None:
        raise SystemExit(f"FATAL: {cell}/pred.root lacks the label branches")
    from sklearn.metrics import roc_auc_score
    old = lp.AUC_STRIDE
    lp.AUC_STRIDE = 1
    try:
        out = {"accuracy": float((p.argmax(1) == y).mean()), "n_jets": int(y.size),
               "macro_auc_ovr": float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in range(10)])),
               "macro_auc_ovo": float(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovo",
                                                    average="macro")),
               **lp.jc1_per_class(p, y)}
    finally:
        lp.AUC_STRIDE = old
    return out


def jc2b_metrics(cell: pathlib.Path, lp) -> dict:
    fd = cell / "features"
    z, y = np.load(fd / "logits.npy").astype(np.float64), np.load(fd / "label188.npy").astype(np.int64)
    if z.shape != (y.size, 188):
        raise SystemExit(f"FATAL: {fd}: logits {z.shape} for {y.size} jets")
    old = lp.AUC_STRIDE
    lp.AUC_STRIDE = 1
    try:
        return lp.native_metrics(softmax(z), y, np.arange(188))
    finally:
        lp.AUC_STRIDE = old


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--trees", nargs="+", default=["leg_top", "leg_qg", "leg2", "leg_jc2b"])
    a = ap.parse_args(argv)
    bm = _load("bench_metrics", "experiments/FT/bench_metrics.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    lp = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
    doc = {"script": "experiments/FT/budget_metrics.py", "root": str(a.root),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "cells": {}, "missing": []}
    for tree in a.trees:
        for cell in sorted((a.root / tree).glob("*/N*/s*")):
            if cell.name.endswith(".lock") or ".partial." in cell.name:
                continue
            key = f"{tree}/{cell.relative_to(a.root / tree)}"
            if not (cell / "DONE").exists():
                doc["missing"].append(key)
                continue
            if tree in ("leg_top", "leg_qg"):
                m = bm.cell_metrics(cell, probe)
                if tree == "leg_qg":
                    m["herwig"] = bm.cell_metrics(cell, probe, test_set="herwig")
            elif tree == "leg2":
                m = jc1_metrics(cell, lp)
            else:
                m = jc2b_metrics(cell, lp)
            m["early_stop"] = json.loads((cell / "early_stop.json").read_text())
            m["early_stop"].pop("metrics", None)
            man = json.loads((cell / "ft_manifest.json").read_text())
            m["recipe"] = {k: man.get(k) for k in ("lr", "head_lr_mult", "lr_schedule", "weight_decay",
                                                   "max_epochs", "steps_per_epoch", "early_stop_patience",
                                                   "subset", "init", "checkpoint")}
            ic = cell / "init_checkpoint.json"
            if ic.exists():
                m["init_checkpoint"] = json.loads(ic.read_text())
            doc["cells"][key] = m
            print(key, json.dumps({k: m[k] for k in ("auc", "macro_auc_ovr") if k in m}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{a.out}: {len(doc['cells'])} cells, {len(doc['missing'])} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
