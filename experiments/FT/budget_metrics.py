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
Every cell also carries its early_stop.json (best epoch, validation metric, epochs run, stopped early;
a full-set cell of ParT's recipe has best_epoch.json instead, read into the same keys) and its manifest's
recipe. A cell without DONE is listed as missing; the read-out never fills it. With --expect-fixed the
cells are the one rule's at each row's recorded rate (scripts/build_ft_budget_jobs.py paper_fixed_cells, from
ft_v2/fixed and ft_v2/fixed_rate; --root names the main tree only), and one not yet started is missing too,
so a partial read-out names every cell it lacks.

    python3 experiments/FT/budget_metrics.py --root /data/results/ft_v2/budget --out budget_metrics.json
    python3 experiments/FT/budget_metrics.py --root /data/results/ft_v2/lrsweep_v2 --out lrsweep_metrics.json
    python3 experiments/FT/budget_metrics.py --root /data/results/ft_v2/fixed --expect-fixed --out fixed_metrics.json
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


def fixed_keys() -> dict:
    """{tree/init/N<N>/s<S>: cell directory} of every cell of the one rule at every size, at its row's recorded rate
    (build_ft_budget_jobs.paper_fixed_cells: ft_v2/fixed, or ft_v2/fixed_rate where the row's search moved its rate)."""
    b = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    return {"/".join(pathlib.Path(c["out"]).parts[-4:]): pathlib.Path(c["out"]) for c in b.paper_fixed_cells()}


def early_stop(cell: pathlib.Path) -> dict:
    """The cell's early_stop.json without its per-epoch metrics; a full-set cell of ParT's recipe (20 epochs, no
    stopping, build_ft_jobs) has best_epoch.json instead, read into the same keys."""
    if (cell / "early_stop.json").exists():
        es = json.loads((cell / "early_stop.json").read_text())
        es.pop("metrics", None)
        return es
    be = json.loads((cell / "best_epoch.json").read_text())
    return {"source": "best_epoch.json", "best_epoch": be["epoch"], "best_metric": be["metric"],
            "epochs_completed": be["n_epochs"], "stopped_early": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--trees", nargs="+", default=["leg_top", "leg_qg", "leg2", "leg_jc2b"])
    ap.add_argument("--expect-fixed", action="store_true",
                    help="the cells are the one rule's (build_ft_budget_jobs.fixed_cells); one not DONE is missing")
    a = ap.parse_args(argv)
    bm = _load("bench_metrics", "experiments/FT/bench_metrics.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    lp = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
    doc = {"script": "experiments/FT/budget_metrics.py", "root": str(a.root),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "cells": {}, "missing": []}
    expect = fixed_keys() if a.expect_fixed else None
    if expect is not None:
        doc["expected"] = len(expect)
        # the cells sit in the main tree (--root) and the rerun trees beside it (fixed_rate, fixed_rate2): resolved
        # from --root, so the jobs' --root /data/results/ft_v2/fixed reads the same folders and a test's tmp root
        # reads none of /data (the full suite of 14184f3 spent hours here reading every finished cell on /data)
        b = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
        main_tree, ft_root = pathlib.Path(b.FIXED_ROOT), pathlib.Path(b.FIXED_ROOT).parent
        expect = {k: a.root / p.relative_to(main_tree) if p.is_relative_to(main_tree)
                  else a.root.parent / p.relative_to(ft_root) for k, p in expect.items()}
    for tree in a.trees:
        found = [(f"{tree}/{c.relative_to(a.root / tree)}", c) for c in sorted((a.root / tree).glob("*/N*/s*"))] \
            if expect is None else [(k, expect[k]) for k in sorted(expect) if k.startswith(tree + "/")]
        for key, cell in found:
            if cell.name.endswith(".lock") or ".partial." in cell.name:
                continue
            if not (cell / "DONE").exists():
                doc["missing"].append(key)
                continue
            if tree == "leg_jc2b" and not (cell / "features" / "logits.npy").exists():
                # a superseded cell whose test logits were removed (features_manifest.py --set jc2b-1e-2-logits): never
                # read again, listed as missing so a preview leaves it out
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
            m["early_stop"] = early_stop(cell)
            man = json.loads((cell / "ft_manifest.json").read_text())
            m["recipe"] = {k: man.get(k) for k in ("lr", "head_lr_mult", "lr_schedule", "weight_decay",
                                                   "max_epochs", "steps_per_epoch", "early_stop_patience",
                                                   "subset", "init", "checkpoint")}
            m["gpu"] = man.get("gpu_device_name")       # every tree, for the compute table
            # what compared cells must share (PI 2026-11-15; tests/test_comparisons_identical.py reads it)
            m["setup"] = {k: man.get(k) for k in ("batch_size", "samples_per_epoch", "samples_per_epoch_val", "val_rule",
                                                  "val", "val_sha256", "subset_sha256", "data_config", "network",
                                                  "num_classes", "ckpt_rule", "training_code_sha256", "weaver_version", "repo_commit",
                                                  "torch_version")}
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
