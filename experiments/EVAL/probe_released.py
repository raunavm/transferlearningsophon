#!/usr/bin/env python3
"""The v4 linear probes (probe_bal.py) of a model whose features live under another root: the released
Sophon checkpoint (2026-11-07, PI: "Released Sophon checkpoint as a reference row in the frozen-probe
tables"), extracted by scripts/build_jc2bal_jobs.py sophon into <root>/<model>/{jc2bal,jc1,top,qg}/.

The same fits and metrics as fits_v4: balanced JetClass-II at every training size (jc2bal_cells), and
JetClass, top and quark/gluon with Herwig (old_cells, ParT's one-vs-one AUC on JetClass at the largest
size); --coarse adds the coarse label sets of probe_bal.py --coarse. The output file has fits_v4's
layout, cells[<checkpoint>][jc2bal|jc1|top|qg|qg_herwig][<n>], plus jc2bal_coarse when asked, so
mlst_v4.load_fits reads it.

    python3 experiments/EVAL/probe_released.py --model sophon-released --checkpoints released \\
        --root /data/results/eval/v2_linprobe_sophon --out .../fits_v4 --probs-out .../fits_v4_probs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pb = _load("probe_bal", "experiments/EVAL/probe_bal.py")
lp = pb.lp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--root", required=True, type=pathlib.Path, help="the features' root: <root>/<model>/<dataset>/...")
    ap.add_argument("--coarse", nargs="*", choices=sorted(pb.COARSE_MAPS), default=[])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--probs-out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    out = a.out / f"{a.model}.json"
    if out.exists():
        print(f"{out} exists")
        return 0
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    helpers = (pb.lp_eval_arm(), _load("probe", "experiments/EVAL/probe.py"), lr.rung_maps()["L162"], names)
    doc = {"model": a.model, "checkpoints": a.checkpoints, "feature_root": str(a.root), "c_grid": list(pb.C_GRID_V4),
           "auc_stride": 1, "selection": "validation cross-entropy", "max_iter": lp.MAX_ITER,
           "converged_grad": lp.CONVERGED_GRAD,
           "balanced_manifest_sha256": hashlib.sha256((pb.BAL / "manifest.json").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "probe_bal_sha256": hashlib.sha256((REPO / "experiments/EVAL/probe_bal.py").read_bytes()).hexdigest(),
           "cells": {}}
    if a.coarse:
        doc["coarse_maps"] = {t: pb.COARSE_MAPS[t] for t in a.coarse}
    for ckpt in a.checkpoints:
        save = lp.saver(a.probs_out, a.model, ckpt)
        if save:
            save.out = a.probs_out
        c = {"jc2bal": pb.jc2bal_cells(a.model, ckpt, save, sizes=lp.train_sizes(a.root / a.model, "jc2bal"),
                                       root=a.root)}
        c.update(pb.old_cells(a.model, ckpt, helpers, save, root=a.root))
        if a.coarse:
            c["jc2bal_coarse"] = pb.coarse_cells(a.model, ckpt, a.coarse, save, root=a.root)
        doc["cells"][ckpt] = c
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"{out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
