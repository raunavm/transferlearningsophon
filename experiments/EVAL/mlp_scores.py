#!/usr/bin/env python3
"""MLP probes with their per-jet test scores (PI 2026-11-23, LEDGER DL-9: "MLP fits for the three nulls ... CPU only,
one model per fit family, same protocol as tab:mlp"; SC-1: the paired test-jet bootstrap needs every network's scores
on the same jets). The nonlinear probe of tab:mlp (experiments/EVAL/probe.py fit_mlp: one hidden layer of 256, three
seeds, trained to a plateau, the class probabilities or the logit difference averaged over the seeds) is fitted as
the family it refits fitted it, on the same features, labels and test jets, and its test scores are kept, so that
experiments/STATS/paired_bootstrap.py --task npz tests MLP differences under SC-1. Nothing is trained but the probe.

Families (each on one CPU model, PF-1; the job pins it and the pod refuses any other):
  jc2bal   probe_bal.mlp_cells' first block (fits_v5_mlp): the largest class-balanced JetClass-II training set,
           the 188 native classes, scored on the 903,922 balanced test jets; p (n, 188)
  coarse   the same fit on the 17 merged groups of the 17-class labels (probe_bal.coarse_lut, the task of tab:coarse's
           linear probes, probe_bal.py --coarse R16_Q1), 17 outputs; p (n, 17), y the group
  bench    linear_probe_v2.mlp_cells, cell set v4 (tab:mlp's top and quark/gluon fit): top tagging and quark/gluon at
           their largest training set, the quark/gluon probe also scored on the Herwig jets; s, the logit difference.
           The missing inputs take the substitution the linear probes of Fig. 2 and Table 2 use (FILL: random draws on
           top, the training mean on quark/gluon; probe_bal.py --fill's features), like with like with the
           linear-probe nulls these fits stand beside (CS-8; third review of render 5f4e1b8, B3(c)), not tab:mlp's zeros
Output: <out>/<model>.json (the metrics as the refitted family records them, the fit records, the CPU model, and
each metric beside the stored fit it repeats, where one exists) and <out>/probs/<model>/<ckpt>/<name>.npz (rows, y and
p or s, the names of the linear probes' files, under <ckpt>.<fill> for bench); an existing <model>.json is never
overwritten.

    python3 experiments/EVAL/mlp_scores.py --family jc2bal --model mtx-l188-s1 --out /data/results/eval/bs3_mlp/jc2bal
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
FAMILIES = ("jc2bal", "coarse", "bench")
COARSE = "R16_Q1"
FILL_ROOT = pathlib.Path("/data/results/eval/v2_linprobe_fill")
FILL = {"top": "itop4", "qg": "nqg"}     # build_refit_v5_jobs.FILL_TAG: the substitutions chosen on validation
# the stored fits each family repeats, for the comparison recorded beside every metric
STORED = {"jc2bal": pathlib.Path("/data/results/eval/v2_linprobe_bal/fits_v5_mlp"), "coarse": None, "bench": None}
COMPARED = {"jc2bal": ("accuracy", "macro_auc_ovr", "macro_auc_resonant", "macro_auc_qcd")}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save(out: pathlib.Path, model: str, ckpt: str, name: str, y: np.ndarray, *, p=None, s=None) -> str:
    """<out>/probs/<model>/<ckpt>/<name>.npz with rows (0..n-1, the test split's order), y and p or s, float32."""
    d = out / "probs" / model / ckpt
    d.mkdir(parents=True, exist_ok=True)
    arr = {"rows": np.arange(y.size, dtype=np.int64), "y": np.asarray(y, dtype=np.int16)}
    if p is not None:
        arr["p"] = np.asarray(p, dtype=np.float32)
    if s is not None:
        arr["s"] = np.asarray(s, dtype=np.float32)
    tmp = d / f"{name}.tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, d / f"{name}.npz")
    return str(d / f"{name}.npz")


def jc2bal(pb, probe, model: str, ckpt: str, out: pathlib.Path, coarse: bool) -> dict:
    """probe_bal.mlp_cells' JetClass-II block (coarse: on the 17 merged groups), with the test scores saved."""
    n = max(pb.lp.train_sizes(pb.NEW_ROOT / model, "jc2bal"))
    xtr, ytr, _ = pb.load(model, "jc2bal", f"train_N{n}", ckpt)
    xva, yva, _ = pb.load(model, "jc2bal", "val", ckpt)
    xte, yte, _ = pb.load(model, "jc2bal", "test", ckpt)
    t0 = time.time()
    if coarse:
        lut = pb.coarse_lut(COARSE)
        k = int(lut.max()) + 1
        p, meta = probe.fit_mlp(xtr, lut[ytr], xva, lut[yva], xte, n_classes=k)
        cell = {**pb.coarse_metrics(p, lut[yte]), "n_classes": k}
        path = save(out, model, ckpt, f"jc2bal_{COARSE}_N{n}", lut[yte], p=p)
        key = "jc2bal_coarse"
    else:
        p, meta = probe.fit_mlp(xtr, ytr, xva, yva, xte, n_classes=pb.N_NATIVE)
        cell = pb.lp.native_metrics(p, yte, np.unique(ytr))
        path = save(out, model, ckpt, f"jc2bal_N{n}", yte, p=p)
        key = "jc2bal"
    return {key: {**cell, **pb.lp.mlp_fit_record(meta), "mlp": meta, "n_train": int(ytr.size), "train_size": n,
                  "n_val": int(yva.size), "scores": path, "seconds": round(time.time() - t0, 1)}}


def bench(pb, probe, model: str, ckpt: str, out: pathlib.Path, fill_root: pathlib.Path = FILL_ROOT) -> dict:
    """linear_probe_v2.mlp_cells (cell set v4) on top and quark/gluon under the linear probes' substitution (FILL),
    Herwig scored by the quark/gluon probe."""
    lp, root, cells = pb.lp, fill_root / model, {}
    for ds in ("top", "qg"):
        tag = f"{ckpt}.{FILL[ds]}"
        n = max(lp.train_sizes(root, ds))
        xva, yva, _ = lp.load_split(root / ds / "val" / tag)
        xte, yte, _ = lp.load_split(root / ds / "test" / tag)
        xtr, ytr, _ = lp.load_split(root / ds / f"train_N{n}" / tag)
        her = lp.load_split(root / "qg" / "herwig" / tag) if ds == "qg" else None
        xs = xte if her is None else np.concatenate([xte, her[0]])
        t0 = time.time()
        s, meta = probe.fit_mlp(xtr, ytr, xva, yva, xs)
        nte = int(yte.size)
        cells[ds] = {**lp.binary_score_metrics(s[:nte], yte, probe, threshold=0.0), **lp.mlp_fit_record(meta),
                     "mlp": meta, "n_train": int(ytr.size), "n_val": int(yva.size), "train_size": n, "fill": FILL[ds],
                     "scores": save(out, model, tag, f"{ds}_N{n}", yte, s=s[:nte]), "seconds": round(time.time() - t0, 1)}
        if her is not None:
            cells["qg_herwig"] = {**lp.binary_score_metrics(s[nte:], her[1], probe, threshold=0.0),
                                  **lp.mlp_fit_record(meta), "n_train": int(ytr.size), "train_size": n, "fill": FILL[ds],
                                  "scores": save(out, model, tag, f"qg_herwig_N{n}", her[1], s=s[nte:])}
        print(f"  {model} {ckpt} mlp {ds}: auc={cells[ds]['auc']:.5f} r30={cells[ds]['r30']:.1f} "
              f"({cells[ds]['seconds']} s)", flush=True)
    return cells


def stored(family: str, model: str, ckpt: str, cells: dict, root=None) -> dict:
    """Each metric of the refit beside the stored fit's ({} without one): {cell: {metric: [stored, refit]}}."""
    root = STORED[family] if root is None else root
    f = root / f"{model}.json" if root else None
    if f is None or not f.exists():
        return {}
    old = json.loads(f.read_text())["cells"].get(ckpt, {})
    return {c: {k: [old[c].get(k), v.get(k)] for k in COMPARED[family]} for c, v in cells.items() if c in old}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default="best70")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    out = a.out / f"{a.model}.json"
    if out.exists():
        print(f"{out} exists")
        return 0
    pb = _load("probe_bal", "experiments/EVAL/probe_bal.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    cpu = _load("latent_scale_probe", "experiments/EVAL/latent_scale_probe.py").cpu_model()
    a.out.mkdir(parents=True, exist_ok=True)
    cells = (bench(pb, probe, a.model, a.checkpoint, a.out) if a.family == "bench"
             else jc2bal(pb, probe, a.model, a.checkpoint, a.out, coarse=a.family == "coarse"))
    doc = {"model": a.model, "family": a.family, "checkpoints": [a.checkpoint], "probe": "mlp",
           "mlp_threads": probe.MLP_THREADS, "mlp_seeds": list(probe.MLP_SEEDS), "mlp_schedule": probe.MLP_SCHEDULE,
           "cpu_model": cpu, "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "cells": {a.checkpoint: cells}, "stored": stored(a.family, a.model, a.checkpoint, cells)}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, out)
    print(f"{out} written; against the stored fit: {json.dumps(doc['stored'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
