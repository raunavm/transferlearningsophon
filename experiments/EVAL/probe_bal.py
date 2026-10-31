#!/usr/bin/env python3
"""Linear and nonlinear probes, v4 (plan 2.3, 3.2, 3.4, 3.5; 2026-10-31).

What changes from linear_probe_v2.py's fits_v3, whose fitting and metrics this reuses:
  * JetClass-II is scored on the class-balanced sets (experiments/FT/stage_jc2_balanced.py, dataset
    jc2bal): native labels throughout, the four nested training sizes, macro AUC on EVERY test jet;
  * the class pairs (jc2bal_pairs) train on up to 1e5 jets per side and test on up to 2e5 per side,
    each side at its natural composition, with C chosen on a tenth of the training rows held out;
  * the C grid extends two decades upward (1e-3 ... 1e4): 148 of the 1,482 fits_v3 fits chose its
    old edge, 100;
  * JetClass (jc1) is scored on all 2M test jets, with ParT's one-vs-one macro AUC beside ours;
  * the missing-feature variants (checkpoint tags <tag>.<variant>, input_variants.py) are probed on
    jc2bal at its largest training set;
  * --old-only DATASET... fits only those of JetClass/top/quark-gluon (the kinematic-only models,
    which are probed on top alone);
  * --hidden probes the head's 512-unit hidden layer, ReLU(W1 x + b1) of the class token x with the
    checkpoint's own first head layer, on jc2bal and quark/gluon at their largest training set.
Top and quark/gluon (and Herwig) read the features linear_probe_v2.py read (--old-root). Every cell's
fitted probe and test probabilities are saved (--probs-out), so per-class metrics and the bootstrap
need no refit.

    python3 experiments/EVAL/probe_bal.py --model mtx-l188-s1 --checkpoints best70 \\
        --variants mtop mqg itop iqg --out /data/results/eval/v2_linprobe_bal/fits_v4 \\
        --probs-out /data/results/eval/v2_linprobe_bal/fits_v4_probs
    python3 experiments/EVAL/probe_bal.py --model mtx-l188-s1 --checkpoints best70 --mlp --out .../fits_v4_mlp
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
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
NEW_ROOT = pathlib.Path("/data/results/eval/v2_linprobe_bal")
OLD_ROOT = pathlib.Path("/data/results/eval/v2_linprobe")
BAL = pathlib.Path("/data/finetune/jc2_bal")
C_GRID_V4 = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PAIR_VAL_FRACTION = 0.1
PAIR_SEED = 20261031
N_NATIVE = 188


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


lp = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
# v4 settings of the reused functions, set once here: fit() reads C_GRID, and multiclass_metrics,
# native_metrics and jc1_per_class read AUC_STRIDE, when called
lp.C_GRID = C_GRID_V4
lp.AUC_STRIDE = 1


def load(model: str, ds: str, split: str, ckpt: str, root: pathlib.Path = NEW_ROOT):
    x, y, m = lp.load_split(root / model / ds / split / ckpt)
    return x, y, m


def ovo_auc(p: np.ndarray, y: np.ndarray) -> float:
    """ParT's JetClass AUC: scikit-learn's macro one-vs-one (Hand and Till)."""
    from sklearn.metrics import roc_auc_score
    q = p / p.sum(1, keepdims=True)
    return float(roc_auc_score(y, q, multi_class="ovo", average="macro"))


def hidden(x: np.ndarray, manifest: dict) -> np.ndarray:
    """The head's hidden layer, ReLU(W1 x + b1), with the checkpoint's own first head layer."""
    import torch
    sd = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    w, b = sd["mod.fc.0.0.weight"].double().numpy(), sd["mod.fc.0.0.bias"].double().numpy()
    if w.shape != (512, 128):
        raise SystemExit(f"FATAL: {manifest['checkpoint']}: head layer {w.shape}")
    return np.maximum(x.astype(np.float64) @ w.T + b, 0.0).astype(np.float32)


def jc2bal_cells(model: str, ckpt: str, save, sizes=None, transform=None, name="jc2bal") -> dict:
    xva, yva, _ = load(model, "jc2bal", "val", ckpt)
    xte, yte, mte = load(model, "jc2bal", "test", ckpt)
    if transform:
        xva, xte = transform(xva, mte), transform(xte, mte)
    cells = {}
    for n in sizes or lp.train_sizes(NEW_ROOT / model, "jc2bal"):
        xtr, ytr, _ = load(model, "jc2bal", f"train_N{n}", ckpt)
        if transform:
            xtr = transform(xtr, mte)
        t0 = time.time()
        sc, clf, rec = lp.fit(xtr, ytr, xva, yva, N_NATIVE)
        p = lp.full_proba(clf, sc.transform(xte), N_NATIVE)
        cells[str(n)] = {**lp.native_metrics(p, yte, np.unique(ytr)), **rec, "n_train": int(ytr.size),
                         "n_val": int(yva.size), "n_classes_in_train": int(np.unique(ytr).size),
                         "train_label_counts": np.bincount(ytr, minlength=N_NATIVE).tolist(),
                         "seconds": round(time.time() - t0, 1)}
        if save:
            save(f"{name}_N{n}", sc, clf, rec, np.arange(yte.size), yte, p=p)
        print(f"  {model} {ckpt} {name} N{n}: macro_auc={cells[str(n)]['macro_auc_ovr']:.5f} C={rec['C']}",
              flush=True)
    return cells


def pair_index() -> dict:
    z = np.load(BAL / "pairs_index.npz")
    out = {}
    for k in z.files:
        split, pair, side = k.split("/")
        out.setdefault(split, {}).setdefault(pair, {})[side] = z[k]
    return out


def pair_cells(model: str, ckpt: str, probe, save) -> dict:
    xtr_all, _, _ = load(model, "jc2bal_pairs", "train", ckpt)
    xte_all, _, _ = load(model, "jc2bal_pairs", "test", ckpt)
    idx = pair_index()
    cells = {}
    for pair, sides in idx["train"].items():
        rows = np.concatenate([sides["signal"], sides["background"]])
        y = np.concatenate([np.ones(sides["signal"].size, np.int64), np.zeros(sides["background"].size, np.int64)])
        r = np.random.default_rng([PAIR_SEED, sum(map(ord, pair))])
        isval = r.random(rows.size) < PAIR_VAL_FRACTION
        te = idx["test"][pair]
        trow = np.concatenate([te["signal"], te["background"]])
        yte = np.concatenate([np.ones(te["signal"].size, np.int64), np.zeros(te["background"].size, np.int64)])
        sc, clf, rec = lp.fit(xtr_all[rows[~isval]], y[~isval], xtr_all[rows[isval]], y[isval], 2)
        p = lp.full_proba(clf, sc.transform(xte_all[trow]), 2)
        cells[pair] = {**lp.binary_metrics(p, yte, probe), **rec, "n_train": int((~isval).sum()),
                       "n_val": int(isval.sum()), "n_test_signal": int(te["signal"].size),
                       "n_test_background": int(te["background"].size)}
        if save:
            save(f"jc2bal_pairs_{pair}", sc, clf, rec, trow, yte, s=p[:, lp.SIGNAL_COLUMN])
        print(f"  {model} {ckpt} pair {pair}: auc={cells[pair]['auc']:.5f} C={rec['C']}", flush=True)
    return cells


def old_cells(model: str, ckpt: str, helpers, save, datasets=("jc1", "top", "qg")) -> dict:
    """JetClass, top and quark/gluon (with Herwig) on fits_v3's features, the v4 grid and stride."""
    cells = lp.probe_cells(OLD_ROOT / model, ckpt, list(datasets), helpers, save,
                           pathlib.Path(lp.JC2_PARQUET))
    for n, c in cells.get("jc1", {}).items():
        if save:
            z = np.load(save.out / model / ckpt / f"jc1_N{n}.npz")
            c["macro_auc_ovo"] = ovo_auc(z["p"].astype(np.float64), z["y"].astype(np.int64))
    return cells


def mlp_cells(model: str, ckpt: str, probe, datasets) -> dict:
    """The nonlinear probe (probe.fit_mlp) at the largest training set of each dataset."""
    cells = {}
    if "jc2bal" in datasets:
        n = max(lp.train_sizes(NEW_ROOT / model, "jc2bal"))
        xtr, ytr, _ = load(model, "jc2bal", f"train_N{n}", ckpt)
        xva, yva, _ = load(model, "jc2bal", "val", ckpt)
        xte, yte, _ = load(model, "jc2bal", "test", ckpt)
        p, meta = probe.fit_mlp(xtr, ytr, xva, yva, xte, n_classes=N_NATIVE)
        cells["jc2bal"] = {**lp.native_metrics(p, yte, np.unique(ytr)), **lp.mlp_fit_record(meta), "mlp": meta,
                           "n_train": int(ytr.size), "train_size": n}
        xtr_all, _, _ = load(model, "jc2bal_pairs", "train", ckpt)
        xte_all, _, _ = load(model, "jc2bal_pairs", "test", ckpt)
        idx = pair_index()
        for pair, sides in idx["train"].items():
            rows = np.concatenate([sides["signal"], sides["background"]])
            y = np.concatenate([np.ones(sides["signal"].size, np.int64), np.zeros(sides["background"].size, np.int64)])
            r = np.random.default_rng([PAIR_SEED, sum(map(ord, pair))])
            isval = r.random(rows.size) < PAIR_VAL_FRACTION
            te = idx["test"][pair]
            trow = np.concatenate([te["signal"], te["background"]])
            yte = np.concatenate([np.ones(te["signal"].size, np.int64), np.zeros(te["background"].size, np.int64)])
            s, meta = probe.fit_mlp(xtr_all[rows[~isval]], y[~isval], xtr_all[rows[isval]], y[isval], xte_all[trow])
            cells[f"pair_{pair}"] = {**lp.binary_score_metrics(s, yte, probe, threshold=0.0),
                                     **lp.mlp_fit_record(meta), "mlp": meta}
            print(f"  {model} {ckpt} mlp pair {pair}: auc={cells[f'pair_{pair}']['auc']:.5f}", flush=True)
    for ds in [d for d in datasets if d in ("jc1", "top", "qg")]:
        k = lp.DATASETS[ds][0]
        n = max(lp.train_sizes(OLD_ROOT / model, ds))
        xtr, ytr, _ = lp.load_split(OLD_ROOT / model / ds / f"train_N{n}" / ckpt)
        xva, yva, _ = lp.load_split(OLD_ROOT / model / ds / "val" / ckpt)
        xte, yte, _ = lp.load_split(OLD_ROOT / model / ds / "test" / ckpt)
        if ds == "jc1":
            p, meta = probe.fit_mlp(xtr, ytr, xva, yva, xte, n_classes=k)
            cells[ds] = {**lp.multiclass_metrics(p, yte, k, 0, lp_eval_arm()), **lp.jc1_per_class(p, yte),
                         "macro_auc_ovo": ovo_auc(p, yte), **lp.mlp_fit_record(meta), "mlp": meta}
        else:
            her = lp.load_split(OLD_ROOT / model / "qg" / "herwig" / ckpt) if ds == "qg" else None
            xs = xte if her is None else np.concatenate([xte, her[0]])
            s, meta = probe.fit_mlp(xtr, ytr, xva, yva, xs)
            cells[ds] = {**lp.binary_score_metrics(s[:yte.size], yte, probe, threshold=0.0),
                         **lp.mlp_fit_record(meta), "mlp": meta}
            if her is not None:
                cells["qg_herwig"] = {**lp.binary_score_metrics(s[yte.size:], her[1], probe, threshold=0.0),
                                      **lp.mlp_fit_record(meta)}
        print(f"  {model} {ckpt} mlp {ds} done", flush=True)
    return cells


_EVAL_ARM = []


def lp_eval_arm():
    if not _EVAL_ARM:
        _EVAL_ARM.append(_load("eval_arm", "experiments/EVAL/eval_arm.py"))
    return _EVAL_ARM[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--mlp-datasets", nargs="+", default=["jc2bal", "jc1", "top", "qg"])
    ap.add_argument("--skip-old", action="store_true", help="no JetClass/top/quark-gluon refit")
    ap.add_argument("--old-only", nargs="+", metavar="DATASET",
                    help="only these of jc1/top/qg, no balanced JetClass-II (the kinematic-only models: top)")
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
    probe = _load("probe", "experiments/EVAL/probe.py")
    helpers = (lp_eval_arm(), probe, lr.rung_maps()["L162"], names)
    doc = {"model": a.model, "checkpoints": a.checkpoints, "variants": a.variants, "c_grid": list(C_GRID_V4),
           "auc_stride": 1, "selection": "validation cross-entropy", "max_iter": lp.MAX_ITER,
           "converged_grad": lp.CONVERGED_GRAD, "pair_val_fraction": PAIR_VAL_FRACTION,
           "balanced_manifest_sha256": None if a.old_only else
           hashlib.sha256((BAL / "manifest.json").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "cells": {}}
    for ckpt in a.checkpoints:
        save = lp.saver(a.probs_out, a.model, ckpt)
        if save:
            save.out = a.probs_out
        if a.mlp:
            doc["probe"] = "mlp"
            doc["cells"][ckpt] = mlp_cells(a.model, ckpt, probe, a.mlp_datasets)
            continue
        if a.old_only:
            doc["cells"][ckpt] = old_cells(a.model, ckpt, helpers, save, a.old_only)
            continue
        c = {"jc2bal": jc2bal_cells(a.model, ckpt, save), "jc2bal_pairs": pair_cells(a.model, ckpt, probe, save)}
        if not a.skip_old:
            c.update(old_cells(a.model, ckpt, helpers, save))
        nmax = max(lp.train_sizes(NEW_ROOT / a.model, "jc2bal"))
        for v in a.variants:
            vs = lp.saver(a.probs_out, a.model, f"{ckpt}.{v}")
            c[f"jc2bal.{v}"] = jc2bal_cells(a.model, f"{ckpt}.{v}", vs, sizes=[nmax])
        if a.hidden:
            man = json.loads((NEW_ROOT / a.model / "jc2bal" / "test" / ckpt / "manifest.json").read_text())
            hs = lp.saver(a.probs_out, a.model, f"{ckpt}.hidden")
            c["jc2bal.hidden"] = jc2bal_cells(a.model, ckpt, hs, sizes=[nmax], transform=lambda x, m: hidden(x, man),
                                              name="jc2bal_hidden")
            n = max(lp.train_sizes(OLD_ROOT / a.model, "qg"))
            xtr, ytr, _ = lp.load_split(OLD_ROOT / a.model / "qg" / f"train_N{n}" / ckpt)
            xva, yva, _ = lp.load_split(OLD_ROOT / a.model / "qg" / "val" / ckpt)
            xte, yte, _ = lp.load_split(OLD_ROOT / a.model / "qg" / "test" / ckpt)
            sc, clf, rec = lp.fit(hidden(xtr, man), ytr, hidden(xva, man), yva, 2)
            p = lp.full_proba(clf, sc.transform(hidden(xte, man)), 2)
            c["qg.hidden"] = {str(n): {**lp.binary_metrics(p, yte, probe), **rec, "n_train": int(ytr.size)}}
            if hs:
                hs(f"qg_hidden_N{n}", sc, clf, rec, np.arange(yte.size), yte, s=p[:, lp.SIGNAL_COLUMN])
        doc["cells"][ckpt] = c
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"{out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
