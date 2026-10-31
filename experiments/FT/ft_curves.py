#!/usr/bin/env python3
"""ROC points and per-class tables from the stored per-jet scores of the v2 fine-tunes and of the
linear probes (plan 2026-10-29, Step 1.7; the appendix ROC figure and the ParT-format tables).

  cells   one fine-tuning tree, <tree>/<init>/N<N>/s<S>/, every cell with DONE (a cell without it
          is listed under "skipped"; a cell whose training loss went to NaN is fatal):
     jc1      leg 2, JetClass (pred.root, 10 classes), in the form of ParT's Table 1: accuracy, the
              macro AUC one-vs-rest and one-vs-one (ParT's convention is one-vs-one) on every
              JC1_AUC_STRIDE-th jet (leg2_metrics' macro AUC jets, so the two agree), and for each
              signal class against QCD on the discriminant S/(S+QCD), on every jet, its AUC and its
              rejection at ParT's working point (eps_S 50%; 99% for H->lvqq', 99.5% for t->blv)
     top, qg  accuracy, AUC, R50 and R30 (q/g also on its Herwig test set, "qg_herwig")
     jc2_188  JetClass-II at 188 classes: each resonant class against the summed 27 QCD classes,
              discriminant p_c / (p_c + sum of the QCD probabilities), its AUC and rejection at 50%
          every comparison also carries its ROC points (below).
  probes  the linear probes' saved test outputs (experiments/EVAL/linear_probe_v2.py --probs-out),
          <probs-root>/<model>/<checkpoint>/<dataset>_N<N>.npz, at each dataset's largest training
          set: the same blocks, on the jets the probe saved (jc1 and jc2_188: every 4th test jet);
          the v4 probes' balanced JetClass-II (jc2bal, as jc2_188, every test jet) and class pairs
          (jc2bal_pairs.<pair>, as top).

ROC POINTS. At each signal efficiency of EPS_GRID: eps_B by linear interpolation on the ROC, the
rejection 1/eps_B capped at the number of background jets, n_bkg_pass = eps_B * N_bkg rounded, and
is_bound when fewer than one background jet passes (the rejection is then a lower bound) --
experiments/EVAL/probe.rejection_at's convention, so a working point read here equals bench_metrics'.
Scores are the softmax of the stored logits, as the read-outs compute them.

    python3 experiments/FT/ft_curves.py cells --tree /data/results/ft_v2/best70/leg2 --task jc1 \\
        --out /data/results/ft_v2/curves/best70_leg2.json
    python3 experiments/FT/ft_curves.py probes --probs-root /data/results/eval/v2_linprobe/fits_v3_probs \\
        --out /data/results/ft_v2/curves/linprobe_fits_v3.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")   # class order, QCD classes, eps_S
BM = _load("bench_metrics", "experiments/FT/bench_metrics.py")         # softmax, label order, NaN check
LEG2 = _load("leg2_metrics", "experiments/FT/leg2_metrics.py")         # pred.root reader
SLIM = _load("slim_outputs", "experiments/FT/slim_outputs.py")         # leg-1 slim cache reader

EPS_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20)) + (0.97, 0.98, 0.99, 0.995)
TASKS = ("jc1", "top", "qg", "jc2_188")
JC2_CLASSES = 188
JC1_AUC_STRIDE = 4               # leg2_metrics.py --auc-stride (its default), the tables' macro AUC
FEATURES_188 = "features_v2"     # the 188-class leg's cache, as leg 1's
PROBE_FILE = re.compile(r"^(jc1|jc2_188|jc2bal|top|qg|qg_herwig)_N(\d+)\.npz$")
# the v4 probes' class pairs on the balanced sets (experiments/EVAL/probe_bal.py), one size each
PAIR_FILE = re.compile(r"^jc2bal_pairs_(.+)\.npz$")


def roc_points(y: np.ndarray, score: np.ndarray, grid=EPS_GRID) -> dict:
    """The ROC at each signal efficiency of grid (y = 1 signal); see the docstring."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    n_bkg = int((y == 0).sum())
    eps_b = np.interp(np.asarray(grid, dtype=np.float64), tpr, fpr)
    raw = np.divide(1.0, eps_b, out=np.full_like(eps_b, np.inf), where=eps_b > 0)
    return {"eps_s": [float(e) for e in grid], "eps_b": eps_b.tolist(),
            "rejection": np.minimum(raw, float(n_bkg)).tolist(),
            "n_bkg_pass": np.rint(eps_b * n_bkg).astype(int).tolist(),
            "is_bound": (raw >= n_bkg).tolist(),
            "n_signal": int((y == 1).sum()), "n_background": n_bkg}


def working_point(y: np.ndarray, score: np.ndarray, eps_s: float) -> dict:
    r = roc_points(y, score, (eps_s,))
    return {k: (r[k][0] if isinstance(r[k], list) else r[k]) for k in r}


def binary_block(y: np.ndarray, score: np.ndarray) -> dict:
    """Top and q/g: accuracy, the AUC to full precision, R50 and R30, the ROC points."""
    from sklearn.metrics import roc_auc_score
    if not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise SystemExit(f"FATAL: binary labels expected, found {np.unique(y)[:5].tolist()}")
    return {"accuracy": float(((score > 0.5) == (y == 1)).mean()), "auc": float(roc_auc_score(y, score)),
            "n_jets": int(y.size), "r50": working_point(y, score, 0.5), "r30": working_point(y, score, 0.3),
            "roc": roc_points(y, score)}


def summed(p: np.ndarray, y: np.ndarray, bkg) -> tuple[np.ndarray, np.ndarray]:
    """(jets of the background classes, their summed probability), computed once for every class."""
    return np.isin(y, bkg), p[:, list(bkg)].astype(np.float64).sum(1)


def versus(p: np.ndarray, y: np.ndarray, c: int, bkg) -> tuple[np.ndarray, np.ndarray]:
    """(signal flag, p_c / (p_c + summed background)) on the jets of c and of the background
    (bkg = summed()); a jet with both at 0 scores the tie, 0.5 (as linear_probe_v2.per_class)."""
    isb, pb = bkg
    m = (y == c) | isb
    num = p[m, c].astype(np.float64)
    den = num + pb[m]
    return (y[m] == c).astype(int), np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)


def class_block(p, y, c: int, bkg, eps_s: float) -> dict:
    from sklearn.metrics import roc_auc_score
    t, d = versus(p, y, c, bkg)
    return {"auc": float(roc_auc_score(t, d)), "rejection": working_point(t, d, eps_s), "roc": roc_points(t, d)}


def jc1_block(p: np.ndarray, y: np.ndarray, names: list[str], auc_stride: int = 1) -> dict:
    """ParT's Table 1 on these jets, the macro AUCs on every auc_stride-th; names are the columns'
    label names, checked against ParT's order."""
    from sklearn.metrics import roc_auc_score
    if list(names) != list(LP.JC1_CLASSES):
        raise SystemExit(f"FATAL: JetClass columns {names} are not {list(LP.JC1_CLASSES)}")
    p = p.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    qcd = names.index("label_QCD")
    bkg = summed(p, y, (qcd,))
    pa, ya = p[::auc_stride], y[::auc_stride]
    return {"accuracy": float((p.argmax(1) == y).mean()), "n_jets": int(y.size),
            "auc_stride": auc_stride, "n_jets_auc": int(ya.size),
            "macro_auc_ovr": float(roc_auc_score(ya, pa, multi_class="ovr", average="macro")),
            "macro_auc_ovo": float(roc_auc_score(ya, pa, multi_class="ovo", average="macro")),
            "per_class": {names[c]: {"eps_s": LP.JC1_EPS_S.get(names[c], 0.5),
                                     **class_block(p, y, c, bkg, LP.JC1_EPS_S.get(names[c], 0.5))}
                          for c in range(len(names)) if c != qcd}}


def jc2_188_block(p: np.ndarray, y: np.ndarray) -> dict:
    """Each resonant class present against the summed 27 QCD classes, at eps_S 50%."""
    if p.shape != (y.size, JC2_CLASSES) or y.min() < 0 or y.max() >= JC2_CLASSES:
        raise SystemExit(f"FATAL: JetClass-II scores {p.shape} with labels {y.min()}..{y.max()}: "
                         f"not the {JC2_CLASSES}-class task")
    qcd = summed(p, y, LP.QCD_NATIVE)
    return {"accuracy": float((p.argmax(1) == y).mean()), "n_jets": int(y.size),
            "per_class": {str(int(c)): class_block(p, y, int(c), qcd, 0.5)
                          for c in np.unique(y) if c not in LP.QCD_NATIVE}}


def cell_scores(cell: pathlib.Path, task: str, test_set: str = "pythia"):
    """(rows into the test split, labels, scores, class names or None) of one fine-tuning cell:
    scores are (n, K) probabilities (jc1, jc2_188) or the (n,) signal probability (top, qg)."""
    if task == "jc1":
        names, onehot, scores = LEG2.read_pred_root(cell / "pred.root")
        if not np.all(onehot.sum(axis=1) == 1):
            raise SystemExit(f"FATAL: {cell}/pred.root labels are not one-hot")
        p = scores.astype(np.float64)
        p /= p.sum(1, keepdims=True)
        return np.arange(len(p)), onehot.argmax(1), p, [f"label_{n}" for n in names]
    if task in ("top", "qg"):
        fd = cell / "features"
        f_logits, f_labels = BM.TEST_FILES[test_set]
        logits, y = np.load(fd / f_logits), np.load(fd / f_labels).astype(np.int64)
        if logits.shape != (y.size, 2):
            raise SystemExit(f"FATAL: {fd}/{f_logits} is {logits.shape} for {y.size} labels")
        return np.arange(y.size), y, BM.softmax(logits)[:, BM.SIGNAL_COLUMN], None
    if task == "jc2_188":
        fd = cell / FEATURES_188
        # the slim cache keeps every auc_stride-th row (1 for this leg); the extraction itself
        # read every stride-th test jet (4: every 4th of the first 2M, the probe's AUC jets)
        auc_stride = json.loads((fd / "slim.json").read_text())["auc_stride"] if (fd / "slim.json").exists() else 1
        ext_stride = json.loads((fd / "extract_manifest.json").read_text()).get("stride", 1)
        _, logits = SLIM.read_leg1(fd, auc_stride)
        y = np.load(fd / "label188.npy").astype(np.int64)
        rows = np.arange(y.size)[::auc_stride]
        return rows * ext_stride, y[rows], BM.softmax(logits), None
    raise SystemExit(f"FATAL: unknown task {task}")


def cell_blocks(cell: pathlib.Path, task: str) -> dict:
    if task == "jc1":
        _, y, p, names = cell_scores(cell, task)
        return {"jc1": jc1_block(p, y, names, JC1_AUC_STRIDE)}
    if task == "jc2_188":
        _, y, p, _ = cell_scores(cell, task)
        return {"jc2_188": jc2_188_block(p, y)}
    out = {}
    for test_set in ("pythia", "herwig") if task == "qg" else ("pythia",):
        _, y, s, _ = cell_scores(cell, task, test_set)
        out[task if test_set == "pythia" else "qg_herwig"] = binary_block(y, s)
    return out


def discover(tree: pathlib.Path):
    """([(init, N, seed, cell)] with DONE, [cells without]); .partial.* attempts and .lock skipped."""
    done, skipped = [], []
    for cell in sorted(tree.glob("*/N*/s*")):
        if not cell.is_dir() or ".partial." in cell.name or cell.name.endswith(".lock"):
            continue
        if not (cell / "DONE").exists():
            skipped.append(str(cell))
            continue
        if BM.diverged(cell):
            raise SystemExit(f"FATAL: {cell} trained to a NaN loss; not a result")
        done.append((cell.parent.parent.name, cell.parent.name, cell.name, cell))
    return done, skipped


def probe_blocks(npz: pathlib.Path, ds: str) -> dict:
    z = np.load(npz)
    y = z["y"].astype(np.int64)
    if "s" in z:
        return binary_block(y, z["s"].astype(np.float64))
    p = z["p"].astype(np.float64)
    return jc1_block(p, y, list(LP.JC1_CLASSES)) if ds == "jc1" else jc2_188_block(p, y)


def largest(ckpt_dir: pathlib.Path) -> dict:
    """{dataset: (largest N, file)} of one probe's saved outputs; a v4 class pair is the dataset
    jc2bal_pairs.<pair>, with N 0 (its training size is in the fit file)."""
    out = {}
    for f in ckpt_dir.glob("*.npz"):
        m = PROBE_FILE.match(f.name)
        if m and int(m.group(2)) > out.get(m.group(1), (0, None))[0]:
            out[m.group(1)] = (int(m.group(2)), f)
        elif (m := PAIR_FILE.match(f.name)):
            out[f"jc2bal_pairs.{m.group(1)}"] = (0, f)
    return out


def write_json(path: pathlib.Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, allow_nan=False))   # compact: ~10^6 ROC numbers per tree
    if path.exists():
        raise SystemExit(f"FATAL: {path} appeared while this ran; not overwritten ({tmp} kept)")
    os.replace(tmp, path)
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells")
    c.add_argument("--tree", required=True, type=pathlib.Path, help="<tree>/<init>/N<N>/s<S>/")
    c.add_argument("--task", required=True, choices=TASKS)
    c.add_argument("--out", required=True, type=pathlib.Path)
    p = sub.add_parser("probes")
    p.add_argument("--probs-root", required=True, type=pathlib.Path)
    p.add_argument("--models", nargs="+", help="default: every model directory under --probs-root")
    p.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; use a new output")
    doc = {"script": "experiments/FT/ft_curves.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "eps_grid": list(EPS_GRID)}
    if a.cmd == "cells":
        if a.task in ("top", "qg"):
            BM.check_label_order(a.task)
        cells, skipped = discover(a.tree)
        if not cells:
            raise SystemExit(f"FATAL: no DONE cell under {a.tree}")
        res = {}
        for init, n, seed, cell in cells:
            res.setdefault(init, {}).setdefault(n, {})[seed] = cell_blocks(cell, a.task)
            print(f"  {init:16} {n:10} {seed}", flush=True)
        doc.update(tree=str(a.tree), task=a.task, skipped=skipped, cells=res)
    else:
        res = {}
        for md in sorted(d for d in a.probs_root.iterdir() if d.is_dir() and (not a.models or d.name in a.models)):
            for cd in sorted(d for d in md.iterdir() if d.is_dir()):
                for ds, (n, f) in sorted(largest(cd).items()):
                    res.setdefault(md.name, {}).setdefault(cd.name, {})[ds] = {"n_train": n, "file": str(f),
                                                                                **probe_blocks(f, ds)}
                    print(f"  {md.name:20} {cd.name:10} {ds:10} N={n}", flush=True)
        if a.models and set(a.models) - set(res):
            raise SystemExit(f"FATAL: no saved outputs for {sorted(set(a.models) - set(res))}")
        doc.update(probs_root=str(a.probs_root), cells=res)
    write_json(a.out, doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
