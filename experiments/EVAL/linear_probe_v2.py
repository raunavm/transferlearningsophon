"""Linear probes of the v2 pretrained models on the downstream tasks (PI, 2026-10-09).

A probe is a logistic regression on the frozen class-token features (the 128-d input of the
output MLP, the latent Sophon transfers from), fitted on the training subsets the full
fine-tuning uses and scored on the same test jets with the same metric code:

  jc2   JetClass-II held-out subsets, the 162-way vocabulary (QCD merged). Test: the first 2M
        jets of the TEST2M stream, read with native labels and mapped to the 162 groups;
        accuracy on every jet, one-vs-rest macro AUC on every 4th.
  jc2_188  the same features and jets with the 188 native classes (PI, 2026-10-29: the
        JetClass-II task is the full 188-class task). The cached training and validation
        labels hold the 162 groups (they were extracted through the L162 config), so their
        native labels are read from the subsets' parquet files, row for row, and checked
        against the cache. Besides the macro AUC over all classes it reports the macros over
        the 161 resonant and the 27 QCD classes and over the classes present in training,
        the AUC of every class, and each resonant class's rejection of the summed QCD
        classes; the pairs (dataset "jc2_188_pairs") add three QCD flavour pairs.
  jc1   JetClass, 10 classes (leg 2). Test: the 2M jets of leg 2, the same two metrics.
  top   top tagging, top = signal. Test: top_test.parquet. Accuracy, AUC, ln(1-AUC), R50, R30
        (experiments/FT/bench_metrics.py).
  qg    quark/gluon, quark = signal. Test: Pythia chunks 18-19; the same probe is also scored
        on Herwig chunks 0-1 (dataset "qg_herwig").

Training sizes and subsets are fine-tuning's (seed-1 subsets). Features are standardised on
the training subset; the L2 strength C is chosen from C_GRID on the validation set by its
cross-entropy (probabilities floored at 1e-12, so a class absent from a small training subset
costs every C the same); warm-started upward in C. No test jet is seen before scoring.
A class absent from the training subset gets probability 0, so its one-vs-rest AUC is 0.5.

The objective is scikit-learn's LogisticRegression (lbfgs, L2): C x the summed log-loss plus
||W||^2 / 2, intercept unpenalised, softmax over the classes present for k > 2 and one sigmoid
logit for two classes. It is minimised by full-batch L-BFGS in float64 with torch (on the GPU
when there is one): scikit-learn took ~0.13 s per iteration at 10^4 jets on the cluster's CPUs,
hours per checkpoint at 10^6 (2026-10-10). tests/test_linprobe_v2.py checks the two agree.

Input (scripts/build_linprobe_jobs.py writes it with experiments/EVAL/extract_v2.py):
    <root>/<model>/<dataset>/<split>/<checkpoint>/{features.npy, label188.npy, manifest.json}
    split = train_N<N> | val | test (| herwig for qg)
Output: <out>/<model>.json, written last and atomically:
    {"model", "checkpoints", "cells": {checkpoint: {dataset: {N: metrics}}}, ...}
and, per cell, <probs-out>/<model>/<checkpoint>/<dataset>_N<N>.npz: the fitted probe
(coef, intercept, scaler_mean, scaler_scale, C) and its test outputs on the scored rows (rows,
y, and s, the signal score of a binary task, or p, the class probabilities on the AUC rows of a
multi-class task at its largest training set only). --mlp instead fits the nonlinear probe of
experiments/EVAL/probe.py (decision D6: beside every linear null) on the binary tasks at their
largest training set.

    python3 experiments/EVAL/linear_probe_v2.py --root /data/results/eval/v2_linprobe \\
        --model mtx-l188-s1 --out /data/results/eval/v2_linprobe/fits
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
# 2026-10-29: 100 added; 139 of 1,083 fits of fits3000 chose the old upper edge, 10
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
# 2026-10-10: at 1000 the 162- and 10-class probes at 1e5-1e6 jets stopped short of the
# tolerance (untrained trunk, JetClass-II 1e5, C=1: objective 4e-5 above its minimum,
# reached at ~1900 iterations); 3000 lets them converge
# 2026-10-29: 10000; at 3000 two fits (untrained trunk, JetClass-II 1e6, C=10) stopped short
MAX_ITER = 10000
GRAD_TOL = 1e-6                      # L-BFGS stops when max |gradient| falls below it
CONVERGED_GRAD = 1e-5                # a fit is converged if it ends below this, before MAX_ITER
AUC_STRIDE = 4                       # leg 1 and leg 2: macro AUC on every 4th test jet
P_FLOOR = 1e-12
# dataset -> (number of classes, the class index that is QCD/background for eval_arm.metrics)
DATASETS = {"jc2": (162, 161), "jc2_188": (188, None), "jc1": (10, 0), "top": (2, 0), "qg": (2, 0)}
DATA_DIR = {"jc2_188": "jc2"}        # jc2_188 reads jc2's features
QCD_NATIVE = tuple(range(161, 188))  # the 27 QCD classes of JetClass-II (label_QCD_bbccss .. light)
JC2_PARQUET = "/data/finetune/jc2_v2"
# JetClass (leg 2) in its config's order; ParT quotes each class's rejection of QCD at 50% signal
# efficiency, except H->lvqq' at 99% and t->blv at 99.5% (ParT, Table 1)
JC1_CLASSES = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql",
               "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl")
JC1_EPS_S = {"label_Hqql": 0.99, "label_Tbl": 0.995}
SIGNAL_COLUMN = 1                    # top and quark, as experiments/FT/bench_metrics.py
# Binary probes on the JetClass-II features between classes the coarser vocabularies merge (the
# X->bc probe; the flavour pair's b vs c in two- and four-prong decays), signal first, by native
# class name: trained on the largest training subset's jets of those classes, C chosen on the
# validation set's, scored on the first 2M TEST2M jets' (dataset "jc2_pairs").
PAIRS = {"bc_vs_bq_cs": (("label_X_bc",), ("label_X_bq", "label_X_cs")),
         "bb_vs_cc": (("label_X_bb",), ("label_X_cc",)),
         "bbqq_vs_ccqq": (("label_X_YY_bbqq",), ("label_X_YY_ccqq",))}
# QCD flavour pairs on native labels (jc2_188 only): S1's QCD bb vs cc, and QCD jets with a b
# quark, or with a c quark and no b, against light QCD (u, d quarks and gluons)
QCD_PAIRS = {"qcd_bb_vs_cc": (("label_QCD_bb",), ("label_QCD_cc",)),
             "qcd_b_vs_light": ("qcd_with_b", ("label_QCD_light",)),
             "qcd_c_vs_light": ("qcd_with_c_no_b", ("label_QCD_light",))}
# leg 1 scores the first 2,000,000 jets of TEST2M (extract_features.py --max-jets keeps exactly
# that many); extract_v2.py stops at the end of the batch that crosses its --max-jets
TEST_FIRST = {"jc2": 2_000_000}
WORKING_POINTS = {"r50": 0.5, "r30": 0.3}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_split(d: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(features float32, labels int64, manifest) of one extracted split and checkpoint."""
    man = d / "manifest.json"
    if not man.exists():
        raise SystemExit(f"FATAL: {d} has no manifest.json: the extraction did not finish")
    m = json.loads(man.read_text())
    x = np.load(d / "features.npy").astype(np.float32)
    y = np.load(d / "label188.npy").astype(np.int64)
    rows = np.load(d / "rows.npy")
    if not np.array_equal(rows, np.arange(rows.size)):
        raise SystemExit(f"FATAL: {d}: the feature rows are not the stream's, in order")
    if x.ndim != 2 or x.shape != (y.size, 128) or m["n_feature_rows"] != y.size:
        raise SystemExit(f"FATAL: {d}: features {x.shape}, {y.size} labels, manifest "
                         f"{m['n_feature_rows']} rows")
    if m["n_feature_rows"] != m["n_stream"]:
        raise SystemExit(f"FATAL: {d} kept {m['n_feature_rows']} of {m['n_stream']} jets; "
                         "a probe split must keep every jet")
    if not np.isfinite(x).all():
        raise SystemExit(f"FATAL: {d} holds non-finite features")
    return x, y, m


def train_sizes(model_dir: pathlib.Path, dataset: str) -> list[int]:
    ns = sorted(int(m.group(1)) for p in (model_dir / dataset).glob("train_N*")
                if (m := re.fullmatch(r"train_N(\d+)", p.name)))
    if not ns:
        raise SystemExit(f"FATAL: {model_dir / dataset} has no train_N* splits")
    return ns


def full_proba(clf, x: np.ndarray, k: int, chunk: int = 200_000) -> np.ndarray:
    """P(class | x) over all k classes; a class the probe never saw gets 0."""
    out = np.zeros((x.shape[0], k), dtype=np.float64)
    cols = clf.classes_.astype(np.int64)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk, cols] = clf.predict_proba(x[i:i + chunk])
    return out


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.maximum(p[np.arange(y.size), y], P_FLOOR)).mean())


class Logit:
    """A fitted L2 logistic regression with scikit-learn's attributes (classes_, coef_,
    intercept_, predict_proba) and objective, minimised by full-batch L-BFGS in torch."""

    def __init__(self, classes: np.ndarray, device: str):
        self.classes_, self.device = classes, device
        m = 1 if classes.size == 2 else classes.size
        self.coef_, self.intercept_ = np.zeros((m, 128)), np.zeros(m)

    def fit(self, x, y, c: float, tol: float = GRAD_TOL) -> int:
        """Minimise mean log-loss + ||W||^2 / (2 C N) (scikit-learn's objective over N) from the
        current weights; returns the iterations used and keeps the final max |gradient| in
        grad_max_ (L-BFGS also stops on a stalled loss or its evaluation budget, so the iteration
        count alone does not show convergence)."""
        import torch
        dev, f64 = self.device, torch.float64
        X = torch.as_tensor(x, dtype=f64, device=dev)
        idx = torch.as_tensor(np.searchsorted(self.classes_, y), device=dev)
        W = torch.tensor(self.coef_, dtype=f64, device=dev, requires_grad=True)
        b = torch.tensor(self.intercept_, dtype=f64, device=dev, requires_grad=True)
        lam = 1.0 / (2.0 * c * X.shape[0])
        binary = self.classes_.size == 2
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=tol,
                                tolerance_change=1e-12, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X @ W.T + b
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], idx.to(f64))
                    if binary else torch.nn.functional.cross_entropy(z, idx)) + lam * (W * W).sum()
            loss.backward()
            return loss
        opt.step(closure)
        closure()
        self.grad_max_ = float(max(W.grad.abs().max(), b.grad.abs().max()))
        self.coef_, self.intercept_ = W.detach().cpu().numpy(), b.detach().cpu().numpy()
        return int(opt.state[opt._params[0]]["n_iter"])

    def predict_proba(self, x) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64) @ self.coef_.T + self.intercept_
        if self.classes_.size == 2:
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            return np.stack([1.0 - p1, p1], axis=1)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit(xtr, ytr, xva, yva, k: int) -> tuple[object, object, dict]:
    """(scaler, classifier at the chosen C, record of the search)."""
    from sklearn.preprocessing import StandardScaler
    if np.unique(ytr).size < 2:
        raise SystemExit("FATAL: the training subset holds one class")
    sc = StandardScaler().fit(xtr)
    xtr_s, xva_s = sc.transform(xtr), sc.transform(xva)
    clf = Logit(np.unique(ytr), device())
    search, best = {}, None
    for c in C_GRID:
        n_iter = clf.fit(xtr_s, ytr, c)
        ce = cross_entropy(full_proba(clf, xva_s, k), yva)
        search[str(c)] = {"val_cross_entropy": ce, "n_iter": n_iter, "grad_max": clf.grad_max_,
                          "converged": n_iter < MAX_ITER and clf.grad_max_ <= CONVERGED_GRAD}
        if best is None or ce < best[0]:
            best = (ce, c, clf.coef_.copy(), clf.intercept_.copy())
    _, c, coef, icpt = best
    clf.coef_, clf.intercept_ = coef, icpt
    return sc, clf, {"C": c, "search": search, "converged": search[str(c)]["converged"],
                     "device": clf.device}


def multiclass_metrics(p: np.ndarray, y: np.ndarray, k: int, qcd: int, eval_arm) -> dict:
    idx = np.arange(0, y.size, AUC_STRIDE)
    m = eval_arm.metrics(p[idx], y[idx], k, qcd)
    return {"accuracy": float((p.argmax(1) == y).mean()),
            "accuracy_on_auc_subsample": m["accuracy"], "macro_auc_ovr": m["macro_auc_ovr"],
            "n_classes_present": m["n_classes_present"], "n_jets": int(y.size),
            "n_jets_auc": int(idx.size), "auc_stride": AUC_STRIDE}


def _rejection(y: np.ndarray, score: np.ndarray, eps_s: float) -> dict:
    """1/eps_B at signal efficiency eps_s (y = 1 signal), with the background jets behind it."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    f = float(np.interp(eps_s, tpr, fpr))
    n_bkg = int((y == 0).sum())
    return {"rejection": 1.0 / max(f, 1e-12), "eps_b": f, "n_bkg_pass": int(round(f * n_bkg)),
            "is_bound": bool(f < 1.0 / max(n_bkg, 1))}


def per_class(p: np.ndarray, y: np.ndarray, bkg: tuple, eps_s: dict) -> tuple[dict, dict]:
    """({class: one-vs-rest AUC}, {signal class: rejection of the summed background classes at
    its signal efficiency}) on the given jets; the discriminant is p_c / (p_c + sum_bkg p)."""
    from sklearn.metrics import roc_auc_score
    auc = {int(c): float(roc_auc_score((y == c).astype(int), p[:, c])) for c in np.unique(y)}
    pb, isb = p[:, list(bkg)].sum(1), np.isin(y, bkg)
    rej = {}
    for c in np.unique(y):
        if c in bkg:
            continue
        m = (y == c) | isb
        num, den = p[m, c], p[m, c] + pb[m]
        d = np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)
        rej[int(c)] = _rejection((y[m] == c).astype(int), d, eps_s.get(int(c), 0.5))
    return auc, rej


def native_metrics(p: np.ndarray, y: np.ndarray, train_classes: np.ndarray) -> dict:
    """jc2_188: accuracy on every jet; on the AUC rows the macro AUC over all classes present,
    over the resonant and the QCD classes, and over the classes present in training (an absent
    class scores 0.5), each class's AUC and each resonant class's rejection of summed QCD."""
    idx = np.arange(0, y.size, AUC_STRIDE)
    auc, rej = per_class(p[idx], y[idx], QCD_NATIVE, {})
    mean = lambda cs: float(np.mean([auc[c] for c in cs])) if cs else None
    res = [c for c in auc if c not in QCD_NATIVE]
    return {"accuracy": float((p.argmax(1) == y).mean()),
            "accuracy_on_auc_subsample": float((p[idx].argmax(1) == y[idx]).mean()),
            "macro_auc_ovr": mean(list(auc)), "macro_auc_resonant": mean(res),
            "macro_auc_qcd": mean([c for c in auc if c in QCD_NATIVE]),
            "macro_auc_trained": mean([c for c in auc if c in set(train_classes.tolist())]),
            "n_classes_absent_train": int(sum(c not in set(train_classes.tolist()) for c in auc)),
            "n_classes_present": len(auc), "per_class_auc": auc, "rejection_vs_qcd": rej,
            "n_jets": int(y.size), "n_jets_auc": int(idx.size), "auc_stride": AUC_STRIDE}


def jc1_per_class(p: np.ndarray, y: np.ndarray) -> dict:
    """JetClass in the form of ParT's Table 1: each class's AUC and its rejection of QCD at ParT's
    working point, on the AUC rows."""
    idx = np.arange(0, y.size, AUC_STRIDE)
    eps = {i: JC1_EPS_S.get(n, 0.5) for i, n in enumerate(JC1_CLASSES)}
    auc, rej = per_class(p[idx], y[idx], (0,), eps)
    return {"per_class_auc": auc, "rejection_vs_qcd": rej,
            "eps_s": {int(c): eps[c] for c in rej}}


def binary_metrics(p: np.ndarray, y: np.ndarray, probe) -> dict:
    return binary_score_metrics(p[:, SIGNAL_COLUMN], y, probe)


def binary_score_metrics(score: np.ndarray, y: np.ndarray, probe, threshold: float = 0.5) -> dict:
    if not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise SystemExit(f"FATAL: binary labels expected, found {np.unique(y)[:5].tolist()}")
    l1m, censored, auc = probe.log1m_auc(y, score)
    out = {"accuracy": float(((score > threshold) == (y == 1)).mean()), "auc": auc, "log1m_auc": l1m,
           "log1m_auc_censored": censored, "n_jets": int(y.size),
           "n_signal": int((y == SIGNAL_COLUMN).sum())}
    for name, eps_s in WORKING_POINTS.items():
        rej, eps_b, bound, n_pass, rel = probe.rejection_at(y, score, eps_s)
        out.update({name: rej, f"{name}_eps_b": eps_b, f"{name}_is_bound": bound,
                    f"{name}_n_bkg_pass": n_pass, f"{name}_rel_stat": rel if math.isfinite(rel) else None})
    return out


def jc2_truth(y_native: np.ndarray, l162: dict) -> np.ndarray:
    """TEST2M's native labels -> the 162 groups, as experiments/FT/leg1_metrics.py maps them."""
    lut = np.full(max(max(l162), int(y_native.max())) + 1, -1, dtype=np.int64)
    for a, g in l162.items():
        lut[a] = g
    t = lut[y_native]
    if (y_native < 0).any() or (t < 0).any():
        raise SystemExit("FATAL: TEST2M holds native labels absent from the committed map")
    return t


def pair_groups(names: dict, l162: dict) -> dict:
    """{pair: (signal L162 groups, background L162 groups)}; each class must be a group of its own."""
    nat = {n: i for i, n in names.items()}
    out = {}
    for pair, sides in PAIRS.items():
        groups = tuple(tuple(l162[nat[c]] for c in side) for side in sides)
        for side in sides:
            for c in side:
                if sum(1 for v in l162.values() if v == l162[nat[c]]) != 1:
                    raise SystemExit(f"FATAL: {c} shares its 162-way group; {pair} needs it alone")
        out[pair] = groups
    return out


def native_labels(split: str, cached: np.ndarray, l162: dict, parquet: pathlib.Path) -> np.ndarray:
    """The native labels of a jc2 training or validation split, read from the subset's parquet in
    row order; the cache holds their 162 groups, and the two must agree on every row."""
    import pyarrow.parquet as pq
    f = parquet / ("val.parquet" if split == "val" else f"{split}_s1.parquet")
    y = pq.read_table(f, columns=["jet_label"]).column(0).to_numpy().astype(np.int64)
    if y.size != cached.size or not np.array_equal(jc2_truth(y, l162), cached):
        raise SystemExit(f"FATAL: {f}: its labels do not map row for row onto the cached 162 groups")
    return y


def native_pair_groups(names: dict) -> dict:
    """{pair: (signal native classes, background native classes)} of PAIRS and QCD_PAIRS."""
    nat = {n: i for i, n in names.items()}
    sets = {"qcd_with_b": tuple(c for c in QCD_NATIVE if "b" in names[c].removeprefix("label_QCD_")),
            "qcd_with_c_no_b": tuple(c for c in QCD_NATIVE
                                     if "c" in (f := names[c].removeprefix("label_QCD_")) and "b" not in f)}
    side = lambda v: sets[v] if isinstance(v, str) else tuple(nat[c] for c in v)
    return {pair: (side(a), side(b)) for pair, (a, b) in {**PAIRS, **QCD_PAIRS}.items()}


def pair_probes(xtr, ytr, xva, yva, xte, yte, groups: dict, probe, save=None) -> dict:
    """The pair probes on group- or class-labelled jets: binary, signal = 1."""
    cells = {}
    for pair, (sig, bkg) in groups.items():
        sel = lambda y: np.isin(y, sig + bkg)
        b = lambda y: np.isin(y, sig).astype(np.int64)
        (mtr, mva, mte) = (sel(ytr), sel(yva), sel(yte))
        if min(np.isin(ytr[mtr], sig).sum(), np.isin(ytr[mtr], bkg).sum()) < 10:
            raise SystemExit(f"FATAL: {pair}: fewer than 10 training jets of a side")
        sc, clf, rec = fit(xtr[mtr], b(ytr[mtr]), xva[mva], b(yva[mva]), 2)
        p = full_proba(clf, sc.transform(xte[mte]), 2)
        cells[pair] = {**binary_metrics(p, b(yte[mte]), probe), **rec, "n_train": int(mtr.sum()),
                       "n_val": int(mva.sum())}
        if save:
            save(pair, sc, clf, rec, np.flatnonzero(mte), b(yte[mte]), s=p[:, SIGNAL_COLUMN])
    return cells


def saver(out: pathlib.Path | None, model: str, checkpoint: str):
    """save(name, scaler, probe, record, rows, y, s=..., p=...) -> <out>/<model>/<checkpoint>/<name>.npz"""
    if out is None:
        return None

    def save(name, sc, clf, rec, rows, y, s=None, p=None):
        d = out / model / checkpoint
        d.mkdir(parents=True, exist_ok=True)
        arr = {"rows": np.asarray(rows, dtype=np.int64), "y": np.asarray(y, dtype=np.int16),
               "coef": clf.coef_, "intercept": clf.intercept_, "classes": clf.classes_,
               "scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "C": np.float64(rec["C"])}
        if s is not None:
            arr["s"] = np.asarray(s, dtype=np.float32)
        if p is not None:
            arr["p"] = np.asarray(p, dtype=np.float16)
        tmp = d / f"{name}.tmp.npz"
        np.savez(tmp, **arr)
        os.replace(tmp, d / f"{name}.npz")
    return save


def load_dataset(model_dir: pathlib.Path, ds: str, split: str, checkpoint: str, l162: dict,
                 parquet: pathlib.Path):
    """(features, labels, manifest) of one split as the dataset defines its labels."""
    x, y, m = load_split(model_dir / DATA_DIR.get(ds, ds) / split / checkpoint)
    if split == "test" and ds in TEST_FIRST_OF:
        first = TEST_FIRST[TEST_FIRST_OF[ds]]
        if y.size < first:
            raise SystemExit(f"FATAL: {model_dir.name}/{ds}/test holds {y.size} jets, "
                             f"fewer than the {first} fine-tuning scores")
        x, y = x[:first], y[:first]
    if ds == "jc2" and split == "test":
        y = jc2_truth(y, l162)
    elif ds == "jc2_188" and split != "test":
        y = native_labels(split, y, l162, parquet)
    return x, y, m


TEST_FIRST_OF = {"jc2": "jc2", "jc2_188": "jc2"}


def probe_cells(model_dir: pathlib.Path, checkpoint: str, datasets, helpers, save=None,
                parquet: pathlib.Path = pathlib.Path(JC2_PARQUET)) -> dict:
    eval_arm, probe, l162, names = helpers
    cells = {}
    for ds in datasets:
        k, qcd = DATASETS[ds]
        xva, yva, _ = load_dataset(model_dir, ds, "val", checkpoint, l162, parquet)
        xte, yte, mte = load_dataset(model_dir, ds, "test", checkpoint, l162, parquet)
        her = load_split(model_dir / ds / "herwig" / checkpoint) if ds == "qg" else None
        sizes = train_sizes(model_dir, DATA_DIR.get(ds, ds))
        for n in sizes:
            t0 = time.time()
            xtr, ytr, mtr = load_dataset(model_dir, ds, f"train_N{n}", checkpoint, l162, parquet)
            if ytr.size != n:
                raise SystemExit(f"FATAL: {model_dir.name}/{ds}/train_N{n} holds {ytr.size} jets")
            for y in (ytr, yva, yte):
                if y.min() < 0 or y.max() >= k:
                    raise SystemExit(f"FATAL: {model_dir.name}/{ds}: labels outside 0..{k - 1}")
            sc, clf, rec = fit(xtr, ytr, xva, yva, k)
            p = full_proba(clf, sc.transform(xte), k)
            if ds == "jc2_188":
                m = native_metrics(p, yte, np.unique(ytr))
            elif k > 2:
                m = multiclass_metrics(p, yte, k, qcd, eval_arm)
                if ds == "jc1":
                    m.update(jc1_per_class(p, yte))
            else:
                m = binary_metrics(p, yte, probe)
            cell = {**m, **rec, "n_train": int(ytr.size), "n_classes_in_train": int(np.unique(ytr).size),
                    "n_val": int(yva.size), "train_checkpoint_sha256": mtr["checkpoint_sha256"],
                    "test_checkpoint_sha256": mte["checkpoint_sha256"],
                    "test_label_sha256": hashlib.sha256(yte.tobytes()).hexdigest(),
                    "seconds": round(time.time() - t0, 1)}
            if mtr["checkpoint_sha256"] != mte["checkpoint_sha256"]:
                raise SystemExit(f"FATAL: {model_dir.name}/{ds}: train and test features come "
                                 "from different checkpoint files")
            cells.setdefault(ds, {})[str(n)] = cell
            if save and k == 2:
                save(f"{ds}_N{n}", sc, clf, rec, np.arange(yte.size), yte, s=p[:, SIGNAL_COLUMN])
            elif save and n == max(sizes):
                idx = np.arange(0, yte.size, AUC_STRIDE)
                save(f"{ds}_N{n}", sc, clf, rec, idx, yte[idx], p=p[idx])
            if her is not None:
                ph = full_proba(clf, sc.transform(her[0]), k)
                cells.setdefault("qg_herwig", {})[str(n)] = {**binary_metrics(ph, her[1], probe), **rec,
                                                             "n_train": int(ytr.size)}
                if save:
                    save(f"qg_herwig_N{n}", sc, clf, rec, np.arange(her[1].size), her[1],
                         s=ph[:, SIGNAL_COLUMN])
            print(f"  {model_dir.name} {checkpoint} {ds} N={n}: C={rec['C']} "
                  + " ".join(f"{key}={m[key]:.5f}" for key in ("accuracy", "macro_auc_ovr", "auc")
                             if key in m and m[key] is not None)
                  + f" ({cell['seconds']} s)", flush=True)
            if ds in ("jc2", "jc2_188") and n == max(sizes):
                groups = pair_groups(names, l162) if ds == "jc2" else native_pair_groups(names)
                psave = (lambda pair, *a, **kw: save(f"{ds}_pairs_{pair}_N{n}", *a, **kw)) if save else None
                cells[f"{ds}_pairs"] = {str(n): pair_probes(xtr, ytr, xva, yva, xte, yte, groups, probe,
                                                            psave)}
    return cells


def mlp_cells(model_dir: pathlib.Path, checkpoint: str, helpers,
              parquet: pathlib.Path = pathlib.Path(JC2_PARQUET)) -> dict:
    """The nonlinear probe (experiments/EVAL/probe.py fit_mlp: one hidden layer of 256, three
    seeds, trained to a plateau) on the binary tasks at their largest training set: the jc2_188
    pairs, top and quark/gluon (also scored on Herwig). Its score is a logit difference."""
    _, probe, l162, names = helpers
    cells = {}
    for ds in ("jc2_188", "top", "qg"):
        xva, yva, _ = load_dataset(model_dir, ds, "val", checkpoint, l162, parquet)
        xte, yte, _ = load_dataset(model_dir, ds, "test", checkpoint, l162, parquet)
        n = max(train_sizes(model_dir, DATA_DIR.get(ds, ds)))
        xtr, ytr, _ = load_dataset(model_dir, ds, f"train_N{n}", checkpoint, l162, parquet)
        if ds == "jc2_188":
            tasks = {pair: (np.isin, sig, bkg) for pair, (sig, bkg) in native_pair_groups(names).items()}
        else:
            tasks = {ds: None}
        for task, spec in tasks.items():
            if spec is None:
                mtr, mva, mte = (np.ones(y.size, bool) for y in (ytr, yva, yte))
                b = lambda y: y
            else:
                _, sig, bkg = spec
                mtr, mva, mte = (np.isin(y, sig + bkg) for y in (ytr, yva, yte))
                b = lambda y, sig=sig: np.isin(y, sig).astype(np.int64)
            her = load_split(model_dir / "qg" / "herwig" / checkpoint) if task == "qg" else None
            xs = xte[mte] if her is None else np.concatenate([xte[mte], her[0]])
            t0 = time.time()
            score, meta = probe.fit_mlp(xtr[mtr], b(ytr[mtr]), xva[mva], b(yva[mva]), xs)
            nte = int(mte.sum())
            cells[task] = {**binary_score_metrics(score[:nte], b(yte[mte]), probe, threshold=0.0),
                           "mlp": meta, "n_train": int(mtr.sum()), "n_val": int(mva.sum()),
                           "train_size": n, "seconds": round(time.time() - t0, 1)}
            if her is not None:
                cells["qg_herwig"] = {**binary_score_metrics(score[nte:], her[1], probe, threshold=0.0),
                                      "mlp": meta, "n_train": int(mtr.sum()), "train_size": n}
            print(f"  {model_dir.name} {checkpoint} mlp {task}: auc={cells[task]['auc']:.5f} "
                  f"converged={meta['all_converged']} ({cells[task]['seconds']} s)", flush=True)
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--model", required=True, help="e.g. mtx-l188-s1 or init-s1")
    ap.add_argument("--checkpoints", nargs="+", default=["best70", "best70_bn"])
    ap.add_argument("--datasets", nargs="+", default=["jc2", "jc1", "top", "qg"], choices=list(DATASETS))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--probs-out", type=pathlib.Path, default=None,
                    help="where each cell's fitted probe and test outputs go (default: none)")
    ap.add_argument("--jc2-parquet", type=pathlib.Path, default=pathlib.Path(JC2_PARQUET),
                    help="the JetClass-II subsets whose native labels jc2_188 reads")
    ap.add_argument("--mlp", action="store_true",
                    help="fit the nonlinear probe on the binary tasks at their largest training set")
    a = ap.parse_args(argv)
    out = a.out / f"{a.model}.json"
    if out.exists():
        print(f"{out} exists: every probe of {a.model} is fitted")
        return 0
    # the fits are float64, which TF32 never touches; switched off and recorded all the same
    tf32 = _load("extract_features", "experiments/EVAL/extract_features.py").strict_fp32()
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    helpers = (_load("eval_arm", "experiments/EVAL/eval_arm.py"),
               _load("probe", "experiments/EVAL/probe.py"), lr.rung_maps()["L162"], names)
    model_dir = a.root / a.model
    doc = {"model": a.model, "root": str(a.root), "checkpoints": a.checkpoints,
           "datasets": a.datasets, "c_grid": list(C_GRID), "max_iter": MAX_ITER,
           "converged_grad": CONVERGED_GRAD, "auc_stride": AUC_STRIDE,
           "selection": "validation cross-entropy",
           "threads": os.environ.get("OMP_NUM_THREADS"), "tf32": tf32,
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    if a.mlp:
        doc.update(probe="mlp", datasets=["jc2_188_pairs", "top", "qg", "qg_herwig"],
                   cells={c: mlp_cells(model_dir, c, helpers, a.jc2_parquet) for c in a.checkpoints})
    else:
        doc["cells"] = {c: probe_cells(model_dir, c, a.datasets, helpers, saver(a.probs_out, a.model, c),
                                       a.jc2_parquet) for c in a.checkpoints}
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"{out} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
