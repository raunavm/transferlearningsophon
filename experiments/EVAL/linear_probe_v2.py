"""Linear probes of the v2 pretrained models on the downstream tasks (PI, 2026-10-09).

A probe is a logistic regression on the frozen class-token features (the 128-d input of the
output MLP, the latent Sophon transfers from), fitted on the training subsets the full
fine-tuning uses and scored on the same test jets with the same metric code:

  jc2   JetClass-II held-out subsets, the 162-way vocabulary (fine-tuning leg 1). Test: the
        2M-jet TEST2M stream read with native labels and mapped to the 162 groups; accuracy
        on every jet, one-vs-rest macro AUC on every 4th (experiments/FT/leg1_metrics.py).
  jc1   JetClass, 10 classes (leg 2). Test: the 2M jets of leg 2, the same two metrics.
  top   top tagging, top = signal. Test: top_test.parquet. Accuracy, AUC, ln(1-AUC), R50, R30
        (experiments/FT/bench_metrics.py).
  qg    quark/gluon, quark = signal. Test: Pythia chunks 18-19; the same probe is also scored
        on Herwig chunks 0-1 (dataset "qg_herwig").

Training sizes and subsets are fine-tuning's (seed-1 subsets). Features are standardised on
the training subset; the L2 strength C is chosen from C_GRID on the validation set by its
cross-entropy (probabilities floored at 1e-12, so a class absent from a small training subset
costs every C the same); warm-started upward in C. No test jet is seen before scoring.
A class absent from the training subset gets probability 0.

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
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
MAX_ITER = 1000
AUC_STRIDE = 4                       # leg 1 and leg 2: macro AUC on every 4th test jet
P_FLOOR = 1e-12
# dataset -> (number of classes, the class index that is QCD/background for eval_arm.metrics)
DATASETS = {"jc2": (162, 161), "jc1": (10, 0), "top": (2, 0), "qg": (2, 0)}
SIGNAL_COLUMN = 1                    # top and quark, as experiments/FT/bench_metrics.py
# Binary probes on the JetClass-II features between classes the coarser vocabularies merge (the
# X->bc probe; the flavour pair's b vs c in two- and four-prong decays), signal first, by native
# class name: trained on the largest training subset's jets of those classes, C chosen on the
# validation set's, scored on the first 2M TEST2M jets' (dataset "jc2_pairs").
PAIRS = {"bc_vs_bq_cs": (("label_X_bc",), ("label_X_bq", "label_X_cs")),
         "bb_vs_cc": (("label_X_bb",), ("label_X_cc",)),
         "bbqq_vs_ccqq": (("label_X_YY_bbqq",), ("label_X_YY_ccqq",))}
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

    def fit(self, x, y, c: float, tol: float = 1e-6) -> int:
        """Minimise mean log-loss + ||W||^2 / (2 C N) (scikit-learn's objective over N) from the
        current weights; returns the iterations used."""
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
        search[str(c)] = {"val_cross_entropy": ce, "n_iter": n_iter, "converged": n_iter < MAX_ITER}
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


def binary_metrics(p: np.ndarray, y: np.ndarray, probe) -> dict:
    if not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise SystemExit(f"FATAL: binary labels expected, found {np.unique(y)[:5].tolist()}")
    score = p[:, SIGNAL_COLUMN]
    l1m, censored, auc = probe.log1m_auc(y, score)
    out = {"accuracy": float((p.argmax(1) == y).mean()), "auc": auc, "log1m_auc": l1m,
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


def pair_probes(xtr, ytr, xva, yva, xte, yte, groups: dict, probe) -> dict:
    """The PAIRS probes on group-labelled jets: binary, signal = 1."""
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
    return cells


def probe_cells(model_dir: pathlib.Path, checkpoint: str, datasets, helpers) -> dict:
    eval_arm, probe, l162, names = helpers
    cells = {}
    for ds in datasets:
        k, qcd = DATASETS[ds]
        xva, yva, _ = load_split(model_dir / ds / "val" / checkpoint)
        xte, yte, mte = load_split(model_dir / ds / "test" / checkpoint)
        if ds in TEST_FIRST:
            if yte.size < TEST_FIRST[ds]:
                raise SystemExit(f"FATAL: {model_dir.name}/{ds}/test holds {yte.size} jets, "
                                 f"fewer than the {TEST_FIRST[ds]} fine-tuning scores")
            xte, yte = xte[:TEST_FIRST[ds]], yte[:TEST_FIRST[ds]]
        if ds == "jc2":
            yte = jc2_truth(yte, l162)
        her = load_split(model_dir / ds / "herwig" / checkpoint) if ds == "qg" else None
        for n in train_sizes(model_dir, ds):
            t0 = time.time()
            xtr, ytr, mtr = load_split(model_dir / ds / f"train_N{n}" / checkpoint)
            if ytr.size != n:
                raise SystemExit(f"FATAL: {model_dir.name}/{ds}/train_N{n} holds {ytr.size} jets")
            for y in (ytr, yva, yte):
                if y.min() < 0 or y.max() >= k:
                    raise SystemExit(f"FATAL: {model_dir.name}/{ds}: labels outside 0..{k - 1}")
            sc, clf, rec = fit(xtr, ytr, xva, yva, k)
            p = full_proba(clf, sc.transform(xte), k)
            m = (multiclass_metrics(p, yte, k, qcd, eval_arm) if k > 2 else binary_metrics(p, yte, probe))
            cell = {**m, **rec, "n_train": int(ytr.size), "n_classes_in_train": int(np.unique(ytr).size),
                    "n_val": int(yva.size), "train_checkpoint_sha256": mtr["checkpoint_sha256"],
                    "test_checkpoint_sha256": mte["checkpoint_sha256"],
                    "test_label_sha256": hashlib.sha256(yte.tobytes()).hexdigest(),
                    "seconds": round(time.time() - t0, 1)}
            if mtr["checkpoint_sha256"] != mte["checkpoint_sha256"]:
                raise SystemExit(f"FATAL: {model_dir.name}/{ds}: train and test features come "
                                 "from different checkpoint files")
            cells.setdefault(ds, {})[str(n)] = cell
            if her is not None:
                ph = full_proba(clf, sc.transform(her[0]), k)
                cells.setdefault("qg_herwig", {})[str(n)] = {**binary_metrics(ph, her[1], probe), **rec,
                                                             "n_train": int(ytr.size)}
            print(f"  {model_dir.name} {checkpoint} {ds} N={n}: C={rec['C']} "
                  + " ".join(f"{key}={m[key]:.5f}" for key in ("accuracy", "macro_auc_ovr", "auc")
                             if key in m and m[key] is not None)
                  + f" ({cell['seconds']} s)", flush=True)
            if ds == "jc2" and n == max(train_sizes(model_dir, ds)):
                cells["jc2_pairs"] = {str(n): pair_probes(xtr, ytr, xva, yva, xte, yte,
                                                          pair_groups(names, l162), probe)}
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--model", required=True, help="e.g. mtx-l188-s1 or init-s1")
    ap.add_argument("--checkpoints", nargs="+", default=["best70", "best70_bn"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    out = a.out / f"{a.model}.json"
    if out.exists():
        print(f"{out} exists: every probe of {a.model} is fitted")
        return 0
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    helpers = (_load("eval_arm", "experiments/EVAL/eval_arm.py"),
               _load("probe", "experiments/EVAL/probe.py"), lr.rung_maps()["L162"], names)
    model_dir = a.root / a.model
    doc = {"model": a.model, "root": str(a.root), "checkpoints": a.checkpoints,
           "datasets": a.datasets, "c_grid": list(C_GRID), "max_iter": MAX_ITER,
           "auc_stride": AUC_STRIDE, "selection": "validation cross-entropy",
           "threads": os.environ.get("OMP_NUM_THREADS"),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "cells": {c: probe_cells(model_dir, c, a.datasets, helpers) for c in a.checkpoints}}
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"{out} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
