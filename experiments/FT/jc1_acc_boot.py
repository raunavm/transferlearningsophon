#!/usr/bin/env python3
"""A paired bootstrap over the JetClass test jets of the fine-tuned accuracy (PI 2026-11-09: a JetClass difference
stands when it exceeds 3 sigma, sigma^2 = the run-to-run spread^2 + a bootstrap over the test jets^2, as
experiments/EVAL/sculpting_boot.py).

Inputs: <root>/leg2/<init>-s<k>/N<n>/s1/pred.root of each configuration's runs, every one DONE. The cells predict
the same test jets in the same order: their label columns must agree jet by jet, or FATAL. Replica 0 is the test
set itself and must reproduce the cell's accuracy in --metrics (budget_metrics.py's read-out of the same cells),
FATAL beyond NOMINAL_TOL. Each of the --replicas resamples the test jets with replacement ONCE, and the same
resample serves every run (paired).

DIFFERENCES: configuration minus the 188-class reference, paired by run index (seed). For each quantity, d_i the
nominal paired value of run index i:
  accuracy     d_i = acc(config, i) - acc(188, i)
  error_ratio  d_i = (1 - acc(config, i)) / (1 - acc(188, i)) - 1   (the share of more errors)
  delta     mean_i d_i
  run_sd    SD of d_i over the indices (ddof 1)
  boot_sd   SD over the replicas of that replica's mean_i d_i (ddof 1)
  sigma     sqrt(run_sd^2 + boot_sd^2); pass = |delta| > 3 sigma

    python3 experiments/FT/jc1_acc_boot.py --root /data/results/ft_v2/budget_long --n 1000000 \\
        --metrics /data/results/ft_v2/budget_long/budget_long_metrics.json --replicas 200 --out jc1_acc_boot_1e6.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

MIN_REPLICAS = 200
NOMINAL_TOL = 1e-12
JC1_CLASSES = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql",
               "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl")
REF = "l188"
CONFIGS = ("l162", "r42q1", "r16q1")
RUNS = (1, 2, 3)


def correct(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Per jet: whether the highest score is the true class (budget_metrics.jc1_metrics), and the true class."""
    import uproot
    t = uproot.open(path)["Events"]
    p = np.stack([t[f"score_{c}"].array(library="np") for c in JC1_CLASSES], 1).astype(np.float64)
    y = np.stack([t[c].array(library="np") for c in JC1_CLASSES], 1).argmax(1)
    return p.argmax(1) == y, y


def summary(d: np.ndarray, boot: np.ndarray) -> dict:
    """d: nominal paired values per run index; boot: (replicas, runs) of the same."""
    delta, run_sd = float(d.mean()), float(d.std(ddof=1))
    boot_sd = float(boot.mean(1).std(ddof=1))
    sigma = float(np.hypot(run_sd, boot_sd))
    return {"per_run": d.tolist(), "delta": delta, "run_sd": run_sd, "boot_sd": boot_sd, "sigma": sigma,
            "z": delta / sigma if sigma else None, "pass": bool(abs(delta) > 3 * sigma)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--metrics", required=True, type=pathlib.Path)
    ap.add_argument("--replicas", type=int, default=MIN_REPLICAS)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.replicas < MIN_REPLICAS:
        raise SystemExit(f"FATAL: {a.replicas} replicas, at least {MIN_REPLICAS}")
    stored = json.loads(a.metrics.read_text())["cells"]
    ok, y0, acc = {}, None, {}
    for init in (REF,) + CONFIGS:
        for k in RUNS:
            key = f"leg2/{init}-s{k}/N{a.n}/s1"
            cell = a.root / key
            if not (cell / "DONE").exists():
                raise SystemExit(f"FATAL: {cell} not DONE")
            c, y = correct(cell / "pred.root")
            if y0 is None:
                y0 = y
            elif y.shape != y0.shape or not np.array_equal(y, y0):
                raise SystemExit(f"FATAL: {cell}: its test jets differ from the first cell's")
            acc[key] = float(c.mean())
            if abs(acc[key] - stored[key]["accuracy"]) > NOMINAL_TOL:
                raise SystemExit(f"FATAL: {key}: accuracy {acc[key]} vs stored {stored[key]['accuracy']}")
            ok[(init, k)] = c.astype(np.float64)
    n = y0.size
    rng = np.random.default_rng(a.seed)
    names = list(ok)
    mat = np.stack([ok[m] for m in names])                       # (cells, jets)
    boot_acc = np.empty((a.replicas, len(names)))
    for r in range(a.replicas):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        boot_acc[r] = mat @ w / n
    col = {m: i for i, m in enumerate(names)}
    nominal = np.array([ok[m].mean() for m in names])
    diffs = {}
    for conf in CONFIGS:
        ref_i = [col[(REF, k)] for k in RUNS]
        con_i = [col[(conf, k)] for k in RUNS]
        d_acc = nominal[con_i] - nominal[ref_i]
        b_acc = boot_acc[:, con_i] - boot_acc[:, ref_i]
        d_err = (1 - nominal[con_i]) / (1 - nominal[ref_i]) - 1
        b_err = (1 - boot_acc[:, con_i]) / (1 - boot_acc[:, ref_i]) - 1
        diffs[f"{conf}-{REF}"] = {"accuracy_ref": nominal[ref_i].tolist(), "accuracy_config": nominal[con_i].tolist(),
                                  "accuracy": summary(d_acc, b_acc), "error_ratio": summary(d_err, b_err)}
        print(conf, json.dumps({q: {k: diffs[f"{conf}-{REF}"][q][k] for k in ("delta", "run_sd", "boot_sd", "pass")}
                                for q in ("accuracy", "error_ratio")}), flush=True)
    doc = {"script": "experiments/FT/jc1_acc_boot.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "root": str(a.root), "n": a.n, "n_test_jets": int(n), "replicas": a.replicas, "seed": a.seed,
           "rule": "pass = |delta| > 3 sigma, sigma^2 = run_sd^2 + boot_sd^2 (PI 2026-11-09)",
           "accuracy": acc, "diffs": diffs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; never overwritten")
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
