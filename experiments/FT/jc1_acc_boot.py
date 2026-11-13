#!/usr/bin/env python3
"""A paired bootstrap over the JetClass test jets of the fine-tuned accuracy (PI 2026-11-09: a JetClass difference
stands when it exceeds 3 sigma, sigma^2 = the run-to-run spread^2 + a bootstrap over the test jets^2, as
experiments/EVAL/sculpting_boot.py).

Inputs: <root>/leg2/<init>-s<k>/N<n>/s1/pred.root of each configuration's runs, every one DONE. The cells predict
the same test jets in the same order: their label columns must agree jet by jet, or FATAL. Replica 0 is the test
set itself and must reproduce the cell's accuracy in --metrics (budget_metrics.py's read-out of the same cells),
FATAL beyond NOMINAL_TOL. Each of the --replicas resamples the test jets with replacement ONCE, and the same
resample serves every run (paired).

DIFFERENCES (DIFFS): configuration minus reference -- each label set minus 188 classes, and each mass-output model
minus its label set (round 5, decision L3) -- paired by the run indices both have (ARM_RUNS). For each quantity, d_i
the nominal paired value of run index i:
  accuracy     d_i = acc(config, i) - acc(188, i)
  error_ratio  d_i = (1 - acc(config, i)) / (1 - acc(188, i)) - 1   (the share of more errors)
  delta     mean_i d_i
  run_sd    SD of d_i over the indices (ddof 1)
  boot_sd   SD over the replicas of that replica's mean_i d_i (ddof 1)
  sigma     sqrt(run_sd^2 + boot_sd^2); pass = |delta| > 3 sigma
and beside it, under "l3", the rule of decision L3 (PI 2026-11-13), which reads the sides unpaired over all their runs:
  delta     mean accuracy of the configuration's runs minus the reference's
  sd_config, sd_reference   the SD of each side over its runs (ddof 1)
  boot_sd   SD over the replicas of that replica's difference of the two means
  sigma     sqrt(sd_config^2 + sd_reference^2 + boot_sd^2); pass = |delta| > 3 sigma

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
# configuration minus reference, by name; and the run indices of each arm (the mass-output models have two)
DIFFS = {**{f"{c}-{REF}": (c, REF) for c in CONFIGS},
         "l162mass-l162": ("l162mass", "l162"), "r16q1mass-r16q1": ("r16q1mass", "r16q1")}
ARM_RUNS = {**{a: RUNS for a in (REF,) + CONFIGS}, "l162mass": (1, 2), "r16q1mass": (1, 2)}


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


def l3(nom_c: np.ndarray, nom_r: np.ndarray, boot_c: np.ndarray, boot_r: np.ndarray) -> dict:
    """Decision L3, the sides unpaired: nom_* the nominal value of each run of a side, boot_* (replicas, runs)."""
    sd = lambda v: float(v.std(ddof=1)) if v.size > 1 else None  # noqa: E731
    delta, sd_c, sd_r = float(nom_c.mean() - nom_r.mean()), sd(nom_c), sd(nom_r)
    boot_sd = float((boot_c.mean(1) - boot_r.mean(1)).std(ddof=1))
    sigma = float(np.sqrt(sd_c ** 2 + sd_r ** 2 + boot_sd ** 2)) if sd_c is not None and sd_r is not None else None
    return {"delta": delta, "n_config": int(nom_c.size), "n_reference": int(nom_r.size), "sd_config": sd_c,
            "sd_reference": sd_r, "boot_sd": boot_sd, "sigma": sigma, "z": delta / sigma if sigma else None,
            "pass": bool(sigma is not None and abs(delta) > 3 * sigma)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--metrics", required=True, type=pathlib.Path)
    ap.add_argument("--replicas", type=int, default=MIN_REPLICAS)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--diffs", nargs="+", choices=list(DIFFS), default=list(DIFFS))
    a = ap.parse_args(argv)
    if a.replicas < MIN_REPLICAS:
        raise SystemExit(f"FATAL: {a.replicas} replicas, at least {MIN_REPLICAS}")
    stored = json.loads(a.metrics.read_text())["cells"]
    ok, y0, acc = {}, None, {}
    arms = list(dict.fromkeys(x for d in a.diffs for x in DIFFS[d]))
    for init in arms:
        for k in ARM_RUNS[init]:
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
    for name in a.diffs:
        conf, ref = DIFFS[name]
        ks = [k for k in ARM_RUNS[conf] if k in ARM_RUNS[ref]]
        ref_i = [col[(ref, k)] for k in ks]
        con_i = [col[(conf, k)] for k in ks]
        d_acc = nominal[con_i] - nominal[ref_i]
        b_acc = boot_acc[:, con_i] - boot_acc[:, ref_i]
        d_err = (1 - nominal[con_i]) / (1 - nominal[ref_i]) - 1
        b_err = (1 - boot_acc[:, con_i]) / (1 - boot_acc[:, ref_i]) - 1
        all_c = [col[(conf, k)] for k in ARM_RUNS[conf]]
        all_r = [col[(ref, k)] for k in ARM_RUNS[ref]]
        diffs[name] = {"runs": ks, "accuracy_ref": nominal[ref_i].tolist(), "accuracy_config": nominal[con_i].tolist(),
                       "accuracy": summary(d_acc, b_acc), "error_ratio": summary(d_err, b_err),
                       "l3": {"accuracy": l3(nominal[all_c], nominal[all_r], boot_acc[:, all_c], boot_acc[:, all_r])}}
        print(name, json.dumps({q: {k: diffs[name][q][k] for k in ("delta", "run_sd", "boot_sd", "pass")}
                                for q in ("accuracy", "error_ratio")}),
              json.dumps({k: diffs[name]["l3"]["accuracy"][k] for k in ("delta", "sigma", "pass")}), flush=True)
    doc = {"script": "experiments/FT/jc1_acc_boot.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "root": str(a.root), "n": a.n, "n_test_jets": int(n), "replicas": a.replicas, "seed": a.seed,
           "rule": "pass = |delta| > 3 sigma, sigma^2 = run_sd^2 + boot_sd^2 (PI 2026-11-09)",
           "rule_l3": "pass = |delta| > 3 sigma, sigma^2 = sd_config^2 + sd_reference^2 + boot_sd^2, the sides unpaired "
                      "over all their runs (decision L3, 2026-11-13)",
           "accuracy": acc, "diffs": diffs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; never overwritten")
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
