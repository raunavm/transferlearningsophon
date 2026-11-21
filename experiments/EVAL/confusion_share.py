#!/usr/bin/env python3
"""Does the share of the 188-class probe's confusions that a 17-group label set keeps apart order those label sets'
JetClass-II linear-probe accuracy? Specified by the PI on 2026-11-21, fixed here before any of its data was read
(commit time of this file); the run reads only the inputs below.

Reference   the linear probe on the 188-class network's frozen features (fits_v5, the largest training set), runs
            mtx-l188-s1, -s2, -s3, on the class-balanced JetClass-II test jets (its saved probabilities, p in
            fits_v5_probs/<run>/best70/jc2bal_N1000000.npz). For each run and each ordered pair of resonant classes
            i != j (jet_label 0-160; QCD, 161-187, left out): the rate r_ij = (test jets of class i whose most probable
            class is j) / (test jets of class i). Pair weight w_{ij} = (r_ij + r_ji) / 2, averaged over the three runs.
Measure     for each grouping g, S_g = sum of w over the unordered resonant pairs that g puts in different groups /
            sum of w over every unordered resonant pair.
Groupings   the nine 17-group label sets only, so the number of groups is fixed: 17 classes (R16_Q1), the three
            flavor-boundary sets (FLAV_F0 no flavor boundary, FLAV_F1 b-quark boundary, FLAV_F1R random flavor boundary),
            the five random partitions (RAND2_p1..p5).
Accuracy    each grouping's JetClass-II linear-probe accuracy at the largest training set, fits_v5, the mean over its
            runs (17 classes: 3; each flavor-boundary set: 2; each random partition: 1).
Pass        Spearman rank correlation of S_g with accuracy over the nine groupings >= 0.8. Pass: one appendix figure
            (S_g against accuracy) and one sentence that the ordering is consistent with it. Otherwise claim nothing.

    python3 experiments/EVAL/confusion_share.py --root /data/results/eval/v2_linprobe_bal --out OUT.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
N_RES, N_ALL = 161, 188
REFERENCE = ("mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3")
PROBS = "fits_v5_probs/{run}/best70/jc2bal_N1000000.npz"
FITS = "fits_v5/{run}.json"
PASS_RHO = 0.8
# grouping: (map file, column, runs)
GROUPINGS = {
    "17 classes": ("configs/labelmaps/rand_label_map.v2.csv", "R16_Q1", ("mtx-r16q1-s1", "mtx-r16q1-s2", "mtx-r16q1-s3")),
    "no flavor boundary": ("configs/labelmaps/flavour_pair_map.v2.csv", "FLAV_F0", ("mtx-flavf0-s1", "mtx-flavf0-s2")),
    "b-quark boundary": ("configs/labelmaps/flavour_pair_map.v2.csv", "FLAV_F1", ("mtx-flavf1-s1", "mtx-flavf1-s2")),
    "random flavor boundary": ("configs/labelmaps/flavour_pair_map.v2.csv", "FLAV_F1R", ("mtx-flavf1r-s1", "mtx-flavf1r-s2")),
    **{f"random partition {k}": ("configs/labelmaps/rand_label_map.v2.csv", f"RAND2_p{k}", (run,))
       for k, run in ((1, "mtx-rand2p1-s1"), (2, "mtx-rand2p2-s2"), (3, "mtx-rand2p3-s1"), (4, "mtx-rand2p4-s2"),
                      (5, "mtx-rand2p5-s2"))},
}


def group_of(path: str, column: str) -> np.ndarray:
    rows = list(csv.DictReader(open(REPO / path)))
    g = np.full(N_ALL, -1, np.int64)
    for r in rows:
        g[int(r["jet_label"])] = int(r[column])
    if (g < 0).any() or len(set(g.tolist())) != 17:
        raise SystemExit(f"FATAL: {path}:{column} is not a 17-group map of the 188 classes")
    return g


def confusion_rates(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """r[i, j] over resonant i, j: the fraction of class-i test jets whose most probable class is j."""
    pred = p.argmax(1)
    r = np.zeros((N_RES, N_RES))
    for i in range(N_RES):
        m = y == i
        if not m.any():
            raise SystemExit(f"FATAL: no test jets of class {i}")
        r[i] = np.bincount(pred[m], minlength=N_ALL)[:N_RES] / m.sum()
    np.fill_diagonal(r, 0.0)
    return r


def spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))     # no ties expected among nine real values
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    w, inputs = np.zeros((N_RES, N_RES)), {}
    for run in REFERENCE:
        f = a.root / PROBS.format(run=run)
        d = np.load(f)
        p, y = d["p"].astype(np.float32), d["y"].astype(np.int64)
        if p.shape != (y.size, N_ALL):
            raise SystemExit(f"FATAL: {f}: probabilities {p.shape} for {y.size} jets")
        r = confusion_rates(p, y)
        w += (r + r.T) / 2 / len(REFERENCE)
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    iu = np.triu_indices(N_RES, 1)
    total = w[iu].sum()
    out = {"script": "experiments/EVAL/confusion_share.py", "inputs_sha256": inputs, "pass_rho": PASS_RHO,
           "groupings": {}}
    for name, (path, col, runs) in GROUPINGS.items():
        g = group_of(path, col)[:N_RES]
        apart = g[iu[0]] != g[iu[1]]
        accs = []
        for run in runs:
            fits = json.loads((a.root / FITS.format(run=run)).read_text())
            accs.append(float(fits["cells"]["best70"]["jc2bal"]["1000000"]["accuracy"]))
        out["groupings"][name] = {"share_apart": float(w[iu][apart].sum() / total),
                                  "pairs_apart": float(apart.mean()), "accuracy": float(np.mean(accs)),
                                  "accuracy_runs": dict(zip(runs, accs))}
    s = [v["share_apart"] for v in out["groupings"].values()]
    acc = [v["accuracy"] for v in out["groupings"].values()]
    out["spearman"] = spearman(s, acc)
    out["pass"] = out["spearman"] >= PASS_RHO
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("spearman", "pass")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
