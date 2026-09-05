#!/usr/bin/env python3
"""Cross-domain flavour probe: is axis B (QCD_bb vs QCD_cc) axis A (X_bb vs
X_cc) measured out of domain?

Reproduces experiments/EVAL/probe.py's split and linear-probe protocol
exactly (SPLIT_SEED, 60/20/20, StandardScaler + LogisticRegression with C
chosen on the validation split), so the in-domain AUCs must reproduce the
published probe_bvc_v2 numbers. Then each fitted probe is applied unchanged to
the OTHER domain's test split, and the raw-space weight directions are compared.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/Users/raunavmendiratta/transferlearningsophon")
from src.stats.bootstrap import event_bootstrap, ci  # noqa: E402

S = pathlib.Path(__file__).resolve().parent
ARMS = {"L162_s1b": "mtx-l162-s1b", "R16_Q1_s2": "mtx-r16q1-s2",
        "R16_Q1_s3": "mtx-r16q1-s3", "R16_Q1_s4": "mtx-r16q1-s4"}
TASKS = {"A_resonant": (0, 1), "B_qcd": (169, 181)}  # (signal, background)
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SPLIT_SEED = 20260822
B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def make_splits(n, rng_seed=SPLIT_SEED):
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return perm[:a], perm[a:b], perm[b:]


data = {a: np.load(S / f"{d}.npz") for a, d in ARMS.items()}
L = data["L162_s1b"]["L"]
rows = data["L162_s1b"]["rows"]
for a, z in data.items():
    assert np.array_equal(z["L"], L) and np.array_equal(z["rows"], rows), a
print(f"{len(ARMS)} arms row-aligned, {L.shape[0]:,} flavour-axis jets")

# task index sets, in the same order probe.py would produce them
task = {}
for t, (sig, bkg) in TASKS.items():
    idx = np.where((L == sig) | (L == bkg))[0]
    y = (L[idx] == sig).astype(np.int64)
    tr, va, te = make_splits(idx.size)
    task[t] = {"idx": idx, "y": y, "tr": tr, "va": va, "te": te}
    print(f"{t}: {idx.size:,} jets ({y.sum():,} signal), "
          f"split {tr.size:,}/{va.size:,}/{te.size:,}")

# fit one linear probe per (arm, task), keep scaler + clf
fits = {}
for a in ARMS:
    X_all = data[a]["X"].astype(np.float64)
    for t, T in task.items():
        X = X_all[T["idx"]]
        y = T["y"]
        sc = StandardScaler().fit(X[T["tr"]])
        best, best_auc, best_C = None, -1.0, None
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
            clf.fit(sc.transform(X[T["tr"]]), y[T["tr"]])
            auc = roc_auc_score(y[T["va"]], clf.decision_function(sc.transform(X[T["va"]])))
            if auc > best_auc:
                best, best_auc, best_C = clf, auc, C
        # raw-space direction: w_std / sigma
        w_raw = best.coef_[0] / sc.scale_
        fits[(a, t)] = {"sc": sc, "clf": best, "C": best_C, "w_raw": w_raw}

# score every (fit_task -> eval_task) on the eval task's TEST split
scores = {}   # (a, fit_t, eval_t) -> test scores
auc = {}
for a in ARMS:
    for ft in TASKS:
        for et in TASKS:
            T = task[et]
            X = data[a]["X"].astype(np.float64)[T["idx"]][T["te"]]
            f = fits[(a, ft)]
            s = f["clf"].decision_function(f["sc"].transform(X))
            scores[(a, ft, et)] = s
            auc[(a, ft, et)] = float(roc_auc_score(T["y"][T["te"]], s))

print("\n=== AUC on the eval task's test split (rows: probe fitted on) ===")
print(f"{'arm':10s} {'fit on':12s} {'-> A test':>10s} {'-> B test':>10s}"
      f"  {'log(1-AUC) A':>13s} {'B':>8s}")
for a in ARMS:
    for ft in TASKS:
        aA, aB = auc[(a, ft, "A_resonant")], auc[(a, ft, "B_qcd")]
        print(f"{a:10s} {ft:12s} {aA:10.5f} {aB:10.5f}  "
              f"{np.log(1-aA):13.3f} {np.log(1-aB):8.3f}")

print("\n=== cosine between raw-space directions w_A and w_B, per arm ===")
cos = {}
for a in ARMS:
    wA, wB = fits[(a, "A_resonant")]["w_raw"], fits[(a, "B_qcd")]["w_raw"]
    cos[a] = float(wA @ wB / (np.linalg.norm(wA) * np.linalg.norm(wB)))
    print(f"{a:10s} cos(w_A, w_B) = {cos[a]:+.4f}   "
          f"C_A={fits[(a,'A_resonant')]['C']}  C_B={fits[(a,'B_qcd')]['C']}")

# cross-arm cosines: does L162's A direction look like each R16 seed's?
print("\n=== cross-arm cosines are meaningless (different feature bases); skipped ===")

# transfer efficiency: fraction of in-domain log(1-AUC) reached out-of-domain
print("\n=== transfer: out-of-domain log(1-AUC) minus in-domain log(1-AUC) (0 = perfect transfer) ===")
for a in ARMS:
    dAB = np.log(1 - auc[(a, "A_resonant", "B_qcd")]) - np.log(1 - auc[(a, "B_qcd", "B_qcd")])
    dBA = np.log(1 - auc[(a, "B_qcd", "A_resonant")]) - np.log(1 - auc[(a, "A_resonant", "A_resonant")])
    print(f"{a:10s} A-probe on B: {dAB:+.3f}    B-probe on A: {dBA:+.3f}")


# paired bootstrap arm contrasts, L162 minus R16 seed, on each (fit, eval) cell
def contrast(a, b, ft, et):
    T = task[et]
    yy = T["y"][T["te"]]
    sa, sb = scores[(a, ft, et)], scores[(b, ft, et)]

    def stat(r):
        return (np.log(max(1 - roc_auc_score(yy[r], sa[r]), 1e-12))
                - np.log(max(1 - roc_auc_score(yy[r], sb[r]), 1e-12)))
    point = stat(np.arange(yy.size))
    dist = event_bootstrap(np.arange(yy.size), stat, b=B_BOOT)
    lo, hi = ci(dist)
    return float(point), float(lo), float(hi)


print(f"\n=== paired bootstrap ({B_BOOT} resamples): dlog(1-AUC), L162 minus R16 seed ===")
out = {"auc": {f"{a}|{ft}->{et}": v for (a, ft, et), v in auc.items()},
       "cos_wA_wB": cos, "contrasts": {}}
for ft in TASKS:
    for et in TASKS:
        print(f"-- probe fitted on {ft}, evaluated on {et} test --")
        for b in [x for x in ARMS if x != "L162_s1b"]:
            p, lo, hi = contrast("L162_s1b", b, ft, et)
            out["contrasts"][f"{ft}->{et}|L162-{b}"] = [p, lo, hi]
            print(f"   L162 - {b}: {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

(S / "xdomain_results.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {S/'xdomain_results.json'}")
