#!/usr/bin/env python3
"""Score one arm's pred.root files and measure its PRETRAINING-SEED variance.

WHY THIS EXISTS SEPARATELY FROM experiments/E1/eval_e1.py
---------------------------------------------------------
eval_e1.py hardcodes JetClass-I's 10 classes and its label names. The arms here
have K = 17 or 162 and a `type: custom` label block with a single truth_label
expression, so none of that transfers. This reads whatever weaver actually
wrote.

WHAT IT IS FOR
--------------
Two things, and the second is the reason it was written now:

  1. docs/GATES.md requires cached ROC arrays -- "a run without cached ROC
     arrays is a failed run even if its AUC is right".

  2. sigma_pretrain FOR THIS ARM, MEASURED. DECISIONS_PENDING item 13 turns on
     whether L162's single seed can be reported against R16_Q1's spread. The
     only sigma_pretrain in the project (0.003704) comes from E1's Arm S -- a
     different arm, a different dataset, a different K. Handing several seeds
     of ONE arm to this script replaces that borrowed number with a measured
     one. Point estimates are on the full test set; the seed spread is over
     runs, not over jets, and the two must never be added in quadrature
     without saying so.

CROSS-ARM COMPARISON IS NOT POSSIBLE HERE, BY CONSTRUCTION
----------------------------------------------------------
L162 emits 162 scores and R16_Q1 emits 17, over different partitions of the
same 188 native labels. Their accuracies are not on the same scale and their
macro AUCs average over different class counts. The controlled cross-arm
endpoint is the frozen probe (experiments/EVAL/probe.py), which is defined on
NATIVE labels and is therefore identical for every arm. Comparing two numbers
out of this script across arms is the single easiest mistake to make with it.

Usage:
    python3 experiments/EVAL/eval_arm.py --k 17 --arm R16_Q1 \
        --pred s2=/data/results/mtx/mtx-r16q1-s2/eval/pred.root \
               s3=/data/results/mtx/mtx-r16q1-s3/eval/pred.root \
        --out /data/results/eval/arm_r16q1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

EPS_S = 0.5  # signal efficiency at which rejection is quoted


def _order_key(name: str) -> tuple:
    """Sort score branches by their trailing integer when they have one.

    Lexicographic order puts score_10 before score_2, which would permute the
    class axis silently -- every per-class number would then be attributed to
    the wrong class while the macro average stayed identical and looked fine.
    """
    m = re.search(r"(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


def load_pred(path: pathlib.Path, k: int):
    """Return (probs (N,k) row-normalised, truth (N,) int)."""
    import uproot
    with uproot.open(str(path)) as f:
        tree = f[f.keys()[0].split(";")[0]]
        keys = [b.split(";")[0] for b in tree.keys()]
        score = sorted([b for b in keys if b.startswith("score_")], key=_order_key)
        if len(score) != k:
            raise SystemExit(
                f"FATAL: {path} has {len(score)} score_* branches, expected k={k}.\n"
                f"  score branches: {score}\n"
                f"  ALL branches:   {keys}")
        truth_cands = [b for b in keys if b in
                       ("truth_label", "_label_", "label", "y_true")]
        arr = tree.arrays(score + truth_cands, library="np")
    # float32, not float64. The test set is 27.4M jets, so at K=162 the score
    # matrix alone is 17.8 GB in float64 against 8.9 GB in float32, and every
    # metric below is rank-based -- neither AUC nor a quantile threshold can
    # tell the two apart at this precision. In-place ops keep it to one copy.
    probs = np.stack([arr[b] for b in score], axis=1).astype(np.float32)
    np.clip(probs, 1e-12, None, out=probs)
    probs /= probs.sum(axis=1, keepdims=True)
    if not truth_cands:
        raise SystemExit(f"FATAL: no truth branch in {path}; saw {keys}")
    truth = np.asarray(arr[truth_cands[0]]).astype(np.int64).ravel()
    if truth.shape[0] != probs.shape[0]:
        raise SystemExit(f"FATAL: {path} truth {truth.shape} vs probs {probs.shape}")
    return probs, truth


def metrics(probs, truth, k, qcd):
    from sklearn.metrics import roc_auc_score, roc_curve
    present = np.unique(truth)
    acc = float((probs.argmax(1) == truth).mean())
    # Restrict the macro average to classes actually present. roc_auc_score
    # raises on an absent class, and quietly substituting 0.5 for it would drag
    # the macro toward chance by an amount that depends on the test sample.
    #
    # Dropping columns breaks sklearn's sum-to-1 precondition for multiclass
    # OvR, so the surviving columns are renormalised. On a full test set every
    # class is present and this is a no-op; it only bites on a subsample, which
    # is exactly where it would otherwise raise. per_class_auc below uses the
    # RAW columns and is the more primitive number of the two.
    if present.size == probs.shape[1]:
        # The normal case on a full test set: every class occurs, the slice
        # would be an identity copy, and at 27.4M x 162 that copy is 8.9 GB.
        # load_pred already normalised, so check that rather than redo it --
        # but CHECK it, because passing raw scores here would otherwise reach
        # sklearn as a bare "scores must sum to 1" error naming nothing.
        head = probs[: min(1000, probs.shape[0])]
        if not np.allclose(head.sum(axis=1), 1.0, atol=1e-3):
            raise SystemExit("FATAL: probs rows do not sum to 1 -- metrics() "
                             "expects load_pred()'s normalised output")
        sub = probs
    else:
        sub = probs[:, present]
        sub = sub / sub.sum(axis=1, keepdims=True)
    if present.size == 2:
        # sklearn's multiclass path rejects a 2-column score array; binary
        # wants the positive class's column alone.
        macro = float(roc_auc_score((truth == present[1]).astype(int), sub[:, 1]))
    else:
        macro = float(roc_auc_score(truth, sub, multi_class="ovr",
                                    average="macro", labels=present.tolist()))
    per = {}
    for c in present:
        yb = (truth == c).astype(int)
        per[int(c)] = float(roc_auc_score(yb, probs[:, c]))
    rej = {}
    for c in present:
        if c == qcd:
            continue
        m = (truth == c) | (truth == qcd)
        d = probs[m, c] / (probs[m, c] + probs[m, qcd])
        y = (truth[m] == c).astype(int)
        fpr, tpr, _ = roc_curve(y, d)
        f = float(np.interp(EPS_S, tpr, fpr))
        n_bkg = int((truth == qcd).sum())
        # 1/N_bkg is the resolvable floor. Past it the number is a statement
        # about the sample size, not the model, so it is flagged as a bound.
        rej[int(c)] = {"rejection": float(1.0 / max(f, 1e-12)),
                       "eps_b": f,
                       "is_bound": bool(f < 1.0 / max(n_bkg, 1))}
    return {"accuracy": acc, "macro_auc_ovr": macro, "n_classes_present": int(present.size),
            "per_class_auc": per, f"rejection_vs_qcd_eff{EPS_S}": rej}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, metavar="SEED=PATH")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--qcd-index", type=int, default=None,
                    help="default k-1: every rung puts QCD_ALL last "
                         "(configs/labelmaps/rung_label_maps.v1.csv)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qcd = args.k - 1 if args.qcd_index is None else args.qcd_index
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"FATAL: --pred wants SEED=PATH, got {spec!r}")
        seed, path = spec.split("=", 1)
        print(f"--- {seed}: {path}", flush=True)
        probs, truth = load_pred(pathlib.Path(path), args.k)
        m = metrics(probs, truth, args.k, qcd)
        m["n_jets"] = int(truth.shape[0])
        m["pred_path"] = path
        per_seed[seed] = m
        print(f"    n={m['n_jets']:,}  acc={m['accuracy']:.5f}  "
              f"macro AUC={m['macro_auc_ovr']:.5f}", flush=True)

    res = {"arm": args.arm, "k": args.k, "qcd_index": qcd, "eps_s": EPS_S,
           "per_seed": per_seed}

    # sigma_pretrain, on the scale docs/STATISTICS.md does inference on.
    if len(per_seed) >= 2:
        aucs = np.array([v["macro_auc_ovr"] for v in per_seed.values()])
        lg = np.log10(np.maximum(1.0 - aucs, 1e-12))
        accs = np.array([v["accuracy"] for v in per_seed.values()])
        res["seed_spread"] = {
            "n_seeds": int(aucs.size),
            "macro_auc_mean": float(aucs.mean()),
            # ddof=1: these are a SAMPLE of pretraining seeds, not the population.
            "macro_auc_sd": float(aucs.std(ddof=1)),
            "log10_1m_auc_mean": float(lg.mean()),
            "sigma_pretrain_log10_1m_auc": float(lg.std(ddof=1)),
            "accuracy_mean": float(accs.mean()),
            "accuracy_sd": float(accs.std(ddof=1)),
            "accuracy_range": [float(accs.min()), float(accs.max())],
        }
        s = res["seed_spread"]
        print(f"\n{s['n_seeds']} seeds: macro AUC {s['macro_auc_mean']:.5f} "
              f"+/- {s['macro_auc_sd']:.5f}")
        print(f"  sigma_pretrain (log10(1-AUC)) = {s['sigma_pretrain_log10_1m_auc']:.6f}")
        print(f"  accuracy {s['accuracy_mean']:.5f} +/- {s['accuracy_sd']:.5f} "
              f"range {s['accuracy_range'][0]:.5f}..{s['accuracy_range'][1]:.5f}")
        print("  THIS ARM ONLY. Do not reuse it for an arm with a different K.")
    else:
        print("\nonly one seed given -- no seed spread computed")

    (out / "eval_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out/'eval_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
