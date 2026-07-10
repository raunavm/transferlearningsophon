#!/usr/bin/env python3
"""E1 evaluation: metrics + preprocessing tax from weaver pred.root files.

Point estimates on the FULL test set; bootstrap CIs on a stratified subsample
(cap for tractability — the 1e-4 TPR CI is inherently noisy, reported as-is).
Tax = Arm P - Arm S, paired (same resampled indices), with bootstrap CIs.

Gate reference (external line only): published ParT macro AUC = 0.9877
(arXiv:2202.03772 Table 1). VALIDATED if Arm P macro AUC within ~0.002.

NOTE: branch names are auto-detected; confirm against the smoke pred_smoke.root
(score_* branches + the 10 label_* one-hots) before trusting the full run.
"""
import argparse
import json
import os

import numpy as np
import uproot
from sklearn.metrics import roc_auc_score, roc_curve

LABELS = ["label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
          "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"]
NAMES = [l.replace("label_", "") for l in LABELS]
QCD = 0
PUBLISHED_PART_AUC = 0.9877


def load(path):
    """Return probs (N,10) row-normalized, truth (N,) int in [0,10)."""
    with uproot.open(path) as f:
        tree = f[f.keys()[0].split(";")[0]]
        keys = [k.split(";")[0] for k in tree.keys()]
        score_keys = [f"score_{l}" for l in LABELS]
        if not all(k in keys for k in score_keys):  # fall back to any score_* in order
            sc = sorted(k for k in keys if k.startswith("score_"))
            assert len(sc) == 10, f"expected 10 score_ branches, found {sc}"
            score_keys = sc
        arr = tree.arrays(score_keys + [l for l in LABELS if l in keys], library="np")
    probs = np.stack([arr[k] for k in score_keys], axis=1).astype(np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(1, keepdims=True)
    truth = np.argmax(np.stack([arr[l] for l in LABELS], axis=1), axis=1)
    return probs, truth


def macro_auc(probs, truth):
    return float(roc_auc_score(truth, probs, multi_class="ovr", average="macro",
                               labels=list(range(10))))


def per_class_auc(probs, truth):
    out = {}
    for k in range(10):
        yb = (truth == k).astype(int)
        out[NAMES[k]] = float(roc_auc_score(yb, probs[:, k])) if yb.sum() else float("nan")
    return out


def _sig_vs_qcd(probs, truth, s):
    """Discriminant p_S/(p_S+p_QCD) on jets in {S, QCD}; returns fpr, tpr sorted."""
    m = (truth == s) | (truth == QCD)
    d = probs[m, s] / (probs[m, s] + probs[m, QCD])
    y = (truth[m] == s).astype(int)
    fpr, tpr, _ = roc_curve(y, d)
    return fpr, tpr


def rejection_at_eff(probs, truth, s, eff):
    fpr, tpr = _sig_vs_qcd(probs, truth, s)
    f = np.interp(eff, tpr, fpr)
    return float(1.0 / max(f, 1e-12))


def tpr_at_fpr(probs, truth, s, fpr_t):
    fpr, tpr = _sig_vs_qcd(probs, truth, s)
    m = fpr <= fpr_t
    return float(tpr[m].max()) if m.any() else 0.0


def full_metrics(probs, truth):
    acc = float((probs.argmax(1) == truth).mean())
    rej05 = {NAMES[s]: rejection_at_eff(probs, truth, s, 0.5) for s in range(1, 10)}
    rej03 = {NAMES[s]: rejection_at_eff(probs, truth, s, 0.3) for s in range(1, 10)}
    tpr3 = {NAMES[s]: tpr_at_fpr(probs, truth, s, 1e-3) for s in range(1, 10)}
    tpr4 = {NAMES[s]: tpr_at_fpr(probs, truth, s, 1e-4) for s in range(1, 10)}
    return {
        "accuracy": acc,
        "macro_auc_ovr": macro_auc(probs, truth),
        "per_class_auc": per_class_auc(probs, truth),
        "rejection_vs_qcd_eff0.5": rej05,
        "rejection_vs_qcd_eff0.3": rej03,
        "mean_rejection_eff0.3": float(np.mean(list(rej03.values()))),
        "tpr_at_fpr_1e-3_vs_qcd": tpr3,
        "tpr_at_fpr_1e-4_vs_qcd": tpr4,
    }


def boot_indices(truth, n_boot, cap, seed=0):
    """Stratified subsample (<=cap), then n_boot resample-with-replacement index sets."""
    rng = np.random.default_rng(seed)
    if len(truth) > cap:
        per = cap // 10
        idx = np.concatenate([rng.choice(np.where(truth == k)[0], per, replace=False)
                              for k in range(10)])
    else:
        idx = np.arange(len(truth))
    return idx, [rng.choice(idx, len(idx), replace=True) for _ in range(n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-p", required=True)
    ap.add_argument("--arm-s", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--boot-cap", type=int, default=2_000_000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pP, tP = load(args.arm_p)
    pS, tS = load(args.arm_s)
    assert len(tP) == len(tS) and np.array_equal(tP, tS), \
        "Arm P/S test rows differ — bootstrap tax requires identical, aligned test jets"

    mP, mS = full_metrics(pP, tP), full_metrics(pS, tS)
    mP["gate_pass"] = bool(abs(mP["macro_auc_ovr"] - PUBLISHED_PART_AUC) <= 0.002)
    mP["published_part_auc"] = PUBLISHED_PART_AUC
    mP["auc_gap_to_published"] = float(mP["macro_auc_ovr"] - PUBLISHED_PART_AUC)

    # paired bootstrap on shared indices
    idx, boots = boot_indices(tP, args.n_boot, args.boot_cap)
    aucP, aucS, dauc, drej = [], [], [], []
    for b in boots:
        aP, aS = macro_auc(pP[b], tP[b]), macro_auc(pS[b], tS[b])
        rP = np.mean([rejection_at_eff(pP[b], tP[b], s, 0.3) for s in range(1, 10)])
        rS = np.mean([rejection_at_eff(pS[b], tS[b], s, 0.3) for s in range(1, 10)])
        aucP.append(aP); aucS.append(aS); dauc.append(aP - aS)
        drej.append(np.log10(rP) - np.log10(rS))

    def ci(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x)),
                "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}

    tax = {
        "n_boot": args.n_boot, "boot_n": int(len(idx)),
        "arm_p_macro_auc": ci(aucP), "arm_s_macro_auc": ci(aucS),
        "tax_macro_auc_P_minus_S": ci(dauc),
        "tax_log10_mean_rejection03_P_minus_S": ci(drej),
    }

    json.dump(mP, open(f"{args.out_dir}/metrics_arm_p.json", "w"), indent=2)
    json.dump(mS, open(f"{args.out_dir}/metrics_arm_s.json", "w"), indent=2)
    json.dump(tax, open(f"{args.out_dir}/tax.json", "w"), indent=2)

    print(f"Arm P  macro AUC = {mP['macro_auc_ovr']:.4f}  (published {PUBLISHED_PART_AUC}, "
          f"gap {mP['auc_gap_to_published']:+.4f})  GATE {'PASS' if mP['gate_pass'] else 'FAIL'}")
    print(f"Arm S  macro AUC = {mS['macro_auc_ovr']:.4f}")
    print(f"Tax (P-S) macro AUC = {tax['tax_macro_auc_P_minus_S']['mean']:+.4f} "
          f"CI95 {tax['tax_macro_auc_P_minus_S']['ci95']}")
    print(f"Arm P acc={mP['accuracy']:.4f} meanRej@0.3={mP['mean_rejection_eff0.3']:.1f} | "
          f"Arm S acc={mS['accuracy']:.4f} meanRej@0.3={mS['mean_rejection_eff0.3']:.1f}")


if __name__ == "__main__":
    main()
