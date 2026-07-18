#!/usr/bin/env python3
"""Phase A §6-A.5(b): test floor σ_test,Δ from the E1 proxy pair (G0 artifact).

Scores independently-seeded JC1-trained E1 checkpoints on the frozen JC2 eval
split via the P0-construction linear probe (§11.3: standardized L2 logistic on
g10sem, trained on release-train embeddings, C selected on release-val), then
event-bootstraps (B=2000, seed 7777) the paired difference of log-rejection.

Consumes the cached E8 embeddings (extract_jc2_embeddings.py output) — the E8
extraction IS the required scoring pass, so this costs no GPU time.

Statistic constructions recorded (PLAN §12 lists eps_S=0.50 with FPR grid
{1e-4, 1e-3}; both readings are recorded so the estimand sheet can bind one):
  A. d = log R_A - log R_B,  R = 1/FPR at fixed signal efficiency 0.5
  B. d_TPR(WP) = log TPR_A - log TPR_B at fixed FPR = WP in {1e-3, 1e-4}
Primary proxy pair: arms_s1 vs arms_s2 (same arm, independent seeds).
Robustness pairs: (arms_s1,arms_s3), (arms_s2,arms_s3), (armp_s1,armp_s3).

No per-event UIDs exist in the release -> each jet is its own resampling unit
(recorded; refresh on the real pair post-unblinding per §16.4).

Run (in-cluster, after e8-extract-probe):
  python3 scripts/phase_a/test_floor.py --emb-dir /data/results/e8/emb \
      --out /data/results/phase_a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.E8.probe_transfer import (  # noqa: E402
    N_VAL_SEL, TRAIN_SEED, VAL_SEED, fit_linear, load_g10sem, load_split, subsample)
from src.stats.bootstrap import ci, event_bootstrap  # noqa: E402

B_BOOT = 2000
BOOT_SEED = 7777
QCD = 9
PROBE_TRAIN_SIZE = 1_000_000          # balanced top rung = P0 probe training set
PRIMARY_PAIR = ("arms_s1", "arms_s2")
ROBUSTNESS_PAIRS = [("arms_s1", "arms_s3"), ("arms_s2", "arms_s3"),
                    ("armp_s1", "armp_s3")]
FULL_TEST_JETS = 33_500_000           # 335 files x 100k — for the sigma projection


def probe_scores(emb_dir, backbone, remap):
    """P0-construction probe -> signal score (1 - P(QCD)) on the FULL test split."""
    Xtr, ytr = load_split(emb_dir, backbone, "train", remap)
    Xva, yva = load_split(emb_dir, backbone, "val", remap)
    Xte, yte = load_split(emb_dir, backbone, "test", remap)
    tr = subsample(ytr, PROBE_TRAIN_SIZE, 10, np.random.default_rng(TRAIN_SEED))
    va = (subsample(yva, N_VAL_SEL, 10, np.random.default_rng(VAL_SEED))
          if len(yva) > N_VAL_SEL else np.arange(len(yva)))
    sc, clf, C = fit_linear(Xtr[tr], ytr[tr], Xva[va], yva[va])
    prob = clf.predict_proba(sc.transform(Xte))
    print(f"  {backbone}: probe C={C}, n_train={len(tr):,}, n_test={len(yte):,}")
    return 1.0 - prob[:, QCD], yte


def log_rej_at_effs(score, is_sig, eff=0.5):
    thr = np.quantile(score[is_sig], 1.0 - eff)
    fpr = float(np.mean(score[~is_sig] >= thr))
    return np.log(1.0 / fpr) if fpr > 0 else np.inf


def log_tpr_at_fpr(score, is_sig, wp):
    thr = np.quantile(score[~is_sig], 1.0 - wp)
    tpr = float(np.mean(score[is_sig] >= thr))
    return np.log(tpr) if tpr > 0 else -np.inf


def paired_stat(sa, sb, y, kind, wp=None):
    is_sig = y != QCD

    def stat(rows):
        ys = is_sig[rows]
        if kind == "R_eps0.5":
            return (log_rej_at_effs(sa[rows], ys) - log_rej_at_effs(sb[rows], ys))
        return (log_tpr_at_fpr(sa[rows], ys, wp) - log_tpr_at_fpr(sb[rows], ys, wp))

    point = stat(np.arange(len(y)))
    dist = event_bootstrap(np.arange(len(y)), stat, b=B_BOOT, seed=BOOT_SEED)
    fin = dist[np.isfinite(dist)]
    lo, hi = ci(fin)
    return {"point": float(point), "sigma": float(fin.std(ddof=1)),
            "ci95": [lo, hi], "n_boot_finite": int(len(fin))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    remap = load_g10sem()
    backbones = sorted({b for p in [PRIMARY_PAIR] + ROBUSTNESS_PAIRS for b in p})
    scores, y = {}, None
    for b in backbones:
        scores[b], y = probe_scores(args.emb_dir, b, remap)

    n_qcd = int((y == QCD).sum())
    res = {"n_test": int(len(y)), "n_qcd_test": n_qcd,
           "expected_qcd_survivors": {"1e-3": n_qcd * 1e-3, "1e-4": n_qcd * 1e-4},
           "probe": "P0 construction (linear, train=release-train 1M balanced, C on val)",
           "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                         "unit": "jet (no event UIDs in release — documented)"},
           "pairs": {}}

    for pair in [PRIMARY_PAIR] + ROBUSTNESS_PAIRS:
        a, b = pair
        tag = f"{a}|{b}"
        primary = pair == PRIMARY_PAIR
        print(f"== pair {tag}{' (PRIMARY)' if primary else ''} ==")
        entry = {"primary": primary,
                 "R_eps0.5": paired_stat(scores[a], scores[b], y, "R_eps0.5")}
        if primary:
            for wp in (1e-3, 1e-4):
                entry[f"TPR@FPR{wp:g}"] = paired_stat(scores[a], scores[b], y, "tpr", wp)
        res["pairs"][tag] = entry
        print(f"   d={entry['R_eps0.5']['point']:+.4f}  sigma={entry['R_eps0.5']['sigma']:.4f}")

    prim = res["pairs"][f"{PRIMARY_PAIR[0]}|{PRIMARY_PAIR[1]}"]
    scale = float(np.sqrt(len(y) / FULL_TEST_JETS))
    res["sigma_test_delta"] = {
        "R_eps0.5": prim["R_eps0.5"]["sigma"],
        "TPR@FPR0.001": prim.get("TPR@FPR0.001", {}).get("sigma"),
        "TPR@FPR0.0001": prim.get("TPR@FPR0.0001", {}).get("sigma"),
        "projection_full_partition": {
            "factor_sqrt_N": scale,
            "R_eps0.5_projected": prim["R_eps0.5"]["sigma"] * scale,
            "note": ("sigma ~ 1/sqrt(N_eval) projection to the full 335-file "
                     "partition; refresh on the real pair post-unblinding (§16.4)")}}

    (out / "test_floor.json").write_text(json.dumps(res, indent=2))
    lines = ["# Phase A test floor σ_test,Δ — A.5(b)  (G0 evidence)", "",
             f"- eval split: E8-extracted test files, {len(y):,} jets ({n_qcd:,} QCD)",
             f"- primary pair {PRIMARY_PAIR}: σ(R@εS=0.5) = {prim['R_eps0.5']['sigma']:.4f} "
             f"(log-rejection units), d = {prim['R_eps0.5']['point']:+.4f}, "
             f"CI95 {prim['R_eps0.5']['ci95']}"]
    for wp in ("0.001", "0.0001"):
        k = f"TPR@FPR{wp}"
        if k in prim:
            lines.append(f"- {k}: σ = {prim[k]['sigma']:.4f}, d = {prim[k]['point']:+.4f}, "
                         f"CI95 {prim[k]['ci95']}")
    lines += [f"- projected σ on the full partition: "
              f"{res['sigma_test_delta']['projection_full_partition']['R_eps0.5_projected']:.4f}",
              "- robustness pairs in test_floor.json", ""]
    (out / "test_floor.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
