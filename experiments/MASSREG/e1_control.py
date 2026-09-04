#!/usr/bin/env python3
"""E1 -- the mass-regression paper's CONTROL figure.

WHAT IT SHOWS, AND WHY IT COMES FIRST
-------------------------------------
Before we can claim the class-gated mass ESTIMATOR sculpts the QCD spectrum, we
have to show our pipeline reproduces the behaviour GloParT already publishes for
its DISCRIMINANT: cutting harder on D_S does NOT sculpt QCD m_SD, because the
discriminant is mass-decorrelated by construction (flat reweighting over
jet_pt x jet_sdmass). That is DP-2026-104 Fig. 7 / thesis Fig. 11.16.

If our version of their figure does not come out flat, the disagreement is in
OUR pipeline and every downstream number is worthless. So this is a gate, not a
result: PASS = ratio flat within uncertainty; FAIL = stop and diagnose.

Note the asymmetry the paper turns on. Decorrelation is applied to the
discriminant. It is NOT applied to the regressed mass, which is a different
head with a different target -- see docs/LIT_MASS_REGRESSION.md section 7. This
figure establishes the first half; E2/E3 measure the second.

THE DISCRIMINANT
----------------
GloParT's own definition (DP-2026-104 p.8):

    D_S = sum_{i in S} a_i p_i / ( sum_{i in S} a_i p_i + sum_{i in QCD} p_i )

The QCD index set is READ FROM configs/labelmaps/rung_label_maps.v1.csv rather
than hardcoded as range(161, 188). Hardcoding is how "161 resonant + 27 QCD"
silently becomes wrong if the map is ever regenerated, and the map is the
committed artefact the rest of the study already trusts.

Usage:
    python3 experiments/MASSREG/e1_control.py \
        --features /data/results/eval/sophon-public/features_massreg \
        --out /data/results/massreg/e1_control
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
LABEL_MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
# The efficiencies DP-2026-104 Fig. 7 plots, plus the inclusive reference.
EPS_B = [0.05, 0.01, 0.005]


def qcd_indices(path: pathlib.Path = LABEL_MAP) -> np.ndarray:
    """The native labels that are QCD, from the committed map."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    idx = [int(r["jet_label"]) for r in rows if r["class_name"].startswith("label_QCD_")]
    if not idx:
        raise SystemExit(f"FATAL: no label_QCD_* rows in {path}")
    return np.asarray(sorted(idx), dtype=np.int64)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def discriminant(p: np.ndarray, sig: np.ndarray, qcd: np.ndarray,
                 alpha: np.ndarray | None = None) -> np.ndarray:
    """D_S per DP-2026-104 p.8. alpha defaults to uniform over the signal set."""
    a = np.ones(sig.size) if alpha is None else np.asarray(alpha, dtype=np.float64)
    if a.size != sig.size:
        raise SystemExit(f"FATAL: alpha has {a.size} entries for {sig.size} signal classes")
    num = (p[:, sig] * a[None, :]).sum(axis=1)
    den = num + p[:, qcd].sum(axis=1)
    # den == 0 needs a real guard: it is not a rounding artefact but a jet the
    # model put entirely outside signal-and-QCD, and 0/0 would come back as nan
    # and silently drop out of every histogram below.
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="dir holding logits.npy, label188.npy, observers.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signal", nargs="+", type=int, default=None,
                    help="native label ids of the signal set (default: label_X_bb=0)")
    ap.add_argument("--mass-branch", default="jet_sdmass")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--mass-range", type=float, nargs=2, default=[20.0, 500.0],
                    help="the study's own selection window")
    args = ap.parse_args()

    d = pathlib.Path(args.features)
    logits = np.load(d / "logits.npy")
    L = np.load(d / "label188.npy")
    obs = np.load(d / "observers.npz")
    if args.mass_branch not in obs:
        raise SystemExit(f"FATAL: {args.mass_branch} not in {sorted(obs)}")
    m = obs[args.mass_branch].astype(np.float64)

    if not (logits.shape[0] == L.shape[0] == m.shape[0]):
        raise SystemExit(f"FATAL: row mismatch logits {logits.shape[0]} "
                         f"label {L.shape[0]} mass {m.shape[0]}")

    qcd = qcd_indices()
    sig = np.asarray(args.signal if args.signal else [0], dtype=np.int64)
    if np.intersect1d(sig, qcd).size:
        raise SystemExit("FATAL: signal set overlaps the QCD set")

    p = softmax(logits.astype(np.float64))
    D = discriminant(p, sig, qcd)

    is_qcd = np.isin(L, qcd)
    # The spectrum is measured on QCD jets INSIDE the selection window; the
    # window is part of the study's definition, not a plotting choice.
    lo, hi = args.mass_range
    keep = is_qcd & (m > lo) & (m < hi)
    mq, Dq = m[keep], D[keep]
    if mq.size < 1000:
        raise SystemExit(f"FATAL: only {mq.size} QCD jets in window; too few")

    edges = np.linspace(lo, hi, args.bins + 1)
    incl, _ = np.histogram(mq, bins=edges)

    results = {
        "n_qcd_in_window": int(mq.size),
        "n_total": int(L.shape[0]),
        "signal_labels": sig.tolist(),
        "n_qcd_classes": int(qcd.size),
        "mass_branch": args.mass_branch,
        "bin_edges": edges.tolist(),
        "inclusive_counts": incl.tolist(),
        "working_points": {},
    }

    print(f"{mq.size:,} QCD jets in [{lo}, {hi}]  ({qcd.size} QCD classes)")
    for eps in EPS_B:
        # Threshold defined on QCD ONLY -- this is a background efficiency.
        thr = float(np.quantile(Dq, 1.0 - eps))
        sel = Dq >= thr
        cnt, _ = np.histogram(mq[sel], bins=edges)
        # Ratio to inclusive, normalised so a flat (unsculpted) spectrum sits at
        # 1.0 regardless of the efficiency. Bins with no inclusive entry are nan,
        # not 0 -- an empty bin is missing information, not a measured zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(incl > 0, cnt / np.maximum(incl, 1), np.nan)
        ratio = ratio / np.nanmean(ratio)
        # Poisson error on the ratio, propagated from both counts.
        with np.errstate(divide="ignore", invalid="ignore"):
            err = ratio * np.sqrt(np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), np.nan)
                                  + 1.0 / np.maximum(incl, 1))
        finite = np.isfinite(ratio) & np.isfinite(err) & (err > 0)
        # Flatness as a number, so the gate is decided rather than eyeballed.
        chi2 = float(np.sum(((ratio[finite] - 1.0) / err[finite]) ** 2))
        ndf = int(finite.sum() - 1)
        results["working_points"][f"eps_B={eps}"] = {
            "threshold": thr, "n_selected": int(sel.sum()),
            "counts": cnt.tolist(), "ratio": np.where(np.isfinite(ratio), ratio, None).tolist(),
            "ratio_err": np.where(np.isfinite(err), err, None).tolist(),
            "chi2_vs_flat": chi2, "ndf": ndf,
            "chi2_per_ndf": chi2 / ndf if ndf > 0 else None,
        }
        print(f"  eps_B={eps:<6} thr={thr:.5f}  n={int(sel.sum()):>8,}  "
              f"chi2/ndf vs flat = {chi2/ndf:.2f}" if ndf > 0 else "")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_control.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'e1_control.json'}")
    print("GATE: ratio flat within uncertainty => PASS. Read chi2_per_ndf, and "
          "read it against the bin count -- a large chi2 on 50 bins with "
          "millions of jets can still be a small sculpting effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
