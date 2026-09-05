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

The figure MUST be made in p_T bins. Decorrelation is trained at fixed p_T on a
flat (p_T, m_SD) stream; on the natural falling test spectrum m and p_T are
correlated and the tagger's efficiency varies with p_T, so integrating over p_T
reintroduces mass dependence (low-mass QCD is low-p_T: suppressed at the low
end, enhanced at the high end). The inclusive run of 2026-09-05 showed exactly
that -- chi2/ndf 99 with a 0.61 lowest bin -- and it is a property of the
figure, not the model. DP-2026-104 Fig. 7 is p_T-binned. The threshold is set
per p_T bin on that bin's QCD jets; the ratio is normalised within the bin, so
flatness does not depend on that choice, but per-bin thresholds keep n_selected
comparable across bins. A single bin over the whole window reproduces the
inclusive figure.
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
# DP-2026-104 shows 400-600 and 1000-1500; extended to cover 200 < p_T < 2500.
PT_EDGES = [200.0, 400.0, 600.0, 1000.0, 1500.0, 2500.0]
# A working point with fewer selected jets than this cannot support a 50-bin
# ratio: its chi2 is noise and must not enter the gate.
MIN_SELECTED = 500


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


def spectrum(mq: np.ndarray, Dq: np.ndarray, edges: np.ndarray) -> dict:
    """The ratio-to-inclusive panel at each EPS_B for ONE set of QCD jets.

    The threshold is a quantile of Dq, i.e. a background efficiency defined on
    exactly the jets passed in -- so calling this per p_T bin gives per-bin
    working points. The ratio is normalised to its mean, so a flat (unsculpted)
    spectrum sits at 1.0 regardless of efficiency. Bins with no inclusive entry
    are nan, not 0: an empty bin is missing information, not a measured zero.
    """
    incl, _ = np.histogram(mq, bins=edges)
    out = {"inclusive_counts": incl.tolist(), "working_points": {}}
    for eps in EPS_B:
        # Threshold defined on QCD ONLY -- this is a background efficiency.
        thr = float(np.quantile(Dq, 1.0 - eps)) if Dq.size else float("nan")
        sel = Dq >= thr
        cnt, _ = np.histogram(mq[sel], bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(incl > 0, cnt / np.maximum(incl, 1), np.nan)
        ratio = ratio / np.nanmean(ratio) if np.isfinite(np.nanmean(ratio)) else ratio
        # Poisson error on the ratio, propagated from both counts.
        with np.errstate(divide="ignore", invalid="ignore"):
            err = ratio * np.sqrt(np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), np.nan)
                                  + 1.0 / np.maximum(incl, 1))
        finite = np.isfinite(ratio) & np.isfinite(err) & (err > 0)
        # Flatness as a number, so the gate is decided rather than eyeballed.
        chi2 = float(np.sum(((ratio[finite] - 1.0) / err[finite]) ** 2))
        ndf = int(finite.sum() - 1)
        out["working_points"][f"eps_B={eps}"] = {
            "threshold": thr, "n_selected": int(sel.sum()),
            "thin": bool(sel.sum() < MIN_SELECTED),
            "counts": cnt.tolist(),
            "ratio": np.where(np.isfinite(ratio), ratio, None).tolist(),
            "ratio_err": np.where(np.isfinite(err), err, None).tolist(),
            "chi2_vs_flat": chi2, "ndf": ndf,
            "chi2_per_ndf": chi2 / ndf if ndf > 0 else None,
        }
    return out


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
    ap.add_argument("--pt-branch", default="jet_pt")
    ap.add_argument("--pt-edges", type=float, nargs="+", default=PT_EDGES,
                    help="p_T bin edges; a single [lo, hi] pair reproduces the inclusive figure")
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

    if args.pt_branch not in obs:
        raise SystemExit(f"FATAL: {args.pt_branch} not in {sorted(obs)}")
    pt = obs[args.pt_branch].astype(np.float64)[keep]
    edges = np.linspace(lo, hi, args.bins + 1)
    pt_edges = np.asarray(args.pt_edges, dtype=np.float64)
    if pt_edges.size < 2 or np.any(np.diff(pt_edges) <= 0):
        raise SystemExit(f"FATAL: --pt-edges must be >= 2 increasing values, got {args.pt_edges}")

    results = {
        "n_qcd_in_window": int(mq.size),
        "n_total": int(L.shape[0]),
        "signal_labels": sig.tolist(),
        "n_qcd_classes": int(qcd.size),
        "mass_branch": args.mass_branch,
        "pt_branch": args.pt_branch,
        "bin_edges": edges.tolist(),
        "pt_edges": pt_edges.tolist(),
        "min_selected": MIN_SELECTED,
        "pt_bins": [],
    }

    print(f"{mq.size:,} QCD jets in [{lo}, {hi}]  ({qcd.size} QCD classes)")
    for plo, phi in zip(pt_edges[:-1], pt_edges[1:]):
        inbin = (pt >= plo) & (pt < phi)
        wp = spectrum(mq[inbin], Dq[inbin], edges)
        results["pt_bins"].append({"pt_lo": float(plo), "pt_hi": float(phi),
                                   "n_qcd": int(inbin.sum()), **wp})
        print(f"\np_T in [{plo:.0f}, {phi:.0f}) : {int(inbin.sum()):,} QCD jets")
        for k, v in wp["working_points"].items():
            flag = "  THIN -- excluded from the gate" if v["thin"] else ""
            c = v["chi2_per_ndf"]
            print(f"  {k:<12} n={v['n_selected']:>8,}  chi2/ndf vs flat = "
                  f"{c:.2f}{flag}" if c is not None else f"  {k:<12} n={v['n_selected']:>8,}  (no ndf)")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_control.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'e1_control.json'}")
    worst = max((v["chi2_per_ndf"] for b in results["pt_bins"]
                 for v in b["working_points"].values()
                 if not v["thin"] and v["chi2_per_ndf"] is not None), default=None)
    print(f"GATE: worst chi2/ndf over non-thin p_T-binned working points = "
          f"{worst:.2f}" if worst is not None else "GATE: no non-thin working point")
    print("Ratio flat within uncertainty => PASS. Read chi2/ndf against the bin "
          "count -- a large chi2 on 50 bins with millions of jets can still be "
          "a small sculpting effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
