#!/usr/bin/env python3
"""Reproduce the PUBLISHED bb/cc tagging anchors with OUR evaluation code.

WHY THIS RUNS BEFORE ANY ARM-WISE REJECTION IS QUOTED
-----------------------------------------------------
`docs/PRD_PLAN.md` §6.3 and §8.1: the public Sophon checkpoint must first
reproduce the numbers its own authors published, using OUR discriminant
construction, OUR selection and OUR ROC code. If it does, a later arm-wise
rejection is a measurement. If it does not, every arm-wise rejection inherits
an unknown bias and no amount of internal consistency will reveal it, because
every arm would carry the same bias and the ORDERING could still look sane.

THE ANCHORS -- arXiv:2503.00118 Table A1, Sophon column. QCD background
rejection 1/eps_B at fixed signal efficiency:

    X->bb vs QCD   eps_S = 60 %    300
    X->bb vs QCD   eps_S = 40 %    810
    X->cc vs QCD   eps_S = 60 %    110
    X->cc vs QCD   eps_S = 40 %    320

Table A1 quotes TWO SIGNIFICANT FIGURES, so agreement is judged on a relative
tolerance, never on equality. The default 25 % is deliberately loose: the
published numbers come from the authors' own Delphes sample and selection, and
this check is asking "is our evaluation code right", not "is our test set
theirs". A 2x miss is a bug; a 10 % miss is not.

THE DISCRIMINANT is the one the Sophon paper describes and that
`experiments/MASSREG/e1_control.py` already implements -- one published signal
node against UNDIVIDED QCD:

    D_S = sum_{i in S} p_i / ( sum_{i in S} p_i + sum_{j in QCD} p_j )

The QCD index set is read from `configs/labelmaps/rung_label_maps.v1.csv`, not
hardcoded as range(161, 188), for the reason e1_control records.

Runs on CACHED logits. Zero GPU.

Run:  python3 experiments/EVAL/anchors.py --features /data/results/eval/sophon-public/features_massreg \
        --out /data/results/eval/sophon-public/anchors.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# arXiv:2503.00118 Table A1, Sophon column. (signal class_name, eps_S) -> 1/eps_B
PUBLISHED = {
    ("label_X_bb", 0.60): 300.0,
    ("label_X_bb", 0.40): 810.0,
    ("label_X_cc", 0.60): 110.0,
    ("label_X_cc", 0.40): 320.0,
}
# The study's selection (docs/GROUND_TRUTH.md), which is also the range the
# bb/cc jets span in 2503.00118 App. A: 200 < pT < 2500, 20 < m_SD < 500.
PT_LO, PT_HI, MSD_LO, MSD_HI = 200.0, 2500.0, 20.0, 500.0
CHUNK = 500_000


def _e1_control():
    """Reuse the committed discriminant rather than writing a second one."""
    p = ROOT / "experiments" / "MASSREG" / "e1_control.py"
    spec = importlib.util.spec_from_file_location("e1_control", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def signal_index(class_name: str) -> int:
    import csv
    with (ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv").open() as f:
        for r in csv.DictReader(f):
            if r["class_name"] == class_name:
                return int(r["jet_label"])
    raise SystemExit(f"FATAL: {class_name} not in the label map")


# The (m_SD, pT) grid the background is reweighted on. Coarse enough that every
# populated cell holds enough QCD jets to estimate a ratio, fine enough to
# remove the gross mismatch (QCD median m_SD 57 GeV vs signal 141 GeV, measured).
MSD_EDGES = np.linspace(MSD_LO, MSD_HI, 25)
PT_EDGES = np.geomspace(PT_LO, PT_HI, 21)


def match_weights(m_sig, pt_sig, m_bkg, pt_bkg):
    """Per-jet background weights making QCD's (m_SD, pT) density match signal's.

    WHY THIS IS NOT OPTIONAL. In the raw test sample the signal and QCD
    kinematics are wildly different -- measured medians m_SD 141 vs 57 GeV and
    pT 1059 vs 854 -- so an UNWEIGHTED ROC lets the discriminant separate on
    jet mass, which is not flavour tagging. arXiv:2405.12972 App. B applies a
    two-dimensional reweighting on (m_SD, pT) "to ensure consistent
    distributions, hence reducing the dependence of the tagger response on jet
    mass and p_T", so a number compared against Table A1 must do the same.

    Cells with no background jets get zero weight: they cannot contribute a
    background estimate, and pretending otherwise would divide by zero.
    """
    hs, _, _ = np.histogram2d(m_sig, pt_sig, bins=[MSD_EDGES, PT_EDGES])
    hb, _, _ = np.histogram2d(m_bkg, pt_bkg, bins=[MSD_EDGES, PT_EDGES])
    ratio = np.zeros_like(hs)
    nz = hb > 0
    ratio[nz] = (hs[nz] / hs.sum()) / (hb[nz] / hb.sum())
    i = np.clip(np.digitize(m_bkg, MSD_EDGES) - 1, 0, len(MSD_EDGES) - 2)
    j = np.clip(np.digitize(pt_bkg, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    return ratio[i, j]


def rejection(d_sig: np.ndarray, d_bkg: np.ndarray, eps_s: float,
              w_bkg: np.ndarray | None = None) -> tuple[float, float, float]:
    """1/eps_B at a threshold fixed by the SIGNAL efficiency.

    The threshold is the (1 - eps_s) quantile of the signal discriminant, so
    exactly eps_s of signal passes; eps_B is then measured on background. Doing
    it the other way round (fixing eps_B) answers a different question and is
    the easiest way to quote a number that cannot be compared to Table A1.
    """
    thr = float(np.quantile(d_sig, 1.0 - eps_s))
    passed = d_bkg >= thr
    if w_bkg is None:
        num, den = float(passed.sum()), float(d_bkg.size)
    else:
        num, den = float(w_bkg[passed].sum()), float(w_bkg.sum())
    eps_b = num / den if den > 0 else 0.0
    return (float("inf") if num == 0 else 1.0 / eps_b), thr, eps_b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir with logits.npy, label188.npy, observers.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="relative agreement required against the published value")
    a = ap.parse_args(argv)

    e1 = _e1_control()
    d = pathlib.Path(a.features)
    logits = np.load(d / "logits.npy", mmap_mode="r")
    label = np.load(d / "label188.npy")
    obs = np.load(d / "observers.npz")
    qcd = e1.qcd_indices()
    print(f"{logits.shape[0]:,} jets, {logits.shape[1]} classes, {qcd.size} QCD nodes", flush=True)
    if logits.shape[1] != 188:
        raise SystemExit(f"FATAL: expected 188 logit columns, got {logits.shape[1]}")

    sel = ((obs["jet_pt"] > PT_LO) & (obs["jet_pt"] < PT_HI)
           & (obs["jet_sdmass"] > MSD_LO) & (obs["jet_sdmass"] < MSD_HI))
    print(f"selection {PT_LO}<pT<{PT_HI}, {MSD_LO}<m_SD<{MSD_HI}: "
          f"{sel.sum():,} of {sel.size:,} ({100*sel.mean():.2f}%)", flush=True)

    results, ok = [], True
    for class_name in ("label_X_bb", "label_X_cc"):
        sig = np.asarray([signal_index(class_name)], dtype=np.int64)
        if np.intersect1d(sig, qcd).size:
            raise SystemExit("FATAL: signal set overlaps the QCD set")
        # Only signal and QCD jets enter a "S vs QCD" ROC; everything else is
        # neither, and including it would silently change eps_B's denominator.
        keep = sel & (np.isin(label, sig) | np.isin(label, qcd))
        rows = np.flatnonzero(keep)
        D = np.empty(rows.size, dtype=np.float64)
        for i in range(0, rows.size, CHUNK):
            idx = rows[i:i + CHUNK]
            p = e1.softmax(np.asarray(logits[idx], dtype=np.float32))
            D[i:i + idx.size] = e1.discriminant(p, sig, qcd)
        is_sig = np.isin(label[rows], sig)
        d_sig, d_bkg = D[is_sig], D[~is_sig]
        print(f"{class_name}: {d_sig.size:,} signal, {d_bkg.size:,} QCD", flush=True)
        # An empty side is a data or selection error, not a result. Without this
        # np.quantile raises a bare IndexError from inside numpy, which on a
        # cluster job reads as a code bug rather than "this cache has no X_cc".
        if d_sig.size == 0 or d_bkg.size == 0:
            raise SystemExit(f"FATAL: {class_name}: {d_sig.size} signal and "
                             f"{d_bkg.size} QCD jets pass the selection; "
                             f"a rejection cannot be defined")
        msd, jpt = obs["jet_sdmass"][rows], obs["jet_pt"][rows]
        w = match_weights(msd[is_sig], jpt[is_sig], msd[~is_sig], jpt[~is_sig])
        print(f"  reweighted QCD: {int((w > 0).sum()):,} of {w.size:,} jets in populated cells",
              flush=True)
        for eps_s in (0.60, 0.40):
            rej, thr, eps_b = rejection(d_sig, d_bkg, eps_s, w_bkg=w)
            raw, _, raw_eps = rejection(d_sig, d_bkg, eps_s)
            pub = PUBLISHED[(class_name, eps_s)]
            rel = abs(rej - pub) / pub
            agree = rel <= a.tolerance
            ok &= agree
            results.append(dict(signal=class_name, eps_s=eps_s, rejection=round(rej, 1),
                                rejection_unweighted=round(raw, 1),
                                published=pub, rel_diff=round(rel, 3), agrees=bool(agree),
                                threshold=round(thr, 6), eps_b=eps_b,
                                n_signal=int(d_sig.size), n_qcd=int(d_bkg.size)))
            print(f"  eps_S={eps_s:.0%}: 1/eps_B = {rej:8.1f} (unweighted {raw:8.1f})  "
                  f"published {pub:6.0f}  rel {rel:+.1%}  {'OK' if agree else 'MISMATCH'}",
                  flush=True)

    out = dict(features=str(d), selection=dict(pt=[PT_LO, PT_HI], msd=[MSD_LO, MSD_HI]),
               n_jets_total=int(label.size), n_selected=int(sel.sum()),
               tolerance=a.tolerance, all_agree=bool(ok), anchors=results)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}\nALL ANCHORS AGREE: {ok}", flush=True)
    # A disagreement is a RESULT, not a job failure. Returning non-zero made the
    # first run burn its whole backoffLimit re-deriving the same numbers.
    return 0


if __name__ == "__main__":
    sys.exit(main())
