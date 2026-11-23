#!/usr/bin/env python3
"""SC-2 check of the transfer-head term (PI 2026-11-23, LEDGER SC-2, #1) on rows that have per-jet scores and a paired
bootstrap: the signal efficiency at a fixed background efficiency, eps_S at eps_B = 1%, the metric of tab:tests'
transfer-head rows, for two sides of networks scored on the same test jets.

Per network, the unpaired analytic variance of tests/test_xbx_claims_l3.py: eps_S (1 - eps_S) / n_S + (d eps_S / d
eps_B)^2 eps_B (1 - eps_B) / n_B, the slope a central difference of the ROC between eps_B / K and eps_B K (K = 10^0.2,
the spacing of the transfer head's stored curve, experiments/XBS/xbx.py EPS_B_GRID); a side's term is the mean of its
networks' SDs, and the two sides' terms add in quadrature. Against it: the paired bootstrap SD of the difference of the
side means over B resamples of the test jets drawn within each class and shared by every network
(experiments/STATS/paired_bootstrap.py counts, its seed), and each network's own bootstrap SD beside its analytic SD.

    python3 experiments/STATS/analytic_check.py --diff NAME=A1.npz,A2.npz/B1.npz,B2.npz ... --out <file>.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
EPS_B = 0.01
K = 10 ** 0.2


def _pb():
    spec = importlib.util.spec_from_file_location("paired_bootstrap", REPO / "experiments/STATS/paired_bootstrap.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class EffScorer:
    """eps_S at a background efficiency on the count-weighted ROC (truth: signal), sorting once; eps_S interpolated
    in eps_B between the distinct scores (the ROC of paired_bootstrap.RejScorer with the axes swapped)."""

    def __init__(self, truth: np.ndarray, score: np.ndarray):
        self.order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
        ss = np.asarray(score, dtype=np.float64)[self.order]
        self.pos = np.asarray(truth, dtype=bool)[self.order].astype(np.float64)
        self.gid = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]).astype(np.int64)
        self.g = int(self.gid[-1]) + 1

    def __call__(self, w=None, eps_b: float = EPS_B) -> float:
        ws = np.ones(self.pos.size) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        tp = np.cumsum(np.bincount(self.gid, weights=ws * self.pos, minlength=self.g))
        fp = np.cumsum(np.bincount(self.gid, weights=ws * (1.0 - self.pos), minlength=self.g))
        return float(np.interp(eps_b, np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]]))


def analytic(sc: EffScorer, n_s: int, n_b: int) -> tuple[float, float, float]:
    """(eps_S, its analytic SD, the slope) on the full sample."""
    e = sc()
    slope = (sc(eps_b=EPS_B * K) - sc(eps_b=EPS_B / K)) / (EPS_B * K - EPS_B / K)
    return e, float(np.sqrt(e * (1 - e) / n_s + slope ** 2 * EPS_B * (1 - EPS_B) / n_b)), slope


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["rows"].astype(np.int64), z["y"].astype(np.int64), z["s"].astype(np.float64)


def check(diffs: dict, b: int, seed: int) -> dict:
    pb = _pb()
    paths = list(dict.fromkeys(p for a, c in diffs.values() for p in a + c))
    rows0, y0, _ = load(paths[0])
    if set(np.unique(y0).tolist()) != {0, 1}:
        raise SystemExit("FATAL: binary labels expected")
    c = pb.counts(y0, b, seed)
    n_s, n_b = int((y0 == 1).sum()), int((y0 == 0).sum())
    net = {}
    for p in paths:
        rows, y, s = load(p)
        if not (np.array_equal(rows, rows0) and np.array_equal(y, y0)):
            raise SystemExit(f"FATAL: {p} is not scored on the jets of {paths[0]}")
        sc = EffScorer(y == 1, s)
        e, sd, slope = analytic(sc, n_s, n_b)
        rep = np.array([sc(c[k]) for k in range(b)])
        net[p] = {"eps_s": e, "analytic_sd": sd, "slope": slope, "boot_sd": float(rep.std(ddof=1)), "rep": rep}
    out = {}
    for name, (a, bb) in diffs.items():
        va, vb = [net[p]["eps_s"] for p in a], [net[p]["eps_s"] for p in bb]
        ta, tb = float(np.mean([net[p]["analytic_sd"] for p in a])), float(np.mean([net[p]["analytic_sd"] for p in bb]))
        rep = np.mean([net[p]["rep"] for p in a], axis=0) - np.mean([net[p]["rep"] for p in bb], axis=0)
        sd_a, sd_b = (float(np.std(v, ddof=1)) if len(v) > 1 else None for v in (va, vb))
        boot, t = float(rep.std(ddof=1)), float(np.hypot(ta, tb))
        run2 = (sd_a or 0.0) ** 2 + (sd_b or 0.0) ** 2
        out[name] = {"a": a, "b": bb, "delta": float(np.mean(va) - np.mean(vb)), "sd_a": sd_a, "sd_b": sd_b,
                     "boot_sd_paired": boot, "analytic_sd_unpaired": t, "analytic_over_paired": t / boot,
                     "sigma_paired": float(np.sqrt(run2 + boot ** 2)), "sigma_analytic": float(np.sqrt(run2 + t ** 2))}
    return {"metric": f"eps_S at eps_B = {EPS_B:g}", "slope_K": K, "b": b, "seed": seed, "n_s": n_s, "n_b": n_b,
            "networks": {p: {k: v for k, v in d.items() if k != "rep"} for p, d in net.items()}, "diffs": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="append", required=True, metavar="NAME=A1,A2/B1,B2")
    ap.add_argument("--b", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; use a new output")
    diffs = {}
    for d in a.diff:
        name, sides = d.split("=", 1)
        x, y = sides.split("/")
        diffs[name] = (x.split(","), y.split(","))
    doc = {"script": "experiments/STATS/analytic_check.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), **check(diffs, a.b, a.seed)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, a.out)
    for k, v in doc["diffs"].items():
        print(f"{k}: delta {v['delta']:.5f} paired {v['boot_sd_paired']:.5f} analytic {v['analytic_sd_unpaired']:.5f} "
              f"ratio {v['analytic_over_paired']:.2f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
