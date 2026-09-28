#!/usr/bin/env python3
"""Write the binned pass/fail counts every real-data top fit is made from.

The fits in peak_fit.py see the jets only through these counts: for each score,
the (m_SD, pT) bins of the main cut and of the validation band, built by
peak_fit's own map, cut and binning functions. Exported once from the merged
jets, they let every fit be redone and checked anywhere without the 200 GB
dataset. Nothing is fitted here.

The counts are checked against the histograms.npz of the run they are for (its
mass projections), so they are the bins that run fitted, or nothing is written.

Usage (in the fit job's environment, after merge_shards.py):
    python3 experiments/AOJ/export_fit_bins.py --jets merged/jets.npz \\
        --results results.json --histograms histograms.npz \\
        --scores NAME=scores_NAME.npz ... --out bins.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("peak_fit", HERE / "peak_fit.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

KEYS = ("m_edges", "i", "j", "n_pass", "n_fail", "rho", "pt")


def fit_bins(z, mass, pt, peak, eff):
    """(main, validation) bins exactly as peak_fit.analyse and peak_fit.validation make them."""
    cfg = P.PEAKS[peak]
    passed = P.passes(z, mass, pt, P.build_map(z, mass, pt, eff))
    main = P._bins(mass, pt, passed, cfg["fit_range"])
    outer = P.build_map(z, mass, pt, (P.VALIDATION_OFFSET + 1) * eff)
    inner = P.build_map(z, mass, pt, P.VALIDATION_OFFSET * eff)
    region = ~P.passes(z, mass, pt, inner)
    val = P._bins(mass[region], pt[region], P.passes(z, mass, pt, outer)[region], cfg["fit_range"])
    return main, val


def projection(b, key):
    return np.bincount(b["i"], weights=b[key], minlength=len(b["m_edges"]) - 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jets", required=True)
    ap.add_argument("--results", required=True, help="results.json of the run the bins are for")
    ap.add_argument("--histograms", required=True, help="histograms.npz of the same run")
    ap.add_argument("--scores", nargs="+", required=True, metavar="NAME=scores.npz")
    ap.add_argument("--peak", default="top", choices=list(P.PEAKS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = json.loads(pathlib.Path(a.results).read_text())
    hist = np.load(a.histograms)
    j = np.load(a.jets)
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    mass, pt = mass[ok], pt[ok]
    if int(ok.sum()) != res["n_jets"]:
        raise SystemExit(f"FATAL: {int(ok.sum())} jets in the fit region, results.json has {res['n_jets']}")
    key = dict(W="two_prong_logodds", top="three_prong_logodds")[a.peak]
    cms_key = dict(W="aoj_pn_WvsQCD", top="aoj_pn_TvsQCD")[a.peak]
    scores = {"reference": P.logit(j[cms_key])[ok]}
    for item in a.scores:
        name, path = item.split("=", 1)
        scores[name] = np.load(path)[key].astype(float)[ok]
    if set(scores) - {"reference"} != set(res["models"]):
        raise SystemExit("FATAL: --scores does not name exactly the models of results.json")
    out = {}
    for name, z in scores.items():
        main_b, val_b = fit_bins(z, mass, pt, a.peak, res["eff"])
        for part, b, prefix in (("main", main_b, ""), ("validation", val_b, "validation_")):
            for k in ("n_pass", "n_fail"):
                stored = hist[f"{name}_{a.peak}_{prefix}{k}"]
                if not np.array_equal(projection(b, k), stored):
                    raise SystemExit(f"FATAL: {name} {part} {k} differs from {a.histograms}; "
                                     "these are not the bins that run fitted")
            for k in KEYS:
                out[f"{name}|{part}|{k}"] = b[k]
        print(f"{name:14s} main {len(main_b['n_pass'])} bins, {main_b['n_pass'].sum():.0f} pass; "
              f"validation {len(val_b['n_pass'])} bins; match the run's histograms", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(scores)} scores x (main, validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
