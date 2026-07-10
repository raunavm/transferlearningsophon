#!/usr/bin/env python3
"""E1 probe: JetClass-1 data integrity + (jet_pt, jet_sdmass) inspection.

Confirms all 10 classes are present, reports per-class jet_pt / jet_sdmass
ranges & quantiles, and the fraction of jets that would fall OUTSIDE the
provisional Arm S reweight bins (data_arm_s.yaml). Read-only. Writes one JSON.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import uproot

CLASSES = ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
           "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu"]

# Provisional Arm S bin edges — MUST match data_arm_s.yaml. The outside-fraction
# check tells us whether these edges need widening before training.
PT_EDGES = [499, 560, 625, 700, 785, 880, 1001]
SDMASS_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300, 350,
                400, 450, 500, 550]


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--files-per-class", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {"root_dir": args.root_dir, "files_per_class": args.files_per_class,
              "pt_edges": PT_EDGES, "sdmass_edges": SDMASS_EDGES, "per_class": {}}
    all_pt, all_sd = [], []

    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(args.root_dir, f"{cls}_*.root")))[:args.files_per_class]
        if not files:
            report["per_class"][cls] = {"error": "NO FILES FOUND"}
            print(f"[FAIL] {cls}: no files")
            continue
        pts, sds, n = [], [], 0
        for f in files:
            with uproot.open(f"{f}:tree") as t:
                arr = t.arrays(["jet_pt", "jet_sdmass"], library="np")
                pts.append(arr["jet_pt"]); sds.append(arr["jet_sdmass"]); n += len(arr["jet_pt"])
        pt = np.concatenate(pts); sd = np.concatenate(sds)
        all_pt.append(pt); all_sd.append(sd)
        out_pt = float(np.mean((pt < PT_EDGES[0]) | (pt > PT_EDGES[-1])))
        out_sd = float(np.mean((sd < SDMASS_EDGES[0]) | (sd > SDMASS_EDGES[-1])))
        report["per_class"][cls] = {
            "files": len(files), "n_jets": int(n),
            "jet_pt": {"min": float(pt.min()), "max": float(pt.max()),
                       "q001": q(pt, 0.001), "q999": q(pt, 0.999)},
            "jet_sdmass": {"min": float(sd.min()), "max": float(sd.max()),
                           "q001": q(sd, 0.001), "q999": q(sd, 0.999)},
            "frac_pt_outside_bins": out_pt, "frac_sdmass_outside_bins": out_sd,
        }
        print(f"[ok] {cls}: {n:,} jets | pt[{pt.min():.0f},{pt.max():.0f}] "
              f"sd[{sd.min():.1f},{sd.max():.1f}] | outside pt={out_pt:.4%} sd={out_sd:.4%}")

    if all_pt:
        pt = np.concatenate(all_pt); sd = np.concatenate(all_sd)
        report["overall"] = {
            "frac_pt_outside_bins": float(np.mean((pt < PT_EDGES[0]) | (pt > PT_EDGES[-1]))),
            "frac_sdmass_outside_bins": float(np.mean((sd < SDMASS_EDGES[0]) | (sd > SDMASS_EDGES[-1]))),
            "verdict_bins_ok": bool(
                np.mean((pt < PT_EDGES[0]) | (pt > PT_EDGES[-1])) < 0.005
                and np.mean((sd < SDMASS_EDGES[0]) | (sd > SDMASS_EDGES[-1])) < 0.005),
        }
        print(f"\nOVERALL outside: pt={report['overall']['frac_pt_outside_bins']:.4%} "
              f"sd={report['overall']['frac_sdmass_outside_bins']:.4%} "
              f"| bins_ok={report['overall']['verdict_bins_ok']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fo:
        json.dump(report, fo, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
