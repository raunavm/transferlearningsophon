#!/usr/bin/env python3
"""Candidate jets for Fig. 1's event display (PI brief 2026-11-12): from the first JetClass-II TEST file of each
family, the first N jets of one class that pass the pretraining selection, with their particles and every aux_*
field, as JSON. The figure's three jets are chosen from these by hand (fig1_design.py names them).

    python3 experiments/FIGS/pick_fig1_jets.py --out /data/results/fig1_jets/<sha7>/candidates.json
"""
import argparse
import json
import pathlib

import awkward as ak
import numpy as np

# (file of the test split, jet_label, class name): scripts/build_eval_jobs.py's test files start at these
PICKS = (("/jc2/jet_data/Res2P_0250.parquet", 0, "label_X_bb"),
         ("/jc2/jet_data/Res34P_1075.parquet", 63, "label_X_YY_qqqq"),
         ("/jc2/jet_data/QCD_0350.parquet", 187, "label_QCD_light"))
PART = ("part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi", "part_charge")
N = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    doc = {"selection": "200 < jet_pt < 2500, 20 < jet_sdmass < 500, 400 < jet_pt < 900, 25 to 70 particles", "jets": []}
    for path, label, name in PICKS:
        arr = ak.from_parquet(path)
        print(path, len(arr), "fields:", arr.fields)
        keep = ((arr.jet_label == label) & (arr.jet_pt > 400) & (arr.jet_pt < 900) & (arr.jet_sdmass > 20)
                & (arr.jet_sdmass < 500) & (ak.num(arr.part_px) >= 25) & (ak.num(arr.part_px) <= 70))
        rows = np.flatnonzero(ak.to_numpy(keep))[:N]
        print(name, "label", label, "passing", int(ak.sum(keep)), "kept rows", rows[:5])
        for r in rows:
            j = arr[int(r)]
            rec = {"class_name": name, "file": path, "row": int(r)}
            for f in arr.fields:
                if f.startswith(("jet_", "aux_")) or f in PART:
                    v = ak.to_list(j[f])
                    rec[f] = v
            doc["jets"].append(rec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc))
    print("wrote", a.out, len(doc["jets"]), "jets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
