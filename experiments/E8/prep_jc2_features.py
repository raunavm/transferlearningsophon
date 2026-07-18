#!/usr/bin/env python3
"""JC2 -> per-g10-class preprocessed feature arrays for Phase-2 fine-tuning (E8).

Fine-tuning (unlike the frozen probe) needs the full (N,128,17) preprocessed
features, not the 128-d embeddings, since backprop flows through the backbone.
This caches them once per preprocessing so every FT run reuses them (the JC1
`train_finetune_sweep.py` pattern: load_npy_features reads `*_features.npy`).

For each split it writes, per g10 class c and per preprocessing:
  <out_part>/<split>/g10c{c}_{features,lorentz,masks,labels}.npy   (ParT preproc)
  <out_sophon>/<split>/g10c{c}_{features,lorentz,masks,labels}.npy (Sophon preproc)
so ParT + Arm P fine-tune from features_part and Arm S from features_sophon.
Class-balanced by construction (one file per class); `--cap-per-class` bounds
size/RAM. Reuses the extractor's `process_file` (identical preprocessing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.E8.extract_jc2_embeddings import process_file  # noqa: E402


def g10_map():
    j = json.loads((REPO / "experiments/E2/labels/label_maps.json").read_text())
    return np.asarray(j["g10sem"]["map"], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--files", nargs="+", required=True, help="JC2 parquet paths")
    ap.add_argument("--out-part", required=True)
    ap.add_argument("--out-sophon", required=True)
    ap.add_argument("--cap-per-class", type=int, default=120000)
    args = ap.parse_args()

    remap = g10_map()
    nC = int(remap.max()) + 1
    acc = {p: {c: {"f": [], "lv": [], "m": []} for c in range(nC)} for p in ("part", "sophon")}
    counts = {c: 0 for c in range(nC)}

    for fi, fp in enumerate(args.files):
        if all(counts[c] >= args.cap_per_class for c in range(nC)):
            print("all classes at cap — stopping early")
            break
        pf, plv, sf, slv, masks, lab188, _ = process_file(fp, True)  # both preprocs
        g = remap[lab188]
        for c in range(nC):
            if counts[c] >= args.cap_per_class:
                continue
            sel = np.where(g == c)[0][: args.cap_per_class - counts[c]]
            if len(sel) == 0:
                continue
            acc["part"][c]["f"].append(pf[sel].astype(np.float16))
            acc["part"][c]["lv"].append(plv[sel].astype(np.float16))
            acc["part"][c]["m"].append(masks[sel])
            acc["sophon"][c]["f"].append(sf[sel].astype(np.float16))
            acc["sophon"][c]["lv"].append(slv[sel].astype(np.float16))
            acc["sophon"][c]["m"].append(masks[sel])
            counts[c] += len(sel)
        print(f"[{fi + 1}/{len(args.files)}] {Path(fp).name}  "
              f"counts={[counts[c] for c in range(nC)]}")

    for preproc, outdir in (("part", args.out_part), ("sophon", args.out_sophon)):
        od = Path(outdir) / args.split
        od.mkdir(parents=True, exist_ok=True)
        for c in range(nC):
            if not acc[preproc][c]["f"]:
                print(f"  WARN {preproc} class {c}: 0 jets")
                continue
            f = np.concatenate(acc[preproc][c]["f"])
            lv = np.concatenate(acc[preproc][c]["lv"])
            m = np.concatenate(acc[preproc][c]["m"])
            np.save(od / f"g10c{c}_features.npy", f)
            np.save(od / f"g10c{c}_lorentz.npy", lv)
            np.save(od / f"g10c{c}_masks.npy", m)
            np.save(od / f"g10c{c}_labels.npy", np.full(len(f), c, np.int64))
        (od / "prep_meta.json").write_text(json.dumps(
            {"split": args.split, "preproc": preproc, "cap_per_class": args.cap_per_class,
             "per_class_counts": {str(c): counts[c] for c in range(nC)},
             "n_files_read": len(args.files),
             "note": "class-balanced per-g10-class features; FeatureDataset/load_npy_features format"},
            indent=2))
        print(f"saved {preproc}/{args.split}: {[counts[c] for c in range(nC)]}")


if __name__ == "__main__":
    main()
