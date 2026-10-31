#!/usr/bin/env python3
"""Mass-response distributions for the mass figure (plan Part 5, 2026-10-31).

From an extract_v2.py --keep-head-outputs checkpoint directory (head_outputs.npz): the distribution of
ln(m / m_SD^gen) for resonant and for QCD jets with a generator-level match, for the uncorrected
soft-drop mass (m = m_SD^reco) and, for a model with a mass output, for its corrected mass
(m = m_SD^reco exp(y_pred), the output regressing ln(m_SD^gen / m_SD^reco)). Normalised histograms on
a fixed grid, so the runs of a configuration can be averaged.

    python3 experiments/EVAL/mass_response_hist.py --head <dir> --out <file.json>
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

BINS = np.linspace(-1.0, 1.0, 101)
QCD_FIRST = 161


def hist(x: np.ndarray) -> list:
    h = np.histogram(np.clip(x, BINS[0], BINS[-1] - 1e-9), bins=BINS)[0].astype(np.float64)
    return (h / h.sum()).tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    z = np.load(a.head / "head_outputs.npz")
    gen, reco = z["genjet_sdmass"].astype(np.float64), z["jet_sdmass"].astype(np.float64)
    ok = gen > 0                                     # unmatched generator jets are a hard 0.0
    lab = z["label188"].astype(np.int64)
    doc = {"head": str(a.head), "bins": BINS.tolist(), "n_matched": int(ok.sum()),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "hist": {}}
    groups = {"resonant": ok & (lab < QCD_FIRST), "qcd": ok & (lab >= QCD_FIRST)}
    base = np.log(reco[ok] / gen[ok])
    for g, m in groups.items():
        doc["hist"][f"uncorrected/{g}"] = hist(np.log(reco[m] / gen[m]))
        if "mass" in z.files:
            doc["hist"][f"node/{g}"] = hist(np.log(reco[m] / gen[m]) + z["mass"][m].astype(np.float64))
    doc["uncorrected_median"] = float(np.median(base))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc) + "\n")
    print(a.out, {k: round(float(np.argmax(v)), 1) for k, v in doc["hist"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
