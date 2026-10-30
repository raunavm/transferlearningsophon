#!/usr/bin/env python3
"""Mass sculpting of one run's output layer: how far a cut on its resonant-vs-QCD score
reshapes the QCD soft-drop mass distribution (plan 2026-10-29, A6 and step 1.5).

On the QCD test jets (native labels 161..187) of an extract_v2.py --keep-head-outputs
checkpoint, the score is s = 1 - p_QCD, p_QCD the summed softmax probability of the
run's QCD output nodes (anomaly.node_roles: the single QCD node of the 162-, 43- and
17-class label sets, the 27 QCD nodes of the 188-class one, from
configs/labelmaps/rung_label_maps.v1.csv). s is the max-shifted softmax sum over the non-QCD
nodes, in float64, so it keeps its precision where p_QCD is close to 1, which is most QCD jets.
At each QCD efficiency in WORKING_POINTS the cut keeps the jets at or above the k-th
largest score, k = round(eff n); the jet_sdmass histograms (BINS, normalised) before
and after the cut are compared by the Jensen-Shannon divergence in base 2 (0 = the
same shape, 1 = disjoint). The run with and without the mass output are paired by seed
in the table code, not here.

    python3 experiments/EVAL/mass_sculpting.py --head <dir with head_outputs.npz> --out <file.json>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD_FILE = "head_outputs.npz"
QCD_LABELS = (161, 188)                      # native labels [161, 188)
WORKING_POINTS = (0.1, 0.01)                 # QCD efficiency of the cut
BINS = np.arange(20, 501, 20)                # jet_sdmass, GeV: the selection's range


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def score(logits: np.ndarray, qcd_nodes) -> np.ndarray:
    """1 - p_QCD = the summed softmax probability of every node outside `qcd_nodes`."""
    z = np.asarray(logits, dtype=np.float64)
    q = np.zeros(z.shape[1], dtype=bool)
    q[sorted(qcd_nodes)] = True
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e[:, ~q].sum(axis=1) / e.sum(axis=1)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in base 2 of two normalised histograms."""
    m = (p + q) / 2
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))   # noqa: E731
    return 0.5 * kl(p) + 0.5 * kl(q)


def hist(x: np.ndarray) -> np.ndarray:
    h = np.histogram(x, bins=BINS)[0].astype(np.float64)
    return h / h.sum()


def sculpting(s: np.ndarray, msd: np.ndarray) -> dict:
    """Before/after histograms, JSD and passing counts at each working point."""
    before = hist(msd)
    desc = np.sort(s)[::-1]
    out = {}
    for eff in WORKING_POINTS:
        k = max(int(round(eff * s.size)), 1)
        cut = float(desc[k - 1])
        sel = s >= cut
        after = hist(msd[sel])
        out[str(eff)] = {"threshold": cut, "n_pass": int(sel.sum()),
                         "eff_realised": float(sel.mean()), "hist_after": after.tolist(),
                         "jsd": jsd(before, after)}
    return {"hist_before": before.tolist(), "working_points": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", required=True, type=pathlib.Path,
                    help="an extract_v2.py --keep-head-outputs checkpoint directory")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="the JSON file written")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"{a.out} exists; not overwritten")
        return 0
    man = json.loads((a.head / "manifest.json").read_text())
    rung = man["rung"]
    if rung == "none":
        raise SystemExit(f"FATAL: {a.head}: a label set outside the contraction tree has no QCD node")
    qcd_nodes = sorted(_load("anomaly", "experiments/EVAL/anomaly.py").node_roles(rung)[2])
    z = np.load(a.head / HEAD_FILE)
    lab = z["label188"].astype(np.int64)
    qcd = (lab >= QCD_LABELS[0]) & (lab < QCD_LABELS[1])
    logits = z["logits"][qcd]
    if logits.shape[1] != man["num_classes"] or max(qcd_nodes) >= logits.shape[1]:
        raise SystemExit(f"FATAL: {logits.shape[1]} logits for {rung}'s {man['num_classes']} classes")
    s, msd = score(logits, qcd_nodes), z["jet_sdmass"][qcd].astype(np.float64)
    doc = {"head": str(a.head), "run_dir": man["run_dir"], "rung": rung, "tag": man["tag"],
           "checkpoint_sha256": man.get("checkpoint_sha256"),
           "head_outputs_sha256": (man.get("head_outputs") or {}).get("sha256"),
           "qcd_nodes": qcd_nodes, "n_qcd": int(qcd.sum()), "n_jets": int(lab.size),
           "score": "1 - p_QCD, p_QCD = summed softmax probability of the QCD nodes",
           "bins_jet_sdmass": BINS.tolist(), "divergence": "Jensen-Shannon, base 2",
           **sculpting(s, msd),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(a.out)
    for eff, w in doc["working_points"].items():
        print(f"QCD efficiency {eff}: {w['n_pass']:,} of {doc['n_qcd']:,} pass, JSD {w['jsd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
