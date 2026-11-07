#!/usr/bin/env python3
"""Mass sculpting of one run's output layer: how far a cut on its resonant-vs-QCD score
reshapes the QCD soft-drop mass distribution (plan 2026-10-29, A6 and step 1.5).

On the QCD test jets (native labels 161..187) of an extract_v2.py --keep-head-outputs
checkpoint, the score is s = 1 - p_QCD, p_QCD the summed softmax probability of the
run's QCD output nodes (anomaly.node_roles: the single QCD node of the 162-, 43- and
17-class label sets, the 27 QCD nodes of the 188-class one, from
configs/labelmaps/rung_label_maps.v1.csv; for the three 17-class flavour-boundary partitions,
whose extraction manifest says rung 'none', the QCD node of --label-set's column of FLAV_MAP,
by the same rule). s is the max-shifted softmax sum over the non-QCD
nodes, in float64, so it keeps its precision where p_QCD is close to 1, which is most QCD jets.
At each QCD efficiency in WORKING_POINTS the cut keeps the jets at or above the k-th
largest score, k = round(eff n); the jet_sdmass histograms (BINS, normalised) before
and after the cut are compared by the Jensen-Shannon divergence in base 2 (0 = the
same shape, 1 = disjoint). At each working point the resonant test jets (native labels
0..160) passing the same cut give the signal efficiency eff_sig, and 1 / eff_realised the
background rejection. sig_eff_50 is the cut that keeps half of the resonant jets (the k-th
largest resonant score, k = round(n_sig / 2)) with its QCD efficiency, rejection R50 and the
divergence there, JSD50 (DisCo, arXiv:2001.05310 Fig. 3, plots 1/JSD50 against R50). The
run with and without the mass output are paired by seed in the table code, not here.

    python3 experiments/EVAL/mass_sculpting.py --head <dir with head_outputs.npz> --out <file.json> [--label-set FLAV_F0]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD_FILE = "head_outputs.npz"
QCD_LABELS = (161, 188)                      # native labels [161, 188); below them resonant
FLAV_MAP = "configs/labelmaps/flavour_pair_map.v2.csv"   # FLAV_F0, FLAV_F1, FLAV_F1R columns
SIG_EFF = 0.5                                # the signal efficiency of DisCo's R50 / JSD50
WORKING_POINTS = (0.1, 0.01)                 # QCD efficiency of the cut
# 2026-10-31: the divergence and the Jensen-Shannon DISTANCE (its square root, the quantity CMS
# reports, arXiv:2604.09809 Sec. 6.2 and Fig. 9) on a grid of QCD efficiencies, for the curve
EFF_GRID = tuple(float(x) for x in np.round(np.logspace(-3, np.log10(0.5), 16), 6))
BINS = np.arange(20, 501, 20)                # jet_sdmass, GeV: the selection's range


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def qcd_nodes(label_set: str) -> list[int]:
    """The QCD output nodes of a contraction-tree rung (anomaly.node_roles) or of a
    flavour-boundary partition (a column of FLAV_MAP), a node being QCD only if every
    native class in it is."""
    if not label_set.startswith("FLAV_"):
        return sorted(_load("anomaly", "experiments/EVAL/anomaly.py").node_roles(label_set)[2])
    rows = list(csv.DictReader((REPO / FLAV_MAP).open()))
    if label_set not in rows[0]:
        raise SystemExit(f"FATAL: {label_set} not a column of {FLAV_MAP}")
    isq = {}
    for r in rows:
        isq.setdefault(int(r[label_set]), []).append(r["class_name"].startswith("label_QCD_"))
    return sorted(n for n, q in isq.items() if all(q))


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


def sculpting(s: np.ndarray, msd: np.ndarray, s_sig: np.ndarray) -> dict:
    """Before/after histograms, JSD, passing counts, signal efficiency and rejection at each
    working point; the same at the cut keeping SIG_EFF of the resonant jets (scores s_sig)."""
    before = hist(msd)
    desc = np.sort(s)[::-1]
    out = {}
    for eff in WORKING_POINTS:
        k = max(int(round(eff * s.size)), 1)
        cut = float(desc[k - 1])
        sel = s >= cut
        after = hist(msd[sel])
        sig = s_sig >= cut
        out[str(eff)] = {"threshold": cut, "n_pass": int(sel.sum()),
                         "eff_realised": float(sel.mean()), "hist_after": after.tolist(),
                         "jsd": jsd(before, after), "n_sig_pass": int(sig.sum()),
                         "eff_sig": float(sig.mean()), "rejection": 1 / float(sel.mean())}
    k = max(int(round(SIG_EFF * s_sig.size)), 1)
    cut = float(np.sort(s_sig)[::-1][k - 1])
    sel, sig = s >= cut, s_sig >= cut
    after = hist(msd[sel])
    d50 = jsd(before, after)
    at50 = {"eff_sig_target": SIG_EFF, "threshold": cut, "n_sig_pass": int(sig.sum()),
            "eff_sig": float(sig.mean()), "n_pass": int(sel.sum()), "eff_realised": float(sel.mean()),
            "rejection": 1 / float(sel.mean()), "hist_after": after.tolist(),
            "jsd": d50, "js_distance": float(np.sqrt(d50))}
    curve = []
    for eff in EFF_GRID:
        k = max(int(round(eff * s.size)), 1)
        sel = s >= float(desc[k - 1])
        d = jsd(before, hist(msd[sel]))
        curve.append({"eff": eff, "eff_realised": float(sel.mean()), "n_pass": int(sel.sum()),
                      "jsd": d, "js_distance": float(np.sqrt(d))})
    return {"hist_before": before.tolist(), "working_points": out, "sig_eff_50": at50, "curve": curve}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", required=True, type=pathlib.Path,
                    help="an extract_v2.py --keep-head-outputs checkpoint directory")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="the JSON file written")
    ap.add_argument("--label-set", default=None,
                    help="for a run outside the contraction tree (manifest rung 'none'): its "
                         "flavour-boundary partition, a column of FLAV_MAP")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"{a.out} exists; not overwritten")
        return 0
    man = json.loads((a.head / "manifest.json").read_text())
    rung = man["rung"]
    if rung == "none" and not a.label_set:
        raise SystemExit(f"FATAL: {a.head}: a label set outside the contraction tree has no QCD node; "
                         "name it with --label-set")
    if rung != "none" and a.label_set not in (None, rung):
        raise SystemExit(f"FATAL: --label-set {a.label_set} for a {rung} run")
    label_set = a.label_set or rung
    nodes = qcd_nodes(label_set)
    if not nodes:
        raise SystemExit(f"FATAL: {label_set} has no output node that is QCD alone")
    z = np.load(a.head / HEAD_FILE)
    lab = z["label188"].astype(np.int64)
    qcd = (lab >= QCD_LABELS[0]) & (lab < QCD_LABELS[1])
    res = (lab >= 0) & (lab < QCD_LABELS[0])
    logits = z["logits"]
    if logits.shape[1] != man["num_classes"] or max(nodes) >= logits.shape[1]:
        raise SystemExit(f"FATAL: {logits.shape[1]} logits for {label_set}'s {man['num_classes']} classes")
    s_all = score(logits, nodes)
    s, msd = s_all[qcd], z["jet_sdmass"][qcd].astype(np.float64)
    doc = {"head": str(a.head), "run_dir": man["run_dir"], "rung": rung, "label_set": label_set,
           "tag": man["tag"],
           "checkpoint_sha256": man.get("checkpoint_sha256"),
           "head_outputs_sha256": (man.get("head_outputs") or {}).get("sha256"),
           "qcd_nodes": nodes, "n_qcd": int(qcd.sum()), "n_sig": int(res.sum()), "n_jets": int(lab.size),
           "score": "1 - p_QCD, p_QCD = summed softmax probability of the QCD nodes",
           "signal": "resonant test jets, native labels 0..160; eff_sig at the same cut",
           "bins_jet_sdmass": BINS.tolist(), "divergence": "Jensen-Shannon, base 2; js_distance = its square root",
           **sculpting(s, msd, s_all[res]),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(a.out)
    for eff, w in doc["working_points"].items():
        print(f"QCD efficiency {eff}: {w['n_pass']:,} of {doc['n_qcd']:,} pass, JSD {w['jsd']:.4f}, "
              f"signal efficiency {w['eff_sig']:.4f}")
    w = doc["sig_eff_50"]
    print(f"signal efficiency {w['eff_sig']:.4f}: R50 {w['rejection']:.1f}, JSD50 {w['jsd']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
