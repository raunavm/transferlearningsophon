#!/usr/bin/env python3
"""Mass sculpting of the fine-tuned JetClass-II networks at 10^6 jets (plan 2026-11-08, step 2), on the jets
the pretraining heads' sculpting uses, so the two are read side by side with one floor.

WHY THESE JETS (2026-11-08). On the class-balanced test set weighted back to the natural QCD mix the QCD jets
had a Kish n_eff of about 14,400, a floor of 0.10-0.17 at 1% QCD efficiency against distances of 0.044-0.056:
noise. Each cell is instead scored on the first 2,000,000 TEST2M jets, natural composition, unweighted.

Cells: experiments/FT/roc_points.ft_cells (DONE, no NaN loss, one per pretraining run); model mtx-<init>, or
scratch-s<S> from scratch. Inputs: <head-root>/<model>/head_outputs.npz (scripts/build_head_jobs.py, the ft-head
job: extract_v2.py --keep-head-outputs with the cell's net_best_epoch_state.pt), checked by roc_points.ft_head
(188 class outputs of the cell's own checkpoint, sha256), and its labels and jet_sdmass equal to the pretraining
heads' (--ref), or FATAL. Score, cuts, histograms and divergences: mass_sculpting.head_doc as it is (1 - the
summed softmax of the 27 native QCD outputs, nodes 161..187 of L188; no weights), so each document has
sculpting_v3's schema and the tables' floor is F10b's; beside it model, cell, source_tree (budget, budget_long or
budget_long_fulldecay) and method.

    python3 experiments/EVAL/sculpting_ft.py --tree /data/results/ft_v2/budget/leg_jc2b --inits l188-s1 ... \\
        --out-dir /data/results/eval/v2_head/sculpting_ft
One JSON per model (<out-dir>/<model>.json), written whole; one that exists is skipped, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
REF = pathlib.Path("/data/results/eval/v2_head/mtx-l188-s1/test/best70")   # build_head_jobs OBS_MODEL's head


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("mass_sculpting", "experiments/EVAL/mass_sculpting.py")
RP = _load("roc_points", "experiments/FT/roc_points.py")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=True, type=pathlib.Path, help="<tree>/<init>/N<N>/s<S>/")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--inits", nargs="+", required=True)
    ap.add_argument("--head-root", type=pathlib.Path, default=RP.FT_HEAD_ROOT)
    ap.add_argument("--ref", type=pathlib.Path, default=REF, help="a pretraining head's directory: the jets")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    ref = np.load(a.ref / MS.HEAD_FILE)
    ref = {k: ref[k] for k in ("label188", "jet_sdmass")}
    for init, seed, cell in RP.ft_cells(a.tree, a.n, a.inits):
        model = f"{init}-{seed}" if init.startswith("scratch") else f"mtx-{init}"
        out = a.out_dir / f"{model}.json"
        if out.exists():
            print(f"skip {out}: exists", flush=True)
            continue
        head, _man = RP.ft_head(a.head_root, model, cell)
        z = np.load(head / MS.HEAD_FILE)
        if not all(np.array_equal(z[k], v) for k, v in ref.items()):
            raise SystemExit(f"FATAL: {head}: not the jets of {a.ref} (labels or jet_sdmass differ)")
        doc = {"model": model, "cell": str(cell), "source_tree": RP.source_tree(cell), "n_train": a.n,
               "method": "fine-tuning", **MS.head_doc(head, "L188"),
               "mass_sculpting_sha256": hashlib.sha256((REPO / "experiments/EVAL/mass_sculpting.py").read_bytes()).hexdigest(),
               "script": "experiments/EVAL/sculpting_ft.py",
               "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
        MS.write(out, doc)
        wp = doc["working_points"]
        print(f"{model} ({doc['source_tree']}): JSD " + ", ".join(f"{e} {w['jsd']:.5f}" for e, w in wp.items())
              + f", JSD50 {doc['sig_eff_50']['jsd']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
