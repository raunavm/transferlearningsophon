#!/usr/bin/env python3
"""The released Sophon checkpoint's CPU probes (PI 2026-11-23, LEDGER SH-13: "inference and probes only"), each run by
the very function that ran it for our networks, pointed at Sophon's stored features (SOPHON_ROOT, model
sophon-released, checkpoint released). Nothing here trains a network or re-extracts a feature.

  mlp        probe_bal.mlp_cells (tab:mlp's MLP; --datasets jc1 top qg now, jc2bal once the pair features exist: its
             JetClass-II block also fits the pairs) with probe_bal's roots set to SOPHON_ROOT -> <out>/<model>.json
  pairs      probe_bal.pair_cells on the v5 grid (the pair linear probes of Figs. 5, 10, 11), the probabilities kept as
             fits_v5_probs keeps ours -> <out>/<model>.json, <probs-out>/<model>/<ckpt>/jc2bal_pairs_<pair>.npz
  recall     Table 15: perclass_recall.probe_pred on JetClass (the saved v5 probe re-scored on the stored test features,
             its correct count asserted against the fit) and perclass_recall.recall_doc on the test jets of our
             networks (asserted equal to mtx-l188-s1's), as one more network -> <out>/probe_jc1.json
  massview   Table 4: a v2-layout view of sophon-public/features_massreg's first N_MASS rows (the TEST2M mass jets,
             asserted equal in label, jet pT, m_SD, generator-level m_SD and eta to the observers atlas_resolution.py
             reads), so atlas_resolution.py fits it unchanged -> <out>/features.npy, label188.npy, rows.npy, manifest.json

    python3 experiments/EVAL/probe_sophon.py mlp --datasets jc1 top qg --out /data/results/eval/bs3_sophon/fits_v5_mlp
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SOPHON_ROOT = pathlib.Path("/data/results/eval/v2_linprobe_sophon")
MODEL, CKPT = "sophon-released", "released"
MASSREG = pathlib.Path("/data/results/eval/sophon-public/features_massreg")
OBS = pathlib.Path("/data/results/eval/v2_head/mtx-l188-s1/test/best70/head_outputs.npz")   # build_head_jobs.OBS_MODEL
N_MASS = 2_000_000
OUR_JC1 = pathlib.Path("/data/results/eval/v2_linprobe/mtx-l188-s1/jc1/test/best70/label188.npy")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sophon_probe_bal(root: pathlib.Path = SOPHON_ROOT):
    """probe_bal with every root it reads at call time pointed at root (its load()'s default root is bound at
    definition, so load is wrapped)."""
    pb = _load("probe_bal", "experiments/EVAL/probe_bal.py")
    pb.NEW_ROOT = pb.OLD_ROOT = root
    pb.load = functools.partial(pb.load, root=root)
    return pb


def write(doc: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    print(f"{path} written", flush=True)


def head(pb, mode: str) -> dict:
    return {"model": MODEL, "checkpoints": [CKPT], "mode": mode, "feature_root": str(SOPHON_ROOT),
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "probe_bal_sha256": hashlib.sha256((REPO / "experiments/EVAL/probe_bal.py").read_bytes()).hexdigest()}


def run_mlp(datasets, out: pathlib.Path) -> dict:
    pb = sophon_probe_bal()
    probe = _load("probe", "experiments/EVAL/probe.py")
    doc = {**head(pb, "mlp"), "probe": "mlp", "mlp_threads": probe.MLP_THREADS, "datasets": list(datasets),
           "cpu_model": _load("latent_scale_probe", "experiments/EVAL/latent_scale_probe.py").cpu_model(),
           "cells": {CKPT: pb.mlp_cells(MODEL, CKPT, probe, list(datasets))}}
    write(doc, out / f"{MODEL}.json")
    return doc


def run_pairs(out: pathlib.Path, probs_out: pathlib.Path) -> dict:
    pb = sophon_probe_bal()
    pb.set_grid("v5")
    probe = _load("probe", "experiments/EVAL/probe.py")
    save = pb.lp.saver(probs_out, MODEL, CKPT)
    doc = {**head(pb, "pairs"), **pb.grid_record("v5"), "pair_val_fraction": pb.PAIR_VAL_FRACTION,
           "cells": {CKPT: {"jc2bal_pairs": pb.pair_cells(MODEL, CKPT, probe, save)}}}
    write(doc, out / f"{MODEL}.json")
    return doc


def run_recall(out: pathlib.Path, our_labels: pathlib.Path = OUR_JC1) -> dict:
    pr = _load("perclass_recall", "experiments/EVAL/perclass_recall.py")
    pr.PROBE_BAL = SOPHON_ROOT
    k, ds, _root, ft = pr.TASKS["jc1"]
    pr.TASKS["jc1"] = (k, ds, SOPHON_ROOT, ft)
    pred, y, rec = pr.probe_pred("jc1", MODEL, CKPT)
    ours = np.load(our_labels).astype(np.int64)
    if not np.array_equal(y, ours):
        pr.fatal(f"{MODEL}'s JetClass test jets are not our networks' ({our_labels})")
    pb = _load("paired_bootstrap", "experiments/STATS/paired_bootstrap.py")
    doc = pr.recall_doc({MODEL: pred}, y, k, {"SOPHON": [MODEL]}, pb.B_DEFAULT, pb.SEED_DEFAULT, pb.counts)
    doc["models"][MODEL].update(rec)
    doc.update({"script": "experiments/EVAL/probe_sophon.py recall (perclass_recall.probe_pred, recall_doc)",
                "task": "jc1", "mode": "probe", "n_train": pr.N_PROBE, "classes": list(pr.CLASSES_JC1),
                "same_jets_as": str(our_labels)})
    pr.write(doc, out / "probe_jc1.json")
    return doc


def run_massview(out: pathlib.Path, src: pathlib.Path = MASSREG, obs: pathlib.Path = OBS, n: int = N_MASS) -> dict:
    """The first n rows of the v1 cache as a v2 cache that holds every row of a uniform prefix (mass_resolution
    .v2_prefix), after checking they are the observers' jets."""
    o = np.load(obs)
    if not np.array_equal(o["rows"], np.arange(n)):
        raise SystemExit(f"FATAL: {obs}: rows are not 0..{n - 1}")
    lab = np.load(src / "label188.npy", mmap_mode="r")[:n]
    if not np.array_equal(lab, o["label188"]):
        raise SystemExit(f"FATAL: {src}: its first {n} labels are not the observers'")
    ob = np.load(src / "observers.npz")
    for k in ("jet_pt", "jet_sdmass", "genjet_sdmass", "jet_eta"):
        if not np.array_equal(np.asarray(ob[k][:n], np.float64), np.asarray(o[k], np.float64)):
            raise SystemExit(f"FATAL: {src}: {k} of its first {n} jets differs from the observers'")
    f = np.load(src / "features.npy", mmap_mode="r")
    out.mkdir(parents=True, exist_ok=True)
    # the labels in the observers' dtype: atlas_resolution compares the sha256 of their bytes
    lab = np.asarray(lab, dtype=o["label188"].dtype)
    for name, arr in (("features", np.asarray(f[:n])), ("label188", lab), ("rows", np.arange(n, dtype=np.int64))):
        tmp = out / f"{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{name}.npy")
    src_man = json.loads((src / "extract_manifest.json").read_text())
    man = {"prefix_features": n, "n_stream": n, "view_of": str(src), "rows": f"0..{n - 1} of the source",
           "source_manifest": src_man, "checkpoint_sha256": src_man.get("checkpoint_sha256"),
           "label188_sha256": hashlib.sha256(np.ascontiguousarray(lab).tobytes()).hexdigest(),
           "checked_against": str(obs), "script": "experiments/EVAL/probe_sophon.py massview"}
    write(man, out / "manifest.json")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("mlp", "pairs", "recall", "massview"))
    ap.add_argument("--datasets", nargs="+", default=["jc1", "top", "qg"])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--probs-out", type=pathlib.Path)
    a = ap.parse_args(argv)
    done = a.out / ("manifest.json" if a.mode == "massview" else "probe_jc1.json" if a.mode == "recall" else f"{MODEL}.json")
    if done.exists():
        print(f"{done} exists")
        return 0
    if a.mode == "pairs" and a.probs_out is None:
        raise SystemExit("FATAL: pairs needs --probs-out")
    {"mlp": lambda: run_mlp(a.datasets, a.out), "pairs": lambda: run_pairs(a.out, a.probs_out),
     "recall": lambda: run_recall(a.out), "massview": lambda: run_massview(a.out)}[a.mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
