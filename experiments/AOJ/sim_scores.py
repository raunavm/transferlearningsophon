#!/usr/bin/env python3
"""The real-data scores, on JetClass-II simulation: what separates model from domain.

The real-data yield of one pretrained model is (tops in the data) x (that model's
efficiency on REAL tops at the cut). The two cannot be told apart in data alone,
so each model's score is computed here, by the same code, on simulated jets with a
known label: JetClass-II test files read through
configs/finetune/JetClassII_base_selAspenOpenJets.yaml (the AspenOpenJets staging
selection, no reweighting). experiments/AOJ/realdata_checks.py then applies each
model's DATA cut to them.

Input: one experiments/EVAL/extract_features.py --save-logits directory per model,
all over the SAME file list with one loader worker, so their rows are the same jets
in the same order. Output, per model, <out>/scores_<name>.npz: the scores of
discriminants.SCORES named in --scores (float32 log-odds), and once per directory
<out>/jets.npz: jet_pt, jet_eta, jet_sdmass (float32) and the native label
(int16). A second model must reproduce jets.npz exactly or is refused -- a model
scored on other jets cannot be compared jet for jet.

Run:  python3 experiments/AOJ/sim_scores.py --name l188-s1 --rung L188 \\
          --extract-dir /scratch/extract/l188-s1 --out /data/results/aoj/sim_v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


disc = _load("discriminants")
JET_COLUMNS = ("jet_pt", "jet_eta", "jet_sdmass")


def jets_digest(jets: dict) -> str:
    """sha256 over the jet columns in a fixed order: which jets, in which order."""
    h = hashlib.sha256()
    for k in (*JET_COLUMNS, "label"):
        h.update(np.ascontiguousarray(jets[k]).tobytes())
    return h.hexdigest()


def reduce_extract(ext: pathlib.Path, rung: str, scores) -> tuple[dict, dict, dict]:
    """(jets, scores, manifest) of one extraction directory."""
    manifest = json.loads((ext / "extract_manifest.json").read_text())
    logits = np.load(ext / "logits.npy", mmap_mode="r")
    # a mass-output model's last column is the mass regression, not a class
    lo, hi = manifest.get("logit_columns", {}).get("class_logits", [0, logits.shape[1]])
    obs = np.load(ext / "observers.npz")
    missing = [k for k in JET_COLUMNS if k not in obs.files]
    if missing:
        raise SystemExit(f"FATAL: {ext} has no observer(s) {missing}")
    label = np.load(ext / "label188.npy")
    n = len(label)
    if logits.shape[0] != n or any(len(obs[k]) != n for k in JET_COLUMNS):
        raise SystemExit(f"FATAL: {ext}: {logits.shape[0]:,} logit rows, {n:,} labels, "
                         f"observers {[len(obs[k]) for k in JET_COLUMNS]}")
    x = np.asarray(logits[:, lo:hi])
    out = {s: disc.contrast(x, rung, s).astype(np.float32) for s in scores}
    jets = {k: np.asarray(obs[k], dtype=np.float32) for k in JET_COLUMNS}
    jets["label"] = label.astype(np.int16)
    return jets, out, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rung", required=True, help="label-map column of this head, e.g. L188")
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", nargs="+", choices=list(disc.SCORES), default=["three_prong", "prong_only"])
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    jets, scores, manifest = reduce_extract(pathlib.Path(a.extract_dir), a.rung, a.scores)
    out.mkdir(parents=True, exist_ok=True)
    jets_path = out / "jets.npz"
    if jets_path.exists():
        have = np.load(jets_path)
        for k, v in jets.items():
            if not np.array_equal(have[k], v):
                raise SystemExit(f"FATAL: {a.name}'s {k} differs from {jets_path}; it was scored on "
                                 "other jets (another file list, order or loader)")
    else:
        # Several models finish at once (the jobs run three per GPU, six jobs), so the
        # file is written under a private name and renamed into place: a reader sees
        # all of it or none. Whoever wins, every model records the digest of ITS jets
        # (jets_sha256) and realdata_checks.py refuses a model whose digest differs.
        tmp = out / f".jets.{a.name}.npz"
        np.savez(tmp, **jets)
        os.replace(tmp, jets_path)
    np.savez(out / f"scores_{a.name}.npz", **{f"{s}_logodds": v for s, v in scores.items()})
    summary = dict(name=a.name, rung=a.rung, n_jets=int(len(jets["label"])),
                   checkpoint=manifest.get("checkpoint"), checkpoint_sha256=manifest.get("checkpoint_sha256"),
                   data_config=manifest.get("data_config"), data_config_sha256=manifest.get("data_config_sha256"),
                   n_test_files=manifest.get("n_test_files"),
                   label188_sha256=hashlib.sha256(jets["label"].astype(np.int16).tobytes()).hexdigest(),
                   jets_sha256=jets_digest(jets),
                   scores={s: dict(numerator=disc.nodes(a.rung, disc.SCORES[s][0]),
                                   denominator=disc.nodes(a.rung, disc.SCORES[s][1])) for s in a.scores})
    (out / f"scores_{a.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("name", "rung", "n_jets", "checkpoint")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
