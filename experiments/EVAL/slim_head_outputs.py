#!/usr/bin/env python3
"""The reduced head-output file of a fine-tuned network (PI 2026-11-21, storage): only what the sculpting read-out and
Table A10 read, and each jet's class label: label188, jet_sdmass, rows and the score 1 - p_QCD
(mass_sculpting.score on the label set's QCD nodes, float64), instead of every class logit (about 0.8 GB a network).

Reads an extract_v2.py --keep-head-outputs checkpoint directory (pod-local), writes <out>/head_outputs.npz and its
manifest, and refuses unless mass_sculpting.head_doc gives the same document, number for number, from the full and the
reduced file (only the file paths and hashes differ).

    python3 experiments/EVAL/slim_head_outputs.py --full X/net_best_epoch_state --label-set L188 --out OUT/<model>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mass_sculpting", REPO / "experiments/EVAL/mass_sculpting.py")
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)
DIFFER = {"head", "head_outputs_sha256"}      # the only keys of head_doc that name the file read


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, type=pathlib.Path, help="a --keep-head-outputs checkpoint directory")
    ap.add_argument("--label-set", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    full = MS.head_doc(a.full, a.label_set)
    z = np.load(a.full / MS.HEAD_FILE)
    nodes = MS.qcd_nodes(a.label_set)
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "head_outputs.tmp.npz"
    np.savez(tmp, rows=z["rows"], label188=z["label188"], jet_sdmass=z["jet_sdmass"],
             score_notqcd=MS.score(z["logits"], nodes), score_label_set=np.array(a.label_set),
             qcd_nodes=np.asarray(nodes, dtype=np.int64))
    man = json.loads((a.full / "manifest.json").read_text())
    tmp_dir = a.out / ".slim_check"
    tmp_dir.mkdir(exist_ok=True)
    shutil.copy(tmp, tmp_dir / MS.HEAD_FILE)
    (tmp_dir / "manifest.json").write_text(json.dumps(man))
    slim = MS.head_doc(tmp_dir, a.label_set)
    shutil.rmtree(tmp_dir)
    diff = sorted(k for k in set(full) | set(slim) if k not in DIFFER and full.get(k) != slim.get(k))
    if diff:
        tmp.unlink()
        raise SystemExit(f"FATAL: the reduced file changes {diff}")
    tmp.rename(a.out / MS.HEAD_FILE)
    man["head_outputs"] = {**(man.get("head_outputs") or {}), "reduced": "label188, jet_sdmass, rows, score_notqcd "
                           f"(1 - p_QCD on {a.label_set}'s QCD nodes); the logits are not kept",
                           "sha256": hashlib.sha256((a.out / MS.HEAD_FILE).read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{a.out}: reduced, head_doc identical ({len(full)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
