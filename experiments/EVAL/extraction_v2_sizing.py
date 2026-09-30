#!/usr/bin/env python3
"""What the v2 extraction needs, from the measured test-split class counts.

Inputs: class_counts.py's test_class_counts.json (every selected test jet per
native class, all 335 files), the committed v1 probe results (the largest
rejection any model reached at 90 % signal efficiency, per task), and the
2,000,000-jet v1 label vector (the prefix whose rows the label-recovery and
anomaly analyses read). Output, per probe task: the background jets in the
split and in its test part at each candidate split, the passing jets expected
at the largest v1 rejection, whether that reaches MIN_PASS; and the storage the
extraction writes for the v2 grid (configs/arms/v2_grid.json) under the
checkpoint plans considered, against the volume's free space.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST_FRACTIONS = (0.2, 0.5, 0.6)            # v1's split, and two v2 candidates
PLANS = {  # checkpoints with features, checkpoints with head scores
    "primary features, heads at 11": (1, 11),
    "features at all 11": (11, 11),
}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--counts", required=True, type=pathlib.Path)
    ap.add_argument("--probe-files", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--prefix-labels", required=True, type=pathlib.Path)
    ap.add_argument("--free-bytes", type=float, required=True,
                    help="free space on /data now (df)")
    ap.add_argument("--size-bytes", type=float, required=True, help="volume size (df)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    cc = _load("class_counts", "experiments/EVAL/class_counts.py")
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    counts = json.loads(a.counts.read_text())
    prefix = np.bincount(np.load(a.prefix_labels).astype(np.int64), minlength=188)
    tasks = {f: cc.needs(np.asarray(counts["selected_per_class"]),
                         np.asarray(counts["in_window_per_class"]), a.probe_files, f)
             for f in TEST_FRACTIONS}
    grid = json.loads((REPO / "configs/arms/v2_grid.json").read_text())
    n_models = sum(x["runs"] for x in grid["arms"] if x["num_classes"] is not None)
    qcd, signals = xv.anomaly_classes()
    storage = {name: cc.storage(counts, prefix, xv.probe_classes(), qcd + list(signals.values()),
                                n_models, nf, nh)
               for name, (nf, nh) in PLANS.items()}
    used = a.size_bytes - a.free_bytes
    line85 = 0.85 * a.size_bytes
    for s in storage.values():
        s["fits_under_85pc"] = bool(used + s["bytes_total"] < line85)
        s["headroom_to_85pc_bytes"] = line85 - used
    res = {"inputs": {"counts": {"path": str(a.counts), "sha256": hashlib.sha256(
               a.counts.read_bytes()).hexdigest()},
                      "probe_files": [str(p) for p in a.probe_files],
                      "prefix_labels_sha256": hashlib.sha256(
                          np.load(a.prefix_labels).tobytes()).hexdigest()},
           "min_pass": cc.MIN_PASS, "eps_s": cc.EPS_S,
           "tasks_by_test_fraction": {str(f): t for f, t in tasks.items()},
           "n_models_v2_grid": n_models, "storage": storage,
           "volume": {"size_bytes": a.size_bytes, "free_bytes": a.free_bytes}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
