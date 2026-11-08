#!/usr/bin/env python3
"""Every test jet of the sibling pairs (probe_bal.SIBLING_PAIRS: X->ee and X->mumu), PI 2026-11-05.

The balanced test set holds 5,000 jets of each, so the X->ee vs X->mumu ROC stops at a rejection of
250 and the models that separate the pair have no curve at all. This stages ALL of their jets in the
fine-tuning test files (JC2_ROLES["ft_test"], the files the balanced test set is drawn from), with
the pretraining selection, as stage_jc2_balanced.py writes its sets (its census, writer and row ids):

  test.parquet     every selected test jet of the sibling classes, rows in a random order
  manifest.json    counts per class, the sources' sha256, the seed
  DONE

    python3 experiments/FT/stage_jc2_siblings.py --out /data/finetune/jc2_sib --workers 8
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SEED = 20261105


def _load(name, rel):
    """Registered in sys.modules, so the census workers can unpickle its functions by name."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    if getattr(sys.modules.get(name), "__file__", None) == spec.origin:
        return sys.modules[name]    # one module object per process, whichever file loaded it first
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SB = _load("stage_jc2_balanced", "experiments/FT/stage_jc2_balanced.py")


def sibling_labels(names: dict) -> list:
    """The native labels of every class in probe_bal.SIBLING_PAIRS, in label order."""
    pairs = _load("probe_bal", "experiments/EVAL/probe_bal.py").SIBLING_PAIRS
    label = {v: k for k, v in names.items()}
    return sorted(label[n] for sides in pairs.values() for side in sides for n in side)


def plan(src: np.ndarray, lab: np.ndarray, labels: list) -> np.ndarray:
    """The source ids of every jet of the given labels."""
    return np.sort(src[np.isin(lab, labels)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default="/jc2/jet_data")
    a = ap.parse_args(argv)
    if (a.out / "DONE").exists():
        raise SystemExit(f"FATAL: {a.out} is complete; staged data is never rebuilt")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    labels = sibling_labels(names)
    files = SB.MS.jc2_role_files("ft_test", a.root)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} source files missing, e.g. {missing[:3]}")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(SB.census_file, files, chunksize=4))
    src = np.concatenate([i * SB.SRC_MULT + rows for i, (_p, rows, _l) in enumerate(res)])
    lab = np.concatenate([l for _p, _r, l in res]).astype(np.int64)
    ids = plan(src, lab, labels)
    print(f"census: {lab.size:,} selected test jets, {ids.size:,} of labels {labels} ({time.time() - t0:.0f} s)", flush=True)
    out = SB.write(a.out, "test.parquet", ids, files, np.random.default_rng([SEED, 1]))
    manifest = {"script": "experiments/FT/stage_jc2_siblings.py", "seed": SEED, "files": files,
                "selection": [SB.MS.PT_LO, SB.MS.PT_HI, SB.MS.MSD_LO, SB.MS.MSD_HI],
                "labels": {str(c): names[c] for c in labels},
                "counts": {str(c): int((lab == c).sum()) for c in labels},
                "outputs": {"test.parquet": out}, "sha256": {"test.parquet": SB.sha256(a.out / "test.parquet")},
                "seconds": round(time.time() - t0, 1)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"DONE: {out['rows']:,} rows ({manifest['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
