#!/usr/bin/env python3
"""Which paper runs' sensitivity checkpoint (bestval, the global best epoch) is their primary (best70,
the first maximum within epochs 70-79), by extract_v2.py's own resolvers (2026-11-07, PI: "Checkpoint
sensitivity, bestval only"). Where the two are one file, the bestval probes are the best70 probes and
need no refit.

    python3 experiments/EVAL/checkpoint_identity.py --runs configs/arms/v2_paper_runs.txt \\
        --root /data/results/mtx_v2 --out /data/results/eval/v2_ckpt_identity/identity.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")


def identity(run_dir: pathlib.Path) -> dict:
    (_, b70), (_, bv) = xv.resolve_checkpoints(run_dir, ["best70", "bestval"])
    return {"best70": b70.name, "bestval": bv.name, "same_file": b70.resolve() == bv.resolve()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, type=pathlib.Path)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    runs = [l.strip() for l in a.runs.read_text().splitlines() if l.strip() and not l.startswith("#")]
    rec = {r: identity(a.root / r) for r in runs}
    doc = {"runs": rec, "n_runs": len(rec), "n_same_file": sum(v["same_file"] for v in rec.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    print(f"{doc['n_same_file']} of {doc['n_runs']} runs: bestval is best70's file; {a.out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
