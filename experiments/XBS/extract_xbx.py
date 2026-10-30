#!/usr/bin/env python3
"""experiments/EVAL/extract_v2.py, unchanged, for X->bb and X->bs: every jet of a staged split,
its class-token features and, for a model with an output layer, the head columns of
experiments/XBS/xbx.py xbx_columns in the model's vocabulary instead of extract_v2's anomaly
columns.

extract_v2.py is used as it is, not changed for X->bx: the probe extractions clone it at
their own tags, and their pins must stay current (tests/test_spec_pins.py). Its own flags
select every jet (FLAGS), and its head-score function -- which run() looks up at call time --
is replaced by xbx_columns for the vocabulary --arm. The vocabulary's nodes are written to <out>/xbx.json, once: a split
extracted before (the QCD evaluation jets every signal shares) keeps its record.
experiments/XBS/fit_xbx.py refuses a split whose features or head scores are not every jet's,
in order.

    python3 experiments/XBS/extract_xbx.py --arm L188 <extract_v2.py's own arguments>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
# every jet keeps its features and its head scores; --max-jets far beyond any split turns off
# extract_v2's check that the stream outlasts the head prefix, which these flags make moot
FLAGS = ["--feature-classes", "--prefix-features", "2000000000", "--no-anomaly-rows",
         "--head-prefix", "2000000000", "--diag-stride", "1", "--max-jets", "2000000000"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EX = _load("extract_v2", "experiments/EVAL/extract_v2.py")
X = _load("xbx", "experiments/XBS/xbx.py")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    a, rest = ap.parse_known_args(argv)
    clash = [f for f in FLAGS if f.startswith("--") and f in rest]
    if clash:
        raise SystemExit(f"FATAL: {clash} are set by extract_xbx.py")
    groups = X.node_groups(a.arm)
    k = int(rest[rest.index("--num-classes") + 1])
    if k and k != groups["k"]:
        raise SystemExit(f"FATAL: {a.arm} has {groups['k']} classes, the model {k}")
    EX.head_score_columns = lambda logits, rung, signals: X.xbx_columns(logits, groups)
    rc = EX.main(rest + FLAGS)
    meta = pathlib.Path(rest[rest.index("--out") + 1]) / "xbx.json"
    if not meta.exists():
        meta.write_text(json.dumps({"arm": a.arm, "groups": groups}, indent=1))
    elif json.loads(meta.read_text())["arm"] != a.arm:
        raise SystemExit(f"FATAL: {meta} records another vocabulary than {a.arm}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
