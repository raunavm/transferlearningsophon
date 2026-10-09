"""The v2 linear probes in the fine-tuning read-outs' format, so experiments/FIGS/make_tables.py
reads them as it reads fine-tuning: one file set per checkpoint rule, `linprobe` (the primary,
best70) and `linprobe_bn` (its BatchNorm twin; the untrained trunk's init in both), with the
cells of experiments/FT/{leg1,leg2,bench}_metrics.py keyed init / N / "s1" (a probe has no
fine-tuning seed; s1 is the seed-1 training subset it was fitted on):

  finetune/<rule>_leg1_metrics.json       {"cells": {init: {N: {"s1": cell}}}}   JetClass-II
  finetune/<rule>_leg2_metrics.json       the same, JetClass
  benchmarks/<rule>_bench_metrics.json    {"cells": {"top"|"qg": {init: {N: {"s1": cell}}}}}
  benchmarks/<rule>_bench_metrics_herwig.json   {"cells": {"qg": ...}}, the q/g probes on Herwig
  finetune/<rule>_pair_metrics.json       {"cells": {pair: {init: {N: {"s1": cell}}}}}, the binary
                                          probes between JetClass-II classes (linear_probe_v2.PAIRS)

An init is the model name without "mtx-" (l188-s1, init-s1), as the fine-tuning read-outs name
it. Every model of the grid and every untrained trunk must have a fit; a missing one is fatal.

    python3 experiments/EVAL/linear_probe_summary.py --fits /data/results/eval/v2_linprobe/fits \\
        --out experiments/FIGS/data/v2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RULES = {"linprobe": "best70", "linprobe_bn": "best70_bn"}
FILES = {"leg1": ("finetune", "leg1_metrics.json", "jc2"),
         "leg2": ("finetune", "leg2_metrics.json", "jc1"),
         "bench": ("benchmarks", "bench_metrics.json", ("top", "qg")),
         "herwig": ("benchmarks", "bench_metrics_herwig.json", ("qg_herwig",)),
         "pairs": ("finetune", "pair_metrics.json", None)}


def expected_models() -> list[str]:
    spec = importlib.util.spec_from_file_location("build_linprobe_jobs", REPO / "scripts/build_linprobe_jobs.py")
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    return [m for m, *_ in bl.models()]


def summarise(fits: pathlib.Path, models: list[str]) -> dict:
    """{(sub, file name): document} for every rule."""
    docs = {m: json.loads((fits / f"{m}.json").read_text()) for m in models
            if (fits / f"{m}.json").exists()}
    missing = [m for m in models if m not in docs]
    if missing:
        raise SystemExit(f"FATAL: no linear-probe fit for {missing}")
    out = {}
    for rule, ckpt in RULES.items():
        for leg, (sub, fname, ds) in FILES.items():
            cells = {}
            for m, d in docs.items():
                c = d["cells"].get("init" if m.startswith("init-") else ckpt)
                if c is None:
                    raise SystemExit(f"FATAL: {m}'s fit has no {ckpt} probes")
                init = m.removeprefix("mtx-")
                if ds is None:
                    for n, per in c["jc2_pairs"].items():
                        for pair, v in per.items():
                            cells.setdefault(pair, {}).setdefault(init, {})[n] = {"s1": v}
                elif isinstance(ds, str):
                    cells[init] = {n: {"s1": v} for n, v in c[ds].items()}
                else:
                    for one in ds:
                        key = "qg" if one == "qg_herwig" else one
                        cells.setdefault(key, {})[init] = {n: {"s1": v} for n, v in c[one].items()}
            out[(sub, f"{rule}_{fname}")] = {
                "rule": rule, "checkpoint": ckpt, "kind": "linear probe on frozen class-token features",
                "fits": {m: hashlib.sha256((fits / f"{m}.json").read_bytes()).hexdigest() for m in docs},
                "cells": cells}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fits", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path, help="experiments/FIGS/data/v2")
    a = ap.parse_args(argv)
    for (sub, name), doc in summarise(a.fits, expected_models()).items():
        d = a.out / sub
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, d / name)
        print(f"{d / name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
