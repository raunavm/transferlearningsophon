#!/usr/bin/env python3
"""Leg 2 of the fine-tuning legs: the DOMAIN-SHIFT curve, as a real artifact.

WHY THIS EXISTS WHEN LEG 2 ALREADY HAS NUMBERS. weaver prints a "Test metric"
line into each leg-2 cell's predict.log, so the leg-2 table could be, and first
was, read by eye off 54 logs. docs/REVIEW.md requires that any claim "X equals N"
be reproducible by a committed script, and a number transcribed from a log is not
-- there is no record of which cells were included, which were skipped, or
whether a `.partial` directory was counted twice. This turns the same logs into
leg2_metrics.json with the same shape leg1_metrics.py emits, so the two legs can
be plotted and compared by one reader.

IT PARSES, SO IT IS STRICT ABOUT PARSING. A log that yields no metric is an
ERROR, not a cell quietly missing from the mean: a silently dropped cell shrinks
a denominator and moves a published average with nothing to show for it. Cells
whose directory name carries `.partial.` are interrupted attempts superseded by a
finished cell and ARE skipped -- by name, since a partial directory can hold a
complete-looking log from before the interruption.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st

METRIC = re.compile(r"Test metric ([0-9.]+)")


def cell_metric(log: pathlib.Path) -> float:
    """The LAST Test metric in the log. weaver prints one per predict pass, and a
    retried pass appends rather than truncating, so the last is the live one."""
    hits = METRIC.findall(log.read_text(errors="replace"))
    if not hits:
        raise SystemExit(f"FATAL: no 'Test metric' in {log}. A cell that cannot "
                         "be parsed must not be silently dropped from a mean.")
    return float(hits[-1])


def discover(root: pathlib.Path):
    """(init, N, seed, predict.log). Reference runs sit one level shallower."""
    out = []
    for log in sorted(root.rglob("predict.log")):
        rel = log.relative_to(root).parts[:-1]
        if any(".partial." in p for p in rel):
            continue
        if len(rel) == 3:
            out.append((rel[0], rel[1], rel[2], log))
        elif len(rel) == 1:
            out.append((rel[0], "ref", "s1", log))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    cells = discover(a.root)
    if not cells:
        raise SystemExit(f"FATAL: no predict.log under {a.root}")
    res = {}
    for init, n, seed, log in cells:
        res.setdefault(init, {}).setdefault(n, {})[seed] = {"accuracy": cell_metric(log)}
        print(f"  {init:16} {n:10} {seed:4} acc={res[init][n][seed]['accuracy']:.5f}",
              flush=True)

    summary = {}
    for init, per_n in res.items():
        for n, per_seed in per_n.items():
            v = [c["accuracy"] for c in per_seed.values()]
            summary.setdefault(init, {})[n] = {
                "n_seeds": len(v),
                "accuracy_mean": st.mean(v),
                "accuracy_sd": st.stdev(v) if len(v) > 1 else None,
            }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "leg2_metrics.json").write_text(
        json.dumps({"cells": res, "summary": summary}, indent=1))
    print(f"\n{len(cells)} cells -> {a.out / 'leg2_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
