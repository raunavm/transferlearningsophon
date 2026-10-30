#!/usr/bin/env python3
"""What the v2 jobs have finished, read off /data every 30 min by the CronJob status-raunav (plan
2026-10-29, A9). The pod cannot list jobs (no RBAC), so it counts what the jobs leave behind: a
fine-tuning cell's DONE (<tree>/<init>/N<N>/s<S>/DONE; interrupted attempts, .partial.*, are not
counted), a probe refit's <model>.json (written last), and so on, each against the number expected,
with the newest one's time. A directory that does not exist yet counts 0. Written atomically.

    python3 experiments/MTX/status_v2.py --out /data/results/status/status.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil

FT = "/data/results/ft_v2"
EVAL = "/data/results/eval"
CELL = "*/N*/s*/DONE"
# name: (directory, glob of one completion file under it, number expected).
# Fine-tuning: the 27 paper runs x 4 training sizes x 1 fine-tuning seed per leg; the V100
# from-scratch references 3 seeds x 4 sizes; the 188-class leg the 16 runs reported for
# fine-tuning x 4 sizes; the top-tagging recipe check 3 runs and 3 references at 1e4. Probes
# and X->bx: the 27 runs and the 3 untrained trunks (<model>.json, written last); heads: the 16 runs
# (sculpting.json, their last step); the masking control: the 12 hierarchy runs and the 3
# trunks x 2 masks (<mask>/fits/<model>.json).
ENTRIES = {
    "ft best70/leg1": (f"{FT}/best70/leg1", CELL, 108),
    "ft best70/leg2": (f"{FT}/best70/leg2", CELL, 108),
    "ft best70/leg_top": (f"{FT}/best70/leg_top", CELL, 108),
    "ft best70/leg_qg": (f"{FT}/best70/leg_qg", CELL, 108),
    "ft best70/leg1_188": (f"{FT}/best70/leg1_188", CELL, 64),
    "ft ref_v100/jc2/leg1": (f"{FT}/ref_v100/jc2/leg1", CELL, 12),
    "ft ref_v100/jc1/leg2": (f"{FT}/ref_v100/jc1/leg2", CELL, 12),
    "ft ref_v100/bench/leg_top": (f"{FT}/ref_v100/bench/leg_top", CELL, 12),
    "ft ref_v100/bench/leg_qg": (f"{FT}/ref_v100/bench/leg_qg", CELL, 12),
    "ft ref_v100/jc2/leg1_188": (f"{FT}/ref_v100/jc2/leg1_188", CELL, 12),
    "ft recipe_check": (f"{FT}/recipe_check/leg_top_e100/leg_top", CELL, 6),
    "probe fits_v3": (f"{EVAL}/v2_linprobe/fits_v3", "*.json", 30),
    "v2_head": (f"{EVAL}/v2_head", "*/sculpting.json", 16),
    "v2_fillcontrol fits": (f"{EVAL}/v2_fillcontrol", "*/fits/*.json", 30),
    "xbx fits_ihep": (f"{EVAL}/xbx/fits_ihep", "*.json", 30),
}


def count(root: pathlib.Path, pattern: str) -> tuple[int, float | None]:
    """(completion files, newest mtime) under root; none under .partial.* or written as .tmp."""
    hits = [p for p in root.glob(pattern)
            if not any(".partial." in q for q in p.relative_to(root).parts) and ".tmp" not in p.name]
    return len(hits), max((p.stat().st_mtime for p in hits), default=None)


def _iso(t: float | None) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def status(entries: dict = ENTRIES, data: str = "/data") -> dict:
    out = {}
    for name, (root, pattern, expected) in entries.items():
        n, newest = count(pathlib.Path(root), pattern)
        out[name] = {"done": n, "expected": expected, "complete": n >= expected, "newest": _iso(newest),
                     "path": f"{root}/{pattern}"}
    du = shutil.disk_usage(data) if os.path.isdir(data) else None
    return {"written_at": _iso(dt.datetime.now(dt.timezone.utc).timestamp()),
            "data_used_pct": None if du is None else round(100 * du.used / du.total, 1),
            "entries": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    doc = status()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, a.out)
    for name, e in doc["entries"].items():
        print(f"{name:28} {e['done']:4d} / {e['expected']:<4d} newest {e['newest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
