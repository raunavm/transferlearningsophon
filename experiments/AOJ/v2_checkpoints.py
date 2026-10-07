#!/usr/bin/env python3
"""Link each v2 grid model a real-data shard scores to its checkpoint file.

The file is the one experiments/EVAL/extract_v2.py resolves for the same run and tag
(resolve_checkpoints: best70 the first maximum within epochs 70-79, wavg the weight
average, best70_bn best70's BatchNorm twin checked against its record), so the real data
reads exactly the checkpoints the frozen readouts of v2 read. A run without DONE is
refused: its window best and weight average are written at its end.

<links>/<name>.pt points at the file and <record> (JSON, name -> run_dir, tag,
checkpoint) says which. A record an earlier attempt of the shard wrote must say the same,
or nothing is linked: every model of a shard is scored from one file.

ONE FILE, SCORED ONCE. Where two names resolve to the same file (bestval is best70's epoch,
and then bestval_bn is best70_bn's twin), the later name is not scored: extract_v2.aliases,
the extraction's own rule, names the first, recorded as same_file_as, and <links>/<name>.same
holds that name, so the shard links the later name's scores to the first's.

Usage (in the shard pod, from the repository root):
    python3 experiments/AOJ/v2_checkpoints.py --links /workspace/ckpt --record OUT/checkpoints.json \\
        l188-s1-best70=/data/results/mtx_v2/mtx-l188-s1:best70 ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resolve(items: list[str]) -> dict:
    """{name: {run_dir, tag, checkpoint}} for NAME=RUN_DIR:TAG items."""
    xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
    out = {}
    for item in items:
        name, spec = item.split("=", 1)
        run_dir, tag = spec.rsplit(":", 1)
        if not (pathlib.Path(run_dir) / "DONE").exists():
            raise SystemExit(f"FATAL: {run_dir} has no DONE; the run is not complete")
        [(t, path)] = xv.resolve_checkpoints(pathlib.Path(run_dir), [tag])
        out[name] = dict(run_dir=run_dir, tag=t, checkpoint=str(path))
    for name, first in xv.aliases([(n, pathlib.Path(r["checkpoint"])) for n, r in out.items()]).items():
        if first != name:
            out[name]["same_file_as"] = first
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--links", required=True, type=pathlib.Path)
    ap.add_argument("--record", required=True, type=pathlib.Path)
    ap.add_argument("models", nargs="+", metavar="NAME=RUN_DIR:TAG")
    a = ap.parse_args(argv)
    got = resolve(a.models)
    if a.record.exists() and json.loads(a.record.read_text()) != got:
        raise SystemExit(f"FATAL: {a.record} records other checkpoints than resolve now; "
                         "a shard scores every model from one file")
    a.links.mkdir(parents=True, exist_ok=True)
    for name, r in got.items():
        link = a.links / f"{name}.pt"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(r["checkpoint"])
        (a.links / f"{name}.same").unlink(missing_ok=True)
        if "same_file_as" in r:
            (a.links / f"{name}.same").write_text(r["same_file_as"])
    if not a.record.exists():
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps(got, indent=1))
    print(f"{len(got)} checkpoints linked in {a.links}, recorded in {a.record}; "
          f"{sum('same_file_as' in r for r in got.values())} the file of another")
    return 0


if __name__ == "__main__":
    sys.exit(main())
