#!/usr/bin/env python3
"""Join the per-shard outputs of the full real-data run into the inputs of ONE fit.

The 80 AspenOpenJets files are scored in ten shards of eight
(scripts/build_aoj_jobs.py), each writing jets.npz, scores_<model>.npz,
scores_<model>.json and closure.json. The top peak is fitted once, on all of
them, so the fit sees the whole sample and the decorrelation map is built from
every jet -- not ten fits averaged afterwards.

Refuses, rather than repairs, anything that would make the joined arrays mean
something different from one pass over all the files:
  - a model scored in some shards and not others (its scores would be shorter
    than jets.npz, or silently misaligned against it);
  - a model scored from a different checkpoint in different shards;
  - score and jet row counts that disagree inside a shard;
  - the same collision event in two shards (a file staged twice).

Closure hard flags are carried through as the union over shards, each tagged
with its shard, so peak_fit.py's veto sees every one.

Run:  python3 experiments/AOJ/merge_shards.py --shards S0 S1 ... --out OUT
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def models_in(shard: pathlib.Path) -> set[str]:
    return {p.stem.removeprefix("scores_") for p in shard.glob("scores_*.npz")}


def merge(shards: list[pathlib.Path], out: pathlib.Path) -> dict:
    names = [models_in(s) for s in shards]
    everywhere = set.intersection(*names)
    partial = set.union(*names) - everywhere
    if partial:
        raise SystemExit(f"FATAL: {sorted(partial)} scored in some shards only: "
                         + "; ".join(f"{s.name} lacks {sorted(partial - n)}"
                                     for s, n in zip(shards, names) if partial - n))
    if not everywhere:
        raise SystemExit("FATAL: no model is scored in any shard")

    jets = [dict(np.load(s / "jets.npz")) for s in shards]
    keys = set(jets[0])
    for s, j in zip(shards, jets):
        if set(j) != keys:
            raise SystemExit(f"FATAL: {s.name}/jets.npz has columns {sorted(j)}, expected {sorted(keys)}")
    n = [len(j["event"]) for j in jets]

    checkpoints = {}
    for name in sorted(everywhere):
        shas = {json.loads((s / f"scores_{name}.json").read_text()).get("checkpoint_sha256")
                for s in shards}
        if len(shas) != 1:
            raise SystemExit(f"FATAL: {name} was scored from different checkpoints across shards: {shas}")
        checkpoints[name] = shas.pop()
        for s, k in zip(shards, n):
            got = {f: len(v) for f, v in np.load(s / f"scores_{name}.npz").items()}
            if set(got.values()) != {k}:
                raise SystemExit(f"FATAL: {s.name}/scores_{name}.npz has {got} rows, jets.npz has {k:,}")

    ids = np.concatenate([np.stack([j["run"], j["lumi"], j["event"]], axis=1) for j in jets])
    if len(np.unique(ids, axis=0)) != len(ids):
        raise SystemExit(f"FATAL: {len(ids) - len(np.unique(ids, axis=0)):,} (run, lumi, event) "
                         "triples occur more than once across the shards")

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "jets.npz", **{k: np.concatenate([j[k] for j in jets]) for k in sorted(keys)})
    for name in sorted(everywhere):
        parts = [np.load(s / f"scores_{name}.npz") for s in shards]
        np.savez(out / f"scores_{name}.npz",
                 **{f: np.concatenate([p[f] for p in parts]) for f in parts[0].files})

    hard = []
    for s in shards:
        c = s / "closure.json"
        if not c.exists():
            raise SystemExit(f"FATAL: {s.name} has no closure.json; the input check did not run there")
        hard += [f"{s.name}: {flag}" for flag in json.loads(c.read_text())["hard_flags"]]
    (out / "closure.json").write_text(json.dumps({"hard_flags": hard, "per_shard": True}, indent=2))

    manifest = dict(shards=[str(s) for s in shards], jets_per_shard=n, n_jets=int(sum(n)),
                    models=sorted(everywhere), checkpoint_sha256=checkpoints,
                    closure_hard_flags=hard)
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard output directories, in order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = merge([pathlib.Path(s) for s in a.shards], pathlib.Path(a.out))
    print(f"merged {len(m['shards'])} shards, {m['n_jets']:,} jets, {len(m['models'])} models; "
          f"closure hard flags: {m['closure_hard_flags'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
