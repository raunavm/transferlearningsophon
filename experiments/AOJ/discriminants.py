#!/usr/bin/env python3
"""Two-prong and three-prong discriminants, the SAME physical score at every
label granularity.

    two_prong   = sum P(hadronic two-parton two-prong nodes)
                  / (that + sum P(QCD nodes))
    three_prong = the same with the hadronic three-parton three-prong nodes

WHY NODE SELECTION IS BY STRUCTURAL NAME AND NOT BY INDEX. A 17-class head has
one output called `2P_HAD_2PARTON`; a 43-class head has three
(`2P_HAD_2PARTON|B`, `|C`, `|LG`); a 162- or 188-class head has ten native
classes that contract into it. Picking outputs by index, or by a hand-written
class list per head, builds a DIFFERENT discriminant per model and then compares
them. Here the numerator is defined ONCE, as a coefficient vector over the 188
native classes -- 1 on every class whose structural name (the part before `|`)
is the requested one -- and each head realises it through
scripts/build_usecase_survival.py's criterion: constructible iff the vector is
constant on every group of that head's partition (Sophon's class-division
property, arXiv:2405.12972). members() refuses a head on which it is not.

QCD nodes come from experiments/EVAL/anomaly.py::node_roles -- a node is QCD only
if EVERY native class in it is -- so the denominator is the one the anomaly
scores already use.

P_sig / (P_sig + P_QCD) = sigmoid(logsumexp(sig logits) - logsumexp(QCD logits)):
the softmax normaliser cancels, so the score is computed from raw logits in log
space and PERSISTED AS THE LOG-ODDS in float16. A probability in float16 has a
spacing of 5e-4 on [0.5, 1), which is where a 1 % working point lives, so the
tail would collapse into ties; the log-odds keeps its ordering.

Run (after experiments/EVAL/extract_features.py --save-logits):
    python3 experiments/AOJ/discriminants.py --name sophon-public --rung L188 \
        --extract-dir /scratch/extract/sophon-public \
        --staged /scratch/staged/RunG_batch0.parquet ... --out /data/results/aoj/feasibility_v1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp

REPO = pathlib.Path(__file__).resolve().parents[2]
STRUCTURES = {"two_prong": "2P_HAD_2PARTON", "three_prong": "3P_HAD_3PARTON"}
# Rungs whose node names carry the structural prefix. The finer ones (L188, L162)
# name nodes after native classes, so their members come through these.
NAMED_RUNGS = ["R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]
JET_FLOATS = ["jet_pt", "jet_eta", "jet_sdmass", "aoj_jet_pt", "aoj_jet_eta",
              "aoj_pn_WvsQCD", "aoj_pn_TvsQCD", "aoj_pn_HbbvsQCD"]
JET_INTS = ["run", "lumi", "event"]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_anomaly = _load("experiments/EVAL/anomaly.py", "anomaly")
_survival = _load("scripts/build_usecase_survival.py", "build_usecase_survival")


def native_classes(structure: str) -> set[int]:
    """Native labels whose structural name is `structure`, checked to be the same
    at every rung that names its nodes -- that is what makes it granularity-
    independent rather than a property of one column."""
    out = set()
    for r in _anomaly.read_map():
        names = {r[f"{g}_name"].split("|")[0] for g in NAMED_RUNGS}
        if len(names) != 1:
            raise SystemExit(f"FATAL: {r['class_name']} has structural names {names} "
                             f"across {NAMED_RUNGS}; the prefix is not rung-independent")
        if names == {structure}:
            out.add(int(r["jet_label"]))
    if not out:
        raise SystemExit(f"FATAL: no native class has structural name {structure!r}")
    return out


def members(rung: str, structure: str) -> list[int]:
    """Output indices of a `rung` head that realise `structure`."""
    _, groups = _survival.read_map()
    if rung not in groups:
        raise SystemExit(f"FATAL: rung {rung} not in {_survival.RUNGS}")
    coeff = {i: 1.0 for i in native_classes(structure)}
    if not _survival.group_constant(coeff, groups[rung]):
        raise SystemExit(f"FATAL: {structure} is not constructible at {rung}: a group "
                         "straddles it, so no sum of this head's outputs equals it")
    return sorted(int(g) for g, labs in groups[rung].items() if coeff.get(labs[0]))


def qcd_nodes(rung: str) -> list[int]:
    return sorted(_anomaly.node_roles(rung)[2])


def n_outputs(rung: str) -> int:
    return len(_survival.read_map()[1][rung])


def log_odds(logits: np.ndarray, rung: str, structure: str) -> np.ndarray:
    """log[ sum P(structure nodes) / sum P(QCD nodes) ], from raw logits."""
    if logits.ndim != 2 or logits.shape[1] != n_outputs(rung):
        raise SystemExit(f"FATAL: logits {logits.shape} are not a {rung} head "
                         f"({n_outputs(rung)} outputs)")
    x = logits.astype(np.float64)
    return logsumexp(x[:, members(rung, structure)], axis=1) - logsumexp(x[:, qcd_nodes(rung)], axis=1)


def score(logits: np.ndarray, rung: str, structure: str) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds(logits, rung, structure)))


def staged_jets(paths) -> dict:
    """Per-jet columns of the staged files, in the order they were given."""
    import awkward as ak
    cols = JET_FLOATS + JET_INTS
    parts = [ak.from_parquet(p, columns=cols) for p in paths]
    return {c: np.concatenate([ak.to_numpy(t[c]) for t in parts]) for c in cols}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rung", required=True, help="label-map column of this head, e.g. L188")
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--staged", nargs="+", required=True,
                    help="staged parquet files, in the --data-test order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ext, out = pathlib.Path(a.extract_dir), pathlib.Path(a.out)
    logits = np.load(ext / "logits.npy", mmap_mode="r")
    obs = np.load(ext / "observers.npz")
    jets = staged_jets(a.staged)

    # ROW ALIGNMENT, asserted and not assumed. extract_features.py reads files in
    # order with one worker and no shuffling, and the config's selection repeats
    # the staging one, so row i of the logits must be row i of the staged files.
    # run/lumi/event are taken from the staged files because extract_features.py
    # casts observers to float32, which cannot hold a CMS event number.
    n = len(jets["event"])
    if logits.shape[0] != n:
        raise SystemExit(f"FATAL: {logits.shape[0]:,} logit rows vs {n:,} staged jets")
    for k in ("jet_sdmass", "aoj_jet_pt"):
        if not np.array_equal(obs[k], jets[k].astype(np.float32)):
            raise SystemExit(f"FATAL: observer {k} differs from the staged files; the "
                             "logits are not row-aligned with them")

    out.mkdir(parents=True, exist_ok=True)
    lo = {s: log_odds(np.asarray(logits), a.rung, st) for s, st in STRUCTURES.items()}
    np.savez(out / f"scores_{a.name}.npz",
             **{f"{s}_logodds": v.astype(np.float16) for s, v in lo.items()})

    jets_path = out / "jets.npz"
    if jets_path.exists():           # a second model: must be the same jets
        if not np.array_equal(np.load(jets_path)["event"], jets["event"]):
            raise SystemExit(f"FATAL: {jets_path} holds different jets than {a.name}")
    else:
        np.savez(jets_path, **{k: jets[k].astype(np.float32) for k in ("jet_pt", "jet_eta",
                 "jet_sdmass", "aoj_jet_pt", "aoj_jet_eta")},
                 **{k: jets[k].astype(np.float16) for k in JET_FLOATS if k.startswith("aoj_pn_")},
                 **{k: jets[k] for k in JET_INTS})

    manifest = json.loads((ext / "extract_manifest.json").read_text())
    summary = dict(
        name=a.name, rung=a.rung, n_jets=n, n_outputs=n_outputs(a.rung),
        checkpoint=manifest.get("checkpoint"), checkpoint_sha256=manifest.get("checkpoint_sha256"),
        members={s: members(a.rung, st) for s, st in STRUCTURES.items()},
        qcd_nodes=qcd_nodes(a.rung),
        median_logodds={s: float(np.median(v)) for s, v in lo.items()})
    (out / f"scores_{a.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
