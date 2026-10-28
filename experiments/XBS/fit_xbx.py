#!/usr/bin/env python3
"""X->bb and X->bs per model (docs/PRESPEC_2026-09.md A5): every readout of one frozen model,
scored on the evaluation jets inside the m_SD window.

Input, from experiments/XBS/extract_xbx.py on the staged sets (experiments/XBS/stage_xbx.py):
    <root>/<model>/<sig>/{train,val,eval}/<checkpoint>/   the matched pair and the signal's evaluation jets
    <root>/<model>/qcd/eval/<checkpoint>/                 the QCD evaluation jets
Per signal:
    zero_shot_xbb, zero_shot_xbq   Eq. 4 from the model's own outputs: the node holding X->bb,
                                   and Sophon's control for X->bs, the node holding X->bq,
                                   each against the QCD nodes (no training)
    linear                         the paper-wide linear probe (experiments/EVAL/linear_probe_v2.py
                                   fit: standardised, L2 chosen on the validation pair)
    head                           Sophon's head, its first layer the model's own, one per seed
                                   (experiments/XBS/xbx.py train_head)
each with experiments/XBS/xbx.py endpoints. A model without an output layer (the untrained
network) has the linear probe only.
Output: <out>/<model>.json, written last and atomically.
The from-scratch baseline (scripts/build_xbx_jobs.py) is scored the same way from its own
outputs (extract_features.py --save-logits on the two evaluation sets): `scratch`.

    python3 experiments/XBS/fit_xbx.py model --root /data/results/eval/xbx --model mtx-l188-s1 \\
        --signals xbb_jc2 --checkpoint best70 --out /data/results/eval/xbx/fits
    python3 experiments/XBS/fit_xbx.py scratch --sig <cell>/features_sig --qcd <cell>/features_qcd \\
        --out <cell>/endpoints.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
WINDOW = (150.0, 230.0)       # Sophon's (Fig. 2) and A5's
SEEDS = (1, 2, 3)             # A5: three head seeds per model


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


X = _load("xbx", "experiments/XBS/xbx.py")
LP = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")


def load(d: pathlib.Path) -> dict:
    """Features, classes, m_SD and (if the model has an output layer) head columns of one split,
    every jet of the staged set in its stored order (LP.load_split's checks)."""
    x, y, man = LP.load_split(d)
    out = {"x": x, "y": y, "msd": np.load(d / "observers.npz")["jet_sdmass"].astype(np.float64), "manifest": man}
    if man.get("has_head"):
        h = np.load(d / "head_scores.npz")
        if not np.array_equal(h["rows"], np.arange(y.size)):
            raise SystemExit(f"FATAL: {d}: the head scores are not every jet's, in order")
        out["head"] = {k: h[k] for k in ("logp_xbb", "logp_xbq", "logp_qcd")}
    return out


def first_layer(man: dict) -> dict:
    """The model's own first output layer, from the checkpoint the extraction recorded."""
    import torch
    path = pathlib.Path(man["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != man["checkpoint_sha256"]:
        raise SystemExit(f"FATAL: {path} is not the checkpoint the features came from")
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: sd[k] for k in (f"{X.FC0}.weight", f"{X.FC0}.bias")}


def fit_signal(model_dir: pathlib.Path, sig: str, ckpt: str, window, seeds) -> dict:
    tr, va, ev, qcd = (load(model_dir / s / part / ckpt) for s, part in
                       ((sig, "train"), (sig, "val"), (sig, "eval"), ("qcd", "eval")))
    for name, d, want in (("train", tr, {0, 1}), ("val", va, {0, 1}), ("eval", ev, {1}), ("qcd", qcd, {0})):
        if set(np.unique(d["y"]).tolist()) != want:
            raise SystemExit(f"FATAL: {sig} {name}: classes {np.unique(d['y'])}, expected {sorted(want)}")
    if (tr["y"] == 1).sum() != (tr["y"] == 0).sum():
        raise SystemExit(f"FATAL: {sig}: the training pair is not balanced")
    ws, wb = X.in_window(ev["msd"], *window), X.in_window(qcd["msd"], *window)
    score = lambda s, b: X.endpoints(s[ws], b[wb])
    out = {"n_train_per_class": int((tr["y"] == 1).sum()), "n_val": int(va["y"].size),
           "repetition": X.HEAD_SAMPLES / tr["y"].size, "window": list(window)}
    sc, clf, rec = LP.fit(tr["x"], tr["y"], va["x"], va["y"], 2)
    p = lambda d: clf.predict_proba(sc.transform(d["x"]))[:, 1]
    out["linear"] = {**score(p(ev), p(qcd)), "C": rec["C"], "converged": rec["converged"], "search": rec["search"]}
    if "head" in ev:
        for key, col in (("zero_shot_xbb", "logp_xbb"), ("zero_shot_xbq", "logp_xbq")):
            z = lambda d: X.zero_shot(d["head"][col], d["head"]["logp_qcd"])
            out[key] = score(z(ev), z(qcd))
        fc0 = first_layer(tr["manifest"])
        out["head"] = []
        for s in seeds:
            h = X.train_head(tr["x"], tr["y"], fc0, seed=s, n_samples=X.HEAD_SAMPLES)
            out["head"].append({"seed": s, **score(X.head_scores(h, ev["x"]), X.head_scores(h, qcd["x"]))})
    return out


def scratch(sig: pathlib.Path, qcd: pathlib.Path, window) -> dict:
    """A two-class model's own signal probability on the two evaluation sets, in the window."""
    def score(d, want):
        y = np.load(d / "label188.npy")
        if set(np.unique(y).tolist()) != {want}:
            raise SystemExit(f"FATAL: {d}: classes {np.unique(y)}, expected {want}")
        z = np.load(d / "logits.npy").astype(np.float64)
        p = X.softmax(z)[:, 1]
        return p[X.in_window(np.load(d / "observers.npz")["jet_sdmass"].astype(np.float64), *window)]
    return {"window": list(window), **X.endpoints(score(sig, 1), score(qcd, 0))}


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--root", required=True, type=pathlib.Path)
    m.add_argument("--model", required=True)
    m.add_argument("--signals", nargs="+", required=True)
    m.add_argument("--checkpoint", required=True, help="best70, or init for an untrained network")
    m.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    m.add_argument("--out", required=True, type=pathlib.Path)
    s = sub.add_parser("scratch")
    s.add_argument("--sig", required=True, type=pathlib.Path)
    s.add_argument("--qcd", required=True, type=pathlib.Path)
    s.add_argument("--out", required=True, type=pathlib.Path)
    for p in (m, s):
        p.add_argument("--window", nargs=2, type=float, default=WINDOW)
    a = ap.parse_args(argv)
    if a.cmd == "scratch":
        write_json(a.out, scratch(a.sig, a.qcd, a.window))
        return 0
    res = {"model": a.model, "checkpoint": a.checkpoint, "window": list(a.window), "seeds": a.seeds,
           "device": LP.device(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "signals": {s: fit_signal(a.root / a.model, s, a.checkpoint, a.window, a.seeds) for s in a.signals}}
    write_json(a.out / f"{a.model}.json", res)
    print(json.dumps({s: {k: v.get("neg_ln_eps_s") if isinstance(v, dict) else [h["neg_ln_eps_s"] for h in v]
                          for k, v in r.items() if k in ("zero_shot_xbb", "zero_shot_xbq", "linear", "head")}
                      for s, r in res["signals"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
