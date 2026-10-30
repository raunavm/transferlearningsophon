#!/usr/bin/env python3
"""X->bb and X->bs per model (docs/PRESPEC_2026-09.md A5): every readout of one frozen model,
scored on the evaluation jets inside the m_SD window.

Input, from experiments/XBS/extract_xbx.py on the staged sets (experiments/XBS/stage_xbx.py):
    <root>/<model>/<sig>/{train,val,eval}/<checkpoint>/   the matched pair and the signal's evaluation jets
    <root>/<model>/qcd/eval/<checkpoint>/                 the QCD evaluation jets
Per signal:
    zero_shot_xbb                  Eq. 4 from the model's own outputs, the X->bb class against
                                   the QCD nodes (no training), where X->bb is a class of its own
                                   (188 and 162 classes); elsewhere null with zero_shot_xbb_reason
                                   "cannot be formed", and the merged node that holds X->bb is
                                   scored as group_node_xbb (a different discriminant, not Eq. 4)
    zero_shot_xbq                  Sophon's control for X->bs: the node holding X->bq against the
                                   QCD nodes (A5: bq at 188 and 162 classes, the merged group below,
                                   which at most label sets is X->bb's node, so the same number as
                                   group_node_xbb; nodes records both)
    linear                         the paper-wide linear probe (experiments/EVAL/linear_probe_v2.py
                                   fit: standardised, L2 chosen on the validation pair)
    head                           Sophon's head, its first layer the model's own, one per seed
                                   (experiments/XBS/xbx.py train_head)
each with experiments/XBS/xbx.py endpoints. A model without an output layer (the untrained
network) has the linear probe only. Recorded beside them: the window and mX (both given
explicitly and refused unless the window is Sophon's 150-230 GeV scaled by mX/200,
xbx.window_for), A5's eps_B* and the jets
in the window, and the head's training length (N per class, its 2.5 M samples, the repetition
factor 2.5 M / 2N).
Output: <out>/<model>.json, written last and atomically; an existing one is never overwritten.
The from-scratch baseline (scripts/build_xbx_jobs.py) is scored the same way from its own
outputs (extract_features.py --save-logits on the two evaluation sets): `scratch`, with the
examples it saw from its fine-tuning manifest.

    python3 experiments/XBS/fit_xbx.py model --root /data/results/eval/xbx --model mtx-l188-s1 \\
        --signals xbs --checkpoint best70 --window 93.75 143.75 --mx 125 \\
        --out /data/results/eval/xbx/fits_ihep/xbs
    python3 experiments/XBS/fit_xbx.py scratch --sig <cell>/features_sig --qcd <cell>/features_qcd \\
        --window 93.75 143.75 --mx 125 --ft-manifest <cell>/ft_manifest.json --out <cell>/endpoints.json
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
           "head_samples": X.HEAD_SAMPLES, "repetition": X.HEAD_SAMPLES / tr["y"].size,
           "window": list(window), "n_sig_in_window": int(ws.sum()), "n_qcd_in_window": int(wb.sum()),
           "eps_b_star": 1e-3 if wb.sum() >= X.A5_QCD_FOR_1E3 else 1e-2}
    sc, clf, rec = LP.fit(tr["x"], tr["y"], va["x"], va["y"], 2)
    p = lambda d: clf.predict_proba(sc.transform(d["x"]))[:, 1]
    out["linear"] = {**score(p(ev), p(qcd)), "C": rec["C"], "converged": rec["converged"], "search": rec["search"]}
    if "head" in ev:
        z = lambda d, col: X.zero_shot(d["head"][col], d["head"]["logp_qcd"])
        g = X.node_groups(json.loads((model_dir / sig / "eval" / "xbx.json").read_text())["arm"])
        out["nodes"] = {k: g[k] for k in ("xbb", "xbq", "xbb_own")}
        if g["xbb_own"]:
            out["zero_shot_xbb"] = score(z(ev, "logp_xbb"), z(qcd, "logp_xbb"))
        else:
            out["zero_shot_xbb"] = None
            out["zero_shot_xbb_reason"] = "cannot be formed: X->bb is not a class of its own in this label set"
            out["group_node_xbb"] = score(z(ev, "logp_xbb"), z(qcd, "logp_xbb"))
        out["zero_shot_xbq"] = score(z(ev, "logp_xbq"), z(qcd, "logp_xbq"))
        fc0 = first_layer(tr["manifest"])
        out["head"] = []
        for s in seeds:
            h = X.train_head(tr["x"], tr["y"], fc0, seed=s, n_samples=X.HEAD_SAMPLES)
            out["head"].append({"seed": s, **score(X.head_scores(h, ev["x"]), X.head_scores(h, qcd["x"]))})
    return out


def scratch(sig: pathlib.Path, qcd: pathlib.Path, window, ft_manifest: pathlib.Path | None = None) -> dict:
    """A two-class model's own signal probability on the two evaluation sets, in the window, and
    the examples its training saw (epochs x samples per epoch, from the fine-tuning manifest)."""
    def score(d, want):
        y = np.load(d / "label188.npy")
        if set(np.unique(y).tolist()) != {want}:
            raise SystemExit(f"FATAL: {d}: classes {np.unique(y)}, expected {want}")
        z = np.load(d / "logits.npy").astype(np.float64)
        p = X.softmax(z)[:, 1]
        return p[X.in_window(np.load(d / "observers.npz")["jet_sdmass"].astype(np.float64), *window)]
    out = {"window": list(window), **X.endpoints(score(sig, 1), score(qcd, 0))}
    if ft_manifest is not None:
        m = json.loads(ft_manifest.read_text())
        n, spe, ep = int(m["n_train"]), int(m["samples_per_epoch"]), int(m["epochs"])
        out["training"] = {"n_train_per_class": n // 2, "samples_per_epoch": spe, "epochs": ep,
                           "examples_seen": spe * ep, "passes": spe * ep / n}
    return out


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
    s.add_argument("--ft-manifest", type=pathlib.Path, default=None)
    for p in (m, s):
        p.add_argument("--window", nargs=2, type=float, required=True, metavar=("LO", "HI"))
        p.add_argument("--mx", type=float, required=True, help="the signal's resonance mass, GeV")
    a = ap.parse_args(argv)
    if not np.allclose(a.window, X.window_for(a.mx)):
        raise SystemExit(f"FATAL: window {a.window} is not {X.window_for(a.mx)}, Sophon's scaled to mX = {a.mx:g}")
    target = a.out if a.cmd == "scratch" else a.out / f"{a.model}.json"
    if target.exists():
        raise SystemExit(f"FATAL: {target} exists; a fit is never overwritten")
    if a.cmd == "scratch":
        write_json(target, {"mx": a.mx, **scratch(a.sig, a.qcd, a.window, a.ft_manifest)})
        return 0
    res = {"model": a.model, "checkpoint": a.checkpoint, "window": list(a.window), "mx": a.mx, "seeds": a.seeds,
           "device": LP.device(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "signals": {s: fit_signal(a.root / a.model, s, a.checkpoint, a.window, a.seeds) for s in a.signals}}
    write_json(target, res)
    print(json.dumps({s: {k: v.get("neg_ln_eps_s") if isinstance(v, dict) else [h["neg_ln_eps_s"] for h in v]
                          for k, v in r.items() if k in ("zero_shot_xbb", "group_node_xbb", "zero_shot_xbq",
                                                          "linear", "head") and v is not None}
                      for s, r in res["signals"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
