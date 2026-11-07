#!/usr/bin/env python3
"""Rejection of each QCD heavy-flavor subclass by the X->bb discriminant of the balanced JetClass-II
linear probe (2026-11-07, PI: "one small appendix table, from existing probabilities").

The probe is probe_bal.py's 188-class linear probe at the largest balanced training set (fits_v4_probs
<model>/<checkpoint>/jc2bal_N<n>.npz). Its discriminant is Sophon's, D = p(X->bb) / (p(X->bb) + sum of
the 27 QCD class probabilities) (sophon Eq. (4); linear_probe_v2.per_class). Jets are ranked by
logit(D) = z_bb - logsumexp(z_QCD), from the probe's saved weights applied to the stored test features,
because the saved float16 probabilities tie near D = 1 where the working points sit; the recomputed
probabilities are checked against the saved ones.

Signal: the X->bb test jets. Background, each subclass of QCD_SUBCLASSES (the QCD classes by the number
of b and c hadrons their names count) and all QCD, at signal efficiencies of 50 % and 30 %:
  balanced  the background jets as the class-balanced test set holds them (up to 5,000 per class),
            probe.rejection_at (the paper's R50 / R30)
  natural   each jet weighted by its class's selected test count (census.json ft_test) over its count in
            the balanced test set, so a subclass has the natural mix of its classes; rel_stat is Kish's
            1/sqrt(n_eff) of the passing weights

    python3 experiments/EVAL/qcd_hf_rejection.py --model mtx-l188-s1 --checkpoints best70 \\
        --probs-root /data/results/eval/v2_linprobe_bal/fits_v4_probs \\
        --features-root /data/results/eval/v2_linprobe_bal --out /data/results/eval/v2_linprobe_bal/fits_v4_qcdhf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SIGNAL = "label_X_bb"
SUBCLASSES = ("bb", "b", "c", "light")
WORKING_POINTS = {"r50": 0.5, "r30": 0.3}          # linear_probe_v2.WORKING_POINTS
CENSUS = pathlib.Path("/data/finetune/jc2_bal/census.json")
P_TOL = 2e-3                                        # recomputed vs saved float16 probabilities


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def class_names() -> dict:
    with (REPO / "configs/labelmaps/rung_label_maps.v1.csv").open() as f:
        return {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}


def qcd_subclasses(names: dict) -> dict:
    """{subclass: native QCD classes}: bb = two or more b, b = one b, c = no b and one or more c, light =
    neither (s and light quarks, gluons), from the class name label_QCD_<hadron content>."""
    out = {s: [] for s in SUBCLASSES}
    for i, n in sorted(names.items()):
        if not n.startswith("label_QCD_"):
            continue
        f = n.removeprefix("label_QCD_")
        nb, nc = f.count("b"), f.count("c")
        out["bb" if nb >= 2 else "b" if nb == 1 else "c" if nc else "light"].append(i)
    return out


def log_odds(z: np.ndarray, classes: np.ndarray, sig: int, qcd: list) -> np.ndarray:
    """logit(D) = z_sig - logsumexp(z_QCD) from the probe's logits over its classes."""
    col = {int(c): j for j, c in enumerate(classes)}
    missing = [c for c in [sig, *qcd] if c not in col]
    if missing:
        raise SystemExit(f"FATAL: the probe never saw classes {missing}")
    zq = z[:, [col[c] for c in qcd]]
    m = zq.max(1)
    return z[:, col[sig]] - (m + np.log(np.exp(zq - m[:, None]).sum(1)))


def weighted_rejection(y: np.ndarray, s: np.ndarray, w: np.ndarray, eps_s: float) -> dict:
    """1/eps_B at signal efficiency eps_s with background weights w (signal weight 1), interpolated on the
    weighted ROC as probe.rejection_at; is_bound when no background jet passes."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s, sample_weight=np.where(y == 1, 1.0, w))
    eps_b = float(np.interp(eps_s, tpr, fpr))
    cut = thr[np.searchsorted(tpr, eps_s)]
    wp = w[(y == 0) & (s >= cut)]
    rel = float(math.sqrt((wp ** 2).sum()) / wp.sum()) if wp.size else None
    return {"rejection": 1.0 / eps_b if eps_b > 0 else None, "eps_b": eps_b, "is_bound": eps_b <= 0,
            "n_bkg_pass": int(wp.size), "rel_stat": rel}


def rejections(score, y, sig: int, groups: dict, weight: np.ndarray, probe) -> dict:
    """{background: {balanced: {...}, natural: {...}}} at WORKING_POINTS; weight is per native class."""
    out = {}
    s_sig = y == sig
    for name, cls in groups.items():
        m = s_sig | np.isin(y, cls)
        yy, ss = s_sig[m].astype(np.int64), score[m]
        bal, nat = {}, {}
        for k, eps in WORKING_POINTS.items():
            rej, eps_b, bound, n_pass, rel = probe.rejection_at(yy, ss, eps)
            bal[k] = {"rejection": rej, "eps_b": eps_b, "is_bound": bool(bound), "n_bkg_pass": n_pass,
                      "rel_stat": rel if math.isfinite(rel) else None}
            nat[k] = weighted_rejection(yy, ss, weight[y[m]], eps)
        out[name] = {"classes": [int(c) for c in cls], "n_bkg": int((yy == 0).sum()), "n_signal": int(yy.sum()),
                     "balanced": bal, "natural": nat}
    return out


def probe_file(d: pathlib.Path) -> pathlib.Path:
    """The largest balanced training set's probe of a checkpoint directory (jc2bal_N<n>.npz)."""
    ns = [int(m.group(1)) for p in d.glob("jc2bal_N*.npz") if (m := re.fullmatch(r"jc2bal_N(\d+)\.npz", p.name))]
    if not ns:
        raise SystemExit(f"FATAL: {d} holds no jc2bal_N*.npz")
    return d / f"jc2bal_N{max(ns)}.npz"


def cell(probs: pathlib.Path, feats: pathlib.Path, names: dict, natural: np.ndarray, probe) -> dict:
    z = np.load(probs)
    x = np.load(feats / "features.npy").astype(np.float64)
    y = np.load(feats / "label188.npy").astype(np.int64)
    rows = z["rows"]
    if not np.array_equal(rows, np.arange(y.size)) or not np.array_equal(z["y"].astype(np.int64), y):
        raise SystemExit(f"FATAL: {probs} and {feats} hold different test jets")
    logit = ((x - z["scaler_mean"]) / z["scaler_scale"]) @ z["coef"].T + z["intercept"]
    e = np.exp(logit - logit.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    dp = float(np.abs(p - z["p"].astype(np.float64)[:, z["classes"]]).max())
    if dp > P_TOL:
        raise SystemExit(f"FATAL: {probs}: the saved weights reproduce its probabilities only to {dp:.2e}")
    nat = {n: i for i, n in names.items()}
    sub = qcd_subclasses(names)
    groups = {**sub, "qcd_all": sorted(c for v in sub.values() for c in v)}
    counts = np.bincount(y, minlength=len(names)).astype(np.float64)
    weight = np.divide(natural, counts, out=np.zeros_like(counts), where=counts > 0)
    score = log_odds(logit, z["classes"], nat[SIGNAL], groups["qcd_all"])
    return {"probs": str(probs), "probs_sha256": hashlib.sha256(probs.read_bytes()).hexdigest(),
            "features": str(feats), "max_abs_p_diff": dp, "n_test": int(y.size),
            "background": rejections(score, y, nat[SIGNAL], groups, weight, probe)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--probs-root", required=True, type=pathlib.Path)
    ap.add_argument("--features-root", required=True, type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path, default=CENSUS)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    out = a.out / f"{a.model}.json"
    if out.exists():
        print(f"{out} exists")
        return 0
    names = class_names()
    census = json.loads(a.census.read_text())
    natural = np.asarray(census["counts"]["ft_test"], dtype=np.float64)
    probe = _load("probe", "experiments/EVAL/probe.py")
    doc = {"model": a.model, "signal": SIGNAL, "subclasses": {k: [names[c] for c in v] for k, v in qcd_subclasses(names).items()},
           "working_points": WORKING_POINTS, "discriminant": "p(X->bb) / (p(X->bb) + sum of the 27 QCD p)",
           "census": str(a.census), "census_sha256": hashlib.sha256(a.census.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "cells": {}}
    for ck in a.checkpoints:
        doc["cells"][ck] = cell(probe_file(a.probs_root / a.model / ck), a.features_root / a.model / "jc2bal" / "test" / ck,
                                names, natural, probe)
        r = doc["cells"][ck]["background"]
        print(f"  {a.model} {ck}: R30 " + " ".join(f"{k}={v['balanced']['r30']['rejection']:.4g}" for k, v in r.items()), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"{out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
