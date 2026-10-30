#!/usr/bin/env python3
"""Paired bootstrap over test jets for ratios of (1 - AUC) between models (plan 2026-10-29, A4 D1-3:
the test-sample uncertainty, given beside the spread over pretraining runs).

Every model must be scored on the same jets: equal rows and labels, asserted (after --take-rows-of,
which reduces every model to one model's rows, e.g. a fine-tune's 2M JetClass jets to the probe's
every 4th). B resamples of the jets are drawn once -- seeded, with replacement within each class, so
every resample keeps the class counts -- and shared by every model, so a ratio's spread over the
resamples carries the correlation of two models scored on the same jets. A resample is a count per
jet, scored by src/stats/paired.AucScorer (the count-weighted Mann-Whitney statistic over the full
sample's score order, a tie counting one half): nothing is sorted again per resample.

Inputs (--model NAME=PATH):
  <...>.npz    a linear probe's test outputs (linear_probe_v2.py --probs-out): rows, y, and s
               (binary) or p (class probabilities)
  <cell>/      a fine-tuning cell, read as experiments/FT/ft_curves.py reads it for --task (jc1: on
               every JC1_AUC_STRIDE-th jet, the jets of leg2_metrics' macro AUC)
Metrics (per model, 1 - AUC; its SD over the resamples):
  auc          binary tasks: the signal score's AUC
  macro        class probabilities: the mean one-vs-rest AUC over the classes present
               (linear_probe_v2.native_metrics' macro_auc_ovr)
  class_<c>    with --per-class, each class's one-vs-rest AUC
and for each pair A/B the ratio (1 - AUC_A) / (1 - AUC_B), its SD and the SD of its logarithm.

    python3 experiments/STATS/paired_bootstrap.py --task top \\
        --model l188-s1=<tree>/l188-s1/N1200000/s1 --model r16q1-s1=<tree>/r16q1-s1/N1200000/s1 \\
        --pair r16q1-s1/l188-s1 --out /data/results/bootstrap/ft/top_N1200000.json
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.stats import paired as P  # noqa: E402

B_DEFAULT = 200
SEED_DEFAULT = 20261029
TASKS = ("jc1", "top", "qg", "qg_herwig", "jc2_188", "npz")


@functools.cache
def _ft_curves():
    spec = importlib.util.spec_from_file_location("ft_curves", REPO / "experiments/FT/ft_curves.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(path: pathlib.Path, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, labels, scores) of one model: scores (n,) for a binary task, (n, K) otherwise."""
    if path.suffix == ".npz":
        z = np.load(path)
        return (z["rows"].astype(np.int64), z["y"].astype(np.int64),
                z["s"].astype(np.float64) if "s" in z else z["p"].astype(np.float64))
    if task == "npz":
        raise SystemExit(f"FATAL: {path} is not an .npz; name the fine-tuning task with --task")
    ft, base = _ft_curves(), "qg" if task == "qg_herwig" else task
    if base in ("top", "qg"):
        ft.BM.check_label_order(base)
    if ft.BM.diverged(path):
        raise SystemExit(f"FATAL: {path} trained to a NaN loss; not a result")
    rows, y, s, _ = ft.cell_scores(path, base, "herwig" if task == "qg_herwig" else "pythia")
    keep = slice(None, None, ft.JC1_AUC_STRIDE if task == "jc1" else 1)
    return rows[keep].astype(np.int64), y[keep].astype(np.int64), s[keep]


def align(name: str, model: tuple, ref: str, rows0: np.ndarray, y0: np.ndarray, take: bool) -> np.ndarray:
    """The model's scores on the reference model's jets: reduced to its rows when take, and its
    rows and labels asserted equal to the reference's."""
    rows, y, s = model
    if take:
        order = np.argsort(rows, kind="stable")
        i = order[np.minimum(np.searchsorted(rows, rows0, sorter=order), rows.size - 1)]
        if not np.array_equal(rows[i], rows0):
            raise SystemExit(f"FATAL: {name} lacks rows of {ref}")
        rows, y, s = rows[i], y[i], s[i]
    if not (np.array_equal(rows, rows0) and np.array_equal(y, y0)):
        raise SystemExit(f"FATAL: {name} and {ref} are not scored on the same jets (rows and labels)")
    return s


def counts(y: np.ndarray, b: int, seed: int) -> np.ndarray:
    """(b, n) resample counts per jet, drawn with replacement within each class; resample k is
    a function of (seed, k) alone."""
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    out = np.empty((b, y.size), dtype=np.uint8)
    for k in range(b):
        rng, cnt = np.random.default_rng([seed, k]), np.zeros(y.size, dtype=np.int64)
        for idx in strata:
            cnt[idx] = np.bincount(rng.integers(0, idx.size, idx.size), minlength=idx.size)
        if cnt.max() > 255:
            raise SystemExit("FATAL: a jet drawn more than 255 times; counts do not fit uint8")
        out[k] = cnt
    return out


def one_minus_auc(score: np.ndarray, truth: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
    """(1 - AUC on the sample, 1 - AUC on each resample)."""
    sc = P.AucScorer(truth, score)
    return 1.0 - sc.auc(), 1.0 - np.array([sc.auc(c[k]) for k in range(c.shape[0])])


def metrics(y: np.ndarray, s: np.ndarray, c: np.ndarray, per_class: bool) -> dict:
    """{key: (value, resampled values)}: auc for a score, macro (and class_<k>) for probabilities."""
    if s.ndim == 1:
        return {"auc": one_minus_auc(s, y == 1, c)}
    per = {int(k): one_minus_auc(s[:, k], y == k, c) for k in np.unique(y)}
    out = {"macro": (float(np.mean([v for v, _ in per.values()])), np.mean([r for _, r in per.values()], axis=0))}
    if per_class:
        out.update({f"class_{k}": v for k, v in per.items()})
    return out


def _sd(x: np.ndarray) -> float | None:
    return float(np.std(x, ddof=1)) if np.all(np.isfinite(x)) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--task", default="npz", choices=TASKS, help="how a fine-tuning cell is read")
    ap.add_argument("--pair", action="append", default=[], metavar="A/B", help="ratio (1-AUC_A)/(1-AUC_B)")
    ap.add_argument("--take-rows-of", metavar="NAME", help="reduce every model to this model's rows")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--b", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; use a new output")
    paths = dict(m.split("=", 1) for m in a.model)
    if len(paths) != len(a.model):
        raise SystemExit("FATAL: a model name is given twice")
    pairs = list(dict.fromkeys(tuple(p.split("/", 1)) for p in a.pair))
    unknown = {n for p in pairs for n in p} - set(paths) | ({a.take_rows_of} - set(paths) - {None})
    if unknown:
        raise SystemExit(f"FATAL: {sorted(unknown)} are not models")
    # one model in memory at a time: the reference model's jets fix the resamples
    ref = a.take_rows_of or next(iter(paths))
    first = load(pathlib.Path(paths[ref]), a.task)
    rows, y = first[0], first[1]
    c = counts(y, a.b, a.seed)
    res = {}
    for n in paths:
        s = align(n, first if n == ref else load(pathlib.Path(paths[n]), a.task), ref, rows, y,
                  bool(a.take_rows_of))
        res[n] = metrics(y, s, c, a.per_class)
        print(f"  {n:20} " + " ".join(f"{k}: 1-AUC={v:.6g}" for k, (v, _) in res[n].items()
                                      if not k.startswith("class_")), flush=True)
    doc = {"script": "experiments/STATS/paired_bootstrap.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "task": a.task, "b": a.b, "seed": a.seed, "resampling": "test jets, with replacement within each class",
           "n_jets": int(y.size), "class_counts": {int(k): int((y == k).sum()) for k in np.unique(y)},
           "jets_sha256": hashlib.sha256(rows.tobytes() + y.tobytes()).hexdigest(),
           "models": {n: {"input": paths[n], **{k: {"one_minus_auc": v, "sd": _sd(r)} for k, (v, r) in m.items()}}
                      for n, m in res.items()},
           "pairs": {}}
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in pairs:
            doc["pairs"][f"{p}/{q}"] = {k: {"ratio": v / res[q][k][0] if res[q][k][0] > 0 else None,
                                            "sd": _sd(r / res[q][k][1]),
                                            "ln_ratio_sd": _sd(np.log(r / res[q][k][1]))}
                                        for k, (v, r) in res[p].items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
