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

Round 5 (decision L3, PI 2026-11-13: |delta| > 3 sigma, sigma^2 = the SD^2 of each side over its runs + the
bootstrap SD^2 over test jets):
  --headline   adds the tables' headline metric, scored on the same resamples: accuracy (the argmax share) for
               class probabilities, R30 for a binary score (1 / eps_B at 30% signal efficiency, eps_B interpolated
               on the ROC as ft_curves.roc_points, capped at the background count); not used in the ratios
  --diff NAME=A1,A2/B1,B2,B3   the difference of the mean of each metric over models A minus over models B (1 - AUC
               for the AUC metrics): delta, sd_a and sd_b (each side's SD over its models, ddof 1), boot_sd (the SD
               over the resamples of that resample's difference of the two means), sigma = sqrt(sd_a^2 + sd_b^2 +
               boot_sd^2), z and pass = |delta| > 3 sigma (null sigma, no pass, for a side of one model)
  --task jc1_all   a JetClass cell on every one of its 2,000,000 test jets, the jets of the read-out's macro AUC
               (budget_metrics.py, the figures' JetClass AUC per decision L2)

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
TASKS = ("jc1", "jc1_all", "top", "qg", "qg_herwig", "jc2_188", "jc2b", "npz")
EPS_S_R30 = 0.3
HEADLINE = ("accuracy", "r30")          # not ratios of 1 - AUC: kept out of the pairs


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
    ft, base = _ft_curves(), {"qg_herwig": "qg", "jc1_all": "jc1"}.get(task, task)
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


class RejScorer:
    """1 / eps_B at signal efficiency eps_s on the count-weighted ROC (truth: signal), sorting once: the ROC of
    ft_curves.roc_points (sklearn's roc_curve, one point per distinct score, eps_B interpolated in eps_S) with
    every point kept, capped at the (weighted) background count."""

    def __init__(self, truth: np.ndarray, score: np.ndarray, eps_s: float):
        self.order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
        ss = np.asarray(score, dtype=np.float64)[self.order]
        self.pos = np.asarray(truth, dtype=bool)[self.order].astype(np.float64)
        self.gid = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]).astype(np.int64)
        self.g, self.eps_s = int(self.gid[-1]) + 1, eps_s

    def __call__(self, w=None) -> float:
        ws = np.ones(self.pos.size) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        tp = np.cumsum(np.bincount(self.gid, weights=ws * self.pos, minlength=self.g))
        fp = np.cumsum(np.bincount(self.gid, weights=ws * (1.0 - self.pos), minlength=self.g))
        eps_b = float(np.interp(self.eps_s, np.r_[0.0, tp / tp[-1]], np.r_[0.0, fp / fp[-1]]))
        return min(1.0 / eps_b, float(fp[-1])) if eps_b > 0 else float(fp[-1])


def headline(y: np.ndarray, s: np.ndarray, c: np.ndarray) -> dict:
    """{"accuracy" | "r30": (value, resampled values)}: the argmax share for probabilities, R30 for a score."""
    if s.ndim == 1:
        sc = RejScorer(y == 1, s, EPS_S_R30)
        return {"r30": (sc(), np.array([sc(c[k]) for k in range(c.shape[0])]))}
    ok = (s.argmax(1) == y).astype(np.float64)
    w = c.astype(np.float64)
    return {"accuracy": (float(ok.mean()), (w @ ok) / w.sum(1))}


def metrics(y: np.ndarray, s: np.ndarray, c: np.ndarray, per_class: bool, head: bool = False) -> dict:
    """{key: (value, resampled values)}: auc for a score, macro (and class_<k>) for probabilities; with head, the
    headline metric beside them."""
    if s.ndim == 1:
        out = {"auc": one_minus_auc(s, y == 1, c)}
    else:
        per = {int(k): one_minus_auc(s[:, k], y == k, c) for k in np.unique(y)}
        out = {"macro": (float(np.mean([v for v, _ in per.values()])), np.mean([r for _, r in per.values()], axis=0))}
        if per_class:
            out.update({f"class_{k}": v for k, v in per.items()})
    if head:
        out.update(headline(y, s, c))
    return out


def group_diff(res: dict, a: list, b: list) -> dict:
    """Decision L3 for the mean over models a minus the mean over models b, per metric both sides have."""
    out = {}
    for k in [k for k in res[a[0]] if all(k in res[m] for m in a + b)]:
        va, vb = (np.array([res[m][k][0] for m in side]) for side in (a, b))
        rep = np.mean([res[m][k][1] for m in a], axis=0) - np.mean([res[m][k][1] for m in b], axis=0)
        sd_a, sd_b = (float(v.std(ddof=1)) if v.size > 1 else None for v in (va, vb))
        boot_sd = _sd(rep)
        sigma = (float(np.sqrt(sd_a ** 2 + sd_b ** 2 + boot_sd ** 2))
                 if None not in (sd_a, sd_b, boot_sd) else None)
        delta = float(va.mean() - vb.mean())
        out[k] = {"quantity": k if k in HEADLINE else "one_minus_auc", "delta": delta, "n_a": len(a), "n_b": len(b),
                  "sd_a": sd_a, "sd_b": sd_b, "boot_sd": boot_sd, "sigma": sigma, "z": delta / sigma if sigma else None,
                  "pass": bool(sigma is not None and abs(delta) > 3 * sigma)}
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
    ap.add_argument("--headline", action="store_true", help="accuracy (probabilities) or R30 (a score) beside the AUC")
    ap.add_argument("--diff", action="append", default=[], metavar="NAME=A1,A2/B1,B2",
                    help="mean over models A minus mean over models B, per metric (decision L3)")
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
    diffs = {}
    for d in a.diff:
        name, sides = d.split("=", 1)
        diffs[name] = [side.split(",") for side in sides.split("/", 1)]
        if len(diffs[name]) != 2 or not all(diffs[name]):
            raise SystemExit(f"FATAL: --diff {d}: NAME=A1,A2/B1,B2 expected")
    if len(diffs) != len(a.diff):
        raise SystemExit("FATAL: a --diff name is given twice")
    unknown = ({n for p in pairs for n in p} | {n for ab in diffs.values() for side in ab for n in side}) - set(paths) \
        | ({a.take_rows_of} - set(paths) - {None})
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
        res[n] = metrics(y, s, c, a.per_class, a.headline)
        print(f"  {n:20} " + " ".join(f"{k}: {'' if k in HEADLINE else '1-AUC='}{v:.6g}" for k, (v, _) in res[n].items()
                                      if not k.startswith("class_")), flush=True)
    doc = {"script": "experiments/STATS/paired_bootstrap.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "task": a.task, "b": a.b, "seed": a.seed, "resampling": "test jets, with replacement within each class",
           "n_jets": int(y.size), "class_counts": {int(k): int((y == k).sum()) for k in np.unique(y)},
           "jets_sha256": hashlib.sha256(rows.tobytes() + y.tobytes()).hexdigest(),
           "models": {n: {"input": paths[n], **{k: {"value" if k in HEADLINE else "one_minus_auc": v, "sd": _sd(r)}
                                                for k, (v, r) in m.items()}}
                      for n, m in res.items()},
           "pairs": {}}
    if diffs:
        doc["rule_l3"] = ("delta = mean over A - mean over B; pass = |delta| > 3 sigma, sigma^2 = sd_a^2 + sd_b^2 + "
                          "boot_sd^2 (decision L3, 2026-11-13)")
        doc["diffs"] = {name: {"a": ab[0], "b": ab[1], **group_diff(res, *ab)} for name, ab in diffs.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in pairs:
            doc["pairs"][f"{p}/{q}"] = {k: {"ratio": v / res[q][k][0] if res[q][k][0] > 0 else None,
                                            "sd": _sd(r / res[q][k][1]),
                                            "ln_ratio_sd": _sd(np.log(r / res[q][k][1]))}
                                        for k, (v, r) in res[p].items() if k not in HEADLINE}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
