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

tab:tests (PI 2026-11-23, LEDGER SP-12 step 4; SC-1 (1)), on the same resamples:
  --native-groups   class probabilities of JetClass-II: macro_resonant and macro_qcd, the mean one-vs-rest AUC over
               the resonant and over the QCD classes present (linear_probe_v2.native_metrics' groups, QCD_NATIVE)
  --headline-only   the headline metrics alone (no AUC), for a table that prints accuracy only
  --step NAME=A1,A2/B1,B2/C1,C2   the step from B to A against the step from C to B, (mean A - mean B) -
               (mean B - mean C), per metric: boot_sd over the resamples as for --diff; the run term is
               SD_A^2 + 4 SD_B^2 + SD_C^2 (each mean's run SD, weighted by its coefficient squared), recorded as
               sd_a^2 = SD_A^2 + 2 SD_B^2 (the first step) and sd_b^2 = 2 SD_B^2 + SD_C^2 (the second), so sigma is
               still sqrt(sd_a^2 + sd_b^2 + boot_sd^2)
  --second-task TASK   every model's cell also scored on TASK's test set (other jets, their own resamples from
               seed + 1, shared by every model): <metric>_second, the headline metric there, and <metric>_ratio, it
               over that on --task, per model and per resample (e.g. R30 on Herwig over R30 on Pythia); a --diff side
               may name <model>@second, the model's headline metrics on TASK's jets (Herwig minus Pythia jets); a
               difference of a side on TASK's jets and one on --task's is unpaired (other jets, resampled
               independently): boot_sd^2 = boot_sd_a^2 + boot_sd_b^2, each side's mean's bootstrap variance
  --cache DIR  each model's metrics and resampled metrics in DIR/<model>.npz, read instead of recomputed when its
               record (input, task, jets, b, seed, options) is the same, refused when it differs; with --cache-only
               the run fills the cache and writes no --out (a large set split over pods, then one run reads them all)

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
HEADLINE = ("accuracy", "r50", "r30")   # not ratios of 1 - AUC: kept out of the pairs
EPS_S_R50 = 0.5


@functools.cache
def _ft_curves():
    spec = importlib.util.spec_from_file_location("ft_curves", REPO / "experiments/FT/ft_curves.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@functools.cache
def _qcd_native() -> frozenset:
    """linear_probe_v2.QCD_NATIVE, the QCD classes native_metrics' macro_auc_qcd averages over."""
    spec = importlib.util.spec_from_file_location("linear_probe_v2", REPO / "experiments/EVAL/linear_probe_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return frozenset(m.QCD_NATIVE)


def is_auc(k: str) -> bool:
    """A metric stored as 1 - AUC (the AUC, the macro AUC and its groups, a class's AUC); every other is stored as is."""
    return k in ("auc", "macro") or k.startswith(("macro_", "class_"))


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
    """{metric: (value, resampled values)}: the argmax share for probabilities; for a two-class score every metric the
    tables print (PI 2026-11-21): accuracy (the argmax of the two outputs, signal when s > 0.5), R50 and R30."""
    if s.ndim == 1:
        out = {}
        for key, eps in (("r50", EPS_S_R50), ("r30", EPS_S_R30)):
            sc = RejScorer(y == 1, s, eps)
            out[key] = (sc(), np.array([sc(c[k]) for k in range(c.shape[0])]))
        ok = ((s > 0.5) == (y == 1)).astype(np.float64)
        w = c.astype(np.float64)
        out["accuracy"] = (float(ok.mean()), (w @ ok) / w.sum(1))
        return out
    ok = (s.argmax(1) == y).astype(np.float64)
    w = c.astype(np.float64)
    return {"accuracy": (float(ok.mean()), (w @ ok) / w.sum(1))}


def metrics(y: np.ndarray, s: np.ndarray, c: np.ndarray, per_class: bool, head: bool = False,
            groups: bool = False, head_only: bool = False) -> dict:
    """{key: (value, resampled values)}: auc for a score, macro (and class_<k>) for probabilities; with head, the
    headline metric beside them; groups: macro_resonant and macro_qcd (native_metrics' groups); head_only: the
    headline metrics alone."""
    if head_only:
        return headline(y, s, c)
    if s.ndim == 1:
        out = {"auc": one_minus_auc(s, y == 1, c)}
    else:
        per = {int(k): one_minus_auc(s[:, k], y == k, c) for k in np.unique(y)}
        mean = lambda ks: (float(np.mean([per[k][0] for k in ks])), np.mean([per[k][1] for k in ks], axis=0))  # noqa: E731
        out = {"macro": mean(list(per))}
        if groups:
            qcd = _qcd_native()
            out["macro_resonant"] = mean([k for k in per if k not in qcd])
            out["macro_qcd"] = mean([k for k in per if k in qcd])
        if per_class:
            out.update({f"class_{k}": v for k, v in per.items()})
    if head:
        out.update(headline(y, s, c))
    return out


def _verdict(k: str, delta: float, sd_a, sd_b, rep: np.ndarray, boot_sd=False) -> dict:
    boot_sd = _sd(rep) if boot_sd is False else boot_sd
    sigma = float(np.sqrt(sd_a ** 2 + sd_b ** 2 + boot_sd ** 2)) if None not in (sd_a, sd_b, boot_sd) else None
    return {"quantity": "one_minus_auc" if is_auc(k) else k, "delta": delta, "sd_a": sd_a, "sd_b": sd_b,
            "boot_sd": boot_sd, "sigma": sigma, "z": delta / sigma if sigma else None,
            "pass": bool(sigma is not None and abs(delta) > 3 * sigma)}


def _runs(res: dict, side: list, k: str) -> tuple:
    """(run values, resampled mean over the side's models, run SD or None for one model) of metric k."""
    v = np.array([res[m][k][0] for m in side])
    return v, np.mean([res[m][k][1] for m in side], axis=0), (float(v.std(ddof=1)) if v.size > 1 else None)


def group_diff(res: dict, a: list, b: list, unpaired: bool = False) -> dict:
    """Decision L3 for the mean over models a minus the mean over models b, per metric both sides have. unpaired: the
    sides are scored on different jets, resampled independently; boot_sd^2 is the sum of each side's mean's bootstrap
    variance (boot_sd_a^2 + boot_sd_b^2), not the variance of the difference over resamples."""
    out = {}
    for k in [k for k in res[a[0]] if all(k in res[m] for m in a + b)]:
        (va, ra, sd_a), (vb, rb, sd_b) = _runs(res, a, k), _runs(res, b, k)
        ba, bb = _sd(ra), _sd(rb)
        boot = (float(np.hypot(ba, bb)) if None not in (ba, bb) else None) if unpaired else False
        out[k] = _verdict(k, float(va.mean() - vb.mean()), sd_a, sd_b, ra - rb, boot)
        out[k].update({"n_a": len(a), "n_b": len(b)})
        if unpaired:
            out[k].update({"resampling": "unpaired", "boot_sd_a": ba, "boot_sd_b": bb})
    return out


def step_diff(res: dict, a: list, b: list, c: list) -> dict:
    """(mean A - mean B) - (mean B - mean C) per metric (--step): the run term SD_A^2 + 4 SD_B^2 + SD_C^2, recorded
    as sd_a^2 = SD_A^2 + 2 SD_B^2 and sd_b^2 = 2 SD_B^2 + SD_C^2; None where a side has one model."""
    out = {}
    for k in [k for k in res[a[0]] if all(k in res[m] for m in a + b + c)]:
        (va, ra, sa), (vb, rb, sb), (vc, rc, sc) = (_runs(res, side, k) for side in (a, b, c))
        sd_a, sd_b = ((float(np.sqrt(sa ** 2 + 2 * sb ** 2)), float(np.sqrt(2 * sb ** 2 + sc ** 2)))
                      if None not in (sa, sb, sc) else (None, None))
        out[k] = _verdict(k, float(va.mean() - 2 * vb.mean() + vc.mean()), sd_a, sd_b, ra - 2 * rb + rc)
        out[k].update({"sd_runs": {"a": sa, "b": sb, "c": sc}, "n_a": len(a), "n_b": len(b), "n_c": len(c)})
    return out


def _sd(x: np.ndarray) -> float | None:
    return float(np.std(x, ddof=1)) if np.all(np.isfinite(x)) else None


def _jets(rows: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(rows.tobytes() + y.tobytes()).hexdigest()


def cache_read(f: pathlib.Path, meta: dict) -> dict | None:
    """A model's cached {metric: (value, resamples)}, None if absent; refused if it was made otherwise (meta)."""
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    got = json.loads(str(z["meta"]))
    if got != meta:
        raise SystemExit(f"FATAL: {f} was made from {got}, not {meta}")
    return {k[2:]: (float(z[k]), z["r:" + k[2:]]) for k in z.files if k.startswith("v:")}


def cache_write(f: pathlib.Path, meta: dict, m: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta, sort_keys=True)), **{f"v:{k}": np.array(v) for k, (v, _) in m.items()},
             **{f"r:{k}": r for k, (_, r) in m.items()})
    os.replace(tmp, f)


def _sides(spec: str, n: int, flag: str) -> list[list[str]]:
    name, sides = spec.split("=", 1)
    out = [side.split(",") for side in sides.split("/")]
    if len(out) != n or not all(all(x) for x in out):
        raise SystemExit(f"FATAL: {flag} {spec}: NAME=" + "/".join(["A1,A2"] * n) + " expected")
    return [name] + out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--task", default="npz", choices=TASKS, help="how a fine-tuning cell is read")
    ap.add_argument("--pair", action="append", default=[], metavar="A/B", help="ratio (1-AUC_A)/(1-AUC_B)")
    ap.add_argument("--take-rows-of", metavar="NAME", help="reduce every model to this model's rows")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--headline", action="store_true", help="accuracy (probabilities) or R30 (a score) beside the AUC")
    ap.add_argument("--headline-only", action="store_true", help="the headline metrics alone, no AUC")
    ap.add_argument("--native-groups", action="store_true", help="JetClass-II's macro AUC over its resonant, QCD classes")
    ap.add_argument("--diff", action="append", default=[], metavar="NAME=A1,A2/B1,B2",
                    help="mean over models A minus mean over models B, per metric (decision L3)")
    ap.add_argument("--step", action="append", default=[], metavar="NAME=A1,A2/B1,B2/C1,C2",
                    help="(mean A - mean B) - (mean B - mean C), per metric")
    ap.add_argument("--second-task", choices=TASKS, help="each headline metric's ratio, this task's over --task's")
    ap.add_argument("--cache", type=pathlib.Path, help="per-model resampled metrics, read or written")
    ap.add_argument("--cache-only", action="store_true", help="fill --cache; no --out")
    ap.add_argument("--b", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    if (a.out is None) != a.cache_only or (a.cache_only and a.cache is None):
        raise SystemExit("FATAL: --out, or --cache-only with --cache")
    if a.out is not None and a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists; use a new output")
    if a.second_task and not (a.headline or a.headline_only):
        raise SystemExit("FATAL: --second-task takes the ratio of the headline metrics: add --headline")
    paths = dict(m.split("=", 1) for m in a.model)
    if len(paths) != len(a.model):
        raise SystemExit("FATAL: a model name is given twice")
    pairs = list(dict.fromkeys(tuple(p.split("/", 1)) for p in a.pair))
    diffs = {d[0]: d[1:] for d in (_sides(x, 2, "--diff") for x in a.diff)}
    steps = {d[0]: d[1:] for d in (_sides(x, 3, "--step") for x in a.step)}
    if len(diffs) != len(a.diff) or len(steps) != len(a.step) or set(diffs) & set(steps):
        raise SystemExit("FATAL: a --diff or --step name is given twice")
    named = set(paths) | ({f"{n}@second" for n in paths} if a.second_task else set())
    unknown = ({n for p in pairs for n in p} | {n for ab in [*diffs.values(), *steps.values()] for side in ab
                                                for n in side}) - named | ({a.take_rows_of} - set(paths) - {None})
    if unknown:
        raise SystemExit(f"FATAL: {sorted(unknown)} are not models")
    # one model in memory at a time: the reference model's jets fix the resamples
    ref = a.take_rows_of or next(iter(paths))
    first = load(pathlib.Path(paths[ref]), a.task)
    rows, y = first[0], first[1]
    second = load(pathlib.Path(paths[ref]), a.second_task) if a.second_task else None
    opts = {"task": a.task, "second_task": a.second_task, "b": a.b, "seed": a.seed, "take_rows_of": a.take_rows_of,
            "per_class": a.per_class, "headline": a.headline, "headline_only": a.headline_only,
            "native_groups": a.native_groups, "jets_sha256": _jets(rows, y),
            "jets2_sha256": _jets(*second[:2]) if second else None}
    c = c2 = None
    res = {}
    for n in paths:
        meta = {**opts, "input": paths[n]}
        got = cache_read(a.cache / f"{n}.npz", meta) if a.cache else None
        if got is None:
            c = counts(y, a.b, a.seed) if c is None else c
            s = align(n, first if n == ref else load(pathlib.Path(paths[n]), a.task), ref, rows, y,
                      bool(a.take_rows_of))
            got = metrics(y, s, c, a.per_class, a.headline, a.native_groups, a.headline_only)
            if second:
                # other jets: their own resamples (seed + 1), shared by every model
                c2 = counts(second[1], a.b, a.seed + 1) if c2 is None else c2
                s2 = align(n, second if n == ref else load(pathlib.Path(paths[n]), a.second_task), ref, *second[:2],
                           bool(a.take_rows_of))
                h2 = headline(second[1], s2, c2)
                got.update({f"{k}_second": v for k, v in h2.items()})
                got.update({f"{k}_ratio": (v / got[k][0], r / got[k][1]) for k, (v, r) in h2.items() if k in got})
            if a.cache:
                cache_write(a.cache / f"{n}.npz", meta, got)
        res[n] = got
        print(f"  {n:20} " + " ".join(f"{k}: {'1-AUC=' if is_auc(k) else ''}{v:.6g}" for k, (v, _) in res[n].items()
                                      if not k.startswith("class_")), flush=True)
    if a.cache_only:
        print(f"cached {len(res)} models in {a.cache}")
        return 0
    doc = {"script": "experiments/STATS/paired_bootstrap.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "task": a.task, "b": a.b, "seed": a.seed, "resampling": "test jets, with replacement within each class",
           "n_jets": int(y.size), "class_counts": {int(k): int((y == k).sum()) for k in np.unique(y)},
           "jets_sha256": opts["jets_sha256"],
           "models": {n: {"input": paths[n], **{k: {"one_minus_auc" if is_auc(k) else "value": v, "sd": _sd(r)}
                                                for k, (v, r) in m.items()}}
                      for n, m in res.items()},
           "pairs": {}}
    if second:
        doc.update({"second_task": a.second_task, "second_seed": a.seed + 1, "second_n_jets": int(second[1].size),
                    "second_jets_sha256": opts["jets2_sha256"]})
    if diffs or steps:
        doc["rule_l3"] = ("delta = mean over A - mean over B; pass = |delta| > 3 sigma, sigma^2 = sd_a^2 + sd_b^2 + "
                          "boot_sd^2 (decision L3, 2026-11-13)")
        # with --second-task, <model>@second is the model's headline metrics on the second task's jets
        view = {**res, **{f"{n}@second": {k.removesuffix("_second"): v for k, v in m.items() if k.endswith("_second")}
                          for n, m in res.items()}}
        # a side on the second task's jets against one on the first's: no paired resampling exists (unpaired)
        on2 = lambda side: any(n.endswith("@second") for n in side)  # noqa: E731
        doc["diffs"] = {name: {"a": ab[0], "b": ab[1], **group_diff(view, *ab, unpaired=on2(ab[0]) != on2(ab[1]))}
                        for name, ab in diffs.items()}
    if steps:
        doc["rule_step"] = ("delta = (mean A - mean B) - (mean B - mean C); sigma^2 = SD_A^2 + 4 SD_B^2 + SD_C^2 + "
                            "boot_sd^2, written as sd_a^2 = SD_A^2 + 2 SD_B^2 and sd_b^2 = 2 SD_B^2 + SD_C^2")
        doc["diffs"].update({name: {"a": abc[0], "b": abc[1], "c": abc[2], "step": True, **step_diff(res, *abc)}
                             for name, abc in steps.items()})
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in pairs:
            doc["pairs"][f"{p}/{q}"] = {k: {"ratio": v / res[q][k][0] if res[q][k][0] > 0 else None,
                                            "sd": _sd(r / res[q][k][1]),
                                            "ln_ratio_sd": _sd(np.log(r / res[q][k][1]))}
                                        for k, (v, r) in res[p].items() if is_auc(k)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
