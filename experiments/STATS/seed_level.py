#!/usr/bin/env python3
"""Seed-level inference on the four-granularity frozen-probe ladder.

Implements docs/PRESPEC_2026-09.md sections 2 and 3 for the frozen probes and
nothing else. Every choice below -- endpoint, level order, direction, contrast
family, equivalence bound, family sizes -- is a module constant fixed by that
document. The only run-time options are which files to read and which seed
indices to drop for a recorded reason; no flag changes a test.

INPUT     one probe_results.json per pretraining-seed index, written by
          experiments/EVAL/probe.py. Arms `l188-sN`, `l162-sN`, `r42q1-sN`,
          `r16q1-sN` are the 188-, 162-, 43- and 17-class models; `l162-s1b` is
          seed index 1 of the 162-class model.
          With --label-recovery, additionally one label_recovery.json per seed
          index, written by experiments/EVAL/label_recovery.py (prediction S9).
ENDPOINT  `log1m_auc` exactly as probe.py stores it: the NATURAL log of 1 - AUC,
          floored at the sample's resolution (`log1m_auc_censored` marks a
          bound). LOWER is better. The S9 block has its OWN endpoint, balanced
          accuracy, where HIGHER is better; it says so on every field.
UNIT      the pretraining seed. Contrasts are paired by seed index (2.1, 2.2).

A MULTI-CLAUSE PREDICTION IS REPORTED CLAUSE BY CLAUSE. C1 is three clauses and
S9 is two. A prediction that contains a predicted-EQUAL clause beside a
directional one cannot be answered by the directional test alone: 2.5 sends the
equal clause to an equivalence test, and a trend test that fires says nothing
about it. Each clause carries its own verdict, and the composite verdict names
which clauses hold -- "confirmed in clauses 1-2, inconclusive in clause 3" --
rather than collapsing to a bare "confirmed" or "rejected".

ORDER OF OPERATIONS is part of the pre-specification (2.6): the minimum
detectable effect and the per-level seed standard deviations are printed BEFORE
any contrast, and that section prints no level mean and no p-value.

SIGN CONVENTION. Levels run fine -> coarse (188, 162, 43, 17). "Performance
falls with coarser labels" means log(1 - AUC) INCREASES along that order, so
the trend tests use alternative="increasing" with that explicit order, and every
paired difference is (coarser - finer): positive = the coarser label set is
worse.

The three defects experiments/FT/leg_stats.py lists are respected. Values are
read at full precision from the per-arm cells -- never from a rounded or
pre-aggregated block, and never from probe.py's own `contrasts`, whose
intervals are over jets, not seeds. Every spread is named for what it is. No
ratio is printed as a multiple of a standard deviation: only t, df and p.

Usage:
    python3 experiments/STATS/seed_level.py /data/results/eval/probe_ladder_v1 \
        --out /data/results/eval/probe_ladder_v1/stats \
        [--drop-pairs 2 --drop-reason "162-class seed 2 trained on a different GPU model"]

    # S9, on its own or beside the ladder; writes s9_label_recovery.json
    python3 experiments/STATS/seed_level.py \
        --label-recovery /data/results/eval/label_recovery_ladder_v1 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.stats.inference import holm, paired_t, tost          # noqa: E402
from src.stats.mde import mde_total, seed_term_multiplier     # noqa: E402
from src.stats.trend import isotonic_trend, max_t_trend       # noqa: E402

PRESPEC = REPO / "docs" / "PRESPEC_2026-09.md"
STATS_MODULES = ("src/stats/trend.py", "src/stats/inference.py", "src/stats/mde.py")

LEVELS = (188, 162, 43, 17)        # fine -> coarse: the order of every trend test
TREND_DIRECTION = "increasing"     # of log(1-AUC) along LEVELS == performance falls
TREND_FAMILY = "marcus"
ARM_LEVEL = {"l188": 188, "l162": 162, "r42q1": 43, "r16q1": 17}
ARM_ALIAS = {"l162-s1b": "l162-s1"}
ARM_RE = re.compile(r"^(l188|l162|r42q1|r16q1)-s([1-9]\d*)$")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)   # PRESPEC 2.1: five per configuration
PROBES = ("linear", "mlp")
ENDPOINT = "log1m_auc"
LOG_BASE = math.e                  # probe.py log1m_auc uses np.log
ALPHA, POWER = 0.05, 0.80
# The headline working point for background rejection, fixed blind in
# docs/PRESPEC_2026-09.md before any re-run number existed, because 50 % signal
# efficiency leaves no background jets at the three finer vocabularies and the
# metric is then a statement about the size of the test sample. Rejection is
# summarised at EVERY point the probe recorded; this one is labelled headline.
HEADLINE_EPS_S = "0.90"
PAIRS = [(a, b) for i, a in enumerate(LEVELS) for b in LEVELS[i + 1:]]   # (finer, coarser)

SEEN_TASKS = ("bvc_resonant", "bvc_qcd", "retained_topology", "ee_vs_mm")   # PRESPEC 1
CONTROL_TASKS = ("bvc_4prong", "visible_content")                          # C4's two class pairs
# PRESPEC 3 writes C1 as THREE clauses. The trend test answers the first, its
# arg-max contrast locates the second, and the third -- "188 ~ 162 ~ 43" -- is a
# PREDICTED NULL, which 2.5 sends to an equivalence test. Reporting the trend
# test alone as "C1" overstates the prediction: a ladder can fall with its step
# in the predicted place and still have the three fine levels differ.
C1 = {"task": "bvc_resonant", "predicted_step": ([188, 162, 43], [17]),
      "predicted_equal": (188, 162, 43),
      "prediction": "performance falls with coarser labels; the step is 43 -> 17; "
                    "188 ~ 162 ~ 43",
      "clauses": ("performance falls with coarser labels",
                  "the step is 43 -> 17",
                  "188 ~ 162 ~ 43")}
# Which other PRESPEC 3 predictions carry a predicted-EQUAL clause, i.e. one
# that 2.5 sends to an equivalence test. Transcribed from the document's wording,
# never inferred from it:
#   C2 "188 ~ 162 > 43 ~ 17", C3 "equivalent across the four granularities" and
#     S5 "same ordering as C2" do. All three are fine-tuning on the community
#     benchmarks, which this script does not read; they stay pending, and the
#     equivalence machinery below is what they will use.
#   S2 "equivalent across granularities" does, and equivalence() runs it.
#   S1 ("188 best; then 162 >= 43 > 17"), S3 ("steps at 162->43 and 43->17") and
#     S4 ("gap shrinks with size but does not vanish") do NOT. ">=" admits a
#     difference, and naming where the steps are is not a claim that the other
#     rungs are equal. Neither gets an equivalence test, because the document
#     does not predict one -- it would be a test this analysis invented.
S1 = {"task": "bvc_qcd", "predicted_step": None}   # 188 best, then 162 >= 43 > 17: no single step
S2_TASKS = ("retained_topology", "ee_vs_mm")
CONFIRMATORY = ("C1", "C2", "C3", "C4", "C5")
# Predicted sign of (random-label control - 17-class model, same seed index) in
# log(1-AUC), draws 1, 2, 3 (PRESPEC 3, "C4 in detail"). 0 = predicted no difference.
C4_PREDICTED = {"bvc_4prong": (-1, -1, 0), "visible_content": (+1, 0, +1)}


def tost_bound(log_base: float = LOG_BASE, ratio: float = 1.1) -> float:
    """PRESPEC 2.5's +-ln(1.1) in the endpoint's log base: log_base(1.1)."""
    return math.log(ratio) / math.log(log_base)


def parse_arm(name: str) -> tuple[int, int]:
    """Arm key -> (granularity level, seed index). Anything else is refused."""
    m = ARM_RE.match(ARM_ALIAS.get(name, name))
    if not m:
        raise SystemExit(f"FATAL: arm name {name!r} does not parse as "
                         f"<{'|'.join(ARM_LEVEL)}>-s<seed index> (aliases: {ARM_ALIAS}). "
                         f"An unrecognised arm must not be silently dropped from a ladder.")
    return ARM_LEVEL[m.group(1)], int(m.group(2))


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_ladder(src, parse=parse_arm) -> dict:
    """Tidy table: one row per (task, probe kind, level, seed index).

    `src` is a directory holding s*/probe_results.json, or a list of files.
    Refuses if the files disagree on `row_alignment_sha256` or `n_jets_total`
    (the models were then not scored on the same jets), if an arm name does not
    parse, or if a (level, seed) cell appears twice. Missing cells are not an
    error here; `missing_cells` reports them.

    `parse` maps an arm name to (cell, seed index) and defaults to the ladder's
    four granularities. The mass-output 2x2 passes `parse_mass_arm`, whose cell
    is a string naming one corner of the 2x2 rather than an integer level. It
    is a parameter and not a second loader because every guard above -- one
    alignment hash, one jet count, no duplicated cell -- has to hold identically
    for both, and two copies of it would be two things to keep in agreement.
    """
    paths = [pathlib.Path(p) for p in ([src] if isinstance(src, (str, pathlib.Path)) else src)]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob("s*/probe_results.json"))
    if not paths:
        raise SystemExit(f"FATAL: no probe_results.json found under {src}")
    docs = {p: json.loads(p.read_text()) for p in paths}
    for key in ("row_alignment_sha256", "n_jets_total"):
        seen = {str(p): d.get(key) for p, d in docs.items()}
        if None in seen.values() or len(set(seen.values())) != 1:
            raise SystemExit(f"FATAL: input files disagree on (or lack) `{key}`; a paired "
                             f"contrast across them would compare different jets.\n"
                             + "\n".join(f"  {p}: {v}" for p, v in seen.items()))
    rows, keys, skipped, checkpoints = [], set(), [], {}
    for p, d in docs.items():
        checkpoints.update(d.get("arm_checkpoints") or {})
        for task, t in d["tasks"].items():
            if t.get("skipped"):
                skipped.append({"file": str(p), "task": task})
                continue
            for arm, entry in t["arms"].items():
                level, seed = parse(arm)
                for kind in PROBES:
                    if (task, kind, level, seed) in keys:
                        raise SystemExit(f"FATAL: duplicated cell task={task} probe={kind} "
                                         f"level={level} seed={seed} (arm {arm!r} in {p})")
                    keys.add((task, kind, level, seed))
                    e = entry[kind]
                    # Rejection at the default signal efficiency; the flat fields
                    # mirror the task's first working point, which travels with them.
                    at = e["rejection_at"].get(f"{d['eps_s_default']:.2f}")
                    rej = at or e
                    rows.append({
                        "task": task, "probe": kind, "level": level, "seed": seed, "arm": arm,
                        ENDPOINT: float(e[ENDPOINT]), "censored": bool(e["log1m_auc_censored"]),
                        "auc": float(e["auc"]), "rejection": float(rej["rejection"]),
                        "rejection_is_bound": bool(rej["rejection_is_bound"]),
                        "rejection_eps_s": float(d["eps_s_default"] if at else e["rejection_eps_s"]),
                        # Every working point the probe recorded, not just the
                        # default one. docs/PRESPEC_2026-09.md fixes 90 % signal
                        # efficiency as the HEADLINE point for this metric,
                        # because 50 % is censored on this task; the flat fields
                        # above stay at the default so earlier outputs and the
                        # table generator keep reading what they always read.
                        "rejection_points": {
                            eps: {"rejection": float(v["rejection"]),
                                  "is_bound": bool(v["rejection_is_bound"]),
                                  "n_bkg_pass": int(v["n_bkg_pass"]),
                                  "rel_stat_err": float(v["rel_stat_err"])}
                            for eps, v in sorted(e["rejection_at"].items())},
                        "file": str(p)})
    first = next(iter(docs.values()))
    return {"rows": rows, "files": [{"path": str(p), "sha256": _sha(p)} for p in paths],
            "row_alignment_sha256": first["row_alignment_sha256"],
            "n_jets_total": first["n_jets_total"], "arm_checkpoints": checkpoints,
            "skipped_tasks": skipped}


def missing_cells(rows, seeds) -> dict:
    """{"task/probe": [[level, seed], ...]} for every expected cell that is absent."""
    have = {(r["task"], r["probe"], r["level"], r["seed"]) for r in rows}
    out = {}
    for task in sorted({r["task"] for r in rows}):
        for kind in PROBES:
            miss = [[lv, s] for s in seeds for lv in LEVELS if (task, kind, lv, s) not in have]
            if miss:
                out[f"{task}/{kind}"] = miss
    return out


def index_cells(rows, drop=()) -> dict:
    """(task, probe, level, seed) -> row, without the dropped seed indices."""
    return {(r["task"], r["probe"], r["level"], r["seed"]): r for r in rows
            if r["seed"] not in drop}


def _cell(cells, task, kind, level, seed, field=ENDPOINT):
    r = cells.get((task, kind, level, seed))
    return None if r is None else r[field]


def paired_diffs(cells, task, kind, fine, coarse, seeds) -> dict:
    """(coarser - finer) by seed index, over the seeds that have both cells."""
    used = [s for s in seeds if _cell(cells, task, kind, fine, s) is not None
            and _cell(cells, task, kind, coarse, s) is not None]
    d = [_cell(cells, task, kind, coarse, s) - _cell(cells, task, kind, fine, s) for s in used]
    bound = any(_cell(cells, task, kind, lv, s, "censored") for lv in (fine, coarse) for s in used)
    return {"seeds": used, "d": d, "is_bound": bool(bound)}


def sign_flip_p(d) -> dict:
    """Exact two-sided sign-flip p on paired differences, with its floor 2/2^n."""
    d = np.asarray(d, dtype=float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    sums, obs = np.abs(signs @ d), abs(d.sum())
    return {"p": float((sums >= obs - 1e-12 * max(1.0, obs)).mean()),
            "floor": 2.0 / 2 ** len(d), "n_arrangements": 2 ** len(d)}


def pair_contrast(pd: dict) -> dict:
    """Primary: paired t on n-1 df. Beside it: the exact sign-flip p and its floor."""
    n = len(pd["d"])
    base = {"n_pairs": n, "seeds": pd["seeds"], "is_bound": pd["is_bound"]}
    if n < 2:
        return {**base, "estimable": False, "reason": "fewer than 2 complete seed pairs"}
    if np.ptp(pd["d"]) == 0:    # e.g. both levels censored at the same resolution floor
        return {**base, "estimable": False, "mean_diff": float(np.mean(pd["d"])),
                "reason": "identical paired differences (zero variance): t undefined"}
    t = paired_t(pd["d"], ALPHA)
    return {**base, "estimable": True, "mean_diff": t["mean"],
            "sd_diff": float(np.std(pd["d"], ddof=1)), "t": t["t"], "df": n - 1,
            "p": t["p"], "ci95": list(t["ci"]), "sign_flip": sign_flip_p(pd["d"])}


def pairwise_table(cells, task, kind, seeds) -> list[dict]:
    """All six (coarser - finer) contrasts of one task, Holm-corrected within the table."""
    rows = [{"fine": a, "coarse": b, **pair_contrast(paired_diffs(cells, task, kind, a, b, seeds))}
            for a, b in PAIRS]
    est = [r for r in rows if r["estimable"]]
    for r, rej in zip(est, holm([r["p"] for r in est], ALPHA) if est else []):
        r["holm_reject"] = bool(rej)
    return rows


def mde_row(cells, task, kind, seeds) -> dict:
    """PRESPEC 2.6, from the 17-class-minus-162-class paired differences."""
    pd = paired_diffs(cells, task, kind, 162, 17, seeds)
    n = len(pd["d"])
    row = {"task": task, "probe": kind, "n_pairs": n, "seeds": pd["seeds"]}
    if n < 2:
        return {**row, "estimable": False, "reason": "fewer than 2 complete seed pairs"}
    sd = float(np.std(pd["d"], ddof=1))
    mde = mde_total(sd, n, 0.0, ALPHA, POWER)
    return {**row, "estimable": True, "sd_paired_diff": sd,
            "multiplier": seed_term_multiplier(n, ALPHA, POWER), "mde": mde,
            "mde_as_ratio_of_1m_auc": LOG_BASE ** mde}


def level_summary(cells, task, kind, seeds) -> list[dict]:
    out = []
    for lv in LEVELS:
        rs = [cells[(task, kind, lv, s)] for s in seeds if (task, kind, lv, s) in cells]
        y = [r[ENDPOINT] for r in rs]
        rej = [r["rejection"] for r in rs]
        out.append({"level": lv, "n_seeds": len(rs),
                    "mean": float(np.mean(y)) if rs else None,
                    "seed_sd": float(np.std(y, ddof=1)) if len(rs) > 1 else None,
                    "mean_auc": float(np.mean([r["auc"] for r in rs])) if rs else None,
                    "rejection_median": float(np.median(rej)) if rs else None,
                    "rejection_range": [min(rej), max(rej)] if rs else None,
                    "rejection_eps_s": rs[0]["rejection_eps_s"] if rs else None,
                    "n_rejection_bound": sum(r["rejection_is_bound"] for r in rs),
                    "rejection_points": rejection_points(rs),
                    "headline_eps_s": HEADLINE_EPS_S,
                    "n_censored": sum(r["censored"] for r in rs)})
    return out


def rejection_points(rs) -> dict:
    """Per working point: median rejection, its seed range, how many seeds hit
    the cap, and the mean number of background jets left. A point where any seed
    is bound is not a property of the models, so `is_bound` travels with it and
    the table generator refuses to print a bound cell as a bare number."""
    eps_all = sorted({e for r in rs for e in r.get("rejection_points", {})})
    out = {}
    for eps in eps_all:
        vs = [r["rejection_points"][eps] for r in rs if eps in r.get("rejection_points", {})]
        if not vs:
            continue
        rej = [v["rejection"] for v in vs]
        left = [v["n_bkg_pass"] for v in vs]
        out[eps] = {"n_seeds": len(vs), "median": float(np.median(rej)),
                    "range": [min(rej), max(rej)],
                    "n_bound": sum(v["is_bound"] for v in vs),
                    "mean_n_bkg_pass": float(np.mean(left)),
                    "is_headline": eps == HEADLINE_EPS_S}
    return out


def trend_test(cells, task, kind, seeds, predicted_step=None) -> dict:
    """Max-T trend test (Marcus family, exact, blocks = seed index) + isotonic companion.

    Direction is fixed: log(1-AUC) INCREASES along LEVELS (fine -> coarse), i.e.
    performance falls with coarser labels. A seed block with a missing cell is
    dropped whole and reported; with fewer than 2 complete blocks nothing is run.
    """
    have = {s: [lv for lv in LEVELS if _cell(cells, task, kind, lv, s) is not None] for s in seeds}
    complete = [s for s in seeds if len(have[s]) == len(LEVELS)]
    out = {"task": task, "probe": kind, "levels_fine_to_coarse": list(LEVELS),
           "alternative": f"{ENDPOINT} {TREND_DIRECTION} along levels = performance falls "
                          f"with coarser labels",
           "blocks_used": complete,
           "blocks_dropped_incomplete": {str(s): [lv for lv in LEVELS if lv not in have[s]]
                                         for s in seeds if s not in complete}}
    if len(complete) < 2:
        return {**out, "run": False, "reason": "fewer than 2 complete seed blocks — not run"}
    trip = [(cells[(task, kind, lv, s)], lv, s) for s in seeds for lv in have[s]]
    y, levels, blocks = zip(*[(r[ENDPOINT], lv, s) for r, lv, s in trip])
    mt = max_t_trend(y, levels, blocks, alternative=TREND_DIRECTION, family=TREND_FAMILY,
                     exact=True, order=list(LEVELS))
    iso = isotonic_trend(y, levels, blocks, increasing=TREND_DIRECTION == "increasing",
                         exact=True, order=list(LEVELS))
    if mt["n_blocks"] != len(complete):
        raise SystemExit("FATAL: trend test and loader disagree on the complete seed blocks")
    step = [list(mt["step"][0]), list(mt["step"][1])]
    return {**out, "run": True, "family": TREND_FAMILY, "method": mt["method"],
            "n_blocks": mt["n_blocks"], "n_arrangements": mt["n_arrangements"],
            "stat": mt["stat"], "p": mt["p"], "p_min": mt["p_min"],
            # PRESPEC addition: a single step at one end cannot go below ~(1/4)^blocks
            "end_step_p_bound": 0.25 ** mt["n_blocks"],
            "argmax_step": step,
            "predicted_step": None if predicted_step is None else [list(x) for x in predicted_step],
            "argmax_is_predicted_step": None if predicted_step is None
            else step == [list(x) for x in predicted_step],
            "contrasts_localisation_only": mt["contrasts"],
            "n_censored_cells": sum(r["censored"] for r, _, s in trip if s in complete),
            # exchangeability is the null, not equal means: spreads go beside the test
            "seed_sd_per_level": [float(np.std([cells[(task, kind, lv, s)][ENDPOINT]
                                                for s in complete], ddof=1)) for lv in LEVELS],
            "isotonic": {"stat": iso["stat"], "p": iso["p"], "fitted": iso["fit"]["fitted"],
                         "pooled": iso["fit"]["pooled"], "means": iso["fit"]["means"]}}


def tost_pair(cells, task, kind, fine, coarse, seeds, bound) -> dict | None:
    """One level pair's two one-sided tests, or None when it is not estimable."""
    pd = paired_diffs(cells, task, kind, fine, coarse, seeds)
    if len(pd["d"]) < 2 or np.ptp(pd["d"]) == 0:    # zero variance: see pair_contrast
        return None
    r = tost(pd["d"], bound=bound, alpha=ALPHA)
    return {"fine": fine, "coarse": coarse, "n_pairs": len(pd["d"]), "is_bound": pd["is_bound"],
            "mean_diff": float(np.mean(pd["d"])), "p": r["p"], "ci90": list(r["ci_1m2a"]),
            "equivalent_at_target": r["equivalent"],
            "smallest_bound_passed": float(max(abs(x) for x in r["ci_1m2a"]))}


def tost_verdict(equivalent: bool, bound: float, x: float) -> str:
    """PRESPEC 2.5's wording. `x` is the smallest bound passed. Never "no effect"."""
    return (f"equivalent within ±{bound:.4f} (±10% in 1−AUC); smallest bound passed ±{x:.4f}"
            if equivalent else
            f"inconclusive at ±10%; equivalent within ±{x:.4f} "
            f"(a factor {LOG_BASE ** x:.3f} in 1−AUC)")


def equivalence(cells, task, kind, seeds) -> dict:
    """S2: TOST on the largest pairwise paired difference, bound +-ln(1.1).

    The pair is chosen by rule -- largest |mean paired difference| among the six
    -- and `smallest_bound_passed` is the half-width at which its (1-2a) interval
    just fits. The maximum over all six pairs is reported beside it as a
    companion, because the pair with the largest mean is not necessarily the
    pair with the widest interval. The wording is PRESPEC 2.5's, never "no effect".
    """
    bound = tost_bound()
    rows = [r for r in (tost_pair(cells, task, kind, a, b, seeds, bound) for a, b in PAIRS)
            if r is not None]
    out = {"task": task, "probe": kind, "target_bound": bound, "log_base": "e",
           "n_pairs_estimable": len(rows), "n_pairs_total": len(PAIRS)}
    if not rows:
        return {**out, "run": False, "reason": "no level pair with 2 complete seed pairs and "
                                               "non-identical differences — not run"}
    top = max(rows, key=lambda r: abs(r["mean_diff"]))
    return {**out, "run": True, "largest_pair": top, "p": top["p"],
            "verdict": tost_verdict(top["equivalent_at_target"], bound,
                                    top["smallest_bound_passed"]),
            "all_pairs_companion": {"max_p": max(r["p"] for r in rows),
                                    "smallest_bound_passed_by_all": max(
                                        r["smallest_bound_passed"] for r in rows)},
            "pairs": rows}


def equivalence_set(cells, task, kind, levels, seeds) -> dict:
    """A predicted-EQUAL SET of levels: TOST on every pair inside it (PRESPEC 2.5).

    C1's third clause, "188 ~ 162 ~ 43", is a null over a SET, so it holds only
    if EVERY pair inside that set is equivalent at ±ln(1.1). That is an
    intersection-union test: the set's p is the MAXIMUM of the pair p-values and
    takes no multiplicity correction, because the set null is rejected only when
    every component null is. A pair that does not reach the target reports the
    smallest bound it does pass -- never "no effect", and never a bare
    "rejected", which is what a directional test would have produced here.
    """
    if list(levels) != [lv for lv in LEVELS if lv in levels]:
        raise SystemExit(f"FATAL: the predicted-equal set {list(levels)} must be a subset of "
                         f"{list(LEVELS)} given fine -> coarse; the pair differences are "
                         f"coarser − finer and would otherwise change sign silently")
    bound = tost_bound()
    pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]]
    rows = [r for r in (tost_pair(cells, task, kind, a, b, seeds, bound) for a, b in pairs)
            if r is not None]
    out = {"task": task, "probe": kind, "levels": list(levels), "target_bound": bound,
           "log_base": "e", "n_pairs_estimable": len(rows), "n_pairs_total": len(pairs)}
    if not rows:
        return {**out, "run": False, "reason": "no pair inside the set has 2 complete seed pairs "
                                               "and non-identical differences — not run"}
    worst = max(rows, key=lambda r: r["smallest_bound_passed"])
    all_eq = len(rows) == len(pairs) and all(r["equivalent_at_target"] for r in rows)
    return {**out, "run": True, "all_equivalent": all_eq,
            "p": max(r["p"] for r in rows),          # intersection-union: the worst pair
            "n_equivalent": sum(r["equivalent_at_target"] for r in rows),
            "not_equivalent": [[r["fine"], r["coarse"]] for r in rows
                               if not r["equivalent_at_target"]],
            "widest_pair": [worst["fine"], worst["coarse"]],
            "smallest_bound_passed_by_all": worst["smallest_bound_passed"],
            "verdict": tost_verdict(all_eq, bound, worst["smallest_bound_passed"]),
            "pairs": rows}


def sign_agreement(linear: list[dict], mlp: list[dict]) -> dict:
    """S6: does the MLP probe order every pair of levels the way the linear probe does?"""
    pairs = [{"fine": a["fine"], "coarse": a["coarse"],
              "linear": int(np.sign(a["mean_diff"])), "mlp": int(np.sign(b["mean_diff"]))}
             for a, b in zip(linear, mlp) if a["estimable"] and b["estimable"]]
    for p in pairs:
        p["agree"] = p["linear"] == p["mlp"]
    return {"n_agree": sum(p["agree"] for p in pairs), "n_pairs": len(pairs), "pairs": pairs}


def c4_sign_pattern(diffs: dict, ci: dict | None = None) -> dict:
    """C4: sign pattern of (random-label control - 17-class model) over the six cells.

    diffs[task] = three differences in log(1-AUC), draws 1, 2, 3, each paired
    with the 17-class model of seed index 1, 2, 3. ci[task] (optional) = three
    (lo, hi) intervals, e.g. probe.py's paired bootstrap over jets.

    Four cells carry a signed prediction and two are predicted zero
    (C4_PREDICTED). Only the signed cells can match a sign; `p_at_least` is the
    exact probability of at least `n_match` matches among them when every
    difference is symmetric about zero, i.e. Binomial(4, 1/2), floor 1/16. A
    predicted-zero cell is scored only when an interval is supplied (consistent
    = the interval covers zero) and never enters that probability.

    DESCRIPTIVE ONLY. There is ONE run per draw, so a cell has no seed-level
    replicate and its interval, if any, is over jets, not over pretraining
    seeds. The binomial also treats the cells as independent, which the two
    draw-1 cells are not: they share both models.
    """
    cells = []
    for task, pred in C4_PREDICTED.items():
        if len(diffs[task]) != len(pred):
            raise SystemExit(f"FATAL: C4 needs {len(pred)} draws for {task}, got {len(diffs[task])}")
        for i, (d, s) in enumerate(zip(diffs[task], pred)):
            iv = None if ci is None else [float(x) for x in ci[task][i]]
            cells.append({"task": task, "draw": i + 1, "seed": i + 1, "diff": float(d), "ci": iv,
                          "predicted_sign": s,
                          "sign_matches": bool(np.sign(d) == s) if s else None,
                          "zero_consistent": (iv[0] <= 0.0 <= iv[1]) if (not s and iv) else None})
    signed = [c for c in cells if c["predicted_sign"]]
    k, m = sum(c["sign_matches"] for c in signed), len(signed)
    return {"cells": cells, "n_signed_cells": m, "n_match": k,
            "p_at_least": sum(math.comb(m, j) for j in range(k, m + 1)) / 2 ** m,
            "p_floor": 1 / 2 ** m,
            "zero_cells_consistent": [c["zero_consistent"] for c in cells if not c["predicted_sign"]],
            "inference_level": "descriptive: one run per draw"}


# ------------------------------------------- C5: granularity x mass output

# The 2x2. Keys are the four corners, not integer levels: a mass-output model
# and its plain twin share a granularity, so keying on the level alone would
# collide and load_ladder's duplicated-cell guard would (correctly) refuse the
# file. Keeping the corners as strings is what lets the ladder loader stay
# strict about its own four levels.
MASS_CELL = {"l162": "162", "l162mass": "162+mass",
             "r16q1": "17", "r16q1mass": "17+mass"}
MASS_ARM_RE = re.compile(r"^(l162mass|r16q1mass|l162|r16q1)-s([1-9]\d*)$")
MASS_LEVELS = (162, 17)                 # the two granularities that have a twin
C5_TASK = "bvc_resonant"                # PRESPEC C5: "the b-versus-c probe"
C5_DIRECTION = ("two-sided; the written expectation is that the mass output "
                "helps more at 17 classes than at 162")


def parse_mass_arm(name: str) -> tuple[str, int]:
    """Arm key -> (2x2 corner, seed index). Anything else is refused.

    Deliberately a different function from `parse_arm` rather than a widening
    of it: `parse_arm` must go on refusing every name that is not one of the
    ladder's four granularities, because an unrecognised arm silently dropped
    from a ladder is the failure this whole module is built to prevent.
    """
    m = MASS_ARM_RE.match(ARM_ALIAS.get(name, name))
    if not m:
        raise SystemExit(f"FATAL: arm name {name!r} does not parse as "
                         f"<{'|'.join(MASS_CELL)}>-s<seed index> (aliases: {ARM_ALIAS}). "
                         f"An unrecognised arm must not be silently dropped from the 2x2.")
    return MASS_CELL[m.group(1)], int(m.group(2))


def mass_gain(cells, task, kind, level, seeds) -> dict:
    """(with mass output - without), by seed index, at one granularity.

    Negative is better: the endpoint is log(1 - AUC).
    """
    plain, mass = str(level), f"{level}+mass"
    used = [s for s in seeds if _cell(cells, task, kind, plain, s) is not None
            and _cell(cells, task, kind, mass, s) is not None]
    d = [_cell(cells, task, kind, mass, s) - _cell(cells, task, kind, plain, s) for s in used]
    bound = any(_cell(cells, task, kind, c, s, "censored") for c in (plain, mass) for s in used)
    return {"level": level, "seeds": used, "d": d, "is_bound": bool(bound)}


def mass_did(cells, task, kind, seeds) -> dict:
    """C5: the difference-in-differences, by seed index.

    did_s = (162+mass - 162)_s - (17+mass - 17)_s, over the seed indices that
    have all FOUR corners. A seed missing any one corner is dropped whole
    rather than contributing a half-difference -- the quantity is the
    interaction, and a partial seed cannot carry one.

    Sign: the endpoint is log(1 - AUC) and lower is better, so the written
    expectation ("the mass output helps more at 17 classes") is did > 0. The
    test itself is two-sided, as pre-registered.
    """
    fine, coarse = MASS_LEVELS
    used = [s for s in seeds
            if all(_cell(cells, task, kind, c, s) is not None
                   for c in (str(fine), f"{fine}+mass", str(coarse), f"{coarse}+mass"))]
    d = [(_cell(cells, task, kind, f"{fine}+mass", s) - _cell(cells, task, kind, str(fine), s))
         - (_cell(cells, task, kind, f"{coarse}+mass", s) - _cell(cells, task, kind, str(coarse), s))
         for s in used]
    bound = any(_cell(cells, task, kind, c, s, "censored") for s in used
                for c in (str(fine), f"{fine}+mass", str(coarse), f"{coarse}+mass"))
    return {"seeds": used, "d": d, "is_bound": bool(bound)}


def c5_analysis(cells, seeds, tasks) -> dict:
    """C5 on the pre-registered task, and the same quantity on every other.

    Only `C5_TASK` is confirmatory: docs/PRESPEC_2026-09.md fixed C5 on the
    b-versus-c probe before any of these numbers existed. The other tasks are
    reported because the 2x2 measured them and withholding a measured cell is
    its own kind of selection, but they are labelled exploratory, they carry no
    verdict, and they enter no multiplicity family. Nothing downstream may
    promote one of them.
    """
    def block(task):
        out = {"task": task, "probes": {}}
        for kind in PROBES:
            did = mass_did(cells, task, kind, seeds)
            out["probes"][kind] = {
                "did": pair_contrast(did),
                "gain_by_level": {str(lv): pair_contrast(mass_gain(cells, task, kind, lv, seeds))
                                  for lv in MASS_LEVELS}}
        return out

    conf = block(C5_TASK) if C5_TASK in tasks else {"task": C5_TASK, "probes": {}}
    linear = conf["probes"].get("linear", {}).get("did", {})
    p = linear.get("p") if linear.get("estimable") else None
    return {"prediction": C5_DIRECTION, "task": C5_TASK, "endpoint": ENDPOINT,
            "did_definition": "(162+mass − 162) − (17+mass − 17), paired by seed index",
            "expected_sign_if_written_expectation_holds": "+",
            "confirmatory": conf, "p": p,
            "exploratory": [block(t) for t in tasks if t != C5_TASK],
            "exploratory_note": "measured by the same jobs; no verdict, no multiplicity family"}


def holm_family(entries: list[tuple[str, float | None]], alpha: float = ALPHA) -> list[dict]:
    """Holm over the FULL family, pending members included.

    A test is judged at the family size from the start: with m members the
    smallest p must beat alpha/m. `reject_whatever_pending` takes every pending
    p as 1, `reject_possible` takes it as 0; they coincide once nothing is pending.
    """
    m = len(entries)
    worst = holm([1.0 if p is None else p for _, p in entries], alpha)
    best = holm([0.0 if p is None else p for _, p in entries], alpha)
    return [{"test": name, "status": "pending" if p is None else "available", "p_raw": p,
             "family_size": m, "threshold_if_smallest": alpha / m,
             "reject_whatever_pending": None if p is None else bool(w),
             "reject_possible": None if p is None else bool(b)}
            for (name, p), w, b in zip(entries, worst, best)]


# ------------------------------------------------- multi-clause predictions

def clause(n: int, text: str, test: str, verdict: str, p=None, detail: str = "") -> dict:
    """One clause of a prediction, with the test that answers it and its verdict.

    `verdict` is one of: confirmed; not confirmed; inconclusive (2.5's word for
    an equivalence test that did not reach ±10% -- never "no effect" and never
    "rejected"); not run.
    """
    if verdict not in ("confirmed", "not confirmed", "inconclusive", "not run"):
        raise SystemExit(f"FATAL: {verdict!r} is not one of the four clause verdicts")
    return {"n": n, "text": text, "test": test, "verdict": verdict, "p": p, "detail": detail}


def composite_verdict(clauses: list[dict]) -> str:
    """"confirmed in clauses 1-2, inconclusive in clause 3" -- in clause order.

    Consecutive clauses that share a verdict are merged into one span, so the
    sentence names exactly which clauses hold. This is the string a table prints
    instead of collapsing a three-clause prediction to one word.
    """
    parts, i = [], 0
    while i < len(clauses):
        j = i
        while j + 1 < len(clauses) and clauses[j + 1]["verdict"] == clauses[i]["verdict"]:
            j += 1
        nums = [c["n"] for c in clauses[i:j + 1]]
        span = f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else str(nums[0])
        parts.append(f"{clauses[i]['verdict']} in {'clauses' if len(nums) > 1 else 'clause'} {span}")
        i = j + 1
    return ", ".join(parts)


def overall_verdict(clauses: list[dict]) -> str:
    """confirmed / partially confirmed / not confirmed, over all clauses."""
    v = [c["verdict"] for c in clauses]
    if all(x == "confirmed" for x in v):
        return "confirmed"
    if not any(x == "confirmed" for x in v):
        return "not confirmed"
    return "partially confirmed"


def compose(prediction: str, clauses: list[dict], **extra) -> dict:
    """The block every multi-clause prediction carries beside its tests."""
    return {"prediction": prediction, "clauses": clauses,
            "composite_verdict": composite_verdict(clauses),
            "overall": overall_verdict(clauses),
            "n_clauses": len(clauses),
            "n_clauses_confirmed": sum(c["verdict"] == "confirmed" for c in clauses), **extra}


def c1_clauses(trend: dict, equal: dict, holm_entry: dict, family_size: int) -> dict:
    """C1's three clauses (PRESPEC 3) and the composite verdict over them.

    clause 1  the trend test, judged at the FULL confirmatory family threshold
              (2.7), so the verdict holds whatever the pending members give.
    clause 2  the arg-max contrast against the predicted step. Localisation
              only: the addition of 2026-09-18 says the per-contrast adjusted
              p-values locate the step and are not separate tests, so this
              clause carries no p of its own.
    clause 3  the predicted null "188 ~ 162 ~ 43", as an equivalence test over
              the set (2.5). A trend test cannot answer it in either direction:
              failing to reject a difference is not equivalence, and the max-T
              statistic is driven by the step, not by the three fine levels.
    """
    if not trend["run"]:
        c_1 = clause(1, C1["clauses"][0], "max-T trend test", "not run", detail=trend["reason"])
        c_2 = clause(2, C1["clauses"][1], "arg-max contrast of the trend test", "not run",
                     detail=trend["reason"])
    else:
        rejected = bool(holm_entry.get("reject_whatever_pending"))
        c_1 = clause(1, C1["clauses"][0],
                     f"max-T trend test ({trend['method']}, {trend['n_blocks']} seed blocks)",
                     "confirmed" if rejected else "not confirmed", p=trend["p"],
                     detail=(f"Holm at the full confirmatory family of {family_size}: must beat "
                             f"{ALPHA / family_size:.3g} as the smallest; "
                             + ("rejected whatever the pending tests give" if rejected
                                else "does not reach that threshold")))
        step = trend["argmax_step"]
        c_2 = clause(2, C1["clauses"][1], "arg-max contrast of the trend test",
                     "confirmed" if trend["argmax_is_predicted_step"] else "not confirmed",
                     detail=f"arg-max {step[0]} | {step[1]}, predicted "
                            f"{trend['predicted_step'][0]} | {trend['predicted_step'][1]}; "
                            f"localisation only, no separate p")
    if not equal["run"]:
        c_3 = clause(3, C1["clauses"][2], "equivalence (TOST) over the set", "not run",
                     detail=equal["reason"])
    else:
        c_3 = clause(3, C1["clauses"][2],
                     f"equivalence (TOST) at ±ln(1.1) on all {equal['n_pairs_total']} pairs "
                     f"inside the set, intersection-union",
                     "confirmed" if equal["all_equivalent"] else "inconclusive",
                     p=equal["p"], detail=equal["verdict"])
    return compose(C1["prediction"], [c_1, c_2, c_3], clause3_equivalence=equal)


# ------------------------------------------- S9: label recovery across the tree

# The eight rungs of the contraction tree, FINE -> COARSE (docs/DECISIONS.md D3;
# the same order experiments/EVAL/label_recovery.py writes).
RUNGS = ("L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1")
RECOVERY_FILE = "label_recovery.json"
# S9's endpoint is NOT the ladder's. Balanced accuracy is higher-is-better, and
# S9 is written about "the advantage of finer models", so every difference in
# this block runs FINER MODEL - COARSER MODEL: positive = the finer model
# recovers that rung better. The probe ladder's differences run the other way
# (coarser - finer on log(1-AUC)); nothing is shared between the two conventions
# except the pairing and the paired t.
S9 = {"prediction": "each model wins at and below its own granularity; the advantage of finer "
                    "models decays to zero at the coarser model's own level",
      "clauses": ("each model wins at and below its own granularity",
                  "the advantage of finer models decays to zero at the coarser model's "
                  "own level")}


def load_recovery(src) -> dict:
    """Tidy table: one row per (rung, probe kind, model level, seed index).

    `src` is a directory holding s*/label_recovery.json (or s*.json), or a list
    of files, in the schema experiments/EVAL/label_recovery.py writes. Refuses
    if the files disagree on `row_alignment_sha256` or `n_test` (the models were
    then not scored on the same jets), if an arm name does not parse, if a
    (rung, level, seed) cell appears twice, if a rung is not one of the eight, or
    if an arm's `own_rung` is not the same in every seed file. A rung the probe
    skipped is recorded, not silently dropped.
    """
    paths = [pathlib.Path(p) for p in ([src] if isinstance(src, (str, pathlib.Path)) else src)]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob(f"s*/{RECOVERY_FILE}")) or sorted(paths[0].glob("s*.json"))
    if not paths:
        raise SystemExit(f"FATAL: no {RECOVERY_FILE} found under {src}")
    docs = {p: json.loads(p.read_text()) for p in paths}
    for key in ("row_alignment_sha256", "n_test"):
        seen = {str(p): d.get(key) for p, d in docs.items()}
        if None in seen.values() or len(set(seen.values())) != 1:
            raise SystemExit(f"FATAL: label-recovery files disagree on (or lack) `{key}`; a "
                             f"paired contrast across them would compare different jets.\n"
                             + "\n".join(f"  {p}: {v}" for p, v in seen.items()))
    rows, keys, skipped, own = [], set(), [], {}
    for p, d in docs.items():
        for arm, entry in d["arms"].items():
            level, seed = parse_arm(arm)
            if own.setdefault(level, entry["own_rung"]) != entry["own_rung"]:
                raise SystemExit(f"FATAL: {arm} in {p} says its own vocabulary is "
                                 f"{entry['own_rung']!r}; another file says {own[level]!r}")
            for rung, cell in entry["rungs"].items():
                if rung not in RUNGS:
                    raise SystemExit(f"FATAL: rung {rung!r} in {p} is not one of {list(RUNGS)}")
                if "linear" not in cell:
                    skipped.append({"file": str(p), "arm": arm, "rung": rung,
                                    "reason": cell.get("skipped", "no probe in the cell")})
                    continue
                for kind in PROBES:
                    if (rung, kind, level, seed) in keys:
                        raise SystemExit(f"FATAL: duplicated cell rung={rung} probe={kind} "
                                         f"level={level} seed={seed} (arm {arm!r} in {p})")
                    keys.add((rung, kind, level, seed))
                    rows.append({"rung": rung, "probe": kind, "level": level, "seed": seed,
                                 "arm": arm, "own_rung": entry["own_rung"],
                                 "accuracy": float(cell[kind]), "n_groups": int(cell["n_groups"]),
                                 "chance": float(cell["chance"]),
                                 "not_recovered": bool(cell["not_recovered"]),
                                 "mlp_converged": bool(cell["mlp_converged"]),
                                 "mlp_below_linear": bool(cell["mlp_below_linear"]),
                                 "file": str(p)})
    first = next(iter(docs.values()))
    return {"rows": rows, "files": [{"path": str(p), "sha256": _sha(p)} for p in paths],
            "row_alignment_sha256": first["row_alignment_sha256"],
            "n_used": first["n_used"], "n_train": first["n_train"], "n_test": first["n_test"],
            "chance_sigma": first.get("chance_sigma"), "own_rung": own, "skipped_cells": skipped}


def index_recovery(rows, drop=()) -> dict:
    """(rung, probe, level, seed) -> row, without the dropped seed indices."""
    return {(r["rung"], r["probe"], r["level"], r["seed"]): r for r in rows
            if r["seed"] not in drop}


def missing_recovery_cells(rcells, levels, rungs, seeds) -> dict:
    """{"rung/probe": [[level, seed], ...]} for every expected cell that is absent."""
    out = {}
    for rung in rungs:
        for kind in PROBES:
            miss = [[lv, s] for s in seeds for lv in levels if (rung, kind, lv, s) not in rcells]
            if miss:
                out[f"{rung}/{kind}"] = miss
    return out


def recovery_diff(rcells, rung, kind, a, b, seeds) -> dict:
    """(model a - model b) balanced accuracy at one rung, by seed index.

    Higher balanced accuracy is better, so a POSITIVE difference means model `a`
    recovers that rung better. Callers pass the finer model as `a`, which makes
    the difference "the advantage of the finer model" that S9 is written about.
    """
    used = [s for s in seeds
            if (rung, kind, a, s) in rcells and (rung, kind, b, s) in rcells]
    return {"seeds": used, "is_bound": False,   # balanced accuracy has no censoring
            "d": [rcells[(rung, kind, a, s)]["accuracy"] - rcells[(rung, kind, b, s)]["accuracy"]
                  for s in used]}


def recovery_row(rcells, rung, kind, fine, coarse, seeds) -> dict:
    """One rung of one model pair: paired t on n-1 df, sign-flip p beside it."""
    r = {"rung": rung, **pair_contrast(recovery_diff(rcells, rung, kind, fine, coarse, seeds))}
    r.pop("is_bound")           # nothing here can be a bound; see recovery_diff
    if r["estimable"]:
        lo, hi = r["ci95"]
        r["advantage"] = ("finer better" if lo > 0 else
                          "coarser better" if hi < 0 else "not distinguishable")
    else:
        r["advantage"] = None
    return r


def recovery_pair_table(rcells, fine, coarse, kind, seeds, rungs) -> list[dict]:
    """The eight rungs of one model pair, Holm-corrected within that table (2.7)."""
    rows = [recovery_row(rcells, rung, kind, fine, coarse, seeds) for rung in rungs]
    est = [r for r in rows if r["estimable"]]
    for r, rej in zip(est, holm([r["p"] for r in est], ALPHA) if est else []):
        r["holm_reject"] = bool(rej)
    return rows


def _in_bracket(rung: str, lo, hi) -> bool:
    """Is `rung` inside the crossover bracket [lo, hi]? `hi=None` is open-ended."""
    if lo is None:
        return False            # the advantage never stops being demonstrable
    i = RUNGS.index(rung)
    return RUNGS.index(lo) <= i and (hi is None or i <= RUNGS.index(hi))


def crossover(rows: list[dict], coarser_own_rung: str) -> dict:
    """Where the finer model's advantage reaches zero, with its uncertainty.

    Rows run fine -> coarse. Three rungs are read off the SAME per-rung 95%
    paired intervals, walking that order:

      crossover_rung    the first rung whose interval no longer excludes zero:
                        the advantage stops being demonstrable here. This is the
                        level S9 is about.
      sign_change_rung  the first rung whose point estimate is <= 0.
      excluded_rung     the first rung whose interval lies at or below zero: the
                        advantage is demonstrably gone.

    `crossover_bracket` = [crossover_rung, excluded_rung] is the uncertainty on
    a discrete ladder -- the crossover is at or after the first and at or before
    the second -- and it is what replaces reading a level off a plot. Any of the
    three is None when its condition never occurs on the ladder, and None is a
    finding: an advantage that never stops being demonstrable contradicts S9 as
    directly as one that stops at the wrong rung.

    The classification uses the RAW interval, not the Holm-corrected one. Holm
    is a statement about the family of eight rung tests; "where does the
    advantage reach zero" is a statement about each rung on its own, and the
    corrected flag travels beside every row for anyone who wants the other read.
    """
    est = [r for r in rows if r["estimable"]]

    def first(pred):
        return next((r["rung"] for r in est if pred(r)), None)

    cross = first(lambda r: r["ci95"][0] <= 0.0)
    excluded = first(lambda r: r["ci95"][1] <= 0.0)
    return {"crossover_rung": cross, "excluded_rung": excluded,
            "sign_change_rung": first(lambda r: r["mean_diff"] <= 0.0),
            "crossover_bracket": [cross, excluded],
            "last_rung_with_advantage": next(
                (r["rung"] for r in reversed(est) if r["advantage"] == "finer better"), None),
            "coarser_own_rung": coarser_own_rung,
            "crossover_at_coarser_own_rung": cross == coarser_own_rung,
            "coarser_own_rung_in_bracket": _in_bracket(coarser_own_rung, cross, excluded),
            "n_rungs_estimable": len(est), "n_rungs": len(rows),
            "per_rung_advantage": {r["rung"]: r["advantage"] for r in rows}}


def wins_at_and_below_own(rcells, level, own_rung, others, kind, seeds, rungs) -> dict:
    """S9's first clause for ONE model: "wins at and below its own granularity".

    "Below" is coarser: the model's own rung and every rung further down the
    tree. "Wins" is scored as "is not beaten": at those rungs a coarser
    vocabulary is a function of a finer one, so the models should tie, and a tie
    cannot be demonstrated by failing to reject. A cell counts against the clause
    only when ANOTHER model is ahead by more than the paired 95% interval; a cell
    merely behind on the point estimate is counted separately and not scored.
    """
    checked = [r for r in rungs if RUNGS.index(r) >= RUNGS.index(own_rung)]
    cells = []
    for rung in checked:
        for other in others:
            r = pair_contrast(recovery_diff(rcells, rung, kind, level, other, seeds))
            cells.append({"rung": rung, "other_level": other, "estimable": r["estimable"],
                          "mean_diff": r.get("mean_diff"), "p": r.get("p"),
                          "ci95": r.get("ci95"),
                          "behind_on_point_estimate": (r["mean_diff"] < 0.0) if r["estimable"]
                          else None,
                          "beaten": (r["ci95"][1] < 0.0) if r["estimable"] else None})
    est = [c for c in cells if c["estimable"]]
    beaten = [c for c in est if c["beaten"]]
    return {"level": level, "own_rung": own_rung, "probe": kind, "rungs_checked": checked,
            "n_cells": len(cells), "n_estimable": len(est), "n_beaten": len(beaten),
            "n_behind_on_point_estimate": sum(c["behind_on_point_estimate"] for c in est),
            "holds": bool(est) and not beaten,
            "beaten_by": [{"rung": c["rung"], "level": c["other_level"],
                           "mean_diff": c["mean_diff"], "p": c["p"]} for c in beaten],
            "cells": cells}


def s9_analysis(data: dict, seeds, drop=()) -> dict:
    """S9's two clauses from the label-recovery ladder, on BOTH probes (D6).

    The second clause is a null -- "the advantage decays to ZERO" -- and D6
    forbids reporting a linear-probe null without the nonlinear probe beside it,
    because a linear probe only lower-bounds mutual information. The linear
    probe is primary (as for S1, clarification 4 of 2026-09-19) and the MLP
    result travels in the same structure and in the clause's detail line.

    The unit is the pretraining seed and pairs are by seed index (2.1, 2.2); the
    primary test at every rung is the paired t on n-1 degrees of freedom with the
    exact sign-flip p and its floor beside it (2.4).
    """
    rcells = index_recovery(data["rows"], drop)
    own = data["own_rung"]
    levels = [lv for lv in LEVELS if lv in own]
    rungs = [r for r in RUNGS if any(k[0] == r for k in rcells)]
    pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]]
    out = {"endpoint": {"field": "balanced accuracy", "lower_is_better": False,
                        "difference": "finer model − coarser model, by seed index",
                        "probe_primary": "linear", "probe_companion": "mlp (D6)"},
           "rungs_fine_to_coarse": list(rungs), "levels_fine_to_coarse": levels,
           "own_rung": {str(lv): own[lv] for lv in levels}, "seeds_used": list(seeds),
           "missing_cells": missing_recovery_cells(rcells, levels, rungs, seeds),
           "skipped_cells": data["skipped_cells"],
           "pairs": {}, "wins": {}}
    for fine, coarse in pairs:
        block = {}
        for kind in PROBES:
            table = recovery_pair_table(rcells, fine, coarse, kind, seeds, rungs)
            block[kind] = {"fine": fine, "coarse": coarse, "rungs": table,
                           "crossover": crossover(table, own[coarse])}
        out["pairs"][f"{fine}_vs_{coarse}"] = block
    for lv in levels:
        out["wins"][str(lv)] = {
            kind: wins_at_and_below_own(rcells, lv, own[lv], [x for x in levels if x != lv],
                                        kind, seeds, rungs) for kind in PROBES}

    def by_probe(kind, field):
        return [out["pairs"][k][kind]["crossover"][field] for k in out["pairs"]]

    wins = {k: [out["wins"][str(lv)][k] for lv in levels] for k in PROBES}
    n_pairs = len(pairs)
    if not any(w["n_estimable"] for w in wins["linear"]):
        c_1 = clause(1, S9["clauses"][0], "paired t at every rung at or below the model's own",
                     "not run", detail="no rung has 2 complete seed pairs")
    else:
        hold = sum(w["holds"] for w in wins["linear"])
        c_1 = clause(1, S9["clauses"][0],
                     "paired t on n−1 df at every rung at or below the model's own vocabulary; "
                     "a model fails only where another is ahead by more than the 95% interval",
                     "confirmed" if hold == len(levels) else "not confirmed",
                     detail=f"{hold} of {len(levels)} models are never beaten there (linear "
                            f"probe); nonlinear probe: {sum(w['holds'] for w in wins['mlp'])} "
                            f"of {len(levels)}")
    if not any(by_probe("linear", "n_rungs_estimable")):
        c_2 = clause(2, S9["clauses"][1], "crossover of the per-rung paired differences",
                     "not run", detail="no rung has 2 complete seed pairs")
    else:
        at = sum(bool(x) for x in by_probe("linear", "crossover_at_coarser_own_rung"))
        inside = sum(bool(x) for x in by_probe("linear", "coarser_own_rung_in_bracket"))
        at_mlp = sum(bool(x) for x in by_probe("mlp", "crossover_at_coarser_own_rung"))
        c_2 = clause(2, S9["clauses"][1],
                     "crossover of the per-rung paired differences, with its bracket",
                     "confirmed" if at == n_pairs else "not confirmed",
                     detail=f"{at} of {n_pairs} model pairs cross over exactly at the coarser "
                            f"model's own rung and {inside} of {n_pairs} have that rung inside "
                            f"the crossover bracket (linear probe); nonlinear probe: {at_mlp} "
                            f"of {n_pairs}")
    return {**out, **compose(S9["prediction"], [c_1, c_2])}


# ---------------------------------------------------------------- report

def _p(x) -> str:
    return f"{x:.3g}"


def format_contrast(r: dict) -> str:
    """One pairwise line: t, df, p and the sign-flip p with its floor. No other unit exists."""
    head = f"    {r['coarse']:>3d} − {r['fine']:<3d} "
    if not r["estimable"]:
        return f"{head} not estimable ({r['reason']}; n={r['n_pairs']})"
    sf = r["sign_flip"]
    return (f"{head} diff={r['mean_diff']:+.4f}  t={r['t']:+.2f} df={r['df']} p={_p(r['p'])} "
            f"CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
            f"sign-flip p={_p(sf['p'])} (floor {_p(sf['floor'])})  n={r['n_pairs']}"
            f"{'  Holm-reject' if r.get('holm_reject') else ''}"
            f"{'  [BOUND: a cell reached AUC=1]' if r['is_bound'] else ''}")


def format_did(label: str, r: dict) -> str:
    """One difference-in-differences (or one-level gain) line, in the same unit as the rest."""
    if not r["estimable"]:
        return f"    {label:<22} not estimable ({r['reason']}; n={r['n_pairs']})"
    sf = r["sign_flip"]
    return (f"    {label:<22} diff={r['mean_diff']:+.4f}  t={r['t']:+.2f} df={r['df']} "
            f"p={_p(r['p'])} CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
            f"sign-flip p={_p(sf['p'])} (floor {_p(sf['floor'])})  n={r['n_pairs']}"
            f"{'  [BOUND: a cell reached AUC=1]' if r['is_bound'] else ''}")


def format_c5(c5: dict) -> list[str]:
    """C5 printed with its definition above it, because a DiD sign is easy to read backwards."""
    out = ["  C5  mass output x granularity, " + c5["task"],
           f"      prediction: {c5['prediction']}",
           f"      difference-in-differences = {c5['did_definition']}",
           f"      endpoint {c5['endpoint']}, LOWER IS BETTER, so the written "
           f"expectation is a {c5['expected_sign_if_written_expectation_holds']} sign"]
    for kind in PROBES:
        blk = c5["confirmatory"]["probes"].get(kind)
        if not blk:
            out.append(f"    {kind}: no cells")
            continue
        out.append(f"    {kind}:")
        out.append(format_did("DiD (162 − 17)", blk["did"]))
        for lv in MASS_LEVELS:
            out.append(format_did(f"gain at {lv} classes", blk["gain_by_level"][str(lv)]))
    if c5["exploratory"]:
        out.append("    EXPLORATORY -- the same difference-in-differences on the other tasks the")
        out.append("    2x2 measured, linear probe. No verdict, and no multiplicity family:")
        for blk in c5["exploratory"]:
            lin = blk["probes"].get("linear", {}).get("did", {})
            out.append(format_did(blk["task"], lin) if lin else f"    {blk['task']}: no cells")
    return out


def format_trend(name: str, r: dict, family_size: int) -> list[str]:
    out = [f"  {name}: {r['task']}, {r['probe']} probe — alternative: {r['alternative']}"]
    if r["blocks_dropped_incomplete"]:
        out.append("    seed blocks dropped whole (missing levels): " + "; ".join(
            f"seed {s}: {lv}" for s, lv in r["blocks_dropped_incomplete"].items()))
    if not r["run"]:
        return out + [f"    {r['reason']} (complete blocks: {r['blocks_used']})"]
    out.append(f"    max-T={r['stat']:.3f}  global p={_p(r['p'])}  ({r['method']}, "
               f"{r['n_arrangements']:,} arrangements, {r['n_blocks']} seed blocks; "
               f"smallest attainable p {_p(r['p_min'])}; single end-step bound "
               f"{_p(r['end_step_p_bound'])})")
    if r["end_step_p_bound"] > ALPHA / family_size:
        out.append(f"    UNDERPOWERED, not negative: with {r['n_blocks']} seed blocks a single "
                   f"end-step cannot reach the family threshold {_p(ALPHA / family_size)}")
    a, b = r["argmax_step"]
    pred = "" if r["predicted_step"] is None else (
        f" — predicted step {r['predicted_step'][0]} | {r['predicted_step'][1]}: "
        f"{'YES' if r['argmax_is_predicted_step'] else 'NO'}")
    out.append(f"    arg-max contrast: {a} | {b}{pred}")
    out.append("    per-contrast adjusted p (localisation only, not separate tests):")
    for c in r["contrasts_localisation_only"]:
        out.append(f"      {str(c['lower']):>16s} | {str(c['upper']):<16s} "
                   f"diff={c['difference']:+.4f} t={c['t']:+.2f} p_adj={_p(c['p_adj'])}")
    out.append("    seed SD per level, blocks used: " + "  ".join(
        f"{lv}: {sd:.4f}" for lv, sd in zip(LEVELS, r["seed_sd_per_level"])))
    iso = r["isotonic"]
    out.append(f"    isotonic companion: E²={iso['stat']:.3f} p={_p(iso['p'])} "
               f"pooled levels {iso['pooled']}")
    if r["n_censored_cells"]:
        out.append(f"    {r['n_censored_cells']} cell(s) reached AUC=1: their endpoint is a bound")
    return out


def format_equivalence_set(e: dict, indent: str = "      ") -> list[str]:
    """Every pair inside a predicted-equal set, then the set's verdict."""
    if not e["run"]:
        return [f"{indent}{e['reason']}"]
    out = [f"{indent}TOST bound ±ln(1.1) = ±{e['target_bound']:.5f}; all "
           f"{e['n_pairs_total']} pairs inside "
           f"{{{', '.join(str(x) for x in e['levels'])}}} must pass"]
    for r in e["pairs"]:
        tail = ("equivalent at ±10%" if r["equivalent_at_target"] else
                f"NOT equivalent at ±10%; equivalent within ±{r['smallest_bound_passed']:.4f} "
                f"(a factor {LOG_BASE ** r['smallest_bound_passed']:.3f} in 1−AUC)")
        out.append(f"{indent}  {r['coarse']:>3d} − {r['fine']:<3d} diff={r['mean_diff']:+.4f}  "
                   f"90% CI [{r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}]  TOST p={_p(r['p'])}  "
                   f"n={r['n_pairs']}  {tail}"
                   f"{'  [BOUND: a cell reached AUC=1]' if r['is_bound'] else ''}")
    out.append(f"{indent}set verdict (intersection-union p={_p(e['p'])}): {e['verdict']}")
    return out


def format_clauses(name: str, block: dict) -> list[str]:
    """The clause-by-clause verdict, and the composite sentence above it."""
    out = [f"  {name} is {block['n_clauses']} clauses — {block['composite_verdict'].upper()} "
           f"({block['overall']})"]
    for c in block["clauses"]:
        out.append(f"    clause {c['n']}: \"{c['text']}\" — {c['verdict'].upper()}")
        out.append(f"      {c['test']}" + ("" if c["p"] is None else f", p={_p(c['p'])}"))
        if c["detail"]:
            out.append(f"      {c['detail']}")
    return out


def format_recovery_row(r: dict) -> str:
    """One rung of one model pair. Same units as the ladder: t, df, p, no sigma."""
    head = f"      {r['rung']:<8s}"
    if not r["estimable"]:
        return f"{head} not estimable ({r['reason']}; n={r['n_pairs']})"
    sf = r["sign_flip"]
    return (f"{head} diff={r['mean_diff']:+.4f}  t={r['t']:+.2f} df={r['df']} p={_p(r['p'])}  "
            f"CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
            f"sign-flip p={_p(sf['p'])} (floor {_p(sf['floor'])})  n={r['n_pairs']}  "
            f"{r['advantage']}{'  Holm-reject' if r.get('holm_reject') else ''}")


def format_s9(s9: dict) -> list[str]:
    """The S9 block: the clauses, then every model pair's ladder and crossover."""
    out = ["  endpoint: balanced accuracy, HIGHER is better; differences are FINER model − "
           "COARSER model,",
           "  paired by seed index: positive = the finer model recovers that rung better",
           "  rungs fine→coarse: " + " ".join(s9["rungs_fine_to_coarse"]),
           "  each model's own vocabulary: " + "  ".join(
               f"{lv}-class: {s9['own_rung'][str(lv)]}" for lv in s9["levels_fine_to_coarse"])]
    if s9["missing_cells"]:
        for key, cells in s9["missing_cells"].items():
            out.append(f"  MISSING cells, {key}: " + "; ".join(f"{lv}-class seed {s}"
                                                               for lv, s in cells))
    for sk in s9["skipped_cells"]:
        out.append(f"  cell {sk['arm']} / {sk['rung']} was skipped by the probe: {sk['reason']}")
    out += format_clauses("S9", s9)
    for key, per_probe in s9["pairs"].items():
        for kind in PROBES:
            b = per_probe[kind]
            c = b["crossover"]
            role = "" if kind == "linear" else " (beside the linear probe; D6)"
            lo, hi = c["crossover_bracket"]
            out.append(f"    {b['fine']}-class over {b['coarse']}-class, {kind} probe{role}")
            out += [format_recovery_row(r) for r in b["rungs"]]
            out.append(f"      crossover at {lo}; bracket [{lo}, {hi}]; point estimate reaches "
                       f"zero at {c['sign_change_rung']}")
            out.append(f"      the {b['coarse']}-class model's own rung is "
                       f"{c['coarser_own_rung']} — crossover there: "
                       f"{'YES' if c['crossover_at_coarser_own_rung'] else 'NO'}"
                       f" (inside the bracket: "
                       f"{'yes' if c['coarser_own_rung_in_bracket'] else 'no'})")
    out.append("    wins at and below each model's own vocabulary:")
    for lv, per_probe in s9["wins"].items():
        for kind in PROBES:
            w = per_probe[kind]
            beaten = "; ".join(f"{b['rung']} by the {b['level']}-class model "
                               f"(diff={b['mean_diff']:+.4f}, p={_p(b['p'])})"
                               for b in w["beaten_by"])
            out.append(f"      {lv}-class, {kind}: own rung {w['own_rung']}, "
                       f"{len(w['rungs_checked'])} rungs at or below it, {w['n_estimable']} of "
                       f"{w['n_cells']} cells estimable — beaten in {w['n_beaten']} "
                       f"(behind on the point estimate in {w['n_behind_on_point_estimate']})"
                       + (f": {beaten}" if beaten else ""))
    return out


def format_holm(rows: list[dict]) -> list[str]:
    out = []
    for h in rows:
        if h["status"] == "pending":
            out.append(f"    {h['test']:28s} pending")
            continue
        verdict = ("rejected whatever the pending tests give" if h["reject_whatever_pending"]
                   else "may be rejected, depending on the pending tests" if h["reject_possible"]
                   else "not rejected")
        out.append(f"    {h['test']:28s} raw p={_p(h['p_raw'])}  must beat "
                   f"{_p(h['threshold_if_smallest'])} if smallest of {h['family_size']}: {verdict}")
    return out


def _refuse_overwrite(out_file: pathlib.Path) -> pathlib.Path:
    if out_file.exists():
        raise SystemExit(f"FATAL: {out_file} exists; an earlier look at the data is a record. "
                         f"Refusing to overwrite it -- give a new --out.")
    return out_file


def run_ladder(a, argv=None) -> int:
    out_file = _refuse_overwrite(pathlib.Path(a.out) / "seed_level_results.json")

    data = load_ladder(a.inputs)
    rows = data["rows"]
    for r in rows:
        r["dropped_pair"] = r["seed"] in a.drop_pairs
    all_seeds = sorted(set(EXPECTED_SEEDS) | {r["seed"] for r in rows})
    seeds = [s for s in all_seeds if s not in a.drop_pairs]
    cells = index_cells(rows, a.drop_pairs)
    tasks = sorted({r["task"] for r in rows}, key=lambda t: (t in CONTROL_TASKS, t))
    missing = missing_cells(rows, seeds)

    res = {"provenance": {
               "inputs": data["files"], "script_sha256": _sha(__file__),
               "prespec_sha256": _sha(PRESPEC) if PRESPEC.exists() else None,
               "stats_modules_sha256": {m: _sha(REPO / m) for m in STATS_MODULES},
               "row_alignment_sha256": data["row_alignment_sha256"],
               "n_jets_total": data["n_jets_total"],
               "arm_checkpoints": data["arm_checkpoints"], "argv": list(argv or sys.argv[1:])},
           "endpoint": {"field": ENDPOINT, "log_base": "e (natural log, probe.py np.log)",
                        "lower_is_better": True, "difference": "coarser − finer, by seed index"},
           "levels_fine_to_coarse": list(LEVELS), "seeds_used": seeds,
           "dropped_pairs": {"seeds": a.drop_pairs, "reason": a.drop_reason},
           "missing_cells": missing, "skipped_tasks": data["skipped_tasks"], "table": rows}

    print("SEED-LEVEL INFERENCE, four-granularity frozen probes (PRESPEC_2026-09 §2–3)")
    print(f"  inputs: {len(data['files'])} file(s); row alignment "
          f"{data['row_alignment_sha256'][:16]}; {data['n_jets_total']:,} jets")
    print(f"  endpoint: {ENDPOINT} = NATURAL log of (1 − AUC), lower is better; levels "
          f"fine→coarse {list(LEVELS)}; differences are coarser − finer, paired by seed index")
    print(f"  seed indices in the design: {seeds}")
    if a.drop_pairs:
        print(f"  DROPPED seed indices {a.drop_pairs} from every contrast. Reason: {a.drop_reason}")
    if missing:
        common = len({json.dumps(v) for v in missing.values()}) == 1 and len(missing) == 2 * len(tasks)
        for key, cellsm in ([("every task, both probes", next(iter(missing.values())))]
                            if common else missing.items()):
            by_seed = {s: [lv for lv, s2 in cellsm if s2 == s] for s in sorted({c[1] for c in cellsm})}
            print(f"  MISSING cells, {key}: " + "; ".join(f"seed {s}: levels {lv}"
                                                          for s, lv in by_seed.items()))
    for s in data["skipped_tasks"]:
        print(f"  task {s['task']} was SKIPPED by the probe in {s['file']}")

    # ---- 1. what the design can detect. No level mean and no p-value here. ----
    print("\n== 1. MINIMUM DETECTABLE EFFECT AND SEED SPREAD (printed before any contrast) ==")
    print(f"  MDE at {POWER:.0%} power, two-sided {ALPHA:.0%}, from the standard deviation of the "
          f"17-class − 162-class paired differences;\n  already-seen tasks only (PRESPEC §1); "
          f"seed term only, no test-resampling floor")
    res["mde"] = [mde_row(cells, t, k, seeds) for t in SEEN_TASKS if t in tasks for k in PROBES]
    for m in res["mde"]:
        print(f"    {m['task']:18s} {m['probe']:6s} " + (
            f"n={m['n_pairs']} pairs  SD(paired diff)={m['sd_paired_diff']:.4f}  "
            f"×{m['multiplier']:.2f}  MDE={m['mde']:.4f} "
            f"(a factor {m['mde_as_ratio_of_1m_auc']:.3f} in 1−AUC)" if m["estimable"]
            else f"not estimable ({m['reason']}; n={m['n_pairs']})"))
    res["levels"] = {t: {k: level_summary(cells, t, k, seeds) for k in PROBES} for t in tasks}
    print(f"  per-level standard deviation of {ENDPOINT} over pretraining seeds (n seeds):")
    for t in tasks:
        for k in PROBES:
            print(f"    {t:18s} {k:6s} " + "  ".join(
                f"{r['level']}: " + ("n/a" if r["seed_sd"] is None else f"{r['seed_sd']:.4f}")
                + f" ({r['n_seeds']})" for r in res["levels"][t][k]))

    # ---- 2. contrasts ----
    print("\n== 2. CONTRASTS ==")
    print("  per-level summary (rejection is the median over seeds; 'bound' counts seeds at the cap):")
    for t in tasks:
        tag = "  [control task: tabulated only; C4 needs the random-label control]" \
            if t in CONTROL_TASKS else ""
        for k in PROBES:
            print(f"    {t} / {k}{tag}")
            for r in res["levels"][t][k]:
                if not r["n_seeds"]:
                    print(f"      {r['level']:>3d}-class  no seeds")
                    continue
                print(f"      {r['level']:>3d}-class  n={r['n_seeds']}  mean {ENDPOINT}={r['mean']:+.4f}"
                      f"  mean AUC={r['mean_auc']:.5f}  1/eps_B@{r['rejection_eps_s']:.0%}="
                      f"{r['rejection_median']:.1f} [{r['rejection_range'][0]:.1f}, "
                      f"{r['rejection_range'][1]:.1f}] bound in {r['n_rejection_bound']}"
                      f"  censored in {r['n_censored']}")
                for eps, pt in r["rejection_points"].items():
                    mark = " <- HEADLINE" if pt["is_headline"] else ""
                    note = (f"  CENSORED in {pt['n_bound']}/{pt['n_seeds']}"
                            if pt["n_bound"] else "")
                    print(f"        eps_s={eps}  1/eps_B={pt['median']:9.1f} "
                          f"[{pt['range'][0]:.1f}, {pt['range'][1]:.1f}]  "
                          f"background jets left {pt['mean_n_bkg_pass']:5.1f}"
                          f"{note}{mark}")

    print("\n  CONFIRMATORY")
    c1 = trend_test(cells, C1["task"], "linear", seeds, C1["predicted_step"])
    # C5 only exists once the mass-output 2x2 has been probed. Its p joins the
    # confirmatory family in place of C5's `None`; without --mass it stays
    # pending exactly as C2 and C3 do, and the family size never changes.
    c5 = None
    if a.mass:
        mdata = load_ladder(a.mass, parse=parse_mass_arm)
        if mdata["row_alignment_sha256"] != data["row_alignment_sha256"]:
            raise SystemExit(
                "FATAL: the mass-output 2x2 was scored on a different set of test jets "
                f"than the ladder ({mdata['row_alignment_sha256'][:16]} vs "
                f"{data['row_alignment_sha256'][:16]}). C5 and C1 would not be the same "
                "measurement and must not share a multiplicity family.")
        mcells = index_cells(mdata["rows"], a.drop_pairs)
        c5 = c5_analysis(mcells, seeds, sorted({r["task"] for r in mdata["rows"]}))
        c5["provenance"] = {"inputs": mdata["files"],
                            "row_alignment_sha256": mdata["row_alignment_sha256"],
                            "n_jets_total": mdata["n_jets_total"],
                            "arm_checkpoints": mdata["arm_checkpoints"]}
    computed = {"C1": c1.get("p"), "C5": None if c5 is None else c5["p"]}
    fam = holm_family([(c, computed.get(c)) for c in CONFIRMATORY])
    # C1's third clause is a predicted null and needs 2.5's equivalence test; the
    # trend test above answers only the first two. The three verdicts and the
    # composite live inside res["confirmatory"]["C1"] beside the trend result,
    # so nothing that already reads that block moves.
    c1_equal = equivalence_set(cells, C1["task"], "linear", C1["predicted_equal"], seeds)
    c1.update(c1_clauses(c1, c1_equal, next(h for h in fam if h["test"] == "C1"),
                         len(CONFIRMATORY)))
    res["confirmatory"] = {"C1": c1, "holm_family": fam}
    if c5 is not None:
        res["confirmatory"]["C5"] = c5
    print("\n".join(format_trend("C1", c1, len(CONFIRMATORY))))
    print("\n".join(format_clauses("C1", c1)))
    print("\n".join(format_equivalence_set(c1_equal)))
    if c5 is not None:
        print("\n".join(format_c5(c5)))
    print(f"  Holm over the full confirmatory family of {len(CONFIRMATORY)} at {ALPHA:.0%}:")
    print("\n".join(format_holm(fam)))

    print("\n  SECONDARY")
    s1 = trend_test(cells, S1["task"], "linear", seeds, S1["predicted_step"])
    s2 = {t: {k: equivalence(cells, t, k, seeds) for k in PROBES} for t in S2_TASKS}
    sec = holm_family([("S1 " + S1["task"], s1.get("p"))]
                      + [(f"S2 {t}", s2[t]["linear"].get("p")) for t in S2_TASKS])
    print("\n".join(format_trend("S1", s1, len(sec))))
    for t in S2_TASKS:
        for k in PROBES:
            e = s2[t][k]
            role = "" if k == "linear" else " (beside the linear probe; not in the Holm table)"
            print(f"  S2: {t}, {k} probe{role} — predicted equivalent, TOST bound "
                  f"±ln(1.1) = ±{e['target_bound']:.5f} in natural-log units")
            if not e["run"]:
                print(f"    {e['reason']}")
                continue
            top, comp = e["largest_pair"], e["all_pairs_companion"]
            print(f"    largest pairwise difference {top['coarse']} − {top['fine']}: "
                  f"{top['mean_diff']:+.4f}  90% CI [{top['ci90'][0]:+.4f}, {top['ci90'][1]:+.4f}]"
                  f"  TOST p={_p(top['p'])}  n={top['n_pairs']}"
                  f"{'  [BOUND: a cell reached AUC=1]' if top['is_bound'] else ''}")
            print(f"    {e['verdict']}")
            print(f"    companion over all {e['n_pairs_estimable']} pairs: max TOST p="
                  f"{_p(comp['max_p'])}; all equivalent within ±{comp['smallest_bound_passed_by_all']:.4f}")
    pw = {t: {k: pairwise_table(cells, t, k, seeds) for k in PROBES}
          for t in tasks if t not in CONTROL_TASKS}
    s6 = {t: sign_agreement(pw[t]["linear"], pw[t]["mlp"]) for t in pw}
    for t, g in s6.items():
        bad = [f"{p['coarse']}−{p['fine']}" for p in g["pairs"] if not p["agree"]]
        print(f"  S6: {t}: " + (f"MLP orders {g['n_agree']} of {g['n_pairs']} level pairs as the "
                                f"linear probe does" + (f"; disagree: {bad}" if bad else "")
                                if g["n_pairs"] else "no estimable level pair — not run"))
    print("  Holm within the secondary table (S1, S2 on the linear probe):")
    print("\n".join(format_holm(sec)))
    res["secondary"] = {"S1": s1, "S2": s2, "S6": s6, "holm_family": sec}

    print("\n  PAIRWISE, coarser − finer by seed index — EXPLORATORY (not listed in PRESPEC §3);"
          "\n  primary = paired t on n−1 df; Holm within each table of six")
    for t in pw:
        for k in PROBES:
            if not any(r["estimable"] for r in pw[t][k]):
                print(f"    {t} / {k}: no pair of levels has 2 complete seed pairs — not run")
                continue
            print(f"    {t} / {k}")
            print("\n".join(format_contrast(r) for r in pw[t][k]))
    res["pairwise_exploratory"] = pw

    print("\n  C4 is pending: the random-label control has no features yet. c4_sign_pattern() is "
          "implemented\n  and tested on a synthetic fixture; with one run per draw it is descriptive.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_file}")
    return 0


def run_s9(a, argv=None) -> int:
    """S9 on the label-recovery ladder. Its own inputs, its own output file."""
    out_file = _refuse_overwrite(pathlib.Path(a.out) / "s9_label_recovery.json")

    data = load_recovery(a.label_recovery)
    all_seeds = sorted(set(EXPECTED_SEEDS) | {r["seed"] for r in data["rows"]})
    seeds = [s for s in all_seeds if s not in a.drop_pairs]
    s9 = s9_analysis(data, seeds, a.drop_pairs)
    res = {"provenance": {
               "inputs": data["files"], "script_sha256": _sha(__file__),
               "prespec_sha256": _sha(PRESPEC) if PRESPEC.exists() else None,
               "stats_modules_sha256": {m: _sha(REPO / m) for m in STATS_MODULES},
               "row_alignment_sha256": data["row_alignment_sha256"],
               "n_used": data["n_used"], "n_train": data["n_train"], "n_test": data["n_test"],
               "chance_sigma": data["chance_sigma"], "argv": list(argv or sys.argv[1:])},
           "dropped_pairs": {"seeds": a.drop_pairs, "reason": a.drop_reason},
           "secondary": {"S9": s9}, "table": data["rows"]}

    print("\n== 3. S9: LABEL RECOVERY ACROSS THE CONTRACTION TREE "
          "(PRESPEC_2026-09 §3, probe = both, D6) ==")
    print(f"  inputs: {len(data['files'])} file(s); row alignment "
          f"{data['row_alignment_sha256'][:16]}; {data['n_test']:,} test jets per cell")
    if a.drop_pairs:
        print(f"  DROPPED seed indices {a.drop_pairs} from every contrast. "
              f"Reason: {a.drop_reason}")
    print("\n".join(format_s9(s9)))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_file}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="*",
                    help="a directory holding s*/probe_results.json, or the files")
    ap.add_argument("--out", required=True, help="directory for seed_level_results.json")
    ap.add_argument("--drop-pairs", nargs="*", type=int, default=[],
                    help="seed indices to exclude from every contrast (hardware-mismatched pairs)")
    ap.add_argument("--drop-reason", default=None, help="required with --drop-pairs; recorded")
    ap.add_argument("--label-recovery", nargs="+", default=None, metavar="PATH",
                    help="S9: a directory holding s*/label_recovery.json, or the files. "
                         "Written to s9_label_recovery.json in --out; the ladder inputs are "
                         "optional when this is given")
    ap.add_argument("--mass", nargs="+", default=None, metavar="PATH",
                    help="C5: a directory holding s*/probe_results.json for the granularity x "
                         "mass-output 2x2, or the files. Reported inside the ladder's "
                         "confirmatory block; without it C5 stays pending")
    a = ap.parse_args(argv)
    if a.mass and not a.inputs:
        ap.error("--mass reports C5 inside the ladder's confirmatory family, so it needs the "
                 "ladder inputs too; C5 and C1 are corrected together or not at all")
    if a.drop_pairs and not a.drop_reason:
        raise SystemExit("FATAL: --drop-pairs needs --drop-reason; a dropped pair is reported "
                         "with why it was dropped (PRESPEC 2.2)")
    if not a.inputs and not a.label_recovery:
        ap.error("give the probe_results.json inputs, --label-recovery, or both")
    rc = run_ladder(a, argv) if a.inputs else 0
    return rc or (run_s9(a, argv) if a.label_recovery else 0)


if __name__ == "__main__":
    raise SystemExit(main())
