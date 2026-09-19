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
ENDPOINT  `log1m_auc` exactly as probe.py stores it: the NATURAL log of 1 - AUC,
          floored at the sample's resolution (`log1m_auc_censored` marks a
          bound). LOWER is better.
UNIT      the pretraining seed. Contrasts are paired by seed index (2.1, 2.2).

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
PAIRS = [(a, b) for i, a in enumerate(LEVELS) for b in LEVELS[i + 1:]]   # (finer, coarser)

SEEN_TASKS = ("bvc_resonant", "bvc_qcd", "retained_topology", "ee_vs_mm")   # PRESPEC 1
CONTROL_TASKS = ("bvc_4prong", "visible_content")                          # C4's two class pairs
C1 = {"task": "bvc_resonant", "predicted_step": ([188, 162, 43], [17])}
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


def load_ladder(src) -> dict:
    """Tidy table: one row per (task, probe kind, level, seed index).

    `src` is a directory holding s*/probe_results.json, or a list of files.
    Refuses if the files disagree on `row_alignment_sha256` or `n_jets_total`
    (the models were then not scored on the same jets), if an arm name does not
    parse, or if a (level, seed) cell appears twice. Missing cells are not an
    error here; `missing_cells` reports them.
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
                level, seed = parse_arm(arm)
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
                    "n_censored": sum(r["censored"] for r in rs)})
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


def equivalence(cells, task, kind, seeds) -> dict:
    """S2: TOST on the largest pairwise paired difference, bound +-ln(1.1).

    The pair is chosen by rule -- largest |mean paired difference| among the six
    -- and `smallest_bound_passed` is the half-width at which its (1-2a) interval
    just fits. The maximum over all six pairs is reported beside it as a
    companion, because the pair with the largest mean is not necessarily the
    pair with the widest interval. The wording is PRESPEC 2.5's, never "no effect".
    """
    bound = tost_bound()
    rows = []
    for a, b in PAIRS:
        pd = paired_diffs(cells, task, kind, a, b, seeds)
        if len(pd["d"]) < 2 or np.ptp(pd["d"]) == 0:    # zero variance: see pair_contrast
            continue
        r = tost(pd["d"], bound=bound, alpha=ALPHA)
        rows.append({"fine": a, "coarse": b, "n_pairs": len(pd["d"]), "is_bound": pd["is_bound"],
                     "mean_diff": float(np.mean(pd["d"])), "p": r["p"], "ci90": list(r["ci_1m2a"]),
                     "equivalent_at_target": r["equivalent"],
                     "smallest_bound_passed": float(max(abs(x) for x in r["ci_1m2a"]))})
    out = {"task": task, "probe": kind, "target_bound": bound, "log_base": "e",
           "n_pairs_estimable": len(rows), "n_pairs_total": len(PAIRS)}
    if not rows:
        return {**out, "run": False, "reason": "no level pair with 2 complete seed pairs and "
                                               "non-identical differences — not run"}
    top = max(rows, key=lambda r: abs(r["mean_diff"]))
    x = top["smallest_bound_passed"]
    verdict = (f"equivalent within ±{bound:.4f} (±10% in 1−AUC); smallest bound passed ±{x:.4f}"
               if top["equivalent_at_target"] else
               f"inconclusive at ±10%; equivalent within ±{x:.4f} "
               f"(a factor {LOG_BASE ** x:.3f} in 1−AUC)")
    return {**out, "run": True, "largest_pair": top, "p": top["p"], "verdict": verdict,
            "all_pairs_companion": {"max_p": max(r["p"] for r in rows),
                                    "smallest_bound_passed_by_all": max(
                                        r["smallest_bound_passed"] for r in rows)},
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="a directory holding s*/probe_results.json, or the files")
    ap.add_argument("--out", required=True, help="directory for seed_level_results.json")
    ap.add_argument("--drop-pairs", nargs="*", type=int, default=[],
                    help="seed indices to exclude from every contrast (hardware-mismatched pairs)")
    ap.add_argument("--drop-reason", default=None, help="required with --drop-pairs; recorded")
    a = ap.parse_args(argv)
    if a.drop_pairs and not a.drop_reason:
        raise SystemExit("FATAL: --drop-pairs needs --drop-reason; a dropped pair is reported "
                         "with why it was dropped (PRESPEC 2.2)")

    out_file = pathlib.Path(a.out) / "seed_level_results.json"
    if out_file.exists():
        raise SystemExit(f"FATAL: {out_file} exists; an earlier look at the data is a record. "
                         f"Refusing to overwrite it -- give a new --out.")

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

    print("\n  CONFIRMATORY")
    c1 = trend_test(cells, C1["task"], "linear", seeds, C1["predicted_step"])
    fam = holm_family([("C1", c1.get("p"))] + [(c, None) for c in CONFIRMATORY[1:]])
    res["confirmatory"] = {"C1": c1, "holm_family": fam}
    print("\n".join(format_trend("C1", c1, len(CONFIRMATORY))))
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


if __name__ == "__main__":
    raise SystemExit(main())
