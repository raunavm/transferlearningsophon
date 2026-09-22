#!/usr/bin/env python3
"""The INFERENCE layer for the fine-tuning legs: gaps with honest uncertainty.

WHY THIS EXISTS. leg1_metrics.py and leg2_metrics.py produce the MEASUREMENTS --
per-cell accuracy and AUC. Nothing in this repo produced the INFERENCE, so the
gaps and their uncertainties were formed by hand into docs/PRD_PLAN.md and
carried no artifact, no script and no test. An audit (2026-09-12) found three
defects in that hand layer, all of which this script makes structurally
impossible to repeat:

  1. ROUNDING. The leg-2 significance column reproduced only from per-arm means
     first rounded to 4 decimals: published 44.5 against 39.1 at full precision
     at N=1e6, and 27.7 against 29.4 at N=1e5. Leg 1's column, one line above in
     the same table, was full precision -- so the table mixed two denominators.
     THE FIX IS STRUCTURAL: this script reads `cells` and recomputes every mean
     itself. It never reads the `summary` block, so a rounded intermediate
     cannot enter. read_cells() raises if `summary` is passed to it.

  2. A DENOMINATOR THAT WAS NOT WHAT IT WAS CALLED. docs/PRD_PLAN.md named the
     unit "the R16_Q1 pretraining-seed SD". It is the SD of three ARM MEANS,
     which estimates sigma^2_pre + sigma^2_ft/n_ft, not sigma^2_pre. A one-way
     random-effects decomposition of the committed cells shows the pretraining
     component is not merely conflated but ABSENT: F = 0.41 / 1.64 / 0.79 across
     the three leg-2 cells, and the estimated sigma^2_pre is NEGATIVE at two of
     them. In every cell the unit used was SMALLER than the fine-tuning noise on
     a single run of the quantity being compared. This script reports the
     decomposition beside every gap and refuses to name anything a
     pretraining-seed SD unless F > 1 and the estimate is positive.

  3. 2-df RATIOS PRINTED WITH A GREEK SIGMA. The quoted values were gap divided
     by an SD estimated from 3 seeds, then read as Gaussian z. On 2 df the t
     tails are heavy enough that "44.5 sigma" is p = 5e-4 and the in-domain
     low-N headline "8.7 sigma" is p = 0.013. This script emits t, df and a
     two-sided p, and NEVER emits a bare sigma. `format_row` has no sigma field.

THE DESIGN CONSTRAINT THAT OUTRANKS ALL THREE. The leg cells this script reads
hold ONE 162-class pretraining seed (mtx-l162-s1b) against 3 of the 17-class
model, so at the pretraining-seed level there are C(4,1) = 4 label arrangements
and the smallest attainable one-sided permutation p is 1/4 = 0.25. NO pretraining-seed-level claim of any
strength is expressible from this design, whatever denominator is chosen. That
is a property of the design, not of the analysis, and it does not go away by
picking a better test -- it goes away when L162 seeds 2-5 finish. Every row this
script emits therefore carries an explicit `inference_level`:

    "fine-tuning-seed"   -- valid now. Answers: for THESE particular pretrained
                            checkpoints, is the gap larger than fine-tuning
                            noise? This is a real and publishable question; it
                            is just narrower than the one the paper asks.
    "pretraining-seed"   -- the question the paper actually asks. Requires >= 2
                            pretraining seeds on BOTH sides. Emitted as
                            `blocked` with the permutation floor until L162's
                            remaining seeds land.

WHAT IS NOT RECOMPUTED HERE. The gap VALUES were all confirmed correct by the
audit's re-derivation; only the uncertainty layer was wrong. This script
recomputes the gaps anyway, from cells, so that the numbers in the paper and the
numbers in the artifact cannot drift apart.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import re
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[2]

# The three contrasts the paper draws, and what each is allowed to claim.
#
# `i1_clean` records whether the contrast varies exactly one thing (invariant
# I1). pretrained_vs_scratch does NOT: scratch fine-tunes at start_lr 5e-4 and
# every pretrained init at 1e-4 (the documented recipe), so the contrast varies
# initialisation AND rate. That is why leg 1 shows scratch BEATING both
# pretrained inits at N=1e4 -- a rate artefact, not a transfer result. The
# granularity contrast uses 1e-4 on both sides and is unaffected.
CONTRASTS = {
    "granularity":        {"a": ["l162-s1b"], "b": ["r16q1-s2", "r16q1-s3", "r16q1-s4"],
                           "i1_clean": True},
    "l162_vs_sophon":     {"a": ["l162-s1b"], "b": ["sophon-public"],
                           "i1_clean": False},   # different pretraining CORPUS, not vocabulary
    "pretrained_vs_scratch": {"a": ["r16q1-s2", "r16q1-s3", "r16q1-s4"], "b": ["scratch"],
                              "i1_clean": False},  # start_lr differs; see above
}

# The grid of the two committed legs. main() does NOT use it: each metrics file
# has its own grid (top ends at N1200000, q/g at N1600000, wave 2 adds N1000),
# so the sizes are read from the file -- see sizes_in().
SIZES = ["N10000", "N100000", "N1000000"]


def sizes_in(cells: dict, arms: list[str]) -> list[str]:
    """Every N<digits> size ANY arm of the contrast has, in numeric order.

    The union, not the intersection: a size one arm lacks then reaches
    arm_seed_values and is fatal there, instead of vanishing from the table.
    Non-size keys (leg 2's `ref`) are not part of any contrast.
    """
    found = {s for arm in arms for s in cells[arm] if re.fullmatch(r"N\d+", s)}
    return sorted(found, key=lambda s: int(s[1:]))


def read_cells(doc: dict) -> dict:
    """Return the per-cell block, refusing the pre-aggregated summary.

    The rounding defect entered through `summary`, whose accuracy_mean is a
    float that a hand computation then re-rounded. Taking `cells` and
    aggregating here is what makes defect 1 unrepeatable, so this function
    treats a summary-shaped input as a hard error rather than a convenience.
    """
    if "cells" not in doc:
        raise SystemExit("FATAL: no `cells` block. leg_stats recomputes every "
                         "mean from raw cells by design; it must not be handed "
                         "a pre-aggregated summary.")
    return doc["cells"]


def arm_seed_values(cells: dict, arm: str, size: str, metric: str) -> list[float]:
    """The fine-tuning-seed values for one arm at one N, full precision."""
    if arm not in cells:
        raise SystemExit(f"FATAL: arm {arm} absent from cells. Present: {sorted(cells)}")
    if size not in cells[arm]:
        raise SystemExit(f"FATAL: arm {arm} has no {size}")
    block = cells[arm][size]
    out = []
    for seed in sorted(block):
        cell = block[seed]
        if metric not in cell:
            raise SystemExit(f"FATAL: {arm}/{size}/{seed} has no {metric!r}. "
                             f"A cell that cannot be read must not be silently "
                             f"dropped from a mean. Keys: {sorted(cell)}")
        out.append(float(cell[metric]))
    return out


def variance_components(groups: list[list[float]]) -> dict:
    """One-way random-effects decomposition over pretraining seeds.

    groups[i] holds the fine-tuning-seed values for pretraining seed i. Returns
    the between/within mean squares, the F ratio, and the implied pretraining
    variance component. A NEGATIVE component is reported as-is and flagged
    rather than clamped to zero: clamping it to zero would hide exactly the
    finding that the pretraining-seed signal is undetectable, which is the
    defect this script was written for.
    """
    k = len(groups)
    if k < 2:
        return {"estimable": False, "reason": f"only {k} pretraining seed(s)"}
    n = len(groups[0])
    if any(len(g) != n for g in groups):
        return {"estimable": False, "reason": "unbalanced fine-tuning-seed counts"}

    means = [st.mean(g) for g in groups]
    ms_between = n * st.variance(means)
    ms_within = st.mean([st.variance(g) for g in groups])
    if ms_within <= 0:
        return {"estimable": False, "reason": "zero within-group variance"}

    f = ms_between / ms_within
    var_pre = (ms_between - ms_within) / n
    return {
        "estimable": True,
        "k_pretraining_seeds": k,
        "n_finetuning_seeds": n,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "sd_of_arm_means": st.stdev(means),
        "pooled_finetuning_sd": math.sqrt(ms_within),
        "F": f,
        "df": [k - 1, k * (n - 1)],
        "var_pretraining_hat": var_pre,
        # The single sentence the paper needs. If this is False, nothing in the
        # cell may be described as a pretraining-seed standard deviation.
        "pretraining_variance_detected": bool(f > 1.0 and var_pre > 0),
    }


def welch(xs: list[float], ys: list[float]) -> dict:
    """Welch's t on two independent samples. Returns t, df and a two-sided p.

    p is computed from the t survival function via the regularised incomplete
    beta, so this module needs no scipy -- the cluster image carries numpy but
    the analysis pods are minimal, and an import that fails at hour 30 of a run
    is worse than forty lines of arithmetic.
    """
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"estimable": False, "reason": f"need >=2 per side, got {nx} and {ny}"}
    mx, my = st.mean(xs), st.mean(ys)
    vx, vy = st.variance(xs), st.variance(ys)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return {"estimable": False, "reason": "zero pooled variance"}
    se = math.sqrt(se2)
    t = (mx - my) / se
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"estimable": True, "t": t, "df": df, "se": se,
            "p_two_sided": _t_sf(abs(t), df) * 2,
            "ci95": _ci(mx - my, se, df)}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz). NR 6.4."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise SystemExit("FATAL: incomplete beta did not converge. Refusing to "
                     "return an unconverged p-value.")


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """Upper-tail P(T > t) for Student's t."""
    return 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))


def _t_ppf95(df: float) -> float:
    """Two-sided 95% critical value, by bisection on the survival function."""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_sf(mid, df) > 0.025:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _ci(diff: float, se: float, df: float) -> list[float]:
    h = _t_ppf95(df) * se
    return [diff - h, diff + h]


def permutation_floor(n_a: int, n_b: int) -> float:
    """Smallest attainable one-sided permutation p at the pretraining-seed level.

    With n_a and n_b pretraining seeds there are C(n_a+n_b, n_a) arrangements,
    and the most extreme one carries p = 1/C. At 1 against 3 this is 0.25: no
    pretraining-seed claim is expressible, however large the gap. Reported so
    the constraint appears in the artifact rather than only in a comment.
    """
    return 1.0 / math.comb(n_a + n_b, n_a)


def contrast(cells: dict, name: str, spec: dict, size: str, metric: str) -> dict:
    a_groups = [arm_seed_values(cells, arm, size, metric) for arm in spec["a"]]
    b_groups = [arm_seed_values(cells, arm, size, metric) for arm in spec["b"]]
    a_flat = [v for g in a_groups for v in g]
    b_flat = [v for g in b_groups for v in g]
    gap = st.mean(a_flat) - st.mean(b_flat)

    n_pre_a, n_pre_b = len(a_groups), len(b_groups)
    pre_level = n_pre_a >= 2 and n_pre_b >= 2

    row = {
        "contrast": name, "size": size, "metric": metric,
        "gap": gap,
        "mean_a": st.mean(a_flat), "mean_b": st.mean(b_flat),
        "arms_a": spec["a"], "arms_b": spec["b"],
        "n_pretraining_seeds": {"a": n_pre_a, "b": n_pre_b},
        "i1_clean": spec["i1_clean"],
        # The honest, currently-valid test: pool fine-tuning seeds and ask
        # whether the gap exceeds fine-tuning noise for THESE checkpoints.
        "finetuning_seed_test": welch(a_flat, b_flat),
        "variance_components": {
            "a": variance_components(a_groups),
            "b": variance_components(b_groups),
        },
    }

    if pre_level:
        row["inference_level"] = "pretraining-seed"
        row["pretraining_seed_test"] = welch([st.mean(g) for g in a_groups],
                                             [st.mean(g) for g in b_groups])
    else:
        row["inference_level"] = "fine-tuning-seed"
        row["pretraining_seed_test"] = {
            "estimable": False,
            "reason": (f"{n_pre_a} pretraining seed(s) on side A and {n_pre_b} "
                       f"on side B; >=2 needed on both"),
            "permutation_floor_one_sided": permutation_floor(n_pre_a, n_pre_b),
        }
    return row


def format_row(r: dict) -> str:
    """One human-readable line. There is deliberately NO sigma field.

    The audit's third defect was a 2-df ratio printed with a Greek sigma. The
    only way to stop that recurring is for the formatter to be unable to
    express it, so this prints t, df and p or nothing at all.
    """
    ft = r["finetuning_seed_test"]
    if ft.get("estimable"):
        lo, hi = ft["ci95"]
        stat = (f"t={ft['t']:+.2f} df={ft['df']:.1f} p={ft['p_two_sided']:.2g} "
                f"CI95=[{lo:+.5f},{hi:+.5f}]")
    else:
        stat = f"not estimable ({ft.get('reason')})"
    flag = "" if r["i1_clean"] else "  [I1: >1 variable]"
    lvl = "" if r["inference_level"] == "pretraining-seed" else \
        f"  [level={r['inference_level']}]"
    return f"  {r['contrast']:24s} {r['size']:10s} gap={r['gap']:+.6f}  {stat}{flag}{lvl}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # NAME=PATH, repeatable. For bench_metrics.json, whose cells are keyed by
    # dataset first, NAME picks the dataset: --leg top=... --leg qg=...
    ap.add_argument("--leg", action="append", metavar="NAME=PATH", default=None)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    results = {"metric": a.metric, "legs": {}}
    legs = a.leg or [f"{leg}={REPO / 'experiments/FIGS/data' / f'{leg}_metrics.json'}"
                     for leg in ("leg1", "leg2")]
    for item in legs:
        leg, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"FATAL: --leg wants NAME=PATH, got {item!r}")
        p = pathlib.Path(path)
        if not p.exists():
            raise SystemExit(f"FATAL: no {p}")
        cells = read_cells(json.loads(p.read_text()))
        cells = cells.get(leg, cells)
        rows = []
        print(f"\n=== {leg} ===")
        for name, spec in CONTRASTS.items():
            if any(arm not in cells for arm in spec["a"] + spec["b"]):
                print(f"  {name:24s} SKIPPED: arm absent from {leg}")
                continue
            for size in sizes_in(cells, spec["a"] + spec["b"]):
                r = contrast(cells, name, spec, size, a.metric)
                rows.append(r)
                print(format_row(r))
        results["legs"][leg] = rows

    # The one sentence that governs how any of this may be quoted.
    blocked = [r for leg in results["legs"].values() for r in leg
               if r["inference_level"] != "pretraining-seed"]
    if blocked:
        floors = {r["pretraining_seed_test"]["permutation_floor_one_sided"]
                  for r in blocked if "permutation_floor_one_sided" in r["pretraining_seed_test"]}
        print(f"\nPRETRAINING-SEED INFERENCE BLOCKED on {len(blocked)} of "
              f"{sum(len(v) for v in results['legs'].values())} rows. Smallest "
              f"attainable one-sided permutation p: {sorted(floors)}. "
              f"No pretraining-seed claim may be quoted from these rows at any "
              f"strength until L162 seeds 2-5 finish.")
    results["pretraining_seed_inference_blocked_rows"] = len(blocked)

    if a.out:
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
