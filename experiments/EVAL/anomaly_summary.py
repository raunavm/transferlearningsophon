#!/usr/bin/env python3
"""The anomaly-detection table as the paper reports it: mean +/- sd over seeds.

READS    anomaly_merge.py's artifact (experiments/FIGS/data/anomaly_merged_v4/
         anomaly_results.json) and, once it exists, the merged class-sum rerun
         (--class-sum-rerun), which adds the family class_sum_matched.
WRITES   <out>/anomaly_summary.json, refusing to overwrite. Per family x signal
         x injection x label set: the five per-seed values of ln sigma_min and
         of max SIC, their mean and sd (ddof=1). No hypothesis test.

CONVENTIONS, and where each comes from.
  within a seed  the stored value, which anomaly.py already made the MEDIAN over
                 the 10 resamplings -- arXiv:2604.20965 Sec. V.4 reports the
                 median over its 10 training sets. Nothing is re-aggregated.
  across seeds   mean and sample sd (n-1) over the five pretraining seeds, the
                 unit of inference (PRESPEC 2.1).
  injection      N_sig = 2000 is primary. The Mahalanobis, kNN and class-sum
                 scores are fixed functions of the jet, so N_sig only moves the
                 sampling noise (measured here: see `injection_agreement`), and
                 N_sig = 4000 exceeds the X->YY->bbb test jets, which is skipped
                 there. 4000 is recorded for reference.
  max SIC        with the 20 % background-statistics cut, arXiv:2511.14832
                 Sec. III.4 (anomaly.sic_curve).

NOT DETECTED. A (family, signal) is flagged when the seed-mean max SIC is below
NOT_DETECTED_MAX_SIC at EVERY label set. A score with no power has
eps_S = eps_B at every threshold, so its SIC is sqrt(eps_B) <= 1 -- the
"random" term ARGOS subtracts (arXiv:2511.14832 eq. 6) -- and its max SIC is 1,
reached with no cut; sigma_min then sits at its ceiling sigma_t = 5 (ln 1.61),
because without a cut the Asimov Z is ~ S/sqrt(B) = sigma_0. max SIC < 1.3 means
the best cut raises the significance by under 30 %: the signal would already
need ~5/1.3 = 3.8 sigma before any selection, which is not anomaly detection.
The output records the range of thresholds over which the flagged set would
not change, so the reader can see how much the choice matters.

EXCLUDED. iad_hgb, whose sigma_min is not the published definition (see
EXCLUDED below and anomaly.score_iad). It is dropped, not corrected.

SUPERSEDED. class_sum leaves out only the signal's own output node, which is 1
native class at 188 and 162 classes but 3-12 at 43 and 10-29 at 17, so its
cross-label-set comparison mixes the label-set effect with a change of
estimator. class_sum_matched (anomaly.py) removes the same native classes at
every label set. With the rerun supplied, class_sum is marked superseded and
kept beside it for reproducibility; the rerun must reproduce the committed
draws (rng seeds) and, at 17 classes, where the two estimators coincide, the
committed class_sum values, or nothing is written.

Usage:
    python3 experiments/EVAL/anomaly_summary.py \
        --anomaly experiments/FIGS/data/anomaly_merged_v4/anomaly_results.json \
        [--class-sum-rerun <anomaly_cs_merged_v1>/anomaly_results.json] \
        --out experiments/FIGS/data/anomaly_merged_v4/analysis_v3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]

LEVELS = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
SEEDS = (1, 2, 3, 4, 5)
PRIMARY, REFERENCE = "2000", "4000"
FAMILIES = ("class_sum", "knn", "mahalanobis")
RERUN_FAMILY = "class_sum_matched"
NOT_DETECTED_MAX_SIC = 1.3
EXCLUDED = {"iad_hgb": (
    "sigma_min is computed from the efficiency curve of ONE classifier trained "
    "at the injected N_sig and scanned in sigma_0 with that curve held fixed, and "
    "the classifier is scored on its own training rows. arXiv:2604.20965 Sec. V.4 "
    "trains a classifier per injection, takes the median Asimov Z over trainings "
    "and interpolates to Z = 5. Too optimistic by about a factor 2; a correct "
    "rerun (per-injection grid, held-out scoring) costs ~10x the original run, so "
    "the family is dropped from the paper rather than rerun.")}
CONFIG_KEYS = ("row_alignment_sha256", "sigma_t", "stat_cut", "min_bkg_pass",
               "trainings", "n_bkg", "n_template")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _level_seed(arm: str, ad: dict, parse_arm) -> tuple[int, int]:
    """(level, seed) from the arm name (seed_level's convention, l162-s1b = seed
    1), cross-checked against the label set the artifact records."""
    level, seed = parse_arm(arm)
    if LEVELS.get(ad.get("rung")) != level:
        raise SystemExit(f"FATAL: {arm} records label set {ad.get('rung')!r}, "
                         f"which is not the {level}-class set its name says")
    return level, seed


def family_block(doc: dict, fam: str, parse_arm) -> dict:
    """{signal: {injection: {...}}} for one family, per level over the five seeds."""
    by = {}
    for arm, ad in doc["arms"].items():
        level, seed = _level_seed(arm, ad, parse_arm)
        for sig, per_n in ad["signals"].items():
            for n in (PRIMARY, REFERENCE):
                by.setdefault(sig, {}).setdefault(n, {}).setdefault(level, {})[seed] = (
                    arm, per_n.get(n, {}))
    out = {}
    for sig, per_n in by.items():
        out[sig] = {}
        for n, per_level in per_n.items():
            cells = [c for lv in per_level.values() for _a, c in lv.values()]
            if all("skipped" in c for c in cells):
                out[sig][n] = {"skipped": cells[0]["skipped"]}
                continue
            levels = {}
            for level in LEVELS.values():
                got = per_level.get(level, {})
                if tuple(sorted(got)) != SEEDS:
                    raise SystemExit(f"FATAL: {fam}/{sig}/N={n} at {level} classes "
                                     f"has seeds {sorted(got)}, not {list(SEEDS)}")
                arms = [got[s][0] for s in SEEDS]
                vals = [got[s][1].get(fam) for s in SEEDS]
                if not all(v and "sigma_min" in v for v in vals):
                    raise SystemExit(f"FATAL: {fam}/{sig}/N={n} has no sigma_min "
                                     f"for some of {arms}")
                sm = [float(v["sigma_min"]) for v in vals]
                if not all(math.isfinite(x) and x > 0 for x in sm):
                    raise SystemExit(f"FATAL: {fam}/{sig}/N={n} {arms}: sigma_min "
                                     f"{sm} has no finite log")
                ln = [math.log(x) for x in sm]
                ms = [float(v["max_sic"]) for v in vals]
                levels[str(level)] = {
                    "arms": arms, "ln_sigma_min": ln, "max_sic": ms,
                    "ln_sigma_min_mean": float(np.mean(ln)),
                    "ln_sigma_min_sd": float(np.std(ln, ddof=1)),
                    "max_sic_mean": float(np.mean(ms)),
                    "max_sic_sd": float(np.std(ms, ddof=1)),
                    "n_at_ceiling": sum(bool(v.get("at_ceiling")) for v in vals)}
            means = {lv: e["max_sic_mean"] for lv, e in levels.items()}
            out[sig][n] = {
                "levels": levels,
                "max_sic_mean_by_level": means,
                "max_sic_max_over_seeds_and_levels": max(
                    x for e in levels.values() for x in e["max_sic"]),
                "not_detected": all(m < NOT_DETECTED_MAX_SIC for m in means.values())}
    return out


def classes_removed(doc: dict, key: str) -> dict:
    """{signal: {level: n}}; a function of the tree, so every cell must agree."""
    got = {}
    for ad in doc["arms"].values():
        for sig, per_n in ad["signals"].items():
            for c in per_n.values():
                if key in c:
                    got.setdefault(sig, {}).setdefault(LEVELS[ad["rung"]], set()).add(c[key])
    for sig, per in got.items():
        if any(len(v) != 1 for v in per.values()):
            raise SystemExit(f"FATAL: cells disagree on {key} for {sig}: {per}")
    return {s: {str(lv): per[lv].pop() for lv in LEVELS.values() if lv in per}
            for s, per in got.items()}


# The rerun rebuilds the logits from the cached features on whichever node it
# lands on, and a matrix product is reproducible only on one node: the last
# digits differ, and a jet that sits exactly on a threshold can change side. On
# the 2026-09-29 rerun 143 of 145 cells at 17 classes agreed to 1e-9 and two
# differed by 3e-5 in log; the spread between pretraining seeds is ~0.1. A
# real disagreement (a different score, a different sample) is far larger.
REPRO_TOL = 1e-4


def check_rerun(doc: dict, rerun: dict) -> dict:
    """The rerun must be the committed run's draws, or its numbers are not comparable."""
    for k in CONFIG_KEYS:
        if doc.get(k) != rerun.get(k):
            raise SystemExit(f"FATAL: the class-sum rerun differs on {k!r} "
                             f"({rerun.get(k)!r} vs {doc.get(k)!r})")
    if set(doc["arms"]) != set(rerun["arms"]):
        raise SystemExit(f"FATAL: the rerun has models {sorted(rerun['arms'])}, the "
                         f"committed run {sorted(doc['arms'])}")
    n_seeds = n_same = n_exact = 0
    worst = 0.0
    for arm, ad in doc["arms"].items():
        for sig, per_n in ad["signals"].items():
            for n, c in per_n.items():
                r = rerun["arms"][arm]["signals"].get(sig, {}).get(n)
                if r is None:
                    raise SystemExit(f"FATAL: the rerun has no {arm}/{sig}/N={n}")
                if c.get("rng_seeds") != r.get("rng_seeds"):
                    raise SystemExit(f"FATAL: {arm}/{sig}/N={n} drew different "
                                     "resamplings in the rerun")
                n_seeds += 1
                # At 17 classes the signal's group IS its node, so the two
                # estimators coincide and the rerun must return the committed value.
                if ad["rung"] == "R16_Q1" and "sigma_min" in c.get("class_sum", {}):
                    old, new = c["class_sum"], r.get(RERUN_FAMILY, {})
                    if "sigma_min" not in new:
                        raise SystemExit(f"FATAL: the rerun has no {RERUN_FAMILY} "
                                         f"sigma_min at {arm}/{sig}/N={n}")
                    for m in ("sigma_min", "max_sic"):
                        d = abs(math.log(new[m]) - math.log(old[m]))
                        worst = max(worst, d)
                        if d > REPRO_TOL:
                            raise SystemExit(
                                f"FATAL: {arm}/{sig}/N={n}: {RERUN_FAMILY} {m} "
                                f"{new[m]!r} does not reproduce the committed "
                                f"class_sum {old[m]!r} at 17 classes")
                    n_same += 1
                    n_exact += all(abs(math.log(new[m]) - math.log(old[m])) <= 1e-9
                                   for m in ("sigma_min", "max_sic"))
    return {"cells_with_identical_rng_seeds": n_seeds,
            "cells_at_17_classes_reproducing_class_sum": n_same,
            "cells_at_17_classes_identical_to_1e-9": n_exact,
            "tolerance_abs_log": REPRO_TOL,
            "max_abs_log_difference_at_17_classes": worst}


def summarise(doc: dict, rerun: dict | None = None) -> dict:
    parse_arm = _load("seed_level", "experiments/STATS/seed_level.py").parse_arm
    fams = {f: family_block(doc, f, parse_arm) for f in FAMILIES}
    res = {"families": fams, "excluded_families": dict(EXCLUDED), "superseded": {}}
    cr = {"class_sum": classes_removed(doc, "classes_removed")}
    if rerun is not None:
        res["reproduction"] = check_rerun(doc, rerun)
        fams[RERUN_FAMILY] = family_block(rerun, RERUN_FAMILY, parse_arm)
        cr[RERUN_FAMILY] = classes_removed(rerun, "classes_removed_matched")
        uneven = {s: v for s, v in cr[RERUN_FAMILY].items() if len(set(v.values())) != 1}
        if uneven:
            raise SystemExit(f"FATAL: {RERUN_FAMILY} removed different native "
                             f"classes by label set: {uneven}")
        res["superseded"]["class_sum"] = {
            "by": RERUN_FAMILY,
            "reason": "class_sum removes a different number of native classes at "
                      "each label set (classes_removed_by_level); class_sum_matched "
                      "removes the same ones everywhere. Kept for reproducibility."}
    res["classes_removed_by_level"] = cr

    # How much the primary-injection choice matters, measured rather than asserted.
    diffs = [abs(b["levels"][lv]["ln_sigma_min_mean"] - a["levels"][lv]["ln_sigma_min_mean"])
             for f in fams.values() for per_n in f.values()
             if "levels" in (a := per_n[PRIMARY]) and "levels" in (b := per_n[REFERENCE])
             for lv in a["levels"]]
    res["injection_agreement"] = {
        "max_abs_diff_ln_sigma_min_mean_2000_vs_4000": max(diffs),
        "skipped_at_4000": sorted({s for f in fams.values() for s, per_n in f.items()
                                   if "skipped" in per_n[REFERENCE]})}

    # the highest level mean per (family, signal): flagged iff below the threshold
    peak = {(f, s): max(per_n[PRIMARY]["max_sic_mean_by_level"].values())
            for f, blk in fams.items() for s, per_n in blk.items()}
    flagged = [p for p in peak.values() if p < NOT_DETECTED_MAX_SIC]
    kept = [p for p in peak.values() if p >= NOT_DETECTED_MAX_SIC]
    res["not_detected_rule"] = {
        "threshold_max_sic": NOT_DETECTED_MAX_SIC,
        "statistic": "seed-mean max SIC at N_sig = 2000, below the threshold at "
                     "every label set",
        "reason": "no discriminating power gives max SIC = 1 (no cut; sigma_min "
                  "at its ceiling 5, ln 1.61); below 1.3 the best cut raises the "
                  "significance by under 30 %, i.e. a signal would need ~3.8 sigma "
                  "before selection (arXiv:2511.14832 Sec. III.4 max SIC, eq. 6 "
                  "random baseline sqrt(eps_B))",
        "flag_set_unchanged_for_thresholds_in": [max(flagged, default=None),
                                                 min(kept, default=None)],
        "not_detected": sorted(f"{f}|{s}" for (f, s), p in peak.items()
                               if p < NOT_DETECTED_MAX_SIC)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anomaly", required=True, type=pathlib.Path)
    ap.add_argument("--class-sum-rerun", type=pathlib.Path, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    out_file = a.out / "anomaly_summary.json"
    if out_file.exists():
        raise SystemExit(f"FATAL: {out_file} exists; refusing to overwrite -- give "
                         "a new --out.")
    doc = json.loads(a.anomaly.read_text())
    rerun = (json.loads(a.class_sum_rerun.read_text())
             if a.class_sum_rerun else None)
    res = summarise(doc, rerun)
    res = {"provenance": {
               "inputs": {"anomaly": {"path": str(a.anomaly), "sha256": _sha(a.anomaly)},
                          "class_sum_rerun": (
                              {"path": str(a.class_sum_rerun),
                               "sha256": _sha(a.class_sum_rerun)}
                              if a.class_sum_rerun else None)},
               "script_sha256": _sha(__file__),
               "row_alignment_sha256": doc["row_alignment_sha256"],
               "resamplings_per_seed": doc["trainings"],
               "argv": list(argv if argv is not None else sys.argv[1:])},
           "conventions": {
               "within_seed": "stored median over the resamplings (anomaly.py)",
               "across_seeds": "mean and sd (ddof=1) over seeds 1-5; no tests",
               "primary_injection": PRIMARY, "reference_injection": REFERENCE,
               "ln": "natural log of sigma_min; lower is better"},
           **res}

    print(f"ln sigma_min, mean +/- sd over seeds, N_sig = {PRIMARY}  "
          f"[max SIC mean]  (* = not detected)")
    for fam, blk in res["families"].items():
        tag = " (superseded)" if fam in res["superseded"] else ""
        print(f"  {fam}{tag}")
        for sig, per_n in blk.items():
            c = per_n[PRIMARY]
            row = "  ".join(f"{lv:>3s}: {e['ln_sigma_min_mean']:+.3f}+/-"
                            f"{e['ln_sigma_min_sd']:.3f} [{e['max_sic_mean']:.2f}]"
                            for lv, e in c["levels"].items())
            print(f"    {sig:18s}{'*' if c['not_detected'] else ' '} {row}")
    print(f"excluded: {sorted(res['excluded_families'])}; injection agreement "
          f"{res['injection_agreement']['max_abs_diff_ln_sigma_min_mean_2000_vs_4000']:.3f}")
    a.out.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
