#!/usr/bin/env python3
"""Merge per-arm anomaly runs into one artifact, then apply the cross-arm pass.

WHY THE ANOMALY TABLE IS BUILT THIS WAY NOW. The grid is ~1,440 cells at ~43 s,
and the single-process version died after ~25 h with one arm of four finished
(ledger: eval-anomaly-died-partial). The pod was garbage-collected before it
could be inspected, so the cause was never determined -- which is the point: a
run whose failure mode cannot be diagnosed must not also be a run whose failure
costs everything. Arms are independent up to one final normalisation, so they
now run as separate jobs and this merges them. Three consequences:

  - a death costs ONE arm, not the wave;
  - arms run in PARALLEL, so the wall clock is one arm (~25 h), not four;
  - an arm already on disk is simply not re-run. l162-s1b completed on
    2026-09-09 and does not need recomputing.

WHAT A PER-ARM RUN CANNOT DO, AND WHY THIS FILE EXISTS RATHER THAN A SHELL `cat`.
The regret that the vocabulary ablation is actually about is
`sigma_min / min over ARMS at fixed (signal, N_sig, family)`. An arm run on its
own takes that minimum over itself and reports regret exactly 1.000 for every
family -- "no regret from coarsening the vocabulary", the null under test,
manufactured by the aggregation. That is the defect audit-2-anomaly found when
the normalisation was inside the arm loop, and splitting arms across jobs
reintroduces it unless the merge recomputes. So the per-arm `regret` and
`regret_n_arms` fields are STALE BY CONSTRUCTION and are overwritten here by
anomaly.cross_arm_regret(), the same function the single-process path calls.

THE INPUTS MUST BE COMMENSURABLE OR THE MERGE IS MEANINGLESS. Cells from runs
with different trainings, background sizes, statistics cuts or -- above all -- a
different test-row ordering are not comparable, and concatenating them produces
a table with no error bar and no way to notice. Every shared configuration key
is compared and a mismatch is fatal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

# Keys that must agree across every per-arm run for the cells to be comparable.
# row_alignment_sha256 is the load-bearing one: it says the arms scored the SAME
# jets in the SAME order, which is what makes a cross-arm minimum meaningful.
SHARED = ["row_alignment_sha256", "sigma_t", "stat_cut", "min_bkg_pass",
          "trainings", "n_bkg", "n_template"]


def _anomaly():
    spec = importlib.util.spec_from_file_location(
        "anomaly", REPO / "experiments/EVAL/anomaly.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def merge(payloads: list[tuple[str, dict]]) -> dict:
    """Combine per-arm payloads. `payloads` is [(label, parsed json), ...]."""
    if not payloads:
        raise SystemExit("FATAL: nothing to merge")
    base_label, base = payloads[0]
    merged = {k: base.get(k) for k in SHARED}
    for label, d in payloads[1:]:
        for k in SHARED:
            if d.get(k) != base.get(k):
                raise SystemExit(
                    f"FATAL: {label} and {base_label} disagree on {k!r} "
                    f"({d.get(k)!r} vs {base.get(k)!r}). Cells from runs that "
                    "differ in configuration or in test-row ordering are not "
                    "comparable and must not be merged.")
    merged["arms"] = {}
    seen = {}
    for label, d in payloads:
        arms = d.get("arms")
        if not arms:
            raise SystemExit(f"FATAL: {label} has no 'arms' block")
        for arm, ad in arms.items():
            if arm in seen:
                raise SystemExit(
                    f"FATAL: arm {arm!r} appears in both {seen[arm]} and "
                    f"{label}. Silently keeping one would hide which run the "
                    "published number came from.")
            seen[arm] = label
            merged["arms"][arm] = ad
    merged["merged_from"] = {a: seen[a] for a in sorted(seen)}
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-arm anomaly_results.json files, or directories "
                         "containing one")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    payloads = []
    for raw in a.inputs:
        p = pathlib.Path(raw)
        if p.is_dir():
            p = p / "anomaly_results.json"
        if not p.exists():
            raise SystemExit(f"FATAL: no {p}")
        payloads.append((str(p), json.loads(p.read_text())))

    merged = merge(payloads)
    arms = sorted(merged["arms"])
    print(f"merged {len(arms)} arms: {', '.join(arms)}")
    if len(arms) < 2:
        # Not fatal -- a one-arm merge is a legitimate intermediate -- but the
        # regret column it produces is degenerate and must not be published.
        print("WARNING: fewer than two arms. Cross-arm regret is normalised "
              "against a single arm, so every regret will be 1.000. This "
              "artifact is not publishable as a vocabulary ablation.")

    an = _anomaly()
    an.cross_arm_regret(merged)
    bad, unmeasured = an.null_guard(merged)

    # RAGGED GRIDS ARE THE EXPECTED CASE HERE, AND THEY ARE NOT UNIFORMLY FATAL.
    #
    # cross_arm_regret takes its minimum over the arms PRESENT in each
    # (signal, N_sig, family) cell, and stamps regret_n_arms. A cell only one arm
    # reached therefore normalises against itself and reports regret 1.000 -- the
    # audit-2-anomaly defect, reappearing per-cell instead of per-arm. The
    # stamp already existed; nothing read it, so the degenerate cells were
    # indistinguishable from measured ones in the artifact.
    #
    # This is not hypothetical. The l162-s1b payload this merge is wired to read
    # holds FIVE of six signals -- label_X_YY_qqqq is absent, because that job
    # died at ~25 h (ledger: eval-anomaly-died-partial) -- and anomaly.py writes
    # a complete-looking anomaly_results.json after EVERY signal, so an existence
    # check cannot tell a finished arm from a partial one. Any arm killed by its
    # deadline lands here the same way.
    #
    # Reported, not refused: a partial merge is a legitimate intermediate and
    # refusing it would throw away the only copy of expensive work.
    per_arm_signals = {arm: set(ad["signals"]) for arm, ad in merged["arms"].items()}
    universe = sorted(set().union(*per_arm_signals.values())) if per_arm_signals else []
    incomplete = {a: sorted(set(universe) - s) for a, s in per_arm_signals.items()
                  if set(universe) - s}
    counts = {}
    for ad in merged["arms"].values():
        for sig, per_n in ad["signals"].items():
            for n_sig, agg in per_n.items():
                for v in (agg or {}).values():
                    if isinstance(v, dict) and "regret_n_arms" in v:
                        counts[(sig, n_sig)] = max(counts.get((sig, n_sig), 0),
                                                   v["regret_n_arms"])
    degenerate = sorted(k for k, n in counts.items() if n < 2)
    # ONE ARM IS NOT THE ONLY WAY A REGRET CELL CAN BE EMPTY OF MEANING, and the
    # check above misses the other way. `regret_n_arms >= 2` -- the filter this
    # file tells the reader to publish on -- is satisfied by THREE R16_Q1 SEEDS
    # with no L162 arm present, and that cell's regret then measures seed-to-seed
    # variation inside ONE vocabulary while reading as a vocabulary ablation.
    #
    # It is not hypothetical: label_X_YY_qqqq is exactly this in the first real
    # merge (2026-09-15). l162-s1b lost that signal when its job died at ~25 h,
    # so the signal survives on the three R16_Q1 seeds alone, passes
    # regret_n_arms = 3, and would have gone into the table as a measured cost of
    # coarsening. The quantity the ablation is about is defined ACROSS RUNGS, so
    # the rung -- not the arm -- is what has to be counted.
    per_sig_rungs: dict[str, set] = {}
    for arm, ad in merged["arms"].items():
        rung = ad.get("rung", arm)
        for sig in ad["signals"]:
            per_sig_rungs.setdefault(sig, set()).add(rung)
    one_rung = sorted(s for s, r in per_sig_rungs.items() if len(r) < 2)

    merged["completeness"] = {
        "signals_seen": universe,
        "arms_missing_signals": incomplete,
        "cells_normalised_against_one_arm": [list(k) for k in degenerate],
        "rungs_per_signal": {s: sorted(r) for s, r in sorted(per_sig_rungs.items())},
        "signals_with_one_rung": one_rung,
    }
    if one_rung:
        print(f"WARNING: {len(one_rung)} signal(s) are carried by only ONE RUNG, "
              f"so their cross-arm regret measures SEED variation, not the cost "
              f"of coarsening -- and they PASS regret_n_arms >= 2. Exclude them "
              f"from any vocabulary claim:")
        for s in one_rung:
            print(f"  {s}: rungs {sorted(per_sig_rungs[s])}")
    if incomplete:
        print("WARNING: the grid is RAGGED. Arms missing signals:")
        for arm, miss in sorted(incomplete.items()):
            print(f"  {arm}: missing {', '.join(miss)}")
    if degenerate:
        print(f"WARNING: {len(degenerate)} (signal, N_sig) cells were reached by "
              f"only ONE arm. Their regret is 1.000 by construction, not by "
              f"measurement, and must not be read as 'no cost to coarsening'. "
              f"Filter on regret_n_arms >= 2 before publishing.")
        for sig, n_sig in degenerate[:10]:
            print(f"  {sig} N_sig={n_sig}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "anomaly_results.json").write_text(json.dumps(merged, indent=2))
    print(f"wrote {a.out / 'anomaly_results.json'}")
    print(f"null not flat: {len(bad)}   null unmeasured: {len(unmeasured)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
