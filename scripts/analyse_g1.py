#!/usr/bin/env python3
"""Decide G1: does the learning-rate optimum move with K?

THE GATE (docs/GATES.md)
------------------------
  Pass: the LR optimum is the SAME for L162 and R16_Q1, so one LR is
        defensible across the whole ladder.
  Kill: the optimum MOVES with K. A single LR then confounds granularity with
        tuning, every arm needs its own sweep, the compute model changes
        materially, and per CLAUDE.md that is a YELLOW escalation -- surfaced to
        the PI, never absorbed silently.

L162 and R16_Q1 are the ladder's ENDPOINTS. If the optimum does not move
between them, it does not move in between; that is the whole reason only two
arms are swept.

WHAT IT READS
-------------
weaver writes one line per epoch to <run>/train.log:

    Epoch #2: Current validation metric: 0.42349 (best: 0.46864)

`Current` is that epoch's validation accuracy; `best` is the running maximum,
which is also weaver's checkpoint-selection rule. We take the BEST over the 16
epochs, because that is the number the full-budget runs will be selected on.
Reading the last epoch instead would score a run on wherever it happened to
land, which is noise.

WHY BEST-OVER-EPOCHS AND NOT FINAL
----------------------------------
docs/GATES.md's G2 asks separately whether fixed-step and best-over-grid order
the arms differently. Using `best` here keeps G1 consistent with weaver's own
selection rule; if G2 later shows the two disagree, that is G2's result, not a
reason to have picked differently here.

Usage:
    python3 scripts/analyse_g1.py <dir-of-run-logs>

where the directory holds one subdir per run, named g1-<arm>-lr<tag>, each
containing train.log -- i.e. exactly what /data/results/g1/ looks like.
"""
from __future__ import annotations

import pathlib
import re
import sys

# tag -> the actual rate, for reporting. Must match scripts/build_g1_jobs.py.
RATES = {"25e5": 2.5e-4, "5e4": 5e-4, "1e3": 1e-3}
ARMS = ["l162", "r16q1"]

# Below this gap in validation accuracy, an arm's argmax is not resolved.
# HEURISTIC, not a measured sigma `[I]`. Two anchors: binomial SE on accuracy
# ~0.72 over the 1,280,000 validation jets is ~0.0004, and E1's three Arm P
# seeds land within 0.0002 of each other on test accuracy at FULL budget.
# Run-to-run spread at 20% budget is larger than either, and G1 has one seed
# per point so it cannot measure it. 0.005 is deliberately an order above the
# statistical floor: it flags "too close to call" rather than pretending the
# argmax resolved something.
TIE_MARGIN = 0.005

# Budget per point, from scripts/build_g1_jobs.py EPOCHS. A run short of this
# has not finished, and its `best` is a LOWER BOUND -- the completed runs show
# the decay phase (epochs 12-15) producing the single largest gains, so a run
# stopped before it is systematically understated. Issuing a verdict on a mixed
# set of finished and unfinished runs is the easiest way to get a confident
# wrong answer out of this script.
EXPECTED_EPOCHS = 16

EPOCH_RE = re.compile(
    r"Epoch #(\d+): Current validation metric: ([0-9.]+) \(best: ([0-9.]+)\)")


def read_run(log: pathlib.Path) -> dict | None:
    if not log.exists():
        return None
    rows = EPOCH_RE.findall(log.read_text(errors="replace"))
    if not rows:
        return None
    # weaver writes each validation line to train.log TWICE (verified on
    # g1-r16q1-lr5e4: 6 epochs -> 12 matching lines). Keying by epoch number
    # collapses that, and also does the right thing for a RESUMED run, where
    # an epoch legitimately recurs and the later value is the live one.
    # Without this, n_epochs reports 32 for a 16-epoch run. `best` is a max so
    # the verdict itself was never affected -- the epoch count was.
    by_epoch: dict[int, tuple[float, float]] = {}
    for e, cur, best in rows:
        by_epoch[int(e)] = (float(cur), float(best))
    epochs = [(e, c, b) for e, (c, b) in sorted(by_epoch.items())]
    return {
        "n_epochs": len(epochs),
        "last_epoch": epochs[-1][0],
        "best": max(b for _, _, b in epochs),
        "final": epochs[-1][1],
        "curve": [c for _, c, _ in epochs],
    }


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1])

    results: dict[str, dict[str, dict]] = {}
    missing = []
    for arm in ARMS:
        results[arm] = {}
        for tag in RATES:
            r = read_run(root / f"g1-{arm}-lr{tag}" / "train.log")
            if r is None:
                missing.append(f"g1-{arm}-lr{tag}")
            else:
                results[arm][tag] = r

    if missing:
        print(f"MISSING ({len(missing)}): {', '.join(missing)}")
        print("Reporting on what exists; the verdict below is NOT final until "
              "all six runs are in.\n")

    for arm in ARMS:
        print(f"=== {arm.upper()} ===")
        if not results[arm]:
            print("  no runs yet\n")
            continue
        for tag in sorted(RATES, key=lambda t: RATES[t]):
            r = results[arm].get(tag)
            if r is None:
                print(f"  lr={RATES[tag]:<8.1e} (missing)")
                continue
            print(f"  lr={RATES[tag]:<8.1e} best={r['best']:.5f} "
                  f"final={r['final']:.5f} epochs={r['n_epochs']}")
        print()

    # the verdict
    argmax = {}
    for arm in ARMS:
        if len(results[arm]) == len(RATES):
            argmax[arm] = max(results[arm], key=lambda t: results[arm][t]["best"])

    if len(argmax) < len(ARMS):
        print("VERDICT: incomplete — cannot decide until all six runs finish.")
        return 0

    incomplete = {f"g1-{arm}-lr{tag}": r["n_epochs"]
                  for arm in ARMS for tag, r in results[arm].items()
                  if r["n_epochs"] < EXPECTED_EPOCHS}
    if incomplete:
        print(f"*** PROVISIONAL — {len(incomplete)} of {len(ARMS) * len(RATES)} "
              f"runs have not finished: "
              f"{', '.join(f'{k} ({v}/{EXPECTED_EPOCHS})' for k, v in incomplete.items())}")
        print("    Their `best` is a LOWER BOUND. The decay phase (epochs 12-15) "
              "produces the largest\n    single gains, so an unfinished run is "
              "understated by more than the margins below.\n    Do NOT act on the "
              "verdict until every run reads 16/16.\n")

    # How resolved is each arm's argmax? The verdict is a comparison of two
    # argmaxes, so if either arm's top two rates are separated by less than the
    # run-to-run noise, the argmax is arbitrary and the PASS/KILL flips on
    # nothing. G1 runs ONE seed per point, so it cannot estimate that noise
    # from itself -- report the margin and refuse to call it resolved.
    margins = {}
    for arm, tag in argmax.items():
        ranked = sorted((results[arm][t]["best"] for t in results[arm]), reverse=True)
        margins[arm] = ranked[0] - ranked[1]

    same = len(set(argmax.values())) == 1
    for arm, tag in argmax.items():
        print(f"{arm.upper():7s} optimum: lr={RATES[tag]:.1e}  "
              f"(margin over 2nd best: {margins[arm]:.5f})")

    unresolved = {a: m for a, m in margins.items() if m < TIE_MARGIN}

    if same:
        tag = next(iter(argmax.values()))
        print(f"\nVERDICT: PASS — both endpoints optimise at lr={RATES[tag]:.1e}. "
              f"A single LR is defensible across the ladder. Use it for the "
              f"full-budget arms.")
    else:
        print("\nVERDICT: KILL/BRANCH — the optimum MOVES with K.")
        print("  A single LR would confound granularity with tuning.")
        print("  Per docs/GATES.md this needs a per-arm LR sweep, which changes")
        print("  the compute model materially. YELLOW: surface to the PI, do not")
        print("  proceed to the full-budget arms on one rate.")

    if unresolved:
        print(f"\nUNRESOLVED: {', '.join(unresolved)} — top two rates differ by "
              f"less than {TIE_MARGIN} validation accuracy, so that arm's "
              f"optimum is not distinguishable from a tie and the verdict "
              f"above turns on a difference this sweep cannot measure. "
              f"G1 runs one seed per point and cannot estimate its own noise; "
              f"resolving it needs a second seed at the top two rates.")

    # A grid-edge optimum means the sweep did not bracket the true best.
    edges = {a: t for a, t in argmax.items() if RATES[t] in (min(RATES.values()),
                                                            max(RATES.values()))}
    if edges:
        print(f"\nCAVEAT: optimum sits at a GRID EDGE for {', '.join(edges)}. "
              f"The true optimum may lie outside 2.5e-4..1e-3, so 'same optimum' "
              f"may just mean 'both pinned to the same edge'. Extend the grid "
              f"before treating a PASS as strong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
