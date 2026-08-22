#!/usr/bin/env python3
"""Locate each arm's OWN learning-rate optimum, and say whether it is bracketed.

WHY THIS IS SEPARATE FROM analyse_g1.py
---------------------------------------
analyse_g1.py answers the G1 GATE question -- does the optimum MOVE with K? --
and it is deliberately hardcoded to the two ladder endpoints at the three grid
rates, because that is the comparison the gate is defined on. That question is
answered: KILL, the optimum moves. Widening that script to cover the per-arm
sweeps would blur the gate's evidence with the follow-up work.

This script answers the DIFFERENT question the KILL created: for each arm
separately, where is its optimum, and do we actually know it?

WHAT "BRACKETED" MEANS, AND WHY IT IS THE POINT
-----------------------------------------------
An argmax at the edge of a grid is not an optimum -- it is the best of what was
tried, with the true best possibly outside. Because the study compares arms AT
their optima, an unbracketed rate is a rate we do not know, and training an
80-epoch arm against one risks the whole budget on an untested guess.

An arm is BRACKETED when a point on each side of its argmax TRAINED and scored
worse.

A DIVERGENCE DOES NOT BRACKET. This script said the opposite until 2026-08-22,
on the reasoning that a rate which cannot be trained puts a ceiling on the
optimum. That reasoning assumed divergence was a property of the rate. It is
not. g1-r42q1-lr5e4 went nan at iteration 2 at seed 1; g1-r42q1-lr5e4-s2, the
same arm at the same rate at seed 2, cleared 16,000 iterations clean with
AvgAcc 0.638. Every divergence on record -- R42_Q1 at 5e-4, R42_Q1 at 1e-3,
L162 at 2e-3 -- is a SEED 1 run, and the only one retried at another seed
trained.

So divergence is stochastic and a single-seed failure says nothing definite
about the rate. Treating it as a ceiling silently converted "we got unlucky
once" into "this rate is unusable", which is how L162 came to look bracketed
above when nothing above its argmax has ever been shown to be worse. A diverged
rate is reported here as UNSTABLE, with the seed count, and does not close a
bracket.

TIES
----
The seed spread was measured twice on R16_Q1 under an otherwise identical
recipe (0.01319 and 0.01580). Gaps below that floor are ties, not orderings, so
a "bracket" whose margins are inside the noise is reported as a SOFT bracket --
the shape is right but the evidence does not resolve it.

Usage:
    python3 scripts/analyse_lr_sweep.py <dir-of-run-logs>

where the directory holds one subdir per run named g1-<arm>-lr<tag>[-s<seed>],
i.e. exactly what /data/results/g1/ looks like.
"""
from __future__ import annotations

import pathlib
import re
import sys

# Measured, not assumed. See scripts/analyse_g1.py for the derivation.
TIE_MARGIN = 0.0158
EXPECTED_EPOCHS = 16

# tag -> rate. Must match scripts/build_lr_sweep.py's tag convention:
# mantissa then exponent, minus sign dropped.
TAGS = {
    "125e6": 1.25e-4, "25e5": 2.5e-4, "5e4": 5e-4,
    "1e3": 1e-3, "14e4": 1.4e-3, "2e3": 2e-3, "4e3": 4e-3,
}
ARMS = ["l188", "l162", "r42q1", "r16q1"]

EPOCH_RE = re.compile(
    r"Epoch #(\d+): Current validation metric: ([0-9.]+) \(best: ([0-9.]+)\)")
RUN_RE = re.compile(r"^g1-(?P<arm>[a-z0-9]+)-lr(?P<tag>[0-9a-z]+)(?:-s(?P<seed>\d+)\w*)?$")


def read_run(d: pathlib.Path) -> dict:
    """Return status for one run directory. Never raises on a broken run."""
    log = d / "train.log"
    if not log.exists():
        return {"status": "no-log"}
    text = log.read_text(errors="replace")
    rows = EPOCH_RE.findall(text)
    if not rows:
        # weaver logs the nan before the CUDA assert kills it, so a run that
        # diverged has a log but no validation line. Distinguish that from a
        # run that simply has not reached its first validation yet.
        return {"status": "diverged" if "nan" in text.lower() else "no-epochs"}
    # weaver writes each validation line TWICE, and a resumed run legitimately
    # repeats an epoch; keying by epoch collapses both.
    by_epoch = {int(e): (float(c), float(b)) for e, c, b in rows}
    return {
        "status": "complete" if len(by_epoch) >= EXPECTED_EPOCHS else "partial",
        "n_epochs": len(by_epoch),
        "best": max(b for _, b in by_epoch.values()),
    }


def collect(root: pathlib.Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m or m["tag"] not in TAGS:
            continue
        r = read_run(d)
        r["seed"] = int(m["seed"] or 1)
        r["name"] = d.name
        out.setdefault(m["arm"], {}).setdefault(m["tag"], []).append(r)
    return out


def verdict(points: dict[str, list[dict]]) -> tuple[str, str]:
    """(status, explanation) for one arm.

    Only TRAINED points can close a bracket. Diverged points are recorded and
    reported, because a rate that failed at one seed and worked at another is a
    real fact about stability, but they carry no information about ordering.
    """
    scored, unstable = {}, {}
    for tag, runs in points.items():
        ok = [r for r in runs if r["status"] in ("complete", "partial")]
        bad = [r for r in runs if r["status"] == "diverged"]
        if ok:
            scored[tag] = max(r["best"] for r in ok)
        if bad:
            unstable[tag] = (len(bad), len(ok) + len(bad))
    if not scored:
        return "NO DATA", "no rate produced a validation number"

    order = sorted(TAGS, key=lambda t: TAGS[t])
    best_tag = max(scored, key=lambda t: scored[t])
    i = order.index(best_tag)
    lower = [t for t in order[:i] if t in scored]
    upper = [t for t in order[i + 1:] if t in scored]

    note = ""
    if unstable:
        note = ("; unstable at " + ", ".join(
            f"{TAGS[t]:.2e} ({n} of {m} seeds nan)"
            for t, (n, m) in sorted(unstable.items(), key=lambda kv: TAGS[kv[0]])))

    if not lower and not upper:
        return "UNBRACKETED (single point)", f"only {TAGS[best_tag]:.2e} ever trained{note}"
    if not lower:
        return ("UNBRACKETED BELOW",
                f"no TRAINED point below {TAGS[best_tag]:.2e}{note}")
    if not upper:
        return ("UNBRACKETED ABOVE",
                f"no TRAINED point above {TAGS[best_tag]:.2e}; the optimum may be "
                f"higher and a divergence there would not prove otherwise{note}")

    lo, up = lower[-1], upper[0]
    ml = scored[best_tag] - scored[lo]
    mu = scored[best_tag] - scored[up]
    detail = (f"beats {TAGS[lo]:.2e} by {ml:.5f} and {TAGS[up]:.2e} by {mu:.5f}")
    soft = [f"{TAGS[t]:.2e}" for t, m in ((lo, ml), (up, mu)) if m < TIE_MARGIN]
    if soft:
        return "SOFT BRACKET", (detail + f"; inside the {TIE_MARGIN} noise floor vs "
                                + ", ".join(soft) + note)
    return "BRACKETED", detail + note


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"FATAL: {root} is not a directory")
    data = collect(root)
    if not data:
        sys.exit(f"no recognisable run directories under {root}")

    print(f"Per-arm learning-rate sweep. Tie margin {TIE_MARGIN} (measured).")
    print("A rate that DIVERGED does NOT bracket: divergence is stochastic "
          "(same arm+rate, seed 1 nan / seed 2 clean).\n")

    for arm in [a for a in ARMS if a in data] + [a for a in data if a not in ARMS]:
        pts = data[arm]
        print(f"=== {arm.upper()} ===")
        for tag in sorted(TAGS, key=lambda t: TAGS[t]):
            if tag not in pts:
                continue
            for r in sorted(pts[tag], key=lambda x: x["seed"]):
                if r["status"] in ("complete", "partial"):
                    flag = "" if r["status"] == "complete" else f"  << only {r['n_epochs']}/{EXPECTED_EPOCHS}"
                    print(f"  lr={TAGS[tag]:<9.2e} seed {r['seed']}  best={r['best']:.5f}{flag}")
                else:
                    print(f"  lr={TAGS[tag]:<9.2e} seed {r['seed']}  {r['status'].upper()}")
        st, why = verdict(pts)
        print(f"  -> {st}: {why}\n")

    ready = [a for a in data if verdict(data[a])[0] in ("BRACKETED", "SOFT BRACKET")]
    blocked = [a for a in data if a not in ready]
    print("READY for full budget:", ", ".join(sorted(ready)) or "none")
    if blocked:
        print("BLOCKED (rate not bracketed):", ", ".join(sorted(blocked)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
