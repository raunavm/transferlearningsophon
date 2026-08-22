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

An arm is BRACKETED when a point on each side of its argmax scored worse. A
DIVERGENCE brackets from above just as well as a low score does: if the rate
above the argmax cannot be trained at all, the optimum is below it. That is how
L162 is bracketed despite its argmax sitting at the top of the grid.

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
    """(status, explanation) for one arm."""
    # Score each rate by its BEST seed that actually trained.
    scored, diverged = {}, set()
    for tag, runs in points.items():
        ok = [r for r in runs if r["status"] in ("complete", "partial")]
        if ok:
            scored[tag] = max(r["best"] for r in ok)
        elif any(r["status"] == "diverged" for r in runs):
            diverged.add(tag)
    if not scored:
        return "NO DATA", "no rate produced a validation number"

    order = sorted(TAGS, key=lambda t: TAGS[t])
    best_tag = max(scored, key=lambda t: scored[t])
    i = order.index(best_tag)
    lower = [t for t in order[:i] if t in scored or t in diverged]
    upper = [t for t in order[i + 1:] if t in scored or t in diverged]

    def margin(t):
        return None if t in diverged else scored[best_tag] - scored[t]

    if not lower and not upper:
        return "UNBRACKETED (single point)", f"only {TAGS[best_tag]:.2e} was tried"
    if not lower:
        return ("UNBRACKETED BELOW",
                f"nothing was run below {TAGS[best_tag]:.2e}; the optimum may be lower")
    if not upper:
        return ("UNBRACKETED ABOVE",
                f"nothing was run above {TAGS[best_tag]:.2e}; the optimum may be higher")

    lo, up = lower[-1], upper[0]
    ml, mu = margin(lo), margin(up)
    soft = [f"{TAGS[t]:.2e} by {m:.5f}" for t, m in ((lo, ml), (up, mu))
            if m is not None and m < TIE_MARGIN]
    detail = (f"beats {TAGS[lo]:.2e} " + (f"by {ml:.5f}" if ml is not None else "(diverged)")
              + f" and {TAGS[up]:.2e} " + (f"by {mu:.5f}" if mu is not None else "(diverged)"))
    if soft:
        return "SOFT BRACKET", detail + f"; inside the {TIE_MARGIN} noise floor vs " + ", ".join(soft)
    return "BRACKETED", detail


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
    print("A rate that DIVERGED brackets from above exactly as a low score does.\n")

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
