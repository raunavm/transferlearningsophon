#!/usr/bin/env python3
"""The fine-tuning rate search, read (PI 2026-11-20/21): for every row (task, training size; pretrained or from scratch)
the validation accuracy at each rate tried, the rate it selects, and whether that rate is the one recorded in
scripts/build_ft_budget_jobs.py (FIXED_ROW_RATE, FIXED_ROW_RATE_SCRATCH: the single source of each row's rate).

Rules, applied to the raw per-cell facts of pull_mlst_data.rate_search_scan (budget/rate_search.json):
  value      pretrained: the mean best validation accuracy of seed 1 of 188 and of 17 classes; from scratch: seed 1
  diverged   a cell whose training loss became non-finite in any attempt (DIVERGED record, or a non-finite loss in its
             log or in a .partial attempt's): the rate counts as lower-scoring than any rate that trained, whatever
             its best check before the blow-up. Crashes, evictions and storage failures are retried and never count.
             A cell of the paper's networks that diverges at the row's recorded rate fails that rate for the row.
  schedule   every attempt's manifest must have the row's rate and its size's schedule (FIXED_STEPS steps in checks of
             FIXED_CHECK); any other cell stops the read-out (FATAL)
  selected   the rate with the highest value; settled once it has a lower-scoring rate on each side, or it is the
             lowest rate tried and that is the floor (FIXED_SEARCH_FLOOR, 1e-5)

    python3 experiments/FT/rate_search.py SCAN.json              # one line per row
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FT_ROOT = "/data/results/ft_v2/"
DIVERGED = "diverged"


def builder():
    spec = importlib.util.spec_from_file_location("build_ft_budget_jobs", REPO / "scripts/build_ft_budget_jobs.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    return b


def _rel(path: str) -> str:
    if not path.startswith(FT_ROOT):
        raise SystemExit(f"FATAL: {path} is not under {FT_ROOT}")
    return path[len(FT_ROOT):]


def cell_state(rec: dict | None, lr: str, n: int, b) -> tuple:
    """(value: float | DIVERGED | None while it runs, [schedule errors]) of one cell's scan record."""
    if rec is None:
        return None, []
    errs = []
    for a in rec["attempts"]:
        m = a.get("manifest")
        if m is None:
            continue
        spe, ep = (int(m[k]) if m.get(k) not in (None, "") else 0 for k in ("steps_per_epoch", "max_epochs"))
        if m.get("lr") != lr or spe != b.FIXED_CHECK[n] or ep * spe != b.FIXED_STEPS[n]:
            errs.append(f"{a['dir']}: lr {m.get('lr')}, {ep} checks of {spe} steps; the row's rule is lr {lr}, "
                        f"{b.FIXED_STEPS[n]} steps in checks of {b.FIXED_CHECK[n]}")
    if rec.get("superseded"):
        errs.append(f"{rec['attempts'][-1]['dir']}: SUPERSEDED, never read")
    if rec.get("diverged_record") or any(a.get("first_nonfinite_check") is not None for a in rec["attempts"]):
        return DIVERGED, errs
    if rec["done"]:
        return rec["attempts"][-1]["best_metric"], errs
    return None, errs


def evaluate(scan: dict, b=None) -> list:
    """One dict per row of build_ft_budget_jobs.fixed_search_plan, in its order."""
    b = b or builder()
    cells = scan["cells"]
    paper = {}
    for c in b.paper_fixed_cells():
        if c["task"] != "topkin":
            paper.setdefault((c["task"], c["n"], "pre" if c["init"] else "scratch"), []).append(c)
    rows = []
    for (task, n, kind), rates in b.fixed_search_plan().items():
        vals, errs = {}, []
        for lr, dirs in rates.items():
            states = []
            for d in dirs:
                v, e = cell_state(cells.get(_rel(d)), lr, n, b)
                states.append(v)
                errs += e
            vals[lr] = DIVERGED if DIVERGED in states else (None if None in states else sum(states) / len(states))
        recorded, original = b.row_rate(task, n, kind), b.original_rate(task, kind)
        rerun_diverged = []
        if recorded is not None:     # every paper cell at the recorded rate: one divergence fails the rate for the row
            for c in paper.get((task, n, kind), []):
                v, e = cell_state(cells.get(_rel(c["out"])), recorded, n, b)
                errs += e
                if v == DIVERGED:
                    rerun_diverged.append(_rel(c["out"]))
            if rerun_diverged and recorded in vals:
                vals[recorded] = DIVERGED
        num = {lr: v for lr, v in vals.items() if isinstance(v, float)}
        # ties at full precision (PI 2026-11-22): rates exactly equal at the top keep the rate the row was first run at
        # (no rerun; the methods then needs a clause, so the row says so); a tie without it is flagged for the PI
        tied = [lr for lr, v in num.items() if v == max(num.values())] if num else []
        best = (original if original in tied else tied[0]) if tied else None
        if len(tied) > 1:
            errs.append(f"exact tie at {num[best]!r} between {tied}: kept {best}" + (
                " (the rate the row was first run at, no rerun; the methods needs a clause)" if best == original
                else " (the original rate is not among them: ask the PI)"))
        complete = all(v is not None for v in vals.values())
        order = list(vals)
        below = lambda lr: vals[lr] == DIVERGED or num.get(lr, math.inf) < num[best] or (lr in tied and lr != best)  # noqa: E731
        lower = [lr for lr in order if best and float(lr) < float(best) and below(lr)]
        upper = [lr for lr in order if best and float(lr) > float(best) and below(lr)]
        if not complete:
            status = "running"
        elif best is None:
            status = "every rate diverged"
        elif lower and upper:
            status = "settled"
        elif best == order[0] and best == b.FIXED_SEARCH_FLOOR:
            status = "settled at the floor"
        elif best == order[0]:
            status = "edge: lowest rate"
        else:
            status = "edge: highest rate"
        settled = status.startswith("settled")
        if settled and recorded != best:
            errs.append(f"selected {best}, recorded {recorded}: record the row's rate in FIXED_ROW_RATE"
                        + ("_SCRATCH" if kind == "scratch" else ""))
        if not settled and recorded is not None:
            errs.append(f"recorded {recorded} but the row is {status}")
        rows.append({"task": task, "n": n, "kind": kind, "values": vals, "best": best, "status": status, "tied": tied,
                     "recorded": recorded, "original": original, "rerun_diverged": rerun_diverged, "issues": errs})
    return rows


def fmt(row: dict) -> str:
    v = " ".join(f"{lr}:{'div' if x == DIVERGED else ('...' if x is None else f'{x:.4f}')}" for lr, x in row["values"].items())
    rec = f" recorded {row['recorded']}" if row["recorded"] else ""
    return f"{row['task']:4s} {row['n']:>7d} {row['kind']:7s} {row['status']:22s} best {row['best']}{rec} | {v}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rows = evaluate(json.loads(pathlib.Path(argv[0]).read_text()))
    for r in rows:
        print(fmt(r))
        for e in r["issues"]:
            print("    ISSUE", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
