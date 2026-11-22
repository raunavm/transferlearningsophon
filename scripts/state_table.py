#!/usr/bin/env python3
"""The 24-row table of docs/STATE.md: every searched row (task, size, pretrained / from scratch) from the rate search
(experiments/FT/rate_search.py on a scan), its original rate, and its rerun: none needed, needed but not yet built, or
the jobs that hold it with cells done of total (the cluster's completed and failed indices). Markdown. Reads only.
    python3 scripts/state_table.py SCAN.json
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "FT"))
import build_ft_budget_jobs as B  # noqa: E402
import rate_search as RS  # noqa: E402
from project_schedule import indices, spec_cells  # noqa: E402


def main(argv=None) -> int:
    scan = json.loads(pathlib.Path((argv or sys.argv[1:])[0]).read_text())
    jobs = {j["metadata"]["name"]: j for j in json.loads(subprocess.run(
        ["kubectl", "get", "jobs", "-n", "cms-ml", "-o", "json"], check=True, capture_output=True, text=True).stdout)["items"]}
    # rerun cells by folder -> (job, index): every fixed-rate spec; a job no longer on the cluster is marked GONE (its
    # cells' DONE markers on /data are then the record)
    where = {}
    for spec in (ROOT / "experiments/FT/k8s/budget").glob("job-ft-budget-fixed-rate*-raunav.yaml"):
        name = spec.name.removeprefix("job-").removesuffix(".yaml")
        for i, c in spec_cells(spec).items():
            if name in jobs or c["path"] not in where:
                where[c["path"]] = (name, i)
    rate_cells = B.fixed_rate_cells()
    print("| task | size | kind | status | best | original | rerun |")
    print("|---|---|---|---|---|---|---|")
    for r in RS.evaluate(scan):
        task, n, kind = r["task"], r["n"], r["kind"]
        orig = B.original_rate(task, kind)
        if r["best"] == orig and r["status"].startswith("settled"):
            rerun = "none (best is the original rate)"
        elif (task, n, kind) not in B.FIXED_RATE_ORDER:
            rerun = "not built (row not settled)" if not r["status"].startswith("settled") else "NOT BUILT: settled, needs recording"
        else:
            cells = [c for c in rate_cells if (c["task"].removesuffix("kin"), c["n"], "pre" if c["init"] else "scratch")
                     == (task, n, kind)]
            per_job = {}
            for c in cells:
                job, i = where.get(c["out"], ("NO JOB", -1))
                st = jobs.get(job, {}).get("status", {})
                done = i in indices(st.get("completedIndexes")) | indices(st.get("failedIndexes"))
                d, t = per_job.get(job, (0, 0))
                per_job[job] = (d + done, t + 1)
            rerun = "; ".join(f"{j.removeprefix('ft-budget-fixed-rate-').removesuffix('-raunav')} "
                              + (f"{d}/{t}" if j in jobs else f"GONE from the cluster ({t} cells)") for j, (d, t) in per_job.items())
        vals = " ".join(f"{lr}:{'div' if x == RS.DIVERGED else ('…' if x is None else f'{x:.4f}')}" for lr, x in r["values"].items())
        print(f"| {task} | {n:.0e} | {kind} | {r['status']} | {r['best']} ({vals}) | {orig} | {rerun} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
