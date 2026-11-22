#!/usr/bin/env python3
"""Projection of the rate search and its reruns in the PI's order (2026-11-22: the 1e6 rows, then the 1e3-1e5 search
cells, then part A of every 1e3-1e5 rerun, then part B), built from the sources of truth, never from a typed list:
  - the guard's launch list (job order) and each job's spec (its cells: task, network, size, folder);
  - the cluster: each job's done, failed (a diverged cell, not owed), running and pending indices, and each running
    pod's start;
  - the rate search (experiments/FT/rate_search.py on the latest scan): every row not settled, the search step it
    still needs at an edge, and the rerun it needs when its best is not its original rate (cells from the builder:
    fixed_cells, split A/B as fixed_rate_parts);
  - cell hours: the median finished attempt of the same task, size and kind on the same GPU pool
    (experiments/FIGS/data/v2/budget/ft_attempts.json, joined to cells by folder, and the cluster's finished pods).
Greedy list scheduling on a fixed number of GPUs per pool. The assumptions are printed with the result. Reads only.
    python3 scripts/project_schedule.py LAUNCH_LIST SCAN.json --gen 25 --a100 3
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import heapq
import json
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "FT"))
import build_ft_budget_jobs as B  # noqa: E402
import rate_search as RS  # noqa: E402

CELL_RE = re.compile(r"^ *(\d+)\) TASK=(\w+); INIT=([\w-]*);.*?N=(\d+);.*?LR=([\w.-]+);.*?CELL=(\S+?);;", re.M)
UTC = dt.timezone.utc
T = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)  # noqa: E731
CUTOFF = dt.datetime(2026, 11, 25, 18, 0, tzinfo=UTC)                           # PI 2026-11-22: the B cut-off
GROUPS = ("1e6", "search", "A", "B")


def neighbour(lr: str, up: bool) -> str:
    """The next rate of the search's 1-3 ladder (1e-3 -> 3e-3 -> 1e-2; 1e-3 -> 3e-4)."""
    m, e = lr.split("e")
    m, e = int(m), int(e)
    if up:
        return f"3e{e}" if m == 1 else f"1e{e + 1}"
    return f"3e{e - 1}" if m == 1 else f"1e{e}"


def kind_of(init: str) -> str:
    return "pre" if init else "scratch"


def group_of(task: str, n: int, init: str, path: str) -> str:
    if n >= 1_000_000:
        return "1e6"
    if "/fixed_sweep" in path:
        return "search"
    return "A" if not init or init.startswith(B.FIXED_RATE_A_INITS) else "B"


def spec_cells(spec: pathlib.Path) -> dict:
    return {int(m[1]): {"task": m[2].removesuffix("kin"), "init": m[3], "n": int(m[4]), "lr": m[5], "path": m[6]}
            for m in CELL_RE.finditer(spec.read_text())}


def kubectl_json(*args) -> dict:
    return json.loads(subprocess.run(["kubectl", "get", "-n", "cms-ml", *args, "-o", "json"], check=True,
                                     capture_output=True, text=True).stdout)


def indices(s: str) -> set:
    out = set()
    for part in filter(None, (s or "").split(",")):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("launch_list")
    ap.add_argument("scan")
    ap.add_argument("--gen", type=int, required=True, help="general GPUs assumed held")
    ap.add_argument("--a100", type=int, required=True, help="A100s assumed held (at most 8)")
    a = ap.parse_args(argv)
    now = dt.datetime.now(UTC).replace(microsecond=0)

    # every cell folder the specs name -> (task, size, kind), to read hours by folder
    known = {}
    for spec in (ROOT / "experiments" / "FT" / "k8s" / "budget").glob("job-ft-budget-fixed-*-raunav.yaml"):
        for c in spec_cells(spec).values():
            known[c["path"].removeprefix("/data/results/ft_v2/")] = (c["task"], c["n"], kind_of(c["init"]))
    hours = collections.defaultdict(list)
    att = json.loads((ROOT / "experiments/FIGS/data/v2/budget/ft_attempts.json").read_text())["cells"]
    for path, atts in att.items():
        for x in atts:
            if path in known and x["end_kind"] == "done":
                pool = "a100" if "A100" in (x["gpu"] or "") else "gen"
                hours[known[path] + (pool,)].append((T(x["end"]) - T(x["start"])).total_seconds() / 3600)

    jobs = {j["metadata"]["name"]: j for j in kubectl_json("jobs")["items"]}
    pods = collections.defaultdict(dict)
    for p in kubectl_json("pods")["items"]:
        job = p["metadata"].get("labels", {}).get("job-name", "")
        idx = p["metadata"].get("annotations", {}).get("batch.kubernetes.io/job-completion-index")
        if job and idx is not None:
            pods[job][int(idx)] = p

    order, plan = [], {}
    for line in pathlib.Path(a.launch_list).read_text().split():
        name, spec = line.split("|")[:2]
        if name in plan:
            continue
        sp = ROOT / spec
        pool = "a100" if "nvidia.com/a100" in sp.read_text() else "gen"
        cells = spec_cells(sp)
        st = jobs.get(name, {}).get("status", {})
        finished = indices(st.get("completedIndexes")) | indices(st.get("failedIndexes"))
        for i, c in cells.items():
            p = pods[name].get(i)
            if i in finished:
                cs = (p or {}).get("status", {}).get("containerStatuses") or [{}]
                term = cs[0].get("state", {}).get("terminated")
                if term and i in indices(st.get("completedIndexes")):
                    hours[(c["task"], c["n"], kind_of(c["init"]), pool)].append(
                        (T(term["finishedAt"]) - T(term["startedAt"])).total_seconds() / 3600)
                continue
            run = p and p["status"].get("phase") == "Running"
            start = T(p["status"]["startTime"]) if run else None
            plan[(name, i)] = {**c, "pool": pool, "start": start, "group": group_of(c["task"], c["n"], c["init"], c["path"]),
                               "job": name, "deps": set()}
        order.append(name)

    # the rows still searching: what each still needs, assuming its current best holds
    assume = ["each unsettled row's current best holds: its pending neighbours score lower",
              "a row at the edge of the rates tried needs one more step (3x past it), which then scores lower",
              "a rerun starts when its row's search cells have all finished"]
    future = []
    for r in RS.evaluate(json.loads(pathlib.Path(a.scan).read_text())):
        task, n, kind = r["task"], r["n"], r["kind"]
        row_cells = {k for k, c in plan.items() if (c["task"], c["n"], kind_of(c["init"])) == (task, n, kind)
                     and "/fixed_sweep" in c["path"]}
        if r["status"].startswith("settled"):
            if r["best"] != r["recorded"]:
                future.append(("RECORD", f"{task} {n} {kind}: settled at {r['best']}, not recorded"))
            continue
        best = r["best"]
        tried = list(r["values"])
        step = []
        if r["status"].startswith("edge"):
            nxt = neighbour(best, "highest" in r["status"])
            if nxt not in tried:
                inits = B.FIXED_SEARCH_INITS[kind]
                step = [{"task": task, "n": n, "init": x, "lr": nxt, "path": "/fixed_sweep/future", "pool": "gen",
                         "start": None, "group": "1e6" if n >= 1_000_000 else "search", "job": f"(step {task} {n} {kind})",
                         "deps": set()} for x in inits]
        if best == B.original_rate(task, kind):
            future += [("STEP", c) for c in step]
            continue
        if kind == "pre":
            rr = [c for c in B.fixed_cells({n}) if c["task"] in (task, f"{task}kin") and c["init"]]
        else:
            rr = [{"task": task, "init": "", "n": n} for _ in B.SCRATCH_SEEDS]
        dep_keys = row_cells | {("future", id(c)) for c in step}
        for c in rr:
            g = "1e6" if n >= 1_000_000 else group_of(task, n, c["init"], "/fixed_rate/")
            future.append(("RERUN", {"task": task, "n": n, "init": c["init"], "lr": best, "path": "/fixed_rate/future",
                                     "pool": "gen", "start": None, "group": g, "deps": dep_keys,
                                     "job": f"(rerun {task} {n} {kind} at {best}{'' if g == '1e6' else ', ' + g})"}))
        future += [("STEP", c) for c in step]

    # the queue: jobs in list order, each job's indices in order; future cells after the last job of their group
    queue = sorted(plan, key=lambda k: (order.index(k[0]), k[1]))
    for what, c in future:
        if what == "RECORD":
            print("NEEDS ACTION:", c)
            continue
        key = ("future", id(c))
        plan[key] = c
        last = max((j for j, k in enumerate(queue) if GROUPS.index(plan[k]["group"]) <= GROUPS.index(c["group"])),
                   default=-1)
        queue.insert(last + 1, key)

    def est(c):
        for key in ((c["task"], c["n"], kind_of(c["init"]), c["pool"]),
                    (c["task"], c["n"], kind_of(c["init"]), "a100" if c["pool"] == "gen" else "gen")):
            if hours.get(key):
                f = 1.0 if key[3] == c["pool"] else (1.6 if c["pool"] == "gen" else 1 / 1.6)
                return statistics.median(hours[key]) * f
        same_n = [h for k, v in hours.items() if k[1] == c["n"] and k[3] == c["pool"] for h in v]
        return statistics.median(same_n)

    # simulate: running cells keep going; a pool starts a cell only while it runs fewer than its GPUs
    cap = {"gen": a.gen, "a100": a.a100}
    busy = collections.Counter()
    events, end_at = [], {}
    t = now
    for k in queue:
        c = plan[k]
        if c["start"]:
            left = max(est(c) - (now - c["start"]).total_seconds() / 3600, 0.25)
            heapq.heappush(events, (now + dt.timedelta(hours=left), k))
            busy[c["pool"]] += 1
    waiting = [k for k in queue if not plan[k]["start"]]
    gpu_h = sum(max(est(plan[k]) - (now - plan[k]["start"]).total_seconds() / 3600, 0.25) if plan[k]["start"]
                else est(plan[k]) for k in queue)
    while waiting or events:
        started = True
        while started:
            started = False
            for k in waiting:
                c = plan[k]
                if busy[c["pool"]] < cap[c["pool"]] and all(d in end_at and end_at[d] <= t for d in c["deps"]):
                    heapq.heappush(events, (t + dt.timedelta(hours=est(c)), k))
                    busy[c["pool"]] += 1
                    waiting.remove(k)
                    started = True
                    break
        if not events:
            print("STUCK:", [plan[k]["job"] for k in waiting][:5])
            break
        t, k = heapq.heappop(events)
        end_at[k] = t
        busy[plan[k]["pool"]] -= 1

    fmt = lambda x: x.strftime("%a %d %H:%M UTC")  # noqa: E731
    print(f"now {fmt(now)}; GPUs assumed: {a.gen} general, {a.a100} A100")
    print("assumptions: " + "; ".join(assume) + "; hours: median finished attempt, same task/size/kind/pool "
          "(other pool x1.6 or /1.6, else same size)")
    left = lambda k: (max(est(plan[k]) - (now - plan[k]["start"]).total_seconds() / 3600, 0.25)  # noqa: E731
                      if plan[k]["start"] else est(plan[k]))
    jc2b6 = sum(left(k) for k in queue if (plan[k]["task"], plan[k]["n"]) == ("jc2b", 1_000_000))
    print(f"remaining GPU-hours {gpu_h:.0f}: JetClass-II 1e6 {jc2b6:.0f} (counted apart from the 2,000 ceiling), "
          f"the rest {gpu_h - jc2b6:.0f}")
    for key in sorted({(c["task"], c["n"], kind_of(c["init"]), c["pool"]) for c in plan.values()}):
        print(f"  hours {' '.join(map(str, key)):28s} {est(dict(task=key[0], n=key[1], init='x' if key[2] == 'pre' else '', pool=key[3])):5.2f}"
              f"  (finished attempts measured: {len(hours.get(key, []))})")
    by_job = collections.defaultdict(list)
    for k in queue:
        by_job[plan[k]["job"]].append(k)
    for job, ks in by_job.items():
        n_run = sum(1 for k in ks if plan[k]["start"])
        print(f"  {plan[ks[0]]['group']:6s} {job:48s} cells left {len(ks):3d} (running {n_run:2d})  done {fmt(max(end_at[k] for k in ks))}")
    for g in GROUPS:
        ks = [k for k in queue if plan[k]["group"] == g]
        if ks:
            e = max(end_at[k] for k in ks)
            flag = "" if g not in ("A", "B") else ("  BEFORE the cut-off" if e <= CUTOFF else "  AFTER the cut-off (Wed 25 18:00 UTC)")
            print(f"group {g:6s} {len(ks):3d} cells, {sum(est(plan[k]) for k in ks):5.0f} GPU-h at most, last ends {fmt(e)}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
