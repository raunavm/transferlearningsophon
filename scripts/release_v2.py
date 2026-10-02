#!/usr/bin/env python3
"""Release the suspended v2 pretraining jobs a few at a time.

Every grid spec (scripts/build_mtx_launch.py --v2) is created suspended, so
creating the grid starts nothing. This script reads the grid order -- tier, then
run index, then the order of the arms in configs/arms/v2_grid.json -- asks
kubectl which of those jobs exist, which are suspended and which have pods
pending or running, and prints it. With --apply it then unsuspends jobs in that
order, one `kubectl patch` per job and RELEASE_GAP_S apart, while fewer than
--max-pending pods of the grid are pending.

With --create-tier N it instead creates the jobs of tier N that do not exist yet,
in grid order, one `kubectl create -f` per committed grid spec (suspended), and
skips every job that exists. Never `kubectl apply` a grid spec: on a job already
released it sets suspend back to true, and the Job controller kills the running pod.

It reads, creates missing grid jobs and unsuspends grid jobs; it never deletes or
changes anything else. Without --apply it only prints.

Run:  python3 scripts/release_v2.py [--max-pending 3] [--apply]
      python3 scripts/release_v2.py --create-tier 1 [--apply]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAMESPACE = "cms-ml"
RELEASE_GAP_S = 75
UNSUSPEND = '{"spec":{"suspend":false}}'
GRID_DIR = ROOT / "experiments" / "MTX" / "k8s" / "v2" / "grid"


def grid_rows() -> list:
    """[(tier, job name)] of the grid, in the order the jobs are released."""
    spec = importlib.util.spec_from_file_location("build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    rows = [(int(a.get("tier", 1)), run, i, b.v2_job_name(a["name"], run))
            for i, a in enumerate(b.v2_arms()) for run in range(1, int(a["runs"]) + 1)]
    return [(tier, name) for tier, *_, name in sorted(rows)]


def grid_order() -> list:
    """The grid's job names, in the order they are released."""
    return [name for _, name in grid_rows()]


def _get(kind: str) -> list:
    out = subprocess.run(["kubectl", "get", kind, "-n", NAMESPACE, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["items"]


def cluster_state(names: list) -> tuple:
    """({job name: job} for the grid's jobs that exist, {job name: [pod phase]})."""
    want = set(names)
    jobs = {j["metadata"]["name"]: j for j in _get("jobs") if j["metadata"]["name"] in want}
    phases = {}
    for p in _get("pods"):
        job = p["metadata"].get("labels", {}).get("job-name")
        if job in want:
            phases.setdefault(job, []).append(p["status"].get("phase"))
    return jobs, phases


def status(job, phases: list) -> str:
    if job is None:
        return "not created"
    done = {c["type"] for c in job.get("status", {}).get("conditions", []) if c.get("status") == "True"}
    if done & {"Complete", "Failed"}:
        return "complete" if "Complete" in done else "failed"
    if job["spec"].get("suspend"):
        return "suspended"
    return f"released: {phases.count('Pending')} pending, {phases.count('Running')} running"


def survey(names: list) -> tuple:
    """Print every grid job's status; return (pods pending, suspended jobs in release order)."""
    jobs, phases = cluster_state(names)
    pending = sum(p.count("Pending") for p in phases.values())
    queue = []
    for name in names:
        s = status(jobs.get(name), phases.get(name, []))
        print(f"  {name:44s} {s}")
        if s == "suspended":
            queue.append(name)
    print(f"{pending} grid pods pending, {len(queue)} jobs suspended")
    return pending, queue


def release(name: str) -> None:
    assert "raunav" in name, name
    cmd = ["kubectl", "patch", "job", name, "-n", NAMESPACE, "--type", "merge", "-p", UNSUSPEND]
    print("+ " + " ".join(cmd[:-1]) + f" '{UNSUSPEND}'", flush=True)
    subprocess.run(cmd, check=True)


def spec_file(name: str) -> pathlib.Path:
    """The committed grid spec of a job, refused unless it creates that job, suspended."""
    path = GRID_DIR / f"job-{name}.yaml"
    d = yaml.safe_load(path.read_text())
    if d["metadata"]["name"] != name or "raunav" not in name or d["spec"].get("suspend") is not True:
        raise SystemExit(f"{path}: not the suspended grid job {name}")
    return path


def create(path: pathlib.Path) -> None:
    cmd = ["kubectl", "create", "-f", str(path), "-n", NAMESPACE]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def create_missing(tier: int, apply: bool) -> None:
    """Create the jobs of `tier` that do not exist, in grid order; every spec is checked
    before the first is created."""
    names = [name for t, name in grid_rows() if t == tier]
    if not names:
        raise SystemExit(f"the grid has no tier {tier}")
    have = {j["metadata"]["name"] for j in _get("jobs")}
    todo = [spec_file(name) for name in names if name not in have]
    print(f"tier {tier}: {len(names) - len(todo)} of {len(names)} jobs exist; "
          f"{'creating' if apply else 'would create'} {len(todo)}, suspended:")
    for path in todo:
        if apply:
            create(path)
        else:
            print(f"  {path.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-pending", type=int, default=3,
                    help="release only while fewer grid pods than this are pending")
    ap.add_argument("--create-tier", type=int, default=None, metavar="N",
                    help="create the missing jobs of tier N (suspended) instead of releasing")
    ap.add_argument("--apply", action="store_true", help="create or patch; without it, only print")
    a = ap.parse_args(argv)
    if a.create_tier is not None:
        create_missing(a.create_tier, a.apply)
        return 0
    names = grid_order()
    pending, queue = survey(names)
    if not a.apply:
        print(f"would release, {RELEASE_GAP_S} s apart, while fewer than {a.max_pending} pods are pending:")
        for name in queue[:max(a.max_pending - pending, 0)]:
            print(f"  {name}")
        return 0
    # Each release is followed by a fresh survey: the released job's pod counts as pending
    # until it is scheduled, and pods of jobs released earlier may have started meanwhile.
    while queue and pending < a.max_pending:
        release(queue[0])
        time.sleep(RELEASE_GAP_S)
        pending, queue = survey(names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
