"""scripts/release_v2.py against a fake kubectl on PATH: it releases suspended grid
jobs in grid order while fewer than --max-pending grid pods are pending on the job's
GPU product, one patch per job; with --create-tier it creates a tier's missing jobs, suspended, from the
committed specs with `kubectl create`; and it calls kubectl for nothing else."""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("release_v2", "scripts/release_v2.py")
B = _load("build_mtx_launch_release", "scripts/build_mtx_launch.py")

# Answers `get jobs|pods -n NS -o json` from a state file; a patch updates the job's
# spec and gives it a Pending pod, as the Job controller would; `create -f` adds the
# file's job (no pod while it is suspended) and fails if it exists. Anything else fails.
FAKE = """#!{python}
import json, os, sys
path = os.environ["FAKE_KUBE_STATE"]
st = json.load(open(path))
args = sys.argv[1:]
with open(os.environ["FAKE_KUBE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "get" and args[1] in ("jobs", "pods"):
    print(json.dumps({{"items": st[args[1]]}}))
elif args[:2] == ["patch", "job"]:
    job = next(j for j in st["jobs"] if j["metadata"]["name"] == args[2])
    job["spec"].update(json.loads(args[args.index("-p") + 1])["spec"])
    st["pods"].append({{"metadata": {{"labels": {{"job-name": args[2]}}}}, "status": {{"phase": "Pending"}}}})
    json.dump(st, open(path, "w"))
elif args[:2] == ["create", "-f"]:
    import yaml
    d = yaml.safe_load(open(args[2]))
    if any(j["metadata"]["name"] == d["metadata"]["name"] for j in st["jobs"]):
        sys.exit(1)
    st["jobs"].append({{"metadata": {{"name": d["metadata"]["name"]}}, "spec": {{"suspend": d["spec"].get("suspend")}},
                       "status": {{}}}})
    json.dump(st, open(path, "w"))
else:
    sys.exit(1)
"""


def _job(name, suspend, cond=None):
    return {"metadata": {"name": name}, "spec": {"suspend": suspend},
            "status": {"conditions": [{"type": cond, "status": "True"}] if cond else []}}


def _pod(job, phase):
    return {"metadata": {"labels": {"job-name": job}}, "status": {"phase": phase}}


@pytest.fixture
def kube(tmp_path, monkeypatch):
    """A fake cluster: grid job 0 complete, 1 running, 2 not created, 3-6 suspended,
    7 released with a pending pod; another user's suspended job and pending pod."""
    names = R.grid_order()
    st = {"jobs": [_job(names[0], False, "Complete"), _job(names[1], False),
                   *[_job(n, True) for n in names[3:7]], _job(names[7], False),
                   _job("someone-else", True)],
          "pods": [_pod(names[1], "Running"), _pod(names[7], "Pending"), _pod("someone-else", "Pending")]}
    (tmp_path / "state.json").write_text(json.dumps(st))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "kubectl").write_text(FAKE.format(python=sys.executable))
    (bin_ / "kubectl").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KUBE_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("FAKE_KUBE_LOG", str(tmp_path / "calls.log"))
    return names, tmp_path / "calls.log"


def _calls(log):
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def _patched(log):
    return [c[2] for c in _calls(log) if c[0] == "patch"]


def test_the_order_is_tier_then_run_index_then_the_registry():
    names = R.grid_order()
    arms = B.v2_arms()
    want = [B.v2_job_name(a["name"], r)
            for t in sorted({int(a.get("tier", 1)) for a in arms})
            for r in range(1, max(int(a["runs"]) for a in arms) + 1)
            for a in arms if int(a.get("tier", 1)) == t and r <= int(a["runs"])]
    assert names == want and len(names) == sum(int(a["runs"]) for a in arms)
    tier1 = [a for a in arms if int(a.get("tier", 1)) == 1]
    assert names[:len(tier1)] == [B.v2_job_name(a["name"], 1) for a in tier1]


def test_apply_releases_in_order_until_the_pending_cap(kube, monkeypatch, capsys):
    names, log = kube
    sleeps = []
    # the script's own `time`, not the module every thread shares: a daemon thread an
    # earlier test left in this process (pretrain_v2's MemMonitor) also calls time.sleep
    monkeypatch.setattr(R, "time", types.SimpleNamespace(sleep=sleeps.append))
    assert R.main(["--apply"]) == 0
    # one grid pod pending, cap 3: two releases, the first suspended jobs in grid order
    assert _patched(log) == names[3:5] and sleeps == [R.RELEASE_GAP_S] * 2
    out = capsys.readouterr().out
    assert f"{names[2]:44s} not created" in out and f"{names[0]:44s} complete" in out
    assert "released: 0 pending, 1 running" in out


def test_kubectl_is_asked_for_nothing_but_reads_and_unsuspending_grid_jobs(kube, monkeypatch):
    names, log = kube
    monkeypatch.setattr(R, "time", types.SimpleNamespace(sleep=lambda s: None))
    R.main(["--apply", "--max-pending", "10"])
    assert _patched(log) == names[3:7]                      # every suspended grid job, none other
    for c in _calls(log):
        if c[0] == "get":
            assert c == ["get", c[1], "-n", "cms-ml", "-o", "json"] and c[1] in ("jobs", "pods")
        else:
            assert c == ["patch", "job", c[2], "-n", "cms-ml", "--type", "merge",
                         "-p", '{"spec":{"suspend":false}}'] and "raunav" in c[2] and c[2] in names


def test_nothing_is_released_at_the_cap(kube, monkeypatch):
    names, log = kube
    monkeypatch.setattr(R, "time", types.SimpleNamespace(sleep=lambda s: pytest.fail("slept without releasing")))
    assert R.main(["--apply", "--max-pending", "1"]) == 0
    assert _patched(log) == []


def test_the_cap_is_per_gpu_product(kube, monkeypatch):
    """Three 3090 pods pending stop the 3090 jobs, not the first L40 job (run index 4)."""
    names, log = kube
    gpu = R.gpu_of()
    l40 = [n for n in names if gpu[n] != gpu[names[0]]]
    assert l40 and all(gpu[n] == gpu[l40[0]] for n in l40)
    path = log.parent / "state.json"
    st = json.loads(path.read_text())
    st["pods"] += [_pod(names[3], "Pending"), _pod(names[4], "Pending")]   # with names[7]: three on 3090
    for j in st["jobs"]:
        if j["metadata"]["name"] in (names[3], names[4]):
            j["spec"]["suspend"] = False
    st["jobs"] += [_job(l40[0], True), _job(l40[1], True)]
    path.write_text(json.dumps(st))
    sleeps = []
    monkeypatch.setattr(R, "time", types.SimpleNamespace(sleep=sleeps.append))
    assert R.main(["--apply", "--max-pending", "3"]) == 0
    assert _patched(log) == l40[:2] and sleeps == [R.RELEASE_GAP_S] * 2   # names[5], names[6] wait
    assert R.main(["--apply", "--max-pending", "3"]) == 0 and _patched(log) == l40[:2]


def test_the_default_is_a_dry_run(kube):
    names, log = kube
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_v2.py")],
                       capture_output=True, text=True, env=os.environ)
    assert r.returncode == 0, r.stderr
    assert _patched(log) == [] and {c[0] for c in _calls(log)} == {"get"}
    plan = r.stdout[r.stdout.index("would release"):].split()
    assert names[3] in plan and names[4] in plan and names[5] not in plan


def _created(log):
    return [c[2] for c in _calls(log) if c[0] == "create"]


def _tier(n):
    return [name for t, name in R.grid_rows() if t == n]


def test_create_makes_a_tiers_missing_jobs_suspended_in_grid_order(kube):
    names, log = kube
    want = [n for n in _tier(1) if n not in {names[0], names[1], *names[3:8]}]
    assert names[2] in want and R.main(["--create-tier", "1", "--apply"]) == 0
    assert _created(log) == [str(R.GRID_DIR / f"job-{n}.yaml") for n in want]
    for c in _calls(log):
        assert c == ["get", "jobs", "-n", "cms-ml", "-o", "json"] or c == [
            "create", "-f", c[2], "-n", "cms-ml"], c                       # never apply, patch or delete
    st = json.loads((log.parent / "state.json").read_text())
    made = {j["metadata"]["name"]: j for j in st["jobs"]}
    assert all(made[n]["spec"]["suspend"] is True for n in want)
    assert made[names[1]]["spec"]["suspend"] is False                       # a released job is left alone
    assert len(st["pods"]) == 3                                             # nothing started
    assert R.main(["--create-tier", "1", "--apply"]) == 0 and len(_created(log)) == len(want)


def test_create_without_apply_only_prints(kube, capsys):
    names, log = kube
    assert R.main(["--create-tier", "2"]) == 0
    assert {c[0] for c in _calls(log)} == {"get"}
    out = capsys.readouterr().out
    tier2 = _tier(2)
    assert f"tier 2: 0 of {len(tier2)} jobs exist; would create {len(tier2)}, suspended:" in out
    assert [ln.strip() for ln in out.splitlines() if ln.strip().startswith("experiments/")] == [
        f"experiments/MTX/k8s/v2/grid/job-{n}.yaml" for n in tier2]


def test_create_refuses_the_tier_if_any_spec_would_start_its_job(kube, monkeypatch, tmp_path):
    names, log = kube
    grid = tmp_path / "grid"
    grid.mkdir()
    tier2 = _tier(2)
    for n in tier2:
        (grid / f"job-{n}.yaml").write_text((R.GRID_DIR / f"job-{n}.yaml").read_text())
    last = grid / f"job-{tier2[-1]}.yaml"
    assert last.read_text().count("  suspend: true\n") == 1
    last.write_text(last.read_text().replace("  suspend: true\n", ""))
    monkeypatch.setattr(R, "GRID_DIR", grid)
    with pytest.raises(SystemExit, match="not the suspended grid job"):
        R.main(["--create-tier", "2", "--apply"])
    assert _created(log) == []
    with pytest.raises(SystemExit, match="no tier 9"):
        R.main(["--create-tier", "9", "--apply"])
