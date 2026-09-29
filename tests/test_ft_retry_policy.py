"""Retries that survive a flaky cluster (scripts/build_ft_jobs.py, 2026-09-29).

Pinned here, by running the emitted scripts under bash with the wave-3 stubs:
  * an attempt killed with its pod (eviction, a lost node) is not a failure;
  * two FAILED attempts of one cell, or six attempts of any kind, stop the job
    with the halt code the pod failure policy turns into FailJob;
  * a run whose training loss went to NaN fails its attempt instead of being
    marked DONE, and one retry is allowed;
and, on the specs: which carry the policy, and that every other launched spec
is untouched.
"""
import importlib.util
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("wave3_harness_retry", "tests/test_wave3_specs.py")
B = T.B
LEGS = "job-ft-legs-baseline-scratch-v2-raunav.yaml"
SHARD = "ft-legs-baseline-scratch-v2-raunav"
INITS = B.INITS_LATER["scratch-v2"]


@pytest.fixture(scope="module")
def refs():
    return B.build(B.PIN_REFS, wave3=True, bench_v2=True, later=["scratch-v2", "mpm-s1-v2"])


def _run(text, tmp_path, nan_in=None, pod=None):
    """The emitted script under bash on the data tree in `tmp_path`; the weaver
    stub writes train.log, with a NaN loss for the cells whose model prefix
    contains `nan_in`. A later pod of the same job gets its own stubs, `pod`."""
    stubs = tmp_path / pod if pod else tmp_path
    env, _ = T._shell_env(stubs, INITS)
    py = stubs / "bin" / "python3"
    old = "    P=$(next_after --model-prefix \"$@\")\n"
    assert old in py.read_text()
    py.write_text(py.read_text().replace(old, old + (
        '    L=$(next_after --log "$@")\n'
        '    if [ -n "${NAN_IN:-}" ] && [[ "$P" == *"${NAN_IN}"* ]]; then\n'
        '      echo "INFO: Train AvgLoss: nan, AvgAcc: 0.137" > "$L"\n'
        '    else echo "INFO: Train AvgLoss: 0.5, AvgAcc: 0.8" > "$L"; fi\n')))
    if nan_in:
        env["NAN_IN"] = nan_in
    return subprocess.run(["bash", "-c", T._redirect(text, stubs, tmp_path)], capture_output=True,
                          text=True, env=env, cwd=tmp_path, timeout=600)


def _cell(tmp_path, n=1000, s=1):
    return tmp_path / "data/results/ft/w2b/leg1/scratch-v2" / f"N{n}" / f"s{s}"


def _attempt(path: pathlib.Path, failed=False, traceback=False):
    path.mkdir(parents=True)
    (path / "stdout.log").write_text("epoch 3 ...\n" + ("Traceback (most recent call last):\n" if traceback else ""))
    if failed:
        (path / "ATTEMPT_FAILED").write_text("rc=1 pod=p node=n\n")


def _fail_mark(tmp_path):
    return tmp_path / "data/results/ft/w2b" / f"FAILED.{SHARD}"


# ------------------------------------------------------------------ the specs

def test_the_policy_is_on_exactly_the_specs_being_re_created(refs):
    v3 = B.build(B.PIN_W3, bench_v3=True)
    robust = {n for n, t in {**refs, **v3}.items() if "podFailurePolicy" in t}
    assert robust == {"job-ft-legs-baseline-scratch-v2-raunav.yaml",
                      "job-ft-legs-baseline-mpm-s1-v2-raunav.yaml",
                      "job-ft-legs-bench-baseline-mpm-s1-v2-raunav.yaml",
                      "job-ft-legs-bench-v3-last-c-raunav.yaml",
                      "job-ft-legs-bench-v3-last-e-raunav.yaml"}
    for name in robust:
        d = yaml.safe_load({**refs, **v3}[name])["spec"]
        assert d["backoffLimit"] >= B.ROBUST_BACKOFF
        assert d["podFailurePolicy"]["rules"] == [
            {"action": "FailJob",
             "onExitCodes": {"containerName": "main", "operator": "In", "values": [B.EXIT_HALT]}},
            {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}]
        assert d["template"]["spec"]["restartPolicy"] == "Never"   # the policy requires it
    # a job that ran, or is running, under the old logic keeps its record
    for name in ("job-ft-legs-bench-baseline-scratch-v2-raunav.yaml",
                 *(f"job-ft-legs-bench-v3-last-{s}-raunav.yaml" for s in "abd")):
        assert (K8S / name).read_text() == {**refs, **v3}[name], name


def test_no_other_spec_on_disk_carries_the_policy():
    have = {p.name for p in K8S.glob("job-*.yaml") if "podFailurePolicy" in p.read_text()}
    # every v2 spec carries it (audit 2026-09-29): its name says so
    v2 = {p.name for p in K8S.glob("job-ft-v2-*.yaml")} | {"job-ft-subsets-jc2-v2-raunav.yaml"}
    assert v2 <= have
    assert have - v2 == {"job-ft-legs-baseline-scratch-v2-raunav.yaml",
                    "job-ft-legs-baseline-mpm-s1-v2-raunav.yaml",
                    "job-ft-legs-bench-baseline-mpm-s1-v2-raunav.yaml",
                    "job-ft-legs-bench-v3-last-c-raunav.yaml",
                    "job-ft-legs-bench-v3-last-e-raunav.yaml"}


def test_the_unlaunched_self_supervised_groups_get_it_too():
    later = B.build(B.PIN_REFS, wave3=True, bench_v2=True, later=["mpm-s2", "mpm-s3"])
    assert all("podFailurePolicy" in t and f"HALT={B.EXIT_HALT}" in t for t in later.values())


# ------------------------------------------------------------------ the shell

def test_evicted_attempts_do_not_count_and_the_cell_finishes(refs, tmp_path):
    c = _cell(tmp_path)
    for i in range(4):
        _attempt(c.parent / f"s1.partial.{100 + i}")        # killed pods: no marker, no traceback
    _attempt(c)                                              # and the one in place
    r = _run(refs[LEGS], tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (c / "DONE").exists() and not _fail_mark(tmp_path).exists()


def test_two_failed_attempts_halt_with_the_policy_code(refs, tmp_path):
    c = _cell(tmp_path)
    _attempt(c.parent / "s1.partial.100", failed=True)
    _attempt(c.parent / "s1.partial.101", traceback=True)   # failed before the marker existed
    r = _run(refs[LEGS], tmp_path)
    assert r.returncode == B.EXIT_HALT, r.stdout[-2000:]
    assert "2 failed attempts" in _fail_mark(tmp_path).read_text()
    assert not (c / "DONE").exists()


def test_the_attempt_in_place_counts_as_well(refs, tmp_path):
    c = _cell(tmp_path)
    _attempt(c.parent / "s1.partial.100", failed=True)
    _attempt(c, failed=True)
    assert _run(refs[LEGS], tmp_path).returncode == B.EXIT_HALT


def test_six_attempts_of_any_kind_halt(refs, tmp_path):
    c = _cell(tmp_path)
    for i in range(5):
        _attempt(c.parent / f"s1.partial.{100 + i}")
    _attempt(c)
    r = _run(refs[LEGS], tmp_path)
    assert r.returncode == B.EXIT_HALT and "6 in all" in _fail_mark(tmp_path).read_text()


def test_a_failed_marker_from_an_earlier_pod_halts_at_the_top(refs, tmp_path):
    _fail_mark(tmp_path).parent.mkdir(parents=True)
    _fail_mark(tmp_path).write_text("x: 2 failed attempts\n")
    r = _run(refs[LEGS], tmp_path)
    assert r.returncode == B.EXIT_HALT and not list((tmp_path / "data/results/ft/w2b").rglob("DONE"))


def test_a_nan_run_fails_its_attempt_is_retried_once_then_halts(refs, tmp_path):
    c = _cell(tmp_path, 1000, 2)
    r = _run(refs[LEGS], tmp_path, nan_in="N1000/s2")
    assert r.returncode == 1 and "diverged (NaN training loss)" in r.stdout
    assert (c / "ATTEMPT_FAILED").read_text().startswith("rc=1 ") and not (c / "DONE").exists()
    assert (_cell(tmp_path, 1000, 1) / "DONE").exists()      # the cells before it are kept
    # the next pod retries it once; a second divergence stops the job
    r = _run(refs[LEGS], tmp_path, nan_in="N1000/s2", pod="pod2")
    assert r.returncode == 1 and len(list(c.parent.glob("s2.partial.*"))) == 1
    r = _run(refs[LEGS], tmp_path, nan_in="N1000/s2", pod="pod3")
    assert r.returncode == B.EXIT_HALT and "2 failed attempts" in _fail_mark(tmp_path).read_text()


def test_a_nan_run_that_converges_on_its_retry_is_kept(refs, tmp_path):
    c = _cell(tmp_path, 1000, 2)
    assert _run(refs[LEGS], tmp_path, nan_in="N1000/s2").returncode == 1
    r = _run(refs[LEGS], tmp_path, pod="pod2")
    assert r.returncode == 0, r.stdout[-2000:]
    assert (c / "DONE").exists() and not (c / "ATTEMPT_FAILED").exists()
    assert [p for p in c.parent.glob("s2.partial.*") if (p / "ATTEMPT_FAILED").exists()]
