"""The v2 read-out specs (scripts/build_ft_jobs.py, "v2 read-outs"): one per
checkpoint rule and leg, derived from the read-out that ran
(job-ft-scratch-v2-leg1-metrics-raunav.yaml), each expecting every cell the
generator emitted for its rule, and each run under bash with stubs.
"""
import importlib.util
import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_jobs_v2ro", "scripts/build_ft_jobs.py")
MTX = B._mtx_launch()
HAND = K8S / "job-ft-scratch-v2-leg1-metrics-raunav.yaml"
LEGS = ("leg1", "leg2", "bench")
POLICY = {"rules": [
    {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In", "values": [B.EXIT_HALT]}},
    {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}]}


@pytest.fixture(scope="module")
def ro():
    return B.build(B.PIN_V2_READOUT, v2_readouts=True)


@pytest.fixture(scope="module")
def trial():
    return B.build(B.PIN, v2_trial_readouts=True)


def _name(rule, leg, scope=""):
    return f"job-ft-v2-{rule}{'-' + scope if scope else ''}-{leg}-metrics-raunav.yaml"


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _calls(text):
    """Each `python3 experiments/FT/<script>.py ...` the script runs, as (script, argv)."""
    live = "\n".join(l for l in _args(text).splitlines() if not l.lstrip().startswith("#"))
    return [(m.group(1), shlex.split(m.group(2)))
            for m in re.finditer(r"python3 experiments/FT/(\w+)\.py (.*)", live.replace("\\\n", " "))]


def _flag(argv, key):
    """Every value after `key`, or True when it takes none."""
    out = []
    for i, a in enumerate(argv):
        if a == key:
            vals = []
            for b in argv[i + 1:]:
                if b.startswith("--"):
                    break
                vals.append(b)
            out.append(" ".join(vals) or True)
    return out


def test_the_generated_scratch_readout_is_the_one_that_ran_but_for_the_listed_differences():
    """v2_readout(None, "leg1") against the hand-written spec that ran (ledger
    ft-v2-scratch-leg1-metrics), at the tag it ran at. Each intended difference is
    applied to the spec that ran; what is left must be equal, script byte for byte."""
    fname, text = B.v2_readout(None, "leg1", "mtx-s1.98")
    ran, got = yaml.safe_load(HAND.read_text()), yaml.safe_load(text)
    c = ran["spec"]["template"]["spec"]["containers"][0]
    # 1. the pod failure policy of the v2 jobs: exit 42 fails the Job at once, an
    #    eviction is not counted. It names the container `main`, job()'s name.
    ran["spec"]["podFailurePolicy"] = POLICY
    assert c["name"] == "readout"
    c["name"] = "main"
    # 2. the clone reads its tag from REPO_REF, as every spec of this generator does
    #    (build() asserts it), with job()'s env block, NODE_NAME and POD_NAME included.
    c["env"] = [{"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
                {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                {"name": "REPO_REF", "value": "mtx-s1.98"}]
    script = c["args"][0]
    assert script.count('--branch "mtx-s1.98"') == 1
    script = script.replace('--branch "mtx-s1.98"', '--branch "${REPO_REF}"')
    # 3. a failed precondition halts the Job (exit 42) rather than burning a retry.
    assert script.count("exit 1; }") == 2
    script = script.replace("exit 1; }", "exit ${HALT}; }").replace(
        "set -euo pipefail\n",
        f"set -euo pipefail\nHALT={B.EXIT_HALT}   # the pod failure policy fails the Job at once on this code\n")
    c["args"] = [script]
    assert got == ran
    # ...and the comments only otherwise. It is not re-emitted: its own file is its record.
    assert yaml.safe_load(HAND.read_text())["metadata"]["name"] == got["metadata"]["name"]
    assert fname not in B.build(B.PIN_V2_READOUT, v2_readouts=True)


def test_one_readout_per_rule_leg_and_scope(ro):
    # PI, 2026-10-09: one fine-tuning per pretrained model, from the primary; the grid is
    # tier 1 alone, so each read-out covers every cell once
    assert B.V2_RULES == ("best70",) and B.V2_READOUT_LEGS == LEGS
    assert B.V2_READOUT_SCOPES == ("",)
    assert set(ro) == {_name(r, leg, "") for r in B.V2_RULES for leg in LEGS}


def test_the_expected_cells_are_every_paper_run_at_every_size():
    # the 27 runs the paper reports (configs/arms/v2_paper_runs.txt), of the 37 in the grid
    expected = json.loads((ROOT / B.V2_EXPECTED).read_text())
    assert len(B.v2_runs()) == 37
    runs = {n for n, d, *_ in B.v2_runs() if d.rsplit("/", 1)[1] in B.v2_paper_runs()}
    assert len(runs) == 27
    assert set(expected) == {f"best70/{leg}" for leg in ("leg1", "leg2", "leg_top", "leg_qg")} | {"scratch/leg1"}
    for key, cells in expected.items():
        if key != "scratch/leg1":
            assert {c.split("/")[0] for c in cells} == runs and len(cells) == 4 * len(runs)


def test_the_readout_specs_on_disk_are_the_generators(ro, trial):
    want = {**ro, **trial}
    for p in K8S.glob("job-ft-v2-*-metrics-raunav.yaml"):
        assert p.read_text() == want[p.name], p.name
    assert set(trial) <= {p.name for p in K8S.glob("job-ft-v2-*-metrics-raunav.yaml")}


def test_every_readout_is_a_cpu_job_of_mine_at_its_own_pin_with_the_retry_policy(ro):
    assert B.PIN_V2_READOUT == "mtx-s2.12"
    for name, t in ro.items():
        d = yaml.safe_load(t)
        assert name == f"job-{d['metadata']['name']}.yaml" and name.endswith("-raunav.yaml")
        assert d["spec"]["backoffLimit"] == 1 and d["spec"]["podFailurePolicy"] == POLICY
        pod = d["spec"]["template"]["spec"]
        c = pod["containers"][0]
        assert c["name"] == "main" and c["image"] == MTX.V2_IMAGE
        assert "nvidia.com/gpu" not in c["resources"]["limits"] and "tolerations" not in pod
        assert {e["name"]: e.get("value") for e in c["env"]}["REPO_REF"] == B.PIN_V2_READOUT
        assert [v["name"] for v in pod["volumes"]] == ["data"] == [m["name"] for m in c["volumeMounts"]]
        assert 'values: ["us-west"]' in t
        a = _args(t)
        assert f"pip install --no-cache-dir -q {MTX.V2_PYARROW}\n" in a
        assert a.index("git clone") > a.index("OUT=") and "rm -" not in a


def test_each_readout_reads_its_rule_and_expects_every_cell_emitted_for_it(ro):
    expected = json.loads((ROOT / B.V2_EXPECTED).read_text())
    outs = {"/data/results/ft_v2/scratch_leg1_metrics"}            # the scratch reference's
    for rule, leg, scope in [(r, leg, sc) for r in B.V2_RULES for leg in LEGS for sc in B.V2_READOUT_SCOPES]:
            t = ro[_name(rule, leg, scope)]
            tag = rule + (f"_{scope}" if scope else "")
            out = f"/data/results/ft_v2/{tag}_{leg}_metrics"
            assert f"OUT={out}\n" in _args(t) and out not in outs
            outs.add(out)
            for f in B.V2_READOUT_FILES[leg]:
                assert f"#   experiments/FIGS/data/v2/{B.V2_READOUT_COPY[leg]}/{tag}_{f}\n" in t
            calls = _calls(t)
            for script, argv in calls:
                src = (ROOT / f"experiments/FT/{script}.py").read_text()
                assert [a for a in argv if a.startswith("--") and f'"{a}"' not in src] == [], script
                assert _flag(argv, "--expect-cells") == [f"{B.V2_EXPECTED}:{rule}{'@' + scope if scope else ''}"]
            if leg == "leg1":
                [(script, argv)] = calls
                assert script == "leg1_metrics" and expected[f"{rule}/leg1"]
                assert f"ROOT={B.V2_ROOT}/{rule}/leg1\n" in _args(t)
                assert _flag(argv, "--root") == [f"${{ROOT}} {B.V2_ROOT}/scratch/leg1"]
                assert _flag(argv, "--auc-stride") == [str(B.LEG1_AUC_STRIDE)]
            elif leg == "leg2":
                [(script, argv)] = calls
                assert script == "leg2_metrics" and expected[f"{rule}/leg2"]
                assert f"ROOT={B.V2_ROOT}/{rule}/leg2\n" in _args(t) and _flag(argv, "--macro-auc") == [True]
                assert _flag(argv, "--ref-init") == [f"{B.W3_ROOT}/leg2/scratch-v2"]
                assert _flag(argv, "--sha-table") == [B.V2_SHA_TABLE]
            else:
                assert expected[f"{rule}/leg_top"] and expected[f"{rule}/leg_qg"]
                assert f"ROOT={B.V2_ROOT}/{rule}\n" in _args(t)
                refs = {d: f"{B.BENCH_V2_ROOT}/leg_{d}/scratch-v2" for d in ("top", "qg")}
                nmax = f"{B.BENCH_SIZES['top'][-1]} {B.BENCH_SIZES['qg'][-1]}"
                got = {(bool(_flag(a, "--herwig")), tuple(_flag(a, "--features-dir"))):
                       (tuple(_flag(a, "--ref-init")), tuple(_flag(a, "--sizes"))) for s, a in calls}
                assert got == {(False, ()): ((refs["top"], refs["qg"]), ()),
                               (True, ()): ((refs["qg"],), ()),
                               (False, ("features_last",)): ((), (nmax,)),
                               (True, ("features_last",)): ((), (nmax,))}
                assert {s for s, _ in calls} == {"bench_metrics"} and len(calls) == 4


def test_the_reference_cells_checked_are_exactly_the_from_scratch_cells_read(ro):
    ref = B.INITS_LATER[B.SCRATCH_REF]
    want = {"leg1": {f"{B.V2_ROOT}/scratch/leg1/{n}/N{N}/s{s}" for leg, n, N, s in B.cells_legs(ref) if leg == "leg1"},
            "leg2": {f"{B.W3_ROOT}/leg2/{n}/N{N}/s{s}" for leg, n, N, s in B.cells_legs(ref) if leg == "leg2"},
            "bench": {f"{B.BENCH_V2_ROOT}/{leg}/{n}/N{N}/s{s}" for leg, n, N, s in B.cells_bench(ref)}}
    assert {k: len(v) for k, v in want.items()} == {"leg1": 12, "leg2": 12, "bench": 24}
    for rule in B.V2_RULES:
        for leg in LEGS:
            words = re.search(r"for c in (.*); do", _args(ro[_name(rule, leg)])).group(1)
            got = subprocess.run(["bash", "-c", f"echo {words}"], capture_output=True, text=True).stdout.split()
            assert set(got) == want[leg] and len(got) == len(want[leg])


# ------------------------------------------------------------------ the shell
# python3 is a stub that logs each call and writes the tables a read-out writes
# (KILL_AT=k SIGKILLs the job at its k-th read-out, as an eviction does; FAIL=1
# fails it); git and pip are stubs; /data and /workspace live in tmp_path.

def _stub(bindir, name, body):
    p = bindir / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    calls = tmp_path / "calls.log"
    _stub(bindir, "git", 'case "$1" in clone) mkdir -p "${@: -1}";; *) echo deadbeef;; esac\n')
    _stub(bindir, "pip", "exit 0\n")
    _stub(bindir, "python3", f'echo "PY $*" >> {calls}\n' + r'''
next_after () { local key=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$key" ] && { echo "$2"; return; }; shift; done; }
case "$*" in
  "-c "*) exit 0;;
  *_metrics.py*)
    [ -z "${FAIL:-}" ] || { echo "FATAL: read-out failed"; exit 1; }
    k=$(grep -c "_metrics.py" "__CALLS__")
    [ "${k}" != "${KILL_AT:-}" ] || { kill -9 $PPID; exit 137; }
    O=$(next_after --out "$@"); mkdir -p "$O"
    case "$*" in
      *leg1_metrics.py*) f=leg1_metrics;;
      *leg2_metrics.py*) f=leg2_metrics;;
      *) f=bench_metrics
         case "$*" in *--herwig*) f=${f}_herwig;; esac
         case "$*" in *features_last*) f=${f}_last;; esac;;
    esac
    echo "{}" > "$O/$f.json";;
esac
exit 0
'''.replace("__CALLS__", str(calls)))
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", REPO_REF="test"), calls


def _tree(tmp_path, rule, leg, skip_ref=False):
    """One DONE cell of the rule per tree the read-out reads, and every reference cell."""
    data = tmp_path / "data"
    trees = {"leg1": ["leg1"], "leg2": ["leg2"], "bench": ["leg_top", "leg_qg"]}[leg]
    for t in trees:
        cell = data / f"results/ft_v2/{rule}/{t}/l188-s1/N1000/s1"
        cell.mkdir(parents=True)
        (cell / "DONE").touch()
        (cell / "predict.log").touch()
    words = re.search(r"for c in (.*); do", B._v2_readout_script(rule, leg)).group(1)
    refs = subprocess.run(["bash", "-c", f"echo {words}"], capture_output=True, text=True).stdout.split()
    for i, c in enumerate(refs):
        p = data / c[len("/data/"):]
        p.mkdir(parents=True)
        if not (skip_ref and i == len(refs) - 1):
            (p / "DONE").touch()
    return data


def _run(text, tmp_path, **env_extra):
    env, calls = _env(tmp_path)
    env.update(env_extra)
    script = (_args(text).replace("/workspace", str(tmp_path / "workspace"))
              .replace("/data", str(tmp_path / "data")))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env,
                       cwd=tmp_path, timeout=120)
    return r, calls.read_text() if calls.exists() else ""


@pytest.mark.parametrize("rule,leg", [("best70", "leg1"), ("best70", "leg2"), ("best70", "bench")])
def test_a_readout_runs_under_bash_writes_its_tables_and_refuses_a_second_run(ro, tmp_path, rule, leg):
    data = _tree(tmp_path, rule, leg)
    r, calls = _run(ro[_name(rule, leg)], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = data / f"results/ft_v2/{rule}_{leg}_metrics"
    assert sorted(p.name for p in out.iterdir()) == sorted(B.V2_READOUT_FILES[leg])
    assert calls.count("_metrics.py") == len(B.V2_READOUT_FILES[leg])
    assert not list((data / "results/ft_v2").glob("*.staging.*"))
    if leg == "leg1":
        assert f"--root {data}/results/ft_v2/{rule}/leg1 {data}/results/ft_v2/scratch/leg1 " in calls
    # applied again, it refuses at once (exit 42 fails the Job) and reads nothing
    (tmp_path / "calls.log").unlink()
    r, calls = _run(ro[_name(rule, leg)], tmp_path)
    assert r.returncode == B.EXIT_HALT and "use a new output directory" in r.stdout
    assert "_metrics.py" not in calls


@pytest.mark.parametrize("leg", LEGS)
def test_a_missing_tree_or_reference_cell_halts_and_a_failed_readout_is_counted(ro, tmp_path, leg):
    text = ro[_name("best70", leg)]
    r, calls = _run(text, tmp_path / "none")
    assert r.returncode == B.EXIT_HALT and "FATAL: no " in r.stdout and "_metrics.py" not in calls
    _tree(tmp_path / "ref", "best70", leg, skip_ref=True)
    r, calls = _run(text, tmp_path / "ref")
    assert r.returncode == B.EXIT_HALT and "1 reference cells not DONE" in r.stdout
    assert "_metrics.py" not in calls
    # a read-out that fails (missing expected cells, a NaN cell...) is not a halt:
    # the Job's one retry is counted, as a crash is
    _tree(tmp_path / "fail", "best70", leg)
    r, _ = _run(text, tmp_path / "fail", FAIL="1")
    assert r.returncode == 1 and "read-out failed" in r.stdout


def test_a_jetclass_cell_with_a_log_but_no_done_halts(ro, tmp_path):
    data = _tree(tmp_path, "best70", "leg2")
    half = data / "results/ft_v2/best70/leg2/l188-s2/N1000/s1"
    half.mkdir(parents=True)
    (half / "predict.log").touch()
    r, calls = _run(ro[_name("best70", "leg2")], tmp_path)
    assert r.returncode == B.EXIT_HALT and f"NOT DONE: {half}" in r.stdout and "_metrics.py" not in calls


def test_an_evicted_benchmark_readout_leaves_no_partial_output_and_its_retry_completes(ro, tmp_path):
    data = _tree(tmp_path, "best70", "bench")
    out = data / "results/ft_v2/best70_bench_metrics"
    r, _ = _run(ro[_name("best70", "bench")], tmp_path, KILL_AT="3")
    assert r.returncode == -9 and not out.exists()
    [staged] = (data / "results/ft_v2").glob("best70_bench_metrics.staging.*")
    assert len(list(staged.iterdir())) == 2
    (tmp_path / "calls.log").unlink()
    r, calls = _run(ro[_name("best70", "bench")], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert sorted(p.name for p in out.iterdir()) == sorted(B.V2_READOUT_FILES["bench"])
    assert calls.count("bench_metrics.py") == 4


# ------------------------------------------------------------------ the pin

def test_the_readouts_pin_a_tag_still_to_be_made_and_the_tree_carries_what_they_run():
    B.verify_pin(B.PIN_V2_READOUT, True, B.V2_READOUT_NEEDED)
    for path, flag in B.V2_READOUT_NEEDED.items():
        assert flag in (ROOT / path).read_text(), path
    with pytest.raises(SystemExit, match="has no --no-such-flag"):
        B.verify_pin(B.PIN_V2_READOUT, True, {"experiments/FT/bench_metrics.py": "--no-such-flag"})
    cli = [sys.executable, str(ROOT / "scripts/build_ft_jobs.py"), "--v2-readouts", "--check-only"]
    r = subprocess.run(cli + ["--pin-not-yet-tagged"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count(f"checked (pin {B.PIN_V2_READOUT})") == 3      # 1 rule x 3 legs
    r = subprocess.run(cli + ["--pin-not-yet-tagged", "--v2"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "emitted alone" in r.stderr


# ------------------------------------------------------------------ the read-out of the finished runs

def test_the_trial_reads_the_runs_fine_tuned_before_the_split_and_expects_all_their_cells(trial):
    paper = [s for s, _ in B.v2_groups()]
    assert B.v2_trial_inits() == [s for s in paper if s not in B.V2_SPLIT_RUNS] and len(B.v2_trial_inits()) == 18
    full = json.loads((ROOT / B.V2_EXPECTED).read_text())
    got = json.loads((ROOT / B.V2_TRIAL_EXPECTED).read_text())
    assert got == B.v2_trial_expected_cells()
    assert set(got) == {k for k in full if k.startswith("best70/")}
    for k, cells in got.items():
        assert cells == [c for c in full[k] if c.split("/")[0] in B.v2_trial_inits()] and len(cells) == 18 * 4
    for leg in LEGS:
        name = f"job-ft-v2-best70-done18-{leg}-metrics-raunav.yaml"
        a = _args(trial[name])
        assert f"OUT={B.V2_ROOT}/best70_done18_{leg}_metrics\n" in a
        runs = re.findall(r"python3 experiments/FT/\w+_metrics\.py[^\n]*", a.replace("\\\n", " "))
        assert len(runs) == len(B.V2_READOUT_FILES[leg])
        for r in runs:
            want = B.v2_trial_inits() + ([B.SCRATCH_REF] if leg == "leg1" else [])
            assert "--inits " + " ".join(want) + " --expect-cells " + B.V2_TRIAL_EXPECTED + ":best70" in r, r
        assert B.V2_EXPECTED + ":" not in a
        env = {e["name"]: e.get("value") for e in
               yaml.safe_load(trial[name])["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["REPO_REF"] == B.PIN_V2_TRIAL[leg]


def _final(leg):
    return B.v2_readout("best70", leg, B.PIN_V2_TRIAL[leg])[1]


def test_the_trial_differs_from_the_final_readout_only_where_it_must(trial):
    import difflib
    for leg in LEGS:
        a = _final(leg).splitlines()
        b = trial[f"job-ft-v2-best70-done18-{leg}-metrics-raunav.yaml"].splitlines()
        changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0)
                   if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        for l in changed:
            assert ("done18" in l or "ft_v2_expected_cells.json" in l or "best70_" in l
                    or "ft-v2-best70-" in l or "--expect-cells" in l or "find ${ROOT}" in l), l


def test_a_split_run_still_fine_tuning_does_not_stop_the_trial(trial, tmp_path):
    data = tmp_path / "data"
    for t in ("leg2",):
        cell = data / f"results/ft_v2/best70/{t}/l188-s3/N1000/s1"
        cell.mkdir(parents=True)
        (cell / "DONE").touch()
        (cell / "predict.log").touch()
    words = re.search(r"for c in (.*); do", B._v2_readout_script("best70", "leg2")).group(1)
    for c in subprocess.run(["bash", "-c", f"echo {words}"], capture_output=True, text=True).stdout.split():
        (data / c[len("/data/"):]).mkdir(parents=True)
        (data / c[len("/data/"):] / "DONE").touch()
    for i in B.v2_trial_inits():
        (data / f"results/ft_v2/best70/leg2/{i}").mkdir(parents=True, exist_ok=True)
    half = data / "results/ft_v2/best70/leg2/l188-s1/N1000/s1"       # a split run, mid-way
    half.mkdir(parents=True)
    (half / "predict.log").touch()
    r, calls = _run(trial["job-ft-v2-best70-done18-leg2-metrics-raunav.yaml"], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NOT DONE" not in r.stdout and "--inits l188-s3 " in calls
    assert (data / "results/ft_v2/best70_done18_leg2_metrics").is_dir()
    r, _ = _run(B.v2_readout("best70", "leg2", B.PIN_V2_READOUT)[1], tmp_path)   # the final one still halts
    assert r.returncode == B.EXIT_HALT and f"NOT DONE: {half}" in r.stdout


def test_the_trial_pins_a_tag_whose_tree_carries_inits():
    # each leg as it ran; leg 2 again after eval_arm's tie (as the final read-outs)
    assert B.PIN_V2_TRIAL == {"leg1": "mtx-s2.11", "leg2": "mtx-s2.12", "bench": "mtx-s2.11"}
    assert B.PIN_V2_READOUT == "mtx-s2.12" and B.EVAL_ARM_TIE.items() <= B.V2_READOUT_NEEDED.items()
    for leg, pin in B.PIN_V2_TRIAL.items():
        B.verify_pin(pin, True, B.V2_TRIAL_NEEDED)
    B.verify_pin(B.PIN_V2_READOUT, True, B.V2_READOUT_NEEDED)
