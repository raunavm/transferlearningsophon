"""Retries that survive a flaky cluster (scripts/build_ft_jobs.py, 2026-09-29;
cells resume since 2026-10-01).

Pinned here, by running an emitted script under bash with the wave-3 stubs:
  * an attempt killed with its pod (eviction, a lost node) is not a failure, and
    the cell RESUMES from its last completed epoch: six evictions no longer halt;
  * the resumed cell continues from its epoch and its best-validation checkpoint
    is the best over the whole run, not of the last attempt;
  * two FAILED attempts of one cell stop the job with the halt code the pod
    failure policy turns into FailJob, and a failed attempt is never resumed;
  * attempts that complete no epoch stop the job after STALL_LIMIT of them;
  * a step that fails on a CUDA device fault, or with the GPU no longer
    answering, is the node's fault: recorded, not counted against the cell,
    which resumes; NODE_FAULT_LIMIT of them halt; a pod whose GPU does not
    answer is refused before it touches a cell;
  * a read-out failure after training redoes only the read-out (TRAINED);
  * a halt renames what it counted, so removing the marker restarts the job;
  * cells are claimed atomically and a cell held by another job halts the job;
  * a run whose training loss went to NaN fails its attempt, is retried once
    from the start, and halts the job if it diverges again;
and, on the specs: which carry the policy, which take the resumable logic, and
that the specs of jobs that ran under the 2026-09-29 logic are their records.
"""
import importlib.util
import json
import pathlib
import subprocess
import zipfile

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
# A resumable legs spec: the scratch-v2 baseline group's script, one seed, under a
# name the 2026-09-29 records do not hold -- exactly how a new spec is emitted.
SHARD = "ft-legs-baseline-scratch-resume-raunav"
INITS = [("scratch-v2", "", 0, [1])]
CELL = "leg1/scratch-v2/N1000/s1"              # 50 epochs, the first cell


@pytest.fixture(scope="module")
def spec():
    assert B.resumable(SHARD)
    gpu = dict(gpu=True, cpu="4", memory="88Gi", shm="8Gi", backoff=1, pin=B.PIN_RESUME,
               exclude_hosts=B.BAD_NODES + B.LOST_GPU_NODES)
    script = B.robust_script(B.legs_w3(INITS, SHARD), SHARD)
    return B.job(SHARD, B._fill(script, B.PIN_RESUME, inits=INITS), **B._retry_kw(SHARD, gpu),
                 header=B._retry_note(SHARD))


@pytest.fixture(scope="module")
def refs():
    return B.build(B.PIN_REFS, wave3=True, bench_v2=True, later=["scratch-v2", "mpm-s1-v2"])


def _run(text, tmp_path, n, **env_extra):
    """Pod number n of the job: its own stubs, the shared data tree in tmp_path."""
    if not (tmp_path / "data").exists():
        T._shell_env(tmp_path, INITS)                       # the data tree, once
    stubs = tmp_path / f"pod{n}"
    env, calls = T._shell_env(stubs, INITS)
    env.update({k: str(v) for k, v in env_extra.items()})
    r = subprocess.run(["bash", "-c", T._redirect(text, stubs, tmp_path)], capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=600)
    return r, calls.read_text() if calls.exists() else ""


def _cell(tmp_path, rel=CELL):
    return tmp_path / "data/results/ft/w2b" / rel


def _fail_mark(tmp_path):
    return tmp_path / "data/results/ft/w2b" / f"FAILED.{SHARD}"


def _ledger(cell):
    return [int(ln.rsplit("from_epoch=", 1)[1]) for ln in (cell / "ATTEMPTS").read_text().splitlines()]


def _exact_epochs(cell):
    return [int(ln.split("Epoch #")[1].split(":")[0])
            for ln in (cell / "train.log").read_text().splitlines() if "exact validation metric" in ln]


def _weaver_calls(calls, rel=CELL):
    return [ln for ln in calls.splitlines() if "ft_weaver.py" in ln and f"/{rel}/net" in ln]


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
    # every job that ran, or is running, keeps its record byte for byte -- the five
    # that ran under the 2026-09-29 logic included
    for name in (*robust, "job-ft-legs-bench-baseline-scratch-v2-raunav.yaml",
                 *(f"job-ft-legs-bench-v3-last-{s}-raunav.yaml" for s in "abd")):
        assert (K8S / name).read_text() == {**refs, **v3}[name], name
        assert not B.resumable(yaml.safe_load({**refs, **v3}[name])["metadata"]["name"])


def test_no_other_spec_on_disk_carries_the_policy():
    have = {p.name for p in K8S.glob("job-*.yaml") if "podFailurePolicy" in p.read_text()}
    # every v2 spec carries it (audit 2026-09-29): its name says so; and every
    # hand-written read-only job since (inspection, read-outs)
    v2 = {p.name for p in K8S.glob("job-ft-v2-*.yaml")} | {"job-ft-subsets-jc2-v2-raunav.yaml"}
    hand = {p.name for p in K8S.glob("job-ft-inspect-*.yaml")} | {"job-ft-bench-v3-metrics-raunav.yaml"}
    assert v2 <= have and hand <= have
    assert have - v2 - hand == {"job-ft-legs-baseline-scratch-v2-raunav.yaml",
                                "job-ft-legs-baseline-mpm-s1-v2-raunav.yaml",
                                "job-ft-legs-bench-baseline-mpm-s1-v2-raunav.yaml",
                                "job-ft-legs-bench-v3-last-c-raunav.yaml",
                                "job-ft-legs-bench-v3-last-e-raunav.yaml"}
    for name in hand:
        assert "readOnly: true" in (K8S / name).read_text() or "metrics" in name, name


def test_the_unlaunched_self_supervised_groups_resume_and_avoid_the_lost_gpu():
    later = B.build(B.PIN_RESUME, wave3=True, bench_v2=True, later=["mpm-s2", "mpm-s3"])
    assert len(later) == 4
    for text in later.values():
        assert "podFailurePolicy" in text and f"HALT={B.EXIT_HALT}" in text
        assert "cell_resume.py prepare" in text and "ft_weaver.py --seed" in text
        assert "seed_weaver.py" not in T._args(text)
        assert set(B.GPU_LOST_LATER) <= T._excluded(text)
    with pytest.raises(SystemExit, match="predates"):
        B.build(B.PIN_REFS, wave3=True, later=["mpm-s2"])


def test_the_resumable_logic_replaces_every_restart_and_cap(spec):
    live = "\n".join(ln for ln in T._args(spec).splitlines() if not ln.lstrip().startswith("#"))
    for gone in (".partial.$(date", '"${a}" -ge 6', "seed_weaver.py", "^Traceback", "attempt_ok",
                 "mkdir ${OUT}.lock"):
        assert gone not in live, gone
    assert live.count("cell_resume.py lock --path ${OUT}.lock --owner ${SHARD}") == 2
    assert live.count("touch ${OUT}/TRAINED") == 2
    assert live.index("python3 experiments/FT/gpu_probe.py ||") < live.index("for spec in ${INITS}")
    rules = yaml.safe_load(spec)["spec"]["podFailurePolicy"]["rules"]
    assert rules[1] == {"action": "Count", "onExitCodes": {"containerName": "main", "operator": "In",
                                                           "values": [B.EXIT_NODE_FAULT]}}
    assert live.count("cell_resume.py prepare --dir ${OUT}") == 2
    assert live.count("cell_resume.py best --dir ${OUT} --epochs ${EP}") == 2
    assert live.count('[ "${E}" -ge 0 ] || python3 experiments/FT/smoke_checks.py manifest') == 2
    assert live.count("tee ${OUT}/stdout.log") == 2
    # the best epoch is fixed before anything reads net_best_epoch_state.pt
    assert live.index("cell_resume.py best") < live.index("net_best_epoch_state.pt")
    assert B.RETRY_NOTE_RESUME in spec and set(B.GPU_LOST_LATER) <= T._excluded(spec)


# ------------------------------------------------------------------ the shell

def test_a_complete_run_keeps_weavers_own_best_epoch(spec, tmp_path):
    r, calls = _run(spec, tmp_path, 1)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    c = _cell(tmp_path)
    assert (c / "DONE").exists() and _ledger(c) == [-1]
    rec = json.loads((c / "best_epoch.json").read_text())
    assert rec["epoch"] == 49 and not rec["resumed"] and not rec["restored"]
    assert T._done_cells(tmp_path / "data/results/ft/w2b") == set(B.cells_legs(INITS))
    assert "--load-epoch" not in calls


def test_six_evictions_no_longer_halt_and_the_cell_finishes(spec, tmp_path):
    for n, kill in enumerate((5, 12, 20, 27, 35, 44), start=1):       # six pods, each evicted
        r, _ = _run(spec, tmp_path, n, KILL_IN=CELL, KILL_AT=kill)
        assert r.returncode == -9, r.stdout[-2000:] + r.stderr[-2000:]
    r, calls = _run(spec, tmp_path, 7)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    c = _cell(tmp_path)
    assert (c / "DONE").exists() and not _fail_mark(tmp_path).exists()
    assert not list(c.parent.glob("s1.partial.*"))
    assert _ledger(c) == [-1, 4, 11, 19, 26, 34, 43]
    assert _exact_epochs(c) == list(range(50))            # one trajectory, every epoch once
    assert "--load-epoch 43" in _weaver_calls(calls)[0]


def test_a_resumed_cell_continues_from_its_epoch_and_keeps_the_runs_best(spec, tmp_path):
    # epoch 2 is the best of the run; the attempt that resumes after epoch 6
    # never sees it, and weaver (restarted at 0) would keep epoch 7
    metrics = " ".join(["0.3", "0.4", "0.9"] + ["0.5"] * 47)
    r, _ = _run(spec, tmp_path, 1, KILL_IN=CELL, KILL_AT=7, METRICS=metrics)
    assert r.returncode == -9
    c = _cell(tmp_path)
    assert _exact_epochs(c) == list(range(7))
    r, calls = _run(spec, tmp_path, 2, METRICS=metrics)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    run = _weaver_calls(calls)
    assert len(run) == 1 and "--load-epoch 6" in run[0]
    assert _ledger(c) == [-1, 6] and _exact_epochs(c) == list(range(50))
    assert (c / "train.log.cut.1").read_text().strip() == "[t] INFO: Epoch #7 training"
    assert (c / "stdout.log.1").exists() and (c / "stdout.log").exists()
    rec = json.loads((c / "best_epoch.json").read_text())
    assert rec["epoch"] == 2 and rec["resumed"] and rec["restored"]
    with zipfile.ZipFile(c / "net_best_epoch_state.pt") as z:
        assert z.read("-") == b"state 2\n"
    assert not list(c.glob("net_epoch-*"))                 # pruned as before


def test_a_checkpoint_cut_short_is_not_resumed_from(spec, tmp_path):
    r, _ = _run(spec, tmp_path, 1, KILL_IN=CELL, KILL_AT=7)
    c = _cell(tmp_path)
    (c / "net_epoch-6_optimizer.pt").write_bytes((c / "net_epoch-6_optimizer.pt").read_bytes()[:20])
    r, calls = _run(spec, tmp_path, 2)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "--load-epoch 5" in _weaver_calls(calls)[0] and _exact_epochs(c) == list(range(50))


def test_attempts_without_progress_halt_with_the_policy_code(spec, tmp_path):
    r, _ = _run(spec, tmp_path, 1, KILL_IN=CELL, KILL_AT=3)
    for n in (2, 3, 4):                                    # each dies in the epoch it resumes into
        r, _ = _run(spec, tmp_path, n, KILL_IN=CELL, KILL_AT=3)
        assert r.returncode == -9
    r, _ = _run(spec, tmp_path, 5)
    assert r.returncode == B.EXIT_HALT, r.stdout[-2000:]
    assert f"no epoch completed in the last {B.STALL_LIMIT} attempts" in _fail_mark(tmp_path).read_text()
    assert not (_cell(tmp_path) / "DONE").exists()
    r, _ = _run(spec, tmp_path, 6)                         # and the marker stops every later pod
    assert r.returncode == B.EXIT_HALT


def test_two_real_failures_still_halt_and_a_failure_is_never_resumed(spec, tmp_path):
    c = _cell(tmp_path)
    r, _ = _run(spec, tmp_path, 1, NAN_IN=CELL)
    assert r.returncode == 1 and "diverged (NaN training loss)" in r.stdout
    assert (c / "ATTEMPT_FAILED").read_text().startswith("rc=1 ") and not (c / "DONE").exists()
    r, calls = _run(spec, tmp_path, 2, NAN_IN=CELL)
    assert r.returncode == 1
    assert "--load-epoch" not in _weaver_calls(calls)[0]   # started again, not resumed
    assert len(list(c.parent.glob("s1.partial.*"))) == 1
    r, _ = _run(spec, tmp_path, 3, NAN_IN=CELL)
    assert r.returncode == B.EXIT_HALT and "2 failed attempts" in _fail_mark(tmp_path).read_text()


def test_a_nan_run_that_converges_on_its_retry_is_kept(spec, tmp_path):
    c = _cell(tmp_path)
    assert _run(spec, tmp_path, 1, NAN_IN=CELL)[0].returncode == 1
    r, _ = _run(spec, tmp_path, 2)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (c / "DONE").exists() and not (c / "ATTEMPT_FAILED").exists()
    assert [p for p in c.parent.glob("s1.partial.*") if (p / "ATTEMPT_FAILED").exists()]


def test_an_eviction_after_a_failure_resumes_and_the_failure_still_counts(spec, tmp_path):
    c = _cell(tmp_path)
    assert _run(spec, tmp_path, 1, NAN_IN=CELL)[0].returncode == 1          # failure 1
    assert _run(spec, tmp_path, 2, KILL_IN=CELL, KILL_AT=9)[0].returncode == -9
    r, calls = _run(spec, tmp_path, 3, NAN_IN=CELL)                        # resumes, diverges
    assert "--load-epoch 8" in _weaver_calls(calls)[0] and r.returncode == 1
    r, _ = _run(spec, tmp_path, 4)
    assert r.returncode == B.EXIT_HALT and "2 failed attempts" in _fail_mark(tmp_path).read_text()


def test_a_readout_failure_after_training_redoes_only_the_readout(spec, tmp_path):
    c = _cell(tmp_path)
    r, _ = _run(spec, tmp_path, 1, EXTRACT_FAIL_IN=f"{CELL}/features_v2")
    assert r.returncode == 1 and (c / "ATTEMPT_FAILED").exists() and (c / "TRAINED").exists()
    r, calls = _run(spec, tmp_path, 2)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "--load-epoch 49" in _weaver_calls(calls)[0]          # no epoch trained again
    assert _exact_epochs(c) == list(range(50)) and (c / "DONE").exists()
    assert (c / "ATTEMPT_FAILED.1").exists() and not list(c.parent.glob("s1.partial.*"))


def test_a_cuda_fault_is_the_nodes_and_the_cell_resumes(spec, tmp_path):
    c = _cell(tmp_path)
    r, _ = _run(spec, tmp_path, 1, CUDA_FAULT_IN=CELL, FAULT_AT=5)
    assert r.returncode == B.EXIT_NODE_FAULT, r.stdout[-2000:] + r.stderr[-2000:]
    assert len(list(c.glob("NODE_FAULT.*"))) == 1 and not (c / "ATTEMPT_FAILED").exists()
    faults = (tmp_path / "data/results/ft/w2b" / f"NODE_FAULTS.{SHARD}").read_text()
    assert "node=n" in faults and CELL in faults
    r, calls = _run(spec, tmp_path, 2)
    assert r.returncode == 0 and "--load-epoch 4" in _weaver_calls(calls)[0]
    assert (c / "DONE").exists()


def test_node_faults_halt_at_their_limit(spec, tmp_path):
    for n in range(1, B.NODE_FAULT_LIMIT + 1):
        r, _ = _run(spec, tmp_path, n, CUDA_FAULT_IN=CELL, FAULT_AT=3 * n)
        assert r.returncode == B.EXIT_NODE_FAULT
    r, _ = _run(spec, tmp_path, 9)
    assert r.returncode == B.EXIT_HALT
    assert f"{B.NODE_FAULT_LIMIT} node faults (nodes n)" in _fail_mark(tmp_path).read_text()


def test_a_pod_whose_gpu_does_not_answer_is_refused_before_any_cell(spec, tmp_path):
    r, calls = _run(spec, tmp_path, 1, GPU_DEAD=1)
    assert r.returncode == B.EXIT_NODE_FAULT
    root = tmp_path / "data/results/ft/w2b"
    assert "preflight" in (root / f"NODE_FAULTS.{SHARD}").read_text()
    assert not (root / "leg1").exists() and "ft_weaver.py" not in calls


def test_a_halted_job_restarts_once_its_marker_is_removed(spec, tmp_path):
    for n in (1, 2):
        assert _run(spec, tmp_path, n, NAN_IN=CELL)[0].returncode == 1
    assert _run(spec, tmp_path, 3)[0].returncode == B.EXIT_HALT
    c = _cell(tmp_path)
    assert len(list(c.parent.glob("s1.halted.partial.*"))) == 2 and not c.exists()
    _fail_mark(tmp_path).unlink()                          # the cause is fixed
    r, _ = _run(spec, tmp_path, 4)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (c / "DONE").exists() and _ledger(c) == [-1]


def test_a_stalled_job_restarts_once_its_marker_is_removed(spec, tmp_path):
    for n in range(1, 5):
        _run(spec, tmp_path, n, KILL_IN=CELL, KILL_AT=3)
    assert _run(spec, tmp_path, 5)[0].returncode == B.EXIT_HALT
    _fail_mark(tmp_path).unlink()
    r, calls = _run(spec, tmp_path, 6)
    assert r.returncode == 0 and "--load-epoch 2" in _weaver_calls(calls)[0]
    assert list(_cell(tmp_path).glob("ATTEMPTS.halted.*"))


def test_cells_are_claimed_atomically_and_a_foreign_lock_halts_the_job(spec, tmp_path):
    T._shell_env(tmp_path, INITS)                           # the data tree first
    root = tmp_path / "data/results/ft/w2b"
    own, foreign, ownerless = (root / "leg1/scratch-v2/N1000/s1.lock", root / "leg1/scratch-v2/N10000/s1.lock",
                               root / "leg2/scratch-v2/N1000/s1.lock")
    for d, owner in ((own, SHARD), (foreign, "ft-v2-legs-bestval-t1a-raunav"), (ownerless, None)):
        d.mkdir(parents=True)
        if owner:
            (d / "owner").write_text(owner + "\n")
    r, _ = _run(spec, tmp_path, 1)
    assert r.returncode == B.EXIT_HALT
    assert "re-entering" in r.stdout and "held by ft-v2-legs-bestval-t1a-raunav" in r.stdout
    assert "held by unknown" in r.stdout and "2 cells are held by another job's lock" in r.stdout
    assert (_cell(tmp_path) / "DONE").exists() and not own.exists() and foreign.exists()
    assert not list(root.rglob("*.lock.partial.*"))
