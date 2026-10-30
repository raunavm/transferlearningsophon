"""The from-scratch reference on subset s1 and the 50-epoch leg-2 check (scripts/build_ft_jobs.py,
"the from-scratch reference on subset s1, and leg 2 at 50 epochs"; 2026-10-30): two Indexed Jobs
on V100, one cell at a time, whose pod index names one cell; under the bash harness each cell's
training command is the V100 reference's (or the launched pretrained cell's) but for the stated
change (subset s1, or 50 epochs); and their read-outs, which wait in the pod.
"""
import importlib.util
import pathlib
import re
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


B = _load("build_ft_jobs_extra", "scripts/build_ft_jobs.py")
V = _load("v2ft_harness_extra", "tests/test_ft_v2_finetune_specs.py")
W = _load("wave3_harness_extra", "tests/test_wave3_specs.py")
R = _load("v2ro_harness_extra", "tests/test_ft_v2_readout_specs.py")
GPU = ("job-ft-v2-ref-v100-s1-raunav.yaml", "job-ft-v2-leg2e50-raunav.yaml")
READ = ("job-ft-v2-ref-v100-s1-leg2-metrics-raunav.yaml", "job-ft-v2-ref-v100-s1-bench-metrics-raunav.yaml",
        "job-ft-v2-leg2e50-metrics-raunav.yaml")
V100 = "Tesla-V100-SXM2-32GB"


@pytest.fixture(scope="module")
def specs():
    return B.build(B.PIN_V2_EXTRA, v2_extra=True)


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _live(text):
    return "\n".join(l for l in _args(text).splitlines() if not l.lstrip().startswith("#"))


def _words(text, var):
    words = re.search(rf"for {var} in (.*); do", _args(text)).group(1)
    return subprocess.run(["bash", "-c", f"echo {words}"], capture_output=True, text=True).stdout.split()


# ------------------------------------------------------------------ the specs

def test_the_specs_on_disk_are_the_generators(specs):
    assert set(specs) == set(GPU + READ)
    for name, text in specs.items():
        assert (K8S / name).read_text() == text, name


def test_the_cells_are_seeds_2_3_on_subset_s1_below_the_full_sets_and_the_50_epoch_check():
    assert B.V2_REF_S1_ROOT == "/data/results/ft_v2/ref_v100_s1"
    assert B.LEG2E50_ROOT == "/data/results/ft_v2/recipe_check/leg2_e50"
    s1 = B.ref_s1_cells()
    want = ([("leg2", f"{B.V2_REF_S1_ROOT}/jc1", n, s) for s in (2, 3) for n in (1000, 10000, 100000, 1000000)]
            + [(t, f"{B.V2_REF_S1_ROOT}/bench", n, s) for t in ("leg_top", "leg_qg")
               for s in (2, 3) for n in (1000, 10000, 100000)])
    assert [(t, c[0], c[4], c[5]) for t, c in s1] == want
    # all from scratch: no run directory, no checkpoint head; seed 1 and the full sets are not rerun
    assert all(c[1:4] == ("", B.SCRATCH_REF, 0) for _, c in s1)
    assert not any(c[5] == 1 for _, c in s1)
    assert not any(c[4] in (B.BENCH_SIZES["top"][-1], B.BENCH_SIZES["qg"][-1]) for _, c in s1)
    e50 = B.leg2e50_cells()
    assert [(c[0], c[2], c[4], c[5]) for c in e50] == (
        [(B.LEG2E50_ROOT, f"l188-s{s}", 1_000_000, 1) for s in (1, 2, 3)]
        + [(B.LEG2E50_ROOT, B.SCRATCH_REF, 1_000_000, s) for s in (1, 2, 3)])
    assert [c[1] for c in e50[:3]] == [f"{B.MTX_V2}/mtx-l188-s{s}" for s in (1, 2, 3)] and not any(c[1] for c in e50[3:])
    assert {c[3] for c in e50[:3]} == {188} and B.LEG2E50_EPOCHS == 50 and B.EPOCHS[1_000_000] == 10


def test_each_is_an_indexed_job_of_mine_on_v100_one_cell_at_a_time(specs):
    cells = {GPU[0]: B.ref_s1_cells(), GPU[1]: B.leg2e50_cells()}
    assert (len(cells[GPU[0]]), len(cells[GPU[1]])) == (20, 6)
    for name in GPU:
        d = yaml.safe_load(specs[name])
        assert d["metadata"]["name"] == name[4:-5] and d["metadata"]["name"].endswith("-raunav")
        s = d["spec"]
        assert s["completionMode"] == "Indexed" and s["completions"] == len(cells[name]) and s["parallelism"] == 1
        assert s["backoffLimitPerIndex"] == B.ROBUST_BACKOFF and "backoffLimit" not in s
        assert s["podFailurePolicy"]["rules"][0] == {
            "action": "FailIndex", "onExitCodes": {"containerName": "main", "operator": "In", "values": [B.EXIT_HALT]}}
        pod = s["template"]["spec"]
        c = pod["containers"][0]
        env = {e["name"]: e.get("value") for e in c["env"]}
        assert env["REPO_REF"] == B.PIN_V2_EXTRA == "mtx-s2.20" and env["GPU_PRODUCT"] == V100
        assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
        terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
        expr = {e["key"]: e for t in terms for e in t["matchExpressions"]}
        assert expr["nvidia.com/gpu.product"]["values"] == [V100]
        assert expr["topology.kubernetes.io/region"]["values"] == ["us-west"]
        assert set(B._v2_off_nodes()) | set(B.GPU_LOST_LATER) <= set(expr["kubernetes.io/hostname"]["values"])
        assert {v["persistentVolumeClaim"]["claimName"] for v in pod["volumes"] if v["name"] == "data"} == {"transfer-learning-vol"}
        # every HALT line of the spec fails this pod's index, not the Job
        assert "fails the Job at once" not in _args(specs[name])
        assert _args(specs[name]).count("fails this pod's index at once") == (2 if name == GPU[0] else 1)


def test_the_table_names_each_cell_once_with_its_tree(specs):
    for name, cs in ((GPU[0], B.ref_s1_cells()), (GPU[1], [("leg2", c) for c in B.leg2e50_cells()])):
        rows = re.findall(r"^ +(\d+)\) ROOT_OUT=(\S*); RUN_DIR=(\S*); INITS=\"([^\"]*)\"; CELL_N=(\d+);"
                          r"(?: CELL_TREE=(\S+);)?;$", _args(specs[name]), re.M)
        assert [int(r[0]) for r in rows] == list(range(len(cs)))
        for (_, root, run, inits, n, tree), (t, (r0, d, i, k, n0, s)) in zip(rows, cs):
            assert (root, run, int(n), inits) == (r0, d, n0, f"{i}:{'/workspace/ckpt/' + i + '.pt' if d else ''}:{k}:{s}")
            assert tree == (t if name == GPU[0] else "")


def test_subset_s1_and_the_epochs_are_the_only_changes_in_the_scripts(specs):
    s1, e50 = _live(specs[GPU[0]]), _live(specs[GPU[1]])
    # subset s1 in place of sS, on JetClass and on both benchmarks
    assert s1.count("${SUB1}/train_N${N}_s1.parquet") == 2 and "train_N${N}_s${S}" not in s1
    assert "DSEED=1 " in s1 and "DSEED=${S}" not in s1
    assert e50.count("${SUB1}/train_N${N}_s1.parquet") == 2 and "train_N${N}_s${S}" not in e50
    # the launched epochs on the reference; 50 on the check
    assert "EP=$(epochs_for ${N})" in s1 and "--num-epochs 20" in s1 and "best --dir ${OUT} --epochs 20" in s1
    assert "EP=50 " in e50 and "EP=$(epochs_for ${N})" not in e50 and "recipe_check=leg2_e50" in e50
    # the V100 references' recipe: 1e-3, no multiplier; no weight decay on JetClass, flat+decay on the benchmarks
    assert 'WD=""; RECIPE="head_lr_mult=1 weight_decay=0 recipe=part-from-scratch"; RULE=none' in s1
    assert 'SCHED=""; RECIPE="head_lr_mult=1 lr_schedule=flat+decay recipe=part-from-scratch"; RULE=none' in s1
    assert s1.count("LR=1e-3; MULT=()") == 2 and "LR=1e-3; MULT=()" in e50
    # leg 1 and the JetClass-II subsets are not touched
    for gone in ("/leg1/", "TEST2M", "jc2_v2", "JetClassII_L162"):
        assert gone not in s1 and gone not in e50, gone


# ------------------------------------------------------------------ the shell

def _run_index(text, tmp, monkeypatch, idx):
    monkeypatch.setenv("JOB_COMPLETION_INDEX", str(idx))
    return V._v2_run(text, tmp, [])


def _lines(calls, tmp, key, cell):
    """The calls of one cell that contain key, with tmp written T and the script path dropped."""
    out = [l.replace(str(tmp), "T") for l in calls.splitlines() if key in l and f"/{cell}/net " in l]
    return [" ".join(l.split()[2:]) if l.startswith("PY ") else " ".join(l.split()) for l in out]


def _done(tmp):
    data = tmp / "data"
    return sorted(str(p.parent.relative_to(data)) for p in data.rglob("DONE") if "results" in str(p))


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    """The training and prediction commands of the V100 references (jc1, bench) and of the
    launched leg-2 cell of l188-s1 at N=1e6, under the bash harness."""
    out = {}
    for part in ("jc1", "bench"):
        tmp = tmp_path_factory.mktemp(f"ref-{part}")
        text = (K8S / f"job-ft-v2-ref-v100-{part}-raunav.yaml").read_text()
        env, cl = W._shell_env(tmp, [])
        r = subprocess.run(["bash", "-c", W._redirect(text, tmp, tmp)], capture_output=True, text=True,
                           env=env, cwd=tmp, timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        out[part] = (tmp, cl.read_text())
    tmp = tmp_path_factory.mktemp("launched")
    text = (K8S / "job-ft-v2-legs-best70-l188-s1-n1000000-raunav.yaml").read_text()
    r, calls = V._v2_run(text, tmp, V._inits(text))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    out["launched"] = (tmp, calls)
    return out


def _ref_s1_cell(idx):
    tree, (root, _, init, _, n, s) = B.ref_s1_cells()[idx]
    return tree, root, f"{tree}/{init}/N{n}/s{s}", n, s


@pytest.mark.parametrize("idx", [0, 7, 9, 19])
def test_a_reference_index_runs_its_one_cell_by_the_v100_references_commands_on_subset_s1(
        specs, refs, tmp_path, monkeypatch, idx):
    tree, root, cell, n, s = _ref_s1_cell(idx)
    r, calls = _run_index(specs[GPU[0]], tmp_path, monkeypatch, idx)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert _done(tmp_path) == [f"{root}/{cell}"[len("/data/"):]]
    [train] = _lines(calls, tmp_path, "weaver.py", cell)
    assert f"--seed {s} " in train and f"train_N{n}_s1.parquet" in train and "lr_mult" not in train
    assert "--start-lr 1e-3" in train and "--lr-scheduler" not in train and "--load-model" not in train
    assert ("weight_decay 0.01" in train) == (tree != "leg2")
    man = next(l for l in calls.splitlines() if "smoke_checks.py manifest" in l)
    assert "lr=1e-3 head_lr_mult=1 " in man and "recipe=part-from-scratch" in man and "ckpt_rule=none" in man
    assert f"train_N{n}_s1.parquet" in man and "ft_v2.py resolve" not in calls
    # the V100 reference's command for the same seed and size, on subset s1 and in this tree
    part = "jc1" if tree == "leg2" else "bench"
    rtmp, rcalls = refs[part]
    want = [l.replace(f"train_N{n}_s{s}.", f"train_N{n}_s1.").replace(f"{B.V2_REF_ROOT}/", f"{B.V2_REF_S1_ROOT}/")
            for l in _lines(rcalls, rtmp, "weaver.py", cell)]
    assert [train.split()] == [l.split() for l in want]
    if tree == "leg2":
        pred = _lines(calls, tmp_path, "WEAVER --predict", cell)
        assert pred and pred == [l.replace(f"{B.V2_REF_ROOT}/", f"{B.V2_REF_S1_ROOT}/")
                                 for l in _lines(rcalls, rtmp, "WEAVER --predict", cell)]
    if tree == "leg_qg":
        assert list((tmp_path / "data").rglob("logits_herwig.npy"))
    # an index whose cell is DONE skips it and completes, before the GPU checks
    again = tmp_path / "again"
    (again / "data" / _done(tmp_path)[0]).mkdir(parents=True)
    (again / "data" / _done(tmp_path)[0] / "DONE").touch()
    r, calls = _run_index(specs[GPU[0]], again, monkeypatch, idx)
    assert r.returncode == 0 and "(DONE)" in r.stdout and "weaver.py" not in calls and "gpu_probe.py" not in calls


@pytest.mark.parametrize("idx", [0, 5])
def test_a_check_index_trains_50_epochs_at_the_launched_or_the_reference_recipe(
        specs, refs, tmp_path, monkeypatch, idx):
    root, run, init, k, n, s = B.leg2e50_cells()[idx]
    cell = f"leg2/{init}/N{n}/s{s}"
    r, calls = _run_index(specs[GPU[1]], tmp_path, monkeypatch, idx)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert _done(tmp_path) == [f"{root}/{cell}"[len("/data/"):]]
    [train] = _lines(calls, tmp_path, "weaver.py", cell)
    assert "--num-epochs 50 " in train and f"--seed {s} " in train and f"train_N{n}_s1.parquet" in train
    man = next(l for l in calls.splitlines() if "smoke_checks.py manifest" in l)
    assert "epochs=50 " in man and "recipe_check=leg2_e50" in man
    if run:
        assert "lr=1e-4 head_lr_mult=50 weight_decay=0.01 wave=v2 ckpt_rule=best70" in man
        rtmp, rcalls = refs["launched"]
        old = f"{B.V2_ROOT}/best70/"
    else:
        assert "lr=1e-3 head_lr_mult=1 weight_decay=0 recipe=part-from-scratch wave=v2 ckpt_rule=none" in man
        rtmp, rcalls = refs["jc1"]
        old = f"{B.V2_REF_ROOT}/jc1/"
    want = [l.replace("--num-epochs 10 ", "--num-epochs 50 ").replace(f"train_N{n}_s{s}.", f"train_N{n}_s1.")
            .replace(old, f"{B.LEG2E50_ROOT}/") for l in _lines(rcalls, rtmp, "weaver.py", cell)]
    assert [train.split()] == [l.split() for l in want]


def test_an_index_beyond_the_table_halts_its_index(specs, tmp_path, monkeypatch):
    r, calls = _run_index(specs[GPU[0]], tmp_path, monkeypatch, 20)
    assert r.returncode == B.EXIT_HALT and "no cell for index 20" in r.stdout and "weaver" not in calls


# ------------------------------------------------------------------ the read-outs

def test_each_readout_waits_for_exactly_its_cells_then_reads_them(specs):
    s1 = B.ref_s1_cells()
    dirs = lambda trees: sorted(f"{c[0]}/{t}/{c[2]}/N{c[4]}/s{c[5]}" for t, c in s1 if t in trees)
    want = {READ[0]: (dirs({"leg2"}), [("leg2_metrics", f"{B.V2_REF_S1_ROOT}/jc1/leg2", False)],
                      f"{B.V2_REF_S1_ROOT}_leg2_metrics"),
            READ[1]: (dirs({"leg_top", "leg_qg"}), [("bench_metrics", f"{B.V2_REF_S1_ROOT}/bench", False),
                                                    ("bench_metrics", f"{B.V2_REF_S1_ROOT}/bench", True)],
                      f"{B.V2_REF_S1_ROOT}_bench_metrics"),
            READ[2]: (sorted(B._cell_dirs(B.leg2e50_cells(), "leg2")),
                      [("leg2_metrics", f"{B.LEG2E50_ROOT}/leg2", False)], f"{B.LEG2E50_ROOT}_leg2_metrics")}
    for name, (cells, runs, out) in want.items():
        text = specs[name]
        assert sorted(_words(text, "w")) == cells, name
        got = R._calls(text)
        assert [(sc, R._flag(a, "--root"), bool(R._flag(a, "--herwig"))) for sc, a in got] == \
               [(sc, [root], h) for sc, root, h in runs]
        for sc, a in got:
            if sc == "leg2_metrics":
                assert R._flag(a, "--macro-auc") == [True] and R._flag(a, "--auc-stride") == [str(B.LEG1_AUC_STRIDE)]
        assert f"OUT={out}\n" in _args(text)
        d = yaml.safe_load(text)
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert d["spec"]["backoffLimit"] == 1 and "nvidia.com/gpu" not in c["resources"]["limits"]
        assert {e["name"]: e.get("value") for e in c["env"]}["REPO_REF"] == B.PIN_V2_EXTRA
        assert d["metadata"]["name"].endswith("-raunav") and "sleep 1800" in _args(text)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_readout_waits_halts_after_its_limit_and_reads_once_every_cell_is_done(specs, tmp_path, which):
    text = specs[READ[which]]
    dirs = _words(text, "w")
    r, calls = R._run(text.replace(f"-lt {B.WAIT_LIMIT_S} ]", "-lt 0 ]"), tmp_path / "early")
    assert r.returncode == B.EXIT_HALT and f"{len(dirs)} " in r.stdout and "_metrics.py" not in calls
    for c in dirs:
        p = tmp_path / "late/data" / c[len("/data/"):]
        p.mkdir(parents=True, exist_ok=True)
        (p / "DONE").touch()
    r, calls = R._run(text, tmp_path / "late")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "late/data" / re.search(r"^ *OUT=/data/(\S+)$", _args(text), re.M).group(1)
    files = ["leg2_metrics.json"] if which != 1 else ["bench_metrics.json", "bench_metrics_herwig.json"]
    assert sorted(p.name for p in out.iterdir()) == files and not list(out.parent.glob("*.staging.*"))
    # applied again, it refuses at once and reads nothing
    (tmp_path / "late/calls.log").unlink()
    r, calls = R._run(text, tmp_path / "late")
    assert r.returncode == B.EXIT_HALT and "use a new output directory" in r.stdout and "_metrics.py" not in calls


# ------------------------------------------------------------------ the generator

def test_the_pin_is_a_tag_still_to_be_made_and_the_flag_is_emitted_alone():
    assert B.PIN_V2_EXTRA == "mtx-s2.20"
    B.verify_pin(B.PIN_V2_EXTRA, True, B.V2_EXTRA_NEEDED)
    cli = [sys.executable, str(ROOT / "scripts/build_ft_jobs.py"), "--v2-extra", "--check-only", "--pin-not-yet-tagged"]
    r = subprocess.run(cli, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count(f"checked (pin {B.PIN_V2_EXTRA})") == 5
    for other in ("--v2-leg188", "--v2-ref-v100"):
        r = subprocess.run(cli + [other], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "emitted alone" in r.stderr, other


def test_the_leg188_and_reference_specs_are_unchanged_by_the_shared_helpers():
    for name, text in {**B.build(B.PIN_V2_LEG188, v2_leg188=True), **B.v2_ref_v100_specs()}.items():
        assert (K8S / name).read_text() == text, name
