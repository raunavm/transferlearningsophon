"""leg1_188 and the top-tagging recipe check (scripts/build_ft_jobs.py, "leg1_188"; plan
2026-10-29): two Indexed Jobs on V100 whose pod index names one cell, in paper order, with
the launched cells' recipe but for what each changes; their read-outs, which wait in the pod;
the 188-class data config; the final read-outs' wait; and leg1_metrics.py --num-classes 188.
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_jobs_leg188", "scripts/build_ft_jobs.py")
V = _load("v2ft_harness_leg188", "tests/test_ft_v2_finetune_specs.py")
R = _load("v2ro_harness_leg188", "tests/test_ft_v2_readout_specs.py")
GPU = ("job-ft-v2-leg188-raunav.yaml", "job-ft-v2-topcheck-raunav.yaml")
READ = ("job-ft-v2-leg188-metrics-raunav.yaml", "job-ft-v2-topcheck-metrics-raunav.yaml")


@pytest.fixture(scope="module")
def specs():
    return B.build(B.PIN_V2_LEG188, v2_leg188=True)


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


def test_the_188_class_config_is_the_l188_arm_without_its_weights():
    cf = _load("build_finetune_configs_leg188", "scripts/build_finetune_configs.py")
    src = cf.SOURCES["JetClassII_L188_noweight.yaml"]
    header, body = cf.strip_weights(src.read_text(), src, "JetClassII_L188_noweight.yaml")
    assert (ROOT / B.V2_LEG188_DATA).read_text() == header + body
    assert src.read_text().startswith(body.rstrip("\n")) and not re.search(r"^weights:", body, re.M)
    # the labels are the native classes, unchanged (label i is jet_label == i)
    assert "+ 187 * ((jet_label == 187))" in body and "-o num_classes 188" in body


def test_each_is_an_indexed_job_of_mine_on_v100_with_the_retry_policy_per_index(specs):
    cells = {GPU[0]: B.leg188_cells(), GPU[1]: B.topcheck_cells()}
    par = {GPU[0]: B.LEG188_PARALLEL, GPU[1]: B.TOPCHECK_PARALLEL}
    assert (len(cells[GPU[0]]), len(cells[GPU[1]]), par[GPU[0]], par[GPU[1]]) == (76, 6, 6, 3)
    for name in GPU:
        d = yaml.safe_load(specs[name])
        assert d["metadata"]["name"] == name[4:-5] and "raunav" in name
        s = d["spec"]
        assert s["completionMode"] == "Indexed" and s["completions"] == len(cells[name])
        assert s["parallelism"] == par[name] and s["backoffLimitPerIndex"] == B.ROBUST_BACKOFF
        assert "backoffLimit" not in s and "maxFailedIndexes" not in s
        assert s["podFailurePolicy"]["rules"] == [
            {"action": "FailIndex", "onExitCodes": {"containerName": "main", "operator": "In", "values": [B.EXIT_HALT]}},
            {"action": "Count", "onExitCodes": {"containerName": "main", "operator": "In", "values": [B.EXIT_NODE_FAULT]}},
            {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}]
        pod = s["template"]["spec"]
        assert pod["restartPolicy"] == "Never"
        c = pod["containers"][0]
        env = {e["name"]: e.get("value") for e in c["env"]}
        assert env["REPO_REF"] == B.PIN_V2_LEG188 and env["GPU_PRODUCT"] == "Tesla-V100-SXM2-32GB"
        assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
        terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
        expr = {e["key"]: e for t in terms for e in t["matchExpressions"]}
        assert expr["nvidia.com/gpu.product"]["values"] == list(B.V2_SPLIT_GPUS) == ["Tesla-V100-SXM2-32GB"]
        assert set(B._v2_off_nodes()) | set(B.GPU_LOST_LATER) <= set(expr["kubernetes.io/hostname"]["values"])
        assert "exit ${HALT}" in _live(specs[name]) and "fails this pod's index" in _args(specs[name])


def test_the_cells_are_in_paper_order_and_the_table_names_each_once(specs):
    assert B.V2_LEG188_RUNS[:12] == tuple(f"{a}-s{s}" for a in ("l188", "l162", "r42q1", "r16q1") for s in (1, 2, 3))
    assert B.V2_LEG188_RUNS[12:] == ("l162mass-s1", "l162mass-s2", "r16q1mass-s1", "r16q1mass-s2")
    assert {f"mtx-{r}" for r in B.V2_LEG188_RUNS} <= set(B.v2_paper_runs())
    cells = B.leg188_cells()
    assert [(c[2], c[4], c[5]) for c in cells] == (
        [(r, n, 1) for r in B.V2_LEG188_RUNS for n in B.SIZES]
        + [(B.SCRATCH_REF, n, s) for s in (1, 2, 3) for n in B.SIZES])
    assert {c[0] for c in cells[:64]} == {f"{B.V2_ROOT}/best70"} and {c[0] for c in cells[64:]} == {f"{B.V2_REF_ROOT}/jc2"}
    assert all(c[1] for c in cells[:64]) and not any(c[1] for c in cells[64:])
    for name, cs in ((GPU[0], cells), (GPU[1], B.topcheck_cells())):
        rows = re.findall(r"^ +(\d+)\) ROOT_OUT=(\S*); RUN_DIR=(\S*); INITS=\"([^\"]*)\"; CELL_N=(\d+);;$",
                          _args(specs[name]), re.M)
        assert [int(r[0]) for r in rows] == list(range(len(cs)))
        for (_, root, run, inits, n), (r0, d, i, k, n0, s) in zip(rows, cs):
            assert (root, run, int(n)) == (r0, d, n0)
            assert inits == f"{i}:{'/workspace/ckpt/' + i + '.pt' if d else ''}:{k}:{s}"


def test_leg188_is_the_launched_leg1_recipe_at_188_classes(specs):
    live = _live(specs[GPU[0]])
    for want, n in ((f"--data-config {B.V2_LEG188_DATA}", 1), (f"data_config={B.V2_LEG188_DATA}", 1),
                    ("-o num_classes 188 ", 1), ("num_classes=188 ", 1), ("--num-classes 188 --arm FT188_", 1),
                    ("--max-jets 2000000 --stride 4 --save-logits", 1), ("--n 500000 --k 188", 1),
                    ("slim_outputs.py leg1 --dir ${OUT}/features_v2 --stride 1\n", 1),
                    ("OUT=${ROOT_OUT}/leg1_188/${name}/N${N}/s${S}", 1), ("for N in ${CELL_N}; do", 1),
                    ("--steps-per-epoch-val -1", 1), ("cell_resume.py best --dir ${OUT} --epochs ${EP}", 1),
                    ("${SUB2}/train_N${N}_s1.parquet", 2)):
        assert live.count(want) == n, want
    for gone in ("num_classes 162", "--k 162", "JetClassII_L162", "leg2", "/data/finetune/jc1"):
        assert gone not in live, gone
    # the launched leg-1 cell's epochs, samples and recipe, unchanged
    launched = _live((K8S / "job-ft-v2-legs-best70-l188-s1-n1000-raunav.yaml").read_text())
    line = lambda text, key: next(l for l in text.splitlines() if key in l)
    for key in ("epochs_for () {", "samples_for () {", "HEAD_MULT=("):
        assert line(live, key) == line(launched, key), key
    # the pretrained branch as launched; the from-scratch one at the references' 1e-3
    assert line(live, "LR=1e-4; MULT=") == line(launched, "LR=1e-4; MULT=").replace("LR=5e-4", "LR=1e-3")
    assert 'WD="--optimizer-option weight_decay 0.01"' in live and 'RECIPE="head_lr_mult=50 weight_decay=0.01"' in live
    assert 'LR=1e-3; MULT=()' in live and 'WD=""; RECIPE="head_lr_mult=1 weight_decay=0 recipe=part-from-scratch"' in live


def test_the_top_check_is_the_launched_top_recipe_with_100_epochs_on_subset_s1(specs):
    live = _live(specs[GPU[1]])
    assert "for D in top; do" in live and 'top) echo "10000";;' in live and "net_epoch-19" not in live
    assert "--num-epochs 100" in live and "best --dir ${OUT} --epochs 100" in live and "epochs=100" in live
    assert "DSEED=1 " in live and "DSEED=${S}" not in live
    assert 'SCHED="--lr-scheduler none"' in live and 'SCHED=""' in live and "LR=1e-3; MULT=()" in live
    assert B.TOPCHECK_ROOT == "/data/results/ft_v2/recipe_check/leg_top_e100"


# ------------------------------------------------------------------ the shell

def _run_index(text, tmp_path, monkeypatch, idx):
    monkeypatch.setenv("JOB_COMPLETION_INDEX", str(idx))
    return V._v2_run(text, tmp_path, [])


def _weaver(calls):
    return [l for l in calls.splitlines() if "ft_weaver.py" in l]


@pytest.mark.parametrize("idx", [0, 71])
def test_an_index_runs_its_one_cell_under_bash_and_a_done_cell_is_skipped(specs, tmp_path, monkeypatch, idx):
    root, run, init, k, n, s = B.leg188_cells()[idx]
    r, calls = _run_index(specs[GPU[0]], tmp_path, monkeypatch, idx)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    data = tmp_path / "data"
    done = sorted(p.parent.relative_to(data) for p in data.rglob("DONE") if "results" in str(p))
    assert done == [pathlib.Path(f"{root}/leg1_188/{init}/N{n}/s{s}"[len("/data/"):])]
    [w] = _weaver(calls)
    assert f"--seed {s} " in w and f"train_N{n}_s1.parquet" in w and "-o num_classes 188" in w
    assert "gpu_probe.py" in calls and "ft_v2.py verify" in calls
    man = next(l for l in calls.splitlines() if "smoke_checks.py manifest" in l)
    if run:
        assert "--start-lr 1e-4" in w and "weight_decay 0.01" in w and "lr_mult" in w and "--load-model-weights" in w
        assert "lr=1e-4 head_lr_mult=50 weight_decay=0.01 wave=v2 ckpt_rule=best70 task=jc2_188" in man
        assert f"ft_v2.py resolve --run-dir {run}".replace("/data", str(data)) in calls
    else:
        assert "--start-lr 1e-3" in w and "weight_decay" not in w and "lr_mult" not in w and "--load-model" not in w
        assert "lr=1e-3 head_lr_mult=1 weight_decay=0 recipe=part-from-scratch wave=v2 ckpt_rule=none" in man
        assert "ft_v2.py resolve" not in calls
    # an index whose cell is DONE (its pod lost after the marker) skips it and completes,
    # before the GPU checks and the subset hashes
    again = tmp_path / "again"
    (again / "data" / done[0] / "DONE").parent.mkdir(parents=True)
    (again / "data" / done[0] / "DONE").touch()
    r, calls = _run_index(specs[GPU[0]], again, monkeypatch, idx)
    assert r.returncode == 0 and "(DONE)" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert not _weaver(calls) and "gpu_probe.py" not in calls and "ft_v2.py" not in calls


def test_an_index_beyond_the_table_halts_its_index(specs, tmp_path, monkeypatch):
    r, calls = _run_index(specs[GPU[0]], tmp_path, monkeypatch, 76)
    assert r.returncode == B.EXIT_HALT and "no cell for index 76" in r.stdout and not _weaver(calls)


@pytest.mark.parametrize("idx", [1, 4])
def test_a_top_check_index_trains_100_epochs_on_subset_s1(specs, tmp_path, monkeypatch, idx):
    root, run, init, k, n, s = B.topcheck_cells()[idx]
    r, calls = _run_index(specs[GPU[1]], tmp_path, monkeypatch, idx)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    data = tmp_path / "data"
    assert (data / f"{root}/leg_top/{init}/N{n}/s{s}"[len("/data/"):] / "DONE").exists()
    [w] = _weaver(calls)
    assert f"--seed {s} " in w and "--num-epochs 100" in w and "top_sub/train_N10000_s1.parquet" in w
    assert ("--lr-scheduler none" in w) == bool(run) and ("--start-lr 1e-4" if run else "--start-lr 1e-3") in w


# ------------------------------------------------------------------ the read-outs

def test_each_readout_waits_for_exactly_its_cells_then_reads_them(specs):
    l188 = specs[READ[0]]
    assert sorted(_words(l188, "w")) == sorted(B._cell_dirs(B.leg188_cells(), "leg1_188"))
    [(script, argv)] = R._calls(l188)
    assert script == "leg1_metrics" and R._flag(argv, "--num-classes") == ["188"]
    assert R._flag(argv, "--auc-stride") == ["1"] and not R._flag(argv, "--expect-cells")
    assert R._flag(argv, "--root") == [f"${{ROOT}} {B.V2_REF_ROOT}/jc2/leg1_188"]
    assert R._flag(argv, "--probe-labels") == [B.LEG188_PROBE_LABELS]
    assert B.LEG188_PROBE_LABELS == "/data/results/eval/v2_linprobe/init-s1/jc2/test/init/label188.npy"
    assert f"ROOT={B.V2_ROOT}/best70/leg1_188\n" in _args(l188)
    assert f"OUT={B.V2_ROOT}/best70_leg188_leg1_metrics\n" in _args(l188)
    top = specs[READ[1]]
    assert sorted(_words(top, "w")) == sorted(B._cell_dirs(B.topcheck_cells(), "leg_top"))
    [(script, argv)] = R._calls(top)
    assert script == "bench_metrics" and R._flag(argv, "--root") == [B.TOPCHECK_ROOT]
    assert R._flag(argv, "--datasets") == ["top"]
    for t in (l188, top):
        d = yaml.safe_load(t)
        assert d["spec"]["backoffLimit"] == 1 and "nvidia.com/gpu" not in d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        assert {e["name"]: e.get("value") for e in d["spec"]["template"]["spec"]["containers"][0]["env"]}["REPO_REF"] == B.PIN_V2_LEG188_READOUT
        assert "sleep 1800" in _args(t) and f"-lt {6 * 86400} ]" in _args(t)


@pytest.mark.parametrize("which", [0, 1])
def test_a_readout_waits_halts_after_its_limit_and_reads_once_every_cell_is_done(specs, tmp_path, which):
    text = specs[READ[which]]
    dirs = _words(text, "w")
    # nothing DONE yet: it waits; with the limit at zero it halts at once and reads nothing
    r, calls = R._run(text.replace(f"-lt {B.WAIT_LIMIT_S} ]", "-lt 0 ]"), tmp_path / "early")
    assert r.returncode == B.EXIT_HALT and f"{len(dirs)} " in r.stdout and "_metrics.py" not in calls
    for c in dirs + (_words(text, "c") if which == 0 else []):
        p = tmp_path / "late/data" / c[len("/data/"):]
        p.mkdir(parents=True, exist_ok=True)
        (p / "DONE").touch()
    r, calls = R._run(text, tmp_path / "late")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert calls.count("_metrics.py") == 1 and "waiting for" not in r.stdout


def test_the_final_readouts_wait_for_every_cell_of_their_rule_the_four_running_ones_included():
    exp = json.loads((ROOT / B.V2_EXPECTED).read_text())
    for leg, trees in (("leg1", ["leg1"]), ("leg2", ["leg2"]), ("bench", ["leg_top", "leg_qg"])):
        text = (K8S / f"job-ft-v2-best70-{leg}-metrics-raunav.yaml").read_text()
        got = set(_words(text, "w"))
        want = {f"{B.V2_ROOT}/best70/{t}/{c}" for t in trees for c in exp[f"best70/{t}"]}
        assert got == want and len(_words(text, "w")) == len(want)
        assert _args(text).index("for w in") < _args(text).index("ROOT=")
    leg1 = set(_words((K8S / "job-ft-v2-best70-leg1-metrics-raunav.yaml").read_text(), "w"))
    assert {f"{B.V2_ROOT}/best70/leg1/{c}" for c in ("l162-s3/N100000/s1", "l162-s3/N1000000/s1",
                                                    "r16q1-s3/N1000000/s1")} <= leg1


def test_the_pin_is_a_tag_still_to_be_made_and_the_tree_carries_what_the_specs_run():
    assert B.PIN_V2_LEG188 == B.PIN_V2_READOUT == "mtx-s2.19"
    assert B.PIN_V2_LEG188_READOUT == "mtx-s2.20"      # the read-outs had not run when probe.py changed
    B.verify_pin(B.PIN_V2_LEG188, True, B.V2_LEG188_NEEDED)
    B.verify_pin(B.PIN_V2_LEG188_READOUT, True, B.V2_LEG188_NEEDED)
    cli = [sys.executable, str(ROOT / "scripts/build_ft_jobs.py"), "--v2-leg188", "--check-only", "--pin-not-yet-tagged"]
    r = subprocess.run(cli, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count(f"checked (pin {B.PIN_V2_LEG188})") == 2
    assert r.stdout.count(f"checked (pin {B.PIN_V2_LEG188_READOUT})") == 2
    r = subprocess.run(cli + ["--v2-readouts"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "emitted alone" in r.stderr


def test_the_docstring_states_the_v2_from_scratch_rate():
    assert "1e-3 in every v2 from-scratch reference" in " ".join(B.__doc__.split())
    assert B.LR_SCRATCH_PART == "1e-3"


# ------------------------------------------------------------------ leg1_metrics.py --num-classes 188

L1 = _load("leg1_metrics_188", "experiments/FT/leg1_metrics.py")
PROBE = _load("probe_188", "experiments/EVAL/probe.py")


def _cell188(root, init="l188-s1", n="N1000", seed="s1", k=188, absent=(5, 170), labels=None):
    fd = root / init / n / seed / "features_v2"
    fd.mkdir(parents=True)
    lab = (np.tile(np.arange(188), 12) if labels is None else labels).astype(np.int16)
    np.save(fd / "label188.npy", lab)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(lab.size, k)).astype(np.float32)
    ok = lab < k
    z[np.arange(lab.size)[ok], lab[ok]] += 2.0       # informative, not perfect
    np.save(fd / "logits.npy", z)
    L1.SLIM.slim_leg1(fd, 1)                         # as the cells keep it: every row, stride 1
    sub = root.parent / "subsets"
    sub.mkdir(exist_ok=True)
    (sub / "manifest.json").write_text(json.dumps({"class_coverage": {
        f"train_{n}_s1.parquet": {"n_jets": 1000, "n_classes_present": 188 - len(absent),
                                  "classes_absent": list(absent), "counts": [0] * 188}}}))
    (fd.parent / "ft_manifest.json").write_text(json.dumps({"subset": str(sub / f"train_{n}_s1.parquet")}))
    return fd


def test_the_188_path_scores_the_native_classes_with_the_summed_qcd(tmp_path):
    fd = _cell188(tmp_path / "leg1_188")
    cov = L1.subset_coverage(json.loads((fd.parent / "ft_manifest.json").read_text())["subset"], {})
    m = L1.cell_metrics_188(fd, PROBE, 1, cov)
    lab = np.load(fd / "label188.npy").astype(int)
    p = L1.softmax(np.load(fd / "logits_auc.npy"))
    from sklearn.metrics import roc_auc_score
    auc = {c: roc_auc_score((lab == c).astype(int), p[:, c]) for c in range(188)}
    assert m["n_classes_present"] == 188 and set(m["per_class_auc"]) == set(range(188))
    assert np.isclose(m["macro_auc_ovr"], np.mean(list(auc.values())))
    assert np.isclose(m["macro_auc_resonant"], np.mean([auc[c] for c in range(161)]))
    assert np.isclose(m["macro_auc_qcd"], np.mean([auc[c] for c in range(161, 188)]))
    assert np.isclose(m["macro_auc_trained"], np.mean([auc[c] for c in range(188) if c not in (5, 170)]))
    assert m["n_classes_absent_train"] == 2 and m["accuracy"] == pytest.approx((p.argmax(1) == lab).mean())
    # one rejection per resonant class, against ALL 27 QCD classes summed
    assert set(m["rejection_vs_qcd"]) == set(range(161)) and m["eps_s"] == 0.5
    c = 7
    sel, p_qcd = (lab == c) | (lab >= 161), p[:, 161:].sum(axis=1)
    d = p[sel, c] / (p[sel, c] + p_qcd[sel])
    rej, eps_b, bound, n_pass, _ = PROBE.rejection_at((lab[sel] == c).astype(int), d, 0.5)
    got = m["rejection_vs_qcd"][c]
    assert got == pytest.approx({"rejection": rej, "eps_b": eps_b, "n_bkg_pass": n_pass, "is_bound": bool(bound)})
    assert got["n_bkg_pass"] == n_pass and got["is_bound"] is bool(bound)
    # ...which is not the rejection of one QCD class, the 162 path's single index
    one = (lab == c) | (lab == 161)
    d1 = p[one, c] / (p[one, c] + p[one, 161])
    assert PROBE.rejection_at((lab[one] == c).astype(int), d1, 0.5)[1] != pytest.approx(eps_b)
    assert m["n_jets"] == m["n_jets_auc"] == lab.size and m["auc_stride"] == 1


def test_the_188_path_refuses_a_162_head_and_non_native_labels(tmp_path):
    fd = _cell188(tmp_path / "a", k=162)
    with pytest.raises(SystemExit, match="not 188"):
        L1.cell_metrics_188(fd, PROBE, 1, {"classes_absent": []})
    fd = _cell188(tmp_path / "b", labels=np.array([0, 1, 188] * 10))
    with pytest.raises(SystemExit, match="not native"):
        L1.cell_metrics_188(fd, PROBE, 1, {"classes_absent": []})
    fd = _cell188(tmp_path / "c")
    with pytest.raises(SystemExit, match="classes_absent"):
        L1.cell_metrics_188(fd, PROBE, 1, {"n_classes_present": 188})


def test_the_188_cli_writes_every_metric_and_its_seed_summary(tmp_path):
    root = tmp_path / "leg1_188"
    _cell188(root, seed="s1")
    _cell188(root, init="scratch-v2", seed="s2")
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "experiments/FT/leg1_metrics.py"), "--root", str(root), "--out", str(out),
           "--auc-stride", "1", "--num-classes", "188"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / "leg1_metrics.json").read_text())
    assert res["num_classes"] == 188 and res["auc_stride"] == 1
    cell = res["cells"]["l188-s1"]["N1000"]["s1"]
    for k in ("accuracy", "macro_auc_ovr", "macro_auc_resonant", "macro_auc_qcd", "macro_auc_trained",
              "n_classes_absent_train", "per_class_auc", "rejection_vs_qcd", "train_coverage", "label188_sha256"):
        assert k in cell, k
    assert set(cell["rejection_vs_qcd"]["0"]) == {"rejection", "eps_b", "n_bkg_pass", "is_bound"}
    s = res["summary"]["l188-s1"]["N1000"]
    assert s["n_seeds"] == 1 and s["macro_auc_qcd_mean"] == cell["macro_auc_qcd"] and s["macro_auc_qcd_sd"] is None
    # a stride the cells were not slimmed at is refused, and so is the 162 path on these cells
    r = subprocess.run(cmd[:-4] + ["--auc-stride", "4", "--num-classes", "188"], capture_output=True, text=True)
    assert r.returncode != 0 and "every 1th row" in r.stdout + r.stderr
    r = subprocess.run(cmd[:-2], capture_output=True, text=True)
    assert r.returncode != 0 and "162" in r.stdout + r.stderr


def test_the_188_cli_checks_the_cells_are_the_probes_auc_jets(tmp_path):
    root = tmp_path / "leg1_188"
    lab = np.load(_cell188(root) / "label188.npy")
    cmd = [sys.executable, str(ROOT / "experiments/FT/leg1_metrics.py"), "--root", str(root),
           "--auc-stride", "1", "--num-classes", "188", "--probe-labels"]
    # the probe cache holds every test jet; its AUC jets are every 4th of the first 2M
    probe = np.repeat(lab, 4)
    np.save(tmp_path / "ok.npy", probe)
    r = subprocess.run(cmd + [str(tmp_path / "ok.npy"), "--out", str(tmp_path / "a")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((tmp_path / "a/leg1_metrics.json").read_text())["probe_labels"] == str(tmp_path / "ok.npy")
    np.save(tmp_path / "shifted.npy", np.roll(probe, 1))
    r = subprocess.run(cmd + [str(tmp_path / "shifted.npy"), "--out", str(tmp_path / "b")], capture_output=True, text=True)
    assert r.returncode != 0 and "not the linear probe's AUC jets" in r.stdout + r.stderr
    assert not (tmp_path / "b").exists()
    r = subprocess.run(cmd[:6] + ["--probe-labels", str(tmp_path / "ok.npy"), "--out", str(tmp_path / "c")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "needs --num-classes 188" in r.stderr
