"""The v2 pretraining specs (scripts/build_mtx_launch.py v2_spec and friends)
and the v2 run manifest (scripts/write_run_manifest.py --driver pretrain_v2)."""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_mtx_launch as b  # noqa: E402

TAG = "mtx-s1.99"


def _script(spec: str) -> str:
    return yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _brace_count(expr: str) -> int:
    lo, hi = re.search(r"\{(\d+)\.\.(\d+)\}", expr).groups()
    return int(hi) - int(lo) + 1


@pytest.fixture(scope="module")
def grid():
    return b.v2_grid_specs(TAG)


def test_one_spec_per_arm_and_run(grid):
    want = sum(int(a["runs"]) for a in b.v2_arms())
    assert len(grid) == want
    for arm in b.v2_arms():
        for r in range(1, int(arm["runs"]) + 1):
            assert f"job-{b.v2_job_name(arm['name'], r)}.yaml" in grid


def test_every_spec_is_valid_yaml_and_bash(grid):
    for fn, spec in grid.items():
        d = yaml.safe_load(spec)
        assert d["metadata"]["name"] + ".yaml" == fn[len("job-"):]
        assert "raunav" in d["metadata"]["name"] and len(d["metadata"]["name"]) <= 63
        r = subprocess.run(["bash", "-n"], input=_script(spec), text=True, capture_output=True)
        assert r.returncode == 0, (fn, r.stderr)


def test_retry_policy_matches_the_fine_tuning_jobs(grid):
    ft = yaml.safe_load(b._ft_jobs().POD_FAILURE_POLICY_RESUME)["podFailurePolicy"]
    for spec in grid.values():
        d = yaml.safe_load(spec)["spec"]
        assert d["backoffLimit"] == b.V2_BACKOFF == 60
        rules = d["podFailurePolicy"]["rules"]
        assert rules == ft["rules"] and rules == [
            {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}},
            {"action": "Count", "onExitCodes": {"containerName": "main", "operator": "In", "values": [43]}},
            {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}]
        c = d["template"]["spec"]["containers"][0]
        assert c["name"] == "main" and d["template"]["spec"]["restartPolicy"] == "Never"
        s = _script(spec)
        assert "trap on_exit EXIT" in s and "attempts/failed-" in s and "exit ${HALT}" in s
        assert "NODE_FAULT=43" in s and f"-lt {b.NODE_FAULT_LIMIT} ]" in s and b.NODE_FAULT_LIMIT == 6


def test_every_grid_job_is_created_suspended(grid):
    assert all(yaml.safe_load(spec)["spec"]["suspend"] is True for spec in grid.values())
    assert "suspend" not in yaml.safe_load(b.v2_smoke_specs(TAG)["job-mtx2-smoke-3090-raunav.yaml"])["spec"]


def test_the_runtime_is_pinned_by_digest_and_recorded(grid):
    digest = "sha256:db235b515a278198ebc6dc2c607c9c38cfb10e6b46e29ed2847ec9f4af6191e7"
    for spec in grid.values():
        c = yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]
        assert c["image"] == f"gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning@{digest}"
        assert {e["name"]: e.get("value") for e in c["env"]}["IMAGE_DIGEST"] == digest
        assert ":cu121" not in spec


def test_the_gpu_is_probed_before_the_trap_and_the_storage_guard_comes_first(grid):
    for spec in grid.values():
        s = _script(spec)
        probe = s.index("python3 experiments/FT/gpu_probe.py || node_fault preflight")
        assert s.index('[ "${USE}" -le "${CEIL}" ]') < s.index("mkdir -p ${OUT}") < probe
        assert probe < s.index("trap on_exit EXIT") < s.index("python3 experiments/MTX/pretrain_v2.py")
        assert "CEIL=85; [ \"${LAST}\" -lt 0 ] || CEIL=95" in s
        assert b.CUDA_FAULT in s and b.CUDA_FAULT == b._ft_jobs().CUDA_FAULT
        trap = s[s.index("on_exit () {"):s.index("trap on_exit EXIT")]
        assert trap.index('[ ${rc} -ne 4 ] || node_fault "rc=${rc}"') < trap.index("gpu_probe.py")


def test_the_exit_codes_are_the_drivers():
    src = (ROOT / "experiments" / "MTX" / "pretrain_v2.py").read_text()
    code = {k: int(v) for k, v in re.findall(r"^(EXIT_\w+) = (\d+)", src, re.M)}
    assert code == {"EXIT_HALT": b.EXIT_HALT, "EXIT_RETRY": b.EXIT_NONFINITE, "EXIT_NO_GPU": b.EXIT_NO_GPU}
    assert b.EXIT_NO_GPU == 4 and "return pv.EXIT_NO_GPU" in (ROOT / "experiments" / "MTX" / "finalize_v2.py").read_text()


def test_v2_code_loader_validation_and_output(grid):
    for fn, spec in grid.items():
        s = _script(spec)
        run_id = re.search(r"^RUN_ID=(\S+)$", s, re.M).group(1)
        assert f"OUT={b.V2_ROOT}/${{RUN_ID}}" in s and fn == "job-mtx2-" + run_id[len("mtx-"):] + "-raunav.yaml"
        assert re.fullmatch(r"mtx-[a-z0-9]+-s\d", run_id)
        assert "python3 experiments/MTX/pretrain_v2.py" in s and "seed_weaver" not in s
        assert s.count("experiments/MTX/pretrain_v2.py --seed ${SEED} --out ${OUT} --device cuda ") == 1
        # the sidecar must be its config plus histograms before anything trains (exit 42 otherwise)
        assert s.index("sv.sidecar_mismatch(") < s.index("python3 experiments/MTX/pretrain_v2.py")
        assert 'is not ${CFG} plus reweighting histograms"; exit ${HALT}; }' in s
        lofo = "--extra-selection" in s
        win = "--data-windows 3" if lofo else "--data-fraction 0.2"           # PI 2026-10-01
        assert f"--num-workers 5 --fetch-step 1.0 --data-split-num 200 {win}" in s
        assert f"{win} --keep-checkpoints window" in s                      # the manifest records it
        assert s.count(win) == 2 and ("--data-fraction" in s) != lofo
        assert s.count("--extra-selection '") == (2 if lofo else 0)          # training and manifest
        assert "--fetch-by-files" not in s and "--samples-per-epoch 10240000" in s
        assert "--num-epochs 80" in s and "--start-lr 5e-4" in s and "--use-amp" in s
        val = re.search(r"--data-val (.*?) --data-config", s).group(1).split()
        assert [_brace_count(v) for v in val] == [4, 16, 5] and sum(map(_brace_count, val)) == b.N_VAL_FILES
        assert "Res2P_{0200..0203}" in val[0] and "Res34P_{0860..0875}" in val[1] and "QCD_{0280..0284}" in val[2]
        tr = re.search(r"--data-train (.*?) --data-val", s).group(1).split()
        assert sum(map(_brace_count, tr)) == b.N_TRAIN_FILES
        assert "--driver pretrain_v2" in s and "--keep-checkpoints all" not in s
        assert "--keep-checkpoints window --select-on acc" in s and re.search(r"--select-on acc( --extra-selection '[^']*')? \\", s)
        seed = re.search(r"^SEED=(\d+)$", s, re.M).group(1)
        assert run_id.endswith(f"-s{seed}")


def test_objectives_map_to_their_flags(grid):
    for arm in b.v2_arms():
        s = _script(grid[f"job-{b.v2_job_name(arm['name'], 1)}.yaml"])
        assert f"--network-config {b.ARCH[arm['objective']]}" in s
        if arm["objective"] == "mpm":
            assert "--mpm --mpm-mask-rate 0.40" in s and "num_classes" not in s
        else:
            assert f"-o num_classes {arm['num_classes']} " in s
        assert ("--mass-lambda" in s) == (arm.get("mass_lambda") is not None)
        if arm.get("mass_lambda") is not None:
            assert f"--mass-lambda {float(arm['mass_lambda'])}" in s
        assert ("--extra-selection" in s) == bool(arm.get("extra_selection"))
        assert f"CFG={arm['config']}" in s


def test_scheduling_pins_region_gpu_and_resources(grid):
    by_run = {}
    for spec in grid.values():
        t = yaml.safe_load(spec)["spec"]["template"]["spec"]
        terms = t["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]
        ex = {e["key"]: e for e in terms["matchExpressions"]}
        run = int(re.search(r"^SEED=(\d+)$", _script(spec), re.M).group(1))
        gpu = b.V2_GPU_BY_RUN[run]
        by_run.setdefault(run, set()).update(ex["nvidia.com/gpu.product"]["values"])
        assert ex["topology.kubernetes.io/region"]["values"] == ["us-west"]
        assert ex["nvidia.com/gpu.product"]["values"] == [gpu]
        assert ex["kubernetes.io/hostname"]["operator"] == "NotIn"
        assert ex["kubernetes.io/hostname"]["values"] == list(b.V2_BAD_NODES)    # on every product
        env = {e["name"]: e.get("value") for e in t["containers"][0]["env"]}
        assert env["GPU_PRODUCT"] == gpu and env["REPO_REF"] == TAG
        res = t["containers"][0]["resources"]
        assert res["requests"] == res["limits"] and res["requests"]["nvidia.com/gpu"] == "1"
        assert res["requests"]["cpu"] == b.V2_CPU and res["requests"]["memory"] == b.V2_MEM
    # A14 design change 7 (the L40 check passed): runs 1-3 on the RTX 3090, 4-5 on L40,
    # every arm, so each run index is on one product (I7)
    assert b.V2_GPU_BY_RUN == {1: b.V2_GPU, 2: b.V2_GPU, 3: b.V2_GPU, 4: "NVIDIA-L40", 5: "NVIDIA-L40"}
    assert by_run == {r: {g} for r, g in b.V2_GPU_BY_RUN.items()}


def test_there_is_one_way_to_build_the_grid():
    """Runs 4-5 on L40 is the table's default, not a launch option: the products the paper
    reports (experiments/FIGS/make_tables.py reads them from the grid specs) cannot
    depend on how the grid was built."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_mtx_launch.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--v2 " in r.stdout and "l40" not in r.stdout.lower()


def test_another_gpu_product_gets_its_own_run_directory():
    arm = b.v2_arms()[0]
    n1, _ = b.v2_spec(arm, 1, tag=TAG)
    n2, s2 = b.v2_spec(arm, 1, "NVIDIA-L40", tag=TAG)
    assert n1 != n2 and n2.endswith("-l40-raunav")
    assert yaml.safe_load(s2)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][1]["values"] == ["NVIDIA-L40"]


def test_smoke_specs_stay_inside_the_gate():
    specs = b.v2_smoke_specs(TAG)
    assert set(specs) == {"job-mtx2-smoke-3090-raunav.yaml", "job-mtx2-smoke-a-l40-raunav.yaml",
                          "job-mtx2-smoke-a-a6000-raunav.yaml", "job-mtx2-smoke-a-a40-raunav.yaml"}
    for spec in specs.values():
        s = _script(spec)
        lines = [m.group(0) for m in re.finditer(r"python3 experiments/MTX/pretrain_v2\.py --seed .*", s)]
        assert lines
        for line in lines:
            assert int(re.search(r"--num-epochs (\d+)", line).group(1)) <= 3
            assert int(re.search(r"--samples-per-epoch (\d+)", line).group(1)) <= 200_000
        assert f"OUT={b.SMOKE_ROOT}/" in s
        assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
    s = _script(specs["job-mtx2-smoke-3090-raunav.yaml"])
    assert [m.group(1) for m in re.finditer(r"^RUN_ID=(\S+)$", s, re.M)] == ["smoke-a", "smoke-b", "smoke-c", "smoke-a2"]
    assert s.count("kill -9 ${BG}") == 1 and "net_epoch-1_resume.pt" in s
    assert "-o num_classes 188 " in s and "-o num_classes 17 " in s


def test_end_of_run_smoke_covers_the_objectives_the_first_smoke_did_not():
    specs = b.v2_endrun_smoke_specs(TAG)
    assert set(specs) == {"job-mtx2-endrun-3090-raunav.yaml", "job-mtx2-endrun-l40-raunav.yaml"}
    for f, spec in specs.items():
        y = yaml.safe_load(spec)
        assert "raunav" in y["metadata"]["name"] and not y["spec"].get("suspend")
        s = _script(spec)
        assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
        for line in re.findall(r"python3 experiments/MTX/pretrain_v2\.py --seed .*", s):
            assert int(re.search(r"--num-epochs (\d+)", line).group(1)) == 3
            assert int(re.search(r"--samples-per-epoch (\d+)", line).group(1)) == 200_000
            assert "--device cuda" in line
        assert f"OUT={b.ENDRUN_ROOT}/" in s and "/mtx_v2/mtx-" not in s     # never a grid run's directory
        assert "END-OF-RUN CHECK" in s and s.index("END-OF-RUN CHECK") > s.rindex("pretrain_v2.py --seed")
    s = _script(specs["job-mtx2-endrun-3090-raunav.yaml"])
    assert [m.group(1) for m in re.finditer(r"^RUN_ID=(\S+)$", s, re.M)] == ["endrun-mpm", "endrun-mass-b", "endrun-lofo"]
    assert s.count("kill -9 ${BG}") == 1 and "--mpm " in s and "--mass-lambda 5.0" in s
    assert "--data-windows 3" in s and "--extra-selection" in s
    prod = lambda f: yaml.safe_load(specs[f])["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert {"key": "nvidia.com/gpu.product", "operator": "In", "values": [b.V2_GPU_BY_RUN[4]]} in prod(
        "job-mtx2-endrun-l40-raunav.yaml")


def test_dry_run_specs_request_no_gpu():
    for full in (False, True):
        name, spec = b.v2_dryrun_spec(TAG, full)
        t = yaml.safe_load(spec)["spec"]["template"]["spec"]
        assert "nvidia.com/gpu" not in t["containers"][0]["resources"]["requests"]
        s = _script(spec)
        assert "loader_dryrun.py" in s and ("--full-columns" in s) == full and "raunav" in name
        assert s.index('[ "${USE}" -le 85 ]') < s.index("loader_dryrun.py")       # storage guard first
    name, spec = b.v2_dryrun_spec(TAG, False, "lofo", arm="R16_Q1_LOFO4P", epochs=4)
    s = _script(spec)
    arm = b._arm("R16_Q1_LOFO4P")
    assert name == "mtx2-loader-dryrun-lofo-raunav" and "--epochs 4 " in s and '-le 85 ]' in s
    assert f"--data-windows 3 --extra-selection '{arm['extra_selection']}'" in s and "--data-fraction" not in s
    assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0


def test_the_v2_manifest_lists_no_inert_seed(tmp_path):
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "write_run_manifest.py"),
                        "--driver", "pretrain_v2", "--run-id", "x", "--arm", "R16_Q1", "--num-classes", "17",
                        "--seed", "2", "--data-config", "configs/arms/R16_Q1.yaml",
                        "--samples-per-epoch", "10240000", "--num-epochs", "80", "--batch-size", "512",
                        "--num-workers", "5", "--data-split-num", "200", "--fetch-step", "1.0",
                        "--keep-checkpoints", "all", "--val-files", "b.parquet", "a.parquet",
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    eff = m["randomness"]["effective_streams"]
    assert set(eff) == {"trunk_init", "head_init", "data_sampling", "dropout"} and "inert" not in eff
    assert m["data_stream"]["loader"]["data_split_num"] == 200 and m["data_stream"]["loader"]["num_workers"] == 5
    assert m["data_stream"]["validation"]["files"] == ["a.parquet", "b.parquet"]
    assert "70-79" in m["checkpoints"]["robustness"] and m["driver"].endswith("pretrain_v2.py")
    assert "first maximum of val.acc" in m["checkpoints"]["primary"]


def test_the_v1_manifest_is_unchanged_by_default(tmp_path):
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "write_run_manifest.py"),
                        "--run-id", "x", "--arm", "R16_Q1", "--num-classes", "17", "--seed", "2",
                        "--data-config", "configs/arms/R16_Q1.yaml", "--samples-per-epoch", "10240000",
                        "--num-epochs", "80", "--batch-size", "512", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    assert m["randomness"]["effective_streams"]["inert"] == ["trunk_init", "data_sampling"]
    assert "driver" not in m


def test_run_directories_follow_the_fine_tuning_contract():
    assert b.v2_run_id("L162_MASS", 3) == "mtx-l162mass-s3"
    assert b.v2_run_id("RAND2_p1", 1) == "mtx-rand2p1-s1"
    assert b.v2_run_id("L188_LOFO4P", 2) == "mtx-l188lofo4p-s2"
    assert b.v2_job_name("MPM", 1) == "mtx2-mpm-s1-raunav"


def test_every_new_config_gets_a_checked_make_weight_pass():
    specs = b.v2_makeweight_specs(TAG)
    text = "".join(_script(s) for s in specs.values())
    new = b.v2_new_configs()
    assert new and all(c not in b.V1_SIDECAR_CONFIGS for c, _ in new)
    for c, k in new:
        assert text.count(f"run_cfg {c} {k}\n") == 1
    assert text.count(b.HIST_SHA256) >= 2 * len(specs)
    for spec in specs.values():
        d = yaml.safe_load(spec)["spec"]["template"]["spec"]
        assert "nvidia.com/gpu" not in d["containers"][0]["resources"]["requests"]
        assert subprocess.run(["bash", "-n"], input=_script(spec), text=True).returncode == 0
    grid_cfgs = {a["config"] for a in b.v2_arms()}
    assert grid_cfgs <= b.V1_SIDECAR_CONFIGS | {c for c, _ in new}


def test_labelled_make_weight_jobs_get_their_own_names():
    plain, labelled = b.v2_makeweight_specs(TAG), b.v2_makeweight_specs(TAG, label="-v3")
    assert sorted(labelled) == [f.replace("mtx2-makeweight-", "mtx2-makeweight-v3-") for f in sorted(plain)]
    for fn, spec in labelled.items():
        assert yaml.safe_load(spec)["metadata"]["name"] == fn[len("job-"):-len(".yaml")]


def test_a_labelled_dry_run_gets_its_own_name_and_output():
    name, spec = b.v2_dryrun_spec(TAG, False, "s170")
    assert name == "mtx2-loader-dryrun-s170-raunav"
    assert "loader_dryrun/dryrun_s170_seed1.json" in _script(spec)


def test_the_any_gpu_resume_check_pins_a_list_and_stays_inside_the_gate():
    name, spec = b.v2_det_any_spec(TAG)
    t = yaml.safe_load(spec)["spec"]["template"]["spec"]
    ex = {e["key"]: e for e in t["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"][0]["matchExpressions"]}
    assert ex["nvidia.com/gpu.product"]["values"] == list(b.ANY_GPUS)
    s = _script(spec)
    assert [m.group(1) for m in re.finditer(r"^RUN_ID=(\S+)$", s, re.M)] == ["smoke-detany-a", "smoke-detany-b", "smoke-detany-a2"]
    for line in re.findall(r"python3 experiments/MTX/pretrain_v2\.py --seed .*", s):
        assert "--deterministic" in line and "--num-epochs 3 " in line and "--samples-per-epoch 200000 " in line
    assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0


def test_the_job_restarts_from_the_epoch_the_driver_resumes(grid, tmp_path):
    """The spec's LAST (failed-attempt bookkeeping) is pretrain_v2.latest_complete_epoch:
    the newest resume file with its state file beside it."""
    s = _script(next(iter(grid.values())))
    block = s[s.index("LAST=-1"):]
    block = block[:block.index("\ndone\n") + len("\ndone\n")]
    for e, files in ((44, ("net_epoch-40_state.pt", "net_epoch-44_state.pt", "net_epoch-44_resume.pt",
                           "net_epoch-45_resume.pt")),
                     (-1, ("net_epoch-3_state.pt", "net_epoch-4_resume.pt")), (-1, ())):
        d = tmp_path / str(len(files))
        d.mkdir()
        for f in files:
            (d / f).write_text("x")
        r = subprocess.run(["bash", "-c", f"set -euo pipefail\nOUT={d}\n{block}echo $LAST"],
                           text=True, capture_output=True)
        assert r.returncode == 0 and r.stdout.strip() == str(e), (files, r.stdout, r.stderr)


def test_a_held_out_family_manifest_records_its_selection_and_window(tmp_path):
    out, sel = tmp_path / "m.json", "~((jet_label >= 15) & (jet_label < 20))"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "write_run_manifest.py"),
                        "--driver", "pretrain_v2", "--run-id", "x", "--arm", "R16_Q1_LOFO4P", "--num-classes", "17",
                        "--seed", "2", "--data-config", "configs/arms/R16_Q1.yaml",
                        "--samples-per-epoch", "10240000", "--num-epochs", "80", "--batch-size", "512",
                        "--num-workers", "5", "--data-split-num", "200", "--fetch-step", "1.0",
                        "--data-windows", "3", "--extra-selection", sel, "--keep-checkpoints", "window",
                        "--val-files", "a.parquet", "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    assert m["vocabulary"]["controlled_variable"] == "training selection (LOFO)"
    assert m["data_stream"]["selection"].endswith(f" & ({sel})") and m["data_stream"]["extra_selection"] == sel
    ld = m["data_stream"]["loader"]
    assert ld["data_windows"] == 3 and ld["data_fraction"] == 1 / 3 and "1/3 of every file" in ld["window"]
    assert m["checkpoints"]["robustness"].startswith("net_wavg70-79_state.pt") and "200,000" in m["checkpoints"]["robustness"]
    assert m["checkpoints"]["retention"] == "window"


def test_every_lofo_spec_hands_the_manifest_its_selection_and_window(grid):
    lofo = [a for a in b.v2_arms() if a.get("extra_selection")]
    assert lofo
    for arm in lofo:
        s = _script(grid[f"job-{b.v2_job_name(arm['name'], 1)}.yaml"])
        man = s[s.index("write_run_manifest.py"):s.index("--out ${MANIFEST}")]
        assert f"--extra-selection '{arm['extra_selection']}'" in man and "--data-windows 3" in man


def test_the_pre_launch_check_names_every_missing_sidecar(monkeypatch):
    import hashlib
    cfg = "configs/arms/R16_Q1.yaml"
    monkeypatch.setattr(b, "v2_arms", lambda: [{"name": "R16_Q1", "config": cfg},
                                               {"name": "R16_Q1_LOFO4P", "config": cfg}])
    name = f"R16_Q1.{hashlib.md5((ROOT / cfg).read_bytes()).hexdigest()}.auto.yaml"
    assert b.v2_expected_sidecars("HEAD") == {cfg: name}
    assert b.v2_missing_sidecars("HEAD", [name, "other.auto.yaml"]) == []
    assert b.v2_missing_sidecars("HEAD", ["R16_Q1.0000.auto.yaml"]) == [f"{cfg}: {name}"]


def test_the_storage_projection_refuses_what_does_not_fit():
    peak = b.v2_run_peak_gib()
    # the window's 14 state files, the 11 early-epoch ones (EARLY_KEEP), two resume files, records
    assert b.V2_EARLY_STATES == 11
    assert peak == (25 * b.V2_STATE_MIB + 2 * b.V2_RESUME_MIB + b.V2_RECORDS_MIB
                    + b.V2_DIAG_BATCH_MIB) / 1024 < 0.35
    n = sum(int(a["runs"]) for a in b.v2_arms())
    assert b.V2_GRID_RESERVE_GIB == n * peak
    assert b.v2_storage_problem(n, n * peak + 0.1) is None
    assert f"{n} runs" in b.v2_storage_problem(n, n * peak - 0.1)
    # every epoch kept (--keep-checkpoints all) would not fit the ~62 GB to the guard (coordinator, 2026-10-01)
    assert n * 80 * (b.V2_STATE_MIB + b.V2_RESUME_MIB) / 1024 > 62


def test_the_self_supervised_held_out_family_run_reads_a_third_of_each_file(grid):
    arm, parent = b._arm("MPM_LOFO4P"), b._arm("MPM")
    assert arm["objective"] == "mpm" and arm["config"] == parent["config"]
    assert arm["extra_selection"] == b._arm("L188_LOFO4P")["extra_selection"]
    for r in range(1, int(arm["runs"]) + 1):
        s = _script(grid[f"job-{b.v2_job_name('MPM_LOFO4P', r)}.yaml"])
        train = re.search(r"python3 experiments/MTX/pretrain_v2\.py .*", s).group(0)
        assert "--data-windows 3" in train and "--data-fraction" not in s
        assert f"--extra-selection '{arm['extra_selection']}'" in train
        assert "--mpm --mpm-mask-rate 0.40" in train and "num_classes" not in train
        assert f"--network-config {b.ARCH['mpm']}" in train and f"CFG={parent['config']}" in s
        man = s[s.index("write_run_manifest.py"):s.index("--out ${MANIFEST}")]
        assert "--num-classes 0" in man and "--data-windows 3" in man and "--mpm-mask-rate 0.40" in man


def test_the_committed_grid_is_the_generators_at_its_tag():
    files = sorted(b.ROOT.glob("experiments/MTX/k8s/v2/grid/*.yaml"))
    tags = {re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', f.read_text()).group(1) for f in files}
    assert len(tags) == 1
    want = b.v2_grid_specs(tags.pop())
    assert sorted(f.name for f in files) == sorted(want)
    for f in files:
        assert f.read_text() == want[f.name], f.name


def test_the_grid_is_refused_without_the_pre_launch_inputs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_mtx_launch.py"), "--v2", str(tmp_path),
                        "--tag", "HEAD"], capture_output=True, text=True)
    assert r.returncode != 0 and "--sidecar-listing" in r.stderr and not list(tmp_path.iterdir())


def _outside_script(spec: str) -> dict:
    d = yaml.safe_load(spec)
    d["metadata"].pop("name")
    d["spec"].pop("suspend", None)
    d["spec"]["template"]["spec"]["containers"][0].pop("args")
    return d


@pytest.mark.parametrize("run_id", ["mtx-r16q1-s4", "mtx-l162mass-s1", "mtx-mpmlofo4p-s3"])
def test_a_finalize_spec_is_its_runs_grid_spec_running_finalize_v2(grid, run_id):
    name, spec = b.v2_finalize_spec(run_id, TAG)
    g = grid[f"job-mtx2-{run_id[len('mtx-'):]}-raunav.yaml"]
    d = yaml.safe_load(spec)
    assert name == d["metadata"]["name"] == f"mtx2-finalize-{run_id[len('mtx-'):]}-raunav" and len(name) <= 63
    assert "suspend" not in d["spec"]
    # image digest, GPU product and tag, mounts, resources, node affinity, retry policy
    assert _outside_script(spec) == _outside_script(g)
    run = int(run_id.rsplit("-s", 1)[1])
    ex = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"][0]["matchExpressions"][1]
    assert ex["key"] == "nvidia.com/gpu.product" and ex["values"] == [b.V2_GPU_BY_RUN[run]]
    s, t = _script(spec), _script(g)
    assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
    # every line of the grid script is kept (run directory, data files, guards, probe,
    # sidecar) but the training, its manifest and record, and the run's own counts
    kept = set(s.splitlines())
    gone = [ln for ln in t.splitlines() if ln not in kept]
    assert all(ln.startswith(("#", "  --", "MANIFEST=", "[ -f ${MANIFEST} ]")) or any(k in ln for k in (
        "-e${LAST}", "failed attempts after epoch", "NODE_FAULTS", "${ATTEMPT} pod=", 'cp "${SIDECAR}" ${OUT}/',
        "> ${OUT}/reweight_sidecar.sha256", "write_run_manifest.py", "experiments/MTX/pretrain_v2.py"))
        for ln in gone), gone
    assert f"RUN_ID={run_id}" in s and f"OUT={b.V2_ROOT}/${{RUN_ID}}" in s
    assert s.count("PYTHONUNBUFFERED=1 python3 experiments/MTX/finalize_v2.py --out ${OUT} 2>&1 | "
                   "tee -a ${OUT}/train.log") == 1
    assert "pretrain_v2.py" not in s and "write_run_manifest" not in s
    assert s.index("[ -f ${OUT}/recipe.json ] ||") < s.index('[ "${USE}" -le "${CEIL}" ]') < s.index("mkdir -p ${OUT}")
    assert s.index("sv.sidecar_mismatch(") < s.index("sha256sum -c ${OUT}/reweight_sidecar.sha256 ||") < s.index(
        "finalize_v2.py")


def test_a_finalize_spec_is_written_only_for_a_run_of_the_grid():
    with pytest.raises(SystemExit, match="not a run of the v2 grid"):
        b.v2_finalize_spec("mtx-r16q1-s9", TAG)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_mtx_launch.py"), "--v2-finalize",
                        "mtx-r16q1-s4"], capture_output=True, text=True)
    assert r.returncode != 0 and "--tag" in r.stderr


# ------------------------------------------------ the run script under bash, with stubs
# python3 stands in for the GPU probe (GPU=ok, bad, or first: answers only the first
# call of an attempt) and the trainer or finalize_v2 (notes itself in RAN, prints
# TRAIN_SAYS, exits TRAIN_RC); df reports USE_PCT; date numbers the attempts. Files
# under /jc2 and the sidecar are made locally.
STUBS = {
    "python3": """#!/bin/bash
case "$1" in
  experiments/FT/gpu_probe.py)
    n=$(cat "$PROBES" 2>/dev/null || echo 0); echo $((n + 1)) > "$PROBES"
    [ "$GPU" = ok ] || { [ "$GPU" = first ] && [ "$n" -eq 0 ]; } ;;
  experiments/MTX/pretrain_v2.py|experiments/MTX/finalize_v2.py)
    echo "$1" >> "$RAN"; echo "$TRAIN_SAYS"; exit "$TRAIN_RC" ;;
  *) exit 0 ;;
esac
""",
    "df": '#!/bin/bash\nprintf "Use%%\\n %s%%\\n" "$USE_PCT"\n',
    "md5sum": '#!/bin/bash\necho "abc  $1"\n',
    "sha256sum": '#!/bin/bash\necho "def  $1"\n',
    "date": '#!/bin/bash\nn=$(cat "$DATES" 2>/dev/null || echo 0); echo $((n + 1)) > "$DATES"; echo "T$n"\n',
}


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    for name, text in STUBS.items():
        (d / "bin").mkdir(exist_ok=True)
        (d / "bin" / name).write_text(text)
        (d / "bin" / name).chmod(0o755)
    (d / "jc2").mkdir()
    # named by this bash's own brace expansion, as the script names them (bash 3.2 does not pad)
    names = " ".join(g.split("/jc2/jet_data/")[1] for g in b.TRAIN_GLOBS + b.VAL_GLOBS)
    subprocess.run(["bash", "-c", f"touch {names}"], cwd=d / "jc2", check=True)
    (d / "mw").mkdir()
    (d / "mw" / "R16_Q1.abc.auto.yaml").touch()
    (d / "work" / "configs" / "arms").mkdir(parents=True)
    return d


def _attempt(sim, out_root, finalize=False, **env) -> int:
    """One pod's run script for R16_Q1 run 1 (or its finalize script), with OUT under out_root."""
    s = b.v2_script(b._arm("R16_Q1"), 1, run_id="r", out_root=str(out_root), finalize=finalize)
    s = s.replace("/jc2/jet_data/", f"{sim}/jc2/").replace("/data/results/mtx/makeweight/", f"{sim}/mw/")
    (out_root / "probes").unlink(missing_ok=True)
    e = {"PATH": f"{sim}/bin{os.pathsep}{os.environ['PATH']}", "PROBES": str(out_root / "probes"),
         "DATES": str(out_root / "dates"), "RAN": str(out_root / "ran"),
         "GPU": "ok", "TRAIN_RC": "0", "TRAIN_SAYS": "trained",
         "USE_PCT": "50", "POD_NAME": "pod", "NODE_NAME": "node", "GPU_PRODUCT": "gpu", "REPO_REF": TAG}
    out_root.mkdir(exist_ok=True)
    return subprocess.run(["bash", "-c", s], cwd=sim / "work", env={**e, **env},
                          capture_output=True, text=True).returncode


def _probed(out_root) -> bool:
    return (out_root / "probes").exists()


def _resumable(out, epoch):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"net_epoch-{epoch}_resume.pt").touch()
    (out / f"net_epoch-{epoch}_state.pt").touch()


def test_a_run_that_trains_leaves_no_marker(sim, tmp_path):
    assert _attempt(sim, tmp_path) == 0
    out = tmp_path / "r"
    assert os.listdir(out / "attempts") == [] and not (out / "NODE_FAULTS").exists()
    assert (out / "train.log").read_text() == "trained\n" and len((out / "attempts.log").read_text().splitlines()) == 1


def test_node_faults_leave_no_failed_attempt_and_the_sixth_halts(sim, tmp_path):
    out = tmp_path / "r"
    assert _attempt(sim, tmp_path, GPU="bad") == 43                         # refused before any work
    assert not (out / "train.log").exists() and "NODE_FAULT preflight pod=pod node=node" in (
        out / "NODE_FAULTS").read_text()
    assert "NODE_FAULT preflight" in (out / "attempts.log").read_text()
    assert _attempt(sim, tmp_path, GPU="first", TRAIN_RC="1", TRAIN_SAYS="Traceback") == 43   # GPU gone after
    for _ in range(4):                                                      # a CUDA fault in this attempt's log
        assert _attempt(sim, tmp_path, TRAIN_RC="1", TRAIN_SAYS="RuntimeError: CUDA error: unknown error") == 43
    assert os.listdir(out / "attempts") == [] and len((out / "NODE_FAULTS").read_text().splitlines()) == 6
    assert _attempt(sim, tmp_path) == 42 and not _probed(tmp_path)


def test_a_driver_without_a_gpu_is_a_node_fault_whatever_the_probe_says(sim, tmp_path):
    """pretrain_v2 exits 4 (EXIT_NO_GPU) when --device cuda has no usable GPU: the
    node's fault, even when the GPU answers the probe again after it."""
    out = tmp_path / "r"
    _resumable(out, 7)
    say = "FATAL: --device cuda and no usable GPU (torch.cuda.is_available() is False); nothing written"
    for n in (1, 2):
        assert _attempt(sim, tmp_path, TRAIN_RC="4", TRAIN_SAYS=say) == 43
        assert (tmp_path / "probes").read_text().strip() == "1"            # the preflight probe only
        assert len((out / "NODE_FAULTS").read_text().splitlines()) == n
    assert "NODE_FAULT rc=4 pod=pod node=node after_epoch=7" in (out / "NODE_FAULTS").read_text()
    assert os.listdir(out / "attempts") == []                               # two would have halted the run
    assert _attempt(sim, tmp_path) == 0


def test_an_ordinary_failure_is_counted_and_two_at_one_epoch_halt(sim, tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "train.log").write_text("RuntimeError: CUDA error: unknown error\n")   # an earlier attempt's
    assert _attempt(sim, tmp_path, TRAIN_RC="1", TRAIN_SAYS="ValueError: bad input") == 1
    assert _attempt(sim, tmp_path, TRAIN_RC="1", TRAIN_SAYS="ValueError: bad input") == 1
    assert sorted(os.listdir(out / "attempts")) == ["failed-T0-e-1", "failed-T1-e-1"]
    assert not (out / "NODE_FAULTS").exists()
    n = len((out / "attempts.log").read_text().splitlines())
    assert _attempt(sim, tmp_path) == 42 and not _probed(tmp_path)
    assert len((out / "attempts.log").read_text().splitlines()) == n


def test_a_non_finite_loss_is_a_failed_attempt_wherever_it_ran(sim, tmp_path):
    """pretrain_v2 exits 3 on a non-finite loss: counted at the epoch it resumed from even
    when the GPU stops answering after it, so a second one at that epoch halts the run."""
    out = tmp_path / "r"
    _resumable(out, 4)
    nan = "FATAL: non-finite training loss at epoch 5 / CUDA error: unknown error"
    assert _attempt(sim, tmp_path, GPU="first", TRAIN_RC="3", TRAIN_SAYS=nan) == 3
    assert _attempt(sim, tmp_path, TRAIN_RC="3", TRAIN_SAYS=nan) == 3
    assert sorted(os.listdir(out / "attempts")) == ["failed-T0-e4", "failed-T1-e4"]
    assert not (out / "NODE_FAULTS").exists()
    assert _attempt(sim, tmp_path) == 42
    _resumable(out, 5)                                                      # an epoch completed: counting restarts
    assert _attempt(sim, tmp_path) == 0


def test_a_resume_may_go_further_than_a_fresh_start(sim, tmp_path):
    fresh, resume = tmp_path / "fresh", tmp_path / "resume"
    fresh.mkdir()
    assert _attempt(sim, fresh, USE_PCT="86") == 42 and not (fresh / "r").exists()     # nothing written
    assert _attempt(sim, fresh, USE_PCT="85") == 0
    resume.mkdir()
    _resumable(resume / "r", 9)
    assert _attempt(sim, resume, USE_PCT="96") == 42 and not (resume / "r" / "attempts").exists()
    assert _attempt(sim, resume, USE_PCT="95") == 0


def test_finalize_runs_where_the_runs_own_failures_stop_it(sim, tmp_path):
    """The case finalize_v2 is for: every epoch finished and the end step failed twice, so
    the training script halts the run. The finalize script still runs, refuses a directory
    that holds no run, and counts its failed attempts and node faults apart from the run's."""
    out = tmp_path / "r"
    _resumable(out, 79)
    (out / "attempts").mkdir()
    for t in ("T7", "T8"):
        (out / "attempts" / f"failed-{t}-e79").write_text("rc=1\n")
    (out / "NODE_FAULTS").write_text("earlier\n" * 5)
    assert _attempt(sim, tmp_path) == 42 and not (tmp_path / "ran").exists()             # the training script
    assert _attempt(sim, tmp_path, finalize=True) == 42 and not _probed(tmp_path)    # no recipe.json
    (out / "recipe.json").write_text("{}\n")
    assert _attempt(sim, tmp_path, finalize=True, GPU="bad") == 43                  # a node fault of its own
    assert _attempt(sim, tmp_path, finalize=True, TRAIN_RC="4") == 43               # a sixth if shared
    assert (out / "NODE_FAULTS").read_text() == "earlier\n" * 5
    assert len((out / "FINALIZE_NODE_FAULTS").read_text().splitlines()) == 2
    assert _attempt(sim, tmp_path, finalize=True, TRAIN_RC="1", TRAIN_SAYS="ValueError: bad") == 1
    assert sorted(os.listdir(out / "attempts")) == ["failed-T2-finalize", "failed-T7-e79", "failed-T8-e79"]
    assert _attempt(sim, tmp_path, finalize=True) == 0
    assert (tmp_path / "ran").read_text() == "experiments/MTX/finalize_v2.py\n" * 3
    assert " finalize pod=pod " in (out / "attempts.log").read_text() and not list(out.glob("run_manifest*"))
    _attempt(sim, tmp_path, finalize=True, TRAIN_RC="1")
    assert _attempt(sim, tmp_path, finalize=True) == 42                             # two failures of its own halt
