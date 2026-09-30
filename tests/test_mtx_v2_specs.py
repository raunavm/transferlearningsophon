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
    for spec in grid.values():
        d = yaml.safe_load(spec)["spec"]
        assert d["backoffLimit"] == b.V2_BACKOFF
        rules = d["podFailurePolicy"]["rules"]
        assert rules[0] == {"action": "FailJob", "onExitCodes": {
            "containerName": "main", "operator": "In", "values": [42]}}
        assert rules[1] == {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}
        c = d["template"]["spec"]["containers"][0]
        assert c["name"] == "main" and d["template"]["spec"]["restartPolicy"] == "Never"
        s = _script(spec)
        assert "trap 'rc=$?" in s and "attempts/failed-" in s and "exit ${HALT}" in s


def test_v2_code_loader_validation_and_output(grid):
    for fn, spec in grid.items():
        s = _script(spec)
        run_id = re.search(r"^RUN_ID=(\S+)$", s, re.M).group(1)
        assert f"OUT={b.V2_ROOT}/${{RUN_ID}}" in s and fn == "job-mtx2-" + run_id[len("mtx-"):] + "-raunav.yaml"
        assert re.fullmatch(r"mtx-[a-z0-9]+-s\d", run_id)
        assert "python3 experiments/MTX/pretrain_v2.py" in s and "seed_weaver" not in s
        assert "--num-workers 5 --fetch-step 1.0 --data-split-num 200 --data-fraction 0.2" in s
        assert "--data-fraction 0.2 --keep-checkpoints all" in s             # the manifest records it
        assert "--fetch-by-files" not in s and "--samples-per-epoch 10240000" in s
        assert "--num-epochs 80" in s and "--start-lr 5e-4" in s and "--use-amp" in s
        val = re.search(r"--data-val (.*?) --data-config", s).group(1).split()
        assert [_brace_count(v) for v in val] == [4, 16, 5] and sum(map(_brace_count, val)) == b.N_VAL_FILES
        assert "Res2P_{0200..0203}" in val[0] and "Res34P_{0860..0875}" in val[1] and "QCD_{0280..0284}" in val[2]
        tr = re.search(r"--data-train (.*?) --data-val", s).group(1).split()
        assert sum(map(_brace_count, tr)) == b.N_TRAIN_FILES
        assert "--driver pretrain_v2" in s and "--keep-checkpoints all" in s
        assert "--keep-checkpoints all --select-on head_top1_acc" in s and "--select-on head_top1_acc \\" in s
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
    for spec in grid.values():
        t = yaml.safe_load(spec)["spec"]["template"]["spec"]
        terms = t["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]
        ex = {e["key"]: e for e in terms["matchExpressions"]}
        assert ex["topology.kubernetes.io/region"]["values"] == ["us-west"]
        assert ex["nvidia.com/gpu.product"]["values"] == [b.V2_GPU]
        assert "hcc-chase-shor-c4715.unl.edu" in ex["kubernetes.io/hostname"]["values"]
        env = {e["name"]: e.get("value") for e in t["containers"][0]["env"]}
        assert env["GPU_PRODUCT"] == b.V2_GPU and env["REPO_REF"] == TAG
        res = t["containers"][0]["resources"]
        assert res["requests"] == res["limits"] and res["requests"]["nvidia.com/gpu"] == "1"


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


def test_dry_run_specs_request_no_gpu():
    for full in (False, True):
        name, spec = b.v2_dryrun_spec(TAG, full)
        t = yaml.safe_load(spec)["spec"]["template"]["spec"]
        assert "nvidia.com/gpu" not in t["containers"][0]["resources"]["requests"]
        s = _script(spec)
        assert "loader_dryrun.py" in s and ("--full-columns" in s) == full and "raunav" in name


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
    assert "val.head_top1_acc" in m["checkpoints"]["primary"]


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
