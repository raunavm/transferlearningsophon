"""scripts/build_aoj_jobs.py: the full real-data run as committed must be the run
the builder describes, cover every file once, score every model everywhere,
stay off the 3090 pool, and never write or fit the withdrawn two-prong channel."""
import importlib.util
import json
import pathlib
import re
import subprocess

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("scripts/build_aoj_jobs.py", "build_aoj_jobs")
FT = _load("scripts/build_ft_jobs.py", "build_ft_jobs")
SPECS = B.specs()
SHARDS = [p for p in SPECS if "-fit-" not in p.name]
FIT = next(p for p in SPECS if "-fit-" in p.name)


def _script(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


@pytest.mark.parametrize("path", sorted(SPECS), ids=lambda p: p.name)
def test_committed_spec_is_exactly_what_the_builder_writes(path):
    assert path.read_text() == SPECS[path], f"{path.name} was edited by hand; edit the builder"


@pytest.mark.parametrize("path", sorted(SPECS), ids=lambda p: p.name)
def test_every_spec_is_valid_yaml_and_valid_bash(path):
    doc = yaml.safe_load(SPECS[path])
    assert doc["metadata"]["name"].endswith("-raunav")
    r = subprocess.run(["bash", "-n"], input=_script(SPECS[path]), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_shards_cover_all_eighty_files_once_with_the_listed_checksums():
    listed = {f["key"]: f["md5"] for f in json.loads(B.FILES.read_text())["files"]}
    seen = {}
    for p in SHARDS:
        for key, md5 in re.findall(r"(Run[GH]_batch\d+\.h5) ([0-9a-f]{32})", SPECS[p]):
            assert key not in seen, f"{key} in {seen.get(key)} and {p.name}"
            seen[key] = p.name
            assert md5 == listed[key]
    assert set(seen) == set(listed) and len(seen) == 80


def test_the_feasibility_files_keep_their_checksums():
    feas = (REPO / "experiments/AOJ/k8s/job-aoj-feasibility-v2cpu-raunav.yaml").read_text()
    listed = {f["key"]: f["md5"] for f in json.loads(B.FILES.read_text())["files"]}
    pairs = re.findall(r"(Run[GH]_batch\d+\.h5) ([0-9a-f]{32})", feas)
    assert len(pairs) == 8 and all(listed[k] == m for k, m in pairs)


def test_every_shard_scores_every_model_with_its_own_head_width():
    B.verify_heads()
    names = [m.name for m in B.MODELS]
    assert len(names) == len(set(names)) == 31
    for p in SHARDS:
        lines = re.findall(r"^\s+score (\S+) \"(\S+)\" (\d+) (\d) (\S+) (\S+)$", SPECS[p], re.M)
        assert [l[0] for l in lines] == names
        for (name, ckpt, k, reg, arm, rung), m in zip(lines, B.MODELS):
            assert (int(k), int(reg), rung) == (m.k, m.num_reg, m.rung)
            assert ckpt.endswith("net_epoch-79_state.pt") or name == "sophon-public"


def test_the_mass_output_models_carry_their_regression_column():
    mass = [m for m in B.MODELS if "mass" in m.name]
    assert len(mass) == 10 and all(m.num_reg == 1 and m.arm.endswith("_MASS") for m in mass)
    assert all(m.num_reg == 0 for m in B.MODELS if "mass" not in m.name)


def test_the_withdrawn_two_prong_channel_is_neither_written_nor_fitted():
    for p in SHARDS:
        assert "--structures three_prong" in SPECS[p]
    assert "--peaks top" in SPECS[FIT]


def test_shards_stay_off_the_3090_pool_and_off_every_bad_node():
    for p in SHARDS:
        terms = yaml.safe_load(SPECS[p])["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        by_key = {t["key"]: t for t in terms}
        gpu = by_key["nvidia.com/gpu.product"]
        assert gpu["operator"] == "NotIn" and gpu["values"] == ["NVIDIA-GeForce-RTX-3090"]
        hosts = by_key["kubernetes.io/hostname"]
        assert hosts["operator"] == "NotIn"
        assert set(hosts["values"]) >= set(FT.BAD_NODES) | set(FT.LOST_GPU_NODES)


def test_the_fit_refuses_until_every_shard_is_done_and_never_overwrites():
    s = _script(SPECS[FIT])
    assert f"seq 0 {B.N_SHARDS - 1}" in s and "shard${i}/DONE" in s
    assert "results.json exists" in s


def test_a_resumed_shard_skips_scored_models_and_rewrites_nothing():
    s = _script(SPECS[SHARDS[0]])
    assert 'skip $1 (scored)' in s
    assert '[ -f "${OUT}/closure.json" ] || ' in s and "cp -n" in s
    assert '>> "${OUT}/gpu_per_model.txt"' in s, "a resumed shard may change GPU; record it per model"


def test_the_pin_carries_every_flag_the_run_passes():
    B.verify_pin(B.PIN, not_yet_tagged=True)
    for path, flag in B.NEEDED_FLAGS.items():
        assert flag in (REPO / path).read_text()
