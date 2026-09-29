"""The v2 fine-tuning specs (scripts/build_ft_jobs.py, audit 2026-09-29).

The staging job: CPU, the held-out pool only, the space guard, the retry policy,
and a sha256 record that covers every subset any v2 job verifies.
"""
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "experiments" / "FT" / "k8s"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_jobs_v2specs", "scripts/build_ft_jobs.py")
STAGE = "job-ft-subsets-jc2-v2-raunav.yaml"


@pytest.fixture(scope="module")
def stage():
    return B.build(B.PIN_V2, v2_subsets=True)[STAGE]


def _args(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_the_staging_spec_on_disk_is_the_generators(stage):
    assert (K8S / STAGE).read_text() == stage


def test_the_staging_job_is_a_cpu_job_of_mine_with_the_retry_policy(stage):
    d = yaml.safe_load(stage)
    assert d["metadata"]["name"] == "ft-subsets-jc2-v2-raunav"
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in c["resources"]["limits"]
    assert d["spec"]["backoffLimit"] == B.ROBUST_BACKOFF
    rules = d["spec"]["podFailurePolicy"]["rules"]
    assert rules[0]["action"] == "FailJob" and rules[0]["onExitCodes"]["values"] == [B.EXIT_HALT]
    assert rules[1]["action"] == "Ignore"
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["REPO_REF"] == B.PIN_V2
    assert "us-west" in stage


def test_the_staging_job_builds_one_draw_from_the_pool_and_never_rebuilds(stage):
    a = _args(stage)
    assert f"POOL=({B.V2_POOL})" in a
    call = re.search(r"make_subsets.py jc2v2 (.*?)\|\| exit", a, re.S).group(1)
    assert "--seeds 1 " in call and f"--val-size {B.V2_VAL_JETS}" in call
    assert "--sizes " + " ".join(map(str, B.SIZES)) in call
    assert B.V2_VAL_JETS % 512 == 0
    assert "[ -f ${OUT}/DONE ] && {" in a                       # a complete build is never redone
    assert '[ "$p" -lt 85 ] && [ "$g" -ge 100 ]' in a          # CLAUDE.md: nothing > 1 GB past 85%
    assert a.index("mv ${OUT}.staging ${OUT}") < a.index("ft_v2.py hash")


def test_the_sha256_record_covers_every_reused_file_a_v2_job_reads(stage):
    hashed = set(re.search(r"ft_v2.py hash .*?--files (.*?)\|\| exit", _args(stage), re.S)
                 .group(1).replace("\\", " ").split())
    want = set(B.HERWIG_TEST) | {"/data/finetune/jc1/val.parquet", "/data/finetune/top_sub/val.parquet",
                                 "/data/finetune/qg_v2_sub/val.parquet"}
    want |= {f"/data/finetune/jc1/train_N{n}_s1.parquet" for n in B.SIZES}
    want |= {f"/data/finetune/{d}/train_N{n}_s1.parquet"
             for d, b in (("top_sub", "top"), ("qg_v2_sub", "qg")) for n in B.BENCH_SIZES[b]}
    assert want <= hashed
    assert {"${OUT}/train_N*_s1.parquet", "${OUT}/val.parquet"} <= hashed

