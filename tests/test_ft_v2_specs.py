"""The v2 fine-tuning specs (scripts/build_ft_jobs.py, audit 2026-09-29).

The staging job: CPU, the held-out pool only, the space guard, the retry policy,
and a sha256 record that covers every subset any v2 job verifies.
"""
import importlib.util
import pathlib
import re
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


B = _load("build_ft_jobs_v2specs", "scripts/build_ft_jobs.py")
STAGE = "job-ft-subsets-jc2-v2-raunav.yaml"


@pytest.fixture(scope="module")
def stage():
    return B.build(B.PIN_V2_SUBSETS, v2_subsets=True)[STAGE]


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
    assert env["REPO_REF"] == B.PIN_V2_SUBSETS
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



def test_the_staging_spec_on_disk_is_the_job_as_it_ran():
    assert B.V2_SUBSETS_RAN and "--as-dir" not in _args((K8S / STAGE).read_text())


def _stage_run(tmp_path, n, **env_extra):
    """The fixed staging script under bash: make_subsets writes a build with its
    DONE, ft_v2.py hash writes its --out (or SIGKILLs the job, as an eviction does)."""
    T = _load("wave3_harness_stage", "tests/test_wave3_specs.py")
    stubs = tmp_path / f"pod{n}"
    env, _ = T._shell_env(stubs, [])
    py = stubs / "bin" / "python3"
    old = "case \"$*\" in\n"
    py.write_text(py.read_text().replace(old, old + (
        '  *make_subsets.py*)\n'
        '    O=$(next_after --out "$@"); mkdir -p "$O"\n'
        '    for f in train_N1000_s1 train_N10000_s1 train_N100000_s1 train_N1000000_s1 val; do echo x > "$O/$f.parquet"; done\n'
        '    echo "{}" > "$O/manifest.json"; touch "$O/DONE";;\n'
        '  *ft_v2.py\\ hash*)\n'
        '    [ -n "${KILL_HASH:-}" ] && { kill -9 $PPID; exit 137; }\n'
        '    O=$(next_after --out "$@"); echo "{\\"files\\": {}}" > "$O";;\n'), 1))
    (tmp_path / "jc2/jet_data").mkdir(parents=True, exist_ok=True)
    pool = subprocess.run(["bash", "-c", "echo " + B.V2_POOL.replace("/jc2/jet_data", str(tmp_path / "jc2/jet_data"))],
                          capture_output=True, text=True).stdout.split()   # as this bash expands it
    assert len(pool) == 310
    for f in pool:
        pathlib.Path(f).touch()
    env.update(env_extra)
    text = B._fill(B._SUBSETS_JC2_V2_FIXED, "test")
    script = (text.replace("/workspace", str(stubs / "workspace")).replace("/data", str(tmp_path / "data"))
              .replace("/jc2/jet_data", str(tmp_path / "jc2/jet_data")))
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=300)


def test_an_eviction_during_the_hash_no_longer_strands_the_staged_subsets(tmp_path):
    out = tmp_path / "data/finetune/jc2_v2"
    r = _stage_run(tmp_path, 1, KILL_HASH="1")
    assert r.returncode == -9 and not out.exists()          # nothing in place yet
    r = _stage_run(tmp_path, 2)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "DONE").exists() and (out / "ft_v2_subsets_sha256.json").exists()
    r = _stage_run(tmp_path, 3)                              # complete: never rebuilt
    assert r.returncode == B.EXIT_HALT and "never rebuilt" in r.stdout


def test_a_build_in_place_without_its_record_is_only_hashed(tmp_path):
    out = tmp_path / "data/finetune/jc2_v2"
    out.mkdir(parents=True)
    (out / "DONE").touch()
    (out / "manifest.json").write_text("{}")
    r = _stage_run(tmp_path, 1)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "ft_v2_subsets_sha256.json").exists() and not (tmp_path / "data/finetune/jc2_v2.staging").exists()
