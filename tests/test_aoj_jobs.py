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
SHARDS = [p for p in SPECS if "-full-s" in p.name]
FIT = next(p for p in SPECS if p.name == "job-aoj-full-fit-raunav.yaml")
CHECK = next(p for p in SPECS if "-fitcheck-" in p.name)


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
        lines = re.findall(r"^\s+scored (\S+) \"(\S+)\" (\d+) (\d) (\S+) (\S+)(?: & p\d=\$!)?$",
                           SPECS[p], re.M)
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
        gpu = [t for t in terms if t["key"] == "nvidia.com/gpu.product"]
        assert {g["operator"] for g in gpu} == {"Exists", "NotIn"}, "an unlabelled node must not match"
        assert next(g for g in gpu if g["operator"] == "NotIn")["values"] == ["NVIDIA-GeForce-RTX-3090"]
        by_key = {t["key"]: t for t in terms}
        hosts = by_key["kubernetes.io/hostname"]
        assert hosts["operator"] == "NotIn"
        assert set(hosts["values"]) >= set(FT.BAD_NODES) | set(FT.LOST_GPU_NODES) | {
            "ry-gpu-10.sdsc.optiputer.net"}


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


def _run_scoring_block(tmp_path, fail=None):
    """Run the shard's real scoring block under bash with a fake python3 that
    succeeds for every model except `fail`."""
    s = _script(SPECS[SHARDS[0]])
    block = s[s.index("# Every step is chained"):s.rindex('touch "${OUT}/DONE"')]
    block = block.replace("/scratch/", f"{tmp_path}/scratch/")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "python3"
    fake.write_text(f"""#!/bin/bash
args="$*"
if [[ "$args" == *extract_features.py* ]]; then
  [[ -n "{fail or ''}" && "$args" == *"/scratch/extract/{fail or 'NONE'} "* ]] && {{ echo boom; exit 3; }}
  echo "1,000 jets  900 jets/s"; exit 0
fi
if [[ "$args" == *discriminants.py* ]]; then
  name=$(sed -E 's/.*--name ([^ ]+).*/\\1/' <<< "$args"); touch "$OUT/scores_$name.npz"; exit 0
fi
exit 9
""")
    fake.chmod(0o755)
    harness = (f"set -euo pipefail\nexport OUT={tmp_path}/out\nFILES=f\nCFG=c\nGPU=fake\n"
               f"mkdir -p $OUT\n{block}\ntouch \"$OUT/DONE\"\n")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, env=env)


def test_the_parallel_scoring_block_scores_every_model_and_finishes(tmp_path):
    r = _run_scoring_block(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    got = {p.stem.removeprefix("scores_") for p in (tmp_path / "out").glob("scores_*.npz")}
    assert got == {m.name for m in B.MODELS} and (tmp_path / "out" / "DONE").exists()


def test_one_failed_model_inside_a_parallel_group_stops_the_shard_without_done(tmp_path):
    r = _run_scoring_block(tmp_path, fail="l188-s2")
    assert r.returncode != 0 and "FATAL: l188-s2 failed" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "out" / "DONE").exists()
    assert not (tmp_path / "out" / "scores_l188-s2.npz").exists()


def test_the_fit_clones_the_tag_with_the_corrected_merge_and_the_shards_keep_theirs():
    """The shards ran at PIN and their specs are the record of it; only the fit moves."""
    assert f'--branch "{B.FIT_PIN}"' in SPECS[FIT] and f'--branch "{B.PIN}"' not in SPECS[FIT]
    assert all(f'--branch "{B.PIN}"' in SPECS[p] for p in SHARDS)
    B.verify_pin(B.FIT_PIN, not_yet_tagged=True, flags=B.FIT_NEEDED_FLAGS)


def test_the_fit_check_merges_exactly_as_the_fit_did_and_writes_only_its_report():
    """Everything up to the merged jets is the fit's own job, line for line; after
    it the check runs instead of the fit and cannot write where the fit wrote."""
    fit, check = SPECS[FIT], SPECS[CHECK]
    merge = "python3 experiments/AOJ/merge_shards.py --shards ${SHARDS} --out /scratch/merged"
    assert merge in fit and merge in check
    assert "peak_fit.py" not in check and "fit_convergence_check.py" in check
    assert f'--branch "{B.CHECK_PIN}"' in check
    assert f"OUT={B.OUT_ROOT}/fit_convergence_check" in check and f"OUT={B.OUT_ROOT}/fit\n" not in check
    assert yaml.safe_load(check)["spec"]["backoffLimit"] == 1


def test_the_v2_fit_is_the_first_fit_with_only_the_minimiser_pin_output_and_check_changed():
    one, two = B.render_fit(), B.render_fit_v2()
    assert "name: aoj-full-fit-v2-raunav" in two and f'--branch "{B.FIT2_PIN}"' in two
    assert f"OUT={B.OUT_ROOT}/fit_v2\n" in two and f"OUT={B.OUT_ROOT}/fit\n" not in two
    assert '--results "${OUT}/results.json"' in two and '--out "${OUT}/check.json"' in two
    # everything from the merge through the fit command is the first run's, line for line
    body = lambda t: t[t.index("python3 experiments/AOJ/merge_shards.py"):t.index('ls -la "${OUT}"')]
    assert body(one) == body(two)


def test_the_bins_export_merges_exactly_as_the_fit_did_and_fits_nothing():
    fit, bins = SPECS[FIT], B.render_fit_bins()
    merge = "python3 experiments/AOJ/merge_shards.py --shards ${SHARDS} --out /scratch/merged"
    assert merge in fit and merge in bins
    assert "peak_fit.py" not in bins and "export_fit_bins.py" in bins
    assert f'--branch "{B.BINS_PIN}"' in bins and "name: aoj-full-fitbins-raunav" in bins
    assert f"--results {B.OUT_ROOT}/fit_v2/results.json" in bins
    assert f"--histograms {B.OUT_ROOT}/fit_v2/histograms.npz" in bins
    assert f"OUT={B.OUT_ROOT}/fit_v2_bins\n" in bins and yaml.safe_load(bins)["spec"]["backoffLimit"] == 1


def test_the_v3_fit_is_the_first_fit_with_its_checks_appended():
    one, three = B.render_fit(), B.render_fit_v3()
    assert "name: aoj-full-fit-v3-raunav" in three and f'--branch "{B.FIT3_PIN}"' in three
    assert f"OUT={B.OUT_ROOT}/fit_v3\n" in three and f"OUT={B.OUT_ROOT}/fit\n" not in three
    body = lambda t: t[t.index("python3 experiments/AOJ/merge_shards.py"):t.index('ls -la "${OUT}"')]
    assert body(one) == body(three)
    after = three[three.index('ls -la "${OUT}"'):]
    order = [after.index(s) for s in ("fit_convergence_check.py", "export_fit_bins.py", "fit_minimum_diagnostic.py")]
    assert order == sorted(order) and '--histograms "${OUT}/histograms.npz"' in after


# ---- the checks of the real-data section (audit B5, must-fix 9) ----
RESCORE = sorted(p for p in SPECS if "-rescore-s" in p.name and "-p" not in p.name.removesuffix("-raunav.yaml")[-3:])
PARTS = sorted(p for p in SPECS if re.search(r"-rescore-s\d+-p\d+-raunav", p.name))
SIM = sorted(p for p in SPECS if "-sim-g" in p.name)
CHECKS = [p for p in SPECS if p.name == "job-aoj-checks-v1-raunav.yaml"]


def test_each_rescore_shard_is_its_first_run_shard_with_only_the_listed_changes():
    """Staging, row alignment and scoring must be the first run's line for line, so the
    three-prong score can be checked bit for bit against it."""
    assert len(RESCORE) == B.N_SHARDS
    for i, fs in enumerate(B.shards()):
        one, two = B.render_shard(i, fs), B.render_rescore_shard(i, fs)
        s1, s2 = _script(one), _script(two)
        if i in B.RESCORE_RECREATED:
            one = one.replace("values: [" + ", ".join(f'"{b}"' for b in B.BAD_NODES) + "]",
                              "values: [" + ", ".join(f'"{b}"' for b in B.SIM_BAD_NODES) + "]")
            gpu_mem = "              - key: nvidia.com/gpu.memory\n                operator: Gt\n"
            one = one.replace("                operator: Exists\n",
                              "                operator: Exists\n" + gpu_mem
                              + f'                values: ["{B.MIN_GPU_MEMORY_MIB}"]\n', 1)
            assert "patternlab.calit2.optiputer.net" in two
            terms = yaml.safe_load(two)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
            assert {"key": "nvidia.com/gpu.memory", "operator": "Gt", "values": ["10000"]} in terms
        else:
            assert "patternlab" not in two and "gpu.memory" not in two, \
                "a running shard's spec must stay the one it was launched with"
        s1, s2 = _script(one), _script(two)
        assert yaml.safe_load(one)["spec"]["template"]["spec"]["affinity"] == \
            yaml.safe_load(two)["spec"]["template"]["spec"]["affinity"]
        assert s2.replace(f"OUT={B.RESCORE_ROOT}/shard{i}", f"OUT={B.OUT_ROOT}/shard{i}", 1) \
                 .replace(f'--branch "{B.RESCORE_PIN}"', f'--branch "{B.PIN}"') \
                 .replace("--structures three_prong prong_only", "--structures three_prong") \
                 .replace("".join(ln[10:] + "\n" for ln in B._failure_accounting("${OUT}", FT.EXIT_HALT)
                              .splitlines()), "") == s1
        doc = yaml.safe_load(two)
        assert doc["metadata"]["name"] == f"aoj-rescore-s{i}-raunav"
        assert "two_prong" not in s2, "the withdrawn two-prong channel stays unwritten"


@pytest.mark.parametrize("path", RESCORE + PARTS + SIM + CHECKS, ids=lambda p: p.name)
def test_the_check_jobs_carry_the_retry_policy_of_the_fine_tuning_jobs(path):
    doc = yaml.safe_load(SPECS[path])
    spec = doc["spec"]
    assert spec["backoffLimit"] == FT.ROBUST_BACKOFF
    rules = spec["podFailurePolicy"]["rules"]
    assert rules[0] == {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In",
                                                              "values": [FT.EXIT_HALT]}}
    assert rules[1]["action"] == "Ignore" and rules[1]["onPodConditions"] == [{"type": "DisruptionTarget"}]
    assert [c["name"] for c in spec["template"]["spec"]["containers"]] == ["main"]
    terms = spec["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    if path in CHECKS:
        assert "nvidia.com/gpu" not in SPECS[path] and f'--branch "{B.CHECKS_PIN}"' in SPECS[path]
        return
    gpu = [t for t in terms if t["key"] == "nvidia.com/gpu.product"]
    assert {g["operator"] for g in gpu} == {"Exists", "NotIn"}
    assert f'--branch "{B.RESCORE_PIN}"' in SPECS[path]


def _accounting_run(tmp_path, rc):
    """One attempt of the failure accounting, ending with exit code rc."""
    script = "set -euo pipefail\nACC=" + str(tmp_path) + "\n" + B._failure_accounting("${ACC}", 42) + f"exit {rc}\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode


def test_two_failed_attempts_halt_with_the_failjob_code_and_evictions_do_not_count(tmp_path):
    assert _accounting_run(tmp_path, 1) == 1
    assert _accounting_run(tmp_path, 0) == 0          # a successful attempt is not a failure
    assert _accounting_run(tmp_path, 1) == 1
    assert _accounting_run(tmp_path, 0) == 42, "two failed attempts: the next one halts at once"
    assert (tmp_path / "FAILED_ATTEMPTS").read_text().count("rc=1") == 2


def test_the_sim_jobs_score_every_model_once_on_one_file_list_of_test_files_only():
    names = [re.findall(r'^\s+scored (\S+) "', SPECS[p], re.M) for p in SIM]
    flat = [n for g in names for n in g]
    assert sorted(flat) == sorted(m.name for m in B.MODELS) and len(flat) == len(set(flat))
    lists = {re.search(r'FILES="([^"]+)"', SPECS[p]).group(1) for p in SIM}
    assert len(lists) == 1, "every model must read the same files in the same order"
    files = lists.pop().split()
    fam = {"Res34P": (1075, 1289), "QCD": (350, 419)}
    for f in files:
        m = re.fullmatch(r"/jc2/jet_data/(Res34P|QCD)_(\d{4})\.parquet", f)
        assert m and fam[m[1]][0] <= int(m[2]) <= fam[m[1]][1], f"{f} is not a test file"
    assert sum("QCD_" in f for f in files) == 70
    for p in SIM:
        s = _script(SPECS[p])
        assert "--num-workers 1" in s and f"--data-config {B.SIM_CONFIG}" in s
        assert "sim_scores.py" in s and "rm -rf \"/scratch/extract/$1\"" in s


@pytest.mark.parametrize("path", sorted(SPECS), ids=lambda p: p.name)
def test_every_job_reading_parquet_through_weaver_installs_pyarrow_first(path):
    """The image has no pyarrow; weaver swallows the ImportError and dies later with
    "Zero entries loaded" (the first simulation launch, 2026-09-29)."""
    s = _script(SPECS[path])
    # realdata_checks.py imports scripts/stage_aoj.py, whose imports need pyarrow
    uses = [s.find(k) for k in ("extract_features.py", "closure.py", "stage_aoj.py", "realdata_checks.py") if k in s]
    if not uses:
        return
    install = s.find("pip install --no-cache-dir -q pyarrow")
    assert 0 <= install < min(uses), f"{path.name} reads parquet before installing pyarrow"


def test_the_checks_job_waits_for_every_input_and_reads_the_rescore_against_the_first_run():
    s = _script(SPECS[CHECKS[0]])
    assert f"seq 0 {B.N_SHARDS - 1}" in s and f"{B.RESCORE_ROOT}/shard${{i}}/DONE" in s
    assert f"seq 0 {B.N_SIM_JOBS - 1}" in s and f"{B.SIM_ROOT}/attempts/g${{g}}/DONE" in s
    assert s.count(f"exit {FT.EXIT_HALT}; }}") == 2, "a missing input halts; it is not retried"
    merge = "python3 experiments/AOJ/merge_shards.py --shards ${SHARDS} --out /scratch/merged"
    assert merge in s and s.index(merge) < s.index("realdata_checks.py")
    assert "--first-run-shards ${FIRST}" in s and f"{B.OUT_ROOT}/shard${{i}}" in s
    first = "python3 experiments/AOJ/merge_shards.py --shards ${FIRST} --out /scratch/merged_first"
    assert first in s and s.index(first) < s.index("realdata_checks.py")
    assert "--first-run-merged /scratch/merged_first" in s
    assert f"--fit {B.MAIN_FIT} --sim {B.SIM_ROOT}" in s and f"OUT={B.CHECKS_ROOT}" in s
    assert (REPO / B.MAIN_FIT).exists()


def test_the_simulation_jobs_stay_off_the_node_that_failed_every_container_start():
    for p in SIM:
        terms = yaml.safe_load(SPECS[p])["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        hosts = next(t for t in terms if t["key"] == "kubernetes.io/hostname")["values"]
        assert "patternlab.calit2.optiputer.net" in hosts and set(B.BAD_NODES) <= set(hosts)


def test_a_split_shard_scores_every_model_once_across_its_parts_and_only_the_last_marks_it_done(tmp_path):
    for i, n in B.RESCORE_PARTS.items():
        parts = [p for p in PARTS if f"-rescore-s{i}-p" in p.name]
        assert len(parts) == n
        names = [re.findall(r'^\s+scored (\S+) "', SPECS[p], re.M) for p in parts]
        flat = [x for g in names for x in g]
        assert sorted(flat) == sorted(m.name for m in B.MODELS[1:]) and len(flat) == len(set(flat)), \
            "disjoint parts covering every model but the first, which wrote the shard's jets.npz"
        for p in parts:
            s = _script(SPECS[p])
            assert 'touch "${OUT}/DONE"' not in s.replace('touch "${OUT}/DONE"; }', ""), "only done_if_all marks DONE"
            assert "jets.npz\" ] && [ -f \"${OUT}/closure.json\" ] ||" in s
            assert f"OUT={B.RESCORE_ROOT}/shard{i}" in s and "attempts_p" in s
    # done_if_all: DONE appears only once every model of the full list is scored
    s = _script(SPECS[PARTS[0]])
    fn = next(ln for ln in s.splitlines() if ln.startswith("done_if_all ()"))
    all_line = next(ln for ln in s.splitlines() if ln.startswith("ALL_MODELS="))
    run = lambda: subprocess.run(["bash", "-c", f"OUT={tmp_path}\n{all_line}\n{fn}\ndone_if_all"], check=True)
    for m in B.MODELS[:-1]:
        (tmp_path / f"scores_{m.name}.npz").touch()
    run()
    assert not (tmp_path / "DONE").exists()
    (tmp_path / f"scores_{B.MODELS[-1].name}.npz").touch()
    run()
    assert (tmp_path / "DONE").exists()


def test_the_checks_job_runs_one_blas_thread_per_process():
    s = _script(SPECS[CHECKS[0]])
    env = "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    assert env in s and s.index(env) < s.index("python3 experiments/AOJ/realdata_checks.py")


def test_the_read_back_job_mounts_the_volume_read_only_and_writes_nothing():
    doc = yaml.safe_load(SPECS[K8S_READ])
    pod = doc["spec"]["template"]["spec"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert all(m.get("readOnly") for m in pod["containers"][0]["volumeMounts"])
    s = _script(SPECS[K8S_READ])
    assert f"{B.CHECKS_ROOT}/prong_test.json" in s and "BEGIN-TAR" in s and ">" not in s.replace("2>", "")


K8S_READ = B.K8S / "job-aoj-read-checks-v1-raunav.yaml"


# ---- the injection test (2026-10-01) ----
INJ_BINS = B.K8S / "job-aoj-injection-bins-v1-raunav.yaml"


def test_the_injection_bins_job_merges_the_first_run_as_the_fit_did_and_carries_the_retry_policy():
    spec = yaml.safe_load(SPECS[INJ_BINS])["spec"]
    assert spec["backoffLimit"] == FT.ROBUST_BACKOFF and spec["podFailurePolicy"]["rules"][0]["onExitCodes"][
        "values"] == [FT.EXIT_HALT]
    s = _script(SPECS[INJ_BINS])
    assert f"{B.OUT_ROOT}/shard${{i}}/DONE" in s and f"seq 0 {B.N_SHARDS - 1}" in s
    merge = "python3 experiments/AOJ/merge_shards.py --shards ${SHARDS} --out /scratch/merged"
    assert merge in s and s.index(merge) < s.index("injection_test.py bins")
    assert f'--branch "{B.INJECTION_PIN}"' in s and f"--committed {B.COMMITTED_BINS}" in s
    assert (REPO / B.COMMITTED_BINS).exists()
    assert s.index("export OMP_NUM_THREADS=1") < s.index("injection_test.py bins")
    assert s.index("pip install --no-cache-dir -q pyarrow") < s.index("injection_test.py bins")
    assert f"OUT={B.INJECTION_ROOT}" in s and "BEGIN-TAR" in s and "END-TAR" in s


TOYS = sorted(p for p in SPECS if "-injection-toys-" in p.name)


@pytest.mark.parametrize("path", TOYS, ids=lambda p: p.name)
def test_the_toy_jobs_carry_the_retry_policy_and_count_their_own_attempts(path):
    spec = yaml.safe_load(SPECS[path])["spec"]
    assert spec["backoffLimit"] == FT.ROBUST_BACKOFF and spec["podFailurePolicy"]["rules"][0]["onExitCodes"][
        "values"] == [FT.EXIT_HALT]
    study = path.name.removeprefix("job-aoj-injection-toys-").removesuffix("-v1-raunav.yaml")
    s = _script(SPECS[path])
    assert f'"${{OUT}}/attempts/toys_{study}/ATTEMPTS"' in s, "attempts are counted per job, not shared"
    assert f'--branch "{B.TOYS_PINS[study]}"' in s and "nvidia.com/gpu" not in SPECS[path]
    assert s.index("export OMP_NUM_THREADS=1") < s.index("injection_test.py toys")
    assert s.count("injection_test.py toys") == len(B.TOY_STUDIES[study])
    assert s.count("--out \"${OUT}/toys_") == len(B.TOY_STUDIES[study])
    if study in B.NEEDS_EXTRA:
        pre = '[ -f "${OUT}/injection_bins.npz" ] || { echo "FATAL: no injection bins"; exit 42; }'
        assert pre in s and s.index(pre) < s.index("injection_test.py toys")


def test_the_leak_study_leaves_out_the_reference_which_has_no_simulated_efficiency():
    s = _script(SPECS[B.K8S / "job-aoj-injection-toys-top-v1-raunav.yaml"])
    leak = next(ln for ln in s.split("injection_test.py toys")[1:] if "--modes leak" in ln)
    names = re.search(r"--names ([^\\]+)\\", leak).group(1).split()
    assert "reference" not in names and set(names) == {m.name for m in B.MODELS}


FIT5 = B.K8S / "job-aoj-fit-v5-raunav.yaml"
CHECKS2 = B.K8S / "job-aoj-checks-v2-raunav.yaml"


@pytest.mark.parametrize("path", [FIT5, CHECKS2], ids=lambda p: p.name)
def test_the_v5_fit_and_v2_checks_carry_the_retry_policy_and_a_storage_guard_before_writing(path):
    spec = yaml.safe_load(SPECS[path])["spec"]
    assert spec["backoffLimit"] == FT.ROBUST_BACKOFF and spec["podFailurePolicy"]["rules"][0]["onExitCodes"][
        "values"] == [FT.EXIT_HALT]
    s = _script(SPECS[path])
    guard = s.index("df --output=pcent /data")
    assert guard < s.index("git clone") and '-lt 95 ]' in s
    assert "BEGIN-TAR" in s and "nvidia.com/gpu" not in SPECS[path]
    assert s.index("export OMP_NUM_THREADS=1") < s.index("python3 experiments/AOJ/" + (
        "fit_v5.py" if path == FIT5 else "realdata_checks.py"))


def test_the_v2_checks_are_the_v1_checks_against_fit_v5_and_nothing_else_changes():
    one, two = _script(SPECS[CHECKS[0]]), _script(SPECS[CHECKS2])
    assert f"--fit {B.MAIN_FIT2} " in two and f"OUT={B.CHECKS2_ROOT}" in two and f'--branch "{B.CHECKS2_PIN}"' in two
    strip = lambda t: "\n".join(ln for ln in t.splitlines()
                                if not any(k in ln for k in ("OUT=", "--branch", "--fit ", "USED", "FREE_G",
                                                             "PVC used", "BEGIN-TAR", "END-TAR", "tar czf",
                                                             'cd "${OUT}"')) and ln.strip() != "echo")
    assert strip(one) == strip(two).replace(f"--workers {B.CHECKS2_CPU - 1}", "--workers 15")


def test_the_v5_fit_runs_from_the_committed_bins_and_writes_its_own_directory():
    s = _script(SPECS[FIT5])
    assert f'--branch "{B.FIT5_PIN}"' in s and "merge_shards" not in s
    assert f"OUT={B.FIT5_ROOT}" in s and "fit_v5.py --workers" in s
    assert (REPO / "experiments/FIGS/data/aoj_full_v1/fit_v3/bins.npz").exists()


def test_the_injection_read_back_mounts_the_volume_read_only_and_writes_nothing():
    path = B.K8S / "job-aoj-read-injection-v1-raunav.yaml"
    pod = yaml.safe_load(SPECS[path])["spec"]["template"]["spec"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert all(m.get("readOnly") for m in pod["containers"][0]["volumeMounts"])
    s = _script(SPECS[path])
    assert f'cd "{B.INJECTION_ROOT}"' in s and "toys_*.jsonl" in s and ">" not in s.replace("2>", "")
