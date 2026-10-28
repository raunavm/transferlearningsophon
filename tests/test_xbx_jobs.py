"""The X->bb / X->bs job specs (scripts/build_xbx_jobs.py)."""
import importlib.util
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_xbx_jobs_t", ROOT / "scripts/build_xbx_jobs.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
XS = importlib.util.spec_from_file_location("xbx_jt", ROOT / "experiments/XBS/xbx.py")
X = importlib.util.module_from_spec(XS)
XS.loader.exec_module(X)
SPECS = B.build()


def _script(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def _products(text):
    terms = yaml.safe_load(text)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    return {(e["operator"], tuple(e["values"])) for e in terms if e["key"] == "nvidia.com/gpu.product"}


def test_the_specs_on_disk_are_the_generators():
    on_disk = {p.name: p.read_text() for p in (ROOT / "experiments/XBS/k8s").glob("*.yaml")}
    assert on_disk == SPECS


def test_every_paper_model_and_untrained_network_has_an_extraction_and_a_fit():
    paper = [ln.strip() for ln in (ROOT / "configs/arms/v2_paper_runs.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    models = [m for m, *_ in B.models()]
    assert models == paper + ["init-s1", "init-s2", "init-s3"] and len(paper) == 27
    for m in models:
        stem = m.removeprefix("mtx-")
        assert f"job-xbx-x-{stem}-raunav.yaml" in SPECS and f"job-xbx-fit-{stem}-raunav.yaml" in SPECS
    assert len(SPECS) == 2 + 2 * len(models) + len(B.SCRATCH_SIGNALS) * len(B.SCRATCH_SEEDS)


@pytest.mark.parametrize("model,run,arm,k,reg,ckpt", B.models())
def test_an_extraction_scores_its_own_vocabulary_from_its_primary_checkpoint(model, run, arm, k, reg, ckpt):
    s = _script(SPECS[f"job-xbx-x-{model.removeprefix('mtx-')}-raunav.yaml"])
    assert f"--run-dir /data/results/mtx_v2/{run} " in s and f"--checkpoints {ckpt} " in s
    assert f"--num-classes {k} --num-reg {reg} " in s and f"extract_xbx.py --arm {arm} " in s
    if model.startswith("init-"):
        assert ckpt == "init" and k == 0
    else:
        assert ckpt == "best70" and X.node_groups(arm)["k"] == k
    for sig in B.SIGNALS:
        assert f"{B.STAGE}/pairs/{sig}/DONE ]" in s and f"extract {sig}/train " in s and f"extract {sig}/eval " in s
    assert "extract qcd/eval " in s and "--data-config configs/xbx/JetClassII_xbx.yaml" in s


def test_every_spec_is_ours_retries_only_what_a_retry_can_fix_and_parses_as_bash(tmp_path):
    for name, text in SPECS.items():
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav") and d["metadata"]["namespace"] == "cms-ml"
        assert f"--branch \"{B.PIN}\"" in text
        rules = d["spec"]["podFailurePolicy"]["rules"]
        assert {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}} in rules
        assert any(r["action"] == "Ignore" for r in rules)
        f = tmp_path / f"{name}.sh"
        f.write_text(_script(text))
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, name


def test_the_gpu_jobs_stay_off_the_grids_and_the_fine_tunes_gpus_and_the_scratch_runs_on_v100():
    for name, text in SPECS.items():
        if "-stage-" in name:
            assert "nvidia.com/gpu" not in text
        elif "-scratch-" in name:
            assert _products(text) == {("In", (B.SCRATCH_GPU,))}
        else:
            assert _products(text) == {("NotIn", B.NOT_ON)}


def test_the_from_scratch_baseline_is_sophons_recipe_and_resumes():
    s = _script(SPECS["job-xbx-scratch-xbs-s2-raunav.yaml"])
    assert "--start-lr 5e-4 --num-epochs 50" in s and "--optimizer-option weight_decay 0.01" in s
    assert "--optimizer ranger" in s and "--lr-scheduler" not in s          # weaver's flat+decay
    assert "--batch-size 512" in s and "--seed 2 " in s and "-o num_classes 2" in s
    assert "cell_resume.py prepare" in s and "--load-epoch" in s and "cell_resume.py best" in s
    assert "/data/results/xbx/scratch/xbs/s2" in s and "fit_xbx.py scratch" in s
    assert yaml.safe_load(SPECS["job-xbx-scratch-xbs-s2-raunav.yaml"])["spec"]["backoffLimit"] == 20


def test_the_outside_samples_are_checked_before_they_are_converted():
    s = _script(SPECS["job-xbx-stage-ihep-raunav.yaml"])
    for sig, d in B.SAMPLE_DIRS.items():
        assert s.index(f"stage_xbx.py check --files ${{F_{sig}}}") < s.index(f"convert --name {sig}") < s.index(f"pair --signal {sig}")
        assert f"ls {d}/*.root" in s
    j = _script(SPECS["job-xbx-stage-jc2-raunav.yaml"])
    assert j.index("jc2-eval --name qcd") < j.index("pair --signal xbb_jc2")


def test_the_staging_job_runs_the_xbx_tests_in_the_image_first_and_allows_no_skip():
    s = _script(SPECS["job-xbx-stage-jc2-raunav.yaml"])
    i = s.index("python3 -m pytest")
    assert i < s.index("jc2-eval --name qcd") and all(t in s[i:i + 300] for t in B.SELF_TESTS)
    assert 'grep -q "skipped"' in s
