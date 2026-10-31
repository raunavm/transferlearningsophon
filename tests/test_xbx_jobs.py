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
        for pre in ("xbx", "xbx-ihep-xbb", "xbx-ihep-xbs"):
            assert f"job-{pre}-x-{stem}-raunav.yaml" in SPECS and f"job-{pre}-fit-{stem}-raunav.yaml" in SPECS
    scratch = len(B.SCRATCH_SIGNALS) * len(B.SCRATCH_SEEDS)
    # the stand-in chain (and the two superseded IHEP specs) as written, the real samples (one
    # extraction and one fit per sample and model), the dry run, the two report jobs
    assert len(SPECS) == (2 + 2 * len(models) + scratch) + (1 + 4 * len(models) + scratch) + 1 + 2


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
    for sig, other in (("xbb", "xbs"), ("xbs", "xbb")):
        # each sample alone: it needs only its own staging, so it can come without the other
        r = _script(SPECS[f"job-xbx-ihep-{sig}-x-{model.removeprefix('mtx-')}-raunav.yaml"])
        assert r.count(f"extract {sig}/") == 3 and f"/{other}/" not in r and f"extract {other}/" not in r
        assert f"{B.STAGE}/pairs/{sig}/DONE ]" in r and f"{B.STAGE}/{sig}/eval/DONE ]" in r
        assert f"--run-dir /data/results/mtx_v2/{run} " in r and f"--checkpoints {ckpt} " in r and "extract qcd/eval " in r
        f = _script(SPECS[f"job-xbx-ihep-{sig}-fit-{model.removeprefix('mtx-')}-raunav.yaml"])
        assert f"--signals {sig} --checkpoint {ckpt} " in f and f"--out {B.FITS_IHEP}/{sig} || halt" in f


def test_every_spec_is_ours_retries_only_what_a_retry_can_fix_and_parses_as_bash(tmp_path):
    for name, text in SPECS.items():
        d = yaml.safe_load(text)
        assert d["metadata"]["name"].endswith("-raunav") and d["metadata"]["namespace"] == "cms-ml"
        new = name.startswith("job-xbx-ihep-") or "dryrun" in name
        pin = B.PIN_REPORT if "-report-" in name else B.PIN_IHEP if new else B.PIN
        assert f"--branch \"{pin}\"" in text
        rules = d["spec"]["podFailurePolicy"]["rules"]
        assert {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}} in rules
        assert any(r["action"] == "Ignore" for r in rules)
        f = tmp_path / f"{name}.sh"
        f.write_text(_script(text))
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, name


def test_the_gpu_jobs_stay_off_the_grids_and_the_fine_tunes_gpus_and_the_scratch_runs_on_v100():
    for name, text in SPECS.items():
        if "-stage-" in name or "-report-" in name:
            assert "nvidia.com/gpu" not in text
        elif "-scratch-dryrun-" in name:
            # the plumbing check only: any GPU but the fine-tunes' V100s and the grid's RTX 3090s
            assert _products(text) == {("In", B.DRYRUN_GPUS)}
            assert not {B.SCRATCH_GPU, "NVIDIA-GeForce-RTX-3090"} & set(B.DRYRUN_GPUS)
        elif "-scratch-" in name:
            assert _products(text) == {("In", (B.SCRATCH_GPU,))}
        else:
            assert _products(text) == {("NotIn", B.NOT_ON)}


@pytest.mark.parametrize("name,out", [("job-xbx-scratch-xbs-s2-raunav.yaml", "/data/results/xbx/scratch/xbs/s2"),
                                      ("job-xbx-ihep-scratch-xbs-s2-raunav.yaml", "/data/results/xbx/scratch_ihep/xbs/s2")])
def test_the_from_scratch_baseline_is_sophons_recipe_and_resumes(name, out):
    s = _script(SPECS[name])
    assert "--start-lr 5e-4 --num-epochs 50" in s and "--optimizer-option weight_decay 0.01" in s
    assert "--optimizer ranger" in s and "--lr-scheduler" not in s          # weaver's flat+decay
    assert "--batch-size 512" in s and "--seed 2 " in s and "-o num_classes 2" in s
    assert "cell_resume.py prepare" in s and "--load-epoch" in s and "cell_resume.py best" in s
    assert f"OUT={out}\n" in s and "fit_xbx.py scratch" in s
    assert yaml.safe_load(SPECS[name])["spec"]["backoffLimit"] == 20


def test_the_real_sample_jobs_have_their_own_names_and_write_somewhere_new():
    # the report's dry run reads the stand-in fits by design
    old = {n for n in SPECS if not n.startswith("job-xbx-ihep-") and "dryrun" not in n}
    new = set(SPECS) - old - {"job-xbx-report-dryrun-raunav.yaml"}
    names = lambda ns: {yaml.safe_load(SPECS[n])["metadata"]["name"] for n in ns}
    assert not names(old) & names(new) and all(n.startswith("xbx-ihep-") for n in names(new) - {"xbx-scratch-dryrun-raunav"})
    assert B.FITS_IHEP == "/data/results/eval/xbx/fits_ihep" != B.FITS
    for n in new:
        s = _script(SPECS[n])
        assert B.FITS + " " not in s and "/data/results/xbx/scratch/" not in s
        if "-fit-" in n:
            assert f"--out {B.FITS_IHEP}/" in s


def test_every_real_sample_spec_passes_the_window_rule_and_the_mass():
    assert B.MX == 125 and B.X.window_for(B.MX) == (93.75, 143.75)
    for n, text in SPECS.items():
        s = _script(text)
        if "-ihep-" in n and ("-fit-" in n or "-scratch-" in n):
            assert "--window 93.75 143.75 --mx 125 " in s, n
        if "-ihep-stage-" in n:
            assert s.count("--window 93.75 143.75 --files") == 4 and "--mx" not in s
    s = _script(B.build(mx=200.0)["job-xbx-ihep-xbs-fit-l188-s1-raunav.yaml"])
    assert "--window 150 230 --mx 200 " in s


def test_the_real_sample_staging_inspects_and_checks_first():
    s = _script(SPECS["job-xbx-ihep-stage-raunav.yaml"])
    for sig in B.SAMPLE_DIRS:
        i = s.index(f"inspect --file \"${{F_{sig}%%")
        j = s.index("stage_xbx.py check --window", i)
        assert i < j < s.index(f"convert --name {sig}", j) < s.index(f"pair --signal {sig}", j)


def _run_stage(tmp_path, present, failing=()):
    """The staging script from its first staging line, its paths moved under tmp_path and
    stage_xbx.py replaced by a logger whose check fails for the samples in failing: (exit code,
    logged calls)."""
    tmp_path.mkdir(parents=True)
    s = _script(SPECS["job-xbx-ihep-stage-raunav.yaml"])
    s = s[s.index(f"mkdir -p {B.STAGE}"):].replace("/data/xbx", str(tmp_path / "data/xbx"))
    for sig in present:
        d = pathlib.Path(B.SAMPLE_DIRS[sig].replace("/data/xbx", str(tmp_path / "data/xbx")))
        d.mkdir(parents=True)
        for k in range(3):
            (d / f"{sig}_{k}.root").write_text("")
    log = tmp_path / "calls.log"
    head = ("set -euo pipefail\n"
            "halt () { exit 42; }\n"
            "du () { :; }\n"
            f"python3 () {{ echo \"$*\" >> {log}; case \"$*\" in {' | '.join(f'*check_{s}.json*' for s in failing) or '-'}) return 1;; esac; }}\n")
    f = tmp_path / "stage.sh"
    f.write_text(head + s)
    rc = subprocess.run(["bash", str(f)], capture_output=True, text=True).returncode
    return rc, log.read_text() if log.exists() else ""


def test_the_real_sample_staging_stages_whichever_sample_is_present(tmp_path):
    rc, calls = _run_stage(tmp_path / "xbs", ["xbs"])
    assert rc == 0 and "convert --name xbs" in calls and "pair --signal xbs" in calls and "xbb" not in calls
    assert "inspect --file " + str(tmp_path / "xbs/data/xbx/ihep/Xbs_ntuple/xbs_0.root") + "\n" in calls
    rc, calls = _run_stage(tmp_path / "both", ["xbb", "xbs"])
    assert rc == 0 and calls.count("convert --name") == 2
    rc, calls = _run_stage(tmp_path / "none", [])
    assert rc == 42 and calls == ""
    # a sample that fails its check does not stop the other; the job fails at the end
    rc, calls = _run_stage(tmp_path / "bad", ["xbb", "xbs"], failing=["xbb"])
    assert rc == 42 and "convert --name xbs" in calls and "pair --signal xbs" in calls
    assert "check_xbb.json" in calls and "convert --name xbb" not in calls


def test_the_dry_run_proves_the_from_scratch_chain_on_the_stand_in():
    text = SPECS["job-xbx-scratch-dryrun-raunav.yaml"]
    s = _script(text)
    assert yaml.safe_load(text)["metadata"]["name"] == "xbx-scratch-dryrun-raunav"
    assert _products(text) == {("In", B.DRYRUN_GPUS)}
    assert f"PAIR={B.STAGE}/pairs/xbb_jc2\n" in s and f"OUT={B.DRYRUN_ROOT}/xbb_jc2/s1\n" in s
    assert "--num-epochs 2 " in s and "cell_resume.py best --dir ${OUT} --epochs 2" in s
    i, j = s.index("ft_weaver.py"), s.index("extract_features.py")
    assert i < j < s.index("fit_xbx.py scratch") and "--save-logits --max-jets 200000" in s
    assert "--window 150 230 --mx 200 --ft-manifest ${OUT}/ft_manifest.json" in s


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


def test_the_report_jobs_read_the_fits_and_never_overwrite():
    dry = _script(SPECS["job-xbx-report-dryrun-raunav.yaml"])
    assert f"xbx_report.py --fits {B.FITS} --as xbb_jc2=xbb --out /data/results/xbx_dryrun/report_{B.PIN_REPORT}" in dry
    real = _script(SPECS["job-xbx-ihep-report-raunav.yaml"])
    assert f"xbx_report.py --fits {B.FITS_IHEP} --out " in real and "--as" not in real
    for s in (dry, real):
        out = s.split("--out ")[1].split()[0]
        assert f"[ ! -e {out} ]" in s
