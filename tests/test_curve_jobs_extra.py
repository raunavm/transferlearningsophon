"""scripts/build_curve_jobs.py's bootstrap jobs: the flavour and random-partition ratios, and the
fine-tunes' from-scratch baseline on subset s1 (seeds 1-3, scratch-s1-<seed>)."""
import importlib.util
import pathlib
import re
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BC = _load("build_curve_jobs", "scripts/build_curve_jobs.py")
REF, REF_S1 = "/data/results/ft_v2/ref_v100", "/data/results/ft_v2/ref_v100_s1"


@pytest.fixture(scope="module")
def specs():
    return BC.build(BC.PIN)


def _pairs(text):
    """The PAIRS a bootstrap job's every call reports."""
    return re.search(r"PAIRS=\(([^)]*)\)", text).group(1).replace("--pair ", "").split()


def test_the_curves_and_bootstrap_jobs_pin_mtx_s2_20_and_the_launched_status_job_keeps_its_pin(specs):
    assert BC.PIN == "mtx-s2.20"
    for p, text in specs.items():
        c = yaml.safe_load(text)["spec"]
        if p.parent == BC.OUT_DIR:
            assert '--branch "mtx-s2.20"' in c["template"]["spec"]["containers"][0]["args"][0]
        else:
            assert BC._pin_of(text) == "mtx-s2.19"


def test_the_probe_reports_the_flavour_and_random_partition_ratios(specs):
    pairs = _pairs(specs[BC.OUT_DIR / "job-boot-probe-raunav.yaml"])
    for k in (1, 2):
        assert {f"mtx-flavf1-s{k}/mtx-flavf0-s{k}", f"mtx-flavf1r-s{k}/mtx-flavf1-s{k}",
                f"mtx-flavf1r-s{k}/mtx-flavf0-s{k}", f"mtx-flavf0-s{k}/mtx-r16q1-s{k}"} <= set(pairs)
    rand = [m for m, _ in BC.PROBE_MODELS if m.startswith("mtx-rand2p")]
    assert len(rand) == 5
    for m in rand:
        assert f"{m}/mtx-l188-s{m.rsplit('-s', 1)[1]}" in pairs
    # every pair names models of the call; none twice
    models = {m for m, _ in BC.PROBE_MODELS}
    assert all(set(q.split("/")) <= models for q in pairs) and len(pairs) == len(set(pairs))


def test_a_flavour_ratio_over_a_run_not_bootstrapped_is_refused():
    with pytest.raises(AssertionError):
        BC.ratio_pairs(["l188-s1", "flavf1-s1"], "", lambda k: "x")


def test_every_probe_task_reports_every_ratio(specs):
    """One PAIRS for the job, so Herwig at every size and the six pair tasks get them all."""
    text = specs[BC.OUT_DIR / "job-boot-probe-raunav.yaml"]
    assert text.count("PAIRS=(") == 1 and '"${PAIRS[@]}"' in text
    for n in BC.SIZES["qg_herwig"]:
        assert f"          boot qg_herwig {n}\n" in text
    for p in BC.PAIRS_188:
        assert f"          boot jc2_188_pairs_{p} {BC.NMAX}\n" in text


def test_the_fine_tunes_are_over_the_from_scratch_seed_of_their_index_on_s1(specs):
    for job in ("job-boot-ft-raunav.yaml", "job-boot-jc2-raunav.yaml"):
        text = specs[BC.OUT_DIR / job]
        pairs = _pairs(text)
        for r in BC.FT_RUNS:
            assert f"{r}/scratch-s1-{r.rsplit('-s', 1)[1]}" in pairs
        assert "l162mass-s2/l162-s2" in pairs and "r16q1-s3/l188-s3" in pairs
        assert "scratch-v2-s" not in " ".join(pairs) and text.count("PAIRS=(") == 1
    text = specs[BC.OUT_DIR / "job-boot-ft-raunav.yaml"]
    pre, ref = BC.TREES["best70_leg_qg"][0], BC.TREES["ref_v100_bench_leg_qg"][0]
    for n in BC.SIZES["qg_herwig"]:
        assert f"          boot qg_herwig {pre} {ref} {n} " in text


def _boot_calls(tmp_path, monkeypatch, family):
    """Run the job's body with python3 logging its arguments; one dict of models per call."""
    monkeypatch.setattr(BC, "BOOT", str(tmp_path / "boot"))
    monkeypatch.setattr(BC, "_guard", lambda outs, inputs: "")
    log = tmp_path / "calls"
    script = ("set -euo pipefail\nhalt () { exit 42; }\n"
              f"python3 () {{ echo \"$*\" >> {log}; }}\n" + BC.boot_ft_body(family))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "boot" / family).is_dir()
    out = {}
    for line in log.read_text().splitlines():
        task = re.search(r"--task (\S+)", line).group(1)
        n = int(re.search(r"--out \S+_N(\d+)\.json", line).group(1))
        out[(task, n)] = dict(m.split("=", 1) for m in re.findall(r"--model (\S+)", line))
    return out


def test_the_from_scratch_seeds_on_s1_are_read_where_they_trained_on_s1(tmp_path, monkeypatch):
    calls = _boot_calls(tmp_path, monkeypatch, "ft")
    assert len(calls) == 16 and {t for t, _ in calls} == {"jc1", "top", "qg", "qg_herwig"}
    for (task, n), models in calls.items():
        assert len(models) == len(BC.FT_RUNS) + 3 and list(models)[0] == BC.FT_RUNS[0]
        tree = {"jc1": "jc1/leg2", "top": "bench/leg_top"}.get(task, "bench/leg_qg")
        full = task != "jc1" and n == BC.SIZES[task][-1]
        assert models["scratch-s1-1"] == f"{REF}/{tree}/scratch-v2/N{n}/s1"
        for s in (2, 3):
            # seeds 2-3 of ref_v100 train on subsets s2-s3, except at the full top and q/g sets
            assert models[f"scratch-s1-{s}"] == f"{REF if full else REF_S1}/{tree}/scratch-v2/N{n}/s{s}"


def test_jetclass2_reads_all_three_seeds_from_its_reference(tmp_path, monkeypatch):
    """On JetClass-II every from-scratch seed trains on subset s1."""
    calls = _boot_calls(tmp_path, monkeypatch, "jc2")
    assert len(calls) == 4
    for (task, n), models in calls.items():
        assert task == "jc2_188"
        for s in (1, 2, 3):
            assert models[f"scratch-s1-{s}"] == f"{REF}/jc2/{BC.LEG1_188}/scratch-v2/N{n}/s{s}"


def test_the_bootstrap_waits_for_the_from_scratch_seeds_on_s1(specs):
    ft = specs[BC.OUT_DIR / "job-boot-ft-raunav.yaml"]
    assert f"{REF_S1}/jc1/leg2/scratch-v2/N{{1000,10000,100000,1000000}}/s{{2,3}}/DONE" in ft
    for d in ("top", "qg"):
        assert f"{REF_S1}/bench/leg_{d}/scratch-v2/N{{1000,10000,100000}}/s{{2,3}}/DONE" in ft
    assert REF_S1 not in specs[BC.OUT_DIR / "job-boot-jc2-raunav.yaml"]


def test_the_bootstrap_reads_and_waits_for_the_q_g_cells_build_ft_jobs_trains_on_s1(tmp_path, monkeypatch, specs):
    """The q/g cells of build_ft_jobs' from-scratch reference on subset s1 are exactly the seeds 2-3
    the fine-tuning bootstrap reads from REF_S1 (Pythia and Herwig), and boot-ft waits for their DONE."""
    cells = {f"{root}/{t}/{i}/N{n}/s{s}" for t, (root, _, i, _, n, s) in BC.BF.ref_s1_cells() if t == "leg_qg"}
    assert cells and BC.REF_S1 == BC.BF.V2_REF_S1_ROOT == REF_S1
    calls = _boot_calls(tmp_path, monkeypatch, "ft")
    for task in ("qg", "qg_herwig"):
        read = {m for (t, _), models in calls.items() if t == task for m in models.values() if m.startswith(REF_S1)}
        assert read == cells, task
    ft = specs[BC.OUT_DIR / "job-boot-ft-raunav.yaml"]
    waited = ft.split("missing () { local n=0 f; for f in ", 1)[1].split("; do\n", 1)[0].split()
    waits = BC.s1_markers("ref_v100_bench_leg_qg")
    assert len(waits) == 1 and waits[0] in waited
    expanded = subprocess.run(["bash", "-c", f"printf '%s\\n' {waits[0]}"], capture_output=True, text=True,
                              check=True).stdout.split()
    assert set(expanded) == {f"{c}/DONE" for c in cells}
