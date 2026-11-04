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


def test_the_launched_curves_and_bootstrap_jobs_keep_mtx_s2_20_and_the_status_job_keeps_its_pin(specs):
    assert BC.PIN == "mtx-s2.20"
    for p, text in specs.items():
        c = yaml.safe_load(text)["spec"]
        if p.parent == BC.OUT_DIR:
            name = p.name.removeprefix("job-").removesuffix("-raunav.yaml")
            pin = (BC.PIN_ROC_JC1OVR if name == "curves-roc-jc1ovr"
                   else BC.PIN_ROC if name.startswith("curves-roc") or name in BC.UNLAUNCHED_AT_PIN
                   else BC.PIN_FULL if name == "curves-full" else BC.PIN_V4 if name.endswith("-v4") else "mtx-s2.20")
            assert f'--branch "{pin}"' in c["template"]["spec"]["containers"][0]["args"][0]
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


# ------------------------------------------------------------------ the ROC curves jobs (2026-11-02)
ROC_JOBS = ("job-curves-roc-raunav.yaml", "job-curves-roc-kin-raunav.yaml")


def _script(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_the_roc_jobs_are_pinned_cpu_tried_once_and_keep_off_the_bad_nodes(specs):
    launch = BC.BF._mtx_launch()
    for job in ROC_JOBS:
        text = specs[BC.OUT_DIR / job]
        d = yaml.safe_load(text)
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert d["metadata"]["name"] == job.removeprefix("job-").removesuffix(".yaml") and "raunav" in job
        assert d["spec"]["backoffLimit"] == 1 and "nvidia.com/gpu" not in c["resources"]["limits"]
        assert f'--branch "{BC.PIN_ROC}"' in c["args"][0] and f'export REPO_REF="{BC.PIN_ROC}"' in c["args"][0]
        assert "k8s-haosu-10.sdsc.optiputer.net" in text and "configmap" not in text.lower()
        terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        notin = [t["values"] for t in terms if t["operator"] == "NotIn"]
        assert notin == [list(launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)]
    # the launched specs keep their exclusion text
    assert "k8s-haosu-10" not in specs[BC.OUT_DIR / "job-curves-full-raunav.yaml"]


def test_the_roc_jobs_write_only_new_files_under_curves_v2(specs):
    assert BC.ROC_OUT == "/data/results/ft_v2/curves_v2" and BC.NEEDED_ROC == {"experiments/FT/roc_points.py": "def curve_doc"}
    old = {f"{BC.CURVES}/", f"{BC.BOOT}/", f"{BC.FITS_V4}/", f"{BC.PROBS_V4}/"}
    for job in ROC_JOBS:
        script = _script(specs[BC.OUT_DIR / job])
        outs = re.findall(r"--out-dir (\S+)", script) + re.findall(r"> (\S+)", script)
        assert outs and all(o.startswith(BC.ROC_OUT + "/") for o in outs), outs
        assert "--out " not in script and not any(f"--out-dir {o}" in script for o in old)
        assert not re.search(r"\b(rm|mv)\s", script)
    # a family's DONE is written once, after its calls
    script = _script(specs[BC.OUT_DIR / "job-curves-roc-raunav.yaml"])
    for fam in BC.ROC_ORDER:
        done = f'[ -e {BC.ROC_OUT}/{fam}/DONE ] || echo "{BC.PIN_ROC}" > {BC.ROC_OUT}/{fam}/DONE'
        assert script.count(done) == 1 and script.rindex(f"--out-dir {BC.ROC_OUT}/{fam} ") < script.index(done)
    order = [script.index(f"--out-dir {BC.ROC_OUT}/{fam} ") for fam in BC.ROC_ORDER]
    assert order == sorted(order)


def test_the_roc_job_waits_for_the_full_sets_and_every_fit_and_covers_every_family(specs):
    script = _script(specs[BC.OUT_DIR / "job-curves-roc-raunav.yaml"])
    waited = script.split("missing () { local n=0 f; for f in ", 1)[1].split("; do\n", 1)[0].split()
    models = [m for m, _ in BC.PROBE_MODELS]
    assert len(models) == 30 and f"{BC.FITS_V4}/{BC._alt(models)}.json" in waited
    assert f"{BC.FITS_SIB}/{BC._alt(models)}.json" in waited and BC.FITS_SIB.endswith("/fits_v4_sib")
    assert f"{REF}/jc1/leg2/scratch-v2/N1000000/s{{1,2,3}}/DONE" in waited
    assert f"{BC.V2}/best70/leg_qg/{{" + ",".join(BC.FULL_RUNS) + "}/N1600000/s1/DONE" in waited
    calls = BC.roc_calls()
    assert sum(len(c) for c in calls.values()) == 2 + 1 + 2 + 2 + 2 + 1
    assert "--tree /data/results/ft_v2/ref_v100/jc1/leg2 --task jc1 --n 1000000 --inits scratch-v2" in calls["jc1_ft"][1]
    assert calls["qg_ft"][0].endswith("--inits " + " ".join(BC.FULL_RUNS)) and "--n 1600000" in calls["qg_ft"][0]
    assert f"--files {' '.join(BC.ROC_PAIR_FILES)} " in calls["pairs"][0] and BC.PROBS_SIB in calls["pairs"][0]
    rp = _load("roc_points", "experiments/FT/roc_points.py")
    assert {f.removeprefix("jc2bal_pairs_") for f in BC.ROC_PAIR_FILES} == set(rp.PAIR_SIDES)
    # the folder the sibling-pair job writes (2026-11-03: the two builders had spelled it apart)
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    assert f"--probs-out {BC.PROBS_SIB}" in bj.sib_spec("mtx-s9.99", 5)[1] and f"--out {BC.FITS_SIB} " in bj.sib_spec("mtx-s9.99", 5)[1]
    assert calls["pairs"][1].endswith("--files jc2bal_pairs_ee_vs_mm --models init-s1 init-s2 init-s3")
    assert len(BC.ROC_OVR_RUNS) == 12 and calls["jc2_ovr"][0].endswith(" ".join(BC.ROC_OVR_RUNS))
    for c in (c for cs in calls.values() for c in cs):
        assert c.split()[0] in ("ft", "probes") and f"python3 experiments/FT/roc_points.py {c} \\\n" in script


def test_the_kinematic_roc_job_waits_for_its_fine_tuning(specs):
    bfb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    assert BC.KIN_TREE == f"{bfb.BUDGET_ROOT}/{bfb.TASKS['topkin'][4]}"
    assert (BC.KIN_INIT,) == bfb.KIN_RUNS and BC.KIN_N == bfb.FULL["topkin"]
    script = _script(specs[BC.OUT_DIR / "job-curves-roc-kin-raunav.yaml"])
    marker = f"{BC.KIN_TREE}/l188kin-s1/N1200000/s1/DONE"
    assert f"for f in {marker}; do" in script and "after 6 days" in script and "exit 42" in script
    assert (f"roc_points.py ft --family top_ft --tree {BC.KIN_TREE} --task top --n 1200000 --inits l188kin-s1"
            in script) and f"--out-dir {BC.ROC_OUT}/top_ft || halt" in script


def test_the_roc_script_is_checked_at_the_roc_tag_only(monkeypatch):
    seen = []
    monkeypatch.setattr(BC.BF, "verify_pin", lambda pin, nyt, needed: seen.append((pin, dict(needed))))
    monkeypatch.setattr(BC, "build", lambda pin: {})
    monkeypatch.setattr("sys.argv", ["build_curve_jobs.py", "--pin-not-yet-tagged", "--check-only"])
    assert BC.main() == 0
    assert seen == [(BC.PIN, BC.NEEDED), (BC.PIN_ROC, {**BC.NEEDED, **BC.NEEDED_ROC}),
                    (BC.PIN_ROC_JC1OVR, {**BC.NEEDED, **BC.NEEDED_ROC_JC1OVR})]
    assert "experiments/FT/roc_points.py" not in BC.NEEDED
    monkeypatch.setattr("sys.argv", ["build_curve_jobs.py", "--tag", "mtx-s9.99"])
    with pytest.raises(SystemExit, match="is not PIN_ROC"):
        BC.main()


# ------------------------------------------------------------------ the class-against-the-rest job (2026-11-03)
OVR_JOB = "job-curves-roc-jc1ovr-raunav.yaml"


def test_the_class_against_the_rest_job_has_its_own_pin_and_the_launched_roc_jobs_keep_theirs(specs):
    assert BC.PIN_ROC_JC1OVR == "mtx-s2.57" and BC.PIN_ROC == "mtx-s2.53"
    launch = BC.BF._mtx_launch()
    text = specs[BC.OUT_DIR / OVR_JOB]
    d = yaml.safe_load(text)
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert d["metadata"]["name"] == "curves-roc-jc1ovr-raunav" and d["spec"]["backoffLimit"] == 1
    assert "nvidia.com/gpu" not in c["resources"]["limits"] and "configmap" not in text.lower()
    assert f'--branch "{BC.PIN_ROC_JC1OVR}"' in c["args"][0] and f'export REPO_REF="{BC.PIN_ROC_JC1OVR}"' in c["args"][0]
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert [t["values"] for t in terms if t["operator"] == "NotIn"] == [list(launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)]
    for job in ROC_JOBS:
        assert "_ovr" not in specs[BC.OUT_DIR / job].replace("jc2_ovr", "") and f'--branch "{BC.PIN_ROC}"' in specs[BC.OUT_DIR / job]


def test_the_class_against_the_rest_job_swaps_the_family_of_the_jc1_calls_and_writes_each_done(specs):
    calls, old = BC.roc_jc1ovr_calls(), BC.roc_calls()
    assert list(calls) == ["jc1_ft_ovr", "jc1_probe_ovr"]
    for fam in ("jc1_ft", "jc1_probe"):
        assert calls[f"{fam}_ovr"] == [c.replace(f"--family {fam} ", f"--family {fam}_ovr ") for c in old[fam]]
        assert all(f"--family {fam}_ovr " in c for c in calls[f"{fam}_ovr"])
    script = _script(specs[BC.OUT_DIR / OVR_JOB])
    for fam, cs in calls.items():
        done = f'[ -e {BC.ROC_OUT}/{fam}/DONE ] || echo "{BC.PIN_ROC_JC1OVR}" > {BC.ROC_OUT}/{fam}/DONE'
        assert script.count(done) == 1 and script.rindex(f"--out-dir {BC.ROC_OUT}/{fam} ") < script.index(done)
        for c in cs:
            assert f"python3 experiments/FT/roc_points.py {c} \\\n" in script
    outs = re.findall(r"--out-dir (\S+)", script) + re.findall(r"> (\S+)", script)
    assert outs and all(o.startswith((f"{BC.ROC_OUT}/jc1_ft_ovr", f"{BC.ROC_OUT}/jc1_probe_ovr")) for o in outs), outs
    assert "--out " not in script and not re.search(r"\b(rm|mv)\s", script)


def test_the_class_against_the_rest_job_waits_for_the_jetclass_cells_and_the_fits(specs):
    script = _script(specs[BC.OUT_DIR / OVR_JOB])
    waited = script.split("missing () { local n=0 f; for f in ", 1)[1].split("; do\n", 1)[0].split()
    models = [m for m, _ in BC.PROBE_MODELS]
    assert f"{BC.FITS_V4}/{BC._alt(models)}.json" in waited and len(waited) == 3
    assert f"{REF}/jc1/leg2/scratch-v2/N1000000/s{{1,2,3}}/DONE" in waited
    assert f"{BC.V2}/best70/leg2/{{" + ",".join(BC.FULL_RUNS) + "}/N1000000/s1/DONE" in waited
