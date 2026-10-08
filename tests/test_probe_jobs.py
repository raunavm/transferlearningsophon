"""The four-granularity probe jobs: one per seed index, four models each."""
import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_probe_jobs", ROOT / "scripts" / "build_probe_jobs.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def test_forty_three_jobs_all_named_raunav_and_parse():
    jobs = bp.build()
    # 5 probe v1 + 5 label-recovery v1 + 5 probe v2 + 3 random-label control
    # + 5 mass-output 2x2 + 5 mass regression + 1 |V_cb| window (S10)
    # + 14 MLP re-runs (5 ladder v2 + 3 control + 5 mass 2x2 + 1 |V_cb|)
    assert len(jobs) == 43
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"]
        assert fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["backoffLimit"] == 1


def test_each_job_holds_one_seed_index_at_all_four_granularities():
    for fname, text in bp.build().items():
        if any(k in fname for k in ("randcontrol", "mass2x2", "massres", "vcbwindow")):
            continue          # not ladder jobs; each covered by its own test below
        seed = int(fname.split("-s")[-1].split("-")[0])
        line = next(l for l in text.splitlines() if l.strip().startswith("for spec in"))
        runs = line.split("for spec in")[1].split(";")[0].split()
        assert [r.split(":")[1] for r in runs] == ["L188", "L162", "R42_Q1", "R16_Q1"]
        for r in runs:
            name = r.split(":")[0]
            want = "s1b" if (name.startswith("mtx-l162") and seed == 1) else f"s{seed}"
            assert name.endswith(f"-{want}"), (fname, name)
        # the excluded 1e-3 run must never appear
        assert "mtx-l162-s1:" not in line


def test_outputs_are_disjoint_per_seed_and_mlp_cannot_be_skipped():
    outs = [l.strip() for t in bp.build().values() for l in t.splitlines()
            if l.strip().startswith("OUT=")]
    assert len(outs) == len(set(outs)) == 43
    for text in bp.build().values():
        assert "--no-mlp" not in text and "--skip-mlp" not in text


def test_v2_differs_from_v1_only_in_name_output_flag_and_pin():
    """The re-run exists to add 70 % and 90 % signal efficiency to a censored
    metric (docs/PRESPEC_2026-09.md, final section). Anything else that moved
    between v1 and v2 -- the model list, the task list, the bootstrap count, the
    resources -- would make the new rejection numbers incomparable to the
    committed AUCs they sit beside. Stated as an exact rewrite rather than a
    diff count, so a fifth difference cannot hide inside a fourth."""
    jobs = bp.build()
    tasks = " ".join(bp.TASKS)
    for seed in bp.SEEDS:
        v1 = jobs[f"job-probe-ladder-v1-s{seed}-raunav.yaml"]
        v2 = jobs[f"job-probe-ladder-v2-s{seed}-raunav.yaml"]
        rewritten = (
            v1.replace(f"probe-ladder-v1-s{seed}-raunav",
                       f"probe-ladder-v2-s{seed}-raunav")
              .replace(f'--branch "{bp.PIN}"', f'--branch "{bp.PIN_V2}"')
              .replace(f"probe_ladder_v1/s{seed}", f"probe_ladder_v2/s{seed}")
              .replace(f"--tasks {tasks} \\",
                       f"--tasks {tasks} \\\n            --eps-s 0.5 0.7 0.9 \\"))
        assert rewritten == v2, f"seed {seed}: v2 changed something beyond the four"


def test_v1_is_frozen_at_the_tag_and_the_directory_it_ran_at():
    """v1 ran and its results are committed under
    experiments/FIGS/data/probe_ladder_v1/. Re-pinning it or re-pointing its
    output would falsify the ledger, and v2 must not be able to overwrite it."""
    jobs = bp.build()
    for seed in bp.SEEDS:
        v1 = jobs[f"job-probe-ladder-v1-s{seed}-raunav.yaml"]
        assert f'--branch "{bp.PIN}"' in v1 and bp.PIN_V2 not in v1
        assert f"OUT=/data/results/eval/probe_ladder_v1/s{seed}\n" in v1
        assert "--eps-s" not in v1, "v1 must reproduce the default behaviour"
        v2 = jobs[f"job-probe-ladder-v2-s{seed}-raunav.yaml"]
        assert f"OUT=/data/results/eval/probe_ladder_v2/s{seed}\n" in v2
        assert "probe_ladder_v1" not in v2


def test_the_v2_pin_and_its_operating_points():
    assert bp.PIN_V2 == "mtx-s1.51"
    assert bp.EPS_S_V2 == [0.5, 0.7, 0.9]
    assert bp.EPS_S_V2[0] == 0.5, (
        "50 % must stay FIRST: the flat rejection fields mirror the first "
        "operating point and experiments/STATS/seed_level.py resolves "
        "rejection_at through eps_s_default, so v2 must still contain v1")
    for seed in bp.SEEDS:
        v2 = bp.build()[f"job-probe-ladder-v2-s{seed}-raunav.yaml"]
        assert f'--branch "{bp.PIN_V2}"' in v2
        assert "--eps-s 0.5 0.7 0.9" in v2


def test_the_label_recovery_jobs_are_untouched_by_the_rerun():
    """Label recovery has no rejection metric and no censoring defect, so it
    stays at v1, at its original pin, with no operating-point flag."""
    jobs = bp.build()
    for seed in bp.SEEDS:
        t = jobs[f"job-labelrec-ladder-v1-s{seed}-raunav.yaml"]
        assert f'--branch "{bp.PIN}"' in t and bp.PIN_V2 not in t
        assert f"OUT=/data/results/eval/label_recovery_ladder_v1/s{seed}\n" in t
        assert "--eps-s" not in t
    assert len([f for f in jobs if f.startswith("job-labelrec-")]) == 5


def test_windowed_task_is_not_requested():
    assert "bc_vs_rest" not in bp.TASKS
    assert {"bvc_4prong", "visible_content"} <= set(bp.TASKS)


def test_committed_specs_match_the_builder():
    for fname, text in bp.build().items():
        p = bp.K8S / fname
        assert p.exists(), f"run scripts/build_probe_jobs.py: {fname} missing"
        assert p.read_text() == text, f"{fname} drifted from the builder"


def test_the_random_control_pairs_each_draw_with_its_own_seed_index():
    """C4 is a WITHIN-SEED contrast: random draw minus the 17-class model at the
    same seed index. probe.check_alignment only gates arms inside one job, so a
    draw and its reference have to travel together or the contrast is across
    different test jets and nothing would say so."""
    jobs = bp.build()
    assert len(bp.CONTROL_DRAWS) == 3, (
        "three draws, not one: different random partitions merge different class "
        "pairs, so draw-to-draw variation cannot be estimated from one draw")
    for rand, ref, draw in bp.CONTROL_DRAWS:
        t = jobs[f"job-probe-randcontrol-d{draw}-raunav.yaml"]
        line = next(l for l in t.splitlines() if l.strip().startswith("for spec in"))
        assert f"{rand}:RAND" in line and f"{ref}:R16_Q1" in line
        # the draw's seed index and its reference's must agree
        assert rand.split("-s")[-1].rstrip("b") == ref.split("-s")[-1], (rand, ref)
        assert f"OUT=/data/results/eval/probe_ladder_randcontrol/sd{draw}\n" in t


def test_the_random_control_runs_only_the_two_tasks_c4_is_defined_on():
    """The other four probe tasks are not part of C4. Running them here would be
    four more chances to find something in a confirmatory control."""
    assert bp.CONTROL_TASKS == ["bvc_4prong", "visible_content"]
    for _, _, draw in bp.CONTROL_DRAWS:
        t = bp.build()[f"job-probe-randcontrol-d{draw}-raunav.yaml"]
        assert "--tasks bvc_4prong visible_content" in t
        for other in ("bvc_resonant", "retained_topology", "bvc_qcd", "ee_vs_mm"):
            assert other not in t, f"draw {draw} runs {other}, which C4 does not use"


def test_the_random_control_cannot_overwrite_the_ladder_results():
    for _, _, draw in bp.CONTROL_DRAWS:
        t = bp.build()[f"job-probe-randcontrol-d{draw}-raunav.yaml"]
        assert "probe_ladder_v1" not in t and "probe_ladder_v2" not in t
        assert f'--branch "{bp.PIN_V2}"' in t, "must run the probe code the ladder ran"


def test_the_mass_2x2_carries_all_four_corners_of_one_seed_index():
    """C5 is a difference-in-differences: (162+mass − 162) − (17+mass − 17) at one
    seed index. All four corners must be in ONE job, because probe.check_alignment
    gates only the arms inside a job, and a DiD built across two jobs could be
    comparing four models on two different orderings of the test set."""
    jobs = bp.build()
    assert [c for _, c in bp.MASS_CELLS] == ["L162", "L162_MASS", "R16_Q1", "R16_Q1_MASS"]
    for seed in bp.SEEDS:
        t = jobs[f"job-probe-mass2x2-s{seed}-raunav.yaml"]
        line = next(l for l in t.splitlines() if l.strip().startswith("for spec in"))
        runs = line.split("for spec in")[1].split(";")[0].split()
        assert [r.split(":")[1] for r in runs] == ["L162", "L162_MASS", "R16_Q1", "R16_Q1_MASS"]
        # every corner at THIS seed index, and the 162-class seed 1 is the 5e-4
        # repair -- the excluded 1e-3 run has the same seed and must never appear
        want = {f"mtx-l162-s1b" if seed == 1 else f"mtx-l162-s{seed}",
                f"mtx-l162mass-s{seed}", f"mtx-r16q1-s{seed}", f"mtx-r16q1mass-s{seed}"}
        assert {r.split(":")[0] for r in runs} == want, (seed, runs)
        assert "mtx-l162-s1:" not in line


def test_the_mass_2x2_measures_the_same_thing_as_the_ladder_rerun():
    """C5's cells and C1's cells have to be the same measurement or they could not
    sit in one multiplicity family: same tasks, same operating points, same probe
    code. Only the models and the output directory differ."""
    jobs = bp.build()
    tasks = " ".join(bp.TASKS)
    for seed in bp.SEEDS:
        t = jobs[f"job-probe-mass2x2-s{seed}-raunav.yaml"]
        v2 = jobs[f"job-probe-ladder-v2-s{seed}-raunav.yaml"]
        assert f"--tasks {tasks}" in t, "the approved plan specifies the same probes for the 2x2"
        eps = " ".join(str(e) for e in bp.EPS_S_V2)
        assert f"--eps-s {eps}" in t and f"--eps-s {eps}" in v2
        # the 90 % headline operating point has to be among them
        assert "0.9" in bp.EPS_S_V2 or 0.9 in bp.EPS_S_V2


def test_the_mass_2x2_cannot_overwrite_any_other_result():
    for seed in bp.SEEDS:
        t = bp.build()[f"job-probe-mass2x2-s{seed}-raunav.yaml"]
        assert f"OUT=/data/results/eval/probe_ladder_mass2x2/s{seed}\n" in t
        assert "probe_ladder_v1" not in t and "probe_ladder_v2" not in t
        assert "probe_ladder_randcontrol" not in t


def test_the_mass_pin_contains_the_probe_code_the_ladder_ran():
    """A spec that clones a different probe.py from the ladder it is compared
    against runs different code. The 2x2 and the ladder must carry the same
    probe.py; later changes (the converged MLP) reach both through the mlp2
    reruns, which share one pin."""
    import subprocess
    root = str(pathlib.Path(bp.__file__).resolve().parents[1])
    show = lambda tag: subprocess.run(["git", "-C", root, "show", f"{tag}:experiments/EVAL/probe.py"],
                                      capture_output=True, text=True).stdout      # noqa: E731
    assert show(bp.MASS_PIN) and show(bp.MASS_PIN) == show(bp.PIN_V2), (
        f"{bp.MASS_PIN} and {bp.PIN_V2} carry different probe.py; the 2x2 would run "
        f"different probe code from the ladder")


def test_the_mass_regression_carries_six_models_including_the_two_without_a_twin():
    """S7's third clause is about the granularity ladder WITHOUT the mass output
    -- "without the mass output, finer labels give equal or better resolution" --
    so the 188- and 43-class models are needed too, even though neither has a
    mass twin. Four models would answer only two of the three clauses."""
    jobs = bp.build()
    assert [c for _, c in bp.MASSRES_CELLS] == [
        "L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS"]
    for seed in bp.SEEDS:
        t = jobs[f"job-massres-s{seed}-raunav.yaml"]
        line = next(l for l in t.splitlines() if l.strip().startswith("for spec in"))
        runs = line.split("for spec in")[1].split(";")[0].split()
        assert len(runs) == 6
        want = {f"mtx-l188-s{seed}", "mtx-l162-s1b" if seed == 1 else f"mtx-l162-s{seed}",
                f"mtx-r42q1-s{seed}", f"mtx-r16q1-s{seed}",
                f"mtx-l162mass-s{seed}", f"mtx-r16q1mass-s{seed}"}
        assert {r.split(":")[0] for r in runs} == want, (seed, runs)
        assert "mtx-l162-s1:" not in line


def test_the_mass_regression_requires_the_generator_level_mass_cache():
    """The target lives in one place and nowhere else. A job that started without
    it would fail deep inside the probe instead of in its first ten seconds."""
    for seed in bp.SEEDS:
        t = bp.build()[f"job-massres-s{seed}-raunav.yaml"]
        assert bp.MASSRES_OBS in t
        for f in ("observers.npz", "label188.npy", "observers_manifest.json"):
            assert f in t, f
        assert "--observers" in t
        assert f"OUT=/data/results/eval/mass_resolution/s{seed}\n" in t
        # it must not be able to land on any probe result
        assert "probe_ladder" not in t


def test_the_mass_regression_pin_carries_the_script_it_runs():
    """mass_resolution.py is new, so a spec pinned at any earlier tag would clone a
    repository that does not contain it and die on the first line."""
    import subprocess
    root = str(pathlib.Path(bp.__file__).resolve().parents[1])
    ok = subprocess.run(["git", "-C", root, "cat-file", "-e",
                         f"{bp.MASSRES_PIN}:experiments/EVAL/mass_resolution.py"],
                        capture_output=True)
    assert ok.returncode == 0, (
        f"{bp.MASSRES_PIN} does not contain experiments/EVAL/mass_resolution.py")


def test_s10_job_pairs_162_and_17_at_all_five_seeds_on_the_windowed_caches():
    """S10 (docs/PRESPEC_2026-09.md, clarification of 2026-09-27): 162 against 17
    classes at seed indices 1-5, the |V_cb| task only, inside the published window.
    All ten in ONE job, because probe.check_alignment only gates arms inside a job."""
    text = bp.build()["job-probe-vcbwindow-s10-raunav.yaml"]
    line = next(l for l in text.splitlines() if l.strip().startswith("for spec in"))
    runs = line.split("for spec in")[1].split(";")[0].split()
    assert runs == ([f"mtx-l162-{'s1b' if s == 1 else f's{s}'}:L162" for s in bp.SEEDS]
                    + [f"mtx-r16q1-s{s}:R16_Q1" for s in bp.SEEDS])
    assert "mtx-l162-s1:" not in line                 # the excluded 1e-3 run
    assert "/features_vcbwindow_e79_full\n" in text and "/features_e79\n" not in text
    assert "--tasks bc_vs_rest \\\n" in text and "--eps-s 0.6 0.4 \\\n" in text
    assert f'--branch "{bp.MASS_PIN}"' in text


def test_every_other_job_still_reads_the_2m_caches():
    for fname, text in bp.build().items():
        if "vcbwindow" not in fname:
            assert "/features_e79\n" in text and "vcbwindow" not in text, fname


# ---------------------------------------------------------------------------
# The MLP re-runs (mlp2). The first runs capped the MLP at 60 epochs while fits
# were still improving; these re-fit only the MLP, on the caches each run read,
# and copy the linear numbers from that run's own output.
# ---------------------------------------------------------------------------

def _models_line(text):
    return next(l for l in text.splitlines() if l.strip().startswith("for spec in"))


def test_every_probe_set_whose_mlp_the_paper_reads_has_a_rerun():
    """The four-level ladder (5), the random-label control (3), the mass-output
    2x2 (5) and the |V_cb| probe (1); the frozen v1 and the label-recovery and
    mass-regression jobs have no MLP the paper reads and get none."""
    jobs = bp.build()
    reruns = sorted(f for f in jobs if "-mlp2-" in f)
    assert reruns == sorted(f"job-{bp.mlp2_name(n)}.yaml" for n in bp.MLP2_SOURCES)
    assert len(reruns) == 14
    for name in bp.MLP2_SOURCES:
        assert f"job-{name}.yaml" in jobs, name
    assert not any(("v1" in n) or ("labelrec" in n) or ("massres" in n) for n in bp.MLP2_SOURCES)


def test_a_rerun_is_its_source_with_only_name_pin_and_probe_call_moved():
    """Same models, same caches, same resources and region as the run whose
    linear numbers it copies. Checked as: the text before the probe call equals
    the source's with the name and pin rewritten, and the text after it is the
    source's own."""
    jobs = bp.build()
    for name in bp.MLP2_SOURCES:
        src, new = jobs[f"job-{name}.yaml"], jobs[f"job-{bp.mlp2_name(name)}.yaml"]
        pin = re.search(r'--branch "([^"]+)"', src).group(1)
        head_src, head_new = src.split("          OUT=")[0], new.split("          SRC=")[0]
        assert head_new == (head_src.replace(f"name: {name}\n", f"name: {bp.mlp2_name(name)}\n")
                            .replace(f'--branch "{pin}"', f'--branch "{bp.MLP2_PIN}"')), name
        assert src.split("--bootstrap 2000\n")[1] == new.split("--bootstrap 2000\n")[1], name
        assert _models_line(src) == _models_line(new)


def test_a_rerun_reads_its_sources_output_and_writes_beside_it():
    jobs = bp.build()
    old_outs = {l.strip() for f, t in jobs.items() if "-mlp2-" not in f
                for l in t.splitlines() if l.strip().startswith("OUT=")}
    for name in bp.MLP2_SOURCES:
        src, new = jobs[f"job-{name}.yaml"], jobs[f"job-{bp.mlp2_name(name)}.yaml"]
        src_out = re.search(r"^          OUT=(\S+)$", src, re.M).group(1)
        new_out = re.search(r"^          OUT=(\S+)$", new, re.M).group(1)
        assert f"          SRC={src_out}/probe_results.json\n" in new
        head, leaf = src_out.rsplit("/", 1)
        assert new_out == f"{head}_mlp2/{leaf}" and f"OUT={new_out}" not in old_outs
        # never over a result: the job refuses to start if its output exists
        assert '[ ! -e "${OUT}/probe_results.json" ] || {' in new
        # tasks and working points come from the source file, never the command line
        call = new.split("python3 experiments/EVAL/probe.py")[1].split("date -u")[0]
        assert "--mlp-rerun-of ${SRC}" in call and "--bootstrap 2000" in call
        assert "--tasks" not in call and "--eps-s" not in call
        assert f'--branch "{bp.MLP2_PIN}"' in new


def test_the_rerun_pin_is_refused_until_tagged_unless_declared():
    """The pod clones a TAG. mtx-s1.64 is created after the commit, so the build
    must name that explicitly, and the flag it relies on must be in the file."""
    assert bp.MLP2_PIN == "mtx-s1.64"
    with pytest.raises(SystemExit):
        bp.verify_pin("mtx-s0.0-does-not-exist", False, bp.MLP2_NEEDED)
    bp.verify_pin("mtx-s0.0-does-not-exist", True, bp.MLP2_NEEDED)
    with pytest.raises(SystemExit):
        bp.verify_pin("mtx-s0.0-does-not-exist", True,
                      {"experiments/EVAL/probe.py": "--no-such-flag"})


# ---------------------------------------------------------------------------
# v1 errors (audit 2026-09-29): refits of the committed probes with per-jet
# outputs saved, the label-recovery learning curve and the fine-tuning bootstrap.
# ---------------------------------------------------------------------------

def _v1err():
    base = bp.build()
    return base, bp.build_v1err(base)


def test_v1err_jobs_parse_carry_the_retry_policy_and_are_committed():
    base, jobs = _v1err()
    # the 9 probe reruns applied one per pod, 5 label-recovery curves, batches A, A2, B,
    # B2 and B3, and the checks of B2 (another node) and of B3 (another CPU vendor)
    assert len(jobs) == 9 + 5 + 7
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"] and fname == f"job-{d['metadata']['name']}.yaml"
        rules = d["spec"]["podFailurePolicy"]["rules"]
        assert {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In",
                                                      "values": [42]}} in rules
        assert {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]} in rules
        assert d["spec"]["template"]["spec"]["containers"][0]["name"] == "main"
        pin = (bp.V1ERR_PIN3 if "batch-a2" in fname
               else bp.V1ERR_PIN_MASS2 if "batch-b2" in fname
               else bp.MASS2_CHECK_PIN if "mass-pinned-check" in fname
               else bp.V1ERR_PIN_MASS3 if "batch-b3" in fname or "cpufixed-check" in fname
               else bp.V1ERR_PIN2 if any(k in fname for k in ("labelrec-curve", "batch-a"))
               else bp.V1ERR_PIN)
        assert f'--branch "{pin}"' in text and "|| halt" in text
        assert "/data/results/eval/v1err/" in text
        assert (bp.K8S / fname).read_text() == text, f"{fname} not committed as built"


def test_v1err_probe_reruns_are_their_sources_with_scores_saved():
    base, jobs = _v1err()
    singles = [s for s in bp.V1ERR_PROBE_SOURCES if s not in bp.BATCH_A_PROBES]
    assert len(singles) == 9
    for src in singles:
        s, r = base[f"job-{src}.yaml"], jobs[f"job-{bp.v1err_name(src)}.yaml"]
        assert _models_line(s) == _models_line(r)
        for pat in (r"--tasks .+", r"--eps-s .+", r"d=/data/results/eval/\$\{a\}/\S+"):
            assert re.findall(pat, s) == re.findall(pat, r), (src, pat)
        out = re.search(r"OUT=/data/results/eval/(\S+)", s).group(1)
        assert f"OUT=/data/results/eval/v1err/{out}\n" in r and "--save-scores" in r


def test_label_recovery_curve_one_seed_per_job_five_sizes_to_the_whole_pool():
    _, jobs = _v1err()
    curve = {k: v for k, v in jobs.items() if "labelrec-curve" in k}
    assert len(curve) == 5
    assert len(set(bp.CURVE_SIZES)) >= 5 and bp.CURVE_SIZES[-1] == 0
    for text in curve.values():
        assert len(_models_line(text).split()[3:-1]) == 4
        assert "--sizes " + " ".join(map(str, bp.CURVE_SIZES)) in text
        assert "--mlp-rungs L188" in text and "wait ${p} || halt" in text


def test_v1err_pin_needs_the_flags_it_passes():
    assert set(bp.V1ERR_NEEDED) >= {"experiments/EVAL/probe.py",
                                    "experiments/EVAL/mass_resolution.py",
                                    "experiments/STATS/paired_errors.py"}
    bp.verify_pin("mtx-s0.0-does-not-exist", True, bp.V1ERR_NEEDED)


def test_batch_a_runs_the_same_probe_reruns_as_the_single_specs():
    base, jobs = _v1err()
    a = jobs["job-v1err-batch-a-raunav.yaml"]
    d = yaml.safe_load(a)
    assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
    assert f'--branch "{bp.V1ERR_PIN2}"' in a and "class_counts.py" in a and "ft-replicates" in a
    for src in bp.BATCH_A_PROBES:
        single = bp.v1err_probe_spec(base[f"job-{src}.yaml"])
        specs = re.search(r"for spec in (.+); do", single).group(1)
        out = re.search(r"OUT=(\S+)", single).group(1)
        tasks = re.search(r"--tasks (.+) \\", single).group(1)
        eps = re.search(r"--eps-s (.+) \\", single).group(1)
        assert f'run_probe "{specs}" ' in a and f' {out} "{tasks}" "{eps}" &' in a
    assert a.count("run_probe \"") == len(bp.BATCH_A_PROBES)
    assert "for p in ${P}; do wait ${p} || halt; done" in a
    assert (bp.K8S / "job-v1err-batch-a-raunav.yaml").read_text() == a


def test_batch_b_runs_the_five_mass_reruns():
    base, jobs = _v1err()
    b = jobs["job-v1err-batch-b-raunav.yaml"]
    assert b.count('run_massres "') == 5 and "--save-residuals" in b
    for src in bp.V1ERR_MASSRES_SOURCES:
        specs = re.search(r"for spec in (.+); do", base[f"job-{src}.yaml"]).group(1)
        assert f'run_massres "{specs}" ' in b
    assert (bp.K8S / "job-v1err-batch-b-raunav.yaml").read_text() == b
    # B2 is B at the pinned-thread tag, written elsewhere; nothing else differs
    b2 = jobs["job-v1err-batch-b2-raunav.yaml"]
    assert b2.replace("v1err-batch-b2-raunav", "v1err-batch-b-raunav").replace(
        bp.V1ERR_PIN_MASS2, bp.V1ERR_PIN).replace("/mass_resolution_pinned/", "/mass_resolution/") == b
    assert b2.count("/data/results/eval/v1err/mass_resolution_pinned/s") == 5
    assert set(bp.V1ERR_MASS2_NEEDED) >= {"experiments/EVAL/latent_scale_probe.py"}
    # the check: seed 1 of B2 again, at B2's tag, kept off B2's node, guarded
    c = jobs["job-v1err-mass-pinned-check-raunav.yaml"]
    d = yaml.safe_load(c)
    specs = re.search(r"for spec in (.+); do", base["job-massres-s1-raunav.yaml"]).group(1)
    assert c.count('run_massres "') == 1 and f'run_massres "{specs}" ' in c
    assert "/data/results/eval/v1err/mass_resolution_pinned_check/s1 &" in c
    assert f'--branch "{bp.MASS2_CHECK_PIN}"' in c and "df --output=pcent /data" in c
    terms = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert {"key": "kubernetes.io/hostname", "operator": "NotIn", "values": [bp.MASS2_NODE]} in terms
    # B3: B2 at the CPU-path tag, on Intel; its check: seed 1 on AMD
    b3 = jobs["job-v1err-batch-b3-raunav.yaml"]
    assert b3.count("/data/results/eval/v1err/mass_resolution_cpufixed/s") == 5
    k3 = jobs["job-v1err-mass-cpufixed-check-raunav.yaml"]
    assert k3.count('run_massres "') == 1 and "mass_resolution_cpufixed_check/s1 &" in k3
    for text, vendor in ((b3, "Intel"), (k3, "AMD")):
        t3 = yaml.safe_load(text)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        assert {"key": bp.VENDOR, "operator": "In", "values": [vendor]} in t3
        assert f'--branch "{bp.V1ERR_PIN_MASS3}"' in text and "df --output=pcent /data" in text


def test_batch_a2_runs_the_bootstrap_resumably_beside_the_probe_reruns():
    base, jobs = _v1err()
    a2 = jobs["job-v1err-batch-a2-raunav.yaml"]
    a = jobs["job-v1err-batch-a-raunav.yaml"]
    assert f"--b {bp.FT_B} --procs 10 --cache ${{FT}}/cells" in a2 and "run_ft &" in a2
    probe_calls = lambda s: sorted(l for l in s.splitlines() if l.strip().startswith('run_probe "'))
    assert probe_calls(a2) == probe_calls(a) and len(probe_calls(a2)) == len(bp.BATCH_A_PROBES)
    assert (bp.K8S / "job-v1err-batch-a2-raunav.yaml").read_text() == a2


def test_the_v2_probe_tasks_are_every_probe_task_and_the_v2_extraction_keeps_their_rows():
    """The v2 caches must hold every row a v2 probe task reads: an unwindowed task's
    classes over the whole split (the single-pair b-vs-c tasks among them), a
    windowed task's inside its window or everywhere."""
    import importlib.util
    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    probe = load("probe", "experiments/EVAL/probe.py")
    xv = load("extract_v2", "experiments/EVAL/extract_v2.py")
    assert {"bc_vs_bq", "bc_vs_cs"} <= set(bp.V2_TASKS) and len(set(bp.V2_TASKS)) == len(bp.V2_TASKS)
    assert set(bp.V2_TASKS) == set(probe.TASKS)
    anywhere, windowed = xv.probe_feature_rules()
    for task in bp.V2_TASKS:
        spec = probe.TASKS[task]
        cls = set(spec["signal"]) | set(spec["background"])
        if not spec.get("window"):
            assert cls <= set(anywhere), task
        else:
            inside = {x for cl, w in windowed if w == dict(spec["window"]) for x in cl}
            assert cls <= set(anywhere) | inside, task


# ---------------------------------------------------------------------------
# The v2 grid (--v2): one job per run and untrained-trunk reference, every frozen
# readout of every extracted checkpoint, written beside the extraction's run
# directories (/data/results/eval/v2/<analysis>/<run>/<tag>/<readout>).
# ---------------------------------------------------------------------------

def _bx():
    return bp._load_builder("build_extract_jobs")


def _script(text):
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_v2_one_job_per_run_and_reference_with_the_retry_policy_and_the_pin():
    bx, jobs = _bx(), bp.build_v2()
    runs = [r for r, *_ in bx.v2_runs()] + [ref for ref, _ in bx.v2_init_refs()]
    assert sorted(jobs) == sorted(f"job-frozen-v2-{r.removeprefix('mtx-')}-raunav.yaml" for r in runs)
    assert len(jobs) == 93
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"] and fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["backoffLimit"] == bp.V1ERR_BACKOFF
        rules = d["spec"]["podFailurePolicy"]["rules"]
        assert {"action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In",
                                                      "values": [42]}} in rules
        assert {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]} in rules
        assert f'--branch "{bp.V2_GRID_PIN}"' in text and "df --output=pcent /data" in text
        assert "nvidia.com/gpu" not in text


def test_v2_pin_carries_what_the_jobs_run_or_is_refused_until_tagged():
    import subprocess
    assert bp.V2_GRID_PIN == "mtx-s2.00"
    tagged = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{bp.V2_GRID_PIN}"],
                            cwd=ROOT, capture_output=True).returncode == 0
    if tagged:                                                  # tagged 2026-10-08 (e635a5c)
        bp.verify_pin(bp.V2_GRID_PIN, False, bp.V2_GRID_NEEDED)
    else:
        with pytest.raises(SystemExit):
            bp.verify_pin(bp.V2_GRID_PIN, False, bp.V2_GRID_NEEDED)
    bp.verify_pin(bp.V2_GRID_PIN, True, bp.V2_GRID_NEEDED)     # the working tree has every flag


def test_v2_every_checkpoint_and_readout_the_prespec_names():
    """best70, wavg, bestval and both BatchNorm twins for every run (A8, A14); the
    class token and the pooled embedding for every model, the pooled one only for
    the self-supervised runs (section 4); init for the untrained-trunk references."""
    bx, jobs = _bx(), bp.build_v2()
    for run, arm, k, _, _ in bx.v2_runs() + [(ref, None, None, 0, 0) for ref, _ in bx.v2_init_refs()]:
        s = _script(jobs[f"job-frozen-v2-{run.removeprefix('mtx-')}-raunav.yaml"])
        tags = re.search(r"^for t in (.+); do$", s, re.M).group(1).split()
        readouts = re.search(r'READOUTS="([^"]+)"', s).group(1).split()
        if arm is None:
            assert tags == ["init"] and readouts == ["features", "pooled"]
        else:
            assert tags == list(bx.V2_CHECKPOINTS) and set(tags) >= {"best70", "wavg", "bestval",
                                                                     "best70_bn", "bestval_bn"}
            assert readouts == (["pooled"] if k == 0 else ["features", "pooled"]), run
        files = re.search(r"^  for f in (.+); do$", s, re.M).group(1).split()
        assert set(files) == {"label188.npy", "manifest.json", "rows.npy", "observers.npz"} | {
            f"{r}.npy" for r in readouts}
        assert f"CACHE={bx.V2_OUT}/{run};" in s and f"RUN={run};" in s


def test_v2_runs_every_task_at_the_v2_split_and_keeps_the_per_jet_outputs():
    bx = _bx()
    s = _script(bp.build_v2(only=["mtx-r16q1-s2"])["job-frozen-v2-r16q1-s2-raunav.yaml"])
    assert f"--tasks {' '.join(bp.V2_TASKS)} " in s and "--eps-s 0.5 0.7 0.9 " in s
    assert f"--split-fractions {' '.join(map(str, bx.V2_SPLIT_FRACTIONS))} " in s
    assert "--save-scores" in s and "--save-residuals" in s
    assert f"--sizes {' '.join(map(str, bp.CURVE_SIZES))} --mlp-rungs L188" in s
    assert "--own-rung r16q1-s2@$1=R16_Q1 " in s and "--features r16q1-s2@$1=${CACHE}/$1" in s
    assert "--observers ${CACHE}/$1" in s
    # probes first (the headline), the slowest last; one arm per call
    assert s.index("each do_probe") < s.index("each do_mass") < s.index("each do_curve")
    assert s.count("--features ") == 3 and "--no-mlp" not in s


def test_v2_outputs_sit_beside_the_runs_never_on_one_and_disjoint_per_run(monkeypatch, capsys):
    """<root>/<analysis>/<run>/<tag>/<readout>: an analysis directory is never a run's
    name (paired_errors.py indexes <root>/<run>/<tag>/manifest.json), and the specs go
    to experiments/EVAL/k8s/v2/frozen/, as the anomaly specs go to .../v2/anomaly/."""
    import sys
    bx, jobs = _bx(), bp.build_v2()
    runs = [re.search(r"^RUN=(\S+);", _script(t), re.M).group(1) for t in jobs.values()]
    assert len(runs) == len(set(runs)) and not set(bp.V2_GRID_ANALYSES) & set(runs)
    s = _script(jobs["job-frozen-v2-l188-s1-raunav.yaml"])
    assert f"OUT={bx.V2_OUT};" in s
    for a in bp.V2_GRID_ANALYSES:
        assert f"local o=${{OUT}}/{a}/${{RUN}}/$1/$2" in s
    monkeypatch.setattr(sys, "argv", ["x", "--v2", "--pin-not-yet-tagged", "--check", "--only", "init-s1"])
    assert bp.main() == 1          # not written: --check names where it would go
    assert "DRIFT: experiments/EVAL/k8s/v2/frozen/job-frozen-v2-init-s1-raunav.yaml" in capsys.readouterr().out


def test_v2_resources_follow_the_calls_run_side_by_side():
    for fname, text in bp.build_v2().items():
        s = _script(text)
        par = int(re.search(r"(\d+) calls at a time", s).group(1))
        n = len(re.search(r"^for t in (.+); do$", s, re.M).group(1).split()) * len(
            re.search(r'READOUTS="([^"]+)"', s).group(1).split())
        assert par == min(bp.V2_GRID_PAR, n), fname
        lim = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        assert lim == {"memory": f"{16 * par}Gi", "cpu": str(4 * par), "ephemeral-storage": "10Gi"}
        assert "export OMP_NUM_THREADS=4 " in s and "--threads 4" in s


def test_v2_filters_by_tier_and_run():
    bx = _bx()
    tiers = {a["name"]: a["tier"] for a in __import__("json").loads(bx.V2_GRID.read_text())["arms"]}
    want = [r for r, arm, *_ in bx.v2_runs() if tiers[arm] == 1] + [ref for ref, _ in bx.v2_init_refs()]
    assert sorted(bp.build_v2([1])) == sorted(f"job-frozen-v2-{r.removeprefix('mtx-')}-raunav.yaml"
                                              for r in want)
    assert sum(len(bp.build_v2([t])) for t in (1, 2, 3)) == len(bp.build_v2())
    assert sorted(bp.build_v2(only=["mtx-l188-s1", "init-s1"])) == [
        "job-frozen-v2-init-s1-raunav.yaml", "job-frozen-v2-l188-s1-raunav.yaml"]
    assert bp.build_v2([2], only=["mtx-l188-s1"]) == {}
    with pytest.raises(SystemExit):
        bp.build_v2(only=["mtx-l188-s9"])


STUB = '''#!{python}
import json, os, pathlib, sys
a = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps(a) + "\\n")
if os.environ.get("FAIL") and os.environ["FAIL"] in " ".join(a):
    sys.exit(int(os.environ["FAIL_RC"]))
out = pathlib.Path(a[a.index("--out") + 1])
for name in {{"probe.py": ["probe_results.json", "scores.npz"],
             "mass_resolution.py": ["mass_resolution.json", "residuals.npz"],
             "label_recovery_curve.py": ["label_recovery_curve.json"]}}[pathlib.Path(a[0]).name]:
    (out / name).write_text("x")
'''


def _run_v2_body(tmp, run, tags, links, env=None, missing=None):
    """The job's script from RUN= on, its paths moved under tmp, against a stub
    python3 that records each call and writes the outputs the script checks for."""
    import json
    import os
    import subprocess
    import sys
    bx = _bx()
    s = _script(bp.build_v2(only=[run])[f"job-frozen-v2-{run.removeprefix('mtx-')}-raunav.yaml"])
    body = s[s.index("RUN="):].replace(bx.V2_OUT, str(tmp / "v2"))
    halt = s[s.index("halt () "):].splitlines()[0]
    cache = tmp / "v2" / run
    files = re.search(r"^  for f in (.+); do$", s, re.M).group(1).split()
    for t in tags:
        (cache / t).mkdir(parents=True, exist_ok=True)
        for f in files:
            if (t, f) != missing:
                (cache / t / f).write_text("x")
    for t, target in links.items():
        if not (cache / t).is_symlink():
            (cache / t).symlink_to(target, target_is_directory=True)
    stub = tmp / "bin" / "python3"
    stub.parent.mkdir(exist_ok=True)
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(0o755)
    calls = tmp / "calls.jsonl"
    calls.write_text("")
    r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + halt + "\n" + body],
                       env={**os.environ, "PATH": f"{stub.parent}:{os.environ['PATH']}",
                            "CALLS": str(calls), **(env or {})},
                       capture_output=True, text=True)
    return r, [json.loads(l) for l in calls.read_text().splitlines()]


def test_v2_script_fits_each_checkpoint_once_links_aliases_and_resumes(tmp_path):
    """Run the generated shell: bestval and bestval_bn are links (the same epochs as
    best70 and best70_bn), so 3 tags x 2 readouts x 3 analyses are fitted, and each
    link reappears in every analysis. A second attempt redoes only the curve,
    which resumes itself."""
    run = "mtx-l188-s1"
    links = {"bestval": "best70", "bestval_bn": "best70_bn"}
    r, calls = _run_v2_body(tmp_path, run, ["best70", "wavg", "best70_bn"], links)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls) == 18
    got = {(pathlib.Path(c[0]).name, c[c.index("--readout") + 1],
            c[c.index("--features") + 1].split("=")[0]) for c in calls}
    assert got == {(s, ro, f"l188-s1@{t}") for s in ("probe.py", "mass_resolution.py",
                                                    "label_recovery_curve.py")
                   for ro in ("features", "pooled") for t in ("best70", "wavg", "best70_bn")}
    analysis = {"probe.py": "probe", "mass_resolution.py": "mass_resolution",
                "label_recovery_curve.py": "label_recovery_curve"}
    for c in calls:
        tag = c[c.index("--features") + 1].split("@")[1].split("=")[0]
        assert c[c.index("--features") + 1] == f"l188-s1@{tag}={tmp_path}/v2/{run}/{tag}"
        assert c[c.index("--out") + 1] == (f"{tmp_path}/v2/{analysis[pathlib.Path(c[0]).name]}/"
                                           f"{run}/{tag}/{c[c.index('--readout') + 1]}")
    for a in bp.V2_GRID_ANALYSES:
        for t, target in links.items():
            p = tmp_path / "v2" / a / run / t
            assert p.is_symlink() and p.resolve() == (tmp_path / "v2" / a / run / target).resolve()
    r, calls = _run_v2_body(tmp_path, run, ["best70", "wavg", "best70_bn"], links)
    assert r.returncode == 0 and len(calls) == 6
    assert {pathlib.Path(c[0]).name for c in calls} == {"label_recovery_curve.py"}


def test_v2_script_reads_the_self_supervised_runs_pooled_embedding_only(tmp_path):
    r, calls = _run_v2_body(tmp_path, "mtx-mpm-s1", list(_bx().V2_CHECKPOINTS), {})
    assert r.returncode == 0, r.stderr
    assert len(calls) == 15 and {c[c.index("--readout") + 1] for c in calls} == {"pooled"}


def test_v2_script_refuses_an_incomplete_cache_and_halts_on_a_failed_call(tmp_path):
    r, calls = _run_v2_body(tmp_path / "a", "init-s2", ["init"], {}, missing=("init", "pooled.npy"))
    assert r.returncode == 42 and not calls and "FATAL: no" in r.stdout
    r, _ = _run_v2_body(tmp_path / "b", "init-s2", ["init"], {},
                        env={"FAIL": "mass_resolution.py", "FAIL_RC": "1"})
    assert r.returncode == 42 and "HALT: exit 1" in r.stdout          # deterministic: not retried
    r, calls = _run_v2_body(tmp_path / "c", "init-s2", ["init"], {},
                            env={"FAIL": "label_recovery_curve.py", "FAIL_RC": "137"})
    assert r.returncode == 137                                          # a signal: retried
    assert {pathlib.Path(c[0]).name for c in calls} == {"probe.py", "mass_resolution.py",
                                                       "label_recovery_curve.py"}
