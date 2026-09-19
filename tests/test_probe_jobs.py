"""The four-granularity probe jobs: one per seed index, four models each."""
import importlib.util
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_probe_jobs", ROOT / "scripts" / "build_probe_jobs.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def test_twenty_three_jobs_all_named_raunav_and_parse():
    jobs = bp.build()
    # 5 probe v1 + 5 label-recovery v1 + 5 probe v2 + 3 random-label control
    # + 5 mass-output 2x2
    assert len(jobs) == 23
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"]
        assert fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["backoffLimit"] == 1


def test_each_job_holds_one_seed_index_at_all_four_granularities():
    for fname, text in bp.build().items():
        if "randcontrol" in fname or "mass2x2" in fname:
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
    assert len(outs) == len(set(outs)) == 23
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
    """A spec that clones a tag predating probe.py's current commit runs different
    code from the ladder it is compared against. test_spec_pins enforces this in
    general; C5 is the case where it would silently change a confirmatory result."""
    import subprocess
    root = str(pathlib.Path(bp.__file__).resolve().parents[1])
    head = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H",
                           "--", "experiments/EVAL/probe.py"],
                          capture_output=True, text=True).stdout.strip()
    ok = subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", head, bp.MASS_PIN],
                        capture_output=True)
    assert ok.returncode == 0, (
        f"{bp.MASS_PIN} does not contain probe.py at {head[:12]}; the 2x2 would run "
        f"different probe code from the ladder")
