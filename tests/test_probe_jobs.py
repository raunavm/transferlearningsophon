"""The four-granularity probe jobs: one per seed index, four models each."""
import importlib.util
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_probe_jobs", ROOT / "scripts" / "build_probe_jobs.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def test_ten_jobs_all_named_raunav_and_parse():
    jobs = bp.build()
    assert len(jobs) == 10
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"]
        assert fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["backoffLimit"] == 1


def test_each_job_holds_one_seed_index_at_all_four_granularities():
    for fname, text in bp.build().items():
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
    assert len(outs) == len(set(outs)) == 10
    for text in bp.build().values():
        assert "--no-mlp" not in text and "--skip-mlp" not in text


def test_windowed_task_is_not_requested():
    assert "bc_vs_rest" not in bp.TASKS
    assert {"bvc_4prong", "visible_content"} <= set(bp.TASKS)


def test_committed_specs_match_the_builder():
    for fname, text in bp.build().items():
        p = bp.K8S / fname
        assert p.exists(), f"run scripts/build_probe_jobs.py: {fname} missing"
        assert p.read_text() == text, f"{fname} drifted from the builder"
