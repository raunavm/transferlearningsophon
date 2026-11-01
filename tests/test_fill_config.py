"""experiments/FT/fill_config.py: the mean-fill fine-tuning config, and the fill jobs (2026-11-01)."""
import importlib.util
import pathlib
import subprocess

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


fc = _load("fill_config", "experiments/FT/fill_config.py")
MEAN = np.array([-0.35, -0.38, -0.22, -0.22, 0.24, 0.0, 0.478, 0.107, 0.398, 0.01, 0.007, 0.0, 0.015,
                 0.0, 0.033, 0.002, 0.0])                     # mtx-l188-s1's input BatchNorm, rounded


@pytest.mark.parametrize("cfg,n", [("configs/finetune/TopReference.yaml", 10), ("configs/finetune/EnergyFlowQG.yaml", 4)])
def test_every_zero_slot_gets_its_batchnorm_mean_and_nothing_else_changes(cfg, n):
    src = (ROOT / cfg).read_text()
    out, values = fc.fill(src, MEAN)
    assert len(values) == n
    names = [e[0] for e in fc.pf_feature_entries(src)]
    for name, v in values.items():
        assert v == MEAN[names.index(name)]
        assert f"{name}: ak.zeros_like(part_energy) + {v!r}" in out
    changed = [(a, b) for a, b in zip(src.splitlines(), out.splitlines()) if a != b]
    assert len(changed) == n and len(src.splitlines()) == len(out.splitlines())
    assert yaml.safe_load(out)["inputs"] == yaml.safe_load(src)["inputs"]


def test_a_mean_outside_a_clip_range_is_refused():
    bad = MEAN.copy()
    bad[12] = 1.5                                             # d0err is clipped to [0, 1]
    with pytest.raises(SystemExit):
        fc.fill((ROOT / "configs/finetune/TopReference.yaml").read_text(), bad)
    with pytest.raises(SystemExit):
        fc.fill((ROOT / "configs/finetune/TopReference.yaml").read_text(), MEAN[:16])


def test_fill_variants_agree_between_the_job_and_the_probe():
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    pb = _load("probe_bal", "experiments/EVAL/probe_bal.py")
    assert {k: v.split() for k, v in bj.FILL_VARIANTS.items()} == {k: list(v) for k, v in pb.FILLS.items()}
    assert str(pb.FILL_ROOT) == bj.FILL_ROOT


def test_fill_extraction_reads_the_downstream_probes_files_for_every_model():
    bj = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
    lpj = bj._probe_builder()
    _, text = bj.xfill_spec("t", 5)
    job = yaml.safe_load(text)
    script = job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert job["spec"]["completions"] == len(bj.models()) == 30
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    for ds in ("top", "qg"):
        sizes, cfg, extra = lpj.DATASETS[ds]
        for n in sizes:
            assert f'extract {ds} train_N{n} {cfg} "{bj.FILL_VARIANTS[ds]}" "{lpj.TRAIN_DIR[ds]}/train_N{n}_s1.parquet"' in script
        for split, (_m, files) in extra.items():
            assert f'extract {ds} {split} {cfg} "{bj.FILL_VARIANTS[ds]}" "{files}"' in script
    assert f"--impute-ref {bj.REF2}" in script and "--no-pooled" in script
    _, fit = bj.fitfill_spec("t", 5)
    s = yaml.safe_load(fit)["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "--fill" in s and "fits_v4_fill" in s and subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0


def test_fine_tuning_fill_check_is_parts_recipe_zero_against_mean():
    bfb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    cells = bfb.fillcheck_cells()
    assert [c["task"] for c in cells] == ["top", "topmean"]
    for c in cells:
        assert (c["init"], c["n"], c["epochs"], c["spe"], c["patience"], c["lr"]) == \
            ("l188-s1", 1_200_000, 20, 1_200_000, 1000, "1e-4") and float(c["mult"]) == 50.0
    assert cells[0]["out"].endswith("leg_top/l188-s1_zero/N1200000/s1")
    assert cells[1]["out"].endswith("leg_top/l188-s1_mean/N1200000/s1")
    text = bfb.spec("ft-budget-fillcheck-raunav", cells, "t", 2, "48Gi")
    script = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    assert f"topmean) SUB=/data/finetune/top_sub; CFG={bfb.MEANFILL_TOP};" in script
    assert script.index("experiments/FT/fill_config.py") < script.index("experiments/FT/ft_weaver.py")
