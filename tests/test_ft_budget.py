"""The fixed-budget fine-tuning (plan 3.1): early stopping and the cell builder."""
import importlib.util
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


es = _load("early_stop", "experiments/FT/early_stop.py")
bfb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")


def _log(metrics, current):
    lines = []
    for e, v in metrics.items():
        lines.append(f"[INFO] Epoch #{e}: exact validation metric {v!r}\n")
        if e in current:
            lines.append(f"[INFO] Epoch #{e}: Current validation metric: {v:.5f} (best: 0.9)\n")
    return lines


def test_scan_reads_exact_values_and_current_lines():
    m, c = es.scan(_log({0: 0.81, 1: 0.8234567891}, {0}))
    assert m == {0: 0.81, 1: 0.8234567891} and c == {0}


def test_best_epoch_is_weavers_strictly_greater_rule():
    assert es.best_epoch({0: 0.8, 1: 0.9, 2: 0.9, 3: 0.85}) == (1, 0.9)   # a tie keeps the earlier


def test_stop_only_after_patience_epochs_without_improvement():
    metrics = {e: 0.9 if e == 3 else 0.8 for e in range(14)}
    stop, last, be = es.decide(metrics, set(metrics), 10)
    assert (stop, last, be) == (True, 13, 3)
    stop, last, be = es.decide({e: metrics[e] for e in range(13)}, set(range(13)), 10)
    assert not stop and be == 3


def test_an_epoch_counts_only_once_its_current_line_is_logged():
    metrics = {e: 0.9 if e == 3 else 0.8 for e in range(14)}
    stop, last, _ = es.decide(metrics, set(range(13)), 10)       # epoch 13 not complete yet
    assert not stop and last == 12


def test_head_multiplier_is_capped_at_5e3():
    assert float(bfb.head_mult("1e-4")) == 50.0
    assert float(bfb.head_mult("3e-4")) * 3e-4 == pytest.approx(5e-3)
    assert float(bfb.head_mult("1e-3")) == pytest.approx(5.0)


def test_sweep_cells():
    cells = bfb.lrsweep_cells()
    assert len(cells) == 12 and {c["n"] for c in cells} == {10_000} and {c["s"] for c in cells} == {1}
    pre = [c for c in cells if c["init"]]
    assert {c["init"] for c in pre} == {"l188-s1"} and sorted({c["lr"] for c in pre}) == sorted(bfb.SWEEP_PRE)
    assert sorted({c["lr"] for c in cells if not c["init"]}) == sorted(bfb.SWEEP_SCRATCH)
    assert len({c["out"] for c in cells}) == 12


def test_bench_and_jc1_cells():
    runs = bfb.paper_runs()
    assert len(runs) == 27
    bench = bfb.bench_cells("3e-4", "1e-3")
    assert len(bench) == 2 * (27 + 3) * 3
    assert {c["n"] for c in bench} == {1_000, 10_000, 100_000}           # never the full set
    assert {c["s"] for c in bench if c["init"]} == {1}
    assert {c["s"] for c in bench if not c["init"]} == {1, 2, 3}
    jc1 = bfb.bench_cells("3e-4", "1e-3", ("jc1",))
    assert len(jc1) == (27 + 3) * 2 and {c["n"] for c in jc1} == {10_000, 100_000}   # 10^3 dropped
    assert len({c["out"] for c in bench + jc1}) == len(bench) + len(jc1)


@pytest.mark.parametrize("kind", ["lrsweep", "bench", "jc1", "jc2b", "kin"])
def test_spec_is_valid_bash_on_v100(kind, tmp_path, monkeypatch):
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    bfb.main(["--tag", "mtx-test", "--lr-pre", "3e-4", "--lr-scratch", "1e-3", kind])
    text = (tmp_path / f"job-ft-budget-{kind}-raunav.yaml").read_text()
    d = yaml.safe_load(text)
    spec = d["spec"]
    assert spec["completionMode"] == "Indexed" and "suspend" not in spec
    script = spec["template"]["spec"]["containers"][0]["args"][0]
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    gpus = spec["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"][0]["matchExpressions"][1]["values"]
    assert gpus == ["Tesla-V100-SXM2-32GB"]
    assert "--samples-per-epoch ${SPE}" in script and "--num-epochs ${EP}" in script
    assert "--samples-per-epoch-val ${VAL}" in script and "early_stop.py watch" in script
    assert ("EP=40; SPE=256000; PAT=10;" in script) or kind == "kin"
    assert "train_N${N}_s1.parquet" in script and "--model-prefix ${RUN}/net" in script


def test_jc2b_and_kinematic_cells():
    jc2b = bfb.jc2b_cells("3e-4", "1e-3")
    assert len(jc2b) == (16 + 3) * 3 and {c["n"] for c in jc2b} == {10_000, 100_000, 1_000_000}
    assert {c["val"] for c in jc2b} == {20_480} and all(c["epochs"] == 40 for c in jc2b)
    kin = bfb.kin_cells("3e-4")
    full = [c for c in kin if c["n"] == 1_200_000]
    assert len(kin) == 8 and len(full) == 2
    assert all(c["epochs"] == 20 and c["spe"] == 1_200_000 and c["patience"] == 1000 and c["lr"] == "1e-4"
               for c in full)
    assert {c["init"] for c in kin} == set(bfb.KIN_RUNS)
