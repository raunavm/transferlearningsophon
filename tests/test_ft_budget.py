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
    assert all(c["epochs"] == 400 and c["spe"] == 50 * 512 and c["patience"] == 10 for c in cells)
    assert all(c["out"].startswith("/data/results/ft_v2/lrsweep_v2/") for c in cells)   # the first sweep is kept
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
    # quark/gluon's own sweep may choose another from-scratch rate; top and the pretrained cells keep theirs
    per = bfb.bench_cells("1e-3", "1e-3", lr_scratch_task={"qg": "3e-3"})
    assert {c["lr"] for c in per if not c["init"] and c["task"] == "qg"} == {"3e-3"}
    assert {c["lr"] for c in per if not c["init"] and c["task"] == "top"} == {"1e-3"}
    assert {c["lr"] for c in per if c["init"]} == {"1e-3"}


@pytest.mark.parametrize("kind", ["lrsweep", "bench", "jc1", "jc2b", "kin", "fillcheck"])
def test_spec_is_valid_bash_on_any_24gb_gpu(kind, tmp_path, monkeypatch):
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
    assert gpus == list(bfb._launch().V2_FT_GPU) and "Tesla-V100-SXM2-32GB" in gpus   # any GPU that fits (PI, 2026-11-01)
    assert "k8s-haosu-10.sdsc.optiputer.net" in text                      # faulty since 2026-11-02
    assert "--samples-per-epoch ${SPE}" in script and "--num-epochs ${EP}" in script
    assert "--samples-per-epoch-val ${VAL}" in script and "early_stop.py watch" in script
    # 500-step epochs at 1e5 and above, 50-step epochs below; both at most 20,000 steps
    # a job with a cell on a whole training set gets the old full-size memory, the rest their own
    mem = spec["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"]
    assert mem == {"fillcheck": "88Gi", "kin": "88Gi", "jc2b": "88Gi", "jc1": "64Gi"}.get(kind, "48Gi")
    if kind == "fillcheck":
        return
    assert "EP=400; SPE=25600; PAT=10;" in script
    assert ("EP=40; SPE=256000; PAT=10;" in script) == (kind != "lrsweep")
    assert "train_N${N}_s1.parquet" in script and "--model-prefix ${RUN}/net" in script
    # the manifest hashes the resolved checkpoint file, never the run directory (lrsweep, 2026-10-31)
    assert "checkpoint=${RUN_DIR:+/workspace/ckpt/${INIT}.pt}" in script and "checkpoint=${RUN_DIR} " not in script
    assert script.index("${CELL}/DONE ] ||") < script.index("mv ${CELL} ${CELL}.partial")


def test_jc2b_and_kinematic_cells():
    jc2b = bfb.jc2b_cells("3e-4", "1e-3")
    assert len(jc2b) == (16 + 3) * 3 and {c["n"] for c in jc2b} == {10_000, 100_000, 1_000_000}
    assert {c["val"] for c in jc2b} == {20_480}
    assert all((c["epochs"], c["spe"]) == ((400, 25_600) if c["n"] < 100_000 else (40, 256_000)) for c in jc2b)
    assert all(c["epochs"] * c["spe"] == 20_000 * 512 for c in jc2b)
    kin = bfb.kin_cells("3e-4")
    full = [c for c in kin if c["n"] == 1_200_000]
    assert bfb.KIN_RUNS == ("l188kin-s1",) and len(kin) == 4 and len(full) == 1
    assert all(c["epochs"] == 20 and c["spe"] == 1_200_000 and c["patience"] == 1000 and c["lr"] == "1e-4"
               for c in full)
    assert {c["init"] for c in kin} == set(bfb.KIN_RUNS)


def test_a_retry_job_holds_the_same_cells_and_a_failed_clone_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    bfb.main(["--tag", "t", "--lr-pre", "3e-4", "--lr-scratch", "1e-3", "--indices", "13,26", "bench"])
    d = yaml.safe_load((tmp_path / "job-ft-budget-bench-retry-raunav.yaml").read_text())
    assert d["metadata"]["name"] == "ft-budget-bench-retry-raunav" and d["spec"]["completions"] == 2
    script = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    cells = bfb.bench_cells("3e-4", "1e-3", ("top", "qg"))
    assert f"CELL={cells[13]['out']};;" in script and f"CELL={cells[26]['out']};;" in script
    assert cells[0]["out"] not in script
    # a host that cannot be resolved must not use up the index's attempts: 5 tries, a minute apart
    assert 'retry git clone' in script and "for T in 1 2 3 4 5" in script and "sleep 60" in script
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (fake / "sleep").chmod(0o755)
    head = script.split("retry git clone")[0]
    probe = head + 'N=0; f () { N=$((N+1)); [ "$N" -ge 3 ]; }; retry f; echo "calls=$N"\n'
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                         env={"PATH": f"{fake}:/usr/bin:/bin"})
    assert out.returncode == 0 and "calls=3" in out.stdout
