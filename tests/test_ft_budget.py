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


def test_patience_can_start_when_the_learning_rate_starts_to_decay():
    metrics = {e: 0.5 for e in range(20)}       # best at epoch 0, nothing better since
    assert es.decide(metrics, set(metrics), 10)[0] is True
    assert es.decide(metrics, set(metrics), 10, 12)[0] is False
    metrics[22] = 0.4
    assert es.decide(metrics, set(metrics), 10, 12)[0] is True


def test_long_cells_are_the_runs_at_the_limit_with_four_times_the_budget():
    jc2 = bfb.long_cells("jc2b", "1e-3", "1e-3")
    jc1 = bfb.long_cells("jc1", "1e-3", "1e-3")
    assert sorted(c["init"] for c in jc2) == ["", "", "", "r16q1-s1", "r16q1mass-s1", "r16q1mass-s2"]
    assert len(jc1) == len(bfb.paper_runs()) + 3
    for c in jc2 + jc1:
        assert c["n"] == 1_000_000 and c["epochs"] == 160 and c["spe"] == 256_000 and c["patience"] == 10
        assert c["out"].startswith("/data/results/ft_v2/budget_long/")
        assert c.get("patience_from", 0) == (0 if c["init"] else 112)


def test_from_scratch_full_schedule_cells_run_all_160_epochs_into_their_own_tree():
    cells = bfb.long_scratch_full_cells("jc2b", "1e-3") + bfb.long_scratch_full_cells("jc1", "1e-3")
    assert len(cells) == 6 and all(c["init"] == "" and c["n"] == 1_000_000 for c in cells)
    for c in cells:
        assert c["epochs"] == 160 and c["patience_from"] == 160       # no early stop before the last epoch
        assert not es.decide({e: 1.0 if e == 0 else 0.5 for e in range(159)}, set(range(159)), c["patience"],
                             c["patience_from"])[0]
        assert c["out"].startswith("/data/results/ft_v2/budget_long_fulldecay/")


def test_from_scratch_full_schedule_cells_below_1e6_cover_every_task_and_size():
    cells = bfb.scratch_full_cells("1e-3", "3e-3")
    assert sorted({(c["task"], c["n"]) for c in cells}) == sorted(
        [(t, n) for t in ("top", "qg") for n in (1_000, 10_000, 100_000)] + [(t, n) for t in ("jc1", "jc2b") for n in (10_000, 100_000)])
    assert len(cells) == 30 and all(c["init"] == "" and c["patience_from"] == c["epochs"] for c in cells)
    assert {c["lr"] for c in cells if c["task"] == "qg"} == {"3e-3"} and {c["lr"] for c in cells if c["task"] != "qg"} == {"1e-3"}
    assert all(c["epochs"] * c["spe"] // 512 == 20_000 for c in cells)        # the same budget as the cells they replace
    assert all(c["out"].startswith("/data/results/ft_v2/budget_fulldecay/") for c in cells)


def test_pretrained_full_schedule_cells_are_the_old_cells_run_to_their_last_epoch():
    """The 188- and 17-class cells below 1e5 (PI 2026-11-08): the same cell as the one launched, every field but its
    tree and patience_from, which runs it to the last epoch with the best validation epoch, as from scratch."""
    cells = bfb.pre_full_cells("1e-3")
    assert len(cells) == 36 and (cells[0]["task"], cells[0]["n"]) == ("top", 1000)
    assert sorted({(c["task"], c["n"]) for c in cells}) == sorted(
        [(t, n) for t in ("top", "qg") for n in (1_000, 10_000)] + [("jc1", 10_000), ("jc2b", 10_000)])
    old = {(c["task"], c["init"], c["n"]): c for c in bfb.bench_cells("1e-3", "1e-3", ("top", "qg", "jc1"))
           + bfb.jc2b_cells("1e-3", "1e-3") if c["init"]}
    for c in cells:
        assert c["init"].split("-")[0] in ("l188", "r16q1") and c["patience_from"] == c["epochs"] == 400
        o = old[(c["task"], c["init"], c["n"])]
        assert {k: v for k, v in c.items() if k not in ("out", "patience_from")} == {k: v for k, v in o.items() if k != "out"}
        assert c["out"] == o["out"].replace("/ft_v2/budget/", "/ft_v2/budget_prefull/")
        assert not es.decide({e: 1.0 if e == 0 else 0.5 for e in range(399)}, set(range(399)), c["patience"],
                             c["patience_from"])[0]


def test_long_control_cells_are_the_eleven_partition_runs_on_jetclass2_with_the_long_budget():
    ctrl = bfb.long_ctrl_cells("1e-3")
    assert sorted(c["init"] for c in ctrl) == sorted(r for r in bfb.paper_runs() if r.startswith(("rand", "flav")))
    assert len(ctrl) == 11
    for c in ctrl:
        assert (c["task"], c["n"], c["epochs"], c["spe"], c["patience"]) == ("jc2b", 1_000_000, 160, 256_000, 10)
        assert c["out"] == f"/data/results/ft_v2/budget_long/leg_jc2b/{c['init']}/N1000000/s1"
        assert c["lr"] == "1e-3" and c["mult"] == "5.0" and "patience_from" not in c


def test_a_job_can_request_three_cpus_under_its_own_name(tmp_path, monkeypatch):
    """PI 2026-11-06: the unstarted long-budget JetClass cells move to a job requesting 3 CPUs."""
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    bfb.main(["--tag", "mtx-s2.99", "--lr-pre", "1e-3", "--lr-scratch", "1e-3", "--cpu", "3",
              "--name-suffix", "cpu3", "--indices", "0,5", "long-jc1"])
    t = (tmp_path / "job-ft-budget-long-jc1-retry-cpu3-raunav.yaml").read_text()
    assert "name: ft-budget-long-jc1-retry-cpu3-raunav" in t and t.count('cpu: "3"') == 2 and 'cpu: "4"' not in t
    assert "completions: 2" in t


def test_memory_can_be_overridden_and_defaults_to_the_rule(tmp_path, monkeypatch):
    """2026-11-07: the long JetClass cells peaked at 53 GiB, so the waiting ones move to 64Gi; without
    --mem the 1e6 rule (FULL_MEM) still applies."""
    monkeypatch.setattr(bfb, "OUT_DIR", tmp_path)
    common = ["--tag", "mtx-s2.99", "--lr-pre", "1e-3", "--lr-scratch", "1e-3", "--indices", "0,5"]
    bfb.main(common + ["--mem", "64Gi", "--name-suffix", "mem64", "long-jc1"])
    t = (tmp_path / "job-ft-budget-long-jc1-retry-mem64-raunav.yaml").read_text()
    assert t.count('memory: "64Gi"') == 2 and bfb.FULL_MEM not in t
    bfb.main(common + ["long-jc1"])
    assert (tmp_path / "job-ft-budget-long-jc1-retry-raunav.yaml").read_text().count(f'memory: "{bfb.FULL_MEM}"') == 2


def test_the_full_schedule_scratch_readout_waits_for_all_36_cells_and_never_overwrites():
    """job-budget-readout-e (2026-11-07): budget_metrics.py on the two new roots at the tag the cells trained with,
    only once every cell of scratch_full_cells and long_scratch_full_cells has DONE; a read-out short of a cell is
    refused, one already written is skipped, and both land where pull_mlst_data.py takes them."""
    import re
    k8s = ROOT / "experiments/FT/k8s/budget"
    doc = yaml.safe_load((k8s / "job-budget-readout-e-raunav.yaml").read_text())
    assert doc["metadata"]["name"] == "budget-readout-e-raunav" and doc["spec"]["backoffLimit"] == 1
    script = doc["spec"]["template"]["spec"]["containers"][0]["args"][0]
    trained = {re.search(r'name: REPO_REF\n\s+value: "([^"]+)"', (k8s / f"job-ft-budget-{k}-raunav.yaml").read_text()).group(1)
               for k in ("scratch-full", "long-scratch-full")}
    assert re.findall(r'--branch "([^"]+)"', script) == [max(trained, key=lambda t: [int(x) for x in t[5:].split(".")])]
    loop = re.search(r"for c in (.*?); do\n", script).group(1)
    cells = subprocess.run(["bash", "-c", f"for c in {loop}; do echo $c; done"], capture_output=True, text=True,
                           check=True).stdout.split()
    want = bfb.scratch_full_cells("1e-3", "3e-3") + bfb.long_scratch_full_cells("jc2b", "1e-3") + \
        bfb.long_scratch_full_cells("jc1", "1e-3")
    assert len(cells) == 36 and sorted(cells) == sorted(c["out"] for c in want)
    assert '[ -f "${c}/DONE" ] || { echo "FATAL: ${c} not DONE"; exit 42; }' in script
    calls = [ln.split()[1:] for ln in script.splitlines() if ln.strip().startswith("readout /")]
    assert calls == [[bfb.BUDGET_FULL_ROOT, f"{bfb.BUDGET_FULL_ROOT}/budget_fulldecay_metrics.json", "30",
                      "leg_top", "leg_qg", "leg2", "leg_jc2b"],
                     [bfb.LONG_FULL_ROOT, f"{bfb.LONG_FULL_ROOT}/budget_long_fulldecay_metrics.json", "6", "leg2", "leg_jc2b"]]
    assert int(calls[0][2]) == len(bfb.scratch_full_cells("1e-3", "3e-3")) and int(calls[1][2]) == 6
    assert 'if [ -e ${OUT} ]; then echo "skip ${OUT}: exists"; return 0; fi' in script
    assert "--out ${OUT}.new" in script and "not d['missing']" in script and "mv ${OUT}.new ${OUT}" in script
    pull = (ROOT / "experiments/FIGS/pull_mlst_data.py").read_text()
    for root, f in (("budget_fulldecay", "budget_fulldecay_metrics.json"),
                    ("budget_long_fulldecay", "budget_long_fulldecay_metrics.json")):
        assert f'(FT / "{root}", "budget", "{f}")' in pull
