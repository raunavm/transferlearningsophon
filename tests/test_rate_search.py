"""The fine-tuning rate search (PI 2026-11-21): one source of each row's rate (build_ft_budget_jobs.FIXED_ROW_RATE and
FIXED_ROW_RATE_SCRATCH) for the builder, the read-out and Table A15; the one divergence rule; the 1e-5 floor; every
cell on its size's schedule."""
import importlib.util
import json
import pathlib
import re
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
RS = _load("rate_search", "experiments/FT/rate_search.py")
ES = _load("early_stop", "experiments/FT/early_stop.py")
SPECS = ROOT / "experiments/FT/k8s/budget"


def _row(c):
    return ("top" if c["task"] == "topkin" else c["task"], c["n"], "pre" if c["init"] else "scratch")


def test_every_row_the_paper_reads_has_one_rate_its_recorded_one():
    rows = {}
    for c in B.paper_fixed_cells():
        rows.setdefault(_row(c), set()).add(c["lr"])
    for (task, n, kind), rates in rows.items():
        want = B.row_rate(task, n, kind) or B.original_rate(task, kind)
        assert rates == {want}, f"{task} {n} {kind}: reads {sorted(rates)}, the recorded rate is {want}"


def test_a_row_at_a_new_rate_is_read_whole_from_the_rerun_tree():
    for c in B.paper_fixed_cells():
        task, n, kind = _row(c)
        moved = c["lr"] != B.original_rate(task, kind)
        assert c["out"].startswith((B.FIXED_RATE_ROOT if moved else B.FIXED_ROOT) + "/"), c["out"]


def test_every_recorded_rate_is_a_searched_rate_of_its_row():
    plan = B.fixed_search_plan()
    for kind, table in (("pre", B.FIXED_ROW_RATE), ("scratch", B.FIXED_ROW_RATE_SCRATCH)):
        for (task, n), lr in table.items():
            assert lr in plan[(task, n, kind)], f"{task} {n} {kind}: {lr} was never tried"


def _spec_cells(name):
    text = (SPECS / name).read_text()
    return {int(i): cell for i, cell in re.findall(r"^\s*(\d+)\) TASK=.*?CELL=([^;]+);;", text, re.M)}


def test_the_rerun_job_indices_keep_the_launched_order():
    cells = [c["out"] for c in B.fixed_rate_cells()]
    assert _spec_cells("job-ft-budget-fixed-rate-raunav.yaml") == dict(enumerate(cells[:16]))
    launched = _spec_cells("job-ft-budget-fixed-rate-qg1e3-raunav.yaml")
    assert launched == {i: cells[16 + i] for i in range(len(launched))}


def test_the_plan_averages_both_pretrained_networks_and_one_from_scratch_run_at_every_rate():
    for (task, n, kind), rates in B.fixed_search_plan().items():
        for lr, dirs in rates.items():
            assert len(dirs) == len(B.FIXED_SEARCH_INITS[kind]), (task, n, kind, lr, dirs)
            assert float(lr) >= float(B.FIXED_SEARCH_FLOOR), (task, n, kind, lr)


def test_no_superseded_cell_is_read():
    """The old 1e4 neighbours (20,000 steps checked every 50) and the 40,000-step top 1e3 smoke cells."""
    dirs = [d for rates in B.fixed_search_plan().values() for ds in rates.values() for d in ds]
    old_1e4 = [d for d in dirs if d.startswith(B.FIXED_SWEEP_ROOT + "/") and "/N10000/" in d
               and ("/leg_top/" in d or "/leg_qg/" in d)]
    smoke = [d for d in dirs if d in {c["out"] for c in B.fixed_smoke_1e3_cells()}]
    assert not old_1e4 and not smoke, old_1e4 + smoke


def test_every_search_cell_follows_its_sizes_schedule():
    cells = (B.fixed_sweep_rows_cells() + B.fixed_sweep_jc2b_cells() + B.fixed_sweep_all_cells()
             + B.fixed_sweep_edge_cells() + B.fixed_sweep_edge2_cells() + B.fixed_sweep_1e4_cells() + B.fixed_rate_cells())
    for c in cells:
        assert c["spe"] == B.FIXED_CHECK[c["n"]] * B.BATCH and c["epochs"] * B.FIXED_CHECK[c["n"]] == B.FIXED_STEPS[c["n"]], c


# ---------------------------------------------------------------- the rules, on a small made-up search
def _fake(plan, recorded=None, paper=()):
    return types.SimpleNamespace(
        fixed_search_plan=lambda: plan, paper_fixed_cells=lambda: list(paper),
        row_rate=lambda t, n, k: (recorded or {}).get((t, n, k)), original_rate=lambda t, k: "1e-2",
        FIXED_CHECK={1_000: 50}, FIXED_STEPS={1_000: 20_000}, FIXED_SEARCH_FLOOR="1e-5")


def _cell(lr, best=None, done=True, nonfinite=None, check=50, diverged_record=None, partial_nonfinite=False):
    m = {"lr": lr, "steps_per_epoch": str(check), "max_epochs": str(20_000 // check)}
    attempts = [{"dir": "x", "manifest": m, "first_nonfinite_check": nonfinite, "best_metric": best}]
    if partial_nonfinite:
        attempts.insert(0, {"dir": "x.partial.1", "manifest": m, "first_nonfinite_check": 7, "best_metric": 0.99})
    return {"done": done, "superseded": False, "diverged_record": diverged_record, "attempts": attempts}


def _plan(rates):
    return {("top", 1_000, "scratch"): {lr: [f"{RS.FT_ROOT}s/{lr}"] for lr in rates}}


def _scan(cells):
    return {"cells": {f"s/{lr}": c for lr, c in cells.items()}}


def test_a_divergence_in_any_attempt_scores_below_every_rate_that_trained():
    rates = ("1e-3", "3e-3", "1e-2")
    scan = _scan({"1e-3": _cell("1e-3", 0.80), "3e-3": _cell("3e-3", 0.85),
                  "1e-2": _cell("1e-2", 0.80, partial_nonfinite=True)})    # retried to DONE: still a divergence
    row = RS.evaluate(scan, _fake(_plan(rates), {("top", 1_000, "scratch"): "3e-3"}))[0]
    assert row["values"]["1e-2"] == RS.DIVERGED and row["best"] == "3e-3" and row["status"] == "settled"
    assert not row["issues"]


def test_a_best_check_before_the_blow_up_never_counts():
    rates = ("1e-3", "3e-3", "1e-2")
    scan = _scan({"1e-3": _cell("1e-3", 0.80), "3e-3": _cell("3e-3", 0.85),
                  "1e-2": _cell("1e-2", 0.99, done=False, nonfinite=12)})
    row = RS.evaluate(scan, _fake(_plan(rates)))[0]
    assert row["best"] == "3e-3" and row["status"] == "settled"


def test_a_row_still_best_at_its_lowest_rate_settles_only_at_the_floor():
    scan = _scan({"3e-5": _cell("3e-5", 0.9), "1e-4": _cell("1e-4", 0.8)})
    assert RS.evaluate(scan, _fake(_plan(("3e-5", "1e-4"))))[0]["status"] == "edge: lowest rate"
    scan = _scan({"1e-5": _cell("1e-5", 0.9), "3e-5": _cell("3e-5", 0.8)})
    assert RS.evaluate(scan, _fake(_plan(("1e-5", "3e-5"))))[0]["status"] == "settled at the floor"


def test_a_cell_off_its_sizes_schedule_is_an_issue():
    rates = ("1e-3", "3e-3", "1e-2")
    scan = _scan({"1e-3": _cell("1e-3", 0.80), "3e-3": _cell("3e-3", 0.85, check=100), "1e-2": _cell("1e-2", 0.8)})
    row = RS.evaluate(scan, _fake(_plan(rates), {("top", 1_000, "scratch"): "3e-3"}))[0]
    assert any("checks of 100 steps" in e for e in row["issues"])


def test_a_divergence_at_the_recorded_rate_in_the_rerun_fails_that_rate_for_the_row():
    rates = ("1e-3", "3e-3", "1e-2")
    paper = [{"task": "top", "n": 1_000, "init": "", "out": f"{RS.FT_ROOT}r/s2", "lr": "3e-3"}]
    cells = {f"s/{lr}": _cell(lr, v) for lr, v in (("1e-3", 0.82), ("3e-3", 0.85), ("1e-2", 0.80))}
    cells["r/s2"] = _cell("3e-3", None, done=False, nonfinite=3)
    row = RS.evaluate({"cells": cells}, _fake(_plan(rates), {("top", 1_000, "scratch"): "3e-3"}, paper))[0]
    assert row["rerun_diverged"] == ["r/s2"] and row["best"] == "1e-3"
    assert any("recorded 3e-3" in e for e in row["issues"])


def test_a_recorded_rate_other_than_the_selected_one_is_an_issue():
    rates = ("1e-3", "3e-3", "1e-2")
    scan = _scan({"1e-3": _cell("1e-3", 0.80), "3e-3": _cell("3e-3", 0.85), "1e-2": _cell("1e-2", 0.8)})
    row = RS.evaluate(scan, _fake(_plan(rates), {("top", 1_000, "scratch"): "1e-3"}))[0]
    assert any("selected 3e-3, recorded 1e-3" in e for e in row["issues"])


def test_the_watcher_stops_at_the_first_nonfinite_training_loss():
    lines = ["Train AvgLoss: 0.69, AvgAcc: 0.5\n", "Train AvgLoss: 0.41, AvgAcc: 0.8\n", "Train AvgLoss: nan, AvgAcc: 0.5\n"]
    assert ES.nonfinite_check(lines) == 2
    assert ES.nonfinite_check(lines[:2]) is None
    assert ES.nonfinite_check(["Train AvgLoss: inf, AvgAcc: 0.5\n"]) == 0


def test_the_cell_script_records_a_divergence_and_never_retries_it():
    s = B.SCRIPT if hasattr(B, "SCRIPT") else (ROOT / "scripts/build_ft_budget_jobs.py").read_text()
    assert '[ ! -f ${CELL}/DIVERGED ] || { echo "${CELL} diverged: not retried"; exit ${HALT}; }' in s
    assert 'echo "FATAL: ${CELL} diverged (non-finite training loss)"; exit ${HALT}' in s
    assert 'diverged (NaN training loss)"; exit 1' not in s


def test_the_pulled_read_out_is_at_every_rows_recorded_rate():
    """Binds once the paper's read-out (fixed_metrics.json, complete) is pulled."""
    p = ROOT / "experiments/FIGS/data/v2/budget/fixed_metrics.json"
    if not p.exists():
        pytest.skip("the complete one-rule read-out is not pulled yet")
    import sys
    sys.path.insert(0, str(ROOT / "experiments/FIGS"))
    import mlst_results as M
    assert M.rate_mismatches(M._budget_cells(p)) == []


def test_the_new_output_layer_trains_at_5e_3_at_every_rate():
    """The methods sentence (PI 2026-11-21): the head trains at 5e-3 whatever the backbone rate, the 1e-5 floor included."""
    cells = (B.paper_fixed_cells() + B.fixed_sweep_rows_cells() + B.fixed_sweep_jc2b_cells() + B.fixed_sweep_all_cells()
             + B.fixed_sweep_edge_cells() + B.fixed_sweep_edge2_cells() + B.fixed_sweep_1e4_cells() + B.fixed_rate_cells())
    for c in cells:
        if c["init"]:
            assert abs(float(c["lr"]) * float(c["mult"]) - B.HEAD_CAP) < 1e-12, (c["out"], c["lr"], c["mult"])
