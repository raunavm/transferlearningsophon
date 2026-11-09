"""The equal-stopping decision (experiments/FT/equal_stopping.py, PI 2026-11-09): moved when either arm's three-run
mean changes by more than its old sample SD, on the figure's metric; pending until all six new cells exist."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("equal_stopping", ROOT / "experiments/FT/equal_stopping.py")
es = importlib.util.module_from_spec(s)
s.loader.exec_module(es)
bfb_s = importlib.util.spec_from_file_location("bfb_es", ROOT / "scripts/build_ft_budget_jobs.py")
bfb = importlib.util.module_from_spec(bfb_s)
bfb_s.loader.exec_module(bfb)


def _cells(vals: dict) -> dict:
    """vals {(task, n, arm): [three values]} -> read-out cells with the metric and the AUC."""
    out = {}
    for (task, n, arm), v in vals.items():
        met, auc = es.METRIC[task]
        for k, x in zip(es.RUNS, v):
            out[f"{es.TREE[task]}/{arm}-s{k}/N{n}/s1"] = {met: x, auc: 0.9}
    return out


def test_the_sizes_and_metrics_are_the_jobs_and_the_figures():
    assert es.SIZES == bfb.PRE_FULL_SIZES and tuple(es.ARMS.values()) == bfb.PRE_FULL_RUNS
    assert es.METRIC["top"][0] == es.METRIC["qg"][0] == "r30"
    assert es.METRIC["jc1"][0] == es.METRIC["jc2b"][0] == "accuracy"


def test_a_change_beyond_the_old_sd_of_either_arm_moves_the_size():
    old = {(t, n, a): [1.0, 2.0, 3.0] for t, n in es.SIZES for a in ("l188", "r16q1")}     # SD 1.0
    new = dict(old)
    new[("top", 1000, "r16q1")] = [2.1, 3.1, 4.1]       # +1.1 > 1.0: moved
    new[("qg", 1000, "l188")] = [1.9, 2.9, 3.9]         # +0.9: not
    del new[("jc2b", 10000, "l188")]                    # not finished: pending
    rows = {(r["task"], r["n"]): r for r in es.decide(_cells(old), _cells(new))}
    assert rows[("top", 1000)]["moved"] is True and rows[("qg", 1000)]["moved"] is False
    assert rows[("jc2b", 10000)]["moved"] is None
    assert rows[("top", 1000)]["arms"]["17 classes"]["delta"] == pytest.approx(1.1)
