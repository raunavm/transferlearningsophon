"""experiments/FIGS/rev35_stats.py (ledger 35: audit V4, V6, V7, A1)."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "FIGS"))
import rev35_stats as R  # noqa: E402


def test_spearman_exact_matches_the_small_tables():
    assert R.spearman_exact([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == (pytest.approx(1.0), pytest.approx(2 / 120))
    rho, p = R.spearman_exact([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert rho == pytest.approx(0.8) and p == pytest.approx(16 / 120)


def test_t_quantile():
    assert R._t_quantile(0.00135, 1e7) == pytest.approx(3.0, abs=1e-3)
    assert R._t_quantile(0.025, 10) == pytest.approx(2.228139, abs=1e-5)
    assert R._t_quantile(0.00135, 4) == pytest.approx(6.62, abs=0.01)


def test_separation_weights_the_pairs_a_label_set_keeps_apart():
    P = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    assert R.separation(P, np.array([0, 1, 2])) == pytest.approx(1.0)
    assert R.separation(P, np.array([0, 0, 1])) == pytest.approx(0.0)     # the only confused pair merged
    assert R.naive(np.array([0, 0, 1])) == pytest.approx(2 / 3)


def test_every_label_set_has_a_map_column_and_probe_runs():
    g = R.groups()
    assert len(g) == 12 and all(v.shape == (188,) for v in g.values()) and len(set(g["L188"])) == 188
    assert len(set(g["R16_Q1"])) == len(set(g["RAND2_p1"])) == len(set(g["FLAV_F0"])) == 17
    assert [len(R.accuracy(a)) for a in ("L188", "RAND2_p1", "FLAV_F0")] == [3, 1, 2]


def test_the_differences_table_reads_every_row():
    rows = R.v4()
    assert len(rows) == len(R.V4_ROWS) >= 15
    assert all(r["sigma"] > 0 and r["n_first"] >= 1 for r in rows)
    tex = R.table_v4(rows)
    assert r"\label{tab:differences}" in tex and tex.count(r"\\") >= len(rows)
