"""The trend-test size simulation: its statistic, its nulls and its capture."""
import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import trend as T                                   # noqa: E402

spec = importlib.util.spec_from_file_location("trend_size_sim",
                                              ROOT / "experiments/STATS/trend_size_sim.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

ORDER = [188, 162, 43, 17]


@pytest.mark.parametrize("alternative", ["increasing", "decreasing", "two-sided"])
@pytest.mark.parametrize("family", ["marcus", "williams", "changepoint"])
def test_vectorised_statistic_equals_the_trend_test(alternative, family):
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(6, 5, 4)) * np.array([0.03, 0.03, 0.03, 0.16]) + np.array([0, 0.01, 0.05, 0.9])
    got = M.max_t_stats(Y, family, alternative)
    for i, tab in enumerate(Y):
        r = T.max_t_trend(tab.ravel(), ORDER * 5, np.repeat(np.arange(5), 4),
                          alternative=alternative, family=family, exact=False, n_perm=9,
                          rng=0, order=ORDER)
        assert math.isclose(got[i], r["stat"], rel_tol=1e-10), (i, got[i], r["stat"])


def test_table_follows_the_level_order_not_the_input_order():
    call = {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "levels": [17, 43, 162, 188, 188, 162, 43, 17],
            "blocks": [1, 1, 1, 1, 2, 2, 2, 2], "kw": {"order": ORDER}}
    Y, order = M.table(call)
    assert order == ORDER
    assert Y.tolist() == [[4, 3, 2, 1], [5, 6, 7, 8]]


def test_indep_upper_inflates_only_the_noisiest_level_by_the_chi2_bound():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 4)) * np.array([0.03, 0.02, 0.03, 0.16])
    D0, _ = M.null_cov(Y, "indep")
    D1, info = M.null_cov(Y, "indep_upper")
    j = int(np.argmax(np.diag(D0)))
    assert info["inflated_level_index"] == j
    assert info["sd_factor"] == pytest.approx(2.874, abs=1e-3)       # sqrt(4 / chi2_0.025,4)
    ratio = np.diag(D1) / np.diag(D0)
    assert ratio[j] == pytest.approx(info["sd_factor"] ** 2)
    assert np.allclose(np.delete(ratio, j), 1.0)
    assert np.allclose(D0, np.diag(np.diag(D0)))


def test_draws_have_the_requested_covariance_even_when_singular():
    C = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]])
    x = M.draw(C, (200_000,), np.random.default_rng(0))
    assert np.allclose(np.cov(x, rowvar=False), C, atol=0.02)


def test_stored_trends_finds_nested_max_t_results_and_skips_the_isotonic_companion():
    doc = {"confirmatory": {"C1": {"run": True, "stat": 3.0, "p": 1e-5, "n_arrangements": 7,
                                   "method": "exact", "isotonic": {"stat": 0.9, "p": 1e-4}}},
           "secondary": {"S4": {"per_size": {"N1000": {"trend": {"run": False}},
                                             "N10000": {"trend": {"run": True, "stat": 2.0,
                                                                  "p": 0.01, "n_arrangements": 7,
                                                                  "method": "exact"}}}}}}
    found = M.stored_trends(doc)
    assert [p for p, _ in found] == ["confirmatory.C1", "secondary.S4.per_size.N10000.trend"]


def test_a_null_with_equal_spreads_keeps_the_monte_carlo_test_near_its_level():
    """The machinery itself: with exchangeable levels the size must be about 5%."""
    old = M.N_SIZE
    M.N_SIZE = 400
    try:
        C = np.eye(4) * 0.03 ** 2
        r = M._size_chunk((C, 5, ORDER, {"alternative": "increasing", "family": "marcus"},
                           False, [1, 2], 400))
    finally:
        M.N_SIZE = old
    assert 5 <= r <= 40          # 400 draws at 5%: mean 20, 99.9% within this band
