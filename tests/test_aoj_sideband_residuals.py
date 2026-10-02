"""experiments/AOJ/sideband_residuals.py: the passing-jet residual below the top window.

Synthetic histograms check the arithmetic and the refusals; the committed fit checks
that the arrays read are the fit's passing expectation (they must add up to the fitted
yield and to the passing jets), never a value of the residual itself.
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sideband_residuals",
                                              REPO / "experiments/AOJ/sideband_residuals.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

FIT = REPO / "experiments/FIGS/data/aoj_full_v1/fit_v6"


def test_the_residual_sums_the_bins_before_dividing():
    edges = [100.0, 110.0, 120.0, 130.0, 140.0]
    n = [110.0, 90.0, 130.0, 400.0]
    b, s = [100.0, 100.0, 100.0, 400.0], [0.0, 0.0, 0.0, 0.0]
    r = R.residual(n, b, s, edges, 100.0, 130.0)
    assert r["z"] == pytest.approx((330.0 - 300.0) / math.sqrt(300.0))
    assert r["n_bins"] == 3 and r["max_pull_bin_gev"] == [120.0, 130.0]
    assert r["max_abs_pull"] == pytest.approx(3.0)


def test_the_signal_counts_as_expected_passing_jets():
    r = R.residual([120.0], [100.0], [20.0], [0.0, 1.0], 0.0, 1.0)
    assert r["z"] == pytest.approx(0.0)


def test_a_range_off_the_bin_edges_is_refused():
    with pytest.raises(SystemExit, match="bin edges"):
        R.residual([1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0, 2.0], 0.5, 2.0)


def test_the_default_range_is_the_fit_edge_to_the_window_edge():
    P = R.peak_fit().PEAKS[R.PEAK]
    assert R.sideband_edges() == (float(P["fit_range"][0]), float(P["window"][0]))


def test_mean_pulls_rank_bins_by_size():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "h.npz"
        arrays = {}
        for name, n in (("a", [104.0, 100.0, 80.0]), ("b", [104.0, 100.0, 80.0])):
            arrays[f"{name}_top_n_pass"] = np.array(n)
            arrays[f"{name}_top_background"] = np.full(3, 100.0)
            arrays[f"{name}_top_signal"] = np.zeros(3)
            arrays[f"{name}_top_m_edges"] = np.array([0.0, 1.0, 2.0, 3.0])
        np.savez(p, **arrays)
        mp = R.mean_pulls(p, ["a", "b"])
        assert mp["rank"] == [2, 3, 1]
        assert mp["mean_pull"][2] == pytest.approx(-2.0)


@pytest.mark.skipif(not (FIT / "histograms.npz").exists(), reason="fit_v6 not committed")
def test_the_committed_arrays_are_the_fits_passing_expectation():
    hist = np.load(FIT / "histograms.npz")
    fits = json.loads((FIT / "results.json").read_text())["models"]
    names = R.models(hist)
    assert set(names) == set(fits)
    for n in names:
        b, s = hist[f"{n}_top_background"], hist[f"{n}_top_signal"]
        assert s.sum() == pytest.approx(fits[n]["top"]["signal_yield"], rel=1e-6)
        assert hist[f"{n}_top_n_pass"].sum() == pytest.approx(fits[n]["top"]["n_pass"])
        assert (b + s).sum() == pytest.approx(fits[n]["top"]["n_pass"], rel=1e-4)
    res = R.all_residuals(FIT / "histograms.npz")
    assert set(res) == set(names) and all(math.isfinite(r["z"]) for r in res.values())
