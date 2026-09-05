"""Tests for the E1 control's discriminant and its QCD index set.

The index set matters more than it looks. E1's whole job is to show the QCD
spectrum is NOT sculpted, and every count in that figure is selected by
`is_qcd`. Get the index set wrong and the figure is flat for the wrong reason.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "e1_control", REPO / "experiments" / "MASSREG" / "e1_control.py")
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)


def test_qcd_index_set_matches_ground_truth():
    """27 QCD classes at 161..187 -- docs/GROUND_TRUTH.md, 161 resonant + 27."""
    q = e1.qcd_indices()
    assert q.size == 27
    assert q.min() == 161 and q.max() == 187
    np.testing.assert_array_equal(q, np.arange(161, 188))


def test_qcd_indices_are_read_not_hardcoded(tmp_path):
    """A regenerated map must change the answer, or the read is decorative."""
    p = tmp_path / "map.csv"
    p.write_text("jet_label,class_name\n0,label_X_bb\n1,label_QCD_foo\n"
                 "2,label_QCD_bar\n3,label_Y_cc\n")
    np.testing.assert_array_equal(e1.qcd_indices(p), np.array([1, 2]))


def test_qcd_indices_dies_on_a_map_with_no_qcd(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text("jet_label,class_name\n0,label_X_bb\n")
    with pytest.raises(SystemExit):
        e1.qcd_indices(p)


def test_softmax_rows_sum_to_one_and_survive_large_logits():
    z = np.array([[1e3, 1e3 + 1, -1e3], [0.0, 0.0, 0.0]])
    p = e1.softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p)), "max-subtraction must prevent overflow"


def test_discriminant_is_one_for_pure_signal_and_zero_for_pure_qcd():
    p = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    d = e1.discriminant(p, np.array([0, 1]), np.array([2, 3]))
    np.testing.assert_allclose(d, [1.0, 0.0])


def test_discriminant_matches_the_published_ratio_by_hand():
    """D_S = sum a_i p_i / (sum a_i p_i + sum_QCD p_i), DP-2026-104 p.8."""
    p = np.array([[0.3, 0.1, 0.4, 0.2]])
    d = e1.discriminant(p, np.array([0, 1]), np.array([2, 3]))
    np.testing.assert_allclose(d, [(0.3 + 0.1) / (0.3 + 0.1 + 0.4 + 0.2)])


def test_alpha_reweights_the_signal_sum():
    p = np.array([[0.2, 0.2, 0.6, 0.0]])
    d = e1.discriminant(p, np.array([0, 1]), np.array([2, 3]),
                        alpha=np.array([3.0, 1.0]))
    num = 3.0 * 0.2 + 1.0 * 0.2
    np.testing.assert_allclose(d, [num / (num + 0.6)])


def test_alpha_length_mismatch_is_fatal():
    p = np.array([[0.25, 0.25, 0.25, 0.25]])
    with pytest.raises(SystemExit):
        e1.discriminant(p, np.array([0, 1]), np.array([2, 3]),
                        alpha=np.array([1.0]))


def test_zero_denominator_yields_zero_not_nan():
    """A jet with no mass in signal-or-QCD must not become nan and vanish
    from every histogram without a word."""
    p = np.zeros((1, 4))
    p[0, 3] = 0.0
    d = e1.discriminant(p, np.array([0]), np.array([1]))
    assert np.all(np.isfinite(d)) and d[0] == 0.0


def test_discriminant_ignores_classes_in_neither_set():
    """Only the signal and QCD sets enter; a third category must not leak in."""
    p = np.array([[0.2, 0.0, 0.2, 0.6]])  # index 3 is neither
    d = e1.discriminant(p, np.array([0]), np.array([2]))
    np.testing.assert_allclose(d, [0.2 / (0.2 + 0.2)])


# ---------------------------------------------------------------------------
# p_T binning. The 2026-09-05 inclusive figure failed with chi2/ndf 99. The
# diagnosis was that integrating over p_T reintroduces mass dependence when
# the discriminant is decorrelated only at FIXED p_T. These tests encode that
# diagnosis as a synthetic that must sculpt inclusively and be flat per bin.
# ---------------------------------------------------------------------------
MASS_EDGES = np.linspace(20.0, 500.0, 51)


def _two_populations(n=400_000, seed=0):
    """Low-p_T jets are low-mass, high-p_T jets are high-mass. D depends on p_T
    only, so at fixed p_T it is independent of mass -- the decorrelated case."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pt = np.concatenate([rng.uniform(200, 600, half), rng.uniform(600, 2500, half)])
    m = np.concatenate([rng.uniform(20, 200, half), rng.uniform(100, 500, half)])
    D = pt / 2500.0 + 1e-3 * rng.standard_normal(n)
    return m, D, pt


def _worst_chi2(wp):
    return max(v["chi2_per_ndf"] for v in wp["working_points"].values() if not v["thin"])


def test_independent_discriminant_is_flat():
    rng = np.random.default_rng(1)
    m = rng.uniform(20, 500, 300_000)
    D = rng.uniform(0, 1, m.size)
    assert _worst_chi2(e1.spectrum(m, D, MASS_EDGES)) < 1.6


def test_mass_correlated_discriminant_sculpts():
    rng = np.random.default_rng(2)
    m = rng.uniform(20, 500, 300_000)
    D = m / 500.0 + 0.05 * rng.standard_normal(m.size)
    assert _worst_chi2(e1.spectrum(m, D, MASS_EDGES)) > 20


def test_integrating_over_pt_sculpts_when_each_pt_bin_is_flat():
    """THE diagnosis. Same jets, same discriminant: inclusive fails, binned passes."""
    m, D, pt = _two_populations()
    inclusive = _worst_chi2(e1.spectrum(m, D, MASS_EDGES))
    assert inclusive > 20, f"inclusive should sculpt, chi2/ndf={inclusive:.2f}"
    for lo, hi in [(200, 600), (600, 2500)]:
        b = (pt >= lo) & (pt < hi)
        per_bin = _worst_chi2(e1.spectrum(m[b], D[b], MASS_EDGES))
        assert per_bin < 1.6, f"bin [{lo},{hi}) should be flat, chi2/ndf={per_bin:.2f}"


def test_flatness_is_invariant_to_monotone_rescaling_of_the_discriminant():
    """Thresholds are quantiles and ratios are normalised within the set, so
    the working points depend on D only through its ordering."""
    m, D, _ = _two_populations(n=100_000, seed=3)
    a = e1.spectrum(m, D, MASS_EDGES)
    b = e1.spectrum(m, 3.0 * D + 1.0, MASS_EDGES)
    for k in a["working_points"]:
        assert a["working_points"][k]["counts"] == b["working_points"][k]["counts"]
        np.testing.assert_allclose(a["working_points"][k]["chi2_vs_flat"],
                                   b["working_points"][k]["chi2_vs_flat"])


def test_thin_working_point_is_flagged_and_a_fat_one_is_not():
    rng = np.random.default_rng(4)
    m = rng.uniform(20, 500, 20_000)          # 0.5% of 20k = 100 selected: thin
    D = rng.uniform(0, 1, m.size)
    wp = e1.spectrum(m, D, MASS_EDGES)["working_points"]
    assert wp["eps_B=0.005"]["thin"] and wp["eps_B=0.005"]["n_selected"] < e1.MIN_SELECTED
    assert not wp["eps_B=0.05"]["thin"] and wp["eps_B=0.05"]["n_selected"] >= e1.MIN_SELECTED


def test_default_pt_edges_contain_the_two_published_bins():
    """DP-2026-104 Fig. 7 shows 400-600 and 1000-1500 GeV; both must be exact bins."""
    e = e1.PT_EDGES
    assert [400.0, 600.0] == [x for x in e if x in (400.0, 600.0)]
    assert [1000.0, 1500.0] == [x for x in e if x in (1000.0, 1500.0)]
    assert e[0] == 200.0 and e[-1] == 2500.0, "must span the study's own selection"
