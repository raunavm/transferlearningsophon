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
