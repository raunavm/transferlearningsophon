"""CI for experiments/MASSREG/gated_head.py.

The estimator must match colizz/weaver-core-dev @ 7ac799e exactly. Reproducing
CMS-DP-2026-104's published formula instead -- which omits the residual
composition -- yields a larger apparent bias, and that error is invisible in the
output.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "MASSREG"))
g = pytest.importorskip("gated_head")


def test_residual_composition_is_additive_on_the_generic_node():
    """THE regression test for the note-vs-code discrepancy. With zero residuals
    the estimator must return theta_generic untouched. A non-residual mixture
    would return the p-weighted mean of the experts instead."""
    out = g.combine(np.array([1.10, 0.95]), np.zeros((2, 3)),
                    np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    assert out == pytest.approx([1.10, 0.95])


def test_a_confident_gate_selects_that_class_residual():
    out = g.combine(np.array([1.0]), np.array([[0.5, -0.3, 0.2]]),
                    np.array([[1.0, 0.0, 0.0]]))
    assert out == pytest.approx([1.5])


def test_gate_is_a_weighted_mean_of_residuals():
    out = g.combine(np.array([0.0]), np.array([[1.0, 0.0]]), np.array([[0.25, 0.75]]))
    assert out == pytest.approx([0.25])


def test_alpha_weight_matches_the_exporter_factor_two_on_qq():
    """Exporter line 147 carries `2*massCorrHqq*probHqq` over `2*probHqq`."""
    d = np.array([[1.0, 0.0]]); p = np.array([[0.5, 0.5]])
    assert g.combine(np.zeros(1), d, p, alpha=np.array([1.0, 2.0])) == pytest.approx([1 / 3])


def test_qcd_columns_are_excluded_by_omission_from_S():
    """A huge QCD residual with most of the probability must not move the answer
    when its column is not in S -- this is the deployed two-prong behaviour."""
    delta = np.array([[0.4, 99.0]])       # col 1 is QCD
    p = np.array([[0.05, 0.95]])
    out = g.combine(np.zeros(1), delta, p, classes=[0])
    assert out == pytest.approx([0.4])


def test_denominator_is_clamped_not_divided_by_zero():
    out = g.combine(np.array([1.0]), np.array([[2.0, 3.0]]), np.zeros((1, 2)))
    assert np.isfinite(out).all()


def test_oracle_gate_recovers_the_true_class_residual():
    """E5a: same trained residuals, gated on the TRUE label. This is the control
    that attributes any distortion to the train/inference gating mismatch."""
    delta = np.array([[0.5, -0.4], [0.5, -0.4]])
    truth = g.split_loss_mask(np.array([0, 1]), 2).astype(float)
    assert g.combine(np.zeros(2), delta, truth) == pytest.approx([0.5, -0.4])


def test_predicted_mass_is_multiplicative_on_the_ungroomed_mass():
    assert g.predicted_mass(np.array([100.0, 50.0]),
                            np.array([1.2, 0.8])) == pytest.approx([120.0, 40.0])


def test_split_loss_mask_is_one_hot_on_the_true_label():
    m = g.split_loss_mask(np.array([2, 0]), 4)
    assert m.sum() == 2 and m[0, 2] and m[1, 0]


def test_gate_weight_measures_resonance_probability_mass():
    p = np.array([[0.3, 0.2, 0.5]])
    assert g.gate_weight(p, classes=[0, 1]) == pytest.approx([0.5])


def test_mismatched_shapes_raise_rather_than_broadcast():
    with pytest.raises(ValueError):
        g.combine(np.zeros(2), np.zeros((2, 3)), np.zeros((2, 4)))
