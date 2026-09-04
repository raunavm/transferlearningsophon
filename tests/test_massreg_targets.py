"""CI for experiments/MASSREG/targets.py.

Every failure mode here is silent: a wrong tau rule or a missed neutrino still
returns a plausible mass, and the resulting bias would look like a physics result.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "MASSREG"))
t = pytest.importorskip("targets")


def test_two_body_back_to_back_gives_the_textbook_mass():
    """Two 100 GeV massless-ish partons back to back -> m = 2*E."""
    pt = np.array([100.0, 100.0]); eta = np.zeros(2)
    phi = np.array([0.0, np.pi]); m = np.zeros(2)
    pid = np.array([1, -1]); rdp = np.array([True, True]); tdp = np.zeros(2, bool)
    assert t.m_vis_true(pt, eta, phi, m, pid, rdp, tdp) == pytest.approx(200.0, rel=1e-6)


def test_collinear_partons_give_zero_mass():
    pt = np.array([50.0, 50.0]); eta = np.zeros(2); phi = np.zeros(2); m = np.zeros(2)
    pid = np.array([1, 2]); rdp = np.array([True, True]); tdp = np.zeros(2, bool)
    assert t.m_vis_true(pt, eta, phi, m, pid, rdp, tdp) == pytest.approx(0.0, abs=1e-6)


def test_neutrinos_are_excluded():
    """A neutrino carrying half the energy must not enter the visible mass."""
    pt = np.array([100.0, 100.0]); eta = np.zeros(2)
    phi = np.array([0.0, np.pi]); m = np.zeros(2)
    rdp = np.array([True, True]); tdp = np.zeros(2, bool)
    with_nu = t.m_vis_true(pt, eta, phi, m, np.array([1, 14]), rdp, tdp)
    only_q = t.m_vis_true(pt[:1], eta[:1], phi[:1], m[:1], np.array([1]),
                          rdp[:1], tdp[:1])
    assert with_nu == pytest.approx(only_q)


@pytest.mark.parametrize("nu", [12, 14, 16, -12, -14, -16])
def test_every_neutrino_flavour_and_sign_is_excluded(nu):
    pid = np.array([1, nu])
    keep = t.visible_mask(pid, np.array([True, True]), np.zeros(2, bool))
    assert keep.tolist() == [True, False]


def test_the_tau_is_replaced_by_its_daughters_not_double_counted():
    """The tau itself must drop out; its daughters must come in. Counting both
    would roughly double that leg's contribution."""
    pid = np.array([15, 211, 211, 16])          # tau, two pions, tau neutrino
    rdp = np.array([True, False, False, False])
    tdp = np.array([False, True, True, True])
    keep = t.visible_mask(pid, rdp, tdp)
    assert keep.tolist() == [False, True, True, False]


def test_a_tau_flagged_both_ways_is_still_dropped():
    """If a producer flags the tau as its own decay product, the |pid|==15 cut
    must still remove it, or it is counted twice."""
    pid = np.array([15, 211])
    keep = t.visible_mask(pid, np.array([True, False]), np.array([True, True]))
    assert keep.tolist() == [False, True]


def test_empty_visible_set_returns_zero_not_nan():
    pid = np.array([14]); rdp = np.array([True]); tdp = np.zeros(1, bool)
    assert t.m_vis_true(np.array([50.0]), np.zeros(1), np.zeros(1), np.zeros(1),
                        pid, rdp, tdp) == 0.0


def test_m_res_true_reads_the_flagged_resonance_and_qcd_gets_zero():
    mass = np.array([125.0, 4.18, 4.18])
    assert t.m_res_true(mass, np.array([True, False, False])) == pytest.approx(125.0)
    assert t.m_res_true(mass, np.zeros(3, bool)) == 0.0


def test_ungroomed_jet_mass_matches_the_four_vector():
    pt, eta, phi, m = 500.0, 1.2, 0.3, 80.0
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    assert t.m_jet_ungroomed(pt, eta, phi, e) == pytest.approx(m, rel=1e-6)


def test_ungroomed_jet_mass_clamps_instead_of_returning_nan():
    """float32 round-off drives E^2-|p|^2 negative for near-massless jets; an
    unclamped sqrt gives nan, which propagates silently through a ratio target."""
    out = t.m_jet_ungroomed(100.0, 0.0, 0.0, 99.9999)
    assert np.isfinite(out) and out == 0.0


def test_audit_reports_the_assumptions_as_counts():
    pid = np.array([15, 211, 16, 5])
    a = t.audit_flags(pid, np.array([True, False, False, True]),
                      np.array([False, True, True, False]),
                      np.array([False, False, False, False]))
    assert a["n_tau_in_res_decay_prod"] == 1
    assert a["n_nu_in_res_decay_prod"] == 0
    assert a["n_visible"] == 2          # the pion and the b, not the tau or nu
