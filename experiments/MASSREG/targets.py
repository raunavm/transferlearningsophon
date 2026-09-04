"""GloParT's two mass-regression targets, built from JetClass-II's aux gen block.

CMS-DP-2026-104 p.43 defines the regression target as a RATIO

    theta^a_true = m^a_true / m_jet ,   a in {vis, res}

with m_jet "the invariant mass computed from all PF candidates in the jet"
(ungroomed), and the numerator branching on the truth class:

    signal -> m^vis_true, "the invariant mass of all non-neutrino resonance
              daughters represented by quarks or leptons. For tau leptons, the
              corresponding decay daughters are used instead."
    QCD    -> m^gen_SD, gen soft-drop mass excluding neutrinos  (= genjet_sdmass)

v1 used m^res_true, the generator resonance mass, on the signal branch instead.

The E0 schema probe (job massreg-schema-raunav, 2026-08-30) confirmed JetClass-II
carries everything needed: aux_genpart_{pt,eta,phi,mass,pid} plus the flags
isResX / isResY / isResDecayProd / isTauDecayProd / isQcdParton.

THE TAU RULE IS AN ASSUMPTION ABOUT FLAG SEMANTICS, NOT A MEASUREMENT. We take a
tau to be flagged isResDecayProd, and its daughters isTauDecayProd, so the visible
set is

    (isResDecayProd AND |pid| != 15)  OR  isTauDecayProd,   minus neutrinos.

`audit_flags` measures whether that holds. Run it before trusting any number from
here -- a wrong reading of the flags is silent and shifts every signal-branch mass.
"""
from __future__ import annotations

import numpy as np

NEUTRINO_PIDS = (12, 14, 16)
TAU_PID = 15


def _p4(pt, eta, phi, mass):
    """(pt, eta, phi, m) -> (px, py, pz, E). Massless-safe."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, e


def _invmass(px, py, pz, e):
    m2 = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2, 0.0))


def visible_mask(pid, is_res_decay_prod, is_tau_decay_prod):
    """The set summed for m^vis_true, per the DP note's wording.

    A tau is dropped and replaced by its daughters; neutrinos are removed from
    whatever survives. Both arrays are per-particle within one jet.
    """
    apid = np.abs(np.asarray(pid))
    keep = np.asarray(is_res_decay_prod) | np.asarray(is_tau_decay_prod)
    # The tau veto is applied AFTER the union, not inside it. A producer that
    # flags a tau as both isResDecayProd and isTauDecayProd would otherwise
    # reintroduce it through the second term and double-count that leg against
    # its own daughters. Caught by
    # tests/test_massreg_targets.py::test_a_tau_flagged_both_ways_is_still_dropped.
    return keep & (apid != TAU_PID) & ~np.isin(apid, NEUTRINO_PIDS)


def m_vis_true(pt, eta, phi, mass, pid, is_res_decay_prod, is_tau_decay_prod):
    """Invariant mass of the visible resonance daughters of ONE jet."""
    keep = visible_mask(pid, is_res_decay_prod, is_tau_decay_prod)
    if not keep.any():
        return 0.0
    px, py, pz, e = _p4(np.asarray(pt)[keep], np.asarray(eta)[keep],
                        np.asarray(phi)[keep], np.asarray(mass)[keep])
    return float(_invmass(px.sum(), py.sum(), pz.sum(), e.sum()))


def m_res_true(mass, is_res_x, is_res_y=None):
    """The generator resonance mass of ONE jet -- v1's signal branch.

    Returns 0.0 when no resonance is flagged, which is the QCD case and is the
    same sentinel genjet_sdmass uses for 'unmatched'. Mask on > 0 downstream.
    """
    mass = np.asarray(mass)
    sel = np.asarray(is_res_x)
    if is_res_y is not None:
        sel = sel | np.asarray(is_res_y)
    hits = mass[sel]
    return float(hits[0]) if hits.size else 0.0


def m_jet_ungroomed(jet_pt, jet_eta, jet_phi, jet_energy):
    """The DP note's denominator, from the stored jet 4-vector.

    JetClass-II stores no ungroomed jet_mass (E0 schema probe), so it is
    reconstructed. Clamped at 0: E^2 - |p|^2 goes slightly negative in float32
    for near-massless jets, and an unclamped sqrt yields nan rather than 0.
    """
    px, py, pz = (jet_pt * np.cos(jet_phi), jet_pt * np.sin(jet_phi),
                  jet_pt * np.sinh(jet_eta))
    return _invmass(px, py, pz, np.asarray(jet_energy))


def audit_flags(pid, is_res_decay_prod, is_tau_decay_prod, is_res_x):
    """Measure the flag semantics this module assumes, for one jet.

    Returns counts the caller accumulates. Every assumption stated in the module
    docstring shows up here as a number rather than as a belief.
    """
    pid = np.abs(np.asarray(pid))
    rdp = np.asarray(is_res_decay_prod)
    tdp = np.asarray(is_tau_decay_prod)
    return {
        "n_res_decay_prod": int(rdp.sum()),
        "n_tau_decay_prod": int(tdp.sum()),
        "n_tau_in_res_decay_prod": int((rdp & (pid == TAU_PID)).sum()),
        "n_tau_decay_prod_also_res": int((tdp & rdp).sum()),
        "n_nu_in_res_decay_prod": int((rdp & np.isin(pid, NEUTRINO_PIDS)).sum()),
        "n_res_x": int(np.asarray(is_res_x).sum()),
        "n_visible": int(visible_mask(pid, rdp, tdp).sum()),
    }
