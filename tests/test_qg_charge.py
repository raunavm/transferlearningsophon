"""The EnergyFlow q/g staging assigned inverted charge to electrons and muons.

part_charge is an input feature of configs/finetune/EnergyFlowQG.yaml, so this
was a defect in the staged training data, not only in a summary.

PDG sign convention is not uniform across the five charged species the staging
recognises. For 211 (pi+), 321 (K+) and 2212 (p) the positive id carries
positive charge. For the leptons it is inverted: 11 is the ELECTRON (charge
-1), 13 the mu- (charge -1). sign(pdgid) alone is therefore right for the
hadrons and wrong for every lepton constituent.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "stage_downstream", ROOT / "scripts" / "stage_downstream.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXPECTED = {
    211: +1, -211: -1,      # pi+ / pi-
    321: +1, -321: -1,      # K+ / K-
    2212: +1, -2212: -1,    # p / pbar
    11: -1, -11: +1,        # e- / e+   <- the inverted pair
    13: -1, -13: +1,        # mu- / mu+ <- the inverted pair
    22: 0, 130: 0, 2112: 0,  # photon, K0L, neutron
}


@pytest.mark.parametrize("pid,want", sorted(EXPECTED.items()))
def test_charge_matches_pdg(tmp_path, pid, want):
    sd = _load()
    n_const = 4
    pids = np.zeros((1, n_const), dtype=np.float64)
    pids[0, 0] = pid
    X = np.zeros((1, n_const, 4))
    X[..., 0] = 0.0
    X[0, 0, 0] = 10.0            # only the first constituent has pt > 0
    X[..., 3] = pids
    path = tmp_path / "qg.npz"
    np.savez(path, X=X, y=np.array([1]))
    out = sd.load_qg(path)
    charge = out[-1]["part_charge"] if isinstance(out[-1], dict) else None
    if charge is None:                       # positional return
        charge = [a for a in out if hasattr(a, "__len__")][-1]
    got = float(np.asarray(charge[0].to_list() if hasattr(charge[0], "to_list")
                           else charge[0])[0])
    assert got == want, f"PDG {pid}: charge {got:+.0f}, PDG says {want:+d}"


def test_leptons_are_not_signed_like_hadrons():
    """The specific inversion, stated directly so it cannot silently return."""
    for pid in (11, 13):
        assert EXPECTED[pid] == -1, "PDG 11 and 13 are the NEGATIVE leptons"
        assert np.sign(pid) == +1, "plain sign(pdgid) would give +1"
