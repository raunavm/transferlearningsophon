"""experiments/AOJ/injection_test.py: the injection test's bins must be the bins the
checks and the fit build, from the jets the fit saw, or nothing is written."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("injection_test", ROOT / "experiments/AOJ/injection_test.py")
IT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IT)
P = IT.P


def _merged(tmp_path, n=400_000, seed=3):
    """A merged run of two scores: QCD-like jets over the rho window, a top-like peak."""
    rng = np.random.default_rng(seed)
    pt = np.minimum(500.0 * (1 - rng.random(n)) ** (-1 / 4.5), 2499.0)
    rho = rng.uniform(-6.0, -1.8, n)
    mass = pt * np.exp(rho / 2)
    sig = rng.random(n) < 0.01
    mass[sig] = rng.normal(175.0, 12.0, sig.sum())
    d = tmp_path / "merged"
    d.mkdir()
    pn = np.where(sig, 1 - rng.random(n) ** 4, rng.random(n))
    np.savez(d / "jets.npz", jet_sdmass=mass.astype(np.float32), aoj_jet_pt=pt.astype(np.float32),
             aoj_pn_TvsQCD=pn.astype(np.float32))
    np.savez(d / "scores_m.npz", three_prong_logodds=P.logit(np.where(sig, 1 - rng.random(n) ** 3,
                                                                      rng.random(n))).astype(np.float16))
    return d


def _committed(tmp_path, merged, change=False):
    j = np.load(merged / "jets.npz")
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    out = {}
    for name, z in (("reference", P.logit(j["aoj_pn_TvsQCD"])),
                    ("m", np.load(merged / "scores_m.npz")["three_prong_logodds"].astype(float))):
        b = IT.top_bins(z[ok], mass[ok], pt[ok], IT.EFF)
        if change and name == "m":
            b = dict(b, n_pass=b["n_pass"] + 1)
        out.update({f"{name}|main|{k}": b[k] for k in IT.KEYS})
    np.savez(tmp_path / "committed.npz", **out)
    return tmp_path / "committed.npz"


def test_the_bins_are_the_pseudo_window_and_extra_working_points_of_every_score(tmp_path):
    merged = _merged(tmp_path)
    IT.export(merged, _committed(tmp_path, merged), tmp_path / "out.npz", workers=1)
    z = np.load(tmp_path / "out.npz")
    parts = {k.split("|")[1] for k in z.files}
    assert {k.split("|")[0] for k in z.files} == {"reference", "m"}
    assert parts == {"pseudo", *(f"main_eff{e:g}" for e in IT.EXTRA_EFF)}
    for name in ("reference", "m"):
        ps = z[f"{name}|pseudo|m_edges"]
        assert ps[0] == IT.PSEUDO["fit_range"][0] and ps[-1] == IT.PSEUDO["fit_range"][1]
        lo, hi = z[f"{name}|main_eff0.005|n_pass"].sum(), z[f"{name}|main_eff0.02|n_pass"].sum()
        assert 1.5 * lo < hi < 6 * lo, "four times the data efficiency: more passing jets, at most ~4x"


def test_nothing_is_written_when_the_one_per_cent_bins_are_not_the_committed_ones(tmp_path):
    merged = _merged(tmp_path)
    with pytest.raises(SystemExit, match="these are not the jets the fit saw"):
        IT.export(merged, _committed(tmp_path, merged, change=True), tmp_path / "out.npz", workers=1)
    assert not (tmp_path / "out.npz").exists()
