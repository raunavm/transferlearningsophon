"""experiments/AOJ/peak_fit.py on SYNTHETIC spectra -- the four ways the peak
search can lie, each pinned: a real peak missed, a peak from nothing, a peak
faked by a mass-correlated score, and a peak erased by the map itself."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aoj_peak_fit", ROOT / "experiments/AOJ/peak_fit.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

EFF = 0.01
SHAPES = dict(W=(80.0, 8.0), top=(173.0, 14.0))


def _pt(rng, n):
    return np.minimum(500.0 * (1 - rng.random(n)) ** (-1 / 4.5), 2499.0)   # falling, like data


def sample(seed, n_bkg=300_000, n_sig=0, peak="W", correlated=False):
    """QCD-like background flat in rho, plus a Gaussian resonance whose score
    piles up at 1 (1 - U^4: a deliberately MEDIOCRE tagger, true AUC 0.80).
    Returns only jets inside the rho window, as main() does."""
    rng = np.random.default_rng(seed)
    pt = np.r_[_pt(rng, n_bkg), _pt(rng, n_sig)]
    rho_b = rng.uniform(-6.5, -1.5, n_bkg)
    mass = np.r_[pt[:n_bkg] * np.exp(rho_b / 2), rng.normal(*SHAPES[peak], n_sig)]
    score = np.r_[rng.random(n_bkg), 1 - rng.random(n_sig) ** 4]
    if correlated:      # a tagger that simply likes heavy, hard jets -- background only
        score[:n_bkg] += 0.6 * (rho_b + 5.5) / 3.5 + 0.2 * np.log(pt[:n_bkg] / 500)
    is_sig = np.r_[np.zeros(n_bkg, bool), np.ones(n_sig, bool)]
    rho = pf.rho_of(mass, pt)
    ok = (rho > pf.RHO_RANGE[0]) & (rho < pf.RHO_RANGE[1]) & (mass > 20)
    return mass[ok], pt[ok], score[ok], is_sig[ok]


def cut_and_fit(mass, pt, score, peak="W", masked=pf.MASKED):
    passed = pf.passes(score, mass, pt, pf.build_map(score, mass, pt, EFF, masked=masked))
    fit, hist, _ = pf.fit_peak(mass, pt, passed, peak, *SHAPES[peak])
    return fit, hist, passed


def truth_in_fitted_bins(mass, pt, passed, is_sig, peak):
    """(signal passing, what the estimator measures) inside the bins the fit uses.

    Signal left in `fail` is absorbed by q and comes back multiplied by the
    BACKGROUND transfer factor (module docstring), so the estimator's target is
    S_pass - TF * S_fail, a few per cent below S_pass."""
    count = lambda sel: pf._bins(mass[sel], pt[sel], passed[sel], pf.PEAKS[peak]["fit_range"])
    sig, bkg = count(is_sig), count(~is_sig)
    tf = bkg["n_pass"].sum() / bkg["n_fail"].sum()
    return sig["n_pass"].sum(), sig["n_pass"].sum() - tf * sig["n_fail"].sum()


@pytest.mark.parametrize("peak,n_sig", [("W", 6000), ("top", 5000)])
def test_an_injected_peak_is_recovered_within_its_uncertainty(peak, n_sig):
    mass, pt, score, is_sig = sample(11, n_sig=n_sig, peak=peak)
    fit, hist, passed = cut_and_fit(mass, pt, score, peak)
    s_pass, expected = truth_in_fitted_bins(mass, pt, passed, is_sig, peak)
    assert abs(fit["signal_yield"] - expected) < 2.5 * fit["signal_yield_err"], (fit["signal_yield"], expected)
    assert abs(fit["signal_yield"] - s_pass) < 0.08 * s_pass, "the documented bias must stay small"
    assert 0.02 * s_pass < fit["signal_yield_err"] < 0.10 * s_pass, "an uncertainty, not a placeholder"
    assert fit["z_wald"] > 10 and fit["s_over_sqrt_b"] > 10
    assert fit["asymptotic_p"] > 0.01, "signal + background must describe the spectrum"
    assert hist["signal"].sum() == pytest.approx(fit["signal_yield"], rel=1e-6)


def test_no_signal_returns_a_yield_compatible_with_zero():
    pulls = []
    for seed in (21, 22, 23):
        fit, _, _ = cut_and_fit(*sample(seed)[:3])
        pulls.append(fit["signal_yield"] / fit["signal_yield_err"])
        assert abs(fit["s_over_sqrt_b"]) < 3
    assert max(map(abs, pulls)) < 3 and abs(np.mean(pulls)) < 1.5, pulls


def test_a_score_correlated_with_mass_does_not_fake_a_peak_after_the_map():
    mass, pt, score, _ = sample(31, correlated=True)
    window = pf.in_windows(mass, [pf.PEAKS["W"]["window"]])
    low, high = mass < pf.PEAKS["W"]["window"][0], mass > pf.PEAKS["W"]["window"][1]
    # teeth: a flat cut at the same overall efficiency sculpts the spectrum grossly
    flat = score > np.quantile(score, 1 - EFF)
    assert flat[high].mean() > 10 * max(flat[low].mean(), 1e-4)
    # after the map the efficiency is flat THROUGH the masked window ...
    fit, _, passed = cut_and_fit(mass, pt, score)
    assert passed[window].mean() == pytest.approx(EFF, rel=0.25)
    assert passed[low].mean() == pytest.approx(EFF, rel=0.25)
    assert passed[high].mean() == pytest.approx(EFF, rel=0.25)
    # ... and the fit finds nothing
    assert abs(fit["signal_yield"]) < 3 * fit["signal_yield_err"]
    assert fit["asymptotic_p"] > 0.01


def test_the_masked_map_does_not_erase_an_injected_peak_and_an_unmasked_one_does():
    mass, pt, score, is_sig = sample(41, n_sig=15_000)
    masked, _, passed = cut_and_fit(mass, pt, score)
    unmasked, _, passed_u = cut_and_fit(mass, pt, score, masked=())
    _, expected = truth_in_fitted_bins(mass, pt, passed, is_sig, "W")
    assert abs(masked["signal_yield"] - expected) < 2.5 * masked["signal_yield_err"]
    # the unmasked map raises the threshold exactly where the signal is
    assert (passed_u & is_sig).sum() < 0.85 * (passed & is_sig).sum()
    assert unmasked["signal_yield"] < 0.85 * masked["signal_yield"]


def test_the_map_is_built_from_jets_outside_the_mass_windows_only():
    mass, pt, score, _ = sample(51, n_bkg=150_000)
    poisoned = score.copy()
    poisoned[pf.in_windows(mass, pf.MASKED)] += 5.0          # anything at all inside the windows
    a, b = pf.build_map(score, mass, pt, EFF), pf.build_map(poisoned, mass, pt, EFF)
    np.testing.assert_allclose(a["coef"], b["coef"])


def test_only_bins_fully_inside_the_rho_window_are_fitted():
    mass, pt, score, _ = sample(61, n_bkg=100_000)
    b = pf._bins(mass, pt, score > 0.99, pf.PEAKS["top"]["fit_range"])
    m_lo, m_hi = b["m_edges"][b["i"]], b["m_edges"][b["i"] + 1]
    assert (pf.rho_of(m_lo, pf.PT_EDGES[b["j"] + 1]) > pf.RHO_RANGE[0]).all()
    assert (pf.rho_of(m_hi, pf.PT_EDGES[b["j"]]) < pf.RHO_RANGE[1]).all()
    assert m_hi[b["j"] == 0].max() <= 500 * np.exp(-1) , "at pT = 500 the top window is cut at 184 GeV"


def test_validation_fit_passes_on_background_and_the_f_test_starts_at_2_1():
    mass, pt, score, _ = sample(71, n_sig=5000)
    res, _ = pf.validation(score, mass, pt, "W", EFF, n_toys=40)
    assert res["f_test"][0]["order"] == (2, 1) and "signal_yield" not in res
    assert res["toy_p"] > 0.05 and res["band"] == pytest.approx([0.50, 0.51])


def test_auc_is_mann_whitney():
    s = np.array([0.1, 0.4, 0.35, 0.8]); y = np.array([False, False, True, True])
    assert pf.auc(s, y) == pytest.approx(0.75)
    assert pf.auc(s, np.ones(4, bool)) is None


def test_main_says_go_for_a_tagger_and_no_go_for_noise(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(81)
    parts = [sample(82, n_bkg=250_000), sample(83, n_bkg=0, n_sig=6000), sample(84, n_bkg=0, n_sig=5000, peak="top")]
    mass, pt = (np.concatenate([p[k] for p in parts]) for k in (0, 1))
    kind = np.repeat([0, 1, 2], [len(p[0]) for p in parts])
    # A GOOD tagger, Gaussian in log-odds (true AUC 0.98), so the probability the
    # "CMS" score is stored as is not piled within one float16 step of 1. The proxy
    # label is impure -- reference `pass` holds background, reference `fail` holds
    # the untagged signal -- so even this tagger reads well below its true AUC
    # against it, and the 1 - U^4 tagger above reads 0.72.
    tag = lambda k: rng.normal(np.where(kind == k, 3.0, -3.0), 2.0)
    prob = lambda x: (1 / (1 + np.exp(-x))).astype(np.float16)
    np.savez(tmp_path / "jets.npz", jet_sdmass=mass.astype(np.float32), aoj_jet_pt=pt.astype(np.float32),
             aoj_pn_WvsQCD=prob(tag(1)), aoj_pn_TvsQCD=prob(tag(2)))
    np.savez(tmp_path / "good.npz", two_prong_logodds=tag(1).astype(np.float16),
             three_prong_logodds=tag(2).astype(np.float16))
    np.savez(tmp_path / "noise.npz", two_prong_logodds=rng.normal(size=len(kind)).astype(np.float16),
             three_prong_logodds=rng.normal(size=len(kind)).astype(np.float16))
    (tmp_path / "closure.json").write_text(json.dumps(dict(hard_flags=[])))
    monkeypatch.setattr(sys, "argv", ["peak_fit.py", "--jets", str(tmp_path / "jets.npz"), "--toys", "20",
                                      "--closure", str(tmp_path / "closure.json"), "--out", str(tmp_path),
                                      "--scores", f"good={tmp_path/'good.npz'}", f"noise={tmp_path/'noise.npz'}"])
    assert pf.main() == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["pipeline_ok"] and res["verdict"] == dict(good="GO", noise="NO-GO"), capsys.readouterr().out
    assert 75 < res["reference"]["W"]["mean"] < 90 and 160 < res["reference"]["top"]["mean"] < 185
    good = res["models"]["good"]["W"]
    assert good["mean"] == res["reference"]["W"]["mean"] and not good["shape_floated"]
    assert 0.5 < good["efficiency_relative_to_reference"] < 1.5 and good["auc_vs_cms_proxy"] > 0.8
    assert not res["models"]["noise"]["W"]["criteria"]["s_over_sqrt_b"]
    assert "good_W_n_pass" in np.load(tmp_path / "histograms.npz").files


def test_a_hard_closure_flag_vetoes_go_and_a_failed_reference_outranks_both():
    good = dict(W=dict(criteria=dict(peak_position=True, auc=True)))
    bad = dict(W=dict(criteria=dict(peak_position=True, auc=False)))
    assert pf.decide(good, [], True) == ("GO", [])
    assert pf.decide(bad, [], True) == ("NO-GO", ["W:auc"])
    assert pf.decide(good, ["unit:part_d0 median ratio 9.8"], True)[0] == "NO-GO"
    assert pf.decide(good, [], False)[0] == pf.decide(bad, [], False)[0] == "PIPELINE-INVALID"
