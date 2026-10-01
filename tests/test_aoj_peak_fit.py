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
    # every score floats its own shape, starting from the reference's; the peak position
    # criterion reads the fit the yield comes from, not a second fit at START_ORDER
    assert good["shape_floated"] and good["floated_mean"] == good["mean"] and 75 < good["mean"] < 90
    assert good["shape_start"] == [res["reference"]["W"]["mean"], res["reference"]["W"]["width"]]
    var = res["shape_variations"]["W"]
    assert var["pool"] == ["good", "noise"] and var["old_reference"] == res["reference"]["W"]["shape_start"]
    assert set(good["shape_variations"]) == {"pooled", "old_reference"}
    assert all(v["delta_deviance_vs_fitted_shape"] > -1e-3 for v in good["shape_variations"].values())
    for f in (res["reference"]["W"], good, res["models"]["noise"]["W"]):
        assert f["profile_error_ok"] or f["signal_yield_err"] == f["signal_yield_err_hessian"]
    assert 0.5 < good["efficiency_relative_to_reference"] < 1.5 and good["auc_vs_cms_proxy"] > 0.8
    # the working point is 1 % of the SIDEBAND jets; the all-jet efficiency is recorded beside it
    for f in (res["reference"]["W"], good, res["models"]["noise"]["W"]):
        assert f["data_efficiency_sidebands"] == pytest.approx(EFF, abs=1e-3)
        assert 0.0 <= f["data_efficiency_top_window"] <= 1.0 and 0.0 < f["data_efficiency"] < 1.0
    assert not res["models"]["noise"]["W"]["criteria"]["s_over_sqrt_b"]
    assert "good_W_n_pass" in np.load(tmp_path / "histograms.npz").files


def test_the_top_peak_alone_is_fitted_from_a_three_prong_only_score_file(tmp_path, monkeypatch):
    """The full run writes no two-prong score (the W channel is withdrawn), so the
    fitter must run on the top peak alone and never ask for the W inputs. And by
    default the published checkpoint is fitted but not pooled into the shape systematic."""
    rng = np.random.default_rng(91)
    parts = [sample(92, n_bkg=250_000), sample(93, n_bkg=0, n_sig=5000, peak="top")]
    mass, pt = (np.concatenate([p[k] for p in parts]) for k in (0, 1))
    kind = np.repeat([0, 1], [len(p[0]) for p in parts])
    tag = rng.normal(np.where(kind == 1, 3.0, -3.0), 2.0)
    np.savez(tmp_path / "jets.npz", jet_sdmass=mass.astype(np.float32), aoj_jet_pt=pt.astype(np.float32),
             aoj_pn_TvsQCD=(1 / (1 + np.exp(-tag))).astype(np.float16))
    np.savez(tmp_path / "m.npz", three_prong_logodds=tag.astype(np.float16))
    monkeypatch.setattr(sys, "argv", ["peak_fit.py", "--jets", str(tmp_path / "jets.npz"), "--toys", "0",
                                      "--out", str(tmp_path), "--peaks", "top",
                                      "--scores", f"m={tmp_path/'m.npz'}", f"{pf.PUBLISHED}={tmp_path/'m.npz'}"])
    assert pf.main() == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["peaks"] == ["top"] and list(res["reference"]) == ["top"]
    assert list(res["models"]["m"]) == ["top"] and res["verdict"]["m"] == "GO"
    assert pf.PUBLISHED in res["models"] and res["shape_variations"]["top"]["pool"] == ["m"]


def _top_bins(seed, n_sig=5000):
    mass, pt, score, is_sig = sample(seed, n_sig=n_sig, peak="top")
    passed = pf.passes(score, mass, pt, pf.build_map(score, mass, pt, EFF))
    return pf._bins(mass, pt, passed, pf.PEAKS["top"]["fit_range"]), (mass, pt, passed, is_sig)


def test_the_shape_floats_to_the_injected_peak_from_a_wrong_start_and_the_start_does_not_matter():
    """Started from a shape 7 GeV high and 3 GeV too wide -- as the reference's was for
    every real score -- the fit must find the injected peak and end where a start at
    the truth ends."""
    b, (mass, pt, passed, is_sig) = _top_bins(12)
    wrong = pf.fit_binned(b, "top", 180.0, 17.3, float_shape=True)[0]
    right = pf.fit_binned(b, "top", *SHAPES["top"], float_shape=True)[0]
    assert wrong["profile_error_ok"] and not wrong["width_at_bound"] and wrong["shape_start"] == [180.0, 17.3]
    assert abs(wrong["mean"] - 173.0) < 1.5 and abs(wrong["width"] - 14.0) < 1.5
    assert wrong["tf_order"] == right["tf_order"]
    assert max(abs(wrong["mean"] - right["mean"]), abs(wrong["width"] - right["width"])) < 0.1
    assert abs(wrong["signal_yield"] - right["signal_yield"]) < 0.05 * right["signal_yield_err"]
    _, expected = truth_in_fitted_bins(mass, pt, passed, is_sig, "top")
    assert abs(wrong["signal_yield"] - expected) < 2.5 * wrong["signal_yield_err"]
    assert wrong["signal_yield_err"] >= wrong["signal_yield_err_hessian"], "profiling the shape cannot narrow it"


def test_the_quoted_error_is_where_the_deviance_profiled_over_tf_and_shape_rises_by_one():
    """Re-derived with a different minimiser over the shape (Nelder-Mead from an offset
    start). And the fixed-yield fit it rests on gives, at a FIXED shape, the Hessian
    error -- the fixed-shape profile error of fit_minimum_diagnostic."""
    from scipy import optimize
    b, _ = _top_bins(13)
    fit, _, (model, x) = pf.fit_binned(b, "top", *SHAPES["top"], float_shape=True)
    order, tf_norm = tuple(fit["tf_order"]), model.tf_norm
    y0 = fit["signal_yield"]

    def prof(y):
        at = lambda v: pf._Model(b, order, tf_norm, v[0], v[1]).fit_at_yield(y, x)[1]
        return optimize.minimize(at, (fit["mean"] + 0.7, fit["width"] - 0.7), method="Nelder-Mead",
                                 options=dict(xatol=1e-4, fatol=1e-10)).fun
    f0 = prof(y0)
    assert abs(f0 - fit["deviance"] / 2) < 1e-4, "the fitted shape is the profile's minimum"
    for sign, e in ((-1, fit["signal_yield_err_lo"]), (1, fit["signal_yield_err_hi"])):
        assert 2 * (prof(y0 + sign * e) - f0) == pytest.approx(1.0, abs=0.01)
    h = fit["signal_yield_err_hessian"]
    fixed = [2 * (model.fit_at_yield(y0 + d, x)[1] - model.loss(x)[0]) for d in (-h, h)]
    assert fixed == pytest.approx([1.0, 1.0], abs=0.05)


def test_the_order_is_chosen_by_f_tests_between_fits_each_maximised_over_its_own_shape():
    """Re-derived from the trail: every order the F-test visits sits at a minimum of the
    loss in (mean, width) AT THAT ORDER, its deviance is that minimum, and each p-value is
    the F-test between those maximised likelihoods with the two shape parameters in the
    parameter count of both models."""
    from scipy import stats
    b, _ = _top_bins(15)
    tf_norm, window = pf._tf_norm(b, pf.PEAKS["top"]["window"]), pf.PEAKS["top"]["window"]
    order, shape, trail = pf._choose_shape_and_order(b, tf_norm, window, [(180.0, 17.3)])
    assert len(trail) >= 3, "the fixture must visit more than one candidate order"
    dev = lambda o, m, w: 2 * pf._Model(b, tuple(o), tf_norm, m, w).fit()[1]
    for t in trail:
        assert dev(t["order"], t["mean"], t["width"]) == pytest.approx(t["deviance"], abs=1e-6)
        for dm, dw in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
            assert dev(t["order"], t["mean"] + dm, t["width"] + dw) > t["deviance"] - 1e-6, t
    n_par = lambda o: (o[0] + 1) * (o[1] + 1)
    n_other = len(np.unique(b["j"])) + 2
    by_order = {tuple(t["order"]): t for t in trail}
    for t in trail[1:]:
        if "f_test_p" not in t:
            continue
        parent = [by_order[p] for p in ((t["order"][0] - 1, t["order"][1]), (t["order"][0], t["order"][1] - 1))
                  if p in by_order and "admissible" not in by_order[p]]
        dof = len(b["n_pass"]) - n_par(t["order"]) - n_other
        ps = [stats.f.sf(((p["deviance"] - t["deviance"]) / (n_par(t["order"]) - n_par(p["order"])))
                         / (t["deviance"] / dof), n_par(t["order"]) - n_par(p["order"]), dof) for p in parent]
        assert any(p == pytest.approx(t["f_test_p"], rel=1e-9) for p in ps), t
    chosen = by_order[tuple(order)]
    assert (chosen["mean"], chosen["width"]) == shape


def test_the_order_and_shape_do_not_depend_on_where_the_shape_search_starts():
    b, _ = _top_bins(16)
    tf_norm, window = pf._tf_norm(b, pf.PEAKS["top"]["window"]), pf.PEAKS["top"]["window"]
    answers = [pf._choose_shape_and_order(b, tf_norm, window, [s])[:2] for s in ((150.0, 25.0), (190.0, 8.0))]
    assert answers[0][0] == answers[1][0]
    assert np.allclose(answers[0][1], answers[1][1], atol=0.1)


def test_background_only_spectra_fitted_with_a_floating_shape_show_no_significant_peak():
    """The shape free over the whole window can put a Gaussian on any upward fluctuation
    (look-elsewhere); on background alone the yield must still be within 3 of its error,
    or the fit must say it has no profile error."""
    for seed in (101, 102, 103, 104):
        mass, pt, score, _ = sample(seed, n_bkg=200_000)
        passed = pf.passes(score, mass, pt, pf.build_map(score, mass, pt, EFF))
        fit, _, _ = pf.fit_peak(mass, pt, passed, "top", *SHAPES["top"], float_shape=True)
        assert abs(fit["signal_yield"] / fit["signal_yield_err"]) < 3 or not fit["profile_error_ok"], (seed, fit)


def test_a_profile_error_whose_crossing_search_ends_short_of_tolerance_is_not_quoted(monkeypatch):
    b, _ = _top_bins(13)
    monkeypatch.setattr(pf, "MAX_CROSSING_STEPS", 1)
    monkeypatch.setattr(pf, "CROSSING_TOL", 1e-12)
    fit = pf.fit_binned(b, "top", *SHAPES["top"], float_shape=True)[0]
    assert not fit["profile_error_ok"] and fit["signal_yield_err_lo"] is None
    assert fit["signal_yield_err"] == fit["signal_yield_err_hessian"]


def test_a_width_on_its_bound_is_flagged(monkeypatch):
    """A peak narrower than the 3 GeV floor: the width stops there, and the fit says so."""
    monkeypatch.setitem(SHAPES, "top", (173.0, 1.0))
    b, _ = _top_bins(14)
    fit = pf.fit_binned(b, "top", 173.0, 8.0, float_shape=True)[0]
    assert fit["width_at_bound"] and fit["width"] < pf.WIDTH_BOUNDS[0] + 0.05


def test_a_hard_closure_flag_vetoes_go_and_a_failed_reference_outranks_both():
    good = dict(W=dict(criteria=dict(peak_position=True, auc=True)))
    bad = dict(W=dict(criteria=dict(peak_position=True, auc=False)))
    assert pf.decide(good, [], True) == ("GO", [])
    assert pf.decide(bad, [], True) == ("NO-GO", ["W:auc"])
    assert pf.decide(good, ["unit:part_d0 median ratio 9.8"], True)[0] == "NO-GO"
    assert pf.decide(good, [], False)[0] == pf.decide(bad, [], False)[0] == "PIPELINE-INVALID"


def _leaky(seed, n_bkg=600_000, n_sig=40_000, eps=0.05):
    """Top-like signal that a weak tagger mostly FAILS: a fraction eps of it scores like
    signal, the rest like background -- as these taggers do on data tops."""
    rng = np.random.default_rng(seed)
    pt = np.r_[_pt(rng, n_bkg), _pt(rng, n_sig)]
    mass = np.r_[pt[:n_bkg] * np.exp(rng.uniform(-6.5, -1.5, n_bkg) / 2), rng.normal(*SHAPES["top"], n_sig)]
    tagged = rng.random(n_sig) < eps
    score = np.r_[rng.random(n_bkg), np.where(tagged, 1 - rng.random(n_sig) ** 4, rng.random(n_sig))]
    is_sig = np.r_[np.zeros(n_bkg, bool), np.ones(n_sig, bool)]
    rho = pf.rho_of(mass, pt)
    ok = (rho > pf.RHO_RANGE[0]) & (rho < pf.RHO_RANGE[1]) & (pt > pf.PT_RANGE[0]) & (pt < pf.PT_RANGE[1])
    mass, pt, score, is_sig = mass[ok], pt[ok], score[ok], is_sig[ok]
    passed = pf.passes(score, mass, pt, pf.build_map(score, mass, pt, EFF))
    b = pf._bins(mass, pt, passed, pf.PEAKS["top"]["fit_range"])
    h = lambda sel: np.histogram2d(mass[sel], pt[sel], bins=(b["m_edges"], pf.PT_EDGES))[0][b["i"], b["j"]]
    return b, h(is_sig), h(is_sig & passed)


def test_tops_failing_the_cut_bias_the_pass_only_fit_low_and_the_fit_given_the_tops_recovers_them():
    """The fail region's tops are absorbed by q and come back as TF * F in the pass
    background (_Model). Given the tops in each bin, the fit recovers the passing ones."""
    b, tops, tops_pass = _leaky(31)
    s_pass = tops_pass.sum()
    blind = pf.fit_binned(b, "top", *SHAPES["top"])[0]
    told = pf.fit_binned(b, "top", *SHAPES["top"], tops=tops)[0]
    tf = pf._tf_norm(b, pf.PEAKS["top"]["window"])
    assert s_pass - blind["signal_yield"] > 3 * blind["signal_yield_err"], "the fixture must show the leak"
    assert blind["signal_yield"] == pytest.approx(s_pass - tf * (tops - tops_pass).sum(), abs=2.5 * blind["signal_yield_err"])
    assert abs(told["signal_yield"] - s_pass) < 2.5 * told["signal_yield_err"], (told["signal_yield"], s_pass)
    floated = pf.fit_binned(b, "top", 180.0, 17.3, float_shape=True, tops=tops)[0]
    assert abs(floated["signal_yield"] - s_pass) < 2.5 * floated["signal_yield_err"]


def test_the_fit_given_tops_has_the_gradient_of_its_loss_and_without_tops_is_the_pass_only_fit():
    b, tops, _ = _leaky(32, n_bkg=200_000, n_sig=10_000)
    m0 = pf._Model(b, (2, 1), pf._tf_norm(b, pf.PEAKS["top"]["window"]), *SHAPES["top"])
    m1 = pf._Model(b, (2, 1), m0.tf_norm, *SHAPES["top"], tops=tops)
    x, _ = m1.fit()
    g = m1.loss(x * 1.01)[1]
    for k in (0, m1.n_tf + 1):
        d = np.zeros_like(x); d[k] = 1e-6 * max(1.0, abs(x[k]))
        num = (m1.loss(x * 1.01 + d)[0] - m1.loss(x * 1.01 - d)[0]) / (2 * d[k])
        assert g[k] == pytest.approx(num, rel=1e-3, abs=1e-6)
    zero = pf._Model(b, (2, 1), m0.tf_norm, *SHAPES["top"], tops=np.zeros(len(b["n_pass"])))
    assert zero.loss(x)[0] == pytest.approx(m0.loss(x)[0], rel=1e-12)


def test_analyse_at_a_given_shape_holds_it_and_chooses_the_order_by_f_test():
    mass, pt, score, _ = sample(51, n_sig=5000, peak="top")
    fit, _, _ = pf.analyse(score, mass, pt, "top", EFF, 0, shape=SHAPES["top"], float_shape=False)
    assert (fit["mean"], fit["width"]) == SHAPES["top"] and not fit["shape_floated"]
    assert fit["f_test"] and fit["floated_mean"] == SHAPES["top"][0]
