"""AD as a vocabulary ablation (docs/PRD_PLAN.md 4.3).

Every metric here is defined in a paper, so each test checks the implementation
against the DEFINITION rather than against itself.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    s = importlib.util.spec_from_file_location(
        "anomaly", ROOT / "experiments" / "EVAL" / "anomaly.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


an = _load()


def test_stat_cut_is_the_papers_twenty_percent():
    """arXiv:2604.20965: thresholds whose relative stat error on eps_B < 20%.

    Relative Poisson error is 1/sqrt(n), so the floor is n > 25. A wrong floor
    here silently turns max SIC into a fluctuation finder.
    """
    assert an.STAT_CUT == 0.20
    assert an.MIN_BKG_PASS == 25
    assert 1.0 / np.sqrt(an.MIN_BKG_PASS) <= an.STAT_CUT


def test_sic_curve_never_uses_a_threshold_below_the_floor():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(500), np.zeros(5000)])
    s = np.concatenate([rng.normal(2, 1, 500), rng.normal(0, 1, 5000)])
    eps_s, eps_b, sic = an.sic_curve(y, s)
    n_b = int((1 - y).sum())
    assert (eps_b * n_b >= an.MIN_BKG_PASS).all()
    assert np.allclose(sic, eps_s / np.sqrt(eps_b))


def test_sic_is_one_when_the_score_is_pure_noise():
    """The null. A score carrying no signal information has SIC ~ 1."""
    rng = np.random.default_rng(1)
    y = np.concatenate([np.ones(2000), np.zeros(20000)])
    s = rng.normal(0, 1, y.size)          # independent of y
    _, _, sic = an.sic_curve(y, s)
    assert 0.7 < float(np.median(sic)) < 1.4


def test_sigma_min_inverts_max_sic_against_the_target():
    """sigma_min = sigma_t / max SIC, from max(S)(sigma_min) = sigma_t."""
    eps_s = np.array([0.5, 0.4, 0.3])
    eps_b = np.array([0.25, 0.16, 0.09])   # SIC = 1.0, 1.0, 1.0
    assert an.sigma_min(eps_s, eps_b, 1000) == pytest.approx(an.SIGMA_T / 1.0)
    eps_b2 = eps_b / 4                      # SIC doubles
    assert an.sigma_min(eps_s, eps_b2, 1000) == pytest.approx(an.SIGMA_T / 2.0)


def test_argos_matches_its_published_formula():
    """ARGOS = eps_SR/sqrt(eps_BT) - sqrt(eps_BT)  (arXiv:2511.14832).

    Built so the maximising threshold is known: the data sample is shifted, the
    template is not, so ARGOS must be positive and reproduce the formula at the
    threshold it returns.
    """
    rng = np.random.default_rng(2)
    s_data = np.concatenate([rng.normal(0, 1, 9000), rng.normal(4, 1, 1000)])
    s_tmpl = rng.normal(0, 1, 20000)
    a, thr = an.argos(s_data, s_tmpl)
    e_sr = float((s_data >= thr).mean())
    e_bt = float((s_tmpl >= thr).mean())
    assert a == pytest.approx(e_sr / np.sqrt(e_bt) - np.sqrt(e_bt), rel=1e-9)
    assert a > 0, "an injected excess must give ARGOS above the random baseline"


def test_argos_is_signal_blind():
    """It must be computable from (data, template) alone -- no truth labels.

    Checked by signature: passing only the two score arrays has to suffice.
    """
    rng = np.random.default_rng(3)
    out = an.argos(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000))
    assert out is None or isinstance(out, tuple)


def test_node_roles_reproduce_every_arm_width():
    """resonant + pure-QCD nodes must equal the arm's head width.

    188 = 161 + 27, 162 = 161 + 1, 43 = 42 + 1, 17 = 16 + 1 -- an independent
    check on the committed map, since these widths are ground truth.
    """
    for rung, k in (("L188", 188), ("L162", 162), ("R42_Q1", 43), ("R16_Q1", 17)):
        _, res, qcd = an.node_roles(rung)
        assert len(res) + len(qcd) == k, f"{rung}: {len(res)}+{len(qcd)} != {k}"
    _, res188, qcd188 = an.node_roles("L188")
    assert (len(res188), len(qcd188)) == (161, 27), "the released vocabulary"


def test_a_node_mixing_qcd_and_resonant_joins_neither_sum():
    """Putting a mixed node in either sum would define the score to include
    the thing it discriminates against."""
    _, res, qcd = an.node_roles("R1_Q1")
    assert not (res & qcd)


def test_class_sum_leaves_the_signal_node_out():
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 1, (100, 188))
    _, res, _ = an.node_roles("L188")
    node = sorted(res)[0]
    s_with = an.score_class_sum(logits, "L188", sig_node=-1)
    s_without = an.score_class_sum(logits, "L188", sig_node=node)
    assert not np.allclose(s_with, s_without), "the signal node must be excluded"


def test_chi2_sculpting_is_flat_for_an_unbiased_cut_and_large_for_a_mass_cut():
    rng = np.random.default_rng(5)
    m = rng.normal(100, 20, 20000)
    flat = rng.random(m.size) < 0.3                 # independent of m
    sculpt = m > 110                                # cuts directly on m
    c_flat = an.chi2_sculpting(m[flat], m)
    c_sculpt = an.chi2_sculpting(m[sculpt], m)
    assert c_flat < 5, f"an m-independent cut must not sculpt (got {c_flat})"
    assert c_sculpt > c_flat * 10, "a cut on m itself must show up loudly"


def test_signal_suite_spans_two_three_and_four_prongs():
    """PRD_PLAN 4.3 requires >= 6 classes over 2/3/4 prongs, fixed in advance."""
    assert len(an.SIGNAL_SUITE) >= 6
    two = [s for s in an.SIGNAL_SUITE if s.count("_") == 2]
    three = [s for s in an.SIGNAL_SUITE if "_YY_" in s and len(s.split("_")[-1]) == 3]
    four = [s for s in an.SIGNAL_SUITE if "_YY_" in s and len(s.split("_")[-1]) == 4]
    assert two and three and four, "all three prong classes must appear"
    assert 0 in an.N_SIG_SCAN, "the N_sig -> 0 null is required"


def test_ten_trainings_is_the_default():
    assert an.N_TRAININGS >= 10


def _cache(d, n, labels, rng, k, with_logits=True, signal_boost=None):
    d.mkdir(parents=True, exist_ok=True)
    F = rng.normal(size=(n, 16)).astype(np.float32)
    if signal_boost is not None:
        F[signal_boost] += 3.0            # a genuinely separable signal
    np.save(d / "features.npy", F)
    np.save(d / "label188.npy", labels.astype(np.int16))
    if with_logits:
        lg = rng.normal(size=(n, k)).astype(np.float32)
        np.save(d / "logits.npy", lg)
    np.savez(d / "observers.npz",
             jet_pt=rng.uniform(200, 2500, n).astype(np.float32),
             jet_sdmass=rng.uniform(20, 500, n).astype(np.float32),
             jet_eta=rng.uniform(-2.5, 2.5, n).astype(np.float32))
    (d / "extract_manifest.json").write_text('{"arm": "t", "checkpoint": "x"}')


def test_main_runs_end_to_end_and_writes_finite_metrics(tmp_path):
    """Nothing else here drives main(); an unexercised driver is an untested one."""
    import json as _json
    rng = np.random.default_rng(7)
    qcd = sorted(an._probe().qcd_indices())
    sig_lab = 0                                  # label_X_bb
    n = 6000
    lab = np.array(rng.choice(qcd, size=n))
    sidx = np.arange(0, 900)
    lab[sidx] = sig_lab
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, k=188, signal_boost=sidx)
    out = tmp_path / "ad"
    an.main(["--features", f"a={d}", "--rungs", "a=L188", "--out", str(out),
             "--n-bkg", "1500", "--n-template", "1500", "--trainings", "2",
             "--signals", "label_X_bb", "--n-sig", "0", "400"])
    res = _json.loads((out / "anomaly_results.json").read_text())
    per_n = res["arms"]["a"]["signals"]["label_X_bb"]
    assert set(per_n) == {"0", "400"}
    fams = {k for k, v in per_n["400"].items() if isinstance(v, dict) and "max_sic" in v}
    assert {"knn", "mahalanobis", "iad_hgb"} <= fams, f"got {fams}"
    for fam in fams:
        v = per_n["400"][fam]
        assert np.isfinite(v["max_sic"]) and v["max_sic"] > 0
        assert np.isfinite(v["sigma_min"])
        assert "regret" in v and v["regret"] >= 1.0 - 1e-9, "regret is >= 1 by definition"
    # the null carries the signal-blind quantities and NO SIC -- there is no
    # eps_S to build one from, which is exactly why the guard reads ARGOS
    assert per_n["0"]["knn"].get("null") is True
    assert "max_sic" not in per_n["0"]["knn"]
    assert "argos" in per_n["0"]["knn"], "the null must still measure ARGOS"


def test_saturated_scores_are_flagged_not_reported_as_agreement(tmp_path):
    """A trivially separable signal drives every score to the same ceiling.

    max SIC <= sqrt(n_B / MIN_BKG_PASS). Three methods landing on one value is
    clipping, not consensus, and the run must say so rather than let it read as
    'the vocabulary made no difference'.
    """
    import json as _json
    rng = np.random.default_rng(11)
    qcd = sorted(an._probe().qcd_indices())
    n = 6000
    lab = np.array(rng.choice(qcd, size=n))
    sidx = np.arange(0, 900)
    lab[sidx] = 0
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, k=188, signal_boost=sidx)     # +3 sigma in every dim
    out = tmp_path / "ad"
    an.main(["--features", f"a={d}", "--rungs", "a=L188", "--out", str(out),
             "--n-bkg", "1500", "--n-template", "1500", "--trainings", "1",
             "--signals", "label_X_bb", "--n-sig", "400"])
    fams = _json.loads((out / "anomaly_results.json").read_text())
    v = fams["arms"]["a"]["signals"]["label_X_bb"]["400"]["knn"]
    assert v["max_sic"] == pytest.approx(np.sqrt(1500 / an.MIN_BKG_PASS), rel=0.02)
    assert v["at_ceiling"] is True, "a clipped score must be flagged"


def test_class_sum_is_absent_without_logits_rather_than_crashing(tmp_path):
    """A cache with no logits still yields the vocabulary-FREE scores.

    That asymmetry is the measurement: the vocabulary-defined score is the one
    that stops existing, and it must do so as a recorded absence.
    """
    import json as _json
    rng = np.random.default_rng(8)
    qcd = sorted(an._probe().qcd_indices())
    n = 4000
    lab = np.array(rng.choice(qcd, size=n))
    lab[:600] = 0
    d = tmp_path / "arm"
    _cache(d, n, lab, rng, k=17, with_logits=False, signal_boost=np.arange(600))
    out = tmp_path / "ad"
    an.main(["--features", f"a={d}", "--rungs", "a=R16_Q1", "--out", str(out),
             "--n-bkg", "1200", "--n-template", "1200", "--trainings", "1",
             "--signals", "label_X_bb", "--n-sig", "300"])
    per_n = _json.loads((out / "anomaly_results.json").read_text())
    fams = per_n["arms"]["a"]["signals"]["label_X_bb"]["300"]
    assert "class_sum" not in fams
    assert "knn" in fams and "max_sic" in fams["knn"]
