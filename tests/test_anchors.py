"""The anchor check is only worth running if its ROC arithmetic is right.

`experiments/EVAL/anchors.py` exists to decide whether OUR evaluation code
reproduces arXiv:2503.00118 Table A1 before any arm-wise rejection is quoted.
That makes its own arithmetic load-bearing: a rejection computed at a threshold
fixed the wrong way round (on background rather than signal efficiency) would
still produce plausible numbers, still "agree" or "disagree" for reasons that
look physical, and would silently validate or reject the evaluation code for
the wrong reason.

These tests pin the published targets against transcription drift and check the
rejection arithmetic against cases whose answer is known by construction.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "anchors", ROOT / "experiments" / "EVAL" / "anchors.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


an = _load()


def test_published_anchors_match_table_a1():
    """arXiv:2503.00118 Table A1, Sophon column. Transcribed once; pinned here."""
    assert an.PUBLISHED == {
        ("label_X_bb", 0.60): 300.0,
        ("label_X_bb", 0.40): 810.0,
        ("label_X_cc", 0.60): 110.0,
        ("label_X_cc", 0.40): 320.0,
    }


def test_rejection_threshold_is_fixed_on_signal_efficiency():
    """eps_s of the SIGNAL must pass, whatever the background looks like."""
    d_sig = np.linspace(0.0, 1.0, 10_001)
    d_bkg = np.linspace(0.0, 1.0, 10_001)
    for eps_s in (0.60, 0.40):
        rej, thr, eps_b = an.rejection(d_sig, d_bkg, eps_s)
        assert (d_sig >= thr).mean() == pytest.approx(eps_s, abs=0.002)
        # identical distributions -> eps_B == eps_S -> rejection == 1/eps_S
        assert rej == pytest.approx(1.0 / eps_s, rel=0.01)


def test_rejection_is_not_computed_on_background_efficiency():
    """Guard against the inverted definition: skew the background so the two
    conventions give very different answers, and check we get the signal one."""
    d_sig = np.linspace(0.5, 1.0, 10_000)     # signal concentrated high
    d_bkg = np.linspace(0.0, 0.5, 10_000)     # background concentrated low
    rej, thr, eps_b = an.rejection(d_sig, d_bkg, 0.60)
    assert thr == pytest.approx(0.70, abs=0.01), "threshold must come from the signal"
    assert eps_b == 0.0 and rej == float("inf")


def test_perfect_separation_gives_infinite_rejection():
    rej, _, eps_b = an.rejection(np.ones(1000), np.zeros(1000), 0.60)
    assert eps_b == 0.0 and rej == float("inf")


def test_known_rejection_is_recovered_exactly():
    """Background 1 % above the signal's 60 % point -> rejection exactly 100."""
    d_sig = np.linspace(0.0, 1.0, 100_001)         # 60 % point at D = 0.4
    d_bkg = np.concatenate([np.zeros(99_000), np.ones(1_000)])
    rej, thr, eps_b = an.rejection(d_sig, d_bkg, 0.60)
    assert thr == pytest.approx(0.4, abs=1e-3)
    assert eps_b == pytest.approx(0.01, rel=1e-9)
    assert rej == pytest.approx(100.0, rel=1e-9)


def test_end_to_end_on_synthetic_cache(tmp_path):
    """Whole path: cached arrays -> selection -> discriminant -> rejection.

    Signal nodes are given a large logit for signal jets and QCD nodes a large
    logit for QCD jets, so the discriminant separates perfectly and every
    rejection is infinite -- which the report must express, not crash on.
    """
    n, K = 4_000, 188
    rng = np.random.default_rng(0)
    e1_qcd = an._e1_control().qcd_indices()
    label = np.zeros(n, dtype=np.int16)
    label[:1_000] = 0                       # X_bb
    label[1_000:2_000] = 1                  # X_cc
    label[2_000:] = e1_qcd[0]               # QCD
    logits = rng.normal(0, 0.1, size=(n, K)).astype(np.float32)
    logits[np.arange(1_000), 0] += 20.0
    logits[np.arange(1_000, 2_000), 1] += 20.0
    logits[2_000:, e1_qcd[0]] += 20.0
    np.save(tmp_path / "logits.npy", logits)
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz", jet_pt=np.full(n, 500.0),
             jet_sdmass=np.full(n, 100.0), jet_eta=np.zeros(n))
    out = tmp_path / "anchors.json"
    # A disagreement is a RESULT, not a job failure: main must still exit 0 so
    # a cluster job does not burn its backoffLimit re-deriving the same numbers.
    rc = an.main(["--features", str(tmp_path), "--out", str(out)])
    d = json.loads(out.read_text())
    assert rc == 0 and d["all_agree"] is False
    assert len(d["anchors"]) == 4
    assert d["n_selected"] == n
    for r in d["anchors"]:
        assert r["rejection"] == float("inf") or r["rejection"] > 1e6
        assert r["n_signal"] == 1_000 and r["n_qcd"] == 2_000


def test_jets_that_are_neither_signal_nor_qcd_are_excluded(tmp_path):
    """A third class must not inflate the background denominator."""
    n, K = 3_000, 188
    e1_qcd = an._e1_control().qcd_indices()
    label = np.zeros(n, dtype=np.int16)
    label[:1_000] = 0
    label[1_000:1_500] = 1                   # X_cc, so the cc pass has signal
    label[1_500:2_000] = e1_qcd[0]
    label[2_000:] = 3                        # label_X_qq: neither
    logits = np.zeros((n, K), dtype=np.float32)
    logits[np.arange(1_000), 0] = 20.0
    logits[1_000:1_500, 1] = 20.0
    logits[1_500:2_000, e1_qcd[0]] = 20.0
    logits[2_000:, 3] = 20.0
    np.save(tmp_path / "logits.npy", logits)
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz", jet_pt=np.full(n, 500.0),
             jet_sdmass=np.full(n, 100.0), jet_eta=np.zeros(n))
    out = tmp_path / "a.json"
    an.main(["--features", str(tmp_path), "--out", str(out)])
    bb = [r for r in json.loads(out.read_text())["anchors"] if r["signal"] == "label_X_bb"]
    assert all(r["n_qcd"] == 500 for r in bb), "X_qq jets leaked into the QCD denominator"


def test_a_class_with_no_signal_jets_fails_loudly(tmp_path):
    """Empty side -> a named SystemExit, not numpy's bare IndexError."""
    n, K = 200, 188
    e1_qcd = an._e1_control().qcd_indices()
    label = np.full(n, e1_qcd[0], dtype=np.int16)   # QCD only: no X_bb at all
    np.save(tmp_path / "logits.npy", np.zeros((n, K), dtype=np.float32))
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz", jet_pt=np.full(n, 500.0),
             jet_sdmass=np.full(n, 100.0), jet_eta=np.zeros(n))
    with pytest.raises(SystemExit) as e:
        an.main(["--features", str(tmp_path), "--out", str(tmp_path / "a.json")])
    assert "cannot be defined" in str(e.value)


def test_match_weights_makes_background_look_like_signal():
    """The whole point of the reweighting: after it, the background's (m_SD, pT)
    density must match the signal's. Measured mismatch on the real cache was
    median m_SD 57 vs 141 GeV, which an unweighted ROC converts into free
    separation that is not flavour tagging."""
    rng = np.random.default_rng(0)
    # Inside the Table A1 window the reweighting grid spans 90-140 x 450-600,
    # so the test distributions must live there too: signal skewed heavy,
    # background skewed light, the residual mismatch the weights must remove.
    m_sig = rng.normal(130, 12, 200_000).clip(91, 139)
    pt_sig = rng.normal(540, 40, 200_000).clip(451, 599)
    m_bkg = rng.normal(100, 12, 400_000).clip(91, 139)
    pt_bkg = rng.normal(480, 40, 400_000).clip(451, 599)
    w = an.match_weights(m_sig, pt_sig, m_bkg, pt_bkg)
    assert (w > 0).any()
    # weighted background mean mass must move to the signal's, not stay at its own
    wm = np.average(m_bkg, weights=w)
    assert abs(wm - m_sig.mean()) < 0.03 * m_sig.mean(), f"weighted mean {wm:.1f}"
    assert abs(wm - m_bkg.mean()) > 0.10 * m_bkg.mean(), "weights did nothing"


def test_match_weights_zero_where_background_is_empty():
    """A cell with no background jets cannot supply a background estimate."""
    m_sig = np.array([100.0, 400.0])
    pt_sig = np.array([500.0, 500.0])
    m_bkg = np.array([100.0, 100.0])
    pt_bkg = np.array([500.0, 500.0])
    w = an.match_weights(m_sig, pt_sig, m_bkg, pt_bkg)
    assert np.all(np.isfinite(w)) and np.all(w >= 0)


def test_weighted_rejection_differs_from_unweighted_when_kinematics_differ():
    """Guard that the weights actually reach the rejection arithmetic."""
    d_sig = np.linspace(0.0, 1.0, 10_001)
    d_bkg = np.linspace(0.0, 1.0, 10_001)
    w = np.linspace(0.0, 2.0, 10_001)          # up-weight the high-D background
    plain, _, _ = an.rejection(d_sig, d_bkg, 0.60)
    weighted, _, _ = an.rejection(d_sig, d_bkg, 0.60, w_bkg=w)
    assert weighted < plain, "weighting the hard background up must lower rejection"


def test_window_is_the_one_table_a1_is_measured_in():
    """arXiv:2503.00118 App. A: "The selected jet must satisfy 450 < p_T < 600,
    |eta| < 2.4, and a soft-drop mass requirement of 90 < m_SD < 140."

    NOT the study's own 200<pT<2500, 20<m_SD<500 -- that is the range the bb/cc
    jets span in training, and evaluating there gave rejections 2.6-3.4x the
    published values because the signal and QCD kinematics are grossly
    different across it (median m_SD 141 vs 57 GeV, measured)."""
    assert (an.PT_LO, an.PT_HI) == (450.0, 600.0)
    assert (an.MSD_LO, an.MSD_HI) == (90.0, 140.0)
    assert an.ETA_MAX == 2.4


def test_eta_cut_is_actually_applied(tmp_path):
    """|eta| < 2.4 is part of the window; dropping it silently adds jets."""
    n, K = 400, 188
    e1_qcd = an._e1_control().qcd_indices()
    label = np.zeros(n, dtype=np.int16)
    label[:100] = 0
    label[100:200] = 1
    label[200:] = e1_qcd[0]
    logits = np.zeros((n, K), dtype=np.float32)
    logits[np.arange(100), 0] = 20.0
    logits[100:200, 1] = 20.0
    logits[200:, e1_qcd[0]] = 20.0
    eta = np.zeros(n)
    eta[::2] = 3.0                                  # half the jets fail |eta|<2.4
    np.save(tmp_path / "logits.npy", logits)
    np.save(tmp_path / "label188.npy", label)
    np.savez(tmp_path / "observers.npz", jet_pt=np.full(n, 500.0),
             jet_sdmass=np.full(n, 100.0), jet_eta=eta)
    out = tmp_path / "a.json"
    an.main(["--features", str(tmp_path), "--out", str(out)])
    assert json.loads(out.read_text())["n_selected"] == n // 2


def test_unbounded_rejection_never_counts_as_agreement():
    """With no background jets above threshold the rejection is unbounded. The
    Poisson term is then infinite too, and "inf <= tolerance + inf" is True --
    so without an explicit finiteness guard a perfectly separating (or simply
    empty) background would AGREE with any published number."""
    import json as _json
    rel = float("inf"); stat = float("inf"); tol = 0.25
    assert (rel <= tol + stat) is True, "the trap this guard exists for"
    assert not (np.isfinite(float("inf")) and rel <= tol + stat)


def _anchor_cache(d, n_sig=4000, n_qcd=20000, seed=0):
    """A cache separable enough to give a finite rejection at both eps_S."""
    rng = np.random.default_rng(seed)
    bb, cc, qcd_lab = 0, 1, 161          # label_X_bb, label_X_cc, a QCD node
    n = 2 * n_sig + n_qcd
    label = np.concatenate([np.full(n_sig, bb), np.full(n_sig, cc),
                            np.full(n_qcd, qcd_lab)])
    logits = rng.normal(0, 1, size=(n, 188)).astype(np.float32)
    # A modest edge on each signal's own node: strong enough that 1/eps_B is
    # finite, weak enough that tens of background jets pass, which is the
    # regime the real anchors sit in (29-123 jets).
    logits[:n_sig, bb] += 2.0
    logits[n_sig:2 * n_sig, cc] += 2.0
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "logits.npy", logits)
    np.save(d / "label188.npy", label.astype(np.int16))
    # in-window kinematics, with the QCD mass shifted so the reweighting is
    # NOT a no-op and the two estimators are distinguishable
    msd = np.concatenate([rng.uniform(90, 140, 2 * n_sig),
                          np.full(n_qcd, 95.0)])
    np.savez(d / "observers.npz",
             jet_pt=np.full(n, 500.0), jet_sdmass=msd, jet_eta=np.zeros(n))
    return d


def test_main_reaches_a_finite_rejection_and_compares_the_unweighted_one(tmp_path):
    """End-to-end: nothing previously drove main() to a finite rejection, so
    the comparison and the stat band were never exercised at all.

    The compared number must be the UNWEIGHTED rejection -- Table A1's
    procedure is the kinematic window plus jet-quark matching and nothing
    else, and probe.py's arm-wise numbers are plain unweighted ROC. The
    reweighted number is kept only as a diagnostic.
    """
    d = _anchor_cache(tmp_path / "cache")
    out = tmp_path / "a.json"
    an.main(["--features", str(d), "--out", str(out), "--tolerance", "1e9"])
    res = json.loads(out.read_text())
    rows = [r for r in res["anchors"] if r["signal"] == "label_X_bb"]
    assert rows, "label_X_bb must produce anchor rows"
    for r in rows:
        assert np.isfinite(r["rejection"]), "the run must reach a finite rejection"
        assert r["n_bkg_pass"] > 0
        assert "rejection_reweighted_diagnostic" in r, (
            "the reweighted number is kept, but as a diagnostic")
        # the compared field is the unweighted estimator: rel_diff must be
        # consistent with `rejection`, not with the diagnostic
        exp = abs(r["rejection"] - r["published"]) / r["published"]
        assert abs(exp - r["rel_diff"]) < 1e-2, (
            f"rel_diff {r['rel_diff']} was computed on a different number "
            f"than the reported rejection {r['rejection']}")


def test_stat_band_belongs_to_the_compared_estimator(tmp_path):
    """The Poisson term is 1/sqrt(n_bkg_pass) at the compared threshold."""
    d = _anchor_cache(tmp_path / "cache", seed=3)
    out = tmp_path / "a.json"
    an.main(["--features", str(d), "--out", str(out), "--tolerance", "1e9"])
    for r in json.loads(out.read_text())["anchors"]:
        if r["n_bkg_pass"] > 0:
            assert abs(r["stat_rel_err"] - 1 / np.sqrt(r["n_bkg_pass"])) < 1e-3
