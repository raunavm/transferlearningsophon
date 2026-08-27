"""CI tests for experiments/EVAL/latent_scale_probe.py.

The probe answers a question whose wrong answer is publishable-looking: does the
TRUNK carry absolute momentum, or only its inputs? Every failure mode here is
silent -- a probe that quietly fits the wrong target, or splits differently from
probe.py, still prints a plausible R^2.

Run:  python3 -m pytest tests/test_latent_scale_probe.py -v
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "EVAL"))

lsp = pytest.importorskip("latent_scale_probe")


def test_split_is_identical_to_the_classification_probe():
    """Both probes must put the same jet in the same split. If they drift, a
    scale number and a b-vs-c number computed on 'the same' test jets are not on
    the same jets, and no error says so."""
    probe = pytest.importorskip("probe")
    assert lsp.SPLIT_SEED == probe.SPLIT_SEED
    for n in (1000, 60_565):
        a = [x.tolist() for x in lsp.make_splits(n)]
        b = [x.tolist() for x in probe.make_splits(n)]
        assert a == b, f"splits diverge at n={n}"


def test_split_is_deterministic_and_partitions_every_row():
    tr, va, te = lsp.make_splits(1000)
    allrows = np.concatenate([tr, va, te])
    assert sorted(allrows.tolist()) == list(range(1000))
    assert [x.tolist() for x in lsp.make_splits(1000)] == [tr.tolist(), va.tolist(), te.tolist()]


def test_targets_are_the_three_the_spec_names():
    obs = {"jet_pt": np.array([200.0, 2500.0]), "jet_eta": np.array([-2.0, 1.5])}
    t = lsp.targets(obs)
    assert set(t) == {"ln_jet_pt", "abs_jet_eta", "jet_eta_signed"}
    assert np.allclose(t["ln_jet_pt"], np.log([200.0, 2500.0]))
    assert np.allclose(t["abs_jet_eta"], [2.0, 1.5])
    assert np.allclose(t["jet_eta_signed"], [-2.0, 1.5])


def test_signed_eta_is_not_the_same_target_as_abs_eta():
    """The negative control is only a control if it actually differs from the
    affirmative one. A refactor that made both |eta| would make the control pass
    by construction and prove nothing."""
    obs = {"jet_pt": np.array([500.0, 500.0]), "jet_eta": np.array([-2.0, 2.0])}
    t = lsp.targets(obs)
    assert not np.allclose(t["abs_jet_eta"], t["jet_eta_signed"])


def test_targets_reject_non_positive_pt():
    obs = {"jet_pt": np.array([200.0, 0.0]), "jet_eta": np.array([1.0, 1.0])}
    with pytest.raises(AssertionError):
        lsp.targets(obs)


def test_r2_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=500), rng.normal(size=500)
    assert lsp.r2(y, p) == pytest.approx(sk.r2_score(y, p), abs=1e-12)


def test_r2_is_zero_for_the_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert lsp.r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_a_representation_carrying_the_target_beats_its_own_null():
    """End-to-end on a fixture where the answer is known by construction:
    dim 0 encodes ln pT, nothing encodes the SIGN of eta. The affirmative target
    must clear its permutation null and the negative control must not."""
    rng = np.random.default_rng(0)
    n = 1500
    lnpt = rng.uniform(np.log(200), np.log(2500), n)
    eta = rng.uniform(-2.5, 2.5, n)
    F = rng.normal(size=(n, 16))
    F[:, 0] += 3.0 * lnpt
    tr, va, te = lsp.make_splits(n)

    hit = lsp.probe_one(F, np.log(np.exp(lnpt)), tr, va, te, do_null=True)
    assert hit["r2_minus_null"] > 0.5, hit

    miss = lsp.probe_one(F, eta, tr, va, te, do_null=True)
    assert abs(miss["ridge"]["r2"] - miss["null"]["r2_shuf_mean"]) < 0.05, miss


def test_permutation_null_is_near_zero_not_near_one():
    """A null that came back high would mean the estimator fits noise at this
    sample size, and every R^2 in the output would be uninterpretable."""
    rng = np.random.default_rng(1)
    n = 1200
    F = rng.normal(size=(n, 16))
    y = rng.normal(size=n)
    tr, va, te = lsp.make_splits(n)
    out = lsp.probe_one(F, y, tr, va, te, do_null=True)
    assert out["null"]["r2_shuf_mean"] < 0.05, out["null"]
