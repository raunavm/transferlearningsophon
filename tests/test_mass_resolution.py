"""S7's frozen-feature mass regression: the four pre-registered choices, pinned.

Every test here corresponds to a decision that was free until the PI fixed it
(DECISIONS_PENDING item 40). They exist so the definition cannot drift after the
numbers land, which is the whole reason it was fixed in advance.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("mr", ROOT / "experiments/EVAL/mass_resolution.py")
mr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mr)

N = 4000
SHA = "ab" * 32


# ------------------------------------------------------------- choice 1

def test_both_68_percent_widths_equal_the_standard_deviation_on_a_gaussian():
    """The headline statistic is calibrated: on a Gaussian residual the smallest
    and the central 68 % intervals coincide, and both equal the standard
    deviation. So the CMS-vs-ATLAS fork costs nothing in the easy case and can
    only matter where the distribution is skewed -- which is the regime both
    collaborations say they chose a quantile statistic for."""
    g = np.random.default_rng(0).normal(0.0, 1.0, 200_000)
    r = mr.resolution(g)
    assert r["sigma_eff"] == pytest.approx(1.0, abs=0.02)
    assert r["sigma68_central"] == pytest.approx(1.0, abs=0.02)
    assert r["sd"] == pytest.approx(1.0, abs=0.02)


def test_the_smallest_interval_tracks_the_mode_when_the_residual_is_skewed():
    """The two 68 % conventions are NOT the same statistic. On a right-skewed
    residual the smallest interval hugs the mode and the central one is dragged
    by the tail, so the smallest one is narrower. This is the fork the
    pre-registration had to resolve rather than leave open."""
    rng = np.random.default_rng(11)
    skewed = np.concatenate([rng.normal(0, 0.5, 90_000), rng.exponential(3.0, 30_000)])
    r = mr.resolution(skewed)
    assert r["sigma_eff"] < r["sigma68_central"], "the smallest interval hugs the mode"
    lo, hi = r["sigma_eff_interval"]
    assert lo < 0.0 < hi, "and it contains the mode"


def test_the_two_statistics_can_rank_two_models_in_opposite_directions():
    """THE REASON THIS CHOICE WAS PRE-REGISTERED. A model with a tighter core and a
    heavier tail beats its rival on the 68 % half-width and loses to it on the
    standard deviation. A mass-output model is trained to regress mass, so a
    tighter core with a worse tail is exactly what it might produce -- and picking
    the statistic after seeing that would be picking the answer."""
    rng = np.random.default_rng(1)
    tight_core_heavy_tail = np.concatenate([rng.normal(0, 0.8, 190_000),
                                            rng.normal(0, 6.0, 10_000)])
    plain = rng.normal(0, 1.0, 200_000)
    a, b = mr.resolution(tight_core_heavy_tail), mr.resolution(plain)
    assert a["sigma_eff"] < b["sigma_eff"], "the effective resolution prefers the tight core"
    assert a["sd"] > b["sd"], "the standard deviation prefers the other one"


def test_the_standard_deviation_always_travels_with_the_headline():
    """Reporting one without the other is what would let the choice be made after
    the fact, so both are always present, and so is the sample size."""
    r = mr.resolution(np.random.default_rng(2).normal(0, 1, 5000))
    for k in ("sigma_eff", "sigma68_central", "sd", "median", "fractional",
              "tail_fraction", "n"):
        assert k in r, k


# ------------------------------------------------------------- choice 2

def test_centering_removes_everything_a_class_only_predictor_could_know():
    """Why the target is centered at all: class implies mass, so without this a
    better CLASSIFIER would score as a better mass regressor -- the one confound
    that would make a label-vocabulary study uninterpretable."""
    rng = np.random.default_rng(3)
    lab = rng.integers(0, 20, N)
    y = lab * 3.0                      # target determined ENTIRELY by class
    tr = np.arange(N)
    yc, usable, info = mr.class_center(y, lab, tr)
    assert usable.all()
    assert np.abs(yc).max() < 1e-9, "a class-determined target must centre to zero"
    assert info["n_classes_used"] == 20


def test_centering_keeps_the_within_class_variation_it_is_supposed_to_keep():
    rng = np.random.default_rng(4)
    lab = rng.integers(0, 10, N)
    within = rng.normal(0, 0.5, N)
    yc, _, _ = mr.class_center(lab * 5.0 + within, lab, np.arange(N))
    # the class offsets are gone; the within-class scatter survives intact
    assert yc.std() == pytest.approx(within.std(), rel=0.05)


def test_class_means_come_from_the_training_split_alone():
    """Fitting the centering on all rows would leak the test set into its own
    centering."""
    rng = np.random.default_rng(5)
    lab = np.zeros(N, dtype=int)
    y = np.concatenate([np.full(N // 2, 10.0), np.full(N // 2, 20.0)])
    tr = np.arange(N // 2)                      # train sees only the 10.0 half
    yc, _, _ = mr.class_center(y, lab, tr)
    assert yc[:N // 2] == pytest.approx(0.0)
    assert yc[N // 2:] == pytest.approx(10.0), "test rows are centred by the TRAIN mean"


def test_a_class_with_too_few_training_jets_is_dropped_and_counted():
    """A mean from a handful of jets is noise, and noise that differs per class.
    Those jets leave rather than being centred badly, and the count is reported."""
    rng = np.random.default_rng(6)
    lab = np.zeros(N, dtype=int)
    lab[:5] = 1                                  # class 1 has 5 jets, below the floor
    y = rng.normal(0, 1, N)
    _, usable, info = mr.class_center(y, lab, np.arange(N))
    assert info["n_classes_present"] == 2
    assert info["n_classes_used"] == 1
    assert info["n_jets_dropped_small_class"] == 5
    assert not usable[:5].any() and usable[5:].all()


# ------------------------------------------------------------- choice 3

def test_family_means_are_subsumed_by_native_class_centering():
    """docs/DOWNSTREAM_SUITE.md asks for a within-class resolution WITH FAMILY
    MEANS REMOVED, which reads like two operations. It is one: every family is a
    union of native classes, so centering at 188 has already removed them, and
    doing both would subtract the same thing twice."""
    rng = np.random.default_rng(7)
    lab = rng.integers(0, 30, N)
    family = lab // 10                            # families are unions of classes
    y = family * 7.0 + lab * 0.5 + rng.normal(0, 0.3, N)
    yc, _, _ = mr.class_center(y, lab, np.arange(N))
    # after class centering, the family means are already zero
    for f in range(3):
        assert abs(yc[family == f].mean()) < 0.05, f"family {f} mean survived"


# ------------------------------------------------------------- choice 4

def test_the_tail_is_reported_and_never_trimmed():
    rng = np.random.default_rng(8)
    res = np.concatenate([rng.normal(0, 0.1, 9000), np.full(1000, 5.0)])
    r = mr.resolution(res)
    assert r["tail_fraction"] == pytest.approx(0.1, abs=0.01)
    assert r["tail_at"] == mr.TAIL_AT == 1.0
    assert r["n"] == 10_000, "every jet is still in the sample; nothing was trimmed"


# --------------------------------------------------- alignment and validity

def _write_observers(d: pathlib.Path, n=N, unmatched=0, sha=SHA):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    gen = rng.uniform(20, 300, n)
    gen[:unmatched] = 0.0                      # a hard 0.0 == never matched
    np.savez(d / "observers.npz", genjet_sdmass=gen, jet_sdmass=rng.uniform(20, 300, n))
    np.save(d / "label188.npy", rng.integers(0, 20, n).astype(np.int16))
    (d / "observers_manifest.json").write_text(json.dumps({"label188_sha256": sha, "n_jets": n}))
    return d


def test_unmatched_jets_are_dropped_as_invalid(tmp_path):
    """genjet_sdmass is a hard 0.0 when the jet was never matched. That is a
    validity cut, not a tail cut, and it is not one of the four choices."""
    obs = mr.load_observers(_write_observers(tmp_path / "o", unmatched=400))
    assert obs["n"] == N
    assert obs["n_valid"] == N - 400
    assert not obs["valid"][:400].any()
    assert np.isfinite(obs["y"][obs["valid"]]).all()


def test_an_arm_whose_rows_are_not_the_observers_rows_is_refused(tmp_path):
    """The observer cache verified alignment against the twenty granularity caches
    and could NOT verify it against the ten mass-output ones, which did not exist
    when it ran. S7 regresses one cached array against another, so an unchecked
    row offset would be invisible in every number it produces."""
    d = tmp_path / "arm"
    d.mkdir()
    (d / "extract_manifest.json").write_text(json.dumps({"label188_sha256": "cd" * 32,
                                                         "n_jets": N}))
    with pytest.raises(SystemExit, match="does not match the observers"):
        mr.check_alignment("l162mass-s1", d, SHA, N)


def test_an_arm_with_no_alignment_digest_at_all_is_refused(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "extract_manifest.json").write_text(json.dumps({"n_jets": N}))
    with pytest.raises(SystemExit, match="carries no label188_sha256"):
        mr.check_alignment("l162mass-s1", d, SHA, N)


def test_a_matching_arm_passes_and_records_what_it_checked(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "extract_manifest.json").write_text(json.dumps(
        {"label188_sha256": SHA, "n_jets": N, "checkpoint_sha256": "ef" * 32}))
    prov = mr.check_alignment("l162mass-s1", d, SHA, N)
    assert prov["label188_sha256"] == SHA and prov["n_jets"] == N
    assert prov["checkpoint_sha256"] == "ef" * 32


def test_observers_without_a_digest_cannot_anchor_anything(tmp_path):
    d = _write_observers(tmp_path / "o")
    (d / "observers_manifest.json").write_text(json.dumps({"n_jets": N}))
    with pytest.raises(SystemExit, match="no label188_sha256"):
        mr.load_observers(d)


# ------------------------------------------------------------------ D6

def test_both_probes_are_always_reported(tmp_path):
    """D6: a linear result never stands on its own, and that does not stop being
    true because the endpoint is a regression rather than a classification."""
    rng = np.random.default_rng(10)
    n = 1200
    F = rng.normal(0, 1, (n, 8))
    y = F[:, 0] * 0.7 + rng.normal(0, 0.3, n)
    tr, va, te = mr.make_splits(n)
    r = mr.probe_arm(F, y, tr, va, te)
    assert set(mr.PROBES) <= set(r)
    for p in mr.PROBES:
        assert r[p]["sigma_eff"] > 0 and r[p]["sd"] > 0
    # and the spread of the target itself, or a resolution means nothing
    assert r["target"]["sigma_eff"] > r["ridge"]["sigma_eff"], (
        "a probe that learned something must beat the uninformed spread")


def test_the_scale_travels_with_the_width():
    """Every source in the field pairs the width with a scale statistic. A twin
    that is narrower but mis-scaled is not a better one, and a referee will ask."""
    off = np.random.default_rng(12).normal(0.4, 0.2, 50_000)
    r = mr.resolution(off)
    assert r["median"] == pytest.approx(0.4, abs=0.02)
    assert r["mode_of_eff_interval"] == pytest.approx(0.4, abs=0.05)


def test_the_residual_is_reported_as_a_fraction_because_it_is_a_log_ratio():
    """The probe predicts a log mass-ratio, so exponentiating the residual gives
    the response m_pred/m_true the field quantifies. `fractional` is that
    percentage, which is how every cited source quotes a mass resolution."""
    r = mr.resolution(np.random.default_rng(13).normal(0, 0.10, 100_000))
    assert r["fractional"] == pytest.approx(np.expm1(r["sigma_eff"]), rel=1e-9)
    assert r["fractional"] == pytest.approx(0.105, abs=0.01), "~10% for a 0.10 log width"
