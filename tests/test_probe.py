"""The probe's task definitions and its two correctness guards.

The most valuable test here is test_collapse_claims_match_the_label_map: the
probe asserts, in prose that reaches the paper, that b-vs-c resonant survives at
R42_Q1 and dies at R16_Q1, and that QCD b-vs-c survives ONLY at L188. Those
claims are what make each rung load-bearing, and they are checked here against
configs/labelmaps/rung_label_maps.v1.csv rather than believed.
"""
from __future__ import annotations

import csv
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MAPS = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def probe():
    return _probe()


@pytest.fixture(scope="module")
def rungs():
    if not MAPS.exists():
        pytest.skip("label map absent")
    rows = list(csv.DictReader(MAPS.open()))
    return {int(r["jet_label"]): r for r in rows}


def test_collapse_claims_match_the_label_map(probe, rungs):
    """Each task must be COLLAPSED exactly at the rungs it claims, and distinct
    at every other rung. This is the paper's rung argument, checked."""
    all_arms = ["L188", "L162", "R42_Q1", "R16_Q1"]
    for task, spec in probe.TASKS.items():
        sig_ids, bkg_ids = spec["signal"], spec["background"]
        assert rungs[sig_ids[0]]["class_name"] == spec["names"][0]
        # A one-class background names itself; a multi-class background (the
        # published discriminants have several) names the SET, so only the
        # single-class case can be checked against the map by name.
        if len(bkg_ids) == 1:
            assert rungs[bkg_ids[0]]["class_name"] == spec["names"][1]
        for arm in all_arms:
            # Collapsed at a rung iff the signal shares its group with ANY
            # background class: that is exactly when the arm can no longer
            # separate the two sides, whatever the background's size.
            sg = {rungs[i][arm] for i in sig_ids}
            bg = {rungs[i][arm] for i in bkg_ids}
            collapsed = bool(sg & bg)
            claimed = arm in spec["collapsed_at"]
            assert collapsed == claimed, (
                f"{task}: at {arm} the sides are "
                f"{'collapsed' if collapsed else 'distinct'} "
                f"(signal groups {sorted(sg)} vs background {sorted(bg)}) but "
                f"the task claims {'collapsed' if claimed else 'distinct'}")


def test_tasks_are_arm_independent(probe):
    """A task defined per-arm would not be a controlled contrast."""
    for spec in probe.TASKS.values():
        assert all(isinstance(v, int) for v in spec["signal"] + spec["background"])
        assert set(spec["signal"]).isdisjoint(spec["background"])


def test_alignment_guard_fires(probe):
    import hashlib
    a = np.arange(100, dtype=np.int16)
    arms = {
        "A": {"L": a, "label_sha": hashlib.sha256(a.tobytes()).hexdigest()},
        "B": {"L": a[::-1].copy(),
              "label_sha": hashlib.sha256(a[::-1].copy().tobytes()).hexdigest()},
    }
    with pytest.raises(SystemExit) as e:
        probe.check_alignment(arms)
    assert e.value.code == 2
    same = {"A": arms["A"], "B": dict(arms["A"])}
    assert probe.check_alignment(same) == arms["A"]["label_sha"]


def test_rejection_past_the_cap_is_flagged_a_bound(probe):
    """Beyond 1/N_bkg the value is an artefact of sample size, never a value."""
    y = np.r_[np.ones(500, int), np.zeros(500, int)]
    perfect = np.r_[np.ones(500), np.zeros(500)] * 1.0
    r, eps_b, bound = probe.rejection_at(y, perfect)
    assert bound is True and r <= 500.0
    rng = np.random.default_rng(0)
    r2, _, bound2 = probe.rejection_at(y, rng.random(1000))
    assert bound2 is False and r2 < 10


def test_splits_are_deterministic_and_partition(probe):
    """Same jets must land in the same split for every arm, or the comparison
    is confounded by the split rather than by the arm."""
    tr, va, te = probe.make_splits(1000)
    tr2, va2, te2 = probe.make_splits(1000)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2)
    allidx = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allidx), np.arange(1000)), "splits must partition"
