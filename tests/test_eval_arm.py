"""Tests for the per-arm scorer, chiefly its class-axis ordering.

The ordering test is the load-bearing one. A lexicographic sort of
score_0..score_16 puts score_10 before score_2, which permutes the class axis.
Every per-class AUC would then be attributed to the wrong class, the macro
average would be numerically IDENTICAL, and nothing would error.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_arm", REPO / "experiments" / "EVAL" / "eval_arm.py")
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)


def test_score_branches_sort_numerically_not_lexicographically():
    names = [f"score_{i}" for i in range(17)]
    got = sorted(sorted(names), key=ea._order_key)      # inner sort = worst case
    assert got == names
    assert got.index("score_2") < got.index("score_10")


def test_order_key_falls_back_to_name_without_a_trailing_integer():
    names = ["score_QCD", "score_Hbb", "score_3"]
    got = sorted(names, key=ea._order_key)
    assert got[0] == "score_3", "numbered branches sort ahead of named ones"
    assert got[1:] == ["score_Hbb", "score_QCD"]


def test_macro_auc_is_one_for_a_perfect_classifier():
    truth = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[truth] * 0.9 + 0.05
    m = ea.metrics(probs, truth, 3, qcd=2)
    assert m["macro_auc_ovr"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["n_classes_present"] == 3


def test_absent_classes_are_excluded_rather_than_scored_as_chance():
    """Class 2 never occurs. Substituting 0.5 for it would drag the macro to
    0.75 on an otherwise perfect classifier."""
    truth = np.array([0, 0, 1, 1])
    probs = np.zeros((4, 3))
    probs[truth == 0, 0] = 0.9
    probs[truth == 1, 1] = 0.9
    probs += 0.05
    m = ea.metrics(probs, truth, 3, qcd=2)
    assert m["n_classes_present"] == 2
    assert m["macro_auc_ovr"] == pytest.approx(1.0)
    assert set(m["per_class_auc"]) == {0, 1}


def test_qcd_class_gets_no_rejection_entry_against_itself():
    truth = np.array([0, 0, 1, 1])
    probs = np.array([[.8, .2], [.7, .3], [.3, .7], [.2, .8]])
    m = ea.metrics(probs, truth, 2, qcd=1)
    rej = m[f"rejection_vs_qcd_eff{ea.EPS_S}"]
    assert set(rej) == {0}, "QCD must not be scored against itself"


def test_rejection_past_the_sample_floor_is_flagged_as_a_bound():
    """With few background jets the rejection is a statement about N, not the
    model, and must be marked so it is never quoted as a value."""
    rng = np.random.default_rng(0)
    truth = np.concatenate([np.zeros(500), np.ones(20)]).astype(int)
    p_sig = np.concatenate([rng.uniform(.9, 1., 500), rng.uniform(0., .01, 20)])
    probs = np.stack([p_sig, 1 - p_sig], axis=1)
    m = ea.metrics(probs, truth, 2, qcd=1)
    assert m[f"rejection_vs_qcd_eff{ea.EPS_S}"][0]["is_bound"] is True


def test_bad_pred_spec_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["eval_arm.py", "--pred", "no-equals-sign",
                         "--k", "2", "--arm", "X", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        ea.main()
