"""The inference layer. Each test here pins one defect an audit actually found.

The measurements were right and the inference was wrong, so these tests guard
the uncertainty layer specifically: that means are never taken from a rounded
intermediate, that an undetectable pretraining-seed component is reported as
undetectable rather than clamped, that nothing is ever printed as a bare sigma,
and that the t arithmetic is correct.
"""
import importlib.util
import json
import math
import pathlib
import statistics as st

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "experiments/FIGS/data"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "leg_stats", REPO / "experiments/FT/leg_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _mod()


def _cells(leg):
    return L.read_cells(json.loads((DATA / f"{leg}_metrics.json").read_text()))


# ---------------------------------------------------------------- defect 1
# The published leg-2 column reproduced only from 4-dp-rounded per-arm means.
# This is the regression test: it pins BOTH numbers, so anyone who reintroduces
# rounding gets a failure naming the exact value they reintroduced.

def test_rounding_and_full_precision_give_materially_different_answers():
    cells = _cells("leg2")
    a = L.arm_seed_values(cells, "l162-s1b", "N1000000", "accuracy")
    arms = [L.arm_seed_values(cells, f"r16q1-s{i}", "N1000000", "accuracy")
            for i in (2, 3, 4)]
    means = [st.mean(g) for g in arms]

    full = (st.mean(a) - st.mean(means)) / st.stdev(means)
    rounded_means = [round(m, 4) for m in means]
    rounded = (round(st.mean(a), 4) - st.mean(rounded_means)) / st.stdev(rounded_means)

    assert full == pytest.approx(39.118, abs=0.01), (
        "full-precision ratio moved; the committed artifact changed")
    assert rounded == pytest.approx(44.456, abs=0.01), (
        "this is the WRONG value that was published as 44.5; it is pinned so "
        "that reintroducing 4-dp rounding fails loudly rather than silently")
    assert abs(full - rounded) > 5.0, (
        "rounding must remain materially different, else this test is vacuous")


def test_read_cells_refuses_a_pre_aggregated_summary():
    doc = json.loads((DATA / "leg2_metrics.json").read_text())
    with pytest.raises(SystemExit, match="cells"):
        L.read_cells({"summary": doc["summary"]})


def test_every_mean_comes_from_cells_not_from_the_summary_block():
    """The summary carries accuracy_mean; leg_stats must not consult it."""
    doc = json.loads((DATA / "leg2_metrics.json").read_text())
    cells = L.read_cells(doc)
    for arm in ("l162-s1b", "r16q1-s2"):
        for size in L.SIZES:
            recomputed = st.mean(L.arm_seed_values(cells, arm, size, "accuracy"))
            assert recomputed == pytest.approx(
                doc["summary"][arm][size]["accuracy_mean"], abs=1e-12), (
                f"{arm}/{size}: recomputation from cells must agree with the "
                f"summary at FULL precision -- if it does not, one of them is "
                f"rounded and the artifact is internally inconsistent")


# ---------------------------------------------------------------- defect 2
# The denominator was named "pretraining-seed SD" while carrying no detectable
# pretraining-seed variance.

def test_the_pretraining_component_is_reported_absent_where_it_is_absent():
    cells = _cells("leg2")
    groups = [L.arm_seed_values(cells, f"r16q1-s{i}", "N1000000", "accuracy")
              for i in (2, 3, 4)]
    vc = L.variance_components(groups)
    assert vc["estimable"]
    assert vc["F"] == pytest.approx(0.786, abs=0.01)
    assert vc["var_pretraining_hat"] < 0, (
        "the committed data give a NEGATIVE pretraining variance component at "
        "this cell; if this ever goes positive the headline denominator "
        "becomes defensible and the prose must be revisited")
    assert vc["pretraining_variance_detected"] is False


def test_a_negative_variance_component_is_never_clamped_to_zero():
    """Clamping would hide the finding. Two groups whose means coincide but
    whose within-group scatter is large must report a negative component."""
    groups = [[0.50, 0.60, 0.55], [0.55, 0.50, 0.60], [0.60, 0.55, 0.50]]
    vc = L.variance_components(groups)
    assert vc["estimable"] and vc["var_pretraining_hat"] < 0
    assert vc["pretraining_variance_detected"] is False


def test_the_denominator_used_is_smaller_than_the_finetuning_noise():
    """The specific pathology: the sigma unit was below the noise on a single
    run of the quantity being compared, in every leg-2 cell."""
    cells = _cells("leg2")
    for size in L.SIZES:
        groups = [L.arm_seed_values(cells, f"r16q1-s{i}", size, "accuracy")
                  for i in (2, 3, 4)]
        vc = L.variance_components(groups)
        assert vc["sd_of_arm_means"] < vc["pooled_finetuning_sd"], (
            f"{size}: the unit published as a pretraining-seed SD is "
            f"{vc['sd_of_arm_means']:.6g} against a fine-tuning floor of "
            f"{vc['pooled_finetuning_sd']:.6g}")


def test_a_single_pretraining_seed_cannot_be_decomposed():
    assert L.variance_components([[0.1, 0.2, 0.3]])["estimable"] is False


# ---------------------------------------------------------------- defect 3
# 2-df ratios printed with a Greek sigma.

def test_no_row_can_be_rendered_as_a_sigma():
    cells = _cells("leg2")
    r = L.contrast(cells, "granularity", L.CONTRASTS["granularity"],
                   "N1000000", "accuracy")
    line = L.format_row(r)
    assert "sigma" not in line.lower() and "σ" not in line
    assert "sigma" not in json.dumps(r).lower()
    assert "p=" in line and "df=" in line and "CI95=" in line


def test_the_t_arithmetic_matches_scipy_where_scipy_is_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1.0, 2.0, 2.1, 4.9, 9.5, 30.0, 200.0):
        for t in (0.0, 0.5, 2.97, 7.42, 38.5, 120.0):
            assert L._t_sf(t, df) == pytest.approx(
                float(scipy_stats.t.sf(t, df)), rel=1e-9, abs=1e-300)
        assert L._t_ppf95(df) == pytest.approx(
            float(scipy_stats.t.ppf(0.975, df)), rel=1e-9)


def test_the_t_arithmetic_matches_published_table_values_without_scipy():
    """So the guard survives on the cluster image, which has no scipy."""
    for df, crit in ((1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228),
                     (30, 2.042), (100, 1.984)):
        assert L._t_ppf95(float(df)) == pytest.approx(crit, abs=0.001)


# ---------------------------------------------- the design constraint itself

def test_pretraining_seed_inference_is_blocked_while_l162_has_one_seed():
    cells = _cells("leg1")
    r = L.contrast(cells, "granularity", L.CONTRASTS["granularity"],
                   "N1000000", "accuracy")
    assert r["inference_level"] == "fine-tuning-seed"
    assert r["pretraining_seed_test"]["estimable"] is False
    assert r["pretraining_seed_test"]["permutation_floor_one_sided"] == 0.25


def test_the_permutation_floor_is_the_combinatorial_one():
    assert L.permutation_floor(1, 3) == pytest.approx(0.25)
    assert L.permutation_floor(1, 1) == pytest.approx(0.5)
    assert L.permutation_floor(5, 5) == pytest.approx(1 / 252)


def test_the_lr_confounded_contrast_is_flagged_and_the_granularity_one_is_not():
    assert L.CONTRASTS["granularity"]["i1_clean"] is True
    assert L.CONTRASTS["pretrained_vs_scratch"]["i1_clean"] is False
    cells = _cells("leg1")
    r = L.contrast(cells, "pretrained_vs_scratch",
                   L.CONTRASTS["pretrained_vs_scratch"], "N10000", "accuracy")
    assert "[I1: >1 variable]" in L.format_row(r)
    assert r["gap"] < 0, (
        "at N=1e4 in leg 1 scratch BEATS the pretrained arms; this is the "
        "learning-rate artefact, and the sign is pinned so that a silent "
        "recipe change shows up here")


# ------------------------------------------------------- the published gaps

@pytest.mark.parametrize("leg,size,expected", [
    ("leg1", "N10000", 0.084812), ("leg1", "N100000", 0.040152),
    ("leg1", "N1000000", 0.024552), ("leg2", "N10000", 0.062023),
    ("leg2", "N100000", 0.009962), ("leg2", "N1000000", 0.007719),
])
def test_the_six_granularity_gaps_reproduce_from_cells(leg, size, expected):
    r = L.contrast(_cells(leg), "granularity", L.CONTRASTS["granularity"],
                   size, "accuracy")
    assert r["gap"] == pytest.approx(expected, abs=5e-6)


def test_l162_beats_public_sophon_in_leg2_and_ties_it_in_leg1():
    """Published as a tie in BOTH legs. Leg 2 is not a tie: p = 0.042."""
    leg2 = L.contrast(_cells("leg2"), "l162_vs_sophon",
                      L.CONTRASTS["l162_vs_sophon"], "N1000000", "accuracy")
    leg1 = L.contrast(_cells("leg1"), "l162_vs_sophon",
                      L.CONTRASTS["l162_vs_sophon"], "N1000000", "accuracy")
    assert leg2["finetuning_seed_test"]["p_two_sided"] < 0.05
    assert leg1["finetuning_seed_test"]["p_two_sided"] > 0.10
    lo, _ = leg2["finetuning_seed_test"]["ci95"]
    assert lo > 0, "leg 2's 95% CI excludes zero, so 'tie' is the wrong word"


def test_a_cell_missing_the_metric_is_fatal_not_dropped(tmp_path):
    bad = {"cells": {"x": {"N10000": {"s1": {"not_accuracy": 1.0}}}}}
    with pytest.raises(SystemExit, match="has no"):
        L.arm_seed_values(L.read_cells(bad), "x", "N10000", "accuracy")
