"""The label-recovery claims the paper may make, pinned against the five-seed,
four-label-set analysis that make_paper_figures.py now draws
(experiments/FIGS/data/label_recovery_ladder_v1/analysis/s9_label_recovery.json).

It replaced a two-model file (one 162-class seed against four 17-class seeds) on
2026-09-22. The two agree on the headline and disagree on a secondary shape
claim, so the shape claim is no longer made and nothing here asserts it."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FINER = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1"]
DECAY = ["L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]
PROBES = ("linear", "mlp")


@pytest.fixture(scope="module")
def fig(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "make_paper_figures", REPO / "experiments/FIGS/make_paper_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = tmp_path_factory.mktemp("figs")
    return m, m.fig1_label_recovery(m.LABEL_RECOVERY)


@pytest.mark.parametrize("probe", PROBES)
def test_the_lead_ends_at_the_17_class_models_own_level(fig, probe):
    """Both probes: the 162-class model's advantage is last seen one level finer
    than the 17-class set, and the sign changes at the 17-class set itself."""
    c = fig[1]["crossover"][probe]
    assert c["crossover_rung"] == "R16_Q1"
    assert c["crossover_at_coarser_own_rung"] is True


@pytest.mark.parametrize("probe", PROBES)
def test_no_level_at_or_below_the_17_class_set_significantly_favours_162(fig, probe):
    """Not "the lead is gone at every coarser level" point by point: on the
    nonlinear probe the 2-class level shows +0.003 for the 162-class model, which
    is why the analysis records its last rung with an advantage as the 2-class
    one there. None of it survives correction, and that is the claim pinned."""
    cells = fig[1]["contrasts"][probe]
    for r in ("R16_Q1", "R3_VIS", "R1_Q1"):
        assert not (cells[r]["mean_diff"] > 0 and cells[r]["holm_reject"]), (r, cells[r])


@pytest.mark.parametrize("probe", PROBES)
def test_the_finer_levels_favour_the_162_class_model_after_correction(fig, probe):
    cells = fig[1]["contrasts"][probe]
    for r in FINER:
        assert cells[r]["mean_diff"] > 0, (r, cells[r]["mean_diff"])
        assert cells[r]["holm_reject"] is True, (r, cells[r]["p"])


@pytest.mark.parametrize("probe", PROBES)
def test_the_lead_shrinks_level_by_level_down_to_the_crossover(fig, probe):
    d = [fig[1]["contrasts"][probe][r]["mean_diff"] for r in DECAY]
    assert all(a > b for a, b in zip(d, d[1:])), d


@pytest.mark.parametrize("probe", PROBES)
def test_the_reversal_at_the_17_class_level_is_not_a_significant_win(fig, probe):
    """GUARD AGAINST OVERCLAIMING. The sign flips at the 17-class model's own
    level, but after correction the reversal is not significant on either probe,
    so the paper may say the lead DISAPPEARS there -- not that the 17-class model
    wins there. The older test pinned the flip at >2 sigma on four seeds; five
    paired seeds under Holm do not support that."""
    cell = fig[1]["contrasts"][probe]["R16_Q1"]
    assert cell["mean_diff"] < 0
    assert cell["holm_reject"] is False


def test_the_preregistered_label_recovery_prediction_is_recorded_as_failed(fig):
    """The figure shows a clean crossover, but the pre-registered prediction about
    it was not confirmed. The caption must not imply otherwise."""
    assert fig[1]["composite_verdict"] == "not confirmed in clauses 1-2"


def test_both_figures_are_written(fig):
    m, _ = fig
    m.fig2_transfer(REPO / "experiments/FIGS/data/leg1_metrics.json",
                    REPO / "experiments/FIGS/data/leg2_metrics.json")
    for stem in ("label_recovery_crossover", "transfer_curves"):
        for ext in ("pdf", "png"):
            assert (m.OUT / f"{stem}.{ext}").stat().st_size > 5000
