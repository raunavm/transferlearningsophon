"""The figure script is where measured numbers become the paper's claims, so
the claims themselves are pinned here rather than only the plotting mechanics."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNGS_FINER = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1"]
RUNGS_AT_OR_BELOW = ["R16_Q1", "R3_VIS", "R1_Q1"]


@pytest.fixture(scope="module")
def gaps(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "make_paper_figures", REPO / "experiments/FIGS/make_paper_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = tmp_path_factory.mktemp("figs")
    return m, m.fig1_label_recovery(REPO / "experiments/FIGS/data/label_recovery_v3.json")


def test_the_crossover_is_where_the_paper_says_it_is(gaps):
    """L162 leads at every rung FINER than R16_Q1 and trails at R16_Q1 and
    below. If a re-measurement moves the sign flip, the paper's cleanest figure
    is telling a different story and this must fail."""
    _, g = gaps
    for r in RUNGS_FINER:
        assert g[r][0] > 0, f"{r} should favour L162, got {g[r][0]:+.4f}"
    for r in RUNGS_AT_OR_BELOW:
        assert g[r][0] < 0, f"{r} should favour R16_Q1, got {g[r][0]:+.4f}"


def test_the_advantage_decays_monotonically_to_the_crossover_then_returns(gaps):
    """The shape is decay-to-a-minimum, NOT monotone decay all the way down --
    a distinction worth pinning because the loose phrasing is easy to write.

    From the L162 rung the lead falls monotonically and crosses zero at R16_Q1,
    which is the minimum. BELOW R16_Q1 the gap is negative but shrinks back
    toward zero (-0.0042 -> -0.0037 -> -0.0022), and it must: R1_Q1 is a binary
    split, so there is almost nothing left for either vocabulary to distinguish
    and both arms have to converge. Asserting monotone decay across all eight
    rungs would fail on a correct result."""
    _, g = gaps
    decay = [g[r][0] for r in ["L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]]
    assert all(a > b for a, b in zip(decay, decay[1:])), f"not monotone: {decay}"
    assert decay[-1] == min(g[r][0] for r in g), "R16_Q1 should be the minimum"
    below = [abs(g[r][0]) for r in RUNGS_AT_OR_BELOW]
    assert all(a >= b for a, b in zip(below, below[1:])), \
        f"gap should shrink toward zero below the crossover: {below}"


def test_the_sign_flip_is_resolved_against_the_seed_scatter(gaps):
    """At n=3 seeds this gap was inside the noise. The fourth seed is what makes
    it quotable, so the sigma is pinned, not just the sign."""
    _, g = gaps
    assert g["R16_Q1"][1] < -2.0, f"flip is only {g['R16_Q1'][1]:.1f} sigma"
    assert g["L188"][1] > 10.0, f"fine-rung lead is only {g['L188'][1]:.1f} sigma"


def test_both_figures_are_written(gaps):
    m, _ = gaps
    m.fig2_transfer(REPO / "experiments/FIGS/data/leg1_metrics.json",
                    REPO / "experiments/FIGS/data/leg2_metrics.json")
    for stem in ("label_recovery_crossover", "transfer_curves"):
        for ext in ("pdf", "png"):
            assert (m.OUT / f"{stem}.{ext}").stat().st_size > 5000
