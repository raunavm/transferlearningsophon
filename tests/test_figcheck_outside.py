"""figcheck fails a figure with a plotted point outside its axes (PI 2026-11-11: nothing off the axes)."""
import pathlib
import sys

import numpy as np
import pytest

fitz = pytest.importorskip("fitz")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments/FIGS/figstyle"), str(ROOT / "experiments/FIGS")]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figcheck  # noqa: E402


def _axes():
    fig, ax = plt.subplots()
    ax.set_title("(a) panel")
    return fig, ax


def test_points_inside_the_axes_pass():
    fig, ax = _axes()
    ax.plot([1, 2, 3], [1, 4, 9], label="curve")
    ax.axhline(5)
    ax.fill_between([1, 3], [1, 1], [2, 2])
    ax.stairs([1, 2], [0, 1, 2], baseline=None)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    assert figcheck.outside(ax) == []
    plt.close(fig)


@pytest.mark.parametrize("draw", [
    lambda ax: ax.plot([1, 2, 3], [1, 4, 90], label="curve"),                       # a curve through the top edge
    lambda ax: ax.plot([5], [5], marker="o", linestyle="none", clip_on=False),     # a marker right of the frame
    lambda ax: ax.scatter([1], [-3]),
    lambda ax: ax.fill_between([1, 3], [1, 1], [20, 20]),
    lambda ax: ax.stairs([1, 20], [0, 1, 2], baseline=None),
    lambda ax: ax.axhline(50),
])
def test_anything_outside_the_axes_is_found(draw):
    fig, ax = _axes()
    draw(ax)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    assert figcheck.outside(ax), "a point outside the axes went unseen"
    plt.close(fig)


def test_a_log_axis_ignores_the_zero_baseline_and_nan():
    fig, ax = _axes()
    ax.plot([1, 2, 3], [np.nan, 10, 100])
    ax.stairs([1, 10], [1, 2, 3], baseline=0)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.5, 200)
    assert figcheck.outside(ax) == []
    plt.close(fig)


def test_the_check_reports_it_as_a_finding():
    fig, ax = _axes()
    ax.plot([1, 2], [1, 50], label="curve")
    ax.set_ylim(0, 10)
    rep = figcheck.check(fig, "x", 15.0, 6.5)
    assert rep["findings"]["outside"] and rep["n_findings"] >= 1
    plt.close(fig)


def test_macro_auc_is_a_banned_term():
    assert figcheck.notation("188 classes, macro AUC 0.985") and not figcheck.notation("mean AUC over classes 0.985")
