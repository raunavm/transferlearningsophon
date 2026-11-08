"""The gallery's uncertainty arithmetic (experiments/FIGS/figstyle/gallery_bands.py) and its change markers
(gallery_review.CHANGES, shown per page by gallery_figures.py and build_gallery.py)."""
import ast
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments/FIGS/figstyle"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FIG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("gallery_bands")


def test_one_run_carries_the_binomial_term_only():
    k, n = np.array([20.0, 400.0]), 10_000
    rel = B.rejection_rel_error([[n / 20, n / 400]], [k], n)
    assert rel == pytest.approx(np.sqrt((1 - k / n) / k))


def test_the_binomial_term_is_the_68_percent_spread_of_a_rejection():
    """1/eps_B over test sets drawn from the same efficiency: its relative SD is the binomial term."""
    rng = np.random.default_rng(3)
    n, eps = 50_000, 0.002
    k = rng.binomial(n, eps, size=200_000)
    rej = n / k
    expected = B.rejection_rel_error([[1 / eps]], [[n * eps]], n)[0]
    assert np.std(rej) / np.mean(rej) == pytest.approx(expected, rel=0.03)


def test_the_seed_sd_and_the_binomial_term_add_in_quadrature():
    rej = np.array([[100.0], [110.0], [90.0]])
    k, n = np.array([[50.0], [45.0], [55.0]]), 5_000
    seed = np.std(rej, ddof=1) / np.mean(rej)
    binom = np.sqrt((1 - 50 / n) / 50)            # the mean passing count, not divided by the runs: one test set
    assert B.rejection_rel_error(rej, k, n)[0] == pytest.approx(np.hypot(seed, binom))


def test_the_reference_is_taken_in_only_while_the_series_keep_half_the_axis():
    assert B.extend_to(0.99, 1.01, [1.0, 1.005]) == (0.99, 1.01)          # inside already
    assert B.extend_to(0.99, 1.01, [0.98]) == (0.98, 1.01)                # span 0.02 of 0.03: readable
    assert B.extend_to(0.99, 1.01, [0.88, 0.95]) is None                  # 0.02 of 0.13: arrows instead
    assert B.extend_to(0.1, 0.2, [np.nan, 0.15]) == (0.1, 0.2)


def test_the_spread_band_is_the_sd_over_three_runs_the_range_over_two_and_none_for_one():
    per = np.array([[1.0, 4.0], [2.0, 6.0], [6.0, 8.0]])
    lo, hi = B.spread_band(per)
    sd = per.std(axis=0, ddof=1)
    assert lo == pytest.approx(per.mean(axis=0) - sd) and hi == pytest.approx(per.mean(axis=0) + sd)
    lo, hi = B.spread_band(per[:2])
    assert list(lo) == [1.0, 4.0] and list(hi) == [2.0, 6.0]
    assert B.spread_band(per[:1]) is None


def test_every_change_marker_names_a_gallery_page_and_says_new_or_changed():
    tree = ast.parse((FIG / "gallery_figures.py").read_text())
    order = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "ORDER")
    changes = _load("gallery_review").CHANGES
    assert set(changes) <= set(order) and "F10c" in order
    assert all(v.startswith(("NEW: ", "CHANGED: ")) and v.endswith(".") for v in changes.values())
    assert changes["F10c"].startswith("NEW: ")
