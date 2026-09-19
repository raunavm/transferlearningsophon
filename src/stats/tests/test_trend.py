"""Unit tests for src/stats/trend.py (STATISTICS §2: multiple-contrast trend test).

Each test pins the module to a hand-worked value, to an independent brute-force
re-derivation, or to a property the permutation test must have (size, exact ≈
Monte-Carlo, invariance). Every generator is seeded explicitly.
Run: python3 -m pytest src/stats/tests/test_trend.py
"""
import itertools

import numpy as np
import pytest

from src.stats import contrast_matrix, isotonic_fit, isotonic_trend, max_t_trend
from src.stats.trend import _isotonic

LEVELS = [17, 43, 162, 188]
FAMILIES = ["williams", "marcus", "changepoint"]


def _rcbd(b, means, rng, block_sd=10.0, noise=1.0):
    """b seed blocks x 4 levels with strong block effects."""
    lv, bl = np.tile(LEVELS, b), np.repeat(np.arange(b), 4)
    y = np.tile(means, b) + np.repeat(rng.normal(0, block_sd, b), 4) + rng.normal(0, noise, 4 * b)
    return y, lv, bl


# ---- contrasts ----
@pytest.mark.parametrize("family", FAMILIES)
def test_contrasts_sum_to_zero_unit_norm(family):
    for k, n in [(3, None), (4, None), (5, None), (4, [5, 4, 5, 3])]:
        c = contrast_matrix(k, family, n)
        assert c.shape == (k * (k - 1) // 2 if family == "marcus" else k - 1, k)
        assert np.allclose(c.sum(1), 0.0) and np.allclose((c ** 2).sum(1), 1.0)

def test_contrast_rows_hand_worked():
    unit = lambda v: np.array(v) / np.linalg.norm(v)
    w = contrast_matrix(4, "williams")
    assert np.allclose(w, [unit([-1, 0, 0, 1]), unit([-1, 0, .5, .5]), unit([-1, 1/3, 1/3, 1/3])])
    assert np.allclose(contrast_matrix(4, "changepoint")[1], unit([-.5, -.5, .5, .5]))
    # unbalanced (Bretz 2006): pooled sets are n-weighted — n = [2, 1, 3]
    assert np.allclose(contrast_matrix(3, "williams", [2, 1, 3])[1], unit([-1, 1/4, 3/4]))

def test_marcus_contains_williams_and_changepoint():
    m = contrast_matrix(4, "marcus")
    for family in ("williams", "changepoint"):
        for row in contrast_matrix(4, family):
            assert np.isclose(m, row).all(1).any()


# ---- max-T: locating a planted step ----
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pos", [1, 2, 3])
def test_planted_step_is_located(family, pos):
    means = np.array([0.0] * pos + [-3.0] * (4 - pos))   # drops between pos-1 and pos
    y, lv, bl = _rcbd(4, means, np.random.default_rng(100 + pos), noise=0.2)
    r = max_t_trend(y, lv, bl, alternative="decreasing", family=family)
    lower, upper = r["step"]
    assert upper == LEVELS[pos:] and r["p"] < 0.01
    if family != "williams":                             # williams pins lower to the control
        assert lower == LEVELS[:pos]
    assert r["contrasts"][r["argmax"]]["difference"] < -2.5

def test_direction_is_respected():
    y, lv, bl = _rcbd(4, [0, 0, -3, -3], np.random.default_rng(5), noise=0.2)
    assert max_t_trend(y, lv, bl, alternative="increasing")["p"] > 0.9
    assert max_t_trend(y, lv, bl, alternative="two-sided")["p"] < 0.01
    # reversing the level order turns the decrease into an increase
    r = max_t_trend(y, lv, bl, alternative="increasing", order=LEVELS[::-1])
    assert r["p"] < 0.01 and r["step"] == ([188, 162], [43, 17])


# ---- size under the null, strong block effects ----
def test_size_controlled_under_block_effects():
    sim, perm = np.random.default_rng(2024), np.random.default_rng(99)
    n_sim, rej_t, rej_iso = 2000, 0, 0
    for _ in range(n_sim):
        y, lv, bl = _rcbd(5, np.zeros(4), sim)
        rej_t += max_t_trend(y, lv, bl, exact=False, n_perm=199, rng=perm)["p"] <= 0.05
        rej_iso += isotonic_trend(y, lv, bl, exact=False, n_perm=199, rng=perm)["p"] <= 0.05
    assert rej_t / n_sim <= 0.07 and rej_iso / n_sim <= 0.07
    assert rej_t / n_sim >= 0.03                          # and not vacuously conservative


# ---- exact enumeration ----
def _brute_force(table, c, directed):
    """Max statistic of every arrangement, from two-way residuals — no shared code."""
    b, k = table.shape
    out = []
    for combo in itertools.product(itertools.permutations(range(k)), repeat=b):
        t = np.array([row[list(p)] for row, p in zip(table, combo)])
        resid = t - t.mean(1, keepdims=True) - t.mean(0) + t.mean()
        s2 = (resid ** 2).sum() / ((b - 1) * (k - 1))
        out.append(directed(c @ t.mean(0) / np.sqrt(s2 * (c ** 2).sum(1) / b)))
    return np.array(out)                                  # out[0] is the identity = observed

@pytest.mark.parametrize("alternative,directed", [("decreasing", np.negative), ("two-sided", np.abs)])
def test_exact_p_and_p_min_match_brute_force(alternative, directed):
    y, lv, bl = _rcbd(3, [0, -.5, -1, -1], np.random.default_rng(7))
    r = max_t_trend(y, lv, bl, alternative=alternative)
    stats = _brute_force(y.reshape(3, 4), contrast_matrix(4, "marcus"), directed)
    tmax = stats.max(1)
    assert r["method"] == "exact" and r["n_arrangements"] == 24 ** 3 == len(tmax)
    assert np.isclose(r["stat"], tmax[0])
    assert np.isclose(r["p"], np.mean(tmax >= tmax[0] - 1e-9))
    p_adj = [np.mean(tmax >= s - 1e-9) for s in stats[0]]
    assert np.allclose([c["p_adj"] for c in r["contrasts"]], p_adj)
    # smallest attainable p = share of arrangements tied at the maximum, NOT 1/N:
    # relabelling levels inside a pooled set leaves that contrast's t unchanged
    assert np.isclose(r["p_min"], np.mean(tmax >= tmax.max() - 1e-9)) and r["p_min"] > 1 / 24 ** 3

def test_exact_matches_monte_carlo():
    y, lv, bl = _rcbd(4, [0, -.3, -.9, -1.0], np.random.default_rng(12))
    n_perm = 40_000
    for fn in (max_t_trend, isotonic_trend):
        exact, mc = fn(y, lv, bl), fn(y, lv, bl, exact=False, n_perm=n_perm, rng=321)
        assert exact["method"] == "exact" and mc["method"] == "monte-carlo"
        assert 0.005 < exact["p"] < 0.5                   # a p-value worth comparing
        se = np.sqrt(exact["p"] * (1 - exact["p"]) / n_perm)
        assert abs(exact["p"] - mc["p"]) < 4 * se
        assert mc["p_min"] == 1 / (n_perm + 1) and mc["n_arrangements"] == n_perm

def test_five_by_four_design_is_enumerated():
    n_arr = 24 ** 5
    assert n_arr == 7_962_624
    y, lv, bl = _rcbd(5, [0, -3, -3, -3], np.random.default_rng(1), noise=0.2)   # step 17 | 43
    r = max_t_trend(y, lv, bl)                                                  # no rng needed
    assert r["method"] == "exact" and r["rng"] is None
    assert r["n_arrangements"] == n_arr and r["n_blocks"] == 5
    assert r["step"] == ([17], [43, 162, 188])
    # a pure end step cannot reach p_min: the 6^5 arrangements that keep level 17
    # on top of every block share the numerator, so p is of order (1/4)^5 ~ 1e-3
    assert r["p_min"] < r["p"] < (1 / 4) ** 5
    # all four levels ordered in every block: the observed arrangement is the extreme
    y, lv, bl = _rcbd(5, [0, -3, -6, -9], np.random.default_rng(1), noise=0.2)
    r, iso = max_t_trend(y, lv, bl), isotonic_trend(y, lv, bl)
    assert np.isclose(iso["p_min"], 1 / n_arr) and iso["p"] == iso["p_min"]   # sorted is unique
    ties = r["p_min"] * n_arr
    assert np.isclose(ties, round(ties)) and 1 <= round(ties) <= 6 and r["p"] == r["p_min"]


# ---- invariance, missing cells ----
def test_invariant_to_block_constants():
    y, lv, bl = _rcbd(4, [0, 0, -1, -1], np.random.default_rng(3))
    shift = np.repeat([100.0, -7.0, 0.5, 42.0], 4)
    a, b = max_t_trend(y, lv, bl), max_t_trend(y + shift, lv, bl)
    assert np.isclose(a["stat"], b["stat"]) and np.isclose(a["p"], b["p"]) and a["step"] == b["step"]
    ia, ib = isotonic_trend(y, lv, bl), isotonic_trend(y + shift, lv, bl)
    assert np.isclose(ia["stat"], ib["stat"]) and np.isclose(ia["p"], ib["p"])

def test_missing_cell_drops_block_and_reports_it():
    y, lv, bl = _rcbd(5, [0, 0, 0, -3], np.random.default_rng(4), noise=0.2)
    gone = ~((bl == 2) & (lv == 43))
    r = max_t_trend(y[gone], lv[gone], bl[gone])
    assert r["blocks_dropped"] == [2] and r["n_blocks"] == 4 and r["n_arrangements"] == 24 ** 4
    full = max_t_trend(y[bl != 2], lv[bl != 2], bl[bl != 2])
    assert np.isclose(r["stat"], full["stat"]) and np.isclose(r["p"], full["p"])
    y_nan = y.copy()
    y_nan[(bl == 2) & (lv == 43)] = np.nan                # a non-finite endpoint is a missing cell
    assert isotonic_trend(y_nan, lv, bl)["blocks_dropped"] == [2]

def test_duplicate_cell_rejected():
    y, lv, bl = _rcbd(3, np.zeros(4), np.random.default_rng(6))
    with pytest.raises(ValueError, match="more than one endpoint"):
        max_t_trend(np.append(y, 0.0), np.append(lv, 17), np.append(bl, 0))


# ---- isotonic regression ----
def test_isotonic_hand_worked():
    # [1,3,2,4]: 3>2 violates, pool to 2.5. SS about 2.5: raw 5.0, isotonic 4.5
    f = isotonic_fit([1, 3, 2, 4], LEVELS, increasing=True)
    assert np.allclose(f["fitted"], [1, 2.5, 2.5, 4]) and f["pooled"] == [[17], [43, 162], [188]]
    assert np.isclose(f["explained"], 0.9)
    # cascade: [3,1,2,0] -> [2,2,2,0] -> pool to 1 < 2 -> pool everything to 1.5
    f = isotonic_fit([3, 1, 2, 0], LEVELS, increasing=True)
    assert np.allclose(f["fitted"], 1.5) and f["pooled"] == [LEVELS] and np.isclose(f["explained"], 0)
    # weights = n per level: means [1,3,2], n [1,1,3] -> (3*1 + 2*3)/4 = 2.25
    f = isotonic_fit([1, 3, 2, 2, 2], [1, 2, 3, 3, 3], increasing=True)
    assert np.allclose(f["fitted"], [1, 2.25, 2.25]) and f["n"] == [1, 1, 3]
    # decreasing is the default direction, as in max_t_trend
    f = isotonic_fit([4, 2, 3, 1], LEVELS)
    assert np.allclose(f["fitted"], [4, 2.5, 2.5, 1]) and f["pooled"] == [[17], [43, 162], [188]]
    # already monotone: nothing pooled, everything explained
    f = isotonic_fit([4, 3, 2, 1], LEVELS)
    assert f["pooled"] == [[v] for v in LEVELS] and np.isclose(f["explained"], 1.0)

def test_e_bar_squared_hand_worked():
    # two identical blocks [1,3,2,4]: isotonic SS 2*4.5, within-block SS 2*5.0
    r = isotonic_trend([1, 3, 2, 4] * 2, LEVELS * 2, [0] * 4 + [1] * 4, increasing=True)
    assert np.isclose(r["stat"], 0.9) and r["n_arrangements"] == 24 ** 2
    assert r["fit"]["pooled"] == [[17], [43, 162], [188]]

def test_isotonic_exact_p_matches_brute_force():
    sk = pytest.importorskip("sklearn.isotonic")
    table = np.random.default_rng(14).normal(size=(3, 3)) + [0, .5, 1]
    r = isotonic_trend(table.ravel(), np.tile([1, 2, 3], 3), np.repeat(np.arange(3), 3),
                       increasing=True)
    e2 = []
    for combo in itertools.product(itertools.permutations(range(3)), repeat=3):
        t = np.array([row[list(p)] for row, p in zip(table, combo)])
        fit = sk.IsotonicRegression().fit_transform(np.arange(3), t.mean(0))
        e2.append(3 * ((fit - t.mean()) ** 2).sum() / ((t - t.mean(1, keepdims=True)) ** 2).sum())
    e2 = np.array(e2)                                     # e2[0] is the identity = observed
    assert r["n_arrangements"] == 6 ** 3 == len(e2) and np.isclose(r["stat"], e2[0])
    assert np.isclose(r["p"], np.mean(e2 >= e2[0] - 1e-12))

def test_isotonic_matches_sklearn():
    sk = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(8)
    for k in (3, 4, 6):
        for _ in range(30):
            m, w = rng.normal(size=k), rng.integers(1, 6, k).astype(float)
            ref = sk.IsotonicRegression().fit_transform(np.arange(k), m, sample_weight=w)
            assert np.allclose(_isotonic(m, w), ref)
    batch = rng.normal(size=(50, 4))                      # vectorised over arrangements
    assert np.allclose(_isotonic(batch, np.ones(4))[7], _isotonic(batch[7], np.ones(4)))


# ---- RNG discipline ----
def test_reproducible_given_seed_and_seed_recorded():
    y, lv, bl = _rcbd(5, [0, 0, -.5, -1], np.random.default_rng(9))
    a = max_t_trend(y, lv, bl, exact=False, n_perm=2000, rng=123)
    b = max_t_trend(y, lv, bl, exact=False, n_perm=2000, rng=np.random.default_rng(123))
    assert a["p"] == b["p"] and [c["p_adj"] for c in a["contrasts"]] == [c["p_adj"] for c in b["contrasts"]]
    assert a["rng"]["seed"] == 123 and b["rng"]["seed"] is None
    assert a["rng"]["state"] == b["rng"]["state"] == np.random.default_rng(123).bit_generator.state
    c = max_t_trend(y, lv, bl, exact=False, n_perm=2000, rng=124)
    assert c["rng"]["state"] != a["rng"]["state"]

def test_monte_carlo_refuses_without_explicit_rng():
    y, lv, bl = _rcbd(6, np.zeros(4), np.random.default_rng(10))   # 24^6 > EXACT_MAX
    for kwargs in ({"blocks": bl}, {"blocks": bl[:20], "exact": False}, {"blocks": None}):
        n = len(kwargs["blocks"]) if kwargs["blocks"] is not None else 24
        with pytest.raises(ValueError, match="explicit rng"):
            max_t_trend(y[:n], lv[:n], **kwargs)
        with pytest.raises(ValueError, match="explicit rng"):
            isotonic_trend(y[:n], lv[:n], **kwargs)


# ---- unblocked variant ----
def test_unblocked_path():
    rng = np.random.default_rng(11)
    lv = np.repeat(LEVELS, [5, 4, 5, 3])                  # unbalanced independent groups
    y = np.where(lv == 17, 0.0, -3.0) + rng.normal(0, 0.5, len(lv))
    r = max_t_trend(y, lv, None, n_perm=9999, rng=1)
    assert r["method"] == "monte-carlo" and r["n_blocks"] is None and r["blocks_dropped"] == []
    assert r["step"] == ([17], [43, 162, 188]) and r["p_min"] == 1 / 10_000 and r["p"] < 1e-3
    assert isotonic_trend(y, lv, None, n_perm=9999, rng=1)["p"] < 1e-3
    with pytest.raises(ValueError, match="blocked design only"):
        max_t_trend(y, lv, None, exact=True)

def test_blocking_removes_seed_variance():
    # same paired data: within-block permutation sees the step, the unblocked
    # reference is swamped by the seed-to-seed spread
    y, lv, bl = _rcbd(5, [0, 0, 0, -1.5], np.random.default_rng(13), noise=0.3)
    assert max_t_trend(y, lv, bl)["p"] < 0.01
    assert max_t_trend(y, lv, None, n_perm=1999, rng=2)["p"] > 0.2
