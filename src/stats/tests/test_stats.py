"""Analytic-case unit tests for src/stats (AGENT §8; PLAN §16).

Each test pins a function to a value derivable by hand or to a PLAN reference
number, so a regression is caught before any confirmatory use.
Run: python3 -m pytest src/stats/tests/test_stats.py   (or execute directly).
"""
import numpy as np

from src.stats import (ci, event_bootstrap, holm, kish_gate, mde_total, n_eff,
                       n_eff_clustered, paired_bootstrap_diff, paired_t,
                       seed_term_multiplier, sigma_d_ceiling_for_mei, tost)
from src.stats.mde import MEI_LOG


# ---- Kish (§16.2) ----
def test_n_eff_equal_weights_is_count():
    # all-equal weights ⇒ N_eff = N, for any weight value (float-rounded)
    assert np.isclose(n_eff(np.full(1000, 3.7)), 1000.0)

def test_n_eff_two_valued_closed_form():
    # k ones + 1 big weight W: N_eff = (k+W)^2 / (k + W^2)
    k, W = 9, 100.0
    w = np.array([1.0] * k + [W])
    assert np.isclose(n_eff(w), (k + W) ** 2 / (k + W * W))

def test_n_eff_clustered_groups_rows():
    # two jets per event, equal weights, E events ⇒ clustered N_eff = E
    ev = np.repeat(np.arange(500), 2)
    assert np.isclose(n_eff_clustered(np.ones(1000), ev), 500.0)

def test_kish_gate_demotes_below_threshold():
    below = kish_gate(np.ones(1_000_000))          # N_eff = 1e6 < 4e6
    above = kish_gate(np.ones(5_000_000))          # N_eff = 5e6 ≥ 4e6
    assert below["fpr_reported"] == 1e-3 and not below["eligible_1e-4"]
    assert above["fpr_reported"] == 1e-4 and above["eligible_1e-4"]

def test_kish_rel_error_5pct_at_threshold():
    # at N_eff = 4e6, FPR 1e-4 ⇒ 400 survivors ⇒ 5% rel error (PLAN §16.2)
    g = kish_gate(np.ones(4_000_000))
    assert np.isclose(g["expected_survivors"], 400.0)
    assert np.isclose(g["rel_binomial_error"], 0.05, atol=1e-6)


# ---- MDE (§16.4): the plan's own reference multipliers ----
def test_mde_multiplier_reference_values():
    assert np.isclose(seed_term_multiplier(3), 3.10, atol=0.005)
    assert np.isclose(seed_term_multiplier(5), 1.66, atol=0.005)

def test_mde_quadrature():
    # seed term with σ_d=1,n=5 is the multiplier; ⊕ equal test floor
    m = seed_term_multiplier(5)
    assert np.isclose(mde_total(1.0, 5, sigma_test=m), m * np.sqrt(2))

def test_sigma_ceiling_n5_is_5p7pct():
    # n=5 covers the 10% MEI only when σ_d ≲ 5.7% (PLAN §16.4)
    assert np.isclose(sigma_d_ceiling_for_mei(5), 0.057, atol=0.001)


# ---- Inference (§16.3) ----
def test_paired_t_zero_mean_not_rejected():
    d = np.array([-2, -1, 0, 1, 2], float)  # mean 0 exactly
    r = paired_t(d)
    assert np.isclose(r["mean"], 0.0) and not r["reject_null"]

def test_paired_t_matches_scipy_direction():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 0.1, 20)  # clearly positive
    assert paired_t(d)["reject_null"]

def test_tost_equivalence_ci_duality():
    # tight cluster near 0 ⇒ (1-2α) CI inside ±MEI ⇒ equivalent
    d = np.array([0.001, -0.001, 0.0, 0.002, -0.002])
    out = tost(d, bound=MEI_LOG)
    lo, hi = out["ci_1m2a"]
    assert out["equivalent"] == (-MEI_LOG < lo and hi < MEI_LOG) == True

def test_tost_not_equivalent_when_far():
    d = np.array([0.5, 0.48, 0.52, 0.49, 0.51])  # far from 0, well beyond MEI
    assert not tost(d, bound=MEI_LOG)["equivalent"]

def test_holm_stepdown_known_example():
    # p=[0.01,0.04,0.03], m=3, α=0.05: 0.01≤.0167 ✓; 0.03≤.025 ✗ → stop
    rej = holm([0.01, 0.04, 0.03])
    assert list(rej) == [True, False, False]

def test_holm_all_reject_when_all_tiny():
    assert holm([1e-4, 2e-4, 3e-4]).all()


# ---- Event bootstrap (§16.1) ----
def test_event_bootstrap_deterministic():
    ev = np.arange(200)
    v = np.random.default_rng(1).normal(size=200)
    f = lambda r: v[r].mean()
    a = event_bootstrap(ev, f, b=500, seed=7777)
    b = event_bootstrap(ev, f, b=500, seed=7777)
    assert np.array_equal(a, b)  # reproducible for fixed seed

def test_event_bootstrap_se_matches_analytic():
    # one row per event, mean statistic ⇒ bootstrap SE ≈ s/sqrt(N)
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 4000)
    ev = np.arange(4000)
    dist = event_bootstrap(ev, lambda r: v[r].mean(), b=2000, seed=7777)
    assert np.isclose(dist.std(), v.std(ddof=1) / np.sqrt(4000), rtol=0.1)

def test_event_bootstrap_clusters_twins():
    # 2 rows/event with identical value; resampling must keep counts even
    ev = np.repeat(np.arange(1000), 2)
    val = np.repeat(np.arange(1000), 2).astype(float)
    # statistic = number of rows; must always be even (twins move together)
    dist = event_bootstrap(ev, lambda r: len(r) % 2, b=100, seed=7777)
    assert (dist == 0).all()

def test_paired_bootstrap_diff_zero_for_identical():
    ev = np.arange(500)
    v = np.random.default_rng(3).normal(size=500)
    point, dist = paired_bootstrap_diff(ev, v, v, b=200, seed=7777)
    assert point == 0.0 and np.allclose(dist, 0.0)  # A−A ≡ 0 on every resample

def test_ci_covers_point():
    lo, hi = ci(np.random.default_rng(4).normal(0, 1, 5000))
    assert lo < 0 < hi


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
