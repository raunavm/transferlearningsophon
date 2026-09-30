"""src/stats/paired.py: every weighted metric against its unweighted definition,
the ratio against a case with a known answer, the Poisson interval against
published values, and the refusal to pair two runs whose streams differ."""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.stats import paired as P

REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _two_class(n=4000, sep=1.5, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    s = rng.normal(size=n) + sep * y
    if ties:
        s = np.round(s, 1)
    return y, s


@pytest.mark.parametrize("ties", [False, True])
def test_auc_unit_weights_is_sklearn(ties):
    y, s = _two_class(ties=ties)
    assert P.AucScorer(y, s).auc() == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_integer_weights_is_expanded_sample():
    y, s = _two_class(n=600, ties=True, seed=3)
    w = next(P.boot_counts(y.size, 1, seed=11))
    rows = np.repeat(np.arange(y.size), w.astype(int))
    assert P.AucScorer(y, s).auc(w) == pytest.approx(roc_auc_score(y[rows], s[rows]), abs=1e-12)


def test_one_minus_auc_floors_at_one_pair():
    y = np.r_[np.zeros(50, bool), np.ones(40, bool)]
    s = np.r_[np.zeros(50), np.ones(40)]
    sc = P.AucScorer(y, s)
    assert sc.auc() == 1.0
    assert sc() == pytest.approx(1.0 / (50 * 40))


def test_macro_auc_matches_eval_arm_including_absent_classes():
    eval_arm = _load("eval_arm", "experiments/EVAL/eval_arm.py")
    rng = np.random.default_rng(5)
    n, k = 3000, 6
    y = rng.integers(0, 5, n)                 # class 5 absent -> renormalised subset
    z = rng.normal(size=(n, k)) + 1.2 * np.eye(k)[y]
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ref = eval_arm.metrics(p, y, k, 0)["macro_auc_ovr"]
    assert 1 - P.MacroAucScorer(y, p)() == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("eps", [0.5, 0.7, 0.9])
def test_eps_b_matches_probe_rejection_at(eps):
    probe = _load("probe", "experiments/EVAL/probe.py")
    y, s = _two_class(n=5000, sep=2.5, seed=9)
    r, eps_b, *_ = probe.rejection_at(y.astype(int), s, eps)
    assert P.EpsBScorer(y, s, eps).eps_b() == pytest.approx(eps_b, abs=1e-15)


def test_eps_b_integer_weights_is_expanded_sample():
    probe = _load("probe", "experiments/EVAL/probe.py")
    y, s = _two_class(n=800, sep=2.0, seed=2)
    w = next(P.boot_counts(y.size, 1, seed=4))
    rows = np.repeat(np.arange(y.size), w.astype(int))
    _, eps_b, *_ = probe.rejection_at(y[rows].astype(int), s[rows], 0.9)
    assert P.EpsBScorer(y, s, 0.9).eps_b(w) == pytest.approx(eps_b, abs=1e-15)


def test_sigma_eff_matches_mass_resolution():
    mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
    rng = np.random.default_rng(1)
    res = np.r_[rng.normal(size=900) * 0.1, rng.normal(size=100)]
    assert P.SigmaEffScorer(res)() == pytest.approx(mr.sigma_eff(res)[0], abs=1e-15)
    w = next(P.boot_counts(res.size, 1, seed=8))
    rows = np.repeat(np.arange(res.size), w.astype(int))
    assert P.SigmaEffScorer(res)(w) == pytest.approx(mr.sigma_eff(res[rows])[0], abs=1e-15)


def test_boot_counts_depend_only_on_n_seed_and_index():
    a = list(P.boot_counts(100, 3, seed=7))
    b = list(P.boot_counts(100, 3, seed=7))
    assert all((x == y).all() for x, y in zip(a, b))
    assert all(x.sum() == 100 for x in a)
    assert not (a[0] == a[1]).all()


def test_paired_ratio_known_answer_and_error_terms():
    # coarse = fine * exp(d_k) for runs k: the ratio is exp(mean d), the run SD is SD(d),
    # and with replicate vectors that do not move the test term is zero.
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fine = {k: np.full(11, 0.01 * (k + 1)) for k in range(5)}
    coarse = {k: fine[k] * math.exp(d[k]) for k in range(5)}
    r = P.paired_ratio(fine, coarse)
    assert r["ratio"] == pytest.approx(math.exp(0.3))
    assert r["ln_run_sd"] == pytest.approx(d.std(ddof=1))
    assert r["ln_test_se"] == pytest.approx(0.0, abs=1e-15)
    assert r["ln_combined_se"] == pytest.approx(d.std(ddof=1) / math.sqrt(5))
    assert r["run_range"] == pytest.approx([math.exp(0.1), math.exp(0.5)])
    assert r["dof"] == pytest.approx(4.0)


def test_paired_ratio_test_term_is_the_replicate_spread_of_the_run_mean():
    rng = np.random.default_rng(0)
    fine = {k: np.exp(rng.normal(size=201) * 0.05) for k in range(3)}
    coarse = {k: fine[k] * 2.0 for k in range(3)}
    shift = rng.normal(size=200) * 0.1
    for k in coarse:
        coarse[k] = coarse[k].copy()
        coarse[k][1:] *= np.exp(shift)            # common to every run: does not average away
    r = P.paired_ratio(fine, coarse)
    assert r["ratio"] == pytest.approx(2.0)
    assert r["ln_test_se"] == pytest.approx(shift.std(ddof=1), rel=1e-12)


def test_paired_ratio_uses_explicit_pairs():
    fine = {1: np.full(3, 0.1), 2: np.full(3, 0.2)}
    coarse = {"d1": np.full(3, 0.2), "d2": np.full(3, 0.4)}
    r = P.paired_ratio(fine, coarse, pairs={"d1": 1, "d2": 2})
    assert r["ratio"] == pytest.approx(2.0)
    assert r["pairs"] == {"d1": "1", "d2": "2"}


def test_bootstrap_test_error_is_right_for_a_known_auc():
    # Two independent models on the same 2000 jets: the replicate SD of ln(1-AUC)
    # agrees with the Hanley-McNeil standard error within 20 %.
    y, s = _two_class(n=2000, sep=1.0, seed=12)
    sc = P.AucScorer(y, s)
    v = P.replicates(sc, y.size, b=400, seed=3)
    a = 1 - v[0]
    n1, n0 = y.sum(), (~y).sum()
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    assert np.std(v[1:], ddof=1) == pytest.approx(se, rel=0.2)


def test_garwood_interval_published_values():
    # Garwood 68.27 %: k = 0 -> [0, 1.841]; k = 10 -> [6.89, 14.27] (PDG Table 40.3, two decimals).
    assert P.poisson_interval(0) == pytest.approx((0.0, 1.8410), abs=1e-3)
    lo, hi = P.poisson_interval(10)
    assert (lo, hi) == pytest.approx((6.89, 14.27), abs=5e-3)


def test_rejection_interval_reproduces_the_audit_pooled_value():
    # audit B3: 188 classes, 10+9+6+9+11 = 45 of 5 x 11,876 -> 1,320 (1,130-1,550)
    r = P.rejection_interval(5 * 11876, 45)
    assert round(r["rejection"], -1) == 1320
    assert [round(x, -1) for x in r["interval"]] == [1130, 1550]
    assert P.rejection_interval(11876, 0)["is_bound"]


def _write_stream(d: pathlib.Path, rows):
    """Records in experiments/MTX/pretrain_v2.py's format: sha256 is
    sha256(files_sha256 + rows_sha256), which stream_ids.load_stream checks."""
    import hashlib
    (d / "stream").mkdir(parents=True)
    for e, r in enumerate(rows):
        (d / "stream" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"run": d.name, "epoch": e, "seed_data": 1, "seed_dropout": 1,
             "files_sha256": "f", "rows_sha256": r,
             "sha256": hashlib.sha256(("f" + r).encode()).hexdigest(), "n_jets": 10}))


def test_refuses_v2_runs_whose_streams_differ(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _write_stream(a, ["x", "y"])
    _write_stream(b, ["x", "y"])
    _write_stream(c, ["x", "z"])
    assert P.stream_pairing(a, b) == "identical"
    with pytest.raises(SystemExit, match="not a pair|not paired|differ"):
        P.stream_pairing(a, c)
    v = {k: np.full(3, 0.1) for k in "ab"}
    w = {k: np.full(3, 0.2) for k in "ab"}
    with pytest.raises(SystemExit):
        P.paired_ratio({"a": v["a"]}, {"c": w["a"]}, pairs={"c": "a"},
                       run_dirs={"a": a, "c": c})
    r = P.paired_ratio({"a": v["a"]}, {"b": w["a"]}, pairs={"b": "a"},
                       run_dirs={"a": a, "b": b})
    assert r["stream_pairing"] == "identical"


def test_v1_runs_pair_by_index_and_say_so(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert P.stream_pairing(tmp_path / "a", tmp_path / "b") == "v1"
    _write_stream(tmp_path / "c", ["x"])
    with pytest.raises(SystemExit, match="recorded no training"):
        P.stream_pairing(tmp_path / "a", tmp_path / "c")


def test_macro_auc_skips_a_class_absent_from_a_resampling():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500, int), np.ones(500, int), [2]]      # class 2: one jet
    p = rng.random((y.size, 3))
    p /= p.sum(1, keepdims=True)
    sc = P.MacroAucScorer(y, p)
    w = np.ones(y.size)
    w[-1] = 0                                                  # the resampling missed it
    assert np.isfinite(sc(w))
    assert sc.auc(w) == pytest.approx(np.mean([s.auc(w) for s in sc.scorers[:2]]))
