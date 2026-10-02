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


def _write_stream(d: pathlib.Path, rows, best=None):
    """Records in experiments/MTX/pretrain_v2.py's format: sha256 is
    sha256(files_sha256 + rows_sha256), which stream_ids.load_stream checks.
    `best`: the run's best-validation epoch, written as best_epoch.json."""
    import hashlib
    (d / "stream").mkdir(parents=True)
    for e, r in enumerate(rows):
        (d / "stream" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"run": d.name, "epoch": e, "seed_data": 1, "seed_dropout": 1,
             "files_sha256": "f", "rows_sha256": r,
             "sha256": hashlib.sha256(("f" + r).encode()).hexdigest(), "n_jets": 10}))
    if best is not None:
        (d / "best_epoch.json").write_text(json.dumps({"epoch": best}))


def _rows(diverge_at=None, n=80):
    return [f"x{e}" if diverge_at is None or e < diverge_at else f"y{e}" for e in range(n)]


def test_a_pair_whose_streams_differ_is_excluded_and_the_rest_are_used(tmp_path):
    # A7: pair 2 diverges at epoch 75, after both best-validation epochs (40, 55)
    # and inside the weight average (70-79); pair 3 diverges at 50, inside 0-55.
    d = {}
    for k, (div, best_f, best_c) in {1: (None, 40, 41), 2: (75, 40, 55), 3: (50, 40, 55)}.items():
        _write_stream(tmp_path / f"f{k}", _rows(), best_f)
        _write_stream(tmp_path / f"c{k}", _rows(div), best_c)
        d[f"f{k}"], d[f"c{k}"] = tmp_path / f"f{k}", tmp_path / f"c{k}"
    rng = np.random.default_rng(1)
    fine = {f"f{k}": np.exp(rng.normal(size=21) * 0.01) * 0.1 for k in (1, 2, 3)}
    coarse = {f"c{k}": fine[f"f{k}"] * math.exp(0.1 * k) for k in (1, 2, 3)}
    pairs = {f"c{k}": f"f{k}" for k in (1, 2, 3)}
    r = P.paired_ratio(fine, coarse, pairs=pairs, run_dirs=d, checkpoint="bestval")
    assert r["stream_pairing"] == "identical" and r["n_runs"] == 2
    assert r["pairs"] == {"c1": "f1", "c2": "f2"}
    assert [(x["pair"], x["first_bad_epoch"], x["upto_epoch"]) for x in r["excluded_pairs"]] == \
        [(["c3", "f3"], 50, 55)]
    assert r["ratio"] == pytest.approx(math.exp(0.15))
    w = P.paired_ratio(fine, coarse, pairs=pairs, run_dirs=d, checkpoint="wavg")
    assert w["pairs"] == {"c1": "f1"} and w["n_runs"] == 1
    assert {x["pair"][0]: x["first_bad_epoch"] for x in w["excluded_pairs"]} == {"c2": 75, "c3": 50}
    none = P.paired_ratio({"f3": fine["f3"]}, {"c3": coarse["c3"]}, pairs={"c3": "f3"},
                          run_dirs=d, checkpoint="wavg")
    assert none["n_runs"] == 0 and "not_computed" in none and len(none["excluded_pairs"]) == 1


def test_run_dirs_must_exist_and_hold_their_stream_records(tmp_path):
    # A mistyped root used to pass every v2 pair as an unchecked v1 pair.
    _write_stream(tmp_path / "a", _rows(), 10)
    (tmp_path / "empty").mkdir()
    _write_stream(tmp_path / "nobest", _rows())
    v = {"a": np.full(3, 0.1)}
    for other, msg in ((tmp_path / "missing", "does not exist"),
                       (tmp_path / "empty", "no stream record"),
                       (tmp_path / "nobest", "best-validation epoch is unknown")):
        with pytest.raises(SystemExit, match=msg):
            P.paired_ratio(v, {"b": np.full(3, 0.2)}, pairs={"b": "a"},
                           run_dirs={"a": tmp_path / "a", "b": other}, checkpoint="bestval")
    with pytest.raises(SystemExit, match="needs the checkpoint"):
        P.paired_ratio(v, {"b": np.full(3, 0.2)}, pairs={"b": "a"},
                       run_dirs={"a": tmp_path / "a", "b": tmp_path / "a"})
    r = P.paired_ratio(v, {"b": np.full(3, 0.2)}, pairs={"b": "a"})
    assert r["stream_pairing"] == "unchecked" and "excluded_pairs" not in r


def _replicate_matrix(rng, point, shared_sd, ind_sd, b):
    """runs x (1 + b): point estimates, then replicates around them with a test
    fluctuation shared by every run and one of each run's own."""
    n = len(point)
    reps = (np.asarray(point)[:, None] + rng.normal(0, shared_sd, b)[None, :]
            + rng.normal(0, ind_sd, (n, b)))
    return np.c_[point, reps]


def test_the_error_counts_independent_test_noise_once():
    # var(mean) = max(s^2, v_ind)/n + v_shared, and the quadrature sum it replaces
    # exceeds it by exactly min(s^2, v_ind)/n.
    rng = np.random.default_rng(3)
    for point in ([0.10, 0.11, 0.09, 0.10, 0.105],          # runs agree: s^2 < v_ind
                  [0.0, 0.3, -0.2, 0.25, 0.1]):              # runs disagree: s^2 > v_ind
        L = _replicate_matrix(rng, point, 0.01, 0.05, 2000)
        r = P.paired_log({k: L[k] for k in range(5)})
        C = np.cov(L[:, 1:])
        off = (C.sum() - np.trace(C)) / 20
        assert r["v_shared_unclipped"] == pytest.approx(off, rel=1e-12)
        assert r["v_ind"] == pytest.approx(np.trace(C) / 5 - off, rel=1e-12)
        assert r["v_ind"] == pytest.approx(0.05 ** 2, rel=0.1)
        assert r["v_shared"] == pytest.approx(0.01 ** 2, rel=0.3)
        s2 = np.var(point, ddof=1)
        assert r["ln_combined_se"] ** 2 == pytest.approx(max(s2, r["v_ind"]) / 5 + r["v_shared"])
        # the test SE of the mean is v_ind/n + v_shared, so the old sum is larger by min(s^2, v_ind)/n
        assert r["ln_test_se"] ** 2 == pytest.approx(r["v_ind"] / 5 + r["v_shared_unclipped"])
        old = s2 / 5 + r["ln_test_se"] ** 2
        assert old - r["ln_combined_se"] ** 2 == pytest.approx(min(s2, r["v_ind"]) / 5)
        assert r["dof"] == pytest.approx(r["ln_combined_se"] ** 4 / ((s2 / 5) ** 2 / 4))


def test_shared_covariance_below_zero_is_clipped_and_recorded():
    rng = np.random.default_rng(4)
    z = rng.normal(size=500)
    L = np.c_[[0.0, 0.01], np.vstack([z, -z]) * 0.05]       # two runs, anticorrelated replicates
    r = P.paired_log({0: L[0], 1: L[1]})
    assert r["v_shared_unclipped"] < 0 and r["v_shared"] == 0.0
    assert r["v_ind"] == pytest.approx(np.trace(np.cov(L[:, 1:])) / 2)


def test_one_run_keeps_its_full_test_variance():
    rng = np.random.default_rng(5)
    v = np.r_[0.2, 0.2 + rng.normal(size=300) * 0.03]
    r = P.paired_log({1: v})
    assert r["ln_combined_se"] == pytest.approx(np.std(v[1:], ddof=1))
    assert r["ln_combined_se"] == r["ln_test_se"] and r["dof"] == math.inf


def _coverage(va, ve, vek, n, b, trials, seed):
    """Fraction of 95 % intervals that hold the true mean ln ratio (0), for the
    corrected error and for the quadrature sum it replaced."""
    from scipy.stats import t as student
    rng = np.random.default_rng(seed)
    hit_new = hit_old = 0
    for _ in range(trials):
        point = (rng.normal(0, math.sqrt(va), n) + rng.normal(0, math.sqrt(ve))
                 + rng.normal(0, math.sqrt(vek), n))
        L = _replicate_matrix(rng, point, math.sqrt(ve), math.sqrt(vek), b)
        r = P.paired_log({k: L[k] for k in range(n)})
        lo, hi = r["ci95"]
        hit_new += lo <= 1.0 <= hi
        s2 = np.var(point, ddof=1)
        comb = math.sqrt(s2 / n + r["ln_test_se"] ** 2)
        dof = comb ** 4 / ((s2 / n) ** 2 / (n - 1))
        hit_old += abs(r["ln_ratio"]) <= student.ppf(0.975, dof) * comb
    return hit_new / trials, hit_old / trials


def test_coverage_is_ninety_five_percent_where_the_quadrature_sum_overcovers():
    # ln r_k = a_k + e + e_k with var a = var e_k = 1, var e = 0.06, five runs:
    # the independent test noise is half the run spread, as in the b vs c cells.
    new, old = _coverage(1.0, 0.06, 1.0, n=5, b=200, trials=4000, seed=11)
    assert 0.935 <= new <= 0.965
    assert old - new >= 0.01 and old >= 0.96


def _two_groups(rng, n1, n2, va, vg, vu, b):
    """Two groups of units (runs of two vocabularies): ln y = run effect + test
    noise the group's units share + each unit's own; replicates redraw the test
    noise. Groups are independent of each other, so the noise one group shares
    survives in their difference."""
    out = []
    for n in (n1, n2):
        pt = rng.normal(0, math.sqrt(va), n) + rng.normal(0, math.sqrt(vg)) + rng.normal(0, math.sqrt(vu), n)
        reps = (pt[:, None] + rng.normal(0, math.sqrt(vg), b)[None, :]
                + rng.normal(0, math.sqrt(vu), (n, b)))
        out.append(np.c_[pt, reps])
    return np.vstack(out), [1 / n1] * n1 + [-1 / n2] * n2, [range(n1), range(n1, n1 + n2)]


def test_two_group_error_takes_the_test_noise_within_groups():
    rng = np.random.default_rng(21)
    L, w, groups = _two_groups(rng, 5, 3, 0.5, 1.0, 1.0, 500)
    e = P.combined_error(L, w, groups)
    C = np.cov(L[:, 1:])
    num = 0.0
    for g in (list(range(5)), list(range(5, 8))):
        Cg = C[np.ix_(g, g)]
        k = len(g)
        o = (Cg.sum() - np.trace(Cg)) / (k * (k - 1))
        num += (k - 1) * (np.trace(Cg) / k - max(o, 0.0))
    assert e["v_ind"] == pytest.approx(num / 6, rel=1e-12)
    pt = L[:, 0]
    s2 = (((pt[:5] - pt[:5].mean()) ** 2).sum() + ((pt[5:] - pt[5:].mean()) ** 2).sum()) / 6
    w = np.array(w)
    assert e["ln_test_se"] ** 2 == pytest.approx(w @ C @ w, rel=1e-12)
    assert e["ln_combined_se"] ** 2 == pytest.approx(
        (w ** 2).sum() * max(s2 - e["v_ind"], 0.0) + e["ln_test_se"] ** 2, rel=1e-12)
    assert e["ln_combined_se"] >= e["ln_test_se"]


def test_two_group_coverage_is_ninety_five_percent_when_a_group_shares_test_noise():
    # The form this replaced took one shared covariance for all units, which a
    # difference cancels: with the group-shared test noise as large as each
    # unit's own it covered 75 % (verification of 2026-10-01).
    from scipy.stats import t as student
    rng = np.random.default_rng(13)
    hit = hit_old = 0
    trials = 2000
    for _ in range(trials):
        L, w, groups = _two_groups(rng, 5, 5, 0.5, 1.0, 1.0, 200)
        e = P.combined_error(L, w, groups)
        hit += abs(e["estimate"]) <= P._t975(e["dof"]) * e["ln_combined_se"]
        C = np.cov(L[:, 1:])
        d, o = np.trace(C) / 10, (C.sum() - np.trace(C)) / 90
        s2 = e["ln_spread_sd"] ** 2
        old = math.sqrt(0.4 * max(s2, d - max(o, 0.0)))
        hit_old += abs(e["estimate"]) <= student.ppf(0.975, 8) * old
    assert 0.93 <= hit / trials <= 0.97
    assert hit_old / trials < 0.85


def test_two_group_error_is_never_below_its_test_error_on_the_v1_replicates():
    # The verifier's check (2026-10-01): two vocabularies' runs as unpaired groups,
    # on the committed probe replicates; the old form fell below the contrast's own
    # test SE in 99 of 124 cells.
    base = REPO / "experiments/FIGS/data/paired_v1err/probes"
    z = np.load(base / "replicates_probes.npz")
    meta = json.loads((base / "replicates_probes.json").read_text())["meta"]
    models = json.loads((REPO / "configs/analysis/contrasts.v1.json").read_text())["models"]
    cells = {}
    for k in meta:
        fam, task, kind, model, metric = k.split("|")
        arm = models[model][0]
        if arm in ("L188", "L162", "R42_Q1", "R16_Q1"):
            cells.setdefault((task, kind, metric), {}).setdefault(arm, []).append(k)
    n = 0
    for per in cells.values():
        arms = sorted(per)
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                V = [z[k] for k in per[b] + per[a]]
                if any(np.any(v <= 0) for v in V):
                    continue
                nb, na = len(per[b]), len(per[a])
                e = P.combined_error(np.log(V), [1 / nb] * nb + [-1 / na] * na,
                                     [range(nb), range(nb, nb + na)])
                assert e["ln_combined_se"] >= e["ln_test_se"] * (1 - 1e-12)
                n += 1
    assert n >= 120


def test_separate_spreads_follow_the_welch_formula():
    rng = np.random.default_rng(22)
    L, w, groups = _two_groups(rng, 5, 3, 0.5, 0.3, 1.0, 400)
    L[5:, 0] += rng.normal(0, 1.0, 3)                      # the three runs vary more
    e = P.combined_error(L, w, groups, separate=True)
    C = np.cov(L[:, 1:])
    terms = []
    for g in (list(range(5)), list(range(5, 8))):
        k, Cg = len(g), C[np.ix_(g, g)]
        v_ind = np.trace(Cg) / k - max((Cg.sum() - np.trace(Cg)) / (k * (k - 1)), 0.0)
        s2 = np.var(L[g, 0], ddof=1)
        terms.append((k, s2, v_ind, (np.asarray(w)[g] ** 2).sum()))
    var = sum(c * max(s2 - vi, 0.0) for _, s2, vi, c in terms) + e["ln_test_se"] ** 2
    assert e["ln_combined_se"] ** 2 == pytest.approx(var, rel=1e-12)
    assert e["dof"] == pytest.approx(var ** 2 / sum((c * s2) ** 2 / (k - 1) for k, s2, _, c in terms),
                                     rel=1e-12)
    assert e["v_ind"] == pytest.approx([t[2] for t in terms], rel=1e-12)
    assert e["ln_combined_se"] >= e["ln_test_se"]
    # equal-size groups without resampling: the pooled variance, fewer degrees of freedom
    pt = rng.normal(0, 1, (8, 1)) * np.r_[np.ones(4), 3 * np.ones(4)][:, None]
    w2, g2 = [1 / 4] * 4 + [-1 / 4] * 4, [range(4), range(4, 8)]
    pooled, sep = P.combined_error(pt, w2, g2), P.combined_error(pt, w2, g2, separate=True)
    assert sep["ln_combined_se"] == pytest.approx(pooled["ln_combined_se"], rel=1e-12)
    assert sep["dof"] < pooled["dof"] == pytest.approx(6)
    assert "not_computed" in P.combined_error(pt[:5], [1 / 4] * 4 + [-1.0], [range(4), [4]],
                                              separate=True)


def test_unequal_groups_with_unequal_run_variance_cover_as_welch_does():
    # A13: five runs against three that leave a family out, whose run variance
    # is three times the parent's; anomaly sigma_min carries no resampling
    # (B = 0). The pooled spread covered 0.91 (verification of 2026-10-01). The
    # Welch-Satterthwaite interval covers 0.941 for normal data in exactly this
    # configuration (its own small-sample error at about two degrees of freedom).
    rng = np.random.default_rng(31)
    hit = {False: 0, True: 0}
    trials = 10000
    for _ in range(trials):
        pt = np.r_[rng.normal(0, math.sqrt(0.5), 5), rng.normal(0, math.sqrt(1.5), 3)]
        for sep in hit:
            e = P.combined_error(pt[:, None], [1 / 5] * 5 + [-1 / 3] * 3, [range(5), range(5, 8)],
                                 separate=sep)
            hit[sep] += abs(e["estimate"]) <= P._t975(e["dof"]) * e["ln_combined_se"]
    assert 0.93 <= hit[True] / trials <= 0.96
    assert hit[False] / trials < 0.925
    # with test noise as well (a group-shared part and each run's own, B = 200)
    hit = 0
    trials = 2000
    for _ in range(trials):
        L, w, groups = _two_groups(rng, 5, 3, 0.0, 0.3, 1.0, 200)
        L[:, 0] += np.r_[rng.normal(0, math.sqrt(0.5), 5), rng.normal(0, math.sqrt(1.5), 3)]
        e = P.combined_error(L, w, groups, separate=True)
        hit += abs(e["estimate"]) <= P._t975(e["dof"]) * e["ln_combined_se"]
    assert 0.935 <= hit / trials <= 0.97


def test_without_resampling_the_error_is_the_spread_over_runs():
    # anomaly sigma_min: one value per run, no bootstrap
    pt = np.log([1.2, 1.5, 1.4, 2.0, 2.4, 2.1])
    e = P.combined_error(pt[:, None], [1 / 3] * 3 + [-1 / 3] * 3, [range(3), range(3, 6)])
    s2 = (np.var(pt[:3], ddof=1) + np.var(pt[3:], ddof=1)) / 2
    assert math.isnan(e["ln_test_se"]) and e["v_ind"] == 0.0
    assert e["ln_combined_se"] == pytest.approx(math.sqrt(s2 * 2 / 3)) and e["dof"] == pytest.approx(4)
    assert "not_computed" in P.combined_error(pt[:1, None], [1.0])
    r = P.paired_ratio({k: np.array([1.0 + 0.1 * k]) for k in range(4)},
                       {k: np.array([1.5 + 0.12 * k]) for k in range(4)})
    assert r["n_boot"] == 0 and r["ln_combined_se"] == pytest.approx(r["ln_run_se"])


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


def test_fixed_effects_reports_the_named_columns_and_floors_at_the_test_noise():
    # six strata (tasks) x five units (partitions): a stratum intercept, a unit
    # effect shared by every stratum, and one merge effect per stratum.
    rng = np.random.default_rng(6)
    merged = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0],
                       [1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1]])
    X, y, strata = np.zeros((30, 16)), np.zeros(30), []
    for i in range(30):
        s, u = divmod(i, 5)
        X[i, s] = 1.0
        if u:
            X[i, 5 + u] = 1.0
        X[i, 10 + s] = merged[s, u]
        y[i] = -3 - 0.2 * s + 0.5 * (u == 0) + 0.3 * merged[s, u]
        strata.append(s)
    shared = rng.normal(0, 0.05, (6, 400))              # test noise a stratum's units share
    own = rng.normal(0, 0.01, (30, 400))
    L = np.c_[y, y[:, None] + np.repeat(shared, 5, axis=0) + own]
    f = P.fixed_effects(L, X, strata, {f"e{s}": 10 + s for s in range(6)})
    assert set(f["coefficients"]) == {f"e{s}" for s in range(6)} and f["resid_dof"] == 14
    for c in f["coefficients"].values():
        assert c["ln_ratio"] == pytest.approx(0.3, abs=1e-12)
        # the shared noise cancels: the bootstrap SE is set by the units' own noise
        assert c["ln_test_se"] < 0.02
    # no residual at all: the error is the coefficient's own test error, which for
    # exchangeable units is [(X' W X)^-1]_jj with W = 1 / v_ind of each unit's stratum
    assert f["dispersion"] == pytest.approx(0.0, abs=1e-20)
    assert f["v_run"] == {s: 0.0 for s in range(6)}
    v_ind = []
    for s in range(6):
        C = np.cov(L[5 * s:5 * s + 5, 1:])
        v_ind += [np.trace(C) / 5 - max((C.sum() - np.trace(C)) / 20, 0.0)] * 5
    assert v_ind[0] == pytest.approx(0.01 ** 2, rel=0.2)
    A = np.linalg.inv(X.T @ (X / np.array(v_ind)[:, None]))
    for s in range(6):
        c = f["coefficients"][f"e{s}"]
        assert c["ln_combined_se"] == c["ln_test_se"]
        assert c["ln_combined_se"] == pytest.approx(math.sqrt(A[10 + s, 10 + s]), rel=0.1)
    # unit noise beyond the test noise, the same in every stratum, and units that
    # share test noise unevenly within a stratum (the merging ones more): the
    # per-stratum run variances recover the planted 0.03^2 on average (each has
    # about three degrees of freedom) and no coefficient's error falls below its
    # own test error
    extra = np.zeros((30, 401))
    extra[:, 0] = np.random.default_rng(8).normal(0, 0.03, 30)
    lump = rng.normal(0, 0.04, (6, 401))
    uneven = np.repeat(lump, 5, axis=0) * merged.reshape(30, 1)
    g = P.fixed_effects(L + extra + uneven, X, strata, {f"e{s}": 10 + s for s in range(6)})
    assert 0.5 * 0.03 ** 2 < np.mean(list(g["v_run"].values())) < 2 * 0.03 ** 2
    assert all(2 < k < 4 for k in g["stratum_dof"].values())
    for s in range(6):
        c = g["coefficients"][f"e{s}"]
        assert c["ln_test_se"] > 0.03                       # the lump does not cancel
        assert c["ln_combined_se"] > c["ln_test_se"]


# The committed A10 design (PRESPEC, redraw of 2026-10-01): each probe pair's
# merge column over the five partitions, in the order b vs c two-prong,
# bbqq/ccqq, visible content, two- vs four-prong, e vs mu, X->bc vs X->bq,
# X->bc vs X->cs.
_PATS = [(0, 2, 4), (0, 2), (3, 4), (0, 1), (0, 1, 3), (0, 1, 4), (0, 4)]
# The v1 17-class linear probes on those tasks (bvc_resonant, bvc_4prong,
# visible_content, retained_topology, ee_vs_mm; the two X->bc probes, new in v2,
# take bc_vs_rest's and bvc_qcd's): run variance of ln(1 - AUC) beyond the test
# noise, and each run's own test variance (verification of 2026-10-01). A unit
# is the mean of two runs.
_V1_RUN = np.array([2.50e-2, 2.62e-2, 4.18e-3, 8.0e-3, 5.89e-2, 1.75e-3, 1.37e-2])
_V1_IND = np.array([1.26e-3, 1.53e-3, 3.73e-3, 1.60e-3, 2.47e-3, 2.89e-4, 1.13e-3])
_V2_TEST = 0.2 / 0.7         # v2 scores 70 % of each task's jets, v1 20 %


def _joint_fits(v_run, v_ind, v_shared, trials, seed, b=200, effect=0.2):
    """Per draw of the task x partition units (a run effect, test noise shared
    within a task and each unit's own; the partition effects are zero, which
    the fit does not see: it is equivariant in them), yield
    (fixed_effects, the previous round's form, the per-pair rule, the one
    pooled run variance it replaced), each {task: (estimate - effect, se, dof)}.
    The previous form subtracted the leakage (signed weights g) and clipped the
    run variance stratum by stratum; the pooled form took one run variance for
    every task and the pooled residual degrees of freedom."""
    T, A = len(_PATS), 5
    merged = np.array([[a in p for a in range(A)] for p in _PATS], float)
    n, p = T * A, 2 * T + A - 1
    X, task = np.zeros((n, p)), np.repeat(np.arange(T), A)
    for i, (t, a) in enumerate((t, a) for t in range(T) for a in range(A)):
        X[i, t] = 1.0
        if a:
            X[i, T + a - 1] = 1.0
        X[i, T + A - 1 + t] = merged[t, a]
    rng = np.random.default_rng(seed)
    mu = X @ np.r_[rng.normal(-3, 1, T), np.zeros(A - 1), np.full(T, effect)]
    report = {t: T + A - 1 + t for t in range(T)}
    D = np.array([task == t for t in range(T)], float)
    for _ in range(trials):
        def noise(size):                   # shared within a task, and each unit's own
            return (rng.normal(0, 1, (T, size)) * np.sqrt(v_shared)[:, None])[task] \
                + rng.normal(0, 1, (n, size)) * np.sqrt(v_ind[task])[:, None]
        pt = mu + rng.normal(0, 1, n) * np.sqrt(v_run[task]) + noise(1)[:, 0]
        L = np.c_[pt, pt[:, None] + noise(b)]
        f = P.fixed_effects(L, X, task, report)
        new = {t: (f["coefficients"][t]["ln_ratio"] - effect, f["coefficients"][t]["ln_combined_se"],
                   f["coefficients"][t]["dof"]) for t in range(T)}
        C = np.cov(L[:, 1:])
        W = np.empty(n)
        for t in range(T):
            Ct = C[np.ix_(task == t, task == t)]
            W[task == t] = 1 / (np.trace(Ct) / A - max((Ct.sum() - np.trace(Ct)) / (A * (A - 1)), 0))
        rows = np.linalg.inv(X.T @ (W[:, None] * X)) @ (X.T * W)
        M = np.eye(n) - X @ rows
        r = M @ L[:, 0]
        As = [M.T @ ((W * d)[:, None] * M) for d in D]
        G = np.array([np.diag(a) @ D.T for a in As])
        Q, tau = D @ (W * r ** 2), np.array([(a * C).sum() for a in As])
        v = np.clip(np.linalg.solve(G, Q - tau), 0, None)
        S = np.diag(D.T @ v) + C
        k = np.array([np.trace(x) ** 2 / (x * x.T).sum() for x in (a @ S for a in As)])
        MWM = M.T @ (W[:, None] * M)
        rss = float((W * r ** 2).sum())
        vp = max(rss - (MWM * C).sum(), 0.0) / np.trace(MWM)
        prev, pooled, pair = {}, {}, {}
        for t in range(T):
            j = report[t]
            c, tv = D @ rows[j] ** 2, (rows[j] @ L[:, 1:]).var(ddof=1)
            g = np.linalg.solve(G.T, c)
            var = c @ v + tv
            prev[t] = (rows[j] @ L[:, 0] - effect, math.sqrt(var), var ** 2 / ((g * Q) ** 2 / k).sum())
            var = c.sum() * vp + tv
            pooled[t] = (prev[t][0], math.sqrt(var),
                         var ** 2 / ((c.sum() * rss / np.trace(MWM)) ** 2 / (n - p)))
            m = [i for i in range(t * A, t * A + A) if merged[t, i - t * A]]
            s = [i for i in range(t * A, t * A + A) if not merged[t, i - t * A]]
            e = P.combined_error(L[m + s], [1 / len(m)] * len(m) + [-1 / len(s)] * len(s),
                                 [range(len(m)), range(len(m), A)])
            pair[t] = (e["estimate"] - effect, e["ln_combined_se"], e["dof"])
        yield new, prev, pair, pooled


def _joint_coverage(v_run, v_ind, v_shared, trials, seed):
    """Per task, the fraction of 95 % intervals that hold the merge effect, for
    each form _joint_fits yields."""
    hit = np.zeros((4, len(_PATS)))
    for forms in _joint_fits(v_run, v_ind, v_shared, trials, seed):
        for i, form in enumerate(forms):
            hit[i] += [abs(b) <= P._t975(d) * se for b, se, d in form.values()]
    return hit / trials


def test_joint_fit_covers_each_task_with_the_v1_per_task_variances():
    # The run variance of ln(1 - AUC) differs about 35x between the v1 tasks; one
    # value pooled over them covered 0.77 to 0.998 task by task. The previous
    # round's per-task form subtracted the leakage between tasks and covered up
    # to 0.98 where the run variance dominates (verification of 2026-10-01). The
    # joint fit must cover 95 % within simulation error (3000 trials, SE 0.004)
    # where the run variance dominates, and elsewhere as the per-pair rule does
    # on the same draws: visible content's test noise is as large as its run
    # variance, and neither form goes below the test error, so both over-cover.
    new, prev, pair, pooled = _joint_coverage(_V1_RUN / 2, _V1_IND / 2, _V1_IND / 4,
                                              trials=3000, seed=1)
    run_dominated = _V1_RUN / 2 > 2 * _V1_IND
    assert run_dominated.sum() == 6
    assert np.all(np.abs(new[run_dominated] - 0.95) <= 0.012), new
    assert np.all(np.abs(new - pair) <= 0.015), (new, pair)
    assert np.all(new >= 0.938), new
    # the check bites on both sides
    assert np.sum(prev[run_dominated] > 0.962) >= 2, prev
    assert pooled.min() < 0.85 and pooled.max() > 0.99, pooled


def test_joint_fit_covers_each_task_with_equal_variances():
    # run, test and within-task shared variance all 0.01 in every task: the test
    # noise is half of each unit's, so, as for the per-pair rule, the floor at the
    # test error over-covers
    new, prev, pair, pooled = _joint_coverage(np.full(7, 0.01), np.full(7, 0.01),
                                              np.full(7, 0.01), trials=2000, seed=2)
    assert np.all(np.abs(new - pair) <= 0.015), (new, pair)
    assert new.min() >= 0.938 and new.max() <= 0.98, new
    assert pooled.min() >= 0.93, pooled                       # the pooled form was right here


def test_joint_fit_degrees_of_freedom_stay_above_one():
    # The v1 variances at the v2 test fraction: subtracting the leakage of a noisy
    # task's run variance from a quiet task's spread gave dof < 1 in 0.5 % of
    # coefficients, t near 10^4 at the lowest, and an OverflowError in math.exp
    # once in 10^4 fits (verification of 2026-10-01).
    lo_new, lo_prev = [], []
    for new, prev, _, _ in _joint_fits(_V1_RUN / 2, _V1_IND * _V2_TEST / 2,
                                       _V1_IND * _V2_TEST / 4, trials=1500, seed=3):
        lo_new += [d for _, _, d in new.values()]
        lo_prev += [d for _, _, d in prev.values()]
    assert min(lo_new) >= 1.0
    assert sum(d < 1 for d in lo_prev) >= 10


# ------------------------------------------------------------------- A14 (v2)
def test_two_runs_take_one_degree_of_freedom_and_never_fall_below_their_spread():
    # Two runs at ln 0 and 0.2 (s^2 = 0.02), replicates anticorrelated with variance
    # 0.01 each: the contrast's test variance is 0, v_ind = 0.01, so the clipped
    # form gives 0.5 (0.02 - 0.01) = 0.005, below the observed 0.5 s^2 = 0.01.
    k = 200
    a = math.sqrt(0.01 * (2 * k - 1) / (2 * k))          # sample variance exactly 0.01
    z = np.r_[np.full(k, a), np.full(k, -a)]
    L = np.c_[[0.0, 0.2], np.vstack([z, -z])]
    old = P.combined_error(L, [0.5, 0.5])
    assert old["v_ind"] == pytest.approx(0.01) and old["ln_combined_se"] ** 2 == pytest.approx(0.005)
    assert old["dof"] == pytest.approx(0.25)
    new = P.combined_error(L, [0.5, 0.5], small_sample_rule=True)
    assert new["ln_combined_se"] ** 2 == pytest.approx(0.01) and new["dof"] == 1.0
    # without resampling: var = s^2 / 2 at one degree of freedom, as before
    e = P.combined_error(np.array([[0.1], [0.3]]), [0.5, 0.5], small_sample_rule=True)
    assert e["ln_combined_se"] == pytest.approx(0.1) and e["dof"] == 1.0
    # the rule leaves three runs alone
    L3 = np.c_[[0.0, 0.2, 0.1], np.vstack([z, -z, z])]
    assert P.combined_error(L3, [1 / 3] * 3, small_sample_rule=True) == P.combined_error(L3, [1 / 3] * 3)


def test_a_two_run_group_of_a_welch_error_takes_the_floor_and_one_degree_of_freedom():
    # Group 1: two runs at ln 0 and 0.2 (s^2 = 0.02), replicates anticorrelated with
    # variance 0.01 each, so v_ind = 0.01 and the clipped term 0.5 (0.02 - 0.01) =
    # 0.005 is below the observed 0.5 s^2 = 0.01. Group 2: three runs at 1.0, 1.1, 1.2
    # with no test noise (s^2 = 0.01, term 0.01 / 3). The contrast's test variance is 0.
    k = 200
    a = math.sqrt(0.01 * (2 * k - 1) / (2 * k))          # sample variance exactly 0.01
    z = np.r_[np.full(k, a), np.full(k, -a)]
    zero = np.zeros(2 * k)
    L = np.c_[[0.0, 0.2, 1.0, 1.1, 1.2], np.vstack([z, -z, zero, zero, zero])]
    w, groups = [0.5, 0.5, -1 / 3, -1 / 3, -1 / 3], [[0, 1], [2, 3, 4]]
    for kw in ({"separate": True}, {"pools": [[0], [1]]}):
        old = P.combined_error(L, w, groups, **kw)
        assert old["ln_combined_se"] ** 2 == pytest.approx(0.005 + 0.01 / 3)
        assert "two_run_rule" not in old
        new = P.combined_error(L, w, groups, small_sample_rule=True, **kw)
        assert new["ln_combined_se"] ** 2 == pytest.approx(0.01 + 0.01 / 3)
        assert new["dof"] == 1.0 and new["two_run_rule"] is True     # Satterthwaite gives 1.68
    # groups of three and more are left alone
    L3 = np.c_[[0.0, 0.2, 0.1, 1.0, 1.1, 1.2], np.vstack([z, -z, z, zero, zero, zero])]
    g3 = [[0, 1, 2], [3, 4, 5]]
    w3 = [1 / 3] * 3 + [-1 / 3] * 3
    assert (P.combined_error(L3, w3, g3, separate=True, small_sample_rule=True)
            == P.combined_error(L3, w3, g3, separate=True))


def test_two_run_coverage_is_at_most_six_percent_excluded():
    # Null data, two runs: run SD 1, each run's own test SD 0.1 to 1 of it and a
    # tenth of that variance shared by both runs. The form without the rule
    # excluded 7-11 % at 0.1-0.7 (measured 2026-10-02); with it, at most 6 %.
    rng = np.random.default_rng(7)
    trials, b = 2000, 200
    for ratio in (0.1, 0.3, 0.5, 1.0):
        vek, ve = ratio ** 2, 0.1 * ratio ** 2
        excl = {False: 0, True: 0}
        for _ in range(trials):
            pt = rng.normal(0, 1, 2) + rng.normal(0, math.sqrt(ve)) + rng.normal(0, math.sqrt(vek), 2)
            L = _replicate_matrix(rng, pt, math.sqrt(ve), math.sqrt(vek), b)
            for rule in excl:
                r = P.paired_log({1: L[0], 2: L[1]}, small_sample_rule=rule)
                lo, hi = r["ci95"]
                excl[rule] += not lo <= 1.0 <= hi
        assert excl[True] / trials <= 0.06, (ratio, excl)
        if ratio == 0.3:
            assert excl[False] / trials > 0.075, excl            # the check bites


def test_one_run_pair_is_not_computed_under_the_small_sample_rule(tmp_path):
    v = np.r_[0.2, 0.2 + np.random.default_rng(5).normal(size=50) * 0.03]
    assert P.paired_log({1: v}, small_sample_rule=True)["not_computed"] == P.ONE_PAIR
    assert "not_computed" not in P.paired_log({1: v})          # v1: the test error alone
    # left with one pair by the stream check (pair 2 diverges at epoch 30, inside 0-45)
    d = {}
    for k, div in ((1, None), (2, 30)):
        _write_stream(tmp_path / f"f{k}", _rows(), 40)
        _write_stream(tmp_path / f"c{k}", _rows(div), 45)
        d[f"f{k}"], d[f"c{k}"] = tmp_path / f"f{k}", tmp_path / f"c{k}"
    fine = {f"f{k}": v * 0.5 for k in (1, 2)}
    coarse = {f"c{k}": v for k in (1, 2)}
    r = P.paired_ratio(fine, coarse, pairs={"c1": "f1", "c2": "f2"}, run_dirs=d,
                       checkpoint="bestval", small_sample_rule=True)
    assert r["n_runs"] == 1 and r["not_computed"] == P.ONE_PAIR
    assert [x["first_bad_epoch"] for x in r["excluded_pairs"]] == [30]


def test_best70_reads_the_selected_epoch_within_70_to_79(tmp_path):
    _write_stream(tmp_path / "a", _rows(), 40)
    _write_stream(tmp_path / "b", _rows(74), 41)
    for d, e in (("a", 72), ("b", 73)):
        (tmp_path / d / "best_window_epoch.json").write_text(json.dumps({"epoch": e}))
    assert P.checkpoint_epoch(tmp_path / "a", "best70") == 72
    assert P.stream_check([tmp_path / "a", tmp_path / "b"], "best70") is None       # up to 73
    assert P.stream_check([tmp_path / "a", tmp_path / "b"], ("best70", "bestval")) is None
    assert P.stream_check([tmp_path / "a", tmp_path / "b"], ("wavg", "best70"))["first_bad_epoch"] == 74
    (tmp_path / "a" / "best_window_epoch.json").unlink()
    with pytest.raises(SystemExit, match="selected epoch within 70-79 is unknown"):
        P.checkpoint_epoch(tmp_path / "a", "best70")


def test_pools_keep_each_sides_run_variance_from_the_replicate_runs():
    # P1 as A14 reads it, without resampling: two merging partitions and three
    # splitting ones, two runs each. Each side's run variance is its runs' spread
    # about their partition's mean, pooled over its partitions.
    y = np.array([[0.10, 0.14], [0.30, 0.26], [0.0, 0.05], [0.02, 0.01], [-0.03, -0.01]])
    L = y.reshape(-1, 1)
    w = [1 / 4] * 4 + [-1 / 6] * 6
    e = P.combined_error(L, w, [range(2 * i, 2 * i + 2) for i in range(5)],
                         pools=[[0, 1], [2, 3, 4]])
    within = ((y[:, 0] - y[:, 1]) ** 2 / 2)                   # one degree of freedom each
    s2m, s2s = within[:2].sum() / 2, within[2:].sum() / 3
    cm, cs = 4 / 16, 6 / 36
    var = cm * s2m + cs * s2s
    assert e["estimate"] == pytest.approx(y[:2].mean() - y[2:].mean())
    assert e["ln_combined_se"] ** 2 == pytest.approx(var)
    assert e["dof"] == pytest.approx(var ** 2 / ((cm * s2m) ** 2 / 2 + (cs * s2s) ** 2 / 3))
    assert e["pool_dof"] == [2, 3] and e["ln_spread_sd"] == pytest.approx([math.sqrt(s2m), math.sqrt(s2s)])
    # one run per partition leaves no replicate: not computed
    assert "not_computed" in P.combined_error(y[:, :1], [1 / 2] * 2 + [-1 / 3] * 3,
                                              [[i] for i in range(5)], pools=[[0, 1], [2, 3, 4]])


def test_p1_welch_covers_with_run_or_test_noise_dominant():
    # null merge effect, five partitions x two runs, test noise shared within the
    # task: the v1 run-dominated case (b vs c two-prong) and an equal one
    rng = np.random.default_rng(9)
    for vr, vi in ((2.5e-2, 1.26e-3), (1.0, 1.0)):
        for nm in (2, 3):
            hit, trials = 0, 1500
            for _ in range(trials):
                sh = rng.normal(0, math.sqrt(vi / 2), 201)
                pt = rng.normal(0, math.sqrt(vr), 10) + rng.normal(0, math.sqrt(vi), 10) + sh[0]
                L = np.c_[pt, pt[:, None] + sh[None, 1:] + rng.normal(0, math.sqrt(vi), (10, 200))]
                w = [1 / (2 * nm)] * (2 * nm) + [-1 / (2 * (5 - nm))] * (2 * (5 - nm))
                e = P.combined_error(L, w, [range(2 * i, 2 * i + 2) for i in range(5)],
                                     pools=[list(range(nm)), list(range(nm, 5))])
                hit += abs(e["estimate"]) <= P._t975(e["dof"]) * e["ln_combined_se"]
            assert 0.935 <= hit / trials <= 0.99, (vr, vi, nm, hit / trials)


def _fieller_closed_form(N, D, t):
    """Roots of (Nbar - f Dbar)^2 = t^2 (s_NN - 2 f s_ND + f^2 s_DD) / n."""
    n = len(N)
    S = np.cov(np.vstack([N, D]))
    a = D.mean() ** 2 - t ** 2 * S[1, 1] / n
    b = -2 * (N.mean() * D.mean() - t ** 2 * S[0, 1] / n)
    c = N.mean() ** 2 - t ** 2 * S[0, 0] / n
    return sorted(np.roots([a, b, c]).real)


def test_fieller_is_the_closed_form_without_resampling():
    from scipy.stats import t as student
    N = np.array([0.30, 0.33, 0.30, 0.33, 0.30])
    D = np.array([0.36, 0.38, 0.40, 0.42, 0.44])
    f = P.fieller(N[:, None], D[:, None])
    assert f["fraction"] == pytest.approx(N.mean() / D.mean())
    assert f["ci95"] == pytest.approx(_fieller_closed_form(N, D, student.ppf(0.975, 4)), abs=1e-9)
    assert f["denominator"]["dof"] == pytest.approx(4) and f["n_runs"] == 5
    # a denominator whose interval holds 0: no interval
    u = P.fieller(N[:, None], (D - D.mean() + 0.01)[:, None])
    assert u["ci95"] is None and "unbounded" in u["note"]
    # two runs under the rule: t at one degree of freedom
    two = P.fieller(N[:2, None], D[:2, None], small_sample_rule=True)
    assert two["ci95"] == pytest.approx(_fieller_closed_form(N[:2], D[:2], student.ppf(0.975, 1)), abs=1e-9)
    assert P.fieller(N[:1, None], D[:1, None], small_sample_rule=True)["not_computed"] == P.ONE_PAIR


def test_fieller_with_resampling_inverts_the_combined_error():
    rng = np.random.default_rng(4)
    N = _replicate_matrix(rng, [0.30, 0.35, 0.28, 0.33, 0.31], 0.01, 0.02, 300)
    D = _replicate_matrix(rng, [0.40, 0.42, 0.39, 0.45, 0.41], 0.01, 0.02, 300)
    f = P.fieller(N, D)
    for x in f["ci95"]:                       # on each edge the contrast is at its 95 % bound
        e = P.combined_error(N - x * D, np.full(5, 0.2))
        assert abs(e["estimate"]) == pytest.approx(P._t975(e["dof"]) * e["ln_combined_se"], rel=1e-6)
    assert f["ci95"][0] < f["fraction"] < f["ci95"][1]


def test_the_a14_labels():
    ln11 = math.log(1.1)
    # an interval that excludes 0 and lies within +-ln 1.1 meets two rules A14 does
    # not order: labelled for both, in both labels, and counted as dependent
    assert P.checkpoint_label(0.01, 0.05) == P.DEPENDS_UNDER_10
    assert P.checkpoint_label(0.01, ln11 + 1e-9) == P.DEPENDS
    assert P.checkpoint_label(-0.2, -0.01) == P.DEPENDS
    assert set(P.DEPENDENT) == {P.DEPENDS, P.DEPENDS_UNDER_10}
    assert P.checkpoint_label(-0.05, 0.05) == "robust"
    assert P.checkpoint_label(-0.05, ln11 + 1e-9) == "inconclusive"
    assert P.p1_label(-0.3, 0.09) == "merging costs nothing"
    assert P.p1_label(0.01, 0.05) == "merging costs, under 10%"
    assert P.p1_label(0.01, 0.30) == "merging costs"
    assert P.p1_label(-0.01, 0.30) == "inconclusive"
    assert P.beats_label(0.001) == "beats" and P.beats_label(-0.001) == "inconclusive"
    assert P.equal_label(-0.09, 0.09) == "equal" and P.equal_label(-0.09, 0.1) == "inconclusive"
    assert [P.threshold_label(*x) for x in ((0.1, 0.2), (-0.2, -0.1), (-0.1, 0.1), (0.1, 0.3, 0.2))] == \
        ["holds", "fails", "inconclusive", "inconclusive"]
    assert [P.equivalence_label(*x) for x in ((-0.1, 0.1, 0.2), (0.3, 0.4, 0.2), (-0.1, 0.3, 0.2),
                                              (-0.1, 0.1, 0.0))] == \
        ["holds", "fails", "inconclusive", "not evaluable"]
    lo, hi = P.bounds(1.0, 0.1, 4)
    assert (lo, hi) == pytest.approx((1 - 0.27764451, 1 + 0.27764451), abs=1e-7)   # t(0.975, 4) = 2.776
    assert P.bounds(0.0, 1.0, math.inf, 0.90)[1] == pytest.approx(1.6448536, abs=1e-6)


def test_two_runs_under_the_a14_rule_exclude_0_far_less_often_than_5_percent():
    # Why checkpoint_dependence counts the two-run results apart: with no true
    # difference, two runs (run SD 1, test SD `ratio` of it, a tenth of the test
    # variance shared by both runs) under the A14 floor and Student t at 1 degree
    # of freedom exclude 0 well under 5 % of the time, so 5 % of them would
    # overstate the dependent results the null gives.
    rng = np.random.default_rng(7)
    b, trials = 50, 2000
    for ratio, below in ((0.1, 0.03), (1.0, 0.005)):
        vk, vs = 0.9 * ratio ** 2, 0.1 * ratio ** 2
        hits = 0
        for _ in range(trials):
            sh = rng.normal(0, math.sqrt(vs), b + 1)
            pt = rng.normal(0, 1, 2) + sh[0] + rng.normal(0, math.sqrt(vk), 2)
            L = np.c_[pt, pt[:, None] + sh[None, 1:] + rng.normal(0, math.sqrt(vk), (2, b))]
            r = P.paired_log({1: L[0], 2: L[1]}, small_sample_rule=True)
            assert r["two_run_rule"] and r["dof"] == 1.0
            lo, hi = P.bounds(r["ln_ratio"], r["ln_combined_se"], r["dof"])
            hits += lo > 0 or hi < 0
        assert hits / trials < below, ratio


def test_fixed_effects_weights_each_unit_by_its_run_plus_test_variance():
    # the six-stratum design above; run variances given apart from the fit
    rng = np.random.default_rng(12)
    merged = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0],
                       [1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1]])
    X, strata = np.zeros((30, 16)), []
    for i in range(30):
        s, u = divmod(i, 5)
        X[i, s] = 1.0
        if u:
            X[i, 5 + u] = 1.0
        X[i, 10 + s] = merged[s, u]
        strata.append(s)
    pt = X @ rng.normal(0, 0.3, 16) + rng.normal(0, 0.05, 30)
    L = np.c_[pt, pt[:, None] + rng.normal(0, 0.02, (30, 300))]
    rv = np.repeat([1e-3, 4e-3, 1e-4, 2e-3, 5e-4, 3e-3], 5)
    f = P.fixed_effects(L, X, strata, {"e0": 10, "e3": 13}, run_var=rv)
    v_ind = np.empty(30)
    for s in range(6):
        C = np.cov(L[5 * s:5 * s + 5, 1:])
        v_ind[5 * s:5 * s + 5] = np.trace(C) / 5 - max((C.sum() - np.trace(C)) / 20, 0.0)
    W = 1 / (v_ind + rv)
    beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * pt))
    assert f["coefficients"]["e0"]["ln_ratio"] == pytest.approx(beta[10], abs=1e-12)
    assert f["coefficients"]["e3"]["ln_ratio"] == pytest.approx(beta[13], abs=1e-12)
    # without run_var: 1 / v_ind, as before
    g = P.fixed_effects(L, X, strata, {"e0": 10})
    b0 = np.linalg.solve(X.T @ (X / v_ind[:, None]), X.T @ (pt / v_ind))
    assert g["coefficients"]["e0"]["ln_ratio"] == pytest.approx(b0[10], abs=1e-12)
