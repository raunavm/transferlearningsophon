"""experiments/STATS/analytic_check.py (PI 2026-11-23, LEDGER SC-2 #1): the transfer-head rows' analytic term for eps_S at
eps_B = 1%, against the paired bootstrap of two sides scored on the same jets."""
import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analytic_check", REPO / "experiments/STATS/analytic_check.py")
AC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AC)


def test_eff_scorer_is_the_signal_fraction_above_the_background_cut():
    r = np.random.default_rng(0)
    y = np.r_[np.ones(20000, int), np.zeros(80000, int)]
    s = np.r_[r.normal(2, 1, 20000), r.normal(0, 1, 80000)]
    sc = AC.EffScorer(y == 1, s)
    cut = np.quantile(s[y == 0], 1 - AC.EPS_B)
    assert abs(sc() - (s[y == 1] > cut).mean()) < 2e-3
    w = np.ones(y.size)
    assert sc(w) == sc()
    w[y == 0] = 2.0                     # doubling every background jet's weight moves nothing
    assert abs(sc(w) - sc()) < 1e-12


def test_analytic_sd_is_the_binomial_terms_with_the_roc_slope():
    r = np.random.default_rng(1)
    y = np.r_[np.ones(5000, int), np.zeros(50000, int)]
    s = np.r_[r.normal(2, 1, 5000), r.normal(0, 1, 50000)]
    sc = AC.EffScorer(y == 1, s)
    e, sd, slope = AC.analytic(sc, 5000, 50000)
    want = np.sqrt(e * (1 - e) / 5000 + slope ** 2 * AC.EPS_B * (1 - AC.EPS_B) / 50000)
    assert slope > 0 and abs(sd - want) < 1e-15


def test_two_correlated_sides_pair_below_the_unpaired_term(tmp_path):
    r = np.random.default_rng(2)
    n_s, n_b = 4000, 40000
    y = np.r_[np.ones(n_s, int), np.zeros(n_b, int)]
    base = np.r_[r.normal(2, 1, n_s), r.normal(0, 1, n_b)]
    paths = []
    for k in range(4):
        p = tmp_path / f"m{k}.npz"
        np.savez(p, rows=np.arange(y.size), y=y, s=(base + 0.2 * r.normal(size=y.size) + (0.05 if k < 2 else 0)).astype(np.float32))
        paths.append(str(p))
    doc = AC.check({"a-b": (paths[:2], paths[2:])}, b=50, seed=3)
    d = doc["diffs"]["a-b"]
    assert d["analytic_sd_unpaired"] > d["boot_sd_paired"] > 0
    assert d["sigma_analytic"] >= d["sigma_paired"]
    for v in doc["networks"].values():         # one network: analytic and bootstrap SD of the same order
        assert 0.5 < v["analytic_sd"] / v["boot_sd"] < 2


def test_main_parses_sides_of_paths(tmp_path):
    r = np.random.default_rng(4)
    y = np.r_[np.ones(3000, int), np.zeros(30000, int)]
    paths = []
    for k in range(4):
        p = tmp_path / "a" / f"m{k}.npz"
        p.parent.mkdir(exist_ok=True)
        np.savez(p, rows=np.arange(y.size), y=y, s=np.r_[r.normal(2, 1, 3000), r.normal(0, 1, 30000)].astype(np.float32))
        paths.append(str(p))
    out = tmp_path / "out" / "check.json"
    assert AC.main(["--diff", f"x-y={paths[0]},{paths[1]}:{paths[2]},{paths[3]}", "--b", "20", "--out", str(out)]) == 0
    assert list(__import__("json").loads(out.read_text())["diffs"]) == ["x-y"]
