"""experiments/AOJ/injection_test.py: the injection test's bins must be the bins the
checks and the fit build, from the jets the fit saw, or nothing is written."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("injection_test", ROOT / "experiments/AOJ/injection_test.py")
IT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IT)
P = IT.P


def _merged(tmp_path, n=400_000, seed=3):
    """A merged run of two scores: QCD-like jets over the rho window, a top-like peak."""
    rng = np.random.default_rng(seed)
    pt = np.minimum(500.0 * (1 - rng.random(n)) ** (-1 / 4.5), 2499.0)
    rho = rng.uniform(-6.0, -1.8, n)
    mass = pt * np.exp(rho / 2)
    sig = rng.random(n) < 0.01
    mass[sig] = rng.normal(175.0, 12.0, sig.sum())
    d = tmp_path / "merged"
    d.mkdir()
    pn = np.where(sig, 1 - rng.random(n) ** 4, rng.random(n))
    np.savez(d / "jets.npz", jet_sdmass=mass.astype(np.float32), aoj_jet_pt=pt.astype(np.float32),
             aoj_pn_TvsQCD=pn.astype(np.float32))
    np.savez(d / "scores_m.npz", three_prong_logodds=P.logit(np.where(sig, 1 - rng.random(n) ** 3,
                                                                      rng.random(n))).astype(np.float16))
    return d


def _committed(tmp_path, merged, change=False):
    j = np.load(merged / "jets.npz")
    mass, pt = j["jet_sdmass"].astype(float), j["aoj_jet_pt"].astype(float)
    rho = P.rho_of(mass, pt)
    ok = (rho > P.RHO_RANGE[0]) & (rho < P.RHO_RANGE[1]) & (pt > P.PT_RANGE[0]) & (pt < P.PT_RANGE[1])
    out = {}
    for name, z in (("reference", P.logit(j["aoj_pn_TvsQCD"])),
                    ("m", np.load(merged / "scores_m.npz")["three_prong_logodds"].astype(float))):
        b = IT.top_bins(z[ok], mass[ok], pt[ok], IT.EFF)
        if change and name == "m":
            b = dict(b, n_pass=b["n_pass"] + 1)
        out.update({f"{name}|main|{k}": b[k] for k in IT.KEYS})
    np.savez(tmp_path / "committed.npz", **out)
    return tmp_path / "committed.npz"


def test_the_bins_are_the_pseudo_window_and_extra_working_points_of_every_score(tmp_path):
    merged = _merged(tmp_path)
    IT.export(merged, _committed(tmp_path, merged), tmp_path / "out.npz", workers=1)
    z = np.load(tmp_path / "out.npz")
    parts = {k.split("|")[1] for k in z.files}
    assert {k.split("|")[0] for k in z.files} == {"reference", "m"}
    assert parts == {"pseudo", *(f"main_eff{e:g}" for e in IT.EXTRA_EFF)}
    for name in ("reference", "m"):
        ps = z[f"{name}|pseudo|m_edges"]
        assert ps[0] == IT.PSEUDO["fit_range"][0] and ps[-1] == IT.PSEUDO["fit_range"][1]
        lo, hi = z[f"{name}|main_eff0.005|n_pass"].sum(), z[f"{name}|main_eff0.02|n_pass"].sum()
        assert 1.5 * lo < hi < 6 * lo, "four times the data efficiency: more passing jets, at most ~4x"


def test_nothing_is_written_when_the_one_per_cent_bins_are_not_the_committed_ones(tmp_path):
    merged = _merged(tmp_path)
    with pytest.raises(SystemExit, match="these are not the jets the fit saw"):
        IT.export(merged, _committed(tmp_path, merged, change=True), tmp_path / "out.npz", workers=1)
    assert not (tmp_path / "out.npz").exists()


# ---- the toys ----
DATA = ROOT / "experiments/FIGS/data/aoj_full_v1"


@pytest.fixture(scope="module")
def committed():
    IT._init_toys(str(DATA / "fit_v3/bins.npz"), None, str(DATA / "fit_v4/results.json"), None)
    return IT._T


def test_the_top_generator_is_the_main_fit_without_its_signal_and_the_template_sums_to_one(committed):
    b = IT.region_bins("top", "l162-s4", committed["committed"], None)
    top = IT._top_fit("l162-s4")
    tr = IT.truth("top", "l162-s4", b, top, IT._start("top", "l162-s4"))
    assert tr["order"] == tuple(top["tf_order"]) and (tr["mean"], tr["width"]) == (top["mean"], top["width"])
    assert tr["template"].sum() == pytest.approx(1.0) and (tr["template"] >= 0).all()
    model = P._Model(b, tr["order"], P._tf_norm(b, P.PEAKS["top"]["window"]), top["mean"], top["width"])
    x, _ = model.fit()
    _, s, q, mu = model.expect(x)
    assert np.allclose(tr["background"] + s, mu) and np.allclose(tr["fail"], q)
    assert s.sum() == pytest.approx(top["signal_yield"], rel=1e-6)


def test_toys_inject_what_they_say_and_the_leak_puts_the_failing_signal_in_the_fail_region():
    b = dict(m_edges=np.arange(100.0, 131.0, 5.0), i=np.arange(6), j=np.zeros(6, int),
             n_pass=np.full(6, 10.0), n_fail=np.full(6, 1000.0), rho=np.zeros(6), pt=np.full(6, 600.0))
    tr = dict(background=np.full(6, 1e5), fail=np.full(6, 1e7), template=np.full(6, 1 / 6))
    rng = np.random.default_rng(1)
    boot = [IT.toy_bins(b, tr, 6e4, "bootstrap", rng) for _ in range(50)]
    assert np.mean([t["n_pass"].sum() for t in boot]) == pytest.approx(6.6e5, rel=2e-3)
    leak = [IT.toy_bins(b, tr, 6e4, "leak", rng, eps=0.25) for _ in range(50)]
    assert np.mean([t["n_fail"].sum() for t in leak]) - 6e7 == pytest.approx(6e4 * 3, rel=0.2)
    data = IT.toy_bins(b, tr, 600.0, "data", rng)
    assert (data["n_fail"] == b["n_fail"]).all() and data["n_pass"].sum() - 60 == pytest.approx(600, abs=100)


def test_the_tasks_are_split_over_shards_once_each_with_the_real_data_once_per_score():
    every = IT.tasks(["top", "pseudo"], ["bootstrap", "data"], ["a", "b"], [1000.0, 2000.0], 3, ["fixed"], 7, 0, 1)
    assert len(every) == 2 * 2 * 2 * (2 * 3) + 2 * 2   # regions x modes x scores x sizes x toys, + one real-data fit per region and score
    parts = [IT.tasks(["top", "pseudo"], ["bootstrap", "data"], ["a", "b"], [1000.0, 2000.0], 3, ["fixed"], 7, k, 3)
             for k in range(3)]
    keys = [IT.task_key(t) for p in parts for t in p]
    assert sorted(keys) == sorted(IT.task_key(t) for t in every) and len(set(keys)) == len(keys)
    assert sum(t["toy"] == -1 and t["size"] == 0 for t in every) == 4


def test_a_restarted_run_skips_what_it_wrote_and_toys_are_reproducible(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(IT, "_run_task", lambda t: calls.append(IT.task_key(t)) or dict(t, key=IT.task_key(t)))
    monkeypatch.setattr(IT, "_init_toys", lambda *a: None)
    argv = ["toys", "--names", "l162-s4", "--sizes", "1000", "--toys", "3", "--variants", "fixed",
            "--out", str(tmp_path / "t.jsonl")]
    IT.main(argv)
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == 3 and len(calls) == 3
    (tmp_path / "t.jsonl").write_text("\n".join(lines[:2]) + "\n")
    calls.clear()
    IT.main(argv)
    assert len(calls) == 1 and len((tmp_path / "t.jsonl").read_text().splitlines()) == 3


def test_the_same_task_draws_the_same_toy(committed):
    t = dict(region="top", mode="bootstrap", name="l162-s4", size=2000.0, toy=4, seed=11, variants=["fixed"])
    a, b = IT._run_task(t), IT._run_task(t)
    assert a["fits"]["fixed"]["y"] == b["fits"]["fixed"]["y"]
    c = IT._run_task(dict(t, toy=5))
    assert c["fits"]["fixed"]["y"] != a["fits"]["fixed"]["y"]
