"""experiments/AOJ/realdata_checks.py: the checks behind the real-data section.
Synthetic jets only; the fit machinery itself is tested in test_aoj_peak_fit.py."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aoj_realdata_checks", ROOT / "experiments/AOJ/realdata_checks.py")
RC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RC)
P = RC.P


def jets(n=400_000, seed=0):
    """Falling pT and m_SD spectra covering the rho window."""
    rng = np.random.default_rng(seed)
    pt = 500.0 * np.exp(rng.exponential(0.45, n))
    pt = pt[pt < 2500][: n // 2]
    rho = rng.uniform(-5.6, -1.9, len(pt))
    mass = pt * np.exp(rho / 2)
    keep = (mass > 20) & (mass < 500)
    return mass[keep], pt[keep], rng


def test_jet_level_fit_acceptance_is_the_binned_one():
    mass, pt, _ = jets()
    for fr in (P.PEAKS["top"]["fit_range"], RC.PSEUDO["fit_range"]):
        b = P._bins(mass, pt, np.ones(len(mass), bool), fr)
        assert RC.fit_acceptance(mass, pt, fr).sum() == b["n_pass"].sum() + b["n_fail"].sum()


def test_a_cut_independent_of_mass_leaves_the_shape_unchanged_and_one_that_prefers_the_window_does_not():
    mass, pt, rng = jets(seed=1)
    acc = RC.fit_acceptance(mass, pt, P.PEAKS["top"]["fit_range"])
    flat = rng.random(len(mass)) < 0.05
    sc = RC.shape_change(flat, mass, acc, (140.0, 220.0))
    assert abs(sc["delta"]) < 4 * sc["err"] and 0 < sc["err"] < 0.1
    win = P.in_windows(mass, [(140.0, 220.0)])
    sculpted = rng.random(len(mass)) < np.where(win, 0.06, 0.05)
    sc = RC.shape_change(sculpted, mass, acc, (140.0, 220.0))
    assert sc["delta"] == pytest.approx(0.2, abs=4 * sc["err"]) and sc["delta"] > 4 * sc["err"]


def test_the_map_closure_detects_a_score_sculpted_inside_the_masked_window():
    """The top window is masked when the map is built, so structure of the score
    narrower than it is interpolated over, not removed: exactly what the closure on
    simulated QCD must see. A score with no mass structure must close."""
    mass, pt, rng = jets(seed=2)
    acc = RC.fit_acceptance(mass, pt, P.PEAKS["top"]["fit_range"])
    z = rng.normal(size=len(mass)) + 0.3 * P.rho_of(mass, pt)
    passed = P.passes(z, mass, pt, P.build_map(z, mass, pt, 0.01))
    sc = RC.shape_change(passed, mass, acc, (140.0, 220.0))
    assert abs(sc["delta"]) < 4 * sc["err"]
    z2 = z + 0.8 * P.in_windows(mass, [(150.0, 200.0)])
    passed2 = P.passes(z2, mass, pt, P.build_map(z2, mass, pt, 0.01))
    sc2 = RC.shape_change(passed2, mass, acc, (140.0, 220.0))
    assert sc2["delta"] > 1.0 and sc2["delta"] > 5 * sc2["err"]


def test_overall_calibration_passes_the_target_of_all_jets():
    mass, pt, rng = jets(seed=3)
    z = rng.normal(size=len(mass)) + 0.6 * P.in_windows(mass, [(150.0, 200.0)])   # sculpts the window
    side = P.passes(z, mass, pt, P.build_map(z, mass, pt, 0.01))
    assert side.mean() > 0.012, "the fixture must pass more than 1 % overall at 1 % of the sidebands"
    eff, got = RC._calibrate_overall(z, mass, pt, 0.01)
    assert eff < 0.01 and got == pytest.approx(0.01, abs=2e-5)


def test_asimov_injection_adds_the_stated_yield_in_the_pt_categories_present():
    mass, pt, rng = jets(seed=4)
    b = P._bins(mass, pt, rng.random(len(mass)) < 0.01, RC.PSEUDO["fit_range"])
    per_pt = {f"{P.PT_EDGES[j]:g}-{P.PT_EDGES[j + 1]:g}": 100.0 * (j + 1) for j in range(8)}
    per_pt["500-550"] = -50.0
    inj, cats = RC.inject_asimov(b, 280.0, 11.0, per_pt)
    assert set(cats) == set(np.unique(b["j"]).tolist()) and 0 not in cats, "500-550 GeV has no bin here"
    added = inj["n_pass"] - b["n_pass"]
    for j, y in cats.items():
        assert added[b["j"] == j].sum() == pytest.approx(y, rel=1e-12)
        assert y <= 100.0 * (j + 1) * (1 + 1e-9), "never more than the category's norm"
    top_cat = max(cats)
    assert cats[top_cat] == pytest.approx(100.0 * (top_cat + 1), rel=1e-3), \
        "at high pT every mass bin of the pseudo range is inside the rho window"
    assert np.array_equal(inj["n_fail"], b["n_fail"])


def test_label_sets_and_their_summaries():
    assert [RC.label_set(n) for n in ("l188-s1", "l162-s1b", "r42q1-s3", "r16q1-s2", "l162mass-s5",
                                      "r16q1mass-s4", "sophon-public")] == \
        ["188", "162", "43", "17", "162+mass", "17+mass", "published"]
    rows = {"l188-s1": dict(v=1.0), "l188-s2": dict(v=3.0), "sophon-public": dict(v=99.0)}
    got = RC.by_label_set(rows, "v")
    assert got == {"188": dict(n=2, mean=2.0, sd=pytest.approx(np.sqrt(2)), min=1.0, max=3.0)}


def _sim_dir(tmp_path, tamper=False):
    SS = RC._load("sim_scores", RC.HERE / "sim_scores.py")
    rng = np.random.default_rng(5)
    n = 1000
    j = dict(jet_pt=rng.uniform(500, 2500, n).astype(np.float32), jet_eta=rng.uniform(-2, 2, n).astype(np.float32),
             jet_sdmass=rng.uniform(20, 500, n).astype(np.float32), label=rng.integers(0, 188, n).astype(np.int16))
    np.savez(tmp_path / "jets.npz", **j)
    for name in ("l188-s1", "r16q1-s1"):
        np.savez(tmp_path / f"scores_{name}.npz", three_prong_logodds=rng.normal(size=n).astype(np.float32))
        digest = SS.jets_digest(j) if not (tamper and name == "r16q1-s1") else "0" * 64
        (tmp_path / f"scores_{name}.json").write_text(json.dumps(dict(jets_sha256=digest)))
    return tmp_path


def test_load_sim_keeps_the_acceptance_and_labels_and_refuses_a_model_on_other_jets(tmp_path):
    s = RC.load_sim(_sim_dir(tmp_path))
    rho = P.rho_of(s["mass"], s["pt"])
    assert ((rho > -5.5) & (rho < -2.0)).all() and s["n_acc"] == len(s["mass"])
    assert (s["qcd"] == (s["label"] >= 161)).all()
    assert (s["three"] == (s["three_b"] | s["three_nob"])).all() and not (s["three_b"] & s["three_nob"]).any()
    assert set(s["scores"]) == {"l188-s1", "r16q1-s1"}
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="scored on other jets"):
        RC.load_sim(_sim_dir(other, tamper=True))


def test_the_head_defects_name_existing_runs_only():
    runs = {f"{a}-s{s}" for a in RC.LABEL_SET for s in "12345"} | {"l162-s1b"}
    assert set(RC.HEAD_DEFECTS) <= runs and len(RC.HEAD_DEFECTS) == 4


def _shard(d, scores, rows):
    d.mkdir(parents=True)
    np.savez(d / "jets.npz", event=np.arange(5), jet_sdmass=np.ones(5, np.float32))
    for name, v in scores.items():
        np.savez(d / f"scores_{name}.npz", three_prong_logodds=np.asarray(v, np.float16))
    (d / "closure.json").write_text(json.dumps(dict(rows=rows)))
    return d


def test_the_rescore_is_checked_bit_for_bit_against_the_first_run(tmp_path):
    rows = [dict(feature="part_d0", median=1.0)]
    first = _shard(tmp_path / "a" / "shard0", {"m1": [1, 2, 3, 4, 5], "m2": [0] * 5}, rows)
    same = _shard(tmp_path / "b" / "shard0", {"m1": [1, 2, 3, 4, 5], "m2": [0] * 5},
                  [dict(rows[0], quantiles_aoj=[0.0], n_aoj=9)])
    rep = RC.step_reproduce([same], [first])
    assert rep["all_identical"] and rep["shards"]["shard0"]["same_closure"]
    other = _shard(tmp_path / "c" / "shard0", {"m1": [1, 2, 3, 4, 6], "m2": [0] * 5}, rows)
    rep = RC.step_reproduce([other], [first])
    assert not rep["all_identical"] and rep["shards"]["shard0"]["three_prong_identical"] == {"m1": False, "m2": True}


def test_domain_reads_each_models_data_cut_off_the_data_and_applies_it_to_simulation():
    """A model whose simulated score is its data score shifted up passes more simulated
    QCD at its data cut than 1 %; the simulated cut brings it back to 1 % of the QCD
    sidebands. Efficiencies are counted on the right subsets."""
    mass, pt, rng = jets(n=500_000, seed=6)
    n = len(mass)
    three = sorted(RC.D.native_classes("3P_HAD_3PARTON"))
    label = np.where(rng.random(n) < 0.7, 170, rng.choice(three, n))
    qcd = label >= 161
    z_data = rng.normal(size=n)
    z_sim = np.where(qcd, rng.normal(size=n) + 0.5, rng.normal(size=n) + 2.0)
    data = dict(mass=mass, pt=pt, scores={"m": {"three_prong": z_data.astype(np.float16)}})
    sim = dict(mass=mass, pt=pt, label=label, qcd=qcd, three=np.isin(label, three),
               top_like=np.zeros(n, bool), three_b=np.isin(label, three), three_nob=np.zeros(n, bool),
               scores={"m": {"three_prong": z_sim.astype(np.float32)}})
    fit = {"models": {"m": {"top": dict(signal_yield=1000.0, signal_yield_err=100.0)}}}
    RC._W.update(data=data, sim=sim, fit=fit,
                 class_names={int(r["jet_label"]): r["class_name"] for r in RC.D._anomaly.read_map()})
    name, out = RC._domain_one("m")
    r = out["three_prong"]
    assert r["qcd_at_data_cut"]["eff"] > 0.02, "the shifted QCD must pass more than 1 % at the data cut"
    assert r["qcd_at_sim_cut"]["eff"] == pytest.approx(0.01, abs=0.002)
    acc_top = RC.fit_acceptance(mass, pt, P.PEAKS["top"]["fit_range"]) & P.in_windows(mass, [(140.0, 220.0)])
    assert r["signal_at_data_cut"]["n"] == int((sim["three"] & acc_top).sum())
    assert r["signal_at_data_cut"]["eff"] > r["signal_at_sim_cut"]["eff"] > r["qcd_at_sim_cut"]["eff"]
    assert sum(v["n"] for v in r["per_class_at_data_cut"].values()) == r["signal_at_data_cut"]["n"]
    assert "prong_only" not in out, "a score not persisted for the data is skipped, not faked"


def test_the_selection_chain_counts_each_stage_and_refuses_a_staging_record_that_disagrees(tmp_path):
    mass, pt, rng = jets(seed=7)
    extra = 1000                                     # staged jets outside the rho window
    m_all = np.r_[mass, np.full(extra, 450.0)]
    pt_all = np.r_[pt, np.full(extra, 600.0)]
    merged = tmp_path / "merged"
    merged.mkdir()
    np.savez(merged / "jets.npz", jet_sdmass=m_all.astype(np.float32), aoj_jet_pt=pt_all.astype(np.float32),
             aoj_pn_TvsQCD=rng.random(len(m_all)).astype(np.float16))
    np.savez(merged / "scores_m.npz", three_prong_logodds=rng.normal(size=len(m_all)).astype(np.float16))
    data = RC.load_data(merged)
    shard = tmp_path / "shard0" / "staging"
    shard.mkdir(parents=True)
    (shard / "RunG_batch0.stats.json").write_text(json.dumps(dict(n_jets_read=5 * len(m_all),
                                                                  counters=dict(n_jets=len(m_all)))))
    b = P._bins(data["mass"], data["pt"], np.ones(len(data["mass"]), bool), P.PEAKS["top"]["fit_range"])
    n_fit = float(b["n_pass"].sum() + b["n_fail"].sum())
    fit = dict(n_jets=data["n_window"], models={"m": {"top": dict(n_pass=1.0, n_fail=n_fit - 1)}},
               reference={"top": dict(n_pass=2.0, n_fail=n_fit - 2)})
    got = RC.step_selection([tmp_path / "shard0"], data, fit)
    assert (got["n_dataset"], got["n_staged"]) == (5 * len(m_all), len(m_all))
    assert got["n_rho_window"] == data["n_window"] <= len(m_all) - extra
    assert got["n_fit"] == int(n_fit) and got["n_pt_bins"] == 8 and got["abs_eta_max"] == 2.4
    assert [s["stage"] for s in got["steps"]] == ["dataset", "staged", "rho window", "fit"]
    (shard / "RunG_batch0.stats.json").write_text(json.dumps(dict(n_jets_read=1, counters=dict(n_jets=7))))
    with pytest.raises(SystemExit, match="staging stats say"):
        RC.step_selection([tmp_path / "shard0"], data, fit)
