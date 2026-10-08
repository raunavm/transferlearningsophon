"""experiments/STATS/paired_errors.py end to end on synthetic probe outputs:
replicates from per-jet scores, every comparison formed from paired runs, the
random control paired draw k with run k, a Poisson interval on every
rejection, every v2 run formed into the contrasts of configs/analysis/, and the
stream check of amendment A7."""
import hashlib
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("paired_errors", ROOT / "experiments/STATS/paired_errors.py")
pe = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pe)


V1 = pe.load_spec(pe.CONTRASTS["v1"])
V2 = pe.load_spec(pe.CONTRASTS["v2"])
RAND = [f"RAND2_p{i}" for i in range(1, 6)]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_model_names_are_read_from_the_contrasts_files():
    assert pe.parse_model("l162-s1b", V1) == ("L162", 1, None)
    assert pe.parse_model("r16q1mass-s4", V1) == ("R16_Q1_MASS", 4, None)
    assert pe.parse_model("rand-d3-s3", V1) == ("RAND", 3, None)
    assert pe.parse_model("rand-d1-s1b", V1) == ("RAND", 1, None)
    for bad in ("mpm-s1", "l162-s1", "l188-s1@bestval"):    # a baseline, the superseded run, a v2 name
        with pytest.raises(SystemExit):
            pe.parse_model(bad, V1)
    # the v2 names the regex could not read (audit 2026-09-29, stats-2)
    assert pe.parse_model("rand2p3-s2@wavg", V2) == ("RAND2_p3", 2, "wavg")
    assert pe.parse_model("flavf0-s1@bestval", V2) == ("FLAV_F0", 1, "bestval")
    assert pe.parse_model("r16q1masslm-s4@bestval", V2) == ("R16_Q1_MASS_LM", 4, "bestval")
    assert pe.parse_model("r63q1-s2@wavg", V2) == ("R63_Q1", 2, "wavg")
    assert pe.parse_model("l188lofo4p-s1@wavg", V2) == ("L188_LOFO4P", 1, "wavg")
    assert pe.parse_model("mpm-v2-s3@bestval", V2) == ("MPM", 3, "bestval")
    # A14: the primary checkpoint, and the self-supervised arm that leaves the family out
    assert pe.parse_model("flavf1r-s5@best70", V2) == ("FLAV_F1R", 5, "best70")
    assert pe.parse_model("mpmlofo4p-v2-s2@best70", V2) == ("MPM_LOFO4P", 2, "best70")
    assert V2["models"]["mpm-v2-s1"] == ("MPM", 1, "mtx-mpm-s1")
    assert V2["models"]["mpmlofo4p-v2-s3"] == ("MPM_LOFO4P", 3, "mtx-mpmlofo4p-s3")
    for bad in ("r29q1-s1", "l188-s1@e079", "l188-s6@bestval", "rand2p1-s3@bestval",
                "mpm-v2-s4@best70", "mpmlofo4p-s1@best70"):
        with pytest.raises(SystemExit):
            pe.parse_model(bad, V2)


def test_two_runs_with_one_name_are_fatal(tmp_path):
    # Until 2026-10-02 every self-supervised arm's runs were mpm-v2-s<k>, so the
    # leave-one-family-out arm's runs silently replaced the self-supervised arm's.
    arm = {"objective": "mpm", "runs": 2}
    (tmp_path / "g.json").write_text(json.dumps({"arms": [{**arm, "name": "MPM"},
                                                          {**arm, "name": "MPM_LOFO4P"}]}))
    assert sorted(pe.grid_models(tmp_path / "g.json")) == ["mpm-v2-s1", "mpm-v2-s2",
                                                           "mpmlofo4p-v2-s1", "mpmlofo4p-v2-s2"]
    (tmp_path / "g.json").write_text(json.dumps({"arms": [{**arm, "name": "MPM"},
                                                          {**arm, "name": "M_PM"}]}))
    with pytest.raises(SystemExit, match="names a run of MPM and of M_PM"):
        pe.grid_models(tmp_path / "g.json")


def test_every_v2_run_name_is_a_model_of_the_contrasts_file():
    # the fine-tuning inits and the extraction runs, as their builders name them
    ft = _load("build_ft_jobs_paired", "scripts/build_ft_jobs.py")
    assert {n for n, *_ in ft.v2_runs()} == set(V2["models"])
    assert {d.rsplit("/", 1)[1] for _, d, *_ in ft.v2_runs()} == {d for *_, d in V2["models"].values()}
    ex = _load("build_extract_jobs_paired", "scripts/build_extract_jobs.py")
    assert {r for r, *_ in ex.v2_runs()} <= {d for *_, d in V2["models"].values()}


def _probe_dir(d: pathlib.Path, arms: dict, seed: int):
    """probe.py's two outputs for one task, with scores whose separation is `arms[a]`."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = 3000
    y = (np.arange(n) % 2).astype(np.int8)
    rows = np.arange(n) * 3
    saved = {"bvc_resonant|rows": rows, "bvc_resonant|y": y}
    T = {"eps_s": [0.5, 0.9], "arms": {}}
    for arm, sep in arms.items():
        T["arms"][arm] = {}
        for kind in ("linear", "mlp"):
            s = rng.normal(size=n) + sep * y
            saved[f"bvc_resonant|{kind}|{arm}"] = s
            T["arms"][arm][kind] = {"auc": float(roc_auc_score(y, s)), "log1m_auc_censored": False}
    d.mkdir(parents=True)
    np.savez_compressed(d / "scores.npz", **saved)
    J = {"tasks": {"bvc_resonant": T},
         "scores_npz_sha256": hashlib.sha256((d / "scores.npz").read_bytes()).hexdigest()}
    (d / "probe_results.json").write_text(json.dumps(J))


def test_probe_ratios_pair_runs_and_the_random_draws(tmp_path):
    dirs = []
    for s in (1, 2, 3):
        arms = {f"l188-s{s}": 2.5, f"r16q1-s{s}": 1.5}
        _probe_dir(tmp_path / f"s{s}", arms, s)
        dirs.append(tmp_path / f"s{s}")
    _probe_dir(tmp_path / "rand", {"rand-d1-s1b": 2.0, "rand-d2-s2": 2.0}, 9)
    dirs.append(tmp_path / "rand")
    vec, meta, prov = pe.probe_replicates(dirs, b=50, seed=1)
    out = tmp_path / "r.npz"
    pe.save(out, vec, meta, prov, 50, 1)
    v2, m2, _ = pe.load([out])
    res = pe.ratios(v2, m2)
    rows = {(r["kind"], r["metric"], r["fine"], r["coarse"]): r for r in res["ratios"]}
    r = rows[("linear", "1-auc", "188", "17")]
    assert r["n_runs"] == 3 and r["ratio"] > 1.5 and r["ln_test_se"] > 0
    assert r["fine_models"] == ["l188-s1", "l188-s2", "l188-s3"]
    rr = rows[("linear", "1-auc", "17", "random")]
    assert rr["n_runs"] == 2 and rr["ratio"] < 1
    assert rr["pairs"] == {"rand-d1-s1b": "r16q1-s1", "rand-d2-s2": "r16q1-s2"}
    assert rr["stream_pairing"] == "unchecked"
    dd = rows[("linear", "1-auc", "random draw 1", "random draw 2")]
    assert dd["n_runs"] == 1 and dd["run_sd_proxy_17"] > 0
    # the draws borrow the 17-class run variance, less its own test noise, twice
    var_a = max(dd["run_sd_proxy_17"] ** 2 - dd["run_v_ind_proxy_17"], 0.0)
    assert dd["ln_combined_se_with_proxy"] ** 2 == pytest.approx(2 * var_a + dd["ln_test_se"] ** 2)
    assert dd["dof_with_proxy"] >= 2      # three 17-class runs: at least their two degrees of freedom
    rej = [x for x in res["rejections"] if x["level"] == "188" and x["eps_s"] == 0.9
           and x["kind"] == "linear"]
    assert len(rej) == 1 and len(rej[0]["per_run"]) == 3
    assert rej[0]["pooled"]["n_bkg"] == 3 * 1500
    for c in rej[0]["per_run"]:
        lo, hi = c["interval"]
        assert lo <= c["rejection"] <= hi


def test_a_score_file_that_is_not_the_recorded_one_is_refused(tmp_path):
    _probe_dir(tmp_path / "s1", {"l188-s1": 2.0}, 1)
    (tmp_path / "s1" / "probe_results.json").write_text(json.dumps(
        {**json.loads((tmp_path / "s1" / "probe_results.json").read_text()),
         "scores_npz_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="not the file"):
        pe.probe_replicates([tmp_path / "s1"], b=5)


def test_models_on_different_jets_are_not_paired():
    v = {"probe|t|linear|l188-s1|1-auc": np.full(3, 0.1),
         "probe|t|linear|r16q1-s1|1-auc": np.full(3, 0.2)}
    m = {"probe|t|linear|l188-s1|1-auc": {"jets": "a"},
         "probe|t|linear|r16q1-s1|1-auc": {"jets": "b"}}
    with pytest.raises(SystemExit, match="same jets"):
        pe.ratios(v, m)


def _stream(d: pathlib.Path, rows, best, best70=None):
    import hashlib as _h
    (d / "stream").mkdir(parents=True)
    for e, r in enumerate(rows):
        (d / "stream" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"run": d.name, "epoch": e, "seed_data": 1, "seed_dropout": 1, "files_sha256": "f",
             "rows_sha256": r, "sha256": _h.sha256(("f" + r).encode()).hexdigest(),
             "n_jets": 1}))
    (d / "best_epoch.json").write_text(json.dumps({"epoch": best}))
    if best70 is not None:
        (d / "best_window_epoch.json").write_text(json.dumps({"epoch": best70}))


def test_v2_pairs_whose_streams_differ_are_reported_and_left_out(tmp_path):
    # A7: run 1 agrees at every epoch; run 2 diverges at epoch 30, inside both
    # checkpoints; the ratio is formed from run 1 and run 2 is reported.
    rng = np.random.default_rng(0)
    v, m = {}, {}
    for k, div in ((1, None), (2, 30)):
        _stream(tmp_path / "runs" / f"mtx-l188-s{k}", [f"x{e}" for e in range(80)], 40)
        _stream(tmp_path / "runs" / f"mtx-r16q1-s{k}",
                [f"x{e}" if div is None or e < div else f"y{e}" for e in range(80)], 45)
        for arm, x in (("l188", 0.1), ("r16q1", 0.2)):
            key = f"probe|t|linear|{arm}-s{k}@bestval|1-auc"
            v[key], m[key] = x * np.exp(rng.normal(size=21) * 0.01), {"jets": "a"}
    rows = pe.ratios(v, m, tmp_path / "runs")["ratios"]
    r = next(x for x in rows if x["fine"] == "188" and x["coarse"] == "17")
    assert r["stream_pairing"] == "identical" and r["n_runs"] == 1
    assert r["pairs"] == {"r16q1-s1@bestval": "l188-s1@bestval"}
    assert [(x["pair"], x["first_bad_epoch"], x["upto_epoch"]) for x in r["excluded_pairs"]] == \
        [(["r16q1-s2@bestval", "l188-s2@bestval"], 30, 45)]
    # A14: the one pair left is not formed into a contrast
    assert r["not_computed"] == pe.P.ONE_PAIR and "ratio" not in r
    # a wrong root is fatal; it used to turn every v2 pair into an unchecked v1 pair
    with pytest.raises(SystemExit, match="does not exist"):
        pe.ratios(v, m, tmp_path / "elsewhere")
    with pytest.raises(SystemExit, match="recorded no training stream"):
        pe.ratios({k.replace("@bestval", ""): x for k, x in v.items()},
                  {k.replace("@bestval", ""): x for k, x in m.items()}, tmp_path / "runs")


def test_reproduction_reports_the_largest_auc_difference(tmp_path):
    _probe_dir(tmp_path / "s1", {"l188-s1": 2.0}, 1)
    J = json.loads((tmp_path / "s1" / "probe_results.json").read_text())
    J["tasks"]["bvc_resonant"]["arms"]["l188-s1"]["mlp"]["auc"] += 1e-4
    (tmp_path / "ref.json").write_text(json.dumps(J))
    r = pe.reproduction([tmp_path / "s1"], {str(tmp_path / "s1"): str(tmp_path / "ref.json")})
    w = r[str(tmp_path / "s1")]["max_abs_dauc"]
    assert w["linear"] == 0.0 and w["mlp"] == pytest.approx(1e-4)


def test_ft_cells_are_cached_and_reused(tmp_path, monkeypatch):
    calls = []
    real = pe.P.replicates

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pe.P, "replicates", counting)
    import uproot  # noqa: F401  (leg 2 reader import is lazy; this test uses leg 1)
    rng = np.random.default_rng(0)
    d = tmp_path / "leg1" / "l188-s1" / "N1000" / "s1" / "features_v2"
    d.mkdir(parents=True)
    lab = rng.integers(0, 188, 4000).astype(np.int16)
    np.save(d / "label188.npy", lab)
    np.save(d / "logits.npy", rng.normal(size=(4000, 162)).astype(np.float32))
    job = ("leg1", "l188-s1", "N1000", d, 4, 20, 1, tmp_path / "cache")
    k1, v1, m1 = pe._ft_cell(job)
    k2, v2, m2 = pe._ft_cell(job)
    assert len(calls) == 1 and k1 == k2 and np.array_equal(v1, v2) and m1 == m2


def test_ft_inits_are_models_or_named_baselines_and_nothing_else(tmp_path):
    # the committed v1 read-outs: 264 paired cells, the rest baselines by name
    cj = {leg: json.loads((ROOT / f"experiments/FIGS/data/w2b_{leg}_metrics_v2.json").read_text())
          for leg in ("leg1", "leg2")}
    jobs, base = pe.ft_jobs(tmp_path / "l1", tmp_path / "l2", cj, V1)
    assert len(jobs) == 264
    assert base == ["mpm-s1", "ref_e1arms-s1", "ref_e1arms-s2", "ref_e1arms-s3",
                    "scratch", "sophon-public"]
    cj["leg1"]["cells"]["l188-s9"] = {"N1000": {"s1": {}}}
    with pytest.raises(SystemExit, match="neither a model nor a baseline"):
        pe.ft_jobs(tmp_path / "l1", tmp_path / "l2", cj, V1)
    # v2: the cell's own record names the checkpoint rule, and must match the tree
    v2 = {"leg1": {"cells": {"rand2p1-s2": {"N1000": {"s1": {}}}}}, "leg2": {"cells": {}}}
    cell = tmp_path / "v2" / "rand2p1-s2" / "N1000" / "s1"
    cell.mkdir(parents=True)
    with pytest.raises(SystemExit, match="checkpoint rule None"):
        pe.ft_jobs(tmp_path / "v2", tmp_path / "v2", v2, V2, "bestval")
    (cell / "init_checkpoint.json").write_text(json.dumps({"rule": "wavg"}))
    with pytest.raises(SystemExit, match="'wavg'; this tree is 'bestval'"):
        pe.ft_jobs(tmp_path / "v2", tmp_path / "v2", v2, V2, "bestval")
    jobs, _ = pe.ft_jobs(tmp_path / "v2", tmp_path / "v2", v2, V2, "wavg")
    assert [j[1] for j in jobs] == ["rand2p1-s2@wavg"]


def test_v2_probe_arms_are_named_by_their_extraction_checkpoint(tmp_path):
    # Both checkpoints of one run probed side by side: two models, not a duplicate.
    for tag, sha in (("bestval", "a" * 64), ("wavg", "b" * 64)):
        d = tmp_path / "eval" / "mtx-rand2p1-s1" / tag
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"run_dir": "/data/results/mtx_v2/mtx-rand2p1-s1", "tag": tag, "checkpoint_sha256": sha}))
    index = pe.extraction_index(tmp_path / "eval", V2)
    assert index == {"a" * 64: ["rand2p1-s1@bestval"], "b" * 64: ["rand2p1-s1@wavg"]}
    _probe_dir(tmp_path / "p", {"best": 2.0, "avg": 2.1}, 1)
    J = json.loads((tmp_path / "p" / "probe_results.json").read_text())
    J["arm_checkpoints"] = {"best": "a" * 64, "avg": "b" * 64}
    (tmp_path / "p" / "probe_results.json").write_text(json.dumps(J))
    vec, meta, prov = pe.probe_replicates([tmp_path / "p"], b=5, seed=1, index=index)
    assert {k.split("|")[3] for k in vec} == {"rand2p1-s1@bestval", "rand2p1-s1@wavg"}
    assert prov["duplicates"] == []
    J["arm_checkpoints"]["avg"] = "c" * 64
    (tmp_path / "p" / "probe_results.json").write_text(json.dumps(J))
    with pytest.raises(SystemExit, match="no extraction manifest holds"):
        pe.probe_replicates([tmp_path / "p"], b=5, seed=1, index=index)


def test_one_epoch_selected_by_both_rules_is_a_model_under_both_names(tmp_path):
    # Where a run's global best epoch is also its first maximum within 70-79,
    # extract_v2.py extracts that file once under best70, lists both tags in the
    # manifest and links bestval to best70, and anomaly_heads.py reports both
    # directories. Every replicate family then holds the checkpoint under both
    # names, so '<run>@bestval' exists for every run (the A14 sensitivity rows
    # leave none out) and its ratio to '<run>@best70' is exactly 1.
    ex2 = _load("extract_v2_paired", "experiments/EVAL/extract_v2.py")
    ext, sha = tmp_path / "ext", {}
    c = {"features": np.zeros((2, 4), np.float16), "rows": np.arange(2), "label188": np.zeros(2, np.int16),
         "head_rows": np.arange(0), "head_label188": np.zeros(0, np.int16), "head": {}, "has_head": False}
    for k in (1, 2):
        run = tmp_path / "runs" / f"mtx-r16q1-s{k}"
        alias = ex2.aliases([("best70", run / "net_epoch-73_state.pt"),
                             ("bestval", run / "net_epoch-73_state.pt"),
                             ("wavg", run / "net_wavg70-79_state.pt")])
        assert alias == {"best70": "best70", "bestval": "best70", "wavg": "wavg"}
        sha[k] = {t: hashlib.sha256(f"{k}{t}".encode()).hexdigest() for t in ("best70", "wavg")}
        ex2.write(ext / run.name, {"checkpoints": {"best70": c, "wavg": c}, "n_stream": 2,
                                   "label188_sha256": "l"},
                  {"run_dir": str(run), "checkpoints": {t: {"checkpoint_sha256": s}
                                                        for t, s in sha[k].items()}}, alias)
    assert (ext / "mtx-r16q1-s1" / "bestval").is_symlink()
    index = pe.extraction_index(ext, V2)
    assert index == {**{sha[k]["best70"]: [f"r16q1-s{k}@best70", f"r16q1-s{k}@bestval"] for k in (1, 2)},
                     **{sha[k]["wavg"]: [f"r16q1-s{k}@wavg"] for k in (1, 2)}}
    names = {f"r16q1-s{k}@{t}" for k in (1, 2) for t in ("best70", "bestval", "wavg")}

    # probes: one arm per checkpoint file, and run 1's bestval directory probed too
    _probe_dir(tmp_path / "p", {"b1": 2.0, "w1": 2.1, "b2": 2.2, "w2": 2.3, "v1": 2.0}, 1)
    J = json.loads((tmp_path / "p" / "probe_results.json").read_text())
    J["arm_checkpoints"] = {"b1": sha[1]["best70"], "w1": sha[1]["wavg"], "b2": sha[2]["best70"],
                            "w2": sha[2]["wavg"], "v1": sha[1]["best70"]}
    (tmp_path / "p" / "probe_results.json").write_text(json.dumps(J))
    vec, meta, prov = pe.probe_replicates([tmp_path / "p"], b=20, seed=1, index=index)
    assert {k.split("|")[3] for k in vec} == names
    for k in (1, 2):
        for kind in ("linear", "mlp"):
            for metric in ("1-auc", "eps_b@0.50", "eps_b@0.90"):
                a, b = (vec[f"probe|bvc_resonant|{kind}|r16q1-s{k}@{t}|{metric}"] for t in ("best70", "bestval"))
                assert np.array_equal(a, b)
    assert [x["model"] for x in prov["duplicates"]] == [f"probe|bvc_resonant|{kind}|r16q1-s1@best70"
                                                        for kind in ("linear", "mlp")]
    res = pe.ratios(vec, meta)
    ck = [r for r in res["ratios"] if r["contrast"] == "checkpoint" and r["checkpoint"] == "bestval/best70"]
    assert len(ck) == 6 and all(r["fine_arm"] == "R16_Q1" and r["n_runs"] == 2 for r in ck)
    for r in ck:
        assert r["ratio"] == 1.0 and r["ci95"] == [1.0, 1.0] and r["ln_combined_se"] == 0.0
        assert r["checkpoint_label"] == "robust"
    # one file under two names cannot depend on the checkpoint: counted apart, not in N
    dep = res["checkpoint_dependence"]["bestval/best70"]["models"]
    assert (dep["identical_checkpoint"], dep["n"]) == (6, 0)

    # mass probes: the same, and the bestval directory scored as its own arm is kept once
    md, rng = tmp_path / "m", np.random.default_rng(3)
    md.mkdir()
    z, A = {"rows": np.arange(500), "label188": np.zeros(500, np.int16)}, {}
    for arm, s in (("b1", sha[1]["best70"]), ("w1", sha[1]["wavg"]), ("v1", sha[1]["best70"])):
        A[arm] = {"provenance": {"checkpoint_sha256": s}}
        for kind in ("ridge", "mlp"):
            z[f"{kind}|{arm}"] = rng.normal(0, 0.1, 500)
            A[arm][kind] = {"sigma_eff": pe.P.SigmaEffScorer(z[f"{kind}|{arm}"])()}
    np.savez(md / "residuals.npz", **z)
    (md / "mass_resolution.json").write_text(json.dumps({"arms": A}))
    mv, _, mprov = pe.mass_replicates([md], b=20, seed=1, index=index)
    assert {k.split("|")[3] for k in mv} == {f"r16q1-s1@{t}" for t in ("best70", "bestval", "wavg")}
    for kind in ("ridge", "mlp"):
        assert np.array_equal(mv[f"mass|resolution|{kind}|r16q1-s1@best70|sigma_eff"],
                              mv[f"mass|resolution|{kind}|r16q1-s1@bestval|sigma_eff"])
    assert [x["arm"] for x in mprov["duplicates"]] == ["v1", "v1"]

    # anomaly: anomaly_heads.py names a result by its directory, so both names
    # come with best70's checkpoint; given only the directory that holds the
    # files, the other name takes its value; a name that is not the
    # checkpoint's is fatal
    def entries(k, tags):
        return {t: (sha[k]["wavg" if t == "wavg" else "best70"], 1.3 + 0.1 * k, 3.0) for t in tags}
    _anomaly_file(tmp_path / "an.json", {f"r16q1-s{k}": entries(k, ("best70", "bestval", "wavg"))
                                         for k in (1, 2)})
    av, _, _ = pe.anomaly_replicates([tmp_path / "an.json"], index)
    assert {k.split("|")[3] for k in av} == names
    _anomaly_file(tmp_path / "an1.json", {"r16q1-s1": entries(1, ("best70", "wavg"))})
    av1, _, _ = pe.anomaly_replicates([tmp_path / "an1.json"], index)
    assert av1["anomaly|label_X_YY_bbbb|class_sum_matched|r16q1-s1@bestval|sigma_min@2000"].tolist() == \
        [entries(1, ("best70",))["best70"][1]]
    _anomaly_file(tmp_path / "bad.json", {"r16q1-s1": {"wavg": (sha[1]["best70"], 1.3, 3.0)}})
    with pytest.raises(SystemExit, match="is the checkpoint of r16q1-s1@best70 and r16q1-s1@bestval"):
        pe.anomaly_replicates([tmp_path / "bad.json"], index)


def test_partition_design_reads_probe_pairs_and_agrees_with_the_merge_table():
    # On the committed files, whichever draw they hold: every balance pair in one
    # entry, on a task that holds it, merged by exactly the partitions the merge
    # table (rand_v2_selection.json, read here directly) says merge it.
    d = pe.partition_design(V2, RAND)
    pp = json.loads((ROOT / V2["probe_pairs"]).read_text())
    sel = json.loads((ROOT / V2["partition_merges"]).read_text())
    assert sorted(n for x in d for n in x["pairs"]) == sorted(sel["pairs"])
    for x in d:
        assert sorted(x["merged"] + x["split"]) == RAND and 2 <= len(x["merged"]) <= 3
        for name in x["pairs"]:
            a, b = (c.removeprefix("label_") for c in sel["pairs"][name])
            assert f"{a}|{b}" in pp["probe_tasks"][x["task"]]["sub_pairs"]
            i = list(sel["pairs"]).index(name)
            assert x["merged"] == [r for r in RAND
                                   if sel["merge_vectors"][str(pp["partition_seeds"][r])][i]]
            if "balance_pairs" in pp:
                assert x["task"] == pp["balance_pairs"][name]["task"]


def test_each_probe_pair_is_read_on_its_own_task(tmp_path):
    # 2026-10-01: X->bc vs X->bq and vs X->cs have single-pair tasks of their own
    # and are still in the mixed |V_cb| task; probe_pairs.v2.json names each
    # pair's own task (balance_pairs), and that task is the one read.
    pairs = {"bb/cc": ["label_X_bb", "label_X_cc"], "bc/bq": ["label_X_bc", "label_X_bq"],
             "bc/cs": ["label_X_bc", "label_X_cs"]}
    tasks = {"bvc_resonant": ["X_bb|X_cc"], "bc_vs_bq": ["X_bc|X_bq"], "bc_vs_cs": ["X_bc|X_cs"],
             "bc_vs_rest": ["X_bc|X_bq", "X_bc|X_cs", "X_bc|QCD"]}
    own = {"bb/cc": "bvc_resonant", "bc/bq": "bc_vs_bq", "bc/cs": "bc_vs_cs"}
    vectors = {"11": [1, 1, 1], "12": [1, 1, 1], "13": [0, 0, 0], "14": [0, 0, 0], "15": [0, 0, 0]}
    seeds = {f"RAND2_p{i}": 10 + i for i in range(1, 6)}
    status = {arm: {t: {"merged": [sp for sp in sub if vectors[str(seeds[arm])][0]],
                        "split": [sp for sp in sub if not vectors[str(seeds[arm])][0]]}
                    for t, sub in tasks.items()} for arm in seeds}
    pp = {"probe_tasks": {t: {"sub_pairs": sub} for t, sub in tasks.items()},
          "balance_pairs": {k: {"classes": v, "task": own[k]} for k, v in pairs.items()},
          "partition_seeds": seeds, "status": status}
    (tmp_path / "pp.json").write_text(json.dumps(pp))
    (tmp_path / "sel.json").write_text(json.dumps({"pairs": pairs, "merge_vectors": vectors}))
    spec = {**V2, "probe_pairs": str(tmp_path / "pp.json"), "partition_merges": str(tmp_path / "sel.json")}
    d = pe.partition_design(spec, RAND)
    assert [(x["task"], x["pairs"], x["merged"]) for x in d] == [
        ("bvc_resonant", ["bb/cc"], ["RAND2_p1", "RAND2_p2"]),
        ("bc_vs_bq", ["bc/bq"], ["RAND2_p1", "RAND2_p2"]),
        ("bc_vs_cs", ["bc/cs"], ["RAND2_p1", "RAND2_p2"])]
    # without the pair's own task named, a pair in two tasks is fatal
    (tmp_path / "pp.json").write_text(json.dumps({k: v for k, v in pp.items() if k != "balance_pairs"}))
    with pytest.raises(SystemExit, match="is in 2 probe tasks"):
        pe.partition_design(spec, RAND)
    # an own task that does not hold the pair, or a pair the file does not know, is fatal
    for bad, msg in (({**pp["balance_pairs"], "bc/bq": {"classes": pairs["bc/bq"], "task": "bc_vs_cs"}},
                      "does not hold it"),
                     ({k: v for k, v in pp["balance_pairs"].items() if k != "bc/cs"},
                      "not a balance pair")):
        (tmp_path / "pp.json").write_text(json.dumps({**pp, "balance_pairs": bad}))
        with pytest.raises(SystemExit, match=msg):
            pe.partition_design(spec, RAND)


TAGS = ("best70", "wavg", "bestval")
BETWEEN = {"wavg/best70", "bestval/best70"}


def _v2_replicates(effect: float, quality: dict, b: int = 40, seed: int = 0, shift: dict | None = None,
                   dose_effect: float = 0.0):
    """Synthetic replicates for every v2 model at the three checkpoints: the
    probe tasks of the random partitions and one fine-tuning cell. ln m = task
    level + run + partition quality + `effect` where a random partition merges
    the task's pair + dose_effect x (1 - the partition's dose of the pair's axis,
    the pair's classes left out) (+ shift[arm] at the weight average), with test
    noise shared by every model and each model's own."""
    shift = shift or {}
    rng = np.random.default_rng(seed)
    design = pe.partition_design(V2, RAND)
    doses = pe.axis_doses(V2)
    vec, meta = {}, {}
    groups = [("probe", t, "linear", "1-auc") for t in dict.fromkeys(x["task"] for x in design)] + \
        [("ft", "leg1", "N1000", "1-macro_auc")]
    for i, (fam, task, kind, metric) in enumerate(groups):
        shared = rng.normal(0, 0.02, b)
        pair = next((n for x in design if x["task"] == task for n in x["pairs"]), None)
        axis = None if pair is None else doses["probe_pairs"][pair]["axis"]
        for model, (arm, run, _) in V2["models"].items():
            dose = (0.0 if axis is None or arm not in RAND else
                    1.0 - doses["doses"][arm][axis]["excluding"][pair]["realised"])
            for tag in TAGS:
                merged = any(x["task"] == task and arm in x["merged"] for x in design)
                mu = (-3.0 - 0.2 * i + 0.02 * run + quality.get(arm, 0.0) + effect * merged
                      + dose_effect * dose + (shift.get(arm, 0.0) if tag == "wavg" else 0.0))
                point = mu + rng.normal(0, 0.005)
                key = f"{fam}|{task}|{kind}|{model}@{tag}|{metric}"
                vec[key] = np.exp(np.r_[point, point + shared + rng.normal(0, 0.01, b)])
                meta[key] = {"jets": f"jets-{task}", "n": 1000}
    return vec, meta


def test_every_v2_model_is_formed_into_the_contrasts_of_the_amendments():
    vec, meta = _v2_replicates(effect=0.3, quality={"RAND2_p1": 0.5}, dose_effect=0.2)
    res = pe.ratios(vec, meta)                      # v2: the models carry a checkpoint
    rows = res["ratios"]
    assert res["contrasts"]["version"] == "v2" and not [r for r in rows if "not_computed" in r]
    # no model is left out silently
    assert res["models_in_no_contrast"] == []
    assert {c for r in rows for c in ("fine_models", "coarse_models", "models")
            for m in r.get(c, []) if m.startswith("mpm-v2-")} == {"fine_models", "coarse_models"}
    # every contrast whose arms are all in the grid forms rows (P2 forms a verdict block)
    assert V2["pending_arms"] == {}
    ids = {c["id"] for c in V2["contrasts"]} - {"p2"}
    assert {r["contrast"] for r in rows} == ids
    once = {"checkpoint", "mass_lambda_fraction"}
    for cid in ids - once:                           # at each checkpoint and between them (A14)
        assert {r["checkpoint"] for r in rows if r["contrast"] == cid} == set(TAGS) | BETWEEN, cid
    assert {r["checkpoint"] for r in rows if r["contrast"] == "checkpoint"} == BETWEEN
    assert {r["checkpoint"] for r in rows if r["contrast"] == "mass_lambda_fraction"} == set(TAGS)
    got = {(r["contrast"], r["family"], r["task"], r["fine"], r["coarse"], r["checkpoint"]): r
           for r in rows}
    # A12: the 64- and 30-class levels sit in the ladder
    assert ("ladder", "probe", "bvc_resonant", "64", "30", "best70") in got
    # A11: the matched-lambda cost against the 162-class cost, run by run, and the
    # fraction of the excess cost the matched lambda removes
    a11 = got[("mass_cost_vs_162", "probe", "bvc_resonant",
               "162 classes: mass output over none, lambda 5",
               "17 classes: mass output over none, matched lambda", "best70")]
    assert a11["n_runs"] == 5 and a11["weights"]["L162"] == 1
    frac = got[("mass_lambda_fraction", "probe", "bvc_resonant", None, None, "best70")]
    assert frac["n_runs"] == 5 and frac["runs"] == [1, 2, 3, 4, 5] and "fraction" in frac
    # A14 design change 1: the flavour pair at five runs, paired with the 17-class runs
    p2 = got[("flavour_blind_vs_17", "probe", "bvc_resonant", "17", "flavour-blind 17 (F0)", "best70")]
    assert p2["n_runs"] == 5 and "fine_run_spread" not in p2
    assert [(b["kind"], b["checkpoint"], b["runs"]) for b in res["p2_verdict"]] == \
        [("linear", t, [1, 2, 3, 4, 5]) for t in sorted(TAGS)]
    # A13: the family left out against its parent, unpaired on runs 1-3, saying why;
    # the ladder among them descriptive
    a13 = got[("family_unseen_vs_seen", "ft", "leg1", "188", "188, family unseen", "wavg")]
    assert a13["stream_pairing"].startswith("exempt") and (a13["n_fine_runs"], a13["n_coarse_runs"]) == (3, 3)
    assert a13["fine_models"] == ["l188-s1@wavg", "l188-s2@wavg", "l188-s3@wavg"]
    ladder = got[("family_unseen_ladder", "ft", "leg1", "188, family unseen", "17, family unseen", "best70")]
    assert ladder["reading"].startswith("descriptive") and ladder["n_runs"] == 3
    # the self-supervised models against every vocabulary, in fine-tuning only, Welch on runs 1-3
    ss = [r for r in rows if r["contrast"].startswith("self_supervised")]
    assert {r["family"] for r in ss} == {"ft"}
    assert {(r["contrast"], r["fine_arm"]) for r in ss if r["checkpoint"] == "best70"} == \
        {("self_supervised_vs_vocabulary", a) for a in ("L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1")} | \
        {("self_supervised_family_unseen_vs_vocabulary", a + "_LOFO4P")
         for a in ("L188", "L162", "R42_Q1", "R16_Q1")}
    assert all((r["n_fine_runs"], r["n_coarse_runs"]) == (3, 3) for r in ss)
    assert ("family_unseen_vs_seen", "ft", "leg1", "self-supervised", "self-supervised, family unseen",
            "best70") in got
    # every model at the weight average, and at the global best, against the primary
    a8 = [r for r in rows if r["contrast"] == "checkpoint" and r["family"] == "ft"]
    grid = {c["name"] for c in json.loads((ROOT / "configs/arms/v2_grid.json").read_text())["arms"]}
    for t in BETWEEN:
        assert {r["fine_arm"] for r in a8 if r["checkpoint"] == t} == grid
    assert {r["checkpoint_label"] for r in a8} <= {*pe.P.DEPENDENT, "robust", "inconclusive"}
    for t in BETWEEN:
        dep = res["checkpoint_dependence"][t]
        assert dep["models"]["n"] == sum(1 for r in rows if r["contrast"] == "checkpoint"
                                         and r["checkpoint"] == t)
        assert dep["results"]["expected_under_null"] == pytest.approx(0.05 * dep["results"]["n"])
        # the results with two runs (random against semantic is on runs 1-2) apart
        two = [r for r in rows if r["checkpoint"] == t and "checkpoint_label" in r
               and r["contrast"] != "checkpoint" and r.get("two_run_rule")]
        assert two and {r["contrast"] for r in two} >= {"random_vs_semantic"}
        assert all(r["dof"] == 1.0 for r in two)
        assert dep["results"]["two_run_rule"]["n"] == len(two)
        assert dep["results"]["other"]["n"] == dep["results"]["n"] - len(two)
    # A10 P1: partition 1 is better overall by 0.5 in ln, so the per-pair contrast
    # of a pair it merges is confounded (0.3 + 0.5/n_merged + the dose term); the
    # joint model, with a partition effect shared by every task, recovers the
    # planted 0.3 on every pair, and its dose reading the planted 0.2 on every axis.
    design = pe.partition_design(V2, RAND)
    bb = next(x for x in design if x["pairs"] == ["bb/cc"])
    p1 = got[("partition_split_vs_merged", "probe", "bvc_resonant", "partitions that split bb/cc",
              "partitions that merge bb/cc", "best70")]
    dose = p1["axis_doses"]["bb/cc"]["realised"]
    assert p1["axis_doses"]["bb/cc"]["axis"] == "b<->c" and sorted(dose) == RAND
    shift = 0.2 * (np.mean([1 - dose[a] for a in bb["merged"]]) - np.mean([1 - dose[a] for a in bb["split"]]))
    assert p1["ln_ratio"] == pytest.approx(
        0.3 + shift + (0.5 / len(bb["merged"]) if "RAND2_p1" in bb["merged"] else -0.5 / len(bb["split"])),
        abs=0.03)
    assert p1["p1_label"] == "merging costs" and p1["pool_dof"] == [len(bb["merged"]), len(bb["split"])]
    tasks = {x["task"] for x in design}
    for cid in ("partition_joint", "partition_joint_dose"):
        joint = [r for r in rows if r["contrast"] == cid and r["checkpoint"] == "best70" and r["task"]]
        assert len(joint) == len(design)
        axes = [r for r in rows if r["contrast"] == cid and r["checkpoint"] == "best70" and not r["task"]]
        assert all(r["resid_dof"] == 5 * len(tasks) - (len(tasks) + 4 + len(design) + len(axes))
                   for r in joint + axes)
        for r in joint:
            assert r["tasks_left_out"] == [] and r["v_run_within_partition"] >= 0
            assert r["v_run_task"] >= 0 and 1 < r["task_dof"] < 5
            if cid == "partition_joint_dose":
                assert r["ln_ratio"] == pytest.approx(0.3, abs=0.03), r["probe_pairs"]
                assert r["ci95"][0] < math.exp(0.3) < r["ci95"][1]
        if cid == "partition_joint_dose":
            assert {r["axis"] for r in axes} == {"b<->c", "e<->mu", "c<->light", "b<->light"}
            for r in axes:                  # one or two tasks per axis: within three of its own SE
                assert abs(r["ln_ratio"] - 0.2) < 3 * r["ln_combined_se"], r["axis"]
                assert r["coarse"] == f"{r['axis']} merged throughout (dose 0)"
        else:
            assert axes == []


def test_every_result_is_also_formed_between_the_checkpoints():
    # A14: the weight average shifts 17 classes by +0.2 and 188 by 0 in ln m, so the
    # 17-over-188 ratio moves by exp(0.2) between checkpoints and depends on the
    # checkpoint; the 162-over-188 ratio does not move and is robust; the global
    # best (bestval) does not move either.
    vec, meta = _v2_replicates(effect=0.3, quality={}, shift={"R16_Q1": 0.2})
    res = pe.ratios(vec, meta)
    rows = res["ratios"]
    got = {(r["contrast"], r["family"], r["task"], r["fine"], r["coarse"], r["checkpoint"]): r
           for r in rows}
    k = ("ladder", "probe", "bvc_resonant", "188", "17")
    b, w, d = (got[k + (t,)] for t in ("best70", "wavg", "wavg/best70"))
    assert d["ln_ratio"] == pytest.approx(w["ln_ratio"] - b["ln_ratio"], abs=1e-12)
    assert d["ln_ratio"] == pytest.approx(0.2, abs=0.02) and d["checkpoint_label"] == pe.P.DEPENDS
    assert d["fine_models"][0] == "l188-s1@wavg/best70" and d["ln_combined_se"] >= d["ln_test_se"]
    lo, hi = pe.P.bounds(d["ln_ratio"], d["ln_combined_se"], d["dof"])
    assert [math.exp(lo), math.exp(hi)] == pytest.approx(d["ci95"], rel=1e-12)
    flat = got[("ladder", "probe", "bvc_resonant", "188", "162", "wavg/best70")]
    assert abs(flat["ln_ratio"]) < 0.02 and flat["checkpoint_label"] == "robust"
    assert got[k + ("bestval/best70",)]["checkpoint_label"] == "robust"
    # every result between the checkpoints carries the label; none at one checkpoint
    labelled = [r for r in rows if "ln_combined_se" in r and "ln_ratio" in r]
    assert all(("checkpoint_label" in r) == (r["checkpoint"] in BETWEEN) for r in labelled)
    # the 17-class model itself depends on the checkpoint at the weight average, in
    # every cell, and the count says so
    a8 = [r for r in rows if r["contrast"] == "checkpoint" and r["checkpoint"] == "wavg/best70"]
    assert {r["fine_arm"] for r in a8 if r["checkpoint_label"] == pe.P.DEPENDS} == {"R16_Q1"}
    dep = res["checkpoint_dependence"]["wavg/best70"]["models"]
    assert dep["dependent"] == sum(1 for r in a8 if r["fine_arm"] == "R16_Q1")
    # the unpaired, linear and partition contrasts are formed between checkpoints too
    for cid in ("family_unseen_vs_seen", "mass_cost_vs_162", "partition_split_vs_merged",
                "partition_joint", "random_vs_semantic", "self_supervised_vs_vocabulary"):
        assert any(r["contrast"] == cid and r["checkpoint"] == "wavg/best70" for r in rows), cid
    # no rejection interval is formed for the ratio of two checkpoints
    assert not BETWEEN & {x["checkpoint"] for x in pe.ratios(*_eps_cell())["rejections"]}


def test_the_dependent_results_are_counted_against_five_percent():
    # Ten results: two dependent (one of them also within +-ln 1.1, which counts),
    # four of the ten with two runs (A14's floor at 1 degree of freedom: the 5 %
    # does not apply to them, so they are also counted on their own).
    labels = [pe.P.DEPENDS, pe.P.DEPENDS_UNDER_10] + ["robust"] * 7 + ["inconclusive"]
    two = [False, True, True, False, True, False, False, True, False, False]
    rows = ([{"contrast": "ladder", "checkpoint": "wavg/best70", "checkpoint_label": x,
              **({"two_run_rule": True} if t else {})} for x, t in zip(labels, two)] +
            [{"contrast": "checkpoint", "checkpoint": "wavg/best70", "checkpoint_label": "robust",
              "censored_models": ["m"]}] +
            [{"contrast": "ladder", "checkpoint": "best70"}])
    dep = pe.checkpoint_dependence(rows, ["wavg/best70", "bestval/best70"])
    r = dep["wavg/best70"]["results"]
    assert (r["dependent"], r["n"], r["robust"], r["inconclusive"]) == (2, 10, 7, 1)
    assert r["dependent_under_10pc"] == 1
    assert r["expected_under_null"] == pytest.approx(0.5)
    assert r["binomial_tail_p"] == pytest.approx(1 - 0.95 ** 10 - 10 * 0.05 * 0.95 ** 9)   # 0.0861
    assert r["two_run_rule"] == {"dependent": 1, "n": 4}
    assert (r["other"]["dependent"], r["other"]["n"]) == (1, 6)
    assert r["other"]["expected_under_null"] == pytest.approx(0.3)
    assert r["other"]["binomial_tail_p"] == pytest.approx(1 - 0.95 ** 6)
    assert dep["wavg/best70"]["models"] == {
        "dependent": 0, "n": 1, "expected_under_null": 0.05, "binomial_tail_p": 1.0,
        "dependent_under_10pc": 0, "robust": 1, "inconclusive": 0, "censored": 1,
        "identical_checkpoint": 0, "two_run_rule": {"dependent": 0, "n": 0},
        "other": {"dependent": 0, "n": 1, "expected_under_null": 0.05, "binomial_tail_p": 1.0}}
    assert dep["bestval/best70"]["results"]["n"] == 0
    assert dep["bestval/best70"]["results"]["binomial_tail_p"] is None


def test_a_result_whose_two_checkpoints_are_one_file_is_counted_apart():
    # bestval links to best70 when the global best lies in 70-79: the ratio is exactly
    # 1 with no error, labelled robust, and can never depend on the checkpoint, so it
    # stays out of N and of the 5 % expectation.
    same = {"contrast": "ladder", "checkpoint": "bestval/best70", "checkpoint_label": "robust",
            "ln_ratio": 0.0, "ln_combined_se": 0.0}
    rows = ([dict(same) for _ in range(3)] +
            [{"contrast": "ladder", "checkpoint": "bestval/best70", "checkpoint_label": x,
              "ln_ratio": v, "ln_combined_se": 0.02}
             for x, v in ((pe.P.DEPENDS, 0.3), ("robust", 0.01))])
    r = pe.checkpoint_dependence(rows, ["bestval/best70"])["bestval/best70"]["results"]
    assert r["identical_checkpoint"] == 3
    assert (r["dependent"], r["n"], r["robust"]) == (1, 2, 1)
    assert r["expected_under_null"] == pytest.approx(0.1)
    assert (r["other"]["n"], r["other"]["expected_under_null"]) == (2, pytest.approx(0.1))


def _eps_cell():
    v, m = {}, {}
    for k in (1, 2):
        for arm in ("l188", "r16q1"):
            for tag in ("best70", "wavg"):
                key = f"probe|t|linear|{arm}-s{k}@{tag}|eps_b@0.90"
                v[key] = np.full(5, 0.01 * k)
                m[key] = {"jets": "a", "n_bkg": 1000, "k_pass": 10.0 * k}
    return v, m


def test_the_comparison_between_checkpoints_checks_streams_up_to_the_later(tmp_path):
    # run 3 of 17 classes diverges at epoch 75: after both global best epochs (40,
    # 45) and both selected epochs within 70-79 (72, 73), inside the weight average
    # (70-79). It pairs at best70, bestval and bestval/best70 only. Run 4 diverges
    # at epoch 73, the later selected epoch: it pairs at bestval only.
    rng = np.random.default_rng(1)
    v, m = {}, {}
    for k, div in ((1, None), (2, None), (3, 75), (4, 73)):
        _stream(tmp_path / "runs" / f"mtx-l188-s{k}", [f"x{e}" for e in range(80)], 40, 72)
        _stream(tmp_path / "runs" / f"mtx-r16q1-s{k}",
                [f"x{e}" if div is None or e < div else f"y{e}" for e in range(80)], 45, 73)
        for arm, x in (("l188", 0.1), ("r16q1", 0.2)):
            for tag in TAGS:
                key = f"probe|t|linear|{arm}-s{k}@{tag}|1-auc"
                v[key], m[key] = x * np.exp(rng.normal(size=21) * 0.01), {"jets": "a"}
    res = pe.ratios(v, m, tmp_path / "runs")
    by = {r["checkpoint"]: r for r in res["ratios"] if r["contrast"] == "ladder"}
    assert by["bestval"]["n_runs"] == 4 and by["bestval"]["excluded_pairs"] == []
    for t, n, bad in (("best70", 3, [(73, 73)]), ("bestval/best70", 3, [(73, 73)]),
                      ("wavg", 2, [(75, 79), (73, 79)]), ("wavg/best70", 2, [(75, 79), (73, 79)])):
        assert by[t]["n_runs"] == n, t
        assert [(x["first_bad_epoch"], x["upto_epoch"]) for x in by[t]["excluded_pairs"]] == bad, t
    # no run has finished (no DONE): nothing to report, and nothing fails
    assert res["selected_epochs"] == {} and res["a11_shares"] == {}


def _anomaly_file(path, models: dict):
    """anomaly_heads.py's output for {arm: {tag: (sha, sigma_min, max_sic)}}, one
    signal, one N_sig, one score."""
    J = {"labels_sha256": "l" * 64, "n_bkg": 200000, "n_template": 200000, "trainings": 10,
         "models": {}}
    for arm, per in models.items():
        J["models"][arm] = {"rung": "L188", "checkpoints": {
            tag: {"checkpoint_sha256": sha, "anomaly": {"label_X_YY_bbbb": {"2000": {
                "class_sum_matched": {"sigma_min": s, "max_sic": x, "at_ceiling": False},
                "rng_seeds": [1]}}}}
            for tag, (sha, s, x) in per.items()}}
    path.write_text(json.dumps(J))


def test_the_unseen_family_sensitivity_has_an_error(tmp_path):
    # A13: sigma_min of the family left out, 188 classes against the same
    # vocabulary with the family left out (three runs), unpaired; values alone (no
    # resampling), so the error is the spread over runs. A14: the parent's runs
    # 1-3 only, one GPU product on both sides (runs 4-5 train on L40).
    seen = {1: 1.30, 2: 1.40, 3: 1.35, 4: 1.25, 5: 1.45}
    unseen = {1: 2.0, 2: 2.3, 3: 4.9}
    models, ext = {}, tmp_path / "ext"
    for arm, per in (("l188", seen), ("l188lofo4p", unseen)):
        for k, s in per.items():
            models[f"{arm}-s{k}"] = {}
            for tag, f in (("best70", 1.0), ("wavg", 1.02)):
                sha = hashlib.sha256(f"{arm}{k}{tag}".encode()).hexdigest()
                d = ext / f"mtx-{arm}-s{k}" / tag
                d.mkdir(parents=True)
                (d / "manifest.json").write_text(json.dumps(
                    {"run_dir": f"/r/mtx-{arm}-s{k}", "tag": tag, "checkpoint_sha256": sha}))
                models[f"{arm}-s{k}"][tag] = (sha, s * f, 1.2 if s > 4 else 3.0)
    _anomaly_file(tmp_path / "an.json", models)
    vec, meta, prov = pe.anomaly_replicates([tmp_path / "an.json"], pe.extraction_index(ext, V2))
    key = "anomaly|label_X_YY_bbbb|class_sum_matched|l188lofo4p-s3@best70|sigma_min@2000"
    assert vec[key].tolist() == [4.9] and meta[key]["censored"]
    pe.save(tmp_path / "an.npz", vec, meta, prov, 0, None)
    vp, mp, pp = pe.probe_replicates([_probe_one(tmp_path)], b=5, seed=1)
    pe.save(tmp_path / "p.npz", vp, mp, pp, 5, 1)
    v, m, _ = pe.load([tmp_path / "an.npz", tmp_path / "p.npz"])   # B = 0 beside B = 5
    rows = [r for r in pe.ratios({k: x for k, x in v.items() if k.startswith("anomaly")},
                                 {k: x for k, x in m.items() if k.startswith("anomaly")})["ratios"]
            if r["contrast"] == "family_unseen_vs_seen"]
    r = next(x for x in rows if x["checkpoint"] == "best70")
    a, b = np.log([seen[k] for k in (1, 2, 3)]), np.log(list(unseen.values()))
    # each arm keeps its own spread (Welch): the three runs that leave the family
    # out vary far more than the parent's
    va, vb = np.var(a, ddof=1) / 3, np.var(b, ddof=1) / 3
    assert r["runs"] == [1, 2, 3] and r["fine_models"] == [f"l188-s{k}@best70" for k in (1, 2, 3)]
    assert r["ln_ratio"] == pytest.approx(b.mean() - a.mean())
    assert r["ln_combined_se"] == pytest.approx(math.sqrt(va + vb))
    assert r["dof"] == pytest.approx((va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2))
    assert r["stream_pairing"].startswith("exempt") and r["censored_models"] == ["l188lofo4p-s3@best70"]
    assert "not detected" in r["note"] and r["n_boot"] == 0
    d = next(x for x in rows if x["checkpoint"] == "wavg/best70")
    assert d["ln_ratio"] == pytest.approx(0.0, abs=1e-12)


def _probe_one(tmp_path):
    _probe_dir(tmp_path / "probe", {"l188-s1": 2.0}, 3)
    return tmp_path / "probe"


def test_a_contrast_names_only_grid_arms_or_pending_ones(tmp_path):
    spec = json.loads((ROOT / "configs/analysis/contrasts.v2.json").read_text())
    bad = {**spec, "contrasts": spec["contrasts"] + [{"id": "x", "kind": "pairs",
                                                       "pairs": [["L188", "L118"]]}]}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match=r"compares \['L118'\]"):
        pe.load_spec(tmp_path / "bad.json")
    # an arm still listed as pending once the grid has it (F1r, 2026-10-01) is fatal
    stale = {**spec, "pending_arms": {**spec["pending_arms"], "FLAV_F1R": "?"}}
    (tmp_path / "stale.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match=r"\['FLAV_F1R'\] are model arms now"):
        pe.load_spec(tmp_path / "stale.json")
    # F1r is a grid arm: both of its contrasts form
    assert V2["models"]["flavf1r-s2"] == ("FLAV_F1R", 2, "mtx-flavf1r-s2")
    v, m = {}, {}
    for arm in ("flavf1", "flavf1r", "flavf0"):
        for k in (1, 2):
            key = f"probe|bvc_resonant|linear|{arm}-s{k}@bestval|1-auc"
            v[key], m[key] = np.full(5, {"flavf1": 0.1, "flavf1r": 0.12, "flavf0": 0.13}[arm]), {"jets": "a"}
    got = {r["contrast"]: r for r in pe.ratios(v, m, spec=V2)["ratios"]}
    assert got["flavour_cut_alignment"]["ratio"] == pytest.approx(1.2)
    assert got["flavour_random_cut_vs_blind"]["ratio"] == pytest.approx(0.13 / 0.12)


# ------------------------------------------------------------------- A14 (v2)
def _values(ln: dict, tag: str = "best70", kind: str = "linear"):
    """Replicates without resampling (B = 0) from {(task, model): ln m} at `tag`."""
    v, m = {}, {}
    for (task, model), x in ln.items():
        key = f"probe|{task}|{kind}|{model}@{tag}|1-auc"
        v[key], m[key] = np.array([math.exp(x)]), {"jets": f"jets-{task}"}
    return v, m


def _slug(arm: str) -> str:
    return arm.lower().replace("_", "")


def test_p1_is_welch_on_the_runs_that_replicate_each_partition():
    d = next(x for x in pe.partition_design(V2, RAND) if x["pairs"] == ["bb/cc"])
    y = {"RAND2_p1": (0.10, 0.14), "RAND2_p2": (0.30, 0.26), "RAND2_p3": (0.0, 0.05),
         "RAND2_p4": (0.02, 0.01), "RAND2_p5": (-0.03, -0.01)}
    v, m = _values({(d["task"], f"{_slug(a)}-s{r}"): x - 3 for a, ys in y.items()
                    for r, x in enumerate(ys, 1)})
    r = next(x for x in pe.ratios(v, m)["ratios"] if x["contrast"] == "partition_split_vs_merged")
    M, S = d["merged"], d["split"]
    within = {a: (ys[0] - ys[1]) ** 2 / 2 for a, ys in y.items()}         # one dof per partition
    s2m, s2s = (sum(within[a] for a in g) / len(g) for g in (M, S))
    cm, cs = 1 / (2 * len(M)), 1 / (2 * len(S))                         # sum of w^2 per side
    var = cm * s2m + cs * s2s
    assert r["ln_ratio"] == pytest.approx(np.mean([np.mean(y[a]) for a in M]) -
                                          np.mean([np.mean(y[a]) for a in S]))
    assert r["ln_combined_se"] ** 2 == pytest.approx(var)
    assert r["dof"] == pytest.approx(var ** 2 / ((cm * s2m) ** 2 / len(M) + (cs * s2s) ** 2 / len(S)))
    assert r["p1_label"] == pe.P.p1_label(*pe.P.bounds(r["ln_ratio"], r["ln_combined_se"], r["dof"]))
    doses = pe.axis_doses(V2)
    assert r["axis_doses"] == {"bb/cc": {"axis": "b<->c", "realised": {
        a: doses["doses"][a]["b<->c"]["excluding"]["bb/cc"]["realised"] for a in RAND}}}
    # one run per partition: no replicate, not computed
    v1 = {k: x for k, x in v.items() if "-s1@" in k}
    r1 = next(x for x in pe.ratios(v1, {k: m[k] for k in v1})["ratios"]
              if x["contrast"] == "partition_split_vs_merged")
    assert "not_computed" in r1 and "p1_label" not in r1


def test_the_joint_fit_weights_by_run_plus_test_variance_and_leaves_out_a_censored_task():
    vec, meta = _v2_replicates(effect=0.3, quality={"RAND2_p1": 0.5})
    meta["probe|ee_vs_mm|linear|rand2p3-s1@best70|1-auc"]["censored"] = True
    rows = [r for r in pe.ratios(vec, meta)["ratios"]
            if r["contrast"] == "partition_joint" and r["checkpoint"] == "best70"]
    design = pe.partition_design(V2, RAND)
    tasks = list(dict.fromkeys(x["task"] for x in design))
    out = [r for r in rows if r["task"] == "ee_vs_mm"]
    assert len(out) == 1 and "left out of the joint fit" in out[0]["not_computed"]
    assert out[0]["censored_models"] == ["rand2p3-s1@best70"]
    fit = [r for r in rows if r["task"] != "ee_vs_mm"]
    assert all(r["tasks_left_out"] == ["ee_vs_mm"] and "ee_vs_mm" not in r["tasks_in_fit"] for r in fit)
    assert all(r["resid_dof"] == 5 * 6 - (6 + 4 + 6) for r in fit)
    for r in fit:
        assert r["ln_ratio"] == pytest.approx(0.3, abs=0.03)
    # the task's run variance from the runs that replicate each partition, by hand
    pts = np.log([[vec[f"probe|bvc_resonant|linear|{_slug(a)}-s{k}@best70|1-auc"] for k in (1, 2)]
                  for a in RAND])                                    # partitions x runs x (1 + B)
    s2 = float(np.mean((pts[:, 0, 0] - pts[:, 1, 0]) ** 2 / 2))
    vind = np.mean([np.cov(p[:, 1:]).trace() / 2 - max(np.cov(p[:, 1:])[0, 1], 0.0) for p in pts])
    bb = next(r for r in fit if r["task"] == "bvc_resonant")
    assert bb["v_run_within_partition"] == pytest.approx(max(s2 - vind, 0.0), rel=1e-9)


def _semantic(cells_offset: tuple, task="bvc_resonant"):
    """17- and 43-class runs 1-2 at ln m = -3; the cells of bb/cc lower by
    `cells_offset` (run 1, run 2); the other partitions at -3."""
    doses = pe.axis_doses(V2)
    cells = [x["partition"] for x in doses["cells"] if x["pair"] == "bb/cc"]
    ln = {}
    for r in (1, 2):
        for a in ["R16_Q1", "R42_Q1"] + RAND:
            ln[(task, f"{_slug(a)}-s{r}")] = -3.0 - (cells_offset[r - 1] if a in cells else 0.0)
    return _values(ln), cells


@pytest.mark.parametrize("offset, axis, pair", [((0.5, 0.52), "beats", "inconclusive"),
                                                ((0.01, 0.012), "inconclusive", "equal")])
def test_random_against_semantic_reads_the_two_accounts(offset, axis, pair):
    # two runs: 95 % at t(1) = 12.71, 90 % at t(1) = 6.31. Contrasts 0.5, 0.52:
    # mean 0.51, SE 0.01, 95 % [0.383, 0.637] (beats), 90 % [0.447, 0.573] (not
    # within +-ln 1.1). Contrasts 0.01, 0.012: mean 0.011, SE 0.001, 95 %
    # [-0.0017, 0.0237], 90 % [0.0047, 0.0173] (equal).
    (v, m), cells = _semantic(offset)
    rows = [r for r in pe.ratios(v, m)["ratios"] if r["contrast"] == "random_vs_semantic"]
    acc = next(r for r in rows if r["fine"].startswith("the cells of bb/cc"))
    assert acc["coarse"] == "17" and acc["partitions"] == cells and acc["account_cells"] == ["bb/cc"]
    assert acc["n_runs"] == 2 and acc["dof"] == 1.0
    assert acc["ln_ratio"] == pytest.approx(np.mean(offset))
    assert acc["ln_combined_se"] == pytest.approx(abs(offset[0] - offset[1]) / 2)
    assert (acc["axis_account"], acc["pair_account"]) == (axis, pair)
    # each cell alone carries the accounts; a partition that splits the pair does not
    one = next(r for r in rows if r["partitions"] == [cells[0]] and r["coarse"] == "17")
    assert one["account_cells"] == ["bb/cc"] and one["axis_account"] == axis and one["merges"]
    other = next(r for r in rows if r["partitions"] == ["RAND2_p2"] and r["coarse"] == "17")
    assert "axis_account" not in other and "account_cells" not in other and not other["merges"]
    # against the 43-class model: rows by merge status, no accounts
    r43 = [r for r in rows if r["coarse"] == "43"]
    assert {r["fine"] for r in r43} >= {"partitions that merge bb/cc", "partitions that split bb/cc"}
    assert not any("axis_account" in r for r in r43)


def _p2_values(d_offset: float, d_spread: float = 0.0, gap: float = 1.0):
    """P2's five roles, runs 1-5, ln(1 - AUC). Two-prong: R42 = -4, R16 = R42 + gap
    + 0.01 k, F0 = R16 +- 0.001, F1 = F0 - 0.8 - 0.01 k, F1R = F1 + d_offset +
    0.001 k +- d_spread. Four-prong: F0 = -3, F1 = -3.5 - 0.01 k."""
    ln = {}
    for k in range(1, 6):
        r42 = -4.0
        r16 = r42 + gap + 0.01 * k
        f0 = r16 + 0.001 * (-1) ** k
        f1 = f0 - 0.8 - 0.01 * k
        f1r = f1 + d_offset + 0.001 * k + d_spread * (-1) ** k
        for arm, x in (("R42_Q1", r42), ("R16_Q1", r16), ("FLAV_F0", f0), ("FLAV_F1", f1), ("FLAV_F1R", f1r)):
            ln[("bvc_resonant", f"{_slug(arm)}-s{k}")] = x
        ln[("bvc_4prong", f"flavf0-s{k}")] = -3.0
        ln[("bvc_4prong", f"flavf1-s{k}")] = -3.5 - 0.01 * k
    return _values(ln)


@pytest.mark.parametrize("d_offset, d_spread, gap, withdrawal, d_label", [
    (0.6, 0.0, 1.0, "not withdrawn", "holds"),          # F1R - F1 = 0.6 > gap/4 = 0.2575
    (0.05, 0.0, 1.0, "withdrawn", "holds"),             # 0 < F1R - F1 = 0.05 < gap/4
    (0.25, 0.1, 1.0, "inconclusive", "holds"),          # 95 % [0.096, 0.370]: above 0, spans gap/4
    (0.6, 0.0, -0.5, "not evaluable: the 43- to 17-class gap is not positive", "holds")])
def test_the_p2_verdict_block(d_offset, d_spread, gap, withdrawal, d_label):
    from scipy.stats import t as student
    v, m = _p2_values(d_offset, d_spread, gap)
    blocks = pe.ratios(v, m)["p2_verdict"]
    assert len(blocks) == 1
    b = blocks[0]
    assert (b["kind"], b["checkpoint"], b["runs"]) == ("linear", "best70", [1, 2, 3, 4, 5])
    k = np.arange(1, 6)
    t95, t90 = student.ppf(0.975, 4), student.ppf(0.95, 4)

    def check(clause, x, level_t, key):
        assert clause["per_run"] == pytest.approx(list(x), abs=1e-12)
        assert clause["estimate"] == pytest.approx(x.mean(), abs=1e-12)
        se = x.std(ddof=1) / math.sqrt(5)
        assert clause["ln_combined_se"] == pytest.approx(se, abs=1e-12) and clause["dof"] == pytest.approx(4)
        assert clause[key] == pytest.approx([x.mean() - level_t * se, x.mean() + level_t * se], abs=1e-12)
    g = gap + 0.01 * k
    check(b["gap"], g, t95, "ci95")
    assert b["margin"]["value"] == pytest.approx(g.mean() / 4)
    cl = b["clauses"]
    check(cl["a"], 0.5 + 0.01 * k, t95, "ci95")
    check(cl["b"], 0.8 + 0.01 * k - g / 2, t95, "ci95")
    check(cl["c"], 0.001 * (-1) ** k, t90, "ci90")
    check(cl["d"], d_offset + 0.001 * k + d_spread * (-1) ** k, t95, "ci95")
    assert cl["a"]["label"] == "holds" and cl["b"]["label"] == "holds"   # (b) is 0.3 or 1.05 above 0
    assert cl["c"]["label"] == ("holds" if gap > 0 else "not evaluable")
    assert cl["c"]["margin"] == b["margin"]["value"]
    assert cl["d"]["label"] == d_label
    assert b["withdrawal"]["label"] == withdrawal


def test_a11_reads_the_fraction_with_its_fieller_interval():
    k = np.arange(1, 6)
    c162 = 0.10 + 0.01 * (k % 2)                        # 162 classes, lambda 5
    c17 = 0.50 + 0.02 * (k - 3)                          # 17 classes, lambda 5
    c17m = 0.25 + 0.015 * (k % 3)                        # 17 classes, matched lambda
    ln = {}
    for i, r in enumerate(k):
        for arm, x in (("L162", -4.0), ("L162_MASS", -4.0 + c162[i]), ("R16_Q1", -3.0),
                       ("R16_Q1_MASS", -3.0 + c17[i]), ("R16_Q1_MASS_LM", -3.0 + c17m[i])):
            ln[("bvc_resonant", f"{_slug(arm)}-s{r}")] = x
    v, m = _values(ln)
    r = next(x for x in pe.ratios(v, m)["ratios"] if x["contrast"] == "mass_lambda_fraction")
    N, D = c17 - c17m, c17 - c162
    assert r["runs"] == [1, 2, 3, 4, 5] and r["fraction"] == pytest.approx(N.mean() / D.mean())
    assert r["ci95"] == pytest.approx(pe.P.fieller(N[:, None], D[:, None])["ci95"], abs=1e-12)
    assert r["numerator"]["estimate"] == pytest.approx(N.mean())
    assert r["denominator"]["estimate"] == pytest.approx(D.mean())


def test_the_selected_epochs_and_the_a11_shares_are_read_from_finished_runs(tmp_path):
    def run(name, epochs, done=True, grad=True, best=(74, 33)):
        d = tmp_path / name
        (d / "metrics").mkdir(parents=True)
        for e, (cls, reg, g_cls, g_reg, cos) in enumerate(epochs):
            rec = {"epoch": e, "train": {"loss": cls + 5 * reg, "loss_cls": cls, "loss_reg": reg}}
            if grad:
                rec["grad_diag"] = {"grad_norm": {"loss_cls": g_cls, "lambda_loss_reg": g_reg},
                                    "cosine": cos}
            (d / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps(rec))
        (d / "best_window_epoch.json").write_text(json.dumps({"epoch": best[0]}))
        (d / "best_epoch.json").write_text(json.dumps({"epoch": best[1]}))
        if done:
            (d / "DONE").write_text("{}")
    run("mtx-l162mass-s1", [(1.0, 0.03, 2.0, 0.5, 0.1), (0.8, 0.03, 2.0, 1.0, -0.1)])
    run("mtx-l162mass-s2", [(1.0, 0.02, 1.0, 0.5, 0.3)])
    run("mtx-r16q1masslm-s1", [(1.0, 0.1, 1.0, 1.0, 0.0)], grad=False)
    run("mtx-r16q1mass-s1", [(1.0, 0.1, 1.0, 1.0, 0.0)], done=False)      # still training
    run("mtx-l188-s1", [(1.0, 0.0, 1.0, 1.0, 0.0)], best=(71, 70))         # no mass output
    sh = pe.mass_shares(V2, tmp_path)
    assert set(sh) == {"L162_MASS", "R16_Q1_MASS_LM"}
    s1 = sh["L162_MASS"]["runs"][1]
    x = np.mean([5 * 0.03 / 1.0, 5 * 0.03 / 0.8])                         # lambda L_reg / L_cls
    rho = np.mean([0.25, 0.5])
    assert s1["loss_ratio"] == pytest.approx(x) and s1["loss_share"] == pytest.approx(x / (1 + x))
    assert s1["grad_norm_ratio"] == pytest.approx(rho) and s1["grad_share"] == pytest.approx(rho / (1 + rho))
    assert s1["grad_cosine"] == pytest.approx(0.0) and s1["epochs"] == 2 and s1["grad_epochs"] == 2
    assert sh["L162_MASS"]["lambda"] == 5.0
    assert sh["L162_MASS"]["loss_ratio"]["mean"] == pytest.approx(np.mean([x, 0.1]))
    assert sh["L162_MASS"]["loss_ratio"]["n_runs"] == 2
    lm = sh["R16_Q1_MASS_LM"]
    assert lm["lambda"] == 1.74 and lm["runs"][1]["loss_ratio"] == pytest.approx(0.174)
    assert "grad_share" not in lm["runs"][1] and lm["loss_share"]["sd"] is None
    sel = pe.selected_epochs(V2, tmp_path)
    assert sel["L188"] == {1: {"best70": 71, "bestval": 70}}
    assert sel["L162_MASS"] == {1: {"best70": 74, "bestval": 33}, 2: {"best70": 74, "bestval": 33}}
    assert "R16_Q1_MASS" not in sel
    # no run yet: nothing
    (tmp_path / "empty").mkdir()
    assert pe.mass_shares(V2, tmp_path / "empty") == {} and pe.selected_epochs(V2, tmp_path / "empty") == {}


# ------------------------------------------- A14 frozen readouts: twins, pooled, untrained
def test_the_twins_the_pooled_readout_and_the_untrained_trunk_are_named_apart():
    # the BatchNorm twins are checkpoints of every run; the pooled embedding is a
    # readout suffix; the untrained trunk is a reference model at its own tag only
    assert pe.parse_model("l188-s1@best70_bn", V2) == ("L188", 1, "best70_bn")
    assert pe.parse_model("r16q1-s4@bestval_bn:pooled", V2) == ("R16_Q1", 4, "bestval_bn:pooled")
    assert pe.parse_model("mpm-v2-s2@best70:pooled", V2) == ("MPM", 2, "best70:pooled")
    assert pe.parse_model("init-s3@init", V2) == ("INIT", 3, "init")
    assert pe.parse_model("init-s5@init:pooled", V2) == ("INIT", 5, "init:pooled")
    assert V2["reference_models"]["init-s1"] == ("INIT", 1, None) and "init-s1" not in V2["models"]
    for bad in ("l188-s1@init",            # a reference tag on a grid run
                "init-s1@best70",          # a model checkpoint on a reference
                "init-s6@init",            # run index 6 has no untrained trunk
                "l188-s1@best70:mlp",      # not a readout
                "l188-s1@best70:features", # the first readout takes no suffix
                "l188-s1@best70:",
                "l188-s1@wavg_bn"):        # the weight average has its statistics already
        with pytest.raises(SystemExit):
            pe.parse_model(bad, V2)
    # v1 has the class token only
    with pytest.raises(SystemExit, match="class token only"):
        pe._with_readout(["l162-s1b"], "pooled", "x")
    assert pe._with_readout(["l162-s1b"], "features", "x") == ["l162-s1b"]
    assert pe.base_checkpoint("bestval_bn:pooled", V2) == "bestval"
    assert pe.base_checkpoint("wavg", V2) == "wavg"
    assert pe.readouts_of(["l188-s1@best70", "l188-s1@wavg:pooled", "l162-s1b"]) == ["", ":pooled"]
    assert pe.between_checkpoints(V2, ["", ":pooled"]) == [
        "wavg/best70", "bestval/best70", "wavg:pooled/best70:pooled", "bestval:pooled/best70:pooled"]


def test_the_untrained_trunk_is_named_by_its_extraction_directory(tmp_path):
    def man(d, run_dir, tag, sha):
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"run_dir": f"/data/results/mtx_v2/{run_dir}", "tag": tag, "tags": [tag],
             "checkpoint_sha256": sha}))
    man(tmp_path / "e" / "init-s2" / "init", "mtx-l188-s2", "init", "i" * 64)
    man(tmp_path / "e" / "mtx-l188-s2" / "best70", "mtx-l188-s2", "best70", "b" * 64)
    man(tmp_path / "e" / "mtx-l188-s2" / "best70_bn", "mtx-l188-s2", "best70_bn", "n" * 64)
    assert pe.extraction_index(tmp_path / "e", V2) == {
        "i" * 64: ["init-s2@init"], "b" * 64: ["l188-s2@best70"], "n" * 64: ["l188-s2@best70_bn"]}
    # the untrained trunk of run 2 filed as run 3's reference is refused
    man(tmp_path / "f" / "init-s3" / "init", "mtx-l188-s2", "init", "i" * 64)
    with pytest.raises(SystemExit, match="not that run index's reference"):
        pe.extraction_index(tmp_path / "f", V2)


def test_the_two_readouts_of_one_checkpoint_are_two_models(tmp_path):
    # probe.py writes one directory per readout; the arm and its checkpoint digest are
    # the same in both, and only the recorded readout tells them apart
    d = tmp_path / "e" / "mtx-r16q1-s1" / "best70"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"run_dir": "/data/results/mtx_v2/mtx-r16q1-s1", "tag": "best70", "checkpoint_sha256": "a" * 64}))
    index = pe.extraction_index(tmp_path / "e", V2)
    for readout, sep in (("features", 2.0), ("pooled", 1.5)):
        _probe_dir(tmp_path / readout, {"r16q1-s1@best70": sep}, 1)
        J = json.loads((tmp_path / readout / "probe_results.json").read_text())
        J.update(arm_checkpoints={"r16q1-s1@best70": "a" * 64}, readout=readout)
        (tmp_path / readout / "probe_results.json").write_text(json.dumps(J))
    vec, _, prov = pe.probe_replicates([tmp_path / "features", tmp_path / "pooled"], b=5, seed=1,
                                       index=index)
    assert {k.split("|")[3] for k in vec} == {"r16q1-s1@best70", "r16q1-s1@best70:pooled"}
    assert prov["duplicates"] == []
    assert not np.array_equal(vec["probe|bvc_resonant|linear|r16q1-s1@best70|1-auc"],
                              vec["probe|bvc_resonant|linear|r16q1-s1@best70:pooled|1-auc"])


def _frozen_replicates(b: int = 30, seed: int = 0):
    """Replicates of one probe task for the 188- and 17-class runs 1-3 at the primary,
    the weight average and the primary's BatchNorm twin, through both readouts, and
    the untrained trunk of runs 1-3 through both. ln m = level + 0.02 run (+0.3 at 17
    classes) (+0.1 through the pooled embedding) (+0.15 at the twin of 17 classes)."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 0.02, b)
    vec, meta = {}, {}
    models = [(f"{a}-s{k}", tag, ro, -3.0 + 0.02 * k + (0.3 if a == "r16q1" else 0.0)
               + (0.1 if ro else 0.0) + (0.15 if a == "r16q1" and tag == "best70_bn" else 0.0))
              for a in ("l188", "r16q1") for k in (1, 2, 3)
              for tag in ("best70", "wavg", "best70_bn") for ro in ("", ":pooled")]
    models += [(f"init-s{k}", "init", ro, -1.0 + 0.02 * k) for k in (1, 2, 3) for ro in ("", ":pooled")]
    for m, tag, ro, mu in models:
        key = f"probe|bvc_resonant|linear|{m}@{tag}{ro}|1-auc"
        point = mu + rng.normal(0, 0.003)
        vec[key] = np.exp(np.r_[point, point + shared + rng.normal(0, 0.005, b)])
        meta[key] = {"jets": "j", "n": 1000}
    return vec, meta


def test_contrasts_are_formed_within_one_readout_and_at_the_twins():
    vec, meta = _frozen_replicates()
    res = pe.ratios(vec, meta)
    rows = res["ratios"]
    lad = {r["checkpoint"]: r for r in rows if r["contrast"] == "ladder"}
    # every checkpoint, through each readout, and between the checkpoints within one readout
    assert set(lad) == {"best70", "wavg", "best70_bn", "best70:pooled", "wavg:pooled",
                        "best70_bn:pooled", "wavg/best70", "wavg:pooled/best70:pooled"}
    for tag, r in lad.items():
        ro = ":pooled" in tag
        assert all(m.endswith(":pooled") == ro for m in r["fine_models"] + r["coarse_models"]), tag
    # the 17-over-188 ratio: exp(0.3) at the primary in either readout, exp(0.45) at the twin
    assert lad["best70"]["ln_ratio"] == pytest.approx(0.3, abs=0.01)
    assert lad["best70:pooled"]["ln_ratio"] == pytest.approx(0.3, abs=0.01)
    assert lad["best70_bn"]["ln_ratio"] == pytest.approx(0.45, abs=0.01)
    # the A14 label within each readout, and the count of each readout apart
    assert lad["wavg:pooled/best70:pooled"]["checkpoint_label"] == "robust"
    assert "checkpoint_label" not in lad["best70_bn"]
    dep = res["checkpoint_dependence"]
    assert set(dep) == {"wavg/best70", "bestval/best70", "wavg:pooled/best70:pooled",
                        "bestval:pooled/best70:pooled"}
    assert dep["wavg:pooled/best70:pooled"]["results"]["n"] == 1
    assert dep["wavg:pooled/best70:pooled"]["models"]["n"] == 2          # 188 and 17
    # the untrained trunk enters no contrast, says so, and still has its points
    assert res["models_in_no_contrast"] == sorted(f"init-s{k}@init{ro}" for k in (1, 2, 3)
                                                  for ro in ("", ":pooled"))
    assert res["point_values"]["probe|bvc_resonant|linear|1-auc"]["init-s1@init"] > 0


def test_a_twin_is_stream_checked_through_its_parent(tmp_path):
    # run 2 of 17 classes diverges at epoch 73, after the 188-class selected epoch 72
    # and at the 17-class one: at best70 and at its twin the pair is left out alike
    v, m = _frozen_replicates()
    v = {k: x for k, x in v.items() if "init" not in k and ":pooled" not in k}
    for k in (1, 2, 3):
        _stream(tmp_path / f"mtx-l188-s{k}", [f"x{e}" for e in range(80)], 40, 72)
        _stream(tmp_path / f"mtx-r16q1-s{k}", [f"x{e}" if k != 2 or e < 73 else f"y{e}" for e in range(80)],
                45, 73)
    res = pe.ratios(v, {k: m[k] for k in v}, tmp_path)
    lad = {r["checkpoint"]: r for r in res["ratios"] if r["contrast"] == "ladder"}
    for t in ("best70", "best70_bn"):
        assert lad[t]["n_runs"] == 2, t
        assert [(x["first_bad_epoch"], x["upto_epoch"]) for x in lad[t]["excluded_pairs"]] == [(73, 73)], t
