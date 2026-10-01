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
    for bad in ("r29q1-s1", "l188-s1@e079", "l188-s6@bestval", "rand2p1-s3@bestval"):
        with pytest.raises(SystemExit):
            pe.parse_model(bad, V2)


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


def _stream(d: pathlib.Path, rows, best):
    import hashlib as _h
    (d / "stream").mkdir(parents=True)
    for e, r in enumerate(rows):
        (d / "stream" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"run": d.name, "epoch": e, "seed_data": 1, "seed_dropout": 1, "files_sha256": "f",
             "rows_sha256": r, "sha256": _h.sha256(("f" + r).encode()).hexdigest(),
             "n_jets": 1}))
    (d / "best_epoch.json").write_text(json.dumps({"epoch": best}))


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
    assert index == {"a" * 64: "rand2p1-s1@bestval", "b" * 64: "rand2p1-s1@wavg"}
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


def _v2_replicates(effect: float, quality: dict, b: int = 40, seed: int = 0, shift: dict | None = None):
    """Synthetic replicates for every v2 model at both checkpoints: six probe tasks
    and one fine-tuning cell. ln m = task level + run + partition quality +
    `effect` where a random partition merges the task's pair (+ shift[arm] at the
    weight average), with test noise shared by every model and each model's own."""
    shift = shift or {}
    rng = np.random.default_rng(seed)
    design = pe.partition_design(V2, RAND)
    vec, meta = {}, {}
    groups = [("probe", t, "linear", "1-auc") for t in dict.fromkeys(x["task"] for x in design)] + \
        [("ft", "leg1", "N1000", "1-macro_auc")]
    for i, (fam, task, kind, metric) in enumerate(groups):
        shared = rng.normal(0, 0.02, b)
        for model, (arm, run, _) in V2["models"].items():
            for tag in ("bestval", "wavg"):
                merged = any(x["task"] == task and arm in x["merged"] for x in design)
                mu = (-3.0 - 0.2 * i + 0.02 * run + quality.get(arm, 0.0) + effect * merged
                      + (shift.get(arm, 0.0) if tag == "wavg" else 0.0))
                point = mu + rng.normal(0, 0.005)
                key = f"{fam}|{task}|{kind}|{model}@{tag}|{metric}"
                vec[key] = np.exp(np.r_[point, point + shared + rng.normal(0, 0.01, b)])
                meta[key] = {"jets": f"jets-{task}", "n": 1000}
    return vec, meta


def test_every_v2_model_is_formed_into_the_contrasts_of_the_amendments():
    vec, meta = _v2_replicates(effect=0.3, quality={"RAND2_p1": 0.5})
    res = pe.ratios(vec, meta)                      # v2: the models carry a checkpoint
    rows = res["ratios"]
    assert res["contrasts"]["version"] == "v2" and not [r for r in rows if "not_computed" in r]
    # no model is left out silently (the self-supervised runs enter through A8 only)
    assert res["models_in_no_contrast"] == []
    assert {c for r in rows for c in ("fine_models", "coarse_models", "models")
            for m in r.get(c, []) if m.startswith("mpm-v2-")} == {"fine_models", "coarse_models"}
    # every contrast whose arms are all in the grid forms rows; one that names an
    # arm still pending (F1r until its grid entry) forms none
    ids = {c["id"]: c for c in V2["contrasts"]
           if not {a for pr in c.get("pairs", []) for a in pr} & set(V2["pending_arms"])}
    assert set(V2["pending_arms"]) <= {"FLAV_F1R"}
    assert len(ids) == len(V2["contrasts"]) - 2 * bool(V2["pending_arms"])
    assert {r["contrast"] for r in rows} == set(ids)
    for cid in set(ids) - {"checkpoint"}:            # at both checkpoints and between them (A8)
        assert {r["checkpoint"] for r in rows if r["contrast"] == cid} == \
            {"bestval", "wavg", "wavg/bestval"}, cid
    assert {r["checkpoint"] for r in rows if r["contrast"] == "checkpoint"} == {"wavg/bestval"}
    got = {(r["contrast"], r["family"], r["task"], r["fine"], r["coarse"], r["checkpoint"]): r
           for r in rows}
    # A12: the 64- and 30-class levels sit in the ladder
    assert ("ladder", "probe", "bvc_resonant", "64", "30", "bestval") in got
    # A11: the matched-lambda cost against the 162-class cost, run by run
    a11 = got[("mass_cost_vs_162", "probe", "bvc_resonant",
               "162 classes: mass output over none, lambda 5",
               "17 classes: mass output over none, matched lambda", "bestval")]
    assert a11["n_runs"] == 5 and a11["weights"]["L162"] == 1
    # A10 P2: F0 against the 17-class runs, with their spread over all five
    p2 = got[("flavour_blind_vs_17", "probe", "bvc_resonant", "17", "flavour-blind 17 (F0)", "bestval")]
    assert p2["n_runs"] == 2 and p2["fine_run_spread"]["n_runs"] == 5
    # A13: the family left out against its parent, unpaired and saying why
    a13 = got[("family_unseen_vs_seen", "ft", "leg1", "188", "188, family unseen", "wavg")]
    assert a13["stream_pairing"].startswith("exempt") and (a13["n_fine_runs"], a13["n_coarse_runs"]) == (5, 3)
    assert ("family_unseen_ladder", "ft", "leg1", "188, family unseen", "17, family unseen",
            "bestval") in got
    # A8: every model at the weight average against its best-validation checkpoint
    a8 = [r for r in rows if r["contrast"] == "checkpoint" and r["family"] == "ft"]
    assert {r["fine_arm"] for r in a8} == {c["name"] for c in json.loads(
        (ROOT / "configs/arms/v2_grid.json").read_text())["arms"]}
    assert all(isinstance(r["robust_to_checkpoint"], bool) for r in a8)
    # A10 P1: partition 1 is better overall by 0.5 in ln, so the per-pair contrast
    # of a pair it merges is confounded (0.3 + 0.5/n_merged); the joint model, with
    # a partition effect shared by every task, recovers the planted 0.3 on every pair.
    design = pe.partition_design(V2, RAND)
    bb = next(x for x in design if x["pairs"] == ["bb/cc"])
    p1 = got[("partition_split_vs_merged", "probe", "bvc_resonant", "partitions that split bb/cc",
              "partitions that merge bb/cc", "bestval")]
    assert p1["ln_ratio"] == pytest.approx(
        0.3 + (0.5 / len(bb["merged"]) if "RAND2_p1" in bb["merged"] else -0.5 / len(bb["split"])),
        abs=0.03)
    joint = [r for r in rows if r["contrast"] == "partition_joint" and r["checkpoint"] == "bestval"]
    tasks = {x["task"] for x in design}
    assert len(joint) == len(design)
    assert all(r["resid_dof"] == 5 * len(tasks) - (len(tasks) + 4 + len(design)) for r in joint)
    for r in joint:
        assert r["ln_ratio"] == pytest.approx(0.3, abs=0.03), r["probe_pairs"]
        assert r["ci95"][0] < math.exp(0.3) < r["ci95"][1]
        assert r["v_run_task"] >= 0 and 1 < r["task_dof"] < 5


def test_every_result_is_also_formed_between_the_checkpoints():
    # A8: the weight average shifts 17 classes by +0.2 and 188 by 0 in ln m, so the
    # 17-over-188 ratio moves by exp(0.2) between checkpoints and is not robust;
    # the 162-over-188 ratio does not move.
    vec, meta = _v2_replicates(effect=0.3, quality={}, shift={"R16_Q1": 0.2})
    rows = pe.ratios(vec, meta)["ratios"]
    got = {(r["contrast"], r["family"], r["task"], r["fine"], r["coarse"], r["checkpoint"]): r
           for r in rows}
    k = ("ladder", "probe", "bvc_resonant", "188", "17")
    b, w, d = (got[k + (t,)] for t in ("bestval", "wavg", "wavg/bestval"))
    assert d["ln_ratio"] == pytest.approx(w["ln_ratio"] - b["ln_ratio"], abs=1e-12)
    assert d["ln_ratio"] == pytest.approx(0.2, abs=0.02) and d["robust_to_checkpoint"] is False
    assert d["fine_models"][0] == "l188-s1@wavg/bestval" and d["ln_combined_se"] >= d["ln_test_se"]
    flat = got[("ladder", "probe", "bvc_resonant", "188", "162", "wavg/bestval")]
    assert abs(flat["ln_ratio"]) < 0.02
    # every result between the checkpoints says whether it is robust
    assert all(isinstance(r["robust_to_checkpoint"], bool) for r in rows
               if r["checkpoint"] == "wavg/bestval")
    # the unpaired, linear and partition contrasts are formed between checkpoints too
    for cid in ("family_unseen_vs_seen", "mass_cost_vs_162", "partition_split_vs_merged",
                "partition_joint"):
        assert any(r["contrast"] == cid and r["checkpoint"] == "wavg/bestval" for r in rows), cid
    # no rejection interval is formed for the ratio of two checkpoints
    assert "wavg/bestval" not in {x["checkpoint"] for x in pe.ratios(
        *_eps_cell())["rejections"]}


def _eps_cell():
    v, m = {}, {}
    for k in (1, 2):
        for arm in ("l188", "r16q1"):
            for tag in ("bestval", "wavg"):
                key = f"probe|t|linear|{arm}-s{k}@{tag}|eps_b@0.90"
                v[key] = np.full(5, 0.01 * k)
                m[key] = {"jets": "a", "n_bkg": 1000, "k_pass": 10.0 * k}
    return v, m


def test_the_comparison_between_checkpoints_checks_streams_up_to_the_later(tmp_path):
    # run 3 of 17 classes diverges at epoch 75: after both best-validation epochs
    # (40, 45), inside the weight average (70-79). It pairs at bestval only.
    rng = np.random.default_rng(1)
    v, m = {}, {}
    for k, div in ((1, None), (2, None), (3, 75)):
        _stream(tmp_path / "runs" / f"mtx-l188-s{k}", [f"x{e}" for e in range(80)], 40)
        _stream(tmp_path / "runs" / f"mtx-r16q1-s{k}",
                [f"x{e}" if div is None or e < div else f"y{e}" for e in range(80)], 45)
        for arm, x in (("l188", 0.1), ("r16q1", 0.2)):
            for tag in ("bestval", "wavg"):
                key = f"probe|t|linear|{arm}-s{k}@{tag}|1-auc"
                v[key], m[key] = x * np.exp(rng.normal(size=21) * 0.01), {"jets": "a"}
    rows = pe.ratios(v, m, tmp_path / "runs")["ratios"]
    by = {r["checkpoint"]: r for r in rows if r["contrast"] == "ladder"}
    assert by["bestval"]["n_runs"] == 3 and "excluded_pairs" in by["bestval"]
    assert by["bestval"]["excluded_pairs"] == []
    for t in ("wavg", "wavg/bestval"):
        assert by[t]["n_runs"] == 2
        assert [(x["first_bad_epoch"], x["upto_epoch"]) for x in by[t]["excluded_pairs"]] == [(75, 79)]


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
    # A13: sigma_min of the family left out, 188 classes (five runs) against the
    # same vocabulary with the family left out (three runs), unpaired; values
    # alone (no resampling), so the error is the spread over runs.
    seen = {1: 1.30, 2: 1.40, 3: 1.35, 4: 1.25, 5: 1.45}
    unseen = {1: 2.0, 2: 2.3, 3: 4.9}
    models, ext = {}, tmp_path / "ext"
    for arm, per in (("l188", seen), ("l188lofo4p", unseen)):
        for k, s in per.items():
            models[f"{arm}-s{k}"] = {}
            for tag, f in (("bestval", 1.0), ("wavg", 1.02)):
                sha = hashlib.sha256(f"{arm}{k}{tag}".encode()).hexdigest()
                d = ext / f"mtx-{arm}-s{k}" / tag
                d.mkdir(parents=True)
                (d / "manifest.json").write_text(json.dumps(
                    {"run_dir": f"/r/mtx-{arm}-s{k}", "tag": tag, "checkpoint_sha256": sha}))
                models[f"{arm}-s{k}"][tag] = (sha, s * f, 1.2 if s > 4 else 3.0)
    _anomaly_file(tmp_path / "an.json", models)
    vec, meta, prov = pe.anomaly_replicates([tmp_path / "an.json"], pe.extraction_index(ext, V2))
    key = "anomaly|label_X_YY_bbbb|class_sum_matched|l188lofo4p-s3@bestval|sigma_min@2000"
    assert vec[key].tolist() == [4.9] and meta[key]["censored"]
    pe.save(tmp_path / "an.npz", vec, meta, prov, 0, None)
    vp, mp, pp = pe.probe_replicates([_probe_one(tmp_path)], b=5, seed=1)
    pe.save(tmp_path / "p.npz", vp, mp, pp, 5, 1)
    v, m, _ = pe.load([tmp_path / "an.npz", tmp_path / "p.npz"])   # B = 0 beside B = 5
    rows = [r for r in pe.ratios({k: x for k, x in v.items() if k.startswith("anomaly")},
                                 {k: x for k, x in m.items() if k.startswith("anomaly")})["ratios"]
            if r["contrast"] == "family_unseen_vs_seen"]
    r = next(x for x in rows if x["checkpoint"] == "bestval")
    a, b = np.log(list(seen.values())), np.log(list(unseen.values()))
    # each arm keeps its own spread (Welch): the three runs that leave the family
    # out vary far more than the parent's five
    va, vb = np.var(a, ddof=1) / 5, np.var(b, ddof=1) / 3
    assert r["ln_ratio"] == pytest.approx(b.mean() - a.mean())
    assert r["ln_combined_se"] == pytest.approx(math.sqrt(va + vb))
    assert r["dof"] == pytest.approx((va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 2))
    assert r["stream_pairing"].startswith("exempt") and r["censored_models"] == ["l188lofo4p-s3@bestval"]
    assert "not detected" in r["note"] and r["n_boot"] == 0
    d = next(x for x in rows if x["checkpoint"] == "wavg/bestval")
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
