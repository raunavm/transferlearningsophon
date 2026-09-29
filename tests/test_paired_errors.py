"""experiments/STATS/paired_errors.py end to end on synthetic probe outputs:
replicates from per-jet scores, every comparison formed from paired runs, the
random control paired draw k with run k, and a Poisson interval on every
rejection."""
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("paired_errors", ROOT / "experiments/STATS/paired_errors.py")
pe = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pe)


def test_model_names_parse_to_level_and_run():
    assert pe.parse_model("l162-s1b") == ("162", 1)
    assert pe.parse_model("r16q1mass-s4") == ("17+mass", 4)
    assert pe.parse_model("rand-d3-s3") == ("random", 3)
    assert pe.parse_model("rand-d1-s1b") == ("random", 1)
    with pytest.raises(SystemExit):
        pe.parse_model("mpm-s1")


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
    assert dd["ln_combined_se_with_proxy"] > dd["ln_test_se"]
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


def test_v2_pairs_are_refused_when_their_streams_differ(tmp_path):
    import hashlib as _h
    for name, rows in (("mtx-l188-s1", ["a", "b"]), ("mtx-r16q1-s1", ["a", "c"])):
        d = tmp_path / "runs" / name / "stream"
        d.mkdir(parents=True)
        for e, r in enumerate(rows):
            (d / f"epoch-{e:03d}.json").write_text(json.dumps(
                {"run": name, "epoch": e, "seed_data": 1, "seed_dropout": 1, "files_sha256": "f",
                 "rows_sha256": r, "sha256": _h.sha256(("f" + r).encode()).hexdigest(),
                 "n_jets": 1}))
    v = {"probe|t|linear|l188-s1|1-auc": np.full(3, 0.1),
         "probe|t|linear|r16q1-s1|1-auc": np.full(3, 0.2)}
    m = {k: {"jets": "a"} for k in v}
    with pytest.raises(SystemExit, match="not a pair"):
        pe.ratios(v, m, tmp_path / "runs")


def test_reproduction_reports_the_largest_auc_difference(tmp_path):
    _probe_dir(tmp_path / "s1", {"l188-s1": 2.0}, 1)
    J = json.loads((tmp_path / "s1" / "probe_results.json").read_text())
    J["tasks"]["bvc_resonant"]["arms"]["l188-s1"]["mlp"]["auc"] += 1e-4
    (tmp_path / "ref.json").write_text(json.dumps(J))
    r = pe.reproduction([tmp_path / "s1"], {str(tmp_path / "s1"): str(tmp_path / "ref.json")})
    w = r[str(tmp_path / "s1")]["max_abs_dauc"]
    assert w["linear"] == 0.0 and w["mlp"] == pytest.approx(1e-4)
