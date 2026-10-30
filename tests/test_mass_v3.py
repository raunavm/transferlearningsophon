"""The v3 mass analyses (plan 2026-10-29, A6): mass_resolution.py's uncorrected reference,
median centring, the mass head on the raw target and the binned response, and
mass_sculpting.py's QCD mass distribution under cuts on 1 - p_QCD."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
ms = _load("mass_sculpting", "experiments/EVAL/mass_sculpting.py")
an = _load("anomaly", "experiments/EVAL/anomaly.py")


# ------------------------------------------------------------- mass_resolution.py

def test_the_class_median_comes_from_the_training_rows_alone():
    y = np.array([1.0, 2.0, 30.0, 5.0, 5.0, 100.0])
    lab = np.array([0, 0, 0, 3, 3, 3])
    med = mr.class_median(y, lab, np.array([0, 1, 2, 3, 4]))
    assert med[0] == 2.0 and med[3] == 5.0, "the median, not the mean, and row 5 is not training"
    assert np.isnan(med[1]) and med.shape == (188,)


def test_the_response_is_the_exponentiated_median_residual():
    r = mr.response(np.random.default_rng(0).normal(0.1, 0.05, 20_000))
    assert r["median_response"] == pytest.approx(np.exp(r["median"])) == pytest.approx(np.exp(0.1), rel=0.01)
    assert r["sigma_eff"] == pytest.approx(0.05, rel=0.05)


def test_binned_groups_and_bins():
    rng = np.random.default_rng(1)
    n = 6000
    lab = rng.choice([0, 15, 161, 187], n)
    res = rng.normal(0, np.where(lab >= 161, 0.3, 0.1))
    obs = {"jet_pt": rng.uniform(200, 2500, n), "genjet_sdmass": rng.uniform(20, 500, n)}
    b = mr.binned(res, lab, obs)
    assert b["group"]["resonant"]["n"] == int((lab < 161).sum())
    assert b["group"]["qcd"]["sigma_eff"] == pytest.approx(0.3, rel=0.1)
    assert b["group"]["resonant"]["sigma_eff"] == pytest.approx(0.1, rel=0.1)
    names = mr.topology_names()
    assert names.shape == (188,) and len(set(names)) == 17 and set(names[161:]) == {"QCD_ALL"}
    assert set(b["topology_R16_Q1"]) == {names[0], names[15], "QCD_ALL"}
    assert list(b["jet_pt"]) == ["200-400", "400-700", "700-1000", "1000-2500"]
    assert list(b["genjet_sdmass"]) == ["20-50", "50-100", "100-150", "150-250", "250-500"]
    assert sum(c["n"] for c in b["jet_pt"].values()) == n
    pt = obs["jet_pt"]
    assert b["jet_pt"]["400-700"]["n"] == int(((pt >= 400) & (pt < 700)).sum())
    # a bin of fewer than two jets carries only its count; an absent variable is left out
    one = mr.binned(res[:1], lab[:1], {"jet_pt": np.array([300.0])})
    assert one["jet_pt"]["200-400"] == {"n": 1} and "genjet_sdmass" not in one


def _head_dir(d: pathlib.Path, n=4000, mass=True, seed=2, rows=None):
    """An extract_v2.py --keep-head-outputs checkpoint: a right-skewed log mass-ratio (its
    class mean above its class median) and a mass head that sees through half of it."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    lab = rng.choice([0, 5, 161, 170], n).astype(np.int16)
    sd = rng.uniform(30, 400, n)
    y = rng.normal(np.where(lab >= 161, -0.2, 0.05), 0.1) + rng.exponential(0.3, n) - 0.3
    gen = sd * np.exp(y)
    gen[:40] = 0.0                                       # unmatched
    z = {"rows": np.arange(n) if rows is None else rows, "label188": lab,
         "logits": rng.normal(size=(n, 17)).astype(np.float16),
         "jet_pt": rng.uniform(200, 2500, n).astype(np.float32), "jet_eta": np.zeros(n, np.float32),
         "jet_sdmass": sd.astype(np.float32), "genjet_sdmass": gen.astype(np.float32),
         "genjet_pt": np.ones(n, np.float32)}
    if mass:
        z["mass"] = (0.5 * y + rng.normal(0, 0.05, n)).astype(np.float32)
    np.savez(d / mr.HEAD_FILE, **z)
    (d / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "ab" * 32, "rung": "R16_Q1",
                                                 "num_classes": 17, "run_dir": "/x", "tag": "best70"}))
    return lab


def _feature_cache(d: pathlib.Path, lab: np.ndarray, extra=300):
    """A linear-probe cache: every row of a stream a few jets longer than the observers'."""
    d.mkdir(parents=True)
    n = lab.size + extra
    full = np.concatenate([lab, np.zeros(extra, np.int16)])
    np.save(d / "features.npy", np.random.default_rng(4).normal(size=(n, 8)).astype(np.float16))
    np.save(d / "rows.npy", np.arange(n))
    np.save(d / "label188.npy", full)
    (d / "manifest.json").write_text(json.dumps({"prefix_features": 2_000_000_000, "n_stream": n,
                                                 "checkpoint_sha256": "cd" * 32}))


def test_the_uncorrected_mass_and_the_head_are_scored_on_the_probes_test_jets(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    lab = _head_dir(tmp_path / "head")
    _feature_cache(tmp_path / "feat", lab)
    out = tmp_path / "m"
    assert mr.main(["--observers", str(tmp_path / "head"), "--features", f"A={tmp_path / 'feat'}",
                    "--head", f"best70={tmp_path / 'head'}", "--out", str(out), "--save-residuals"]) == 0
    res = json.loads((out / mr.OUT_JSON).read_text())
    assert not (out / "mass_resolution.json").exists()
    assert not (out / (mr.OUT_JSON + ".tmp")).exists()      # written whole, then renamed
    z = np.load(tmp_path / "head" / mr.HEAD_FILE)
    gen, sd = z["genjet_sdmass"].astype(np.float64), z["jet_sdmass"].astype(np.float64)
    assert res["n_jets_valid"] == 3960
    rz = np.load(out / mr.OUT_RESIDUALS)
    te = rz["rows"]
    y = np.log(gen[te] / sd[te])
    assert np.array_equal(rz["label188"], z["label188"][te])
    # the uncorrected mass: y_pred = 0 on the raw target, the same test jets as every probe
    assert res["uncorrected"]["sigma_eff"] == pytest.approx(mr.sigma_eff(-y)[0])
    assert res["uncorrected"]["n"] == res["heads"]["best70"]["n"] == res["arms"]["A"]["mlp"]["n"] == te.size
    # the head on the raw target: half of y seen, so its residual is -y/2 and it beats no correction
    h = res["heads"]["best70"]
    assert np.allclose(rz["head|best70"], z["mass"][te] - y)
    assert h["sigma_eff"] == pytest.approx(mr.sigma_eff(rz["head|best70"])[0])
    assert h["sigma_eff"] < res["uncorrected"]["sigma_eff"]
    assert h["median_response"] == pytest.approx(np.exp(np.median(z["mass"][te] - y)))
    assert h["provenance"]["cache"] == "head" and set(h["binned"]) == {"group", "topology_R16_Q1",
                                                                       "jet_pt", "genjet_sdmass"}
    # the frozen probes: median-centred by default, the mean-centred fit beside them
    arm = res["arms"]["A"]
    assert set(arm["mean_centred"]) == {"ridge", "mlp", "target"} and set(arm["binned"]) == {"mlp"}
    assert abs(arm["target"]["median"]) < abs(arm["mean_centred"]["target"]["median"])
    assert arm["binned"]["mlp"]["group"]["qcd"]["n"] == int((z["label188"][te] >= 161).sum())
    assert arm["provenance"]["label188_sha256"] == res["row_alignment_sha256"]
    # an existing result is never overwritten
    before = (out / mr.OUT_JSON).read_bytes()
    assert mr.main(["--observers", str(tmp_path / "head"), "--out", str(out)]) == 0
    assert (out / mr.OUT_JSON).read_bytes() == before


def test_a_head_without_a_mass_output_or_on_other_jets_is_refused(tmp_path):
    lab = _head_dir(tmp_path / "head")
    _head_dir(tmp_path / "plain", mass=False)
    with pytest.raises(SystemExit, match="no mass output"):
        mr.load_head("x", tmp_path / "plain", lab.size, hashlib.sha256(lab.tobytes()).hexdigest())
    _head_dir(tmp_path / "other", seed=9)
    with pytest.raises(SystemExit, match="are not the observers'"):
        mr.load_head("x", tmp_path / "other", lab.size, hashlib.sha256(lab.tobytes()).hexdigest())
    m, prov = mr.load_head("x", tmp_path / "head", lab.size, hashlib.sha256(lab.tobytes()).hexdigest())
    assert m.shape == (lab.size,) and prov["n_jets"] == lab.size


def test_observers_from_head_outputs_must_be_the_stream_in_order(tmp_path):
    _head_dir(tmp_path / "bad", rows=np.arange(4000)[::-1])
    with pytest.raises(SystemExit, match="not the stream's, in order"):
        mr.load_observers(tmp_path / "bad")
    lab = _head_dir(tmp_path / "ok")
    obs = mr.load_observers(tmp_path / "ok")
    assert obs["n"] == 4000 and obs["n_valid"] == 3960
    assert obs["label188_sha256"] == hashlib.sha256(lab.tobytes()).hexdigest()
    assert set(obs["obs"]) == {"jet_pt", "jet_eta", "jet_sdmass", "genjet_sdmass", "genjet_pt"}


# ------------------------------------------------------------- mass_sculpting.py

def test_the_score_is_one_minus_p_qcd_and_keeps_its_precision():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(500, 17)) * 3
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert np.allclose(ms.score(z, [16]), 1 - p[:, 16])
    assert np.allclose(ms.score(z, [3, 16]), 1 - p[:, [3, 16]].sum(1))
    sure = np.zeros((1, 17))
    sure[0, 16] = 40.0                                   # p_QCD = 1 to float64 precision
    assert ms.score(sure, [16])[0] == pytest.approx(16 * np.exp(-40.0), rel=1e-9)


def test_the_divergence_is_zero_for_one_shape_and_one_for_disjoint_ones():
    p = np.array([0.2, 0.3, 0.5, 0.0])
    assert ms.jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert ms.jsd(np.array([1.0, 0, 0, 0]), np.array([0, 0, 0.5, 0.5])) == pytest.approx(1.0)
    q = np.array([0.4, 0.1, 0.1, 0.4])
    assert ms.jsd(p, q) == pytest.approx(ms.jsd(q, p)) and 0 < ms.jsd(p, q) < 1


def test_a_score_that_follows_the_mass_sculpts_and_one_that_does_not_does_not():
    rng = np.random.default_rng(6)
    msd = rng.uniform(20, 500, 200_000)
    flat = ms.sculpting(rng.uniform(size=msd.size), msd)
    tied = ms.sculpting(msd / 500, msd)
    for eff in ms.WORKING_POINTS:
        f, t = flat["working_points"][str(eff)], tied["working_points"][str(eff)]
        assert f["n_pass"] == round(eff * msd.size) and f["eff_realised"] == pytest.approx(eff)
        assert f["jsd"] < 0.01 < 0.5 < t["jsd"]
        assert len(f["hist_after"]) == len(ms.BINS) - 1 == 24
        assert sum(f["hist_after"]) == pytest.approx(1.0)
    assert len(flat["hist_before"]) == 24


def _sculpt_dir(d: pathlib.Path, rung: str, k: int, n=3000):
    rng = np.random.default_rng(7)
    d.mkdir(parents=True)
    lab = rng.choice([0, 15, 161, 175, 187], n).astype(np.int16)
    np.savez(d / ms.HEAD_FILE, rows=np.arange(n), label188=lab,
             logits=rng.normal(size=(n, k)).astype(np.float16),
             jet_sdmass=rng.uniform(20, 500, n).astype(np.float32))
    (d / "manifest.json").write_text(json.dumps({"rung": rung, "num_classes": k, "run_dir": "/r",
                                                 "tag": "best70", "checkpoint_sha256": "ef" * 32,
                                                 "head_outputs": {"sha256": "01" * 32}}))
    return lab


@pytest.mark.parametrize("rung,k", [("R16_Q1", 17), ("L162", 162), ("R42_Q1", 43), ("L188", 188)])
def test_sculpting_reads_qcd_jets_and_the_label_sets_qcd_nodes(tmp_path, rung, k):
    lab = _sculpt_dir(tmp_path / "h", rung, k)
    out = tmp_path / "s.json"
    assert ms.main(["--head", str(tmp_path / "h"), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    want = sorted(an.node_roles(rung)[2])
    assert doc["qcd_nodes"] == want and len(want) == (27 if rung == "L188" else 1)
    assert doc["n_qcd"] == int((lab >= 161).sum()) and doc["n_jets"] == lab.size
    for w in doc["working_points"].values():
        assert 0 < w["n_pass"] <= doc["n_qcd"] and 0 <= w["jsd"] <= 1
    before = out.read_bytes()
    assert ms.main(["--head", str(tmp_path / "h"), "--out", str(out)]) == 0
    assert out.read_bytes() == before


def test_a_label_set_outside_the_tree_is_refused(tmp_path):
    _sculpt_dir(tmp_path / "h", "none", 17)
    with pytest.raises(SystemExit, match="no QCD node"):
        ms.main(["--head", str(tmp_path / "h"), "--out", str(tmp_path / "s.json")])
