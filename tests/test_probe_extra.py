"""The probes added 2026-11-07 (PI): coarse label sets (probe_bal.py --coarse), the QCD heavy-flavor
rejection of the X->bb discriminant (qcd_hf_rejection.py), bestval against best70
(checkpoint_identity.py), the released Sophon model (probe_released.py), and their jobs."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pb = _load("probe_bal", "experiments/EVAL/probe_bal.py")
qh = _load("qcd_hf_rejection", "experiments/EVAL/qcd_hf_rejection.py")
BJ = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")


def test_the_coarse_label_sets_map_every_native_class_and_qcd_to_one_group():
    luts = {t: pb.coarse_lut(t) for t in pb.COARSE_MAPS}
    assert {t: int(l.max()) + 1 for t, l in luts.items()} == {"R42_Q1": 43, "R16_Q1": 17, "FLAV_F0": 17}
    for l in luts.values():
        assert l.shape == (188,) and np.unique(l[161:]).size == 1 and l[161] not in l[:161]
    # the hierarchy nests: every 43-class group lies inside one 17-class group
    assert all(np.unique(luts["R16_Q1"][luts["R42_Q1"] == g]).size == 1 for g in range(43))
    assert BJ.COARSE == tuple(pb.COARSE_MAPS)


def test_the_coarse_probe_scores_the_groups_of_a_separable_task(tmp_path, monkeypatch):
    lut = pb.coarse_lut("R16_Q1")
    r = np.random.default_rng(0)
    centres = r.normal(size=(17, 128)) * 8

    def split(n):
        y = r.integers(0, 188, n)
        return (centres[lut[y]] + r.normal(size=(n, 128))).astype(np.float32), y.astype(np.int64)
    data = {"train_N3000": split(3000), "val": split(800), "test": split(1500)}
    (tmp_path / "m" / "jc2bal" / "train_N3000").mkdir(parents=True)
    monkeypatch.setattr(pb, "load", lambda model, ds, sp, ck, root=pb.NEW_ROOT: (*data[sp], {}))
    monkeypatch.setattr(pb.lp, "C_GRID", (1.0,))
    save = pb.lp.saver(tmp_path / "probs", "m", "best70")
    cell = pb.coarse_cells("m", "best70", ["R16_Q1"], save, root=tmp_path)["R16_Q1"]["3000"]
    assert cell["n_classes"] == cell["n_classes_present"] == 17 and cell["n_jets"] == 1500
    assert cell["accuracy"] > 0.99 and cell["macro_auc_ovr"] > 0.999 and sum(cell["train_label_counts"]) == 3000
    z = np.load(tmp_path / "probs" / "m" / "best70" / "jc2bal_R16_Q1_N3000.npz")
    assert np.array_equal(z["y"], lut[data["test"][1]]) and z["p"].shape == (1500, 17)


def test_the_qcd_subclasses_split_the_27_qcd_classes_by_b_and_c_content():
    sub = qh.qcd_subclasses(qh.class_names())
    names = qh.class_names()
    assert {k: len(v) for k, v in sub.items()} == {"bb": 9, "b": 9, "c": 6, "light": 3}
    assert sorted(c for v in sub.values() for c in v) == list(range(161, 188))
    got = {k: {names[c].removeprefix("label_QCD_") for c in v} for k, v in sub.items()}
    assert {"bb", "bbccss"} <= got["bb"] and {"b", "bc", "bccss"} <= got["b"]
    assert got["c"] == {"ccss", "ccs", "cc", "css", "cs", "c"} and got["light"] == {"ss", "s", "light"}


def test_the_log_odds_is_the_logit_of_sophons_discriminant():
    r = np.random.default_rng(1)
    z = r.normal(size=(50, 6)) * 3
    classes = np.array([0, 3, 161, 162, 170, 187])
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    d = p[:, 0] / (p[:, 0] + p[:, 2:].sum(1))
    assert np.allclose(qh.log_odds(z, classes, 0, [161, 162, 170, 187]), np.log(d / (1 - d)))
    with pytest.raises(SystemExit, match="never saw"):
        qh.log_odds(z, classes, 0, [161, 186])


def test_unit_weights_reproduce_the_unweighted_rejection_and_weights_change_the_mix():
    probe = _load("probe", "experiments/EVAL/probe.py")
    r = np.random.default_rng(2)
    y = np.r_[np.ones(2000, np.int64), np.zeros(8000, np.int64)]
    s = r.normal(size=y.size) + 2.0 * y
    got = qh.weighted_rejection(y, s, np.ones(y.size), 0.3)
    assert got["eps_b"] == pytest.approx(probe.rejection_at(y, s, 0.3)[1])
    # half the background is easy; weighting it up raises the rejection
    s[2000:6000] -= 3.0
    w = np.r_[np.ones(6000), np.full(4000, 0.01)]
    assert qh.weighted_rejection(y, s, w, 0.3)["rejection"] > qh.weighted_rejection(y, s, np.ones(y.size), 0.3)["rejection"]


def test_a_cell_recomputes_the_saved_probabilities_and_refuses_other_jets(tmp_path):
    probe = _load("probe", "experiments/EVAL/probe.py")
    names = qh.class_names()
    r = np.random.default_rng(3)
    y = np.r_[np.zeros(400), r.integers(161, 188, 1600)].astype(np.int64)
    x = r.normal(size=(y.size, 128)).astype(np.float16)
    coef, icpt = r.normal(size=(188, 128)) * 0.1, r.normal(size=188)
    mean, scale = x.astype(np.float64).mean(0), x.astype(np.float64).std(0)
    lg = ((x.astype(np.float64) - mean) / scale) @ coef.T + icpt
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    probs = tmp_path / "jc2bal_N1000000.npz"
    np.savez(probs, rows=np.arange(y.size), y=y.astype(np.int16), coef=coef, intercept=icpt, classes=np.arange(188),
             scaler_mean=mean, scaler_scale=scale, C=np.float64(1), p=p.astype(np.float16))
    feats = tmp_path / "test"
    feats.mkdir()
    np.save(feats / "features.npy", x)
    np.save(feats / "label188.npy", y.astype(np.int16))
    natural = np.ones(188)
    c = qh.cell(probs, feats, names, natural, probe)
    assert set(c["background"]) == {"bb", "b", "c", "light", "qcd_all"} and c["max_abs_p_diff"] < qh.P_TOL
    b = c["background"]["qcd_all"]
    assert b["n_signal"] == 400 and b["n_bkg"] == 1600 and set(b["balanced"]) == set(b["natural"]) == {"r50", "r30"}
    assert qh.probe_file(tmp_path) == probs
    np.save(feats / "label188.npy", np.roll(y, 1).astype(np.int16))
    with pytest.raises(SystemExit, match="different test jets"):
        qh.cell(probs, feats, names, natural, probe)


def test_bestval_is_reported_as_best70s_file_where_the_rules_agree(tmp_path):
    ci = _load("checkpoint_identity", "experiments/EVAL/checkpoint_identity.py")
    for run, best in (("a", 74), ("b", 40)):
        d = tmp_path / run
        (d / "metrics").mkdir(parents=True)
        vals = {e: 0.1 for e in range(80)}
        vals[74], vals[best] = 0.8, 0.9 if best != 74 else 0.8
        for e, v in vals.items():
            (d / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps({"epoch": e, "selection": {"metric": "val.acc", "value": v}}))
            (d / f"net_epoch-{e}_state.pt").write_text("x")
        (d / "best_epoch.json").write_text(json.dumps({"epoch": best, "metric": "val.acc"}))
        (d / "best_window_epoch.json").write_text(json.dumps({"epoch": 74, "metric": "val.acc", "window": [70, 79]}))
    runs = tmp_path / "runs.txt"
    runs.write_text("# header\na\nb\n")
    out = tmp_path / "o" / "identity.json"
    assert ci.main(["--runs", str(runs), "--root", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["n_runs"] == 2 and doc["n_same_file"] == 1
    assert doc["runs"]["a"] == {"best70": "net_epoch-74_state.pt", "bestval": "net_epoch-74_state.pt", "same_file": True}
    assert doc["runs"]["b"]["bestval"] == "net_epoch-40_state.pt" and not doc["runs"]["b"]["same_file"]


def test_the_released_probe_reads_every_split_under_its_own_root(tmp_path, monkeypatch):
    pr = _load("probe_released", "experiments/EVAL/probe_released.py")
    (tmp_path / "bal").mkdir()
    (tmp_path / "bal" / "manifest.json").write_text("{}")
    for n in (1000, 1000000):
        (tmp_path / "feat" / "s" / "jc2bal" / f"train_N{n}").mkdir(parents=True)
    monkeypatch.setattr(pr.pb, "BAL", tmp_path / "bal")
    seen = {}
    monkeypatch.setattr(pr.pb, "jc2bal_cells", lambda m, ck, save, sizes=None, root=None: seen.setdefault("jc2bal", (sizes, root)) and {"1000": {}})
    monkeypatch.setattr(pr.pb, "old_cells", lambda m, ck, h, save, root=None: seen.setdefault("old", root) and {"jc1": {}, "top": {}, "qg": {}, "qg_herwig": {}})
    monkeypatch.setattr(pr.pb, "coarse_cells", lambda m, ck, t, save, root=None: seen.setdefault("coarse", (tuple(t), root)) and {"R16_Q1": {}})
    assert pr.main(["--model", "s", "--checkpoints", "released", "--root", str(tmp_path / "feat"), "--coarse", "R16_Q1",
                    "--out", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out" / "s.json").read_text())
    assert seen == {"jc2bal": ([1000, 1000000], tmp_path / "feat"), "old": tmp_path / "feat",
                    "coarse": (("R16_Q1",), tmp_path / "feat")}
    assert set(doc["cells"]["released"]) == {"jc2bal", "jc1", "top", "qg", "qg_herwig", "jc2bal_coarse"}


def test_the_new_jobs_write_new_folders_and_pin_the_released_checkpoint():
    import yaml
    paper = [l for l in BJ.PAPER_RUNS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for spec, name, gpu in ((BJ.coarse_spec("t", 5), "jc2bal-coarse-raunav", True), (BJ.qcdhf_spec("t"), "jc2bal-qcdhf-raunav", False),
                            (BJ.ckptid_spec("t"), "ckpt-identity-raunav", False), (BJ.sophon_spec("t"), "sophon-probe-raunav", True)):
        d = yaml.safe_load(spec[1])
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert spec[0] == d["metadata"]["name"] == name and "raunav" in name and "configmap" not in spec[1].lower()
        assert ("nvidia.com/gpu" in c["resources"]["limits"]) == gpu and "--branch t " in c["args"][0]
    a = yaml.safe_load(BJ.coarse_spec("t", 5)[1])["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "--coarse R42_Q1 R16_Q1 FLAV_F0" in a and f"--out {BJ.LP_BAL}/fits_v4_coarse " in a
    assert f"--probs-out {BJ.LP_BAL}/fits_v4_coarse_probs" in a and 'MODEL=init-s1; CKPTS="init init_bn";;' in a
    a = yaml.safe_load(BJ.qcdhf_spec("t")[1])["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert a.count("qcd_hf_rejection.py") == len(BJ.models()) == len(paper) + 3 and f"--out {BJ.LP_BAL}/fits_v4_qcdhf" in a
    assert "--model init-s2 --checkpoints init init_bn " in a
    a = yaml.safe_load(BJ.sophon_spec("t")[1])["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert BJ.SOPHON_SHA256 in a and "--checkpoints /workspace/sophon/released.pt" in a and "-le 85 ]" in a
    assert a.count("\nextract ") == 6 + 6 + 6 + 7 and "extract jc1 test configs/finetune/JetClassI_sophon_noweight.yaml 0 ${TEST1}" in a
    assert "extract qg herwig configs/finetune/EnergyFlowQG.yaml 0 /data/finetune/qg_herwig/" in a
    assert "OUT=/data/results/eval/v2_linprobe_sophon/sophon-released/$1/$2" in a and "v2_linprobe_bal/" not in a.split("extract ()")[1]
