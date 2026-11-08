"""experiments/FT/roc_points.py: the per-class ROC curves of the paper's ROC figures and their
20-background-jet cut (ROC redesign, 2026-11-02)."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RP = _load("roc_points", "experiments/FT/roc_points.py")
V4 = _load("mlst_v4", "experiments/FIGS/mlst_v4.py")


def _src(tmp_path):
    f = tmp_path / "scores.bin"
    f.write_bytes(b"x")
    return f


def test_the_grid_and_the_cut_constant():
    g = RP.EPS_GRID
    assert len(g) == 207 and g[0] == 0.001 and g[9] == 0.010 and g[10] == 0.015 and g[-1] == 0.995
    assert all(b > a for a, b in zip(g, g[1:])) and {0.3, 0.5} <= set(g)
    assert RP.MIN_BKG_PASS == V4.JC1_MIN_BKG == 20
    assert RP.JC1_SIGNALS == tuple(c for c, _ in V4.JC1_COLS)
    assert RP.FT_TASK["jc1_ft_ovr"] == "jc1"


def test_rejection_above_the_cut_equals_ft_curves(tmp_path):
    """Above the cut the rejection is uncapped there too (<= N_B/20 < N_B), so the two agree."""
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(5000, int), np.zeros(5000, int)]
    s = rng.normal(size=y.size) + 1.5 * y
    c = RP.curve(y, s, ["a"], ["b"], "d", 1, _src(tmp_path))
    old = RP.FC.roc_points(y, s, RP.EPS_GRID)["rejection"]
    keep = [i for i, e in enumerate(RP.EPS_GRID) if e >= c["cut"]["eps_s"]]
    assert keep and keep[0] < RP.EPS_GRID.index(0.3)
    for i in keep:
        assert c["rejection"][i] == pytest.approx(old[i], rel=1e-12)
        assert c["n_bkg_pass"][i] >= RP.MIN_BKG_PASS
    assert c["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert c["n_signal"] == c["n_background"] == 5000 and c["cut"]["n_bkg_pass"] == RP.MIN_BKG_PASS


def test_tied_background_jets_at_the_cut(tmp_path):
    """10 background jets above every signal jet, 15 tied at 1.5 inside the signal range: the cut is
    the vertex where 25 pass, at eps_S 0.5, and every grid point from there has >= 20 behind it."""
    sig = np.linspace(1.0, 2.0, 100)
    bkg = np.r_[np.full(10, 3.0), np.full(15, 1.5), np.zeros(975)]
    y = np.r_[np.ones(100, int), np.zeros(1000, int)]
    c = RP.curve(y, np.r_[sig, bkg], ["a"], ["b"], "d", 1, _src(tmp_path))
    assert c["cut"] == pytest.approx({"eps_s": 0.5, "eps_b": 0.025, "rejection": 40.0, "n_bkg_pass": 25})
    for e, n, r in zip(RP.EPS_GRID, c["n_bkg_pass"], c["rejection"]):
        if e >= 0.5:
            assert n >= RP.MIN_BKG_PASS and r <= 1000 / RP.MIN_BKG_PASS
    assert c["n_bkg_pass_full_eff"] == 25


def test_a_completely_separated_curve_has_no_cut(tmp_path):
    y = np.r_[np.ones(300, int), np.zeros(5000, int)]
    s = np.r_[np.linspace(1, 2, 300), np.linspace(0, 0.5, 5000)]
    c = RP.curve(y, s, ["a"], ["b"], "d", 1, _src(tmp_path))
    assert c["cut"] is None and c["auc"] == pytest.approx(1.0) and c["n_bkg_pass_full_eff"] == 0
    assert set(c["rejection"]) == {None}                     # eps_B = 0 everywhere on the grid
    # one outlier background jet above the lowest signal jet: still no point with 20 behind it
    s[300] = 1.05
    assert RP.curve(y, s, ["a"], ["b"], "d", 1, _src(tmp_path))["cut"] is None


def _jc1_rows():
    """Columns in JetClass order (QCD, Hbb, Hcc, ...). Hand values on D = p_Hbb/(p_Hbb+p_QCD):
    S1 0.5, S2 1/3, B1 4/9, B2 1/9 -> AUC 3/4 (raw p_Hbb gives 3/8); the Hcc jet with p_Hbb 0.9 and
    the other classes' jets are not in the Hbb curve."""
    # (class, p_QCD, p_Hbb, p_Hcc): S1, S2, B1, B2, the Hcc jet
    rows = ((1, 0.1, 0.1, 0.8), (1, 0.6, 0.3, 0.1), (0, 0.5, 0.4, 0.1), (0, 0.8, 0.1, 0.1), (2, 0.05, 0.9, 0.05))
    p, y = [], []
    for c, q, b, cc in rows:
        r = np.zeros(10)
        r[:3] = q, b, cc
        p.append(r)
        y.append(c)
    for c in range(3, 10):                    # one jet of every other signal class
        p.append(np.full(10, 0.1))
        y.append(c)
    return np.array(p), np.array(y)


def test_jetclass_probe_discriminant_and_files(tmp_path):
    p, y = _jc1_rows()
    d = tmp_path / "probs" / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc1_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float16))
    RP.main(["probes", "--family", "jc1_probe", "--probs-root", str(tmp_path / "probs"), "--files", "jc1_N1000000",
             "--out-dir", str(tmp_path / "jc1_probe"), "--allow-out", str(tmp_path)])
    doc = json.loads((tmp_path / "jc1_probe" / "mtx-l188-s1.json").read_text())
    assert list(doc["curves"]) == list(RP.JC1_SIGNALS)
    hbb = doc["curves"]["label_Hbb"]
    assert hbb["auc"] == pytest.approx(0.75) and (hbb["n_signal"], hbb["n_background"]) == (2, 2)
    assert hbb["signal"] == ["label_Hbb"] and hbb["background"] == ["label_QCD"] and hbb["n_train"] == 1_000_000
    assert hbb["cut"] is None                                  # two background jets never reach 20
    assert doc["family"] == "jc1_probe" and doc["checkpoint"] == "best70" and doc["seed"] == 1
    assert doc["eps_grid"] == list(RP.EPS_GRID) and doc["min_bkg_pass"] == 20 and doc["method"] == "linear probe"


JC1_FILE_ORDER = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")   # pred.root columns


def _jc1_probs(n=4000, seed=4):
    """Random probabilities over the 10 JetClass columns, each jet boosted in its own class's column."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    z = rng.normal(size=(n, 10))
    z[np.arange(n), y] += 1.5
    p = np.exp(z)
    return p / p.sum(1, keepdims=True), y


def _assert_jc1_hand_aucs(curves, p, y):
    """Each of the nine signals against QCD on D = p_c/(p_c+p_QCD), c the class's own file column
    (Zqq and Wqq sit in the opposite order to the paper's)."""
    q = p / p.sum(1, keepdims=True)
    for s in RP.JC1_SIGNALS:
        c = JC1_FILE_ORDER.index(s.removeprefix("label_"))
        m = (y == c) | (y == 0)
        want = roc_auc_score(y[m] == c, q[m, c] / (q[m, c] + q[m, 0]))
        assert curves[s]["auc"] == pytest.approx(want, abs=1e-12), s


def test_jetclass_probe_maps_every_class_to_its_own_column(tmp_path):
    p, y = _jc1_probs()
    d = tmp_path / "probs" / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc1_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float16))
    RP.main(["probes", "--family", "jc1_probe", "--probs-root", str(tmp_path / "probs"), "--files", "jc1_N1000000",
             "--out-dir", str(tmp_path / "jc1_probe"), "--allow-out", str(tmp_path)])
    doc = json.loads((tmp_path / "jc1_probe" / "mtx-l188-s1.json").read_text())
    _assert_jc1_hand_aucs(doc["curves"], p.astype(np.float16).astype(np.float64), y)


def test_jetclass_fine_tuning_reads_pred_root_in_file_order(tmp_path, monkeypatch):
    p, y = _jc1_probs(seed=6)
    c = tmp_path / "leg2" / "l188-s1" / "N1000000" / "s1"
    c.mkdir(parents=True)
    (c / "pred.root").touch()                 # hashed as the curves' source; its contents come from the patch
    (c / "DONE").touch()
    monkeypatch.setattr(RP.FC.LEG2, "read_pred_root", lambda path: (list(JC1_FILE_ORDER), np.eye(10)[y], p))
    out = tmp_path / "jc1_ft"
    RP.main(["ft", "--family", "jc1_ft", "--tree", str(tmp_path / "leg2"), "--task", "jc1", "--n", "1000000",
             "--out-dir", str(out), "--allow-out", str(tmp_path)])
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    assert list(doc["curves"]) == list(RP.JC1_SIGNALS) and doc["method"] == "fine-tuning"
    assert doc["curves"]["label_Wqq"]["source"]["file"] == str(c / "pred.root")
    _assert_jc1_hand_aucs(doc["curves"], p, y)


def _assert_jc1_ovr_hand_aucs(doc, p, y):
    """All ten classes, in the paper's order, each against all nine others on p_c renormalised over the
    ten classes, c the class's own file column."""
    assert list(doc["curves"]) == ["label_QCD", *RP.JC1_SIGNALS] and len(doc["curves"]) == 10
    q = p / p.sum(1, keepdims=True)
    for name, c in doc["curves"].items():
        col = JC1_FILE_ORDER.index(name.removeprefix("label_"))
        assert c["auc"] == pytest.approx(roc_auc_score(y == col, q[:, col]), abs=1e-12), name
        assert (c["n_signal"], c["n_background"]) == ((y == col).sum(), (y != col).sum())
        assert c["signal"] == [name] and c["background"] == ["every other class"]


def test_jetclass_probes_one_class_against_all_the_others(tmp_path):
    p, y = _jc1_probs(seed=7)
    # a jet-by-jet scale changes the rank of raw p_c, so only the renormalised score passes
    p = p * np.random.default_rng(8).uniform(0.2, 1.0, (y.size, 1))
    d = tmp_path / "probs" / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc1_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float32))
    RP.main(["probes", "--family", "jc1_probe_ovr", "--probs-root", str(tmp_path / "probs"), "--files", "jc1_N1000000",
             "--out-dir", str(tmp_path / "jc1_probe_ovr"), "--allow-out", str(tmp_path)])
    doc = json.loads((tmp_path / "jc1_probe_ovr" / "mtx-l188-s1.json").read_text())
    assert doc["family"] == "jc1_probe_ovr" and doc["method"] == "linear probe" and doc["checkpoint"] == "best70"
    assert doc["curves"]["label_QCD"]["n_train"] == 1_000_000 and doc["eps_grid"] == list(RP.EPS_GRID)
    _assert_jc1_ovr_hand_aucs(doc, p.astype(np.float32).astype(np.float64), y)
    raw = roc_auc_score(y == 1, p[:, 1])
    assert doc["curves"]["label_Hbb"]["auc"] != pytest.approx(raw, abs=1e-6)


def test_jetclass_fine_tuning_one_class_against_all_the_others_reads_pred_root(tmp_path, monkeypatch):
    p, y = _jc1_probs(seed=9)
    c = tmp_path / "leg2" / "l188-s1" / "N1000000" / "s1"
    c.mkdir(parents=True)
    (c / "pred.root").touch()
    (c / "DONE").touch()
    monkeypatch.setattr(RP.FC.LEG2, "read_pred_root", lambda path: (list(JC1_FILE_ORDER), np.eye(10)[y], p))
    out = tmp_path / "jc1_ft_ovr"
    RP.main(["ft", "--family", "jc1_ft_ovr", "--tree", str(tmp_path / "leg2"), "--task", "jc1", "--n", "1000000",
             "--out-dir", str(out), "--allow-out", str(tmp_path)])
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    assert doc["method"] == "fine-tuning" and doc["curves"]["label_Wqq"]["source"]["file"] == str(c / "pred.root")
    _assert_jc1_ovr_hand_aucs(doc, p, y)
    # the signal-vs-QCD family is not touched by the switch
    out = tmp_path / "jc1_ft"
    RP.main(["ft", "--family", "jc1_ft", "--tree", str(tmp_path / "leg2"), "--task", "jc1", "--n", "1000000",
             "--out-dir", str(out), "--allow-out", str(tmp_path)])
    assert list(json.loads((out / "mtx-l188-s1.json").read_text())["curves"]) == list(RP.JC1_SIGNALS)


def test_a_separable_jetclass_class_has_auc_one_and_no_cut(tmp_path):
    rng = np.random.default_rng(10)
    y = rng.integers(0, 10, 3000)
    p = rng.uniform(0.01, 0.02, (y.size, 10))
    p[y == 9, 9] = 10.0              # label_Tbl above every other jet's score
    cv = RP.jc1_ovr_curves(p, y, RP.LP.JC1_CLASSES, 1, _src(tmp_path))
    assert cv["label_Tbl"]["auc"] == pytest.approx(1.0) and cv["label_Tbl"]["cut"] is None
    assert cv["label_QCD"]["auc"] < 0.9


def test_a_jetclass_file_with_other_columns_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="JetClass columns"):
        RP.jc1_ovr_curves(np.ones((4, 10)), np.arange(4), list(RP.LP.JC1_CLASSES)[::-1], 1, _src(tmp_path))


def test_untrained_trunks_are_written_per_checkpoint(tmp_path):
    p, y = _jc1_rows()
    for ck in ("init", "init_bn"):
        d = tmp_path / "probs" / "init-s2" / ck
        d.mkdir(parents=True)
        np.savez(d / "jc1_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float16))
    RP.main(["probes", "--family", "jc1_probe", "--probs-root", str(tmp_path / "probs"), "--files", "jc1_N1000000",
             "--models", "init-s2", "--out-dir", str(tmp_path / "jc1_probe"), "--allow-out", str(tmp_path)])
    assert sorted(f.name for f in (tmp_path / "jc1_probe").iterdir()) == ["init-s2.json", "initbn-s2.json"]
    assert json.loads((tmp_path / "jc1_probe" / "initbn-s2.json").read_text())["checkpoint"] == "init_bn"


def test_jetclass2_one_vs_rest_background_is_every_other_class(tmp_path):
    rng = np.random.default_rng(3)
    y = rng.choice([0, 1, 170], size=3000)
    p = rng.random((y.size, 188)) + 3 * np.eye(188)[y]
    d = tmp_path / "probs" / "mtx-r16q1-s3" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc2bal_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float16))
    RP.main(["probes", "--family", "jc2_ovr", "--probs-root", str(tmp_path / "probs"), "--files", "jc2bal_N1000000",
             "--out-dir", str(tmp_path / "jc2_ovr"), "--allow-out", str(tmp_path)])
    doc = json.loads((tmp_path / "jc2_ovr" / "mtx-r16q1-s3.json").read_text())
    assert sorted(doc["curves"], key=int) == ["0", "1", "170"] and doc["seed"] == 3
    c = doc["curves"]["0"]
    q = p.astype(np.float16).astype(np.float64)
    q /= q.sum(1, keepdims=True)
    assert c["n_background"] == int((y != 0).sum()) and c["n_signal"] == int((y == 0).sum())
    assert c["auc"] == pytest.approx(roc_auc_score(y == 0, q[:, 0]), abs=1e-12)
    assert c["signal"] == ["label_X_bb"] and c["background"] == ["every other class"]


def test_pairs_read_their_training_size_from_the_fit_and_the_sibling_root(tmp_path):
    y = np.r_[np.ones(50, int), np.zeros(50, int)]
    s = np.r_[np.linspace(0.6, 1, 50), np.linspace(0, 0.7, 50)]
    for root, pair, n in (("probs", "bb_vs_cc", 180_000), ("probs_sib", "ee_vs_mm", 9_575)):
        d = tmp_path / root / "mtx-l188-s1" / "best70"
        d.mkdir(parents=True)
        np.savez(d / f"jc2bal_pairs_{pair}.npz", rows=np.arange(y.size), y=y.astype(np.int16), s=s.astype(np.float32))
        fits = tmp_path / root.replace("probs", "fits")
        fits.mkdir()
        (fits / "mtx-l188-s1.json").write_text(json.dumps({"cells": {"best70": {"jc2bal_pairs": {pair: {"n_train": n}}}}}))
    RP.main(["probes", "--family", "pairs", "--probs-root", str(tmp_path / "probs"), str(tmp_path / "probs_sib"),
             "--fits-root", str(tmp_path / "fits"), str(tmp_path / "fits_sib"), "--files", "jc2bal_pairs_*",
             "--out-dir", str(tmp_path / "pairs"), "--allow-out", str(tmp_path)])
    cv = json.loads((tmp_path / "pairs" / "mtx-l188-s1.json").read_text())["curves"]
    assert cv["bb_vs_cc"]["n_train"] == 180_000 and cv["ee_vs_mm"]["n_train"] == 9_575
    assert cv["ee_vs_mm"]["signal"] == ["label_X_ee"] and cv["ee_vs_mm"]["background"] == ["label_X_mm"]
    assert cv["bb_vs_cc"]["auc"] == pytest.approx(roc_auc_score(y, s.astype(np.float32)))


def _pair_files(tmp_path, pairs):
    y = np.r_[np.ones(50, int), np.zeros(50, int)]
    s = np.r_[np.linspace(0.6, 1, 50), np.linspace(0, 0.7, 50)]
    d = tmp_path / "probs" / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fits").mkdir(exist_ok=True)
    for pair in pairs:
        np.savez(d / f"jc2bal_pairs_{pair}.npz", rows=np.arange(y.size), y=y.astype(np.int16), s=s.astype(np.float32))
    (tmp_path / "fits" / "mtx-l188-s1.json").write_text(json.dumps(
        {"cells": {"best70": {"jc2bal_pairs": {p: {"n_train": 1000} for p in pairs}}}}))


def _pairs_main(tmp_path, *files):
    return RP.main(["probes", "--family", "pairs", "--probs-root", str(tmp_path / "probs"), "--fits-root",
                    str(tmp_path / "fits"), "--files", *files, "--models", "mtx-l188-s1",
                    "--out-dir", str(tmp_path / "pairs"), "--allow-out", str(tmp_path)])


def test_a_named_pair_missing_from_every_root_halts_instead_of_being_dropped(tmp_path):
    _pair_files(tmp_path, ["bb_vs_cc"])
    with pytest.raises(SystemExit, match=r"no \['jc2bal_pairs_ee_vs_mm'\] of mtx-l188-s1 best70"):
        _pairs_main(tmp_path, "jc2bal_pairs_bb_vs_cc", "jc2bal_pairs_ee_vs_mm")
    assert not (tmp_path / "pairs" / "mtx-l188-s1.json").exists()


def test_an_existing_pair_file_without_a_curve_its_inputs_give_is_refused_not_skipped(tmp_path):
    _pair_files(tmp_path, ["bb_vs_cc"])
    _pairs_main(tmp_path, "jc2bal_pairs_bb_vs_cc")
    out = tmp_path / "pairs" / "mtx-l188-s1.json"
    before = out.read_text()
    _pairs_main(tmp_path, "jc2bal_pairs_bb_vs_cc")          # the same inputs: skipped
    _pair_files(tmp_path, ["bb_vs_cc", "ee_vs_mm"])
    with pytest.raises(SystemExit, match=r"exists without the curves \['ee_vs_mm'\]; move it aside"):
        _pairs_main(tmp_path, "jc2bal_pairs_bb_vs_cc", "jc2bal_pairs_ee_vs_mm")
    assert out.read_text() == before


def _top_cell(tree, init, seed, n=500, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    c = tree / init / "N1200000" / f"s{seed}"
    (c / "features").mkdir(parents=True)
    y = (rng.random(n) < 0.5).astype(np.int16)
    # both columns random: softmax(logits)[1] is then a monotone function of neither raw logit
    logits = np.c_[rng.normal(size=n), rng.normal(size=n) + 2 * y].astype(np.float32)
    np.save(c / "features" / "logits.npy", logits)
    np.save(c / "features" / "label188.npy", y)
    (c / "DONE").touch()
    return y, logits


def test_fine_tuning_names_one_file_per_run_and_from_scratch_seed(tmp_path):
    tree = tmp_path / "leg_top"
    y, logits = _top_cell(tree, "l188-s1", 1)
    _top_cell(tree, "scratch-v2", 1)
    _top_cell(tree, "scratch-v2", 2, rng_seed=1)
    out = tmp_path / "top_ft"
    args = ["ft", "--family", "top_ft", "--tree", str(tree), "--task", "top", "--n", "1200000",
            "--out-dir", str(out), "--allow-out", str(tmp_path)]
    RP.main(args)
    assert sorted(f.name for f in out.iterdir()) == ["mtx-l188-s1.json", "scratch-v2-s1.json", "scratch-v2-s2.json"]
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    assert list(doc["curves"]) == ["top"] and doc["method"] == "fine-tuning" and doc["checkpoint"] == "best70"
    assert doc["curves"]["top"]["signal"] == ["label_Top"] and doc["curves"]["top"]["n_train"] == 1_200_000
    top, s = doc["curves"]["top"], RP.BM.softmax(logits)[:, 1]
    assert top["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    i = next(i for i, e in enumerate(RP.EPS_GRID) if e >= top["cut"]["eps_s"])
    assert top["rejection"][i] == pytest.approx(RP.FC.roc_points(y, s, RP.EPS_GRID)["rejection"][i], rel=1e-12)
    assert json.loads((out / "scratch-v2-s2.json").read_text())["checkpoint"] is None
    _top_cell(tree, "l188-s1", 2)                               # a second fine-tuning of one run
    with pytest.raises(SystemExit, match="more than one fine-tuning"):
        RP.main(args)


def test_quark_gluon_scores_pythia_and_herwig_from_their_own_files(tmp_path):
    rng = np.random.default_rng(5)
    tree = tmp_path / "leg_qg"
    fd = tree / "l188-s1" / "N1600000" / "s1" / "features"
    fd.mkdir(parents=True)
    want = {}
    for key, f_logits, f_labels, n, shift in (("qg", "logits.npy", "label188.npy", 600, 2.0),
                                              ("qg_herwig", "logits_herwig.npy", "label_herwig.npy", 400, 0.7)):
        y = (rng.random(n) < 0.5).astype(np.int16)
        z = np.c_[rng.normal(size=n), rng.normal(size=n) + shift * y].astype(np.float32)
        np.save(fd / f_logits, z)
        np.save(fd / f_labels, y)
        want[key] = (f_logits, y.size, roc_auc_score(y, RP.BM.softmax(z)[:, 1]))
    (fd.parent / "DONE").touch()
    out = tmp_path / "qg_ft"
    RP.main(["ft", "--family", "qg_ft", "--tree", str(tree), "--task", "qg", "--n", "1600000",
             "--out-dir", str(out), "--allow-out", str(tmp_path)])
    cv = json.loads((out / "mtx-l188-s1.json").read_text())["curves"]
    assert list(cv) == ["qg", "qg_herwig"]
    for key, (f_logits, n, auc) in want.items():
        assert cv[key]["auc"] == pytest.approx(auc, abs=1e-12) and cv[key]["n_signal"] + cv[key]["n_background"] == n
        assert pathlib.Path(cv[key]["source"]["file"]).name == f_logits
        assert cv[key]["signal"] == ["label_quark"] and cv[key]["background"] == ["label_gluon"]


def test_an_existing_file_is_skipped_never_overwritten(tmp_path, capsys):
    tree = tmp_path / "leg_top"
    _top_cell(tree, "l188-s1", 1)
    out = tmp_path / "top_ft"
    out.mkdir()
    (out / "mtx-l188-s1.json").write_text("kept")
    RP.main(["ft", "--family", "top_ft", "--tree", str(tree), "--task", "top", "--n", "1200000", "--inits", "l188-s1",
             "--out-dir", str(out), "--allow-out", str(tmp_path)])
    assert (out / "mtx-l188-s1.json").read_text() == "kept" and "skip " in capsys.readouterr().out


def test_an_output_outside_the_curves_tree_is_refused(tmp_path):
    for out, allow in ((tmp_path / "top_ft", None), (tmp_path / "pairs", tmp_path), (tmp_path / "x" / "top_ft", tmp_path)):
        args = ["ft", "--family", "top_ft", "--tree", str(tmp_path), "--task", "top", "--n", "1", "--out-dir", str(out)]
        with pytest.raises(SystemExit, match="FATAL: --out-dir"):
            RP.main(args + (["--allow-out", str(allow)] if allow else []))
    assert RP.out_dir(RP.OUT_ROOT / "jc2_ovr", "jc2_ovr", None) == RP.OUT_ROOT.resolve() / "jc2_ovr"
    assert RP.out_dir(RP.OUT_ROOT / "pairs_sib2", "pairs", None) == RP.OUT_ROOT.resolve() / "pairs_sib2"
    with pytest.raises(SystemExit):
        RP.out_dir(RP.OUT_ROOT / "pairsx", "pairs", None)


def test_the_long_jetclass_job_writes_what_the_readers_prefer_once_every_cell_is_done():
    """scripts/build_gallery_data_job.py --mode long-jc1: every long-budget JetClass cell DONE first, then the
    curves of mlst_v4.JC1_LONG into <family>_long and ft_curves.py's table into a new file."""
    GD = _load("build_gallery_data_job", "scripts/build_gallery_data_job.py")
    fn, name, dirs = GD.MODES["long-jc1"]
    assert name == "gallery-data-long-jc1-raunav" and dirs.split() == [f"{f}_long" for f in V4.JC1_LONG]
    lines = [ln.strip() for ln in fn().splitlines()]
    runs = [r.removeprefix("mtx-") for r in V4.paper_models()]
    cells = lines[0].removeprefix("for c in ").split(";")[0].split()
    assert sorted(cells) == sorted([f"{GD.LONG_JC1_TREE}/{r}/N1000000/s1" for r in runs]
                                   + [f"{GD.LONG_JC1_TREE}/scratch/N1000000/s{k}" for k in (1, 2, 3)])
    assert len(cells) == 30 and 'exit 42' in lines[0]
    inits = " ".join(sorted(runs + ["scratch"]))
    for i, fam in enumerate(V4.JC1_LONG):
        call, done = lines[1 + 2 * i], lines[2 + 2 * i]
        assert call == (f"python3 experiments/FT/roc_points.py ft --family {fam} --tree {GD.LONG_JC1_TREE} --task jc1 "
                        f"--n 1000000 --inits {inits} --out-dir {GD.ROC}/{fam}_long")
        assert done == f"echo done > {GD.ROC}/{fam}_long/DONE"
        assert RP.out_dir(pathlib.Path(f"{GD.ROC}/{fam}_long"), fam, None).name == f"{fam}_long"
        assert RP.FT_TASK[fam] == "jc1"
    assert GD.LONG_JC1_FULL == "/data/results/ft_v2/curves/full_budget_long_leg2.json"
    assert lines[-1] == (f"[ -e {GD.LONG_JC1_FULL} ] || python3 experiments/FT/ft_curves.py cells --tree "
                         f"{GD.LONG_JC1_TREE} --task jc1 --n 1000000 --out {GD.LONG_JC1_FULL}")
    assert len(lines) == 6


def test_the_full_schedule_scratch_job_writes_new_families_the_readers_prefer_once_every_cell_is_done():
    """scripts/build_gallery_data_job.py --mode scratch-full (2026-11-07): the six from-scratch cells at 1e6 DONE
    first, then the families the old from-scratch cells fed, each into a new <family>_fulldecay, and ParT's
    per-class table into a new file; no output any other mode writes."""
    GD = _load("build_gallery_data_job", "scripts/build_gallery_data_job.py")
    bfb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    fn, name, dirs = GD.MODES["scratch-full"]
    fams = ("jc2_ft_ovr",) + V4.JC1_LONG
    assert name == "gallery-data-scratch-full-raunav" and dirs.split() == [f"{f}{V4.SCRATCH_FULL}" for f in fams]
    others = {d for m, (_, _, ds) in GD.MODES.items() if m != "scratch-full" for d in ds.split()}
    assert not others & set(dirs.split())
    lines = [ln.strip() for ln in fn().splitlines()]
    cells = lines[0].removeprefix("for c in ").split(";")[0].split()
    assert sorted(cells) == sorted(c["out"] for c in bfb.long_scratch_full_cells("jc2b", "1e-3")
                                   + bfb.long_scratch_full_cells("jc1", "1e-3"))
    assert len(cells) == 6 and "exit 42" in lines[0]
    trees = {"jc2_ft_ovr": (GD.FULL_JC2_TREE, "jc2b"), "jc1_ft": (GD.FULL_JC1_TREE, "jc1"), "jc1_ft_ovr": (GD.FULL_JC1_TREE, "jc1")}
    for i, fam in enumerate(fams):
        tree, task = trees[fam]
        assert tree.startswith(bfb.LONG_FULL_ROOT + "/") and RP.FT_TASK[fam] == task
        assert lines[1 + 2 * i] == (f"python3 experiments/FT/roc_points.py ft --family {fam} --tree {tree} --task {task} "
                                    f"--n 1000000 --inits scratch --out-dir {GD.ROC}/{fam}{V4.SCRATCH_FULL}")
        assert lines[2 + 2 * i] == f"echo done > {GD.ROC}/{fam}{V4.SCRATCH_FULL}/DONE"
        assert RP.out_dir(pathlib.Path(f"{GD.ROC}/{fam}{V4.SCRATCH_FULL}"), fam, None).name == f"{fam}{V4.SCRATCH_FULL}"
    assert GD.FULL_JC1 == "/data/results/ft_v2/curves/full_budget_long_fulldecay_leg2.json" != GD.LONG_JC1_FULL
    assert lines[-1] == (f"[ -e {GD.FULL_JC1} ] || python3 experiments/FT/ft_curves.py cells --tree "
                         f"{GD.FULL_JC1_TREE} --task jc1 --n 1000000 --out {GD.FULL_JC1}")
    assert len(lines) == 8
    spec = (ROOT / "experiments/FT/k8s/curves/job-gallery-data-scratch-full-raunav.yaml").read_text()
    assert "name: gallery-data-scratch-full-raunav" in spec and '--branch "mtx-s2.85"' in spec
    assert all(ln in spec for ln in lines)
