"""The v4 probes (experiments/EVAL/probe_bal.py): grid, stride, ParT's AUC and the hidden layer."""
import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_bal", ROOT / "experiments/EVAL/probe_bal.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def test_v4_grid_extends_two_decades_and_scores_every_jet():
    assert pb.lp.C_GRID == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    assert pb.lp.AUC_STRIDE == 1


def test_ovo_auc_is_sklearns_macro_one_vs_one():
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(0)
    y = r.integers(0, 4, 2000)
    p = r.random((2000, 4)) + np.eye(4)[y]
    assert pb.ovo_auc(p, y) == pytest.approx(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovo"))


def test_jetclass_ovo_auc_only_at_the_size_whose_probabilities_are_saved(tmp_path, monkeypatch):
    # probe_cells saves JetClass's probabilities at the largest size only; jc2bal-fit crashed reading
    # jc1_N1000.npz (2026-11-01)
    r = np.random.default_rng(2)
    y = r.integers(0, 10, 500)
    p = r.random((500, 10)) + np.eye(10)[y]
    d = tmp_path / "m" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc1_N1000000.npz", p=p.astype(np.float16), y=y)
    cells = {"jc1": {str(n): {} for n in (1000, 10000, 100000, 1000000)}, "top": {"1000": {}}}
    monkeypatch.setattr(pb.lp, "probe_cells", lambda *a, **k: cells)
    save = lambda *a, **k: None
    save.out = tmp_path
    out = pb.old_cells("m", "best70", None, save)
    assert out["jc1"]["1000000"]["macro_auc_ovo"] == pytest.approx(pb.ovo_auc(p.astype(np.float16).astype(np.float64), y))
    assert all("macro_auc_ovo" not in out["jc1"][n] for n in ("1000", "10000", "100000"))


def test_hidden_layer_is_relu_of_the_checkpoints_first_head_layer(tmp_path):
    r = np.random.default_rng(1)
    w, b = r.normal(size=(512, 128)), r.normal(size=512)
    sd = {"mod.fc.0.0.weight": torch.tensor(w), "mod.fc.0.0.bias": torch.tensor(b),
          "mod.fc.1.weight": torch.zeros(188, 512)}
    path = tmp_path / "ckpt.pt"
    torch.save(sd, path)
    x = r.normal(size=(10, 128)).astype(np.float32)
    h = pb.hidden(x, {"checkpoint": str(path)})
    assert h.shape == (10, 512) and (h >= 0).all()
    assert np.allclose(h, np.maximum(x.astype(np.float64) @ w.T + b, 0), atol=1e-4)


def test_logit_takes_inputs_wider_than_the_class_token():
    r = np.random.default_rng(2)
    x = r.normal(size=(400, 512))
    y = (x[:, 0] > 0).astype(np.int64)
    clf = pb.lp.Logit(np.unique(y), "cpu")
    clf.fit(x, y, 1.0)
    assert clf.coef_.shape == (1, 512) and (clf.predict_proba(x).argmax(1) == y).mean() > 0.9


def test_the_kinematic_models_are_probed_on_top_only_with_their_own_inputs():
    import yaml
    spec = importlib.util.spec_from_file_location("build_jc2bal_jobs", ROOT / "scripts/build_jc2bal_jobs.py")
    BJ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BJ)
    name, text = BJ.kin_spec("mtx-s2.29")
    d = yaml.safe_load(text)
    assert d["metadata"]["name"] == name == "jc2bal-kin-raunav" and d["spec"]["completions"] == 2
    a = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "configs/finetune/TopReference_kin.yaml" in a and "--old-only top" in a and "--mlp-datasets top" in a
    assert "0) MODEL=mtx-l188kin-s1; RUNG=L188; K=188;;" in a and "r16q1kin" not in a
    assert "${RUN_DIR}/DONE ] ||" in a and "/top_sub/train_N1200000_s1.parquet" not in a      # in the loop
    assert "for N in 1000 10000 100000 1200000; do" in a
    assert "TopReference_kin.yaml" in (ROOT / "configs/finetune/TopReference_kin.yaml").name


def test_every_job_that_loads_data_with_weaver_has_shared_memory():
    # the reference job's DataLoader worker died of a bus error in the default 64 MB /dev/shm (2026-11-01)
    import yaml
    spec = importlib.util.spec_from_file_location("build_jc2bal_jobs", ROOT / "scripts/build_jc2bal_jobs.py")
    BJ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BJ)
    for _, text in (BJ.ref_spec("t"), BJ.extract_spec("t", 5), BJ.fit_spec("t", 5, False), BJ.fit_spec("t", 5, True),
                    BJ.kin_spec("t")):
        pod = yaml.safe_load(text)["spec"]["template"]["spec"]
        assert {"name": "dshm", "mountPath": "/dev/shm"} in pod["containers"][0]["volumeMounts"]
        assert {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}} in pod["volumes"]


def test_a_fit_job_of_one_model_keeps_its_model_and_avoids_the_bad_nodes():
    # index 29 of jc2bal-fit failed on k8s-haosu-10 (CUDA launch failure, 2026-11-02): rerun alone
    import yaml
    spec = importlib.util.spec_from_file_location("build_jc2bal_jobs", ROOT / "scripts/build_jc2bal_jobs.py")
    BJ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BJ)
    whole = yaml.safe_load(BJ.fit_spec("t", 5, False)[1])
    name, text = BJ.fit_spec("t", 5, False, (29,), "jc2bal-fit2-raunav")
    one = yaml.safe_load(text)
    assert name == "jc2bal-fit2-raunav" and one["spec"]["completions"] == 1
    model = BJ.models()[29][0]
    script = one["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert f"  0) MODEL={model};" in script and "  1) MODEL=" not in script
    assert f"  29) MODEL={model};" in whole["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "k8s-haosu-10.sdsc.optiputer.net" in text


def test_the_sibling_pair_separates_a_separable_pair_and_saves_consistent_rows(tmp_path, monkeypatch):
    """X->ee vs X->mumu on the balanced sets: the other classes' jets are not used, the saved rows
    index the test extraction, y = 1 on the electron side, and a separable pair gives AUC 1."""
    probe = pb._load("probe", "experiments/EVAL/probe.py")
    r = np.random.default_rng(4)
    names = {0: "label_X_bb", 10: "label_X_ee", 11: "label_X_mm"}

    def split(n):
        y = r.choice([0, 10, 11], size=n)
        x = r.normal(size=(n, 128)).astype(np.float32)
        x[:, 0] += 6.0 * (y == 10) - 6.0 * (y == 11)
        return x, y.astype(np.int64)
    data = {"train_N1000000": split(900), "test": split(600)}
    monkeypatch.setattr(pb, "load", lambda model, ds, sp, ck, root=pb.NEW_ROOT: (*data[sp], {}))
    save = pb.lp.saver(tmp_path, "m", "best70")
    cell = pb.sibling_pair_cells("m", "best70", probe, save, names)["ee_vs_mm"]
    ytr, yte = data["train_N1000000"][1], data["test"][1]
    assert cell["auc"] == pytest.approx(1.0, abs=1e-12) and cell["n_train"] + cell["n_val"] == int(np.isin(ytr, (10, 11)).sum())
    assert (cell["n_test_signal"], cell["n_test_background"]) == (int((yte == 10).sum()), int((yte == 11).sum()))
    assert cell["train_set"] == "jc2bal/train_N1000000" and cell["test_set"] == "jc2bal/test"
    z = np.load(tmp_path / "m" / "best70" / "jc2bal_pairs_ee_vs_mm.npz")
    assert np.array_equal(yte[z["rows"]], np.where(z["y"] == 1, 10, 11))
    assert z["s"][z["y"] == 1].min() > z["s"][z["y"] == 0].max()
    assert pb.SIBLING_PAIRS == {"ee_vs_mm": (("label_X_ee",), ("label_X_mm",))}


def test_the_sibling_job_is_cpu_covers_every_model_and_avoids_the_bad_nodes():
    import yaml
    spec = importlib.util.spec_from_file_location("build_jc2bal_jobs", ROOT / "scripts/build_jc2bal_jobs.py")
    BJ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BJ)
    name, text = BJ.sib_spec("t", 5)
    d = yaml.safe_load(text)
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert name == d["metadata"]["name"] == "jc2bal-sib-raunav" and d["spec"]["completions"] == len(BJ.models()) == 30
    assert d["spec"]["parallelism"] == 5 and "nvidia.com/gpu" not in c["resources"]["limits"]
    assert c["resources"]["limits"]["memory"] == "16Gi" and c["resources"]["limits"]["cpu"] == "4"
    assert "k8s-haosu-10.sdsc.optiputer.net" in text and "configmap" not in text.lower()
    a = c["args"][0]
    assert "--sibling-pairs" in a and f"--out {BJ.LP_BAL}/fits_v4_sib " in a and f"--probs-out {BJ.LP_BAL}/fits_v4_sib_probs" in a
    assert '/${MODEL}/DONE ] || { echo "FATAL: ${MODEL} is not extracted"; exit 42; }' in a
    assert 'MODEL=init-s1; CKPTS="init init_bn";;' in a and 'MODEL=mtx-l188-s1; CKPTS="best70";;' in a
    # the CPU jobs launched before keep their text: no node exclusion unless asked
    assert "k8s-haosu-10" not in BJ.fit_spec("t", 5, True)[1]
