"""The UMAP figures' inputs (experiments/EVAL/umap_class_token.py, scripts/build_umap_jobs.py)."""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("umap_class_token", ROOT / "experiments/EVAL/umap_class_token.py")
uc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uc)


def _ext(d: pathlib.Path, y, pt):
    d.mkdir(parents=True)
    np.save(d / "label188.npy", y)
    np.savez(d / "observers.npz", jet_pt=pt)


def test_extractions_of_other_jets_are_refused(tmp_path):
    y, pt = np.repeat(np.arange(10), 5), np.arange(50.0)
    _ext(tmp_path / "a", y, pt)
    _ext(tmp_path / "b", y, pt)
    assert np.array_equal(uc.aligned([tmp_path / "a", tmp_path / "b"]), y)
    _ext(tmp_path / "c", y, pt[::-1].copy())
    with pytest.raises(SystemExit):
        uc.aligned([tmp_path / "a", tmp_path / "c"])


def test_the_jetclass_names_follow_the_config_labels():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/finetune/JetClassI_sophon_noweight.yaml").read_text())
    assert [f"label_{n}" for n in uc.JC1] == cfg["labels"]["value"]
    top = yaml.safe_load((ROOT / "configs/finetune/TopReference.yaml").read_text())["labels"]["value"]
    qg = yaml.safe_load((ROOT / "configs/finetune/EnergyFlowQG.yaml").read_text())["labels"]["value"]
    assert top == ["label_QCD", "label_Top"] and qg == ["label_gluon", "label_quark"]
    assert uc.BENCH == {"top": ("QCD", "top"), "qg": ("gluon", "quark")}


def test_embedding_is_reproducible():
    pytest.importorskip("umap")
    x = np.random.default_rng(0).normal(size=(300, 16))
    assert np.array_equal(uc.embed(x), uc.embed(x))


def test_the_umap_job_reads_both_models_frozen_and_fine_tuned_and_pins_umap():
    import yaml
    spec = importlib.util.spec_from_file_location("build_umap_jobs", ROOT / "scripts/build_umap_jobs.py")
    BU = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BU)
    d = yaml.safe_load(BU.spec("t", "/data/results/ft_v2/budget_long", "umap_v1"))
    c = d["spec"]["template"]["spec"]["containers"][0]
    a = c["args"][0]
    assert d["metadata"]["name"] == "umap-v1-raunav" and d["spec"]["backoffLimit"] == 1
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1" and "pip install --no-cache-dir -q umap-learn==0.5.7" in a
    for m, k in (("l188-s1", 188), ("r16q1-s1", 17)):
        assert f"CELL=/data/results/ft_v2/budget_long/leg2/{m}/N1000000/s1" in a
        assert f"x frozen mtx-{m} ${{CK}} {k}" in a and f"x ft mtx-{m} ${{CELL}}/net_best_epoch_state.pt 10" in a
    assert "--num-workers 1" in a and "--out-dir /data/results/eval/umap_v1" in a and "configmap" not in a.lower()
