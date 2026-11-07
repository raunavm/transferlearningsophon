"""The UMAP figures' inputs (experiments/EVAL/umap_class_token.py, scripts/build_umap_jobs.py) and their
gallery pages (experiments/FIGS/figstyle/gallery_labelset.py, gallery_figures.py)."""
import ast
import importlib.util
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("umap_class_token", ROOT / "experiments/EVAL/umap_class_token.py")
uc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uc)


def _ext(d: pathlib.Path, y, pt, cfg="c.yaml", feats=None):
    d.mkdir(parents=True)
    np.save(d / "label188.npy", y)
    np.savez(d / "observers.npz", jet_pt=pt)
    if feats is not None:
        np.save(d / "features.npy", feats)
    (d / "extract_manifest.json").write_text(json.dumps(
        {"checkpoint": str(d / "ck.pt"), "checkpoint_sha256": "s", "data_config": cfg, "stride": 1}))


def _bu():
    spec = importlib.util.spec_from_file_location("build_umap_jobs", ROOT / "scripts/build_umap_jobs.py")
    BU = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BU)
    return BU


def test_extractions_of_other_jets_are_refused(tmp_path):
    y, pt = np.repeat(np.arange(10), 5), np.arange(50.0)
    _ext(tmp_path / "a", y, pt)
    _ext(tmp_path / "b", y, pt)
    assert np.array_equal(uc.aligned([tmp_path / "a", tmp_path / "b"]), y)
    _ext(tmp_path / "c", y, pt[::-1].copy())
    with pytest.raises(SystemExit):
        uc.aligned([tmp_path / "a", tmp_path / "c"])


def test_the_group_names_follow_the_config_labels():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/finetune/JetClassI_sophon_noweight.yaml").read_text())
    assert [f"label_{n}" for n in uc.JC1] == cfg["labels"]["value"]
    top = yaml.safe_load((ROOT / "configs/finetune/TopReference.yaml").read_text())["labels"]["value"]
    qg = yaml.safe_load((ROOT / "configs/finetune/EnergyFlowQG.yaml").read_text())["labels"]["value"]
    assert top == ["label_QCD", "label_Top"] and qg == ["label_gluon", "label_quark"]
    assert uc.SETS["top"] == ("top", (((0,), "QCD"), ((1,), "top")))
    assert uc.SETS["qg"] == ("qg", (((0,), "gluon"), ((1,), "quark")))
    assert uc.SETS["jc1"][1] == tuple(((k,), n) for k, n in enumerate(uc.JC1))
    assert uc.SETS["two_prong"][1] == uc.T.SETS["two_prong"] and uc.SETS["leptons"][1] == uc.T.SETS["leptons"]
    assert uc.SETS["two_prong"][1][3] == (tuple(range(161, 188)), "QCD")     # the 27 QCD classes as one group
    assert uc.UMAP == {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 1, "standardized": True}
    assert uc.N_PER_GROUP == 2000


def test_embedding_is_reproducible():
    pytest.importorskip("umap")
    x = np.random.default_rng(0).normal(size=(300, 16))
    assert np.array_equal(uc.embed(x), uc.embed(x))


def test_every_panel_of_a_set_embeds_the_same_jets(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(uc, "N_PER_GROUP", 3)
    monkeypatch.setattr(uc, "embed", lambda x: (seen.append(np.array(x)), np.zeros((len(x), 2)))[1])
    y, pt = np.repeat([0, 1], 10), np.arange(20.0)
    for m in uc.MODELS:
        for kind in uc.KINDS:
            _ext(tmp_path / "f" / "top" / m / kind, y, pt, feats=np.arange(20.0)[:, None] * np.ones((1, 4)))
    assert uc.main(["--feat-dir", str(tmp_path / "f"), "--out-dir", str(tmp_path / "o"), "--sets", "top"]) == 0
    docs = [json.loads((tmp_path / "o" / f"top_{m}_{kind}.json").read_text()) for m in uc.MODELS for kind in uc.KINDS]
    assert len({d["rows_sha256"] for d in docs}) == 1 and all(np.array_equal(s, seen[0]) for s in seen)
    assert np.bincount(y[seen[0][:, 0].astype(int)]).tolist() == [3, 3]
    d = docs[1]
    assert (d["set"], d["task"], d["model"], d["features"], d["groups"]) == ("top", "top", uc.MODELS[0], "ft", ["QCD", "top"])
    assert d["umap"] == uc.UMAP and d["checkpoint"].endswith(f"top/{uc.MODELS[0]}/ft/ck.pt") and d["n_per_group"] == 3


def test_extractions_through_other_data_configs_are_refused(tmp_path):
    y, pt = np.repeat([0, 1], 10), np.arange(20.0)
    for m in uc.MODELS:
        for kind in uc.KINDS:
            _ext(tmp_path / "qg" / m / kind, y, pt, cfg="other.yaml" if kind == "frozen" else "c.yaml")
    with pytest.raises(SystemExit, match="different data configs"):
        uc.main(["--feat-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--sets", "qg"])


def test_the_umap_job_reads_both_models_frozen_and_fine_tuned_on_every_task():
    import yaml
    BU = _bu()
    d = yaml.safe_load(BU.spec("t", "umap_v2"))
    c = d["spec"]["template"]["spec"]["containers"][0]
    a = c["args"][0]
    assert d["metadata"]["name"] == "umap-v2-raunav" and d["spec"]["backoffLimit"] == 1
    # CPU only: no GPU request, no GPU toleration, torch threads pinned to the quota
    assert "nvidia.com/gpu" not in c["resources"]["limits"] and c["resources"]["limits"]["cpu"] == "16"
    assert "tolerations" not in d["spec"]["template"]["spec"] and "export OMP_NUM_THREADS=14 " in a
    assert "gpu_probe" not in a and "pip install --no-cache-dir -q umap-learn==0.5.7 pyarrow==25.0.1" in a
    assert 'python3 -c "import pyarrow" || ' in a        # weaver reads the parquet test sets through it
    assert '[ ! -f /data/results/eval/umap_v2/DONE ]' in a and '[ "${USED}" -le 85 ]' in a and "retry clone" in a
    assert "ry-gpu-04.sdsc.optiputer.net" in yaml.safe_dump(d["spec"]["template"]["spec"]["affinity"])
    jc2b = "configs/data/JetClassII_base.yaml 1 /data/finetune/jc2_bal/test.parquet"
    qg = "configs/finetune/EnergyFlowQG.yaml 10 /data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet"
    for i, (m, k) in enumerate((("l188-s1", 188), ("r16q1-s1", 17))):
        assert f"C_jc1_{i}=/data/results/ft_v2/budget_long/leg2/{m}/N1000000/s1; done_cell" in a
        assert f"C_jc2_{i}=$(jc2_cell {m})" in a
        assert f"C_top_{i}=/data/results/ft_v2/best70/leg_top/{m}/N1200000/s1; done_cell" in a
        assert f"C_qg_{i}=/data/results/ft_v2/best70/leg_qg/{m}/N1600000/s1; done_cell" in a
        for task, kt, tail in (("jc1", 10, "configs/finetune/JetClassI_sophon_noweight.yaml 10 ${JC1}"), ("jc2", 188, jc2b),
                               ("top", 2, "configs/finetune/TopReference.yaml 20 /data/finetune/top/top_test.parquet"),
                               ("qg", 2, qg)):
            # the frozen and the fine-tuned model read the same jets through the same config
            assert f"x {task} frozen mtx-{m} ${{CK{i}}} {k} {tail}" in a.splitlines()
            assert f"x {task} ft mtx-{m} ${{C_{task}_{i}}}/net_best_epoch_state.pt {kt} {tail}" in a.splitlines()
        for man in (f"v2_linprobe/mtx-{m}/jc1/", f"v2_linprobe_bal/mtx-{m}/jc2bal/", f"v2_linprobe/mtx-{m}/top/",
                    f"v2_linprobe/mtx-{m}/qg/"):
            assert f"/data/results/eval/{man}test/best70/manifest.json" in a
    # the long budget first, else the standard one; every cell checked before the first extraction
    assert "for R in /data/results/ft_v2/budget_long /data/results/ft_v2/budget; do" in a
    assert a.rindex("done_cell ${C_") < a.index("\nx jc1 frozen")
    assert "[ -f ${D}/extract_manifest.json ] && return 0" in a and "--num-workers 1" in a
    assert "umap_class_token.py --feat-dir /data/results/eval/umap_v2/features --out-dir /data/results/eval/umap_v2" in a
    assert "configmap" not in a.lower()


def _assigned(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def test_the_gallery_has_five_umap_pages_none_in_the_paper():
    fig = ROOT / "experiments/FIGS/figstyle"
    new = ["UMAP-jc1", "UMAP-jc2-had", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg"]
    order, paper = _assigned(fig / "gallery_figures.py", "ORDER"), _assigned(fig / "gallery_figures.py", "PAPER")
    assert [p for p in order if p.startswith("UMAP")] == new and not [p for p in paper if p.startswith("UMAP")]
    pages = _assigned(fig / "gallery_labelset.py", "UMAP_PAGES")
    assert list(pages) == new
    assert [v[0] for v in pages.values()] == ["jc1", "two_prong", "leptons", "top", "qg"] == list(uc.SETS)
    assert _assigned(fig / "gallery_labelset.py", "UMAP_LOOK") == "LOOK: to be written after the embeddings are seen"
    assert _assigned(fig / "gallery_labelset.py", "UMAP_SETTINGS") == uc.UMAP
