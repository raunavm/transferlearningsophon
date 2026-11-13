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


def test_the_umap_job_extracts_the_new_models_and_links_the_earlier_ones():
    import yaml
    BU = _bu()
    sha = "ab" * 20
    d = yaml.safe_load(BU.spec(sha, "umap_v3", "umap_v2", 89))
    c = d["spec"]["template"]["spec"]["containers"][0]
    a = c["args"][0]
    assert d["metadata"]["name"] == "umap-v3-raunav" and d["spec"]["backoffLimit"] == 1
    # CPU only: no GPU request, no GPU toleration, torch threads pinned to the quota
    assert "nvidia.com/gpu" not in c["resources"]["limits"] and c["resources"]["limits"]["cpu"] == "8"
    assert "tolerations" not in d["spec"]["template"]["spec"] and "export OMP_NUM_THREADS=7 " in a
    # the code of a pushed commit; the storage line before anything is written
    assert f"git -C tls checkout -q {sha}" in a and "--branch" not in a and "retry clone" in a
    assert '[ "${USED}" -le 89 ]' in a and "configmap" not in a.lower()
    # four new models x four tasks x frozen and fine-tuned, one per pod; 188 and 17 classes come from umap_v2
    new = [m for m, _k, _r in BU.MODELS if m not in BU.REUSABLE]
    assert new == ["l162-s1", "r42q1-s1", "l162mass-s1", "r16q1mass-s1"]
    assert d["spec"]["completionMode"] == "Indexed" and d["spec"]["completions"] == 32 and d["spec"]["parallelism"] <= 12
    assert '[ "${N}" -eq 32 ]' in a and "-mindepth 4 -maxdepth 4 -name extract_manifest.json" in a
    assert "x jc1 frozen mtx-l188-s1" not in a and "x jc2 ft mtx-r16q1-s1 " not in a
    jc2b = "configs/data/JetClassII_base.yaml 2 /data/finetune/jc2_bal/test.parquet"
    for i, (m, k, reg) in enumerate(BU.MODELS):
        if m in BU.REUSABLE:
            assert f"/data/results/eval/umap_v2/features/{{jc1,jc2,top,qg}}/mtx-{m}/{{frozen,ft}}; do [ -f" in a
            continue
        assert f"C_jc2_{i}=$(jc2_cell {m})" in a
        # the frozen mass-output checkpoint is read with its regression output; the fine-tuned cell has none
        assert f"x jc2 frozen mtx-{m} ${{CK{i}}} {k} {reg} {jc2b} ;;" in a
        assert f"x jc2 ft mtx-{m} ${{C_jc2_{i}}}/net_best_epoch_state.pt 188 0 {jc2b} ;;" in a
        assert f"x top ft mtx-{m} ${{C_top_{i}}}/net_best_epoch_state.pt 2 0 " in a
    assert (k, reg) == (17, 1) and "--num-reg ${REG}" in a
    # the long JetClass-II extractions first; every cell checked before the first extraction
    assert a.index("0) x jc2 frozen mtx-l162-s1") < a.index("x jc1 frozen mtx-l162-s1") < a.index("x qg ft mtx-r16q1mass-s1")
    assert a.rindex("done_cell ${C_") < a.index("0) x jc2")
    # the earlier extractions linked (not copied, not changed) by the pod that fits, before the fit
    link = a.index("ln -s ${D} ${L}")
    assert a.index("mkdir /data/results/eval/umap_v3/fit.lock") < link < a.index("umap_class_token.py")
    assert "/umap_v2/features/{jc1,jc2,top,qg}/mtx-{l188-s1,r16q1-s1}/{frozen,ft}" in next(
        ln for ln in a.splitlines() if "ln -s" in ln)
    # the X->bx class token of every model, checked before the extractions start
    assert "/data/results/eval/xbx/mtx-{l188-s1,l162-s1,r42q1-s1,r16q1-s1,l162mass-s1,r16q1mass-s1}/{xbb,xbs,qcd}/eval/best70" in a
    assert "umap_class_token.py --feat-dir /data/results/eval/umap_v3/features --out-dir /data/results/eval/umap_v3" in a
    assert d["spec"]["activeDeadlineSeconds"] >= 43200
    # without --reuse every model is extracted
    assert yaml.safe_load(BU.spec(sha, "umap_x"))["spec"]["completions"] == 48


def _xbx(root, model, rng_seed=0, n=(2500, 2600, 9000)):
    r = np.random.default_rng(rng_seed)
    for k, (split, m) in enumerate(zip(uc.XBX_SPLITS, n)):
        d = root / model / split / "eval" / uc.XBX_CKPT
        d.mkdir(parents=True)
        msd = np.random.default_rng(10 + k).uniform(100, 300, m).astype(np.float32)   # the same jets for every model
        np.save(d / "features.npy", r.normal(size=(m, 128)).astype(np.float16))
        np.save(d / "label188.npy", np.full(m, 0 if split == "qcd" else 1, np.int16))
        np.save(d / "rows.npy", np.arange(m))
        np.savez(d / "observers.npz", jet_sdmass=msd)
        (d / "manifest.json").write_text(json.dumps({"n_feature_rows": m, "n_stream": m, "checkpoint": f"{model}.pt",
                                                     "checkpoint_sha256": "s", "data_config": "x.yaml"}))


def test_the_xbb_and_xbs_grids_embed_the_same_jets_in_the_window_for_every_model(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(uc, "N_PER_GROUP", 50)
    monkeypatch.setattr(uc, "embed", lambda x: (seen.append(np.array(x)), np.zeros((len(x), 2)))[1])
    for i, m in enumerate(("ma", "mb")):
        _xbx(tmp_path / "xbx", m, rng_seed=i)
    args = ["--feat-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--sets", "xbb", "xbs", "--models", "ma",
            "mb", "--xbx-root", str(tmp_path / "xbx")]
    assert uc.main(args) == 0
    # one grid per task (decision L12): X->bb against QCD, X->bs against QCD
    for name, sig in (("xbb", "bb"), ("xbs", "bs")):
        docs = [json.loads((tmp_path / "o" / f"{name}_{m}_frozen.json").read_text()) for m in ("ma", "mb")]
        assert len({d["rows_sha256"] for d in docs}) == 1 and docs[0]["groups"] == [sig, "QCD"]
        assert (docs[0]["set"], docs[0]["task"]) == (name, "xbx") and docs[1]["checkpoint"] == "mb.pt"
        assert docs[0]["window"] == [150.0, 230.0] and docs[0]["mx"] == 200.0
        assert np.bincount(docs[0]["group"]).tolist() == [50, 50] and not (tmp_path / "o" / f"{name}_ma_ft.json").exists()
    x, g, msd, _ = uc.xbx_inputs(tmp_path / "xbx", "ma")
    assert ((msd > 150) & (msd < 230)).all() and x.shape == (g.size, 128) and set(g.tolist()) == {0, 1, 2}
    # another model's jets (another m_SD order) are refused
    o = tmp_path / "xbx" / "mb" / "qcd" / "eval" / "best70" / "observers.npz"
    np.savez(o, jet_sdmass=np.load(o)["jet_sdmass"][::-1].copy())
    with pytest.raises(SystemExit, match="not those of"):
        uc.main(args)


def _assigned(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def test_the_gallery_has_five_umap_pages_none_in_the_paper():
    fig = ROOT / "experiments/FIGS/figstyle"
    new = ["UMAP-jc1", "UMAP-jc2-had", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg"]
    order, paper = _assigned(fig / "gallery_figures.py", "ORDER"), _assigned(fig / "gallery_figures.py", "PAPER")
    # ordered by tier (gallery_placement): the hadronic JetClass-II page in the appendix, the others in the supplement,
    # then the supplement's further grids from umap_v3 (round 3, L19): X->bb and X->bs against QCD, and the mass
    # networks under their label sets on every task (gallery_labelset.UMAP_MORE)
    mass = [f"UMAP-mass-{p.removeprefix('UMAP-')}" for p in (new[1], new[0], *new[2:])] + ["UMAP-mass-xbx"]
    assert [p for p in order if p.startswith("UMAP")] == [new[1], new[0], *new[2:], "UMAP-xbx", *mass]
    assert not [p for p in paper if p.startswith("UMAP")]
    pages = _assigned(fig / "gallery_labelset.py", "UMAP_PAGES")
    assert list(pages) == new
    assert [v[0] for v in pages.values()] == ["jc1", "two_prong", "leptons", "top", "qg"] == list(uc.SETS)[:5]
    assert list(uc.SETS)[5:] == ["xbb", "xbs"]   # round 5 (L12): drawn once the gallery has their pages
    read = _assigned(fig / "gallery_labelset.py", "UMAP_READ")      # written after the embeddings were seen (round 4)
    assert list(read) == new and all(v.endswith(".") and "LOOK" not in v for v in read.values())
    assert _assigned(fig / "gallery_labelset.py", "UMAP_SETTINGS") == uc.UMAP
