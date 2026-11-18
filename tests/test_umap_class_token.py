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


def test_the_umap_job_extracts_the_one_rule_cells_and_links_the_rest():
    import yaml
    BU = _bu()
    sha = "ab" * 20
    d = yaml.safe_load(BU.spec(sha, "umap_v4", "umap_v3", 90))
    c = d["spec"]["template"]["spec"]["containers"][0]
    a = c["args"][0]
    assert d["metadata"]["name"] == "umap-v4-raunav" and d["spec"]["backoffLimit"] == 1
    # CPU only: no GPU request, no GPU toleration, torch threads pinned to the quota
    assert "nvidia.com/gpu" not in c["resources"]["limits"] and c["resources"]["limits"]["cpu"] == "8"
    assert c["resources"]["limits"]["memory"] == BU.MEM
    assert "tolerations" not in d["spec"]["template"]["spec"] and "export OMP_NUM_THREADS=7 " in a
    # the code of a pushed commit; the storage line before anything is written
    assert f"git -C tls checkout -q {sha}" in a and "--branch" not in a and "retry clone" in a
    assert '[ "${USED}" -le 90 ]' in a and "configmap" not in a.lower()
    # the one rule (PI 2026-11-15): only the JetClass and JetClass-II fine-tuned cells are extracted anew, six models x
    # two, one per pod; every frozen extraction and the full-set top and quark/gluon ones come from umap_v3
    assert d["spec"]["completionMode"] == "Indexed" and d["spec"]["completions"] == 12 and d["spec"]["parallelism"] <= 6
    assert '[ "${N}" -eq 12 ]' in a and "-mindepth 4 -maxdepth 4 -name extract_manifest.json" in a
    assert "x jc2 frozen" not in a and "x jc1 frozen" not in a and "x top " not in a and "x qg " not in a
    assert "budget" not in a and "jc2_cell" not in a
    jc2b = "configs/data/JetClassII_base.yaml 2 /data/finetune/jc2_bal/test.parquet"
    old = "/data/results/eval/umap_v3/features"
    for i, (m, k, reg) in enumerate(BU.MODELS):
        for task, leg in (("jc2", "leg_jc2b"), ("jc1", "leg2")):
            assert f"C_{task}_{i}=/data/results/ft_v2/fixed/{leg}/{m}/N1000000/s1; done_cell ${{C_{task}_{i}}}" in a
        assert f"x jc2 ft mtx-{m} ${{C_jc2_{i}}}/net_best_epoch_state.pt 188 0 {jc2b} ;;" in a
        assert f"x jc1 ft mtx-{m} ${{C_jc1_{i}}}/net_best_epoch_state.pt 10 0 " in a
        # every reused extraction must name the checkpoint this job would read
        for task in BU.TASKS:
            assert f"same_ckpt {old}/{task}/mtx-{m}/frozen ${{CK{i}}}" in a
        for task in ("top", "qg"):
            assert f"same_ckpt {old}/{task}/mtx-{m}/ft ${{C_{task}_{i}}}/net_best_epoch_state.pt" in a
    # every cell and every reused extraction checked before the first extraction
    assert a.rindex("done_cell ${C_") < a.index("0) x jc2") and a.rindex("same_ckpt /data") < a.index("0) x jc2")
    assert a.index("0) x jc2 ft mtx-l188-s1") < a.index("x jc1 ft mtx-l188-s1")
    # the earlier extractions linked (not copied, not changed) by the pod that fits, before the fit
    link = a.index("ln -s /data/results/eval/umap_v3/${D} ${L}")
    assert a.index("mkdir /data/results/eval/umap_v4/fit.lock") < link < a.index("umap_class_token.py")
    linked = next(ln for ln in a.splitlines() if "ln -s" in ln).split("; do")[0].split()[3:]
    assert len(linked) == 6 * 6 and "features/jc2/mtx-l188-s1/ft" not in linked and "features/top/mtx-l188-s1/ft" in linked
    # the X->bx class token of every model, checked before the extractions start
    assert "/data/results/eval/xbx/mtx-{l188-s1,l162-s1,r42q1-s1,r16q1-s1,l162mass-s1,r16q1mass-s1}/{xbb,xbs,qcd}/eval/best70" in a
    assert "umap_class_token.py --feat-dir /data/results/eval/umap_v4/features --out-dir /data/results/eval/umap_v4" in a
    assert d["spec"]["activeDeadlineSeconds"] >= 43200
    # without --reuse every extraction is made
    assert yaml.safe_load(BU.spec(sha, "umap_x"))["spec"]["completions"] == 48


def _bash_fn(script: str, name: str) -> str:
    """The one-line bash function name of a generated script."""
    return next(ln for ln in script.splitlines() if ln.strip().startswith(f"{name} () {{"))


def test_the_umap_job_refuses_an_unfinished_cell_and_a_reused_extraction_of_another_checkpoint(tmp_path):
    import subprocess
    import yaml
    BU = _bu()
    a = yaml.safe_load(BU.spec("ab" * 20, "umap_v4", "umap_v3", 90))["spec"]["template"]["spec"]["containers"][0]["args"][0]
    fns = _bash_fn(a, "done_cell") + "\n" + _bash_fn(a, "same_ckpt") + "\n"
    run = lambda body: subprocess.run(["bash", "-c", "set -euo pipefail\n" + fns + body], capture_output=True, text=True)  # noqa: E731
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "net_best_epoch_state.pt").write_bytes(b"x")
    r = run(f"done_cell {cell}\necho after")
    assert r.returncode == 42 and "is not done" in r.stdout and "after" not in r.stdout
    (cell / "DONE").write_text("")
    assert run(f"done_cell {cell}").returncode == 0
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "extract_manifest.json").write_text(json.dumps({"checkpoint": "/data/a.pt"}))
    assert run(f"same_ckpt {ext} /data/a.pt").returncode == 0
    r = run(f"same_ckpt {ext} /data/b.pt\necho after")
    assert r.returncode == 42 and "is not of /data/b.pt" in r.stdout and "after" not in r.stdout


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
    # PI 2026-11-14 (step 5): the appendix holds two grids that cover every task (frozen, fine-tuned), and every
    # per-task grid, the hadronic JetClass-II one first, is in the supplement
    assert [p for p in order if p.startswith("UMAP")] == ["UMAP-frozen", "UMAP-ft", new[1], new[0], *new[2:], "UMAP-xbx",
                                                          *mass]
    assert _assigned(fig / "gallery_labelset.py", "UMAP_TASKS")["UMAP-frozen"][1][0] == ("two_prong", "JetClass-II")
    assert not [p for p in paper if p.startswith("UMAP")]
    pages = _assigned(fig / "gallery_labelset.py", "UMAP_PAGES")
    assert list(pages) == new
    assert [v[0] for v in pages.values()] == ["jc1", "two_prong", "leptons", "top", "qg"] == list(uc.SETS)[:5]
    assert list(uc.SETS)[5:] == ["xbb", "xbs"]   # round 5 (L12): drawn once the gallery has their pages
    read = _assigned(fig / "gallery_labelset.py", "UMAP_READ")      # written after the embeddings were seen (round 4)
    assert list(read) == new and all(v.endswith(".") and "LOOK" not in v for v in read.values())
    assert _assigned(fig / "gallery_labelset.py", "UMAP_SETTINGS") == uc.UMAP
