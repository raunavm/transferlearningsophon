"""X->bb / X->bs staging (experiments/XBS/stage_xbx.py) and its data config
(configs/xbx/JetClassII_xbx.yaml), on synthetic JetClass-II-like files."""
import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_xbx_t", ROOT / "experiments/XBS/stage_xbx.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)


def test_the_config_is_the_base_config_but_for_its_label_and_reweighting():
    a = yaml.safe_load((ROOT / "configs/data/JetClassII_base.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs/xbx/JetClassII_xbx.yaml").read_text())
    for k in ("selection", "preprocess", "inputs"):
        assert b[k] == a[k], k
    kept = {k: v for k, v in a["new_variables"].items() if not k.startswith("label_")}
    assert b["new_variables"] == kept
    assert b["labels"] == {"type": "custom", "value": {"truth_label": "xbx_class"}}
    assert "weights" not in b and set(b["observers"]) == {"jet_pt", "jet_eta", "jet_sdmass"}


def test_the_branches_are_what_the_config_reads():
    assert ST.config_branches() == sorted(
        ["jet_energy", "jet_eta", "jet_pt", "jet_sdmass"]
        + [f"part_{v}" for v in ("charge", "d0err", "d0val", "deta", "dphi", "dzerr", "dzval", "energy",
                                 "isChargedHadron", "isElectron", "isMuon", "isNeutralHadron", "isPhoton",
                                 "px", "py", "pz")])


def _jets(rng, n, label, msd=None):
    cols = ST.config_branches()
    npart = rng.integers(5, 30, n)
    off = np.r_[0, np.cumsum(npart)].astype(np.int32)
    data = {}
    for c in cols:
        if c.startswith("part_"):
            data[c] = pa.ListArray.from_arrays(pa.array(off), pa.array(rng.normal(1, 1, off[-1]).astype(np.float32)))
    data["jet_pt"] = pa.array(rng.uniform(150, 2600, n).astype(np.float32))
    data["jet_sdmass"] = pa.array((rng.uniform(10, 520, n) if msd is None else msd).astype(np.float32))
    data["jet_eta"] = pa.array(rng.normal(0, 1, n).astype(np.float32))
    data["jet_energy"] = pa.array(rng.uniform(300, 3000, n).astype(np.float32))
    data["jet_label"] = pa.array(np.asarray(label, np.int32) * np.ones(n, np.int32))
    data["aux_unused"] = pa.array(np.zeros(n, np.float32))
    return pa.table(data)


@pytest.fixture
def jc2(tmp_path, monkeypatch):
    """Two QCD and two Res2P files per role, QCD with mixed QCD labels, Res2P with X->bb and other
    two-prong classes."""
    rng = np.random.default_rng(0)
    d = tmp_path / "jc2"
    d.mkdir()
    files = {}
    for role, ids in (("ft_heldout", (285, 286, 204, 205)), ("ft_test", (350, 351, 250, 251))):
        for i in ids:
            fam = "QCD" if i >= 285 else "Res2P"
            lab = rng.integers(161, 188, 6000) if fam == "QCD" else rng.choice([0, 1, 4, 6], 6000)
            p = d / f"{fam}_{i:04d}.parquet"
            pq.write_table(_jets(rng, 6000, lab), p)
            files.setdefault((role, "qcd" if fam == "QCD" else "xbb_jc2"), []).append(str(p))
    monkeypatch.setattr(ST, "jc2_files", lambda role, cls: files[(role, cls)])
    return files


def test_the_evaluation_jets_are_the_test_files_jets_of_the_class_in_the_window(jc2, tmp_path):
    root = tmp_path / "xbx"
    ST.jc2_eval("xbb_jc2", root)
    out = root / "xbb_jc2/eval"
    assert (out / "DONE").exists() and json.loads((out / "manifest.json").read_text())["role"] == "ft_test"
    t = pa.concat_tables([pq.read_table(p) for p in sorted(out.glob("*.parquet"))])
    src = pa.concat_tables([pq.read_table(p) for p in jc2[("ft_test", "xbb_jc2")]])
    s = src.to_pandas()
    want = s[(s.jet_label == 0) & (s.jet_pt > 200) & (s.jet_pt < 2500) & (s.jet_sdmass > 50) & (s.jet_sdmass < 300)]
    assert t.num_rows == len(want) and set(np.asarray(t.column("xbx_class"))) == {1}
    assert "jet_label" not in t.column_names and "aux_unused" not in t.column_names
    with pytest.raises(SystemExit, match="never rebuilt"):
        ST.jc2_eval("xbb_jc2", root)


def test_the_pairs_are_matched_balanced_disjoint_and_from_held_out_jets_only(jc2, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "SHARD_ROWS", 400)
    root = tmp_path / "xbx"
    man = ST.pair("xbb_jc2", root, n_train=600, n_val=100)
    out = root / "pairs/xbb_jc2"
    assert man["train_draw"]["n_per_class"] == 600 and man["repetition"] == pytest.approx(ST.X.HEAD_SAMPLES / 1200)
    assert set(man["qcd_files"]) == set(jc2[("ft_heldout", "qcd")])
    assert set(man["signal_files"]) == set(jc2[("ft_heldout", "xbb_jc2")])
    shards = [pq.read_table(out / s) for s in man["train_shards"]]
    assert len(shards) == 3
    for t in shards:                                     # every shard mixes both classes
        assert set(np.asarray(t.column("xbx_class"))) == {0, 1}
    tr, va = pa.concat_tables(shards), pq.read_table(out / "val.parquet")
    for t, n in ((tr, 600), (va, 100)):
        y = np.asarray(t.column("xbx_class"))
        assert (y == 1).sum() == (y == 0).sum() == n
        c = ST.X._cells(np.asarray(t.column("jet_pt")), np.asarray(t.column("jet_sdmass")), ST.X.grid_edges())
        assert np.array_equal(np.bincount(c[y == 1], minlength=700), np.bincount(c[y == 0], minlength=700))
    # disjoint: no jet in both training and validation (the energies are continuous, so unique)
    assert not set(np.asarray(tr.column("jet_energy"))) & set(np.asarray(va.column("jet_energy")))
    # the same draw again, bit for bit
    man2 = ST.pair("xbb_jc2", tmp_path / "again", n_train=600, n_val=100)
    assert man2["sha256"] == man["sha256"]


def _outside_sample(path, rng, n, drop=()):
    """An outside sample with its own branch layout: ROOT with the particles under one counter, or
    parquet in float64 with an event-level branch."""
    t = _jets(rng, n, 0, msd=rng.normal(125, 10, n)).drop(["jet_label", "aux_unused"])
    t = t.drop([c for c in drop])
    if str(path).endswith(".parquet"):
        t = t.cast(pa.schema([(f.name, pa.list_(pa.float64()) if pa.types.is_list(f.type) else pa.float64())
                              for f in t.schema]))
        pq.write_table(t.append_column("event_weight", pa.array(np.ones(n))), path)
        return
    import awkward as ak
    import uproot
    # plain (non-nullable) arrays: uproot cannot write awkward's option types
    lists = {c: t.column(c).combine_chunks() for c in t.column_names if c.startswith("part_")}
    counts = np.diff(next(iter(lists.values())).offsets.to_numpy())
    part = ak.zip({c[len("part_"):]: ak.unflatten(v.values.to_numpy(), counts) for c, v in lists.items()})
    jets = {c: t.column(c).to_numpy() for c in t.column_names if c.startswith("jet_")}
    data = {"part": part, **jets, "event_weight": np.ones(n)}
    try:
        with uproot.recreate(path) as f:     # a TTree, as the n-tupliser writes (newer uproot defaults to RNTuple)
            tree = f.mktree("tree", {k: v.type if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
            tree.extend(data)
    except AttributeError as e:                      # uproot 5.1 cannot write under numpy 2
        pytest.skip(f"uproot cannot write ROOT here: {e}")


@pytest.mark.parametrize("ext", ["parquet", "root"])
def test_an_outside_sample_is_checked_then_split_per_jet(jc2, tmp_path, ext):
    rng = np.random.default_rng(1)
    good, bad = tmp_path / f"Xbs_0.{ext}", tmp_path / f"Xbs_bad.{ext}"
    _outside_sample(good, rng, 3000)
    _outside_sample(bad, rng, 100, drop=("part_d0err",))
    rep = ST.check([str(good)])
    assert rep["problems"] == [] and "event_weight" in rep["per_file"][str(good)]["extra_branches"]
    assert 110 < rep["first_file"]["msd_percentiles_5_50_95"][1] < 140
    assert any("part_d0err" in p for p in ST.check([str(bad)])["problems"])
    with pytest.raises(SystemExit, match="does not carry"):
        ST.convert("xbs", [str(bad)], tmp_path / "xbx")
    root = tmp_path / "xbx"
    counts = ST.convert("xbs", [str(good)], root)
    pool = pq.read_table(root / "xbs/pool/Xbs_0.parquet")
    ev = pq.read_table(root / "xbs/eval/Xbs_0.parquet")
    c = counts[str(good)]
    assert pool.num_rows + ev.num_rows == c["selected"] and ev.num_rows == c["eval"]
    assert 0.15 < ev.num_rows / c["selected"] < 0.25
    # the JetClass-II types, whatever the sample's own
    ref = ST.reference_schema(ST.config_branches())
    assert pool.schema.remove_metadata() == ref.remove_metadata()
    assert set(np.asarray(pool.column("xbx_class"))) == {1}
    # a pod that died between pool/ and eval/: redone to the same bytes; a finished one is refused
    man_pool = json.loads((root / "xbs/pool/manifest.json").read_text())["sha256"]
    (root / "xbs/eval/DONE").unlink()
    ST.convert("xbs", [str(good)], root)
    assert json.loads((root / "xbs/pool/manifest.json").read_text())["sha256"] == man_pool
    with pytest.raises(SystemExit, match="never rebuilt"):
        ST.convert("xbs", [str(good)], root)
    # pairs draw from the converted pool against the held-out QCD
    man = ST.pair("xbs", root, n_train=300, n_val=50)
    assert man["signal_files"] == [str(root / "xbs/pool/Xbs_0.parquet")] and man["train_draw"]["n_per_class"] <= 300
