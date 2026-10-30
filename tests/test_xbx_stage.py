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


def _plain(t):
    """{column: awkward array} without option types: part_* jagged, the rest flat."""
    import awkward as ak
    out = {}
    for c in t.column_names:
        col = t.column(c).combine_chunks()
        if pa.types.is_list(col.type):
            out[c] = ak.unflatten(col.values.to_numpy(), np.diff(col.offsets.to_numpy()))
        else:
            out[c] = ak.Array(col.to_numpy())
    return out


def _write_like_jc2(path, t):
    """Written by awkward, as the JetClass-II release files are."""
    import awkward as ak
    ak.to_parquet(ak.zip(_plain(t), depth_limit=1), path)


def _readable_by_weaver(files):
    """Every staged file loads as weaver loads parquet (ak.from_parquet on the config's branches
    and the label), as plain arrays: no option type anywhere. With weaver installed (the image),
    through weaver's own reader too."""
    import awkward as ak
    cols = ST.config_branches() + ["xbx_class"]
    for f in files:
        a = ak.from_parquet(f, columns=cols)
        assert "?" not in str(a.type), (f, str(a.type)[:200])
        assert len(a) == pq.read_metadata(f).num_rows
    try:
        from weaver.utils.data.fileio import _read_files
    except ImportError:
        return
    for f in files:
        assert len(_read_files([f], cols, (0, 1))) == pq.read_metadata(f).num_rows


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
            _write_like_jc2(p, _jets(rng, 6000, lab))
            files.setdefault((role, "qcd" if fam == "QCD" else "xbb_jc2"), []).append(str(p))
    monkeypatch.setattr(ST, "jc2_files", lambda role, cls: files[(role, cls)])
    # as the release's files are: non-nullable fields, awkward's description in the metadata
    s = pq.read_schema(files[("ft_test", "qcd")][0])
    assert b"awkward_array_metadata" in s.metadata and not s.field("part_px").nullable
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
    _readable_by_weaver(sorted(str(p) for p in out.glob("*.parquet")))
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
    _readable_by_weaver([str(out / s) for s in man["train_shards"]] + [str(out / "val.parquet")])
    # the same draw again, bit for bit
    man2 = ST.pair("xbb_jc2", tmp_path / "again", n_train=600, n_val=100)
    assert man2["sha256"] == man["sha256"]


W125 = (93.75, 143.75)      # Sophon's window scaled to mX = 125 GeV


def _zeroed(t, cols):
    """t with the part_* columns cols set to 0 on every particle, as a zero-filled sample has them."""
    for c in cols:
        col = t.column(c).combine_chunks()
        t = t.set_column(t.schema.get_field_index(c), c,
                         pa.ListArray.from_arrays(col.offsets, pa.array(np.zeros(len(col.values), np.float32))))
    return t


def _outside_sample(path, rng, n, drop=(), zero=(), rename=None):
    """An outside sample with its own branch layout: ROOT with the particles under one counter, or
    parquet in float64 with an event-level branch."""
    t = _jets(rng, n, 0, msd=rng.normal(125, 10, n)).drop(["jet_label", "aux_unused"])
    t = _zeroed(t.drop([c for c in drop]), zero)
    if rename:
        t = t.rename_columns([rename.get(c, c) for c in t.column_names])
    if str(path).endswith(".parquet"):
        t = t.cast(pa.schema([(f.name, pa.list_(pa.float64()) if pa.types.is_list(f.type) else pa.float64())
                              for f in t.schema]))
        pq.write_table(t.append_column("event_weight", pa.array(np.ones(n))), path)
        return
    import awkward as ak
    import uproot
    # plain (non-nullable) arrays: uproot cannot write awkward's option types
    arr = _plain(t)
    part = ak.zip({c[len("part_"):]: v for c, v in arr.items() if c.startswith("part_")})
    jets = {c: ak.to_numpy(v) for c, v in arr.items() if c.startswith("jet_")}
    data = {"part": part, **jets, "event_weight": np.ones(n)}
    try:
        with uproot.recreate(path) as f:     # a TTree, as the n-tupliser writes (newer uproot defaults to RNTuple)
            tree = f.mktree("tree", {k: v.type if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
            tree.extend(data)
    except AttributeError as e:                      # uproot 5.1 cannot write under numpy 2
        pytest.skip(f"uproot cannot write ROOT here: {e}")


@pytest.mark.parametrize("ext", ["parquet", "root"])
def test_an_outside_sample_is_checked_then_split_per_jet(jc2, tmp_path, monkeypatch, ext):
    monkeypatch.setattr(ST.X, "A5_EVAL_FLOOR", 100)
    rng = np.random.default_rng(1)
    good, bad = tmp_path / f"Xbs_0.{ext}", tmp_path / f"Xbs_bad.{ext}"
    _outside_sample(good, rng, 3000)
    _outside_sample(bad, rng, 100, drop=("part_d0err",))
    rep = ST.check([str(good)], W125)
    assert rep["problems"] == [] and "event_weight" in rep["per_file"][str(good)]["extra_branches"]
    assert 110 < rep["first_file"]["msd_percentiles_5_50_95"][1] < 140
    assert any("part_d0err" in p for p in ST.check([str(bad)], W125)["problems"])
    with pytest.raises(SystemExit, match="does not carry"):
        ST.convert("xbs", [str(bad)], tmp_path / "xbx", W125)
    root = tmp_path / "xbx"
    counts = ST.convert("xbs", [str(good)], root, W125)
    pool = pq.read_table(root / "xbs/pool/Xbs_0.parquet")
    ev = pq.read_table(root / "xbs/eval/Xbs_0.parquet")
    c = counts[str(good)]
    assert pool.num_rows + ev.num_rows == c["selected"] and ev.num_rows == c["eval"]
    assert 0.15 < ev.num_rows / c["selected"] < 0.25
    # the floor counts exactly the evaluation jets convert writes, inside the window
    m = np.asarray(ev.column("jet_sdmass"))
    assert rep["eval_in_window"] == int(((m > W125[0]) & (m < W125[1])).sum())
    # the JetClass-II types, whatever the sample's own
    ref = ST.reference_schema(ST.config_branches())
    assert pool.schema.remove_metadata() == ref.remove_metadata()
    assert set(np.asarray(pool.column("xbx_class"))) == {1}
    _readable_by_weaver([str(root / "xbs/pool/Xbs_0.parquet"), str(root / "xbs/eval/Xbs_0.parquet")])
    # a pod that died between pool/ and eval/: redone to the same bytes; a finished one is refused
    man_pool = json.loads((root / "xbs/pool/manifest.json").read_text())["sha256"]
    (root / "xbs/eval/DONE").unlink()
    ST.convert("xbs", [str(good)], root, W125)
    assert json.loads((root / "xbs/pool/manifest.json").read_text())["sha256"] == man_pool
    with pytest.raises(SystemExit, match="never rebuilt"):
        ST.convert("xbs", [str(good)], root, W125)
    # pairs draw from the converted pool against the held-out QCD
    man = ST.pair("xbs", root, n_train=300, n_val=50)
    assert man["signal_files"] == [str(root / "xbs/pool/Xbs_0.parquet")] and man["train_draw"]["n_per_class"] <= 300
    _readable_by_weaver([str(root / "pairs/xbs" / s) for s in man["train_shards"]])


def test_a_sample_below_a5s_floor_is_refused(jc2, tmp_path):
    good = tmp_path / "Xbs_0.parquet"
    _outside_sample(good, np.random.default_rng(1), 3000)       # about 600 evaluation jets
    rep = ST.check([str(good)], W125)
    assert len(rep["problems"]) == 1 and rep["problems"][0].startswith("A5 floor")
    assert 0 < rep["eval_in_window"] < ST.X.A5_EVAL_FLOOR == 10_000
    with pytest.raises(SystemExit, match="A5 floor"):
        ST.convert("xbs", [str(good)], tmp_path / "xbx", W125)
    # the window decides it: almost none of a 125 GeV peak lies in Sophon's 150-230 GeV
    assert ST.check([str(good)], (150.0, 230.0))["eval_in_window"] < rep["eval_in_window"] / 10


@pytest.mark.parametrize("zero,why", [
    (ST.TYPE_FLAGS, "no_type_flag"),                         # the top-tagging inputs: every flag off
    (ST.DISPLACEMENT, "charged_no_displacement"),            # the q/g inputs: charged, displacement 0
    (("part_charge",), "flagged_charged_without_charge"),
    (("part_deta",), "part_deta"),
])
def test_a_zero_filled_sample_is_refused(jc2, tmp_path, monkeypatch, zero, why):
    monkeypatch.setattr(ST.X, "A5_EVAL_FLOOR", 100)
    f = tmp_path / "Xbs_0.parquet"
    _outside_sample(f, np.random.default_rng(1), 3000, zero=zero)
    rep = ST.check([str(f)], W125)
    assert [p.split(": zero-fill: ")[1].split(" = ")[0] for p in rep["problems"]] == [why]
    assert rep["zero_fill"][why] == 1.0 and rep["per_file"][str(f)]["zero_fill"][why] == 1.0
    with pytest.raises(SystemExit, match="does not carry"):
        ST.convert("xbs", [str(f)], tmp_path / "xbx", W125)


def test_a_partly_zero_filled_file_is_refused(jc2, tmp_path, monkeypatch):
    """Each file on its own: 30 % of one file's part_deta at 0 fails, though over both files
    no branch is 0 on every particle."""
    monkeypatch.setattr(ST.X, "A5_EVAL_FLOOR", 100)
    rng = np.random.default_rng(3)
    good, bad = tmp_path / "Xbs_0.parquet", tmp_path / "Xbs_1.parquet"
    _outside_sample(good, rng, 3000)
    t = _jets(rng, 3000, 0, msd=rng.normal(125, 10, 3000)).drop(["jet_label", "aux_unused"])
    col = t.column("part_deta").combine_chunks()
    v = col.values.to_numpy().copy()
    v[rng.random(v.size) < 0.3] = 0
    pq.write_table(t.set_column(t.schema.get_field_index("part_deta"), "part_deta",
                                pa.ListArray.from_arrays(col.offsets, pa.array(v))), bad)
    rep = ST.check([str(good), str(bad)], W125)
    assert len(rep["problems"]) == 1 and rep["problems"][0].startswith(f"{bad}: zero-fill: part_deta = ")
    assert 0.25 < rep["zero_fill"]["part_deta"] < 0.35 and rep["per_file"][str(good)]["zero_fill"]["part_deta"] == 0
    assert ST.check([str(good)], W125)["problems"] == []


def test_a_flag_set_on_every_particle_and_a_few_bare_tracks_pass(jc2, tmp_path, monkeypatch):
    """JetClass-II itself: exactly one type flag per particle, and a charged particle without
    displacement at about 8e-5, under the limit."""
    monkeypatch.setattr(ST.X, "A5_EVAL_FLOOR", 100)
    rng = np.random.default_rng(2)
    t = _jets(rng, 3000, 0, msd=rng.normal(125, 10, 3000))
    n = len(t.column("part_px").combine_chunks().values)
    kind = rng.integers(0, 5, n)
    kind[:2] = 0                                             # two charged hadrons with no displacement
    for i, c in enumerate(ST.TYPE_FLAGS):
        col = t.column(c).combine_chunks()
        t = t.set_column(t.schema.get_field_index(c), c,
                         pa.ListArray.from_arrays(col.offsets, pa.array((kind == i).astype(np.float32))))
    assert ST.zero_fill(t)["no_type_flag"] == 0
    d = {c: t.column(c).combine_chunks() for c in ST.DISPLACEMENT}
    for c, col in d.items():
        v = col.values.to_numpy().copy()
        v[:2] = 0
        t = t.set_column(t.schema.get_field_index(c), c, pa.ListArray.from_arrays(col.offsets, pa.array(v)))
    zf = ST.zero_fill(t)
    assert zf["charged_no_displacement"] == pytest.approx(2 / n) and zf["flagged_charged_without_charge"] == 0
    assert set(zf) == {"no_type_flag", "flagged_charged_without_charge", "charged_no_displacement",
                       "part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi"}
    f = tmp_path / "Xbs_0.parquet"
    pq.write_table(t.drop(["jet_label", "aux_unused"]), f)
    rep = ST.check([str(f)], W125)
    assert rep["problems"] == [] and 0 < rep["zero_fill"]["charged_no_displacement"] < ST.ZERO_FILL_MAX


def test_other_branch_names_are_mapped_before_the_check(jc2, tmp_path, monkeypatch):
    monkeypatch.setattr(ST.X, "A5_EVAL_FLOOR", 100)
    f = tmp_path / "Xbs_0.parquet"
    _outside_sample(f, np.random.default_rng(1), 3000, rename={"jet_pt": "fj_pt", "part_px": "pfcand_px"})
    rep = ST.check([str(f)], W125)
    assert any("'jet_pt'" in p and "'part_px'" in p for p in rep["problems"])
    monkeypatch.setattr(ST, "BRANCH_MAP", {"fj_pt": "jet_pt", "pfcand_px": "part_px"})
    assert ST.check([str(f)], W125)["problems"] == []
    root = tmp_path / "xbx"
    ST.convert("xbs", [str(f)], root, W125)
    pool = pq.read_table(root / "xbs/pool/Xbs_0.parquet")
    assert {"jet_pt", "part_px"} <= set(pool.column_names) and "fj_pt" not in pool.column_names
    _readable_by_weaver([str(root / "xbs/pool/Xbs_0.parquet")])


@pytest.mark.parametrize("ext", ["parquet", "root"])
def test_inspect_lists_trees_branches_types_and_entries(tmp_path, capsys, ext):
    f = tmp_path / f"Xbs_0.{ext}"
    _outside_sample(f, np.random.default_rng(1), 500)
    assert ST.main(["inspect", "--file", str(f)]) == 0
    r = json.loads(capsys.readouterr().out)
    (tree, info), = r["trees"].items()
    assert info["entries"] == 500 and "jet_pt" in info["branches"] and "event_weight" in info["branches"]
    assert tree == ("parquet" if ext == "parquet" else "tree")
