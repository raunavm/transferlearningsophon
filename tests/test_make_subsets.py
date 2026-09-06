"""CI for experiments/FT/make_subsets.py on synthetic files.

What a data-efficiency curve needs from its subsets, checked directly:
exact sizes, nesting (N=100 is the prefix of N=1000), the study's selection
applied, determinism under the seed, per-family proportional sampling, and for
JetClass-I a balanced class mix in the largest file.

Run:  python3 -m pytest tests/test_make_subsets.py -v
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location(
        "make_subsets_under_test", ROOT / "experiments" / "FT" / "make_subsets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jc2_file(path: pathlib.Path, n: int, seed: int, family: str):
    rng = np.random.default_rng(seed)
    pt = rng.uniform(100.0, 3000.0, n)              # some outside the window
    msd = rng.uniform(0.0, 600.0, n)
    label = rng.integers(0, 188, n)
    part = pa.array([list(rng.normal(size=int(k))) for k in rng.integers(1, 6, n)])
    t = pa.table({"jet_pt": pt, "jet_sdmass": msd, "jet_label": label,
                  "family": pa.array([family] * n), "part_px": part})
    pq.write_table(t, path)


@pytest.fixture(scope="module")
def jc2_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("jc2")
    files = []
    for fam, nf in (("Res2P", 4), ("Res34P", 8), ("QCD", 4)):
        for i in range(nf):
            p = d / f"{fam}_{i:04d}.parquet"
            _jc2_file(p, 3000, seed=hash((fam, i)) % 10_000, family=fam)
            files.append(str(p))
    return d, files


def test_jc2_subsets_are_exact_nested_selected_and_deterministic(ms, jc2_dir, tmp_path):
    d, files = jc2_dir
    train = [f for f in files if not f.endswith("_0000.parquet")]
    val = [f for f in files if f.endswith("_0000.parquet")]
    sizes, seeds = [100, 1000], [1, 2]
    out = tmp_path / "a"
    m = ms.build_jc2(train, val, out, sizes, seeds, n_files=8, take=0.5,
                     val_size=300, n_val_files=3)
    ms.finish(out, m)
    assert (out / "DONE").exists() and (out / "manifest.json").exists()
    for s in seeds:
        big = pq.read_table(out / f"train_N1000_s{s}.parquet")
        small = pq.read_table(out / f"train_N100_s{s}.parquet")
        assert big.num_rows == 1000 and small.num_rows == 100
        assert small.equals(big.slice(0, 100)), "N=100 must be the prefix of N=1000"
        pt, msd = big.column("jet_pt").to_numpy(), big.column("jet_sdmass").to_numpy()
        assert ((pt > 200) & (pt < 2500) & (msd > 20) & (msd < 500)).all()
        assert big.schema.equals(pq.read_schema(train[0])), "rows are copied whole"
    v = pq.read_table(out / "val.parquet")
    assert v.num_rows == 300
    # different seeds, different jets
    a = pq.read_table(out / "train_N100_s1.parquet").column("jet_pt").to_numpy()
    b = pq.read_table(out / "train_N100_s2.parquet").column("jet_pt").to_numpy()
    assert not np.array_equal(a, b)
    # same seed twice: identical bytes
    out2 = tmp_path / "b"
    ms.finish(out2, ms.build_jc2(train, val, out2, sizes, seeds, n_files=8, take=0.5,
                                 val_size=300, n_val_files=3))
    for p in sorted(out.glob("*.parquet")):
        assert pq.read_table(p).equals(pq.read_table(out2 / p.name)), p.name
    # per-family proportional file choice: 8 files over 3:7:3 train files -> 2:4:2
    man = json.loads((out / "manifest.json").read_text())
    fams = [pathlib.Path(f).name.split("_")[0] for f in man["per_seed"]["1"]["files"]]
    assert sorted(fams) == sorted(["Res2P"] * 2 + ["Res34P"] * 4 + ["QCD"] * 2)
    assert man["outputs"]["train_N1000_s1.parquet"] == 1000


def test_jc2_refuses_a_pool_smaller_than_the_largest_size(ms, jc2_dir, tmp_path):
    d, files = jc2_dir
    with pytest.raises(SystemExit, match="pool has"):
        ms.build_jc2(files[:2], files[:1], tmp_path / "c", [100_000], [1],
                     n_files=2, take=0.1, val_size=10, n_val_files=1)


def test_choose_files_never_drops_a_family(ms):
    fams = {"Res2P": ["a"] * 200, "Res34P": ["b"] * 860, "QCD": ["c"] * 280}
    got = ms.choose_files(fams, 60, np.random.default_rng(0))
    counts = {f: sum(1 for g in got if g == {"Res2P": "a", "Res34P": "b", "QCD": "c"}[f])
              for f in fams}
    assert counts == {"Res2P": 9, "Res34P": 39, "QCD": 13}
    got1 = ms.choose_files(fams, 1, np.random.default_rng(0))
    assert len(got1) == 3, "at least one file per family, always"


def _fake_tree(path: str):
    """What make_subsets.read_root would return for a JetClass-I file: one
    record array, flat label columns, a jagged constituent column. Content is a
    deterministic function of the path."""
    import awkward as ak
    name = pathlib.Path(path).stem                       # e.g. HToBB_001
    cls, idx = name.rsplit("_", 1)
    n = 400 if "train" in path else 200
    rng = np.random.default_rng(abs(hash((cls, idx))) % 10_000)
    labels = {f"label_{c}": np.zeros(n, dtype=np.int32) for c in
              ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]}
    labels[{"HToBB": "label_Hbb", "HToCC": "label_Hcc", "ZJetsToNuNu": "label_QCD"}[cls]][:] = 1
    part = ak.Array([list(rng.normal(size=int(k))) for k in rng.integers(1, 6, n)])
    return ak.Array({"jet_pt": rng.uniform(500, 1000, n), **labels, "part_px": part})


def test_jc1_subsets_are_balanced_nested_and_parquet_round_trips(ms, tmp_path):
    ak = pytest.importorskip("awkward")
    classes = ["HToBB", "HToCC", "ZJetsToNuNu"]
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir()
    val.mkdir()
    for c in classes:
        for i in range(3):
            (train / f"{c}_{i:03d}.root").touch()       # glob targets only
        (val / f"{c}_000.root").touch()
    ms.read_root = _fake_tree                            # uproot is the smoke's job
    out = tmp_path / "out"
    m = ms.build_jc1(str(train), str(val), out, [30, 300], [1], files_per_class=2,
                     val_per_class=50, classes=classes)
    ms.finish(out, m)
    big = ak.from_parquet(out / "train_N300_s1.parquet")
    small = ak.from_parquet(out / "train_N30_s1.parquet")
    assert len(big) == 300 and len(small) == 30
    assert ak.to_list(small) == ak.to_list(big[:30]), "nested prefix"
    for col in ("label_Hbb", "label_Hcc", "label_QCD"):
        assert int(ak.sum(big[col])) == 100, f"{col}: balanced at N_max"
    assert "part_px" in big.fields and ak.num(big["part_px"], axis=1).tolist()[0] >= 1
    assert len(ak.from_parquet(out / "val.parquet")) == 150
    man = json.loads((out / "manifest.json").read_text())
    assert man["per_class_rows"] == 100 and man["outputs"]["val.parquet"] == 150
