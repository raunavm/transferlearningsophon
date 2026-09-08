"""The benchmark subsets must be an unbiased draw, not a prefix of the file.

The top reference dataset stores its rows in BLOCKS OF 10 that share a label
(measured on the staged top_train.parquet: every run length is a multiple of 10,
longest run 180, 1,211,000 rows at 50.0% top overall). A prefix of that file is
therefore class-skewed -- 46.0% top over the first 1,000 rows -- so cutting the
small-N points of a data-scaling curve straight off the front of the file would
hand the smallest, headline cell a biased training set. Nothing would error:
the model trains, the loss falls, and the curve is quietly wrong at its most
important point.

These tests rebuild that block structure synthetically and check that the
shuffle removes the skew, that the unshuffled prefix really is skewed (so the
first test cannot pass vacuously), and that the q/g chunk split reproduces the
1.6M/200k/200k that ParticleNet (1902.08570) calls the recommended splitting.
"""
import importlib.util
import json
import pathlib

import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_subsets", ROOT / "experiments" / "FT" / "make_subsets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ms = _load()
BLOCK = 10


def block_structured(n_rows: int, seed: int = 0) -> pa.Table:
    """A stand-in for top_train.parquet: labels in RANDOM blocks of 10, 50/50
    overall. The blocks must be randomly labelled, not alternating -- adjacent
    blocks sharing a label is what produces the measured runs of 20..180 and the
    small-N skew. Alternating blocks are balanced at every prefix and would make
    these tests pass vacuously."""
    n_blocks = n_rows // BLOCK
    lab = np.repeat(np.arange(n_blocks) % 2, 1)
    np.random.default_rng(seed).shuffle(lab)
    lab = np.repeat(lab, BLOCK).astype(np.int64)
    return pa.table({"label": pa.array(lab),
                     "jet_pt": pa.array(np.arange(n_rows, dtype=np.float32))})


def sorted_pool(n_rows: int) -> pa.Table:
    """The worst case: every signal jet after every background jet."""
    lab = np.concatenate([np.zeros(n_rows // 2), np.ones(n_rows - n_rows // 2)])
    return pa.table({"label": pa.array(lab.astype(np.int64)),
                     "jet_pt": pa.array(np.arange(n_rows, dtype=np.float32))})


def test_block_structure_inflates_the_class_fraction_variance():
    """The defect signature, and the guard on every test below: blocks of 10 make
    a prefix's class fraction sqrt(10) times noisier than an iid draw. That is
    why the first 1,000 rows of top_train came out at 46.0% top rather than
    50.0 +/- 1.6%. If this fails, the fixture no longer has the defect and the
    balance tests below would pass vacuously."""
    n = 1_000
    fr = [block_structured(50_000, seed=s).column("label").to_numpy()[:n].mean()
          for s in range(200)]
    iid_sd = 0.5 / np.sqrt(n)                      # 0.0158
    assert np.std(fr) > 3 * iid_sd, "fixture is not block-structured"
    assert min(fr) < 0.47, "no seed produced a skewed prefix"


def test_shuffled_prefixes_are_balanced_at_every_size():
    pool = block_structured(200_000)
    rng = ms.rng_for(1, "train")
    subs = ms.nested_prefixes(pool, [1_000, 10_000, 100_000], rng)
    for n, tbl in subs.items():
        frac = tbl.column("label").to_numpy().mean()
        # 5 sigma of a fair 50/50 draw at this n.
        tol = 5 * 0.5 / np.sqrt(n)
        assert abs(frac - 0.5) < max(tol, 0.01), f"N={n}: label_frac {frac:.4f}"


def test_shuffle_destroys_the_block_runs():
    """The defect is the run structure; check it is gone, not just the mean."""
    pool = block_structured(100_000)
    rng = ms.rng_for(1, "train")
    lab = ms.nested_prefixes(pool, [100_000], rng)[100_000].column("label").to_numpy()
    runs = [len(list(g)) for _, g in __import__("itertools").groupby(lab)]
    assert max(runs) < BLOCK * 3, f"longest run {max(runs)} still looks blocked"
    assert not all(r % BLOCK == 0 for r in runs)


def test_prefixes_are_nested():
    pool = block_structured(50_000)
    rng = ms.rng_for(2, "train")
    subs = ms.nested_prefixes(pool, [1_000, 10_000], rng)
    small = subs[1_000].column("jet_pt").to_numpy()
    large = subs[10_000].column("jet_pt").to_numpy()
    assert np.array_equal(small, large[:1_000]), "N=1e3 must be the prefix of N=1e4"


def test_qg_split_is_the_recommended_one():
    tr, va, te = ms.QG_TRAIN_CHUNKS, ms.QG_VAL_CHUNKS, ms.QG_TEST_CHUNKS
    assert sorted(tr + va + te) == list(range(20)), "the 20 chunks must be used exactly once"
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    rows = ms.QG_CHUNK_ROWS
    assert (len(tr) * rows, len(va) * rows, len(te) * rows) == (1_600_000, 200_000, 200_000)


def test_bench_test_files_are_the_held_out_ones():
    top = ms.bench_test_files("top", "/data/finetune/top")
    assert top == ["/data/finetune/top/top_test.parquet"]
    qg = ms.bench_test_files("qg", "/data/finetune/qg")
    assert qg == [f"/data/finetune/qg/qg_chunk{i}.parquet" for i in ms.QG_TEST_CHUNKS]
    # A test file must never be a training chunk.
    assert not set(ms.QG_TEST_CHUNKS) & set(ms.QG_TRAIN_CHUNKS)


def test_build_bench_rejects_an_unshuffled_pool(tmp_path, monkeypatch):
    """The guard must fire if nested_prefixes is ever made to stop shuffling."""
    src = tmp_path / "top"
    src.mkdir()
    pq.write_table(sorted_pool(20_000), src / "top_train.parquet")
    pq.write_table(sorted_pool(4_000), src / "top_val.parquet")
    monkeypatch.setattr(ms, "nested_prefixes",
                        lambda t, sizes, rng: {n: t.slice(0, n) for n in sizes})
    with pytest.raises(SystemExit) as e:
        ms.build_bench("top", str(src), tmp_path / "out", [1_000], [1], 1_000)
    assert "not shuffled" in str(e.value)


def test_build_bench_writes_nested_balanced_subsets(tmp_path):
    src = tmp_path / "top"
    src.mkdir()
    pq.write_table(block_structured(60_000), src / "top_train.parquet")
    pq.write_table(block_structured(8_000), src / "top_val.parquet")
    out = tmp_path / "out"
    m = ms.build_bench("top", str(src), out, [1_000, 10_000], [1, 2], 4_000)
    assert m["dataset"] == "top" and m["test_files"] == [str(src / "top_test.parquet")]
    for seed in (1, 2):
        for n in (1_000, 10_000):
            assert (out / f"train_N{n}_s{seed}.parquet").exists()
            assert abs(m["per_seed"][str(seed)]["label_frac"][str(n)] - 0.5) <= 0.05
    a = pq.read_table(out / "train_N1000_s1.parquet").column("jet_pt").to_numpy()
    b = pq.read_table(out / "train_N1000_s2.parquet").column("jet_pt").to_numpy()
    assert not np.array_equal(a, b), "different fine-tuning seeds must draw different jets"
    assert m["val"]["rows"] == 4_000


def test_the_emitted_job_builds_the_grid_item_16e_specifies():
    """DECISIONS_PENDING item 16(e), RESOLVED 2026-09-07: "add N = 1e3 to every
    sweep". N=1e3 is also the cell the block structure would have damaged most --
    a contiguous window of 1,000 top jets has a class-fraction SD of 0.048
    (measured), so nearly a 10-point imbalance swing was routine there.
    """
    spec = ROOT / "experiments" / "FT" / "k8s" / "job-ft-subsets-bench-raunav.yaml"
    if not spec.exists():
        pytest.skip("bench subsets spec not generated")
    args = yaml.safe_load(spec.read_text())["spec"]["template"]["spec"][
        "containers"][0]["args"][0]
    # docs/PRD_PLAN.md 4.1 gives each benchmark its OWN N_max: top's is
    # 2606.14870's 1.2e6 and q/g's is its whole 1.6M training split. A single
    # shared grid would silently retire the top end of both community tables.
    grids = dict(re.findall(r"(top|qg)\)\s+echo \"([\d ]+)\"", args))
    assert [int(x) for x in grids["top"].split()] == [1_000, 10_000, 100_000, 1_200_000]
    assert [int(x) for x in grids["qg"].split()] == [1_000, 10_000, 100_000, 1_600_000]
    # q/g N_max must be exactly the chunk-aligned training split, not a round number.
    assert int(grids["qg"].split()[-1]) == len(ms.QG_TRAIN_CHUNKS) * ms.QG_CHUNK_ROWS
    assert "--sizes $(sizes_for ${D})" in args
    assert "--dataset ${D}" in args and "for D in top qg" in args


def _tiny_qg(src, n_chunks=2, rows=200):
    """A miniature EnergyFlow-shaped source: chunked parquet, 50/50 labels."""
    src.mkdir(parents=True, exist_ok=True)
    for c in range(n_chunks):
        lab = np.tile([1, 0], rows // 2)
        t = pa.table({"label": pa.array(lab, pa.int64()),
                      "x": pa.array(np.arange(rows, dtype=np.float32))})
        pq.write_table(t, src / f"qg_chunk{c}.parquet")
    return src


def test_done_refuses_to_reuse_subsets_built_for_a_different_grid(tmp_path):
    """A bare DONE check made a grid change a silent no-op.

    The failure this closes is not hypothetical: the q/g source has to be
    re-staged (the lepton charge sign was inverted), and without this the
    rebuild would reuse the subsets built from the superseded staging and
    report success.
    """
    out = tmp_path / "sub"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(
        {"mode": "bench", "dataset": "qg", "sizes": [1000, 10000],
         "seeds": [1, 2, 3], "src": "/data/finetune/qg"}))
    (out / "DONE").write_text("2026-09-07T00:00:00Z\n")

    # same request -> no-op, exit 0
    assert ms.main(["bench", "--dataset", "qg", "--src", "/data/finetune/qg",
                    "--out", str(out), "--sizes", "1000", "10000",
                    "--seeds", "1", "2", "3", "--val-size", "10"]) == 0

    # different grid -> refuse
    with pytest.raises(SystemExit) as e:
        ms.main(["bench", "--dataset", "qg", "--src", "/data/finetune/qg",
                 "--out", str(out), "--sizes", "1000", "10000", "100000",
                 "--seeds", "1", "2", "3", "--val-size", "10"])
    assert "different parameters" in str(e.value)

    # different source (the re-stage case) -> refuse
    with pytest.raises(SystemExit) as e:
        ms.main(["bench", "--dataset", "qg", "--src", "/data/finetune/qg_v2",
                 "--out", str(out), "--sizes", "1000", "10000",
                 "--seeds", "1", "2", "3", "--val-size", "10"])
    assert "different parameters" in str(e.value)


def test_done_without_a_manifest_is_an_error(tmp_path):
    out = tmp_path / "sub"
    out.mkdir()
    (out / "DONE").write_text("x\n")
    with pytest.raises(SystemExit) as e:
        ms.main(["bench", "--dataset", "qg", "--src", "/data/finetune/qg",
                 "--out", str(out), "--sizes", "1000", "--seeds", "1",
                 "--val-size", "10"])
    assert "unknown state" in str(e.value)


def test_legacy_manifest_without_src_warns_but_does_not_falsely_refuse(tmp_path, capsys):
    """`src` postdates the manifests on the PVC.

    Treating its absence as a mismatch would condemn every subset directory
    built before the field existed -- a false alarm that teaches the reader to
    ignore the check. The unverifiable field is named instead.
    """
    out = tmp_path / "sub"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(
        {"mode": "bench", "dataset": "top", "sizes": [1000], "seeds": [1]}))
    (out / "DONE").write_text("x\n")
    assert ms.main(["bench", "--dataset", "top", "--src", "/data/finetune/top",
                    "--out", str(out), "--sizes", "1000", "--seeds", "1",
                    "--val-size", "10"]) == 0
    assert "could not be checked" in capsys.readouterr().out

    # a field that IS present and differs still refuses
    (out / "manifest.json").write_text(json.dumps(
        {"mode": "bench", "dataset": "top", "sizes": [1000, 10000], "seeds": [1]}))
    with pytest.raises(SystemExit) as e:
        ms.main(["bench", "--dataset", "top", "--src", "/data/finetune/top",
                 "--out", str(out), "--sizes", "1000", "--seeds", "1",
                 "--val-size", "10"])
    assert "different parameters" in str(e.value)
