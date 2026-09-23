"""experiments/AOJ/merge_shards.py: ten shards must join into what one pass over
all eighty files would have written, or refuse."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merge_shards", REPO / "experiments/AOJ/merge_shards.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def shard(root, i, n, models=("a", "b"), sha="s", flags=(), event0=None):
    d = root / f"shard{i}"
    d.mkdir()
    ev = np.arange(n, dtype=np.int64) + (i * 1000 if event0 is None else event0) + 2**33
    np.savez(d / "jets.npz", run=np.full(n, 280000 + i), lumi=np.ones(n, dtype=np.int64), event=ev,
             jet_sdmass=np.linspace(50, 250, n).astype(np.float32))
    for m in models:
        np.savez(d / f"scores_{m}.npz", three_prong_logodds=np.full(n, i, dtype=np.float16))
        (d / f"scores_{m}.json").write_text(json.dumps(dict(checkpoint_sha256=f"{sha}-{m}")))
    (d / "closure.json").write_text(json.dumps(dict(hard_flags=list(flags))))
    return d


def test_shards_are_concatenated_in_order_with_exact_event_numbers(tmp_path):
    ds = [shard(tmp_path, i, 5 + i) for i in range(3)]
    m = ms.merge(ds, tmp_path / "out")
    j = np.load(tmp_path / "out" / "jets.npz")
    assert m["n_jets"] == 18 and m["jets_per_shard"] == [5, 6, 7] and m["models"] == ["a", "b"]
    assert j["event"].dtype == np.int64 and j["event"].max() > 2**33
    s = np.load(tmp_path / "out" / "scores_a.npz")["three_prong_logodds"]
    assert s.tolist() == [0] * 5 + [1] * 6 + [2] * 7


def test_a_model_missing_from_one_shard_is_refused_and_named(tmp_path):
    ds = [shard(tmp_path, 0, 4), shard(tmp_path, 1, 4, models=("a",))]
    with pytest.raises(SystemExit, match=r"\['b'\] scored in some shards only: shard1 lacks"):
        ms.merge(ds, tmp_path / "out")


def test_a_model_scored_from_two_checkpoints_is_refused(tmp_path):
    ds = [shard(tmp_path, 0, 4), shard(tmp_path, 1, 4, sha="other")]
    with pytest.raises(SystemExit, match="different checkpoints"):
        ms.merge(ds, tmp_path / "out")


def test_an_event_in_two_shards_is_refused(tmp_path):
    ds = [shard(tmp_path, 0, 4, event0=0), shard(tmp_path, 1, 4, event0=0)]
    np.savez(ds[1] / "jets.npz", **{**dict(np.load(ds[0] / "jets.npz"))})
    with pytest.raises(SystemExit, match="more than once"):
        ms.merge(ds, tmp_path / "out")


def test_score_rows_that_disagree_with_the_jets_are_refused(tmp_path):
    ds = [shard(tmp_path, 0, 4)]
    np.savez(ds[0] / "scores_a.npz", three_prong_logodds=np.zeros(3, dtype=np.float16))
    with pytest.raises(SystemExit, match="rows, jets.npz has 4"):
        ms.merge(ds, tmp_path / "out")


def test_closure_hard_flags_survive_as_a_union_tagged_by_shard(tmp_path):
    ds = [shard(tmp_path, 0, 4, flags=("d0 units",)), shard(tmp_path, 1, 4)]
    ms.merge(ds, tmp_path / "out")
    got = json.loads((tmp_path / "out" / "closure.json").read_text())["hard_flags"]
    assert got == ["shard0: d0 units"]


def test_a_shard_without_its_closure_check_is_refused(tmp_path):
    ds = [shard(tmp_path, 0, 4)]
    (ds[0] / "closure.json").unlink()
    with pytest.raises(SystemExit, match="no closure.json"):
        ms.merge(ds, tmp_path / "out")
