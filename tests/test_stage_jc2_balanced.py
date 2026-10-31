"""The class-balanced JetClass-II staging plan (experiments/FT/stage_jc2_balanced.py), on a synthetic census."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_jc2_balanced", ROOT / "experiments/FT/stage_jc2_balanced.py")
sjb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sjb)
N = sjb.N_NATIVE


def _census(per_class, offset):
    lab = np.concatenate([np.full(n, c) for c, n in enumerate(per_class)]).astype(np.int64)
    src = offset * sjb.SRC_MULT + np.arange(lab.size, dtype=np.int64)
    return src, lab


def _plan():
    rich = np.full(N, 7_000)
    rich[[5, 170, 174]] = [40, 300, 9]                  # rare classes, like QCD bcs
    census = {"ft_heldout": _census(rich, 0), "ft_test": _census(rich, 1)}
    groups = {"a_vs_b": ((0,), (1, 2)), "rare_vs_c": ((174,), (3,))}
    old = (sjb.TEST_PER_CLASS, sjb.PAIR_TRAIN_PER_SIDE, sjb.PAIR_TEST_PER_SIDE)
    sjb.TEST_PER_CLASS, sjb.PAIR_TRAIN_PER_SIDE, sjb.PAIR_TEST_PER_SIDE = 5_000, 1_000, 2_000
    try:
        return sjb.plan(census, groups), census
    finally:
        sjb.TEST_PER_CLASS, sjb.PAIR_TRAIN_PER_SIDE, sjb.PAIR_TEST_PER_SIDE = old


def _labels(ids, census):
    src, lab = census
    pos = {int(s): i for i, s in enumerate(src)}
    return lab[[pos[int(i)] for i in ids]]


def test_test_set_is_balanced_with_rare_classes_whole():
    p, c = _plan()
    counts = np.bincount(_labels(p["test.parquet"], c["ft_test"]), minlength=N)
    assert counts[5] == 40 and counts[170] == 300 and counts[174] == 9
    assert set(np.delete(counts, [5, 170, 174])) == {5_000}


def test_training_sets_are_equal_per_class_and_nested():
    p, c = _plan()
    prev = set()
    for n in sjb.SIZES:
        ids = p[f"train_N{n}_s1.parquet"]
        counts = np.bincount(_labels(ids, c["ft_heldout"]), minlength=N)
        full = np.delete(counts, [5, 170, 174])
        assert full.max() - full.min() <= 1 and full.min() == min(n // N, 7_000)
        assert counts[174] in (min(9, n // N), min(9, n // N + 1))
        assert prev <= set(ids.tolist())               # nested
        prev = set(ids.tolist())
    assert len(p["train_N1000_s1.parquet"]) == 1_000   # every class has >= 6 jets at this size


def test_val_is_disjoint_from_training_and_test_from_heldout():
    p, c = _plan()
    assert not set(p["val.parquet"].tolist()) & set(p["train_N1000000_s1.parquet"].tolist())
    assert (p["test.parquet"] // sjb.SRC_MULT == 1).all()
    assert (p["train_N1000000_s1.parquet"] // sjb.SRC_MULT == 0).all()


def test_pairs_are_capped_per_side_natural_within_side():
    p, c = _plan()
    tr = p["_pairs"]["train"]["a_vs_b"]
    assert len(tr[0]) == 1_000 and len(tr[1]) == 1_000
    b = np.bincount(_labels(tr[1], c["ft_heldout"]), minlength=N)
    assert b[1] > 300 and b[2] > 300                   # both classes of the side, ~half each
    rare = p["_pairs"]["test"]["rare_vs_c"]
    assert len(rare[0]) == 9                           # all of a side smaller than the cap
    union = set(p["pairs_test.parquet"].tolist())
    assert all(set(s.tolist()) <= union for side in p["_pairs"]["test"].values() for s in side)
