"""experiments/EVAL/dup_check_v2.py (PI 2026-11-25, LEDGER §28 1.2): a pair of runs is a duplicate when it shares a
checkpoint, a fine-tuned checkpoint or its scores; two different runs are not."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dup_check_v2", ROOT / "experiments/EVAL/dup_check_v2.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _rec(s, ck):
    return {"scores_sha256": D.sha_arr(s), "_sample": D.sample(s), "checkpoint": {"sha256_file": ck}}


def test_pairs_flag_shared_checkpoints_and_scores_only():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(500, 3)), rng.normal(size=(500, 3))
    got = D.pairs({"x": _rec(a, "c1"), "y": _rec(b, "c2"), "z": _rec(a.copy(), "c3"), "w": _rec(b + 1, "c2")})
    by = {(p["a"], p["b"]): p for p in got}
    assert len(got) == 6
    assert not any(by[("x", "y")][k] for k in ("same_checkpoint", "same_scores")) and by[("x", "y")]["frac_identical"] == 0
    assert by[("x", "z")]["same_scores"] and by[("x", "z")]["frac_identical"] == 1 and not by[("x", "z")]["same_checkpoint"]
    assert by[("w", "y")]["same_checkpoint"] and not by[("w", "y")]["same_scores"]
