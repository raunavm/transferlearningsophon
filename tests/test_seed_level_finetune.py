"""S3 and S4 in experiments/STATS/seed_level.py: fine-tuning on JetClass and
JetClass-II, read from synthetic leg-metrics files."""
import contextlib
import importlib.util
import io
import json
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level", REPO / "experiments/STATS/seed_level.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

STEMS = {"l188": 188, "l162": 162, "r42q1": 43, "r16q1": 17}
SIZES = ("N1000", "N10000", "N100000", "N1000000")
# per stem and seed, with an order across stems that changes from seed to seed,
# so a level with no real effect has no consistent rank either
JITTER = {"l188": [0.010, -0.012, 0.004, 0.008, -0.006], "l162": [-0.004, 0.009, 0.006, -0.008, 0.004],
          "r42q1": [0.006, 0.004, -0.010, 0.001, 0.008], "r16q1": [0.012, -0.008, 0.015, -0.004, 0.0]}


def doc(one_minus_auc):
    cells = {}
    for stem in STEMS:
        for s in range(1, 6):
            init = "l162-s1b" if (stem, s) == ("l162", 1) else f"{stem}-s{s}"
            cells[init] = {n: {"s1": {"macro_auc_ovr": 1 - one_minus_auc(stem, n) * math.exp(
                JITTER[stem][s - 1]), "accuracy": 0.5}} for n in SIZES}
    cells["scratch"] = {n: {"s1": {"macro_auc_ovr": 0.9, "accuracy": 0.4}} for n in SIZES}
    return {"cells": cells, "row_alignment_sha256": "ab" * 32}


def shrinking(stem, n):
    """17 and 43 worse than 188 and 162 at every size, by a gap that halves per decade."""
    base = {"N1000": 0.1, "N10000": 0.05, "N100000": 0.03, "N1000000": 0.02}[n]
    gap = {"N1000": 0.8, "N10000": 0.4, "N100000": 0.2, "N1000000": 0.1}[n]
    return base * math.exp(gap if stem in ("r42q1", "r16q1") else 0.0)


def run(tmp_path, which, fn):
    p = tmp_path / "legs.json"
    p.write_text(json.dumps(doc(fn)))
    flag = {"S4": "--finetune-jetclass2", "S3": "--finetune-jetclass"}[which]
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main([flag, str(p), "--out", str(tmp_path / "o")]) == 0
    return json.loads((tmp_path / "o" / "s3_s4_finetune.json").read_text())["secondary"][which]


def test_s4_confirms_all_three_clauses_on_a_gap_that_shrinks_but_stays(tmp_path):
    r = run(tmp_path, "S4", shrinking)
    assert [c["verdict"] for c in r["clauses"]] == ["confirmed"] * 3
    g = r["gap_17_minus_188"]
    assert g["N1000"] > g["N10000"] > g["N100000"] > g["N1000000"] > 0
    assert r["reference_rows"]["N1000"]["scratch"]["macro_auc"] == 0.9


def test_s4_says_not_confirmed_when_the_small_size_shows_nothing(tmp_path):
    def flat_at_1k(stem, n):
        return 0.1 if n == "N1000" else shrinking(stem, n)
    v = [c["verdict"] for c in run(tmp_path, "S4", flat_at_1k)["clauses"]]
    assert v[0] == "not confirmed" and v[2] == "confirmed"


def test_s3_finds_both_steps_only_when_both_exist(tmp_path):
    def two_steps(stem, n):
        return 0.05 * math.exp({"l188": 0, "l162": 0, "r42q1": 0.3, "r16q1": 0.6}[stem])
    r = run(tmp_path, "S3", two_steps)
    assert [c["verdict"] for c in r["clauses"]] == ["confirmed", "confirmed"]
    def one_step(stem, n):
        return 0.05 * math.exp({"l188": 0, "l162": 0, "r42q1": 0.0, "r16q1": 0.6}[stem])
    (tmp_path / "b").mkdir()
    r = run(tmp_path / "b", "S3", one_step)
    assert [c["verdict"] for c in r["clauses"]] == ["not confirmed", "confirmed"]


def test_a_readout_without_macro_auc_is_refused(tmp_path):
    p = tmp_path / "legs.json"
    d = doc(shrinking)
    for per_n in d["cells"].values():
        for per_s in per_n.values():
            per_s["s1"].pop("macro_auc_ovr")
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="macro-auc"):
        S.load_ft_leg(p, "jc1")
