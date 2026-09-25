"""C2, C3 and S5 in experiments/STATS/seed_level.py: the two community benchmarks,
read from synthetic bench_metrics.py outputs beside the planted probe ladder."""
import contextlib
import importlib.util
import io
import json
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _mod("seed_level", "experiments/STATS/seed_level.py")
T = _mod("test_seed_level_fixtures", "tests/test_seed_level.py")

STEMS = ("l188", "l162", "r42q1", "r16q1")
N_MAX = {"qg": "N1600000", "top": "N1200000"}
GPU = "NVIDIA GeForce RTX 3090"
# per stem and seed: identical noise across levels would give zero-variance
# paired differences, which the tests rightly refuse to evaluate
NOISE = {"l188": [0.010, -0.012, 0.004, 0.008, -0.006], "l162": [-0.004, 0.009, 0.006, -0.008, 0.004],
         "r42q1": [0.006, 0.004, -0.010, 0.001, 0.008], "r16q1": [0.012, -0.008, 0.015, -0.004, 0.0]}


def init_name(stem, s):
    return "l162-s1b" if (stem, s) == ("l162", 1) else f"{stem}-s{s}"


def predicted(dataset, stem, s):
    """q/g: 188 ~ 162 at 40, 43 ~ 17 at 30 (the step at 162 -> 43); top: all at 400."""
    base = {"qg": {"l188": 40, "l162": 40, "r42q1": 30, "r16q1": 30},
            "top": dict.fromkeys(STEMS, 400)}[dataset][stem]
    return base * math.exp(NOISE[stem][s - 1])


def cell(r50, gpu=GPU):
    return {"r50": r50, "r50_is_bound": False, "r50_n_bkg_pass": int(2e5 / r50), "r30": 3 * r50,
            "r30_is_bound": False, "auc": 0.9, "accuracy": 0.85, "gpu": gpu}


def bench_doc(r50_of=predicted, test_set="pythia", gpu_of=lambda d, stem, s: GPU):
    cells = {}
    for d, n in N_MAX.items():
        cells[d] = {init_name(stem, s): {n: {"s1": cell(r50_of(d, stem, s), gpu_of(d, stem, s))}}
                    for stem in STEMS for s in range(1, 6)}
        cells[d]["scratch"] = {n: {f"s{i}": cell(20.0) for i in (1, 2, 3)}}
        cells[d]["l162mass-s1"] = {n: {"s1": cell(35.0)}}
    tag = "ab" if test_set == "pythia" else "cd"
    return {"test_set": test_set, "row_alignment_sha256": {"qg": tag * 32, "top": "ef" * 32},
            "script_sha256": "00" * 32, "repo_commit": "deadbeef", "cells": cells}


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    return T.write_ladder(tmp_path_factory.mktemp("ladder"), T.ladder_values(step=1.0))


def run(ladder, tmp_path, doc=None, herwig=None):
    b = tmp_path / "bench_metrics.json"
    b.write_text(json.dumps(doc or bench_doc()))
    extra = ["--bench", str(b)]
    if herwig is not None:
        h = tmp_path / "bench_metrics_herwig.json"
        h.write_text(json.dumps(herwig))
        extra += ["--bench-herwig", str(h)]
    out = tmp_path / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert S.main([str(ladder), "--out", str(out), *extra]) == 0
    return json.loads((out / "seed_level_results.json").read_text()), buf.getvalue()


def test_the_predicted_pattern_confirms_c2_in_all_four_clauses_and_c3(ladder, tmp_path):
    res, text = run(ladder, tmp_path)
    c2, c3 = res["confirmatory"]["C2"], res["confirmatory"]["C3"]
    assert c2["composite_verdict"] == "confirmed in clauses 1-4", c2["composite_verdict"]
    assert c2["argmax_step"] == [[188, 162], [43, 17]]
    assert c3["composite_verdict"] == "confirmed in clause 1", c3["composite_verdict"]
    fam = {h["test"]: h for h in res["confirmatory"]["holm_family"]}
    assert fam["C2"]["status"] == fam["C3"]["status"] == "available" and fam["C5"]["status"] == "pending"
    assert "in rejection at 50% signal efficiency" in c3["verdict"] and "AUC" not in c3["verdict"]
    assert c2["endpoint"] == S.BENCH_ENDPOINT and c2["fine_tuning_seed"] == "s1"


def test_the_endpoint_is_minus_ln_r50_and_only_the_ladder_is_tested(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(bench_doc()))
    d = S.load_bench(p, S.C2)
    assert len(d["rows"]) == 20 and set(d["reference"]) == {"scratch", "l162mass-s1"}
    r = next(r for r in d["rows"] if r["arm"] == "l162-s1b")
    assert (r["level"], r["seed"]) == (162, 1)
    assert r[S.ENDPOINT] == pytest.approx(-math.log(predicted("qg", "l162", 1)))


def test_a_seed_block_fine_tuned_on_two_gpu_models_is_dropped_whole(ladder, tmp_path):
    doc = bench_doc(gpu_of=lambda d, stem, s: "NVIDIA A10" if (stem, s) == ("r42q1", 3) else GPU)
    res, text = run(ladder, tmp_path, doc)
    c2 = res["confirmatory"]["C2"]
    assert c2["seeds_used"] == [1, 2, 4, 5] and "3" in c2["gpu_mismatched_dropped"]
    assert c2["blocks_used"] == [1, 2, 4, 5] and "seed block 3 DROPPED" in text


def test_a_step_in_the_wrong_place_fails_clause_two_only_where_it_should(ladder, tmp_path):
    def late(d, stem, s):
        base = {"l188": 40, "l162": 40, "r42q1": 40, "r16q1": 30} if d == "qg" else dict.fromkeys(STEMS, 400)
        return base[stem] * math.exp(NOISE[stem][s - 1])
    c2 = run(ladder, tmp_path, bench_doc(late))[0]["confirmatory"]["C2"]
    v = {c["n"]: c["verdict"] for c in c2["clauses"]}
    assert v[1] == "confirmed" and v[2] == "not confirmed" and v[3] == "confirmed"
    assert v[4] == "inconclusive"      # 43 vs 17 now differ by ln(40/30)


def test_a_granularity_effect_on_top_tagging_is_inconclusive_and_its_size_is_shown(ladder, tmp_path):
    def top_step(d, stem, s):
        return (predicted(d, stem, s) if d == "qg" else
                {"l188": 400, "l162": 400, "r42q1": 400, "r16q1": 300}[stem] * math.exp(NOISE[stem][s - 1]))
    c3 = run(ladder, tmp_path, bench_doc(top_step))[0]["confirmatory"]["C3"]
    assert c3["composite_verdict"] == "inconclusive in clause 1"
    assert c3["widest_pair"][1] == 17
    assert c3["span_vs_c2"]["top"] == pytest.approx(math.log(4 / 3), abs=0.03)


def test_s5_reads_the_herwig_readout_and_counts_models_that_lose(ladder, tmp_path):
    herwig = bench_doc(lambda d, stem, s: 0.5 * predicted(d, stem, s), test_set="herwig")
    s5 = run(ladder, tmp_path, herwig=herwig)[0]["secondary"]["S5"]
    assert s5["all_lose"]["n_lose"] == s5["all_lose"]["n_models"] == 20
    assert s5["ordering_vs_pythia"]["n_agree"] == 6
    assert s5["trend"]["run"] and s5["trend"]["argmax_is_predicted_step"]


def test_the_wrong_readout_is_refused(ladder, tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(bench_doc()))
    with pytest.raises(SystemExit, match="'pythia' readout"):
        S.load_bench(p, S.S5, "herwig")
    with pytest.raises(SystemExit):
        S.main(["--out", str(tmp_path / "o"), "--bench", str(p)])
    with pytest.raises(SystemExit):
        S.main([str(ladder), "--out", str(tmp_path / "o2"), "--bench-herwig", str(p)])
