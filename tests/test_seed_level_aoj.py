"""The real-data (PRESPEC §6) analysis in experiments/STATS/seed_level.py, on
synthetic peak_fit.py results with the full run's 31 model names."""
import importlib.util
import json
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level", REPO / "experiments/STATS/seed_level.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# Per level and seed: identical noise across levels would give zero-variance
# paired differences, which the tests rightly refuse to evaluate.
NOISE = {"l188": [0.02, -0.03, 0.01, 0.025, -0.015], "l162": [-0.01, 0.02, 0.015, -0.02, 0.01],
         "r42q1": [0.015, 0.01, -0.025, 0.0, 0.02], "r16q1": [0.03, -0.02, 0.04, -0.01, 0.0],
         "l162mass": [0.0, 0.025, -0.01, 0.015, -0.02], "r16q1mass": [-0.02, 0.01, 0.03, 0.02, -0.03]}
STEMS = {"l188": 188, "l162": 162, "r42q1": 43, "r16q1": 17, "l162mass": "162+mass",
         "r16q1mass": "17+mass"}


def results(yield_of, pipeline_ok=True, hard=(), peaks=("top",)):
    models = {"sophon-public": {"top": dict(signal_yield=500.0, signal_yield_err=25.0)}}
    for stem in STEMS:
        for s in range(1, 6):
            name = f"{stem}-s{s}b" if (stem == "l162" and s == 1) else f"{stem}-s{s}"
            y, err = yield_of(stem, s)
            models[name] = {"top": dict(signal_yield=y, signal_yield_err=err,
                                        efficiency_relative_to_reference=y / 1000,
                                        criteria=dict(peak_position=True))}
    return dict(models=models, reference=dict(top=dict(signal_yield=1000.0, signal_yield_err=30.0)),
                verdict={n: "GO" for n in models}, pipeline_ok=pipeline_ok,
                closure_hard_flags=list(hard), n_jets=1_000_000, eff=0.01, peaks=list(peaks))


def predicted(stem, s):
    """188 ~ 162 ~ 43 at ~400, 17 at ~150; the mass output halves the 17-class yield only."""
    base = {"l188": 400, "l162": 400, "r42q1": 400, "r16q1": 150, "l162mass": 400, "r16q1mass": 75}[stem]
    return base * math.exp(NOISE[stem][s - 1]), 15.0


def write(tmp_path, res):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(res))
    return p


def test_the_predicted_pattern_is_confirmed_in_all_three_clauses(tmp_path):
    data = S.load_aoj(write(tmp_path, results(predicted)))
    r = S.aoj_analysis(data, [1, 2, 3, 4, 5])
    assert r["run"] and r["composite_verdict"] == "confirmed in clauses 1-3", r["composite_verdict"]
    assert "AUC" not in r["equivalence"]["verdict"] and "in yield" in r["equivalence"]["verdict"]
    assert r["trend"]["argmax_is_predicted_step"]


def test_the_suffixed_first_162_class_seed_is_seed_one_and_the_public_model_is_a_reference(tmp_path):
    data = S.load_aoj(write(tmp_path, results(predicted)))
    assert {(r["level"], r["seed"]) for r in data["rows"] if r["model"] == "l162-s1b"} == {(162, 1)}
    assert len(data["rows"]) == 30 and list(data["reference_models"]) == ["sophon-public"]


def test_the_mass_output_interaction_has_the_sign_of_what_it_did(tmp_path):
    """Halving the 17-class yield with the mass output and leaving 162 alone: the
    17 gain is about +ln 2 in -ln(yield), the 162 gain about 0, so
    DiD = gain_162 - gain_17 < 0."""
    r = S.aoj_analysis(S.load_aoj(write(tmp_path, results(predicted))), [1, 2, 3, 4, 5])
    m = r["mass_output_2x2"]
    assert abs(m["gain_17"]["mean_diff"] - math.log(2)) < 0.05 and abs(m["gain_162"]["mean_diff"]) < 0.05
    assert m["difference_in_differences"]["estimable"] and m["difference_in_differences"]["mean_diff"] < 0


def test_a_yield_below_its_uncertainty_is_kept_as_a_bound_not_dropped(tmp_path):
    def lost(stem, s):
        return (5.0, 40.0) if stem == "r16q1" else predicted(stem, s)
    data = S.load_aoj(write(tmp_path, results(lost)))
    r17 = [r for r in data["rows"] if r["level"] == 17]
    assert len(r17) == 5 and all(r["censored"] and r[S.ENDPOINT] == -math.log(40.0) for r in r17)
    assert S.aoj_analysis(data, [1, 2, 3, 4, 5])["trend"]["run"]


@pytest.mark.parametrize("kw,why", [(dict(pipeline_ok=False), "does not show the top peak"),
                                    (dict(hard=("shard0: d0 units",)), "closure hard flags")])
def test_nothing_is_tested_when_the_pipeline_or_the_inputs_fail(tmp_path, kw, why):
    r = S.aoj_analysis(S.load_aoj(write(tmp_path, results(predicted, **kw))), [1, 2, 3, 4, 5])
    assert not r["run"] and why in r["reason"]


def test_a_fit_that_includes_the_withdrawn_w_peak_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="top peak only"):
        S.load_aoj(write(tmp_path, results(predicted, peaks=("W", "top"))))


def test_the_command_line_writes_its_own_file_and_never_overwrites(tmp_path):
    p = write(tmp_path, results(predicted))
    assert S.main(["--real-data", str(p), "--out", str(tmp_path / "o")]) == 0
    got = json.loads((tmp_path / "o" / "aoj_top.json").read_text())
    assert got["secondary"]["real_data_top"]["holm_table"][0]["family_size"] == 2
    with pytest.raises(SystemExit, match="exists"):
        S.main(["--real-data", str(p), "--out", str(tmp_path / "o")])
