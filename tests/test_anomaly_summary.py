"""The paper's anomaly-detection table: mean +/- sd over five seeds, no IAD, and
class_sum superseded only by a rerun that reproduces the committed draws."""
import copy
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/FIGS/data/anomaly_merged_v4"


def _load():
    s = importlib.util.spec_from_file_location(
        "anomaly_summary", ROOT / "experiments/EVAL/anomaly_summary.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S = _load()
STEMS = {"L188": "l188", "L162": "l162", "R42_Q1": "r42q1", "R16_Q1": "r16q1"}


def _committed():
    return json.loads((DATA / "anomaly_results.json").read_text())


def _doc(fams=("class_sum", "knn", "mahalanobis", "iad_hgb")):
    """A synthetic merged artifact: four label sets x five seeds, one signal."""
    arms = {}
    for rung, stem in STEMS.items():
        for seed in range(1, 6):
            arm = "l162-s1b" if (stem, seed) == ("l162", 1) else f"{stem}-s{seed}"
            per_n = {}
            for n in ("0", "2000", "4000"):
                cell = {f: ({"null": True} if n == "0" else
                            {"sigma_min": 2.0 + 0.1 * seed, "max_sic": 1.0 + seed / 10,
                             "at_ceiling": False}) for f in fams}
                cell["rng_seeds"] = [len(arm), int(n), seed]
                cell["classes_removed"] = 1 if rung in ("L188", "L162") else 10
                per_n[n] = cell
            arms[arm] = {"rung": rung, "signals": {"label_X_bb": per_n}}
    return {"row_alignment_sha256": "ab", "sigma_t": 5.0, "stat_cut": 0.2,
            "min_bkg_pass": 25, "trainings": 10, "n_bkg": 200000,
            "n_template": 200000, "arms": arms}


def _rerun(doc):
    """The class-sum rerun of `doc`: same draws, class_sum_matched only, equal
    to the committed class_sum at 17 classes (the two estimators coincide)."""
    r = copy.deepcopy(doc)
    for arm, ad in r["arms"].items():
        for per_n in ad["signals"].values():
            for c in per_n.values():
                old = c.pop("class_sum")
                for f in ("knn", "mahalanobis", "iad_hgb"):
                    c.pop(f)
                same = ad["rung"] == "R16_Q1" or "null" in old
                c["class_sum_matched"] = (dict(old) if same else
                                          {**old, "sigma_min": old["sigma_min"] * 1.5})
                c["classes_removed_matched"] = 10
    return r


# ------------------------------------------------------- the committed table

def test_the_summary_reproduces_the_stored_level_means():
    """analysis_v2/anomaly_s5.json stores the level means of ln sigma_min
    (experiments/STATS/seed_level.py). The per-seed summary must give the same
    means from the same artifact, for every family it reports."""
    stored = json.loads((DATA / "analysis_v2/anomaly_s5.json").read_text())[
        "section5"]["level_means_all_injections"]
    res = S.summarise(_committed())
    n_checked = 0
    for fam, blk in res["families"].items():
        for sig, per_n in blk.items():
            for n, c in per_n.items():
                if "skipped" in c:
                    assert f"{fam}|{sig}|{n}" not in stored
                    continue
                for lv, e in c["levels"].items():
                    assert e["ln_sigma_min_mean"] == pytest.approx(
                        stored[f"{fam}|{sig}|{n}"][lv], abs=1e-12)
                    assert len(e["ln_sigma_min"]) == len(e["max_sic"]) == 5
                    n_checked += 1
    assert n_checked == 3 * (6 * 2 - 1) * 4, n_checked


def test_the_per_seed_values_are_the_stored_medians_and_the_sd_is_n_minus_1():
    doc = _committed()
    res = S.summarise(doc)
    e = res["families"]["knn"]["label_X_bb"]["2000"]["levels"]["43"]
    assert e["arms"] == [f"r42q1-s{s}" for s in range(1, 6)]
    want = [math.log(doc["arms"][a]["signals"]["label_X_bb"]["2000"]["knn"]["sigma_min"])
            for a in e["arms"]]
    assert e["ln_sigma_min"] == want
    assert e["ln_sigma_min_sd"] == pytest.approx(np.std(want, ddof=1), rel=1e-12)
    assert res["families"]["class_sum"]["label_X_YY_bbb"]["4000"] == {
        "skipped": "insufficient jets"}


def test_iad_is_absent_from_the_table_and_the_reason_is_recorded():
    res = S.summarise(_committed())
    assert "iad_hgb" not in res["families"]
    assert "iad_hgb" not in json.dumps(res["families"])
    assert "2604.20965" in res["excluded_families"]["iad_hgb"]


def test_not_detected_is_the_light_quark_signals_for_the_feature_space_scores():
    """Recorded as a regression on the committed data, with the margin: no
    threshold between the two bracketing values would change the flagged set."""
    rule = S.summarise(_committed())["not_detected_rule"]
    assert rule["not_detected"] == sorted(
        f"{f}|{s}" for f in ("knn", "mahalanobis")
        for s in ("label_X_qq", "label_X_YY_qqq", "label_X_YY_qqqq"))
    lo, hi = rule["flag_set_unchanged_for_thresholds_in"]
    assert lo < S.NOT_DETECTED_MAX_SIC <= hi


def test_the_committed_summary_is_what_the_script_computes():
    got = json.loads((DATA / "analysis_v3/anomaly_summary.json").read_text())
    assert got["provenance"]["inputs"]["anomaly"]["sha256"] == S._sha(
        DATA / "anomaly_results.json")
    assert got["provenance"]["script_sha256"] == S._sha(S.__file__), (
        "anomaly_summary.py changed since analysis_v3 was written")
    fresh = json.loads(json.dumps(S.summarise(_committed())))
    for k, v in fresh.items():
        assert got[k] == v, k


# ------------------------------------------------------------- the rerun

def test_the_rerun_supersedes_class_sum_and_removes_one_set_at_every_level():
    doc = _doc()
    res = S.summarise(doc, _rerun(doc))
    assert res["superseded"]["class_sum"]["by"] == "class_sum_matched"
    assert "class_sum" in res["families"], "kept for reproducibility"
    assert set(res["classes_removed_by_level"]["class_sum_matched"]["label_X_bb"]
               .values()) == {10}
    assert res["reproduction"]["cells_at_17_classes_reproducing_class_sum"] == 5 * 2
    assert "not_detected" in res["families"]["class_sum_matched"]["label_X_bb"]["2000"]


def test_without_the_rerun_nothing_is_superseded():
    assert S.summarise(_doc())["superseded"] == {}


def test_a_rerun_that_drew_different_resamplings_is_refused():
    doc = _doc()
    r = _rerun(doc)
    r["arms"]["l188-s3"]["signals"]["label_X_bb"]["2000"]["rng_seeds"] = [0, 0]
    with pytest.raises(SystemExit, match="different resamplings"):
        S.summarise(doc, r)


def test_a_rerun_that_does_not_reproduce_class_sum_at_17_classes_is_refused():
    doc = _doc()
    r = _rerun(doc)
    cell = r["arms"]["r16q1-s2"]["signals"]["label_X_bb"]["4000"]
    cell["class_sum_matched"]["sigma_min"] *= 1.01
    with pytest.raises(SystemExit, match="does not reproduce"):
        S.summarise(doc, r)


def test_a_rerun_under_another_configuration_is_refused():
    doc = _doc()
    r = _rerun(doc)
    r["n_bkg"] = 100000
    with pytest.raises(SystemExit, match="n_bkg"):
        S.summarise(doc, r)


def test_a_missing_seed_is_fatal_not_averaged_over_four():
    doc = _doc()
    del doc["arms"]["r42q1-s4"]
    with pytest.raises(SystemExit, match="seeds"):
        S.summarise(doc)


def test_the_output_is_never_overwritten(tmp_path):
    p = tmp_path / "ad.json"
    p.write_text(json.dumps(_doc()))
    assert S.main(["--anomaly", str(p), "--out", str(tmp_path / "o")]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.main(["--anomaly", str(p), "--out", str(tmp_path / "o")])
