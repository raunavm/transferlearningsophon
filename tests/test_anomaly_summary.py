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


RERUN = ROOT / "experiments/FIGS/data/anomaly_cs_merged_v1/anomaly_results.json"


def test_the_committed_summary_is_what_the_script_computes():
    """analysis_v4 (with the class-sum rerun) is, in content, exactly what this
    script writes without --heads; analysis_v3 (before it) likewise, the script
    having changed since only in the rerun check and in adding the --heads mode
    (2026-09-29), which leaves the output without it unchanged."""
    got = json.loads((DATA / "analysis_v4/anomaly_summary.json").read_text())
    assert got["provenance"]["inputs"]["anomaly"]["sha256"] == S._sha(
        DATA / "anomaly_results.json")
    rerun = json.loads(RERUN.read_text())
    fresh = json.loads(json.dumps(S.summarise(_committed(), rerun)))
    for k, v in fresh.items():
        assert got[k] == v, k
    v3 = json.loads((DATA / "analysis_v3/anomaly_summary.json").read_text())
    for k, v in json.loads(json.dumps(S.summarise(_committed()))).items():
        assert v3[k] == v, k


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


def test_a_rerun_off_only_in_the_last_digits_is_accepted_and_counted():
    """Logits rebuilt on another node differ in the last digits; a threshold tie
    can then move sigma_min by ~1e-5. That reproduces; it is counted, not hidden."""
    doc = _doc()
    r = _rerun(doc)
    r["arms"]["r16q1-s2"]["signals"]["label_X_bb"]["4000"]["class_sum_matched"]["sigma_min"] *= 1 + 3e-5
    rep = S.summarise(doc, r)["reproduction"]
    assert rep["cells_at_17_classes_reproducing_class_sum"] == 5 * 2
    assert rep["cells_at_17_classes_identical_to_1e-9"] == 5 * 2 - 1
    assert 2e-5 < rep["max_abs_log_difference_at_17_classes"] < 1e-4


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


# ------------------------------------------------ sigma_min only, heads flagged
def _heads(acc=None, pq=None):
    """anomaly_heads.py's shape: every committed model at e079 and over 70-79."""
    acc = acc or {}
    pq = pq or {}
    models = {}
    for arm in _committed()["arms"]:
        seed = int(arm.rsplit("-s", 1)[1].rstrip("b"))
        h = {"top1_accuracy": acc.get(arm, 0.60 + 0.01 * seed),
             "mean_p_qcd_resonant": pq.get(arm, 0.04 + 0.01 * seed),
             "mean_p_qcd_qcd": 0.8, "median_logodds_res_qcd_on_qcd": -1.0}
        models[arm] = {"rung": _committed()["arms"][arm]["rung"],
                       "checkpoints": {"e079": {"head": h}},
                       "head_over_70_79": {k: {"mean": v, "min": v, "max": v}
                                           for k, v in h.items()}}
    return {"models": models}


def test_with_heads_the_summary_reports_sigma_min_only_and_states_the_rule():
    rerun = json.loads(RERUN.read_text())
    res = S.summarise(_committed(), rerun, _heads())
    d = res["definition"]
    assert d["B"] == 200_000 and d["sigma_t"] == 5.0
    assert "more than 25" in d["threshold_rule"] and "n_B > 25" in d["threshold_rule"]
    lv = res["families"]["class_sum_matched"]["label_X_YY_bbbb"]["2000"]["levels"]["43"]
    assert "max_sic_mean" not in lv and "max_sic_sd" not in lv
    assert lv["sigma_min"] == pytest.approx([math.exp(x) for x in lv["ln_sigma_min"]])
    assert len(lv["sigma_min"]) == 5 and len(lv["arms"]) == 5


def test_an_output_layer_that_never_predicts_qcd_is_flagged_at_that_checkpoint():
    acc = {a: 0.62 + 0.01 * i for i, a in enumerate(
        ["r42q1-s1", "r42q1-s2", "r42q1-s3", "r42q1-s4"])}
    acc["r42q1-s5"] = 0.550
    pq = {"r42q1-s1": 0.02, "r42q1-s2": 0.05, "r42q1-s3": 0.09, "r42q1-s4": 0.131,
          "r42q1-s5": 1e-7}
    res = S.summarise(_committed(), None, _heads(acc, pq))
    f = res["head_flags"]["models"]
    assert f["r42q1-s5"]["e079"]["outlier"]
    assert f["r42q1-s5"]["e079"]["mean_p_qcd_resonant"]["outside_99pc_prediction_interval"]
    assert not any(f[f"r42q1-s{k}"]["e079"]["outlier"] for k in (1, 2, 3, 4))
    assert "not a property of the run" in res["head_flags"]["rule"]


def test_flags_and_sigma_min_are_reported_at_each_checkpoint_of_the_rule():
    heads = _heads()
    for m in heads["models"].values():
        h = m["checkpoints"]["e079"]["head"]
        for tag in ("best70", "wavg", "bestval"):    # A14: primary, robustness, global best
            m["checkpoints"][tag] = {"head": dict(h)}
    res = S.summarise(_committed(), None, heads)
    cell = res["head_flags"]["models"]["l188-s1"]
    assert {"best70", "bestval", "wavg", "e079", "mean_70_79"} <= set(cell)
    assert S.FLAG_TAGS[0] == "best70"
    assert res["checkpoint_rule"]["by_checkpoint"] == {}     # no anomaly cells in these heads


def test_the_checkpoint_rule_table_is_built_from_the_models_the_anomaly_study_scores():
    """v1's anomaly_heads.json holds the 20 anomaly models with anomaly cells and the
    ten mass-output models with head diagnostics only; the table must not come out
    empty because the latter have no anomaly cells (it did, 2026-10-01)."""
    heads = _heads()
    for arm, m in heads["models"].items():
        cell = {"sigma_min": 2.0 + 0.1 * int(arm.rsplit("-s", 1)[1].rstrip("b"))}
        m["checkpoints"]["e079"]["anomaly"] = {"label_X_bb": {S.PRIMARY: {"class_sum": cell}}}
        m["anomaly_mean_70_79"] = {"class_sum": {"label_X_bb": {S.PRIMARY: {
            "ln_sigma_min_mean": math.log(cell["sigma_min"]),
            "ln_sigma_min_per_epoch": [math.log(cell["sigma_min"])] * 10}}}}
    full = S.checkpoint_rule({"class_sum": {"label_X_bb": {}}}, heads)
    heads["models"]["l162mass-s1"] = {"rung": "L162", "checkpoints": {"e079": {"head": {}}}}
    res = S.checkpoint_rule({"class_sum": {"label_X_bb": {}}}, heads)
    assert res["by_checkpoint"] and res["by_checkpoint"] == full["by_checkpoint"]
    e = res["by_checkpoint"]["e079"]["class_sum"]["label_X_bb"][S.PRIMARY]
    assert set(e) == {"188", "162", "43", "17"} and len(e["188"]["ln_sigma_min"]) == 5
    assert res["mean_ln_over_epochs_70_79"] == full["mean_ln_over_epochs_70_79"]


def test_the_committed_checkpoint_rule_summary_is_what_the_script_computes():
    """anomaly_v1err/analysis (with --heads) is exactly what this script writes:
    the v2 generalisation leaves every v1 output unchanged."""
    d = ROOT / "experiments/FIGS/data"
    got = json.loads((d / "anomaly_v1err/analysis/anomaly_summary.json").read_text())
    heads = json.loads((d / "anomaly_v1err/anomaly_heads.json").read_text())
    fresh = json.loads(json.dumps(S.summarise(_committed(), json.loads(RERUN.read_text()), heads)))
    assert set(got) - set(fresh) == {"provenance", "conventions"}
    for k, v in fresh.items():
        assert got[k] == v, k


# ------------------------------------------------------------- v2 (the grid)
GRID = ROOT / "configs/arms/v2_grid.json"
# arm -> (model stem, label set, runs): both run counts, the tree, the self-supervised
# arm off it, and the untrained-trunk reference
V2_ARMS = {"L188": ("l188", "L188", 5), "R29_Q1": ("r29q1", "R29_Q1", 5),
           "MPM": ("mpm", "none", 3), "L188_LOFO4P": ("l188lofo4p", "L188", 3),
           "init": ("init", "none", 5)}


def _v2_doc(arms=V2_ARMS, readout="pooled"):
    """One v2 merge (one checkpoint, one readout), knn and mahalanobis."""
    out = {}
    for stem, rung, runs in arms.values():
        for s in range(1, runs + 1):
            per_n = {n: {**{f: {"sigma_min": 2.0 + 0.1 * s, "max_sic": 1.5 + s / 10,
                                "at_ceiling": False} for f in S.V2_FAMILIES},
                         "rng_seeds": [s, int(n)]} for n in (S.PRIMARY, S.REFERENCE)}
            out[f"{stem}-s{s}"] = {"rung": rung, "cache": {"readout": readout},
                                   "signals": {"label_X_bb": per_n}}
    return {**{k: v for k, v in _doc().items() if k != "arms"}, "arms": out}


def test_v2_levels_are_the_grid_arms_each_with_its_own_runs():
    res = S.summarise(_v2_doc(), grid=GRID)
    assert set(res["families"]) == set(S.V2_FAMILIES) and res["readout"] == "pooled"
    lv = res["families"]["knn"]["label_X_bb"][S.PRIMARY]["levels"]
    assert list(lv) == ["L188", "MPM", "R29_Q1", "L188_LOFO4P", "init"], "the grid's order"
    assert {k: len(e["arms"]) for k, e in lv.items()} == {
        "L188": 5, "MPM": 3, "R29_Q1": 5, "L188_LOFO4P": 3, "init": 5}
    e = lv["L188_LOFO4P"]
    assert e["arms"] == [f"l188lofo4p-s{s}" for s in (1, 2, 3)]
    want = [math.log(2.0 + 0.1 * s) for s in (1, 2, 3)]
    assert e["ln_sigma_min"] == want
    assert e["ln_sigma_min_sd"] == pytest.approx(np.std(want, ddof=1), rel=1e-12)
    assert "classes_removed_by_level" not in res, "no class sum in a v2 merge"


def test_v2_a_missing_run_of_a_three_run_arm_is_fatal():
    doc = _v2_doc()
    del doc["arms"]["l188lofo4p-s2"]
    with pytest.raises(SystemExit, match="seeds"):
        S.summarise(doc, grid=GRID)


def test_v2_a_model_the_study_does_not_score_is_fatal():
    for name in ("l189-s1", "l162mass-s1", "r16q1masslm-s1", "rand2p1-s1", "flavf1-s1"):
        doc = _v2_doc()
        doc["arms"][name] = doc["arms"]["l188-s1"]
        with pytest.raises(SystemExit, match="anomaly study scores"):
            S.summarise(doc, grid=GRID)


def test_v2_the_study_is_the_ladder_its_left_out_family_and_the_self_supervised_arms():
    """V2_ANOMALY_ARMS read off the grid, against the arms written out here."""
    cells = S.v2_ladder(GRID)[0]
    ladder = ["L188", "L162", "R42_Q1", "R16_Q1", "R63_Q1", "R29_Q1"]
    assert sorted(cells) == sorted(ladder + [f"{a}_LOFO4P" for a in ladder[:4]]
                                   + ["MPM", "MPM_LOFO4P", "init"])
    assert cells["init"] == ("none", (1, 2, 3, 4, 5))
    assert cells["MPM_LOFO4P"] == ("none", (1, 2, 3)) and cells["R16_Q1_LOFO4P"] == ("R16_Q1", (1, 2, 3))


def test_v2_a_model_scored_on_the_wrong_label_set_is_fatal():
    doc = _v2_doc()
    doc["arms"]["l188lofo4p-s1"]["rung"] = "L162"
    with pytest.raises(SystemExit, match="label set"):
        S.summarise(doc, grid=GRID)
    doc = _v2_doc()
    doc["arms"]["mpm-s2"]["rung"] = "R16_Q1"     # off the tree is 'none'
    with pytest.raises(SystemExit, match="label set"):
        S.summarise(doc, grid=GRID)


def test_v2_a_merge_mixing_readouts_is_fatal():
    doc = _v2_doc()
    doc["arms"]["mpm-s1"]["cache"]["readout"] = "features"
    with pytest.raises(SystemExit, match="readouts"):
        S.summarise(doc, grid=GRID)
    with pytest.raises(SystemExit, match="v1's"):
        S.summarise(_v2_doc(), rerun=_v2_doc(), grid=GRID)


def test_v2_the_output_ratio_is_read_at_every_checkpoint_and_twin():
    """--heads at best70, wavg, bestval and the BatchNorm twins: class_sum_matched per
    grid arm over that arm's runs; a tag is read only where every model carries it."""
    tags = ("best70", "wavg", "bestval", "best70_bn", "bestval_bn")
    models = {}
    for stem, rung, runs in (("l188", "L188", 5), ("l188lofo4p", "L188", 3)):
        for s in range(1, runs + 1):
            cell = {"class_sum_matched": {"sigma_min": 3.0 + 0.1 * s}}
            h = {"top1_accuracy": 0.6 + 0.01 * s, "mean_p_qcd_resonant": 0.05 + 0.01 * s}
            models[f"{stem}-s{s}"] = {"rung": rung, "checkpoints": {
                t: {"head": h, "anomaly": {"label_X_bb": {S.PRIMARY: cell}}} for t in tags}}
    res = S.summarise(_v2_doc(), heads={"models": models}, grid=GRID)
    by = res["checkpoint_rule"]["by_checkpoint"]
    assert list(by) == list(S.FLAG_TAGS[:5]) and set(by) == set(tags)
    e = by["best70_bn"]["class_sum_matched"]["label_X_bb"][S.PRIMARY]
    assert set(e) == {"L188", "L188_LOFO4P"} and len(e["L188_LOFO4P"]["ln_sigma_min"]) == 3
    assert set(res["head_flags"]["models"]["l188-s1"]) == set(tags)
    del models["l188lofo4p-s3"]["checkpoints"]["wavg"]["anomaly"]
    res = S.summarise(_v2_doc(), heads={"models": models}, grid=GRID)
    assert "wavg" not in res["checkpoint_rule"]["by_checkpoint"], "a tag every run carries"


def test_v2_main_records_the_grid_it_read(tmp_path):
    p = tmp_path / "ad.json"
    p.write_text(json.dumps(_v2_doc()))
    assert S.main(["--anomaly", str(p), "--grid", str(GRID), "--out", str(tmp_path / "o")]) == 0
    got = json.loads((tmp_path / "o" / "anomaly_summary.json").read_text())
    assert got["provenance"]["inputs"]["grid"]["sha256"] == S._sha(GRID)
    assert "each grid arm's runs" in got["conventions"]["across_seeds"]
