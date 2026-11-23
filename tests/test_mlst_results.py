"""The MLST tables' code (experiments/FIGS/mlst_results.py, read by make_tables.build()):
mean +- sample SD over the runs present with a marker for missing runs, a lone run with its
trailing zeros, the probe convergence rule, the merge marks from the label maps, and the
pulled data checked against its index."""
import csv
import importlib.util
import inspect
import collections
import json
import pathlib
import math
import re
import shutil
import statistics
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/FIGS/data/v2"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("mlst_results_t", "experiments/FIGS/mlst_results.py")
MT = _load("make_tables_mlst_t", "experiments/FIGS/make_tables.py")


def test_a_cell_is_the_mean_and_sample_sd_with_missing_runs_marked():
    seen = []
    fmt = lambda m, s: seen.append((m, s)) or "X"
    assert M._cell([0.1, 0.2, 0.3], 3, fmt) == "X"
    assert seen[0][0] == pytest.approx(0.2) and seen[0][1] == pytest.approx(0.1)   # ddof=1
    # two runs present: their mean and sample SD as well (PI 2026-11-12), with the marker for the missing one
    assert M._cell([0.1, None, 0.3], 3, fmt) == r"X \pending{2 of 3}"
    assert seen[1][0] == pytest.approx(0.2) and seen[1][1] == pytest.approx(0.02 ** 0.5)
    assert M._cell([None, None], 2, fmt) == r"\pending{no run yet}"


def test_a_table_cell_is_a_token_of_its_kind_until_its_column_is_printed():
    """With the tables' formatter (M.TAB) a cell is a token: p (mean, SD) for two or more runs, equal runs included
    (SD 0); s for a single run; o for a number that is not a run's (run=False). TAB.two is the same for two values."""
    import numpy as np
    tok = M.V4.tok
    v = [0.1, 0.2, 0.4]
    assert M._cell(v, 3, M.TAB) == tok("p", float(np.mean(v)), float(np.std(v, ddof=1)))
    assert M._cell([0.1, None, 0.4], 3, M.TAB) == tok("p", 0.25, float(np.std([0.1, 0.4], ddof=1))) + r" \pending{2 of 3}"
    assert M._cell([0.7, 0.7], 2, M.TAB) == tok("p", 0.7, 0.0)                 # equal runs: an SD of zero, not one value
    assert M._cell([0.863], 1, M.TAB) == tok("s", 0.863)                       # a single run
    assert M._cell([0.863], 3, M.TAB) == tok("s", 0.863) + r" \pending{1 of 3}"
    assert M._cell([0.2448], 1, M.TAB, run=False) == tok("o", 0.2448)          # the soft-drop mass: no run, no mark
    assert M._cell([0.0593], 1, M.TAB, 1e3) == tok("s", 0.0593 * 1e3)
    assert M.TAB(0.5, 0.01) == tok("p", 0.5, 0.01) and M.TAB(0.5, None) == tok("p", 0.5, 0.0)
    assert M.TAB.two(0.25, 0.75) == tok("p", 0.5, 0.5 / math.sqrt(2)) and M.TAB.two(0.5, 0.5) == tok("p", 0.5, 0.0)
    # a sentence never holds a token: two runs print both values there
    assert M._cell([1.1661, 1.1652], 2, M.TAB, sep=M.TEXT_SEP) == "1.166 and 1.165"
    # mlst_v4's cells: the same kinds
    V4 = M.V4
    assert V4._value_cell([0.94], "accuracy", "top", M.TAB, M.TAB.two) == tok("s", 0.94)
    assert V4._value_cell([1800.0, None, 1820.0], "r30", "top", M.TAB, M.TAB.two) == tok("p", 1810.0, float(np.std([1800.0, 1820.0], ddof=1)))
    assert V4._value_cell([None], "r30", "top", M.TAB, M.TAB.two) == "--"
    assert V4._jc_cell([5000.0], 0, M.TAB, M.TAB.two) == tok("s", 5000.0)
    assert V4._jc_cell([5000.0, 5000.0, 5000.0], 0, M.TAB, M.TAB.two) == tok("p", 5000.0, 0.0)


def test_two_runs_print_both_values_in_a_sentence_and_mean_and_sd_in_a_table():
    # in a sentence: never beyond four significant figures (2026-10-31), even where the difference is smaller
    assert M._cell([1.1661, 1.1652], 2, MT.fmt_pm, sep=" and ") == "1.166 and 1.165"
    assert M._cell([22.515, 22.467], 2, MT.fmt_pm, sep=" and ") == "22.52 and 22.47"
    # in a table (PI 2026-11-12): mean +- sample SD, as three runs; with make_tables.fmt_pm the SD PDG-rounded
    # (0.0038 / sqrt 2 = 0.0027; 0.0008 / sqrt 2 = 0.0006, shown at the mean's fourth significant figure)
    assert M._cell([0.4123, 0.4161], 2, MT.fmt_pm) == r"0.4142\,$\pm$\,0.0027"
    assert M._cell([1.1661, 1.1653], 2, MT.fmt_pm) == r"1.166\,$\pm$\,0.001"
    assert M._cell([0.4123, 0.4161], 2, MT.fmt_pm) == MT.fmt_pm(0.4142, 0.0038 / 2 ** 0.5)
    assert M._cell([0.5, 0.5], 2, MT.fmt_pm) == "0.500"               # fmt_pm: equal runs, the common value


def test_a_lone_run_prints_three_significant_figures():
    assert M._cell([0.86300], 3, MT.fmt_pm) == r"0.863 \pending{1 of 3}"
    assert M._cell([0.96579654592], 1, MT.fmt_pm) == "0.966"
    assert M._cell([0.0593], 1, MT.fmt_pm, 1e3) == "59.3"
    assert M._cell([1802.83], 1, MT.fmt_pm) == "1{,}803" and M._cell([42589.4], 1, MT.fmt_pm) == "42{,}590"
    assert M._cell([0.96579654592], 1, MT.fmt_pm, frac=True) == "0.9658"      # an accuracy or AUC
    # in a generated table the lone run is a token (kind s) and prints with the single-run mark at its column's place
    assert M.uniform_columns("a & " + M._cell([0.86300], 1, M.TAB) + r" \\") == "a & 0.8630" + M.SINGLE + r" \\"


def test_an_unconverged_probe_is_marked_and_counted():
    cell = {"converged": False, "auc": 0.7}
    fits = {"mtx-l188-s1": {"cells": {"best70": {"top": {"1000": cell}}}},
            "init-s1": {"cells": {"init": {"top": {"1000": cell}}}}}
    M.UNCONVERGED_SEEN.clear()
    for m in fits:
        v = M._probe(fits, m, "top", "1000", "auc")
        assert isinstance(v, M.Unconverged) and M._cell([v], 1, MT.fmt_pm).endswith(M.UNCONVERGED)
    assert M.UNCONVERGED_SEEN == {("mtx-l188-s1", "top", "1000"), ("init-s1", "top", "1000")}


def test_a_comparison_is_taken_run_by_run_between_runs_of_the_same_index():
    runs = {"A": ["mtx-a-s1", "mtx-a-s2", "mtx-a-s3"], "B": ["mtx-b-s1", "mtx-b-s3"], M.INIT: ["init-s1", "init-s2"]}
    v = {"mtx-a-s1": 2.0, "mtx-a-s2": 9.0, "mtx-a-s3": 3.0, "mtx-b-s1": 1.0, "mtx-b-s3": 1.0}
    seen = []
    fmt = lambda m, s: seen.append((m, s)) or "X"
    # s2 has no partner: not a missing run; two pairs print both ratios, 2/1 and 3/1
    assert M._ratio(runs, "A", "B", v.get, fmt) == "2.0 and 3.0"
    v["mtx-b-s3"] = None
    assert M._ratio(runs, "A", "B", v.get, fmt) == r"2.00 \pending{1 of 2}"
    runs["B"] = ["mtx-b-s1", "mtx-b-s2", "mtx-b-s3"]
    v.update({"mtx-b-s2": 3.0, "mtx-b-s3": 1.0})
    assert M._ratio(runs, "A", "B", v.get, fmt) == "X"                       # 2/1, 9/3, 3/1
    assert seen[-1][0] == pytest.approx(8 / 3) and seen[-1][1] == pytest.approx((1 / 3) ** 0.5)
    v.update({"init-s1": M.Unconverged(4.0), "init-s2": 3.0})                # 2/4 marked, 9/3
    assert M._ratio(runs, "A", M.INIT, v.get, fmt) == "0.5 and 3.0" + M.UNCONVERGED
    assert M._oma(M.Unconverged(0.9)) == pytest.approx(0.1) and isinstance(M._oma(M.Unconverged(0.9)), M.Unconverged)


def test_the_merge_marks_follow_the_label_maps():
    s = M.merge_status(ROOT)
    qcd = {"qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light"}
    assert set(s["L188"].values()) == {"apart"}
    assert {k for k, v in s["L162"].items() if v == "merged"} == qcd     # 162 classes merge only QCD
    assert s["R42_Q1"] == {"bc_vs_bq_cs": "partly", "bb_vs_cc": "apart", "bbqq_vs_ccqq": "apart",
                           "ee_vs_mm": "merged", **{k: "merged" for k in qcd}}
    assert set(s["R16_Q1"].values()) == {"merged"}
    assert all(s[a][k] == "merged" for a in M.RAND + M.FLAV for k in qcd)
    # X->ee vs X->mumu: merged at 43 and 17 classes and by random partitions 1, 2 and 4; kept apart by the
    # flavour partitions (ee shares a group with the X->qq decays there)
    assert {a for a in M.RAND if s[a]["ee_vs_mm"] == "merged"} == {"RAND2_p1", "RAND2_p2", "RAND2_p4"}
    assert all(s[a]["ee_vs_mm"] == "apart" for a in M.FLAV)
    assert s["L162_MASS"] == s["L162"] and s["R16_Q1_MASS"] == s["R16_Q1"]


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_pulled_data_is_checked_against_its_index(tmp_path):
    assert M.check_index(DATA) == DATA / "index.json"
    d = tmp_path / "v2"
    shutil.copytree(DATA, d)
    f = next((d / "linprobe/fits_v3").glob("*.json"))
    f.write_bytes(f.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="differ from"):
        M.check_index(d)
    f.unlink()
    with pytest.raises(SystemExit, match="lists"):
        M.check_index(d)
    (d / "index.json").unlink()
    with pytest.raises(SystemExit, match="no "):
        M.check_index(d)


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_index_covers_every_kind_of_pulled_file():
    kinds = {k.split("/")[0] + ("/" + k.split("/")[1] if k.startswith("linprobe") else "")
             for k in json.loads((DATA / "index.json").read_text())}
    assert {"compute", "linprobe/fits_v3"} <= kinds <= {
        "compute", "linprobe/fits3000", "linprobe/mass", "linprobe/fits_v3", "linprobe/fits_v3_mlp", "ft_mlst",
        "head", "fillcontrol", "curves", "bootstrap", "xbx", "jc2_test", "linprobe/fits_v3_mlp2",
        "checks",       # 2026-11-09: the TEST2M disjointness proof, pulled from /data/results/checks
        "linprobe/fits_v3_mlp2_initbn", "linprobe/fits_v3_initbn",
        # 2026-10-31 (pull_mlst_data.py v4): the v4 probes and the outputs beside them
        "linprobe/fits_v4", "linprobe/fits_v4_mlp", "jc2bal", "budget", "partrepro", "dup_check",
        # Phase B (2026-11-02): the substitution probes and the X->bs runs from scratch
        "linprobe/fits_v4_fill", "xbx_scratch",
        # the ROC redesign (2026-11-02): the per-class curves and the X->ee vs X->mumu probe
        "curves_v2", "linprobe/fits_v4_sib",
        # the gallery decisions (2026-11-05): the class-token t-SNE, then UMAP and every lepton-pair test jet
        "tsne", "umap", "umap_preview", "linprobe/fits_v4_sib2",
        # 2026-11-06: the census of selected training jets; round 3 UMAP (frozen and fine-tuned, every task)
        "jc2_train", "umap_v2",
        # round 3 (2026-11-06): performance against jet pT, the JetClass confusion matrices
        "perf_vs_pt", "confusion_jc1",
        # 2026-11-07: the coarse targets, the QCD heavy-flavor subclasses, the released Sophon checkpoint's probes,
        # the checkpoint identity check
        "linprobe/fits_v4_coarse", "linprobe/fits_v4_qcdhf", "linprobe/sophon", "linprobe/sophon_qcdhf",
        "ckpt_identity",
        # 2026-11-13: the JetClass full-decay confusion matrices from scratch, and the UMAP grids of every task
        "confusion_jc1_fulldecay", "umap_v3",
        # the rerun's read-outs (PI 2026-11-15/16; pull_mlst_data.RERUN_PULL, RERUN_PULL_V5, L3_RERUN)
        "perf_vs_pt_fixed", "perf_vs_pt_v5", "confusion_jc1_fixed", "umap_v4", "l3_rerun",
        # the v5 refits behind refit_v5_checks/PASS (pull_mlst_data V5_PULL, first pulled 2026-11-22), the X->bx v5
        # probes, per-class recall, the confusion-share test
        "linprobe/fits_v5", "linprobe/fits_v5_coarse", "linprobe/fits_v5_fill", "linprobe/fits_v5_mlp",
        "linprobe/fits_v5_nat", "linprobe/fits_v5_qcdhf", "linprobe/fits_v5_sib2", "linprobe/sophon_v5",
        "linprobe/sophon_v5_fill", "xbx_v5", "perclass_recall", "stats"}


def _ft_tree(d, inits, n_seeds_scratch=3, bound=False, budget=True):
    """The earlier read-outs (best70_*: every size, the natural-mix JetClass-II at 0.5 so that its
    replacement shows) and the fixed-budget read-out below the full sets and for the balanced set."""
    rec = lambda v: {"macro_auc_ovr": v, "macro_auc_qcd": v, "auc": v, "accuracy": v, "r50": 100 * v, "r30": 300 * v,
                     "r50_is_bound": bound, "r30_is_bound": False, "r50_n_bkg_pass": 400, "r30_n_bkg_pass": 120}
    def by_init(sizes, base=0.9):
        out = {i: {f"N{n}": {"s1": rec(base + 0.01 * k)} for n in sizes} for k, i in enumerate(inits)}
        out[M.SCRATCH] = {f"N{n}": {f"s{s}": rec(base - 0.1) for s in range(1, n_seeds_scratch + 1)} for n in sizes}
        return out
    (d / "ft_mlst").mkdir(parents=True)
    w = lambda name, cells: (d / "ft_mlst" / f"best70_{name}.json").write_text(json.dumps({"cells": cells}))
    w("leg188_leg1_metrics", by_init([1000, 1000000], base=0.5))
    w("leg2_metrics", by_init([1000, 1000000]))
    w("bench_metrics", {"top": by_init([1000, 1200000]), "qg": by_init([1000, 1600000])})
    w("bench_metrics_herwig", {"qg": by_init([1000, 1600000])})
    if not budget:
        return
    cells = {}
    for name, v, seeds in [(i, 0.9 + 0.01 * k, [1]) for k, i in enumerate(inits)] + \
            [(M.BUDGET_SCRATCH, 0.8, range(1, n_seeds_scratch + 1))]:
        for tree, n in (("leg_jc2b", 1000), ("leg_jc2b", 1000000), ("leg2", 1000), ("leg_top", 1000), ("leg_qg", 1000)):
            for sd in seeds:
                cells[f"{tree}/{name}/N{n}/s{sd}"] = {**rec(v), **({"herwig": rec(v)} if tree == "leg_qg" else {})}
    (d / "budget").mkdir()
    (d / "budget" / "budget_metrics.json").write_text(json.dumps({"cells": cells}))


def test_the_fine_tuning_tables_read_one_seed_per_run_and_the_scratch_seeds(tmp_path):
    _ft_tree(tmp_path, ["l188-s1", "l188-s2"])
    tag, per, absent = M.load_ft(tmp_path)
    assert absent == []
    assert tag == "best70" and per["jc2"]["l188-s1"][1000][0]["macro_auc_ovr"] == 0.9
    # the natural-mix JetClass-II cells (0.5) are gone at every size; the budget's replace them
    assert per["jc2"]["l188-s1"][1000000][0]["macro_auc_ovr"] == 0.9 and len(per["jc2"][M.SCRATCH][1000]) == 3
    assert per["top"]["l188-s2"][1000][0]["auc"] == 0.91 and per["qg_herwig"]["l188-s1"][1000][0]["auc"] == 0.9
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"]}
    t = M.table_ft(per, runs, MT.fmt_pm, ("L188",))
    # one block per task, and the JetClass-II accuracy beside its AUC (2026-11-07); the supplement's copy
    # (tab:ftfull, 2026-11-13) adds the AUC blocks of the training-size figures' AUC versions
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS) + len(M.FT_FULL_AUC) and r"\pending{2 of 3}" in t and M.SCRATCH_LABEL in t
    assert r"\begin{longtable}" in t and r"\label{tab:ftfull}" in t and "%SPLIT%" not in t
    assert [k for k, *_ in M.FT_FULL_AUC] == ["jc1", "top", "qg", "qg_herwig"]
    assert "1{,}200{,}000" in t and "1{,}600{,}000" in t
    # each task in its own metric (PI 2026-11-02, no 1 - AUC outside JetClass-II): JetClass-II the AUC
    # (0.90 and 0.91: two runs, mean +- SD), top/q-g R30 (300 x AUC: 270 and 273), never R50 (90 and 91)
    assert r"0.905\,$\pm$\,0.007" in t and r"271.5\,$\pm$\,2.1" in t and r"90.5\,$\pm$\,0.7" not in t
    body, fewest = M.table_ft_rej(per, runs, MT.fmt_pm, ("L188",))
    assert fewest == 120 and r"\pending{2 of 3}" in body and body.count(r"\\") == 4
    per["jc2"]["l188-s1"][1000].append(per["jc2"]["l188-s1"][1000][0])
    with pytest.raises(SystemExit, match="fine-tuning seeds"):
        M.table_ft(per, runs, MT.fmt_pm, ("L188",))


def test_the_long_budget_cells_replace_every_cell_of_their_task_model_and_size(tmp_path):
    """budget_long (PI 2026-11-05): at 1e6 its cells replace the fixed-budget JetClass-II cells and the earlier
    JetClass full-set cells of the same model; other sizes and models keep theirs; a model only it has is added."""
    _ft_tree(tmp_path, ["l188-s1", "l188-s2"])
    rec = lambda v: {"macro_auc_ovr": v, "macro_auc_qcd": v, "auc": v, "accuracy": v}
    long = {"leg_jc2b/l188-s1/N1000000/s1": rec(0.97), "leg_jc2b/rand2p1-s1/N1000000/s1": rec(0.95),
            "leg2/l188-s2/N1000000/s1": rec(0.96),
            **{f"leg_jc2b/{M.BUDGET_SCRATCH}/N1000000/s{k}": rec(0.85) for k in (1, 2)}}
    (tmp_path / "budget" / "budget_long_metrics.json").write_text(json.dumps({"cells": long}))
    _, per, _ = M.load_ft(tmp_path)
    assert [r["accuracy"] for r in per["jc2"]["l188-s1"][1000000]] == [0.97]
    assert [r["accuracy"] for r in per["jc2"]["l188-s2"][1000000]] == [0.91]
    assert per["jc2"]["l188-s1"][1000][0]["accuracy"] == 0.9
    assert per["jc2"]["rand2p1-s1"][1000000][0]["accuracy"] == 0.95
    assert [r["accuracy"] for r in per["jc2"][M.SCRATCH][1000000]] == [0.85, 0.85]
    assert [r["accuracy"] for r in per["jc1"]["l188-s2"][1000000]] == [0.96]
    # a tree with a cell still missing is not used at all: no figure mixes the two recipes
    (tmp_path / "budget" / "budget_long_metrics.json").write_text(json.dumps(
        {"cells": long, "missing": ["leg2/l188-s1/N1000000/s1"]}))
    _, per, _ = M.load_ft(tmp_path)
    assert per["jc1"]["l188-s2"][1000000][0]["accuracy"] != 0.96
    assert [r["accuracy"] for r in per["jc2"]["l188-s1"][1000000]] == [0.97]


def test_the_full_schedule_scratch_cells_replace_every_old_scratch_cell_of_their_task_and_size(tmp_path, monkeypatch):
    """PI 2026-11-07: the from-scratch read-outs with the whole flat+decay schedule replace the fixed-budget and the
    long-budget from-scratch cells of the same task and size, each only complete (FATAL otherwise); the pretrained
    cells keep theirs, and with both read-outs pulled no old from-scratch budget cell is left."""
    bfb = _load("build_ft_budget_jobs_t", "scripts/build_ft_budget_jobs.py")
    assert M.SCRATCH_FULL == {"budget_fulldecay_metrics.json": len(bfb.scratch_full_cells("1e-3", "3e-3")),
                              "budget_long_fulldecay_metrics.json": 2 * len(bfb.long_scratch_full_cells("jc1", "1e-3"))}
    _ft_tree(tmp_path, ["l188-s1"])
    rec = lambda v: {"macro_auc_ovr": v, "macro_auc_qcd": v, "auc": v, "accuracy": v}
    S, b = M.BUDGET_SCRATCH, tmp_path / "budget"
    (b / "budget_long_metrics.json").write_text(json.dumps({"cells": {
        "leg_jc2b/l188-s1/N1000000/s1": rec(0.97), **{f"leg_jc2b/{S}/N1000000/s{k}": rec(0.85) for k in (1, 2, 3)}}}))
    below = {f"{t}/{S}/N1000/s{k}": {**rec(0.7), **({"herwig": rec(0.7)} if t == "leg_qg" else {})}
             for t in ("leg_jc2b", "leg2", "leg_top", "leg_qg") for k in (1, 2, 3)}
    at = {f"leg_jc2b/{S}/N1000000/s{k}": rec(0.75) for k in (1, 2, 3)}
    monkeypatch.setattr(M, "SCRATCH_FULL", {"budget_fulldecay_metrics.json": 12, "budget_long_fulldecay_metrics.json": 3})
    w = lambda name, cells, missing=(): (b / name).write_text(json.dumps({"cells": cells, "missing": list(missing)}))
    acc = lambda per, key, n: [r["accuracy"] for r in per[key][M.SCRATCH][n]]
    # neither pulled: the old cells, as before
    _, per, _ = M.load_ft(tmp_path)
    assert acc(per, "top", 1000) == [0.8] * 3 and acc(per, "jc2", 1000000) == [0.85] * 3
    # below 1e6 only: those sizes replaced, every task (the Herwig sub-record too); 1e6 keeps the long cells
    w("budget_fulldecay_metrics.json", below)
    _, per, _ = M.load_ft(tmp_path)
    for key in ("top", "qg", "qg_herwig", "jc1", "jc2", "jc2_qcd"):
        assert acc(per, key, 1000) == [0.7] * 3, key
    assert acc(per, "jc2", 1000000) == [0.85] * 3
    assert per["jc2"]["l188-s1"][1000000][0]["accuracy"] == 0.97 and per["top"]["l188-s1"][1000][0]["accuracy"] == 0.9
    # both: 1e6 too, and the pretrained cells untouched
    w("budget_long_fulldecay_metrics.json", at)
    _, per, _ = M.load_ft(tmp_path)
    assert acc(per, "jc2", 1000000) == [0.75] * 3 and acc(per, "jc2_qcd", 1000000) == [0.75] * 3
    assert per["jc2"]["l188-s1"][1000000][0]["accuracy"] == 0.97
    # refused loudly once present: a missing cell, one cell short, a pretrained cell
    for cells, missing in ((at, ["leg_jc2b/scratch/N1000000/s3"]), (dict(list(at.items())[:2]), ()),
                           ({**dict(list(at.items())[:2]), "leg_jc2b/l188-s1/N1000000/s1": rec(0.9)}, ())):
        w("budget_long_fulldecay_metrics.json", cells, missing)
        with pytest.raises(SystemExit, match="FATAL: .*budget_long_fulldecay_metrics.json"):
            M.load_ft(tmp_path)
    # both pulled but an old from-scratch cell would still be read (JetClass-II at 1e6 not covered): FATAL
    w("budget_long_fulldecay_metrics.json", {k.replace("leg_jc2b", "leg2"): v for k, v in at.items()})
    with pytest.raises(SystemExit, match="old from-scratch budget cells still read"):
        M.load_ft(tmp_path)


def test_the_compute_table_takes_the_from_scratch_gpus_from_the_full_schedule_cells(tmp_path):
    (tmp_path / "budget").mkdir()
    w = lambda name, cells: (tmp_path / "budget" / name).write_text(json.dumps({"cells": cells}))
    w("budget_metrics.json", {"leg_top/scratch/N1000/s1": {"gpu": "Tesla V100-SXM2-32GB"},
                              "leg_top/l188-s1/N1000/s1": {"gpu": "Tesla V100-SXM2-32GB"}})
    old = M.ft_gpu_types(tmp_path)[0]
    assert old["scratch"] == old["l188-s1"]
    w("budget_fulldecay_metrics.json", {"leg_top/scratch/N1000/s1": {"gpu": "NVIDIA L40"}})
    new = M.ft_gpu_types(tmp_path)[0]
    assert new["scratch"] == [M.gpu_name("NVIDIA L40")] and new["l188-s1"] == old["l188-s1"]


def test_a_rejection_that_is_only_a_bound_stops_the_table(tmp_path):
    _ft_tree(tmp_path, ["l188-s1"], bound=True)
    _, per, _ = M.load_ft(tmp_path)
    with pytest.raises(SystemExit, match="only a bound"):
        M.table_ft_rej(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))


def test_no_fine_tuning_read_out_is_no_table(tmp_path):
    assert M.load_ft(tmp_path) == (None, {}, [k for k, *_ in M.FT_TASKS])


def test_a_task_not_yet_read_out_prints_a_marker_and_the_others_print(tmp_path):
    _ft_tree(tmp_path, ["l188-s1"], budget=False)
    (tmp_path / "ft_mlst" / "best70_leg2_metrics.json").unlink()
    tag, per, absent = M.load_ft(tmp_path)
    assert tag == "best70" and absent == ["jc1"] and per["jc1"] == {}
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))
    jc1 = t.split(M.BLOCK_SEP)[3]
    assert "{JetClass} \\\\" in jc1 and r"training jets & \pending{not yet read out}" in jc1
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS) + len(M.FT_FULL_AUC)
    assert M.SCRATCH_LABEL in t


class _Em:
    def __init__(self):
        self.m = {}

    def macro(self, name, body, *a, **k):
        self.m[name] = body


BENCH_VAL = 'val_for () { if [ "$1" -le 10000 ]; then echo 20000; else echo 200000; fi; }\n'


def _recipe_tree(tmp, cells, spec_text="weaver --data-train x --samples-per-epoch-val 20000 -o fc_params '[(512,0.1)]'\n"):
    data = tmp / "experiments/FIGS/data/v2"
    (data / "ft_mlst").mkdir(parents=True)
    (data / "ft_mlst/manifests.json").write_text(json.dumps(cells))
    spec = tmp / M.LEGS_SPEC
    spec.parent.mkdir(parents=True)
    spec.write_text(spec_text)
    (tmp / M.BENCH_SPEC).write_text(BENCH_VAL)
    return data


def _cell(leg, init, n, **kw):
    c = {"leg": leg, "init": init, "n_train": n, "ft_seed": "1", "lr": "1e-4", "head_lr_mult": "50",
         "weight_decay": "0.01", "epochs": "20" if leg in ("top", "qg") else {"1000": "50", "1000000": "10"}[n],
         "lr_schedule": "constant" if leg in ("top", "qg") else None, "recipe": None, "batch_size": "512",
         "ckpt_rule": "best70", "gpu_device_name": "Tesla V100-SXM2-32GB",
         "samples_per_epoch": "10000" if n == "1000" else n,
         "val_jets": "20480" if leg == "1" else None, "val_rule": "fixed-sample-whole-pass" if leg == "1" else "rotating-stream"}
    c.update(kw)
    return c


def test_the_recipe_macros_come_from_every_cell_and_one_stray_value_stops_the_build(tmp_path):
    cells = {f"best70/leg1/l188-s1/N{n}/s1": _cell("1", "l188-s1", n) for n in ("1000", "1000000")}
    cells["best70/leg_top/l188-s1/N1000/s1"] = _cell("top", "l188-s1", "1000")
    cells["best70/leg1/not-a-paper-run/N1000/s1"] = _cell("1", "not-a-paper-run", "1000", lr="9")   # ignored
    em, missing = _Em(), []
    M.ft_recipe(em, _recipe_tree(tmp_path / "a", cells), {"l188-s1"}, str, missing, MT.words)
    assert em.m["MlstFtLr"] == r"\ensuremath{10^{-4}}" and em.m["MlstFtHeadMult"] == "50" and em.m["MlstFtBenchEpochs"] == "20"
    assert em.m["MlstFtEpochsEThree"] == "50" and em.m["MlstFtEpochsESix"] == "10"
    assert em.m["MlstFtGpuTypes"] == "V100" and em.m["MlstFtCellsDone"] == "3"
    assert em.m["MlstFtRefLr"].startswith(r"\pending") and any("V100" in x for x in missing)
    assert em.m["MlstFtMinEpochJets"] == "10000" and em.m["MlstFtValJcii"] == "20480"
    assert em.m["MlstFtHeadDropout"] == "0.1"
    assert em.m["MlstFtValSmall"] == "20000" and em.m["MlstFtValLarge"] == "200000"
    cells["best70/leg1/l188-s1/N1000/s1"]["lr"] = "2e-4"
    with pytest.raises(SystemExit, match="expected one value"):
        M.ft_recipe(_Em(), _recipe_tree(tmp_path / "b", cells), {"l188-s1"}, str, [], MT.words)
    cells["best70/leg1/l188-s1/N1000/s1"]["lr"] = "1e-4"
    with pytest.raises(SystemExit, match="sets a scheduler"):
        M.ft_recipe(_Em(), _recipe_tree(tmp_path / "c", cells, "--lr-scheduler none\n"), {"l188-s1"}, str, [], MT.words)
    cells.update({f"ref_v100/jc2/leg1/scratch-v2/N1000/s{s}": _cell("1", "scratch-v2", "1000", lr="1e-3", ft_seed=str(s),
                                                                       recipe="part-from-scratch", weight_decay="0")
                  for s in (1, 2, 3)})
    cells.update({f"ref_v100/bench/leg_top/scratch-v2/N1000/s{s}": _cell("top", "scratch-v2", "1000", lr="1e-3", ft_seed=str(s),
                                                                         recipe="part-from-scratch") for s in (1, 2, 3)})
    em = _Em()
    M.ft_recipe(em, _recipe_tree(tmp_path / "d", cells), {"l188-s1"}, str, [], MT.words)
    assert em.m["MlstFtRefLr"] == r"\ensuremath{10^{-3}}" and em.m["MlstFtRefSeedsWord"] == "three"
    assert em.m["MlstFtRefWeightDecayBench"] == "0.01"
    cells["ref_v100/jc2/leg1/scratch-v2/N1000/s1"]["weight_decay"] = "0.01"
    with pytest.raises(SystemExit, match="has weight decay"):
        M.ft_recipe(_Em(), _recipe_tree(tmp_path / "e", cells), {"l188-s1"}, str, [], MT.words)


def test_learning_rates_print_as_powers_of_ten():
    assert M._power("1e-4") == r"\ensuremath{10^{-4}}" and M._power("5e-4") == r"\ensuremath{5\times10^{-4}}"


def test_the_batchnorm_recomputed_random_network_is_its_own_row_read_at_init_bn():
    assert M._ck("init-s2") == "init" and M._ck("initbn-s2") == "init_bn" and M._ck("mtx-l188-s2") == M.PRIMARY
    runs = M._paper(ROOT)
    assert runs[M.INIT_BN] == ["initbn-s1", "initbn-s2", "initbn-s3"]
    mlp = {"initbn-s1": {"cells": {"init_bn": {"top": {"auc": 0.97}, "jc2_188": {"macro_auc_qcd": 0.9}}}}}
    assert M._mlp(mlp, "initbn-s1", "top") == pytest.approx(0.03)
    assert M._mlp(mlp, "initbn-s1", "jc2_188_qcd") == pytest.approx(0.1)
    assert M._mlp(mlp, "init-s1", "top") is None


def test_the_sculpting_floor_is_the_divergence_of_a_random_selection():
    """_sculpt_floor's (k - 1) / (8 n ln 2) against a Monte Carlo of the same draw: n jets at random
    from the QCD mass histogram, compared with it by the script's own divergence."""
    import numpy as np
    sc = _load("mass_sculpting_t", "experiments/EVAL/mass_sculpting.py")
    rng = np.random.default_rng(1)
    p = rng.random(24)
    p[3] = 0.0
    p /= p.sum()
    doc = {"hist_before": p.tolist(), "working_points": {"0.01": {"n_pass": 4103}}}
    floor = M._sculpt_floor({"a": doc, "b": doc}, "0.01")
    assert floor == pytest.approx(22 / (8 * 4103 * math.log(2)))
    sims = [sc.jsd(rng.multinomial(4103, p) / 4103, p) for _ in range(4000)]
    assert floor == pytest.approx(float(np.mean(sims)), rel=0.05)
    other = {"hist_before": p.tolist(), "working_points": {"0.01": {"n_pass": 4000}}}
    with pytest.raises(SystemExit):
        M._sculpt_floor({"a": doc, "b": other}, "0.01")


def test_the_jetclass_table_has_parts_rows_and_ours_in_its_column_order():
    V4 = M.V4
    runs = M._paper(ROOT)

    def block(k):
        return {"accuracy": 0.80 + k, "macro_auc_ovo": 0.98 + k, "macro_auc_ovr": 0.98 + k,
                "per_class": {c: {"rejection": {"rejection": 1000.0 * (j + 1) + 10 * k, "n_bkg_pass": 200 // (j + 1)}}
                              for j, (c, _) in enumerate(V4.JC1_COLS)}}
    pre = {m[len("mtx-"):]: {"N1000000": {"s1": {"jc1": block(1e-3 * i)}}} for i, m in enumerate(runs["L188"])}
    ref = {M.SCRATCH: {"N1000000": {f"s{k}": {"jc1": block(1e-3 * k)} for k in (1, 2, 3)}}}
    tex = V4.table_jetclass(pre, ref, runs, ("L188", "R16_Q1"), M.LABEL, M.SCRATCH, 1_000_000,
                            MT.fmt_pm, M._two)
    # the 2 M published rows only (FIGURE_PLAN T2); the 100 M rows are caption context
    # the builder writes no bold: bold is _bold's, on the tables of BOLD (PI 2026-11-12), and never \textbf
    # part "all" (the supplement's tab:jetclassall, page review round 3): no training-jets and parameters columns, the
    # same in every row of ours, which the caption states
    assert r"ParT~\cite{part} & 0.836 & 0.9834 & " in tex and "& 100 M &" not in tex and "2 M" not in tex
    assert r"\textbf" not in tex and M.BOLD_ON not in tex and "jetclass" not in M.BOLD
    scratch = next(line for line in tex.splitlines() if line.startswith(M.SCRATCH_LABEL))
    assert scratch.startswith(f"{M.SCRATCH_LABEL} & {MT.fmt_pm(0.802, 0.001)} & ") and "2.21" not in scratch
    assert r"$R_{99}$" in tex and r"$R_{99.5}$" in tex
    assert M.LABEL["R16_Q1"] not in tex          # no curves for it: no row
    # the one row order (PI 2026-11-12): the published rows, a rule, ours with from scratch last; no "this work" row
    # (the italic "fine-tuned" heading is _float's, MODES); the headers in words
    body = tex.splitlines()
    assert body[-3] == scratch and body[-4].startswith(M.LABEL["L188"] + " & ")
    assert body[-5] == r"\midrule" and r"\cite{" in body[-6] and "this work" not in tex
    assert tex.splitlines()[4].startswith(" & accuracy & AUC & ") and V4.PARAMS_HEAD == "parameters"
    assert "training jets" not in tex and V4.PARAMS_HEAD not in tex
    assert M.MODES["jetclass"] == M.MODES["jetclass_rej"] == "ft"
    # 200 // 9 = 22 QCD jets pass for the last column: unmarked; 200 // 10 would be marked
    assert scratch.count(r"$^\dagger$") == 0
    pre["l188-s1"]["N1000000"]["s1"]["jc1"]["per_class"]["label_Zqq"]["rejection"]["n_bkg_pass"] = 19
    tex = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two)
    row = next(line for line in tex.splitlines() if line.startswith(M.LABEL["L188"]))
    assert row.rstrip(" \\").endswith(r"$^\dagger$") and row.count(r"$^\dagger$") == 1
    # PI 2026-11-12: the main text's part (jets, parameters, accuracy, AUC) and the appendix's (jets, rejections)
    summ = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two, part="summary")
    assert r"\begin{tabular}{llccc}" in summ and r"$R_{50}$" not in summ
    assert rf" & training jets & {V4.PARAMS_HEAD} & accuracy & AUC \\" in summ.splitlines()
    assert next(ln for ln in summ.splitlines() if ln.startswith(M.SCRATCH_LABEL)).count(" & ") == 4
    rej = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two, part="rej")
    assert r"$R_{99.5}$" in rej and "accuracy" not in rej and V4.PARAMS_HEAD not in rej and "params" not in rej
    # "training jets" over two header rows there (one row made the turned page overfull)
    head = rej.splitlines()
    assert head[4].startswith(" & training & ") and head[5].startswith(" & jets & ") and head[6] == r"\midrule"
    assert r"ParT~\cite{part} & 2 M & 5{,}587" in rej
    srow = next(ln for ln in rej.splitlines() if ln.startswith(M.SCRATCH_LABEL))
    assert srow.startswith(M.SCRATCH_LABEL + " & 1 M & ") and srow.count(" & ") == 1 + len(V4.JC1_COLS)
    ref[M.SCRATCH]["N1000000"]["s1"]["jc1"]["macro_auc_ovr"] = 0.5
    with pytest.raises(SystemExit, match="one-vs-rest"):
        V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two)


# ------------------------------------------------------------------ Phase B (2026-11-02): new emitters
class _PEm(_Em):
    """An emitter stand-in that also exposes the names emitted, as Emitter.provenance does."""

    @property
    def provenance(self):
        return self.m


def test_two_significant_figures_keep_their_trailing_zeros_and_carry_over():
    assert [M._sig2(x) for x in (17.1, 0.596, 127, 9.96, 1234.5, 4.0667)] == ["17", "0.60", "130", "10", "1{,}200", "4.1"]


def test_a_prose_twin_states_the_direction_in_two_significant_figures():
    assert M.prose([1.15, 1.17, 1.19], "errors") == r"17\% more errors"
    assert M.prose([0.76, 0.76, 0.77], "errors") == r"24\% fewer errors"
    assert M.prose([4.0, 4.1, 4.1], "errors") == "4.1 times the errors"
    assert M.prose([1.27], "rejection") == r"27\% more rejection" and M.prose([0.72], "rejection") == r"28\% less rejection"
    assert M.prose([0.58], "resolution") == "0.58 times the resolution"
    # PI 2026-11-23, LEDGER SP-12 (step 3): no 2-sigma rule; a mean near 1 is still put in words, and whether the text
    # states it is its tab:tests row's (ratio_words)
    assert M.prose([1.01, 1.02, 1.03], "errors") == r"2.0\% more errors"
    assert M.prose([None, None], "errors").startswith(r"\pending")


def test_two_runs_give_the_value_nearer_one_and_its_twin_the_other():
    assert M.prose([1.041, 1.032], "errors") == r"at least 3.2\% more errors"
    assert M.prose([1.041, 1.032], "errors", farther=True) == r"up to 4.1\% more errors"
    assert M.prose([0.35, 0.24], "errors") == "at most 0.35 times the errors"
    assert M.prose([0.35, 0.24], "errors", farther=True) == "down to 0.24 times the errors"
    # PI 2026-11-23, LEDGER SP-12 (step 3): runs on either side of 1 are no longer a bound in words (the 2-sigma rule is
    # gone); the 3-sigma test of the row decides, and such runs never pass it
    assert M.prose([0.98, 1.03], "errors") == r"at least 2.0\% fewer errors"
    em = _PEm()
    M._emit_ratio(em, "MlstX", [1.041, 1.032], 2, MT.fmt_pm, None, "p", "errors", two_run=True)
    # the ratio words are held for ratio_words(), never emitted beside the ratio
    assert set(em.m) == {"MlstX"} and em.m["MlstX"] == "1.041 and 1.032" and set(M._held(em)) == {"MlstXPr", "MlstXPrMax"}
    assert M._held_word(M._held(em)["MlstXPrMax"]) == r"up to 4.1\% more errors"
    M._emit_ratio(em, "MlstY", [1.15, 1.2, 1.25], 3, MT.fmt_pm, None, "p", "errors")
    assert "MlstYPrMax" not in M._held(em) and M._held_word(M._held(em)["MlstYPr"]) == r"20\% more errors"


def test_every_ratio_word_the_text_uses_reads_a_row_of_the_tested_differences():
    """PI 2026-11-23, LEDGER SP-12 (step 3): the ratio words the text uses are exactly PR_ROWS, each naming a row of
    tab:tests (TESTS) and the null phrase its sentence reads; build() stops otherwise (ratio_words)."""
    used = M.text_ratio_words(ROOT / "paper/journal")
    assert used == set(M.PR_ROWS), (sorted(used - set(M.PR_ROWS)), sorted(set(M.PR_ROWS) - used))
    rows = {(h, a, b) for h, rs in M.TESTS for a, b, _ in rs}
    for name, (head, first, second, null) in M.PR_ROWS.items():
        assert (head, first, second) in rows, name
        assert "three standard deviations" in null and not re.search(r"\d", null), name


def test_a_ratio_word_reads_its_rows_verdict_and_the_build_stops_on_one_without_a_row():
    """PI 2026-11-23, LEDGER SP-12 (step 3): |Delta| > 3 sigma, the direction and value in words; not, the sentence's
    null phrase; no stored bootstrap, \\pending{bootstrap} (\\pending{rerun bootstrap} where the file is stale). A ratio
    with no row has no word; the build stops on a word the text uses without a row, on one whose direction is not its
    row's and on one emitted outside ratio_words."""
    name = "MlstCmpJcTwoMidOverFullPr"
    head, first, second, null = M.PR_ROWS[name]

    def word(v, used=()):
        em = _PEm()
        M._emit_ratio(em, "MlstCmpJcTwoMidOverFull", [1.22, 1.23, 1.24], 3, MT.fmt_pm, None, "p", "errors")
        M._emit_ratio(em, "MlstCmpJcTwoMidOverRes", [1.1, 1.1, 1.1], 3, MT.fmt_pm, None, "p", "errors")     # no row
        M.ratio_words(em, {(head, first, second): v}, set(used), None)
        assert {n for n in em.m if n.endswith(("Pr", "PrMax"))} == set(M.PR_ROWS)
        return em.m[name]
    # an AUC row: more errors is a lower AUC, so a passing delta is negative
    assert word((-0.0045, 1e-4, 1e-4, 1e-4, 1.7e-4)) == r"23\% more errors"          # |delta| / sigma 26: passes
    assert word((-0.0004, 1e-4, 1e-4, 1e-4, 1.7e-4)) == null == M._NO_MORE            # 2.4: fails
    assert word(None) == r"\pending{bootstrap}" and word("stale") == r"\pending{rerun bootstrap}"
    with pytest.raises(SystemExit, match="against the sign"):
        word((0.0045, 1e-4, 1e-4, 1e-4, 1.7e-4))
    with pytest.raises(SystemExit, match="no tab:tests row"):
        word(None, used={name, "MlstCmpJcTwoMidOverResPr"})
    em = _PEm()
    em.macro("MlstXPr", r"5\% more errors")
    with pytest.raises(SystemExit, match="outside ratio_words"):
        M.ratio_words(em, {}, set(), None)


def test_the_jetclass_accuracy_of_the_random_network_is_read_at_the_largest_training_set():
    cell = lambda acc: {"accuracy": acc, "macro_auc_ovr": 0.6, "converged": True}
    fits = {f"initbn-s{k}": {"cells": {"init_bn": {"jc2_188": {"1000000": cell(0.1)},
                                                    "jc1": {"1000": cell(0.2), "1000000": cell(0.46 + 0.01 * k)}}}}
            for k in (1, 2)}
    runs = {a: [] for a in M.ALL_ROWS}
    runs[M.INIT], runs[M.INIT_BN] = list(M.INIT_MODELS), list(M.INIT_BN_MODELS)
    em = _PEm()
    M.probe_abs_macros(em, fits, runs, None, MT.fmt_pm)
    assert em.m["MlstLpJcOneAccInitBn"] == r"0.470 and 0.480 \pending{2 of 3}"
    assert em.m["MlstLpJcTwoAccInitBn"] == r"0.100 and 0.100 \pending{2 of 3}"


def test_a_macro_the_outline_names_is_a_marker_until_its_input_exists():
    em, missing = _PEm(), []
    em.macro("MlstGpuTypeBound", "0.2\\%")
    M.required_pass(em, None, missing)
    assert em.m["MlstGpuTypeBound"] == "0.2\\%"
    assert em.m["MlstFtTopRthirtyFullOverScratch"].startswith(r"\pending")
    # PI 2026-11-23, LEDGER SP-12 (step 3): a twin is a ratio word, emitted only with a tab:tests row (PR_ROWS)
    assert "MlstFtTopRthirtyFullOverScratchPr" not in em.m and "MlstMassHeadResMassOverUncorrPrMax" not in em.m
    assert em.m["MlstMassHeadResMassOverUncorrPr"].startswith(r"\pending")
    assert all("_" not in v.replace(r"\_", "") for v in em.m.values())     # LaTeX-safe markers
    assert any(x.startswith("DesignNTrainSelected") for x in missing)


def _fit(c, ce):
    return {"1000": {"C": c, "search": {str(c): {"val_cross_entropy": ce}}, "accuracy": 0.9, "r30": 10.0 * ce,
                     "r30_is_bound": False}}


def test_the_substitution_is_chosen_once_per_task_on_validation_over_every_model():
    fill = {"mtx-a-s1": {"cells": {"best70": {"top": _fit(1.0, 0.30), "top.ntop": _fit(1.0, 0.20), "top.itop4": _fit(10.0, 0.26)}}},
            "init-s1": {"cells": {"init_bn": {"top": _fit(1.0, 0.40), "top.ntop": _fit(1.0, 0.45), "top.itop4": _fit(1.0, 0.30)}}}}
    s, tot = M.choose_substitution(fill, ["mtx-a-s1", "initbn-s1"], "top")
    assert s == "draws" and tot["zero"] == pytest.approx(0.70) and tot["mean"] == pytest.approx(0.65)
    assert M.choose_substitution(fill, ["mtx-a-s1", "initbn-s1", "mtx-b-s1"], "top") == (None, {})
    cells = {"jc2bal": {"1000000": {"accuracy": 0.8}}, "jc2bal.mtop": {"1000000": {"accuracy": 0.6}}}
    assert M._acc_drop(cells, "jc2bal.mtop") == pytest.approx(25.0) and M._acc_drop(cells, "jc2bal.mqg") is None


def test_the_appendix_probe_tables_read_the_substitution_chosen_on_validation():
    zero = {"1000": {"accuracy": 0.9, "auc": 0.95, "r50": 50.0, "r30": 100.0, "r30_is_bound": False}}
    mean = {"1000": {"accuracy": 0.91, "auc": 0.96, "r50": 60.0, "r30": 200.0, "r30_is_bound": False}}
    fits = {"mtx-l188-s1": {"cells": {"best70": {"top": zero}}}, M.KIN_MODEL: {"cells": {"best70": {"top": zero}}}}
    fill = {"mtx-l188-s1": {"cells": {"best70": {"top": zero, "top.ntop": mean}}}}
    assert M._bench_rec(fits, fill, {"top": "mean"}, "mtx-l188-s1", "top")["r30"] == 200.0
    assert M._bench_rec(fits, fill, {}, "mtx-l188-s1", "top")["r30"] == 100.0           # pending: zero
    assert M._bench_rec(fits, fill, {"top": "mean"}, M.KIN_MODEL, "top")["r30"] == 100.0   # nothing missing
    assert M._subst_head({"top": "mean"}, "top") == "missing inputs set to the training mean"
    assert r"\pending" in M._subst_head({}, "top")
    runs = {a: [] for a in M.ALL_ROWS + (M.INIT, M.INIT_BN)}
    runs["L188"] = ["mtx-l188-s1"]
    t = M.table_probes_n(fits, runs, MT.fmt_pm, ("L188",), fill, {"top": "mean"})
    assert "the training mean" in t and M.LABEL["L188"] + " & 200" in t
    chosen, _ = M.chosen_substitutions({}, runs)
    assert chosen == {"top": None, "qg": None, "qg_herwig": None}


def test_the_bootstrap_of_the_test_jets_has_the_composition_of_the_fits():
    runs = {a: [f"mtx-{a.lower()}-s{k}" for k in (1, 2)] for a in ("L188", "L162", "R42_Q1", "R16_Q1")}
    pairs = {f"{m}/{r}": {"macro": {"ratio": 1.1 + 0.01 * k, "ln_ratio_sd": 0.02}}
             for a in ("L162", "R42_Q1", "R16_Q1") for k, (m, r) in enumerate(zip(runs[a], runs["L188"]))}
    natural = {"probe": {"jc2_188_N1000000": {"b": 200, "pairs": pairs}}}
    body, worst = M.table_boot(natural, runs, balanced=True)
    assert not worst["probe"]                       # balanced fits: the natural bootstrap is never read
    body, worst = M.table_boot(natural, runs, balanced=False)
    assert worst["probe"] == [0.02] * 3
    balanced = {"probe_v4": {"jc2bal_N1000000": {"b": 200, "pairs": pairs}}}
    assert M.table_boot(balanced, runs, balanced=True)[1]["probe"] == [0.02] * 3


def test_fine_tuning_cells_without_a_recorded_gpu_are_counted(tmp_path):
    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "budget_metrics.json").write_text(json.dumps({"cells": {
        "leg_top/l188-s1/N1000/s1": {"gpu": "NVIDIA GeForce RTX 3090"},
        "leg2/l188-s1/N10000/s1": {"recipe": {"lr": 1e-3}}, "leg_jc2b/l188-s1/N10000/s1": {}}}))
    types, nodev = M.ft_gpu_types(tmp_path)
    assert types == {"l188-s1": [M.gpu_name("NVIDIA GeForce RTX 3090")]} and nodev == {"l188-s1": 2}
    runs = {"L188": ["mtx-l188-s1"]}
    t, _ = M.table_compute({}, runs, MT.fmt_int, ft_gpus=types, ft_nodev=nodev)
    assert r"\pending{2 cells without a device}" in t


def _atlas(groomed=0.2, mlp=None, reg=None):
    """An atlas_res file: the groomed mass's resolution is `groomed` over all jets and 0.3 for the
    resonant and the QCD jets alone, so a macro shows which group it read."""
    gr = lambda grp: groomed if grp == "all" else 0.3
    g = {grp: {"inclusive": {"resolution": gr(grp), "median": 1.05},
               "vs_pt": {"edges": [200, 400, 2500], "resolution": [0.21, 0.19], "median": [1.0, 1.1], "n": [500, 500]}}
         for grp in ("all", "resonant", "qcd")}
    est = {"groomed": g}
    if mlp is not None:
        est["mlp"] = {grp: {"inclusive": {"resolution": mlp, "median": 1.0}} for grp in ("all", "resonant", "qcd")}
    if reg is not None:
        est["regressed"] = {grp: {"inclusive": {"resolution": reg, "median": 0.995},
                                  "vs_pt": {"edges": [200, 400, 2500], "resolution": [reg, reg], "median": [1, 1], "n": [500, 500]}}
                            for grp in ("all", "resonant", "qcd")}
    return {"estimators": est}


def test_the_mass_table_and_macros_use_the_atlas_resolution_over_all_jets():
    runs = M._paper(ROOT)
    atlas = {"mtx-l162-s1": _atlas(mlp=0.2), "mtx-l162-s2": _atlas(mlp=0.2),
             "mtx-l162mass-s1": _atlas(mlp=0.15, reg=0.116), "mtx-l162mass-s2": _atlas(mlp=0.15, reg=0.118)}
    t = M.table_mass(atlas, runs, MT.fmt_pm)
    # T5: the overall resolution and median response (FIGURE_PLAN T5), not the groups
    assert "soft-drop mass & 0.200 & 1.05 \\\\" in t and "sigma" not in t and "0.300" not in t
    # two blocks under italic headings the builder writes (PI 2026-11-12): the mass output of the pretraining head
    # (the two networks with one; 0.116 and 0.118 over all jets, mean +- SD), then the perceptron on the frozen
    # features of every network, the random network last
    lines = t.splitlines()
    head, mlp = (lines.index(M._full_row(3, M.HEADING[k], italic=True)) for k in ("head", "mlp"))
    # PI 2026-11-14 (audit 3, L5): the soft-drop mass, a reference, last, below a rule
    assert head < mlp < lines.index("soft-drop mass & 0.200 & 1.05 \\\\") and lines[head - 1] == lines[mlp - 1] == r"\midrule"
    assert [ln.split(" & ")[0] for ln in lines[head + 1:mlp - 1]] == [M.LABEL[a] for a in M.MASS]
    assert lines[head + 1] == M.LABEL["L162_MASS"] + r" & 0.1170\,$\pm$\,0.0014 & 0.995 \\"
    assert [ln.split(" & ")[0] for ln in lines[mlp + 1:-4]] == [M.LABEL[a] for a in M.MASS_ROWS]
    # each mass network directly under its own label set (L5)
    assert M.MASS_ROWS == M.SETS + (M.INIT,) and "mass" not in M.MODES
    assert M.LABEL["L162_MASS"] + " & 0.150 & 1.00 \\\\" in lines[mlp:]        # equal runs: fmt_pm's common value
    em = _PEm()
    M.mass_macros(em, atlas, runs, None, MT.fmt_pm)
    assert em.m["MlstMassHeadResMassOverUncorr"] == "0.580 and 0.590"
    # PI 2026-11-23, LEDGER SP-12 (step 3): the ratio words are held for ratio_words(), which reads their tab:tests rows
    assert M._held_word(M._held(em)["MlstMassHeadResMassOverUncorrPr"]) == "at most 0.59 times the resolution"
    assert M._held_word(M._held(em)["MlstMassHeadResMassOverUncorrPrMax"]) == "down to 0.58 times the resolution"
    assert em.m["MlstMassHeadResMassOverMlpRes"].startswith("0.580 and 0.590")
    assert em.m["MlstMassUncorrected"] == "0.200" and em.m["MlstMassUncorrectedResonant"] == "0.300"
    assert em.m["MlstMassUncorrectedQcd"] == "0.300" and em.m["MlstMassHeadResMassResponse"] == "0.995 and 0.995"
    assert em.m["MlstMassHeadResMassOverUncorrResonant"] == "0.3867 and 0.3933"      # 0.116 / 0.3, 0.118 / 0.3
    d = M.table_mass_detail(atlas, runs, MT.fmt_pm)
    assert "200--400 & 400--2500" in d and "\nsoft-drop mass & " in d
    # the upper part, one tabular (page review 2026-11-13): all, resonant and QCD jets for every estimator, the cell
    # over all jets empty for the mass output and the perceptron (tab:mass holds it), filled for the ridge regression
    upper = d.split("%SPLIT%")[0].splitlines()
    assert upper.count("jets & all & resonant & QCD & all & resonant & QCD \\\\") == 1
    assert "jets & resonant & QCD & resonant & QCD" not in d
    assert next(ln for ln in upper if ln.startswith(M.LABEL["L162_MASS"] + " & ")).startswith(M.LABEL["L162_MASS"] + " &  & ")
    assert not next(ln for ln in reversed(upper) if ln.startswith(M.LABEL["L188"] + " & ")).startswith(M.LABEL["L188"] + " &  & ")
    # the binned blocks of the binned mass figures under italic section rows (_turned_long breaks a page only at a
    # mode heading), the soft-drop mass named with its jet group above the network outputs
    group = dict((g, h) for g, _, h in M.MASS_GROUPS)
    for est, g, _ in M.MASS_DETAIL_BINNED:
        word = {"regressed": M.HEADING["head"], "mlp": M.HEADING["mlp"], "ridge": M.RIDGE}[est]
        assert rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{{word}, {group[g]} jets}}}} \\" in d, (est, g)
    assert f"\nsoft-drop mass, {group[M.MASS_DETAIL_BINNED[0][1]]} jets & " in d
    assert r"\textit{QCD jets}" not in d
    # the supplement's detail table names its estimators as the main-text table's headings do, in \itshape in the upper
    # part (not \textit, at which _turned_long starts a page)
    assert all(rf"{{\itshape {h}}}" in d for h in (M.HEADING["head"], M.HEADING["mlp"], M.RIDGE))
    assert "regression output of the pretrained network" not in d
    assert d.index(M.HEADING["head"]) < d.index(M.HEADING["mlp"]) < d.index(M.RIDGE)
    atlas["mtx-l162-s1"] = _atlas(groomed=0.3, mlp=0.2)
    with pytest.raises(SystemExit, match="groomed"):
        M.table_mass(atlas, runs, MT.fmt_pm)


def test_the_sculpting_floor_distance_is_that_of_a_random_selection():
    import numpy as np
    sc = _load("mass_sculpting_t2", "experiments/EVAL/mass_sculpting.py")
    rng = np.random.default_rng(2)
    p = rng.random(24)
    p /= p.sum()
    sims = [math.sqrt(sc.jsd(rng.multinomial(4103, p) / 4103, p)) for _ in range(4000)]
    assert M._floor_distance(24, 4103) == pytest.approx(float(np.mean(sims)), rel=0.05)


def test_the_floor_sd_is_the_spread_of_random_selections():
    import numpy as np
    sc = _load("mass_sculpting_t4", "experiments/EVAL/mass_sculpting.py")
    rng = np.random.default_rng(4)
    p = rng.random(24)
    p /= p.sum()
    sims = [math.sqrt(sc.jsd(rng.multinomial(410, p) / 410, p)) for _ in range(4000)]
    assert M._floor_sd(24, 410) == pytest.approx(float(np.std(sims)), rel=0.1)


def test_the_floor_of_a_selection_drawn_from_the_reference_jets_is_smaller_by_the_finite_population_factor():
    import numpy as np
    sc = _load("mass_sculpting_t3", "experiments/EVAL/mass_sculpting.py")
    rng = np.random.default_rng(3)
    pop = rng.integers(0, 24, 10000)                   # the reference jets' mass bins
    ref = np.bincount(pop, minlength=24) / pop.size
    sims = [math.sqrt(sc.jsd(np.bincount(rng.choice(pop, 4000, replace=False), minlength=24) / 4000, ref))
            for _ in range(2000)]
    assert M._floor_distance(24, 4000, 10000) == pytest.approx(float(np.mean(sims)), rel=0.05)
    assert M._floor_distance(24, 4000, 10000) == pytest.approx(M._floor_distance(24, 4000) * math.sqrt(0.6))


def test_the_sculpting_window_and_the_floor_count_come_from_the_curves():
    runs = M._paper(ROOT)
    hist = [1.0 / 24] * 24
    floor = M._floor_distance(24, 410)
    sculpt, v2 = {}, {}
    for arm in M.MAIN + M.MASS:
        for j, m in enumerate(runs[arm]):
            d1 = 0.02 * (2 if arm == "R16_Q1" else 1) * (1 + 0.01 * j)
            doc = {"hist_before": hist, "bins_jet_sdmass": list(range(25)),
                   "working_points": {"0.1": {"jsd": 1e-4, "n_pass": 41028}, "0.01": {"jsd": d1 ** 2, "n_pass": 4103}},
                   "curve": [{"eff": 0.001, "n_pass": 410, "js_distance": floor * (1 + 0.01 * (j - 1)), "jsd": 0},
                             {"eff": 0.01, "n_pass": 4103, "js_distance": d1, "jsd": 0}]}
            sculpt[m] = v2[m] = doc
    em = _PEm()
    M.sculpt_macros(em, sculpt, v2, runs, None, MT.fmt_pm)
    assert em.m["MlstSculptDiffEffLo"] == em.m["MlstSculptDiffEffHi"] == r"1.0\%"      # two significant figures
    assert em.m["MlstSculptLowestEff"] == r"0.10\%" and em.m["MlstSculptLowestNPass"] == "410"
    # the two-run configurations sit 0.5 % below the floor: the largest distance from it, in floor SDs
    assert em.m["MlstSculptLowestMaxFloorSd"] == f"{0.005 * floor / M._floor_sd(24, 410):.1f}"
    # PI 2026-11-23, LEDGER SP-12 (step 3): ratio words held for ratio_words(), not emitted here
    assert M._held_word(M._held(em)["MlstSculptCoarseOverFullOnePr"]) == "2.0 times the distance"
    assert em.m["MlstSculptFullOne"].startswith("0.020")
    assert "MlstSculptCoarseMassOverCoarseOnePrMax" in M._held(em) and not any(n.endswith(("Pr", "PrMax")) for n in em.m)
    # a 17/188 difference at the lowest stored efficiency, next to the working point on this grid, joins the window
    for m in runs["R16_Q1"]:
        v2[m]["curve"][0]["js_distance"] *= 2
    em = _PEm()
    M.sculpt_macros(em, sculpt, v2, runs, None, MT.fmt_pm)
    assert em.m["MlstSculptDiffEffLo"] == r"0.10\%" and em.m["MlstSculptDiffEffHi"] == r"1.0\%"


def test_the_sculpting_window_is_the_unbroken_stretch_around_the_working_point():
    grid = [0.001, 0.002, 0.005, 0.008, 0.012, 0.02, 0.05]
    # a gap below and a passing point above the gap at 0.02: neither far point joins
    assert M._window_around(grid, [0.001, 0.005, 0.008, 0.012, 0.05], 0.01) == (0.005, 0.012)
    assert M._window_around(grid, grid, 0.01) == (0.001, 0.05)
    assert M._window_around(grid, [0.012], 0.01) == (0.012, 0.012)
    # the grid point nearest the working point fails, or nothing passes: no window
    assert M._window_around(grid, [0.001, 0.05], 0.01) is None
    assert M._window_around(grid, [], 0.01) is None


@pytest.mark.parametrize("passing,lo,hi", [((0.001, 0.005, 0.01), r"0.50\%", r"1.0\%"), ((0.001,), "none", "none"), ((), "none", "none")])
def test_the_sculpting_window_macros_skip_a_gap_and_say_when_no_point_passes(passing, lo, hi):
    runs = M._paper(ROOT)
    grid = [0.001, 0.003, 0.005, 0.01, 0.02]
    sculpt, v2 = {}, {}
    for arm in M.MAIN + M.MASS:
        for m in runs[arm]:
            curve = [{"eff": e, "n_pass": 400, "js_distance": 0.1 * (2 if arm == "R16_Q1" and e in passing else 1), "jsd": 0}
                     for e in grid]
            sculpt[m] = v2[m] = {"hist_before": [1.0 / 24] * 24, "bins_jet_sdmass": list(range(25)), "curve": curve,
                                 "working_points": {"0.1": {"jsd": 1e-4, "n_pass": 41028}, "0.01": {"jsd": 1e-4, "n_pass": 4103}}}
    em = _PEm()
    M.sculpt_macros(em, sculpt, v2, runs, None, MT.fmt_pm)
    assert em.m["MlstSculptDiffEffLo"] == lo and em.m["MlstSculptDiffEffHi"] == hi


def _jc1_pre(runs, rej):
    """Curves cells: rej(model, class) -> (rejection, passing background jets)."""
    import collections as co
    pre = co.defaultdict(dict)
    for arm in ("L188", "R16_Q1"):
        for m in runs[arm]:
            pc = {c: {"rejection": dict(zip(("rejection", "n_bkg_pass"), rej(m, c)))} for c, _ in M.V4.JC1_COLS}
            pre[m[len("mtx-"):]] = {"N1000000": {"s1": {"jc1": {"accuracy": 0.84, "per_class": pc}}}}
    return pre


def test_a_jetclass_rejection_is_a_ratio_per_class_and_no_rule_counts_the_classes_that_differ():
    """PI 2026-11-23, LEDGER SP-12 (step 3): the 2-sigma-and-counting rule (_rej_diff) and the counts it wrote are gone;
    a difference is stated only from its tab:tests row, and the ratio words are held for ratio_words()."""
    runs = M._paper(ROOT)

    def rej(m, c):
        k = int(m[-1])
        if "r16q1" in m and c == "label_H4q":
            return 850.0 + k, 200
        if "r16q1" in m and c == "label_Tbqq":
            return 700.0, 10                    # a dagger column: never a claim
        if "r16q1" in m and c == "label_Hgg":
            return 1000.0 * (0.97 + 0.03 * (k - 2)), 6000    # within twice its SD
        return 1000.0, 200
    em = _PEm()
    M.jetclass_macros(em, _jc1_pre(runs, rej), runs, None, MT.fmt_pm, MT.words)
    assert not any(k in n for n in em.m for k in ("NRejDrop", "NRejDiff", "RejDropClasses")) and not hasattr(M, "_rej_diff")
    assert M._held_word(M._held(em)["MlstFtJcOneRejHfourqCoarseOverFullPr"]) == r"15\% less rejection"
    assert "MlstFtJcOneRejHfourqCoarseOverFullPr" not in em.m
    assert em.m["MlstFtJcOneNSignal"] == "nine" and em.m["MlstFtJcOneAccRandMinusCoarse"].startswith(r"\pending")


def _bench_per(runs):
    rec = lambda v: {"accuracy": 0.94, "auc": 0.986, "r50": 450.0 + v, "r30": 1800.0 + v,
                     "r50_is_bound": False, "r30_is_bound": False}
    by = {m[len("mtx-"):]: {1200000: [rec(10 * i)]} for i, m in enumerate(runs["L188"])}
    by[M.SCRATCH] = {1200000: [rec(-400), rec(-390), rec(-410)]}
    by[M.KIN_INIT] = {1200000: [rec(500)]}
    return by


def test_the_benchmark_tables_show_the_planned_rows_and_no_bold():
    runs = M._paper(ROOT)
    per = {"top": _bench_per(runs), "qg": _bench_per(runs), "qg_herwig": _bench_per(runs)}
    t = M.V4.table_bench("top", per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two,
                         kin=("kin (diagnostic)", M.KIN_INIT))
    # the published names in words (round-1 fix 2026-11-13: no "f.t." anywhere, L7)
    assert "LorentzNet" not in t and "OmniLearned-s" not in t and r"ParT, fine-tuned~\cite{part}" in t and "f.t." not in t
    assert r"3{,}486\,$\pm$\,157" in t and r"1{,}615\,$\pm$\,93" in t and r"\textbf" not in t and M.BOLD_ON not in t
    assert not {"bench_top", "bench_qg", "bench_qg_herwig"} & set(M.BOLD)      # the task tables left the paper: no bold
    assert r"kin (diagnostic) & 2.21\,M & " in t and "inputs set to zero" not in t
    assert M.SCRATCH_LABEL + r" & 2.21\,M & 0." in t
    assert M.LABEL["R16_Q1"] + r" & 2.21\,M & --" in t        # no run read out: dashes
    # the physics baselines (landscape Table 1, EFN Table 3) and the parameters of each paper's table
    assert r"linear EFP~\cite{toptag} & 1\,k & 0.932 & 0.980 & -- & 384 \\" in t
    assert r"OmniLearned-l~\cite{omnilearned} & 423\,M & " in t and r"ParticleNet~\cite{particlenet} & 366\,k & " in t
    q = M.V4.table_bench("qg", per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two, herwig=True)
    assert r"\multicolumn{4}{c}{Herwig}" in q and "PFN-Ex" in q and q.count(r"\cite{") == 8
    assert r"linear EFP~\cite{efn} & -- & -- & 0.8919 & 29.7 & -- &" in q
    assert r"constituent multiplicity~\cite{efn} & -- & -- & 0.8401 & 19.0 & -- &" in q
    # PI 2026-11-12: quark/gluon in the main text on Pythia only, Herwig in its own table (ours only)
    p = M.V4.table_bench("qg", per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two)
    assert "Herwig" not in p and r"\textbf" not in p and M.BOLD_ON not in p
    h = M.V4.table_bench_herwig(per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two)
    assert r"\cite{" not in h and r" & accuracy & AUC & $R_{50}$ & $R_{30}$ \\" in h
    # the one row order (PI 2026-11-12): the label sets, from scratch last
    assert [ln.split(" & ")[0] for ln in h.splitlines()[4:-2]] == [M.LABEL[a] for a in M.MAIN] + [M.SCRATCH_LABEL]
    # top: the published rows, a rule, the label sets, the kinematic-only network, from scratch last; the header in words
    lines = t.splitlines()
    k = len(lines) - 1 - lines[::-1].index(r"\midrule")
    assert [ln.split(" & ")[0] for ln in lines[k + 1:-2]] == [M.LABEL[a] for a in M.MAIN] + ["kin (diagnostic)", M.SCRATCH_LABEL]
    assert r"\cite{" in lines[k - 1] and "this work" not in t
    assert lines[2] == rf" & {M.V4.PARAMS_HEAD} & accuracy & AUC & $R_{{50}}$ & $R_{{30}}$ \\" and "params" not in t
    # as the build calls it (the tables' formatter): the kinematic-only network's one run is marked in every cell
    tk = M.V4.table_bench("top", per, runs, M.MAIN, M.LABEL, M.SCRATCH, M.TAB, M.TAB.two, kin=(M.LABEL["KIN"], M.KIN_INIT))
    tok = M.V4.tok
    assert M.LABEL["KIN"] + r" & 2.21\,M & " + " & ".join(tok("s", v) for v in (0.94, 0.986, 950.0, 2300.0)) + r" \\" in tk.splitlines()
    kin = next(ln for ln in M.uniform_columns(tk).splitlines() if ln.startswith(M.LABEL["KIN"] + " & "))
    assert kin == M.LABEL["KIN"] + r" & 2.21\,M & 0.9400$^{\S}$ & 0.9860$^{\S}$ & 950$^{\S}$ & 2{,}300$^{\S}$ \\"
    assert M.LABEL["KIN"] == r"\Kfull{} classes, kinematic inputs only"


def test_table_one_has_the_four_trained_label_sets_and_the_qcd_subclass_row():
    surv = json.loads((ROOT / "configs/labelmaps/usecase_survival.v1.json").read_text())
    sizes = MT.vocabulary_sizes(ROOT / "configs/labelmaps/rung_label_maps.v1.csv")
    t = MT.table_usecase(surv, sizes, ["L188", "L162", "R42_Q1", "R16_Q1"])
    assert "discriminant & source & 188 & 162 & 43 & 17 \\\\" in t and " 64 " not in t
    assert r"(GN3X-style) & \cite{atlasgnx}, Sec.~6 & $\bullet$ & --- & --- & ---" in t
    assert r"\ref{sec:" not in t


def test_the_data_equivalence_interpolates_on_a_log_size_axis():
    assert M._equiv_jets([1000, 10000, 100000], [100.0, 200.0, 300.0], 150.0) == "3{,}200"
    assert M._equiv_jets([1000, 10000, 100000], [100.0, 200.0, 300.0], 400.0) == r"$>$100{,}000"
    assert M._equiv_jets([1000, 10000], [None, 200.0], 150.0).startswith(r"\pending")


def test_the_gpu_type_bound_reads_the_one_run_on_its_own_gpu():
    runs = M._paper(ROOT)
    cell = lambda oma: {"cells": {"best70": {"jc2_188": {"1000000": {"macro_auc_ovr": 1 - oma}}}}}
    fits = {"mtx-l188-s1": cell(0.020), "mtx-l188-s2": cell(0.022), "mtx-l188-s3": cell(0.0231),
            **{f"mtx-r16q1-s{k}": cell(0.03) for k in (1, 2, 3)}}
    dev = lambda d: {"epochs": [{"device": d}] * 3}
    comp = {"mtx-l188-s1": dev("V100"), "mtx-l188-s2": dev("V100"), "mtx-l188-s3": dev("RTX 3090")}
    em = _PEm()
    M.V4.gpu_bound_macro(em, fits, runs, comp, None)
    # 17/188 ratios 1.5, 1.364, 1.299: index 3 against the mean of the others, |1.299 / 1.432 - 1|;
    # the 162- and 43-class runs have no fit here, so the 17-class ratio is the bound; never a ratio
    # within one configuration (PI 2026-11-02: 1 - AUC only between pretraining configurations)
    assert em.m["MlstGpuTypeBound"] == "9.3\\%" and "MlstGpuTypeBoundRatio" not in em.m
    # the other two indices differ by (1.5 - 1.364) / 1.432
    assert em.m["MlstGpuOtherSeedsDiff"] == "9.5\\%"
    fits.update({f"mtx-l162-s{k}": cell(c) for k, c in ((1, 0.020), (2, 0.022), (3, 0.0231 * 1.2))})
    em = _PEm()
    M.V4.gpu_bound_macro(em, fits, runs, comp, None)
    assert em.m["MlstGpuTypeBound"] == "20.0\\%"            # the 162-class ratio departs most
    comp["mtx-l188-s3"] = dev("V100")
    em = _PEm()
    M.V4.gpu_bound_macro(em, fits, runs, comp, None)
    assert em.m["MlstGpuTypeBound"].startswith(r"\pending") and em.m["MlstGpuOtherSeedsDiff"].startswith(r"\pending")


def test_the_part_reproduction_table_and_macros(tmp_path):
    d = tmp_path / "v2"
    (d / "partrepro").mkdir(parents=True)
    (d / "index.json").write_text("{}")
    for v, r30 in (("part", 1711.0), ("head", 1577.0), ("rescale", 1224.0)):
        (d / "partrepro" / f"{v}.json").write_text(json.dumps(
            {"accuracy": 0.94, "auc": 0.9858, "r50": 420.0, "r30": r30, "r30_n_bkg_pass": 118}))
    em, missing = _PEm(), []
    t = M.partrepro(em, d, {}, MT.fmt_pm, missing)
    assert em.m["MlstPartreproRthirtyPart"] == "1{,}711" and em.m["MlstPartreproRthirtyRescaleOverPart"] == "0.715"
    assert M._held_word(M._held(em)["MlstPartreproRthirtyRescaleOverPartPr"]) == r"28\% less rejection"     # SP-12 (step 3)
    assert not missing
    # three runs of ParT's script, each a single run and a single change, not cumulative; ParT's published row and our
    # from-scratch row are in the comparison table only (PI 2026-11-12: one table per number). Round-1 fix 2026-11-13:
    # the caption says once that every value is a single run, so no cell carries the single-run mark
    rows = t.splitlines()[4:-2]
    assert [ln.split(" & ")[0] for ln in rows] == [name for _, name in M.PARTREPRO] == [
        "Particle Transformer training script, unchanged", "the same script with Sophon's output head",
        r"the same script with Sophon's jet-$p_{\mathrm{T}}$ rescaling"]
    assert "ParT, published" not in t and M.SCRATCH_LABEL not in t and "+" not in t
    assert rows[0] == r"Particle Transformer training script, unchanged & 0.9400 & 0.9858 & 420 & 1{,}711 & 118 \\"
    assert not any(M.SINGLE in ln for ln in rows) and r"1{,}224 & " in rows[2]
    assert "Every value is from a single run" in M.printed_caption("partrepro")
    # with the fine-tuning read-out, each column at the place of the top-tagging column it is read against (from
    # scratch 1,400 +- 200: hundreds; 420 +- 20: tens), the macros unchanged
    rec = lambda acc, r50, r30: {"accuracy": acc, "auc": 0.9858, "r50": r50, "r30": r30}      # noqa: E731
    per = {"top": {"l188-s1": {1200000: [rec(0.94, 460.0, 1800.0)]},
                   M.SCRATCH: {1200000: [rec(0.939, 400.0, 1200.0), rec(0.940, 420.0, 1400.0), rec(0.941, 440.0, 1600.0)]}}}
    em2 = _PEm()
    t = M.partrepro(em2, d, per, MT.fmt_pm, [], runs={"L188": ["mtx-l188-s1"], "__bench__": ("L188",)})
    assert [ln.split(" & ")[3:5] for ln in t.splitlines()[4:-2]] == [
        ["420", "1{,}700"], ["420", "1{,}600"], ["420", "1{,}200"]]
    assert em2.m == em.m


def test_the_launched_from_scratch_rates_are_the_ones_the_paper_states():
    assert M.V4.launched_scratch_rates(ROOT) == {"top": "1e-3", "qg": "3e-3", "jc1": "1e-3", "jc2b": "1e-3"}
    sweep = DATA / "budget" / "lrsweep_metrics.json"
    if sweep.exists():
        assert M.V4.choose_rates(json.loads(sweep.read_text()))[1] == {"leg_qg": "3e-3", "leg_top": "1e-3"}


def test_the_models_without_a_balanced_nonlinear_probe_read_natural_composition():
    runs = {**{a: [f"mtx-{a.lower().replace('_', '')}-s{k}" for k in (1, 2, 3)] for a in M.MAIN},
            "L162_MASS": ["mtx-l162mass-s1", "mtx-l162mass-s2"], "R16_Q1_MASS": ["mtx-r16q1mass-s1", "mtx-r16q1mass-s2"],
            "RAND2_p1": ["mtx-rand2p1-s1"]}
    rows = M.MAIN + ("L162_MASS", "RAND2_p1")

    def cell(oma, jc2=True):
        rec = {"auc": 1 - oma, "r30": 100.0}
        tasks = {t: rec for t in ("top", "qg", "qg_herwig")}
        tasks["jc1"] = {"macro_auc_ovr": 1 - oma}
        if jc2:
            tasks["jc2_188"] = {"macro_auc_ovr": 1 - oma, "macro_auc_resonant": 1 - oma, "macro_auc_qcd": 1 - oma,
                                "per_class_auc": {"3": 1 - oma}}
            tasks.update({k: {"auc": 1 - oma} for k, *_ in M.MLP_PAIRS})
        return {"cells": {"best70": tasks}}
    models = [m for ms in runs.values() for m in ms]
    oma = lambda m: 0.04 if m.startswith("mtx-l188") else 0.042 if "mass" in m else 0.05
    nat = {m: cell(oma(m)) for m in models}
    mlp = {m: cell(0.03, jc2=any(m in runs[a] for a in M.MAIN)) for m in models}
    t = M.table_mlp(mlp, runs, MT.fmt_pm, rows, nat=nat)
    assert r"\pending" not in t
    blocks = t.split("%SPLIT%")
    assert len(blocks) == 4 and "mixed test sample" in blocks[3]       # PI 2026-11-14: the mixed test sample
    assert next(ln for ln in blocks[0].splitlines() if ln.startswith(M.LABEL["L162_MASS"])).split(" & ")[1:5] == ["--"] * 4
    assert not any(ln.startswith(M.LABEL["L162_MASS"]) for b in blocks[1:3] for ln in b.splitlines())
    assert next(ln for ln in blocks[3].splitlines() if ln.startswith(M.LABEL["RAND2_p1"])).split(" & ")[2] == M._lone(1.25)
    # no configuration lacks a balanced fit: no natural-composition block, whatever nat holds
    assert "natural-composition" not in M.table_mlp(nat, runs, MT.fmt_pm, rows, nat=nat)
    em = _PEm()
    M.mlp_natural_macros(em, nat, runs, None, MT.fmt_pm)
    assert em.m["MlstMlpNatJcTwoResMassOverRes"] == M._cell([0.84, 0.84], 2, MT.fmt_pm, sep=M.TEXT_SEP)
    # PI 2026-11-23, LEDGER SP-12 (step 3): ratio words held for ratio_words()
    assert "MlstMlpNatJcTwoResMassOverResPr" in M._held(em) and "MlstMlpNatJcTwoCoarseMassOverCoarsePrMax" in M._held(em)
    assert M._per_class(M._nat_as_fits(nat), "mtx-l188-s1") == {3: 0.96}


def test_the_batchnorm_pair_bound_leaves_out_the_random_partitions():
    def cell(best, bn):
        one = lambda oma: {"jc2_188": {"1000000": {met: 0.98 for _, met, _ in M.JC2_TASKS}},
                           "jc2_188_pairs": {"1000000": {"bb_vs_cc": {"auc": 1 - oma}}}}
        return {"cells": {"best70": one(best), "best70_bn": one(bn)}}
    runs = {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"], "RAND2_p1": ["mtx-rand2p1-s1"]}
    fits = {"mtx-l188-s1": cell(0.01, 0.01), "mtx-r16q1-s1": cell(0.02, 0.0202), "mtx-rand2p1-s1": cell(0.02, 0.03)}
    em = _PEm()
    M.bn_macros(em, fits, runs, None)
    assert em.m["MlstBnMaxDevPair"] == "1.0\\%" and em.m["MlstBnMaxDev"] == "0.0\\%"
    em = _PEm()
    M.bn_macros(em, fits, {"L188": runs["L188"], "RAND2_p1": runs["RAND2_p1"]}, None)
    assert em.m["MlstBnMaxDevPair"].startswith(r"\pending")


def test_the_top_rejection_at_half_signal_efficiency_masks_a_curve_below_its_cut():
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"], "R16_Q1": ["mtx-r16q1-s1", "mtx-r16q1-s2", "mtx-r16q1-s3"]}
    doc = lambda rej, cut=0.1: {"eps_grid": [0.3, 0.5], "curves": {"top": {"rejection": [9999.0, rej], "cut": {"eps_s": cut}}}}
    top = {**{m: doc(400.0) for m in runs["L188"]}, **{m: doc(r) for m, r in zip(runs["R16_Q1"], (368.0, 372.0, 376.0))}}
    em = _PEm()
    M.top_r50_macro(em, {"top_ft": top}, runs, None, MT.fmt_pm)
    assert em.m["MlstFtTopRfiftyCoarseOverFull"] == MT.fmt_pm(0.93, 0.01)
    assert M._held_word(M._held(em)["MlstFtTopRfiftyCoarseOverFullPr"]) == r"7.0\% less rejection"     # SP-12 (step 3)
    top["mtx-r16q1-s1"] = doc(372.0, cut=0.6)
    em = _PEm()
    M.top_r50_macro(em, {"top_ft": top}, runs, None, MT.fmt_pm)
    assert r"\pending{2 of 3}" in em.m["MlstFtTopRfiftyCoarseOverFull"]
    em = _PEm()
    M.top_r50_macro(em, {}, runs, None, MT.fmt_pm)
    assert "MlstFtTopRfiftyCoarseOverFull" not in em.m and "MlstFtTopRfiftyCoarseOverFull" in {n for n, _, _ in M.REQUIRED}


def test_the_ee_mumu_auc_over_the_partitions_that_merge_the_pair():
    aucs = {"RAND2_p1": 0.99, "RAND2_p2": 0.98, "RAND2_p3": 0.5}
    runs = {**{a: [] for a in M.MAIN + M.FLAV + (M.INIT, M.INIT_BN)}, **{a: [] for a in M.RAND},
            **{a: [f"mtx-{a.lower().replace('_', '')}-s1"] for a in aucs}}
    pair = lambda auc: {"cells": {"best70": {M.PAIR_DS: {M.PAIR_N: {"ee_vs_mm": {
        "auc": auc, "n_train": 9575, "n_val": 510, "n_test_signal": 5000, "n_test_background": 5000}}}}}}
    fits = {runs[a][0]: pair(x) for a, x in aucs.items()}
    status = {a: {"ee_vs_mm": "apart" if a == "RAND2_p3" else "merged"} for a in M.RAND}
    em = _PEm()
    M.eemm_macros(em, fits, runs, None, MT.fmt_pm, MT.fmt_int, status)
    assert em.m["MlstLpEemmAucRandMerged"] == M._cell([0.99, 0.98], 2, MT.fmt_pm, sep=M.TEXT_SEP)
    em = _PEm()
    M.eemm_macros(em, fits, runs, None, MT.fmt_pm, MT.fmt_int, {a: {"ee_vs_mm": "apart"} for a in M.RAND})
    assert "MlstLpEemmAucRandMerged" not in em.m
    assert "MlstLpEemmAucRandMerged" in {n for n, _, _ in M.REQUIRED}


def test_the_inputs_table_reads_which_inputs_each_data_set_lacks():
    t = M.table_inputs(ROOT, None)
    assert r"electric charge & -- & $\bullet$" in t and r"$\sigma_{d_0}$ & -- & --" in t     # one dash everywhere (PI 2026-11-11)
    assert r"$\Delta\eta$ & $\bullet$ & $\bullet$ & -- \\" in t and r"\pending{offsets}" in t
    assert "running mean" not in t          # the substitutions are said once, in the caption (PI 2026-11-12)
    assert "running mean" in M.CAPTIONS["inputs"][1]
    assert t.count(r" \\") == 17 + 1


def test_the_inputs_table_and_macro_read_the_input_normalization_offsets(tmp_path):
    off = {n: -0.1 for n in ("part_charge", *M.FLAGS, "part_d0", "part_d0err", "part_dz", "part_dzerr")}
    off["part_isChargedHadron"] = -0.957
    t = M.table_inputs(ROOT, off)
    assert r"\pending{offsets}" not in t and "$-0.96$" in t
    src = tmp_path / "input_bn_offsets.json"
    src.write_text("{}")
    em = _PEm()
    M.input_bn_macros(em, off, src)
    assert em.m["MlstInputBnZeroSigmaMax"] == "0.96"


def test_input_bn_offsets_put_a_zero_at_minus_mean_over_sigma():
    torch = pytest.importorskip("torch")
    IB = _load("input_bn_offsets", "experiments/EVAL/input_bn_offsets.py")
    names = IB.names()
    assert len(names) == 17 and "part_isChargedHadron" in names and not any("zero" in n for n in names)
    state = {"mod.embed.input_bn.running_mean": torch.full((17,), 0.5),
             "mod.embed.input_bn.running_var": torch.full((17,), 4.0 - IB.EPS)}
    assert IB.offsets(state, names)["part_charge"] == pytest.approx(-0.25)
    with pytest.raises(SystemExit):
        IB.offsets(state, names[:-1])


def test_the_r30_counting_spread_of_a_label_set_ratio():
    runs = {"A": ["mtx-a-s1", "mtx-a-s2"], "B": ["mtx-b-s1", "mtx-b-s2"]}
    rec = {"mtx-a-s1": {"r30": 110, "r30_rel_stat": 0.1}, "mtx-a-s2": {"r30": 90, "r30_rel_stat": 0.1},
           "mtx-b-s1": {"r30": 100, "r30_rel_stat": 0.1}, "mtx-b-s2": {"r30": 100, "r30_rel_stat": 0.1}}
    between, test = M.count_spreads(rec.get, runs, "A", "B")
    assert between == pytest.approx(statistics.stdev([math.log(1.1), math.log(0.9)]))
    assert test == pytest.approx(math.hypot(0.1, 0.1))
    rec["mtx-b-s2"] = None
    assert M.count_spreads(rec.get, runs, "A", "B") == (None, None)


def test_where_the_per_class_cost_sits_follows_the_label_maps():
    runs = M._paper(ROOT)
    q, lep = M._orbits(ROOT)
    assert q["label_X_bb"] == q["label_X_cc"] and lep["label_X_ee"] == lep["label_X_mm"]
    assert lep["label_X_tauhtaue"] == lep["label_X_tauhtaum"] and lep["label_X_bb"] != lep["label_X_cc"]
    names = [r["class_name"] for r in csv.DictReader((ROOT / "configs/labelmaps/rung_label_maps.v1.csv").open())]
    lepton = {n for n in names if n in lep and sum(lep[x] == lep[n] for x in lep) > 1}

    def fit(auc):
        return {"cells": {"best70": {"jc2_188": {"1000000": {"per_class_auc": {str(i): auc(n) for i, n in enumerate(names)}}}}}}
    fits = {m: fit(lambda n: 0.99) for m in runs["L188"]}
    fits.update({m: fit(lambda n: 0.95 if n in lepton else 0.97) for m in runs["R16_Q1"]})
    em = _PEm()
    M.per_class_macros(em, fits, runs, ROOT, None, MT.fmt_pm, lambda n: n)
    assert em.m["MlstPerClassCoarseNDoubled"] == "188" and em.m["MlstPerClassCoarseLeptonMedian"] == "5.00"
    assert em.m["MlstPerClassCoarseTopClass"] in lepton and int(em.m["MlstPerClassCoarseLeptonN"]) > 0
    assert int(em.m["MlstPerClassCoarseQuarkN"]) > 0 and em.m["MlstPerClassCoarseQuarkMedian"] in ("3.00", "4.00", "5.00")
    # from the label maps alone: the classes merged with both a quark- and a lepton-flavor counterpart
    assert em.m["MlstPerClassCoarseQuarkLeptonN"] == "64"
    assert em.m["MlstPerClassRandFlavMergedOverApart"].startswith(r"\pending")


def _pc(aucs: dict) -> dict:
    """A v3-layout probe file with only the per-class AUCs of the largest JetClass-II set."""
    return {"cells": {"best70": {"jc2_188": {"1000000": {"per_class_auc": {str(c): a for c, a in aucs.items()}}}}}}


def test_the_per_class_auc_difference_is_taken_against_the_188_class_run_of_the_same_index():
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"], "R16_Q1": ["mtx-r16q1-s1", "mtx-r16q1-s3"]}
    fits = {"mtx-l188-s1": _pc({0: 0.99, 1: 0.95}), "mtx-l188-s2": _pc({0: 0.98, 1: 0.96}),
            "mtx-r16q1-s1": _pc({0: 0.97, 1: 0.95}), "mtx-r16q1-s3": _pc({0: 0.5, 1: 0.5})}
    d = M.per_class_auc_diff(fits, runs, "R16_Q1")
    # percentage points, 100 (0.97 - 0.99) and 100 (0.95 - 0.95); s3 has no 188-class partner
    assert set(d) == {0, 1} and d[0] == pytest.approx([-2.0]) and d[1] == pytest.approx([0.0])
    assert M.per_class_auc_diff(fits, runs, "L188") == {0: [0.0, 0.0], 1: [0.0, 0.0]}


def test_a_class_near_auc_one_that_keeps_its_auc_but_triples_its_errors_is_not_a_stable_class():
    runs = M._paper(ROOT)
    near_one, quiet = 50, [40, 41, 42, 43]
    def auc(arm, c):
        if c == near_one:                             # 0.03 points of AUC, three times the errors
            return 0.9995 if arm == "L188" else 0.9985 if arm == "R16_Q1" else 0.9995
        if c in quiet or arm == "L188":
            return 0.9
        return 0.89
    fits = {m: _pc({c: auc(arm, c) for c in range(188)}) for arm in M.MAIN for m in runs[arm]}
    sel = M.jc2_roc_classes(fits, runs, ROOT, None)
    assert sel["change"][near_one] < sel["change"][100] and sel["ratio_dev"][near_one] == pytest.approx(2.0)
    assert near_one not in sel["stable"] and set(sel["stable"]) == set(quiet)


def test_the_jetclass2_roc_classes_are_the_largest_drops_and_the_least_changed_among_the_thick_classes():
    runs = M._paper(ROOT)
    big, quiet = list(range(9)), [40, 41, 42, 43]     # drops of 1..9 percentage points at 17; unchanged
    def auc(arm, c):
        if c in quiet or arm == "L188":
            return 0.99
        if c in big:
            return 0.99 - 0.01 * (big.index(c) + 1) if arm == "R16_Q1" else 0.99
        return 0.99 - 0.002                           # every other class: 0.2 points at every coarser level
    fits = {m: _pc({c: auc(arm, c) for c in range(188)}) for arm in M.MAIN for m in runs[arm]}
    counts = [5000] * 188
    counts[big[8]] = 500                               # the largest drop is a thin class: not a candidate
    order = [int(r["jet_label"]) for r in M._class_rows(ROOT)]
    in_order = lambda cs: sorted(cs, key=order.index)
    sel = M.jc2_roc_classes(fits, runs, ROOT, {"counts": {"ft_test": counts}})
    assert sel["hurt"] == in_order(big[:8]) and sel["stable"] == in_order(quiet)
    assert sel["drop"][big[3]] == pytest.approx(4.0) and sel["change"][big[3]] == pytest.approx(4.0)
    assert sel["change"][quiet[0]] == 0 and sel["change"][100] == pytest.approx(0.2)
    assert sel["ratio_dev"][quiet[0]] == 0 and sel["ratio_dev"][100] == pytest.approx(0.2)
    # without the census every class is a candidate
    assert sel["hurt"] != M.jc2_roc_classes(fits, runs, ROOT, None)["hurt"]
    assert big[8] in M.jc2_roc_classes(fits, runs, ROOT, None)["hurt"]
    em = _PEm()
    M.auc_diff_macros(em, fits, runs, ROOT, {"counts": {"ft_test": counts}}, None, lambda n: n)
    names = {int(r["jet_label"]): r["class_name"] for r in M._class_rows(ROOT)}
    assert em.m["MlstRocJcTwoHurtMinDrop"] == "1.0" and em.m["MlstRocJcTwoHurtMaxDrop"] == "8.0"
    assert em.m["MlstRocJcTwoStableMaxChange"] == "0"
    assert em.m["MlstRocJcTwoStableList"] == M._and([names[c] for c in in_order(quiet)])
    assert em.m["MlstPerClassAucDiffCoarseMin"] == "-9.0" and em.m["MlstPerClassAucDiffCoarseMinClass"] == names[big[8]]
    assert float(em.m["MlstPerClassAucDiffCoarseSeedSdMedian"]) < 1e-9     # equal runs (up to rounding of the mean)


def test_the_list_of_configurations_without_a_curve_reads_as_a_noun_phrase():
    assert M.no_curve_phrase(["L188"]) == r"the \Kfull-class model"
    assert M.no_curve_phrase(["L188", "L162"] + list(M.FLAV)) == \
        r"the \Kfull- and \Kres-class models and the three random partitions with controlled flavor boundaries"
    assert M.no_curve_phrase(["R42_Q1", "R16_Q1", "FLAV_F1"]) == \
        r"the \Kmid- and \Kcoarse-class models and the random partition with a $b$-quark boundary"
    assert M._and(["a", "b", "c"]) == "a, b and c" and M._and(["QCD, $b$", "x"]) == "QCD, $b$ and x"


def _curve(cut, n_bkg=200000, sha="s"):
    """One curve of the roc_points.py schema on the grid of _roc_tree (cut None: no point has 20 background jets)."""
    return {"rejection": [1e4, 100.0, 5.0], "n_background": n_bkg, "label_sha256": sha,
            "cut": None if cut is None else {"eps_s": cut, "eps_b": 1e-4, "rejection": 1e4, "n_bkg_pass": 20}}


def _roc_tree(d, curves: dict, family="pairs", min_bkg=20):
    """curves_v2/<family>/<model>.json with {model: {key: curve}} on a three-point grid."""
    (d / "curves_v2" / family).mkdir(parents=True, exist_ok=True)
    for m, keys in curves.items():
        (d / "curves_v2" / family / f"{m}.json").write_text(json.dumps(
            {"eps_grid": [0.1, 0.5, 0.9], "min_bkg_pass": min_bkg, "curves": keys}))


def test_the_roc_curves_are_cut_where_fewer_than_twenty_background_jets_pass(tmp_path):
    runs = M._paper(ROOT)
    keep = {"L188", "L162", "FLAV_F0", "FLAV_F1", "FLAV_F1R"}            # draw no X->ee vs X->mumu curve
    # F4 draws a configuration only where every run survives the cut: one run without a cut, or a cut
    # that leaves fewer than ROC_MIN_POINTS grid points (above 0.9: none; above 0.5: one), remove the
    # curve as a null cut in every run does
    odd = {"FLAV_F1": [0.95, 0.95], "FLAV_F1R": [None, 0.06], "R42_Q1": [0.06, 0.7, 0.06]}
    assert M.ROC_MIN_POINTS == 2
    cur = {}
    for arm in M.MAIN + M.FLAV:
        for k, m in enumerate(runs[arm]):
            cut = odd[arm][k] if arm in odd else None if arm in keep else 0.06
            cur[m] = {"bb_vs_cc": _curve(0.3 + 0.1 * k), "ee_vs_mm": _curve(cut, n_bkg=5000, sha="e")}
    _roc_tree(tmp_path, cur)
    _roc_tree(tmp_path, {m: {"label_Tbqq": _curve(0.2 + 0.1 * k, sha="t"), "label_Tbl": _curve(0.95, sha="l")}
                         for k, m in enumerate(runs["L188"])}, family="jc1_ft")
    roc = M.V4.load_roc(tmp_path)
    assert set(roc) == {"pairs", "jc1_ft"} and M.V4.JC1_MIN_BKG == 20
    assert all(math.isnan(x) for x in M.V4.roc_series(roc["pairs"][runs["FLAV_F1"][0]], "ee_vs_mm")[1])
    first = roc["pairs"][runs["L188"][0]]                                 # cut at 0.3
    eps, rej = M.V4.roc_series(first, "bb_vs_cc")
    assert list(eps) == [0.1, 0.5, 0.9] and math.isnan(rej[0]) and list(rej[1:]) == [100.0, 5.0]
    assert all(math.isnan(x) for x in M.V4.roc_series(first, "ee_vs_mm")[1])           # cut null: nothing drawn
    assert M.V4.roc_series(first, "qg") is None
    assert M.V4.roc_limit(first, "ee_vs_mm") == 250
    em, missing = _PEm(), []
    M.roc_macros(em, roc, runs, None, missing, MT.fmt_int)
    n = len(runs["L188"])
    assert em.m["MlstRocCutBbFull"] == f"{0.3 + 0.1 * (n - 1):.2f}" and em.m["MlstRocCutEemmFull"] == "not reached"
    assert em.m["MlstRocCutEemmCoarse"] == "0.06" and em.m["MlstRocCutEemmMid"] == "not reached"
    assert em.m["MlstRocCutEemmFone"] == "not reached" and em.m["MlstRocCutEemmFoner"] == "not reached"
    assert em.m["MlstRocCutJcOneTbqqFull"] == f"{0.2 + 0.1 * (n - 1):.2f}" and em.m["MlstRocCutJcOneTblFull"] == "not reached"
    assert "MlstRocCutJcOneHbbFull" not in em.m                                       # no such curve: no macro
    assert em.m["MlstRocNBkgBb"] == "200{,}000" and em.m["MlstRocRejMaxBb"] == "10{,}000"
    assert em.m["MlstRocNBkgEemm"] == "5{,}000" and em.m["MlstRocRejMaxEemm"] == "250"
    assert em.m["MlstRocNoCurveEemm"] == M.no_curve_phrase(sorted(keep | {"R42_Q1"})) and not missing
    # every configuration drawing a curve leaves the sentences that name those without one false: a marker
    for m in cur:
        cur[m]["ee_vs_mm"] = _curve(0.06, n_bkg=5000, sha="e")
    _roc_tree(tmp_path, cur)
    em, missing = _PEm(), []
    M.roc_macros(em, M.V4.load_roc(tmp_path), runs, None, missing, MT.fmt_int)
    assert em.m["MlstRocNoCurveEemm"].startswith(r"\pending") and any("MlstRocNoCurveEemm" in x for x in missing)
    # every test jet of the X->ee / X->mumu pair (pairs_sib2) replaces the balanced set for that pair only
    _roc_tree(tmp_path, {m: {"ee_vs_mm": _curve(None if m in runs["L188"] else 0.06, n_bkg=132000, sha="s")}
                         for m in cur}, family="pairs_sib2")
    em, missing = _PEm(), []
    M.roc_macros(em, M.V4.load_roc(tmp_path), runs, None, missing, MT.fmt_int)
    assert em.m["MlstRocNBkgEemm"] == "132{,}000" and em.m["MlstRocRejMaxEemm"] == "6{,}600"
    assert em.m["MlstRocNBkgBb"] == "200{,}000" and em.m["MlstRocCutEemmFull"] == "not reached"
    assert em.m["MlstRocNoCurveEemm"] == M.no_curve_phrase(["L188"])


def test_curves_of_one_key_scored_on_different_jets_another_cut_or_another_grid_stop_the_build(tmp_path):
    _roc_tree(tmp_path, {"mtx-l188-s1": {"top": _curve(0.1, sha="a")}, "mtx-l188-s2": {"top": _curve(0.1, sha="b")}},
              family="top_ft")
    with pytest.raises(SystemExit, match="different jets"):
        M.V4.load_roc(tmp_path)
    _roc_tree(tmp_path / "x", {"mtx-l188-s1": {"top": _curve(0.1)}}, family="top_ft", min_bkg=50)
    with pytest.raises(SystemExit, match="min_bkg_pass"):
        M.V4.load_roc(tmp_path / "x")
    _roc_tree(tmp_path / "g", {"mtx-l188-s1": {"top": _curve(0.1)}}, family="top_ft")
    (tmp_path / "g/curves_v2/top_ft/mtx-l188-s2.json").write_text(json.dumps(
        {"eps_grid": [0.1, 0.5], "min_bkg_pass": 20, "curves": {"top": _curve(0.1)}}))
    with pytest.raises(SystemExit, match="signal-efficiency grids"):
        M.V4.load_roc(tmp_path / "g")


def test_the_ee_mumu_probe_joins_the_v4_pairs_and_gives_its_macros(tmp_path):
    cell = lambda auc, ntr=9575: {"auc": auc, "n_train": ntr, "n_val": 1064, "n_test_signal": 5000,
                                  "n_test_background": 5000, "C": 1.0, "converged": True}
    jc2 = {"converged": True, **{k: 0.9 for k in ("macro_auc_ovr", "macro_auc_trained", "macro_auc_resonant",
                                                    "macro_auc_qcd")}}
    v4 = lambda cks: {"c_grid": [1.0], "cells": {ck: {"jc2bal": {"1000000": jc2}, "jc2bal_pairs": {}} for ck in cks}}
    sib = lambda cks, auc: {"cells": {ck: {"jc2bal_pairs": {"ee_vs_mm": cell(auc)}} for ck in cks}}
    (tmp_path / "linprobe/fits_v4").mkdir(parents=True)
    (tmp_path / "linprobe/fits_v4_sib").mkdir(parents=True)
    # one random partition that merges the pair and one that keeps it apart, from the label maps
    status = M.merge_status(ROOT)
    merged, apart = (next(a for a in M.RAND if status[a]["ee_vs_mm"] == s) for s in ("merged", "apart"))
    rand = {merged: f"mtx-{merged.lower().replace('_', '')}-s1", apart: f"mtx-{apart.lower().replace('_', '')}-s2"}
    for m, cks, auc in (("mtx-l188-s1", ["best70"], 1.0), ("mtx-r42q1-s1", ["best70"], 0.96),
                        ("init-s1", ["init", "init_bn"], 0.7), (rand[merged], ["best70"], 0.9),
                        (rand[apart], ["best70"], 1.0)):
        (tmp_path / "linprobe/fits_v4" / f"{m}.json").write_text(json.dumps(v4(cks)))
        (tmp_path / "linprobe/fits_v4_sib" / f"{m}.json").write_text(json.dumps(sib(cks, auc)))
    (tmp_path / "linprobe/fits_v4_sib" / "mtx-l162-s1.json").write_text(json.dumps(sib(["best70"], 0.99)))
    fits, raw = M.V4.load_fits(tmp_path)
    assert M._pair(fits, "mtx-r42q1-s1", "ee_vs_mm") == 0.96 and M._pair(fits, "initbn-s1", "ee_vs_mm") == 0.7
    assert "mtx-l162-s1" not in fits                     # no v4 file: not merged
    (tmp_path / "linprobe/fits_v4_sib2").mkdir(parents=True)  # every test jet of the pair replaces the balanced set
    (tmp_path / "linprobe/fits_v4_sib2" / "mtx-r42q1-s1.json").write_text(json.dumps(sib(["best70"], 0.955)))
    assert M._pair(M.V4.load_fits(tmp_path)[0], "mtx-r42q1-s1", "ee_vs_mm") == 0.955
    (tmp_path / "linprobe/fits_v4_sib2" / "mtx-r42q1-s1.json").unlink()
    runs = {"L188": ["mtx-l188-s1"], "R42_Q1": ["mtx-r42q1-s1"]}
    assert M.separated(fits, runs, "ee_vs_mm") == {"1"} and M.separated(fits, runs, "bb_vs_cc") == set()
    # the columns follow the class-pair figures' panel order (PAIR_ORDER, PI 2026-11-13): the lepton pair is the last
    assert [k for k, *_ in M.PAIRS_FIG] == list(M.PAIR_ORDER) and M.PAIRS_FIG[-1][0] == "ee_vs_mm"
    # the comparisons the text quotes: a dash and words, never a division by zero
    em = _PEm()
    M.comparisons(em, fits, {**{a: [] for a in M.ALL_ROWS}, **runs, **{a: [m] for a, m in rand.items()}},
                  status, None, MT.fmt_pm)
    undefined = "undefined, since the reference separates the test jets completely"
    word = lambda n: M._held_word(M._held(em)[n])      # noqa: E731  (ratio words held for ratio_words(), SP-12 step 3)
    assert em.m["MlstCmpEemmMidOverFull"] == "--" and word("MlstCmpEemmMidOverFullPr") == undefined
    assert em.m["MlstCmpEemmRandOverFull"] == "--" and word("MlstCmpEemmRandOverFullPr") == undefined
    assert em.m["MlstPairRandMergedOverApartEemm"] == "--" and word("MlstPairRandMergedOverApartEemmPr") == undefined
    assert em.m["MlstPairRandMergedEemmN"] == "1" and em.m["MlstPairRandApartEemmN"] == "1"
    em = _PEm()
    M.eemm_macros(em, fits, {**runs, **{a: [] for a in ("L162", "R16_Q1") + M.FLAV}, M.INIT: ["init-s1"],
                             M.INIT_BN: ["initbn-s1"]}, None, MT.fmt_pm, MT.fmt_int)
    assert em.m["MlstLpEemmAucMid"] == "0.960" and em.m["MlstLpEemmAucInitBn"] == "0.700"
    assert em.m["MlstPairEemmNTrain"] == "9{,}575" and em.m["MlstPairEemmNTestPerSide"] == "5{,}000"
    fits["mtx-r42q1-s1"]["cells"]["best70"]["jc2_188_pairs"]["1000000"]["ee_vs_mm"]["n_train"] = 9580
    em = _PEm()
    M.eemm_macros(em, fits, {**runs, **{a: [] for a in ("L162", "R16_Q1") + M.FLAV}, M.INIT: [], M.INIT_BN: []},
                  None, MT.fmt_pm, MT.fmt_int)
    assert em.m["MlstPairEemmNTrain"] == "9{,}575--9{,}580" and "MlstLpEemmAucInit" not in em.m
    # PI 2026-11-18: no 1 - AUC anywhere, so neither pair table has a ratio column or its note
    assert not hasattr(M, "ZERO_NOTE") and all(r"\rho" not in M.printed_caption(k) for k in ("pairs", "pairs_full"))


def test_the_jetclass2_fine_tuning_block_has_the_ratio_column_at_one_million_jets(tmp_path):
    _ft_tree(tmp_path, ["l188-s1", "r16q1-s1", "l188-s2"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"]}, MT.fmt_pm, ("L188", "R16_Q1"))
    # the blocks (round-1 fix 2026-11-13): the JetClass-II accuracy first, as everywhere, then its AUC with the ratio
    jc2, jc1 = t.split(M.BLOCK_SEP)[1], t.split(M.BLOCK_SEP)[3]
    assert r"& $\rho$ \\" in jc2 and r"\rho" not in jc1
    # (1 - 0.91) / (1 - 0.90) run by run; the 188-class row 1, which its runs agree on by definition (the agreement
    # mark, round-1 fix 2026-11-13, as tab:mlp and tab:bn print it); from scratch has none
    assert r"\Kcoarse{} classes & 0.9100 & 0.9100 & 0.900 \\" in jc2
    assert r"\Kfull{} classes & 0.9000 & 0.9000 & 1.00" + M.AGREE + r" \\" in jc2
    assert jc2.splitlines()[-1].startswith(M.SCRATCH_LABEL) and jc2.splitlines()[-1].endswith(r"& -- \\")
    acc = t.split(M.BLOCK_SEP)[0]                                   # the JetClass-II accuracy, no ratio column
    assert "{JetClass-II, accuracy}" in acc and r"\rho" not in acc
    # a 188-class run without its 10^6 cell: the 188-class ratio cell says how many runs it rests on
    del per["jc2"]["l188-s2"][1000000]
    two = M.table_ft(per, {"L188": ["mtx-l188-s1", "mtx-l188-s2"]}, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)[1]
    # (at 10^3 jets both runs: mean +- SD of 0.90 and 0.92)
    assert r"\Kfull{} classes & 0.910\,$\pm$\,0.014 & 0.9000 \pending{1 of 2} & 1.00" + M.AGREE + r" \pending{1 of 2} \\" in two
    none = M.table_ft(per, {"L188": ["mtx-l188-s3"]}, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)[1]
    assert r"\Kfull{} classes & " + " & ".join([r"\pending{no run yet}"] * 3) + r" \\" in none


def test_the_new_paper_macros_wait_for_their_inputs_and_the_captions_use_the_methods_symbols():
    names = {n for n, _, _ in M.REQUIRED}
    assert {"MlstMinBkgPass", "MlstMassMinJets", "MlstRocNoCurveEemm", "MlstRocCutBcFull", "MlstRocCutJcOneTblFull",
            "MlstRocJcTwoStableMaxChange", "MlstPerClassAucDiffMidMinClass", "MlstLpEemmAucCoarse",
            "MlstPairEemmNTestPerSide", "MlstRocRejMaxEemm"} <= names
    assert M.MASS_MIN_JETS == 100 and M.THIN_TEST == 1000 and M.ROC_PAIRS[-1] == "ee_vs_mm"
    assert not any("epsilon" in cap for _, cap in M.CAPTIONS.values())
    assert "eq:resolution" in M.CAPTIONS["mass"][1] and r"m_{\mathrm{pred}}" not in M.CAPTIONS["mass"][1]
    cap = {k: c for k, (_, c) in M.CAPTIONS.items()}
    # the per-class rejections (and the discriminant they cut on) are in the appendix table (PI 2026-11-12)
    assert "eq:discr" in cap["jetclass_rej"] and "p_c" not in cap["jetclass_rej"] and r"1-g_{\mathrm{QCD}}" in cap["sculpt"]
    # the blank-bin sentence with \MlstMassMinJets fills __BLANK__ when the table is written (bc9ad07)
    assert r"\MlstMassMinJets" in (ROOT / "paper/journal/tables/mlst_mass_detail.tex").read_text()
    assert "__BLANK__" in cap["mass_detail"] and all(
        r"$\nmin$" in cap[k] and "__MINBKG__ " not in cap[k] for k in ("jetclass_rej", "rand_prong"))
    # the rand_prong caption names the threshold and points to tab:jetclassrej, which cites the equation (round-1 fix)
    assert "eq:cut" in cap["jetclass_rej"] and r"table~\ref{tab:jetclassrej}" in cap["rand_prong"]
    assert "ee_vs_mm" not in M.MLP_JC2 and [k for k, *_ in M.PAIRS[:4]][-1] == "ee_vs_mm"


def test_the_groomed_mass_agrees_across_files_up_to_rounding_only():
    # each file recomputes the groomed mass on the same jets; another node rounds the last digits differently
    f = lambda r, v: {"estimators": {"groomed": {"all": {"inclusive": {"resolution": r}, "vs_pt": {"resolution": v}}}}}
    atlas = {"a": f(0.21275067523827068, [0.2287, None]), "b": f(0.21275067523827057, [0.2287 * (1 + 1e-13), None])}
    assert M._groomed(atlas, "all", "resolution") == 0.21275067523827068
    assert M._groomed(atlas, "all", "resolution", "vs_pt") == [0.2287, None]
    atlas["c"] = f(0.2128, [0.2287, None])
    with pytest.raises(SystemExit, match="differs between atlas_res files"):
        M._groomed(atlas, "all", "resolution")


def test_the_long_budget_jetclass_curves_replace_the_old_ones_only_when_they_hold_every_model(tmp_path):
    bfb = _load("build_ft_budget_jobs_t", "scripts/build_ft_budget_jobs.py")
    assert M.V4.JC1_LONG_SCRATCH == tuple(f"s{k}" for k in bfb.SCRATCH_SEEDS)
    runs = M._pretrained(M._paper(ROOT))
    assert sorted(M.V4.paper_models()) == sorted(runs) and len(runs) == 27
    old = {m: {"label_Tbqq": _curve(0.5, sha="o")} for m in runs + [f"scratch-v2-s{k}" for k in (1, 2, 3)]}
    long = {m: {"label_Tbqq": _curve(0.2, sha="n")} for m in runs + [f"scratch-s{k}" for k in (1, 2, 3)]}
    for fam in M.V4.JC1_LONG:
        _roc_tree(tmp_path, old, family=fam)
        # one model short (a from-scratch seed): the old curves, every one of them
        _roc_tree(tmp_path, {m: v for m, v in long.items() if m != "scratch-s3"}, family=f"{fam}_long")
        docs = M.V4.load_family(tmp_path, fam)
        assert set(docs) == set(old) and all(d["curves"]["label_Tbqq"]["label_sha256"] == "o" for d in docs.values())
    assert M.V4.load_roc(tmp_path)["jc1_ft"]["mtx-l188-s1"]["curves"]["label_Tbqq"]["cut"]["eps_s"] == 0.5
    # complete: the long curves replace the old ones whole, the from-scratch seeds under the old reference's names
    for fam in M.V4.JC1_LONG:
        _roc_tree(tmp_path, {"scratch-s3": long["scratch-s3"]}, family=f"{fam}_long")
        docs = M.V4.load_family(tmp_path, fam)
        assert set(docs) == set(old) and all(d["curves"]["label_Tbqq"]["label_sha256"] == "n" for d in docs.values())
    roc = M.V4.load_roc(tmp_path)
    assert "jc1_ft_long" not in roc and roc["jc1_ft"]["scratch-v2-s3"]["curves"]["label_Tbqq"]["cut"]["eps_s"] == 0.2
    # the full-schedule from-scratch curves (<family>_fulldecay, 2026-11-07) replace the long docs' seeds only
    for fam in M.V4.JC1_LONG:
        _roc_tree(tmp_path, {f"scratch-s{k}": {"label_Tbqq": _curve(0.3, sha="n")} for k in (1, 2)},
                  family=f"{fam}{M.V4.SCRATCH_FULL}")
        with pytest.raises(SystemExit, match="not the from-scratch seeds"):
            M.V4.load_family(tmp_path, fam)
        _roc_tree(tmp_path, {"scratch-s3": {"label_Tbqq": _curve(0.3, sha="n")}}, family=f"{fam}{M.V4.SCRATCH_FULL}")
        docs = M.V4.load_family(tmp_path, fam)
        assert set(docs) == set(old)
        assert {m for m, d in docs.items() if d["curves"]["label_Tbqq"]["cut"]["eps_s"] == 0.3} == \
            {f"scratch-v2-s{k}" for k in (1, 2, 3)}
    assert M.V4.load_roc(tmp_path)["jc1_ft"]["mtx-l188-s1"]["curves"]["label_Tbqq"]["cut"]["eps_s"] == 0.2
    # a family without a long counterpart is read as it is
    _roc_tree(tmp_path, {"mtx-l188-s1": {"top": _curve(0.1)}}, family="top_ft")
    _roc_tree(tmp_path, {"mtx-l188-s2": {"top": _curve(0.1)}}, family="top_ft_long")
    assert set(M.V4.load_family(tmp_path, "top_ft")) == {"mtx-l188-s1"}


def test_the_long_budget_jetclass_table_replaces_both_old_trees_only_when_it_holds_every_model(tmp_path):
    runs = M._paper(ROOT)
    blk = lambda acc: {"jc1": {"accuracy": acc}}
    (tmp_path / "curves").mkdir()
    assert M.jc1_full(tmp_path, runs) is None
    pre = {m[len("mtx-"):]: {"N1000000": {"s1": blk(0.1)}} for m in M._pretrained(runs)}
    ref = {M.SCRATCH: {"N1000000": {f"s{k}": blk(0.2) for k in (1, 2, 3)}}}
    for name, cells in (("best70_leg2", pre), ("ref_v100_jc1_leg2", ref)):
        (tmp_path / "curves" / f"full_{name}.json").write_text(json.dumps({"cells": cells}))
    # one from-scratch seed short: the old trees, together
    long = {**{i: {"N1000000": {"s1": blk(0.3)}} for i in pre},
            "scratch": {"N1000000": {"s1": blk(0.4), "s2": blk(0.4)}}}
    f = tmp_path / "curves" / "full_budget_long_leg2.json"
    f.write_text(json.dumps({"cells": long}))
    p, r, src = M.jc1_full(tmp_path, runs)
    assert (p, r, src.name) == (pre, ref, "full_best70_leg2.json")
    # one paper run short: the same
    long["scratch"]["N1000000"]["s3"] = blk(0.4)
    f.write_text(json.dumps({"cells": {k: v for k, v in long.items() if k != "l188-s1"}}))
    assert M.jc1_full(tmp_path, runs)[2].name == "full_best70_leg2.json"
    # complete: the long tree alone, its from-scratch cells under SCRATCH, the old reference not read
    f.write_text(json.dumps({"cells": long}))
    p, r, src = M.jc1_full(tmp_path, runs)
    assert src == f and set(p) == set(pre) and p["l188-s1"]["N1000000"]["s1"]["jc1"]["accuracy"] == 0.3
    assert set(r) == {M.SCRATCH} and set(r[M.SCRATCH]["N1000000"]) == {"s1", "s2", "s3"}
    assert r[M.SCRATCH]["N1000000"]["s1"]["jc1"]["accuracy"] == 0.4
    # the full-schedule from-scratch cells (2026-11-07) replace the long tree's, its pretrained cells kept
    sf = tmp_path / "curves" / "full_budget_long_fulldecay_leg2.json"
    sf.write_text(json.dumps({"cells": {"scratch": {"N1000000": {f"s{k}": blk(0.6) for k in (1, 2, 3)}}}}))
    p, r, src = M.jc1_full(tmp_path, runs)
    assert src == f and p["l188-s1"]["N1000000"]["s1"]["jc1"]["accuracy"] == 0.3
    assert set(r) == {M.SCRATCH} and [r[M.SCRATCH]["N1000000"][f"s{k}"]["jc1"]["accuracy"] for k in (1, 2, 3)] == [0.6] * 3
    sf.write_text(json.dumps({"cells": {"scratch": {"N1000000": {f"s{k}": blk(0.6) for k in (1, 2)}}}}))
    with pytest.raises(SystemExit, match="lacks a from-scratch seed"):
        M.jc1_full(tmp_path, runs)


def test_the_probes_against_size_pool_the_random_partitions_and_add_the_controls_and_the_kinematic_model():
    """tab:probesnfull (2026-11-07; the supplement's copy since 2026-11-13): the flavour-controlled partitions (two
    runs, mean +- SD), the five random partitions pooled into one row (mean +- SD over their runs) and the
    kinematic-only model on top only, in one longtable with a block per task and metric of the training-size figures
    and their AUC versions (PROBES_N_FULL)."""
    rec = lambda a, r: {"macro_auc_ovr": a, "accuracy": a, "auc": a, "r30": r, "r30_is_bound": False, "converged": True}
    runs = {"L188": [f"mtx-l188-s{k}" for k in (1, 2, 3)], "FLAV_F0": ["mtx-flavf0-s1", "mtx-flavf0-s2"],
            **{a: [f"mtx-{a.lower().replace('_', '')}-s1"] for a in M.RAND}}
    v = {m: 0.90 + 0.01 * i for i, m in enumerate(m for ms in runs.values() for m in ms)}
    fits = {m: {"cells": {"best70": {"jc2_188": {"1000": rec(x, 0)}, "top": {"1000": rec(x, 100 * x)}}}}
            for m, x in v.items()}
    fits[M.KIN_MODEL] = {"cells": {"best70": {"top": {"1000": rec(0.9, 133.2)}}}}
    t = M.table_probes_n(fits, M._pooled(runs), MT.fmt_pm, ("L188", "FLAV_F0", M.RAND_POOLED), kin=True)
    assert r"\begin{longtable}" in t and r"\label{tab:probesnfull}" in t and "%SPLIT%" not in t
    # the blocks with fits: JetClass-II accuracy and AUC, top R30 and AUC (the other tasks have none here)
    assert [ds for ds, *_ in M.PROBES_N_FULL] == ["jc2_188"] * 2 + ["jc1"] * 2 + ["top"] * 2 + ["qg"] * 2 + ["qg_herwig"] * 2
    rows = [ln for ln in t.splitlines() if ln.startswith(M.LABEL["FLAV_F0"])]
    assert [r.split(" & ")[1].removesuffix("*").removesuffix(r" \\") for r in rows] == [
        M._cell([f(v["mtx-flavf0-s1"]), f(v["mtx-flavf0-s2"])], 2, MT.fmt_pm) for f in (lambda x: x, lambda x: x, lambda x: 100 * x,
                                                                                      lambda x: x)]
    pooled = [v[runs[a][0]] for a in M.RAND]
    jc2 = next(ln for ln in t.splitlines() if ln.startswith(M.LABEL[M.RAND_POOLED]))
    assert jc2.split(" & ")[1].removesuffix("*").removesuffix(r" \\") == M._cell(pooled, 5, MT.fmt_pm)
    assert not any(ln.startswith(M.LABEL["RAND2_p1"]) for ln in t.splitlines())     # pooled, not one by one
    kin = [ln for ln in t.splitlines() if ln.startswith(M.LABEL["KIN"])]
    assert kin == [M.LABEL["KIN"] + " & 133 \\\\", M.LABEL["KIN"] + " & 0.9000 \\\\"]     # top only: R30, AUC
    assert M.table_probes_n(fits, M._pooled(runs), MT.fmt_pm, ("L188",)).count(M.LABEL["KIN"]) == 0


def test_the_kinematic_model_joins_the_top_rows_of_the_probe_mlp_and_fine_tuning_tables(tmp_path):
    jc2 = {"converged": True, "accuracy": 0.9, **{k: 0.95 for k in ("macro_auc_ovr", "macro_auc_resonant",
                                                                     "macro_auc_qcd")}}
    top = {"accuracy": 0.9163, "auc": 0.9713, "r50": 94.18, "r30": 231.02, "r50_is_bound": False, "r30_is_bound": False}
    fits = {"mtx-l188-s1": {"cells": {"best70": {"jc2_188": {"1000": jc2}, "top": {"1000": top}}}},
            M.KIN_MODEL: {"cells": {"best70": {"top": {"1200000": top}}}}}
    runs = {"L188": ["mtx-l188-s1"]}
    t = M.table_probes(fits, runs, MT.fmt_pm, ("L188",), kin=True)
    lower = t.split("%SPLIT%")[1]
    assert lower.count(M.LABEL["KIN"]) == 1 and t.split("%SPLIT%")[0].count(M.LABEL["KIN"]) == 0
    assert M.LABEL["KIN"] + " & 0.9163 & 0.9713 & 94.2 & 231 & " + " & ".join(["--"] * 8) + r" \\" in lower
    # the MLP probe: the kinematic-only model is in fits_v4_mlp alone (top); a top cell there never replaces
    # fits_v3_mlp2's for a model that has one
    for d, m, cells in (("fits_v3_mlp2", "mtx-l188-s1", {"top": {"auc": 0.96, "r30": 200.0, "r30_is_bound": False}}),
                        ("fits_v4_mlp", "mtx-l188-s1", {"jc2bal": jc2, "top": {"auc": 0.5, "r30": 1.0}}),
                        ("fits_v4_mlp", M.KIN_MODEL, {"top": {"auc": 0.9811, "r30": 716.0, "r30_is_bound": False}})):
        (tmp_path / "linprobe" / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / "linprobe" / d / f"{m}.json").write_text(json.dumps({"cells": {"best70": cells}}))
    mlp = M.load_mlp(tmp_path)
    assert M._mlp_val(mlp, "mtx-l188-s1", "top") == 0.96 and M._mlp_val(mlp, M.KIN_MODEL, "top", "r30") == 716.0
    t = M.table_mlp(mlp, runs, MT.fmt_pm, ("L188",), kin=True)
    row = next(ln for ln in t.splitlines() if ln.startswith(M.LABEL["KIN"]))
    assert row == M.LABEL["KIN"] + " & " + " & ".join(["--"] * 5 + ["0.9811", "716"] + ["--"] * 4) + r" \\"
    assert t.count(M.LABEL["KIN"]) == 1 and M.LABEL["KIN"] not in M.table_mlp(mlp, runs, MT.fmt_pm, ("L188",))
    # fine-tuning: a row before from scratch (the last row of a fine-tuning table) in the task where it has cells, top
    _ft_tree(tmp_path / "ft", ["l188-s1"])
    _, per, _ = M.load_ft(tmp_path / "ft")
    per["top"][M.KIN_INIT] = {1000: [{"r30": 92.4, "r30_is_bound": False, "auc": 0.95}],
                              1200000: [{"r30": 1802.8, "r30_is_bound": False, "auc": 0.97}]}
    blocks = M.table_ft(per, runs, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)
    topb = next(b for b in blocks if "top, " in b)
    lines = topb.splitlines()
    assert lines.index(next(ln for ln in lines if ln.startswith(M.LABEL["KIN"]))) == \
        lines.index(next(ln for ln in lines if ln.startswith(M.SCRATCH_LABEL))) - 1
    assert lines[-1].startswith(M.SCRATCH_LABEL + " & ")
    assert M.LABEL["KIN"] + " & 92.4 & 1{,}803 \\\\" in topb       # from 100 on four significant figures
    # in the two top blocks (R30, and the AUC of the supplement's copy), nowhere else
    assert [b.count(M.LABEL["KIN"]) for b in blocks] == [int("{top, " in b) for b in blocks] and sum("{top, " in b for b in blocks) == 2


def test_the_task_longtables_keep_each_heading_with_its_first_row(tmp_path):
    """tab:ftfull and tab:probesnfull (2026-11-07; the supplement's copies since 2026-11-13) are longtables: a block
    per task, its title and heading unbreakable from its first row (\\* and a rule with no break around it), the rule
    spanning every column."""
    _ft_tree(tmp_path, ["l188-s1"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))
    lines = t.splitlines()
    ncol = int(next(ln for ln in lines if ln.startswith(r"\multicolumn")).split("}")[0].split("{")[1])
    # across the whole line, the row labels on one line (PI 2026-11-11), set in landscape at the body size (D14,
    # round 2: no \small group any more, a plain one)
    assert r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * (ncol - 1) + "@{}}" in lines
    assert lines[0] == r"\begin{landscape}" and r"\end{landscape}" in lines and lines[1] == "{" and not _sizes_in(t)
    titles = [i for i, ln in enumerate(lines) if ln.startswith(r"\multicolumn") and "continued" not in ln]
    assert len(titles) == len(M.FT_TASKS) + 1 + len(M.FT_FULL_AUC)      # JetClass-II: AUC and accuracy; the AUC blocks
    for i in titles:
        assert lines[i].endswith(r"\\*") and lines[i + 1].startswith("training jets") and lines[i + 1].endswith(r"\\*")
        assert lines[i + 2] == M._head_rule(ncol) and r"\multispan{" + str(ncol) + "}" in lines[i + 2]
    assert r"\noalign{\nobreak\vskip\belowrulesep\nobreak}" in M._head_rule(3) and r"\midrule" not in M._head_rule(3)
    assert M._longtable("ft_full", t).count(r"\caption{") == 1 and "__CAPTION__" not in M._longtable("ft_full", t)


def test_a_table_wider_than_the_text_is_split_into_stacked_blocks_that_keep_the_row_labels():
    cols = 12
    tab = "\n".join([r"\begin{tabular}{l" + "c" * cols + "}", r"\toprule",
                     r" & \multicolumn{" + str(cols) + r"}{c}{rejection of QCD} \\", r"\cmidrule(lr){2-" + str(cols + 1) + "}",
                     "model & " + " & ".join(f"class {i}" for i in range(cols)) + r" \\", r"\midrule",
                     "ParT & " + " & ".join(["--"] * 6 + ["1{,}234"] * 6) + r" \\",
                     r"\multicolumn{" + str(cols + 1) + r"}{l}{\textit{this work}} \\",
                     "188 classes & " + " & ".join([r"6{,}600\,$\pm$\,330"] * cols) + r" \\", r"\bottomrule", r"\end{tabular}"])
    blocks = M.split_wide(tab)
    assert len(blocks) > 1
    seen = []
    for b in blocks:
        assert "188 classes & " in b and r"\textit{this work}" in b and "rejection of QCD" in b
        n = len(re.search(r"\\begin\{tabular\}\{l(c+)\}", b).group(1))
        assert rf"\multicolumn{{{n + 1}}}{{l}}{{\textit{{this work}}}}" in b and rf"\cmidrule(lr){{2-{n + 1}}}" in b
        seen += re.findall(r"class (\d+)", b)
    assert seen == [str(i) for i in range(cols)]                  # every column once, in order
    assert "ParT" not in blocks[0] and "ParT" in blocks[-1]       # only dashes there: the row is left out
    small = r"\begin{tabular}{lc}" + "\n" + r"a & b \\" + "\n" + r"\end{tabular}"
    assert M.split_wide(small) == [small]


# ------------------------------------------------------------------ results-book audit (2026-11-07)
def _column(*cells) -> list:
    """The cells of one table column as uniform_columns prints them."""
    t = M.uniform_columns("\n".join(f"r{i} & {c} \\\\" for i, c in enumerate(cells)))
    assert M.V4.TOK_OPEN not in t
    return [ln.split(" & ")[1].removesuffix(" \\\\") for ln in t.split("\n")]


def test_a_standard_deviation_is_rounded_to_one_significant_figure_or_two_if_the_first_is_1():
    """PI 2026-11-12 (the brief's precision rule): _sd_place is the decimal place an SD ends at; negative left of
    the point."""
    for sd, place in ((0.0024, 3), (0.0049, 3), (0.0006, 4), (6.3, 0), (37.0, -1), (0.00004, 5),      # one figure
                      (0.0149, 3), (0.012, 3), (1.2, 1), (12.0, 0), (151.0, -1), (1.95, 1)):         # first digit 1: two
        assert M._sd_place(sd) == place, sd
    # the carry: 0.95 to 0.99 round up to 1.0, which is printed with its two figures at the same place
    assert M._sd_place(0.97) == 1 and M._sd_place(0.0096) == 3 and M._sd_place(96.0) == -1
    assert _column(M.TAB(10.42, 0.97)) == [r"10.4\,$\pm$\,1.0"]
    assert _column(M.TAB(0.7153, 0.0096)) == [r"0.715\,$\pm$\,0.010"]
    # PI 2026-11-14: per cell as the PDG rounds, two figures for an SD whose leading digits are at most 354
    assert _column(M.TAB(0.7153, 0.0024)) == [r"0.7153\,$\pm$\,0.0024"]          # the mean at the SD's place
    assert _column(M.TAB(0.7153, 0.0149)) == [r"0.715\,$\pm$\,0.015"]
    assert _column(M.TAB(1602.0, 81.0)) == [r"1{,}600\,$\pm$\,80"] and _column(M.TAB(1602.0, 151.0)) == [r"1{,}600\,$\pm$\,150"]


def test_a_table_column_prints_at_one_precision_its_standard_deviations_included():
    """uniform_columns (PI 2026-11-12): every cell of a column at one decimal place, the place of the column's
    median SD by the rule above; the whole column one digit finer when an SD would print as zero there; an SD that
    is zero, or still prints as zero, gives the value with the agreement mark; a single run its mark; never "<" and
    never "+- 0"."""
    T, tok = M.TAB, M.V4.tok
    rows = [("x", T(312.4, 6.3), T(0.7153, 0.0024)), ("y", T(307.7, 1.2), T(0.712, 0.0049)),
            ("z", tok("p", 301.0, 0.0), T.two(0.7101, 0.7202)), ("w", T(295.0, 7.0), tok("s", 0.7))]
    t = M.uniform_columns("\n".join([r"\begin{tabular}{lcc}", r"\toprule", r" & a & b \\", r"\midrule",
                                     *(f"{n} & {a} & {b} \\\\" for n, a, b in rows), r"\bottomrule", r"\end{tabular}"]))
    assert M.V4.TOK_OPEN not in t
    # column a: the SDs 6.3, 1.2, 7.0 (the zero left out), median 6.3: units. Column b: 0.0024, 0.0049, 0.0071,
    # median 0.0049: three decimals, also for the single run
    # PI 2026-11-14: a mean +- SD is rounded per cell as the PDG rounds; a single run keeps the column's place
    assert r"x & 312\,$\pm$\,6 & 0.7153\,$\pm$\,0.0024 \\" in t and r"y & 307.7\,$\pm$\,1.2 & 0.712\,$\pm$\,0.005 \\" in t
    assert rf"z & 301{M.AGREE} & 0.715\,$\pm$\,0.007 \\" in t and rf"w & 295\,$\pm$\,7 & 0.700{M.SINGLE} \\" in t
    assert M.AGREE == r"$^{\P}$" and M.SINGLE == r"$^{\S}$"
    # one digit finer for the whole column where an SD would print as zero (the brief's example, 0.98560 +- 0.00004):
    # median 0.0002, four decimals, where 0.00004 prints as 0.0000
    # PI 2026-11-14: per cell as the PDG rounds, so each SD at its own one or two figures
    assert _column(T(0.98560, 0.00004), T(0.9841, 0.0003), T(0.9838, 0.0002)) == [
        r"0.98560\,$\pm$\,0.00004", r"0.98410\,$\pm$\,0.00030", r"0.98380\,$\pm$\,0.00020"]
    assert _column(T(0.9841, 0.0003), T(0.9838, 0.0002)) == [r"0.98410\,$\pm$\,0.00030", r"0.98380\,$\pm$\,0.00020"]
    # one digit, not more: an SD still zero one digit finer prints the value with the agreement mark
    # PI 2026-11-14: per cell as the PDG rounds, so no SD prints as zero and no agreement mark is needed here
    assert _column(T(0.7153, 0.0024), T(0.7120, 0.0049), T(0.9760, 0.00003)) == [
        r"0.7153\,$\pm$\,0.0024", r"0.712\,$\pm$\,0.005", r"0.97600\,$\pm$\,0.00003"]
    assert _column(T(0.9760, 0.00003)) == [r"0.97600\,$\pm$\,0.00003"]           # alone: its own place
    # never past MAX_DECIMALS: an AUC that is 1 to seven digits
    assert M.MAX_DECIMALS == 5 and _column(T(0.99999946, 0.0000005)) == ["1.00000" + M.AGREE]
    # a value below 100 keeps its units beside large ones; without it the column is rounded to hundreds
    assert _column(T(1500.0, 200.0), T(1700.0, 300.0)) == [r"1{,}500\,$\pm$\,200", r"1{,}700\,$\pm$\,300"]
    assert M._column_place([("p", [1500.0, 200.0]), ("p", [1700.0, 300.0])]) == -2
    assert M._column_place([("p", [1500.0, 200.0]), ("p", [1700.0, 300.0]), ("p", [40.0, 150.0])]) == 0
    # no SD in the column (single runs, numbers that are no runs): the place of the values' size, four decimals
    # below 1, two below 10, one below 100, integers from 100 with thousands separators
    assert _column(tok("s", 0.7), tok("o", 0.2448)) == ["0.7000" + M.SINGLE, "0.2448"]
    assert _column(tok("s", 2.346)) == ["2.35" + M.SINGLE] and _column(tok("s", 47.72)) == ["47.7" + M.SINGLE]
    assert _column(tok("s", 1802.83)) == ["1{,}803" + M.SINGLE] and _column(tok("o", 42589.4)) == ["42{,}590"]
    assert not any(x in "".join(_column(T(0.9760, 0.00003), T(301.0, 0.0), tok("s", 0.5))) for x in ("$<$", r"\pm$\,0 "))


def test_identical_runs_print_the_value_once_with_the_agreement_mark_through_float_noise():
    """Three identical runs whose mean is not exact give an SD of ~1e-17, not 0: within 1e-12 of the mean's size
    that is zero, so the cell prints the value with the agreement mark (never "+- 0") and the column's place is not
    set by the noise."""
    for mean, noise in ((2.3, 3e-16), (1234.5, 2e-13), (0.0, 5e-13), (0.7, 0.0)):
        assert M._zero_sd(mean, noise)
    assert not M._zero_sd(0.5, 1e-9) and not M._zero_sd(1234.5, 1e-8)
    tok = M.V4.tok
    assert _column(tok("p", 312.4, 6.3), tok("p", 301.0, 2e-14)) == [r"312\,$\pm$\,6", "301" + M.AGREE]
    assert _column(tok("p", 0.7153, 1e-17)) == ["0.7153" + M.AGREE]
    assert _column(tok("p", 6100.0, 0.0), tok("p", 4800.0, 100.0)) == ["6{,}100" + M.AGREE, r"4{,}800\,$\pm$\,100"]


def test_the_mark_sentences_follow_the_marks_a_printed_table_carries():
    """A caption gets the single-run sentence and the agreement sentence exactly when a printed cell carries the
    mark (PI 2026-11-12: every mark explained, and no other)."""
    single, agree, lower = (note for _, note in M.MARK_NOTES)
    # PI 2026-11-14: the one rejection convention, a lower limit where no background jet passes, explained once
    assert [mark for mark, _ in M.MARK_NOTES] == [M.SINGLE, M.AGREE, r"$\geq$"] and "no background jet passes" in lower
    # the mark tied to "marks" by a non-breaking space, so that it never ends a line alone (round 2, package 3)
    assert single == r" $^{\S}$~marks a value from a single run."
    assert agree == r" $^{\P}$~marks a run spread below the last digit shown."
    assert M._mark_notes("Cap.", "a & 1") == "Cap."
    assert M._mark_notes("Cap.", "a & 1" + M.SINGLE) == "Cap." + single
    assert M._mark_notes("Cap.", "a & 1" + M.AGREE + " & 2" + M.SINGLE) == "Cap." + single + agree
    body = lambda *cells: "\n".join([r"\begin{tabular}{lc}", r"\toprule", r" & a \\", r"\midrule",       # noqa: E731
                                     *(f"r{i} & {c} \\\\" for i, c in enumerate(cells)), r"\bottomrule", r"\end{tabular}"])
    T, tok = M.TAB, M.V4.tok
    for cells, has_single, has_agree in (((T(0.5, 0.01), T(0.6, 0.02)), False, False), ((T(0.5, 0.01), tok("s", 0.6)), True, False),
                                         ((T(0.5, 0.01), T(0.6, 0.0)), False, True), ((tok("s", 0.5), T(0.6, 0.0)), True, True),
                                         ((tok("o", 0.5),), False, False)):
        f = M._float("mass", body(*cells))
        assert (single in f) == has_single and (agree in f) == has_agree, cells
        assert (M.SINGLE in f.replace(single, "")) == has_single and (M.AGREE in f.replace(agree, "")) == has_agree


def test_the_jetclass_table_marks_a_dagger_and_identical_runs_and_bolds_nothing():
    V4 = M.V4
    runs = M._paper(ROOT)

    def block(k, low):
        return {"accuracy": 0.80, "macro_auc_ovo": 0.98, "macro_auc_ovr": 0.98,
                "per_class": {c: {"rejection": {"rejection": (9000.0 if low else 5000.0) + k,
                                                "n_bkg_pass": 5 if low and c == "label_Tbqq" else 200}}
                              for c, _ in V4.JC1_COLS}}
    pre = {m[len("mtx-"):]: {"N1000000": {"s1": {"jc1": block(i, True)}}} for i, m in enumerate(runs["L188"])}
    ref = {M.SCRATCH: {"N1000000": {f"s{k}": {"jc1": block(k, False)} for k in (1, 2, 3)}}}
    tex = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two)
    col = 3 + [c for c, _ in V4.JC1_COLS].index("label_Tbqq")       # name, accuracy, AUC, then the rejections
    cell = lambda name: next(ln for ln in tex.splitlines() if ln.startswith(name)).split(" & ")[col]   # noqa: E731
    assert r"$^\dagger$" in cell(M.LABEL["L188"]) and r"\textbf" not in tex and M.BOLD_ON not in tex
    assert "6{,}061" in tex                                   # ParT's 2 M row, plain
    # three runs that pass the same QCD jets: the value once with the agreement mark of every table, never "+- 0"
    # (PI 2026-11-12); the dagger follows it
    for i, m in enumerate(runs["L188"]):
        pre[m[len("mtx-"):]]["N1000000"]["s1"]["jc1"] = block(0, True)
    tok = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, M.TAB, M.TAB.two, part="rej")
    row = M.uniform_columns(next(ln for ln in tok.splitlines() if ln.startswith(M.LABEL["L188"])))
    assert row.count("9{,}000" + M.AGREE) == len(V4.JC1_COLS) and V4.SAME_PASS not in row and r"\pm" not in row
    assert "9{,}000" + M.AGREE + r"$^\dagger$" in row
    # from scratch last, below the 188-class row
    assert tok.splitlines()[-3].startswith(M.SCRATCH_LABEL + " & 1 M & ")


def test_our_parameter_count_is_the_instantiated_networks():
    torch = pytest.importorskip("torch")
    pytest.importorskip("weaver")
    import types
    import yaml
    spec = importlib.util.spec_from_file_location("arch_t", ROOT / "experiments/E1/ParT_sophon_arch_10c.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    for cfg, n_in in (("TopReference.yaml", 17), ("TopReference_kin.yaml", 7)):
        pf = yaml.safe_load((ROOT / "configs/finetune" / cfg).read_text())["inputs"]["pf_features"]["vars"]
        assert len(pf) == n_in
        for k in (2, 10, 188):
            model, _ = arch.get_model(types.SimpleNamespace(input_dicts={"pf_features": pf}, input_names=[], input_shapes={}),
                                      num_classes=k, fc_params=[(512, 0.1)])
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == M.V4.our_params(k, n_in)
    assert M.V4.our_params(188) == 2_304_688                  # docs/GROUND_TRUTH.md


def test_the_published_rows_carry_their_papers_parameters_and_the_physics_baselines():
    lit = json.loads(M.V4.LIT.read_text())
    top = {r["method"]: r for r in lit["top"]}
    qg = {r["method"]: r for r in lit["qg_exp"]}
    assert (top["linear EFP"]["acc"], top["linear EFP"]["auc"], top["linear EFP"]["r50"], top["linear EFP"]["r30"],
            top["linear EFP"]["params"], top["linear EFP"]["cite"]) == ("0.932", "0.980", None, "384", r"1\,k", "toptag")
    assert (qg["linear EFP"]["auc"], qg["linear EFP"]["r50"], qg["linear EFP"]["acc"]) == ("0.8919", "29.7", None)
    assert (qg["constituent multiplicity"]["auc"], qg["constituent multiplicity"]["r50"]) == ("0.8401", "19.0")
    assert {m: top[m]["params"] for m in ("ParticleNet", "ParT", "ParT-f.t.", "MIParT-L-f.t.", "L-GATr-f.t.",
                                          "OmniLearned-l")} == {
        "ParticleNet": r"366\,k", "ParT": r"2.14\,M", "ParT-f.t.": r"2.14\,M", "MIParT-L-f.t.": r"2.38\,M",
        "L-GATr-f.t.": r"1.1\,M", "OmniLearned-l": r"423\,M"}
    assert qg["PFN-Ex"]["params"] is None and qg["ParticleNet-f.t."]["params"] == r"370\,k"
    assert all(r.get("params_quote") for s in ("top", "qg_exp") for r in lit[s] if r.get("params"))


def test_a_row_label_names_its_network_only_and_wraps_only_at_its_commas():
    """PI 2026-11-12: a row names the network by its classes (or the control's shortest name), then a seed number,
    and never the downstream mode (no "(LP)", "(f.t.)"); published rows keep their names."""
    t = "\n".join([M.LABEL["RAND2_p1"] + r", seed 1 & 1 \\", M.SCRATCH_LABEL + r" & 2 \\", M.LABEL["L188"] + r" & 3 \\",
                   M.LABEL[M.INIT_BN] + r" & 4 \\", r"ParT~\cite{part} & 5 \\"])
    out = M._labels(t).splitlines()
    assert out[0] == r"\mbox{random partition 1,} \mbox{seed 1} & 1 \\"
    assert out[1] == r"\mbox{from scratch} & 2 \\" and out[2] == r"\mbox{\Kfull{} classes} & 3 \\"
    assert out[3] == r"\mbox{random network,} \mbox{BatchNorm recomputed} & 4 \\"
    assert out[4] == r"ParT~\cite{part} & 5 \\"
    assert not any(tag in v for v in list(M.LABEL.values()) + [M.SCRATCH_LABEL] for tag in ("(LP)", "(MLP)", "(f.t.)"))
    assert not any(hasattr(M, name) for name in ("MODE", "MODE_DEF", "_define_tags", "FT_HEADED", "_ft_heading"))


def _legend() -> dict:
    """The figure legends (figstyle/style_v5.LEGEND), read from the source: the module imports matplotlib."""
    import ast
    tree = ast.parse((ROOT / "experiments/FIGS/figstyle/style_v5.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "LEGEND")


def test_the_rows_are_named_as_the_figure_legends_in_one_order():
    """PI 2026-11-12: one row order in every table that lists networks, the names those of the figure legends."""
    plain = lambda t: t.replace(r"\Kfull{}", "188").replace(r"\Kres{}", "162").replace(r"\Kmid{}", "43").replace(r"\Kcoarse{}", "17")   # noqa: E731
    legend = _legend()
    for arm in M.MAIN + M.FLAV + M.MASS:
        assert plain(M.LABEL[arm]) == legend[arm], arm
    assert [plain(M.LABEL[a]) for a in M.MAIN] == ["188 classes", "162 classes", "43 classes", "17 classes"]
    assert [M.LABEL[a] for a in M.FLAV] == ["no flavor boundary", r"$b$-quark boundary", "random flavor boundary"]
    assert M.LABEL[M.RAND_POOLED] == legend["RAND"] == "random partitions"
    assert [plain(M.LABEL[a]) for a in M.MASS] == ["162 classes + mass", "17 classes + mass"]
    assert M.LABEL[M.INIT] == legend["INIT_BN"] == "random network" and M.SCRATCH_LABEL == legend["SCRATCH"] == "from scratch"
    assert M.LABEL[M.SOPHON] == "released Sophon checkpoint"
    # the kinematic-only run: the table's label is the legend's (round-1 fix 2026-11-13: "only" back in the tables)
    assert plain(M.LABEL["KIN"]) == "188 classes, kinematic inputs only" == legend["KIN"]
    # the order (PI 2026-11-13, L5): the label sets, each network with a mass output directly under its own, then the
    # 17-class partitions (flavor boundaries, then the random ones); a table adds the random network (probes) or from
    # scratch (fine-tuning) after them
    assert M.SETS == ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
    assert M.ALL_ROWS == M.SETS + M.FLAV + M.RAND
    assert M._paper_rows(M.ALL_ROWS + (M.INIT,)) == M.SETS + M.FLAV + (M.RAND_POOLED,) + (M.INIT,) == M.XBX_ROWS
    assert M.MASS_ROWS == M.SETS + (M.INIT,) and M.SUMMARY_CONTROLS == M.FLAV + (M.RAND_POOLED,)
    assert M.SIZE_ROWS == M.SETS + M.FLAV + (M.RAND_POOLED,) and M.MASS_EFFECT == ("L162", "L162_MASS", "R16_Q1", "R16_Q1_MASS")


def test_the_mode_is_an_italic_heading_over_the_rows_and_the_partitions_sit_under_a_subheading():
    """PI 2026-11-12: _headings writes the downstream mode once, in italics, before each run of rows that name a
    network, in the brief's words; the partition rows are indented under "17-class partitions"; the released
    checkpoint below its rule belongs to the run above it; a published model's row or a row "(this work)" does not."""
    assert M.HEADING == {"lp": "linear probe on the frozen features", "ft": "fine-tuned",
                         "direct": "pretraining head used directly",
                         "mlp": "multilayer-perceptron probe on the frozen features",
                         "head": "mass output of the pretraining head"}
    assert M.PARTITIONS == "17-class partitions"
    assert M._full_row(3, "x") == r"\multicolumn{3}{@{}l}{x} \\" and M._full_row(3, "x", italic=True) == r"\multicolumn{3}{@{}l}{\textit{x}} \\"
    rows = [r"ParT~\cite{part} & 0 \\", r"\midrule", M.LABEL["L188"] + r" & 1 \\", M.LABEL["R16_Q1"] + r" & 2 \\",
            M.LABEL["FLAV_F0"] + r" & 3 \\", M.LABEL["RAND2_p1"] + r", seed 1 & 4 \\", M.LABEL[M.RAND_POOLED] + r" & 5 \\",
            M.LABEL["L162_MASS"] + r" & 6 \\", M.LABEL[M.INIT] + r" & 7 \\", r"\midrule", M.LABEL[M.SOPHON] + r" & 8 \\"]
    frame = lambda rows: "\n".join([r"\begin{tabular}{lc}", r"\toprule", r" & a \\", r"\midrule", *rows,      # noqa: E731
                                    r"\bottomrule", r"\end{tabular}"])
    lp, part = M._full_row(2, M.HEADING["lp"], italic=True), M._full_row(2, M.PARTITIONS)
    want = rows[:2] + [lp] + rows[2:4] + [part] + [r"\quad " + r for r in rows[4:7]] + rows[7:]
    assert M._headings(frame(rows), "lp") == frame(want)
    # no mode (a table of several modes, or one whose builder wrote the headings): the subheading and indent only
    assert M._headings(frame(rows)) == frame([r for r in want if r != lp])
    assert M._headings(frame(want), "lp") == frame(want)                       # written once, never twice
    # a longtable: the headings stay on the page of the row below them
    star = M._headings(frame(rows), "ft", star=True).split("\n")
    assert M._full_row(2, M.HEADING["ft"], italic=True) + "*" in star and part + "*" in star
    # two runs of rows, a row of anything else between them (the soft-drop mass, a published model): a heading each
    two = [M.LABEL["L188"] + r" & 1 \\", r"soft-drop mass & 2 \\", M.LABEL["L162"] + r" & 3 \\"]
    assert M._headings(frame(two), "mlp").split("\n").count(M._full_row(2, M.HEADING["mlp"], italic=True)) == 2
    ours = [r"from scratch (this work) & 1 \\", rf"fine-tuned from {M.LABEL['L188']} (this work) & 2 \\"]
    assert M._headings(frame(ours), "ft") == frame(ours)
    # which tables get a heading from _float: one mode a table; the builders of the others write their own
    assert M.MODES["coarse"] == M.MODES["qcdhf"] == M.MODES["fill"] == "lp" and M.MODES["lrsweep"] == M.MODES["jetclass_rej"] == "ft"
    # tab:jc1_class_auc: the linear probes first (L1, round-1 fix 2026-11-13); tab:sculpt: the head used directly, then
    # the fine-tuned networks
    # PI 2026-11-23, LEDGER SP-9: the paper's tab:jc1_class_auc is the linear-probe half; its copy keeps both modes
    assert M.MODES["mlp"] == "mlp" and M.MODES["jc1_class_auc"] == "lp" and M._mode("jc1_class_auc_full", 1) == "ft"
    assert M.MODES["sculpt"] == ["direct", "ft"]
    assert not {"probes", "mass", "pairs", "summary", "summary_auc", "published", "labelsets", "xbx", "xbx_rej", "sculpt_boot",
                "probes_n", "ft", "masseffect"} & set(M.MODES)
    assert M.MODES["probes_n_full"] == "lp" and M.MODES["ft_full"] == "ft"


def test_a_table_too_wide_and_too_tall_is_a_landscape_longtable_with_one_caption():
    """PI 2026-11-11: no narrow blocks, nothing shrunk, no wrapped label. A table wider than the upright page and
    taller than the turned one is cut into full-width blocks in the orientation with fewer pages (landscape
    longtables or upright whole blocks), the caption once and "continued" above every later page and block, one
    table number."""
    rows = [M.LABEL[a] + " & " + " & ".join([M.TAB(0.9, 0.001)] * 14) + r" \\" for a in M.ALL_ROWS] * 3
    body = "\n".join([r"\begin{tabular}{l" + "c" * 14 + "}", r"\toprule", " & " + " & ".join(f"c{i}" for i in range(14))
                      + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    t = M._float("probes", body, True)
    assert M.V4.TOK_OPEN not in t and "pretraining &" not in t
    # the orientation with fewer pages (round 6): upright whole blocks, or landscape longtables
    assert t.count(r"\caption") == 1 and t.count(r"\label{tab:probes}") == 1
    assert "(continued)" in t and "adjustbox" not in t and r"}p{" not in t and r"\mbox" not in t
    if r"\begin{longtable}" in t:
        assert t.startswith(r"\begin{landscape}") and t.rstrip().endswith(r"\end{landscape}")
        n = t.count(r"\begin{longtable}{@{\extracolsep{\fill}}l")
        assert t.count(r"\addtocounter{table}{-1}") == n - 1 and t.count(r"\endhead") == n
    else:
        assert t.count(r"\begin{tabular*}{\linewidth}") > 1 and r"\TableContinued" in t


def test_a_turned_longtable_closes_on_its_last_row_and_a_short_second_mode_shares_the_page():
    """Page review round 3 (_turned_long): the closing rule follows the last row, bound to it and with no glue below
    (CLOSE_RULE), and there is no last foot (one taller than the foot lost the repeated "(continued)" heading on a
    page broken in the last chunk); a second mode block of at most SHORT_BLOCK rows does not start a page of its own
    (tab:mlp's linear probes were two rows alone on a page)."""
    assert M.CLOSE_RULE == r"\noalign{\nobreak\vskip\aboverulesep\hrule height\heavyrulewidth}" and M.SHORT_BLOCK == 3
    lp, ft = (M._full_row(15, M.HEADING[k], italic=True) for k in ("lp", "ft"))
    rows = [M.LABEL[a] + " & " + " & ".join([M.TAB(0.9, 0.001)] * 14) + r" \\" for a in M.ALL_ROWS] * 2
    body = "\n".join([r"\begin{tabular}{l" + "c" * 14 + "}", r"\toprule", " & " + " & ".join(f"c{i}" for i in range(14))
                      + r" \\", r"\midrule", lp, *rows, r"\midrule", ft, *rows[:3], r"\bottomrule", r"\end{tabular}"])
    t = M._float("mlp", body, True)
    n = t.count(r"\begin{longtable}")
    assert n >= 1 and t.count(M.CLOSE_RULE) == n and r"\endlastfoot" not in t and r"\bottomrule" not in t
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        if ln == M.CLOSE_RULE:
            assert lines[i - 1].rstrip().endswith(("\\\\", "\\\\*")) and lines[i + 1] == r"\end{longtable}"
    # the three fine-tuned rows stay on the page of the probes: a new page only between the column blocks
    assert t.count(r"\newpage") == n - 1


def test_a_table_that_fits_is_one_tabular_across_the_whole_line():
    body = "\n".join([r"\begin{tabular}{lcc}", r"\toprule", r" & a & b \\", r"\midrule",
                      M.LABEL["L188"] + r" & 1 & 2 \\", r"\bottomrule", r"\end{tabular}"])
    t = M._float("coarse", body)
    assert t.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc@{}}") == 1 and "landscape" not in t
    assert "adjustbox" not in t and r"\begin{table}[t]" in t
    # at the body size (PI 2026-11-12): no size command, no changed column padding; the mode once, in italics
    assert not _sizes_in(t) and "tabcolsep" not in t and r"}p{" not in t
    lines = t.splitlines()
    assert lines[lines.index(M.LABEL["L188"] + r" & 1 & 2 \\") - 1] == M._full_row(3, M.HEADING["lp"], italic=True)


def _sizes_in(tex: str) -> list:
    """The type-size commands of a generated table (not \\smallskip)."""
    return re.findall(r"\\(?:small|footnotesize|scriptsize|tiny)(?![a-zA-Z])", tex)


def test_a_table_is_set_at_the_body_size_unless_its_call_names_a_size_and_whole_fixes_its_orientation():
    """PI 2026-11-12: _float writes no type size of its own; size (tab:probes only) is set after the caption, which
    stays at the body size. whole: that orientation with no cut whatever the width estimate, the
    column padding at colsep_pt. pages: the %SPLIT% parts are pages, each whole under the caption or "(continued)"."""
    wide = "\n".join([r"\begin{tabular}{l" + "c" * 14 + "}", r"\toprule", " & " + " & ".join(f"c{i}" for i in range(14)) + r" \\",
                      r"\midrule", *(M.LABEL[a] + " & " + " & ".join([M.TAB(0.9, 0.001)] * 14) + r" \\" for a in M.MAIN),
                      r"\bottomrule", r"\end{tabular}"])
    spec = r"{@{\extracolsep{\fill}}l" + "c" * 14 + "@{}}"
    auto = M._float("xbx", wide, True)
    # too wide for a turned page: cut into blocks of fewer columns, in whichever orientation takes fewer pages
    assert not _sizes_in(auto) and "tabcolsep" not in auto and spec not in auto
    assert auto.count(r"\begin{tabular*}{\linewidth}") + auto.count(r"\begin{longtable}") > 1
    up = M._float("xbx", wide, True, whole="upright")
    assert "landscape" not in up and up.count(r"\begin{tabular*}{\linewidth}" + spec) == 1 and not _sizes_in(up)
    assert r"\setlength{\tabcolsep}{4pt}" in up                                             # the default colsep_pt
    turned = M._float("xbx", wide, True, whole="turned", colsep_pt=3.5)
    assert turned.startswith(r"\begin{landscape}") and turned.rstrip().endswith(r"\end{landscape}")
    assert turned.count(spec) == 1 and r"\setlength{\tabcolsep}{3.5pt}" in turned and not _sizes_in(turned)
    small = M._float("xbx", wide, True, whole="upright", size=r"\small")
    assert _sizes_in(small) == [r"\small"] and small.index(r"\caption{") < small.index(r"\small") < small.index(r"\begin{tabular*}")
    assert small.replace("\\small\n", "") == up
    # pages: two tabulars joined by %SPLIT% are two whole turned pages, the second under "Table N (continued)"
    tall = wide.replace(r"\bottomrule", "\n".join(M.LABEL[a] + r" & " + " & ".join(["1"] * 14) + r" \\" for a in M.MAIN * 2)
                        + "\n" + r"\bottomrule")           # twelve rows: two of them are more than one turned page
    pages = M._float("xbx", tall + "%SPLIT%" + tall, True, pages=True, whole="turned", colsep_pt=6)
    assert pages.startswith(r"\begin{landscape}") and r"\begin{longtable}" not in pages
    assert pages.count(r"\begin{minipage}{\linewidth}") == 2 and pages.count(spec) == 2
    assert pages.count(r"\captionof{table}{") == 1 and pages.count(r"\label{tab:xbx}") == 1
    assert pages.count(r"\textit{Table~\ref{tab:xbx} (continued)}") == 1 and r"\setlength{\tabcolsep}{6pt}" in pages
    assert pages.index(r"\captionof{table}{") < pages.index(spec) < pages.index("(continued)") < pages.rindex(spec)
    # a page its builder chose shares no page with the one before it: \newpage before the second and before no other;
    # no skip above the caption that heads the first page
    assert pages.count(r"\newpage") == 1 and pages.index(spec) < pages.index(r"\newpage") < pages.index("(continued)")
    assert pages.index(r"\setlength{\abovecaptionskip}{0pt}") < pages.index(r"\captionof{table}{")
    assert r"\newpage" not in turned and r"\newpage" not in up
    # the longtable caption label ends with a full stop, as a float's
    assert "#1{#2. }#3" in M.LT_SETUP and "#2: " not in M.LT_SETUP


def test_the_released_sophon_checkpoint_is_a_reference_row_below_a_rule():
    """PI 2026-11-08: the released checkpoint in the frozen-probe tables, last and below a rule, with no ratio to a
    188-class run; its top and quark/gluon probes keep the missing inputs zero where our runs use the chosen
    substitution."""
    jc2 = lambda a: {"1000000": {"converged": True, "accuracy": 0.4, "macro_auc_ovr": a, "macro_auc_resonant": a,
                                 "macro_auc_qcd": a}}
    bench = lambda r: {"1200000": {"converged": True, "accuracy": 0.9, "auc": 0.98, "r50": r, "r30": 2 * r}}
    jc1 = {"1000000": {"converged": True, "accuracy": 0.7, "macro_auc_ovr": 0.97, "rejection_vs_qcd": {}}}
    doc = lambda ck, a, r: {"cells": {ck: {"jc2_188": jc2(a), "jc1": jc1,
                                           **{ds: bench(r) for ds in ("top", "qg", "qg_herwig")}}}}
    fits = {"mtx-l188-s1": doc("best70", 0.98, 100.0), M.SOPHON_MODEL: doc("released", 0.97, 50.0)}
    runs = {"L188": ["mtx-l188-s1"], M.SOPHON: [M.SOPHON_MODEL]}
    assert M._ck(M.SOPHON_MODEL) == M.SOPHON_CK == "released"
    fill = {"mtx-l188-s1": {"cells": {"best70": {"top.ntop": bench(150.0)}}}}
    chosen = {"top": "mean", "qg": None, "qg_herwig": None}
    t = M.table_probes(fits, runs, MT.fmt_pm, ("L188", M.SOPHON), None, fill, chosen)
    for part in t.split("%SPLIT%"):
        lines = part.strip().splitlines()
        k = next(i for i, ln in enumerate(lines) if ln.startswith(M.LABEL[M.SOPHON] + " & "))
        assert lines[k - 1] == r"\midrule" and lines[k + 1] == r"\bottomrule"
    upper, lower = (p.strip().splitlines() for p in t.split("%SPLIT%"))
    row = lambda lines, lab: next(ln for ln in lines if ln.startswith(lab + " & ")).removesuffix(r" \\").split(" & ")
    assert row(upper, M.LABEL[M.SOPHON])[3] == "--"                                  # no ratio to a 188-class run
    assert row(lower, M.LABEL[M.SOPHON])[4] == M._cell([100.0], 1, MT.fmt_pm)        # top R30, missing inputs zero
    assert row(lower, M.LABEL["L188"])[4] == M._cell([300.0], 1, MT.fmt_pm)          # top R30, the chosen mean
    # this builder is the supplement's copy now (tab:probesfull), set with the tables' formatter: the mode in an italic
    # heading and not on the row, one run a row here and so every value marked
    f = M._float("probes_full", M.table_probes(fits, runs, M.TAB, ("L188", M.SOPHON), None, fill, chosen), True)
    assert "released Sophon checkpoint & " in f and "(LP)" not in f and "outside the controlled comparison" in f
    assert "fetch-by-files" not in f and "kinematics at the head" not in f
    assert rf"\textit{{{M.HEADING['lp']}}}" in f and M.MARK_NOTES[0][1] in f and M.V4.TOK_OPEN not in f
    # the paper's tab:probes says in its caption that the checkpoint's missing inputs are zeros
    assert "and zero for the released Sophon checkpoint" in M.printed_caption("probes")
    co = {M.SOPHON_MODEL: {"cells": {"released": {"jc2bal_coarse": {t: {"1000000": {"accuracy": 0.5, "macro_auc_ovr": 0.9}}
                                                                    for t, _ in M.COARSE_TARGETS}}}}}
    c = M.table_coarse(co, runs, MT.fmt_pm, ("L188", M.SOPHON)).splitlines()
    k = next(i for i, ln in enumerate(c) if ln.startswith(M.LABEL[M.SOPHON] + " & "))
    assert c[k - 1] == r"\midrule" and c[k].split(" & ")[1] == "0.5000"
    # one table of three column groups, accuracy and AUC in each (PI 2026-11-12)
    assert c[2] == (r" & \multicolumn{2}{c}{43 groups} & \multicolumn{2}{c}{17 groups} & "
                    r"\multicolumn{2}{c}{17 groups with no flavor boundary} \\")
    assert c[4] == " & " + " & ".join(["accuracy & AUC"] * 3) + r" \\" and "target" not in "".join(c)


def _r50(r: float, k: int = 400, n: int = 200_000) -> dict:
    """A linear-probe pair record: R50 r with k of n background jets passing at 50% signal efficiency."""
    return {"auc": 0.9, "r50": r, "r50_n_bkg_pass": k, "n_test_background": n}


def test_the_pair_table_groups_the_rows_by_mode_with_the_mode_in_the_group_heading():
    """PI 2026-11-08, 2026-11-12, 2026-11-18, 2026-11-19: the pair table gives R90, the rejection at 90% signal
    efficiency (no AUC, no ratio; the names say R50 for history), one column per pair under one R90 heading in the
    class-pair figures' panel order; the probed and the fine-tuned rows as two row groups under the
    italic headings of every table (HEADING) and no tag on the rows; from scratch has no probe, the random networks no
    fine-tuning; the fine-tuned group is one pending row until the read-out is pulled."""
    # X->ee vs mumu: every network that keeps it apart separates it completely (no background jet passes), as the
    # build's EE_NOTE check requires (PI 2026-11-19, B13, LEDGER TN-7)
    pairs = lambda r: {M.PAIR_N: {k: _r50(r, k=0) if k == "ee_vs_mm" else _r50(r) for k, *_ in M.PAIRS}}      # noqa: E731
    fits = {"mtx-l188-s1": {"cells": {"best70": {M.PAIR_DS: pairs(950.0)}}},
            "mtx-r16q1-s1": {"cells": {"best70": {M.PAIR_DS: pairs(400.0)}}},
            **{m: {"cells": {"init": {M.PAIR_DS: pairs(60.0 + i)}}} for i, m in enumerate(M.INIT_MODELS)}}
    runs = {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"], M.INIT: list(M.INIT_MODELS)}
    ft = {"mtx-l188-s1": {k: (900.0, 0 if k == "ee_vs_mm" else 50, 100_000) for k, *_ in M.PAIRS},
          "mtx-r16q1-s1": {k: (300.0, 80, 100_000) for k, *_ in M.PAIRS},
          **{m: {k: (200.0 + 10 * i, 100, 100_000) for k, *_ in M.PAIRS} for i, m in enumerate(M.FT_SCRATCH_MODELS)}}
    t = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", "R16_Q1", M.INIT), ft)
    # PI 2026-11-19 (list item B13, LEDGER TN-7): the X->ee vs mumu column leaves the table, one sentence in its place
    assert [k for k, *_ in M.PAIRS_TABLE] == [k for k, *_ in M.PAIRS_FIG if k != "ee_vs_mm"]
    n = len(M.PAIRS_TABLE)
    lines = t.splitlines()
    assert lines[2:6] == [rf" & \multicolumn{{{n}}}{{c}}{{$R_{{90}}$}} \\", rf"\cmidrule(lr){{2-{1 + n}}}",
                          " & " + " & ".join(h for _, h, *_ in M.PAIRS_TABLE) + r" \\", r"\midrule"]
    assert [k for k, *_ in M.PAIRS_FIG] == list(M.PAIR_ORDER) and "AUC" not in t and r"\rho" not in t
    lp, f = (lines.index(M._full_row(1 + n, M.HEADING[m], italic=True)) for m in ("lp", "ft"))
    assert lines[lp - 1] == lines[f - 1] == r"\midrule"
    row = lambda group, lab: next(ln for ln in group if ln.startswith(lab + " & ")).removesuffix(r" \\").split(" & ")
    assert [ln.split(" & ")[0] for ln in lines[lp + 1:f - 1]] == [M.LABEL["L188"], M.LABEL["R16_Q1"], M.LABEL[M.INIT]]
    assert [ln.split(" & ")[0] for ln in lines[f + 1:-2]] == [M.LABEL["L188"], M.LABEL["R16_Q1"], M.SCRATCH_LABEL]
    assert all(ln.count(" & ") == n for ln in lines if ln.startswith((M.LABEL["L188"], M.LABEL["R16_Q1"], M.SCRATCH_LABEL)))
    # a merge is the label set's, so the fine-tuned rows carry its mark too (round-1 fix 2026-11-13)
    assert row(lines[f:], M.LABEL["R16_Q1"])[1] == M._cell([300.0], 1, MT.fmt_pm) + r"$^{\dagger}$"
    assert row(lines[f:], M.SCRATCH_LABEL)[1] == M._cell([200.0, 210.0, 220.0], 3, MT.fmt_pm)
    assert row(lines[lp:f], M.LABEL[M.INIT])[1] == M._cell([60.0, 61.0, 62.0], 3, MT.fmt_pm)
    tab = M._float("pairs", t, True, label_cm=M.PAIRS_LABEL_CM)
    assert r"\Kfull{} classes & " in tab and not re.search(r"\((LP|MLP|f\.t\.)\)", tab)
    assert all(rf"\textit{{{M.HEADING[m]}}}" in tab for m in ("lp", "ft")) and "pairs" not in M.MODES
    # the caption: R50 in words, the recomputed random network defined in one sentence (the paper's copy a short one,
    # L16; BN_DEF in the supplement's), and a sentence for each mark a cell carries and for no other (17 classes merge
    # every pair: daggers; no double dagger, no pooled-partition mark, no lower limit here)
    assert r"The rejection at 90\% signal efficiency, $R_{90}$," in tab and "AUC" not in tab
    assert ("BatchNorm recomputed means batch-normalization statistics re-estimated on \\DesignBnJets{} training jets."
            in tab) and M.BN_DEF not in tab and M.BN_DEF in M.printed_caption("pairs_full")
    dagger, ddagger, pooled, limit = (note for _, note in M.KEY_MARKS["pairs"])
    assert dagger in tab and ddagger not in tab and pooled not in tab and limit not in tab and M.POOL_MARK not in tab
    assert "mixed test sample" in M.CAPTIONS["pairs"][1]         # the full account, kept for the text (PI 2026-11-14)
    empty = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188",), {}).splitlines()
    k = empty.index(M._full_row(1 + n, M.HEADING["ft"], italic=True))
    assert empty[k + 1] == M._full_row(1 + n, r"\pending{no run yet}")
    assert M.load_pairs_ft(DATA.parent / "no_such_dir") == {}


def test_the_sculpting_bootstrap_table_reads_the_l3_block_and_refuses_another_sigma():
    """tab:sculptboot under decision L3 (PI 2026-11-13; bootstrap/l3/sculpting_boot_l3.json, the "l3" block of each
    difference): delta = mean over the first network's runs - mean over the second's, the SD over the runs of each
    side, the bootstrap SD over test jets, delta / sigma with sigma^2 the sum of the three squares; six columns, one
    row group per working point, distances x 10^2 at two decimals, no bold (PI 2026-11-12). Every number but the
    bootstrap SD is recomputed from the per-run distances, and a mismatch is fatal. The paired-by-run block of the file,
    the run-SD-only extra row and its dagger are gone (round 2)."""
    assert M.SCULPT_BOOT_L3 == "bootstrap/l3/sculpting_boot_l3.json"
    assert not hasattr(M, "SCULPT_BOOT_EXTRA") and not hasattr(M, "NO_BOOT")

    def x(d, sd_a, sd_b, boot, ra=("mtx-r16q1-s1", "mtx-r16q1-s2"), rb=("mtx-l188-s1", "mtx-l188-s2")):
        sigma = math.sqrt(sd_a ** 2 + sd_b ** 2 + boot ** 2)
        return {"distance": {"runs_config": list(ra), "runs_reference": list(rb), "delta": d, "sd_config": sd_a,
                             "sd_reference": sd_b, "boot_sd": boot, "sigma": sigma, "z": d / sigma,
                             "pass": abs(d) > 3 * sigma}}
    doc = {"replicas": 200, "differences": {"17-188": {"configuration": "r16q1", "reference": "l188", "l3": {
        wp: x(0.0122, 0.002, 0.003, 0.001) if wp == "eps_b=0.01" else x(0.001, 0.002, 0.003, 0.001)
        for wp, _ in M.SCULPT_BOOT_WPS}}}}
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"], "R16_Q1": ["mtx-r16q1-s1", "mtx-r16q1-s2"]}
    t = M.table_sculpt_boot(doc, runs).splitlines()
    assert t[:4] == [r"\begin{tabular}{lccccc}", r"\toprule",
                     r" & $\Delta$ & run SD, first & run SD, second & bootstrap SD & $\Delta/\sigma$ \\", r"\midrule"]
    rows = [ln for ln in t if ln.startswith(M.LABEL["R16_Q1"] + " $-$ " + M.LABEL["L188"] + " & ")]
    assert len(rows) == len(M.SCULPT_BOOT_WPS) and not any(r"\textbf{" in r or M.BOLD_ON in r for r in rows)
    assert "sculpt_boot" not in M.BOLD                 # a table of differences and their sigma: never bold
    # the working points in tab:sculpt's order (round-1 fix 2026-11-13): 10% and 1% QCD efficiency, then 50% signal
    assert [wp for wp, _ in M.SCULPT_BOOT_WPS] == ["eps_b=0.1", "eps_b=0.01", "eps_s=0.5"]
    # PI 2026-11-14: the distances unscaled, as tab:sculpt and the figures print them
    assert rows[1] == M.LABEL["R16_Q1"] + " $-$ " + M.LABEL["L188"] + r" & 0.0122 & 0.0020 & 0.0030 & 0.0010 & 3.26 \\"
    assert r"$^{\dagger}$" not in "\n".join(t)
    # recomputed from the per-run distances: the file's numbers agree, and a changed run is fatal
    dist = {"mtx-l188-s1": 0.040, "mtx-l188-s2": 0.044, "mtx-r16q1-s1": 0.052, "mtx-r16q1-s2": 0.058}
    sculpt = {m: {"working_points": {"0.01": {"jsd": v ** 2}, "0.1": {"jsd": v ** 2}}, "sig_eff_50": {"jsd": v ** 2}}
              for m, v in dist.items()}
    a, b = [dist[m] for m in runs["R16_Q1"]], [dist[m] for m in runs["L188"]]
    ok = {"replicas": 200, "differences": {"17-188": {"configuration": "r16q1", "reference": "l188", "l3": {
        wp: x(statistics.mean(a) - statistics.mean(b), statistics.stdev(a), statistics.stdev(b), 0.001)
        for wp, _ in M.SCULPT_BOOT_WPS}}}}
    assert M.table_sculpt_boot(ok, runs, sculpt) == M.table_sculpt_boot(ok, runs)
    sculpt["mtx-l188-s2"]["working_points"]["0.1"]["jsd"] = 0.045 ** 2
    with pytest.raises(SystemExit, match="differs from the sculpting files"):
        M.table_sculpt_boot(ok, runs, sculpt)
    # the differences in the tables' row order, a partition's indented under the subheading, in every block
    q = {wp: x(0.001, 0.002, 0.003, 0.001) for wp, _ in M.SCULPT_BOOT_WPS}
    more = {"replicas": 200, "differences": {"f0-188": {"configuration": "flavf0", "reference": "l188", "l3": q},
                                             "162-188": {"configuration": "l162", "reference": "l188", "l3": q},
                                             **doc["differences"]}}
    lines = M.table_sculpt_boot(more, {**runs, "L162": ["mtx-l162-s1"], "FLAV_F0": ["mtx-flavf0-s1"]}).splitlines()
    names = [ln.split(" & ")[0] for ln in lines if " $-$ " in ln]
    ref = " $-$ " + M.LABEL["L188"]
    assert names == [M.LABEL["L162"] + ref, M.LABEL["R16_Q1"] + ref, r"\quad " + M.LABEL["FLAV_F0"] + ref] * len(M.SCULPT_BOOT_WPS)
    part = M._full_row(6, M.PARTITIONS) + "*"
    assert lines.count(part) == len(M.SCULPT_BOOT_WPS)
    assert all(lines[i + 1].startswith(r"\quad ") for i, ln in enumerate(lines) if ln == part)
    # a working point's rows never break apart (\\* on every row but its last, page review round 3), and between two
    # working points a space where the page may break, no rule (one there doubled the page's foot rule)
    sep = r"\noalign{\penalty-100}\addlinespace"
    body = lines[lines.index(r"\midrule") + 1:lines.index(r"\bottomrule")]
    blocks = "\n".join(body).split("\n" + sep + "\n")
    assert len(blocks) == len(M.SCULPT_BOOT_WPS) and body.count(sep) == len(M.SCULPT_BOOT_WPS) - 1
    for b in blocks:
        b = b.splitlines()
        assert all(ln.endswith(r"\\*") for ln in b[:-1]) and b[-1].endswith(r" \\") and r"\midrule" not in b
    # sigma that is not the quadrature sum of the three SDs, or a pass that is not |delta| > 3 sigma, is fatal
    doc["differences"]["17-188"]["l3"]["eps_s=0.5"]["distance"]["sigma"] = 0.003
    with pytest.raises(SystemExit, match="quadrature sum"):
        M.table_sculpt_boot(doc, runs)
    doc["differences"]["17-188"]["l3"]["eps_s=0.5"] = {**x(0.001, 0.002, 0.003, 0.001)}
    doc["differences"]["17-188"]["l3"]["eps_s=0.5"]["distance"]["pass"] = True
    with pytest.raises(SystemExit, match="quadrature sum"):
        M.table_sculpt_boot(doc, runs)


@pytest.mark.skipif(not (DATA / "bootstrap/l3/sculpting_boot_l3.json").exists(), reason="no L3 sculpting bootstrap")
def test_the_sculpting_bootstrap_table_is_the_l3_file_recomputed_from_the_per_run_distances():
    """The generated tab:sculptboot: every difference of the L3 file at every working point, Delta and both run SDs
    recomputed from head/sculpting_v3 over every run of each side (2026-11-13: 24 cells, no difference), and none
    beyond three standard deviations (L3; F10b's sentence states none)."""
    runs = M._paper(ROOT)
    # the file the generator reads: the rerun's copy (random partitions pooled) once pulled, else the first
    doc = json.loads((DATA / (M.SCULPT_BOOT_L3_RERUN if (DATA / M.SCULPT_BOOT_L3_RERUN).exists()
                              else M.SCULPT_BOOT_L3)).read_text())
    sculpt = {k.removesuffix("_sculpting"): v for k, v in M._load(DATA / "head/sculpting_v3").items()}
    body = M.table_sculpt_boot(doc, runs, sculpt)       # the recomputation stops the build on a mismatch
    tex = (ROOT / "paper/journal/tables/mlst_sculpt_boot.tex").read_text()
    num = lambda v: f"{v:.4f}".replace("-", "$-$")       # noqa: E731
    for name, d in doc["differences"].items():
        for wp, _ in M.SCULPT_BOOT_WPS:
            q = d["l3"][wp]["distance"]
            assert not q["pass"] and abs(q["z"]) < 3, (name, wp)
            assert " & ".join(num(q[k]) for k in ("delta", "sd_config", "sd_reference", "boot_sd")) in body
    assert len([ln for ln in tex.splitlines() if " $-$ " in ln]) == len(doc["differences"]) * len(M.SCULPT_BOOT_WPS)
    assert "run SD, first & run SD, second & bootstrap SD" in tex and r"$^{\dagger}$" not in tex


def test_the_results_book_is_the_gallery_placement_and_boxes_only_provisional_items(tmp_path, monkeypatch, capsys):
    """PI 2026-11-08: main text, appendix and supplement from the gallery's placement (one source), so each item has
    the map's number; a missing figure or a pending table is fatal unless the placement marks it PROVISIONAL: a box
    keeps such a figure's number, such a table is set with its pending cells; tables joined by " + " share a number; the supplement refers into the paper book
    through xr-hyper, loaded before hyperref, and needs the paper book's .aux."""
    B = _load("make_results_book_t", "experiments/FIGS/make_results_book.py")
    G = B.G
    B.check_spec()
    for tier in G.TIERS:
        figs = [k for k in B.TIERS[tier] if not k.startswith("tab:")]
        assert figs == [k for k, t, _ in G.FIGURES if t == tier]
        assert [k for k in B.TIERS[tier] if k.startswith("tab:")] == [k for k, t, _ in G.TABLES if t == tier]
        assert all(G.number(k) == f"Fig. {B.PREFIX[tier]}{n}" for n, k in enumerate(figs, 1))
    labels = [lab for k, _, _ in G.TABLES for lab in k.split(" + ")]
    assert set(labels) <= set(B.TABLE_FILE)                     # every placed table is one make_tables writes
    (tmp_path / "tables").mkdir()
    (tmp_path / "main.tex").write_text("\\documentclass{article}\n\\usepackage[hidelinks]{hyperref}\n\\title{x}\n")
    for lab in labels:
        (tmp_path / "tables" / f"{B.TABLE_FILE[lab]}.tex").write_text(
            "\\begin{table}[t]\n\\caption{c}\\label{" + lab + "}\n\\begin{tabular}{l}a\\end{tabular}\n\\end{table}\n")
    monkeypatch.setattr(B, "PAPER", tmp_path)
    monkeypatch.setattr(B, "figure", lambda pid, rec, src, first=False: f"% figure {pid}\n")
    monkeypatch.setattr(G, "PROVISIONAL", {"F11": "waits", "tab:mass": "waits"})
    (tmp_path / "tables" / f"{B.TABLE_FILE['tab:mass']}.tex").write_text(
        "\\begin{table}[t]\n\\caption{c}\\label{tab:mass}\n\\begin{tabular}{l}\\pending{x}\\end{tabular}\n\\end{table}\n")
    figs = [k for k, _, _ in G.FIGURES]
    pages = tmp_path / "render"
    pages.mkdir()
    write = lambda ids: (pages / "pages.json").write_text(json.dumps([{"id": i, "drawn": True} for i in ids]))
    write([i for i in figs if i != "F11"])
    for d in (pages, tmp_path / "tables"):        # the figures and the tables stamp the same fine-tuning read-out
        (d / "ft_source.txt").write_text("fixed_metrics.json\n")
    B.build(pages, "paper")
    out = capsys.readouterr().out
    assert "F11 is provisional" in out and "tab:mass is provisional" in out
    book = (tmp_path / "results_book.tex").read_text()
    assert r"\label{fig:F11}" in book and r"\label{tab:mass}" in book and "% figure FJ-ctrl" not in book
    assert r"\pending{x}" in book and book.count("fbox") == 1             # the table as it is, F11 a box
    # the comparison with published models (tab:published) is a main-text table; the task tables it replaced are placed
    # nowhere (PI 2026-11-12)
    assert book.index("% figure F1\n") < book.index("% tables/mlst_published.tex") < book.index(r"{A\arabic{figure}}")
    # PI 2026-11-23, LEDGER SP-9: tab:labelsets in the data release; tab:summaryauc an appendix table
    assert book.index(r"{A\arabic{figure}}") < book.index("% tables/mlst_summary_auc.tex")
    assert not any(f"% tables/mlst_{name}.tex" in book for name in ("jetclass", "bench_top", "bench_qg", "bench_qg_herwig"))
    assert r"\arabic{table}b}" not in book and "externaldocument" not in book     # no 2a/2b (PI 2026-11-12)
    with pytest.raises(SystemExit, match="results_book.aux"):
        B.build(pages, "supplement")
    (tmp_path / "results_book.aux").write_text("")
    B.build(pages, "supplement")
    supp = (tmp_path / "results_supplement.tex").read_text()
    assert supp.index(r"\usepackage{xr-hyper}") < supp.index("{hyperref}") < supp.index(r"\externaldocument{results_book}")
    assert "% figure FJ-ctrl" in supp and "% figure F1\n" not in supp
    write([i for i in figs if i not in ("F11", "F2")])               # F2: a main-text figure, not provisional
    with pytest.raises(SystemExit, match="F2 was not drawn"):
        B.build(pages, "paper")


def test_the_citation_check_finds_a_figure_or_table_that_nothing_else_cites(tmp_path):
    """PI 2026-11-08: anything nothing cites is dropped. A reference inside the label's own float, or in its own
    generated table, or commented out, does not count."""
    C = _load("check_citations_t", "experiments/FIGS/check_citations.py")
    (tmp_path / "sections").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "main.tex").write_text("\\input{sections/a}\n% \\input{sections/old}\n\\input{tables/t1}\n"
                                       "\\input{tables/t2.tex}\n")
    (tmp_path / "sections/a.tex").write_text(
        "See \\cref{fig:x,tab:two}.\n\\begin{figure}\\caption{As \\ref{fig:y}.}\\label{fig:x}\\end{figure}\n"
        "\\begin{figure}\\caption{As figure~\\ref{fig:y}.}\\label{fig:y}\\end{figure}\n% table~\\ref{tab:three}\n")
    (tmp_path / "tables/t1.tex").write_text("\\begin{center}\\captionof{table}{c}\\label{tab:one}\\end{center}\n")
    (tmp_path / "tables/t2.tex").write_text("\\begin{table}\\caption{As table~\\ref{tab:one}}\\label{tab:two}\\end{table}\n"
                                            "\\begin{table}\\caption{d, table~\\ref{tab:three}}\\label{tab:three}\\end{table}\n")
    cited = C.citations(tmp_path / "main.tex")
    assert cited["fig:x"] and cited["fig:y"] and cited["tab:two"] and cited["tab:one"]
    assert cited["tab:three"] == []


def test_the_gallery_citation_check_fails_on_no_sentence_another_tier_or_an_unplaced_item(tmp_path, capsys):
    """check_citations.py --gallery against a fake placement module (BACKS, TIER) and book: clean passes; an item with
    no sentence, one the book places in another tier, one only the book places and one the book does not place fail."""
    import types
    C = _load("check_citations_g", "experiments/FIGS/check_citations.py")
    tiers = {"F1": "main", "tab:a": "main", "F2": "appendix"}
    backs = {k: "A sentence." for k in tiers}
    book = lambda t: types.SimpleNamespace(TIERS=t, PREFIX={"main": "", "appendix": "A"}, check_spec=lambda: None,
                                           M=types.SimpleNamespace(CAPTIONS={"a": ("tab:a", "c")}))
    (tmp_path / "gallery_figures.py").write_text("ORDER = ('F1', 'F2')\n")

    def run(backs, tiers, placed):
        (tmp_path / "fake_placement.py").write_text(f"BACKS = {backs!r}\nTIER = {tiers!r}\n")
        return C.gallery(tmp_path, book(placed))
    ok = {"main": ["F1", "tab:a"], "appendix": ["F2"]}
    assert run(backs, tiers, ok) == 0
    assert run({**backs, "F2": ""}, tiers, ok) == 1 and "no planned citing sentence: F2" in capsys.readouterr().out
    assert run(backs, {**tiers, "F2": "main"}, ok) == 1 and "F2 (appendix here, main in the gallery)" in capsys.readouterr().out
    assert run(backs, tiers, {"main": ["F1", "tab:a"], "appendix": []}) == 1
    assert "placed nowhere in the book: F2" in capsys.readouterr().out
    assert run({**backs, "F3": "S."}, tiers, {**ok, "appendix": ["F2", "F3"]}) == 1
    assert "in no tier of the gallery's map: F3" in capsys.readouterr().out


def test_the_equal_stopping_cells_replace_a_task_and_size_only_complete(tmp_path):
    """PI 2026-11-08/09: per (task, size) the pretrained cells run to their last epoch replace the old ones only when
    every pretrained model is there; the test's six alone leave the old cells; anything else is FATAL."""
    assert M.PRE_FULL_TEST == ("l188-s1", "l188-s2", "l188-s3", "r16q1-s1", "r16q1-s2", "r16q1-s3")
    rec = lambda v: {"accuracy": v, "r30": v}
    models = list(M.PRE_FULL_TEST) + ["l162-s1", "l188kin-s1"]
    base = lambda: {"top": {**{m: {1000: [rec(0.1)], 10000: [rec(0.2)]} for m in models},
                            M.SCRATCH: {1000: [rec(0.0)] * 3}}}
    p = tmp_path / M.PRE_FULL
    w = lambda cells, missing=(): p.write_text(json.dumps({"cells": cells, "missing": list(missing)}))
    assert M.merge_prefull(base(), p) == base() and M.merge_prefull(base(), None) == base()
    # 1e3 moved and was rerun for every model; 1e4 holds only the test's six: 1e3 replaced, 1e4 kept
    w({**{f"leg_top/{m}/N1000/s1": rec(0.5) for m in models}, **{f"leg_top/{m}/N10000/s1": rec(0.6) for m in M.PRE_FULL_TEST}})
    out = M.merge_prefull(base(), p)
    assert all(out["top"][m][1000] == [rec(0.5)] and out["top"][m][10000] == [rec(0.2)] for m in models)
    assert out["top"][M.SCRATCH][1000] == [rec(0.0)] * 3
    for cells, missing in (({f"leg_top/{m}/N1000/s1": rec(0.5) for m in models[:7]}, ()),
                           ({f"leg_top/{m}/N1000/s1": rec(0.5) for m in models}, ["leg_top/l162-s1/N10000/s1"]),
                           ({f"leg_top/scratch/N1000/s1": rec(0.5)}, ())):
        w(cells, missing)
        with pytest.raises(SystemExit, match="FATAL"):
            M.merge_prefull(base(), p)


def _jc1_boot_doc(delta, run_sd, boot_sd):
    s = math.hypot(run_sd, boot_sd)
    x = {"delta": delta, "run_sd": run_sd, "boot_sd": boot_sd, "sigma": s, "z": delta / s, "pass": abs(delta) > 3 * s}
    d = {"accuracy_ref": [0.8475, 0.8463, 0.8467], "accuracy_config": [0.8418, 0.8421, 0.8428],
         "accuracy": dict(x), "error_ratio": dict(x)}
    return d


def test_the_jetclass_accuracy_statement_rests_on_the_3_sigma_bootstrap(tmp_path):
    """PI 2026-11-09: the JetClass statement is the 17-class accuracy at 1e6 with sigma from the run SD and a
    bootstrap of the test jets; a ratio within 3 sigma is 'consistent'; a sigma that is not the hypot is FATAL."""
    (tmp_path / "budget").mkdir()
    f = tmp_path / "budget" / "jc1_acc_boot_N1000000.json"
    doc = {"replicas": 200, "n_test_jets": 2000000,
           "diffs": {"l162-l188": _jc1_boot_doc(-0.0046, 0.0072, 0.0006), "r42q1-l188": _jc1_boot_doc(0.0061, 0.0044, 0.0006),
                     "r16q1-l188": _jc1_boot_doc(0.03015, 0.00627, 0.00075)}}
    f.write_text(json.dumps(doc))
    em = _PEm()
    M.jc1_boot_macros(em, tmp_path, MT.fmt_int, [])
    assert em.m["MlstFtJcOneErrCoarseOverFullBoot"] == r"(3.0\,\ensuremath{\pm}\,0.6)\% more errors"
    assert em.m["MlstFtJcOneErrResOverFullBoot"] == em.m["MlstFtJcOneErrMidOverFullBoot"] == "consistent"
    assert (em.m["MlstFtJcOneAccFullBoot"], em.m["MlstFtJcOneAccCoarseBoot"]) == ("0.847", "0.842")
    doc["diffs"]["r16q1-l188"]["error_ratio"]["sigma"] = 0.001
    f.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="hypot"):
        M.jc1_boot_macros(_PEm(), tmp_path, MT.fmt_int, [])
    missing = []
    M.jc1_boot_macros(_PEm(), tmp_path / "none", MT.fmt_int, missing)
    assert missing and "bootstrap" in missing[0]


def test_the_hidden_layer_rows_sit_at_the_foot_of_the_nonlinear_table_first_block():
    """Two rows alone on a page (PI 2026-11-11): the class-token and hidden-layer linear probes join the first block,
    under a rule, in its JetClass-II AUC and quark/gluon columns."""
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"]}
    rec = {"macro_auc_ovr": 0.98, "auc": 0.88, "r50": 30.0, "r30": 70.0}
    raw = {m: {"cells": {M.PRIMARY: {k: {"1000": rec} for k in ("jc2bal", "jc2bal.hidden", "qg", "qg.hidden")}}}
           for m in runs["L188"]}
    t = M.table_mlp({}, runs, MT.fmt_pm, ("L188",), raw_v4=raw)
    first = t.split("%SPLIT%")[0]
    # under the linear probe's italic heading (D3, page review round 3), named "188 classes, class token" and "188
    # classes, hidden layer of the pretraining head"
    hidden = M.LABEL["L188"] + ", hidden layer of the pretraining head"
    row = next(ln for ln in first.splitlines() if ln.startswith(hidden + " & "))
    cells = row.rstrip(" \\").split(" & ")
    assert len(cells) == 12 and cells[2:8] == ["--"] * 6 and cells[10:] == ["--"] * 2
    lines = first.splitlines()
    k = lines.index(next(ln for ln in lines if ln.startswith(M.LABEL["L188"] + ", class token & ")))
    assert lines[k - 2:k] == [r"\midrule", M._full_row(12, M.HEADING["lp"], italic=True)] and lines[k + 1] == row
    assert "512-unit" not in t and "%SPLIT%" not in t.split(hidden)[0].split("class token")[-1]


# ------------------------------------------------------------------ PI 2026-11-12: the paper's tables
def test_the_paper_tables_pool_the_random_partitions_where_the_first_of_them_stood():
    # in the tables' order (ALL_ROWS, L5) that is after the flavor-boundary partitions, before the references
    assert M._paper_rows(M.ALL_ROWS + (M.INIT,)) == M.SETS + M.FLAV + (M.RAND_POOLED,) + (M.INIT,)
    assert M._paper_rows(M.ALL_ROWS + (M.INIT, M.SOPHON)) == M.SETS + M.FLAV + (M.RAND_POOLED,) + (M.INIT, M.SOPHON)
    assert M._paper_rows(M.MAIN + M.MASS + M.RAND + M.FLAV + (M.INIT,)) == M.MAIN + M.MASS + (M.RAND_POOLED,) + M.FLAV + (M.INIT,)
    assert M._paper_rows(M.MAIN + M.RAND + (M.INIT,)) == M.MAIN + (M.RAND_POOLED, M.INIT)
    assert M._paper_rows(M.MAIN) == M.MAIN
    assert M.LABEL[M.RAND_POOLED] == "random partitions"


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_probe_and_pair_tables_of_the_paper_print_no_ratio_column():
    runs = M._pooled(M._paper(ROOT))
    fits, _ = M.V4.load_fits(DATA)
    t = M.table_probes(fits, runs, MT.fmt_pm, ("L188", M.RAND_POOLED), rho=False)
    assert r"\rho" not in t and r"\multicolumn{5}{c}{JetClass-II}" in t and M.LABEL[M.RAND_POOLED] + " & " in t
    # the paper's tab:probes (2026-11-13): what Table 1 and its AUC table lack, accuracy and R50 on top and
    # quark/gluon (Pythia, Herwig); no ratio, no JetClass-II or JetClass column
    pt = M.table_probes_bench(fits, runs, M.TAB, ("L188", M.RAND_POOLED))
    assert r"\rho" not in pt and "full count" not in pt and "resonant" not in pt and M.V4.TOK_OPEN not in pt
    assert "%SPLIT%" not in pt and "JetClass" not in pt
    assert pt.splitlines()[4] == " & " + " & ".join(["accuracy & $R_{50}$"] * 3) + r" \\"
    p = M.table_pairs({}, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", M.RAND_POOLED))
    assert r"\rho" not in p
    pooled = next(ln for ln in p.splitlines() if ln.startswith(M.LABEL[M.RAND_POOLED] + " & ")).split(" & ")
    # X->ee vs mumu: random partitions 1, 2 and 4 merge it, 3 and 5 keep it apart; the QCD pairs: all five merge them
    # (the columns in the class-pair figures' panel order, PAIRS_FIG)
    # PI 2026-11-19 (B13, LEDGER TN-7): no X->ee vs mumu column, so its pooled-partition mark is gone with it
    keys = [k for k, *_ in M.PAIRS_TABLE]
    assert "ee_vs_mm" not in keys and r"$^{\dagger}$" in pooled[1 + keys.index("qcd_bb_vs_cc")]
    # the pooled-partition mark is the parallel sign: the section sign marks a single run in every table
    assert M.POOL_MARK == r"$^{\parallel}$" and M.POOL_MARK != M.SINGLE and not hasattr(M, "SAME_AUC")


def test_both_resonance_tables_have_the_same_rows_and_the_rejection_table_shows_both_from_scratch_signals():
    """tab:xbx and tab:xbxrej (PI 2026-11-13): one page per signal with the read-outs of tab:xbx (the classifier on the
    frozen features, the linear probe, X->bb's direct discriminant), the rows in the one order (L5), from scratch in
    the classifier's columns; the X->bs closest-class output on the X->bs page (PI 2026-11-21: tab:xbxnode merged in)."""
    runs = M._paper(ROOT)
    end = lambda r30: {"r50": r30 / 3, "r30": r30}      # noqa: E731
    scr = {"xbb": [end(900.0), end(950.0), end(1000.0)], "xbs": [end(600.0), end(610.0), end(620.0)]}
    assert M.XBX_REJ_PAGES == (("xbb", ("head", "linear", "zero_shot_xbb")), ("xbs", ("head", "linear", "zero_shot_xbq")))
    assert not hasattr(M, "XBX_NODE_PAGES")
    t = M.table_xbx_rej({}, scr, runs, M.TAB)
    pages = t.split("%SPLIT%")
    assert len(pages) == 2
    XR = _load("xbx_report_rows_t", "experiments/XBS/xbx_report.py")
    assert tuple(XR.ROWS) == M.XBX_ROWS == M.SETS + M.FLAV + (M.RAND_POOLED,) + (M.INIT,)      # the tables' one order
    for page, (sig, names) in zip(pages, M.XBX_REJ_PAGES):
        lines = page.splitlines()
        assert lines[2] == rf" & \multicolumn{{{2 * len(names)}}}{{c}}{{{M.XBX_SIGNAL[sig]}}} \\"
        assert lines[4] == " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{M.XBX_READOUT[n]}}}" for n in names) + r" \\"
        assert lines[6] == " & " + " & ".join([r"$R_{50}$ & $R_{30}$"] * len(names)) + r" \\"
        rows = [ln.split(" & ")[0] for ln in lines if " & " in ln and not ln.startswith((" ", "&"))]
        assert rows == [M.LABEL[r] for r in XR.ROWS] + [M.SCRATCH_LABEL], sig
        row = lines[-3].rstrip(r" \\").split(" & ")
        assert row[0] == M.SCRATCH_LABEL and "--" not in row[1:3] and row[3:] == ["--"] * (2 * len(names) - 2)
    assert [M.XBX_READOUT[n] for n in ("head", "linear", "zero_shot_xbb")] == ["classifier", "linear probe", "direct"]
    # no X->bb from scratch: no from-scratch row on its page
    only = M.table_xbx_rej({}, {"xbs": scr["xbs"]}, runs, M.TAB).split("%SPLIT%")
    assert M.SCRATCH_LABEL not in only[0] and only[1].splitlines()[-3].startswith(M.SCRATCH_LABEL + " & ")
    # one X->bb seed from scratch beside three X->bs seeds: the single run's values carry the mark
    one = [M.uniform_columns(pg) for pg in M.table_xbx_rej({}, {"xbb": scr["xbb"][:1], "xbs": scr["xbs"]}, runs, M.TAB).split("%SPLIT%")]
    assert one[0].splitlines()[-3].rstrip(r" \\").split(" & ")[1:] == ["300" + M.SINGLE, "900" + M.SINGLE] + ["--"] * 4
    assert one[1].splitlines()[-3].rstrip(r" \\").split(" & ")[1:5] == [r"203.3\,$\pm$\,3.3", r"610\,$\pm$\,10", "--", "--"]
    assert not any(M.SINGLE in M.uniform_columns(pg) for pg in pages)       # three seeds each: no mark
    # the closest-class output on the X->bs page, never from scratch
    assert M.XBX_READOUT["zero_shot_xbq"] in pages[1] and M.XBX_READOUT["zero_shot_xbq"] not in pages[0]


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_summary_table_holds_exactly_the_values_of_figs_2_and_3():
    """tab:summary and tab:summaryauc (PI 2026-11-13, L1, L2) read every run's value as the headline figures do: the
    accessors summary_probe, summary_ft and summary_scratch agree with paper_figures' readers (_f2_probe, _f2_ft,
    scratch_series) on the four tasks those draw, and the tables hold, per mode, the label sets with each mass network
    under its own, the 17-class partitions under their subheading, then the mode's reference. The gallery's own value
    functions and the printed cells, Herwig included: tests/test_gallery_pages.py."""
    P = _load("paper_figures_summary_t", "experiments/FIGS/paper_figures.py")
    I = P.inputs(M)
    runs = M._pooled(M._paper(ROOT))
    chosen, _ = M.chosen_substitutions(I["fill"], runs)
    assert tuple((k, n, ds, met) for k, n, ds, met, *_ in P.F2_PANELS) == M.SUMMARY_PANELS[:4]
    tasks = {k for k, *_ in P.F2_PANELS}
    for key, n, ds, met in M.SUMMARY_GRID + M.SUMMARY_AUC_GRID:
        if key in tasks:
            a, b = P._f2_probe(M, I["fits"], I["fill"], chosen, ds, met), M.summary_probe(I["fits"], I["fill"], chosen, ds, met)
            c, d = P._f2_ft(M, I["per"], key, n, met), M.summary_ft(I["per"], key, n, met)
            for arm in M.SETS + M.FLAV + (M.RAND_POOLED, M.INIT):
                assert [a(m) for m in runs[arm]] == [b(m) for m in runs[arm]]
                if arm != M.INIT:
                    assert [c(m) for m in runs[arm]] == [d(m) for m in runs[arm]]
        assert P.scratch_series(M, I["per"], key, met)[n] == M.summary_scratch(I["per"], key, n, met)
    ctl = [r"\quad " + M.LABEL[a] for a in M.SUMMARY_CONTROLS]
    for grid, key in ((M.SUMMARY_GRID, "summary"), (M.SUMMARY_AUC_GRID, "summary_auc")):
        t = M.table_summary(I["fits"], I["fill"], chosen, I["per"], M._paper(ROOT), M.TAB, grid, key)
        # one part (a turned page) a mode, five columns each
        parts = t.split("%SPLIT%")
        assert len(parts) == 2 and t.count(M.LABEL[M.RAND_POOLED]) == 2
        assert t.count(M.SCRATCH_LABEL + " & ") == t.count(M.LABEL[M.INIT] + " & ") == 1
        assert r"\rho" not in t and r"\textbf" not in t and M.LABEL[M.SOPHON] not in t       # these fits: no Sophon
        for part, mode in zip(parts, ("lp", "ft")):
            lines = part.splitlines()
            assert lines[:6] == [r"\begin{tabular}{lccccc}", r"\toprule",
                                 r" & JetClass-II & JetClass & top tagging & \multicolumn{2}{c}{quark/gluon tagging} \\",
                                 r"\cmidrule(lr){5-6}",
                                 " & " + " & ".join(M.SUMMARY_HEAD[met] + {"qg": ", Pythia", "qg_herwig": ", Herwig"}.get(k, "")
                                                    for k, _, _, met in grid) + r" \\", r"\midrule"]
            # under the italic heading of the mode: the label sets, each mass network under its own, then the
            # partitions indented under their subheading; the random network closes the probes, from scratch the
            # fine-tuned rows
            assert lines[6] == M._full_row(6, M.HEADING[mode], italic=True)
            assert [ln.split(" & ")[0] for ln in lines[7:13]] == [M.LABEL[a] for a in M.SETS]
            assert lines[13] == M._full_row(6, M.PARTITIONS) and part.count(M._full_row(6, M.PARTITIONS)) == 1
            assert [ln.split(" & ")[0] for ln in lines[14:-2]] == ctl + [M.LABEL[M.INIT] if mode == "lp" else M.SCRATCH_LABEL]
            assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]
            rows = [ln.removesuffix(r" \\").split(" & ") for ln in lines[7:13] + lines[14:-2]]
            assert all(len(r) == 6 and "--" not in r for r in rows)
            # bold (PI 2026-11-12): every column of a part has its best value; the random network is a reference row,
            # never bold and never the best; from scratch takes part
            best = rows[:-1] if mode == "lp" else rows
            assert all(any(M.BOLD_ON in r[j] for r in best) for j in range(1, 6)), (key, mode)
            assert mode == "ft" or M.BOLD_ON not in lines[-3]
        # printed: one precision per column and part, no "<", no "+- 0", no bare token; a mark stays outside the bold
        assert M.V4.TOK_OPEN not in t and "$<$" not in t and not re.search(r"\\pm\$\\,0(\.0+)? ", _unbold(t))
        assert all(M.PLAIN.fullmatch(x) for x in _bold_spans(t))


# ------------------------------------------------------------------ PI 2026-11-12: the table fix (published models, label sets, tasks as blocks)
def _jc1_trees(runs: dict, arms) -> tuple:
    """ft_curves cells of the JetClass fine-tuning: the runs of arms (accuracy 0.800, 0.801, ... over a configuration's
    runs) and three from-scratch seeds."""
    def block(k):
        return {"accuracy": 0.80 + k, "macro_auc_ovo": 0.98 + k, "macro_auc_ovr": 0.98 + k,
                "per_class": {c: {"rejection": {"rejection": 1000.0 * (j + 1) + 10 * k, "n_bkg_pass": 200}}
                              for j, (c, _) in enumerate(M.V4.JC1_COLS)}}
    pre = {m[len("mtx-"):]: {"N1000000": {"s1": {"jc1": block(1e-3 * i)}}} for a in arms for i, m in enumerate(runs[a])}
    ref = {M.SCRATCH: {"N1000000": {f"s{k}": {"jc1": block(1e-3 * k)} for k in (1, 2, 3)}}}
    return pre, ref


def _bench_all(runs: dict, arms) -> dict:
    """_bench_per with the runs of every arm of arms: R30 1800, 1810, ... and R50 450, 460, ... over a configuration's
    runs; from scratch 1400 +- 10 and 50 +- 10 over three seeds; the kinematic-only network's one run."""
    by = _bench_per(runs)
    rec = by[M.KIN_INIT][1200000][0]
    for a in arms:
        for i, m in enumerate(runs[a]):
            by[m[len("mtx-"):]] = {1200000: [{**rec, "r50": 450.0 + 10 * i, "r30": 1800.0 + 10 * i}]}
    return by


def _blocks(page: str, ncol: int) -> dict:
    """{heading text: [the rows and rules below it]} of one page of a table of task blocks (_task_pages)."""
    out, name = {}, None
    for ln in page.splitlines():
        m = re.fullmatch(r"\\multicolumn\{%d\}\{@\{\}l\}\{(?:\\textit\{)?(.*?)\}?\} \\\\" % ncol, ln)
        if m:
            name = m.group(1)
            out[name] = []
        elif name is not None and (" & " in ln or ln == r"\midrule"):
            out[name].append(ln)
    return {k: v[:-1] if v and v[-1] == r"\midrule" else v for k, v in out.items()}


def test_the_comparison_with_published_models_has_a_block_a_task_and_our_two_rows_below_the_published_ones():
    """tab:published (PI 2026-11-12, in place of the JetClass, top and quark/gluon tables): blocks JetClass, top
    tagging, quark/gluon tagging; in each the published models, then our network from scratch, then fine-tuned from
    188 classes; on top tagging also the 188-class network with kinematic inputs (one run, marked). Columns:
    parameters, training jets, accuracy, AUC, R50, R30, a dash where a metric does not apply. Two turned pages, cut
    between tasks."""
    runs = M._paper(ROOT)
    pre, ref = _jc1_trees(runs, ("L188",))
    # quark/gluon at its full training set (FULL_N), where tab:labelsets reads the cells this table repeats (L17)
    per = {"top": _bench_per(runs), "qg": {m: {M.FULL_N["qg"]: v[1200000]} for m, v in _bench_per(runs).items()}}
    t = M.table_published(per, runs, pre, ref, M.TAB)
    pages = t.split("%SPLIT%")
    assert M.PUBLISHED_PAGES == (("jc1", "top"), ("qg",)) and len(pages) == 2 and M.V4.TOK_OPEN not in t
    head = rf" & {M.V4.PARAMS_HEAD} & training jets & accuracy & AUC & $R_{{50}}$ & $R_{{30}}$ \\"
    assert all(pg.splitlines()[:4] == [r"\begin{tabular}{lcccccc}", r"\toprule", head, r"\midrule"] for pg in pages)
    assert head == r" & parameters & training jets & accuracy & AUC & $R_{50}$ & $R_{30}$ \\"
    blocks = {**_blocks(pages[0], 7), **_blocks(pages[1], 7)}
    assert list(blocks) == ["JetClass", "top tagging", "quark/gluon tagging"] == [dict(M.TASK_BLOCKS)[k] for k in ("jc1", "top", "qg")]
    assert M._full_row(7, "JetClass") in pages[0] and r"\textit" not in t          # plain block headings, no mode heading
    scratch, ft, kin = M.SCRATCH_LABEL + M.THIS_WORK, f"fine-tuned from {M.LABEL['L188']}{M.THIS_WORK}", M.LABEL["KIN"]
    assert (scratch, ft) == ("from scratch (this work)", r"fine-tuned from \Kfull{} classes (this work)")
    names = {k: [ln.split(" & ")[0] for ln in v] for k, v in blocks.items()}
    # the published rows first (each cited), then ours: from scratch, fine-tuned, and on top the kinematic-only row
    assert names["JetClass"] == [r"ParticleNet~\cite{part}", r"ParT~\cite{part}", scratch, ft]
    assert names["top tagging"][-3:] == [scratch, ft, kin] and names["quark/gluon tagging"][-2:] == [scratch, ft]
    # the published names in words (METHOD_TEX, round-1 fix 2026-11-13: "ParT, fine-tuned", never "ParT-f.t.")
    printed = lambda task: [M.V4.METHOD_TEX.get(k, k) for k in M.V4.LIT_ROWS[task]]      # noqa: E731
    assert [n.split("~")[0] for n in names["top tagging"][:-3]] == printed("top")
    assert [n.split("~")[0] for n in names["quark/gluon tagging"][:-2]] == printed("qg")
    assert not any("f.t." in n for v in names.values() for n in v)
    assert all(r"\cite{" in n for k, v in names.items() for n in v if n not in (scratch, ft, kin))
    assert all(ln.count(" & ") == 6 and r"\midrule" not in ln for v in blocks.values() for ln in v)     # six value columns, no rule
    row = lambda k, name: next(ln for ln in blocks[k] if ln.startswith(name + " & ")).removesuffix(r" \\").split(" & ")   # noqa: E731
    B = M._b
    # JetClass: no R50 and R30 (the per-class rejections are tab:jetclassrej); training jets: the published 2 M, ours 1 M
    assert all(ln.endswith(r" & -- & -- \\") for ln in blocks["JetClass"])
    # bold (PI 2026-11-12): in each block's column the best value and every value within three standard deviations of
    # it, a published value with the SD it prints; A3 (PI 2026-11-14): one printed without an uncertainty is never bold,
    # so ParT's 0.836 and 0.9834 are light, and ours (0.982 and 0.981 +- 0.001) compete with each other
    assert row("JetClass", r"ParT~\cite{part}") == [r"ParT~\cite{part}", r"2.14\,M", r"2\,M", "0.836", "0.9834", "--", "--"]
    assert row("JetClass", scratch)[4] == B(r"0.9820\,$\pm$\,0.0010") and row("JetClass", ft)[4] == B(r"0.9810\,$\pm$\,0.0010")
    assert M.BOLD_ON not in " & ".join(row("JetClass", r"ParticleNet~\cite{part}"))
    # the parameters and the training jets are counts: never bold
    assert not any(M.BOLD_ON in c for v in blocks.values() for ln in v for c in ln.split(" & ")[:3])
    assert row("JetClass", scratch)[1:3] == [r"2.21\,M", r"1\,M"] and row("JetClass", ft)[1:3] == [r"2.21\,M", r"1\,M"]
    # with ParT's 0.836 out of the rule (no uncertainty), our two rows are the best and within 3 SD of each other
    assert row("JetClass", scratch)[3] == B(r"0.8020\,$\pm$\,0.0010") and row("JetClass", ft)[3] == B(r"0.8010\,$\pm$\,0.0010")
    # top and quark/gluon: the task's training set, a dash for an observable that is not trained; the published values
    # as their papers print them, ours at one precision per column of the block
    assert row("top tagging", r"ParT~\cite{part}") == [r"ParT~\cite{part}", r"2.14\,M", r"1.2\,M", "0.940", "0.9858",
                                                      r"413\,$\pm$\,16", r"1{,}602\,$\pm$\,81"]
    assert row("top tagging", r"linear EFP~\cite{toptag}")[1:] == [r"1\,k", r"1.2\,M", "0.932", "0.980", "--", "384"]
    assert row("quark/gluon tagging", r"constituent multiplicity~\cite{efn}")[1:] == ["--", "--", "--", "0.8401", "19.0", "--"]
    assert row("quark/gluon tagging", r"ParT~\cite{part}")[2] == r"1.6\,M" and row("quark/gluon tagging", ft)[2] == r"1.6\,M"
    agree = lambda x: x + M.AGREE      # noqa: E731  three runs with one accuracy and AUC in the fixture
    # A3 (PI 2026-11-14): the published values print no uncertainty here, so the AUC is decided among our two rows
    assert row("top tagging", scratch)[:3] == [scratch, r"2.21\,M", r"1.2\,M"] and row("top tagging", scratch)[3] == agree("0.9400")
    assert row("top tagging", scratch)[4] in (agree("0.9860"), B("0.9860") + M.AGREE)
    assert row("top tagging", scratch)[5:] == [r"50\,$\pm$\,10", r"1{,}400\,$\pm$\,10"]
    assert row("top tagging", ft)[5:] == [r"460\,$\pm$\,10", r"1{,}810\,$\pm$\,10"]
    # the kinematic-only network: one run, every value marked, at the column's place; a one-run diagnostic outside the
    # network list, so never bold (NEVER_BOLD, page review 2026-11-13: its SD of zero made its R50 of 950 bold)
    assert row("top tagging", kin) == [kin, r"2.21\,M", r"1.2\,M", "0.9400" + M.SINGLE, "0.9860" + M.SINGLE,
                                       "950" + M.SINGLE, "2{,}300" + M.SINGLE]
    assert M.LABEL["KIN"] in M.NEVER_BOLD
    # the best values of the block are published ones (named in words, round-1 fix: no "f.t."): L-GATr's accuracy,
    # OmniLearned-l's AUC and R30 (3,486 +- 157; 2,894 +- 84 is 592 away, more than three times 178), and the R50 of
    # ParT fine-tuned (691 +- 15), with MIParT-L's, L-GATr's and OmniLearned-l's within three standard deviations
    # A3 (PI 2026-11-14): a published accuracy or AUC printed without an uncertainty is never bold
    assert row("top tagging", r"L-GATr, fine-tuned~\cite{lgatr}")[3:] == ["0.9446", "0.9879", B(r"651\,$\pm$\,11"),
                                                                          r"2{,}894\,$\pm$\,84"]
    assert row("top tagging", r"OmniLearned-l~\cite{omnilearned}")[3:] == ["0.944", "0.9880", B(r"688\,$\pm$\,9"),
                                                                          B(r"3{,}486\,$\pm$\,157")]
    assert row("top tagging", r"ParT, fine-tuned~\cite{part}")[5] == B(r"691\,$\pm$\,15")
    assert row("top tagging", r"MIParT-L, fine-tuned~\cite{mipart}")[5] == B(r"640\,$\pm$\,10")
    assert sum(M.BOLD_ON in c for ln in blocks["top tagging"] for c in ln.split(" & ")) == 7
    # quark/gluon: the fixture's two rows of ours are above every published value and agree on accuracy and AUC; the
    # accuracy and R50 are tab:labelsets' cells (L17, repeated_cells), the AUC and R30 this table's own
    assert [_unbold(c) for c in row("quark/gluon tagging", scratch)[3:5]] == \
        [_unbold(c) for c in row("quark/gluon tagging", ft)[3:5]] == ["0.9400" + M.AGREE, "0.9860" + M.AGREE]
    assert row("quark/gluon tagging", ft)[4] == B("0.9860") + M.AGREE
    assert row("quark/gluon tagging", ft)[5:] == [B(r"460\,$\pm$\,10"), B(r"1{,}810\,$\pm$\,10")]
    assert row("quark/gluon tagging", scratch)[5:] == [r"50\,$\pm$\,10", r"1{,}400\,$\pm$\,10"]
    same = M.repeated_cells(per, runs)
    assert [same[a, "qg", met] for a in (None, "L188") for met in ("accuracy", "r50")] == [
        row("quark/gluon tagging", scratch)[3], row("quark/gluon tagging", scratch)[5],
        _unbold(row("quark/gluon tagging", ft)[3]), _unbold(row("quark/gluon tagging", ft)[5])]
    assert kin not in pages[1] and M.SINGLE not in pages[1] and M.SINGLE not in "".join(blocks["JetClass"])
    # set: two whole turned pages under one caption and number, the second "(continued)"; the single-run sentence
    f = M._float("published", t, True, pages=True, whole="turned", colsep_pt=6)
    assert f.startswith(r"\begin{landscape}") and r"\begin{longtable}" not in f and f.count(r"\begin{minipage}{\linewidth}") == 2
    assert f.count(r"\captionof{table}{") == 1 and f.count(r"\label{tab:published}") == 1
    assert f.count(r"\textit{Table~\ref{tab:published} (continued)}") == 1 and M.MARK_NOTES[0][1] in f and M.MARK_NOTES[1][1] in f
    assert f.index("top tagging} \\\\") < f.index("(continued)}") < f.index("quark/gluon tagging} \\\\")
    assert rf"\textit{{{M.HEADING['ft']}}}" not in f and M.PARTITIONS not in f and not _sizes_in(f)
    assert M.CAPTIONS["published"][0] == "tab:published" and "published" not in M.MODES
    # the second page starts a page; the caption says what bold marks, once (PI 2026-11-19, restated 2026-11-22:
    # "Bold marks the best value in each column."; published values take part like any other)
    assert f.count(r"\newpage") == 1 and f.index("top tagging} \\\\") < f.index(r"\newpage") < f.index("(continued)}")
    assert M.BOLD["published"]["of"] == "their column and task"
    assert f.count(M.BOLD_NOTE) == 1 and "within three standard deviations" not in f and "never bold." in f
    assert "The kinematic-inputs network is a reference, never bold." in f
    assert r"\textbf" not in f and f.count(M.BOLD_ON) == t.count(M.BOLD_ON) > 0


def _ft_full(runs: dict, arms) -> dict:
    """per of fine-tuning on the full top and quark/gluon training sets (FULL_N): run i of an arm accuracy 0.94 and
    R50 450 + 10 i, R30 1800 + 10 i; from scratch three seeds, R50 50 +- 10 and R30 1400 +- 10."""
    rec = lambda i: {"accuracy": 0.94, "auc": 0.986, "r50": 450.0 + 10 * i, "r30": 1800.0 + 10 * i,    # noqa: E731
                     "r50_is_bound": False, "r30_is_bound": False}
    per = {}
    for task in ("top", "qg", "qg_herwig"):
        n = M.FULL_N[task]
        by = {m[len("mtx-"):]: {n: [rec(i)]} for a in arms for i, m in enumerate(runs[a])}
        by[M.SCRATCH] = {n: [{**rec(0), "r50": 50.0 + d, "r30": 1400.0 + d} for d in (-10, 0, 10)]}
        per[task] = by
    return per


def test_the_label_set_table_holds_the_fine_tuned_rows_the_comparison_table_does_not():
    """tab:labelsets (PI 2026-11-13, plan "tables"): what Table 1 and tab:summaryauc lack of the fine-tuned top and
    quark/gluon networks, the accuracy and R50 (their R30 and AUC are there), read as Table 1 reads them (summary_ft,
    summary_scratch) on the full training sets; one column group a task (top; quark/gluon scored on Pythia and on
    Herwig jets), the family's rows in the one order (L5) under "fine-tuned", the partitions under their subheading,
    from scratch last; one turned page."""
    runs = M._paper(ROOT)
    per = _ft_full(runs, M.SETS + M.FLAV + M.RAND)
    t = M.table_labelsets(per, runs, M.TAB)
    lines = t.splitlines()
    assert "%SPLIT%" not in t and M.V4.TOK_OPEN not in t and "JetClass" not in t
    assert M.BENCH_GROUPS == (("top", "top tagging"), ("qg", "quark/gluon, Pythia jets"), ("qg_herwig", "quark/gluon, Herwig jets"))
    assert lines[:6] == [r"\begin{tabular}{lcccccc}", r"\toprule",
                         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h in M.BENCH_GROUPS) + r" \\",
                         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                         " & " + " & ".join(["accuracy & $R_{50}$"] * 3) + r" \\", r"\midrule"]
    assert lines[6] == M._full_row(7, M.HEADING["ft"], italic=True)
    names = [ln.split(" & ")[0] for ln in lines if " & " in ln and not ln.startswith(" & ")]
    assert names == [M.LABEL[a] for a in M.SETS] + [r"\quad " + M.LABEL[a] for a in M.FLAV + (M.RAND_POOLED,)] + [M.SCRATCH_LABEL]
    assert lines.count(M._full_row(7, M.PARTITIONS)) == 1 and lines[lines.index(M._full_row(7, M.PARTITIONS)) + 1].startswith(r"\quad ")
    row = lambda lab: _unbold(next(ln for ln in lines if ln.startswith(lab + " & "))).removesuffix(r" \\").split(" & ")   # noqa: E731
    # every task's column group has values (quark/gluon at its own training set, 1.6 million)
    assert all(len(row(M.LABEL[a])) == 7 and "--" not in row(M.LABEL[a]) for a in M.SETS)
    # the pooled random partitions: mean and SD over the five, one run each (R50 450 each in the fixture)
    assert row(r"\quad " + M.LABEL[M.RAND_POOLED])[2] == "450" + M.AGREE
    assert row(M.LABEL["L188"])[2:3] == [r"460\,$\pm$\,10"] and row(M.SCRATCH_LABEL)[1:3] == ["0.9400" + M.AGREE, r"50\,$\pm$\,10"]
    # bold (one block, BOLD["labelsets"]): from scratch, far below in R50, is not bold there
    assert M.BOLD["labelsets"] == {} and M.BOLD_ON not in next(ln for ln in lines if ln.startswith(M.SCRATCH_LABEL)).split(" & ")[2]
    assert r"\textbf" not in t
    # a row with no value on any task is left out
    assert M.LABEL["FLAV_F0"] not in M.table_labelsets(_ft_full(runs, M.SETS), runs, M.TAB)
    # set: one turned page at the body size, the bold sentence of a table without blocks
    f = M._float("labelsets", t, True, pages=True, whole="turned", colsep_pt=6)
    assert f.count(r"\label{tab:labelsets}") == 1 and f.count(rf"\textit{{{M.HEADING['ft']}}}") == 1 and not _sizes_in(f)
    assert M.CAPTIONS["labelsets"][0] == "tab:labelsets" and "labelsets" not in M.MODES
    assert f.count(M.BOLD_NOTE) == 1 and "three standard deviations" not in f     # PI 2026-11-19 / 2026-11-22
    assert M.printed_caption("labelsets").startswith("The fine-tuned networks of table~\\ref{tab:summary} and training "
                                                      "from scratch")


def test_the_benchmark_and_jetclass_rows_are_shared_by_the_task_tables_and_the_new_ones():
    """mlst_v4.bench_rows and jetclass_rows: the published rows, and ours with from scratch last (the kinematic-only
    network before it); 'qg_herwig' has no published row; published=True adds each published row's kind."""
    V4, runs = M.V4, M._paper(ROOT)
    per = {"top": _bench_per(runs), "qg": _bench_per(runs), "qg_herwig": _bench_per(runs)}
    lit, ours = V4.bench_rows("top", per, runs, ("L188",), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two,
                              kin=(M.LABEL["KIN"], M.KIN_INIT), published=True)
    assert [name.split("~")[0] for name, *_ in lit] == [V4.METHOD_TEX.get(k, k) for k in V4.LIT_ROWS["top"]]
    assert all(len(r) == 4 and len(r[1]) == 4 for r in lit) and V4.METHOD_TEX["ParT-f.t."] == "ParT, fine-tuned"
    assert {kind for *_, kind in lit} == {"scratch", "pretrained"} and lit[0][2:] == (r"1\,k", "scratch")
    assert [name for name, *_ in ours] == [M.LABEL["L188"], M.LABEL["KIN"], M.SCRATCH_LABEL]
    assert all(len(r) == 3 and len(r[1]) == 4 and r[2] == r"2.21\,M" for r in ours)
    assert ours[1][1] == [V4.tok("s", v) for v in (0.94, 0.986, 950.0, 2300.0)]
    assert ours[2][1][3] == V4.tok("p", 1400.0, 10.0) and ours[0][1][3] == V4.tok("p", 1810.0, 10.0)
    lit, ours = V4.bench_rows("qg", per, runs, M.MAIN, M.LABEL, M.SCRATCH, M.TAB, M.TAB.two, published=True)
    assert [kind for *_, kind in lit][:3] == ["observable", "scratch", "scratch"] and len(lit) == len(V4.LIT_ROWS["qg"])
    assert [name for name, *_ in ours] == [M.LABEL[a] for a in M.MAIN] + [M.SCRATCH_LABEL]
    assert ours[1][1] == ["--"] * 4                                                # no run read out: dashes
    lit, ours = V4.bench_rows("qg_herwig", per, runs, ("L188",), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two)
    assert lit == [] and [name for name, *_ in ours] == [M.LABEL["L188"], M.SCRATCH_LABEL]
    assert all(len(r) == 3 for r in V4.bench_rows("top", per, runs, (), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two)[0])
    # the task table is these rows and nothing else
    t = V4.table_bench("top", per, runs, ("L188",), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two, kin=(M.LABEL["KIN"], M.KIN_INIT))
    lit, ours = V4.bench_rows("top", per, runs, ("L188",), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two, kin=(M.LABEL["KIN"], M.KIN_INIT))
    assert [ln for ln in t.splitlines() if " & " in ln][1:] == [f"{n} & {p} & " + " & ".join(c) + r" \\" for n, c, p in lit + ours]
    pre, ref = _jc1_trees(runs, ("L188", "R16_Q1"))
    lit, ours = V4.jetclass_rows(pre, ref, runs, ("L188", "R16_Q1"), M.LABEL, M.SCRATCH, 1_000_000, M.TAB, M.TAB.two)
    assert [p for p, *_ in lit] == [r"ParticleNet~\cite{part} & 2 M & 366\,k", r"ParT~\cite{part} & 2 M & 2.14\,M"]
    assert [p for p, *_ in ours] == [lab + r" & 1 M & 2.21\,M" for lab in (M.LABEL["L188"], M.LABEL["R16_Q1"], M.SCRATCH_LABEL)]
    assert all(len(cells) == len(marks) == 2 + len(V4.JC1_COLS) for _, cells, marks in lit + ours)
    assert all(c.startswith(V4.TOK_OPEN + "p ") for _, cells, _ in ours for c in cells) and lit[1][1][:2] == ["0.836", "0.9834"]
    assert V4.jetclass_rows(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, M.TAB, M.TAB.two, literature=False)[0] == []


def _probe_fits() -> tuple:
    """Probe fits of one 188-class run, three random networks, the kinematic-only network (top only) and the
    released Sophon checkpoint, with every task of tab:probes."""
    jc2 = lambda a: {"1000000": {"converged": True, "accuracy": 0.4, "macro_auc_ovr": a, "macro_auc_resonant": 0.11,     # noqa: E731
                                 "macro_auc_qcd": 0.22}}
    bench = lambda r: {"1200000": {"converged": True, "accuracy": 0.9, "auc": 0.98, "r50": r, "r30": 2 * r}}   # noqa: E731
    jc1 = {"1000000": {"converged": True, "accuracy": 0.7, "macro_auc_ovr": 0.97}}
    doc = lambda ck, a, r: {"cells": {ck: {"jc2_188": jc2(a), "jc1": jc1,                                      # noqa: E731
                                           **{ds: bench(r) for ds in ("top", "qg", "qg_herwig")}}}}
    fits = {"mtx-l188-s1": doc("best70", 0.98, 100.0), M.SOPHON_MODEL: doc("released", 0.97, 50.0),
            M.KIN_MODEL: {"cells": {"best70": {"top": bench(40.0)}}},
            **{m: doc("init", 0.6, 10.0 + k) for k, m in enumerate(M.INIT_MODELS)}}
    runs = {"L188": ["mtx-l188-s1"], M.INIT: list(M.INIT_MODELS), M.SOPHON: [M.SOPHON_MODEL]}
    return fits, runs


def test_the_probe_table_of_the_paper_holds_what_table_one_lacks_of_top_and_quark_gluon():
    """tab:probes (PI 2026-11-13, plan "tables"): what Table 1 and tab:summaryauc lack of the linear probes, the
    accuracy and R50 on top and quark/gluon (Pythia, Herwig) at each task's largest training set, one column group a
    task, the networks under the italic mode heading, then the random network (the references of L5), the
    kinematic-only network (top only, outside the network list, round 2) and, below a rule, the released checkpoint;
    one turned page at the body size."""
    fits, runs = _probe_fits()
    t = M.table_probes_bench(fits, runs, M.TAB, ("L188", M.INIT, M.SOPHON), kin=True)
    lines = t.splitlines()
    assert "%SPLIT%" not in t and M.V4.TOK_OPEN not in t and r"\rho" not in t and "JetClass" not in t and "probes" not in M.MODES
    assert "0.11" not in t and "0.22" not in t and "0.98" not in t.replace("0.9800", "")      # no JetClass-II or AUC column
    assert lines[:6] == [r"\begin{tabular}{lcccccc}", r"\toprule",
                         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h in M.BENCH_GROUPS) + r" \\",
                         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                         " & " + " & ".join(["accuracy & $R_{50}$"] * 3) + r" \\", r"\midrule"]
    assert lines[6] == M._full_row(7, M.HEADING["lp"], italic=True)
    full, init, sophon, kin = M.LABEL["L188"], M.LABEL[M.INIT], M.LABEL[M.SOPHON], M.LABEL["KIN"]
    S, A, B = M.SINGLE, M.AGREE, M._b
    # each column at its own precision: the one SD of the R50 columns is 1.0 (two figures). Bold (one block,
    # BOLD["probes"]) only among the rows that are no reference: the random network, the kinematic-only network (a
    # one-run diagnostic, NEVER_BOLD since the page review of 2026-11-13) and the released checkpoint are references,
    # so every column holds one value that can be bold, the 188-class run's, and none is
    assert lines[7:-2] == [rf"{full} & 0.9000{S} & 100.0{S} & 0.9000{S} & 100.0{S} & 0.9000{S} & 100.0{S} \\",
                           rf"{init} & 0.9000{A} & 11.0\,$\pm$\,1.0 & 0.9000{A} & 11.0\,$\pm$\,1.0 & 0.9000{A} & 11.0\,$\pm$\,1.0 \\",
                           rf"{kin} & 0.9000{S} & 40.0{S} & -- & -- & -- & -- \\",
                           r"\midrule", rf"{sophon} & 0.9000{S} & 50.0{S} & 0.9000{S} & 50.0{S} & 0.9000{S} & 50.0{S} \\"]
    assert M.BOLD["probes"] == {} and t.count(M.BOLD_ON) == 0 and kin in M.NEVER_BOLD
    # the substitution chosen on validation for our rows; the released checkpoint keeps its missing inputs zero
    fill = {"mtx-l188-s1": {"cells": {"best70": {"top.ntop": {"1200000": {"accuracy": 0.9, "auc": 0.98, "r50": 150.0, "r30": 300.0}}}}}}
    c = M.table_probes_bench(fits, {k: v for k, v in runs.items() if k != M.INIT}, M.TAB, ("L188", M.SOPHON), fill,
                             {"top": "mean"}).splitlines()
    assert c[7:-2] == [rf"{full} & 0.9000{S} & 150{S} & 0.9000{S} & 100{S} & 0.9000{S} & 100{S} \\", r"\midrule",
                       rf"{sophon} & 0.9000{S} & 50{S} & 0.9000{S} & 50{S} & 0.9000{S} & 50{S} \\"]
    # set as the build sets it: one turned page at the body size, the mark sentences and the bold sentence of a table
    # without blocks in the caption
    f = M._float("probes", t, True, pages=True, whole="turned", colsep_pt=6)
    assert "landscape" in f and f.count(r"\label{tab:probes}") == 1 and not _sizes_in(f)
    assert all(note in f for _, note in M.MARK_NOTES[:2]) and f.count(sophon + " & ") == 1 and "(LP)" not in f
    assert M.KEY_MARKS["probes"][0][1] not in f
    # no bold cell: no bold sentence (it is said only where a cell is bold)
    assert "Bold marks" not in f and f.count(M.BOLD_ON) == 0
    # the caption names the networks by Table 1 (round 2, package 3: no pointer to the AUC table)
    assert M.printed_caption("probes").startswith("Linear probes of table~\\ref{tab:summary} on top and quark/gluon")


def test_the_missing_input_table_of_the_paper_leaves_out_what_the_probe_table_prints():
    """tab:fill (PI 2026-11-12: a number has one table): with chosen, the column of the substitution chosen on
    validation and the kinematic-only row are left out (tab:probes prints both); without it (the supplement's
    copies) every substitution and the kinematic-only row, its one value marked."""
    cell = lambda r: {"1200000": {"r30": r, "r30_is_bound": False}}      # noqa: E731
    keys = {t: M.SUBST_KEY[t] for t in ("top", "qg")}
    fill = {m: {"cells": {"best70": {keys[t][s]: cell(100.0 * (i + 1) + 10 * j + k) for i, t in enumerate(keys)
                                     for j, s in enumerate(M.SUBST)}}}
            for k, m in enumerate(("mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"))}
    fits = {M.KIN_MODEL: {"cells": {"best70": {"top": {"1200000": {"r30": 231.02, "r30_is_bound": False}}}}}}
    runs = {"L188": list(fill), M.INIT: []}
    paper = M.table_fill(None, fill, fits, runs, M.TAB, ("L188", M.INIT), drop=False, chosen={"top": "draws", "qg": "mean"}).splitlines()
    assert paper[0] == r"\begin{tabular}{lcccc}" and paper[4] == " & zero & mean & zero & draws \\\\"
    assert paper[2] == r" & \multicolumn{2}{c}{top, $R_{30}$} & \multicolumn{2}{c}{quark/gluon (Pythia), $R_{30}$} \\"
    # PI 2026-11-23, LEDGER SP-9: tab:fill takes in tab:fill_main's rows, the kinematic-only one among them (tab:probes
    # is in the data release); the chosen substitution's column stays Table 1's
    assert paper[8] == M.LABEL["KIN"] + r" & \multicolumn{2}{c}{231" + M.SINGLE + r" (nothing missing)} & -- & -- \\"
    assert paper[7].startswith(M.LABEL[M.INIT] + " & ") and paper[9] == r"\bottomrule"
    tok = M.V4.tok
    assert paper[6] == M.LABEL["L188"] + " & " + " & ".join(tok("p", x, 1.0) for x in (101.0, 111.0, 201.0, 221.0)) + r" \\"
    whole = M.table_fill(None, fill, fits, runs, M.TAB, ("L188", M.INIT), drop=False).splitlines()
    assert whole[0] == r"\begin{tabular}{lcccccc}" and whole[4] == " & " + " & ".join(M.SUBST * 2) + r" \\"
    # the kinematic-only row last, after the references of L5 (the random network), outside the network list (plan
    # "tables", round 2), its one value marked
    assert whole[6].startswith(M.LABEL["L188"] + " & ") and whole[7].startswith(M.LABEL[M.INIT] + " & ")
    assert whole[8] == M.LABEL["KIN"] + r" & \multicolumn{3}{c}{231" + M.SINGLE + r" (nothing missing)} & -- & -- & -- \\"
    assert whole[9] == r"\bottomrule"


def test_the_mass_table_head_block_reads_the_regression_output_over_all_jets_inclusive():
    """tab:mass (PI 2026-11-12: the quantity of the binned mass figure in a table): the block "mass output of the
    pretraining head" is estimators.regressed.all.inclusive of head/atlas_res, for the two networks pretrained with
    a mass output; not the resonant or the QCD jets, not a bin, not the perceptron."""
    runs = M._paper(ROOT)

    def doc(res, med):
        g = lambda v, w: {"inclusive": {"resolution": v, "median": w},     # noqa: E731
                          "vs_pt": {"edges": [200, 400, 2500], "resolution": [0.71, 0.72], "median": [1.71, 1.72], "n": [500, 500]}}
        return {"estimators": {"groomed": {grp: g(0.2448, 0.9294) for grp in ("all", "resonant", "qcd")},
                               "regressed": {"all": g(res, med), "resonant": g(0.51, 1.51), "qcd": g(0.61, 1.61)},
                               "mlp": {grp: g(0.31, 1.31) for grp in ("all", "resonant", "qcd")}}}
    atlas = {m: doc(res, med) for m, res, med in zip(runs["L162_MASS"], (0.1461, 0.1473), (0.99, 1.01))}
    assert len(atlas) == 2
    lines = M.table_mass(atlas, runs, M.TAB).splitlines()
    head, mlp = (lines.index(M._full_row(3, M.HEADING[k], italic=True)) for k in ("head", "mlp"))
    tok = M.V4.tok
    # the soft-drop mass once, below both blocks (a reference, last; L5): a property of the jets, no run and no mark
    assert lines[-3] == "soft-drop mass & " + tok("o", 0.2448) + " & " + tok("o", 0.9294) + r" \\" and lines[-4] == r"\midrule" and head == 4
    assert sum(ln.startswith("soft-drop mass") for ln in lines) == 1
    assert lines[head + 1] == M.LABEL["L162_MASS"] + " & " + M._cell([0.1461, 0.1473], 2, M.TAB) + " & " + M._cell([0.99, 1.01], 2, M.TAB) + r" \\"
    assert lines[head + 2] == M.LABEL["R16_Q1_MASS"] + r" & \pending{no run yet} & \pending{no run yet} \\" and mlp == head + 4
    assert not any(x in " ".join(lines[head:mlp]) for x in ("0.51", "0.61", "0.71", "0.31", "1.71"))
    assert M.LABEL["L162_MASS"] + " & " + tok("p", 0.31, 0.0) + " & " + tok("p", 1.31, 0.0) + r" \\" in lines[mlp:]
    # printed: the head block's two runs as mean +- SD at the column's place, the soft-drop mass unmarked
    f = M._float("mass", "\n".join(lines))
    assert M.LABEL["L162_MASS"] + r" & 0.1467\,$\pm$\,0.0008 & 1.000\,$\pm$\,0.014 \\" in f
    assert r"soft-drop mass & 0.2448 & 0.929 \\" in f and M.SINGLE not in f
    # one network with a value in each block: nothing to compare, no bold and no bold sentence
    assert M.BOLD_ON not in f and "Bold marks" not in f
    assert f.count(rf"\textit{{{M.HEADING['head']}}}") == f.count(rf"\textit{{{M.HEADING['mlp']}}}") == 1 and "landscape" not in f
    # the supplement's detail table reads the same estimator in its first section, over the resonant and the QCD jets,
    # printed there at one precision per column (uniform_columns); its column over all jets is empty for this
    # estimator: the values over all jets are this table's alone (one home per number, 2026-11-13)
    d = M.table_mass_detail(atlas, runs, M.TAB).splitlines()
    k = next(i for i, ln in enumerate(d) if M.HEADING["head"] in ln)
    A = M.AGREE
    assert d[k + 1] == M.LABEL["L162_MASS"] + f" &  & 0.5100{A} & 0.6100{A} &  & 1.51{A} & 1.61{A} \\\\"
    upper = "\n".join(d).split("%SPLIT%")[0]
    assert "0.1467" not in "\n".join(d) and M.V4.TOK_OPEN not in upper


def _paper_tables() -> dict:
    """{label: (caption key, generated file)} of the tables placed in the main text and the appendix."""
    G = _load("gallery_placement_tables_t", "experiments/FIGS/figstyle/gallery_placement.py")
    key = {label: k for k, (label, _) in M.CAPTIONS.items()}
    name = {k: f for k, f, *_ in M.PLACEHOLDERS} | dict(M.LONGTABLES)      # tab:perclass is a longtable (appendix)
    # PI 2026-11-23, LEDGER SP-9: the class map (tab:app-vocabulary, written by appendix_tables.py) joins the appendix
    return {lab: (key[lab], ROOT / "paper/journal/tables" / f"{name[key[lab]]}.tex")
            for item, tier, _ in G.TABLES if tier in (G.MAIN, G.APPENDIX) for lab in item.split(" + ") if lab in key}


def _caption_and_body(tex: str) -> tuple:
    m = re.search(r"\\caption(?:of\{table\})?\{", tex)
    i, depth = m.end(), 1
    while depth:
        depth += (tex[i] == "{") - (tex[i] == "}")
        i += 1
    return tex[m.end():i - 1], tex[:m.start()] + tex[i:]


def test_the_paper_has_twenty_tables_and_the_task_tables_are_placed_nowhere():
    tables = _paper_tables()
    # PI 2026-11-23, LEDGER SP-9: the approved structure's main-text and appendix tables (the class map apart, above)
    assert list(tables) == ["tab:sets", "tab:summary", "tab:published", "tab:mass", "tab:inputs", "tab:lrsweep",
                            "tab:compute", "tab:fill", "tab:mlp", "tab:partrepro", "tab:summaryauc", "tab:tests",
                            "tab:coarse", "tab:jc1_class_auc"]
    assert tables["tab:summaryauc"][1].name == "mlst_summary_auc.tex" and tables["tab:tests"][1].name == "mlst_tests.tex"
    assert tables["tab:published"][1].name == "mlst_published.tex" and tables["tab:compute"][1].name == "compute.tex"
    assert all(k in M.CAPTION_SHORT for k, _ in tables.values())                   # each prints its short caption


def test_the_captions_of_the_paper_tables_are_sentences_that_name_the_networks_one_way():
    """PI 2026-11-12 (the caption brief), shortened 2026-11-13 (round 2, package 3): what the table compares, then
    definitions and run counts, in sentences: no colon-form, no semicolon chain; the networks by one name, never
    "Sophon's network"; one wording of the multi-class AUC; a table that holds the rest of Table 1's networks names
    them by that table; one short run-count sentence."""
    for label, (key, _) in _paper_tables().items():
        cap = M.printed_caption(key)
        assert ": " not in cap and ";" not in cap, label
        assert "Sophon's network" not in cap and "networks pretrained on JetClass-II" not in cap, label
        assert not re.search(r"\((LP|MLP|f\.t\.)\)", cap), label
        assert not re.search(r"\bSD\b", cap) or key == "sculpt_boot", label       # "standard deviation", but for the column names there
        assert not re.search(r"\bcontrols?\b", cap) and "Below the rule" not in cap, label
        assert cap.rstrip().endswith(".") or cap.rstrip().endswith("__"), label
    assert M.PNETS == "the networks pretrained with 188, 162, 43 and 17 classes"
    # the supplement's tables name the networks with PNETS; a paper table that holds the rest of Table 1's networks
    # names them by that table
    assert all(M.PNETS in M.printed_caption(k) for k in ("coarse", "qcdhf", "per_class", "jc1_conf", "jetclass_all", "per_run"))
    assert all("of table~\\ref{tab:summary}" in M.printed_caption(k)
               for k in ("probes", "probes_n", "fill", "ft", "labelsets", "summary_auc"))
    # the detail and mass-effect tables name their definitions by the tables that hold them (round-1 fix 2026-11-13)
    assert "defined as in table~\\ref{tab:mass}" in M.printed_caption("mass_detail")
    assert "those of tables~\\ref{tab:summary}, \\ref{tab:mass}" in M.printed_caption("masseffect")
    # the run counts in one short sentence (round 2): "three runs" covers the label sets, the random network's three
    # initializations and the three seeds from scratch; one count for the mass networks and the flavor-boundary
    # partitions (build() stops if they ever differ); the random partitions pooled
    lead = r" Values are the mean $\pm$ standard deviation over \MlstNRunsWord{} runs"
    assert not hasattr(M, "RUNS_OF") and M._runs("set") == M._runs("set", "init", "scratch") == lead + "."
    assert M._runs("set", "mass") == lead + r", \MlstMassNRunsWord{} per mass network."
    assert M._runs("set", "flav") == lead + r", \MlstMassNRunsWord{} per flavor-boundary partition."
    assert M._runs("set", "flav", "mass", "rand") == M._runs("set", "flav", "mass", "rand", "init", "scratch") == (
        lead + r", \MlstMassNRunsWord{} per mass network and flavor-boundary partition, and over five random "
               "partitions.")
    assert M._runs("set", "rand") == lead + " and over five random partitions."
    # a supplement copy with each random partition on its own row says so ("rand1")
    assert M._runs("set", "rand", "rand1") == M._runs("set", "rand") + " Each random partition is a single run on its own row."
    assert M.printed_caption("summary").endswith(M._runs("set", "flav", "mass", "rand"))
    assert M.printed_caption("mass").endswith(M._runs("set", "mass"))
    assert all(M.printed_caption(k).count("Values are the mean") == 1 for k, _ in _paper_tables().values()
               if k not in ("published", "inputs", "partrepro", "lrsweep", "sculpt", "sculpt_boot"))
    # one definition and wording of a multi-class AUC; on JetClass the sentence on ParT's average over class pairs
    assert M.AUC_OVR == ("AUC is the area under the receiver operating characteristic curve, averaged over the classes, each "
                         "against the rest.")
    assert "averaged on JetClass-II and JetClass over the classes, each against the rest." in M.printed_caption("summary_auc")
    # audit 3 (B): ParT's JetClass AUC is the class-pair average, equal to ours on equal class sizes, and says so
    assert "average over class pairs" in M.printed_caption("published")
    assert "on JetClass averaged over the classes, each against the rest" in M.printed_caption("published")
    # the sentence on the Particle Transformer paper's average over class pairs is in the JetClass tables' captions
    assert M.AUC_JC.strip() in M.printed_caption("jetclass_all") and "Particle Transformer paper" in M.AUC_JC
    # PI 2026-11-19: the pairs as the rejection at 90% signal efficiency, no AUC
    assert "rejection at 90\\% signal efficiency" in M.printed_caption("pairs")
    assert "receiver operating characteristic" not in M.printed_caption("pairs")
    assert "averaged over" not in M.printed_caption("pairs") and "averaged over the target's groups, each against the rest" in M.printed_caption("coarse")
    # the definitions the brief asked for by name: the recomputed random network in the tables that show it (BN_DEF in
    # the supplement's copies, a shorter sentence in the paper's, L16); the flavor boundary in the supplement's
    # captions, and in the paper text for the main and appendix tables, whose network names Table 1's end note defines
    assert M.BN_DEF in M.printed_caption("fill_full") and M.BN_DEF in M.printed_caption("pairs_full")
    assert all(r"\DesignBnJets{} training jets" in M.printed_caption(k) for k in ("fill", "pairs"))
    assert not any("BatchNorm" in M.printed_caption(k) for k in ("probes", "coarse", "qcdhf"))
    assert M.PART_DEF == " A flavor boundary keeps apart decays that differ only in quark flavor."
    assert [lab for lab, (k, _) in _paper_tables().items() if M.PART_DEF in M.printed_caption(k)] == []
    # PI 2026-11-23, LEDGER CA-15: the assertion above holds (tab:sets defines the flavor boundary and captions name it
    # in two or three words), so tab:coarse no longer carries PART_DEF
    assert M.PART_DEF in M.printed_caption("per_run") and M.PART_DEF not in M.printed_caption("coarse")
    # PI 2026-11-22 (Part 2): the networks are defined in tab:sets, not in Table 1's end note
    assert not hasattr(M, "NET_DEF") and "tab:sets" in M.CAPTIONS["sets"][0]
    assert "__SCULPT_CUT__" in M.CAPTION_SHORT["sculpt"] and "resonant" in M.sculpt_fill({"m": {"sig_eff_50": {}}}, {a: [] for a in M.FLAV})["__SCULPT_CUT__"]
    assert all(h in M.printed_caption("mass") for h in ("mass output of the pretraining head", "multilayer-perceptron probe"))
    assert "which are not combined" in M.printed_caption("partrepro") and "cumulative" not in M.printed_caption("partrepro")


def test_every_caption_is_short_and_free_of_the_banned_forms():
    """PI caption brief (2026-11-13, round 2; L16 the same day): the main and appendix captions as printed (mark and
    bold sentences and end notes included) at most 135 words (round 3: Tables 3 and A13 at 134); every supplement
    caption names its rows without "controls", "Below the rule" or "class hierarchy", has no colon-form and no "SD"
    (tab:randprong's fill keeps its colons and is left out)."""
    # L16 (PI 2026-11-13): at most 135 words for every main and appendix table caption, counted with its end notes
    # (the definitions set below the table: Table 1's network names, Table 2's dash, Table A15's Pythia)
    noted = set()
    for label, (key, path) in _paper_tables().items():
        tex = path.read_text()
        cap, _ = _caption_and_body(tex)
        notes = re.findall(r"\\par\\smallskip\{\\noindent (.*?)\}\n", tex)
        n = len(cap.split()) + sum(len(x.split()) for x in notes)
        assert n <= 135, (label, n)
        noted |= {label} if notes else set()
    assert noted == {"tab:summary", "tab:published", "tab:masseffect"}
    G = _load("gallery_placement_caps_t", "experiments/FIGS/figstyle/gallery_placement.py")
    key = {label: k for k, (label, _) in M.CAPTIONS.items()}
    for item, tier, _ in G.TABLES:
        if tier != G.SUPPLEMENT or item == "tab:randprong":
            continue
        cap = M.printed_caption(key[item])
        assert ": " not in cap and not re.search(r"\bSD\b", cap), item
        assert not re.search(r"\bcontrols?\b", cap) and "Below the rule" not in cap and "class hierarchy" not in cap, item
    # the mark sentences tie the mark to "marks" by a non-breaking space (no mark alone at a line end)
    assert all("$~marks " in note for notes in M.KEY_MARKS.values() for _, note in notes)
    assert all("$~marks " in note for _, note in M.MARK_NOTES)
    assert {"pairs_full", "qcdhf_full", "probes_full", "jetclass_all", "jetclass_probe"} <= set(M.KEY_MARKS)
    assert {"per_run", "jetclass_nbkg"} <= set(M.CAPTION_SHORT)


def test_the_generated_paper_tables_follow_the_table_rules():
    """The 20 tables as make_tables.py wrote them (paper/journal/tables, committed): every mark of a table has its
    sentence in the caption and no caption explains a mark the table lacks; never "<" or "+- 0"; no mode tag on a
    row; thousands separators; "parameters"; the full text width, no wrapped row label, nothing scaled, and the body
    size everywhere (2026-11-13: tab:probes, the one exception before, is one turned page at the body size). A bold
    value is read without its wrapper."""
    marks = {M.SINGLE: M.MARK_NOTES[0][1].strip(), M.AGREE: M.MARK_NOTES[1][1].strip()}
    own = (r"\dagger", r"\ddagger", r"\parallel", "^{*}", r"^{\ast}")
    small, tables = set(), _paper_tables()
    for label, (key, path) in tables.items():
        tex = path.read_text()
        cap, body = _caption_and_body(tex)
        assert tex.count(rf"\label{{{label}}}") == 1 and r"\pending" not in tex, label
        for mark, note in marks.items():
            assert (mark in body) == (note in cap), (label, mark)
        notes = dict(M.KEY_MARKS.get(key, ()))
        for mark in own:
            assert (mark in body) == (mark in cap), (label, mark)
            if mark in body:       # a KEY_MARKS sentence, or the caption's own (tab:sculptboot's dagger)
                assert notes.get(mark, f"$^{{{mark}}}$ marks").strip() in cap, (label, mark)
        assert ": " not in cap and ";" not in cap and "Sophon's network" not in tex, label
        assert "$<$" not in tex and not re.search(r"\\pm\$\\,0(\.0+)?\s*(&|\\\\)", _unbold(body)), label
        assert not re.search(r"\((LP|MLP|f\.t\.)\)", tex) and "# params" not in tex and "params" not in body, label
        assert not re.findall(r"(?<![\d.{,])\d{4,}(?![\d}])", _unbold(body)), label         # 1{,}602, never 1602
        # tabular* or longtable, no p column; PI 2026-11-23, LEDGER FG-25: tab:sets (121.7 pt too wide) wraps what each
        # network merges and its runs in p columns, its row labels and class counts unwrapped
        assert r"\begin{tabular}" not in tex and (r"arraybackslash}p{" not in tex or label == "tab:sets"), label
        if label == "tab:sets":
            assert r"{@{\extracolsep{\fill}}ll>{\raggedright\arraybackslash}p{" in tex, label
        assert tex.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}") + tex.count(
            r"\begin{longtable}{@{\extracolsep{\fill}}") >= 1, label
        assert not re.search(r"adjustbox|resizebox|scalebox|\\(footnotesize|scriptsize|tiny)(?![a-zA-Z])", tex), label
        if _sizes_in(tex):
            small.add(label)
    assert small == set()
    read = lambda label: tables[label][1].read_text()      # noqa: E731
    # where the single-run mark is: the kinematic-only and Sophon rows (Sophon in Table 1 and its AUC table too). The
    # reproduction runs and the learning-rate sweep are single runs throughout, which their captions say once (round-1
    # fix 2026-11-13). From scratch on X->bb has its three seeds since 2026-11-13 (eb6c87e; it was one seed, marked),
    # so the resonance tables carry no mark: mean +- SD for both signals, in the classifier columns (X->bb:
    # classifier, linear probe, direct; X->bs: classifier, linear probe)
    assert {lab for lab in tables if M.SINGLE in read(lab)} == {"tab:summary", "tab:summaryauc", "tab:published", "tab:probes"}
    assert all("Every value is from a single run" in read(lab) for lab in ("tab:partrepro", "tab:lrsweep"))
    scratch = next(ln for ln in read("tab:xbx").splitlines() if ln.startswith(M.SCRATCH_LABEL + " & ")).removesuffix(r" \\").split(" & ")
    assert r"\pm" in scratch[1] and r"\pm" in scratch[4] and scratch[2:4] + scratch[5:] == ["--"] * 3 and M.SINGLE not in "".join(scratch)
    # Table A10 (PI 2026-11-18): R50 once in each header, no AUC and no ratio, a lower limit where fewer than 20
    # background jets pass in a run (X->ee vs mumu at 188 classes)
    pairs = read("tab:pairs")
    assert r"{c}{$R_{90}$} \\" in pairs and "AUC" not in _caption_and_body(pairs)[1] and r"\rho" not in pairs
    assert M.PAIR_GT in _caption_and_body(pairs)[1]
    # the rows the brief moved out: BatchNorm recomputed named on its row only in tab:pairs (tab:fill's one random
    # network is the recomputed one, named by the common name, its caption says so); Sophon in the headline tables (on
    # JetClass-II and JetClass) and in tab:probes
    bn, sophon = M.LABEL[M.INIT_BN] + " & ", M.LABEL[M.SOPHON] + " & "
    # PI 2026-11-14: "random network" is the initialized one everywhere; the recomputed one keeps its own name
    assert {lab for lab in tables if bn in read(lab)} == {"tab:pairs", "tab:fill"}
    assert M.LABEL[M.INIT] + " & " not in read("tab:fill") and "BatchNorm recomputed means" in read("tab:fill")
    assert {lab for lab in tables if sophon in read(lab)} == {"tab:summary", "tab:summaryauc", "tab:probes"}
    # tab:mass: the soft-drop mass, the head block (two networks), the perceptron block (seven)
    mass = read("tab:mass").splitlines()
    head, mlp = (mass.index(M._full_row(3, M.HEADING[k], italic=True)) for k in ("head", "mlp"))
    # the soft-drop mass, a reference, last (L5)
    assert mass[mlp + 9].startswith("soft-drop mass & ") and mlp - head == 4 and mass[mlp + 10] == r"\bottomrule"
    # the coarse targets and the b-tagging discriminant left the paper for the supplement (L13), where they are
    # restored (round 3, package 1; with their whole copies): placed and written there, with their pooled
    # random-partitions row
    assert not {"tab:coarse", "tab:qcdhf"} & set(tables)
    G = _load("gallery_placement_coarse_t", "experiments/FIGS/figstyle/gallery_placement.py")
    # tab:qcdhf repeated every row of tab:qcdhffull (audit 3, 2026-11-14): only the whole copy is placed
    assert G.TIER["tab:coarse"] == G.TIER["tab:coarsefull"] == G.TIER["tab:qcdhffull"] == G.SUPPLEMENT
    assert "tab:qcdhf" in G.DROPPED_TABLES
    for k, f in (("coarse", "mlst_coarse"), ("qcdhf", "mlst_qcdhf")):
        assert (k, f) in [(a, b) for a, b, *_ in M.PLACEHOLDERS]
        tex = (ROOT / "paper/journal/tables" / f"{f}.tex").read_text()
        assert tex.count(rf"\label{{{M.CAPTIONS[k][0]}}}") == 1 and r"\quad " + M.LABEL[M.RAND_POOLED] + " & " in tex
    # the retired task tables are still written but placed nowhere
    assert not {"tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig"} & set(tables)


# ------------------------------------------------------------------ PI 2026-11-12, evening: bold, and Table 1 with each task's AUC
def _bold_spans(tex: str) -> list:
    """What each bold wrapper of a printed table holds (M._b: BOLD_ON, the text, a closing brace)."""
    out = []
    while M.BOLD_ON in tex:
        _, rest = tex.split(M.BOLD_ON, 1)
        depth, i = 1, 0
        while depth:
            depth += (rest[i] == "{") - (rest[i] == "}")
            i += 1
        out.append(rest[:i - 1])
        tex = rest[i:]
    return out


def _unbold(tex: str) -> str:
    """A printed table with every bold value light."""
    for x in _bold_spans(tex):
        tex = tex.replace(M._b(x), x)
    return tex


def _frame(rows: list, ncol: int) -> str:
    return "\n".join([r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule", " & " + " & ".join("abcdefgh"[:ncol]) + r" \\",
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def _chosen_bold(key: str, tab: str) -> set:
    """{(row label, column)} of the cells M._bold marks: a token of an upper-case kind, or a published value it
    wrapped. Every other line and cell must be as it was."""
    out = M._bold(key, tab)
    cells = set()
    for a, b in zip(tab.split("\n"), out.split("\n"), strict=True):
        if a == b:
            continue
        pa, pb = M._depth0_split(a, "&"), M._depth0_split(b, "&")
        assert len(pa) == len(pb), a
        for j, (ca, cb) in enumerate(zip(pa, pb)):
            if ca == cb:
                continue
            m = M.TOKEN.search(ca)
            want = (ca.replace(m.group(0), M.V4.TOK_OPEN + m.group(1).upper() + " " + m.group(2) + M.V4.TOK_CLOSE) if m
                    else ca.replace(ca.strip().removesuffix(r"\\").strip(), M._b(ca.strip().removesuffix(r"\\").strip())))
            assert cb == want, (ca, cb)
            cells.add((pa[0].strip(), j))
    return cells


def test_a_bold_value_is_in_the_bold_series_that_is_not_extended_and_its_mark_stays_light():
    """PI 2026-11-12: only the mean and its standard deviation are bold, not the marks. \\textbf's extended bold is
    13% wider and put three tables off the page, so the wrapper is the plain bold series; the width model reads a
    bold value as a light one, so no layout decision depends on which cells are bold."""
    assert M.BOLD_ON == r"{\fontseries{b}\selectfont " and M._b("0.94") == r"{\fontseries{b}\selectfont 0.94}"
    assert r"\textbf" not in M.BOLD_ON and not hasattr(M, "BOLD_SIGMA")      # PI 2026-11-19: the single best, no 3 sigma
    # _render: an upper-case token kind is the lower-case one printed bold
    pm = r"0.500\,$\pm$\,0.010"          # PI 2026-11-14: per cell as the PDG rounds
    assert M._render("p", [0.5, 0.01], 2) == pm and M._render("P", [0.5, 0.01], 2) == M._b(pm)
    assert M._render("s", [0.94], 4) == "0.9400" + M.SINGLE and M._render("S", [0.94], 4) == M._b("0.9400") + M.SINGLE
    assert M._render("p", [0.5, 0.0], 3) == "0.500" + M.AGREE and M._render("P", [0.5, 0.0], 3) == M._b("0.500") + M.AGREE
    assert M._render("P", [0.91472, 0.000001], 5) == M._b("0.91472") + M.AGREE        # an SD that prints as zero
    assert M._render("P", [1802.0, 60.0], -1) == M._b(r"1{,}800\,$\pm$\,60")
    # uniform_columns: the kind's case does not move the column's place
    tok, T = M.V4.tok, M.TAB
    up = lambda t: t.replace(M.V4.TOK_OPEN + "p", M.V4.TOK_OPEN + "P").replace(M.V4.TOK_OPEN + "s", M.V4.TOK_OPEN + "S")   # noqa: E731
    cells = (T(0.98560, 0.00004), T(0.9841, 0.0003), T(0.9838, 0.0002), tok("s", 0.98))
    assert [_unbold(c) for c in _column(*map(up, cells))] == _column(*cells)
    assert _column(up(cells[0]), cells[1], cells[2]) == [M._b(r"0.98560\,$\pm$\,0.00004"), r"0.98410\,$\pm$\,0.00030",
                                                        r"0.98380\,$\pm$\,0.00020"]
    # the estimated width of a cell
    for x in (r"1{,}800\,$\pm$\,60", "0.9400" + M.SINGLE, r"0.98560\,$\pm$\,0.00004", "950"):
        bold = M._b(x.removesuffix(M.SINGLE)) + (M.SINGLE if x.endswith(M.SINGLE) else "")
        assert M._cell_cm(bold) == M._cell_cm(x), x
    # the helpers of these tests
    assert _bold_spans("a & " + M._b(r"1{,}800\,$\pm$\,60") + M.AGREE + " & " + M._b("2")) == [r"1{,}800\,$\pm$\,60", "2"]
    assert _unbold("a & " + M._b(r"1{,}800\,$\pm$\,60") + M.AGREE + " & " + M._b("2")) == r"a & 1{,}800\,$\pm$\,60" + M.AGREE + " & 2"


def test_bold_marks_the_best_value_of_a_column_and_every_value_within_three_standard_deviations():
    """PI 2026-11-12 (the bold rule): in a column the best value is bold and so is every value with
    |mean - best| <= 3 sqrt(SD^2 + SD_best^2), from the unrounded values of the tokens; a single run has SD zero.
    Never bold: the random networks and the released checkpoint (reference rows, which are not the best either), a
    number that is no run's, a plain number in a table without published rows, a dash."""
    T, tok, lab = M.TAB, M.V4.tok, M.LABEL
    assert M.NEVER_BOLD == (lab[M.INIT], lab[M.INIT_BN], lab["KIN"], lab[M.SOPHON], M.SCRATCH_LABEL)
    rows = [f"{lab['L188']} & {T(100.0, 2.0)} & 7 & -- \\\\",                    # the best
            f"{lab['L162']} & {T(95.0, 1.0)} & 9 & -- \\\\",                     # 5 <= 3 sqrt(1 + 4) = 6.7
            f"{lab['R42_Q1']} & {T(93.0, 1.0)} & 8 & {T(5.0, 1.0)} \\\\",        # 7 > 6.7
            f"{lab['R16_Q1']} & {tok('s', 94.0)} & 8 & -- \\\\",                 # one run: 6 <= 3 sqrt(0 + 4) = 6
            r"\multicolumn{4}{@{}l}{" + M.PARTITIONS + r"} \\",
            rf"\quad {lab['FLAV_F0']} & {tok('s', 93.9)} & 8 & -- \\",           # one run: 6.1 > 6
            f"{lab[M.INIT]} & {T(200.0, 1.0)} & 8 & {T(9.0, 1.0)} \\\\",         # above the best: a reference row
            rf"\quad {lab[M.INIT_BN]} & {T(101.0, 1.0)} & 8 & {T(9.0, 1.0)} \\",
            r"\midrule",
            f"{lab[M.SOPHON]} & {tok('s', 300.0)} & 8 & {tok('s', 9.0)} \\\\",
            f"soft-drop mass & {tok('o', 300.0)} & 8 & {tok('o', 9.0)} \\\\"]
    tab = _frame(rows, 3)
    # column 1 as above; column 2 holds plain numbers (no published rows in this table); column 3 one value that
    # could be bold, so none
    assert _chosen_bold("xbx", tab) == {(lab["L188"], 1), (lab["L162"], 1), (lab["R16_Q1"], 1)}
    printed = M.uniform_columns(M._bold("xbx", tab)).split("\n")
    assert printed[4:8] == [rf"{lab['L188']} & " + M._b(r"100.0\,$\pm$\,2.0") + r" & 7 & -- \\",
                            rf"{lab['L162']} & " + M._b(r"95.0\,$\pm$\,1.0") + r" & 9 & -- \\",
                            rf"{lab['R42_Q1']} & 93.0\,$\pm$\,1.0 & 8 & 5.0\,$\pm$\,1.0 \\",
                            rf"{lab['R16_Q1']} & " + M._b("94.0") + M.SINGLE + r" & 8 & -- \\"]
    assert printed[9] == rf"\quad {lab['FLAV_F0']} & 93.9{M.SINGLE} & 8 & -- \\" and sum(M.BOLD_ON in ln for ln in printed) == 3
    # each value with its own SD: b is close to the best in value and resolved from it (0.0008 > 3 sqrt(2) 0.0001), c
    # is far in value and not resolved (0.0204 <= 3 * 0.0100)
    close = _frame([f"a & {T(0.5004, 0.0001)} \\\\", f"b & {T(0.4996, 0.0001)} \\\\", f"c & {T(0.48, 0.01)} \\\\"], 1)
    assert _chosen_bold("xbx", close) == {("a", 1), ("c", 1)}
    # a column with fewer than two values that could be bold has none; the table is returned as it was
    lone = _frame([f"a & {T(1.0, 0.1)} & -- \\\\", r"b & -- & -- \\", f"{lab[M.INIT]} & {T(2.0, 0.1)} & {T(2.0, 0.1)} \\\\"], 2)
    assert M._bold("xbx", lone) == lone
    # rows without a frame (a block of _task_pages) are read from the first line
    block = "\n".join([f"a & {T(1.0, 0.1)} \\\\", f"b & {T(2.0, 0.1)} \\\\"])
    assert _chosen_bold("probes", block) == {("b", 1)}


def test_bold_takes_the_smallest_value_or_the_one_closest_to_one_in_the_columns_named_so():
    """PI 2026-11-12: higher is better for accuracy, AUC, rejections, signal efficiency; lower for the mass resolution
    and the Jensen-Shannon distance of the sculpting table; the median response is best where closest to 1, and the
    test is on |mean - best| there too."""
    T, tok = M.TAB, M.V4.tok
    assert {k: {d: v[d] for d in ("lower", "unity") if d in v} for k, v in M.BOLD.items() if "lower" in v or "unity" in v} == {
        "mass": {"lower": (1,), "unity": (2,)}, "sculpt": {"lower": (1, 3, 6)}, "sculpt_full": {"lower": (1, 3, 6)}}
    # tab:sculpt: a distance column (1) and an efficiency column (2)
    sculpt = _frame([f"x & {T(4.4, 0.3)} & {T(50.0, 1.0)} \\\\",        # the smallest distance, the largest efficiency
                     f"y & {T(5.2, 0.3)} & {T(40.0, 1.0)} \\\\",        # 0.8 <= 3 sqrt(0.18) = 1.27;  10 > 4.2
                     f"z & {T(6.0, 0.1)} & {T(49.0, 1.0)} \\\\",        # 1.6 > 3 sqrt(0.10) = 0.95;   1 <= 4.2
                     "random selection & " + tok("o", 0.9) + r" & -- \\"], 2)       # no run's number: not the best
    assert _chosen_bold("sculpt", sculpt) == {("x", 1), ("y", 1), ("x", 2), ("z", 2)}
    # tab:mass: resolution (1, lower) and median response (2, closest to 1), a block under each italic heading
    head, mlp = (M._full_row(3, M.HEADING[k], italic=True) for k in ("head", "mlp"))
    mass = _frame(["soft-drop mass & " + tok("o", 0.05) + " & " + tok("o", 1.0) + r" \\", r"\midrule", head,
                   f"a & {T(0.20, 0.01)} & {T(1.02, 0.01)} \\\\",       # response: |1.02 - 0.99| = 0.03 <= 3 sqrt(2) 0.01
                   f"b & {T(0.10, 0.01)} & {T(0.99, 0.01)} \\\\",       # the best of both columns
                   r"\midrule", mlp,
                   f"c & {T(0.30, 0.01)} & {T(0.90, 0.01)} \\\\",       # the block's best resolution, above the other block's
                   f"d & {T(0.31, 0.01)} & {T(1.05, 0.005)} \\\\",      # 1.05 is closer to 1 than 0.90: the best response
                   f"{M.LABEL[M.INIT]} & {T(0.01, 0.001)} & {T(1.0, 0.001)} \\\\"], 2)
    assert _chosen_bold("mass", mass) == {("b", 1), ("b", 2), ("a", 2), ("c", 1), ("d", 1), ("d", 2)}
    # the same rows in a table where higher is better: the largest of column 1 is z, and y (0.8 <= 3 sqrt(0.10))
    assert _chosen_bold("xbx", sculpt) == {("z", 1), ("y", 1), ("x", 2), ("z", 2)}


def test_bold_is_decided_in_each_block_of_a_column_and_the_partitions_subheading_starts_no_block():
    """PI 2026-11-12: a block is the rows under one italic mode heading within one task block. The subheading
    "17-class partitions" does not start one; tab:jetclassrej is one block whatever its headings (its two published
    rows above the heading take part)."""
    T, lab = M.TAB, M.LABEL
    lp, ft = (M._full_row(2, M.HEADING[k], italic=True) for k in ("lp", "ft"))
    rows = [lp, f"x & {T(0.50, 0.01)} \\\\", M._full_row(2, M.PARTITIONS), rf"\quad y & {T(0.49, 0.01)} \\",
            f"{lab[M.INIT]} & {T(0.10, 0.01)} \\\\", r"\midrule",
            ft, f"z & {T(0.30, 0.01)} \\\\", f"w & {T(0.10, 0.01)} \\\\"]
    assert _chosen_bold("summary", _frame(rows, 1)) == {("x", 1), (r"\quad y", 1), ("z", 1)}
    # the same rows as one block: z is no longer a best value
    whole = [r"ParT~\cite{part} & 100 & 0.51 \\", r"ParticleNet~\cite{part} & 200 & 0.45 \\", r"\midrule",
             M._full_row(3, M.HEADING["ft"], italic=True),
             f"x & 300 & {T(0.50, 0.01)} \\\\", f"z & 400 & {T(0.30, 0.01)} \\\\"]
    assert M.BOLD["jetclass_rej"] == {"whole": True, "published": True, "skip": (1,)}
    # A3 (PI 2026-11-14): a published value without an uncertainty is never bold and sets no best, so x is the best;
    # column 1 (training jets) is skipped
    assert _chosen_bold("jetclass_rej", _frame(whole, 2)) == {("x", 2)}
    # a table of blocks in its caption's words: these keys (tab:published names its own words, "of")
    # PI 2026-11-23 (LEDGER SP-13, step 4): tab:summaryaucfull, the supplement's copy with every column, bolds as its table
    assert {k for k, v in M.BOLD.items() if v.get("blocks")} == {"summary", "summary_auc", "summary_auc_full", "pairs",
                                                                 "probes_n", "ft", "jc1_class_auc", "jc1_class_auc_full",
                                                                 "mass"}
    assert set(M.BOLD) == {"summary", "summary_auc", "summary_auc_full", "published", "labelsets", "probes", "pairs",
                           "probes_n", "ft",
                           "jc1_class_auc", "mass", "sculpt", "jetclass_rej", "xbx", "xbx_rej", "coarse", "qcdhf", "fill",
                           "masseffect", "labelsets_full", "jc1_class_auc_full", "sculpt_full"}    # the supplement copies (PI 2026-11-14)
    # tab:masseffect is built transposed, so the rule applies along its rows, and its caption says so, with which value
    # is best in each row (the "best" sentence before the bold note); tab:published's words name the task blocks
    assert M.BOLD["masseffect"]["of"] == "their row" and set(M.BOLD["masseffect"]) == {"of", "best"}
    assert M.BOLD["published"]["of"] == "their column and task" and "best" in M.BOLD["sculpt"]
    # no bold in the tables of inputs, recipes and differences
    assert not {"inputs", "partrepro", "lrsweep", "sculpt_boot"} & set(M.BOLD)


def test_a_published_value_takes_part_in_bold_with_the_standard_deviation_it_prints_and_a_count_never():
    """PI 2026-11-12: published rows take part by their printed value; where a published cell prints a standard
    deviation that is its SD, where it prints none its SD is zero. The parameters and the training jets are
    counts."""
    T, tok = M.TAB, M.V4.tok
    assert M.BOLD["published"] == {"of": "their column and task", "published": True, "skip": (1, 2)}
    rows = [r"A~\cite{a} & 2.14\,M & 1.2\,M & 691\,$\pm$\,15 & 0.9877 \\",         # the best R50
            r"B~\cite{b} & 2.38\,M & 1.2\,M & 640\,$\pm$\,10 & 0.9878 \\",         # 51 <= 3 sqrt(100 + 225) = 54.1; the best AUC
            r"C~\cite{c} & 366\,k & 1.2\,M & 397\,$\pm$\,7 & -- \\",               # 294 > 49.7
            r"D~\cite{d} & 1\,k & -- & 3{,}000 & 0.9877 \\",                       # see below
            rf"from scratch (this work) & 2.21\,M & 1.2\,M & {T(461.0, 4.0)} & {tok('s', 0.9878)} \\"]
    block = "\n".join(rows[:3] + rows[4:])
    # R50: 691 +- 15 and 640 +- 10. AUC: the published values print no uncertainty, so they are never bold (A3, PI
    # 2026-11-14), and our single run is then the column's only value: no bold
    assert _chosen_bold("published", block) == {(r"A~\cite{a}", 3), (r"B~\cite{b}", 3)}
    out = M._bold("published", block).split("\n")
    assert out[1] == r"B~\cite{b} & 2.38\,M & 1.2\,M & " + M._b(r"640\,$\pm$\,10") + r" & 0.9878 \\"
    assert out[2] == rows[2] and M.V4.tok("s", 0.9878) in out[3] and T(461.0, 4.0) in out[3]
    # a published value without a printed SD (D's 3,000) is never bold and sets no best
    assert _chosen_bold("published", "\n".join(rows)) == {(r"A~\cite{a}", 3), (r"B~\cite{b}", 3)}
    # plain numbers without an uncertainty are never bold (A3, PI 2026-11-14), in count and value columns alike
    assert _chosen_bold("published", "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\") == set()
    assert _chosen_bold("jetclass_rej", "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\") == set()
    # without published rows a table's plain numbers take no part
    assert M._bold("xbx", "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\") == "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\"


def test_a_column_whose_best_value_prints_as_a_bound_has_no_bold():
    """Table A10: an AUC that rounds to 1.0000 prints "> 0.9999", so the best of its block's column is no printed
    value and the column has no bold there; the other columns and the other block are untouched by that."""
    T = M.TAB
    lp, ft = (M._full_row(3, M.HEADING[k], italic=True) for k in ("lp", "ft"))
    rows = [lp, f"a & {M.NEAR_ONE} & {T(0.90, 0.01)} \\\\", f"b & {T(0.99, 0.001)} & {T(0.80, 0.01)} \\\\",
            f"e & {M.NEAR_ONE}{M.SINGLE} & {T(0.70, 0.01)} \\\\", r"\midrule",
            ft, f"c & {T(0.95, 0.01)} & {T(0.70, 0.01)} \\\\", f"d & {T(0.90, 0.01)} & {T(0.90, 0.01)} \\\\"]
    assert _chosen_bold("pairs", _frame(rows, 2)) == {("a", 2), ("c", 1), ("d", 2)}


def test_the_caption_says_what_bold_marks_once_and_only_where_a_cell_is_bold():
    """_float: one sentence in the caption, PI 2026-11-19 (restated 2026-11-22) "Bold marks the best value in each
    column.", the single best value per block and column and every value tied with it at the printed precision; none
    where no cell is bold or the table is not one of BOLD."""
    T, lab = M.TAB, M.LABEL
    col = blk = M.BOLD_NOTE
    assert col == " Bold marks the best value in each column." and M.BOLD_NOTE_ROW == " Bold marks the best value in each row."
    rows = [f"{lab['L188']} & {T(100.0, 2.0)} \\\\", f"{lab['L162']} & {T(95.0, 1.0)} \\\\", f"{lab['R42_Q1']} & {T(93.0, 1.0)} \\\\",
            f"{lab[M.INIT]} & {T(200.0, 1.0)} \\\\"]
    f = M._float("xbx", _frame(rows, 1))
    assert f.count(col) == 1 and r"\textbf" not in f and M.V4.TOK_OPEN not in f
    # the single best (95 is within three standard deviations of 100 but is not the best: no longer bold)
    assert lab["L188"] + " & " + M._b(r"100.0\,$\pm$\,2.0") + r" \\" in f and lab["L162"] + r" & 95.0\,$\pm$\,1.0 \\" in f
    assert lab["R42_Q1"] + r" & 93.0\,$\pm$\,1.0 \\" in f and lab[M.INIT] + r" & 200.0\,$\pm$\,1.0 \\" in f
    assert f.count(M.BOLD_ON) == 1 and f.index(col) < f.index(M.BOLD_ON)          # the sentence is the caption's
    # values tied at the printed precision are all bold (0.9001 and 0.9004 both print 0.900 beside SDs of 0.010)
    tie = M._float("xbx", _frame([f"{lab['L188']} & {T(0.9001, 0.01)} \\\\", f"{lab['L162']} & {T(0.9004, 0.01)} \\\\",
                                  f"{lab['R42_Q1']} & {T(0.85, 0.01)} \\\\"], 1))
    assert tie.count(M.BOLD_ON) == 2 and lab["R42_Q1"] + r" & 0.850\,$\pm$\,0.010 \\" in tie
    # a table of blocks; its soft-drop row is no run's
    head, mlp = (M._full_row(2, M.HEADING[k], italic=True) for k in ("head", "mlp"))
    mass = _frame([head, f"{lab['L162_MASS']} & {T(0.10, 0.01)} \\\\", f"{lab['R16_Q1_MASS']} & {T(0.20, 0.01)} \\\\", r"\midrule",
                   mlp, f"{lab['L188']} & {T(0.30, 0.01)} \\\\", f"{lab['L162']} & {T(0.50, 0.01)} \\\\"], 1)
    g = M._float("mass", mass)
    assert g.count(M.BOLD_NOTE) == 1 and g.count(M.BOLD_ON) == 2
    assert lab["L162_MASS"] + " & " + M._b(r"0.100\,$\pm$\,0.010") + r" \\" in g and lab["L188"] + " & " + M._b(r"0.300\,$\pm$\,0.010") + r" \\" in g
    # nothing to compare: no bold cell, no sentence
    lone = M._float("xbx", _frame(rows[:1] + rows[3:], 1))
    assert M.BOLD_ON not in lone and "Bold marks" not in lone
    # a table that is not one of BOLD (the supplement's copies, the recipe tables)
    assert "probes_full" not in M.BOLD
    other = M._float("probes_full", _frame(rows, 1))
    assert M.BOLD_ON not in other and "Bold marks" not in other and lab["L188"] + r" & 100.0\,$\pm$\,2.0 \\" in other


def test_table_one_holds_the_headline_metric_of_five_tasks_and_its_auc_table_the_auc():
    """tab:summary and tab:summaryauc (PI 2026-11-13, L1, L2): Table 1 is the main-text headline figure's table, the
    headline metric of five tasks (JetClass-II and JetClass accuracy, R30 on top tagging and on quark/gluon tagging
    scored on Pythia and on Herwig jets) for the linear probes and for fine-tuning; the appendix table holds the AUC
    beside each, in the same rows. The released Sophon checkpoint below a rule on JetClass-II and JetClass only."""
    assert M.SUMMARY_PANELS == (("jc2", 1_000_000, "jc2_188", "accuracy"), ("jc1", 1_000_000, "jc1", "accuracy"),
                                ("top", 1_200_000, "top", "r30"), ("qg", 1_600_000, "qg", "r30"),
                                ("qg_herwig", 1_600_000, "qg_herwig", "r30")) == M.SUMMARY_GRID
    assert M.SUMMARY_AUC == {"jc2": "macro_auc_ovr", "jc1": "macro_auc_ovr", "top": "auc", "qg": "auc", "qg_herwig": "auc"}
    assert M.SUMMARY_AUC_GRID == tuple((k, n, ds, M.SUMMARY_AUC[k]) for k, n, ds, _ in M.SUMMARY_PANELS)
    # PI 2026-11-23, LEDGER SP-9 (merge 5): tab:summaryauc's resonant and QCD columns have their heads
    assert M.SUMMARY_HEAD == {"accuracy": "accuracy", "r30": "$R_{30}$", "macro_auc_ovr": "AUC", "auc": "AUC",
                              "macro_auc_resonant": "AUC, resonant", "macro_auc_qcd": "AUC, QCD"}
    assert M.SUMMARY_NO_SOPHON == ("top", "qg", "qg_herwig") and M.SUMMARY_CONTROLS == M.FLAV + (M.RAND_POOLED,)
    assert not hasattr(M, "SUMMARY_LABEL_CM")                  # the row labels never wrap: no label width
    # the captions: what each compares, the AUC defined once in the table that prints it, no class pairs; shortened in
    # round 2 (package 3); the final round (2026-11-13): the released Sophon checkpoint's dashes explained by pointing
    # to tab:probes, the network names in Table 1's end note (NET_DEF), the flavor boundary in the paper text
    cap, auc = M.printed_caption("summary"), M.printed_caption("summary_auc")
    # PI 2026-11-19 (A): R30 in plain words with why 30%; PI 2026-11-22: the released checkpoint's top and quark/gluon
    # probes use the same substitutions (the same-fill refit), so no zero-filled note
    assert ("$R_{30}$, one over the fraction of background jets kept when the cut keeps 30\\% of the signal jets, the "
            "working point of the top-tagging comparison") in cap and "class pairs" not in cap
    assert re.findall(r"\\ref\{([^}]*)\}", cap) == [] and "zero-filled" not in cap
    assert M.PART_DEF not in cap
    # PI 2026-11-23, LEDGER SP-9 (merge 5): and on JetClass-II over the resonant and the QCD classes
    assert auc.count("averaged on JetClass-II and JetClass over the classes, each against the rest, and on JetClass-II "
                     "also over its resonant and over its QCD classes alone.") == 1
    assert "table~\\ref{tab:summary}" in auc and "class pairs" not in auc
    assert M.BOLD["summary"] == M.BOLD["summary_auc"] == {"blocks": True} and not {"summary", "summary_auc"} & set(M.MODES)
    # the generated tables: two whole turned pages at the body size under one caption, the second starting a page
    # under "(continued)"; the probes' page, then the fine-tuned one
    # PI 2026-11-23, LEDGER SP-9 (merges 4, 5): Table 1's X->bb / X->bs columns, the AUC table's class groups
    head = {"summary": " & accuracy & accuracy & $R_{30}$ & $R_{30}$, Pythia & $R_{30}$, Herwig & $X\\to b\\bar b$ & $X\\to bs$ \\\\",
            "summaryauc": " & AUC & AUC, resonant & AUC, QCD & AUC & AUC & AUC, Pythia & AUC, Herwig \\\\"}
    for name, label in (("mlst_summary.tex", "summary"), ("mlst_summary_auc.tex", "summaryauc")):
        tex = (ROOT / "paper/journal/tables" / name).read_text()
        tabs = re.findall(r"\\begin\{tabular\*\}.*?\\end\{tabular\*\}", tex, re.S)
        assert tex.startswith(r"\begin{landscape}") and len(tabs) == 2 and r"\begin{longtable}" not in tex and not _sizes_in(tex)
        assert tex.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}") == 2 and r"\setlength{\tabcolsep}{6pt}" in tex
        assert tex.count(r"\captionof{table}{") == tex.count(rf"\label{{tab:{label}}}") == 1
        assert tex.count(rf"\textit{{Table~\ref{{tab:{label}}} (continued)}}") == tex.count(r"\newpage") == 1
        assert tex.count(M.BOLD_NOTE.strip()) == 1 and "three standard deviations (run" not in tex
        assert "regroup the 188 classes into 17" not in tex      # PI 2026-11-22: no end note; tab:sets defines them
        for tab, mode in zip(tabs, ("lp", "ft"), strict=True):
            lines = tab.splitlines()
            assert lines[2] == r" & JetClass-II & JetClass & top tagging & \multicolumn{2}{c}{quark/gluon tagging} \\"
            assert lines[3] == r"\cmidrule(lr){5-6}" and lines[4] == head[label]
            # the mode heading and the 17-class partitions subheading, both italic (PI 2026-11-14)
            assert lines[6] == M._full_row(6, M.HEADING[mode], italic=True) and tab.count(r"\textit{") == 2
            assert r"\textit{" + M.PARTITIONS + "}" in tab
            names = [ln.split(" & ")[0] for ln in lines if " & " in ln and not ln.startswith(" & ")]
            refs = [M.LABEL[M.INIT], M.LABEL[M.SOPHON]] if mode == "lp" else [M.SCRATCH_LABEL]
            assert names == [M.LABEL[a] for a in M.SETS] + [r"\quad " + M.LABEL[a] for a in M.SUMMARY_CONTROLS] + refs
            assert lines.count(M._full_row(6, M.PARTITIONS)) == 1
            last = lines[-3].removesuffix(r" \\").split(" & ")
            assert last[0] == refs[-1] and len(last) == 6
            if mode == "lp":       # Sophon below a rule, one run, on JetClass-II and JetClass only; references never bold
                assert lines[-4] == r"\midrule" and last[3:] == ["--"] * 3 and all(c.endswith(M.SINGLE) for c in last[1:3])
                assert M.BOLD_ON not in lines[-3] + lines[-5]


def test_the_generated_tables_are_bold_where_the_rule_applies_and_say_so_once():
    """The 20 tables as make_tables.py wrote them (PI 2026-11-12): bold in the tables that compare networks and in no
    table of inputs, recipes or differences; the caption's sentence once, in the words for a table with blocks or
    without; a bold wrapper holds a value and its standard deviation and nothing else (no mark, no dash, no bound);
    never \\textbf; no bold in a reference row or a count column; the columns where lower or closest to 1 is best are
    the ones their headers name."""
    tables = _paper_tables()
    plain = {"tab:inputs", "tab:partrepro", "tab:lrsweep", "tab:sculptboot"}
    # PI 2026-11-19 / 2026-11-22: one sentence, "in each column" (tab:masseffect, built transposed: "in each row")
    words = ("column", "row")
    notes = [M.BOLD_NOTE.strip(), M.BOLD_NOTE_ROW.strip()]
    refs = M.NEVER_BOLD + M.REFERENCE_ROWS
    for label, (key, path) in tables.items():
        tex = path.read_text()
        cap, body = _caption_and_body(tex)
        assert r"\textbf" not in tex and M.BOLD_ON not in cap, label
        assert (M.BOLD_ON in body) == (key in M.BOLD) == (label not in plain), label
        want = ("row" if M.BOLD[key].get("of") == "their row" else "column") if key in M.BOLD else None
        assert [cap.count(n) for n in notes] == [int(x == want) for x in words], label
        spans = _bold_spans(body)
        assert all(M.PLAIN.fullmatch(x) for x in spans), label
        for ln in body.splitlines():
            cells = ln.split(" & ")
            if len(cells) > 1 and cells[0].removeprefix(r"\quad ").startswith(refs):
                assert M.BOLD_ON not in ln, (label, cells[0])
    read = lambda label: tables[label][1].read_text()      # noqa: E731
    rows = lambda label, n: [c for ln in read(label).splitlines() if len(c := ln.removesuffix(r" \\").split(" & ")) == n]   # noqa: E731
    # the parameters and training jets of tab:published, the training jets of tab:jetclassrej
    assert len(rows("tab:published", 7)) > 20 and not any(M.BOLD_ON in c for r in rows("tab:published", 7) for c in r[:3])
    assert len(rows("tab:jetclassrej", 11)) >= 7 and not any(M.BOLD_ON in c for r in rows("tab:jetclassrej", 11) for c in r[:2])
    # tab:sculpt: the three distance columns are the lower-is-better ones; tab:mass: resolution lower, response to 1
    head = lambda label: next(ln for ln in read(label).splitlines() if ln.startswith(" & ") and "multicolumn" not in ln   # noqa: E731
                              ).removesuffix(r" \\").split(" & ")
    assert tuple(j for j, h in enumerate(head("tab:sculpt")) if h == "distance") == M.BOLD["sculpt"]["lower"]
    assert head("tab:mass") == ["", "resolution", "median response"]
    assert (M.BOLD["mass"]["lower"], M.BOLD["mass"]["unity"]) == ((1,), (2,))
    # Table A10: the X->ee against mumu column, whose best prints as a lower limit, has no bold (wherever the table is cut)
    ee = next(h for k, h, *_ in M.PAIRS if k == "ee_vs_mm")
    cut = [ln.removesuffix(r" \\").split(" & ") for ln in read("tab:pairs").splitlines() if " & " in ln]
    header = next(r for r in cut if ee in r)
    column = [r[header.index(ee)] for r in cut if len(r) == len(header) and r[0].strip() and ee not in r]
    assert len(column) > 10 and any(c.startswith(M.PAIR_GT) for c in column) and not any(M.BOLD_ON in c for c in column)


# ------------------------------------------------------------------ PI 2026-11-13: the mass-effect table (brief B) and one JetClass source (L2)
def test_the_mass_effect_table_has_a_row_per_task_metric_and_mode_and_the_four_networks_as_columns():
    """tab:masseffect (brief B): one summary of the mass-regression head as values, not differences. Rows are the tasks
    and metrics, for linear probe and fine-tuned where both exist (mass resolution and response, JetClass-II, JetClass,
    top, quark/gluon on Pythia and on Herwig jets, X->bb and X->bs, sculpting); columns 162, 162 + mass, 17, 17 + mass
    with their run counts; one turned page a group of blocks (MASS_EFFECT_PAGES)."""
    XR = _load("xbx_report_masseffect_t", "experiments/XBS/xbx_report.py")
    runs = M._paper(ROOT)
    rows = M.masseffect_rows({}, {}, {}, {}, {}, {}, XR, {}, {}, {}, runs)
    lp, mlp, ft = "linear probe", "multilayer-perceptron probe", "fine-tuned"

    def task(metrics, probes=(lp,), extra=()):
        return [f"{w}, {how}" for how in probes for w in metrics] + list(extra) + [f"{w}, {ft}" for w in metrics]
    eff = "signal efficiency (\\%), "
    want = {"mass": [f"{w}, {how}" for how in ("mass output of the pretraining head", mlp, "ridge probe")
                     for w in ("resolution", "median response")],
            "jc2": task(("accuracy", "AUC")),
            "jc1": task(("accuracy", "AUC"), (lp, mlp)),
            "top": task(("$R_{30}$", "AUC"), extra=[f"$R_{{30}}$, {mlp}"]),
            "qg": task(("$R_{30}$", "AUC")), "qg_herwig": task(("$R_{30}$", "AUC")),
            # the read-outs in Table 4's words (round-1 fix 2026-11-13)
            "xbb": [eff + w for w in ("classifier", "linear probe", "direct")],
            "xbs": [eff + w for w in ("classifier", "linear probe")],
            "sculpt": [f"at {at}, {how}" for how in ("network output used directly", ft)
                       for at in ("10\\% QCD efficiency", "1\\% QCD efficiency", "50\\% signal efficiency")]}
    assert list(want) == list(M.MASS_EFFECT_BLOCKS) == [b for page in M.MASS_EFFECT_PAGES for b in page]
    assert M.MASS_EFFECT_BLOCKS["qg_herwig"] == "quark/gluon tagging, Herwig jets"
    assert [(b, lab) for b, lab, _, _ in rows] == [(b, lab) for b, labs in want.items() for lab in labs]
    # the bold direction of each row: the resolution and the Jensen-Shannon distance lower, the response closest to 1
    assert all(d == ("lower" if lab.startswith(("resolution", "at ")) else "unity" if lab.startswith("median") else "higher")
               for _, lab, _, d in rows)
    # the generated table: these rows, under the four networks with their run counts on every page
    tex = (ROOT / "paper/journal/tables/mlst_masseffect.tex").read_text()
    head = [" & " + " & ".join(M.LABEL[a] for a in M.MASS_EFFECT) + r" \\",
            " & " + " & ".join(f"{len(runs[a])} runs" for a in M.MASS_EFFECT) + r" \\"]
    lines = tex.splitlines()
    assert all(lines.count(h) == len(M.MASS_EFFECT_PAGES) for h in head) and head[1] == r" & 3 runs & 2 runs & 3 runs & 2 runs \\"
    printed = [ln.split(" & ") for ln in lines if " & " in ln and not ln.startswith(" & ")]
    assert [p[0] for p in printed] == [lab for _, lab, _, _ in rows] and all(len(p) == 5 for p in printed)
    assert [M._full_row(5, M.MASS_EFFECT_BLOCKS[b]) for b in want] == [ln for ln in lines if ln.startswith(r"\multicolumn{5}{@{}l}{")
                                                                       and "continued" not in ln]


def test_the_mass_effect_table_prints_each_row_at_one_place_and_bolds_along_the_row():
    """table_masseffect builds the table transposed, so that one precision and the bold rule (D1, D19) apply to each
    row: the best value of a row by its direction and every value within three standard deviations of it."""
    runs = {"L162": [f"mtx-l162-s{k}" for k in (1, 2, 3)], "L162_MASS": ["mtx-l162mass-s1", "mtx-l162mass-s2"],
            "R16_Q1": [f"mtx-r16q1-s{k}" for k in (1, 2, 3)], "R16_Q1_MASS": ["mtx-r16q1mass-s1", "mtx-r16q1mass-s2"]}
    arm = {m: a for a, ms in runs.items() for m in ms}

    def value(means: dict, sd: float):
        """Each run's value: the network's mean, and offsets whose sample SD is sd."""
        off = {3: (-sd, 0.0, sd), 2: (-sd / math.sqrt(2), sd / math.sqrt(2))}
        return lambda m: means[arm[m]] + off[len(runs[arm[m]])][runs[arm[m]].index(m)]
    rows = [("mass", "resolution, x", value({"L162": 0.20, "L162_MASS": 0.10, "R16_Q1": 0.21, "R16_Q1_MASS": 0.12}, 0.01), "lower"),
            ("mass", "median response, x", value({"L162": 1.10, "L162_MASS": 0.99, "R16_Q1": 1.05, "R16_Q1_MASS": 1.20}, 0.01), "unity"),
            ("jc2", "accuracy, linear probe", value({"L162": 0.410, "L162_MASS": 0.409, "R16_Q1": 0.30, "R16_Q1_MASS": 0.35}, 0.001), "higher"),
            ("xbb", "signal efficiency, direct", None, "higher")]
    t = M.table_masseffect(rows, runs, M.TAB)
    parts = t.split("%SPLIT%")
    assert len(parts) == len(M.MASS_EFFECT_PAGES) and M.V4.TOK_OPEN not in t
    body = {ln.split(" & ")[0]: ln.removesuffix(r" \\").split(" & ")[1:] for ln in t.splitlines()
            if " & " in ln and not ln.startswith(" & ")}
    bold = {lab: [M.MASS_EFFECT[j] for j, c in enumerate(cells) if M.BOLD_ON in c] for lab, cells in body.items()}
    # median response: |1.05 - 1| - |0.99 - 1| = 0.04 <= 3 sqrt(2) 0.01, |1.10 - 1| - 0.01 = 0.09 and 0.20 - 0.01 are not
    assert bold == {"resolution, x": ["L162_MASS", "R16_Q1_MASS"], "median response, x": ["L162_MASS", "R16_Q1"],
                    "accuracy, linear probe": ["L162", "L162_MASS"], "signal efficiency, direct": []}
    assert body["signal efficiency, direct"] == ["--"] * 4
    # one place a row (its standard deviations), each value the mean of its network's runs
    res = [M.PLAIN.fullmatch(_unbold(c)).groups() for c in body["resolution, x"]]
    assert [float(m) for m, _ in res] == [0.20, 0.10, 0.21, 0.12] and {len(m.split(".")[1]) for m, _ in res} == {3}
    # the rows sit under their blocks' headings, on the page of their block
    first = parts[0].splitlines()
    k = first.index(M._full_row(5, M.MASS_EFFECT_BLOCKS["mass"]))
    assert [ln.split(" & ")[0] for ln in first[k + 1:k + 3]] == ["resolution, x", "median response, x"]
    assert "signal efficiency, direct" in parts[[i for i, pg in enumerate(M.MASS_EFFECT_PAGES) if "xbb" in pg][0]]


def test_one_source_for_the_fine_tuned_jetclass_accuracy_and_auc():
    """L2 (PI 2026-11-13): every table that prints the fine-tuned JetClass accuracy or AUC reads it where Table 1 and
    the figures do, the fine-tuning read-out on one million jets (jc1_values: summary_ft, summary_scratch;
    budget_long_metrics.json, the AUC on every test jet); the curves' blocks keep the per-class rejections."""
    import numpy as np
    runs = M._paper(ROOT)
    pre, ref = _jc1_trees(runs, ("L188",))              # the blocks: accuracy 0.800.., AUC 0.980..
    rec = lambda acc, auc: {"accuracy": acc, "macro_auc_ovr": auc}       # noqa: E731
    acc, auc = [0.8466, 0.8468, 0.8471], [0.98548, 0.98535, 0.98544]
    per = {"jc1": {**{m[len("mtx-"):]: {1_000_000: [rec(a, u)]} for m, a, u in zip(runs["L188"], acc, auc)},
                   M.SCRATCH: {1_000_000: [rec(0.8267, 0.9818), rec(0.8255, 0.9821), rec(0.8279, 0.9817)]}}}
    vals = M.jc1_values(per, runs)
    assert vals("L188", "accuracy") == acc and vals("L188", "macro_auc_ovr") == auc
    assert vals(None, "macro_auc_ovr") == [0.9818, 0.9821, 0.9817] and vals("R16_Q1", "accuracy") is None
    tok = lambda v: M.V4.tok("p", float(np.mean(v)), float(np.std(v, ddof=1)))      # noqa: E731
    _, ours = M.V4.jetclass_rows(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, M.TAB, M.TAB.two, vals=vals)
    _, blocks = M.V4.jetclass_rows(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, M.TAB, M.TAB.two)
    assert ours[0][1][:2] == [tok(acc), tok(auc)] and ours[-1][1][:2] == [tok([0.8267, 0.8255, 0.8279]), tok([0.9818, 0.9821, 0.9817])]
    assert ours[0][1][:2] != blocks[0][1][:2] and [r[1][2:] for r in ours] == [r[1][2:] for r in blocks]
    # Table 1 and its AUC table print the mean of the same runs on JetClass
    for grid, values in ((M.SUMMARY_GRID, acc), (M.SUMMARY_AUC_GRID, auc)):
        ft = M.table_summary({}, {}, {}, per, runs, M.TAB, grid).split("%SPLIT%")[1]
        row = next(ln for ln in ft.splitlines() if ln.startswith(M.LABEL["L188"] + " & ")).split(" & ")
        printed = M.PLAIN.fullmatch(_unbold(row[2])).group(1)
        assert float(printed) == round(float(np.mean(values)), len(printed.split(".")[1]))


def test_every_table_prints_one_fine_tuned_jetclass_value_for_one_quantity():
    """L2 on the generated tables: the fine-tuned 188-class network and from scratch on JetClass print the same
    accuracy and AUC in Table 1, its AUC table, the comparison with published models and the supplement's JetClass
    tables, and the 188-class AUC is the mean of budget_long_metrics.json's macro_auc_ovr over its three runs."""
    tables = ROOT / "paper/journal/tables"
    budget = DATA / "budget/budget_long_metrics.json"
    if not (tables / "mlst_summary_auc.tex").exists():
        pytest.skip("the generated tables are not there")

    def cell(name: str, label: str, col: int, after: str | None = None) -> tuple:
        lines = (tables / name).read_text().splitlines()
        if after:
            lines = lines[next(i for i, ln in enumerate(lines) if after in ln):]
        row = next(ln for ln in lines if ln.startswith(label + " & ")).removesuffix(r" \\").split(" & ")
        return M.PLAIN.fullmatch(_unbold(row[col]).strip()).groups()
    ft = rf"\textit{{{M.HEADING['ft']}}}"
    full, scr = M.LABEL["L188"], M.SCRATCH_LABEL
    pub = {full: f"fine-tuned from {full}{M.THIS_WORK}", scr: scr + M.THIS_WORK}
    # tab:jetclassall (page review round 3): no training-jets and parameters columns, accuracy and AUC first
    for label in (full, scr):
        accs = {cell("mlst_summary.tex", label, 2, ft), cell("mlst_published.tex", pub[label], 3),
                cell("mlst_jetclass.tex", label, 3), cell("mlst_jetclass_all.tex", label, 1)}
        aucs = {cell("mlst_summary_auc.tex", label, 2, ft), cell("mlst_published.tex", pub[label], 4),
                cell("mlst_jetclass.tex", label, 4), cell("mlst_jetclass_all.tex", label, 2)}
        assert len(accs) == len(aucs) == 1, (label, accs, aucs)
    if budget.exists():
        cells = json.loads(budget.read_text())["cells"]
        v = [cells[f"leg2/{m.removeprefix('mtx-')}/N1000000/s1"]["macro_auc_ovr"] for m in M._paper(ROOT)["L188"]]
        printed = cell("mlst_summary_auc.tex", full, 2, ft)[0]
        assert len(v) == 3 and float(printed) == round(statistics.mean(v), len(printed.split(".")[1]))


def test_the_rows_tab_published_repeats_print_as_tab_labelsets_prints_them():
    """L17 (PI 2026-11-13): the fine-tuned 188-class network and from scratch on top and quark/gluon tagging print the
    same accuracy and R50 cells in tab:published and tab:labelsets, bold aside (both read bench_ft_cells through
    repeated_cells), and the label-sets caption says that they repeat."""
    tables = ROOT / "paper/journal/tables"
    if not (tables / "mlst_labelsets.tex").exists():
        pytest.skip("the generated tables are not there")

    def rows(name: str, label: str) -> list:
        return [[_unbold(c).strip() for c in ln.removesuffix(r" \\").split(" & ")]
                for ln in (tables / name).read_text().splitlines() if ln.startswith(label + " & ")]
    full, scr = M.LABEL["L188"], M.SCRATCH_LABEL
    pub = {full: f"fine-tuned from {full}{M.THIS_WORK}", scr: scr + M.THIS_WORK}
    for label in (full, scr):
        lab = rows("mlst_labelsets.tex", label)[0]
        top, qg = rows("mlst_published.tex", pub[label])[1:3]       # the blocks JetClass, top, quark/gluon
        # tab:published: accuracy column 3, R50 column 5; tab:labelsets: top accuracy, R50, then quark/gluon's
        assert [top[3], top[5], qg[3], qg[5]] == lab[1:5], (label, top, qg, lab)
    assert "The rows for \\Kfull{} classes and from scratch repeat those of table~\\ref{tab:published}." in \
        M.printed_caption("labelsets")


# ------------------------------------------------------------------ round 2 (2026-11-13): the restored and new tables
def test_the_compute_table_gives_each_runs_last_validation_accuracy_and_a_last_block_of_means():
    """tab:compute (supplement, body size): a validation-accuracy column per run (the last of its epochs), the
    partitions' runs indented under their subheading and never broken apart; the longtable closed by CLOSE_RULE (no
    last foot, page review round 3); then, on a new page, a two-column table with each network's mean +- SD over its
    runs as the pretraining-curve figures draw it, in L5 order, the flavor-boundary partitions and the pooled random
    partitions under the subheading."""
    ep = lambda acc: [{"device": "NVIDIA GeForce RTX 3090", "train_seconds": 3600.0, "val_seconds": 0.0, "val_acc": a}   # noqa: E731
                      for a in acc]
    comp = {"mtx-l188-s1": {"epochs": ep([0.5, 0.60])}, "mtx-l188-s2": {"epochs": ep([0.5, 0.62])},
            "mtx-flavf0-s1": {"epochs": ep([0.4, 0.55])}, "mtx-rand2p1-s1": {"epochs": ep([0.3, 0.45])},
            "mtx-rand2p2-s2": {"epochs": ep([0.3, 0.47])}}
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"], "FLAV_F0": ["mtx-flavf0-s1"],
            "RAND2_p1": ["mtx-rand2p1-s1"], "RAND2_p2": ["mtx-rand2p2-s2"], "L162": ["mtx-l162-s1"]}
    assert M._val_acc(comp, "mtx-l188-s2") == 0.62 and M._val_acc(comp, "mtx-l162-s1") is None
    t, hours = M.table_compute(comp, runs, MT.fmt_int)
    lines = t.splitlines()
    assert lines[:2] == [r"\begin{landscape}", "{"] and not _sizes_in(t)
    assert any(ln.startswith(M.LABEL["L188"] + ", seed 1 & 0.6") for ln in lines)
    assert next(ln for ln in lines if ln.startswith(M.LABEL["L162"] + ", seed 1 & ")).split(" & ")[1:3] == ["--", r"\pending{training}"]
    # the per-run rows: the partitions indented under their subheading, held together (\\* up to the last row)
    k = lines.index(M._full_row(5, M.PARTITIONS) + "*")
    end = lines.index(r"\end{longtable}")
    assert [ln.split(" & ")[0] for ln in lines[k + 1:end - 1]] == [
        r"\quad " + M.LABEL["FLAV_F0"] + ", seed 1", r"\quad " + M.LABEL["RAND2_p1"] + ", seed 1",
        r"\quad " + M.LABEL["RAND2_p2"] + ", seed 2"]
    assert all(ln.endswith(r"\\*") for ln in lines[k + 1:end - 2]) and lines[end - 2].endswith(r" \\")
    # closed by the rule bound to the last row, no \endlastfoot (a taller last foot lost the repeated heading)
    assert lines[end - 1] == M.CLOSE_RULE and r"\endlastfoot" not in t and r"\bottomrule" not in "\n".join(lines[:end])
    # the summary: a two-column table of its own on the next page, under "(continued)"
    assert lines[end + 1:end + 3] == [r"\newpage", r"\noindent\textit{Table~\ref{tab:compute} (continued)}\par\smallskip"]
    summary = lines[end + 3:]
    assert summary[0] == r"\noindent\begin{tabular}{@{}lc@{}}" and "validation accuracy at the last pretraining epoch" in summary[2]
    block = summary[summary.index(r"\midrule") + 1:summary.index(r"\bottomrule")]
    names = [ln.split(" & ")[0] for ln in block]
    # L162 has no epochs: no row; the partitions under the subheading, indented
    assert names == [M.LABEL["L188"], M._full_row(2, M.PARTITIONS), r"\quad " + M.LABEL["FLAV_F0"],
                     r"\quad " + M.LABEL[M.RAND_POOLED]]
    cell = lambda name: next(ln for ln in block if ln.startswith(name + " & ")).removesuffix(r" \\").split(" & ")[1]   # noqa: E731
    assert cell(M.LABEL["L188"]) == r"0.610\,$\pm$\,0.014" and cell(r"\quad " + M.LABEL[M.RAND_POOLED]) == r"0.460\,$\pm$\,0.014"
    assert cell(r"\quad " + M.LABEL["FLAV_F0"]).startswith("0.55") and M.SINGLE in cell(r"\quad " + M.LABEL["FLAV_F0"])
    # PI 2026-11-23, LEDGER SP-9: this per-run table is the data release's copy (tab:computefull); the appendix's
    # tab:compute holds one row per stage, its caption pointing to the release for each run
    cap = M.CAPTIONS["compute_full"][1]
    assert "validation accuracy" in cap and "80 epochs" in cap and "last part of the table" in cap and "last block" not in cap
    assert M.CAPTIONS["compute_full"][0] == "tab:computefull" and "each stage" in M.CAPTIONS["compute"][1]
    stages = M.table_compute_stages({}, {"L188": ["mtx-l188-s1"]}, MT.fmt_int,
                                    ft_h={"pre": {"V100": 2.4}, "scratch": {}}).splitlines()
    assert [ln.split(" & ")[0] for ln in stages if " & " in ln][1:] == [
        r"pretraining, 0 runs \pending{1 runs without a record}", "fine-tuning of the pretrained networks",
        "fine-tuning from scratch"]


def test_the_confusion_table_holds_the_matrices_and_paired_changes_of_the_confusion_figures(monkeypatch):
    """tab:jc1conf (supplement, round 2): per network a block of the ten true classes by the predicted classes, in the
    figures' class order (style_v5.JC1_CLASSES); 188 classes and from scratch (its full-schedule cells) as percentages,
    every other network as its change from its reference, each run against the reference run of the same pretraining
    seed, in percentage points (gallery_perclass._diff); five predicted classes a part; a missing reference run stops
    the build."""
    assert M.CONF_CLASSES == ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql", "label_Zqq",
                              "label_Wqq", "label_Tbqq", "label_Tbl")
    assert set(M.CONF_TEX) == set(M.CONF_CLASSES) == set(dict(M.JC1_AUC_COLS))
    n = len(M.CONF_CLASSES)

    def doc(diag):
        rows = [[diag if i == j else (1 - diag) / (n - 1) for j in range(n)] for i in range(n)]
        return {"classes": list(reversed(M.CONF_CLASSES)), "row_normalised": [r[::-1] for r in rows[::-1]]}
    import numpy as np
    assert M._conf_matrix(doc(0.8))[0, 0] == pytest.approx(80.0) and M._conf_matrix(doc(0.8)).shape == (n, n)
    # diagonals exact in binary, so that the expected tokens are the table's to the last bit
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2"], "R16_Q1": ["mtx-r16q1-s1", "mtx-r16q1-s2"]}
    conf = {"mtx-l188-s1": doc(0.75), "mtx-l188-s2": doc(0.875), "mtx-r16q1-s1": doc(0.625), "mtx-r16q1-s2": doc(0.8125)}
    scratch = {"scratch-s1": doc(0.5), "scratch-s2": doc(0.625), "scratch-s3": doc(0.75)}
    n_diff = len(M.CONF_DIFF)
    monkeypatch.setattr(M, "CONF_DIFF", (("R16_Q1", "L188"),))
    parts = M.table_jc1_conf(conf, scratch, runs).split("%SPLIT%")
    # round-1 fix 2026-11-13: one network a page, its two halves of predicted classes stacked under its heading; the
    # first network's two halves on two pages, the second under "(continued)"; one decimal place a whole matrix
    titles = [f"{M.LABEL['L188']}, percentage of the true class", f"{M.LABEL['L188']}, percentage of the true class (continued)",
              f"{M.LABEL['R16_Q1']} minus {M.LABEL['L188']}, percentage points", f"{M.SCRATCH_LABEL}, percentage of the true class"]
    assert len(parts) == 4 and all(p.count(r"{\itshape ") == 1 for p in parts)
    assert [p.splitlines()[2] for p in parts] == [M._full_row(6, r"{\itshape " + t + "}") for t in titles]
    first = parts[0].splitlines()
    assert first[4:8] == [r" & \multicolumn{5}{c}{predicted class} \\", r"\cmidrule(lr){2-6}",
                          "true class & " + " & ".join(M.CONF_TEX[c] for c in M.CONF_CLASSES[:5]) + r" \\", r"\midrule"]
    assert parts[1].splitlines()[6] == "true class & " + " & ".join(M.CONF_TEX[c] for c in M.CONF_CLASSES[5:]) + r" \\"
    # every other network: both halves in one tabular, the second below a rule and its own header
    assert [p.count("true class & ") for p in parts] == [1, 1, 2, 2]
    qcd = [next(ln for ln in p.splitlines() if ln.startswith(M.CONF_TEX["label_QCD"] + " & ")).split(" & ")[1]
           for p in (parts[0], parts[2], parts[3])]
    # means 81.25 (75, 87.5), -9.375 (-12.5, -6.25) and 62.5 (50, 62.5, 75), printed rather than left as tokens, each
    # rounded as the PDG rounds (PI 2026-11-14)
    assert qcd == [r"81\,$\pm$\,9", r"-9\,$\pm$\,4", r"62\,$\pm$\,12"] and M.V4.TOK_OPEN not in "".join(parts)
    np.testing.assert_allclose([float(np.std([75.0, 87.5], ddof=1)), float(np.std([-12.5, -6.25], ddof=1))], [8.84, 4.42], atol=0.01)
    # a run without the reference run of its seed stops the build
    del conf["mtx-r16q1-s2"]
    with pytest.raises(SystemExit, match="holds 1 of the 2 runs of R16_Q1"):
        M.table_jc1_conf(conf, scratch, runs)
    assert M.CAPTIONS["jc1_conf"][0] == "tab:jc1conf" and "its own, longer learning-rate schedule" in M.printed_caption("jc1_conf")
    tex = ROOT / "paper/journal/tables/mlst_jc1_conf.tex"
    if tex.exists():
        t = tex.read_text()
        assert t.count(r"\label{tab:jc1conf}") == 1 and not re.search(r"& -\d", t) and "$-$" in t and not _sizes_in(t)
        # a heading a network, the first network's two (its halves on two pages): 2 + 9 differences + from scratch
        assert t.count(r"{\itshape ") == 2 + n_diff + 1 and n_diff == 9
        assert all(f"{{\\itshape 17-class partition with {w} minus {M.LABEL['L188']}" in t
                   for w in ("no flavor boundary", "a $b$-quark boundary", "a random flavor boundary"))
        assert f"{{\\itshape 17-class random partitions minus {M.LABEL['L188']}" in t


def test_the_build_reads_the_l3_bootstraps_apart_from_tab_boot_and_one_run_count_for_mass_and_flavor():
    """build() (round 2): bootstrap/l3 has its own layout (an "l3" block, read by tab:sculptboot) and is left out of
    tab:boot's folders, whose files make_tables crashed on (KeyError 'b'); _runs() writes one count for the mass
    networks and the flavor-boundary partitions, so the build stops if the paper's runs ever give them different
    counts."""
    import inspect
    src = inspect.getsource(M.build)
    assert 'd.name != "l3"' in src and "data / SCULPT_BOOT_L3" in src
    assert "the mass networks and the flavor-boundary partitions have different run counts" in src
    runs = M._paper(ROOT)
    assert len({len(runs[a]) for a in M.MASS + M.FLAV}) == 1
    l3 = DATA / "bootstrap/l3"
    if l3.is_dir():
        boot = {d.name for d in (DATA / "bootstrap").glob("*") if d.is_dir() and "." not in d.name and d.name != "l3"}
        assert "l3" not in boot
        assert all("differences" in json.loads(p.read_text()) or "diffs" in json.loads(p.read_text()) for p in l3.glob("*.json"))


def test_the_jet_pt_values_are_a_csv_file_of_the_data_release():
    """L18 (PI 2026-11-13): set at 8 pt the per-bin table (tab:ptbins) still ran past four pages, so the values of the
    jet-pT figures are data_release/jet_pt_bins.csv (pt_bins_csv), as jetclass2_per_class.csv: one row per network,
    mode, test set, quantity and bin, the mean, the sample SD (empty for one run) and the number of runs; the networks
    in the L5 order, then the mode's reference (from scratch fine-tuned, the random network as a JetClass probe); a pT
    binning that differs between two runs of one folder stops the build. The typeset table is gone."""
    assert M.PT_DIRS == ("jc2", "jc2_ft", "jc1_probe", "jc1", "jc1_fulldecay") and not hasattr(M, "PT_PER_BLOCK")
    assert [(t, q, d) for t, q, d, _ in M.PT_BLOCKS] == [
        ("JetClass-II", "accuracy", (("lp", "jc2"), ("ft", "jc2_ft"))), ("JetClass", "accuracy", (("lp", "jc1_probe"), ("ft", "jc1"))),
        ("JetClass-II", "AUC of X->bb against X->cc", (("lp", "jc2"), ("ft", "jc2_ft")))]
    assert M.PT_REFS == {"jc1_probe": (M.INIT, "jc1_probe", "init-s"), "jc2_ft": (None, "jc2_ft", "scratch-s"),
                         "jc1": (None, "jc1_fulldecay", "scratch-s")}
    assert M.PT_CSV == "data_release/jet_pt_bins.csv" and not hasattr(M, "table_pt_bins")
    assert "pt_bins" not in M.CAPTIONS and "pt_bins" not in [k for k, *_ in M.PLACEHOLDERS]
    bins = lambda a: [{"accuracy": a + 0.01 * i, "bb_vs_cc": {"auc": 0.8}} for i in range(2)]      # noqa: E731
    runs = {**{a: [] for a in M.SETS + M.FLAV + (M.RAND_POOLED,)}, "L188": ["mtx-l188-s1", "mtx-l188-s2"]}
    jc2 = {"mtx-l188-s1": {"pt_edges": [200, 300, 400], "bins": bins(0.30)},
           "mtx-l188-s2": {"pt_edges": [200, 300, 400], "bins": bins(0.32)}}
    jc1 = {m: {"pt_edges": [500, 600, 700], "bins": bins(0.5)} for m in runs["L188"]}
    perf = {"jc2": jc2, "jc2_ft": {**jc2, "scratch-s1": jc2["mtx-l188-s1"]},
            "jc1_probe": {**jc1, "init-s1": jc1["mtx-l188-s1"]}, "jc1": {"mtx-l188-s1": jc1["mtx-l188-s1"]},
            "jc1_fulldecay": {}}
    lines = M.pt_bins_csv(perf, runs).splitlines()
    assert lines[0] == "network,mode,test_set,quantity,pt_low_GeV,pt_high_GeV,mean,sd,n_runs"
    # the two runs: mean 0.31, sample SD 0.01 * sqrt(2); a single run: no SD; a network with no file: no row
    assert lines[1:3] == ["188 classes,linear probe,JetClass-II,accuracy,200,300,0.310000,0.014142,2",
                          "188 classes,linear probe,JetClass-II,accuracy,300,400,0.320000,0.014142,2"]
    assert "from scratch,fine-tuned,JetClass-II,accuracy,200,300,0.300000,,1" in lines
    assert "random network,linear probe,JetClass,accuracy,500,600,0.500000,,1" in lines
    assert "188 classes,fine-tuned,JetClass,accuracy,500,600,0.500000,,1" in lines        # one run of two: n_runs says so
    assert not any(ln.startswith(("162 classes,", "from scratch,fine-tuned,JetClass,", "random network,linear probe,JetClass-II"))
                   for ln in lines)
    assert sum(",AUC of X->bb against X->cc," in ln for ln in lines) == 6 and len(lines) == 1 + 18
    jc2["mtx-l188-s2"] = {**jc2["mtx-l188-s2"], "pt_edges": [200, 300, 500]}
    with pytest.raises(SystemExit, match="different pT binnings"):
        M.pt_bins_csv(perf, runs)
    # the released file: the networks in the L5 order, then the references; JetClass-II in ten bins from 200 to 2500 GeV,
    # JetClass in five from 500 to 1000 GeV; every network of the figures with all of its runs
    f = ROOT / "paper/journal" / M.PT_CSV
    if not f.exists():
        pytest.skip("the data release is not generated")
    rows = list(csv.DictReader(f.open()))
    order = [M._plain(a) for a in M.SETS + M.FLAV + (M.RAND_POOLED,)] + [M.SCRATCH_LABEL, M._plain(M.INIT)]
    assert [n for n in dict.fromkeys(r["network"] for r in rows)] == order
    assert order[:10] == ["188 classes", "162 classes", "162 classes + mass", "43 classes", "17 classes", "17 classes + mass",
                          "no flavor boundary", "b-quark boundary", "random flavor boundary", "random partitions"]
    edges = lambda ts: sorted({(int(r["pt_low_GeV"]), int(r["pt_high_GeV"])) for r in rows if r["test_set"] == ts})  # noqa: E731
    assert [lo for lo, _ in edges("JetClass-II")] + [edges("JetClass-II")[-1][1]] == [200, 300, 400, 550, 700, 850, 1000, 1250, 1500, 2000, 2500]
    assert [lo for lo, _ in edges("JetClass")] + [edges("JetClass")[-1][1]] == [500, 600, 700, 800, 900, 1000]
    pooled = M._pooled(M._paper(ROOT))
    for r in rows:
        arm = next((a for a in M.SETS + M.FLAV + (M.RAND_POOLED,) if M._plain(a) == r["network"]), None)
        if arm is not None:
            assert int(r["n_runs"]) == len(pooled[arm]), r
        assert (r["sd"] == "") == (r["n_runs"] == "1") and 0 < float(r["mean"]) < 1, r


def test_every_placed_table_is_set_at_the_body_size_and_the_per_run_table_names_its_modes_plainly():
    """D14 (round 2, package 2): every table the placement puts in the main text, the appendix or the supplement, as
    make_tables.py wrote it, carries no type-size command (tab:ftfull, tab:perrun, tab:probesnfull and tab:compute were
    \\small, tab:perclass \\scriptsize); tab:perrun's header says "linear probe on the frozen features" and "fine-tuned",
    not "Sophon's network"; tab:perclass sets a mass network's name on two lines so six columns fit a turned page."""
    G = _load("gallery_placement_sizes_t", "experiments/FIGS/figstyle/gallery_placement.py")
    key = {label: k for k, (label, _) in M.CAPTIONS.items()}
    name = {k: f for k, f, *_ in M.PLACEHOLDERS} | dict(M.LONGTABLES)
    for item, _, _ in G.TABLES:
        for lab in item.split(" + "):
            tex = (ROOT / "paper/journal/tables" / f"{name[key[lab]]}.tex").read_text()
            assert not _sizes_in(tex) and not re.search(r"\\(footnotesize|scriptsize|tiny)(?![a-zA-Z])", tex), lab
    per_run = (ROOT / "paper/journal/tables/mlst_per_run.tex").read_text()
    # the headline family's five panels in each mode (round-1 fix 2026-11-13: the per-run table reads as tab:summary)
    assert r"\multicolumn{5}{c}{linear probe on the frozen features} & \multicolumn{5}{c}{fine-tuned}" in per_run
    assert "Sophon's network" not in per_run
    per_class = (ROOT / "paper/journal/tables/mlst_per_class.tex").read_text()
    assert r"\begin{tabular}[b]{@{}c@{}}\Kres{} classes\\+ mass\end{tabular}" in per_class


def test_one_rejection_convention_where_no_background_jet_passes():
    """PI 2026-11-14: a rejection read from any stored file follows mlst_v4.rejection: the value where a background
    jet passes; a lower limit (Bound, printed with a geq sign and explained once) of the background count where none
    does, whatever the file stores (a zero count, a bound flag, the 1e12 sentinel, an infinite value, a null); a dash
    where the count is unknown. The figures draw no lower limit (paper_figures.probe_series)."""
    V = M.V4
    assert V.rejection(250.0) == 250.0 and not isinstance(V.rejection(250.0), V.Bound)
    assert isinstance(V.rejection(400.0, bound=True), V.Bound) and V.rejection(400.0, bound=True) == 400.0
    for stored in (1e12, float("inf")):
        assert V.rejection(stored, True, 0, 103453) == 103453.0 and isinstance(V.rejection(stored, True, 0, 103453), V.Bound)
        assert V.rejection(stored, True, 0) is None
    assert V.rejection(None, n_pass=0, n_bkg=298597) == 298597.0 and V.rejection(None) is None
    assert V.background_count([{"eps_b": 0.0, "n_bkg_pass": 0}, {"eps_b": 0.000165, "n_bkg_pass": 33}]) == 200000
    assert M.Bound is V.Bound and M._rej({"r30": 1e12, "r30_is_bound": True, "r30_n_bkg_pass": 0}, "r30") is None
    assert M.MARK_NOTES[-1][0] == r"$\geq$" and M._mark_notes("C.", r"a & $\geq$5") == "C." + M.MARK_NOTES[-1][1]


def test_every_mass_item_reads_the_truth_free_refit_only(tmp_path):
    """PI 2026-11-21: Table 3, Table A9, Figs. A9 and A10 and the mass detail table read head/atlas_res_v3 (and its
    fine-binned twin) only, never the class-centred head/atlas_res: no generator nor figure module names the old
    folders, and with only the old folders present the mass items stay pending."""
    import re
    old = re.compile(r"""["']head/atlas_res(_fine)?["']""")
    for f in ("experiments/FIGS/mlst_results.py", "experiments/FIGS/mlst_v4.py", "experiments/FIGS/paper_figures.py",
              "experiments/FIGS/figstyle/gallery_physics.py", "experiments/FIGS/figstyle/gallery_captions.py",
              "experiments/FIGS/make_tables.py"):
        p = ROOT / f
        if p.exists():
            hits = [ln for ln in p.read_text().splitlines() if old.search(ln)]
            assert not hits, (f, hits[:3])
    assert (M.MASS_RES, M.MASS_RES_FINE) == ("head/atlas_res_v3", "head/atlas_res_fine_v3")
    assert "head/atlas_res" not in M.V4.V5 and not any(v.startswith("head/atlas_res") for v in M.V4.V5.values())
    (tmp_path / "head" / "atlas_res").mkdir(parents=True)
    (tmp_path / "head" / "atlas_res" / "mtx-l188-s1.json").write_text("{}")
    assert M._load(tmp_path / M.MASS_RES) == {}


def test_the_book_refuses_figures_and_tables_built_from_different_fine_tuning_read_outs(tmp_path, monkeypatch):
    """2026-11-21 (audit): the preview's Fig. 8 drew the one-rule cells and its Table A6 the older trees, from two
    builds; both outputs record mlst_results.ft_source and make_results_book stops when they differ."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mrb_src", ROOT / "experiments/FIGS/make_results_book.py")
    mrb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrb)
    figs, tabs = tmp_path / "figs", tmp_path / "tables"
    figs.mkdir(), tabs.mkdir()
    with pytest.raises(SystemExit, match="no fine-tuning source stamp"):
        mrb.check_ft_source(figs, tabs)
    (figs / "ft_source.txt").write_text(M.FIXED_PARTIAL + " (preview)\n")
    (tabs / "ft_source.txt").write_text("budget, budget_long, fulldecay, prefull (the trees before the one rule)\n")
    with pytest.raises(SystemExit, match="build both with the same data"):
        mrb.check_ft_source(figs, tabs)
    (tabs / "ft_source.txt").write_text(M.FIXED_PARTIAL + " (preview)\n")
    assert mrb.check_ft_source(figs, tabs).startswith(M.FIXED_PARTIAL)
    (tmp_path / "budget").mkdir()
    assert M.ft_source(tmp_path).startswith("budget")
    (tmp_path / "budget" / M.FIXED_PARTIAL).write_text("{}")
    monkeypatch.setenv(M.PREVIEW_ENV, "1")
    assert M.ft_source(tmp_path) == M.FIXED_PARTIAL + " (preview)"
    (tmp_path / "budget" / M.FIXED).write_text("{}")
    assert M.ft_source(tmp_path) == M.FIXED


def test_the_mass_figure_claims_are_the_soft_drop_bootstrap_s():
    """PI 2026-11-21: every better/worse-than-soft-drop claim decided by the paired bootstrap (mass_sd_boot_v3.json) on
    the 15 bins Figs. A9 and A10 draw (all jets: ten jet-pT bins, five generator-mass bins). Fig. A9: each mass network
    better than the soft-drop mass in every bin, less biased in 13 and more in the 20-50 GeV bin; Fig. A10: the MLP probe
    on each mass network better than on its label set alone in every bin."""
    f = DATA / "head" / "mass_sd_boot_v3.json"
    if not f.exists():
        pytest.skip("head/mass_sd_boot_v3.json not pulled")
    d = json.loads(f.read_text())
    fig = lambda r: {k: v for k, v in r.items() if k.startswith(("all|pt_fine:", "all|mgen:"))}    # noqa: E731
    sys.path.insert(0, str(ROOT / "experiments/FIGS/figstyle"))
    import gallery_captions as K
    # PI 2026-11-23, LEDGER SP-9: A17-probe is a figure of the data release, its caption on its page
    a9, a10 = K.CAPTIONS["F9b"], K.RELEASE_CAPTIONS["A17-probe"]
    for g in ("162 classes + mass", "17 classes + mass"):
        r = fig(d["groups"][g]["regressed"])
        assert len(r) == 15 and all(v["pass"] and v["better"] for v in r.values()), g
        better_bias = sum(v["bias"]["pass"] and v["bias"]["better"] for v in r.values())
        worse_bias = [k for k, v in r.items() if v["bias"]["pass"] and not v["bias"]["better"]]
        assert better_bias == 13 and worse_bias == ["all|mgen:20-50"], (g, better_bias, worse_bias)
        diff = fig(d["diffs"][f"{g} - {g.removesuffix(' + mass')}"]["mlp"])
        assert len(diff) == 15 and all(v["resolution"]["pass"] and v["resolution"]["better"] for v in diff.values()), g
    # PI 2026-11-23, LEDGER SP-9 (STRUCTURE_PROPOSAL, F9b): no per-bin table stays, so the bias count is qualitative;
    # PI 2026-11-23, LEDGER CA-15 (CAPTION_CUTS_1123 F9b): the bias clause left the paper caption (1db8bc37) and the page
    # caption (ee2bab77), so the caption states the resolution finding only (the bias counts above still hold)
    assert "better than the soft-drop mass in every bin" in a9 and "less biased" not in a9 and "20 to 50 GeV" not in a9
    assert "better on the networks with an added mass regression" in a10 and "than on the same label sets" in a10
    # PI 2026-11-21: no sentence says a probe beats the soft-drop mass in every bin; the bootstrap must show why
    for g, est in (("188 classes", "mlp"), ("188 classes", "ridge")):
        r = fig(d["groups"][g][est])
        assert any(not (v["pass"] and v["better"]) for v in r.values()), (g, est)
    import gallery_placement as P
    texts = list(K.CAPTIONS.values()) + [r[2] for r in P._FIGURES + P._TABLES if isinstance(r[2], str)]
    texts += [p.read_text() for p in sorted((ROOT / "paper/journal/sections").glob("*.tex"))]
    every = re.compile(r"[^.]*\b(every|each|all)\b[^.]*\bbins?\b[^.]*", re.I)
    for t in texts:
        for s in (re.sub(r"\{[\w:]+\}", "", m.group(0)) for m in every.finditer(t)):
            if (re.search(r"soft-drop|\\msd|groomed", s) and re.search(r"probe|MLP|multilayer|ridge|linear regression", s)
                    and re.search(r"better|narrower|beat|outperform|resolves", s)):
                raise AssertionError(f"a probe against the soft-drop mass in every bin: {s.strip()[:200]}")


def test_the_top_and_quark_gluon_sentences_are_the_every_metric_bootstrap():
    """PI 2026-11-21: Fig. 2's top and quark/gluon sentence is written by hand, R30 only, with a pointer to the tables,
    and the captions of the tables that print accuracy, R50 and AUC name the networks that differ; every claim is
    gallery_labelset.TOP_QG_PROBE_CLAIMS, and the paired bootstrap of the v5 probes (bootstrap/l3_v5/*.all.json)
    must give exactly those passes."""
    sys.path.insert(0, str(ROOT / "experiments/FIGS/figstyle"))
    import gallery_captions as K
    src = (ROOT / "experiments/FIGS/figstyle/gallery_labelset.py").read_text()
    ns = {"json": json}         # one namespace: top_qg_passes reads TOP_QG_METRIC as a global
    exec(src[src.index("TOP_QG_BOOT = ("):src.index("PAGES = [dataclasses.replace")], ns)
    f2 = K.CAPTIONS["F2"]
    assert "{top_qg_probe}" not in f2 and "no network differs from 188 classes in R30 by more than three" in f2
    assert "in accuracy, R50 and AUC some do (tables {tab:probes} and {tab:summaryauc})" in f2
    claims = ns["TOP_QG_PROBE_CLAIMS"]
    assert not [k for k in claims if k[2] == "r30"]                     # Fig. 2 and table 1 print R30: no claim
    key, word = "probes", "17 classes + mass is more accurate"
    assert word.replace("17", "\\Kcoarse{}") in M.printed_caption(key) or word in M.printed_caption(key), key
    # PI 2026-11-23, LEDGER SP-9 and SC-1 (4): Table 1's and its AUC table's captions say what they hold, no finding;
    # their top and quark/gluon nulls are rows of tab:tests
    assert "In $R_{30}$" not in M.printed_caption("summary") and "three standard" not in M.printed_caption("summary_auc")
    assert "random partitions, taken together, a lower $R_{50}$ and AUC" in M.CAPTIONS["probes_full"][1]
    f3 = K.CAPTIONS["F2-ft"]
    assert "no network differs from 188 classes in R30 by more than three" in f3 and "in AUC one does" in f3
    assert not [k for k in ns["TOP_QG_FT_CLAIMS"] if k[2] != "auc"]       # Fig. 3's "in AUC one does"
    assert "differs" not in M.printed_caption("summary_auc")      # PI 2026-11-23, SC-1 (4)
    assert "no pretrained network fine-tuned here differs" in M.printed_caption("labelsets")
    checked = 0
    for boot, sub, want in ((ns["TOP_QG_BOOT"], "l3_v5", claims), (ns["TOP_QG_FT_BOOT"], "l3", ns["TOP_QG_FT_CLAIMS"])):
        files = [(task, DATA / "bootstrap" / sub / name) for task, name in boot]
        if not all(p.exists() for _t, p in files):
            continue
        got = {}
        for task, p in files:
            got.update({(task, d, m): side for (d, m), side in ns["top_qg_passes"](json.loads(p.read_text())).items()})
        assert got == want, (sub, sorted(set(got.items()) - set(want.items())), sorted(set(want.items()) - set(got.items())))
        checked += 1
    if not checked:
        pytest.skip("no every-metric bootstrap pulled (bootstrap/l3_v5 waits for the v5 gate, bootstrap/l3/ft_*.all.json)")


def test_no_sentence_claims_sculpting_at_a_working_point_within_three_sigma_of_the_floor():
    """PI 2026-11-21: no caption or sentence claims sculpting at a working point unless the distance passes 3 sigma above
    the random-selection floor (the run SD and the floor's SD combined, mlst_results.sculpt_above_floor). The audit's
    two cells, 17 classes + mass at 1% QCD efficiency and at 50% signal efficiency, are about 2 sigma above it; and a
    sentence that says a network sculpts (or distorts) the mass at a working point, without a negation, fails here."""
    sys.path.insert(0, str(ROOT / "experiments/FIGS/figstyle"))
    import gallery_captions as K
    import gallery_placement as P
    texts = list(K.CAPTIONS.values()) + [r[2] for r in P._FIGURES + P._TABLES if isinstance(r[2], str)]
    texts += [M.printed_caption(k) for k in M.CAPTIONS]
    texts += [p.read_text() for p in sorted((ROOT / "paper/journal/sections").glob("*.tex"))]
    verb = re.compile(r"\bsculpts?\b|\bsculpting by\b|\bdistorts?\b", re.I)
    wp = re.compile(r"\d+\s?\\?%|\\SculptEff(High|Low)|half (of )?the resonant|working point", re.I)
    neg = re.compile(r"\b(no|not|neither|nor|none|whether|asks)\b", re.I)
    for t in texts:
        t = re.sub(r"(?m)%.*$", "", t)
        t = re.sub(r"\\(ref|label|input|paperfig)\{[^}]*\}|\{(tab|fig|sec):[^}]*\}|\[\[[^\]]*\]\]", "", t)
        for s in re.split(r"(?<=[.;])\s+|\n\s*\n", t):
            assert not (verb.search(s) and wp.search(s) and not neg.search(s)), f"a sculpting claim: {s.strip()[:200]}"
    d = DATA / "head" / "sculpting_v3"
    if not d.exists():
        pytest.skip("head/sculpting_v3 not pulled")
    sc = {p.stem.removesuffix("_sculpting"): json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
    f = M.sculpt_above_floor(sc, ["mtx-r16q1mass-s1", "mtx-r16q1mass-s2"])
    assert not f["0.01"]["pass"] and not f["eps_s=0.5"]["pass"] and 1.0 < f["0.01"]["z"] < 3.0, f
    assert abs(f["0.01"]["distance"] - 0.043) < 0.001 and abs(f["0.01"]["floor"] - 0.031) < 0.001, f
    assert f["0.1"]["pass"], f


def test_every_fine_tuning_attempt_counts_its_hours_by_the_one_rule(tmp_path):
    """PI 2026-11-22, the compute table counts every fine-tuning GPU-hour: a pod's Prometheus running time where a
    snapshot holds its whole run (not one first seen at the snapshot's edge), else the attempt's own times; an
    interrupted attempt outside the snapshots runs to the next attempt's start and is an upper bound; an attempt with
    no start record starts when the one before it was set aside, or, the first, is counted without hours."""
    b = tmp_path / "budget"
    b.mkdir()
    a = lambda start, end, kind, pod, gpu="NVIDIA L40", init="l188-s1": {     # noqa: E731
        "start": start, "end": end, "end_kind": kind, "pod": pod, "gpu": gpu, "init": init}
    cells = {
        "fixed/leg2/l188-s1/N10000/s1": [a("2026-11-10T00:00:00Z", "2026-11-10T05:00:00Z", "next", "p-old"),   # bound, 5 h
                                         a(None, "2026-11-10T07:00:00Z", "next", None, gpu=None),               # 2 h, bound
                                         a("2026-11-10T07:00:00Z", "2026-11-10T08:15:00Z", "done", "p-done")],  # 1.25 h
        "fixed_sweep/leg_top/scratch_lr1e-3/N1000/s1": [a("2026-11-20T00:00:00Z", "2026-11-20T09:00:00Z", "next", "p-prom", init=""),
                                                        a("2026-11-20T09:00:00Z", "2026-11-20T10:00:00Z", "done", "p-edge", init="")],
        "fixed/leg_qg/l162-s2/N10000/s1": [a(None, "2026-11-18T03:00:00Z", "next", None, gpu=None, init=None),  # no hours
                                           a("2026-11-18T03:00:00Z", "2026-11-18T05:00:00Z", "done", "p-qg", init="l162-s2")]}
    (b / "ft_attempts.json").write_text(json.dumps({"root": "/data/results/ft_v2", "cells": cells}))
    pods = {"p-early": {"running_hours": 9.0, "first_running_utc": "2026-11-15T04:05Z"},
            "p-edge": {"running_hours": 0.5, "first_running_utc": "2026-11-15T04:10Z"},     # within 15 min of the edge
            "p-prom": {"running_hours": 2.25, "first_running_utc": "2026-11-20T00:05Z"}}
    (b / "ft_pod_hours_prom_20261122.json").write_text(json.dumps({"pods": pods}))
    h = M.ft_hours(tmp_path)
    assert h["pre"] == collections.Counter({"L40": 5 + 1.25 + 2, "GPU not recorded": 2})    # the qg cell's last, 2 h
    assert h["scratch"] == collections.Counter({"L40": 2.25 + 1.0})       # Prometheus for p-prom; p-edge by its times
    assert h["bound"] == 7.0 and h["n_bound"] == 2 and h["n_unknown"] == 1
    assert h["search"] == 3.25 and h["n_attempts"] == 7 and h["n_cells"] == 3
    assert h["prom_from"] == "15 November 2026"      # the snapshots' earliest sample, 15 Nov 04:05
    assert M.ft_hours(tmp_path / "nothing") is None
    t, _ = M.table_compute({}, {}, MT.fmt_int, ft_h=h)
    lines = t.splitlines()
    k = lines.index(M._full_row(5, M.FT_HOURS_HEAD))
    assert lines[k + 1].startswith(r"\quad of the pretrained networks & -- & \mbox{L40 (8)}, \mbox{GPU not recorded (2)} & 10 & --")
    assert lines[k + 2].startswith(r"\quad from scratch & -- & \mbox{L40 (3)} & 3 & --")


# ------------------------------------------------------------------ PI 2026-11-23, LEDGER SP-9: the approved structure

def _l3_entry(delta, sd_a, sd_b, boot):
    return {"quantity": "accuracy", "delta": delta, "n_a": 3, "n_b": 3, "sd_a": sd_a, "sd_b": sd_b, "boot_sd": boot,
            "sigma": math.sqrt(sd_a ** 2 + sd_b ** 2 + boot ** 2)}


def test_the_tested_differences_table_prints_each_stated_difference_with_sigma_of_runs_and_paired_bootstrap(tmp_path):
    """tab:tests (PI 2026-11-23, LEDGER SP-9, merge 6; PI SC-1 (1)): one row per stated difference, Delta, each side's run
    SD, the paired test-jet bootstrap SD and Delta/sigma with sigma^2 = sd_a^2 + sd_b^2 + boot^2, Delta at sigma's
    place and each SD at its own; \\pending{bootstrap} where no bootstrap is stored, \\pending{rerun bootstrap} where the file's runs or values
    are not the ones Table 1 prints (a fine-tuning run since rerun, SP-11); a sigma that is not the quadrature sum
    stops the build. PI 2026-11-23 (LEDGER SP-13, step 2): Delta, sigma and Delta/sigma only, each side's mean and SD
    being in the value tables."""
    runs = M._paper(ROOT)
    acc = {m: 0.84 + 0.001 * k for k, m in enumerate(runs["L188"] + runs["R16_Q1"])}
    per = {"jc1": {m[len("mtx-"):]: {1_000_000: [{"accuracy": acc[m]}]} for m in acc}}
    doc = {"b": 200, "models": {m: {"accuracy": {"value": v, "sd": 1e-4}} for m, v in acc.items()},
           "diffs": {"17-188": {"a": runs["R16_Q1"], "b": runs["L188"], "accuracy": _l3_entry(-0.00462, 0.00054, 0.00058, 0.00011)}}}
    (tmp_path / "bootstrap" / "l3").mkdir(parents=True)
    (tmp_path / "bootstrap" / "l3" / "ft_jc1_N1000000.json").write_text(json.dumps(doc))
    body, reps = M.table_tests(tmp_path, runs, {}, {}, {}, per)
    assert reps == {200} and body.startswith(r"\begin{tabular}{lccc}") and M.TESTS_HEAD in body
    lines = body.splitlines()
    head = M._full_row(4, "JetClass, accuracy, " + M.HEADING["ft"], italic=True)
    i = lines.index(head + "*")
    rows = lines[i + 1:i + 1 + 7]
    name = lambda a, b: f"{M.LABEL[a]} $-$ {M.LABEL[b]}"         # noqa: E731
    # delta and sigma at sigma's place (PDG: 0.00080 -> 0.0008, four decimals)
    sig = _l3_entry(-0.00462, 0.00054, 0.00058, 0.00011)["sigma"]
    want = name("R16_Q1", "L188") + " & $-$0.0046 & 0.0008 & " + f"{-0.00462 / sig:.2f}".replace("-", "$-$")
    assert any(r.rstrip("*").rstrip().removesuffix(r"\\").rstrip() == want for r in rows), rows
    # a row with no stored bootstrap, and one whose file does not hold the paper's values
    assert any(r.startswith(name("L162", "L188") + r" & \multicolumn{3}{c}{\pending{bootstrap}}") for r in rows)
    stale = {**per, "jc1": {**per["jc1"], runs["L188"][0][len("mtx-"):]: {1_000_000: [{"accuracy": 0.9}]}}}
    assert name("R16_Q1", "L188") + r" & \multicolumn{3}{c}{\pending{rerun bootstrap}}" in M.table_tests(
        tmp_path, runs, {}, {}, {}, stale)[0]
    doc["diffs"]["17-188"]["accuracy"]["sigma"] *= 1.1
    (tmp_path / "bootstrap" / "l3" / "ft_jc1_N1000000.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="quadrature"):
        M.table_tests(tmp_path, runs, {}, {}, {}, per)
    # every row names two networks or references the table can print, and every source is of a kind the builder reads
    for heading, rows_ in M.TESTS:
        for first, second, src in rows_:
            assert all(x in M.TESTS_NAMES or x in M.LABEL for x in (first, second)), (heading, first, second)
            assert src is None or src == M.RERUN or src[0] in ("l3", "file", "mass", "mass_group"), (heading, src)
    assert M.SUMMARY_L3[("ft", "jc1")] == "ft_jc1_N1000000.json"


def test_the_tested_differences_table_reads_the_stored_sculpting_and_mass_bootstraps():
    """tab:tests on the files in the repository: every sculpting difference at its three cuts (the l3 block, sides
    over every run) and the two MLP mass-resolution differences, each sigma the quadrature sum (SC-1 (1)); the
    placement puts it in the appendix, tab:sculptboot dropped (PI 2026-11-23, LEDGER SP-9)."""
    sb = DATA / (M.SCULPT_BOOT_L3_RERUN if (DATA / M.SCULPT_BOOT_L3_RERUN).exists() else M.SCULPT_BOOT_L3)
    mass = DATA / M.MASS_SD_BOOT_TESTS
    if not sb.exists() or not mass.exists():
        pytest.skip("the sculpting or mass bootstrap is not pulled")
    sdoc, mdoc = json.loads(sb.read_text()), json.loads(mass.read_text())
    body, reps = M.table_tests(DATA, M._paper(ROOT), {}, {}, {}, {}, sdoc, mdoc)
    assert reps == {sdoc["replicas"]} | {mdoc["boot"]}
    heads = [ln for ln in body.splitlines() if ln.startswith(r"\multicolumn{4}{@{}l}{\textit{")]
    assert sum(M.TESTS_SCULPT in h for h in heads) == len(M.SCULPT_BOOT_WPS)
    # PI 2026-11-23 (LEDGER SP-13): no mass network and no random partition against 188 classes (no sentence states them)
    unstated = {("l162mass", "l188"), ("r16q1mass", "l188"), ("rand2p", "l188")}
    stated = [d for d in sdoc["differences"].values() if (d["configuration"], d["reference"]) not in unstated]
    for wp, h in M.SCULPT_BOOT_WPS:
        block = body.split(M.TESTS_SCULPT + ", " + h)[1].split(r"\addlinespace")[0]
        assert block.count(r" $-$ ") == len(stated), wp
    for a, b in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")):
        row = next(ln for ln in body.split("jet mass, resolution, " + M.HEADING["mlp"])[1].splitlines()
                   if ln.startswith(f"{M.LABEL[a]} $-$ {M.LABEL[b]} &"))
        assert r"\pending" not in row, row


def test_the_tested_differences_table_keeps_the_stated_rows_and_reads_every_stored_bootstrap():
    """PI 2026-11-23 (LEDGER SP-13): tab:tests keeps a row only if a main-text sentence or caption states it, it backs a
    method check the text relies on, or it is a 3-sigma-passing label-granularity result (one still waiting for its
    bootstrap stays until it lands); the others left (their values stay in the data release), the from-scratch
    comparison with ParT's from-scratch value that
    results_bench states came in. On the repository's files every row with a stored bootstrap prints it (the probes,
    the class pairs, the 17 groups, the step, the substitutions, the full top and quark/gluon sets, the mass rows), the
    rows of the fine-tuning reruns print \\pending{rerun bootstrap}, and only the two rows scored on independent samples
    (Herwig against Pythia jets, PI SC-1a) carry the mark."""
    rows = {(h, a, b): src for h, rs in M.TESTS for a, b, src in rs}
    lp, ft = M.HEADING["lp"], M.HEADING["ft"]
    # SP-13 (c): the label-granularity rows that pass 3 sigma stay, and those still waiting for their bootstrap
    kept = [(f"JetClass-II, accuracy, {lp}", a, "L188") for a in M.FLAV + (M.RAND_POOLED,)]
    kept += [(f"JetClass-II, accuracy, {ft}", a, "L188") for a in ("L162", "R42_Q1", "R16_Q1")]
    kept += [(f"JetClass-II, accuracy, {lp}, " + M._power(f"{n:g}") + " training jets", a, "L188")
             for n in (1000, 10000, 100000) for a in ("R42_Q1", "R16_Q1")]
    kept += [(rf"{M._PAIR['bc_vs_bq_cs']}, AUC, {lp}", "FLAV_F0", "L188"),
             (rf"{M._PAIR['bc_vs_bq_cs']}, AUC, " + M.HEADING["mlp"], "R16_Q1", "L188")]
    kept += [(rf"{h}, signal efficiency (\%), classifier on the frozen features", a, "L188") for _, h in M.SUMMARY_XBX
             for a in M.FLAV]
    assert not [r for r in kept if r not in rows], [r for r in kept if r not in rows]
    gone = [(f"JetClass, accuracy, {lp}", a, "L188") for a in ("L162", "R42_Q1", "R16_Q1")]
    gone += [(f"JetClass-II, accuracy, {ft}, " + M._power("1000") + " training jets", "R16_Q1", "L188")]
    gone += [(f"top tagging, $R_{{30}}$, {ft}, published models", "L188", "PART_PUB"),
             (f"quark/gluon tagging (Pythia), $R_{{30}}$, {ft}, published models", "L188", "PART_PUB"),
             (f"top tagging, $R_{{30}}$, {lp}, missing inputs", "L188", M.INIT),
             (f"quark/gluon tagging (Pythia), $R_{{30}}$, {lp}, missing inputs", "L188", M.INIT),
             (r"$X\to b\bar b$, signal efficiency (\%), classifier on the frozen features", "L188", "SCRATCH"),
             (r"$X\to bs$, signal efficiency (\%), classifier on the frozen features", "RAND", "R16_Q1")]
    assert not [r for r in gone if r in rows], [r for r in gone if r in rows]
    assert not any(h.startswith("JetClass, accuracy, " + ft + ", published") for h, *_ in rows)
    assert rows[(f"top tagging, $R_{{30}}$, {ft}, published models", "SCRATCH", "PART_PUB")] is None
    for task in ("top", "qg"):                 # a mass network against its own label set: fine-tuned top tagging only
        for mode in ("lp", "ft"):
            head = ("top tagging" if task == "top" else "quark/gluon tagging (Pythia)") + ", $R_{30}$, " + M.HEADING[mode]
            assert ((head, "L162_MASS", "L162") in rows) == (task == "top" and mode == "ft"), head
            assert rows[(head, "L162_MASS", "L188")] == ("l3", mode, task, "r30", "l162mass-188", True)
    if not (DATA / "bootstrap" / M._JC2_ALL).exists():
        pytest.skip("the tab:tests bootstraps are not pulled")
    runs = M._paper(ROOT)
    fits, _ = M.V4.load_fits(DATA)
    fill = M.with_sophon_fill(DATA, M._load(DATA / M.V4.pick(DATA, M.FITS_FILL)))
    chosen, _ = M.chosen_substitutions(fill, runs)
    co = M._load(DATA / M.V4.pick(DATA, "linprobe/fits_v4_coarse"))
    sb = DATA / (M.SCULPT_BOOT_L3_RERUN if (DATA / M.SCULPT_BOOT_L3_RERUN).exists() else M.SCULPT_BOOT_L3)
    got = {}
    body, reps = M.table_tests(DATA, runs, fits, fill, chosen, M.load_ft(DATA)[1], json.loads(sb.read_text()),
                               json.loads((DATA / M.MASS_SD_BOOT_TESTS).read_text()), rows_out=got, co=co)
    assert reps == {200}
    stored = [k for k, src in rows.items() if src not in (None, M.RERUN) and (src[0] != "l3" or src[1] == "lp")]
    assert len(stored) > 60 and not [k for k in stored if not isinstance(got[k], tuple)], \
        [k for k in stored if not isinstance(got[k], tuple)]
    for k in stored:
        d, sd_a, sd_b, boot, sig = got[k]
        assert sig == pytest.approx(math.sqrt(sd_a ** 2 + sd_b ** 2 + boot ** 2), rel=1e-9), k
    assert [k for k, src in rows.items() if src == M.RERUN] and all(got[k] == "stale" for k, src in rows.items()
                                                                     if src == M.RERUN)
    marked = [ln.split(" & ")[0] for ln in body.splitlines() if M.INDEPENDENT in ln]
    assert marked == [M.TESTS_NAMES[a] + " $-$ " + M.TESTS_NAMES[b] + M.INDEPENDENT
                      for a, b in (("L188_HERWIG_JETS", "L188_PYTHIA_JETS"), ("SCRATCH_HERWIG_JETS", "SCRATCH_PYTHIA_JETS"))]
    # one line a row: the label, Delta, sigma and Delta/sigma, or the label and the marker across the three
    for ln in body.splitlines():
        if ln.startswith((r"\begin", r"\end", r"\toprule", r"\midrule", r"\bottomrule", r"\noalign", r"\multicolumn{4}")):
            continue
        assert ln.rstrip("*").endswith(r"\\") and (ln.count(" & ") == 3 or r"\multicolumn{3}{c}{\pending{" in ln), ln


def test_a_bootstrap_file_row_stops_the_build_where_its_runs_or_values_are_not_the_papers(tmp_path):
    """PI 2026-11-23 (LEDGER SP-13; RR-7, a rule checked where the data is read): a "file" row of tab:tests prints its
    stored difference (an AUC minus that of 1 - AUC) only if each side holds the paper's runs and each run's value lies
    within a quarter of its bootstrap SD of the value the paper prints (here the class-pair AUC of the figures and
    ratio words); otherwise the build stops. Two independent samples (PI SC-1a): the bootstrap SD is the two samples'
    in quadrature or the build stops, and the row carries the mark."""
    runs = M._paper(ROOT)
    head, (first, second, src) = next((h, r) for h, rs in M.TESTS for r in rs if r[2] and r[2][0] == "file"
                                      and "_pairs_bb_vs_cc" in r[2][1] and r[:2] == ("R42_Q1", "L188"))
    _, path, diff, met = src
    oma = {m: 0.002 + 0.0001 * k for k, m in enumerate(runs["R42_Q1"] + runs["L188"])}
    fits = {m: {"cells": {M._ck(m): {M.PAIR_DS: {M.PAIR_N: {"bb_vs_cc": {"auc": 1 - v}}}}}} for m, v in oma.items()}
    entry = {"quantity": "one_minus_auc", "delta": 0.0003, "sd_a": 0.0001, "sd_b": 0.0001, "boot_sd": 0.00002}
    entry["sigma"] = math.sqrt(0.0001 ** 2 * 2 + 0.00002 ** 2)
    doc = {"b": 200, "models": {m: {met: {"one_minus_auc": v, "sd": 1e-5}} for m, v in oma.items()},
           "diffs": {diff: {"a": runs["R42_Q1"], "b": runs["L188"], met: entry}}}
    f = tmp_path / "bootstrap" / path
    f.parent.mkdir(parents=True)

    def table(fits_, doc_):
        f.write_text(json.dumps(doc_))
        got = {}
        return M.table_tests(tmp_path, runs, fits_, {}, {}, {}, rows_out=got)[0], got[(head, first, second)]
    body, v = table(fits, doc)
    assert v[0] == pytest.approx(-0.0003) and M.INDEPENDENT not in body
    far = {**fits, runs["L188"][0]: {"cells": {"best70": {M.PAIR_DS: {M.PAIR_N: {"bb_vs_cc": {"auc": 0.9}}}}}}}
    with pytest.raises(SystemExit, match="quarter of its bootstrap SD"):
        table(far, doc)
    other = json.loads(json.dumps(doc))
    other["diffs"][diff]["b"] = runs["L162"]
    with pytest.raises(SystemExit, match="not the paper's"):
        table(fits, other)
    unpaired = json.loads(json.dumps(doc))
    unpaired["diffs"][diff][met].update(resampling="unpaired", boot_sd_a=0.00001, boot_sd_b=0.00001)
    with pytest.raises(SystemExit, match="SC-1a"):
        table(fits, unpaired)
    unpaired["diffs"][diff][met].update(boot_sd_a=0.000012, boot_sd_b=0.000016)
    body, v = table(fits, unpaired)
    assert M.INDEPENDENT in body and v[4] == pytest.approx(entry["sigma"])


def test_the_auc_table_leaves_out_top_and_quark_gluon_and_no_text_quotes_them(tmp_path):
    """PI 2026-11-23 (LEDGER SP-13, step 4): the top and quark/gluon AUC columns leave tab:summaryauc on the condition
    that no main-text sentence quotes an AUC value of them; the build checks it where it reads the text
    (dropped_auc_check), so a macro printing such a value in main.tex or a section stops it, a commented one does not."""
    assert not {k for k, *_ in M.SUMMARY_AUC_TABLE} & set(M.SUMMARY_AUC_DROPPED)
    assert set(M.SUMMARY_AUC_DROPPED) == {k for k, *_ in M.SUMMARY_AUC_GRID} - {"jc2", "jc1"}
    # their means stay in the supplement's full copy (every column), which the data release writes with every table
    assert M.SUMMARY_AUC_TABLE == tuple(g for g in M.SUMMARY_AUC_FULL if g[0] in ("jc2", "jc1"))
    assert set(M.SUMMARY_AUC_GRID) <= set(M.SUMMARY_AUC_FULL) and len(M.SUMMARY_AUC_FULL) == 7
    assert M.CAPTIONS["summary_auc_full"][0] == "tab:summaryaucfull"
    assert ("summary_auc_full", "mlst_summary_auc_full", True, None) in M.PLACEHOLDERS
    assert "tables/mlst_summary_auc_full.tex" in inspect.getsource(M.build)
    G = _load("gallery_placement_aucfull_t", "experiments/FIGS/figstyle/gallery_placement.py")
    assert G.TIER["tab:summaryaucfull"] == G.SUPPLEMENT and G.TIER["tab:summaryauc"] == G.APPENDIX
    cap = M.printed_caption("summary_auc_full")
    assert "top tagging" in cap and "Herwig" in cap and ": " not in cap and not re.search(r"\bSD\b", cap)
    M.dropped_auc_check(ROOT / "paper/journal")
    assert "dropped_auc_check(root" in inspect.getsource(M.build)
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text("\\input{sections/a}\n")
    (tmp_path / "sections" / "a.tex").write_text("AUC % \\MlstFtTopAucFull{} in a comment\n")
    M.dropped_auc_check(tmp_path)
    for name in ("MlstFtTopAucFull", "MlstFtQgAucScratch", "MlstFtQgHerwigAucFull", "MlstLpQgAucFull"):
        (tmp_path / "sections" / "a.tex").write_text(f"an AUC of \\{name}{{}}.\n")
        with pytest.raises(SystemExit, match="tab:summaryauc"):
            M.dropped_auc_check(tmp_path)
    (tmp_path / "sections" / "a.tex").write_text("\\MlstFtJcOneAucFull{} and \\MlstLpEemmAucFull{}\n")
    M.dropped_auc_check(tmp_path)


def test_table_one_takes_the_transfer_head_columns_of_the_resonance_table():
    """PI 2026-11-23, LEDGER SP-9 (merge 4): Table 1 gains tab:xbx's transfer-head columns, X->bb and X->bs signal
    efficiency in percent at the one QCD efficiency of the fits, under one heading naming it; the pretrained networks in
    the linear-probe part (a dash for the random network, which has no head), from scratch in the fine-tuned part, a
    dash in every other fine-tuned cell; one QCD efficiency or the build stops; the values are xbx_report's own."""
    XR = _load("xbx_report_t1", "experiments/XBS/xbx_report.py")
    runs = M._paper(ROOT)
    head = lambda e: {"head": [{"seed": 1, "n_s": 10, "n_b": 10, "auc": 0.9, "eps_b_star": 0.01, "eps_s_at_star": e,  # noqa: E731
                                "curve": {"eps_b": [0.01], "eps_s_over_sqrt_eps_b": [1.0]}}]}
    fits = {m: {"xbb": head(0.70 + 0.001 * k), "xbs": head(0.60 + 0.001 * k)}
            for k, m in enumerate(m for a in M.SETS + M.FLAV + M.RAND for m in runs[a])}
    scr = {"xbb": [{"eps_s_at_star": 0.57, "eps_b_star": 0.01}, {"eps_s_at_star": 0.58, "eps_b_star": 0.01}],
           "xbs": [{"eps_s_at_star": 0.55, "eps_b_star": 0.01}, {"eps_s_at_star": 0.56, "eps_b_star": 0.01}]}
    xbx = {"XR": XR, "fits": fits, "scr": scr}
    assert M.summary_xbx(xbx)["star"] == 0.01 and M.summary_xbx({}) == {}
    lp, ft = M.table_summary({}, {}, {}, {}, runs, M.TAB, M.SUMMARY_GRID, "summary", xbx=xbx).split("%SPLIT%")
    for part in (lp, ft):
        lines = part.splitlines()
        assert lines[2].endswith(r"\multicolumn{2}{c}{$\varepsilon_S$ (\%) at $\varepsilon_B = 1\%$} \\")
        assert lines[3] == r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}" and lines[4].endswith(r"& $X\to b\bar b$ & $X\to bs$ \\")
        assert all(ln.count(" & ") == 7 for ln in lines if " & " in ln and not ln.startswith(" & "))
    row = lambda part, name: next(ln for ln in part.splitlines() if ln.startswith(name + " & "))   # noqa: E731
    cells = lambda ln: [c.strip() for c in ln.removesuffix(r"\\").split(" & ")]                 # noqa: E731
    want = XR.efficiencies(M, runs, fits, "xbb", "head")["L188"]
    got = cells(row(lp, M.LABEL["L188"]))[6]
    mean = float(re.sub(r"\{\\fontseries\{b\}\\selectfont |\}", "", got).split(r"\,$\pm$\,")[0])
    assert mean == pytest.approx(100 * statistics.mean(want), abs=0.051)
    assert cells(row(lp, M.LABEL[M.INIT]))[6:] == ["--", "--"]
    assert cells(row(ft, M.LABEL["L188"]))[6:] == ["--", "--"] and "--" not in cells(row(ft, M.SCRATCH_LABEL))[6:]
    assert all(c == "--" for c in cells(row(ft, M.SCRATCH_LABEL))[1:6])          # no headline cell without per
    two = {**xbx, "scr": {"xbb": [{"eps_s_at_star": 0.5, "eps_b_star": 0.001}]}}
    with pytest.raises(SystemExit, match="QCD efficiencies"):
        M.summary_xbx(two)
    # the AUC table gains JetClass-II's resonant and QCD columns under its one heading (merge 5)
    assert M.SUMMARY_AUC_TABLE[:3] == (M.SUMMARY_AUC_GRID[0], ("jc2", 1_000_000, "jc2_188", "macro_auc_resonant"),
                                       ("jc2", 1_000_000, "jc2_188", "macro_auc_qcd"))
    auc = M.table_summary({}, {}, {}, {}, runs, M.TAB, M.SUMMARY_AUC_TABLE, "summary_auc").split("%SPLIT%")[0].splitlines()
    # PI 2026-11-23 (LEDGER SP-13, step 4): JetClass-II and JetClass only, the top and quark/gluon AUC columns out
    assert M.SUMMARY_AUC_TABLE[3:] == (M.SUMMARY_AUC_GRID[1],)
    assert auc[2] == r" & \multicolumn{3}{c}{JetClass-II} & JetClass \\"
    assert auc[3] == r"\cmidrule(lr){2-4}" and auc[4] == r" & AUC & AUC, resonant & AUC, QCD & AUC \\"
