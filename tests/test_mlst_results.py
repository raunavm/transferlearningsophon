"""The MLST tables' code (experiments/FIGS/mlst_results.py, read by make_tables.build()):
mean +- sample SD over the runs present with a marker for missing runs, a lone run with its
trailing zeros, the probe convergence rule, the merge marks from the label maps, and the
pulled data checked against its index."""
import csv
import importlib.util
import json
import pathlib
import math
import re
import shutil
import statistics

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
        "ckpt_identity"}


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
    # one block per task, and the JetClass-II accuracy beside its AUC (2026-11-07)
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS) and r"\pending{2 of 3}" in t and M.SCRATCH_LABEL in t
    assert r"\begin{longtable}" in t and r"\label{tab:ft}" in t and "%SPLIT%" not in t
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
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS)
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
    assert r"ParT~\cite{part} & 2 M & 2.14\,M & 0.836 & 0.9834" in tex and "& 100 M &" not in tex
    assert r"\textbf" not in tex and M.BOLD_ON not in tex and "jetclass" not in M.BOLD
    scratch = next(line for line in tex.splitlines() if line.startswith(M.SCRATCH_LABEL))
    assert MT.fmt_pm(0.802, 0.001) in scratch and " & 1 M & 2.21\\,M & " in scratch       # ten outputs: 2,213,374
    assert r"$R_{99}$" in tex and r"$R_{99.5}$" in tex
    assert M.LABEL["R16_Q1"] not in tex          # no curves for it: no row
    # the one row order (PI 2026-11-12): the published rows, a rule, ours with from scratch last; no "this work" row
    # (the italic "fine-tuned" heading is _float's, MODES); the headers in words
    body = tex.splitlines()
    assert body[-3] == scratch and body[-4].startswith(M.LABEL["L188"] + " & ")
    assert body[-5] == r"\midrule" and r"\cite{" in body[-6] and "this work" not in tex
    assert rf" & training jets & {V4.PARAMS_HEAD} & accuracy & AUC & " in tex and V4.PARAMS_HEAD == "parameters"
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
    # within twice the SD of 1: no direction, the bound instead
    assert M.prose([1.0, 1.006, 0.994], "errors") == r"equal within 1\%"
    assert M.prose([None, None], "errors").startswith(r"\pending")


def test_two_runs_give_the_value_nearer_one_and_its_twin_the_other():
    assert M.prose([1.041, 1.032], "errors") == r"at least 3.2\% more errors"
    assert M.prose([1.041, 1.032], "errors", farther=True) == r"up to 4.1\% more errors"
    assert M.prose([0.35, 0.24], "errors") == "at most 0.35 times the errors"
    assert M.prose([0.35, 0.24], "errors", farther=True) == "down to 0.24 times the errors"
    assert M.prose([0.98, 1.03], "errors") == r"equal within 3\%"
    em = _PEm()
    M._emit_ratio(em, "MlstX", [1.041, 1.032], 2, MT.fmt_pm, None, "p", "errors", two_run=True)
    assert set(em.m) == {"MlstX", "MlstXPr", "MlstXPrMax"} and em.m["MlstX"] == "1.041 and 1.032"
    M._emit_ratio(em, "MlstY", [1.15, 1.2, 1.25], 3, MT.fmt_pm, None, "p", "errors")
    assert "MlstYPrMax" not in em.m and em.m["MlstYPr"] == r"20\% more errors"


def test_two_runs_state_a_direction_only_beyond_twice_the_hierarchy_sd():
    # Methods: both runs must lie further from 1 than twice the SD of the same ratio between levels
    assert M.prose([1.04, 1.08], "rejection", ref_sd=0.03) == r"equal within 8\%"
    assert M.prose([1.04, 1.08], "rejection", farther=True, ref_sd=0.03) == r"equal within 8\%"
    assert M.prose([1.04, 1.08], "rejection", ref_sd=0.01) == r"at least 4.0\% more rejection"
    assert M.prose([0.90, 0.92], "rejection", ref_sd=0.05) == r"equal within 10\%"
    assert M.prose([1.04, 1.08], "rejection", ref_sd=None) == r"at least 4.0\% more rejection"
    assert M.prose([1.15, 1.2, 1.25], "errors", ref_sd=0.2) == r"20\% more errors"     # three runs: their own SD
    em = _PEm()
    M._emit_ratio(em, "MlstX", [1.04, 1.08], 2, MT.fmt_pm, None, "p", "rejection", two_run=True, ref_sd=0.03)
    assert em.m["MlstXPr"] == em.m["MlstXPrMax"] == r"equal within 8\%" and em.m["MlstX"] == "1.040 and 1.080"


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


def test_the_hierarchy_sd_pools_the_levels_over_the_188_class_model_run_by_run():
    runs = {"L188": ["a-s1", "a-s2", "a-s3"], "L162": ["b-s1", "b-s2", "b-s3"], "R42_Q1": ["c-s1", "c-s2"],
            "R16_Q1": ["d-s1"]}
    val = {"a-s1": 1.0, "a-s2": 2.0, "a-s3": 1.0, "b-s1": 1.1, "b-s2": 2.0, "b-s3": 0.9,
           "c-s1": 1.2, "c-s2": 2.4, "d-s1": 5.0}
    # in the logarithm: 162/188 ln 1.1, 0, ln 0.9; 42/188 ln 1.2 twice (variance 0); 16/188 one run, left out
    assert M._level_sd(runs, val.get) == pytest.approx(math.sqrt(statistics.variance([math.log(1.1), 0, math.log(0.9)]) / 2))
    assert M._level_sd(runs, lambda m: 10 * val[m] if m.startswith("b") else val[m]) == pytest.approx(M._level_sd(runs, val.get))
    assert M._level_sd({"L188": ["a-s1"], "L162": ["b-s1"], "R42_Q1": [], "R16_Q1": []}, val.get) is None


def test_a_macro_the_outline_names_is_a_marker_until_its_input_exists():
    em, missing = _PEm(), []
    em.macro("MlstGpuTypeBound", "0.2\\%")
    M.required_pass(em, None, missing)
    assert em.m["MlstGpuTypeBound"] == "0.2\\%"
    assert em.m["MlstFtTopRthirtyFullOverScratch"].startswith(r"\pending")
    assert em.m["MlstFtTopRthirtyFullOverScratchPr"].startswith(r"\pending")
    assert em.m["MlstMassHeadResMassOverUncorrPrMax"].startswith(r"\pending")
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
    assert M._subst_head({"top": "mean"}, "top") == "missing inputs: the training mean"
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
    assert lines.index("soft-drop mass & 0.200 & 1.05 \\\\") < head < mlp and lines[head - 1] == lines[mlp - 1] == r"\midrule"
    assert [ln.split(" & ")[0] for ln in lines[head + 1:mlp - 1]] == [M.LABEL[a] for a in M.MASS]
    assert lines[head + 1] == M.LABEL["L162_MASS"] + r" & 0.1170\,$\pm$\,0.0014 & 0.995 \\"
    assert [ln.split(" & ")[0] for ln in lines[mlp + 1:-2]] == [M.LABEL[a] for a in M.MASS_ROWS]
    assert M.MASS_ROWS == M.MAIN + M.MASS + (M.INIT,) and "mass" not in M.MODES
    assert M.LABEL["L162_MASS"] + " & 0.150 & 1.00 \\\\" in lines[mlp:]        # equal runs: fmt_pm's common value
    em = _PEm()
    M.mass_macros(em, atlas, runs, None, MT.fmt_pm)
    assert em.m["MlstMassHeadResMassOverUncorr"] == "0.580 and 0.590"
    assert em.m["MlstMassHeadResMassOverUncorrPr"] == "at most 0.59 times the resolution"
    assert em.m["MlstMassHeadResMassOverUncorrPrMax"] == "down to 0.58 times the resolution"
    assert em.m["MlstMassHeadResMassOverMlpRes"].startswith("0.580 and 0.590")
    assert em.m["MlstMassUncorrected"] == "0.200" and em.m["MlstMassUncorrectedResonant"] == "0.300"
    assert em.m["MlstMassUncorrectedQcd"] == "0.300" and em.m["MlstMassHeadResMassResponse"] == "0.995 and 0.995"
    assert em.m["MlstMassHeadResMassOverUncorrResonant"] == "0.3867 and 0.3933"      # 0.116 / 0.3, 0.118 / 0.3
    d = M.table_mass_detail(atlas, runs, MT.fmt_pm)
    assert "200--400 & 400--2500" in d and r"\textit{QCD jets}" in d and "\nsoft-drop mass & " in d
    assert "jets & all & resonant & QCD & all & resonant & QCD" in d
    # the supplement's detail table names its estimators as the main-text table's headings do
    assert all(rf"\textit{{{M.HEADING[k]}}}" in d for k in ("head", "mlp")) and "regression output of the pretrained network" not in d
    assert d.index(M.HEADING["head"]) < d.index("ridge regression on the frozen features") < d.index(M.HEADING["mlp"])
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
    assert em.m["MlstSculptCoarseOverFullOnePr"] == "2.0 times the distance"
    assert em.m["MlstSculptFullOne"].startswith("0.020")
    assert "MlstSculptCoarseMassOverCoarseOnePrMax" in em.m
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


def test_a_jetclass_rejection_counts_as_a_drop_only_beyond_twice_its_spread_and_its_counting_error():
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
    assert em.m["MlstFtJcOneCoarseNRejDrop"] == "one" and em.m["MlstFtJcOneCoarseRejDropClasses"] == r"$H\to 4q$"
    assert em.m["MlstFtJcOneRejHfourqCoarseOverFullPr"] == r"15\% less rejection"
    assert em.m["MlstFtJcOneNSignal"] == "nine" and em.m["MlstFtJcOneAccRandMinusCoarse"].startswith(r"\pending")
    r, d, drop = M._rej_diff(_jc1_pre(runs, rej), runs, "R16_Q1", "L188", "label_Tbqq")
    assert len(r) == 3 and not d and not drop


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
    assert "LorentzNet" not in t and "OmniLearned-s" not in t and r"ParT-f.t.~\cite{part}" in t
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
    assert M.LABEL["KIN"] == r"\Kfull{} classes, kinematic inputs"


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
    assert em.m["MlstPartreproRthirtyRescaleOverPartPr"] == r"28\% less rejection"
    assert not missing
    # three runs of ParT's script, each a single run (marked) and a single change, not cumulative; ParT's published
    # row and our from-scratch row are in the comparison table only (PI 2026-11-12: one table per number)
    rows = t.splitlines()[4:-2]
    assert [ln.split(" & ")[0] for ln in rows] == [name for _, name in M.PARTREPRO] == [
        "ParT's training script, unchanged", "ParT's script with Sophon's output head",
        r"ParT's script with Sophon's jet-$p_{\mathrm{T}}$ rescaling"]
    assert "ParT, published" not in t and M.SCRATCH_LABEL not in t and "+" not in t
    assert rows[0] == (r"ParT's training script, unchanged & 0.9400$^{\S}$ & 0.9858$^{\S}$ & 420$^{\S}$ & 1{,}711$^{\S}$"
                       r" & 118$^{\S}$ \\")
    assert all(ln.count(M.SINGLE) == 5 for ln in rows) and r"1{,}224$^{\S}$" in rows[2]
    # with the fine-tuning read-out, each column at the place of the top-tagging column it is read against (from
    # scratch 1,400 +- 200: hundreds; 420 +- 20: tens), the macros unchanged
    rec = lambda acc, r50, r30: {"accuracy": acc, "auc": 0.9858, "r50": r50, "r30": r30}      # noqa: E731
    per = {"top": {"l188-s1": {1200000: [rec(0.94, 460.0, 1800.0)]},
                   M.SCRATCH: {1200000: [rec(0.939, 400.0, 1200.0), rec(0.940, 420.0, 1400.0), rec(0.941, 440.0, 1600.0)]}}}
    em2 = _PEm()
    t = M.partrepro(em2, d, per, MT.fmt_pm, [], runs={"L188": ["mtx-l188-s1"], "__bench__": ("L188",)})
    assert [ln.split(" & ")[3:5] for ln in t.splitlines()[4:-2]] == [
        [f"420{M.SINGLE}", f"1{{,}}700{M.SINGLE}"], [f"420{M.SINGLE}", f"1{{,}}600{M.SINGLE}"],
        [f"420{M.SINGLE}", f"1{{,}}200{M.SINGLE}"]]
    assert em2.m == em.m


def test_the_learning_rate_table_marks_the_chosen_rates():
    cells = {}
    for task, base in (("leg_top", 0.92), ("leg_qg", 0.82)):
        for lr, d in (("1e-4", 0.0), ("3e-4", -0.001), ("1e-3", 0.001)):
            cells[f"{task}/l188-s1_lr{lr}/N10000/s1"] = {"early_stop": {"best_metric": base + d}}
        for lr, d in (("3e-4", -0.002), ("1e-3", 0.0), ("3e-3", 0.001)):
            cells[f"{task}/scratch_lr{lr}/N10000/s1"] = {"early_stop": {"best_metric": base - 0.01 + d}}
    t = M.table_lrsweep({"cells": cells})
    chosen = [ln for ln in t.splitlines() if r"$^{\ast}$" in ln]
    assert len(chosen) == 2 and r"\ensuremath{10^{-3}}" in chosen[0] and r"\ensuremath{3\times10^{-3}}" in chosen[1]
    # from scratch each task keeps its own best rate, starred on that task's accuracy, never on the sum
    cells["leg_top/scratch_lr1e-3/N10000/s1"]["early_stop"]["best_metric"] = 0.92
    assert M.V4.choose_rates({"cells": cells}) == ("1e-3", {"leg_qg": "3e-3", "leg_top": "1e-3"})
    rows = M.table_lrsweep({"cells": cells}).splitlines()
    assert [ln.count(r"$^{\ast}$") for ln in rows if ln.startswith(M.SCRATCH_LABEL)] == [0, 1, 1]
    assert next(ln for ln in rows if ln.startswith(M.SCRATCH_LABEL + r" & \ensuremath{10^{-3}}")).split(" & ")[2].endswith(r"$^{\ast}$")
    assert not any(r"$^{\ast}$" in ln.split(" & ")[-1] for ln in rows if ln.startswith(M.SCRATCH_LABEL))
    sums = [ln for ln in rows if ln.startswith((M.SCRATCH_LABEL, M.LABEL["L188"]))]
    # the sum: four decimals; every value is one run's and carries the single-run mark, before the star
    assert len(sums) == 6 and all(re.search(r"& \d\.\d{4}\$\^\{\\S\}\$(\$\^\{\\ast\}\$)? \\\\$", ln) for ln in sums)
    assert all(ln.count(M.SINGLE) == 3 for ln in sums) and M.SINGLE + r"$^{\ast}$" in "".join(sums)
    # three rows fine-tuned from 188 classes (the label alone: the caption says seed 1), then three from scratch
    assert [ln.split(" & ")[0] for ln in sums] == [M.LABEL["L188"]] * 3 + [M.SCRATCH_LABEL] * 3


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
    assert len(blocks) == 4 and "natural-composition" in blocks[3]
    assert next(ln for ln in blocks[0].splitlines() if ln.startswith(M.LABEL["L162_MASS"])).split(" & ")[1:5] == ["--"] * 4
    assert not any(ln.startswith(M.LABEL["L162_MASS"]) for b in blocks[1:3] for ln in b.splitlines())
    assert next(ln for ln in blocks[3].splitlines() if ln.startswith(M.LABEL["RAND2_p1"])).split(" & ")[2] == M._lone(1.25)
    # no configuration lacks a balanced fit: no natural-composition block, whatever nat holds
    assert "natural-composition" not in M.table_mlp(nat, runs, MT.fmt_pm, rows, nat=nat)
    em = _PEm()
    M.mlp_natural_macros(em, nat, runs, None, MT.fmt_pm)
    assert em.m["MlstMlpNatJcTwoResMassOverRes"] == M._cell([0.84, 0.84], 2, MT.fmt_pm, sep=M.TEXT_SEP)
    assert "MlstMlpNatJcTwoResMassOverResPr" in em.m and "MlstMlpNatJcTwoCoarseMassOverCoarsePrMax" in em.m
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
    assert em.m["MlstFtTopRfiftyCoarseOverFull"] == MT.fmt_pm(0.93, 0.01) and em.m["MlstFtTopRfiftyCoarseOverFullPr"] == r"7.0\% less rejection"
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
    t = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", "R42_Q1"))
    # no ratio to a perfect separation: the pair's group has no rho column
    assert r"\multicolumn{1}{c}{$X\to ee$ vs $\mu\mu$}" in t and t.count(r"\multicolumn{2}") == len(M.PAIRS) - 1
    assert r"\Kmid{} classes & " in t and r"0.9600$^{\dagger}$ & " in t
    # the 188-class run's AUC rounds to 1.0000: "> 0.9999", from one run and marked so
    assert M.NEAR_ONE == r"$>0.9999$" and " & " + M.NEAR_ONE + M.SINGLE + " & " in t
    # the ratio column and its note live in the supplement's copy (PI 2026-11-12: no ratio column in the paper)
    assert M.ZERO_NOTE and M.ZERO_NOTE in M._float("pairs_full", t, True, fill={"__ZERO__": M.zero_note(fits, runs)})
    # the comparisons the text quotes: a dash and words, never a division by zero
    em = _PEm()
    M.comparisons(em, fits, {**{a: [] for a in M.ALL_ROWS}, **runs, **{a: [m] for a, m in rand.items()}},
                  status, None, MT.fmt_pm)
    undefined = "undefined, since the reference separates the test jets completely"
    assert em.m["MlstCmpEemmMidOverFull"] == "--" and em.m["MlstCmpEemmMidOverFullPr"] == undefined
    assert em.m["MlstCmpEemmRandOverFull"] == "--" and em.m["MlstCmpEemmRandOverFullPr"] == undefined
    assert em.m["MlstPairRandMergedOverApartEemm"] == "--" and em.m["MlstPairRandMergedOverApartEemmPr"] == undefined
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
    # no 188-class run separates a pair completely: no note in the caption
    fits["mtx-l188-s1"]["cells"]["best70"]["jc2_188_pairs"]["1000000"]["ee_vs_mm"]["auc"] = 0.99
    assert M.ZERO_NOTE not in M._float("pairs_full", t, True, fill={"__ZERO__": M.zero_note(fits, runs)})
    # the paper's copy has no ratio column and so no such note
    assert "__ZERO__" not in M.printed_caption("pairs") and r"\rho" not in M.printed_caption("pairs")


def test_the_jetclass2_fine_tuning_block_has_the_ratio_column_at_one_million_jets(tmp_path):
    _ft_tree(tmp_path, ["l188-s1", "r16q1-s1", "l188-s2"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"]}, MT.fmt_pm, ("L188", "R16_Q1"))
    jc2, jc1 = t.split(M.BLOCK_SEP)[0], t.split(M.BLOCK_SEP)[3]
    assert r"& $\rho$ \\" in jc2 and r"\rho" not in jc1
    # (1 - 0.91) / (1 - 0.90) run by run; the 188-class row 1; from scratch has none
    assert r"\Kcoarse{} classes & 0.9100 & 0.9100 & 0.900 \\" in jc2 and r"\Kfull{} classes & 0.9000 & 0.9000 & 1.00 \\" in jc2
    assert jc2.splitlines()[-1].startswith(M.SCRATCH_LABEL) and jc2.splitlines()[-1].endswith(r"& -- \\")
    acc = t.split(M.BLOCK_SEP)[1]                                   # the JetClass-II accuracy, no ratio column
    assert "{JetClass-II, accuracy}" in acc and r"\rho" not in acc
    # a 188-class run without its 10^6 cell: the 188-class ratio cell says how many runs it rests on
    del per["jc2"]["l188-s2"][1000000]
    two = M.table_ft(per, {"L188": ["mtx-l188-s1", "mtx-l188-s2"]}, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)[0]
    # (at 10^3 jets both runs: mean +- SD of 0.90 and 0.92)
    assert r"\Kfull{} classes & 0.910\,$\pm$\,0.014 & 0.9000 \pending{1 of 2} & 1.00 \pending{1 of 2} \\" in two
    none = M.table_ft(per, {"L188": ["mtx-l188-s3"]}, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)[0]
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
        r"$\nmin$" in cap[k] and "eq:cut" in cap[k] and "__MINBKG__ " not in cap[k] for k in ("jetclass_rej", "rand_prong"))
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
    """tab:probesn (2026-11-07): the flavour-controlled partitions (two runs, mean +- SD), the five random
    partitions pooled into one row (mean +- SD over their runs) and the kinematic-only model on top only, in one
    longtable with a block per task."""
    rec = lambda a, r: {"macro_auc_ovr": a, "accuracy": a, "r30": r, "r30_is_bound": False, "converged": True}
    runs = {"L188": [f"mtx-l188-s{k}" for k in (1, 2, 3)], "FLAV_F0": ["mtx-flavf0-s1", "mtx-flavf0-s2"],
            **{a: [f"mtx-{a.lower().replace('_', '')}-s1"] for a in M.RAND}}
    v = {m: 0.90 + 0.01 * i for i, m in enumerate(m for ms in runs.values() for m in ms)}
    fits = {m: {"cells": {"best70": {"jc2_188": {"1000": rec(x, 0)}, "top": {"1000": rec(x, 100 * x)}}}}
            for m, x in v.items()}
    fits[M.KIN_MODEL] = {"cells": {"best70": {"top": {"1000": rec(0.9, 133.2)}}}}
    t = M.table_probes_n(fits, M._pooled(runs), MT.fmt_pm, ("L188", "FLAV_F0", M.RAND_POOLED), kin=True)
    assert r"\begin{longtable}" in t and r"\label{tab:probesn}" in t and "%SPLIT%" not in t
    rows = [ln for ln in t.splitlines() if ln.startswith(M.LABEL["FLAV_F0"])]
    assert [r.split(" & ")[1].removesuffix("*").removesuffix(r" \\") for r in rows] == [
        M._cell([f(v["mtx-flavf0-s1"]), f(v["mtx-flavf0-s2"])], 2, MT.fmt_pm) for f in (lambda x: x, lambda x: 100 * x)]
    pooled = [v[runs[a][0]] for a in M.RAND]
    jc2 = next(ln for ln in t.splitlines() if ln.startswith(M.LABEL[M.RAND_POOLED]))
    assert jc2.split(" & ")[1].removesuffix("*").removesuffix(r" \\") == M._cell(pooled, 5, MT.fmt_pm)
    assert not any(ln.startswith(M.LABEL["RAND2_p1"]) for ln in t.splitlines())     # pooled, not one by one
    kin = [ln for ln in t.splitlines() if ln.startswith(M.LABEL["KIN"])]
    assert kin == [M.LABEL["KIN"] + " & 133 \\\\"]                                     # top only, three figures
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
    per["top"][M.KIN_INIT] = {1000: [{"r30": 92.4, "r30_is_bound": False}], 1200000: [{"r30": 1802.8, "r30_is_bound": False}]}
    blocks = M.table_ft(per, runs, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)
    topb = next(b for b in blocks if "top, " in b)
    lines = topb.splitlines()
    assert lines.index(next(ln for ln in lines if ln.startswith(M.LABEL["KIN"]))) == \
        lines.index(next(ln for ln in lines if ln.startswith(M.SCRATCH_LABEL))) - 1
    assert lines[-1].startswith(M.SCRATCH_LABEL + " & ")
    assert M.LABEL["KIN"] + " & 92.4 & 1{,}803 \\\\" in topb       # from 100 on four significant figures
    assert sum(b.count(M.LABEL["KIN"]) for b in blocks) == 1


def test_the_pair_table_gives_the_random_network_both_ways_without_a_ratio():
    cell = lambda auc: {"auc": auc}
    pairs = lambda auc: {M.PAIR_N: {k: cell(auc) for k, *_ in M.PAIRS}}
    fits = {"mtx-l188-s1": {"cells": {"best70": {M.PAIR_DS: pairs(0.95)}}},
            **{m: {"cells": {"init": {M.PAIR_DS: pairs(0.6 + 0.01 * k)}}} for k, m in enumerate(M.INIT_MODELS)},
            **{m: {"cells": {"init_bn": {M.PAIR_DS: pairs(0.7)}}} for m in M.INIT_BN_MODELS}}
    runs = {"L188": ["mtx-l188-s1"], M.INIT: list(M.INIT_MODELS), M.INIT_BN: list(M.INIT_BN_MODELS)}
    raw = M.table_pairs(fits, runs, M.TAB, M.merge_status(ROOT), ("L188", M.INIT, M.INIT_BN))
    t = M.uniform_columns(raw)
    assert "%SPLIT%" not in t and M.V4.TOK_OPEN not in t
    row = lambda lab: next(ln for ln in t.splitlines() if ln.startswith(lab + " & ")).split(" & ")      # noqa: E731
    full, init, bn = row(M.LABEL["L188"]), row(M.LABEL[M.INIT]), row(M.LABEL[M.INIT_BN])
    # one precision per column, from the one SD there (0.01: two figures): 0.61 +- 0.01 over three initializations;
    # identical runs: the value once with the agreement mark, never "+- 0" or "<"; a single run: marked
    assert init[1] == r"0.610\,$\pm$\,0.010" and bn[1] == "0.700" + M.AGREE and full[1] == "0.950" + M.SINGLE
    assert "$<$" not in t and not re.search(r"\\pm\$\\,0(\.0+)? ", t)
    assert all(c.strip(" \\") == "--" for c in init[2::2] + bn[2::2])


def test_the_task_longtables_keep_each_heading_with_its_first_row(tmp_path):
    """tab:ft and tab:probesn (2026-11-07) are longtables: a block per task, its title and heading unbreakable
    from its first row (\\* and a rule with no break around it), the rule spanning every column."""
    _ft_tree(tmp_path, ["l188-s1"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))
    lines = t.splitlines()
    ncol = int(next(ln for ln in lines if ln.startswith(r"\multicolumn")).split("}")[0].split("{")[1])
    # across the whole line, the row labels on one line (PI 2026-11-11), set in landscape
    assert r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * (ncol - 1) + "@{}}" in lines
    assert lines[0] == r"\begin{landscape}" and r"\end{landscape}" in lines and "{\\small" in lines
    titles = [i for i, ln in enumerate(lines) if ln.startswith(r"\multicolumn") and "continued" not in ln]
    assert len(titles) == len(M.FT_TASKS) + 1                  # JetClass-II: AUC and accuracy
    for i in titles:
        assert lines[i].endswith(r"\\*") and lines[i + 1].startswith("training jets") and lines[i + 1].endswith(r"\\*")
        assert lines[i + 2] == M._head_rule(ncol) and r"\multispan{" + str(ncol) + "}" in lines[i + 2]
    assert r"\noalign{\nobreak\vskip\belowrulesep\nobreak}" in M._head_rule(3) and r"\midrule" not in M._head_rule(3)
    assert M._longtable("ft", t).count(r"\caption{") == 1 and "__CAPTION__" not in M._longtable("ft", t)


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
    assert _column(M.TAB(0.7153, 0.0024)) == [r"0.715\,$\pm$\,0.002"]            # the mean at the SD's place
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
    assert r"x & 312\,$\pm$\,6 & 0.715\,$\pm$\,0.002 \\" in t and r"y & 308\,$\pm$\,1 & 0.712\,$\pm$\,0.005 \\" in t
    assert rf"z & 301{M.AGREE} & 0.715\,$\pm$\,0.007 \\" in t and rf"w & 295\,$\pm$\,7 & 0.700{M.SINGLE} \\" in t
    assert M.AGREE == r"$^{\P}$" and M.SINGLE == r"$^{\S}$"
    # one digit finer for the whole column where an SD would print as zero (the brief's example, 0.98560 +- 0.00004):
    # median 0.0002, four decimals, where 0.00004 prints as 0.0000
    assert _column(T(0.98560, 0.00004), T(0.9841, 0.0003), T(0.9838, 0.0002)) == [
        r"0.98560\,$\pm$\,0.00004", r"0.98410\,$\pm$\,0.00030", r"0.98380\,$\pm$\,0.00020"]
    assert _column(T(0.9841, 0.0003), T(0.9838, 0.0002)) == [r"0.9841\,$\pm$\,0.0003", r"0.9838\,$\pm$\,0.0002"]
    # one digit, not more: an SD still zero one digit finer prints the value with the agreement mark
    assert _column(T(0.7153, 0.0024), T(0.7120, 0.0049), T(0.9760, 0.00003)) == [
        r"0.7153\,$\pm$\,0.0024", r"0.7120\,$\pm$\,0.0049", "0.9760" + M.AGREE]
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
    single, agree = (note for _, note in M.MARK_NOTES)
    assert [mark for mark, _ in M.MARK_NOTES] == [M.SINGLE, M.AGREE]
    assert single == r" $^{\S}$ marks a value from a single run."
    assert agree.startswith(r" $^{\P}$ marks a value whose standard deviation over the runs rounds to zero")
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
    col = 5 + [c for c, _ in V4.JC1_COLS].index("label_Tbqq")
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
    # the kinematic-only run: the table's label is the legend's without its last word (PI 2026-11-12: no wrapped label)
    assert plain(M.LABEL["KIN"]) == "188 classes, kinematic inputs" and legend["KIN"] == plain(M.LABEL["KIN"]) + " only"
    # the order: the label sets, the 17-class partitions (flavor boundaries, then the random ones), the networks with a
    # mass output; a table adds the random network (probes) or from scratch (fine-tuning) after them
    assert M.ALL_ROWS == M.MAIN + M.FLAV + M.RAND + M.MASS
    assert M._paper_rows(M.ALL_ROWS + (M.INIT,)) == M.MAIN + M.FLAV + (M.RAND_POOLED,) + M.MASS + (M.INIT,) == M.XBX_ROWS
    assert M.MASS_ROWS == M.MAIN + M.MASS + (M.INIT,) and M.SUMMARY_CONTROLS == M.FLAV + (M.RAND_POOLED,)
    assert M.LABELSET_ROWS == M.MAIN[1:] + M.FLAV + (M.RAND_POOLED,)


def test_the_mode_is_an_italic_heading_over_the_rows_and_the_partitions_sit_under_a_subheading():
    """PI 2026-11-12: _headings writes the downstream mode once, in italics, before each run of rows that name a
    network, in the brief's words; the partition rows are indented under "17-class partitions"; the released
    checkpoint below its rule belongs to the run above it; a published model's row or a row "(this work)" does not."""
    assert M.HEADING == {"lp": "linear probe on the frozen features", "ft": "fine-tuned",
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
    assert M.MODES["mlp"] == "mlp" and M._mode("jc1_class_auc", 0) == "ft" and M._mode("jc1_class_auc", 1) == "lp"
    assert not {"probes", "mass", "pairs", "summary", "published", "labelsets", "xbx", "xbx_rej", "sculpt_boot"} & set(M.MODES)


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
    assert "the released Sophon checkpoint has the missing inputs set to zero" in M.printed_caption("probes")
    co = {M.SOPHON_MODEL: {"cells": {"released": {"jc2bal_coarse": {t: {"1000000": {"accuracy": 0.5, "macro_auc_ovr": 0.9}}
                                                                    for t, _ in M.COARSE_TARGETS}}}}}
    c = M.table_coarse(co, runs, MT.fmt_pm, ("L188", M.SOPHON)).splitlines()
    k = next(i for i, ln in enumerate(c) if ln.startswith(M.LABEL[M.SOPHON] + " & "))
    assert c[k - 1] == r"\midrule" and c[k].split(" & ")[1] == "0.5000"
    # one table of three column groups, accuracy and AUC in each (PI 2026-11-12)
    assert c[2] == (r" & \multicolumn{2}{c}{43 groups} & \multicolumn{2}{c}{17 groups} & "
                    r"\multicolumn{2}{c}{17 groups with no flavor boundary} \\")
    assert c[4] == " & " + " & ".join(["accuracy & AUC"] * 3) + r" \\" and "target" not in "".join(c)


def test_the_pair_table_groups_the_rows_by_mode_with_the_mode_in_the_group_heading():
    """PI 2026-11-08, 2026-11-12: the fine-tuned pairs in the pair table, the probed and the fine-tuned rows as two row
    groups under the italic headings of every table (HEADING) and no tag on the rows, every pair a column group (AUC,
    rho); from scratch has no probe, the random networks no fine-tuning; the fine-tuned group is one pending row until
    the read-out is pulled. The paper's copy (rho=False) says AUC once, over the pairs."""
    pairs = lambda auc: {M.PAIR_N: {k: {"auc": auc} for k, *_ in M.PAIRS}}
    fits = {"mtx-l188-s1": {"cells": {"best70": {M.PAIR_DS: pairs(0.95)}}},
            "mtx-r16q1-s1": {"cells": {"best70": {M.PAIR_DS: pairs(0.90)}}},
            **{m: {"cells": {"init": {M.PAIR_DS: pairs(0.6)}}} for m in M.INIT_MODELS}}
    runs = {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"], M.INIT: list(M.INIT_MODELS)}
    ft = {"mtx-l188-s1": {k: {"auc": 0.98} for k, *_ in M.PAIRS}, "mtx-r16q1-s1": {k: {"auc": 0.96} for k, *_ in M.PAIRS},
          **{m: {k: {"auc": 0.9 + 0.01 * i} for k, *_ in M.PAIRS} for i, m in enumerate(M.FT_SCRATCH_MODELS)}}
    t = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", "R16_Q1", M.INIT), ft)
    assert "%SPLIT%" not in t and t.count(r"\multicolumn{2}{c}{") == len(M.PAIRS)
    lines = t.splitlines()
    lp, f = (lines.index(M._full_row(1 + 2 * len(M.PAIRS), M.HEADING[m], italic=True)) for m in ("lp", "ft"))
    assert lines[lp - 1] == lines[f - 1] == r"\midrule"
    row = lambda group, lab: next(ln for ln in group if ln.startswith(lab + " & ")).removesuffix(r" \\").split(" & ")
    assert [ln.split(" & ")[0] for ln in lines[lp + 1:f - 1]] == [M.LABEL["L188"], M.LABEL["R16_Q1"], M.LABEL[M.INIT]]
    assert [ln.split(" & ")[0] for ln in lines[f + 1:-2]] == [M.LABEL["L188"], M.LABEL["R16_Q1"], M.SCRATCH_LABEL]
    assert row(lines[f:], M.LABEL["R16_Q1"])[1:3] == [M._cell([0.96], 1, MT.fmt_pm, frac=True),
                                                     M._cell([(1 - 0.96) / (1 - 0.98)], 1, MT.fmt_pm)]
    assert row(lines[f:], M.SCRATCH_LABEL)[1] == M._cell([0.9, 0.91, 0.92], 3, MT.fmt_pm, frac=True)
    assert row(lines[lp:f], M.LABEL[M.INIT])[2] == "--"
    tab = M._float("pairs", t, True, fill={"__ZERO__": M.zero_note(fits, runs, ft)}, label_cm=M.PAIRS_LABEL_CM)
    assert r"\Kfull{} classes & " in tab and not re.search(r"\((LP|MLP|f\.t\.)\)", tab)
    assert all(rf"\textit{{{M.HEADING[m]}}}" in tab for m in ("lp", "ft")) and "pairs" not in M.MODES
    # the caption: the binary AUC in words, both random networks defined, and a sentence for each mark a cell
    # carries and for no other (17 classes merge every pair: daggers, no double dagger, no pooled-partition mark)
    assert "area under the receiver operating characteristic curve (AUC)" in tab and "An entry $>0.9999$" in tab
    assert M.BN_DEF_BOTH in tab and r"\DesignBnJets{}" in M.BN_DEF
    dagger, ddagger, pooled = (note for _, note in M.KEY_MARKS["pairs"])
    assert dagger in tab and ddagger not in tab and pooled not in tab and r"\ddagger" not in tab and M.POOL_MARK not in tab
    assert "natural composition" in M.CAPTIONS["pairs"][1]         # the full account, kept for the text
    empty = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188",), {}).splitlines()
    k = empty.index(M._full_row(1 + 2 * len(M.PAIRS), M.HEADING["ft"], italic=True))
    assert empty[k + 1] == M._full_row(1 + 2 * len(M.PAIRS), r"\pending{no run yet}")
    # the paper's copy: no ratio column, "AUC" once in the header over the pairs, the pairs' names below a rule
    p = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", "R16_Q1", M.INIT), ft, rho=False).splitlines()
    n = len(M.PAIRS)
    assert p[2:6] == [rf" & \multicolumn{{{n}}}{{c}}{{AUC}} \\", rf"\cmidrule(lr){{2-{1 + n}}}",
                      " & " + " & ".join(h for _, h, *_ in M.PAIRS) + r" \\", r"\midrule"]
    assert "\n".join(p).count("AUC") == 1 and r"\rho" not in "\n".join(p)
    assert p[6] == M._full_row(1 + n, M.HEADING["lp"], italic=True) and M._full_row(1 + n, M.HEADING["ft"], italic=True) in p
    assert all(ln.count(" & ") == n for ln in p if ln.startswith((M.LABEL["L188"], M.LABEL["R16_Q1"], M.SCRATCH_LABEL)))


def test_an_auc_that_rounds_to_one_prints_as_more_than_0_9999_and_leaves_its_columns_precision_alone():
    """PI 2026-11-12 (Table A10): a mean AUC that rounds to 1.0000 prints "> 0.9999" with no spread, plain text that
    takes no part in the column's precision; a single run's also carries the single-run mark; an AUC below that
    prints as a value."""
    key = M.PAIRS[0][0]
    one = lambda aucs: {"cells": {"best70": {M.PAIR_DS: {M.PAIR_N: {k: {"auc": aucs[k == key]} for k, *_ in M.PAIRS}}}}}   # noqa: E731
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"], "L162": ["mtx-l162-s1"], "R16_Q1": ["mtx-r16q1-s1", "mtx-r16q1-s2"]}
    # the first pair: 188 classes 0.99996, 0.99997, 0.99998 (mean 0.99997); 162 classes one run at 0.99999; 17 classes
    # 0.9990 and 0.9994. Every other pair: 0.9
    fits = {**{m: one((0.9, a)) for m, a in zip(runs["L188"], (0.99996, 0.99997, 0.99998))},
            "mtx-l162-s1": one((0.9, 0.99999)), "mtx-r16q1-s1": one((0.9, 0.9990)), "mtx-r16q1-s2": one((0.9, 0.9994))}
    status = {a: {k: "apart" for k, *_ in M.PAIRS} for a in runs}
    raw = M.table_pairs(fits, runs, M.TAB, status, ("L188", "L162", "R16_Q1"), rho=False)
    row = lambda t, lab: next(ln for ln in t.splitlines() if ln.startswith(lab + " & ")).split(" & ")      # noqa: E731
    assert row(raw, M.LABEL["L188"])[1] == M.NEAR_ONE and row(raw, M.LABEL["L162"])[1] == M.NEAR_ONE + M.SINGLE
    assert M.V4.TOK_OPEN in row(raw, M.LABEL["R16_Q1"])[1] and M.V4.TOK_OPEN not in row(raw, M.LABEL["L188"])[1]
    t = M.uniform_columns(raw)
    # the column's place comes from the one SD left in it (0.0003: four decimals), not from the near-one cells
    assert [row(t, M.LABEL[a])[1] for a in runs] == [M.NEAR_ONE, M.NEAR_ONE + M.SINGLE, r"0.9992\,$\pm$\,0.0003"]
    assert row(t, M.LABEL["L162"])[2] == "0.9000" + M.SINGLE and row(t, M.LABEL["L188"])[2] == "0.9000" + M.AGREE
    assert M.load_pairs_ft(DATA.parent / "no_such_dir") == {}


def test_the_sculpting_bootstrap_table_reads_the_verdicts_and_refuses_another_sigma():
    """Plan 2026-11-08: a sculpting difference stands when |delta| > 3 sigma, sigma^2 = run SD^2 + bootstrap SD^2
    (head/sculpting_boot.json); one row group per working point; no bold (PI 2026-11-12)."""
    x = lambda d, run, boot: {"delta": d, "run_sd": run, "boot_sd": boot, "sigma": math.hypot(run, boot),
                              "z": d / math.hypot(run, boot), "pass": abs(d) > 3 * math.hypot(run, boot)}
    doc = {"replicas": 200, "differences": {"17-188": {"configuration": "r16q1", "reference": "l188", **{
        wp: {"distance": x(0.0122, 0.003, 0.001) if wp == "eps_b=0.01" else x(0.001, 0.003, 0.001)}
        for wp, _ in M.SCULPT_BOOT_WPS}}}}
    runs = {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"]}
    t = M.table_sculpt_boot(doc, runs).splitlines()
    rows = [ln for ln in t if ln.startswith(M.LABEL["R16_Q1"] + " $-$ " + M.LABEL["L188"] + " & ")]
    assert len(rows) == len(M.SCULPT_BOOT_WPS) and not any(r"\textbf{" in r or M.BOLD_ON in r for r in rows)
    assert "sculpt_boot" not in M.BOLD                 # a table of differences and their sigma: never bold
    assert rows[0].split(" & ")[1:4] == ["1.22", "0.30", "0.10"]
    # the differences in the tables' row order, a partition's indented under the subheading, in every block
    q = {wp: {"distance": x(0.001, 0.003, 0.001)} for wp, _ in M.SCULPT_BOOT_WPS}
    more = {"replicas": 200, "differences": {"f0-188": {"configuration": "flavf0", "reference": "l188", **q},
                                             "162-188": {"configuration": "l162", "reference": "l188", **q},
                                             **doc["differences"]}}
    lines = M.table_sculpt_boot(more, {**runs, "L162": ["mtx-l162-s1"], "FLAV_F0": ["mtx-flavf0-s1"]}).splitlines()
    names = [ln.split(" & ")[0] for ln in lines if " $-$ " in ln]
    ref = " $-$ " + M.LABEL["L188"]
    assert names == [M.LABEL["L162"] + ref, M.LABEL["R16_Q1"] + ref, r"\quad " + M.LABEL["FLAV_F0"] + ref] * len(M.SCULPT_BOOT_WPS)
    assert lines.count(M._full_row(5, M.PARTITIONS)) == len(M.SCULPT_BOOT_WPS)
    assert all(lines[i + 1].startswith(r"\quad ") for i, ln in enumerate(lines) if ln == M._full_row(5, M.PARTITIONS))
    doc["differences"]["17-188"]["eps_s=0.5"]["distance"]["sigma"] = 0.003 / math.sqrt(3)
    with pytest.raises(SystemExit, match="hypot"):
        M.table_sculpt_boot(doc, runs)


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
    monkeypatch.setattr(B, "figure", lambda pid, rec, src: f"% figure {pid}\n")
    monkeypatch.setattr(G, "PROVISIONAL", {"F11": "waits", "tab:mass": "waits"})
    (tmp_path / "tables" / f"{B.TABLE_FILE['tab:mass']}.tex").write_text(
        "\\begin{table}[t]\n\\caption{c}\\label{tab:mass}\n\\begin{tabular}{l}\\pending{x}\\end{tabular}\n\\end{table}\n")
    figs = [k for k, _, _ in G.FIGURES]
    pages = tmp_path / "render"
    pages.mkdir()
    write = lambda ids: (pages / "pages.json").write_text(json.dumps([{"id": i, "drawn": True} for i in ids]))
    write([i for i in figs if i != "F11"])
    B.build(pages, "paper")
    out = capsys.readouterr().out
    assert "F11 is provisional" in out and "tab:mass is provisional" in out
    book = (tmp_path / "results_book.tex").read_text()
    assert r"\label{fig:F11}" in book and r"\label{tab:mass}" in book and "% figure FJ-ctrl" not in book
    assert r"\pending{x}" in book and book.count("fbox") == 1             # the table as it is, F11 a box
    # the comparison with published models (tab:published) is a main-text table; the task tables it replaced are placed
    # nowhere (PI 2026-11-12)
    assert book.index("% figure F1\n") < book.index("% tables/mlst_published.tex") < book.index(r"{A\arabic{figure}}")
    assert book.index(r"{A\arabic{figure}}") < book.index("% tables/mlst_labelsets.tex")
    assert not any(f"% tables/mlst_{name}.tex" in book for name in ("jetclass", "bench_top", "bench_qg", "bench_qg_herwig"))
    assert r"\arabic{table}b}" not in book and "externaldocument" not in book     # no 2a/2b (PI 2026-11-12)
    with pytest.raises(SystemExit, match="results_book.aux"):
        B.build(pages, "supplement")
    (tmp_path / "results_book.aux").write_text("")
    B.build(pages, "supplement")
    supp = (tmp_path / "results_supplement.tex").read_text()
    assert supp.index(r"\usepackage{xr-hyper}") < supp.index("{hyperref}") < supp.index(r"\externaldocument{results_book}")
    assert "% figure FJ-ctrl" in supp and "% figure F1\n" not in supp
    write([i for i in figs if i not in ("F11", "F8")])
    with pytest.raises(SystemExit, match="F8 was not drawn"):
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
    row = next(ln for ln in first.splitlines() if "512-unit hidden layer" in ln)
    cells = row.rstrip(" \\").split(" & ")
    assert len(cells) == 12 and cells[2:8] == ["--"] * 6 and cells[10:] == ["--"] * 2
    assert "%SPLIT%" not in t.split("512-unit hidden layer")[0].split("class token")[-1]


# ------------------------------------------------------------------ PI 2026-11-12: the paper's tables
def test_the_paper_tables_pool_the_random_partitions_where_the_first_of_them_stood():
    # in the tables' order (ALL_ROWS) that is after the flavor-boundary partitions and before the mass networks
    assert M._paper_rows(M.ALL_ROWS + (M.INIT,)) == M.MAIN + M.FLAV + (M.RAND_POOLED,) + M.MASS + (M.INIT,)
    assert M._paper_rows(M.ALL_ROWS + (M.INIT, M.SOPHON)) == M.MAIN + M.FLAV + (M.RAND_POOLED,) + M.MASS + (M.INIT, M.SOPHON)
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
    # the paper's tab:probes is the task-block builder: four columns, no ratio, no JetClass-II average but the one
    # over all classes
    pt = M.table_probes_tasks(fits, runs, M.TAB, ("L188", M.RAND_POOLED))
    assert r"\rho" not in pt and "full count" not in pt and "resonant" not in pt and M.V4.TOK_OPEN not in pt
    assert all(pg.splitlines()[2] == r" & accuracy & AUC & $R_{50}$ & $R_{30}$ \\" for pg in pt.split("%SPLIT%"))
    p = M.table_pairs({}, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", M.RAND_POOLED), rho=False)
    assert r"\rho" not in p
    pooled = next(ln for ln in p.splitlines() if ln.startswith(M.LABEL[M.RAND_POOLED] + " & ")).split(" & ")
    # X->ee vs mumu: random partitions 1, 2 and 4 merge it, 3 and 5 keep it apart; the QCD pairs: all five merge them
    keys = [k for k, *_ in M.PAIRS]
    assert M.POOL_MARK in pooled[1 + keys.index("ee_vs_mm")] and r"$^{\dagger}$" in pooled[1 + keys.index("qcd_bb_vs_cc")]
    # the pooled-partition mark is the parallel sign: the section sign marks a single run in every table
    assert M.POOL_MARK == r"$^{\parallel}$" and M.POOL_MARK != M.SINGLE and not hasattr(M, "SAME_AUC")


def test_both_resonance_tables_have_the_same_rows_and_the_rejection_table_shows_both_from_scratch_signals():
    runs = M._paper(ROOT)
    end = lambda r30: {"r50": r30 / 3, "r30": r30}      # noqa: E731
    scr = {"xbb": [end(900.0), end(950.0), end(1000.0)], "xbs": [end(600.0), end(610.0), end(620.0)]}
    t = M.table_xbx_rej({}, scr, runs, M.TAB)
    names = [ln.split(" & ")[0] for ln in t.splitlines() if " & " in ln and not ln.startswith((" ", "&"))]
    XR = _load("xbx_report_rows_t", "experiments/XBS/xbx_report.py")
    assert names == [M.LABEL[r] for r in XR.ROWS] + [M.SCRATCH_LABEL] and tuple(XR.ROWS) == M.XBX_ROWS
    assert M.XBX_ROWS == M.MAIN + M.FLAV + (M.RAND_POOLED,) + M.MASS + (M.INIT,)      # the tables' one order
    row = next(ln for ln in t.splitlines() if ln.startswith(M.SCRATCH_LABEL)).rstrip(r" \\").split(" & ")
    assert row[3:5] == ["--", "--"] and row[7:9] == ["--", "--"]          # direct and closest node: none
    assert "--" not in row[1:3] and "--" not in row[5:7]                     # transfer: both signals
    assert "--" not in M.table_xbx_rej({}, {"xbs": scr["xbs"]}, runs, M.TAB).splitlines()[-3].split(" & ")[5]
    assert all(c == "--" for c in M.table_xbx_rej({}, {"xbs": scr["xbs"]}, runs, M.TAB).splitlines()[-3]
               .rstrip(r" \\").split(" & ")[1:3])                           # no X->bb from scratch yet: dashes
    # one X->bb seed from scratch beside three X->bs seeds (PI 2026-11-12): the single run's values carry the mark
    one = M.uniform_columns(M.table_xbx_rej({}, {"xbb": scr["xbb"][:1], "xbs": scr["xbs"]}, runs, M.TAB))
    assert one.splitlines()[-3].rstrip(r" \\").split(" & ")[1:] == [
        "300" + M.SINGLE, "900" + M.SINGLE, "--", "--", r"203\,$\pm$\,3", r"610\,$\pm$\,10", "--", "--"]
    assert M.SINGLE not in M.uniform_columns(t)                              # three seeds each: no mark


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_summary_table_holds_exactly_the_values_of_figs_2_and_3():
    """tab:summary reads every run's value as Figs. 2 and 3 do (paper_figures._f2_probe, _f2_ft, scratch_series), on
    both panels of each task: its headline metric and its AUC (SUMMARY_GRID, PI 2026-11-12). The gallery's own value
    functions and the printed cells: tests/test_gallery_pages.py."""
    P = _load("paper_figures_summary_t", "experiments/FIGS/paper_figures.py")
    I = P.inputs(M)
    runs = M._pooled(M._paper(ROOT))
    chosen, _ = M.chosen_substitutions(I["fill"], runs)
    assert tuple((k, n, ds, met) for k, n, ds, met, *_ in P.F2_PANELS) == M.SUMMARY_PANELS
    assert M.SUMMARY_GRID == tuple((k, n, ds, m) for k, n, ds, met, *_ in P.F2_PANELS for m in (met, M.SUMMARY_AUC[k]))
    for key, n, ds, met in M.SUMMARY_GRID:
        a, b = P._f2_probe(M, I["fits"], I["fill"], chosen, ds, met), M.summary_probe(I["fits"], I["fill"], chosen, ds, met)
        c, d = P._f2_ft(M, I["per"], key, n, met), M.summary_ft(I["per"], key, n, met)
        for arm in M.MAIN + M.FLAV + (M.RAND_POOLED, M.INIT):
            assert [a(m) for m in runs[arm]] == [b(m) for m in runs[arm]]
            if arm != M.INIT:
                assert [c(m) for m in runs[arm]] == [d(m) for m in runs[arm]]
        assert P.scratch_series(M, I["per"], key, met)[n] == [r[met] for r in I["per"][key][M.SCRATCH][n] if r.get(met) is not None]
    t = M.table_summary(I["fits"], I["fill"], chosen, I["per"], M._paper(ROOT), M.TAB)
    # one part for each mode and pair of tasks (SUMMARY_PAGES): the eight columns of a mode are wider than a turned
    # page at the body size
    parts = t.split("%SPLIT%")
    order = [(mode, tasks) for mode in ("lp", "ft") for tasks in M.SUMMARY_PAGES]
    assert len(parts) == len(order) == 4
    assert t.count(M.LABEL[M.RAND_POOLED]) == 4 and t.count(M.SCRATCH_LABEL + " & ") == t.count(M.LABEL[M.INIT] + " & ") == 2
    assert r"\rho" not in t and r"\textbf" not in t
    ctl = [r"\quad " + M.LABEL[a] for a in M.SUMMARY_CONTROLS]
    for part, (mode, tasks) in zip(parts, order):
        lines = part.splitlines()
        grid = [g for g in M.SUMMARY_GRID if g[0] in tasks]
        # two header rows: the task over its two columns, then the task's metric and its AUC
        assert lines[:6] == [r"\begin{tabular}{lcccc}", r"\toprule",
                             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{M.SUMMARY_TASKS[k]}}}" for k in tasks) + r" \\",
                             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                             " & " + " & ".join(M.SUMMARY_HEAD[met] for *_, met in grid) + r" \\", r"\midrule"]
        assert [M.SUMMARY_HEAD[met] for *_, met in grid][1::2] == ["AUC", "AUC"]
        # under the italic heading of the mode: the label sets, then the partitions indented under their subheading;
        # the random network closes the probes, from scratch the fine-tuned rows
        assert lines[6] == M._full_row(5, M.HEADING[mode], italic=True)
        assert [ln.split(" & ")[0] for ln in lines[7:11]] == [M.LABEL[a] for a in M.MAIN]
        assert lines[11] == M._full_row(5, M.PARTITIONS) and part.count(M._full_row(5, M.PARTITIONS)) == 1
        assert [ln.split(" & ")[0] for ln in lines[12:-2]] == ctl + [M.LABEL[M.INIT] if mode == "lp" else M.SCRATCH_LABEL]
        assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]
        rows = [ln.removesuffix(r" \\").split(" & ") for ln in lines[7:11] + lines[12:-2]]
        assert all(len(r) == 5 and "--" not in r for r in rows)
        # bold (PI 2026-11-12): every column of a part has its best value; the random network is a reference row,
        # never bold and never the best; from scratch takes part
        best = rows[:-1] if mode == "lp" else rows
        assert all(any(M.BOLD_ON in r[j] for r in best) for j in range(1, 5)), (mode, tasks)
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
    per = {"top": _bench_per(runs), "qg": _bench_per(runs)}
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
    assert [n.split("~")[0] for n in names["top tagging"][:-3]] == list(M.V4.LIT_ROWS["top"])
    assert [n.split("~")[0] for n in names["quark/gluon tagging"][:-2]] == list(M.V4.LIT_ROWS["qg"])
    assert all(r"\cite{" in n for k, v in names.items() for n in v if n not in (scratch, ft, kin))
    assert all(ln.count(" & ") == 6 and r"\midrule" not in ln for v in blocks.values() for ln in v)     # six value columns, no rule
    row = lambda k, name: next(ln for ln in blocks[k] if ln.startswith(name + " & ")).removesuffix(r" \\").split(" & ")   # noqa: E731
    B = M._b
    # JetClass: no R50 and R30 (the per-class rejections are tab:jetclassrej); training jets: the published 2 M, ours 1 M
    assert all(ln.endswith(r" & -- & -- \\") for ln in blocks["JetClass"])
    # bold (PI 2026-11-12): in each block's column the best value and every value within three standard deviations of
    # it, a published value with the SD it prints (none: zero). Accuracy: ParT's 0.836 alone (ours 0.802 and 0.801
    # +- 0.001). AUC: ParT's 0.9834 and both of ours (0.982 and 0.981 +- 0.001), not ParticleNet's 0.9820
    assert row("JetClass", r"ParT~\cite{part}") == [r"ParT~\cite{part}", r"2.14\,M", r"2\,M", B("0.836"), B("0.9834"), "--", "--"]
    assert row("JetClass", scratch)[4] == B(r"0.9820\,$\pm$\,0.0010") and row("JetClass", ft)[4] == B(r"0.9810\,$\pm$\,0.0010")
    assert M.BOLD_ON not in " & ".join(row("JetClass", r"ParticleNet~\cite{part}"))
    # the parameters and the training jets are counts: never bold
    assert not any(M.BOLD_ON in c for v in blocks.values() for ln in v for c in ln.split(" & ")[:3])
    assert row("JetClass", scratch)[1:3] == [r"2.21\,M", r"1\,M"] and row("JetClass", ft)[1:3] == [r"2.21\,M", r"1\,M"]
    assert row("JetClass", scratch)[3] == r"0.8020\,$\pm$\,0.0010" and row("JetClass", ft)[3] == r"0.8010\,$\pm$\,0.0010"
    # top and quark/gluon: the task's training set, a dash for an observable that is not trained; the published values
    # as their papers print them, ours at one precision per column of the block
    assert row("top tagging", r"ParT~\cite{part}") == [r"ParT~\cite{part}", r"2.14\,M", r"1.2\,M", "0.940", "0.9858",
                                                      r"413\,$\pm$\,16", r"1{,}602\,$\pm$\,81"]
    assert row("top tagging", r"linear EFP~\cite{toptag}")[1:] == [r"1\,k", r"1.2\,M", "0.932", "0.980", "--", "384"]
    assert row("quark/gluon tagging", r"constituent multiplicity~\cite{efn}")[1:] == ["--", "--", "--", "0.8401", "19.0", "--"]
    assert row("quark/gluon tagging", r"ParT~\cite{part}")[2] == r"1.6\,M" and row("quark/gluon tagging", ft)[2] == r"1.6\,M"
    agree = lambda x: x + M.AGREE      # noqa: E731  three runs with one accuracy and AUC in the fixture
    assert row("top tagging", scratch) == [scratch, r"2.21\,M", r"1.2\,M", agree("0.9400"), agree("0.9860"),
                                           r"50\,$\pm$\,10", r"1{,}400\,$\pm$\,10"]
    assert row("top tagging", ft)[5:] == [r"460\,$\pm$\,10", r"1{,}810\,$\pm$\,10"]
    # the kinematic-only network: one run, every value marked, at the column's place. Its R50 of 950 is the block's
    # best (a single run: SD zero, and no other value within reach), bold with the mark outside
    assert row("top tagging", kin) == [kin, r"2.21\,M", r"1.2\,M", "0.9400" + M.SINGLE, "0.9860" + M.SINGLE,
                                       B("950") + M.SINGLE, "2{,}300" + M.SINGLE]
    # the other columns' best values are published ones: L-GATr's accuracy, OmniLearned-l's AUC and R30 (3,486 +- 157;
    # 2,894 +- 84 is 592 away, more than three times 178)
    assert row("top tagging", r"L-GATr-f.t.~\cite{lgatr}")[3:] == [B("0.9446"), "0.9879", r"651\,$\pm$\,11", r"2{,}894\,$\pm$\,84"]
    assert row("top tagging", r"OmniLearned-l~\cite{omnilearned}")[3:] == ["0.944", B("0.9880"), r"688\,$\pm$\,9",
                                                                          B(r"3{,}486\,$\pm$\,157")]
    assert sum(M.BOLD_ON in c for ln in blocks["top tagging"] for c in ln.split(" & ")) == 4
    # quark/gluon: the fixture's two rows of ours are above every published value and agree on accuracy and AUC
    assert row("quark/gluon tagging", scratch)[3:5] == row("quark/gluon tagging", ft)[3:5] == [B("0.9400") + M.AGREE,
                                                                                               B("0.9860") + M.AGREE]
    assert row("quark/gluon tagging", ft)[5:] == [B(r"460\,$\pm$\,10"), B(r"1{,}810\,$\pm$\,10")]
    assert row("quark/gluon tagging", scratch)[5:] == [r"50\,$\pm$\,10", r"1{,}400\,$\pm$\,10"]
    assert kin not in pages[1] and M.SINGLE not in pages[1] and M.SINGLE not in "".join(blocks["JetClass"])
    # set: two whole turned pages under one caption and number, the second "(continued)"; the single-run sentence
    f = M._float("published", t, True, pages=True, whole="turned", colsep_pt=6)
    assert f.startswith(r"\begin{landscape}") and r"\begin{longtable}" not in f and f.count(r"\begin{minipage}{\linewidth}") == 2
    assert f.count(r"\captionof{table}{") == 1 and f.count(r"\label{tab:published}") == 1
    assert f.count(r"\textit{Table~\ref{tab:published} (continued)}") == 1 and M.MARK_NOTES[0][1] in f and M.MARK_NOTES[1][1] in f
    assert f.index("top tagging} \\\\") < f.index("(continued)}") < f.index("quark/gluon tagging} \\\\")
    assert rf"\textit{{{M.HEADING['ft']}}}" not in f and M.PARTITIONS not in f and not _sizes_in(f)
    assert M.CAPTIONS["published"][0] == "tab:published" and "published" not in M.MODES
    # the second page starts a page; the caption says what bold marks, once, in the words for a table of blocks
    assert f.count(r"\newpage") == 1 and f.index("top tagging} \\\\") < f.index(r"\newpage") < f.index("(continued)}")
    assert f.count(M.BOLD_NOTE.format("each block of a column")) == 1 and M.BOLD_NOTE.format("a column") not in f
    assert r"\textbf" not in f and f.count(M.BOLD_ON) == t.count(M.BOLD_ON) > 0


def test_the_label_set_table_holds_the_fine_tuned_rows_the_comparison_table_does_not():
    """tab:labelsets (PI 2026-11-12: the other label sets' rows in one appendix table; apart from Table 1 a number
    has one table): 162, 43 and 17 classes and the 17-class partitions fine-tuned on JetClass, top and quark/gluon
    tagging, without the 188-class and from-scratch rows that tab:published prints; scored on Herwig jets, which no
    other table has, every row. Accuracy, AUC, R50, R30; a block a task under "fine-tuned on <task>"."""
    runs = M._paper(ROOT)
    arms = M.MAIN + M.FLAV + M.RAND
    pre, ref = _jc1_trees(runs, arms)
    by = _bench_all(runs, arms)
    per = {"top": by, "qg": by, "qg_herwig": by}
    t = M.table_labelsets(per, runs, pre, ref, M.TAB)
    pages = t.split("%SPLIT%")
    assert M.LABELSET_PAGES == (("jc1", "top"), ("qg", "qg_herwig")) and len(pages) == 2 and M.V4.TOK_OPEN not in t
    assert all(pg.splitlines()[:4] == [r"\begin{tabular}{lcccc}", r"\toprule", r" & accuracy & AUC & $R_{50}$ & $R_{30}$ \\",
                                       r"\midrule"] for pg in pages)
    blocks = {**_blocks(pages[0], 5), **_blocks(pages[1], 5)}
    tasks = [f"{M.HEADING['ft']} on {dict(M.TASK_BLOCKS)[k]}" for k in ("jc1", "top", "qg", "qg_herwig")]
    assert list(blocks) == tasks == ["fine-tuned on JetClass", "fine-tuned on top tagging", "fine-tuned on quark/gluon tagging",
                                     "fine-tuned on quark/gluon tagging, scored on Herwig jets"]
    assert all(M._full_row(5, h, italic=True) in t for h in tasks)
    names = {k: [ln.split(" & ")[0] for ln in v] for k, v in blocks.items()}
    rest = [M.LABEL[a] for a in M.LABELSET_ROWS]
    assert rest == [M.LABEL[a] for a in ("L162", "R42_Q1", "R16_Q1")] + [M.LABEL[a] for a in M.FLAV] + [M.LABEL[M.RAND_POOLED]]
    assert names[tasks[0]] == names[tasks[1]] == names[tasks[2]] == rest          # no 188 classes, no from scratch
    assert names[tasks[3]] == [M.LABEL["L188"]] + rest + [M.SCRATCH_LABEL]          # Herwig: every row, from scratch last
    assert all(ln.count(" & ") == 4 for v in blocks.values() for ln in v)
    assert all(ln.endswith(r" & -- & -- \\") for ln in blocks[tasks[0]])
    assert not any(ln.endswith(r" & -- & -- \\") for k in tasks[1:] for ln in blocks[k])
    # the pooled random partitions: mean and SD over the five, one run each (R30 1800 each in the fixture)
    pooled = next(ln for ln in blocks[tasks[1]] if ln.startswith(M.LABEL[M.RAND_POOLED] + " & "))
    # bold (PI 2026-11-12): the best of each block's column and every value within three standard deviations of it,
    # the mark outside. On top tagging that is every row of the fixture (R30 1,810 +- 10, 1,805 +- 7 and 1,800)
    assert pooled.removesuffix(r" \\").split(" & ")[4] == M._b("1{,}800") + M.AGREE
    assert next(ln for ln in blocks[tasks[1]] if ln.startswith(M.LABEL["L162"] + " & ")).removesuffix(r" \\").split(" & ")[3:] == [
        M._b(r"460\,$\pm$\,10"), M._b(r"1{,}810\,$\pm$\,10")]
    assert all(M.BOLD_ON in c for ln in blocks[tasks[1]] for c in ln.removesuffix(r" \\").split(" & ")[3:])
    # on Herwig jets from scratch (R50 50 +- 10, R30 1,400 +- 10) is far from the fine-tuned rows: not bold
    assert next(ln for ln in blocks[tasks[3]] if ln.startswith(M.SCRATCH_LABEL + " & ")).removesuffix(r" \\").split(" & ")[3:] == [
        r"50\,$\pm$\,10", r"1{,}400\,$\pm$\,10"]
    assert M.BOLD_ON not in "".join(c for k in tasks for ln in blocks[k] for c in ln.split(" & ")[:1]) and r"\textbf" not in t
    # no row of ours is in both tables: tab:published has the 188-class and from-scratch rows of the first three tasks
    pub_pre, pub_ref = _jc1_trees(runs, ("L188",))
    pub = M.table_published({"top": by, "qg": by}, runs, pub_pre, pub_ref, M.TAB)
    ours = {ln.split(" & ")[0] for ln in pub.splitlines() if " & " in ln and r"\cite{" not in ln and ln.split(" & ")[0].strip()}
    assert ours == {M.SCRATCH_LABEL + M.THIS_WORK, f"fine-tuned from {M.LABEL['L188']}{M.THIS_WORK}", M.LABEL["KIN"]}
    assert not ours & {n for k in tasks for n in names[k]}
    # set: the partitions indented under their subheading in every block (_headings), two whole turned pages
    f = M._float("labelsets", t, True, pages=True, whole="turned", colsep_pt=6)
    assert f.count(M._full_row(5, M.PARTITIONS)) == 4 and f.count(r"\quad " + M.LABEL["FLAV_F0"] + " & ") == 4
    assert f.count(r"\quad " + M.LABEL[M.RAND_POOLED] + " & ") == 4 and r"\quad " + M.LABEL["L162"] not in f
    assert f.count(rf"\textit{{{M.HEADING['ft']}") == 4 and f.count(r"\begin{minipage}{\linewidth}") == 2
    assert f.count(r"\textit{Table~\ref{tab:labelsets} (continued)}") == 1 and f.count(r"\label{tab:labelsets}") == 1
    assert M.CAPTIONS["labelsets"][0] == "tab:labelsets" and "labelsets" not in M.MODES and not _sizes_in(f)
    assert f.count(M.BOLD_NOTE.format("each block of a column")) == 1 and f.count(r"\newpage") == 1


def test_the_benchmark_and_jetclass_rows_are_shared_by_the_task_tables_and_the_new_ones():
    """mlst_v4.bench_rows and jetclass_rows: the published rows, and ours with from scratch last (the kinematic-only
    network before it); 'qg_herwig' has no published row; published=True adds each published row's kind."""
    V4, runs = M.V4, M._paper(ROOT)
    per = {"top": _bench_per(runs), "qg": _bench_per(runs), "qg_herwig": _bench_per(runs)}
    lit, ours = V4.bench_rows("top", per, runs, ("L188",), M.LABEL, M.SCRATCH, M.TAB, M.TAB.two,
                              kin=(M.LABEL["KIN"], M.KIN_INIT), published=True)
    assert [name.split("~")[0] for name, *_ in lit] == list(V4.LIT_ROWS["top"]) and all(len(r) == 4 and len(r[1]) == 4 for r in lit)
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


def test_the_probe_table_of_the_paper_has_the_tasks_as_row_blocks_under_four_columns():
    """tab:probes (PI 2026-11-12, Table A1): the tasks as row blocks (JetClass-II, JetClass, top, quark/gluon on
    Pythia and scored on Herwig jets) with the same columns in each, accuracy, AUC, R50 and R30, a dash where a
    metric does not apply; the full-count, resonant and QCD averages are the supplement's; the kinematic-only
    network in the top block, after the pretrained rows; the released checkpoint last in every block, below a rule;
    upright on two pages."""
    fits, runs = _probe_fits()
    t = M.table_probes_tasks(fits, runs, M.TAB, ("L188", M.INIT, M.SOPHON), kin=True)
    pages = t.split("%SPLIT%")
    assert M.PROBE_PAGES == (("jc2", "jc1"), ("top", "qg", "qg_herwig")) and len(pages) == 2
    assert all(pg.splitlines()[:4] == [r"\begin{tabular}{lcccc}", r"\toprule", r" & accuracy & AUC & $R_{50}$ & $R_{30}$ \\",
                                       r"\midrule"] for pg in pages)
    assert M.V4.TOK_OPEN not in t and r"\rho" not in t and "0.11" not in t and "0.22" not in t and "probes" not in M.MODES
    blocks = [_blocks(pg, 5) for pg in pages]
    tasks = ["JetClass-II", "JetClass", "top tagging", "quark/gluon tagging", "quark/gluon tagging, scored on Herwig jets"]
    assert [list(b) for b in blocks] == [[f"{M.HEADING['lp']}, {x}" for x in tasks[:2]], [f"{M.HEADING['lp']}, {x}" for x in tasks[2:]]]
    assert all(M._full_row(5, f"{M.HEADING['lp']}, {x}", italic=True) in t for x in tasks)
    rows = {x: b[f"{M.HEADING['lp']}, {x}"] for b in blocks for x in tasks if f"{M.HEADING['lp']}, {x}" in b}
    full, init, sophon, kin = M.LABEL["L188"], M.LABEL[M.INIT], M.LABEL[M.SOPHON], M.LABEL["KIN"]
    for x in tasks:
        names = [ln.split(" & ")[0] for ln in rows[x]]
        assert names == [full] + [kin] * (x == "top tagging") + [init, r"\midrule", sophon], x
        assert all(ln.count(" & ") == 4 for ln in rows[x] if ln != r"\midrule")
    S, A = M.SINGLE, M.AGREE
    # a dash where the task has no single rejection; single runs marked; three identical runs: the agreement mark
    assert rows["JetClass-II"] == [rf"{full} & 0.4000{S} & 0.9800{S} & -- & -- \\", rf"{init} & 0.4000{A} & 0.6000{A} & -- & -- \\",
                                   r"\midrule", rf"{sophon} & 0.4000{S} & 0.9700{S} & -- & -- \\"]
    assert rows["JetClass"][0] == rf"{full} & 0.7000{S} & 0.9700{S} & -- & -- \\"
    # each block at its own precision per column: the one SD of the R50 column is 1.0 (two figures), of R30 2 (one).
    # Bold (PI 2026-11-12) among the rows that are neither the random network nor the released checkpoint: the
    # 188-class run, and the kinematic-only run where it ties it (accuracy, AUC); the mark outside the bold
    B = M._b
    assert rows["top tagging"] == [rf"{full} & {B('0.9000')}{S} & {B('0.9800')}{S} & {B('100.0')}{S} & {B('200')}{S} \\",
                                   rf"{kin} & {B('0.9000')}{S} & {B('0.9800')}{S} & 40.0{S} & 80{S} \\",
                                   rf"{init} & 0.9000{A} & 0.9800{A} & 11.0\,$\pm$\,1.0 & 22\,$\pm$\,2 \\", r"\midrule",
                                   rf"{sophon} & 0.9000{S} & 0.9800{S} & 50.0{S} & 100{S} \\"]
    # a block whose column holds one value that can be bold has no bold: the 188-class run alone
    assert rows["quark/gluon tagging"] == [_unbold(ln) for ln in rows["top tagging"] if not ln.startswith(kin)]
    assert not any(M.BOLD_ON in ln for x in tasks if x != "top tagging" for ln in rows[x])
    # the substitution chosen on validation for our rows; the released checkpoint keeps its missing inputs zero
    fill = {"mtx-l188-s1": {"cells": {"best70": {"top.ntop": {"1200000": {"accuracy": 0.9, "auc": 0.98, "r50": 150.0, "r30": 300.0}}}}}}
    c = _blocks(M.table_probes_tasks(fits, {k: v for k, v in runs.items() if k != M.INIT}, M.TAB, ("L188", M.SOPHON), fill,
                                     {"top": "mean"}).split("%SPLIT%")[1], 5)[f"{M.HEADING['lp']}, top tagging"]
    assert c == [rf"{full} & 0.9000{S} & 0.9800{S} & 150{S} & 300{S} \\", r"\midrule", rf"{sophon} & 0.9000{S} & 0.9800{S} & 50{S} & 100{S} \\"]
    # set as the build sets it: upright, nothing cut, 4 pt between the columns, a size below the body's (the one
    # table set so), both mark sentences and the bold sentence in the caption
    f = M._float("probes", t, True, whole="upright", size=r"\small")
    assert "landscape" not in f and f.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}") == 2
    assert r"\setlength{\tabcolsep}{4pt}" in f and set(_sizes_in(f)) == {r"\small"} and f.count(r"\label{tab:probes}") == 1
    assert f.index("caption") < re.search(r"\\small(?![a-zA-Z])", f).start() and all(note in f for _, note in M.MARK_NOTES)
    assert not _sizes_in(M._float("probes", t, True, whole="upright"))
    assert f.count(sophon + " & ") == 5 and "(LP)" not in f and M.KEY_MARKS["probes"][0][1] not in f
    assert f.count(M.BOLD_NOTE.format("each block of a column")) == 1 and f.count(M.BOLD_ON) == 6


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
    assert paper[2] == r" & \multicolumn{2}{c}{top, $R_{30}$} & \multicolumn{2}{c}{quark/gluon, $R_{30}$} \\"
    assert not any(ln.startswith(M.LABEL["KIN"]) for ln in paper)
    tok = M.V4.tok
    assert paper[6] == M.LABEL["L188"] + " & " + " & ".join(tok("p", x, 1.0) for x in (101.0, 111.0, 201.0, 221.0)) + r" \\"
    whole = M.table_fill(None, fill, fits, runs, M.TAB, ("L188", M.INIT), drop=False).splitlines()
    assert whole[0] == r"\begin{tabular}{lcccccc}" and whole[4] == " & " + " & ".join(M.SUBST * 2) + r" \\"
    # the kinematic-only row after the pretrained rows and before the random network, its one value marked
    assert whole[6].startswith(M.LABEL["L188"] + " & ") and whole[8].startswith(M.LABEL[M.INIT] + " & ")
    assert whole[7] == M.LABEL["KIN"] + r" & \multicolumn{3}{c}{231" + M.SINGLE + r" (nothing missing)} & -- & -- & -- \\"


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
    # the soft-drop mass once, above both blocks: a property of the jets, no run and no mark
    assert lines[4] == "soft-drop mass & " + tok("o", 0.2448) + " & " + tok("o", 0.9294) + r" \\" and lines[5] == r"\midrule" and head == 6
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
    # the supplement's detail table reads the same estimator in its first section
    d = M.table_mass_detail(atlas, runs, M.TAB).splitlines()
    k = next(i for i, ln in enumerate(d) if M.HEADING["head"] in ln)
    assert d[k + 1].startswith(M.LABEL["L162_MASS"] + " & " + M._cell([0.1461, 0.1473], 2, M.TAB) + " & ")


def _paper_tables() -> dict:
    """{label: (caption key, generated file)} of the tables placed in the main text and the appendix."""
    G = _load("gallery_placement_tables_t", "experiments/FIGS/figstyle/gallery_placement.py")
    key = {label: k for k, (label, _) in M.CAPTIONS.items()}
    name = {k: f for k, f, *_ in M.PLACEHOLDERS}
    return {lab: (key[lab], ROOT / "paper/journal/tables" / f"{name[key[lab]]}.tex")
            for item, tier, _ in G.TABLES if tier in (G.MAIN, G.APPENDIX) for lab in item.split(" + ")}


def _caption_and_body(tex: str) -> tuple:
    m = re.search(r"\\caption(?:of\{table\})?\{", tex)
    i, depth = m.end(), 1
    while depth:
        depth += (tex[i] == "{") - (tex[i] == "}")
        i += 1
    return tex[m.end():i - 1], tex[:m.start()] + tex[i:]


def test_the_paper_has_seventeen_tables_and_the_task_tables_are_placed_nowhere():
    tables = _paper_tables()
    assert list(tables) == ["tab:summary", "tab:published", "tab:mass", "tab:xbx", "tab:probes", "tab:coarse", "tab:qcdhf",
                            "tab:fill", "tab:inputs", "tab:partrepro", "tab:lrsweep", "tab:jetclassrej", "tab:labelsets",
                            "tab:pairs", "tab:sculpt", "tab:sculptboot", "tab:xbxrej"]
    assert tables["tab:published"][1].name == "mlst_published.tex" and tables["tab:labelsets"][1].name == "mlst_labelsets.tex"
    assert all(k in M.CAPTION_SHORT for k, _ in tables.values())                   # each prints its short caption


def test_the_captions_of_the_paper_tables_are_sentences_that_name_the_networks_one_way():
    """PI 2026-11-12 (the caption brief): what the table compares, then definitions and run counts, in sentences: no
    colon-form, no semicolon chain; the networks by one name, never "Sophon's network"; one wording of the
    multi-class AUC."""
    for label, (key, _) in _paper_tables().items():
        cap = M.printed_caption(key)
        assert ": " not in cap and ";" not in cap, label
        assert "Sophon's network" not in cap and "networks pretrained on JetClass-II" not in cap, label
        assert not re.search(r"\((LP|MLP|f\.t\.)\)", cap), label
        assert not re.search(r"\bSD\b", cap) or key == "sculpt_boot", label       # "standard deviation", but for the column names there
        assert cap.rstrip().endswith(".") or cap.rstrip().endswith("__"), label
    assert M.PNETS == "the networks pretrained with 188, 162, 43 and 17 classes"
    assert all(M.PNETS in M.printed_caption(k) for k in ("summary", "probes", "coarse", "qcdhf", "fill", "pairs", "sculpt", "xbx",
                                                         "jetclass_rej"))
    # the run counts in one sentence, an item per kind of row
    assert M._runs("set", "flav") == (r" Values are the mean $\pm$ standard deviation over \MlstNRunsWord{} pretraining runs per "
                                      r"label set and \RandFlavRuns{} per flavor-boundary partition.")
    assert M._runs("set", "mass", "three training seeds from scratch on $X\\to bs$").endswith(
        r"label set, \MlstMassNRunsWord{} per network with an added mass output and three training seeds from scratch on $X\to bs$.")
    assert M.printed_caption("probes").endswith(M._runs("set", "flav", "mass", "rand", "init"))
    # one definition and wording of a multi-class AUC; on JetClass the sentence on ParT's average over class pairs
    assert M.AUC_OVR == ("AUC is the area under the receiver operating characteristic curve, averaged over the classes, each "
                         "against the rest.")
    assert M.AUC_OVR in M.printed_caption("probes") and "averaged over class pairs" not in M.printed_caption("published")
    assert "equals the average over class pairs that ParT reports" in M.printed_caption("published")
    assert "area under the receiver operating characteristic curve" in M.printed_caption("pairs")
    assert "averaged over" not in M.printed_caption("pairs") and "averaged over the target's groups, each against the rest" in M.printed_caption("coarse")
    # the definitions the brief asked for by name
    assert M.BN_DEF in M.printed_caption("fill") and M.BN_DEF_BOTH in M.printed_caption("pairs")
    assert not any("BatchNorm" in M.printed_caption(k) for k in ("probes", "coarse", "qcdhf"))
    assert "__SCULPT_CUT__" in M.CAPTION_SHORT["sculpt"] and "resonant" in M.sculpt_fill({"m": {"sig_eff_50": {}}}, {a: [] for a in M.FLAV})["__SCULPT_CUT__"]
    assert all(h in M.printed_caption("mass") for h in ("mass output of the pretraining head", "multilayer perceptron"))
    assert "which are not combined" in M.printed_caption("partrepro") and "cumulative" not in M.printed_caption("partrepro")


def test_the_generated_paper_tables_follow_the_table_rules():
    """The 17 tables as make_tables.py wrote them (paper/journal/tables, committed): every mark of a table has its
    sentence in the caption and no caption explains a mark the table lacks; never "<" or "+- 0"; no mode tag on a
    row; thousands separators; "parameters"; the full text width, no wrapped row label, nothing scaled, and the body
    size everywhere but tab:probes (PI 2026-11-12: the reported exception; tab:summary is at the body size on four
    turned pages). A bold value is read without its wrapper."""
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
            if mark in body:
                assert mark in notes and notes[mark].strip() in cap, (label, mark)
        assert ": " not in cap and ";" not in cap and "Sophon's network" not in tex, label
        assert "$<$" not in tex and not re.search(r"\\pm\$\\,0(\.0+)?\s*(&|\\\\)", _unbold(body)), label
        assert not re.search(r"\((LP|MLP|f\.t\.)\)", tex) and "# params" not in tex and "params" not in body, label
        assert not re.findall(r"(?<![\d.{,])\d{4,}(?![\d}])", _unbold(body)), label         # 1{,}602, never 1602
        assert r"\begin{tabular}" not in tex and r"arraybackslash}p{" not in tex, label      # tabular* or longtable, no p column
        assert tex.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}") + tex.count(
            r"\begin{longtable}{@{\extracolsep{\fill}}") >= 1, label
        assert not re.search(r"adjustbox|resizebox|scalebox|\\(footnotesize|scriptsize|tiny)(?![a-zA-Z])", tex), label
        if _sizes_in(tex):
            small.add(label)
    assert small == {"tab:probes"}
    read = lambda label: tables[label][1].read_text()      # noqa: E731
    # where the single-run mark is: the kinematic-only and Sophon rows, the reproduction runs, the learning-rate
    # sweep. From scratch on X->bb has its three seeds since 2026-11-13 (eb6c87e; it was one seed, marked), so the
    # resonance tables carry no mark: mean +- SD for both signals
    assert {lab for lab in tables if M.SINGLE in read(lab)} == {"tab:published", "tab:probes", "tab:partrepro", "tab:lrsweep"}
    scratch = next(ln for ln in read("tab:xbx").splitlines() if ln.startswith(M.SCRATCH_LABEL + " & "))
    assert r"\pm" in scratch.split(" & ")[1] and r"\pm" in scratch.split(" & ")[3] and M.SINGLE not in scratch
    # Table A10: "> 0.9999" for an AUC that rounds to 1.0000, AUC once in each header, the pooled-partition mark
    pairs = read("tab:pairs")
    assert M.NEAR_ONE in pairs and "1.0000" not in _caption_and_body(pairs)[1] and r"{c}{AUC} \\" in pairs and r"\rho" not in pairs
    # (a longtable writes its header twice, for its first page and for the pages after it)
    assert pairs.count("AUC") - _caption_and_body(pairs)[0].count("AUC") == 2 * pairs.count(r"\endhead")
    # the rows the brief moved out: BatchNorm recomputed only in tab:fill and tab:pairs, Sophon only in tab:probes
    bn, sophon = M.LABEL[M.INIT_BN] + " & ", M.LABEL[M.SOPHON] + " & "
    assert {lab for lab in tables if bn in read(lab)} == {"tab:fill", "tab:pairs"}
    assert {lab for lab in tables if sophon in read(lab)} == {"tab:probes"}
    # tab:mass: the soft-drop mass, the head block (two networks), the perceptron block (seven)
    mass = read("tab:mass").splitlines()
    head, mlp = (mass.index(M._full_row(3, M.HEADING[k], italic=True)) for k in ("head", "mlp"))
    assert mass[head - 2].startswith("soft-drop mass & ") and mlp - head == 4 and mass[mlp + 8] == r"\bottomrule"
    # tab:coarse: one table, three column groups
    assert read("tab:coarse").count(r"\begin{tabular*}") == 1 and read("tab:coarse").count(r"\multicolumn{2}{c}{") == 3
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
    assert r"\textbf" not in M.BOLD_ON and M.BOLD_SIGMA == 3.0
    # _render: an upper-case token kind is the lower-case one printed bold
    pm = r"0.50\,$\pm$\,0.01"
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
    assert M.NEVER_BOLD == (lab[M.INIT], lab[M.INIT_BN], lab[M.SOPHON])
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
        "mass": {"lower": (1,), "unity": (2,)}, "sculpt": {"lower": (1, 3, 6)}}
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
    # 0.51 is the best (published: no SD, zero); x is 0.01 from it, within 3 * 0.01; column 1 (training jets) is skipped
    assert _chosen_bold("jetclass_rej", _frame(whole, 2)) == {(r"ParT~\cite{part}", 2), ("x", 2)}
    # a table of blocks in its caption's words: these keys
    assert {k for k, v in M.BOLD.items() if v.get("blocks")} == {"summary", "published", "labelsets", "probes", "pairs", "mass"}
    assert set(M.BOLD) == {"summary", "published", "labelsets", "probes", "pairs", "mass", "sculpt", "jetclass_rej", "xbx",
                           "xbx_rej", "coarse", "qcdhf", "fill"}
    # no bold in the tables of inputs, recipes and differences
    assert not {"inputs", "partrepro", "lrsweep", "sculpt_boot"} & set(M.BOLD)


def test_a_published_value_takes_part_in_bold_with_the_standard_deviation_it_prints_and_a_count_never():
    """PI 2026-11-12: published rows take part by their printed value; where a published cell prints a standard
    deviation that is its SD, where it prints none its SD is zero. The parameters and the training jets are
    counts."""
    T, tok = M.TAB, M.V4.tok
    assert M.BOLD["published"] == {"blocks": True, "published": True, "skip": (1, 2)}
    rows = [r"A~\cite{a} & 2.14\,M & 1.2\,M & 691\,$\pm$\,15 & 0.9877 \\",         # the best R50
            r"B~\cite{b} & 2.38\,M & 1.2\,M & 640\,$\pm$\,10 & 0.9878 \\",         # 51 <= 3 sqrt(100 + 225) = 54.1; the best AUC
            r"C~\cite{c} & 366\,k & 1.2\,M & 397\,$\pm$\,7 & -- \\",               # 294 > 49.7
            r"D~\cite{d} & 1\,k & -- & 3{,}000 & 0.9877 \\",                       # see below
            rf"from scratch (this work) & 2.21\,M & 1.2\,M & {T(461.0, 4.0)} & {tok('s', 0.9878)} \\"]
    block = "\n".join(rows[:3] + rows[4:])
    # R50: 691 +- 15 and 640 +- 10. AUC: the published 0.9878 and our single run at 0.9878 (both SD zero, equal), not
    # 0.9877
    assert _chosen_bold("published", block) == {(r"A~\cite{a}", 3), (r"B~\cite{b}", 3), (r"B~\cite{b}", 4),
                                                ("from scratch (this work)", 4)}
    out = M._bold("published", block).split("\n")
    assert out[1] == r"B~\cite{b} & 2.38\,M & 1.2\,M & " + M._b(r"640\,$\pm$\,10") + " & " + M._b("0.9878") + r" \\"
    assert out[2] == rows[2] and M.V4.tok("S", 0.9878) in out[3] and T(461.0, 4.0) in out[3]
    # a published value without a printed SD has SD zero: 3,000 (thousands separator read) is the best alone
    assert _chosen_bold("published", "\n".join(rows)) == {(r"D~\cite{d}", 3), (r"B~\cite{b}", 4), ("from scratch (this work)", 4)}
    # plain numbers in the count columns are never bold, in the value columns they are
    assert _chosen_bold("published", "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\") == {("x", 3)}
    assert _chosen_bold("jetclass_rej", "x & 5 & 6 & 7 \\\\\ny & 4 & 5 & 6 \\\\") == {("x", 2), ("x", 3)}
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
    """_float (PI 2026-11-12): bold from the tokens before the column is rounded; one sentence in the caption, "of a
    column" or "of each block of a column"; none where no cell is bold or the table is not one of BOLD."""
    T, lab = M.TAB, M.LABEL
    col, blk = (M.BOLD_NOTE.format(x) for x in ("a column", "each block of a column"))
    assert col == " Bold marks the best value of a column and every value within three standard deviations of it."
    assert blk == " Bold marks the best value of each block of a column and every value within three standard deviations of it."
    rows = [f"{lab['L188']} & {T(100.0, 2.0)} \\\\", f"{lab['L162']} & {T(95.0, 1.0)} \\\\", f"{lab['R42_Q1']} & {T(93.0, 1.0)} \\\\",
            f"{lab[M.INIT]} & {T(200.0, 1.0)} \\\\"]
    f = M._float("xbx", _frame(rows, 1))
    assert f.count(col) == 1 and blk not in f and r"\textbf" not in f and M.V4.TOK_OPEN not in f
    assert lab["L188"] + " & " + M._b(r"100.0\,$\pm$\,2.0") + r" \\" in f and lab["L162"] + " & " + M._b(r"95.0\,$\pm$\,1.0") + r" \\" in f
    assert lab["R42_Q1"] + r" & 93.0\,$\pm$\,1.0 \\" in f and lab[M.INIT] + r" & 200.0\,$\pm$\,1.0 \\" in f
    assert f.count(M.BOLD_ON) == 2 and f.index(col) < f.index(M.BOLD_ON)          # the sentence is the caption's
    # a table of blocks; its soft-drop row is no run's
    head, mlp = (M._full_row(2, M.HEADING[k], italic=True) for k in ("head", "mlp"))
    mass = _frame([head, f"{lab['L162_MASS']} & {T(0.10, 0.01)} \\\\", f"{lab['R16_Q1_MASS']} & {T(0.20, 0.01)} \\\\", r"\midrule",
                   mlp, f"{lab['L188']} & {T(0.30, 0.01)} \\\\", f"{lab['L162']} & {T(0.50, 0.01)} \\\\"], 1)
    g = M._float("mass", mass)
    assert g.count(blk) == 1 and col not in g and g.count(M.BOLD_ON) == 2
    assert lab["L162_MASS"] + " & " + M._b(r"0.100\,$\pm$\,0.010") + r" \\" in g and lab["L188"] + " & " + M._b(r"0.300\,$\pm$\,0.010") + r" \\" in g
    # nothing to compare: no bold cell, no sentence
    lone = M._float("xbx", _frame(rows[:1] + rows[3:], 1))
    assert M.BOLD_ON not in lone and "Bold marks" not in lone
    # a table that is not one of BOLD (the supplement's copies, the recipe tables)
    assert "probes_full" not in M.BOLD
    other = M._float("probes_full", _frame(rows, 1))
    assert M.BOLD_ON not in other and "Bold marks" not in other and lab["L188"] + r" & 100.0\,$\pm$\,2.0 \\" in other


def test_table_one_sets_each_tasks_auc_beside_its_metric_in_eight_columns_a_mode():
    """tab:summary (PI 2026-11-12): JetClass-II accuracy, AUC; JetClass accuracy, AUC; top R30, AUC; quark/gluon R30,
    AUC, for the linear probes and for fine-tuning, as the two panels of a task in Figs. 2 and 3. The AUC is the
    average over the classes, each against the rest, on JetClass-II and JetClass and the binary AUC on top and
    quark/gluon. The eight columns of a mode are wider than a turned page at the body size: two tasks a page."""
    assert M.SUMMARY_AUC == {"jc2": "macro_auc_ovr", "jc1": "macro_auc_ovr", "top": "auc", "qg": "auc"}
    assert [(k, m) for k, _, _, m in M.SUMMARY_GRID] == [("jc2", "accuracy"), ("jc2", "macro_auc_ovr"), ("jc1", "accuracy"),
                                                         ("jc1", "macro_auc_ovr"), ("top", "r30"), ("top", "auc"),
                                                         ("qg", "r30"), ("qg", "auc")]
    # an AUC column is its task's panel (training set, probe data set) with the other metric
    panel = {k: (n, ds) for k, n, ds, _ in M.SUMMARY_PANELS}
    assert all((n, ds) == panel[k] for k, n, ds, _ in M.SUMMARY_GRID)
    assert M.SUMMARY_GRID[0::2] == M.SUMMARY_PANELS and len(M.SUMMARY_GRID) == 8
    assert M.SUMMARY_PAGES == (("jc2", "jc1"), ("top", "qg")) and [k for pg in M.SUMMARY_PAGES for k in pg] == [k for k, *_ in M.SUMMARY_PANELS]
    assert M.SUMMARY_TASKS == {"jc2": "JetClass-II", "jc1": "JetClass", "top": "top tagging", "qg": "quark/gluon tagging"}
    assert M.SUMMARY_HEAD == {"accuracy": "accuracy", "r30": "$R_{30}$", "macro_auc_ovr": "AUC", "auc": "AUC"}
    assert [M.SUMMARY_HEAD[m] for *_, m in M.SUMMARY_GRID] == ["accuracy", "AUC", "accuracy", "AUC", "$R_{30}$", "AUC", "$R_{30}$", "AUC"]
    assert not hasattr(M, "SUMMARY_LABEL_CM")                  # the row labels never wrap: no label width
    # the caption: the multi-class AUC in the one wording of every table, once, for the two tasks it applies to
    cap = M.printed_caption("summary")
    assert cap.count(M.AUC_OVR) == 1 and "On JetClass-II and JetClass, " + M.AUC_OVR in cap and "class pairs" not in cap
    assert "$R_{30}$ is the inverse background efficiency at 30\\% signal efficiency." in cap
    assert M.BOLD["summary"] == {"blocks": True} and "summary" not in M.MODES
    # the generated table: four whole turned pages at the body size under one caption, each later page starting a
    # page under "Table 1 (continued)"; the probes' two pages, then the fine-tuned ones
    tex = (ROOT / "paper/journal/tables/mlst_summary.tex").read_text()
    tabs = re.findall(r"\\begin\{tabular\*\}.*?\\end\{tabular\*\}", tex, re.S)
    assert tex.startswith(r"\begin{landscape}") and len(tabs) == 4 and r"\begin{longtable}" not in tex and not _sizes_in(tex)
    assert tex.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}") == 4 and r"\setlength{\tabcolsep}{6pt}" in tex
    assert tex.count(r"\captionof{table}{") == tex.count(r"\label{tab:summary}") == 1
    assert tex.count(r"\textit{Table~\ref{tab:summary} (continued)}") == tex.count(r"\newpage") == 3
    assert tex.count(M.AUC_OVR) == 1 and tex.count(M.BOLD_NOTE.format("each block of a column").strip()) == 1
    heads = []
    for tab, (mode, tasks) in zip(tabs, [(mode, tasks) for mode in ("lp", "ft") for tasks in M.SUMMARY_PAGES], strict=True):
        lines = tab.splitlines()
        assert lines[2] == " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{M.SUMMARY_TASKS[k]}}}" for k in tasks) + r" \\"
        assert lines[4] == " & " + " & ".join(M.SUMMARY_HEAD[m] for k, _, _, m in M.SUMMARY_GRID if k in tasks) + r" \\"
        assert lines[6] == M._full_row(5, M.HEADING[mode], italic=True) and tab.count(r"\textit{") == 1
        last = lines[-3].split(" & ")
        assert last[0] == (M.LABEL[M.INIT] if mode == "lp" else M.SCRATCH_LABEL) and len(last) == 5
        assert mode == "ft" or M.BOLD_ON not in lines[-3]                  # the random network: a reference row
        heads += lines[4].removesuffix(r" \\").split(" & ")[1:]
    assert len(heads) == 16 and heads[1::2] == ["AUC"] * 8             # eight value columns a mode, every second an AUC


def test_the_generated_tables_are_bold_where_the_rule_applies_and_say_so_once():
    """The 17 tables as make_tables.py wrote them (PI 2026-11-12): bold in the tables that compare networks and in no
    table of inputs, recipes or differences; the caption's sentence once, in the words for a table with blocks or
    without; a bold wrapper holds a value and its standard deviation and nothing else (no mark, no dash, no bound);
    never \\textbf; no bold in a reference row or a count column; the columns where lower or closest to 1 is best are
    the ones their headers name."""
    tables = _paper_tables()
    plain = {"tab:inputs", "tab:partrepro", "tab:lrsweep", "tab:sculptboot"}
    notes = [M.BOLD_NOTE.format(x).strip() for x in ("a column", "each block of a column")]
    refs = M.NEVER_BOLD + ("soft-drop mass", "random selection")
    for label, (key, path) in tables.items():
        tex = path.read_text()
        cap, body = _caption_and_body(tex)
        assert r"\textbf" not in tex and M.BOLD_ON not in cap, label
        assert (M.BOLD_ON in body) == (key in M.BOLD) == (label not in plain), label
        want = notes[bool(M.BOLD[key].get("blocks"))] if key in M.BOLD else None
        assert [cap.count(n) for n in notes] == [int(n == want) for n in notes], label
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
    # Table A10: the X->ee against mumu column, whose best prints as a bound, has no bold (wherever the table is cut)
    ee = next(h for k, h, *_ in M.PAIRS if k == "ee_vs_mm")
    cut = [ln.removesuffix(r" \\").split(" & ") for ln in read("tab:pairs").splitlines() if " & " in ln]
    header = next(r for r in cut if ee in r)
    column = [r[header.index(ee)] for r in cut if len(r) == len(header) and r[0].strip() and ee not in r]
    assert len(column) > 10 and any(c.startswith(M.NEAR_ONE) for c in column) and not any(M.BOLD_ON in c for c in column)
