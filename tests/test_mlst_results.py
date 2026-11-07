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
    assert M._cell([0.1, None, 0.3], 3, fmt) == r"0.10, 0.30 \pending{2 of 3}"
    assert M._cell([None, None], 2, fmt) == r"\pending{no run yet}"


def test_two_runs_print_both_values_to_the_digit_of_their_difference():
    # never beyond four significant figures (2026-10-31), even where the difference is smaller
    assert M._cell([1.1661, 1.1652], 2, MT.fmt_pm) == "1.166, 1.165"
    assert M._cell([22.515, 22.467], 2, MT.fmt_pm, sep=" and ") == "22.52 and 22.47"
    assert M._cell([0.4123, 0.4161], 2, MT.fmt_pm) == "0.4123, 0.4161"
    assert M._cell([0.5, 0.5], 2, MT.fmt_pm) == "0.500, 0.500"


def test_a_lone_run_prints_three_significant_figures():
    assert M._cell([0.86300], 3, MT.fmt_pm) == r"0.863 \pending{1 of 3}"
    assert M._cell([0.96579654592], 1, MT.fmt_pm) == "0.966"
    assert M._cell([0.0593], 1, MT.fmt_pm, 1e3) == "59.3"
    assert M._cell([1802.83], 1, MT.fmt_pm) == "1{,}800" and M._cell([42589.4], 1, MT.fmt_pm) == "42{,}600"


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
        "perf_vs_pt", "confusion_jc1"}


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
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS) - 1 and r"\pending{2 of 3}" in t and "from scratch" in t
    assert r"\begin{longtable}" in t and r"\label{tab:ft}" in t and "%SPLIT%" not in t
    assert "1{,}200{,}000" in t and "1{,}600{,}000" in t
    # each task in its own metric (PI 2026-11-02, no 1 - AUC outside JetClass-II): JetClass-II the AUC
    # (0.90 and 0.91), top/q-g R30 (300 x AUC: 270 and 273)
    assert "0.900, 0.910" in t and "270.0, 273.0" in t and "100, 90" not in t
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
    jc1 = t.split(M.BLOCK_SEP)[2]
    assert "{JetClass} \\\\" in jc1 and r"training jets & \pending{not yet read out}" in jc1
    assert t.count(M.BLOCK_SEP) == len(M.FT_TASKS) - 1
    assert "from scratch" in t


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
    # bold: the best of each column (C1.14); here ParT's 2 M row has the best accuracy and AUC
    assert r"ParT~\cite{part} & 2 M & \textbf{0.836} & \textbf{0.9834}" in tex and "& 100 M &" not in tex
    scratch = next(line for line in tex.splitlines() if line.startswith("from scratch"))
    assert MT.fmt_pm(0.802, 0.001) in scratch and " & 1 M & " in scratch
    assert r"$R_{99}$" in tex and r"$R_{99.5}$" in tex
    assert M.LABEL["R16_Q1"] not in tex          # no curves for it: no row
    # 200 // 9 = 22 QCD jets pass for the last column: unmarked; 200 // 10 would be marked
    assert scratch.count(r"$^\dagger$") == 0
    pre["l188-s1"]["N1000000"]["s1"]["jc1"]["per_class"]["label_Zqq"]["rejection"]["n_bkg_pass"] = 19
    tex = V4.table_jetclass(pre, ref, runs, ("L188",), M.LABEL, M.SCRATCH, 1_000_000, MT.fmt_pm, M._two)
    row = next(line for line in tex.splitlines() if line.startswith(M.LABEL["L188"]))
    assert row.rstrip(" \\").endswith(r"$^\dagger$") and row.count(r"$^\dagger$") == 1
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


def test_the_top_table_footnote_traces_the_gap_to_part():
    n = M.partrepro_note()
    assert all(rf"\MlstPartreproRthirty{v}{{}}" in n for v in ("Part", "Head", "Rescale"))
    assert r"1{,}602\,$\pm$\,81" in n and r"\MlstFtTopRthirtyScratch{}" in n and r"\ref{tab:partrepro}" in n


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
    assert M.LABEL["L162_MASS"] + " & 0.150, 0.150 & 1.00, 1.00" in t
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
    assert "200--400 & 400--2500" in d and "soft-drop mass & all & " in d and "soft-drop mass & QCD & " in d
    assert "estimator & pretraining & all & resonant & QCD & all & resonant & QCD" in d
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


def test_the_benchmark_tables_show_the_planned_rows_and_bold_within_one_sd():
    runs = M._paper(ROOT)
    per = {"top": _bench_per(runs), "qg": _bench_per(runs), "qg_herwig": _bench_per(runs)}
    t = M.V4.table_bench("top", per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two,
                         kin=("kin (diagnostic)", M.KIN_INIT))
    assert "LorentzNet" not in t and "OmniLearned-s" not in t and r"ParT-f.t.~\cite{part}" in t
    assert r"\textbf{3{,}486\,$\pm$\,157}" in t and r"1{,}615\,$\pm$\,93" in t
    assert r"kin (diagnostic) & " in t and "inputs set to zero" not in t and r"from scratch & 0." in t
    assert M.LABEL["R16_Q1"] + r" & --" in t        # no run read out: dashes
    q = M.V4.table_bench("qg", per, runs, M.MAIN, M.LABEL, M.SCRATCH, MT.fmt_pm, M._two, herwig=True)
    assert r"\multicolumn{4}{c}{Herwig}" in q and "PFN-Ex" in q and q.count(r"\cite{") == 6


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
    assert "ParT, published" in t and "1{,}224" in t and not missing


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
    assert [ln.count(r"$^{\ast}$") for ln in rows if ln.startswith("from scratch")] == [0, 1, 1]
    assert next(ln for ln in rows if ln.startswith(r"from scratch & \ensuremath{10^{-3}}")).split(" & ")[2].endswith(r"$^{\ast}$")
    assert not any(r"$^{\ast}$" in ln.split(" & ")[-1] for ln in rows if ln.startswith("from scratch"))


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
    assert next(ln for ln in blocks[3].splitlines() if ln.startswith("random partition")).split(" & ")[2] == M._lone(1.25)
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


def test_the_share_of_the_jetclass_gap_the_partitions_recover():
    runs = M._paper(ROOT)
    rej = lambda m, c: (800.0 if "r16q1" in m else 900.0 if "rand2" in m else 1000.0, 200)
    pre = _jc1_pre(runs, rej)
    for arm in M.RAND:
        for m in runs[arm]:
            pc = {c: {"rejection": dict(zip(("rejection", "n_bkg_pass"), rej(m, c)))} for c, _ in M.V4.JC1_COLS}
            pre[m[len("mtx-"):]] = {"N1000000": {"s1": {"jc1": {"accuracy": 0.84, "per_class": pc}}}}
    em = _PEm()
    M.jetclass_macros(em, pre, runs, None, MT.fmt_pm, MT.words)
    assert em.m["MlstFtJcOneRandGapShareWZ"] == "about half"
    assert M._gap_share([0.54, 0.56]) == "about half" and M._gap_share([0.2, 0.56]) == r"20--56\%"
    assert M._gap_share([]).startswith(r"\pending")


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
    assert r"electric charge & --- & $\bullet$" in t and r"$\sigma_{d_0}$ & --- & ---" in t
    assert r"$\Delta\eta$ & $\bullet$ & $\bullet$ & -- &" in t and r"\pending{offsets}" in t
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
    first = t.split("%SPLIT%")[0]
    assert r"$X\to ee$ vs $\mu\mu$" in first and first.count(r"\multicolumn{2}") == 4
    assert r"\Kmid{} classes & " in first and r"0.960$^{\dagger}$ & -- \\" in first   # no ratio to a perfect separation
    assert M.ZERO_NOTE and M.ZERO_NOTE in M._float("pairs", t, True, fill={"__ZERO__": M.zero_note(fits, runs)})
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
    assert M.ZERO_NOTE not in M._float("pairs", t, True, fill={"__ZERO__": M.zero_note(fits, runs)})


def test_the_jetclass2_fine_tuning_block_has_the_ratio_column_at_one_million_jets(tmp_path):
    _ft_tree(tmp_path, ["l188-s1", "r16q1-s1", "l188-s2"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"], "R16_Q1": ["mtx-r16q1-s1"]}, MT.fmt_pm, ("L188", "R16_Q1"))
    jc2, jc1 = t.split(M.BLOCK_SEP)[0], t.split(M.BLOCK_SEP)[2]
    assert r"$\rho$, $10^{6}$ jets" in jc2 and r"\rho" not in jc1
    # (1 - 0.91) / (1 - 0.90) run by run; the 188-class row 1; from scratch has none
    assert r"\Kcoarse{} classes & 0.910 & 0.910 & 0.900 \\" in jc2 and r"\Kfull{} classes & 0.900 & 0.900 & 1 \\" in jc2
    assert jc2.splitlines()[-1].startswith("from scratch") and jc2.splitlines()[-1].endswith(r"& -- \\")
    # a 188-class run without its 10^6 cell: the 188-class ratio cell says how many runs it rests on
    del per["jc2"]["l188-s2"][1000000]
    two = M.table_ft(per, {"L188": ["mtx-l188-s1", "mtx-l188-s2"]}, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)[0]
    assert r"\Kfull{} classes & 0.900, 0.920 & 0.900 \pending{1 of 2} & 1 \pending{1 of 2} \\" in two
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
    assert "eq:discr" in cap["jetclass"] and "p_c" not in cap["jetclass"] and r"1-g_{\mathrm{QCD}}" in cap["sculpt"]
    assert r"\MlstMassMinJets" in cap["mass_detail"] and all(
        r"$\nmin$" in cap[k] and "eq:cut" in cap[k] and "__MINBKG__ " not in cap[k] for k in ("jetclass", "rand_prong"))
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


def test_the_probes_against_size_pool_the_random_partitions_and_add_the_controls_and_the_kinematic_model():
    """tab:probesn (2026-11-07): the flavour-controlled partitions (two runs, both values), the five random
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
    assert [r.split(" & ")[1].removesuffix(r" \\") for r in rows] == [
        M._cell([f(v["mtx-flavf0-s1"]), f(v["mtx-flavf0-s2"])], 2, MT.fmt_pm) for f in (lambda x: x, lambda x: 100 * x)]
    pooled = [v[runs[a][0]] for a in M.RAND]
    jc2 = next(ln for ln in t.splitlines() if ln.startswith("random partitions (all five)"))
    assert jc2.split(" & ")[1].removesuffix(r" \\") == M._cell(pooled, 5, MT.fmt_pm)
    assert not any(ln.startswith("random partition 1") for ln in t.splitlines())     # pooled, not one by one
    kin = [ln for ln in t.splitlines() if ln.startswith(M.KIN_ROW)]
    assert kin == [M.KIN_ROW + " & 133 \\\\"]                                     # top only, three figures
    assert M.table_probes_n(fits, M._pooled(runs), MT.fmt_pm, ("L188",)).count(M.KIN_ROW) == 0


def test_the_kinematic_model_joins_the_top_rows_of_the_probe_mlp_and_fine_tuning_tables(tmp_path):
    jc2 = {"converged": True, "accuracy": 0.9, **{k: 0.95 for k in ("macro_auc_ovr", "macro_auc_resonant",
                                                                     "macro_auc_qcd")}}
    top = {"accuracy": 0.9163, "auc": 0.9713, "r50": 94.18, "r30": 231.02, "r50_is_bound": False, "r30_is_bound": False}
    fits = {"mtx-l188-s1": {"cells": {"best70": {"jc2_188": {"1000": jc2}, "top": {"1000": top}}}},
            M.KIN_MODEL: {"cells": {"best70": {"top": {"1200000": top}}}}}
    runs = {"L188": ["mtx-l188-s1"]}
    t = M.table_probes(fits, runs, MT.fmt_pm, ("L188",), kin=True)
    lower = t.split("%SPLIT%")[1]
    assert lower.count(M.KIN_ROW) == 1 and t.split("%SPLIT%")[0].count(M.KIN_ROW) == 0
    assert M.KIN_ROW + " & 0.916 & 0.971 & 94.2 & 231 & " + " & ".join(["--"] * 8) + r" \\" in lower
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
    row = next(ln for ln in t.splitlines() if ln.startswith(M.KIN_ROW))
    assert row == M.KIN_ROW + " & " + " & ".join(["--"] * 5 + ["0.981", "716"] + ["--"] * 4) + r" \\"
    assert t.count(M.KIN_ROW) == 1 and M.KIN_ROW not in M.table_mlp(mlp, runs, MT.fmt_pm, ("L188",))
    # fine-tuning: a row after from scratch in the task where it has cells, top
    _ft_tree(tmp_path / "ft", ["l188-s1"])
    _, per, _ = M.load_ft(tmp_path / "ft")
    per["top"][M.KIN_INIT] = {1000: [{"r30": 92.4, "r30_is_bound": False}], 1200000: [{"r30": 1802.8, "r30_is_bound": False}]}
    blocks = M.table_ft(per, runs, MT.fmt_pm, ("L188",)).split(M.BLOCK_SEP)
    topb = next(b for b in blocks if "top, " in b)
    lines = topb.splitlines()
    assert lines.index(next(ln for ln in lines if ln.startswith(M.KIN_ROW))) == \
        lines.index(next(ln for ln in lines if ln.startswith("from scratch"))) + 1
    assert M.KIN_ROW + " & 92.4 & 1{,}800 \\\\" in topb
    assert sum(b.count(M.KIN_ROW) for b in blocks) == 1


def test_the_pair_table_gives_the_random_network_both_ways_without_a_ratio():
    cell = lambda auc: {"auc": auc}
    pairs = lambda auc: {M.PAIR_N: {k: cell(auc) for k, *_ in M.PAIRS}}
    fits = {"mtx-l188-s1": {"cells": {"best70": {M.PAIR_DS: pairs(0.95)}}},
            **{m: {"cells": {"init": {M.PAIR_DS: pairs(0.6 + 0.01 * k)}}} for k, m in enumerate(M.INIT_MODELS)},
            **{m: {"cells": {"init_bn": {M.PAIR_DS: pairs(0.7)}}} for m in M.INIT_BN_MODELS}}
    runs = {"L188": ["mtx-l188-s1"], M.INIT: list(M.INIT_MODELS), M.INIT_BN: list(M.INIT_BN_MODELS)}
    t = M.table_pairs(fits, runs, MT.fmt_pm, M.merge_status(ROOT), ("L188", M.INIT, M.INIT_BN))
    for part in t.split("%SPLIT%"):
        init = next(ln for ln in part.splitlines() if ln.startswith(M.LABEL[M.INIT] + " & "))
        bn = next(ln for ln in part.splitlines() if ln.startswith(M.LABEL[M.INIT_BN] + " & "))
        assert init.split(" & ")[1] == M._cell([0.6 + 0.01 * k for k in range(3)], 3, MT.fmt_pm)
        assert bn.split(" & ")[1] == M._cell([0.7] * 3, 3, MT.fmt_pm)
        assert all(c.strip(" \\") == "--" for c in init.split(" & ")[2::2] + bn.split(" & ")[2::2])


def test_the_task_longtables_keep_each_heading_with_its_first_row(tmp_path):
    """tab:ft and tab:probesn (2026-11-07) are longtables: a block per task, its title and heading unbreakable
    from its first row (\\* and a rule with no break around it), the rule spanning every column."""
    _ft_tree(tmp_path, ["l188-s1"])
    _, per, _ = M.load_ft(tmp_path)
    t = M.table_ft(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))
    lines = t.splitlines()
    ncol = int(next(ln for ln in lines if ln.startswith(r"\multicolumn")).split("}")[0].split("{")[1])
    assert r"\begin{longtable}{>{\raggedright\arraybackslash}p{4cm}" + "c" * (ncol - 1) + "}" in lines
    titles = [i for i, ln in enumerate(lines) if ln.startswith(r"\multicolumn") and "continued" not in ln]
    assert len(titles) == len(M.FT_TASKS)
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
