"""The MLST tables' code (experiments/FIGS/mlst_results.py, read by make_tables.build()):
mean +- sample SD over the runs present with a marker for missing runs, a lone run with its
trailing zeros, the probe convergence rule, the merge marks from the label maps, and the
pulled data checked against its index."""
import csv
import importlib.util
import json
import pathlib
import math
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
                           **{k: "merged" for k in qcd}}
    assert set(s["R16_Q1"].values()) == {"merged"}
    assert all(s[a][k] == "merged" for a in M.RAND + M.FLAV for k in qcd)
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
        "linprobe/fits_v4_fill", "xbx_scratch"}


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
    assert t.count("%SPLIT%") == len(M.FT_TASKS) - 1 and r"\pending{2 of 3}" in t and "from scratch" in t
    assert "1{,}200{,}000" in t and "1{,}600{,}000" in t
    # each task in its own metric (PI 2026-11-02, no 1 - AUC outside JetClass-II): JetClass-II the AUC
    # (0.90 and 0.91), top/q-g R30 (300 x AUC: 270 and 273)
    assert "0.900, 0.910" in t and "270.0, 273.0" in t and "100, 90" not in t
    body, fewest = M.table_ft_rej(per, runs, MT.fmt_pm, ("L188",))
    assert fewest == 120 and r"\pending{2 of 3}" in body and body.count(r"\\") == 4
    per["jc2"]["l188-s1"][1000].append(per["jc2"]["l188-s1"][1000][0])
    with pytest.raises(SystemExit, match="fine-tuning seeds"):
        M.table_ft(per, runs, MT.fmt_pm, ("L188",))


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
    assert "JetClass & \\pending{not yet read out}" in t and t.count("%SPLIT%") == len(M.FT_TASKS) - 1
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
    assert "groomed mass & 0.200 & 1.05 \\\\" in t and "sigma" not in t and "0.300" not in t
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
    assert "200--400 & 400--2500" in d and "groomed mass & all & " in d and "groomed mass & QCD & " in d
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
    assert em.m["MlstSculptDiffEffLo"] == em.m["MlstSculptDiffEffHi"] == r"1\%"
    assert em.m["MlstSculptLowestEff"] == r"0.1\%" and em.m["MlstSculptNAtFloorLowest"] == "6"
    assert em.m["MlstSculptCoarseOverFullOnePr"] == "2.0 times the distance"
    assert em.m["MlstSculptFullOne"].startswith("0.020")
    assert "MlstSculptCoarseMassOverCoarseOnePrMax" in em.m


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
                         kin=("kin (diagnostic)", M.KIN_INIT), zero=r"\FillTopNZero{}")
    assert "LorentzNet" not in t and "OmniLearned-s" not in t and r"ParT-f.t.~\cite{part}" in t
    assert r"\textbf{3{,}486\,$\pm$\,157}" in t and r"1{,}615\,$\pm$\,93" in t
    assert r"kin (diagnostic) & none & " in t and r"from scratch & \FillTopNZero{} & " in t
    assert M.LABEL["R16_Q1"] + r" & \FillTopNZero{} & --" in t        # no run read out: dashes
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
    fits.update({f"mtx-l162-s{k}": cell(c) for k, c in ((1, 0.020), (2, 0.022), (3, 0.0231 * 1.2))})
    em = _PEm()
    M.V4.gpu_bound_macro(em, fits, runs, comp, None)
    assert em.m["MlstGpuTypeBound"] == "20.0\\%"            # the 162-class ratio departs most
    comp["mtx-l188-s3"] = dev("V100")
    em = _PEm()
    M.V4.gpu_bound_macro(em, fits, runs, comp, None)
    assert em.m["MlstGpuTypeBound"].startswith(r"\pending")


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


def test_the_inputs_table_reads_which_inputs_each_data_set_lacks():
    t = M.table_inputs(ROOT, None)
    assert r"electric charge & --- & $\bullet$" in t and r"$\sigma_{d_0}$ & --- & ---" in t
    assert r"$\Delta\eta$ & $\bullet$ & $\bullet$ & -- &" in t and r"\pending{offsets}" in t
    assert t.count(r" \\") == 17 + 1


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
    assert em.m["MlstPerClassRandFlavMergedOverApart"].startswith(r"\pending")
