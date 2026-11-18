"""The reruns of the ROC and class-pair inputs (PI 2026-11-18): the pair tables' R50 and its lower-bound rule
(mlst_results.pair_cell, pair_r50), the jobs that refuse a fine-tuning cell that is not DONE (build_head_jobs
--ft-head-fixed, build_gallery_data_job.py's rerun modes), and mlst_v4.rerun_family, which replaces a ROC family by
its rerun whole and with its group, never model by model."""
import importlib.util
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("mlst_results_rr", "experiments/FIGS/mlst_results.py")
MT = _load("make_tables_mlst_rr", "experiments/FIGS/make_tables.py")
HB = _load("build_head_jobs_rr", "scripts/build_head_jobs.py")
GD = _load("build_gallery_data_job_rr", "scripts/build_gallery_data_job.py")


# ------------------------------------------------------------------ R50 and the lower-bound rule
def test_r50_is_measured_from_20_passing_background_jets_and_is_a_lower_limit_below():
    """A run with fewer than PAIR_MIN_PASS (the figures' 20) background jets passing at 50% signal efficiency gives
    the limit n_background / 20; a cell with such a run prints '>' and the smallest of its runs' values and limits,
    with no spread; every run measured prints _cell's mean +- SD."""
    assert (M.PAIR_EPS, M.PAIR_MIN_PASS, M.PAIR_GT) == (0.5, 20, r"$>$") and M.PAIR_MIN_PASS == M.V4.JC1_MIN_BKG
    assert M.pair_cell([(500.0, 400, 200_000), (520.0, 385, 200_000)], 2, MT.fmt_pm) == M._cell([500.0, 520.0], 2, MT.fmt_pm)
    # none passes (a fit's 1e12 sentinel) or 19 pass: each run the limit 200,000 / 20
    assert M.pair_cell([(1e12, 0, 200_000), (200_000 / 19, 19, 200_000)], 2, MT.fmt_pm) == r"$>$10{,}000"
    # a measured run below the other's limit sets the cell's limit; a curve's null rejection is a run with none passing
    assert M.pair_cell([(None, 0, 200_000), (8_000.0, 25, 200_000)], 2, MT.fmt_pm) == r"$>$8{,}000"
    # a single run is marked, a missing run flagged, a cell with no run pending
    assert M.pair_cell([(None, 0, 128_833)], 1, MT.fmt_pm) == r"$>$6{,}442" + M.SINGLE
    assert M.pair_cell([(None, 0, 200_000), None], 2, MT.fmt_pm) == r"$>$10{,}000" + M.SINGLE + r" \pending{1 of 2}"
    assert M.pair_cell([None, None], 2, MT.fmt_pm) == M._cell([], 2, MT.fmt_pm)


def test_a_pair_curve_gives_its_r50_at_the_grid_point_of_50_percent():
    doc = {"eps_grid": [0.3, 0.5, 0.7],
           "curves": {"bb_vs_cc": {"rejection": [900.0, None, 50.0], "n_bkg_pass": [40, 0, 900], "n_background": 18_000},
                      "qcd_bb_vs_cc": {"rejection": [90.0, 30.0, 5.0], "n_bkg_pass": [600, 1800, 9000],
                                       "n_background": 54_000}}}
    assert M.pair_r50(doc, "bb_vs_cc") == (None, 0, 18_000) and M.pair_r50(doc, "qcd_bb_vs_cc") == (30.0, 1800, 54_000)
    assert M.pair_r50(doc, "ee_vs_mm") is None
    with pytest.raises(SystemExit, match="no point"):
        M.pair_r50({**doc, "eps_grid": [0.3, 0.55, 0.7]}, "bb_vs_cc")


def test_a_column_with_a_lower_limit_has_no_bold_and_the_caption_explains_the_mark():
    rec = lambda r, k: {"auc": 0.9, "r50": r, "r50_n_bkg_pass": k, "n_test_background": 200_000}      # noqa: E731
    one = lambda r, k: {"cells": {"best70": {M.PAIR_DS: {M.PAIR_N: {p: rec(r, k) for p, *_ in M.PAIRS}}}}}  # noqa: E731
    fits = {"mtx-l188-s1": one(1e12, 0), "mtx-l162-s1": one(900.0, 222), "mtx-r42q1-s1": one(400.0, 500)}
    runs = {"L188": ["mtx-l188-s1"], "L162": ["mtx-l162-s1"], "R42_Q1": ["mtx-r42q1-s1"]}
    status = {a: {p: "apart" for p, *_ in M.PAIRS} for a in runs}
    raw = M.table_pairs(fits, runs, M.TAB, status, ("L188", "L162", "R42_Q1"))
    row = lambda t, lab: next(ln for ln in t.splitlines() if ln.startswith(lab + " & ")).split(" & ")      # noqa: E731
    assert row(raw, M.LABEL["L188"])[1] == r"$>$10{,}000" + M.SINGLE
    assert M.V4.TOK_OPEN in row(raw, M.LABEL["L162"])[1]
    # the best of every column is a limit, not a value: no bold
    assert M.BOLD_ON not in M.uniform_columns(M._bold("pairs", raw))
    # without the limit the larger R50 is bold
    measured = M.table_pairs(fits, runs, M.TAB, status, ("L162", "R42_Q1"))
    assert M.BOLD_ON in M.uniform_columns(M._bold("pairs", measured))
    tab = M._float("pairs_full", raw, True)
    assert M.PAIR_GT_NOTE in tab and "$~marks " in M.PAIR_GT_NOTE and ":" not in M.PAIR_GT_NOTE
    assert M.PAIR_GT_NOTE not in M._float("pairs_full", measured, True)
    assert all((M.PAIR_GT, M.PAIR_GT_NOTE) in M.KEY_MARKS[k] for k in ("pairs", "pairs_full"))


# ------------------------------------------------------------------ the jobs refuse cells that are not DONE
def test_the_fixed_head_job_refuses_each_cell_until_it_is_done():
    cells = HB.ft_head_fixed_cells()
    assert [m for m, _c in cells] == [f"mtx-{r}" for r in HB.BFB.paper_runs()] + ["scratch-s1", "scratch-s2", "scratch-s3"]
    assert all(c.startswith("/data/results/ft_v2/fixed/leg_jc2b/") and "/N1000000/s" in c for _m, c in cells)
    d = yaml.safe_load(HB.ft_head_fixed_spec("TAG_TBD", 90))
    assert d["metadata"]["name"] == HB.FT_HEAD_FIXED_NAME and "raunav" in HB.FT_HEAD_FIXED_NAME
    assert d["spec"]["completions"] == len(cells) == 30
    assert {"action": "FailIndex", "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}} \
        in d["spec"]["podFailurePolicy"]["rules"]
    c = d["spec"]["template"]["spec"]["containers"][0]
    a = c["args"][0]
    assert '[ -f ${CELL}/DONE ] && [ -f ${CELL}/net_best_epoch_state.pt ] || { echo "FATAL: ${CELL} not DONE"; exit 42; }' in a
    assert a.index("not DONE") < a.index("extract_v2.py")         # refused before any work
    assert all(f"{i}) MODEL={m}; CELL={cell};;" in a for i, (m, cell) in enumerate(cells))
    # its own root: the superseded networks' outputs keep theirs, and a model with a manifest is never redone
    assert f"OUT={HB.FT_HEAD_FIXED_ROOT}/${{MODEL}}\n" in a and HB.FT_HEAD_FIXED_ROOT != HB.FT_HEAD_ROOT
    assert '[ "${USED}" -le 90 ]' in a and c["resources"]["requests"]["memory"] == HB.FT_HEAD_FIXED_MEM == "16Gi"


@pytest.mark.parametrize("mode, fams", [("jc1-fixed", ("jc1_ft_fixed", "jc1_ft_ovr_fixed")),
                                        ("ft-pairs-fixed", ("pairs_ft_fixed",)),
                                        ("probe-v5", ("jc1_probe_v5", "jc1_probe_ovr_v5", "jc2_ovr_v5", "pairs_v5")),
                                        ("probe-v5-fill", ("top_probe_v5", "qg_probe_v5"))])
def test_the_rerun_curve_jobs_check_every_input_first_and_mark_their_families_done_last(mode, fams):
    fn, name, dirs = GD.MODES[mode]
    lines = [ln.strip() for ln in fn().splitlines()]
    assert lines[0].startswith("for f in ") and lines[0].endswith('|| { echo "FATAL: no ${f}"; exit 42; }; done')
    assert lines[-len(fams):] == [f"echo done > {GD.ROC}/{f}/DONE" for f in fams]
    assert not any("DONE" in ln and "echo done" in ln for ln in lines[:-len(fams)])
    body = lines[1:-len(fams)]
    rules = [ln for ln in body if ln.startswith("python3 experiments/EVAL/refit_checks.py ")]
    curves = [ln for ln in body if ln.startswith("python3 experiments/FT/roc_points.py ")]
    assert len(rules) + len(curves) == len(body) and curves
    # the v5 curves are read before the whole refit's checks job passes (B4): the job applies the refit rules first
    assert body[:len(rules)] == rules and bool(rules) == mode.startswith("probe-v5")
    assert all(ln.endswith('|| { echo "FATAL: the v5 fits fail the refit rules"; exit 42; }') for ln in rules)
    assert set(dirs.split()) == set(fams) and name.endswith("-raunav") and mode in GD.RERUNS
    # every rerun family is one mlst_v4 reads, in one group
    assert set(fams) <= set(M.V4.ROC_RERUN.values())


def test_the_fixed_curve_jobs_need_every_cell_done():
    jc1 = GD.fixed_cells("jc1")
    assert len(jc1) == 30 and all(c["out"].startswith("/data/results/ft_v2/fixed/leg2/") for c in jc1)
    first = GD.MODES["jc1-fixed"][0]().splitlines()[0]
    assert all(f"{c['out']}/DONE" in first for c in jc1)
    first = GD.MODES["ft-pairs-fixed"][0]().splitlines()[0]
    assert all(f"{c}/DONE" in first and f"{HB.FT_HEAD_FIXED_ROOT}/{m}/manifest.json" in first
               for m, c in HB.ft_head_fixed_cells())
    assert f"--head-root {HB.FT_HEAD_FIXED_ROOT} " in GD.MODES["ft-pairs-fixed"][0]()


# ------------------------------------------------------------------ a family read from its rerun whole
def _put(root: pathlib.Path, fam: str, models) -> None:
    d = root / "curves_v2" / fam
    d.mkdir(parents=True, exist_ok=True)
    for m in models:
        (d / f"{m}.json").write_text(json.dumps({"family": fam, "model": m, "eps_grid": [0.5], "curves": {}}))


def test_a_linear_probe_family_takes_its_rerun_whole_and_with_its_group(tmp_path):
    _put(tmp_path, "pairs", ["mtx-l188-s1", "init-s1"])
    assert set(M.V4.load_family(tmp_path, "pairs")) == {"mtx-l188-s1", "init-s1"}
    _put(tmp_path, "pairs_v5", ["mtx-l188-s1"])
    with pytest.raises(SystemExit, match="pull the group together"):
        M.V4.load_family(tmp_path, "pairs")
    for fam in ("jc1_probe_v5", "jc1_probe_ovr_v5", "jc2_ovr_v5"):
        _put(tmp_path, fam, [])
    with pytest.raises(SystemExit, match="pairs_v5 lacks 1 models"):
        M.V4.load_family(tmp_path, "pairs")
    _put(tmp_path, "pairs_v5", ["init-s1"])
    docs = M.V4.load_family(tmp_path, "pairs")
    assert set(docs) == {"mtx-l188-s1", "init-s1"} and {d["family"] for d in docs.values()} == {"pairs_v5"}
    assert M.V4.load_family(tmp_path, "pairs_sib2") == docs      # X->ee vs X->mumu: every test jet, in the same rerun
    assert M.V4.load_family(tmp_path, "top_probe") == {}         # its own group, not pulled


def test_a_fine_tuning_family_takes_the_one_rule_once_every_run_and_seed_is_there(tmp_path):
    every = M.V4.paper_models() + ["scratch-s1", "scratch-s2", "scratch-s3"]
    _put(tmp_path, "jc1_ft", ["mtx-l188-s1"])
    _put(tmp_path, "jc1_ft_fixed", every)
    with pytest.raises(SystemExit, match="pull the group together"):
        M.V4.load_family(tmp_path, "jc1_ft")
    _put(tmp_path, "jc1_ft_ovr_fixed", every[:-1])
    with pytest.raises(SystemExit, match="jc1_ft_ovr_fixed lacks 1 models"):
        M.V4.load_family(tmp_path, "jc1_ft_ovr")
    _put(tmp_path, "jc1_ft_ovr_fixed", every[-1:])
    jc = M.V4.load_family(tmp_path, "jc1_ft")
    # JetClass's from-scratch name, as the long-budget docs had it
    assert {d["family"] for d in jc.values()} == {"jc1_ft_fixed"} and len(jc) == len(every)
    assert {"scratch-v2-s1", "scratch-v2-s2", "scratch-v2-s3"} <= set(jc) and "scratch-s1" not in jc
    _put(tmp_path, "pairs_ft_fixed", every)
    assert set(M.V4.load_family(tmp_path, "pairs_ft")) == set(every)
    assert set(M.load_pairs_ft(tmp_path)) == set(every)
