"""The figures of the linear probes draw the numbers of their tables (tab:probes, tab:probesn,
tab:pairs): they select them with mlst_results.py's own functions. The fits are a small
synthetic tree in tmp_path, so the tests do not depend on how much has been pulled."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SIZES = ("1000", "1000000")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fits_tree(M, runs, root: pathlib.Path) -> pathlib.Path:
    """One fits file per paper run and untrained trunk under root/linprobe/fits_v3, in the layout
    of linear_probe_v2.py: every task of M.TASKS at SIZES, and the pairs at M.PAIR_N."""
    d = root / "linprobe/fits_v3"
    d.mkdir(parents=True)
    models = [m for ms in runs.values() for m in ms]
    for i, m in enumerate(models):
        base = 0.80 if m.startswith("init-") else 0.99 - 1e-3 * i
        cell = {}
        for ds, metric, _ in M.TASKS:
            for k, n in enumerate(SIZES):
                c = cell.setdefault(ds, {}).setdefault(n, {"C": 1.0, "converged": True})
                c[metric] = base - 1e-2 * (len(SIZES) - 1 - k) - 1e-4 * len(c)
        cell[M.PAIR_DS] = {M.PAIR_N: {key: {"auc": base - 2e-4 * j} for j, (key, _, _, _) in enumerate(M.PAIRS)}}
        ck = "init" if m.startswith("init-") else M.PRIMARY
        (d / f"{m}.json").write_text(json.dumps({"cells": {ck: cell}}))
    return d


def test_the_figure_is_drawn_from_the_tables_numbers(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    monkeypatch.setattr(F, "DATA", tmp_path)
    fits = M._load(_fits_tree(M, runs, tmp_path))
    ds, metric, _ = M.TASKS[0]
    pts = F.series(M, fits, runs, ds, metric, tuple(F.CLASSES) + (M.INIT,))
    n = max((p[0] for p in pts["L188"]), default=None)
    assert n is not None
    vals = [1 - M._probe(fits, m, ds, str(n), metric) for m in runs["L188"] if m in fits]
    mean = next(p[1] for p in pts["L188"] if p[0] == n)
    assert mean == pytest.approx(sum(vals) / len(vals))
    # a task with no fits yet: every row empty, no failure
    assert all(v == [] for v in F.series(M, fits, runs, "no_fits_yet", metric, tuple(F.CLASSES)).values())
    paths = F.figure_sizes(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_configuration_figure_draws_the_probe_tables_numbers(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    monkeypatch.setattr(F, "DATA", tmp_path)
    fits = M._load(_fits_tree(M, runs, tmp_path))
    for ds, metric, _ in M.TASKS[:4]:
        rows = F.probe_rows(M, fits, runs, ds, metric)
        assert set(rows) == set(F.CLASSES) | set(M.RAND) | set(M.FLAV) | {M.INIT}
        n = M._largest(fits, ds)
        for arm in ("L188", "RAND2_p1", "FLAV_F1", M.INIT):
            want = [1e3 * (1 - M._probe(fits, m, ds, n, metric)) for m in runs[arm] if m in fits]
            assert want and rows[arm] == pytest.approx(want)
    # a task with no fits yet: every row empty, no failure
    empty = F.probe_rows(M, fits, runs, "no_fits_yet", "auc")
    assert set(empty) == set(rows) and all(v == [] for v in empty.values())
    paths = F.figure_label_sets(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_pairs_figure_draws_the_pairs_table_and_its_merge_marks(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    monkeypatch.setattr(F, "DATA", tmp_path)
    fits = M._load(_fits_tree(M, runs, tmp_path))
    status = M.merge_status(REPO)
    for key, _, _, _ in M.PAIRS:
        rows = F.pair_rows(M, fits, runs, key)
        assert M.INIT not in rows
        want = [1e3 * (1 - fits[m]["cells"][M.PRIMARY][M.PAIR_DS][M.PAIR_N][key]["auc"])
                for m in runs["L188"] if m in fits]
        assert want and rows["L188"] == pytest.approx(want)
        # the 17-class label set merges every pair: open markers
        assert F.FILL[status["R16_Q1"][key]] == "none" and F.FILL[status["L188"][key]] == "full"
    paths = F.figure_pairs(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_schematic_is_read_from_the_label_maps(tmp_path):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    g = F.schematic_groups(REPO)
    row = {c: i for i, (c, _) in enumerate(F.SCHEMATIC_CLASSES)}
    two = [row[c] for c in ("X_bb", "X_bc", "X_bq", "X_cc", "X_cs", "X_ss", "X_qq")]
    assert len({g["188 classes"][i] for i in two}) == len(two)            # every decay its own class
    assert g["43 classes"][row["X_bb"]] == g["43 classes"][row["X_bq"]] != g["43 classes"][row["X_cc"]]
    assert len({g["17 classes"][i] for i in two}) == 1                     # no flavor boundary
    assert len({g["No flavor boundary"][i] for i in two}) == 1
    b = g["$b$-quark boundary"]
    assert b[row["X_YY_bbqq"]] != b[row["X_YY_ccqq"]] == b[row["X_YY_qqqq"]]
    paths = F.figure_schematic(REPO, tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_ranked_classes_are_the_ones_a_merge_degrades_seed_by_seed(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    rung, order = F._class_order(REPO)
    hurt = order[7]
    d = tmp_path / "linprobe/fits_v3"
    d.mkdir(parents=True)
    for m in (m for a in ("L188", "L162", "R42_Q1", "R16_Q1") for m in runs[a]):
        seed = int(m.rsplit("-s", 1)[1])
        # 1 - AUC = 0.01 * seed for every class; the 162-class runs make one class 3x (seed 1) or 5x worse
        oma = {c: 0.01 * seed for c in order}
        if m in runs["L162"]:
            oma[hurt] *= 3.0 if seed == 1 else 5.0
        cell = {"jc2_188": {M.PAIR_N: {"per_class_auc": {str(c): 1 - v for c, v in oma.items()}}}}
        (d / f"{m}.json").write_text(json.dumps({"cells": {M.PRIMARY: cell}}))
    monkeypatch.setattr(F, "DATA", tmp_path)
    fits = F._fits(M)
    ranked = F.ranked_merges(M, fits, runs, REPO)
    name, mean, per = ranked[0][0]
    assert name == rung[hurt]["class_name"]
    n162 = len([m for m in runs["L162"] if m.rsplit("-s", 1)[1] in {r.rsplit("-s", 1)[1] for r in runs["L188"]}])
    assert len(per) == n162 and sorted(per) == pytest.approx(sorted([3.0] + [5.0] * (n162 - 1)))
    assert mean == pytest.approx(sum(per) / len(per))
    # the 162 -> 43 merge undoes it, so the class is not among that merge's most degraded
    assert name not in [r[0] for r in ranked[1]]
    paths = F.figure_per_class_ranked(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_every_class_name_is_typeset():
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    rung, order = F._class_order(REPO)
    tex = {F.class_tex(rung[c]["class_name"]) for c in order}
    assert len(tex) == len(order)
    assert F.class_tex("label_X_YY_bctauhv") == r"$X\to YY\to b c \tau_h \nu$"
    assert F.class_tex("label_QCD_light") == "QCD (light)"


def test_the_missing_feature_figure_draws_ratios_to_all_features(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    d = tmp_path / "linprobe/fits_v4"
    d.mkdir(parents=True)
    n = "1000000"
    for row, ms in runs.items():
        for m in ms:
            seed = int(m.rsplit("-s", 1)[1])
            cell = {"jc2bal": {n: {"macro_auc_ovr": 1 - 0.01 * seed}},
                    "jc2bal.mtop": {n: {"macro_auc_ovr": 1 - 0.03 * seed}},
                    "jc2bal.mqg": {n: {"macro_auc_ovr": 1 - 0.015 * seed}}}
            if row in F.IMPUTED_ROWS:
                cell["jc2bal.itop"] = {n: {"macro_auc_ovr": 1 - 0.012 * seed}}
            (d / f"{m}.json").write_text(json.dumps({"cells": {M._ck(m): cell}}))
    monkeypatch.setattr(F, "DATA", tmp_path)
    raw = M.V4.load_fits(tmp_path)[1]
    rows = F.missing_rows(M, raw, runs, "mtop", ("L188", "R16_Q1", "RAND2_p1"))
    assert rows["L188"] == pytest.approx([3.0] * len(runs["L188"]))
    assert F.missing_rows(M, raw, runs, "mqg", ("R16_Q1",))["R16_Q1"] == pytest.approx([1.5] * len(runs["R16_Q1"]))
    imp = F.missing_rows(M, raw, runs, "itop", ("L188", "L162"))
    assert imp["L188"] == pytest.approx([1.2] * len(runs["L188"])) and imp["L162"] == []
    paths = F.figure_missing_features(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_unsculpted_floor_is_the_small_sample_bias_of_the_divergence():
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    import numpy as np
    before = np.linspace(1.0, 0.2, 24)
    lo, hi = F.jsd_floor(before, [40_000, 4_000])
    # for k bins and n jets the bias of the base-2 JSD is about (k - 1) / (8 n ln 2)
    assert hi == pytest.approx(23 / (8 * 4_000 * np.log(2)), rel=0.15)
    assert lo == pytest.approx(hi / 10, rel=0.2)


def test_the_sculpting_and_response_figures_read_the_curve_files(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    d = tmp_path / "head/curves_v2"
    d.mkdir(parents=True)
    before = [1.0 / 24] * 24
    for row in F.SCULPT_ROWS:
        for m in runs[row]:
            k = 2.0 if row == "R16_Q1" else 1.0
            curve = [{"eff": e, "n_pass": int(4e5 * e), "jsd": k * 1e-3 / e ** 0.5} for e in (0.01, 0.1)]
            wps = {wp: {"hist_after": before} for wp in ("0.1", "0.01")}
            (d / f"{m}_sculpting_v2.json").write_text(json.dumps(
                {"bins_jet_sdmass": list(range(20, 501, 20)), "hist_before": before, "working_points": wps,
                 "curve": curve}))
            hist = {f"{h}/{kind}": [1] * 100 for h in ("uncorrected", "node") for kind in ("resonant", "qcd")}
            (d / f"{m}_response.json").write_text(json.dumps({"bins": [-1 + 0.02 * i for i in range(101)],
                                                              "hist": hist}))
    monkeypatch.setattr(F, "DATA", tmp_path)
    got = F.sculpting_curves(M, runs, F._curves("sculpting_v2"))
    eff, mean, per = got["R16_Q1"]
    assert eff == [0.01, 0.1] and mean == pytest.approx([2e-2, 2e-3 / 0.1 ** 0.5])
    assert len(per) == len(runs["R16_Q1"])
    for fig in (F.figure_sculpting, F.figure_mass_response_hist):
        paths = fig(REPO, tmp_path / "out")
        assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_roc_figures_pool_the_random_partitions_and_take_each_run(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    eps = [0.05 * i for i in range(1, 20)]
    cells = {}
    models = [m for ms in runs.values() for m in ms] + list(M.INIT_MODELS)
    for i, m in enumerate(models):
        roc = {"eps_s": eps, "rejection": [(1 + 0.01 * i) / e for e in eps]}
        cells[m] = {M._ck(m): {ds: {"roc": roc} for ds, _ in F.ROC_PAIRS + F.ROC_BENCH}}
    (tmp_path / "curves").mkdir()
    (tmp_path / "curves/linprobe_fits_v4.json").write_text(json.dumps({"cells": cells}))
    monkeypatch.setattr(F, "DATA", tmp_path)
    rr = F.roc_rows(M, runs, F._probe_curves(), "top", F.ROC_ROWS_BENCH)
    assert len(rr["RAND"][1]) == sum(len(runs[a]) for a in M.RAND)
    assert len(rr["INIT"][1]) == len(M.INIT_MODELS) and len(rr["L188"][1]) == len(runs["L188"])
    for fig in (F.figure_roc_pairs, F.figure_roc_bench):
        paths = fig(REPO, tmp_path / "out")
        assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


def test_the_fine_tuning_roc_reads_the_full_sets_and_each_scratch_seed(tmp_path, monkeypatch):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    runs = M._paper(REPO)
    roc = lambda k: {"roc": {"eps_s": [0.3, 0.5], "rejection": [100.0 * k, 50.0 * k]}}
    (tmp_path / "curves").mkdir()
    for tree, n in F.FT_FULL.items():
        ds = ("top",) if tree == "leg_top" else ("qg", "qg_herwig")
        pre = {m[len("mtx-"):]: {f"N{n}": {"s1": {d: roc(1) for d in ds}}, "N1000": {"s1": {d: roc(9) for d in ds}}}
               for m in runs["L188"] + runs["R16_Q1"]}
        ref = {"scratch-v2": {f"N{n}": {f"s{k}": {d: roc(0.5) for d in ds} for k in (1, 2, 3)}}}
        (tmp_path / "curves" / f"full_best70_{tree}.json").write_text(json.dumps({"cells": pre}))
        (tmp_path / "curves" / f"full_ref_v100_bench_{tree}.json").write_text(json.dumps({"cells": ref}))
    monkeypatch.setattr(F, "DATA", tmp_path)
    cells = F._ft_curves()
    rr = F.roc_rows(M, runs, cells, "qg_herwig", F.ROC_ROWS_FT)
    assert len(rr["SCRATCH"][1]) == 3 and rr["L188"][1][0] == [100.0, 50.0]      # the full set, not N1000
    paths = F.figure_roc_bench_ft(REPO, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
