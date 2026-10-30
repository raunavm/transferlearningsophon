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
