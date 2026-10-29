"""The figure of the linear probes against the training-set size draws the numbers of
Table tab:probesn: it selects them with mlst_results.py's own functions."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "experiments/FIGS/data/v2/linprobe/fits3000"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not DATA.is_dir(), reason="no linear-probe fits in this checkout")
def test_the_figure_is_drawn_from_the_tables_numbers(tmp_path):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    fits, runs = M._load(DATA), M._paper(REPO)
    ds, metric, _ = M.TASKS[0]
    pts = F.series(M, fits, runs, ds, metric)
    n = max((p[0] for p in pts["L188"]), default=None)
    assert n is not None
    vals = [1 - M._probe(fits, m, ds, str(n), metric) for m in runs["L188"] if m in fits]
    mean = next(p[1] for p in pts["L188"] if p[0] == n)
    assert mean == pytest.approx(sum(vals) / len(vals))
    paths = F.figure(REPO, tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


@pytest.mark.skipif(not DATA.is_dir(), reason="no linear-probe fits in this checkout")
def test_the_configuration_figure_draws_the_probe_tables_numbers(tmp_path):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    fits, runs = M._load(DATA), M._paper(REPO)
    for ds, metric, _ in M.TASKS[:4]:
        rows = F.probe_rows(M, fits, runs, ds, metric)
        assert set(rows) == set(F.CLASSES) | set(M.RAND) | set(M.FLAV) | {M.INIT}
        n = M._largest(fits, ds)
        for arm in ("L188", "RAND2_p1", "FLAV_F1"):
            want = [1e3 * (1 - M._probe(fits, m, ds, n, metric)) for m in runs[arm] if m in fits]
            assert rows[arm] == pytest.approx(want)
    paths = F.figure_label_sets(REPO, tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in paths)


@pytest.mark.skipif(not DATA.is_dir(), reason="no linear-probe fits in this checkout")
def test_the_pairs_figure_draws_the_pairs_table_and_its_merge_marks(tmp_path):
    F = _mod("mlst_figures", "experiments/FIGS/mlst_figures.py")
    M = F._mlst()
    fits, runs = M._load(DATA), M._paper(REPO)
    status = M.merge_status(REPO)
    for key, _, _, _ in M.PAIRS:
        rows = F.pair_rows(M, fits, runs, key)
        assert M.INIT not in rows
        want = [1e3 * (1 - fits[m]["cells"][M.PRIMARY]["jc2_pairs"][M.PAIR_N][key]["auc"])
                for m in runs["L188"] if m in fits]
        assert rows["L188"] == pytest.approx(want)
        # the 17-class label set merges every pair: open markers
        assert F.FILL[status["R16_Q1"][key]] == "none" and F.FILL[status["L188"][key]] == "full"
    paths = F.figure_pairs(REPO, tmp_path)
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
