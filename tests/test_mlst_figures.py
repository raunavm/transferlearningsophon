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
