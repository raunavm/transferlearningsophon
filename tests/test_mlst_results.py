"""The MLST tables' code (experiments/FIGS/mlst_results.py, read by make_tables.build()):
mean +- sample SD over the runs present with a marker for missing runs, a lone run with its
trailing zeros, the probe convergence rule, the merge marks from the label maps, and the
pulled data checked against its index."""
import importlib.util
import json
import pathlib
import shutil

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
    assert M._cell([0.1, None, 0.3], 3, fmt) == r"X \pending{2 of 3}"
    assert M._cell([None, None], 2, fmt) == r"\pending{no run yet}"


def test_a_lone_run_keeps_its_trailing_zeros():
    assert M._cell([0.86300], 3, MT.fmt_pm) == r"0.86300 \pending{1 of 3}"
    assert M._cell([0.96579654592], 1, MT.fmt_pm) == "0.96580"
    assert M._cell([0.0593], 1, MT.fmt_pm, 1e3) == "59.300"


def test_an_unconverged_probe_stops_a_trained_run_and_marks_the_untrained_network():
    cell = {"converged": False, "auc": 0.7}
    fits = {"mtx-l188-s1": {"cells": {"best70": {"top": {"1000": cell}}}},
            "init-s1": {"cells": {"init": {"top": {"1000": cell}}}}}
    with pytest.raises(SystemExit, match="did not converge"):
        M._probe(fits, "mtx-l188-s1", "top", "1000", "auc")
    v = M._probe(fits, "init-s1", "top", "1000", "auc")
    assert isinstance(v, M.Unconverged) and M._cell([v], 1, MT.fmt_pm).endswith(M.UNCONVERGED)


def test_the_merge_marks_follow_the_label_maps():
    s = M.merge_status(ROOT)
    assert set(s["L188"].values()) == set(s["L162"].values()) == {"apart"}
    assert s["R42_Q1"] == {"bc_vs_bq_cs": "partly", "bb_vs_cc": "apart", "bbqq_vs_ccqq": "apart"}
    assert set(s["R16_Q1"].values()) == {"merged"}
    assert s["L162_MASS"] == s["L162"] and s["R16_Q1_MASS"] == s["R16_Q1"]


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason="no pulled data")
def test_the_pulled_data_is_checked_against_its_index(tmp_path):
    assert M.check_index(DATA) == DATA / "index.json"
    d = tmp_path / "v2"
    shutil.copytree(DATA, d)
    f = next((d / "linprobe/fits3000").glob("*.json"))
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
    assert kinds == {"compute", "linprobe/fits3000", "linprobe/mass"}
