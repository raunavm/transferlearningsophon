"""The one fine-tuning rule's read-out (PI 2026-11-15) and what the paper reads of it: its 341 cells against the launched
specs, merge_fixed with JetClass's 1e6 cells as its largest set, the preview switch (MLST_FT_PREVIEW), the kinematic-only
network's full top cell, the rate sweep's table, the read-out's expected cells, and the height of A8 and A10."""
import importlib.util
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/FIGS/data/v2"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("mlst_results_fx", "experiments/FIGS/mlst_results.py")
B = _load("build_ft_budget_jobs_fx", "scripts/build_ft_budget_jobs.py")


def _keys() -> list:
    return sorted("/".join(pathlib.Path(c["out"]).parts[-4:]) for c in B.fixed_cells(set(B.FIXED_STEPS)))


def _rec(v):
    return {"macro_auc_ovr": v, "macro_auc_qcd": v, "auc": v, "accuracy": v, "r50": 100 * v, "r30": 300 * v,
            "r50_is_bound": False, "r30_is_bound": False, "r50_n_bkg_pass": 400, "r30_n_bkg_pass": 120}


def _doc(drop=()) -> dict:
    """A read-out of the one rule at 0.7 everywhere, the cells in drop missing."""
    cells = {k: {**_rec(0.7), **({"herwig": _rec(0.7)} if k.startswith("leg_qg/") else {})} for k in _keys() if k not in drop}
    return {"cells": cells, "missing": sorted(drop), "expected": len(_keys())}


def _per() -> dict:
    """The earlier read-outs at 0.9: the full top and quark/gluon sets, and JetClass at 1e6 with ParT's recipe."""
    full = lambda n: {"l188-s1": {n: [_rec(0.9)]}, M.SCRATCH: {n: [_rec(0.8)] * 3}}     # noqa: E731
    return {"jc1": full(1_000_000), "top": full(1_200_000), "qg": full(1_600_000), "qg_herwig": full(1_600_000),
            "jc2": full(1_000_000), "jc2_qcd": full(1_000_000)}


def test_fixed_n_is_every_cell_of_the_launched_one_rule_specs():
    launched = set()
    for f in sorted((ROOT / "experiments/FT/k8s/budget").glob("job-ft-budget-fixed-1e[3456]*-raunav.yaml")):
        launched |= set(re.findall(r"CELL=/data/results/ft_v2/fixed/([^;\s]+)", f.read_text()))
    assert len(launched) == M.FIXED_N == 341
    assert launched == set(_keys())


def test_jetclass_largest_set_is_the_one_rule_cell_and_the_full_benchmarks_keep_parts_recipe(tmp_path):
    p = tmp_path / M.FIXED
    p.write_text(json.dumps(_doc()))
    out = M.merge_fixed(_per(), p)
    assert out["jc1"]["l188-s1"][1_000_000][0]["accuracy"] == 0.7          # not the old 0.9 (ParT's recipe)
    assert [r["accuracy"] for r in out["jc1"][M.SCRATCH][1_000_000]] == [0.7] * 3
    assert out["jc2"]["l188-s1"][1_000_000][0]["accuracy"] == 0.7
    for key, n in (("top", 1_200_000), ("qg", 1_600_000), ("qg_herwig", 1_600_000)):
        assert out[key]["l188-s1"][n][0]["accuracy"] == 0.9 and len(out[key][M.SCRATCH][n]) == 3
    assert out["top"][M.KIN_INIT][1000][0]["accuracy"] == 0.7 and M.KIN_INIT not in out["qg"]
    # a one-rule cell at a full top set is refused
    d = _doc()
    d["cells"]["leg_top/l188-s1/N1200000/s1"] = d["cells"].pop("leg_top/l188-s1/N1000/s1")
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="full sets keep ParT's recipe"):
        M.merge_fixed(_per(), p)


def test_a_partial_readout_is_read_only_as_a_preview(tmp_path):
    drop = {"leg2/l188-s1/N1000000/s1", f"leg2/{M.BUDGET_SCRATCH}/N1000000/s2"}
    p = tmp_path / M.FIXED_PARTIAL
    p.write_text(json.dumps(_doc(drop)))
    with pytest.raises(SystemExit, match="complete or not at all"):
        M.merge_fixed(_per(), p)
    out = M.merge_fixed(_per(), p, preview=True)
    assert 1_000_000 not in out["jc1"]["l188-s1"]                 # missing: absent, its table cell pending
    assert out["jc1"]["l188-s2"][1_000_000][0]["accuracy"] == 0.7
    assert 1_000_000 not in out["jc1"][M.SCRATCH]                 # a from-scratch size short of a seed: left out whole
    assert len(out["jc1"][M.SCRATCH][100_000]) == 3
    d = _doc(drop)
    d["missing"] = d["missing"][:1]                                # a preview names every cell, read or missing
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="names every cell"):
        M.merge_fixed(_per(), p, preview=True)


def test_load_ft_reads_the_partial_readout_only_with_the_preview_switch(tmp_path, monkeypatch):
    (tmp_path / "ft_mlst").mkdir()
    (tmp_path / "budget").mkdir()
    cells = {"top": {"l188-s1": {"N1200000": {"s1": _rec(0.9)}}}, "qg": {"l188-s1": {"N1600000": {"s1": _rec(0.9)}}}}
    (tmp_path / "ft_mlst" / "best70_bench_metrics.json").write_text(json.dumps({"cells": cells}))
    (tmp_path / "budget" / M.FIXED_PARTIAL).write_text(json.dumps(_doc({"leg_top/l188-s1/N1000/s1"})))
    monkeypatch.delenv(M.PREVIEW_ENV, raising=False)
    _, per, _ = M.load_ft(tmp_path)
    assert "l188-s2" not in per["top"]                              # the paper's build never reads a partial read-out
    monkeypatch.setenv(M.PREVIEW_ENV, "1")
    _, per, _ = M.load_ft(tmp_path)
    assert per["top"]["l188-s2"][1000][0]["accuracy"] == 0.7 and 1000 not in per["top"]["l188-s1"]
    assert per["top"]["l188-s1"][1_200_000][0]["accuracy"] == 0.9
    (tmp_path / "budget" / M.FIXED).write_text(json.dumps(_doc()))     # the complete read-out wins, even in a preview
    _, per, _ = M.load_ft(tmp_path)
    assert per["top"]["l188-s1"][1000][0]["accuracy"] == 0.7
    (tmp_path / "budget" / M.FIXED).unlink()
    (tmp_path / "budget" / M.FIXED_PARTIAL).unlink()
    with pytest.raises(SystemExit, match=M.PREVIEW_ENV):
        M.load_ft(tmp_path)


def test_the_kinematic_only_full_top_cell_is_added_alone(tmp_path):
    p = tmp_path / M.KIN_FULL
    p.write_text(json.dumps({"cells": {f"leg_top/{M.KIN_INIT}/N1200000/s1": _rec(0.6)}, "missing": []}))
    assert M.merge_kin({"top": {}}, p)["top"][M.KIN_INIT][1_200_000][0]["accuracy"] == 0.6
    p.write_text(json.dumps({"cells": {f"leg_top/{M.KIN_INIT}/N100000/s1": _rec(0.6)}}))
    with pytest.raises(SystemExit):
        M.merge_kin({"top": {}}, p)
    assert M.merge_kin({"x": 1}, tmp_path / "absent.json") == {"x": 1}


def test_the_readout_names_every_one_rule_cell_and_reads_a_full_set_cell_without_early_stop(tmp_path):
    bm = _load("budget_metrics_fx", "experiments/FT/budget_metrics.py")
    assert sorted(bm.fixed_keys()) == _keys()
    out = tmp_path / "out.json"
    bm.main(["--root", str(tmp_path), "--expect-fixed", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["expected"] == 341 and not d["cells"] and sorted(d["missing"]) == _keys()
    c = tmp_path / "kin"
    c.mkdir()
    (c / "best_epoch.json").write_text(json.dumps({"epoch": 17, "metric": 0.94, "n_epochs": 20, "metrics": [0.9] * 20}))
    assert bm.early_stop(c) == {"source": "best_epoch.json", "best_epoch": 17, "best_metric": 0.94,
                                "epochs_completed": 20, "stopped_early": False}


def test_a8_and_a10_with_their_auc_block_fit_one_page():
    import matplotlib
    matplotlib.use("Agg")
    sys.path[:0] = [str(ROOT / "experiments/FIGS/figstyle"), str(ROOT / "experiments/FIGS")]
    import gallery_labelset as LS
    max_h = float(re.search(r"^MAX_H_CM = ([\d.]+)", (ROOT / "experiments/FIGS/make_results_book.py").read_text(),
                            re.M).group(1))
    slots, h, _, height = LS.size_layout(True)
    assert height <= max_h and len(slots) == len(LS.SIZE5) + len(LS.SIZE_AUC4) == 9
    assert [t[0] for t in LS.SIZE_AUC4] == ["jc2", "jc1", "top", "qg"]
    assert [t[2] for t in LS.SIZE_AUC4] == ["macro_auc_ovr", "macro_auc_ovr", "auc", "auc"]
    assert LS.size_layout(False)[1] == pytest.approx(LS.SIZE_ASPECT * LS.SIZE_W) and LS.size_layout(False)[3] <= max_h
