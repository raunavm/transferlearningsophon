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
    assert {"compute", "linprobe/fits3000", "linprobe/mass"} <= kinds <= {
        "compute", "linprobe/fits3000", "linprobe/mass", "ft_mlst"}


def _ft_tree(d, inits, n_seeds_scratch=3, bound=False):
    rec = lambda v: {"macro_auc_ovr": v, "auc": v, "r50": 100 * v, "r30": 300 * v,
                     "r50_is_bound": bound, "r30_is_bound": False, "r50_n_bkg_pass": 400, "r30_n_bkg_pass": 120}
    def by_init(sizes):
        out = {i: {f"N{n}": {"s1": rec(0.9 + 0.01 * k)} for n in sizes} for k, i in enumerate(inits)}
        out[M.SCRATCH] = {f"N{n}": {f"s{s}": rec(0.8) for s in range(1, n_seeds_scratch + 1)} for n in sizes}
        return out
    (d / "ft_mlst").mkdir(parents=True)
    w = lambda name, cells: (d / "ft_mlst" / f"best70_done18_{name}.json").write_text(json.dumps({"cells": cells}))
    w("leg1_metrics", by_init([1000, 1000000]))
    w("leg2_metrics", by_init([1000, 1000000]))
    w("bench_metrics", {"top": by_init([1000, 1200000]), "qg": by_init([1000, 1600000])})
    w("bench_metrics_herwig", {"qg": by_init([1000, 1600000])})


def test_the_fine_tuning_tables_read_one_seed_per_run_and_the_scratch_seeds(tmp_path):
    _ft_tree(tmp_path, ["l188-s1", "l188-s2"])
    tag, per = M.load_ft(tmp_path)
    assert tag == "best70_done18" and per["jc2"]["l188-s1"][1000][0]["macro_auc_ovr"] == 0.9
    runs = {"L188": ["mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"]}
    t = M.table_ft(per, runs, MT.fmt_pm, ("L188",))
    assert t.count("%SPLIT%") == len(M.FT_TASKS) - 1 and r"\pending{2 of 3}" in t and "from scratch" in t
    assert "1{,}200{,}000" in t and "1{,}600{,}000" in t
    body, fewest = M.table_ft_rej(per, runs, MT.fmt_pm, ("L188",))
    assert fewest == 120 and r"\pending{2 of 3}" in body and body.count(r"\\") == 4
    per["jc2"]["l188-s1"][1000].append(per["jc2"]["l188-s1"][1000][0])
    with pytest.raises(SystemExit, match="fine-tuning seeds"):
        M.table_ft(per, runs, MT.fmt_pm, ("L188",))


def test_a_rejection_that_is_only_a_bound_stops_the_table(tmp_path):
    _ft_tree(tmp_path, ["l188-s1"], bound=True)
    _, per = M.load_ft(tmp_path)
    with pytest.raises(SystemExit, match="only a bound"):
        M.table_ft_rej(per, {"L188": ["mtx-l188-s1"]}, MT.fmt_pm, ("L188",))


def test_no_fine_tuning_read_out_is_no_table(tmp_path):
    assert M.load_ft(tmp_path) == (None, {})


class _Em:
    def __init__(self):
        self.m = {}

    def macro(self, name, body, *a, **k):
        self.m[name] = body


def _recipe_tree(tmp, cells, spec_text="weaver --data-train x\n"):
    data = tmp / "experiments/FIGS/data/v2"
    (data / "ft_mlst").mkdir(parents=True)
    (data / "ft_mlst/manifests.json").write_text(json.dumps(cells))
    spec = tmp / M.LEGS_SPEC
    spec.parent.mkdir(parents=True)
    spec.write_text(spec_text)
    return data


def _cell(leg, init, n, **kw):
    c = {"leg": leg, "init": init, "n_train": n, "ft_seed": "1", "lr": "1e-4", "head_lr_mult": "50",
         "weight_decay": "0.01", "epochs": "20" if leg in ("top", "qg") else {"1000": "50", "1000000": "10"}[n],
         "lr_schedule": "constant" if leg in ("top", "qg") else None, "recipe": None, "batch_size": "512",
         "ckpt_rule": "best70", "gpu_device_name": "Tesla V100-SXM2-32GB"}
    c.update(kw)
    return c


def test_the_recipe_macros_come_from_every_cell_and_one_stray_value_stops_the_build(tmp_path):
    cells = {f"best70/leg1/l188-s1/N{n}/s1": _cell("1", "l188-s1", n) for n in ("1000", "1000000")}
    cells["best70/leg_top/l188-s1/N1000/s1"] = _cell("top", "l188-s1", "1000")
    cells["best70/leg1/not-a-paper-run/N1000/s1"] = _cell("1", "not-a-paper-run", "1000", lr="9")   # ignored
    em, missing = _Em(), []
    M.ft_recipe(em, _recipe_tree(tmp_path / "a", cells), {"l188-s1"}, str, missing, MT.words)
    assert em.m["MlstFtLr"] == "1e-4" and em.m["MlstFtHeadMult"] == "50" and em.m["MlstFtBenchEpochs"] == "20"
    assert em.m["MlstFtEpochsEThree"] == "50" and em.m["MlstFtEpochsESix"] == "10"
    assert em.m["MlstFtGpuTypes"] == "V100" and em.m["MlstFtCellsDone"] == "3"
    assert em.m["MlstFtRefLr"].startswith(r"\pending") and any("V100" in x for x in missing)
    cells["best70/leg1/l188-s1/N1000/s1"]["lr"] = "2e-4"
    with pytest.raises(SystemExit, match="expected one value"):
        M.ft_recipe(_Em(), _recipe_tree(tmp_path / "b", cells), {"l188-s1"}, str, [], MT.words)
    cells["best70/leg1/l188-s1/N1000/s1"]["lr"] = "1e-4"
    with pytest.raises(SystemExit, match="sets a scheduler"):
        M.ft_recipe(_Em(), _recipe_tree(tmp_path / "c", cells, "--lr-scheduler none\n"), {"l188-s1"}, str, [], MT.words)
    cells.update({f"ref_v100/jc2/leg1/scratch-v2/N1000/s{s}": _cell("1", "scratch-v2", "1000", lr="1e-3", ft_seed=str(s),
                                                                       recipe="part-from-scratch") for s in (1, 2, 3)})
    em = _Em()
    M.ft_recipe(em, _recipe_tree(tmp_path / "d", cells), {"l188-s1"}, str, [], MT.words)
    assert em.m["MlstFtRefLr"] == "1e-3" and em.m["MlstFtRefSeedsWord"] == "three"
