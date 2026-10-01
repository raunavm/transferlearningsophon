"""v2 JetClass-II fine-tuning subsets (audit 2026-09-29, must-fix 6 and 7) and the
bookkeeping every v2 job relies on (experiments/FT/ft_v2.py).

  * the file partition: the fine-tuning pool is disjoint from the pretraining
    training files, the pretraining fixed validation sample and the fine-tuning
    test files, and the four roles are exactly the lists the jobs use;
  * a build reads nothing outside the pool, keeps validation and training in
    different files, records each output's native-class coverage, and makes the
    validation set a whole number of batches;
  * the checkpoint rules, the sha256 record, weaver's own run arguments.
"""
import importlib.util
import json
import pathlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("make_subsets_v2", "experiments/FT/make_subsets.py")
FV = _load("ft_v2_under_test", "experiments/FT/ft_v2.py")
B = _load("build_ft_jobs_v2", "scripts/build_ft_jobs.py")


def _names(files):
    return {pathlib.Path(f).name for f in files}


def _expand(brace: str) -> list[str]:
    """bash >= 4 brace expansion, zero padding kept (the pods run bash 5; the
    macOS /bin/bash 3.2 drops the padding, so it cannot be the reference)."""
    out = []
    for tok in brace.split():
        m = re.fullmatch(r"(.*)\{(\d+)\.\.(\d+)\}(.*)", tok)
        if not m:
            out.append(tok)
            continue
        a, b = m.group(2), m.group(3)
        w = max(len(a), len(b)) if a.startswith("0") or b.startswith("0") else 0
        out += [f"{m.group(1)}{i:0{w}d}{m.group(4)}" for i in range(int(a), int(b) + 1)]
    return out


# ------------------------------------------------------------ the partition

def test_the_four_roles_are_disjoint_and_are_the_lists_the_jobs_use():
    roles = {r: _names(MS.jc2_role_files(r)) for r in MS.JC2_ROLES}
    for a in roles:
        for b in roles:
            if a < b:
                assert not roles[a] & roles[b], (a, b)
    # the pretraining training split every training job guards
    train = re.search(r"TRAIN_FILES=\(([^)]*)\)", B.SPLIT_GUARD).group(1)
    val = re.search(r"VAL_FILES=\(([^)]*)\)", B.SPLIT_GUARD).group(1)
    assert roles["pretrain_train"] == _names(_expand(train))
    assert roles["pretrain_fixed_val"] | roles["ft_heldout"] == _names(_expand(val))
    # the fine-tuning test list, verbatim from the feature-extraction spec
    assert roles["ft_test"] == _names(B.test2m_list().split())
    # the 2026-09-29 decision, file by file
    assert len(roles["pretrain_fixed_val"]) == 4 + 16 + 5
    assert {"Res2P_0203.parquet", "Res34P_0875.parquet", "QCD_0284.parquet"} <= roles["pretrain_fixed_val"]
    assert len(roles["ft_heldout"]) == 46 + 199 + 65 == 310


def test_the_staging_job_offers_exactly_the_held_out_files():
    assert _names(_expand(B.V2_POOL)) == _names(MS.jc2_role_files("ft_heldout"))


def test_role_of_a_file_comes_from_its_name():
    assert MS.jc2_role("/x/Res34P_0876.parquet") == "ft_heldout"
    assert MS.jc2_role("/x/QCD_0284.parquet") == "pretrain_fixed_val"
    assert MS.jc2_role("/x/Res2P_0250.parquet") == "ft_test"
    assert MS.jc2_role("/x/Res2P_0199.parquet") == "pretrain_train"
    assert MS.jc2_role("/x/top_test.parquet") is None


# ------------------------------------------------------------ a build

def _jc2_file(path, n, seed):
    rng = np.random.default_rng(seed)
    pq.write_table(pa.table({
        "jet_pt": rng.uniform(100.0, 3000.0, n), "jet_sdmass": rng.uniform(0.0, 600.0, n),
        "jet_label": rng.integers(0, 188, n)}), path)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Three files of every role, named as on the volume."""
    d = tmp_path_factory.mktemp("jc2")
    files = {}
    for role, ranges in MS.JC2_ROLES.items():
        for fam, (a, _) in ranges.items():
            for i in range(a, a + 3):
                p = d / f"{fam}_{i:04d}.parquet"
                _jc2_file(p, 3000, hash((fam, i)) % 10_000)
                files.setdefault(role, []).append(str(p))
    return files


@pytest.mark.parametrize("role", ["pretrain_train", "pretrain_fixed_val", "ft_test"])
def test_a_build_refuses_any_file_outside_the_pool(tree, tmp_path, role):
    with pytest.raises(SystemExit, match="not fine-tuning held-out"):
        MS.build_jc2v2(tree["ft_heldout"] + tree[role][:1], tmp_path / "o", [100], [1],
                       n_files=3, take=0.5, val_size=512, n_val_files=3)


def test_the_validation_set_must_be_whole_batches(tree, tmp_path):
    with pytest.raises(SystemExit, match="multiple of 512"):
        MS.build_jc2v2(tree["ft_heldout"], tmp_path / "o", [100], [1],
                       n_files=3, take=0.5, val_size=500, n_val_files=3)


@pytest.fixture(scope="module")
def built(tree, tmp_path_factory):
    out = tmp_path_factory.mktemp("jc2v2")
    rc = MS.main(["jc2v2", "--pool-files", *tree["ft_heldout"], "--out", str(out),
                  "--sizes", "100", "1000", "--seeds", "1", "--n-files", "4",
                  "--take-fraction", "0.5", "--val-size", "512", "--n-val-files", "3"])
    assert rc == 0
    return out, json.loads((out / "manifest.json").read_text())


def test_a_build_reads_only_the_pool_and_keeps_train_and_val_apart(built):
    out, man = built
    train, val = man["per_seed"]["1"]["files"], man["val"]["files"]
    assert all(MS.jc2_role(f) == "ft_heldout" for f in train + val)
    assert not set(train) & set(val)
    assert man["overlap_files"] == {"pretrain_train": 0, "pretrain_fixed_val": 0,
                                    "ft_test": 0, "train_vs_val": 0}
    assert pq.read_table(out / "train_N100_s1.parquet").equals(
        pq.read_table(out / "train_N1000_s1.parquet").slice(0, 100))
    assert man["outputs"]["val.parquet"] == 512
    assert set(man["sha256"]) == set(man["outputs"])


def test_every_output_records_its_native_class_coverage(built):
    out, man = built
    assert set(man["class_coverage"]) == set(man["outputs"])
    for name, cov in man["class_coverage"].items():
        lab = pq.read_table(out / name).column("jet_label").to_numpy()
        counts = np.bincount(lab, minlength=188)
        assert cov["counts"] == counts.tolist()
        assert cov["n_classes_present"] == int((counts > 0).sum())
        assert cov["min_per_present_class"] == int(counts[counts > 0].min())
        assert cov["median_per_present_class"] == float(np.median(counts[counts > 0]))


def test_class_coverage_by_hand():
    c = MS.class_coverage([0, 0, 0, 5, 7, 7])
    assert c["n_classes_present"] == 3 and c["classes_absent"][:6] == [1, 2, 3, 4, 6, 8]
    assert c["median_per_present_class"] == 2.0 and c["min_per_present_class"] == 1
    assert c["max_per_present_class"] == 3 and c["n_classes_with_one_jet"] == 1
    with pytest.raises(SystemExit):
        MS.class_coverage([188])


# ------------------------------------------------------------ ft_v2.py

def _run_dir(tmp_path, values, kept=range(70, 80), best=None, done=True):
    """A pretrain_v2 run directory: metrics/epoch-EEE.json, best_epoch.json, kept states."""
    run = tmp_path / "mtx-l188-s1"
    (run / "metrics").mkdir(parents=True)
    for e, v in values.items():
        (run / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"epoch": e, "selection": {"metric": "val.acc", "value": v}}))
    if best is None:
        top = max(values.values())
        best = min(e for e, v in values.items() if v == top)
    (run / "best_epoch.json").write_text(json.dumps({"epoch": best, "metric": "val.acc"}))
    for e in kept:
        (run / f"net_epoch-{e}_state.pt").write_bytes(f"epoch {e}".encode())
    if done:
        (run / "DONE").write_text("{}\n")
    return run


def test_bestval_is_the_best_epoch_on_the_fixed_sample_loaded_by_name(tmp_path):
    run = _run_dir(tmp_path, {e: 0.5 + e / 1000 - (0.1 if e > 75 else 0) for e in range(80)})
    rec = FV.resolve(run, "bestval")
    assert rec["epoch"] == 75 and rec["path"].endswith("net_epoch-75_state.pt")
    assert rec["sha256"] == FV.sha256_file(run / "net_epoch-75_state.pt")


def test_bestval_ties_keep_the_first_as_the_driver_does(tmp_path):
    run = _run_dir(tmp_path, {e: 0.7 if e in (72, 77) else 0.5 for e in range(80)})
    assert FV.resolve(run, "bestval")["epoch"] == 72


def test_bestval_refuses_disagreeing_records_a_missing_epoch_and_an_unfinished_run(tmp_path):
    vals = {e: 0.7 if e == 74 else 0.5 for e in range(80)}
    with pytest.raises(SystemExit, match="best_epoch.json says epoch 71"):
        FV.resolve(_run_dir(tmp_path / "a", vals, best=71), "bestval")
    with pytest.raises(SystemExit, match="no record of epochs"):
        FV.resolve(_run_dir(tmp_path / "b", {e: v for e, v in vals.items() if e != 3}), "bestval")
    with pytest.raises(SystemExit, match="not complete"):
        FV.resolve(_run_dir(tmp_path / "c", vals, done=False), "wavg")


def test_a_selected_epoch_that_was_not_kept_is_fatal_not_substituted(tmp_path):
    run = _run_dir(tmp_path, {e: 0.9 if e == 40 else 0.5 for e in range(80)})
    (run / "net_best_epoch_state.pt").write_bytes(b"whatever")
    with pytest.raises(SystemExit, match="not kept"):
        FV.resolve(run, "bestval")


def _wavg(run, epochs=range(70, 80), bad_avg=False):
    (run / FV.WAVG_STATE).write_bytes(b"average")
    (run / FV.WAVG_JSON).write_text(json.dumps({
        "inputs": {str(e): FV.sha256_file(run / f"net_epoch-{e}_state.pt") for e in epochs},
        "sha256": "0" * 64 if bad_avg else FV.sha256_file(run / FV.WAVG_STATE)}))


def test_wavg_is_the_average_of_70_79_whose_inputs_are_the_files_in_the_run(tmp_path):
    run = _run_dir(tmp_path, {e: 0.5 for e in range(80)}, kept=range(60, 80))
    with pytest.raises(SystemExit, match="weight average was not written"):
        FV.resolve(run, "wavg")
    _wavg(run)
    rec = FV.resolve(run, "wavg")
    assert rec["path"].endswith(FV.WAVG_STATE) and rec["epoch"] == "wavg70-79"
    assert sorted(rec["inputs"], key=int) == [str(e) for e in range(70, 80)]
    _wavg(run, epochs=range(69, 79))
    with pytest.raises(SystemExit, match="averages epochs"):
        FV.resolve(run, "wavg")
    _wavg(run, bad_avg=True)
    with pytest.raises(SystemExit, match="not the file"):
        FV.resolve(run, "wavg")
    _wavg(run)
    (run / "net_epoch-75_state.pt").write_bytes(b"changed after averaging")
    with pytest.raises(SystemExit, match="input epoch 75"):
        FV.resolve(run, "wavg")
    with pytest.raises(SystemExit, match="unknown checkpoint rule"):
        FV.resolve(run, "e79")


def test_resolve_links_the_file_and_records_the_choice(tmp_path):
    run = _run_dir(tmp_path, {e: 0.5 for e in range(80)})
    _wavg(run)
    link = tmp_path / "ws" / "l188-s1.pt"
    assert FV.main(["resolve", "--run-dir", str(run), "--rule", "wavg", "--link", str(link)]) == 0
    assert link.read_bytes() == b"average"
    assert json.loads(pathlib.Path(f"{link}.json").read_text())["epoch"] == "wavg70-79"


def test_hash_then_verify_and_a_changed_byte_is_refused(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"abc")
    table = tmp_path / "t.json"
    assert FV.main(["hash", "--out", str(table), "--files", str(f)]) == 0
    assert FV.main(["verify", "--table", str(table), "--files", str(f)]) == 0
    f.write_bytes(b"abd")
    assert FV.main(["verify", "--table", str(table), "--files", str(f)]) == 1
    assert FV.main(["verify", "--table", str(table), "--files", str(tmp_path / "b")]) == 1


def test_weaver_args_are_read_from_the_runs_own_dump(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("[2026-09-19 11:07:00,000] INFO: args:\n"
                   " - ('batch_size', 512)\n - ('num_epochs', 50)\n"
                   " - ('steps_per_epoch', 19)\n - ('steps_per_epoch_val', None)\n"
                   " - ('samples_per_epoch', 10000)\n - ('lr_scheduler', 'flat+decay')\n"
                   "[2026-09-19 11:07:22,942] INFO: Epoch #0 training\n")
    a = FV.weaver_args(log)
    assert a == {"batch_size": 512, "num_epochs": 50, "steps_per_epoch": 19,
                 "steps_per_epoch_val": None, "samples_per_epoch": 10000,
                 "lr_scheduler": "flat+decay"}


def test_a_reused_cell_must_have_trained_on_recorded_files_as_they_were_when_it_ran(tmp_path):
    cell = tmp_path / "c"
    cell.mkdir()
    sub, val = "/data/finetune/top_sub/train_N1000_s1.parquet", "/data/finetune/top_sub/val.parquet"
    table = {sub: {"sha256": "x", "bytes": 1869174, "mtime_utc": "2026-09-07T22:18:07Z"},
             val: {"sha256": "v", "bytes": 9, "mtime_utc": "2026-09-07T22:18:44Z"}}
    man = {"subset": sub, "subset_bytes": 1869174, "written_utc": "2026-09-28T21:01:14Z"}

    def problem(**kw):
        (cell / "ft_manifest.json").write_text(json.dumps({**man, **kw}))
        return FV.ref_cell_problem(cell, table)

    assert problem() is None
    assert problem(subset_sha256="x", val_sha256="v") is None
    assert "bytes" in problem(subset_bytes=5)
    assert "not in the sha256 record" in problem(subset="/elsewhere.parquet")
    assert "after the cell ran" in problem(written_utc="2026-09-07T22:18:30Z")   # val rewritten later
    assert "not the record's" in problem(val_sha256="w")
    assert "no written_utc" in problem(written_utc=None)
    del table[val]
    assert f"{val} is not in the sha256 record" in problem()


def test_the_hash_record_is_written_whole_and_can_name_the_final_paths(tmp_path):
    stage = tmp_path / "jc2_v2.staging"
    stage.mkdir()
    (stage / "val.parquet").write_bytes(b"abc")
    out = stage / "ft_v2_subsets_sha256.json"
    FV.cmd_hash([str(stage / "val.parquet")], out, (str(stage), "/data/finetune/jc2_v2"))
    rec = json.loads(out.read_text())["files"]
    assert list(rec) == ["/data/finetune/jc2_v2/val.parquet"]
    assert rec["/data/finetune/jc2_v2/val.parquet"]["bytes"] == 3
    assert not list(stage.glob("*.tmp"))


# ------------------------------------------------------------ what was staged
# experiments/FT/data/jc2_v2_manifest.json and ft_v2_subsets_sha256.json are
# copies of /data/finetune/jc2_v2/{manifest.json, ft_v2_subsets_sha256.json},
# written by job-ft-subsets-jc2-v2-raunav.

STAGED = ROOT / "experiments/FT/data/jc2_v2_manifest.json"
TABLE = ROOT / "experiments/FT/data/ft_v2_subsets_sha256.json"


def test_the_staged_subsets_read_only_held_out_files_train_and_val_apart():
    man = json.loads(STAGED.read_text())
    assert man["mode"] == "jc2v2" and man["seeds"] == [1]
    train, val = man["per_seed"]["1"]["files"], man["val"]["files"]
    assert all(MS.jc2_role(f) == "ft_heldout" for f in train + val)
    assert not set(train) & set(val)
    assert man["overlap_files"] == {"pretrain_train": 0, "pretrain_fixed_val": 0,
                                    "ft_test": 0, "train_vs_val": 0}
    assert _names(man["pool_files"]) == _names(MS.jc2_role_files("ft_heldout"))
    assert man["outputs"] == {**{f"train_N{n}_s1.parquet": n for n in (1000, 10000, 100000, 1000000)},
                              "val.parquet": B.V2_VAL_JETS}


def test_the_staged_subsets_record_their_coverage():
    man = json.loads(STAGED.read_text())
    assert set(man["class_coverage"]) == set(man["outputs"])
    for name, cov in man["class_coverage"].items():
        assert sum(cov["counts"]) == man["outputs"][name]
        assert cov["n_classes_present"] == sum(c > 0 for c in cov["counts"])


def test_the_sha256_record_holds_the_staged_bytes_and_every_reused_file():
    man = json.loads(STAGED.read_text())
    table = json.loads(TABLE.read_text())["files"]
    for name, sha in man["sha256"].items():
        assert table[f"{B.V2_SUBSETS}/{name}"]["sha256"] == sha, name
    assert set(B.V2_REUSED) <= set(table)


def test_a_readout_fails_when_an_expected_cell_was_not_read(tmp_path):
    f = tmp_path / "cells.json"
    f.write_text(json.dumps({"bestval/leg1": ["l188-s1/N1000/s1", "l188-s1/N10000/s1"]}))
    FV.require_cells({"l188-s1/N1000/s1", "l188-s1/N10000/s1", "extra/N1/s1"}, f"{f}:bestval", "leg1")
    FV.require_cells(set(), None, "leg1")                    # no list given: nothing to check
    with pytest.raises(SystemExit, match="1 expected leg1 cells were not read"):
        FV.require_cells({"l188-s1/N1000/s1"}, f"{f}:bestval", "leg1")
    with pytest.raises(SystemExit, match="lists no cells for wavg/leg1"):
        FV.require_cells(set(), f"{f}:wavg", "leg1")
