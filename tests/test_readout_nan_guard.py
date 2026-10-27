"""No readout takes a fine-tuning cell whose training loss went to NaN.

A diverged run still ends with a best-epoch checkpoint, and before the retry
logic of 2026-09-29 it could be marked DONE (one did diverge, leg1/scratch-v2/
N1000/s2; it crashed before DONE, by luck). leg1_metrics, leg2_metrics and
bench_metrics refuse such a cell at discovery; interrupted attempts beside it
(.partial.*) are skipped as before, whatever their log says.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L1 = _load("leg1_nan", "experiments/FT/leg1_metrics.py")
L2 = _load("leg2_nan", "experiments/FT/leg2_metrics.py")
BM = _load("bench_nan", "experiments/FT/bench_metrics.py")

GOOD = "INFO: Train AvgLoss: 0.52, AvgAcc: 0.81\n"
NAN = "INFO: Train AvgLoss: 0.52, AvgAcc: 0.81\nINFO: Train AvgLoss: nan, AvgAcc: 0.14\n"


def _leg1(root, log):
    cell = root / "scratch-v2" / "N1000" / "s2"
    (cell / "features_v2").mkdir(parents=True)
    for f in ("logits.npy", "label188.npy"):
        (cell / "features_v2" / f).touch()
    (cell / "train.log").write_text(log)
    return cell


def _leg2(root, log):
    cell = root / "scratch-v2" / "N1000" / "s2"
    cell.mkdir(parents=True)
    (cell / "predict.log").touch()
    (cell / "train.log").write_text(log)
    (root / "ref_e1arms-s1").mkdir()
    (root / "ref_e1arms-s1" / "predict.log").touch()       # predict only: no train.log
    return cell


def _bench(root, log):
    cell = root / "leg_top" / "scratch-v2" / "N1000" / "s2"
    cell.mkdir(parents=True)
    (cell / "DONE").touch()
    (cell / "train.log").write_text(log)
    return cell


def test_clean_cells_are_read(tmp_path):
    _leg1(tmp_path / "l1", GOOD)
    assert len(L1.discover(tmp_path / "l1")) == 1
    _leg2(tmp_path / "l2", GOOD)
    assert len(L2.discover(tmp_path / "l2")) == 2
    _bench(tmp_path / "b", GOOD)
    assert len(BM.discover(tmp_path / "b", "top")[0]) == 1


@pytest.mark.parametrize("make, read", [
    (_leg1, lambda r: L1.discover(r)),
    (_leg2, lambda r: L2.discover(r)),
    (_bench, lambda r: BM.discover(r, "top")),
])
def test_a_diverged_cell_is_refused(tmp_path, make, read):
    make(tmp_path, NAN)
    with pytest.raises(SystemExit, match="NaN loss"):
        read(tmp_path)


def test_a_diverged_interrupted_attempt_beside_a_clean_cell_is_ignored(tmp_path):
    cell = _bench(tmp_path, GOOD)
    old = cell.parent / "s2.partial.1790664952"
    old.mkdir()
    (old / "train.log").write_text(NAN)
    assert [c[3] for c in BM.discover(tmp_path, "top")[0]] == [cell]


@pytest.mark.parametrize("make, read", [
    (_leg1, lambda r, i: L1.discover(r, i)),
    (_leg2, lambda r, i: L2.discover(r, i)),
    (_bench, lambda r, i: BM.discover(r, "top", i)[0]),
])
def test_inits_restricts_discovery_so_an_unfinished_run_is_never_opened(tmp_path, make, read):
    # --inits: a read-out of the runs finished so far does not open the cells of a run still
    # fine-tuning (here a diverged one, which would stop it), and reads the finished runs as before
    import shutil
    for name, log in (("run-a", NAN), ("run-b", GOOD)):
        shutil.rmtree(tmp_path / "a" / "ref_e1arms-s1", ignore_errors=True)   # _leg2 makes it each time
        make(tmp_path / "a", log)
        init = next(d for d in (tmp_path / "a").rglob("scratch-v2") if d.is_dir())
        init.rename(init.with_name(name))
    with pytest.raises(SystemExit, match="NaN loss"):
        read(tmp_path / "a", None)
    got = read(tmp_path / "a", {"run-b", "ref_e1arms-s1"})
    assert {c[0] for c in got} <= {"run-b", "ref_e1arms-s1"} and "run-b" in {c[0] for c in got}
    assert read(tmp_path / "a", {"nothing"}) == []
