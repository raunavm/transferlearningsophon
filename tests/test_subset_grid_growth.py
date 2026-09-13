"""The leg grid and the subset grid must not be able to drift apart again.

WHAT WENT WRONG. DECISIONS_PENDING item 25 added N=1e3 to `SIZES` in
scripts/build_ft_jobs.py, so the legs loop over 1000/10000/100000/1000000. The
two subset builders were left calling `--sizes 10000 100000 1000000`, so the
N=1e3 subsets were never written. Nothing failed at build time: the legs would
have run for days and then died on a missing train_N1000_s1.parquet.

TWO GUARDS WERE IN PLACE AND NEITHER FIRED. make_subsets.py compares the stored
manifest against the request and refuses a grid change -- but the JOB SPEC ran
`[ -f ${OUT}/DONE ] && exit 0` in front of it, so the careful check never
executed and the job printed "already built" instead. A crude guard sitting in
front of a careful one is worse than no guard: it produces a confident green.

THE FIX HAS A SHARP EDGE, which is most of what these tests cover. Growing the
grid DOWNWARD is safe because no builder consumes the RNG as a function of
`sizes`, so a new small size is a true nested prefix of the subsets already on
disk. Growing it UPWARD is not: jc1 fixes its per-class draw at
`max(sizes) // len(classes)` and jc2/bench size their pool check against
max(sizes), so raising the top changes the pool itself and the existing subsets
stop being prefixes of it. The refusal must survive.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location(
        "make_subsets_grid", ROOT / "experiments" / "FT" / "make_subsets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bfj():
    spec = importlib.util.spec_from_file_location(
        "build_ft_jobs_grid", ROOT / "scripts" / "build_ft_jobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jc2_file(path: pathlib.Path, n: int, seed: int, family: str):
    rng = np.random.default_rng(seed)
    t = pa.table({"jet_pt": rng.uniform(100.0, 3000.0, n),
                  "jet_sdmass": rng.uniform(0.0, 600.0, n),
                  "jet_label": rng.integers(0, 188, n),
                  "family": pa.array([family] * n)})
    pq.write_table(t, path)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("grid")
    out = []
    for fam, nf in (("Res2P", 3), ("Res34P", 5), ("QCD", 3)):
        for i in range(nf):
            p = d / f"{fam}_{i:04d}.parquet"
            _jc2_file(p, 4000, seed=abs(hash((fam, i))) % 9_999, family=fam)
            out.append(str(p))
    return out


def _build(ms, files, out, sizes):
    """Run the CLI exactly as the job spec does."""
    return ms.main(["jc2", "--train-files", *files[1:], "--val-files", files[0],
                    "--out", str(out), "--sizes", *[str(s) for s in sizes],
                    "--seeds", "1", "2", "--n-files", "6", "--take-fraction", "0.5",
                    "--val-size", "200", "--n-val-files", "1"])


# --------------------------------------------------------- the safe direction

def test_growing_the_grid_downward_writes_only_the_new_size(ms, files, tmp_path):
    out = tmp_path / "g"
    assert _build(ms, files, out, [100, 1000]) == 0
    before = {p.name: p.read_bytes() for p in sorted(out.glob("*.parquet"))}
    assert "train_N10_s1.parquet" not in before

    assert _build(ms, files, out, [10, 100, 1000]) == 0
    after = {p.name: p.read_bytes() for p in sorted(out.glob("*.parquet"))}

    assert "train_N10_s1.parquet" in after and "train_N10_s2.parquet" in after
    for name, blob in before.items():
        assert after[name] == blob, (
            f"{name} was rewritten; growing the grid must not touch a byte of "
            "the subsets a completed run was trained on")


def test_the_size_added_by_the_growth_is_a_true_nested_prefix(ms, files, tmp_path):
    """THE LOAD-BEARING PROPERTY. If this fails, the N=1e3 cell is drawn from a
    different shuffle than the N=1e4 cell and the two are not on one curve."""
    out = tmp_path / "p"
    _build(ms, files, out, [100, 1000])
    _build(ms, files, out, [10, 100, 1000])
    for s in (1, 2):
        small = pq.read_table(out / f"train_N10_s{s}.parquet")
        big = pq.read_table(out / f"train_N100_s{s}.parquet")
        assert small.num_rows == 10
        assert small.equals(big.slice(0, 10))


def test_the_manifest_records_the_grown_grid(ms, files, tmp_path):
    out = tmp_path / "m"
    _build(ms, files, out, [100, 1000])
    _build(ms, files, out, [10, 100, 1000])
    assert json.loads((out / "manifest.json").read_text())["sizes"] == [10, 100, 1000]


def test_a_second_identical_request_is_still_a_no_op(ms, files, tmp_path, capsys):
    out = tmp_path / "n"
    _build(ms, files, out, [100, 1000])
    capsys.readouterr()
    assert _build(ms, files, out, [100, 1000]) == 0
    assert "nothing to do" in capsys.readouterr().out


# ------------------------------------------------------- the unsafe direction

def test_raising_the_maximum_is_still_refused(ms, files, tmp_path):
    """jc1 draws `max(sizes) // len(classes)` per class and jc2 checks the pool
    against max(sizes), so a higher top changes the pool and the subsets on disk
    stop being prefixes of it."""
    out = tmp_path / "u"
    _build(ms, files, out, [100, 1000])
    with pytest.raises(SystemExit, match="different parameters"):
        _build(ms, files, out, [100, 1000, 2000])


def test_dropping_a_size_is_refused(ms, files, tmp_path):
    """Only a strict superset is a growth; anything else is a different grid."""
    out = tmp_path / "d"
    _build(ms, files, out, [100, 1000])
    with pytest.raises(SystemExit, match="different parameters"):
        _build(ms, files, out, [1000])


def test_a_non_size_mismatch_is_untouched_by_the_growth_path(ms, files, tmp_path):
    """The growth branch must not become a hole in the manifest comparison."""
    out = tmp_path / "t"
    _build(ms, files, out, [100, 1000])
    with pytest.raises(SystemExit, match="different parameters"):
        ms.main(["jc2", "--train-files", *files[1:], "--val-files", files[0],
                 "--out", str(out), "--sizes", "100", "1000",
                 "--seeds", "1", "2", "--n-files", "6", "--take-fraction", "0.9",
                 "--val-size", "200", "--n-val-files", "1"])


def test_write_subset_keeps_what_is_already_there(ms, tmp_path):
    (tmp_path / "train_N5_s1.parquet").write_bytes(b"original")
    wrote = ms.write_subset(tmp_path, 5, 1, lambda d: d.write_bytes(b"new"),
                            skip_existing=True)
    assert wrote is False
    assert (tmp_path / "train_N5_s1.parquet").read_bytes() == b"original"
    wrote = ms.write_subset(tmp_path, 5, 1, lambda d: d.write_bytes(b"new"),
                            skip_existing=False)
    assert wrote is True
    assert (tmp_path / "train_N5_s1.parquet").read_bytes() == b"new"


# ------------------------------------------------- the drift that started it

def test_the_subset_builders_track_the_leg_grid(bfj):
    """THE REGRESSION TEST FOR THE ACTUAL BUG. The legs loop over SIZES; the
    subset builders must write exactly those sizes, or the legs die days later
    on a missing parquet."""
    want = " ".join(str(s) for s in bfj.SIZES)
    specs = bfj.build(bfj.PIN)
    for name in ("job-ft-subsets-jc2-raunav.yaml", "job-ft-subsets-jc1-raunav.yaml"):
        args = specs[name]
        assert f"--sizes {want} " in args, (
            f"{name} does not pass SIZES ({want}); the subset grid and the leg "
            "grid have drifted apart again")


def test_no_bare_done_guard_masks_the_manifest_comparison(bfj):
    """make_subsets.py compares the stored manifest against the request. A
    `[ -f DONE ] && exit 0` in front of it turns a real grid change into a
    printed 'already built' and an exit 0."""
    specs = bfj.build(bfj.PIN)
    for name, text in specs.items():
        if "make_subsets.py" not in text:
            continue
        assert "DONE ] && { echo \"already built" not in text, (
            f"{name} short-circuits on DONE before make_subsets.py can compare "
            "the manifest against the request")
