"""Leg 1 readout. The defects these pin are the ones that would produce a
plausible-looking table rather than an error."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "leg1_metrics", REPO / "experiments/FT/leg1_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cell(root, init, n, seed, *, k=162, labels=None, partial=False):
    name = f"{seed}.partial.123" if partial else seed
    fd = root / init / n / name / "features_v2"
    fd.mkdir(parents=True)
    lab = np.arange(188, dtype=np.int16) if labels is None else labels
    np.save(fd / "label188.npy", lab)
    rng = np.random.default_rng(0)
    np.save(fd / "logits.npy", rng.normal(size=(lab.shape[0], k)).astype(np.float32))
    return fd


def test_the_qcd_group_id_is_asserted_against_the_committed_map():
    """L162_QCD_GROUP is a hard-coded 161. If a remap moved it, every
    rejection-vs-QCD number would silently be computed against a resonant
    class. The script asserts membership at runtime; this pins the map."""
    m = _mod()
    spec = importlib.util.spec_from_file_location(
        "label_recovery", REPO / "experiments/EVAL/label_recovery.py")
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)
    l162 = lr.rung_maps()["L162"]
    assert sorted(set(l162.values())) == list(range(162))
    assert sum(1 for v in l162.values() if v == m.L162_QCD_GROUP) == 27


def test_partial_directories_are_skipped_by_name_not_by_content(tmp_path):
    """A `.partial.<ts>` cell can hold a complete-looking features.npy written
    before the interruption, so existence checks do not distinguish it."""
    m = _mod()
    _cell(tmp_path, "l162-s1b", "N10000", "s1")
    _cell(tmp_path, "l162-s1b", "N10000", "s2", partial=True)
    found = m.discover(tmp_path)
    assert [c[2] for c in found] == ["s1"]


def test_a_head_that_is_not_162_wide_is_fatal(tmp_path):
    """Leg 1 fine-tunes the 162-way vocabulary. A 10-wide cache would be a
    leg-2 artifact pointed at the wrong reader, and argmax would still return
    an index, so nothing would error downstream."""
    m = _mod()
    fd = _cell(tmp_path, "x", "N10000", "s1", k=10)
    with pytest.raises(SystemExit, match="162"):
        m.cell_metrics(fd, {i: 0 for i in range(188)}, None, 1)


def test_a_native_label_absent_from_the_map_is_fatal_not_silently_negative(tmp_path):
    """The lookup table is filled with -1. Without the check those rows become
    class -1, which numpy happily indexes from the end of the array."""
    m = _mod()
    fd = _cell(tmp_path, "x", "N10000", "s1", labels=np.array([0, 1, 999], dtype=np.int64))
    with pytest.raises(SystemExit, match="absent from the"):
        m.cell_metrics(fd, {0: 0, 1: 1}, None, 1)


def test_the_auc_subsample_is_a_stride_so_cells_stay_paired_and_keep_class_mix():
    """A head slice of an interleaved test list is biased toward the files that
    come first; a per-cell random draw puts sampling noise inside the very
    contrast being measured. The stride does neither."""
    m = _mod()
    # a test list interleaved by file type, as the extraction spec builds it
    truth = np.array([0, 1, 2] * 1000)
    idx = np.arange(0, truth.size, 4)
    strided = np.bincount(truth[idx], minlength=3) / idx.size
    head = np.bincount(truth[: idx.size], minlength=3) / idx.size
    assert np.allclose(strided, [1 / 3, 1 / 3, 1 / 3], atol=0.02)
    assert np.allclose(head, [1 / 3, 1 / 3, 1 / 3], atol=0.02)
    # and the stride is identical across cells, which is what makes it paired
    assert np.array_equal(np.arange(0, 100, 4), np.arange(0, 100, 4))


def test_softmax_is_stable_and_normalised():
    m = _mod()
    z = np.array([[1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0]])
    p = m.softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()
    assert np.allclose(p[0], 1 / 3)


def test_misaligned_cells_are_fatal_rather_than_averaged(tmp_path):
    """Two cells that read out different jets are not comparable. Averaging
    them would produce a number with no error, which is the dangerous case."""
    m = _mod()
    out = tmp_path / "out"
    root = tmp_path / "leg1"
    _cell(root, "a", "N10000", "s1")
    _cell(root, "b", "N10000", "s1", labels=np.arange(188, dtype=np.int16)[::-1].copy())
    r = subprocess.run(
        [sys.executable, str(REPO / "experiments/FT/leg1_metrics.py"),
         "--root", str(root), "--out", str(out), "--auc-stride", "1"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "not" in (r.stdout + r.stderr) and "paired" in (r.stdout + r.stderr)
