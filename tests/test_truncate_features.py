"""Truncation must preserve row correspondence across every array in a cache.

The zero-fill control compares masked and unmasked features on the same jets,
and probe.py only accepts the pair if their label188 arrays are byte-identical.
If truncation took different rows from different arrays -- or silently produced
a shorter array than asked for -- the control would either fail confusingly or,
worse, compare mismatched rows within one directory.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "truncate_features", ROOT / "experiments" / "EVAL" / "truncate_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tf = _load()


def _cache(d, n):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "features.npy", np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    np.save(d / "label188.npy", np.arange(n, dtype=np.int16) % 188)
    np.savez(d / "observers.npz", jet_pt=np.arange(n, dtype=np.float64))
    (d / "extract_manifest.json").write_text('{"k": 1}')


def test_every_array_is_truncated_to_the_same_prefix(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _cache(src, 1000)
    tf.main(["--src", str(src), "--out", str(out), "--n", "250"])
    f = np.load(out / "features.npy")
    l = np.load(out / "label188.npy")
    o = np.load(out / "observers.npz")["jet_pt"]
    assert f.shape == (250, 3) and l.shape == (250,) and o.shape == (250,)
    # row i of every array must still be the same jet as before
    assert np.array_equal(f, np.load(src / "features.npy")[:250])
    assert np.array_equal(l, np.load(src / "label188.npy")[:250])
    assert np.array_equal(o, np.load(src / "observers.npz")["jet_pt"][:250])


def test_manifest_is_carried_over(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _cache(src, 100)
    tf.main(["--src", str(src), "--out", str(out), "--n", "50"])
    assert (out / "extract_manifest.json").read_text() == '{"k": 1}'


def test_asking_for_more_rows_than_exist_fails_loudly(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _cache(src, 100)
    with pytest.raises(SystemExit) as e:
        tf.main(["--src", str(src), "--out", str(out), "--n", "500"])
    assert "fewer than" in str(e.value)


def test_truncated_labels_match_a_fresh_prefix_bytewise(tmp_path):
    """probe.py compares label188 by sha256, so bytes -- not values -- must match."""
    import hashlib
    src, out = tmp_path / "src", tmp_path / "out"
    _cache(src, 1000)
    tf.main(["--src", str(src), "--out", str(out), "--n", "400"])
    a = np.asarray(np.load(src / "label188.npy")[:400]).tobytes()
    b = np.load(out / "label188.npy").tobytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_empty_source_fails_rather_than_writing_nothing(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    with pytest.raises(SystemExit) as e:
        tf.main(["--src", str(src), "--out", str(out), "--n", "10"])
    assert "no .npy arrays" in str(e.value)
