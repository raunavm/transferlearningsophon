"""experiments/MTX/stream_ids.py: reading v2 stream records and checking pairing."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "MTX"))
import stream_ids  # noqa: E402


def _write(run, epoch, rows, files="f" * 64):
    rec = {"run": run.name, "epoch": epoch, "seed_data": 1, "seed_dropout": 2, "files_sha256": files,
           "rows_sha256": rows, "sha256": hashlib.sha256((files + rows).encode()).hexdigest(), "n_jets": 10}
    (run / "stream").mkdir(parents=True, exist_ok=True)
    (run / "stream" / f"epoch-{epoch:03d}.json").write_text(json.dumps(rec))
    return rec["sha256"]


def test_load_stream_maps_epochs_to_the_combined_hash(tmp_path):
    h = {e: _write(tmp_path / "a", e, f"{e:064x}") for e in range(3)}
    assert stream_ids.load_stream(tmp_path / "a") == h


def test_assert_paired_passes_on_equal_streams_and_names_the_epochs_that_differ(tmp_path):
    for e in range(4):
        _write(tmp_path / "a", e, f"{e:064x}")
        _write(tmp_path / "b", e, f"{e:064x}" if e != 2 else "9" * 64)
    assert stream_ids.assert_paired(tmp_path / "a", tmp_path / "b", epochs=[0, 1, 3]) == [0, 1, 3]
    with pytest.raises(AssertionError, match=r"\[2\]"):
        stream_ids.assert_paired(tmp_path / "a", tmp_path / "b")


def test_an_epoch_missing_from_one_run_is_unpaired(tmp_path):
    for e in range(3):
        _write(tmp_path / "a", e, f"{e:064x}")
    _write(tmp_path / "b", 0, f"{0:064x}")
    with pytest.raises(AssertionError, match=r"\[1, 2\]"):
        stream_ids.assert_paired(tmp_path / "a", tmp_path / "b")


def test_a_tampered_record_is_refused(tmp_path):
    _write(tmp_path / "a", 0, "0" * 64)
    p = tmp_path / "a" / "stream" / "epoch-000.json"
    rec = json.loads(p.read_text())
    rec["rows_sha256"] = "1" * 64
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="sha256"):
        stream_ids.load_stream(tmp_path / "a")
