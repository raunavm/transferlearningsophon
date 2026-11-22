"""experiments/EVAL/storage_scan.py (PI 2026-11-21): a file that lost its tail, a zero-filled tail, a broken zip member
and a file whose recorded sha256 no longer matches are all listed; intact files are not."""
import calendar
import hashlib
import importlib.util
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("storage_scan", ROOT / "experiments/EVAL/storage_scan.py")
SS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SS)


def test_every_kind_of_damage_is_listed_and_intact_files_pass(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    (d / "ok.json").write_text(json.dumps({"a": 1}))
    (d / "cut.json").write_text(json.dumps({"a": list(range(100))})[:-20])
    np.savez(d / "ok.npz", x=np.arange(1000.0))
    np.savez(d / "crc.npz", x=np.arange(1000.0))
    raw = bytearray((d / "crc.npz").read_bytes())
    raw[300] ^= 0xFF                                    # a flipped byte inside the member: its CRC fails
    (d / "crc.npz").write_bytes(bytes(raw))
    np.save(d / "ok.npy", np.arange(5000.0))
    np.save(d / "short.npy", np.arange(5000.0))
    (d / "short.npy").write_bytes((d / "short.npy").read_bytes()[:-800])
    (d / "tail.bin").write_bytes(b"x" * 10000 + b"\0" * 5000)
    (d / "rec.bin").write_bytes(b"payload")
    ok_hash = hashlib.sha256((d / "ok.npy").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({"rec": str(d / "rec.bin"), "rec_sha256": hashlib.sha256(b"x").hexdigest(),
                                                 "ok": str(d / "ok.npy"), "ok_sha256": ok_hash}))
    for p in d.iterdir():
        os.utime(p, (SS.LO + 3600, SS.LO + 3600))
    os.utime(d / "rec.bin", (SS.LO + 60, SS.LO + 60))   # older than its manifest: a damaged file, not a stale record
    monkeypatch.setattr(SS, "ROOTS", (str(d),))
    monkeypatch.setattr(SS, "PREFIX", str(d) + "/")
    out = tmp_path / "out"
    SS.main(["--out", str(out)])
    rows = [ln.split("\t") for ln in (out / "report.tsv").read_text().splitlines()[1:]]
    got = {(pathlib.Path(r[0]).name, r[2]) for r in rows}
    assert got == {("cut.json", "load"), ("crc.npz", "load"), ("short.npy", "load"), ("tail.bin", "zero-filled"),
                   ("rec.bin", "hash differs")}, got
    s = json.loads((out / "summary.json").read_text())
    assert s["files_rehashed"] == 2 and s["files_loaded"] == 9
    assert (out / "absent.tsv").read_text().splitlines() == ["path\trecorded_sha256\tmanifest"]


def test_a_failure_window_loads_only_the_files_written_inside_it_under_its_roots(tmp_path, monkeypatch):
    """guard v7 (PI 2026-11-22): after a storage failure, every file written during it is load-checked: --since/--until
    set the window, --roots the folders, --no-records skips the manifests' hashes."""
    d, other = tmp_path / "data" / "results", tmp_path / "data" / "elsewhere"
    d.mkdir(parents=True)
    other.mkdir()
    lo = calendar.timegm((2026, 11, 23, 1, 0, 0))
    (d / "in.json").write_text(json.dumps({"a": list(range(100))})[:-20])     # cut: fails, written inside
    (d / "before.json").write_text(json.dumps({"a": list(range(100))})[:-20])  # cut, but written before the window
    (d / "ok.json").write_text(json.dumps({"a": 1}))
    (other / "in.json").write_text(json.dumps({"a": list(range(100))})[:-20])  # inside, but not under --roots
    (d / "manifest.json").write_text(json.dumps({"x": str(d / "ok.json"), "x_sha256": "0" * 64}))
    for p in (d / "in.json", d / "ok.json", other / "in.json", d / "manifest.json"):
        os.utime(p, (lo + 600, lo + 600))
    os.utime(d / "before.json", (lo - 600, lo - 600))
    monkeypatch.setattr(SS, "PREFIX", str(tmp_path / "data") + "/")
    for k in ("LO", "HI", "ROOTS"):                     # main() sets them from the flags; restored after the test
        monkeypatch.setattr(SS, k, getattr(SS, k))
    out = tmp_path / "out"
    SS.main(["--out", str(out), "--since", "2026-11-23T01:00:00Z", "--until", "2026-11-23T02:30:00Z",
             "--roots", str(d), "--no-records"])
    rows = [ln.split("\t") for ln in (out / "report.tsv").read_text().splitlines()[1:]]
    assert [(pathlib.Path(r[0]).parent.name, pathlib.Path(r[0]).name, r[2]) for r in rows] == [("results", "in.json", "load")]
    s = json.loads((out / "summary.json").read_text())
    assert s["files_loaded"] == 3 and s["recorded_hashes"] == 0 and s["files_rehashed"] == 0
    assert s["window_utc"] == ["2026-11-23T01:00:00Z", "2026-11-23T02:30:00Z"]


def test_the_window_scan_job_is_ours_cpu_only_on_our_volume_at_a_tag():
    spec_b = importlib.util.spec_from_file_location("build_storage_scan_job", ROOT / "scripts/build_storage_scan_job.py")
    B = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(B)
    name, out, text = B.spec("mtx-s3.39", "2026-11-23T01:00:00Z", "2026-11-23T02:30:00Z")
    assert name == "storage-scan-w20261123t0100-0230-raunav" and len(name) <= 63
    assert out == "/data/results/eval/storage_scan_window/2026-11-23T01:00:00Z_2026-11-23T02:30:00Z"
    assert "--branch mtx-s3.39" in text and "--since 2026-11-23T01:00:00Z --until 2026-11-23T02:30:00Z" in text
    assert "--no-records" in text and "claimName: transfer-learning-vol" in text and "nvidia" not in text
    assert "ConfigMap" not in text and "kind: Secret" not in text
    import pytest
    with pytest.raises(SystemExit):
        B.spec("mtx-s3.39", "2026-11-23T02:30:00Z", "2026-11-23T01:00:00Z")
