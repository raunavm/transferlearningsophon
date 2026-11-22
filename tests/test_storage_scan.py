"""experiments/EVAL/storage_scan.py (PI 2026-11-21): a file that lost its tail, a zero-filled tail, a broken zip member
and a file whose recorded sha256 no longer matches are all listed; intact files are not."""
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
