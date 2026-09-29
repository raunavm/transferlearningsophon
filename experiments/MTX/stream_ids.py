"""Read the per-epoch stream records of v2 pretraining runs, and check pairing.

Each v2 run writes <run_dir>/stream/epoch-EEE.json (experiments/MTX/pretrain_v2.py)
with keys run, epoch, seed_data, seed_dropout, files_sha256, rows_sha256, sha256
and n_jets, where sha256 = sha256(files_sha256 + rows_sha256): the schedule of
files the epoch was drawn from and the jets it consumed, in order. Two runs are
paired at epoch e only if their sha256 values at e match.

Stable interface (the evaluation code imports it):
    load_stream(run_dir) -> {epoch: sha256}
    assert_paired(run_dir_a, run_dir_b, epochs=None) -> list of epochs compared
"""
from __future__ import annotations

import hashlib
import json
import pathlib

KEYS = {"run", "epoch", "seed_data", "seed_dropout", "files_sha256", "rows_sha256", "sha256", "n_jets"}


def load_stream(run_dir) -> dict:
    """{epoch: sha256} for every record in <run_dir>/stream/. A record whose
    keys, epoch or combined hash disagree with its file raises ValueError."""
    out = {}
    for p in sorted(pathlib.Path(run_dir, "stream").glob("epoch-*.json")):
        rec = json.loads(p.read_text())
        if set(rec) != KEYS:
            raise ValueError(f"{p}: keys {sorted(rec)} are not {sorted(KEYS)}")
        e = int(p.stem.split("-", 1)[1])
        if rec["epoch"] != e:
            raise ValueError(f"{p}: holds epoch {rec['epoch']}")
        if rec["sha256"] != hashlib.sha256((rec["files_sha256"] + rec["rows_sha256"]).encode()).hexdigest():
            raise ValueError(f"{p}: sha256 is not sha256(files_sha256 + rows_sha256)")
        out[e] = rec["sha256"]
    return out


def assert_paired(run_dir_a, run_dir_b, epochs=None) -> list:
    """Raise AssertionError unless the two runs consumed the same stream at every
    epoch in `epochs` (default: every epoch either run recorded). An epoch one
    run has not recorded is unpaired. Returns the epochs compared."""
    a, b = load_stream(run_dir_a), load_stream(run_dir_b)
    want = sorted(set(a) | set(b)) if epochs is None else sorted(epochs)
    if not want:
        raise AssertionError(f"no stream records in {run_dir_a} or {run_dir_b}")
    bad = [e for e in want if e not in a or e not in b or a[e] != b[e]]
    if bad:
        raise AssertionError(f"{run_dir_a} and {run_dir_b} are not paired at epochs {bad}")
    return want
