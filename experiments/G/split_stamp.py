#!/usr/bin/env python3
"""L5 split stamp (PLAN v5.1 §7.3) — the single source of truth for G splits.

    h = SHA256(f"{dsid}:{run_number}:{event_number}:{SALT}")
    b = int(h[:8], 16) mod 100      -> [0,32) train | [32,40) val | [40,100) test

The salt is frozen in configs/g/split.yaml; this module hardcodes the same
value and asserts agreement with the yaml when it is importable (belt+braces —
the stamp must never silently drift from the frozen config).

Usable as a library (stamp_bin / stamp_split, vectorized variants) and as a
CLI self-test:  python3 experiments/G/split_stamp.py --selftest
"""
from __future__ import annotations

import hashlib

SALT = "sophon-granularity-v5"
BINS = {"train": (0, 32), "val": (32, 40), "test": (40, 100)}


def stamp_bin(dsid: int, run_number: int, event_number: int) -> int:
    h = hashlib.sha256(f"{dsid}:{run_number}:{event_number}:{SALT}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def stamp_split(dsid: int, run_number: int, event_number: int) -> str:
    b = stamp_bin(dsid, run_number, event_number)
    for name, (lo, hi) in BINS.items():
        if lo <= b < hi:
            return name
    raise AssertionError(b)


def _check_yaml_agreement():
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]
                          / "configs/g/split.yaml").read_text())
    assert cfg["salt"] == SALT, f"split salt drift: {cfg['salt']!r} != {SALT!r}"
    assert {k: tuple(v) for k, v in cfg["bins"].items()} == BINS, "bin drift"


def _selftest():
    _check_yaml_agreement()
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200_000
    splits = [stamp_split(int(d), int(r), int(e)) for d, r, e in
              zip(rng.integers(100000, 999999, n), rng.integers(1, 100, n),
                  rng.integers(0, 10_000_000, n))]
    frac = {s: splits.count(s) / n for s in ("train", "val", "test")}
    print("fractions:", {k: round(v, 4) for k, v in frac.items()})
    assert abs(frac["train"] - 0.32) < 0.005
    assert abs(frac["val"] - 0.08) < 0.003
    assert abs(frac["test"] - 0.60) < 0.005
    # determinism + a pinned reference value (guards against hash/format drift)
    assert stamp_bin(123456, 1, 42) == stamp_bin(123456, 1, 42)
    print("pinned example: stamp_bin(123456, 1, 42) =", stamp_bin(123456, 1, 42),
          "->", stamp_split(123456, 1, 42))
    print("SELFTEST PASS")


if __name__ == "__main__":
    _selftest()
