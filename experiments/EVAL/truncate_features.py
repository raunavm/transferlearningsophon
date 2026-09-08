#!/usr/bin/env python3
"""Write the first N rows of a feature cache to a new directory.

WHY: the zero-fill control (docs/PRD_PLAN.md 4.5) compares masked against
unmasked features on THE SAME JETS. experiments/EVAL/probe.py enforces that by
refusing to run unless every supplied directory has an identical label188
sha256 -- which also means identical LENGTH. The unmasked baselines are 2 M
rows; re-extracting the masked pass at 2 M would cost 5x the GPU for statistics
the control does not need (the primary b-vs-c resonant probe already has
12,866 vs 13,218 jets in the first 400 k, measured). So the masked pass is
extracted at N and the baseline is truncated to the same N here.

This is only valid because extraction is deterministic in file order with
shuffle=False, so a fresh N-row pass sees the same jets as the first N rows of
a longer pass. That is an ASSUMPTION THIS SCRIPT DOES NOT VERIFY -- probe.py's
sha check does, and it will refuse the pair if it is ever false.

Run:  python3 experiments/EVAL/truncate_features.py --src <dir> --out <dir> --n 400000
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args(argv)

    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ref = None
    # Refuse --out == --src. Each array is opened with mmap_mode="r" and then
    # np.save reopens the SAME path for writing while that mapping is live, so
    # the destination is truncated before the mapped data is read: the cache
    # silently becomes all zeros and the run exits 0 claiming it wrote N rows.
    if out.resolve() == src.resolve():
        raise SystemExit(
            f"FATAL: --out equals --src ({src}); this would zero the source "
            f"cache in place. Write to a new directory.")

    for f in sorted(src.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        if arr.shape[0] < a.n:
            sys.exit(f"FATAL: {f.name} has {arr.shape[0]:,} rows, fewer than the "
                     f"{a.n:,} requested")
        np.save(out / f.name, np.asarray(arr[:a.n]))
        n_ref = a.n
        print(f"  {f.name}: {arr.shape[0]:,} -> {a.n:,}", flush=True)
    for f in sorted(src.glob("*.npz")):
        z = np.load(f)
        np.savez(out / f.name, **{k: z[k][:a.n] for k in z.files})
        print(f"  {f.name}: -> {a.n:,}", flush=True)
    for f in src.glob("*.json"):
        shutil.copy2(f, out / f.name)
    if n_ref is None:
        sys.exit(f"FATAL: no .npy arrays in {src}")
    print(f"wrote {out} at {a.n:,} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
