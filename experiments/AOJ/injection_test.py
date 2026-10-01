#!/usr/bin/env python3
"""Signal injection into the real-data top fit: is a known signal recovered without bias?

The checks of 2026-09-30 (realdata_checks.py, injection step) added a Gaussian to the
passing data of a signal-free pseudo-window (250-310 GeV) and fitted it back with the
shape floating: recovered at a median of 0.72 of its size, mean pull -1.7. The fits see
the jets only through their (m_SD, pT) bins, so everything here runs from bins.

  bins    from the merged jets of the first run (the scores the main fit saw): for every
          score, the pseudo-window bins at 1 % (the map with the pseudo window masked, as
          realdata_checks._injection_one builds them) and the top bins at the extra working
          points EXTRA_EFF. The top bins at 1 % are rebuilt too and must equal the committed
          fit_v3/bins.npz, or nothing is written: then these are the jets the fit saw.

Usage (in a job, after merge_shards.py over the first run's shards):
    python3 experiments/AOJ/injection_test.py bins --merged /scratch/merged \\
        --committed experiments/FIGS/data/aoj_full_v1/fit_v3/bins.npz --out OUT/injection_bins.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("peak_fit", HERE / "peak_fit.py")
RC = _load("realdata_checks", HERE / "realdata_checks.py")

KEYS = ("m_edges", "i", "j", "n_pass", "n_fail", "rho", "pt")
EFF = 0.01
EXTRA_EFF = (0.005, 0.02)
PSEUDO = RC.PSEUDO
TOP = P.PEAKS["top"]
BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pseudo_bins(z, mass, pt, eff=EFF):
    """The pseudo-window bins exactly as realdata_checks._injection_one makes them."""
    masked = P.MASKED + (PSEUDO["window"],)
    passed = P.passes(z, mass, pt, P.build_map(z, mass, pt, eff, masked=masked))
    return P._bins(mass, pt, passed, PSEUDO["fit_range"])


def top_bins(z, mass, pt, eff):
    """The top bins at data efficiency `eff`, as peak_fit.analyse makes them."""
    return P._bins(mass, pt, P.passes(z, mass, pt, P.build_map(z, mass, pt, eff)), TOP["fit_range"])


_W: dict = {}


def _init(merged):
    _W["data"] = RC.load_data(pathlib.Path(merged), kinds=("three_prong",))


def _export_one(name):
    d = _W["data"]
    z = P.logit(d["pn"]) if name == "reference" else d["scores"][name]["three_prong"].astype(float)
    m, pt = d["mass"], d["pt"]
    parts = {"main": top_bins(z, m, pt, EFF), "pseudo": pseudo_bins(z, m, pt)}
    parts.update({f"main_eff{e:g}": top_bins(z, m, pt, e) for e in EXTRA_EFF})
    return name, parts


def export(merged, committed, out, workers):
    c = np.load(committed)
    names = sorted({k.split("|")[0] for k in c.files})
    saved = {k: os.environ.get(k) for k in BLAS}
    os.environ.update({k: "1" for k in BLAS})
    try:
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(merged),)) as pool:
                rows = pool.map(_export_one, names, chunksize=1)
        else:
            _init(merged)
            rows = [_export_one(n) for n in names]
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    arrays = {}
    for name, parts in rows:
        for k in KEYS:
            if not np.array_equal(parts["main"][k], c[f"{name}|main|{k}"]):
                raise SystemExit(f"FATAL: {name} top bins at 1 % differ from {committed} ({k}); "
                                 "these are not the jets the fit saw; nothing written")
        for part, b in parts.items():
            if part != "main":
                arrays.update({f"{name}|{part}|{k}": b[k] for k in KEYS})
        print(f"{name:14s} pseudo {len(parts['pseudo']['n_pass'])} bins {parts['pseudo']['n_pass'].sum():.0f} pass; "
              + "; ".join(f"top at {e:g}: {parts[f'main_eff{e:g}']['n_pass'].sum():.0f} pass" for e in EXTRA_EFF)
              + "; top at 1 % = committed", flush=True)
    out = pathlib.Path(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"wrote {out}: {len(names)} scores x (pseudo, {', '.join(f'main_eff{e:g}' for e in EXTRA_EFF)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bins")
    b.add_argument("--merged", required=True, type=pathlib.Path)
    b.add_argument("--committed", required=True, type=pathlib.Path)
    b.add_argument("--out", required=True, type=pathlib.Path)
    b.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "bins":
        export(a.merged, a.committed, a.out, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
