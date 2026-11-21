#!/usr/bin/env python3
"""Each mass estimator against the soft-drop mass, paired on the same jets (PI 2026-11-21): the decision behind every
"better / worse than the soft-drop mass" in Table 3, Table A9, Figs. A9 and A10 and the mass detail table.

Input: the truth-free refit's per-jet values only (experiments/EVAL/atlas_resolution.py --centre global,
/data/results/eval/atlas_res_v3/<model>_perjet.npz: rows, label188, jet_pt, genjet_sdmass, m_groomed, m_regressed for
the networks with the mass output, m_ridge, m_mlp). Every network's file must hold the same jets (rows) or the run stops.

For a group of runs (a label set, a mass network, ...) and an estimator, in each jet group (resonant, qcd, all) and bin
(inclusive; the coarse jet-pT and generator-mass bins and the ten jet-pT bins of the figures): the ATLAS resolution
(atlas_resolution: half the 15.9-84.1 % range of r = m/m_gen over its median) of each run and of the soft-drop mass.
    delta    mean over the runs of the estimator's resolution minus the soft-drop mass's (negative: better)
    boot_sd  SD of delta over BOOT_RESAMPLES jet resamples drawn once per bin (BOOT_SEED) and shared by the soft-drop
             mass and every run, so the comparison is paired
    sd_runs  SD of the estimator's resolution over the runs (ddof 1; the soft-drop mass has no run spread)
    sigma    sqrt(sd_runs^2 + boot_sd^2); pass when |delta| > 3 sigma (one run: no sd_runs, no claim)
A bin with fewer than MIN_JETS jets is left out, as in the tables.

    python3 experiments/EVAL/mass_sd_boot.py --root /data/results/eval/atlas_res_v3 --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AR = _load("atlas_resolution", "experiments/EVAL/atlas_resolution.py")
RB = _load("atlas_rebin", "experiments/EVAL/atlas_rebin.py")
Q = AR.QUANTILES
GROUPS = {"188 classes": ("mtx-l188-s1", "mtx-l188-s2", "mtx-l188-s3"),
          "162 classes": ("mtx-l162-s1", "mtx-l162-s2", "mtx-l162-s3"),
          "43 classes": ("mtx-r42q1-s1", "mtx-r42q1-s2", "mtx-r42q1-s3"),
          "17 classes": ("mtx-r16q1-s1", "mtx-r16q1-s2", "mtx-r16q1-s3"),
          "162 classes + mass": ("mtx-l162mass-s1", "mtx-l162mass-s2"),
          "17 classes + mass": ("mtx-r16q1mass-s1", "mtx-r16q1mass-s2"),
          "no flavor boundary": ("mtx-flavf0-s1", "mtx-flavf0-s2"), "b-quark boundary": ("mtx-flavf1-s1", "mtx-flavf1-s2"),
          "random flavor boundary": ("mtx-flavf1r-s1", "mtx-flavf1r-s2"),
          "random partitions": ("mtx-rand2p1-s1", "mtx-rand2p2-s2", "mtx-rand2p3-s1", "mtx-rand2p4-s2", "mtx-rand2p5-s2"),
          "random network": ("init-s1", "init-s2", "init-s3")}
ESTIMATORS = ("regressed", "ridge", "mlp")
BINNINGS = {"pt": AR.PT_EDGES, "mgen": AR.MGEN_EDGES, "pt_fine": RB.PT_EDGES}


def resolution(r: np.ndarray) -> float:
    lo, m, hi = np.quantile(r, (Q[0], 0.5, Q[1]))
    return 0.5 * (hi - lo) / m


def compare(sd: np.ndarray, est: list, rng_seed: int) -> dict:
    """sd: soft-drop responses of the bin's jets; est: one response array per run on the same jets."""
    n = sd.size
    point = [resolution(r) for r in est]
    delta = float(np.mean(point) - resolution(sd))
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(AR.BOOT_RESAMPLES)
    for b in range(AR.BOOT_RESAMPLES):
        i = rng.integers(0, n, n)
        diffs[b] = np.mean([resolution(r[i]) for r in est]) - resolution(sd[i])
    boot_sd = float(diffs.std(ddof=1))
    sd_runs = float(np.std(point, ddof=1)) if len(point) > 1 else None
    sigma = float(np.hypot(sd_runs, boot_sd)) if sd_runs is not None else None
    return {"n": int(n), "softdrop": resolution(sd), "runs": point, "delta": delta, "boot_sd": boot_sd,
            "sd_runs": sd_runs, "sigma": sigma, "z": delta / sigma if sigma else None,
            "pass": bool(sigma is not None and abs(delta) > 3 * sigma), "better": delta < 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    data, inputs, rows0 = {}, {}, None
    for runs in GROUPS.values():
        for m in runs:
            f = a.root / f"{m}_perjet.npz"
            d = dict(np.load(f))
            if rows0 is None:
                rows0, base = d["rows"], d
            elif not np.array_equal(d["rows"], rows0):
                raise SystemExit(f"FATAL: {f} holds other jets than the first network's")
            data[m] = d
            inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    gen, lab, pt = base["genjet_sdmass"], base["label188"], base["jet_pt"]
    sd_r = base["m_groomed"] / gen
    if not all(np.allclose(data[m]["m_groomed"], base["m_groomed"]) for m in data):
        raise SystemExit("FATAL: the soft-drop mass differs between the networks' files")
    jets = {"resonant": lab < 161, "qcd": lab >= 161, "all": np.ones(lab.size, bool)}
    axes = {"pt": pt, "mgen": gen, "pt_fine": pt}
    out = {"script": "experiments/EVAL/mass_sd_boot.py", "inputs_sha256": inputs, "boot": AR.BOOT_RESAMPLES,
           "seed": AR.BOOT_SEED, "min_jets": AR.MIN_JETS, "groups": {}}
    seed = AR.BOOT_SEED
    for name, runs in GROUPS.items():
        for est in ESTIMATORS:
            key = f"m_{est}"
            if not all(key in data[m] for m in runs):
                continue
            res = {}
            for jg, jm in jets.items():
                bins = {"inclusive": jm}
                for ax, edges in BINNINGS.items():
                    v = axes[ax]
                    bins.update({f"{ax}:{lo}-{hi}": jm & (v >= lo) & (v < hi) for lo, hi in zip(edges[:-1], edges[1:])})
                for b, mask in bins.items():
                    if mask.sum() < AR.MIN_JETS:
                        continue
                    seed += 1        # one resample stream per (group, estimator, jets, bin), shared within it
                    res[f"{jg}|{b}"] = compare(sd_r[mask], [data[m][key][mask] / gen[mask] for m in runs], seed)
            out["groups"].setdefault(name, {})[est] = res
            n_bins = [k for k in res if not k.endswith("|inclusive")]
            print(name, est, "better beyond 3 sigma:", sum(res[k]["pass"] and res[k]["better"] for k in n_bins),
                  "worse beyond 3 sigma:", sum(res[k]["pass"] and not res[k]["better"] for k in n_bins),
                  "of", len(n_bins), "bins", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.rename(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
