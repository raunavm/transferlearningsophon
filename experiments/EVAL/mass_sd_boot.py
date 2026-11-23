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

The same for the bias |median(r) - 1| ("bias"), and, for the claims that compare networks (Fig. A10: a probe on a mass
network against the same probe on its own label set), each pair of DIFFS: delta = mean over A's runs minus mean over
B's, sigma = sqrt(sd_A^2 + sd_B^2 + boot_sd^2). Each bin's resamples are drawn from one seed of that bin alone, so the
soft-drop mass and every network see the same resampled jets: every comparison is paired.

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
DIFFS = (("162 classes + mass", "162 classes"), ("17 classes + mass", "17 classes"),
         # tab:tests (PI 2026-11-23, LEDGER SP-12 step 4): the mass table's other stated differences; written to a new
         # file, every group and the two pairs above as before (the resamples depend on the bin alone)
         ("17 classes", "188 classes"), ("188 classes", "random network"), ("17 classes + mass", "162 classes + mass"))
BINNINGS = {"pt": AR.PT_EDGES, "mgen": AR.MGEN_EDGES, "pt_fine": RB.PT_EDGES}


def stats(r: np.ndarray) -> tuple:
    """(resolution, bias |median - 1|) of a response sample."""
    lo, m, hi = np.quantile(r, (Q[0], 0.5, Q[1]))
    return 0.5 * (hi - lo) / m, abs(m - 1.0)


def resampled(runs: list, seed: int) -> tuple:
    """(point, resamples): per run its (resolution, bias); the mean over runs of each over BOOT_RESAMPLES resamples of
    the bin's jets drawn from seed (the same draws for every caller of the bin)."""
    n = runs[0].size
    point = np.array([stats(r) for r in runs])                      # (runs, 2)
    rng = np.random.default_rng(seed)
    reps = np.empty((AR.BOOT_RESAMPLES, 2))
    for b in range(AR.BOOT_RESAMPLES):
        i = rng.integers(0, n, n)
        reps[b] = np.mean([stats(r[i]) for r in runs], axis=0)
    return point, reps


def verdict(pa: np.ndarray, ra: np.ndarray, pb: np.ndarray, rb: np.ndarray, k: int) -> dict:
    """A minus B in quantity k (0 resolution, 1 bias): delta, the paired bootstrap SD, the run SDs, sigma, pass."""
    delta = float(pa[:, k].mean() - pb[:, k].mean())
    boot_sd = float((ra[:, k] - rb[:, k]).std(ddof=1))
    # A with one run has no run spread: no claim; B with one value is the soft-drop mass, which has none to add
    sds = [float(np.std(pa[:, k], ddof=1)) if len(pa) > 1 else None,
           float(np.std(pb[:, k], ddof=1)) if len(pb) > 1 else 0.0]
    sigma = None if sds[0] is None else float(np.sqrt(sds[0] ** 2 + sds[1] ** 2 + boot_sd ** 2))
    return {"a": pa[:, k].tolist(), "b": pb[:, k].tolist(), "delta": delta, "boot_sd": boot_sd, "sd_a": sds[0],
            "sd_b": sds[1], "sigma": sigma, "z": delta / sigma if sigma else None,
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
    sd_r_all = base["m_groomed"] / gen
    if not all(np.allclose(data[m]["m_groomed"], base["m_groomed"]) for m in data):
        raise SystemExit("FATAL: the soft-drop mass differs between the networks' files")
    jets = {"resonant": lab < 161, "qcd": lab >= 161, "all": np.ones(lab.size, bool)}
    axes = {"pt": pt, "mgen": gen, "pt_fine": pt}
    out = {"script": "experiments/EVAL/mass_sd_boot.py", "inputs_sha256": inputs, "boot": AR.BOOT_RESAMPLES,
           "seed": AR.BOOT_SEED, "min_jets": AR.MIN_JETS, "groups": {}}
    bins = {}
    for jg, jm in jets.items():
        bins[f"{jg}|inclusive"] = jm
        for ax, edges in BINNINGS.items():
            v = axes[ax]
            bins.update({f"{jg}|{ax}:{lo}-{hi}": jm & (v >= lo) & (v < hi) for lo, hi in zip(edges[:-1], edges[1:])})
    bins = {k: m for k, m in bins.items() if m.sum() >= AR.MIN_JETS}
    out["diffs"] = {}
    for i, (b, mask) in enumerate(sorted(bins.items())):
        seed = AR.BOOT_SEED + i          # one stream per bin, shared by the soft-drop mass and every network
        sd_p, sd_r = resampled([sd_r_all[mask]], seed)
        got = {}
        for name, runs in GROUPS.items():
            for est in ESTIMATORS:
                key = f"m_{est}"
                if all(key in data[m] for m in runs):
                    got[(name, est)] = resampled([data[m][key][mask] / gen[mask] for m in runs], seed)
        for (name, est), (p, r) in got.items():
            res = out["groups"].setdefault(name, {}).setdefault(est, {})
            res[b] = {"n": int(mask.sum()), "softdrop": float(sd_p[0, 0]), "softdrop_bias": float(sd_p[0, 1]),
                      **{k: v for k, v in verdict(p, r, sd_p, sd_r, 0).items() if k not in ("b", "sd_b")},
                      "bias": {k: v for k, v in verdict(p, r, sd_p, sd_r, 1).items() if k not in ("b", "sd_b")}}
            res[b]["runs"] = res[b].pop("a")
        for a_name, b_name in DIFFS:
            for est in ESTIMATORS:
                if (a_name, est) in got and (b_name, est) in got:
                    pa, ra = got[(a_name, est)]
                    pb, rb = got[(b_name, est)]
                    out["diffs"].setdefault(f"{a_name} - {b_name}", {}).setdefault(est, {})[b] = {
                        "resolution": verdict(pa, ra, pb, rb, 0), "bias": verdict(pa, ra, pb, rb, 1)}
        print(f"bin {b} done", flush=True)
    for name, ests in out["groups"].items():
        for est, res in ests.items():
            nb = [k for k in res if not k.endswith("|inclusive")]
            print(name, est, "resolution better / worse beyond 3 sigma:", sum(res[k]["pass"] and res[k]["better"] for k in nb),
                  "/", sum(res[k]["pass"] and not res[k]["better"] for k in nb), "; bias better / worse:",
                  sum(res[k]["bias"]["pass"] and res[k]["bias"]["better"] for k in nb), "/",
                  sum(res[k]["bias"]["pass"] and not res[k]["bias"]["better"] for k in nb), "of", len(nb), "bins", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.rename(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
