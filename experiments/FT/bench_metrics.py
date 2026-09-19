#!/usr/bin/env python3
"""The two PUBLISHED BENCHMARKS of the fine-tuning legs, read out in the
community table's format.

WHAT THIS IS AND WHY IT NEEDED WRITING. LEGS_BENCH in scripts/build_ft_jobs.py
fine-tunes every initialisation on the Top Quark Tagging Reference set (top vs
QCD) and on EnergyFlow quark/gluon, and leaves per cell

    <root>/leg_<top|qg>/<init>/N<N>/s<S>/{DONE, ft_manifest.json,
                                          features/logits.npy, features/label188.npy}

with no metric computed by anything in this repo. This is the readout: accuracy,
ROC AUC, and background rejection 1/eps_B at signal efficiency 50% and 30%
(R50, R30) -- the four columns of the Particle Transformer / OmniLearn tables --
plus log(1 - AUC), which D7 makes the inferential quantity.

WHICH CLASS IS THE SIGNAL. `label188.npy` is a misnomer here: extract_features
saves y[label_names[0]], and for a `simple` data config weaver defines that as
the argmax over `labels.value` (weaver utils/data/config.py:115). Both configs
declare `value: [<background>, <signal>]` with `<signal>: label == 1`, so the
integer 1 is top / quark and logit COLUMN 1 is the signal column. logits.npy
carries no names, so that order cannot be checked against the cache; it is
checked against the committed data config at runtime instead (SIGNAL below). A
reversed order would report 1 - AUC with nothing raised, and every value would
still sit in [0.5, 1] and look plausible.

METRICS ARE NOT REDEFINED HERE. `rejection_at` and `log1m_auc` come from
experiments/EVAL/probe.py: linear interpolation on the ROC, the 1/N_bkg cap, the
Poisson band on the surviving background count, the resolution floor on
log(1 - AUC). NOT experiments/EVAL/anchors.py's `rejection`, whose threshold
convention differs. A rejection at the cap is a LOWER BOUND, and with a handful
of surviving background jets it is a count, not a measurement: `is_bound`,
`n_bkg_pass` and `rel_stat` travel with every rejection, into the summary too.

WHAT THE SPREAD IS A SPREAD OVER. Everywhere but one corner a cell's seed names
both the training seed and the training subset (s1..s3). At N_max on top, for
pretrained initialisations only, s1..s9 are nine head re-initialisations on ONE
subset -- the benchmark's own convention, hence the median. The directory naming
is the same, so `train_subsets` in the summary is what tells the two apart.

ROW ALIGNMENT IS VERIFIED, NOT ASSUMED, per dataset, exactly as leg1_metrics.py
does it: one sha256 of label188.npy across every cell, or the run is fatal.

No p-values are computed here. experiments/FT/leg_stats.py owns inference and
reads `cells[<dataset>]`, which has leg 1's cells[init][N][seed] shape.

Run:  python3 experiments/FT/bench_metrics.py --root /data/results/ft \
          --out /data/results/ft/bench_metrics
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]

# dataset -> (data config, [background, signal] in the order it must declare)
SIGNAL = {
    "top": ("configs/finetune/TopReference.yaml", ["label_QCD", "label_Top"]),
    "qg": ("configs/finetune/EnergyFlowQG.yaml", ["label_gluon", "label_quark"]),
}
SIGNAL_COLUMN = 1     # asserted against the committed data config at runtime
WORKING_POINTS = {"r50": 0.5, "r30": 0.3}
METRICS = ["accuracy", "auc", "log1m_auc", "r50", "r30"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def check_label_order(dataset: str) -> None:
    """The committed data config must still say column 1 / integer 1 is signal."""
    rel, want = SIGNAL[dataset]
    text = (REPO / rel).read_text()
    m = re.search(r"^labels:\s*\n\s*type:\s*simple\s*\n\s*value:\s*\[([^\]]*)\]",
                  text, re.M)
    have = [v.strip() for v in m.group(1).split(",")] if m else None
    if have != want or f"{want[SIGNAL_COLUMN]}: label == 1" not in text:
        raise SystemExit(f"FATAL: {rel} declares labels {have}, expected {want} "
                         f"with `{want[SIGNAL_COLUMN]}: label == 1`. The signal "
                         "column is read by position, so a reordering there "
                         "inverts every number here without an error.")


def discover(root: pathlib.Path, dataset: str):
    """([(init, N, seed, cell_dir)] for every DONE cell, [cell_dir without DONE]).

    `.partial.<timestamp>` directories are skipped by name, as in leg 1. A cell
    without DONE is still training or died; it is REPORTED, not dropped quietly,
    so a mean over two seeds is never mistaken for a mean over three.
    """
    done, skipped = [], []
    leg = root / f"leg_{dataset}"
    for init_dir in sorted(p for p in leg.iterdir() if p.is_dir()):
        for n_dir in sorted(p for p in init_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in n_dir.iterdir() if p.is_dir()):
                if ".partial." in seed_dir.name:
                    continue
                if (seed_dir / "DONE").exists():
                    done.append((init_dir.name, n_dir.name, seed_dir.name, seed_dir))
                else:
                    skipped.append(seed_dir)
    return done, skipped


def cell_metrics(cell: pathlib.Path, probe) -> dict:
    fd = cell / "features"
    for f in ("logits.npy", "label188.npy"):
        if not (fd / f).exists():
            raise SystemExit(f"FATAL: {cell} is marked DONE but has no features/{f}")
    lab = np.load(fd / "label188.npy")
    sha = hashlib.sha256(lab.tobytes()).hexdigest()
    logits = np.load(fd / "logits.npy")
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise SystemExit(f"FATAL: {fd} logits are {logits.shape}, not (n, 2). "
                         "Both benchmarks are binary; a different width means "
                         "this cell is not what it claims to be.")
    if logits.shape[0] != lab.shape[0]:
        raise SystemExit(f"FATAL: {fd} has {logits.shape[0]} logits rows and "
                         f"{lab.shape[0]} labels")
    if not np.isin(lab, (0, 1)).all():
        raise SystemExit(f"FATAL: {fd} holds labels outside {{0, 1}}: "
                         f"{np.unique(lab)[:5].tolist()}")
    y = lab.astype(int)
    n_sig = int((y == SIGNAL_COLUMN).sum())
    n_bkg = int(y.size - n_sig)
    if n_sig == 0 or n_bkg == 0:
        raise SystemExit(f"FATAL: {fd} has one class only ({n_sig} signal, "
                         f"{n_bkg} background)")

    score = softmax(logits)[:, SIGNAL_COLUMN]
    l1m, censored, auc = probe.log1m_auc(y, score)
    out = {
        "accuracy": float((logits.argmax(1) == y).mean()),
        "auc": auc,
        "log1m_auc": l1m, "log1m_auc_censored": censored,
    }
    for k, eps_s in WORKING_POINTS.items():
        rej, eps_b, bound, n_pass, rel = probe.rejection_at(y, score, eps_s)
        out.update({k: rej, f"{k}_eps_b": eps_b, f"{k}_is_bound": bound,
                    f"{k}_n_bkg_pass": n_pass,
                    # inf (no background jet passed) is not JSON; null is
                    f"{k}_rel_stat": rel if math.isfinite(rel) else None})
    manifest = cell / "ft_manifest.json"
    out.update({
        "n_jets": int(y.size), "n_signal": n_sig, "n_background": n_bkg,
        # p saturates to exactly 0 or 1 past |z1 - z0| ~ 37 and those jets tie
        "n_score_saturated": int(((score == 0.0) | (score == 1.0)).sum()),
        "train_subset": (json.loads(manifest.read_text()).get("subset")
                         if manifest.exists() else None),
        "label188_sha256": sha,
        "cell": str(cell),
    })
    return out


def _agg(v: list[float]) -> dict:
    a = np.array(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def summarise(per_seed: dict) -> dict:
    c = list(per_seed.values())
    s = {k: _agg([v[k] for v in c]) for k in METRICS}
    for k in WORKING_POINTS:
        # A mean that includes a capped cell is itself only a lower bound.
        s[k]["n_bound"] = sum(v[f"{k}_is_bound"] for v in c)
        s[k]["n_bkg_pass_min"] = min(v[f"{k}_n_bkg_pass"] for v in c)
    s["log1m_auc"]["n_censored"] = sum(v["log1m_auc_censored"] for v in c)
    s["seeds"] = sorted(per_seed)
    s["train_subsets"] = sorted({str(v["train_subset"]) for v in c})
    return s


def _rej(m: dict, k: str) -> str:
    """A bound is printed AS a bound; the bare number would read as measured."""
    if m[f"{k}_is_bound"]:
        return f"{k.upper()}>={m[k]:.0f} [BOUND, {m[f'{k}_n_bkg_pass']} bkg pass]"
    return f"{k.upper()}={m[k]:.1f} ({m[f'{k}_n_bkg_pass']} bkg pass)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path,
                    help="the directory that holds leg_top/ and leg_qg/")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--datasets", nargs="+", default=list(SIGNAL), choices=list(SIGNAL))
    args = ap.parse_args(argv)

    probe = _load("probe", "experiments/EVAL/probe.py")
    res, summary, alignment, skipped = {}, {}, {}, []
    for d in args.datasets:
        if not (args.root / f"leg_{d}").is_dir():
            print(f"{d}: no {args.root / f'leg_{d}'}, nothing read", flush=True)
            continue
        check_label_order(d)
        cells, no_done = discover(args.root, d)
        skipped += [{"cell": str(p), "reason": "no DONE"} for p in no_done]
        for p in no_done:
            print(f"  SKIPPED (no DONE): {p}", flush=True)
        print(f"{d}: {len(cells)} cells", flush=True)

        shas = {}
        for init, n, seed, cell in cells:
            m = cell_metrics(cell, probe)
            shas.setdefault(m["label188_sha256"], []).append(f"{init}/{n}/{seed}")
            res.setdefault(d, {}).setdefault(init, {}).setdefault(n, {})[seed] = m
            print(f"  {init:16} {n:10} {seed:4} acc={m['accuracy']:.5f} "
                  f"AUC={m['auc']:.5f} {_rej(m, 'r50')} {_rej(m, 'r30')}", flush=True)
        if len(shas) > 1:
            for s, where in shas.items():
                print(f"  {s[:16]}  {len(where)} cells, e.g. {where[:3]}")
            raise SystemExit(f"FATAL: {d} cells did not read out the same test "
                             "jets in the same order; the between-init contrast "
                             "is not paired and the table would be meaningless")
        if shas:
            alignment[d] = next(iter(shas))
        for init, per_n in res.get(d, {}).items():
            for n, per_seed in per_n.items():
                summary.setdefault(d, {}).setdefault(init, {})[n] = summarise(per_seed)

    if not res:
        raise SystemExit(f"FATAL: no DONE cells under {args.root}/leg_<dataset>")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                capture_output=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench_metrics.json").write_text(json.dumps(
        {"script": "experiments/FT/bench_metrics.py", "repo_commit": commit,
         # HEAD does not pin a script run from a dirty tree; its own hash does
         "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         "root": str(args.root),
         "signal": {d: {"data_config": SIGNAL[d][0], "background": SIGNAL[d][1][0],
                        "signal": SIGNAL[d][1][1], "signal_label": 1,
                        "signal_logit_column": SIGNAL_COLUMN} for d in res},
         "row_alignment_sha256": alignment,
         "skipped": skipped,
         "cells": res, "summary": summary}, indent=1, allow_nan=False))
    print(f"\nwrote {args.out / 'bench_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
