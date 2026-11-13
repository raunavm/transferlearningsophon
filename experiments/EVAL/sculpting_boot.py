#!/usr/bin/env python3
"""A paired bootstrap over the test jets of the pretraining heads' mass sculpting (plan 2026-11-08, step 2; PI:
a sculpting difference stands when it exceeds 3 sigma, sigma^2 = the run-to-run spread^2 + a bootstrap over the
test jets^2).

Inputs: each run's head outputs (<head-root>/<model>/test/best70/head_outputs.npz, the first 2,000,000 TEST2M
jets, the same jets for every run: labels and jet_sdmass must agree, or FATAL). Score and QCD nodes exactly as
mass_sculpting.py (qcd_nodes of the manifest's rung, or of --label-set for a flavour-boundary run). Each run's
score is computed once.

Replica 0 is the test set itself; it must reproduce the stored sculpting documents behind F10b
(<sculpting-dir>/<model>_sculpting.json, build_head_jobs.SCULPT_V3_DIR: JSD at QCD efficiency 0.01 and 0.1 and at
sig_eff_50, FATAL beyond NOMINAL_TOL, and the same head_outputs sha256). Each of the --replicas resamples the
QCD jets and, independently, the resonant jets with replacement ONCE, and the same indices serve every run
(paired). Per replica and run, at QCD efficiency 1% and
10% the cut keeps the k-th largest QCD score and above (k = round(eff n_QCD)), at signal efficiency 50% the
k-th largest resonant score (k = round(n_res / 2)); the QCD jet_sdmass histograms before and after give the
divergence JSD (base 2) and the distance sqrt(JSD).

DIFFERENCES (DIFFS): configuration minus reference, paired by run index (seed) where both have the run. For
each working point and quantity (distance, jsd), d_i the nominal paired difference of run index i:
  delta     mean_i d_i
  run_sd    SD of d_i over the indices (ddof 1), the run-to-run spread;  run_se = run_sd / sqrt(n_pairs)
  boot_sd   SD over the replicas of that replica's mean_i d_i (ddof 1), the jet term
  sigma     sqrt(run_sd^2 + boot_sd^2) (PI 2026-11-08); the test uses it, pass = |delta| > 3 sigma
  sigma_run_se, z_run_se, pass_run_se   the same with run_se in place of run_sd (the standard error of the
            mean over the run indices), recorded beside it
z = delta / sigma (null where sigma = 0).
Beside them, "l3" (decision L3, 2026-11-13): the sides unpaired over all their runs in --models, delta the difference
of the two means, sigma^2 = SD^2 of the configuration's runs + SD^2 of the reference's + boot_sd^2, boot_sd the SD
over the replicas of that replica's difference of the two means; pass = |delta| > 3 sigma (null sigma, no pass, for a
side with one run).

    python3 experiments/EVAL/sculpting_boot.py --head-root /data/results/eval/v2_head --models mtx-l188-s1 ... \\
        --label-set mtx-flavf0-s1=FLAV_F0 ... --replicas 200 --sculpting-dir /data/results/eval/v2_head/sculpting_v3 \\
        --out /data/results/eval/v2_head/sculpting_boot.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MIN_REPLICAS = 200
NOMINAL_TOL = 1e-12
WPS = (("eps_b=0.01", "eps_b", 0.01), ("eps_b=0.1", "eps_b", 0.1), ("eps_s=0.5", "eps_s", 0.5))
# configuration minus reference: model name parts mtx-<arm>-s<k>
DIFFS = {"17-188": ("r16q1", "l188"), "43-188": ("r42q1", "l188"), "162-188": ("l162", "l188"),
         "17mass-17": ("r16q1mass", "r16q1"),
         "162mass-162": ("l162mass", "l162"), "flavf0-188": ("flavf0", "l188"), "flavf1-188": ("flavf1", "l188"),
         "flavf1r-188": ("flavf1r", "l188")}
QTY = {"distance": np.sqrt, "jsd": lambda x: x}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("mass_sculpting", "experiments/EVAL/mass_sculpting.py")
NB = len(MS.BINS) - 1


def bin_index(msd: np.ndarray) -> np.ndarray:
    """np.histogram(msd, MS.BINS)'s bin of each jet (last bin closed), NB for a jet outside the edges."""
    i = np.searchsorted(MS.BINS, msd, side="right") - 1
    i[msd == MS.BINS[-1]] = NB - 1
    i[(msd < MS.BINS[0]) | (msd > MS.BINS[-1]) | np.isnan(msd)] = NB
    return i


def _hist(b: np.ndarray) -> np.ndarray:
    h = np.bincount(b, minlength=NB + 1)[:NB].astype(np.float64)
    return h / h.sum()


def point_values(sq: np.ndarray, sr: np.ndarray, bq: np.ndarray) -> dict:
    """{wp: (jsd, n_pass)} of one run on one (resampled) set: QCD scores sq with mass bins bq, resonant scores sr."""
    before = _hist(bq)
    nq, nr = sq.size, sr.size
    ks = {name: max(int(round(e * nq)), 1) for name, kind, e in WPS if kind == "eps_b"}
    part = np.partition(sq, sorted({nq - k for k in ks.values()}))
    cuts = {name: part[nq - k] for name, k in ks.items()}
    kr = max(int(round(MS.SIG_EFF * nr)), 1)
    cuts["eps_s=0.5"] = np.partition(sr, nr - kr)[nr - kr]
    out = {}
    for name, cut in cuts.items():
        sel = sq >= cut
        out[name] = (MS.jsd(before, _hist(bq[sel])), int(sel.sum()))
    return out


def load_run(head: pathlib.Path, label_set: str | None) -> dict:
    """mass_sculpting.main's inputs and checks: the run's score of every jet, its labels and jet_sdmass."""
    man = json.loads((head / "manifest.json").read_text())
    rung = man["rung"]
    if rung == "none" and not label_set:
        raise SystemExit(f"FATAL: {head}: a label set outside the contraction tree needs --label-set")
    if rung != "none" and label_set not in (None, rung):
        raise SystemExit(f"FATAL: --label-set {label_set} for a {rung} run")
    ls = label_set or rung
    nodes = MS.qcd_nodes(ls)
    if not nodes:
        raise SystemExit(f"FATAL: {ls} has no output node that is QCD alone")
    z = np.load(head / MS.HEAD_FILE)
    logits = z["logits"]
    if logits.shape[1] != man["num_classes"] or max(nodes) >= logits.shape[1]:
        raise SystemExit(f"FATAL: {logits.shape[1]} logits for {ls}'s {man['num_classes']} classes")
    return {"label_set": ls, "qcd_nodes": nodes, "s": MS.score(logits, nodes),
            "label188": z["label188"].astype(np.int64), "msd": z["jet_sdmass"].astype(np.float64),
            "head_outputs_sha256": (man.get("head_outputs") or {}).get("sha256")}


def pairs_of(models, a: str, b: str) -> list:
    """[(mtx-a-s<k>, mtx-b-s<k>)] over the run indices k both have."""
    seeds = lambda arm: {int(m.rsplit("-s", 1)[1]) for m in models if m.rsplit("-s", 1)[0] == f"mtx-{arm}"}  # noqa: E731
    return [(f"mtx-{a}-s{k}", f"mtx-{b}-s{k}") for k in sorted(seeds(a) & seeds(b))]


def runs_of(models, arm: str) -> list:
    """The runs mtx-<arm>-s<k> among models."""
    return [m for m in models if m.rsplit("-s", 1)[0] == f"mtx-{arm}"]


def unpaired(nominal: dict, boot: dict, a_runs: list, b_runs: list, wp: str, f) -> dict:
    """Decision L3 for one difference, working point and quantity: the sides unpaired over all their runs."""
    va, vb = (np.array([f(nominal[m][wp]) for m in r]) for r in (a_runs, b_runs))
    rep = np.mean([f(boot[m][wp]) for m in a_runs], axis=0) - np.mean([f(boot[m][wp]) for m in b_runs], axis=0)
    sd_a, sd_b = (float(v.std(ddof=1)) if v.size > 1 else None for v in (va, vb))
    delta, boot_sd = float(va.mean() - vb.mean()), float(rep.std(ddof=1))
    sigma = math.sqrt(sd_a ** 2 + sd_b ** 2 + boot_sd ** 2) if sd_a is not None and sd_b is not None else None
    return {"runs_config": list(a_runs), "runs_reference": list(b_runs), "delta": delta, "sd_config": sd_a,
            "sd_reference": sd_b, "boot_sd": boot_sd, "sigma": sigma, "z": delta / sigma if sigma else None,
            "pass": bool(sigma is not None and abs(delta) > 3 * sigma)}


def difference(nominal: dict, boot: dict, pairs: list, wp: str, f) -> dict:
    """The statistics of the docstring for one difference, working point and quantity; nominal[m][wp] a number,
    boot[m][wp] the replicas' array."""
    d = np.array([f(nominal[a][wp]) - f(nominal[b][wp]) for a, b in pairs])
    rep = np.mean([f(boot[a][wp]) - f(boot[b][wp]) for a, b in pairs], axis=0)
    n = len(pairs)
    delta = float(d.mean())
    run_sd = float(d.std(ddof=1)) if n > 1 else None
    run_se = run_sd / math.sqrt(n) if run_sd is not None else None
    boot_sd = float(rep.std(ddof=1))
    out = {"n_pairs": n, "per_pair": d.tolist(), "delta": delta, "run_sd": run_sd, "run_se": run_se,
           "boot_sd": boot_sd, "boot_mean": float(rep.mean())}
    for key, run in (("", run_sd), ("_run_se", run_se)):
        sigma = math.hypot(run, boot_sd) if run is not None else None
        out[f"sigma{key}"] = sigma
        out[f"z{key}"] = delta / sigma if sigma else None
        out[f"pass{key}"] = bool(sigma is not None and abs(delta) > 3 * sigma)
    return out


def stored_jsd(path: pathlib.Path, head_outputs_sha256) -> dict:
    """{wp: JSD} of a stored mass_sculpting.py document, which must be of the same head outputs."""
    doc = json.loads(path.read_text())
    if doc.get("head_outputs_sha256") != head_outputs_sha256:
        raise SystemExit(f"FATAL: {path} is of other head outputs ({doc.get('head_outputs_sha256')})")
    return {"eps_b=0.01": doc["working_points"]["0.01"]["jsd"], "eps_b=0.1": doc["working_points"]["0.1"]["jsd"],
            "eps_s=0.5": doc["sig_eff_50"]["jsd"]}


def bootstrap(runs: dict, msd: np.ndarray, label188: np.ndarray, replicas: int, seed: int,
              stored: dict) -> tuple[dict, dict, dict]:
    """(nominal {m: {wp: jsd}}, n_pass {m: {wp: n}}, boot {m: {wp: array of jsd over the replicas}}); stored
    {m: {wp: JSD}} the documents replica 0 must reproduce."""
    qcd = (label188 >= MS.QCD_LABELS[0]) & (label188 < MS.QCD_LABELS[1])
    res = (label188 >= 0) & (label188 < MS.QCD_LABELS[0])
    bq = bin_index(msd[qcd])
    sq = {m: r["s"][qcd] for m, r in runs.items()}
    sr = {m: r["s"][res] for m, r in runs.items()}
    nominal, n_pass = {}, {}
    for m in runs:
        v = point_values(sq[m], sr[m], bq)
        for wp, x in stored[m].items():
            if abs(v[wp][0] - x) > NOMINAL_TOL:
                raise SystemExit(f"FATAL: {m} {wp}: replica 0 gives JSD {v[wp][0]!r}, the stored document {x!r}")
        nominal[m] = {wp: v[wp][0] for wp in v}
        n_pass[m] = {wp: v[wp][1] for wp in v}
    rng = np.random.default_rng(seed)
    boot = {m: {wp: np.empty(replicas) for wp, *_ in WPS} for m in runs}
    nq, nr = int(qcd.sum()), int(res.sum())
    t0 = time.time()
    for r in range(replicas):
        iq, ir = rng.integers(0, nq, nq), rng.integers(0, nr, nr)
        b = bq[iq]
        for m in runs:
            for wp, (x, _) in point_values(sq[m][iq], sr[m][ir], b).items():
                boot[m][wp][r] = x
        if (r + 1) % 20 == 0:
            print(f"  replica {r + 1}/{replicas}, {time.time() - t0:.0f} s", flush=True)
    return nominal, n_pass, boot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head-root", required=True, type=pathlib.Path)
    ap.add_argument("--models", required=True, nargs="+")
    ap.add_argument("--label-set", nargs="*", default=[], help="<model>=<FLAV_ column> for a run at rung 'none'")
    ap.add_argument("--sculpting-dir", required=True, type=pathlib.Path,
                    help="the stored documents replica 0 must reproduce, <dir>/<model>_sculpting.json")
    ap.add_argument("--replicas", type=int, default=MIN_REPLICAS)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"{a.out} exists; not overwritten")
        return 0
    if a.replicas < MIN_REPLICAS:
        raise SystemExit(f"FATAL: {a.replicas} replicas; at least {MIN_REPLICAS}")
    sets = dict(x.split("=", 1) for x in a.label_set)
    if set(sets) - set(a.models):
        raise SystemExit(f"FATAL: --label-set for models not in --models: {sorted(set(sets) - set(a.models))}")
    runs, ref = {}, None
    for m in a.models:
        runs[m] = load_run(a.head_root / m / "test" / "best70", sets.get(m))
        if ref is None:
            ref = runs[m]
        elif not (np.array_equal(runs[m]["label188"], ref["label188"]) and np.array_equal(runs[m]["msd"], ref["msd"])):
            raise SystemExit(f"FATAL: {m} is not scored on the jets of {a.models[0]}")
        else:
            del runs[m]["label188"], runs[m]["msd"]          # the first run's serve every run
        print(f"  {m}: {runs[m]['label_set']}, {len(runs[m]['qcd_nodes'])} QCD nodes", flush=True)
    stored = {m: stored_jsd(a.sculpting_dir / f"{m}_sculpting.json", r["head_outputs_sha256"]) for m, r in runs.items()}
    nominal, n_pass, boot = bootstrap(runs, ref["msd"], ref["label188"], a.replicas, a.seed, stored)
    diffs = {}
    for name, (x, y) in DIFFS.items():
        pairs = pairs_of(a.models, x, y)
        if not pairs:
            raise SystemExit(f"FATAL: no run index has both mtx-{x} and mtx-{y}")
        rx, ry = runs_of(a.models, x), runs_of(a.models, y)
        diffs[name] = {"configuration": x, "reference": y, "pairs": [list(p) for p in pairs],
                       **{wp: {q: difference(nominal, boot, pairs, wp, f) for q, f in QTY.items()} for wp, *_ in WPS},
                       "l3": {wp: {q: unpaired(nominal, boot, rx, ry, wp, f) for q, f in QTY.items()} for wp, *_ in WPS}}
    lab = ref["label188"]
    doc = {"head_root": str(a.head_root), "sculpting_dir": str(a.sculpting_dir), "replicas": a.replicas, "seed": a.seed,
           "n_jets": int(lab.size), "n_qcd": int(((lab >= MS.QCD_LABELS[0]) & (lab < MS.QCD_LABELS[1])).sum()),
           "n_res": int(((lab >= 0) & (lab < MS.QCD_LABELS[0])).sum()),
           "label188_sha256": hashlib.sha256(lab.tobytes()).hexdigest(),
           "working_points": {wp: {kind: e} for wp, kind, e in WPS},
           "resampling": "QCD and resonant jets each resampled with replacement once per replica, the same indices "
                         "for every run",
           "test": "pass = |delta| > 3 sigma, sigma^2 = run_sd^2 + boot_sd^2, run_sd = run-to-run SD of the paired "
                   "differences, boot_sd = bootstrap SD of the mean paired difference; pass_run_se uses "
                   "run_se = run_sd / sqrt(n_pairs) in place of run_sd",
           "test_l3": "l3: the sides unpaired over all their runs, pass = |delta| > 3 sigma, sigma^2 = sd_config^2 + "
                      "sd_reference^2 + boot_sd^2 (decision L3, 2026-11-13)",
           "runs": {m: {"label_set": r["label_set"], "qcd_nodes": r["qcd_nodes"],
                        "head_outputs_sha256": r["head_outputs_sha256"],
                        **{wp: {"jsd": nominal[m][wp], "distance": math.sqrt(nominal[m][wp]), "n_pass": n_pass[m][wp],
                                "boot_sd_jsd": float(boot[m][wp].std(ddof=1)),
                                "boot_sd_distance": float(np.sqrt(boot[m][wp]).std(ddof=1))} for wp, *_ in WPS}}
                    for m, r in runs.items()},
           "differences": diffs,
           "script": "experiments/EVAL/sculpting_boot.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, a.out)
    for name, d in diffs.items():
        print(f"{name}: " + "; ".join(f"{wp} {d[wp]['distance']['delta']:+.4f} +- {d[wp]['distance']['sigma'] or 0:.4f} "
                                       f"({'pass' if d[wp]['distance']['pass'] else 'fail'})" for wp, *_ in WPS))
        u = d["l3"]
        print("  l3: " + "; ".join(f"{wp} {u[wp]['distance']['delta']:+.4f} +- {u[wp]['distance']['sigma'] or 0:.4f} "
                                    f"({'pass' if u[wp]['distance']['pass'] else 'fail'})" for wp, *_ in WPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
