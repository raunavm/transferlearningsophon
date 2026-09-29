#!/usr/bin/env python3
"""Paired errors on every ratio of two models' metrics the paper quotes.

Audit 2026-09-29 B3 and must-fix 5, 8 and 11(a). The method is in
src/stats/paired.py; this file only feeds it. Three steps:

  replicates   per model, the metric on the test sample and on B bootstrap
               resamplings of it (one resampling per b, shared by every model
               scored on the same jets), from the per-jet outputs:
                 probes        probe.py --save-scores       scores.npz
                 mass probes   mass_resolution.py --save-residuals  residuals.npz
                 fine-tuning   leg 1 logits.npy, leg 2 pred.root (fine-tuning seed s1)
  ratios       every comparison the paper makes, through paired_ratio(), plus a
               Garwood interval on every rejection (per run, and pooled over runs
               with the caveat that pooling runs scored on the same jets
               understates the error).

Metrics are all "lower is better" -- 1 - AUC, eps_B at a fixed signal
efficiency, sigma_eff, 1 - macro AUC -- and every ratio is coarse over fine, so
a ratio above one means the coarser (or control) model is worse. A ratio of
eps_B is the inverse ratio of rejections.

A replicate vector is comparable with another only if both were computed on the
same jets in the same order with the same B and seed; the key `jets` (sha256 of
the row indices and labels) is checked before any pair is formed.

Usage:
  paired_errors.py probe-replicates --probe-dirs DIR... --out R.npz
  paired_errors.py mass-replicates  --mass-dirs DIR...  --out R.npz
  paired_errors.py ft-replicates    --leg1-root D --leg2-root D --out R.npz [--procs 8]
  paired_errors.py ratios --replicates R.npz... --out ratios.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.stats import paired as P  # noqa: E402

B = 1000
SEED = P.SEED_DEFAULT

LEVEL = {"l188": "188", "l162": "162", "r42q1": "43", "r16q1": "17",
         "l162mass": "162+mass", "r16q1mass": "17+mass", "rand": "random"}
MODEL_RE = re.compile(r"^(l188|l162mass|l162|r42q1|r16q1mass|r16q1|rand-d([1-9]))-s([1-9])b?$")

# The comparisons, (fine, coarse): the ladder, the mass output, the random
# control against every vocabulary (draw k paired with run k, whose
# initialisation it shares), and nothing else.
LADDER = ("188", "162", "43", "17")
PAIRS = ([(a, b) for i, a in enumerate(LADDER) for b in LADDER[i + 1:]]
         + [("162", "162+mass"), ("17", "17+mass")]
         + [(lv, "random") for lv in LADDER])


def parse_model(name: str) -> tuple[str, int]:
    """(level, run index) from a model name; l162-s1b and rand-d1-s1b are run 1."""
    m = MODEL_RE.match(name)
    if not m:
        raise SystemExit(f"FATAL: cannot read a level and run from model name {name!r}")
    stem = "rand" if m.group(1).startswith("rand") else m.group(1)
    run = int(m.group(2) or m.group(3))
    return LEVEL[stem], run


def jets_key(rows: np.ndarray, labels: np.ndarray) -> str:
    h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def _sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------- probes
def reproduction(dirs: list[pathlib.Path], committed: dict) -> dict:
    """The refit against the committed result it repeats: largest |dAUC| per
    probe, over every task and model. The linear probe is deterministic and must
    agree exactly; the MLP is refitted with the same seeds and schedule."""
    out = {}
    for d in dirs:
        ref = committed.get(str(d))
        if ref is None:
            continue
        A = json.loads((d / "probe_results.json").read_text())["tasks"]
        R = json.loads(pathlib.Path(ref).read_text())["tasks"]
        worst = {"linear": 0.0, "mlp": 0.0}
        for task, T in A.items():
            for arm, v in T.get("arms", {}).items():
                for kind in worst:
                    worst[kind] = max(worst[kind], abs(v[kind]["auc"] - R[task]["arms"][arm][kind]["auc"]))
        out[str(d)] = {"committed": str(ref), "committed_sha256": _sha(ref),
                       "max_abs_dauc": worst}
    return out


def probe_replicates(dirs: list[pathlib.Path], b: int = B, seed: int = SEED):
    """Replicates of 1 - AUC and eps_B at every working point, per task x probe x model.

    A model scored in two jobs (the 162-class run 1 is in the ladder and in the
    2x2) is kept once, from the first directory given, and the other copy's
    AUC is recorded as a reproducibility check."""
    vec, meta, dup = {}, {}, []
    inputs = []
    for d in dirs:
        J = json.loads((d / "probe_results.json").read_text())
        z = np.load(d / "scores.npz")
        inputs.append({"dir": str(d), "probe_results_sha256": _sha(d / "probe_results.json"),
                       "scores_sha256": _sha(d / "scores.npz")})
        if J.get("scores_npz_sha256") != _sha(d / "scores.npz"):
            raise SystemExit(f"FATAL: {d}/scores.npz is not the file its JSON records")
        for task, T in J["tasks"].items():
            if T.get("skipped"):
                continue
            y, rows = z[f"{task}|y"].astype(bool), z[f"{task}|rows"]
            jk = jets_key(rows, y)
            for arm, A in T["arms"].items():
                for kind in ("linear", "mlp"):
                    s = z[f"{task}|{kind}|{arm}"]
                    base = f"probe|{task}|{kind}|{arm}"
                    if f"{base}|1-auc" in vec:
                        dup.append({"model": base, "dir": str(d),
                                    "auc_here": A[kind]["auc"],
                                    "auc_kept": 1 - vec[f"{base}|1-auc"][0]})
                        continue
                    sc = P.AucScorer(y, s)
                    if abs(sc.auc() - A[kind]["auc"]) > 1e-12:
                        raise SystemExit(f"FATAL: {base}: AUC from scores {sc.auc()} "
                                         f"!= reported {A[kind]['auc']}")
                    vec[f"{base}|1-auc"] = P.replicates(sc, y.size, b, seed)
                    meta[f"{base}|1-auc"] = {"jets": jk, "n": int(y.size),
                                             "censored": bool(A[kind]["log1m_auc_censored"])}
                    for e in T["eps_s"]:
                        es = P.EpsBScorer(y, s, float(e))
                        key = f"{base}|eps_b@{float(e):.2f}"
                        vec[key] = P.replicates(es, y.size, b, seed)
                        nb = int((~y).sum())
                        meta[key] = {"jets": jk, "n": int(y.size), "n_bkg": nb,
                                     "k_pass": float(vec[key][0] * nb)}
                print(f"  {base}", flush=True)
    return vec, meta, {"inputs": inputs, "duplicates": dup}


# ------------------------------------------------------------ mass probes
def mass_replicates(dirs: list[pathlib.Path], b: int = B, seed: int = SEED):
    """Replicates of sigma_eff of the within-class mass residual, per probe x model."""
    vec, meta, inputs = {}, {}, []
    for d in dirs:
        J = json.loads((d / "mass_resolution.json").read_text())
        z = np.load(d / "residuals.npz")
        inputs.append({"dir": str(d), "mass_resolution_sha256": _sha(d / "mass_resolution.json"),
                       "residuals_sha256": _sha(d / "residuals.npz")})
        rows, lab = z["rows"], z["label188"]
        jk = jets_key(rows, lab)
        for arm, A in J["arms"].items():
            for kind in ("ridge", "mlp"):
                res = z[f"{kind}|{arm}"].astype(np.float64)
                sc = P.SigmaEffScorer(res)
                if abs(sc() - A[kind]["sigma_eff"]) > 1e-6:
                    raise SystemExit(f"FATAL: {arm}/{kind}: sigma_eff from residuals "
                                     f"{sc()} != reported {A[kind]['sigma_eff']}")
                key = f"mass|resolution|{kind}|{arm}|sigma_eff"
                vec[key] = P.replicates(sc, res.size, b, seed)
                meta[key] = {"jets": jk, "n": int(res.size)}
                print(f"  {key}", flush=True)
    return vec, meta, {"inputs": inputs}


# ------------------------------------------------------------ fine-tuning
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ft_cell(job):
    """One fine-tuning cell's replicate vector of 1 - macro AUC, on the rows the
    committed readout used (leg1_metrics / leg2_metrics, stride 4)."""
    leg, init, n, path, stride, b, seed = job
    if leg == "leg1":
        lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
        l1 = _load("leg1_metrics", "experiments/FT/leg1_metrics.py")
        l162 = lr.rung_maps()["L162"]
        lab188 = np.load(path / "label188.npy")
        lut = np.full(188, -1, dtype=np.int64)
        for k, v in l162.items():
            lut[k] = v
        truth = lut[lab188]
        idx = np.arange(0, truth.size, stride)
        probs = l1.softmax(np.load(path / "logits.npy")[idx])
        y = truth[idx]
    else:
        l2 = _load("leg2_metrics", "experiments/FT/leg2_metrics.py")
        names, onehot, scores = l2.read_pred_root(path / "pred.root")
        truth = onehot.argmax(1)
        probs = scores.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        idx = np.arange(0, truth.size, stride)
        probs, y = probs[idx], truth[idx]
    sc = P.MacroAucScorer(y, probs)
    v = P.replicates(sc, y.size, b, seed)
    return (f"ft|{leg}|{n}|{init}", v,
            {"jets": jets_key(idx, y), "n": int(y.size), "source": str(path),
             "macro_auc": float(1 - v[0])})


def ft_replicates(leg1_root, leg2_root, cells_json: dict, b: int = B, seed: int = SEED,
                  procs: int = 8, stride: int = 4):
    """Every cell of the committed metrics files at fine-tuning seed s1."""
    jobs = []
    for leg, root in (("leg1", leg1_root), ("leg2", leg2_root)):
        for init, per_n in cells_json[leg]["cells"].items():
            if not MODEL_RE.match(init):
                continue                      # scratch, sophon-public, mpm: not paired rows
            for n, per_seed in per_n.items():
                if "s1" not in per_seed:
                    continue
                cell = pathlib.Path(root) / init / n / "s1"
                jobs.append((leg, init, n, cell / "features_v2" if leg == "leg1" else cell,
                             stride, b, seed))
    vec, meta = {}, {}
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for key, v, m in ex.map(_ft_cell, jobs):
            leg, n, init = key.split("|")[1], key.split("|")[2], key.split("|")[3]
            ref = cells_json[leg]["cells"][init][n]["s1"]["macro_auc_ovr"]
            if abs(m["macro_auc"] - ref) > 1e-9:
                raise SystemExit(f"FATAL: {key}: macro AUC {m['macro_auc']} does not "
                                 f"reproduce the committed {ref}")
            vec[f"{key}|1-macro_auc"], meta[f"{key}|1-macro_auc"] = v, m
            print(f"  {key}  1-macroAUC {v[0]:.6g}  test SD {v[1:].std(ddof=1):.3g}", flush=True)
    return vec, meta, {"cells_sha256": {k: v["sha256"] for k, v in cells_json.items()}}


# ------------------------------------------------------------------ ratios
def _group(meta: dict):
    """{(family, task, kind, metric): {model: key}}."""
    out = {}
    for key in meta:
        fam, task, kind, model, metric = key.split("|")
        out.setdefault((fam, task, kind, metric), {})[model] = key
    return out


def ratios(vec: dict, meta: dict, run_dirs_root: pathlib.Path | None = None) -> dict:
    """Every comparison of PAIRS. With `run_dirs_root` (v2), each model's run
    directory is <root>/mtx-<model> and paired_ratio refuses a pair whose
    realised training streams differ; without it (v1) runs pair by index."""
    rows, rej, models = [], [], {}
    for (fam, task, kind, metric), by_model in sorted(_group(meta).items()):
        per_level = {}
        for model, key in by_model.items():
            lv, run = parse_model(model)
            per_level.setdefault(lv, {})[run] = key
            models.setdefault(f"{fam}|{task}|{kind}|{metric}", {})[model] = float(vec[key][0])
        for fine, coarse in PAIRS:
            if fine not in per_level or coarse not in per_level:
                continue
            pairs = {r: r for r in per_level[coarse] if r in per_level[fine]}
            if not pairs:
                continue
            fk = {r: per_level[fine][r] for r in pairs}
            ck = {r: per_level[coarse][r] for r in pairs}
            jets = {meta[k]["jets"] for k in list(fk.values()) + list(ck.values())}
            if len(jets) != 1:
                raise SystemExit(f"FATAL: {fam}/{task}/{kind}/{metric} {fine} vs {coarse}: "
                                 "the models were not scored on the same jets")
            row = {"family": fam, "task": task, "kind": kind, "metric": metric,
                   "fine": fine, "coarse": coarse,
                   "fine_models": [by_key(by_model, fk[r]) for r in pairs],
                   "coarse_models": [by_key(by_model, ck[r]) for r in pairs]}
            fm = {r: by_key(by_model, fk[r]) for r in pairs}
            cm = {r: by_key(by_model, ck[r]) for r in pairs}
            fv = {fm[r]: vec[fk[r]] for r in pairs}
            cv = {cm[r]: vec[ck[r]] for r in pairs}
            mpairs = {cm[r]: fm[r] for r in pairs}
            dirs = None
            if run_dirs_root is not None:
                dirs = {m: pathlib.Path(run_dirs_root) / f"mtx-{m}"
                        for m in list(fm.values()) + list(cm.values())}
            censored = [m for m in row["fine_models"] + row["coarse_models"]
                        if meta[by_model[m]].get("censored")]
            zero = [m for m in row["fine_models"] + row["coarse_models"]
                    if np.any(vec[by_model[m]] <= 0)]
            if zero:
                row["not_computed"] = (f"{zero} pass no background jet in the test sample "
                                       "or in a resampling; the ratio of rejections is "
                                       "undefined there -- see the Poisson intervals")
            else:
                row.update(P.paired_ratio(fv, cv, pairs=mpairs, run_dirs=dirs))
            if censored:
                row["censored_models"] = censored
                row["note"] = ("an AUC of 1 is floored at one discordant pair; the "
                               "ratio is a bound, not a value")
            rows.append(row)
        # BETWEEN RANDOM DRAWS (must-fix 11a). One run per draw, so no run
        # spread of their own: the test-sample term is measured, and the run
        # term is borrowed from the 17-class models (same head width), whose
        # ln metric varies over five runs by sd17; the difference of two single
        # runs carries 2 * sd17^2 of it.
        draws = per_level.get("random", {})
        if len(draws) > 1 and "17" in per_level and len(per_level["17"]) > 1:
            pts = [float(np.log(vec[k][0])) for k in per_level["17"].values()]
            sd17 = float(np.std(pts, ddof=1)) if all(np.isfinite(pts)) else float("nan")
            ds = sorted(draws)
            for i, di in enumerate(ds):
                for dj in ds[i + 1:]:
                    mi, mj = by_key(by_model, draws[di]), by_key(by_model, draws[dj])
                    row = {"family": fam, "task": task, "kind": kind, "metric": metric,
                           "fine": f"random draw {di}", "coarse": f"random draw {dj}",
                           "fine_models": [mi], "coarse_models": [mj]}
                    if np.any(vec[draws[di]] <= 0) or np.any(vec[draws[dj]] <= 0):
                        row["not_computed"] = "a draw passes no background jet"
                    else:
                        row.update(P.paired_ratio({mi: vec[draws[di]]}, {mj: vec[draws[dj]]},
                                                  pairs={mj: mi}))
                        comb = float(np.sqrt(row["ln_test_se"] ** 2 + 2 * sd17 ** 2))
                        row.update({"run_sd_proxy_17": sd17,
                                    "ln_combined_se_with_proxy": comb,
                                    "ci95_with_proxy": [float(np.exp(row["ln_ratio"] - 1.96 * comb)),
                                                        float(np.exp(row["ln_ratio"] + 1.96 * comb))]})
                    rows.append(row)
        if metric.startswith("eps_b@"):
            for lv, runs in per_level.items():
                cells = []
                for run, key in sorted(runs.items()):
                    m = meta[key]
                    cells.append({"model": by_key(by_model, key), "run": run,
                                  **P.rejection_interval(m["n_bkg"], m["k_pass"])})
                pooled = P.rejection_interval(sum(c["n_bkg"] for c in cells),
                                              sum(c["n_bkg_pass"] for c in cells))
                rej.append({"family": fam, "task": task, "kind": kind,
                            "eps_s": float(metric.split("@")[1]), "level": lv,
                            "per_run": cells, "pooled": pooled,
                            "pooled_caveat": "runs share the test jets, so pooling "
                                             "understates the error"})
    return {"ratios": rows, "rejections": rej, "point_values": models}


def by_key(by_model: dict, key: str) -> str:
    return next(m for m, k in by_model.items() if k == key)


# ----------------------------------------------------- audit B3, the check
# The audit's table (2026-09-29 report, B3): paired ratio, run range, run SD of
# ln r, and h, the median over runs of the single-run test-sample 95 % half-width
# of the log difference (probe.py's within-job contrasts). Our test term is the
# error of the MEAN over runs of ln r on jets they share, so it is compared with
# h / 1.96 as the scale, not expected to equal it.
B3 = [  # (family, task, kind, metric, fine, coarse, ratio, lo, hi, run_sd, h)
    ("probe", "bvc_resonant", "linear", "1-auc", "43", "17", 3.66, 2.92, 4.36, 0.16, 0.145),
    ("probe", "bvc_resonant", "linear", "1-auc", "162", "43", 1.068, 1.04, 1.08, 0.015, 0.088),
    ("probe", "bvc_resonant", "linear", "1-auc", "188", "162", 1.02, 0.96, 1.07, 0.046, 0.084),
    ("probe", "bvc_qcd", "linear", "1-auc", "188", "162", 1.146, 1.11, 1.23, 0.041, 0.094),
    ("probe", "retained_topology", "linear", "1-auc", "162", "43", 1.239, 1.18, 1.30, 0.038, 0.113),
    ("probe", "retained_topology", "linear", "1-auc", "162", "17", 1.537, 1.38, 1.87, 0.116, 0.123),
    ("probe", "bvc_resonant", "mlp", "1-auc", "43", "17", 2.85, 2.45, 3.34, 0.12, 0.145),
    ("probe", "retained_topology", "mlp", "1-auc", "162", "17", 1.50, 1.36, 1.85, 0.12, 0.153),
    ("probe", "bc_vs_rest", "linear", "1-auc", "162", "17", 1.64, 1.51, 1.72, 0.05, None),
    ("probe", "bc_vs_rest", "mlp", "1-auc", "162", "17", 1.50, 1.41, 1.59, 0.04, None),
    ("probe", "bvc_resonant", "linear", "1-auc", "162", "162+mass", 1.110, 1.07, 1.15, 0.027, 0.085),
    ("probe", "bvc_resonant", "linear", "1-auc", "17", "17+mass", 2.99, 1.85, 4.79, 0.37, None),
    ("probe", "bvc_resonant", "mlp", "1-auc", "17", "17+mass", 1.95, 1.39, 2.69, None, None),
    ("ft", "leg1", "N10000", "1-macro_auc", "188", "17", 1.316, 1.30, 1.34, 0.014, None),
    ("ft", "leg1", "N1000000", "1-macro_auc", "188", "17", 1.144, 1.13, 1.16, 0.008, None),
    ("ft", "leg1", "N1000", "1-macro_auc", "188", "43", 0.839, 0.75, 0.90, 0.069, None),
    ("ft", "leg2", "N1000000", "1-macro_auc", "188", "43", 1.020, 1.012, 1.030, 0.007, None),
]


def b3_compare(res: dict) -> list[dict]:
    got = {(r["family"], r["task"], r["kind"], r["metric"], r["fine"], r["coarse"]): r
           for r in res["ratios"]}
    out = []
    for fam, task, kind, metric, fine, coarse, ratio, lo, hi, sd, h in B3:
        r = got.get((fam, task, kind, metric, fine, coarse))
        row = {"cell": f"{fam}/{task}/{kind}/{metric} {coarse}/{fine}",
               "audit": {"ratio": ratio, "run_range": [lo, hi], "ln_run_sd": sd, "h95": h}}
        if r is None or "ratio" not in r:
            row["ours"] = None if r is None else {"not_computed": r.get("not_computed")}
        else:
            row["ours"] = {k: r[k] for k in ("ratio", "run_range", "ln_run_sd", "ln_test_se",
                                             "ln_combined_se", "ci95", "z")}
            row["ratio_agrees_to_1pc"] = abs(r["ratio"] / ratio - 1) < 0.01
            row["ours"]["significance_run_plus_test"] = abs(r["ln_ratio"]) / r["ln_combined_se"]
        out.append(row)
    return out


# ------------------------------------------------------------------ io
def save(path: pathlib.Path, vec: dict, meta: dict, prov: dict, b: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in vec.items()})
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(
        {"b": b, "seed": seed, "npz_sha256": _sha(path), "script_sha256": _sha(__file__),
         "provenance": prov, "meta": meta}, indent=1))
    print(f"wrote {path} ({len(vec)} vectors)")


def load(paths) -> tuple[dict, dict, list]:
    vec, meta, prov = {}, {}, []
    for p in paths:
        p = pathlib.Path(p)
        side = json.loads(p.with_suffix(".json").read_text())
        if side["npz_sha256"] != _sha(p):
            raise SystemExit(f"FATAL: {p} is not the file {p.with_suffix('.json')} describes")
        z = np.load(p)
        clash = set(z.files) & set(vec)
        if clash:
            raise SystemExit(f"FATAL: {sorted(clash)[:3]} appear in two replicate files")
        vec.update({k: z[k] for k in z.files})
        meta.update(side["meta"])
        prov.append({"path": str(p), "sha256": side["npz_sha256"], "b": side["b"],
                     "seed": side["seed"], "provenance": side["provenance"]})
    if len({(x["b"], x["seed"]) for x in prov}) > 1:
        raise SystemExit("FATAL: replicate files were made with different B or seed")
    return vec, meta, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe-replicates", "mass-replicates", "ft-replicates"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, type=pathlib.Path)
        s.add_argument("--b", type=int, default=B)
        s.add_argument("--seed", type=int, default=SEED)
        if name == "probe-replicates":
            s.add_argument("--probe-dirs", nargs="+", required=True, type=pathlib.Path)
            s.add_argument("--committed", nargs="*", default=[],
                           help="DIR=FILE: the committed probe_results each refit repeats")
        elif name == "mass-replicates":
            s.add_argument("--mass-dirs", nargs="+", required=True, type=pathlib.Path)
        else:
            s.add_argument("--leg1-root", required=True, type=pathlib.Path)
            s.add_argument("--leg2-root", required=True, type=pathlib.Path)
            s.add_argument("--leg1-metrics", required=True, type=pathlib.Path)
            s.add_argument("--leg2-metrics", required=True, type=pathlib.Path)
            s.add_argument("--procs", type=int, default=8)
    s = sub.add_parser("ratios")
    s.add_argument("--replicates", nargs="+", required=True, type=pathlib.Path)
    s.add_argument("--run-dirs-root", type=pathlib.Path, default=None,
                   help="v2: the pretraining runs' root; pairs must share their stream")
    s.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)

    if a.cmd == "probe-replicates":
        vec, meta, prov = probe_replicates(a.probe_dirs, a.b, a.seed)
        prov["reproduction"] = reproduction(a.probe_dirs, dict(x.split("=", 1) for x in a.committed))
        save(a.out, vec, meta, prov, a.b, a.seed)
    elif a.cmd == "mass-replicates":
        vec, meta, prov = mass_replicates(a.mass_dirs, a.b, a.seed)
        save(a.out, vec, meta, prov, a.b, a.seed)
    elif a.cmd == "ft-replicates":
        cj = {"leg1": json.loads(a.leg1_metrics.read_text()),
              "leg2": json.loads(a.leg2_metrics.read_text())}
        for k, p in (("leg1", a.leg1_metrics), ("leg2", a.leg2_metrics)):
            cj[k]["sha256"] = _sha(p)
        vec, meta, prov = ft_replicates(a.leg1_root, a.leg2_root, cj, a.b, a.seed, a.procs)
        save(a.out, vec, meta, prov, a.b, a.seed)
    else:
        if a.out.exists():
            raise SystemExit(f"FATAL: {a.out} exists; refusing to overwrite")
        vec, meta, prov = load(a.replicates)
        res = ratios(vec, meta, a.run_dirs_root)
        res = {"provenance": {"replicates": prov, "script_sha256": _sha(__file__),
                              "method_sha256": _sha(REPO / "src" / "stats" / "paired.py")},
               "method": {
                   "ratio": "exp(mean over paired runs of ln(coarse/fine)), coarse over fine; "
                            "above 1 = the coarser or control model is worse",
                   "metrics": "1-auc (probes), eps_b@X (background efficiency at signal "
                              "efficiency X; inverse ratio of rejections), sigma_eff (mass "
                              "probes), 1-macro_auc (fine-tuning)",
                   "run_error": "SD over runs of ln r_k / sqrt(n)",
                   "test_error": "SD over bootstrap resamplings of the test jets, one "
                                 "resampling shared by both models and every run, of the "
                                 "mean ln r_k",
                   "combined": "quadrature sum; 95% interval with Student t at the "
                               "Welch-Satterthwaite degrees of freedom",
                   "pairing": "run k with run k (v1: shared initialisation, not a "
                              "shared stream; audit B1); random draw k with run k",
                   "rejection_interval": "Garwood central 68.27% on the count of "
                                         "passing background jets"},
               **res}
        res["audit_b3_check"] = b3_compare(res)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}: {len(res['ratios'])} ratios, {len(res['rejections'])} rejection cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
