#!/usr/bin/env python3
"""Per-class recall on JetClass-II and JetClass (PI 2026-11-18): the fraction of a class's test jets the network labels
as that class (argmax), for every linear probe and fine-tuned network of the per-class figures (gallery_perclass).

The accuracy is the jet-weighted mean of the recalls, sum_c n_c r_c / N, asserted per network; on a test set with equal
class counts it is also their plain mean, asserted there. The class-balanced JetClass-II test set is NOT equal: 11 QCD
classes hold 224 to 4,783 jets against 5,000 (jc2bal census), so there the plain mean ("mean_recall") differs from the
accuracy and both are recorded. JetClass holds 200,000 jets per class.

  probe   the v5 linear probes at 10^6 training jets (experiments/EVAL/probe_bal.py --grid v5): the logits recomputed in
          float64 from the saved probe (coef, intercept, scaler; fits_v5_probs) on the stored test features, the
          standardisation done as scikit-learn's on float32 features (the saved p is float16, whose argmax ties on
          ~300 jets); the correct count must equal the fit's stored accuracy exactly. Networks: the 27 paper runs at
          best70, the random network (init-s*, checkpoint init) and its BatchNorm-recomputed twin (initbn-s*, init_bn).
  ft      the fixed-recipe fine-tuning cells (scripts/build_ft_budget_jobs.py, /data/results/ft_v2/fixed) at --n:
          leg_jc2b from features/logits.npy (the class_logits columns), leg2 (JetClass) from pred.root, read and argmaxed
          as experiments/FT/budget_metrics.py does. A cell without DONE is never read; --require-all refuses the run
          when any is missing.

Test-jet spread (decision L3): B resamples of the jets with replacement within each class
(experiments/STATS/paired_bootstrap.counts, its seed and B, so the resamples are L3's), shared by every network. Per
network and class the SD of the recall over the resamples; per group of runs the SD of their mean; per difference of
two groups the SD of the difference of the means (boot_sd), with sd_a, sd_b over the runs (ddof 1), sigma =
sqrt(sd_a^2 + sd_b^2 + boot_sd^2) and the classes lower / higher by more than 3 sigma.

Output (atomic): <out>/probe_<task>.json, <out>/ft_<task>_N<n>.json; per-class lists indexed by class (JetClass-II:
jet_label; JetClass: CLASSES_JC1 order).

    python3 experiments/EVAL/perclass_recall.py probe --task jc2 jc1 --out /data/results/eval/perclass_recall
    python3 experiments/EVAL/perclass_recall.py ft --task jc2 jc1 --n 1000000 --require-all \\
        --out /data/results/eval/perclass_recall
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
PROBE_BAL = pathlib.Path("/data/results/eval/v2_linprobe_bal")
PROBE_OLD = pathlib.Path("/data/results/eval/v2_linprobe")
FT_ROOT = pathlib.Path("/data/results/ft_v2/fixed")
N_PROBE = 1_000_000
# task -> (classes, probe dataset name, root of its test features, fine-tuning tree)
TASKS = {"jc2": (188, "jc2bal", PROBE_BAL, "leg_jc2b"), "jc1": (10, "jc1", PROBE_OLD, "leg2")}
CLASSES_JC1 = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
               "label_Tbqq", "label_Tbl")       # linear_probe_v2.JC1_CLASSES, budget_metrics.JC1_CLASSES
INIT_MODELS = ("init-s1", "init-s2", "init-s3")
SCRATCH_SEEDS = (1, 2, 3)               # build_ft_budget_jobs.SCRATCH_SEEDS
CHUNK = 200_000
# the differences of the per-class figures (gallery_perclass): the label sets and mass networks against 188 classes
# (Fig. 4), the mass networks against their own label set, the 17-class partitions against 188 classes
DIFFS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L188"), ("R16_Q1_MASS", "L188"),
         ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"), ("FLAV_F0", "L188"), ("FLAV_F1", "L188"),
         ("FLAV_F1R", "L188"), ("RAND", "L188"))


def fatal(msg: str):
    print(f"FATAL: {msg}", flush=True)
    sys.exit(42)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def paper_groups(root: pathlib.Path = REPO) -> dict:
    """{group: [run]} of configs/arms/v2_paper_runs.txt by grid arm, the five random partitions also pooled as RAND
    (mlst_results._paper and _pooled)."""
    slug = {a["name"].lower().replace("_", ""): a["name"]
            for a in json.loads((root / "configs/arms/v2_grid.json").read_text())["arms"]}
    out: dict = {}
    for ln in (root / "configs/arms/v2_paper_runs.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            out.setdefault(slug[re.fullmatch(r"mtx-([a-z0-9]+)-s\d+", ln.strip()).group(1)], []).append(ln.strip())
    out["RAND"] = [m for a in sorted(out) if a.startswith("RAND2_") for m in out[a]]
    return out


# ------------------------------------------------------------------ predictions
def probe_pred(task: str, model: str, ck: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(argmax class, true class, source record) of one linear probe on its test jets."""
    k, ds, feat_root, _ = TASKS[task]
    npz = PROBE_BAL / "fits_v5_probs" / model / ck / f"{ds}_N{N_PROBE}.npz"
    fdir = feat_root / model / ds / "test" / ck
    fit = PROBE_BAL / "fits_v5" / f"{model}.json"
    for f in (npz, fdir / "features.npy", fit):
        if not f.exists():
            fatal(f"{f} missing")
    z = np.load(npz)
    y = np.load(fdir / "label188.npy").astype(np.int64)
    x = np.load(fdir / "features.npy", mmap_mode="r")
    if not (np.array_equal(np.load(fdir / "rows.npy"), np.arange(y.size)) and x.shape == (y.size, 128)
            and np.array_equal(z["rows"], np.arange(y.size)) and np.array_equal(z["y"].astype(np.int64), y)):
        fatal(f"{npz} and {fdir} do not hold the same test jets in order")
    coef, b, mu, sc, cls = z["coef"], z["intercept"], z["scaler_mean"], z["scaler_scale"], z["classes"]
    pred = np.empty(y.size, np.int64)
    for i in range(0, y.size, CHUNK):
        a = np.asarray(x[i:i + CHUNK], dtype=np.float32).copy()
        a -= mu                                 # StandardScaler.transform on float32 features, in place, as the fit
        a /= sc
        pred[i:i + CHUNK] = cls[(a.astype(np.float64) @ coef.T + b).argmax(1)]
    stored = json.loads(fit.read_text())["cells"][ck][ds][str(N_PROBE)]["accuracy"]
    n_ok = int((pred == y).sum())
    if abs(stored * y.size - n_ok) > 0.5:
        fatal(f"{model} {ck} {task}: {n_ok} correct, the fit stored {stored * y.size:.1f}")
    return pred, y, {"source": str(npz), "accuracy_stored": stored}


def softmax(z: np.ndarray) -> np.ndarray:
    """experiments/FT/budget_metrics.softmax."""
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ft_pred(task: str, cell: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(argmax class, true class, source record) of one fine-tuning cell, as budget_metrics.py scores it."""
    if task == "jc1":
        import uproot
        t = uproot.open(cell / "pred.root")["Events"]
        if not all(c in t.keys() for c in CLASSES_JC1):
            fatal(f"{cell}/pred.root lacks the label branches")
        p = np.stack([t[f"score_{c}"].array(library="np") for c in CLASSES_JC1], 1).astype(np.float64)
        y = np.stack([t[c].array(library="np") for c in CLASSES_JC1], 1).argmax(1).astype(np.int64)
        return p.argmax(1), y, {"source": str(cell / "pred.root")}
    fd = cell / "features"
    man = json.loads((fd / "extract_manifest.json").read_text())
    lo, hi = man.get("logit_columns", {}).get("class_logits", (0, 188))
    z = np.load(fd / "logits.npy", mmap_mode="r")
    y = np.load(fd / "label188.npy").astype(np.int64)
    if hi - lo != 188 or z.shape[0] != y.size:
        fatal(f"{fd}: class logits {lo}:{hi} of {z.shape} for {y.size} jets")
    pred = np.empty(y.size, np.int64)
    for i in range(0, y.size, CHUNK):
        pred[i:i + CHUNK] = softmax(np.asarray(z[i:i + CHUNK, lo:hi], dtype=np.float64)).argmax(1)
    return pred, y, {"source": str(fd / "logits.npy")}


def ft_cells(task: str, n: int, runs: list) -> tuple[dict, list]:
    """({model: cell}, [missing cell]) of the cells build_ft_budget_jobs.fixed_cells plans: each paper run (training
    seed 1, one fine-tuning per run) as mtx-<init>, and from scratch with training seeds 1-3 as scratch-s<k>."""
    # each row at its recorded rate (build_ft_budget_jobs.paper_cells: ft_v2/fixed, or ft_v2/fixed_rate where it moved)
    spec = importlib.util.spec_from_file_location("build_ft_budget_jobs", pathlib.Path(__file__).resolve().parents[2]
                                                  / "scripts/build_ft_budget_jobs.py")
    bfb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bfb)
    cells_at = bfb.paper_cells({"jc2": "jc2b", "jc1": "jc1"}[task], n)
    # the builder's paths are under /data/results/ft_v2; read them under FT_ROOT's parent (a test's tree)
    base = FT_ROOT.parent
    plan = {m: base / pathlib.Path(cells_at[m]).relative_to("/data/results/ft_v2")
            for m in list(runs) + [f"scratch-s{s}" for s in SCRATCH_SEEDS]}
    cells = {m: c for m, c in plan.items() if (c / "DONE").exists()}
    return cells, [str(c.relative_to(FT_ROOT.parent)) for m, c in plan.items() if m not in cells]


# ------------------------------------------------------------------ recall, its identity and its test-jet spread
def recall_doc(preds: dict, y: np.ndarray, k: int, groups: dict, b: int, seed: int, counts) -> dict:
    """Per network the recall of every class and its bootstrap SD, the identity with the accuracy asserted; per group
    the bootstrap SD of the mean; per difference of DIFFS whose groups are complete, L3's terms (module docstring)."""
    names = list(preds)
    ok = np.stack([preds[m] == y for m in names], 1).astype(np.uint8)            # (jets, networks)
    n_c = np.bincount(y, minlength=k)
    if (n_c == 0).any():
        fatal(f"classes {np.flatnonzero(n_c == 0).tolist()} have no test jet")
    c = counts(y, b, seed)                                                       # (b, jets), within-class resamples
    correct = np.zeros((k, len(names)), np.int64)
    boot = np.zeros((b, k, len(names)))                                          # resampled recalls
    for cls in range(k):
        idx = np.flatnonzero(y == cls)
        correct[cls] = ok[idx].sum(0)
        # integer sums below 2^24, exact in float32
        boot[:, cls] = (c[:, idx].astype(np.float32) @ ok[idx].astype(np.float32)).astype(np.float64) / idx.size
    rec = correct / n_c[:, None]
    balanced = bool((n_c == n_c[0]).all())
    models = {}
    for j, m in enumerate(names):
        n_ok = int(ok[:, j].sum())
        acc, weighted, plain = n_ok / y.size, float((n_c * rec[:, j]).sum() / y.size), float(rec[:, j].mean())
        if int(correct[:, j].sum()) != n_ok or abs(weighted - acc) > 1e-12:
            fatal(f"{m}: accuracy {acc} is not the jet-weighted mean of its recalls {weighted}")
        if balanced and abs(plain - acc) > 1e-12:
            fatal(f"{m}: equal class counts, but accuracy {acc} != mean recall {plain}")
        models[m] = {"recall": rec[:, j].tolist(), "boot_sd": boot[:, :, j].std(0, ddof=1).tolist(),
                     "n_correct": n_ok, "accuracy": acc, "mean_recall": plain}
    col = {m: j for j, m in enumerate(names)}
    have = {g: [m for m in ms if m in col] for g, ms in groups.items()}
    full = {g: v for g, v in have.items() if v and len(v) == len(groups[g])}
    out_groups = {g: {"models": v, "boot_sd_mean": boot[:, :, [col[m] for m in v]].mean(2).std(0, ddof=1).tolist()}
                  for g, v in full.items()}
    diffs = {}
    for a, bb in DIFFS:
        if a not in full or bb not in full:
            continue
        ja, jb = [col[m] for m in full[a]], [col[m] for m in full[bb]]
        delta = rec[:, ja].mean(1) - rec[:, jb].mean(1)
        bsd = (boot[:, :, ja].mean(2) - boot[:, :, jb].mean(2)).std(0, ddof=1)
        sd = [rec[:, j].std(1, ddof=1) if len(j) > 1 else None for j in (ja, jb)]
        sigma = np.sqrt(sd[0] ** 2 + sd[1] ** 2 + bsd ** 2) if all(x is not None for x in sd) else None
        diffs[f"{a}-{bb}"] = {
            "a": full[a], "b": full[bb], "delta": delta.tolist(), "boot_sd": bsd.tolist(),
            "sd_a": None if sd[0] is None else sd[0].tolist(), "sd_b": None if sd[1] is None else sd[1].tolist(),
            "sigma": None if sigma is None else sigma.tolist(),
            "lower": [] if sigma is None else np.flatnonzero(delta < -3 * sigma).tolist(),
            "higher": [] if sigma is None else np.flatnonzero(delta > 3 * sigma).tolist()}
    return {"n_jets": int(y.size), "class_counts": n_c.tolist(), "balanced": balanced,
            "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "b": b, "seed": seed,
            "resampling": "test jets, with replacement within each class (paired_bootstrap.counts)",
            "models": models, "groups": out_groups, "diffs": diffs}


def write(doc: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc) + "\n")
    os.replace(tmp, path)
    print(f"{path}: {len(doc['models'])} networks, {len(doc['diffs'])} differences", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "ft"))
    ap.add_argument("--task", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    ap.add_argument("--n", type=int, default=N_PROBE, help="ft: the training-set size of the cells")
    ap.add_argument("--require-all", action="store_true", help="ft: refuse the run if any cell lacks DONE")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    pb = _load("paired_bootstrap", "experiments/STATS/paired_bootstrap.py")
    groups = paper_groups()
    for task in a.task:
        k = TASKS[task][0]
        preds, src, y0, missing = {}, {}, None, []
        if a.mode == "probe":
            jobs = [(m, m, "best70") for g in sorted(groups) if g != "RAND" for m in groups[g]] \
                + [(m, m, "init") for m in INIT_MODELS] + [("initbn" + m[len("init"):], m, "init_bn") for m in INIT_MODELS]
            grp = {**groups, "INIT": list(INIT_MODELS), "INIT_BN": ["initbn" + m[len("init"):] for m in INIT_MODELS]}
        else:
            cells, missing = ft_cells(task, a.n, [m for g in sorted(groups) if g != "RAND" for m in groups[g]])
            if missing and a.require_all:
                fatal(f"{task} N{a.n}: {len(missing)} cells without DONE: {missing}")
            jobs = list(cells.items())
            grp = {**groups, "SCRATCH": sorted(m for m in cells if m.startswith("scratch-"))}
        for job in jobs:
            name = job[0]
            pred, y, rec = probe_pred(task, job[1], job[2]) if a.mode == "probe" else ft_pred(task, job[1])
            if y0 is None:
                y0 = y
            elif not np.array_equal(y, y0):
                fatal(f"{name} is scored on other test jets than {next(iter(preds))}")
            preds[name], src[name] = pred, rec
            print(f"  {task} {name}: accuracy {float((pred == y).mean()):.6f}", flush=True)
        if not preds:
            fatal(f"{task} {a.mode}: no network to read")
        doc = recall_doc(preds, y0, k, grp, pb.B_DEFAULT, pb.SEED_DEFAULT, pb.counts)
        for m, r in src.items():
            doc["models"][m].update(r)
        doc.update({"script": "experiments/EVAL/perclass_recall.py", "task": task, "mode": a.mode,
                    "n_train": N_PROBE if a.mode == "probe" else a.n, "missing": missing,
                    "classes": list(CLASSES_JC1) if task == "jc1" else None,
                    "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
        write(doc, a.out / (f"probe_{task}.json" if a.mode == "probe" else f"ft_{task}_N{a.n}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
