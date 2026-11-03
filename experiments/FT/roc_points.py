#!/usr/bin/env python3
"""Per-class ROC curves for the paper's ROC figures (ROC redesign, PI 2026-11-02): one JSON per
(family, model), background rejection 1/eps_B at each signal efficiency of a fine grid, with the
20-background-jet cut. ft_curves.py's outputs, grid and tests stay as they are; its readers and
discriminants are reused here.

  families (OUT_ROOT/<family>/<model>.json; the job writes <family>/DONE after the family)
    jc1_ft     fine-tuning on JetClass at 10^6 jets (pred.root): each of the nine signal classes
               against QCD on D = p_S/(p_S+p_QCD), p renormalised over the 10 classes (ParT Eq. 2)
    jc1_probe  the linear probes on JetClass at 10^6 (jc1_N1000000.npz), the same nine curves
    pairs      the linear probes on the class pairs of the balanced JetClass-II sets
               (jc2bal_pairs_<pair>.npz: P(signal), signal = the pair's first side)
    top_ft     fine-tuning on top tagging at the full set: softmax(logits)[1]
    qg_ft      ... on quark/gluon at the full set, Pythia ("qg") and Herwig ("qg_herwig")
    jc2_ovr    the linear probes on the balanced JetClass-II test set at 10^6, one-vs-rest:
               p_c against every other class, all 188 native classes (keys "0" ... "187")

ROC AND THE CUT. fpr, tpr = roc_curve(y, D, drop_intermediate=False) with fps = rint(fpr N_B);
eps_B at each grid point by linear interpolation (ft_curves' convention), rejection 1/eps_B (null
where eps_B = 0, never capped). The cut is the first ROC vertex at which at least MIN_BKG_PASS
background jets pass (more than 20 only with tied scores): every grid point at or above its eps_S
has at least 20 background jets behind it, and a reader draws only those. The cut is null when no
vertex below full signal efficiency reaches 20 background jets (the curve lies beyond the limit at
every eps_S the grid holds). The grid stops at 0.995: interpolating at eps_S = 1 returns the last
vertex, eps_B = 1.

Every file is written whole (tmp + os.replace) and a file that exists is skipped, never
overwritten, so a retried pod resumes. --out-dir must be OUT_ROOT/<family> (or <--allow-out>/<family>
in tests).

    python3 experiments/FT/roc_points.py ft --family jc1_ft --tree /data/results/ft_v2/best70/leg2 \\
        --task jc1 --n 1000000 --inits l188-s1 ... --out-dir /data/results/ft_v2/curves_v2/jc1_ft
    python3 experiments/FT/roc_points.py probes --family pairs \\
        --probs-root .../fits_v4_probs .../fits_v4_sib_probs --fits-root .../fits_v4 .../fits_v4_sib \\
        --files 'jc2bal_pairs_*' --models mtx-l188-s1 ... --out-dir /data/results/ft_v2/curves_v2/pairs
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import functools
import hashlib
import importlib.util
import json
import os
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FC = _load("ft_curves", "experiments/FT/ft_curves.py")       # cell readers, versus/summed, write_json
LP, BM = FC.LP, FC.BM
PB = _load("probe_bal", "experiments/EVAL/probe_bal.py")     # SIBLING_PAIRS

OUT_ROOT = pathlib.Path("/data/results/ft_v2/curves_v2")
EPS_GRID = tuple(round(0.001 * i, 3) for i in range(1, 11)) + tuple(round(0.015 + 0.005 * i, 3) for i in range(197))
MIN_BKG_PASS = 20               # the fewest background jets behind a drawn point (mlst_v4.JC1_MIN_BKG)
FT_TASK = {"jc1_ft": "jc1", "top_ft": "top", "qg_ft": "qg"}     # the fine-tuning families and their task
# the nine JetClass signal classes in the paper table's order (mlst_v4.JC1_COLS)
JC1_SIGNALS = ("label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql", "label_Tbqq", "label_Tbl",
               "label_Wqq", "label_Zqq")
DISCR = {"jc1": "p_S/(p_S+p_QCD), p renormalised over the 10 classes (ParT Eq. 2)",
         "binary": "softmax(logits)[1]",
         "pairs": "binary linear probe P(signal)",
         "jc2_ovr": "p_c, one-vs-rest: background = every other class of the balanced test set"}
FT_MANIFEST_KEYS = ("subset", "ft_seed", "lr", "head_lr_mult", "recipe", "gpu_device_name")
PAIR_SIDES = {**LP.PAIRS, **LP.QCD_PAIRS, **PB.SIBLING_PAIRS}
CKPTS = {"init": "init", "init_bn": "initbn"}   # an untrained trunk's checkpoint -> its model name prefix
SIZE = re.compile(r"_N(\d+)$")


@functools.lru_cache(maxsize=None)        # one source file serves every curve of a model
def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def curve(y: np.ndarray, d: np.ndarray, signal, background, discriminant: str, n_train, source: pathlib.Path) -> dict:
    """One ROC curve (y = 1 signal) on the grid, with the cut; see the docstring."""
    from sklearn.metrics import auc, roc_curve
    y = np.asarray(y).astype(np.int64)
    n_s, n_b = int((y == 1).sum()), int((y == 0).sum())
    if n_s == 0 or n_b == 0 or n_s + n_b != y.size:
        raise SystemExit(f"FATAL: {source}: {n_s} signal and {n_b} background jets of {y.size}")
    fpr, tpr, _ = roc_curve(y, d, drop_intermediate=False)
    fps = np.rint(fpr * n_b)
    eps_b = np.interp(np.asarray(EPS_GRID, dtype=np.float64), tpr, fpr)
    hit = np.flatnonzero(fps >= MIN_BKG_PASS)
    i = int(hit[0]) if hit.size else None
    cut = None if i is None or tpr[i] >= 1.0 else {
        "eps_s": float(tpr[i]), "eps_b": float(fpr[i]), "rejection": float(1.0 / fpr[i]), "n_bkg_pass": int(fps[i])}
    return {"signal": list(signal), "background": list(background), "discriminant": discriminant,
            "n_signal": n_s, "n_background": n_b, "n_train": n_train, "auc": float(auc(fpr, tpr)),
            "rejection": [float(1.0 / e) if e > 0 else None for e in eps_b],
            "n_bkg_pass": np.rint(eps_b * n_b).astype(int).tolist(), "cut": cut,
            "n_bkg_pass_full_eff": int(fps[np.flatnonzero(tpr >= 1.0)[0]]),
            "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "source": {"file": str(source), "sha256": sha256(source)}}


def seed_of(model: str) -> int:
    return int(model.rsplit("-s", 1)[1])


def curve_doc(family: str, model: str, method: str, checkpoint, ft_manifest, curves: dict) -> dict:
    return {"family": family, "model": model, "seed": seed_of(model), "method": method, "checkpoint": checkpoint,
            "ft_manifest": ft_manifest, "script": "experiments/FT/roc_points.py",
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "repo_ref": os.environ.get("REPO_REF"), "eps_grid": list(EPS_GRID), "min_bkg_pass": MIN_BKG_PASS,
            "curves": curves}


def jc1_curves(p: np.ndarray, y: np.ndarray, names, n_train, source: pathlib.Path) -> dict:
    """The nine signal classes against QCD; p renormalised over the 10 classes."""
    if list(names) != list(LP.JC1_CLASSES):
        raise SystemExit(f"FATAL: {source}: JetClass columns {list(names)} are not {list(LP.JC1_CLASSES)}")
    p = p.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    qcd = list(names).index("label_QCD")
    bkg = FC.summed(p, y, (qcd,))
    out = {}
    for s in JC1_SIGNALS:
        t, d = FC.versus(p, y, list(names).index(s), bkg)
        out[s] = curve(t, d, [s], ["label_QCD"], DISCR["jc1"], n_train, source)
    return out


def native_names() -> dict:
    with (REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv").open() as f:
        return {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}


def ovr_curves(p: np.ndarray, y: np.ndarray, n_train, source: pathlib.Path) -> dict:
    """Each native class present against every other class of the test set, on p_c."""
    if p.ndim != 2 or p.shape != (y.size, LP.DATASETS["jc2_188"][0]):
        raise SystemExit(f"FATAL: {source}: scores {p.shape} for {y.size} jets, not the 188-class task")
    p = p.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    names = native_names()
    return {str(c): curve((y == c).astype(np.int64), p[:, c], [names[int(c)]], ["every other class"],
                          DISCR["jc2_ovr"], n_train, source) for c in np.unique(y)}


def ft_cells(tree: pathlib.Path, n: int, inits) -> list:
    """[(init, seed, cell)] of every DONE cell at N=n (of these inits; ft_curves.discover's rules,
    on these cells only): a requested init with none is fatal, a cell trained to NaN is fatal."""
    out = []
    for cell in sorted(tree.glob(f"*/N{n}/s*")):
        init = cell.parent.parent.name
        if not cell.is_dir() or ".partial." in cell.name or cell.name.endswith(".lock") or (inits and init not in inits):
            continue
        if not (cell / "DONE").exists():
            print(f"  not finished, skipped: {cell}", flush=True)
            continue
        if BM.diverged(cell):
            raise SystemExit(f"FATAL: {cell} trained to a NaN loss; not a result")
        out.append((init, cell.name, cell))
    have = {i for i, _, _ in out}
    if not out or (inits and set(inits) - have):
        raise SystemExit(f"FATAL: no DONE cell at N{n} under {tree} for {sorted(set(inits or ()) - have) or 'any init'}")
    for init in have:
        if not init.startswith("scratch") and sum(i == init for i, _, _ in out) > 1:
            raise SystemExit(f"FATAL: {init} has more than one fine-tuning at N{n}; one per pretraining run")
    return out


def ft_manifest(cell: pathlib.Path):
    f = cell / "ft_manifest.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    return {k: m.get(k) for k in FT_MANIFEST_KEYS}


def ft_cell_curves(task: str, cell: pathlib.Path, n: int) -> dict:
    if task == "jc1":
        _, y, p, names = FC.cell_scores(cell, "jc1")
        return jc1_curves(p, y, names, n, cell / "pred.root")
    out = {}
    for test_set in ("pythia", "herwig") if task == "qg" else ("pythia",):
        _, y, s, _ = FC.cell_scores(cell, task, test_set)
        out[task if test_set == "pythia" else "qg_herwig"] = curve(
            y, s, [BM.SIGNAL[task][1][BM.SIGNAL_COLUMN]], [BM.SIGNAL[task][1][1 - BM.SIGNAL_COLUMN]],
            DISCR["binary"], n, cell / "features" / BM.TEST_FILES[test_set][0])
    return out


def out_dir(path: pathlib.Path, family: str, allow) -> pathlib.Path:
    roots = [OUT_ROOT.resolve()] + ([allow.resolve()] if allow else [])
    p = path.resolve()
    if p.name != family or p.parent not in roots:
        raise SystemExit(f"FATAL: --out-dir {path} is not {OUT_ROOT}/{family}; this script writes there only")
    return p


def run_ft(a, odir: pathlib.Path) -> None:
    task = FT_TASK[a.family]
    if a.task != task:
        raise SystemExit(f"FATAL: family {a.family} is task {task}, not {a.task}")
    if task in ("top", "qg"):
        BM.check_label_order(task)
    for init, seed, cell in ft_cells(a.tree, a.n, a.inits):
        model = f"{init}-{seed}" if init.startswith("scratch") else f"mtx-{init}"
        path = odir / f"{model}.json"
        if path.exists():
            print(f"skip {path}: exists", flush=True)
            continue
        FC.write_json(path, curve_doc(a.family, model, "fine-tuning", None if init.startswith("scratch") else "best70",
                                      ft_manifest(cell), ft_cell_curves(task, cell, a.n)))


def pair_n_train(fits_roots, model_dir: str, ckpt: str, pair: str, cache: dict) -> int:
    """The pair's training jets from its fit file (<fits-root>/<model>.json)."""
    for root in fits_roots:
        f = root / f"{model_dir}.json"
        if f not in cache:
            cache[f] = json.loads(f.read_text()) if f.exists() else {}
        rec = cache[f].get("cells", {}).get(ckpt, {}).get("jc2bal_pairs", {}).get(pair)
        if rec is not None:
            return int(rec["n_train"])
    raise SystemExit(f"FATAL: no fit of {model_dir} {ckpt} pair {pair} under {[str(r) for r in fits_roots]}")


def probe_curves(family: str, f: pathlib.Path, model_dir: str, ckpt: str, fits_roots, cache: dict) -> dict:
    z = np.load(f)
    y = z["y"].astype(np.int64)
    if family == "pairs":
        pair = f.stem.removeprefix("jc2bal_pairs_")
        if pair not in PAIR_SIDES:
            raise SystemExit(f"FATAL: {f}: unknown pair {pair}")
        side = lambda v: [v] if isinstance(v, str) else list(v)
        sig, bkg = PAIR_SIDES[pair]
        return {pair: curve(y, z["s"].astype(np.float64), side(sig), side(bkg), DISCR["pairs"],
                            pair_n_train(fits_roots, model_dir, ckpt, pair, cache), f)}
    n = int(SIZE.search(f.stem).group(1))
    if family == "jc1_probe":
        return jc1_curves(z["p"], y, LP.JC1_CLASSES, n, f)
    return ovr_curves(z["p"], y, n, f)


def run_probes(a, odir: pathlib.Path) -> None:
    models = a.models or sorted({d.name for r in a.probs_root for d in r.iterdir() if d.is_dir()})
    cache = {}
    for md in models:
        for ckpt in ("init", "init_bn") if md.startswith("init-") else ("best70",):
            model = f"{CKPTS[ckpt]}-{md.removeprefix('init-')}" if ckpt in CKPTS else md
            files = {}
            for root in a.probs_root:
                for f in sorted((root / md / ckpt).glob("*.npz")):
                    if any(fnmatch.fnmatch(f.stem, pat) for pat in a.files):
                        if f.name in files:
                            raise SystemExit(f"FATAL: {f.name} of {md} {ckpt} is in two --probs-roots")
                        files[f.name] = f
            if not files:
                if a.models:
                    raise SystemExit(f"FATAL: no {a.files} of {md} {ckpt} under {[str(r) for r in a.probs_root]}")
                continue
            path = odir / f"{model}.json"
            if path.exists():
                print(f"skip {path}: exists", flush=True)
                continue
            curves = {}
            for f in files.values():
                curves.update(probe_curves(a.family, f, md, ckpt, a.fits_root, cache))
                print(f"  {model:16} {f.name}", flush=True)
            FC.write_json(path, curve_doc(a.family, model, "linear probe", ckpt, None, curves))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("ft")
    f.add_argument("--family", required=True, choices=tuple(FT_TASK))
    f.add_argument("--tree", required=True, type=pathlib.Path, help="<tree>/<init>/N<N>/s<S>/")
    f.add_argument("--task", required=True, choices=("jc1", "top", "qg"))
    f.add_argument("--n", required=True, type=int)
    f.add_argument("--inits", nargs="+", help="only these init directories (default: every one)")
    p = sub.add_parser("probes")
    p.add_argument("--family", required=True, choices=("jc1_probe", "pairs", "jc2_ovr"))
    p.add_argument("--probs-root", required=True, nargs="+", type=pathlib.Path)
    p.add_argument("--fits-root", nargs="*", default=[], type=pathlib.Path, help="pairs: the fit files' roots")
    p.add_argument("--files", required=True, nargs="+", help="file stems or globs, e.g. jc1_N1000000 'jc2bal_pairs_*'")
    p.add_argument("--models", nargs="+", help="default: every model directory under the --probs-roots")
    for s in (f, p):
        s.add_argument("--out-dir", required=True, type=pathlib.Path)
        s.add_argument("--allow-out", type=pathlib.Path, help="tests: another root for --out-dir")
    a = ap.parse_args(argv)
    odir = out_dir(a.out_dir, a.family, a.allow_out)
    (run_ft if a.cmd == "ft" else run_probes)(a, odir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
