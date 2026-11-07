#!/usr/bin/env python3
"""Tagger performance against jet pT and the JetClass confusion matrices (figure round 3, PI
2026-11-06; docs/figstyle/TAGGER_GAP.md), from per-jet scores already stored. No refit.

  jc2  JetClass-II, the frozen linear probe at 1e6 (probe_bal.py's <probs-root>/<model>/best70/
       jc2bal_N1000000.npz: rows, y, p) on the class-balanced test set, in the ten pT bins of
       atlas_rebin.PT_EDGES. Jet pT from the staged test file (--test-parquet): the extraction streamed
       that one file through its data config's selection in file order (extract_v2.py: one worker,
       fetch by files, no shuffling at test time), so stream row i is the file's i-th selected row.
       The probe's labels must equal the file's selected labels and the extraction's label188.npy row
       for row (the staged rows are in random order, so a shifted or permuted row map fails this), or
       the script stops. Per bin: the accuracy (argmax over the 188 outputs) with its binomial spread, the
       macro one-vs-rest AUC over the classes present (linear_probe_v2.native_metrics' macro_auc_ovr), and X->bb against X->cc on p_bb / (p_bb + p_cc)
       on those two classes' jets: its AUC and the rejection at 50% signal efficiency with the
       background jets behind it.
  jc1  JetClass, fine-tuned at 1e6 with the long budget (<tree>/<init>/N1000000/s<S>/pred.root). The
       cell's pred.root was cut to its label and score branches (slim_outputs.py pred), so jet pT is
       read from the JetClass test files in weaver's predict order: the files of each DataLoader worker
       as the cell's predict.log records them (weaver's "Restarted DataIter ... file_list"), each read
       in file order (no selection, no shuffling at test time), batches of PREDICT_BATCH taken from
       the workers in turn. The labels so rebuilt must equal pred.root's jet for jet (each file holds
       one class, so this fixes the batch size, the interleaving and every file boundary), or the
       script stops. Five 100-GeV bins 500-1000. Per bin: the accuracy (argmax over the ten outputs) with
       its binomial spread, the macro one-vs-rest AUC over the ten classes and each signal class's rejection of QCD at ParT's working point on p_S / (p_S + p_QCD)
       (ft_curves.jc1_block's discriminant and eps_S). Beside it, into --confusion-out, the 10x10
       confusion matrix: argmax of the ten outputs, rows the true class in linear_probe_v2.JC1_CLASSES
       order, counts and each row normalised to one.

A rejection is ft_curves.working_point's (eps_B interpolated on the ROC, capped at the background
count, is_bound when under one background jet passes) with rel_stat = 1/sqrt(eps_B N_B), the Poisson
spread of the passing count. Jets outside the edges are counted (n_outside), not binned. Every
model of one task must be scored on the same jets: a jets_sha256 that differs from one already
written stops the script. One JSON per model, written as .new and renamed; one that exists is
skipped.

    python3 experiments/EVAL/perf_vs_pt.py jc2 --out-dir /data/results/eval/perf_vs_pt/jc2
    python3 experiments/EVAL/perf_vs_pt.py jc1 --out-dir /data/results/eval/perf_vs_pt/jc1 \\
        --confusion-out /data/results/eval/confusion_jc1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.EVAL import atlas_rebin  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FC = _load("ft_curves", "experiments/FT/ft_curves.py")      # discriminant, working point, pred.root reader
LP = FC.LP                                                  # JetClass class order and working points

PT_EDGES = {"jc2": list(atlas_rebin.PT_EDGES), "jc1": [500, 600, 700, 800, 900, 1000]}   # GeV
EVAL = pathlib.Path("/data/results/eval")
PROBS_ROOT = EVAL / "v2_linprobe_bal" / "fits_v4_probs"
EXTRACT_ROOT = EVAL / "v2_linprobe_bal"
TEST_PARQUET = pathlib.Path("/data/finetune/jc2_bal/test.parquet")
PROBE_FILE, CKPT = "jc2bal_N1000000.npz", "best70"
JC1_TREE, JC1_N = pathlib.Path("/data/results/ft_v2/budget_long/leg2"), 1_000_000
PREDICT_BATCH = 512          # weaver --predict --batch-size of the fine-tuning jobs (build_ft_budget_jobs.py)
LABEL_MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
BB_CC = ("label_X_bb", "label_X_cc")                        # signal, background


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def paper_models() -> list[str]:
    return [l.strip() for l in (REPO / "configs/arms/v2_paper_runs.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def bin_index(pt: np.ndarray, edges) -> np.ndarray:
    """Each jet's bin, [lo, hi) with the last bin closed; -1 outside the edges."""
    e = np.asarray(edges, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    i = np.searchsorted(e, pt, side="right") - 1
    i[pt == e[-1]] = e.size - 2
    i[(pt < e[0]) | (pt > e[-1])] = -1
    return i


def macro_auc(p: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """(mean one-vs-rest AUC over the classes present, their number); None under two classes."""
    from sklearn.metrics import roc_auc_score
    cs = np.unique(y)
    if cs.size < 2:
        return None, int(cs.size)
    return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in cs])), int(cs.size)


def accuracy(p: np.ndarray, y: np.ndarray) -> dict:
    """{"accuracy": share of jets whose argmax is the true class, "accuracy_stat": its binomial spread
    sqrt(a (1 - a) / n)}; both None for no jets."""
    if y.size == 0:
        return {"accuracy": None, "accuracy_stat": None}
    a = float(np.mean(p.argmax(1) == y))
    return {"accuracy": a, "accuracy_stat": float(np.sqrt(a * (1.0 - a) / y.size))}


def versus(p: np.ndarray, y: np.ndarray, sig: int, bkg, eps_s: float) -> dict | None:
    """sig against the summed bkg classes on p_sig / (p_sig + p_bkg): AUC and the rejection at eps_s
    (ft_curves.working_point) with rel_stat; None unless both sides have jets."""
    from sklearn.metrics import roc_auc_score
    t, d = FC.versus(p, y, sig, FC.summed(p, y, bkg))
    if t.size == 0 or t.min() == t.max():
        return None
    wp = FC.working_point(t, d, eps_s)
    n_pass = wp["eps_b"] * wp["n_background"]
    return {"auc": float(roc_auc_score(t, d)), "eps_s": eps_s, **wp,
            "rel_stat": float(1.0 / np.sqrt(n_pass)) if n_pass > 0 else None}


def binned(p: np.ndarray, y: np.ndarray, pt: np.ndarray, edges, block) -> dict:
    """{"pt_edges", "n_outside", "inclusive": block(all jets), "bins": [block(jets of the bin)]}."""
    b = bin_index(pt, edges)
    bins = []
    for k in range(len(edges) - 1):
        m = b == k
        bins.append({"lo": edges[k], "hi": edges[k + 1], "n": int(m.sum()), **block(p[m], y[m])})
    return {"pt_edges": list(edges), "n_outside": int((b < 0).sum()), "inclusive": block(p, y), "bins": bins}


def jets_sha(y: np.ndarray, pt: np.ndarray) -> str:
    return hashlib.sha256(y.astype(np.int64).tobytes() + pt.astype(np.float64).tobytes()).hexdigest()


def same_jets(out_dir: pathlib.Path, sha: str, model: str) -> None:
    for f in sorted(out_dir.glob("*.json")):
        if json.loads(f.read_text())["jets_sha256"] != sha:
            raise SystemExit(f"FATAL: {model} is not scored on the jets of {f}")


def write(path: pathlib.Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path}", flush=True)


def provenance() -> dict:
    return {"script": "experiments/EVAL/perf_vs_pt.py",
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}


# ------------------------------------------------------------------ JetClass-II, the linear probe

def native_names() -> dict:
    with LABEL_MAP.open() as f:
        return {r["class_name"]: int(r["jet_label"]) for r in csv.DictReader(f)}


def test_jets(parquet: pathlib.Path, config: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """(labels, pT) of the file's rows that pass the data config's selection, in file order: the
    stream extract_v2.py scored."""
    import pyarrow.parquet as pq
    import yaml
    raw = yaml.safe_load(config.read_text())
    if raw.get("test_time_selection"):
        raise SystemExit(f"FATAL: {config} has a test_time_selection; the row map assumes the selection")
    t = pq.read_table(parquet, columns=["jet_label", "jet_pt", "jet_sdmass"])
    cols = {c: t.column(c).to_numpy() for c in t.column_names}
    keep = np.asarray(eval(raw["selection"], {"np": np}, dict(cols)), dtype=bool)
    return cols["jet_label"][keep].astype(np.int64), cols["jet_pt"][keep].astype(np.float64)


def jc2_inputs(model: str, probs_root: pathlib.Path, extract_root: pathlib.Path, parquet: pathlib.Path,
               cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(probabilities, labels, pT, sources) of one model, aligned or FATAL."""
    npz = probs_root / model / CKPT / PROBE_FILE
    ext = extract_root / model / "jc2bal" / "test" / CKPT
    for f in (npz, ext / "manifest.json", ext / "label188.npy"):
        if not f.exists():
            raise SystemExit(f"FATAL: no {f}")
    man = json.loads((ext / "manifest.json").read_text())
    cfg = REPO / man["data_config"]
    if sha256(cfg) != man["data_config_sha256"]:
        raise SystemExit(f"FATAL: {cfg} is not the data config {ext} was extracted with")
    if cfg not in cache:
        cache[cfg] = test_jets(parquet, cfg)
    if parquet not in cache:
        cache[parquet] = sha256(parquet)
    y_file, pt = cache[cfg]
    z = np.load(npz)
    rows, y, p = z["rows"].astype(np.int64), z["y"].astype(np.int64), z["p"].astype(np.float32)
    y_ext = np.load(ext / "label188.npy").astype(np.int64)
    if not (np.array_equal(rows, np.arange(y.size)) and man["n_stream"] == y.size == y_file.size == p.shape[0]
            and np.array_equal(y, y_ext) and np.array_equal(y, y_file)):
        raise SystemExit(f"FATAL: {npz}, {ext} and {parquet} are not the same jets in the same order")
    return p, y, pt, {str(npz): sha256(npz), str(ext / "label188.npy"): sha256(ext / "label188.npy"),
                      str(parquet): cache[parquet]}


def jc2_block(bb: int, cc: int):
    def block(p, y):
        auc, k = macro_auc(p, y)
        return {**accuracy(p, y), "macro_auc_ovr": auc, "n_classes_present": k,
                "class_counts": np.bincount(y, minlength=p.shape[1]).tolist(),
                "bb_vs_cc": versus(p, y, bb, (cc,), 0.5),
                "n_bb": int((y == bb).sum()), "n_cc": int((y == cc).sum())}
    return block


def run_jc2(a) -> None:
    names = native_names()
    bb, cc = (names[n] for n in BB_CC)
    cache = {}
    for model in a.models or paper_models():
        out = a.out_dir / f"{model}.json"
        if out.exists():
            print(f"{out} exists", flush=True)
            continue
        p, y, pt, src = jc2_inputs(model, a.probs_root, a.extract_root, a.test_parquet, cache)
        sha = jets_sha(y, pt)
        same_jets(a.out_dir, sha, model)
        doc = {"task": "jc2", "model": model, "checkpoint": CKPT, "probe": "linear, balanced 1e6 (probe_bal.py)",
               "test_set": "class-balanced JetClass-II test set", "pt": "reconstructed jet pT, GeV",
               "bb_vs_cc": {"signal": BB_CC[0], "background": BB_CC[1], "discriminant": "p_bb / (p_bb + p_cc)"},
               "n_jets": int(y.size), "jets_sha256": sha, "sources": src, **provenance(),
               **binned(p, y, pt, PT_EDGES["jc2"], jc2_block(bb, cc))}
        write(out, doc)


# ------------------------------------------------------------------ JetClass, fine-tuned

def jc1_cells(tree: pathlib.Path, n: int) -> list[tuple[str, pathlib.Path]]:
    """[(model, cell)] of every DONE cell at N=n: mtx-<init>, or scratch-s<S> from scratch."""
    out = []
    for cell in sorted(tree.glob(f"*/N{n}/s*")):
        init = cell.parent.parent.name
        if not cell.is_dir() or ".partial." in cell.name or cell.name.endswith(".lock"):
            continue
        if not (cell / "DONE").exists():
            print(f"  not finished, skipped: {cell}", flush=True)
            continue
        if FC.BM.diverged(cell):
            raise SystemExit(f"FATAL: {cell} trained to a NaN loss; not a result")
        out.append((f"{init}-{cell.name}" if init.startswith("scratch") else f"mtx-{init}", cell))
    models = [m for m, _ in out]
    if len(set(models)) != len(models):
        raise SystemExit(f"FATAL: more than one fine-tuning of one pretraining run at N{n} under {tree}")
    return out


def predict_workers(log: pathlib.Path) -> list[list[str]]:
    """Each DataLoader worker's test files, in order, from weaver's own 'Restarted DataIter
    test_<group>_worker<w>, ..., file_list:' records in the cell's predict.log."""
    found = {}
    for m in re.finditer(r"Restarted DataIter \S*_worker(\d+), load_range=[^\n]*file_list:\n(\{.*?\n\})",
                         log.read_text(errors="replace"), re.S):
        files = sum(json.loads(m.group(2)).values(), [])
        if found.setdefault(int(m.group(1)), files) != files:
            raise SystemExit(f"FATAL: {log}: worker {m.group(1)} is listed with two file lists")
    if not found or sorted(found) != list(range(len(found))):
        raise SystemExit(f"FATAL: {log} does not record each worker's test files; the row order is unknown")
    return [found[w] for w in range(len(found))]


def interleave(lengths, batch: int) -> list[np.ndarray]:
    """Per worker, the stream positions of its jets: torch's DataLoader takes one batch from each
    worker in turn (worker 0 first) and skips a worker once it is exhausted."""
    pos = [np.empty(n, dtype=np.int64) for n in lengths]
    nxt = 0
    for k in range(max(-(-n // batch) for n in lengths)):
        for w, n in enumerate(lengths):
            m = max(0, min(batch, n - k * batch))
            pos[w][k * batch:k * batch + m] = np.arange(nxt, nxt + m)
            nxt += m
    return pos


def read_test_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(labels in JC1_CLASSES order, jet pT) of one JetClass file, every jet in file order (the
    data config's selection is empty)."""
    import uproot
    with uproot.open(path) as f:
        trees = sorted({k.split(";")[0] for k, v in f.items() if getattr(v, "classname", "") == "TTree"})
        if len(trees) != 1:
            raise SystemExit(f"FATAL: {path} holds trees {trees}, not one")
        arr = f[trees[0]].arrays(list(LP.JC1_CLASSES) + ["jet_pt"], library="np")
    onehot = np.stack([arr[c] for c in LP.JC1_CLASSES], axis=1).astype(np.int64)
    if not np.all(onehot.sum(1) == 1):
        raise SystemExit(f"FATAL: {path} labels are not one-hot")
    return onehot.argmax(1), arr["jet_pt"].astype(np.float64)


def weaver_stream(workers: list[list[str]], batch: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(labels, pT, {file: sha256}) of the test jets in weaver's predict order: each worker reads its
    files in order, every jet (no shuffling at test time), and the loader interleaves their batches."""
    per = []
    for files in workers:
        got = [read_test_file(f) for f in files]
        per.append((np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])))
    n = sum(y.size for y, _ in per)
    y, pt = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.float64)
    for (wy, wpt), pos in zip(per, interleave([wy.size for wy, _ in per], batch)):
        y[pos], pt[pos] = wy, wpt
    return y, pt, {f: sha256(pathlib.Path(f)) for files in workers for f in files}


def jc1_inputs(cell: pathlib.Path, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(probabilities in JC1_CLASSES order, labels, pT, sources) of one cell. pred.root keeps no
    observers (experiments/FT/slim_outputs.py pred), so pT comes from the test files in the order
    predict.log records; the labels so rebuilt must equal pred.root's, jet for jet, or FATAL."""
    _, y, p, names = FC.cell_scores(cell, "jc1")
    if list(names) != list(LP.JC1_CLASSES):
        raise SystemExit(f"FATAL: {cell}/pred.root columns {names} are not {list(LP.JC1_CLASSES)}")
    workers = predict_workers(cell / "predict.log")
    key = tuple(map(tuple, workers))
    if key not in cache:
        cache[key] = weaver_stream(workers, PREDICT_BATCH)
    y_file, pt, files = cache[key]
    if not np.array_equal(y, y_file):
        raise SystemExit(f"FATAL: {cell}/pred.root's labels are not the test files' in predict.log's order")
    return p, y, pt, {**{str(cell / f): sha256(cell / f) for f in ("pred.root", "predict.log")}, **files}


def jc1_block(p, y):
    auc, k = macro_auc(p, y)
    return {**accuracy(p, y), "macro_auc_ovr": auc, "n_classes_present": k,
            "class_counts": np.bincount(y, minlength=p.shape[1]).tolist(),
            "vs_qcd": {LP.JC1_CLASSES[c]: versus(p, y, c, (0,), LP.JC1_EPS_S.get(LP.JC1_CLASSES[c], 0.5))
                       for c in range(1, len(LP.JC1_CLASSES))}}


def confusion(p: np.ndarray, y: np.ndarray) -> dict:
    """Counts [true][argmax] and each row normalised to one (null for a class with no jets)."""
    k = p.shape[1]
    c = np.bincount(y * k + p.argmax(1), minlength=k * k).reshape(k, k)
    s = c.sum(1, keepdims=True)
    return {"counts": c.tolist(),
            "row_normalised": [(r / t).tolist() if t else None for r, t in zip(c, s[:, 0])]}


def run_jc1(a) -> None:
    cache = {}
    for model, cell in jc1_cells(a.tree, JC1_N):
        out, cout = a.out_dir / f"{model}.json", a.confusion_out / f"{model}.json"
        if out.exists() and cout.exists():
            print(f"{out}, {cout} exist", flush=True)
            continue
        p, y, pt, src = jc1_inputs(cell, cache)
        sha = jets_sha(y, pt)
        common = {"model": model, "cell": str(cell), "n_train": JC1_N, "n_jets": int(y.size),
                  "classes": list(LP.JC1_CLASSES), "jets_sha256": sha, "sources": src, **provenance()}
        if not out.exists():
            same_jets(a.out_dir, sha, model)
            write(out, {"task": "jc1", **common, "pt": "reconstructed jet pT, GeV",
                        "discriminant": "p_S / (p_S + p_QCD), p normalised over the ten classes",
                        **binned(p, y, pt, PT_EDGES["jc1"], jc1_block)})
        if not cout.exists():
            same_jets(a.confusion_out, sha, model)
            write(cout, {"task": "confusion_jc1", **common, "rows": "true class", "columns": "argmax of the ten outputs",
                         **confusion(p, y)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="task", required=True)
    j2 = sub.add_parser("jc2")
    j2.add_argument("--out-dir", required=True, type=pathlib.Path)
    j2.add_argument("--models", nargs="+", help="default: configs/arms/v2_paper_runs.txt")
    j2.add_argument("--probs-root", type=pathlib.Path, default=PROBS_ROOT)
    j2.add_argument("--extract-root", type=pathlib.Path, default=EXTRACT_ROOT)
    j2.add_argument("--test-parquet", type=pathlib.Path, default=TEST_PARQUET)
    j1 = sub.add_parser("jc1")
    j1.add_argument("--out-dir", required=True, type=pathlib.Path)
    j1.add_argument("--confusion-out", required=True, type=pathlib.Path)
    j1.add_argument("--tree", type=pathlib.Path, default=JC1_TREE)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.task == "jc2":
        run_jc2(a)
    else:
        a.confusion_out.mkdir(parents=True, exist_ok=True)
        run_jc1(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
