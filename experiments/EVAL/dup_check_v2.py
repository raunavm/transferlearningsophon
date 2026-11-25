#!/usr/bin/env python3
"""Duplicate-run check of the rewrite's Tables 2 and 3 (PI 2026-11-25, LEDGER §28 1.2): behind each cell, the runs must
have different checkpoints and different per-jet scores. Reads only; nothing is fitted or trained.

For each cell (row, column) and each run behind it: the pretraining checkpoint the scores came from (path, the sha256
its manifest records and the sha256 of the file itself), for a fine-tuned cell also the sha256 of its fine-tuned best
checkpoint, the sha256 of the per-jet scores, and the cell's metric recomputed from those scores. For each pair of runs
of a cell: equal checkpoints, equal score arrays, the share of identical score values on a fixed sample of jets and the
largest difference. Accuracy is the argmax share (a probe's p, or the fine-tuned logits); a binary score's accuracy is
s > 0.5 (probe) and its Rej30% is paired_bootstrap.RejScorer's; a fine-tuned top or quark/gluon cell is scored by
bench_metrics.cell_metrics, the read-out's own scorer.

    python3 experiments/EVAL/dup_check_v2.py --out /data/results/eval/dup_check/dup_check_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
LP = pathlib.Path("/data/results/eval/v2_linprobe_bal")
SOPHON = pathlib.Path("/data/results/eval/v2_linprobe_sophon")
FT = pathlib.Path("/data/results/ft_v2")
FILL = {"top": "itop4", "qg": "nqg"}                     # build_refit_v5_jobs.FILL_TAG: the substitution chosen on validation
ARMS = {"188 classes": ("l188-s1", "l188-s2", "l188-s3"), "162 classes": ("l162-s1", "l162-s2", "l162-s3"),
        "43 classes": ("r42q1-s1", "r42q1-s2", "r42q1-s3"), "17 classes": ("r16q1-s1", "r16q1-s2", "r16q1-s3"),
        "17 classes, random grouping": ("rand2p1-s1", "rand2p2-s2", "rand2p3-s1", "rand2p4-s2", "rand2p5-s2"),
        "162 classes + mass regression": ("l162mass-s1", "l162mass-s2"),
        "17 classes + mass regression": ("r16q1mass-s1", "r16q1mass-s2")}
SAMPLE = 1_000_000          # score values compared value by value between two runs (a fixed spread of positions)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha_file(p: pathlib.Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sample(a: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(a).reshape(-1)
    idx = np.linspace(0, flat.size - 1, min(SAMPLE, flat.size)).astype(np.int64)
    return flat[idx].astype(np.float64)


def probe_ckpt(run: str) -> dict:
    """The pretraining checkpoint of a run's frozen features (the JetClass-II test extraction's manifest)."""
    for man in (LP / f"mtx-{run}" / "jc2bal" / "test" / "best70" / "manifest.json",
                LP / run.replace("initbn-", "init-") / "jc2bal" / "test" / "init_bn" / "manifest.json"):
        if man.exists():
            m = json.loads(man.read_text())
            ck = m.get("checkpoint")
            return {"path": ck, "sha256_manifest": m.get("checkpoint_sha256"),
                    "sha256_file": sha_file(pathlib.Path(ck)) if ck else None, "manifest": str(man)}
    return {"path": None, "sha256_manifest": None, "sha256_file": None, "manifest": None}


def probe_runs(ds: str, run: str):
    """(npz path, the run's name) of a frozen-feature cell's stored test outputs."""
    if run == "sophon":
        base = SOPHON
        d = base / "fits_v5_probs"
    elif run.startswith("initbn-"):
        return (LP / "fits_v5_probs" / run.replace("initbn-", "init-") / "init_bn" / f"{ds}.npz") if ds.startswith("j") \
            else (LP / "fits_v5_fill_probs" / run.replace("initbn-", "init-") / f"init_bn.{FILL[ds[:3].rstrip('_')]}"
                  / f"{ds}.npz")
    else:
        d = LP / "fits_v5_probs"
    if ds.startswith("j"):
        return (d / (run if run == "sophon" else f"mtx-{run}") / "best70" / f"{ds}.npz") if run != "sophon" \
            else next((d.glob(f"*/*/{ds}.npz")), d / f"{ds}.npz")
    t = "top" if ds.startswith("top") else "qg"
    fd = (LP if run != "sophon" else SOPHON) / "fits_v5_fill_probs"
    return (fd / f"mtx-{run}" / f"best70.{FILL[t]}" / f"{ds}.npz") if run != "sophon" \
        else next(fd.glob(f"*/*.{FILL[t]}/{ds}.npz"), fd / f"{ds}.npz")


def probe_entry(pb, ds: str, run: str) -> dict:
    path = probe_runs(ds, run)
    rec = {"input": str(path)}
    if not path.exists():
        rec["missing"] = True
        return rec
    z = np.load(path)
    y = z["y"].astype(np.int64)
    s = z["p"] if "p" in z.files else z["s"]
    rec.update(scores_sha256=sha_arr(s), n=int(y.size), rows_sha256=sha_arr(z["rows"]), coef_sha256=sha_arr(z["coef"]))
    if s.ndim == 2:
        rec["accuracy"] = float((s.astype(np.float32).argmax(1) == y).mean())
    else:
        rec["accuracy"] = float(((s > 0.5) == (y == 1)).mean())
        rec["r30"] = float(pb.RejScorer(y == 1, s, 0.3)())
    rec["_sample"] = sample(s)
    return rec


def ft_cell(key: str, run: str) -> pathlib.Path:
    """A fine-tuned cell of Table 3: JetClass-II and JetClass at 1e6 at the row's recorded rate (paper_fixed_cells);
    top and quark/gluon on the full sets with ParT's recipe (ft_v2/best70; from scratch ref_v100/bench)."""
    if key in ("jc2b", "jc1"):
        b = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
        init, s = ("", int(run[-1])) if run.startswith("scratch") else (run, 1)
        return next(pathlib.Path(c["out"]) for c in b.paper_fixed_cells()
                    if c["task"] == key and c["n"] == 1_000_000 and c["init"] == init and c["s"] == s)
    leg, n = ("leg_top", 1_200_000) if key == "top" else ("leg_qg", 1_600_000)
    if run.startswith("scratch"):
        return FT / "ref_v100" / "bench" / leg / "scratch-v2" / f"N{n}" / f"s{run[-1]}"
    return FT / "best70" / leg / run / f"N{n}" / "s1"


def ft_entry(bm, probe, key: str, run: str) -> dict:
    cell = ft_cell(key, run)
    rec = {"input": str(cell), "done": (cell / "DONE").exists()}
    ic = cell / "init_checkpoint.json"
    if ic.exists():
        m = json.loads(ic.read_text())
        rec["checkpoint"] = {"path": m.get("path"), "sha256_manifest": m.get("sha256"),
                             "sha256_file": sha_file(pathlib.Path(m["path"])) if m.get("path") else None}
    rec["ft_best_sha256"] = sha_file(cell / "net_best_epoch_state.pt")
    if key == "jc1":
        import uproot
        t = uproot.open(cell / "pred.root")["Events"]
        cls = [k[len("score_"):] for k in t.keys() if k.startswith("score_")]
        s = np.stack([t[f"score_{c}"].array(library="np") for c in cls], 1).astype(np.float32)
        y = np.stack([t[c].array(library="np") for c in cls], 1).argmax(1)
        rec["accuracy"] = float((s.argmax(1) == y).mean())
    elif key == "jc2b":
        s = np.load(cell / "features" / "logits.npy")
        y = np.load(cell / "features" / "label188.npy").astype(np.int64)
        rec["accuracy"] = float((s.argmax(1) == y).mean())
    else:
        s = np.load(cell / "features" / "logits.npy")
        m = bm.cell_metrics(cell, probe)
        rec.update(accuracy=m.get("accuracy"), r30=m.get("r30"))
    rec.update(scores_sha256=sha_arr(s), n=int(s.shape[0]))
    rec["_sample"] = sample(s)
    return rec


def pairs(recs: dict) -> list:
    out = []
    for a, b in itertools.combinations(sorted(r for r in recs if "_sample" in recs[r]), 2):
        ra, rb = recs[a], recs[b]
        ca, cb = (ra.get("checkpoint") or {}).get("sha256_file"), (rb.get("checkpoint") or {}).get("sha256_file")
        out.append({"a": a, "b": b, "same_checkpoint": bool(ca and ca == cb),
                    "same_ft_checkpoint": bool(ra.get("ft_best_sha256") and ra.get("ft_best_sha256") == rb.get("ft_best_sha256")),
                    "same_scores": ra["scores_sha256"] == rb["scores_sha256"],
                    "frac_identical": float((ra["_sample"] == rb["_sample"]).mean()) if ra["_sample"].size == rb["_sample"].size else None,
                    "max_abs_diff": float(np.abs(ra["_sample"] - rb["_sample"]).max()) if ra["_sample"].size == rb["_sample"].size else None})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", default=None, help="cells 'table/row/column' to check (default: every cell)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists")
    pb = _load("paired_bootstrap", "experiments/STATS/paired_bootstrap.py")
    bm = _load("bench_metrics", "experiments/FT/bench_metrics.py")
    probe = _load("probe", "experiments/EVAL/probe.py")
    t2 = {"JetClass-II": "jc2bal_N1000000", "JetClass": "jc1_N1000000", "Top tagging": "top_N1200000",
          "Quark/gluon": "qg_N1600000"}
    t3 = {"JetClass-II": "jc2b", "JetClass": "jc1", "Top tagging": "top", "Quark/gluon": "qg"}
    rows2 = {**ARMS, "untrained network": ("initbn-s1", "initbn-s2", "initbn-s3"), "Sophon (released)": ("sophon",)}
    rows3 = {**ARMS, "trained from scratch": ("scratch-v2-s1", "scratch-v2-s2", "scratch-v2-s3")}
    ckpt = {}
    doc = {"script": "experiments/EVAL/dup_check_v2.py",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "cells": {}}
    for table, cols, rows in (("Table 2", t2, rows2), ("Table 3", t3, rows3)):
        for col, ds in cols.items():
            for row, runs in rows.items():
                name = f"{table}/{row}/{col}"
                if a.only is not None and name not in a.only:
                    continue
                recs = {}
                for r in runs:
                    if table == "Table 2":
                        rec = probe_entry(pb, ds, r)
                        if r not in ckpt and r != "sophon":
                            ckpt[r] = probe_ckpt(r)
                        rec["checkpoint"] = ckpt.get(r)
                    else:
                        rec = ft_entry(bm, probe, ds, r)
                    recs[r] = rec
                cell = {"runs": {r: {k: v for k, v in x.items() if k != "_sample"} for r, x in recs.items()},
                        "pairs": pairs(recs)}
                cell["duplicate"] = any(p["same_checkpoint"] or p["same_ft_checkpoint"] or p["same_scores"]
                                        or (p["frac_identical"] or 0) > 0.5 for p in cell["pairs"])
                doc["cells"][name] = cell
                print(name, "DUPLICATE" if cell["duplicate"] else "ok",
                      json.dumps({r: {k: x.get(k) for k in ("accuracy", "r30")} for r, x in recs.items()}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{a.out}: {len(doc['cells'])} cells, {sum(c['duplicate'] for c in doc['cells'].values())} with a duplicate pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
