#!/usr/bin/env python3
"""Leg 2 of the fine-tuning legs: the DOMAIN-SHIFT curve, as a real artifact.

WHY THIS EXISTS WHEN LEG 2 ALREADY HAS NUMBERS. weaver prints a "Test metric"
line into each leg-2 cell's predict.log, so the leg-2 table could be, and first
was, read by eye off 54 logs. docs/REVIEW.md requires that any claim "X equals N"
be reproducible by a committed script, and a number transcribed from a log is not
-- there is no record of which cells were included, which were skipped, or
whether a `.partial` directory was counted twice. This turns the same logs into
leg2_metrics.json with the same shape leg1_metrics.py emits, so the two legs can
be plotted and compared by one reader.

IT PARSES, SO IT IS STRICT ABOUT PARSING. A log that yields no metric is an
ERROR, not a cell quietly missing from the mean: a silently dropped cell shrinks
a denominator and moves a published average with nothing to show for it. Cells
whose directory name carries `.partial.` are interrupted attempts superseded by a
finished cell and ARE skipped -- by name, since a partial directory can hold a
complete-looking log from before the interruption.

MACRO AUC (--macro-auc, added 2026-09-27). PRESPEC 2.3 puts JetClass inference
on log(1 - macro AUC), and weaver's log prints accuracy only. Every cell's
pred.root holds all 2,000,000 test jets' ten class scores and one-hot labels, so
the macro one-vs-rest AUC is computed from it with the SAME function and the
SAME row stride as leg 1 (experiments/EVAL/eval_arm.metrics, stride 4), and the
accuracy recomputed from pred.root must equal the log's to 5e-4 or the file is
not the pass the log describes. The label sha256 must agree across cells.

REUSED CELLS AND TRUE BOOKKEEPING (audit 2026-09-29). v2 reuses v1's complete
from-scratch cells, whose subsets are unchanged: --ref-init <tree>/<init> adds
one such init directory, and each of its cells must have trained on a subset in
the sha256 record (--sha-table) with the size it recorded when it ran. Every
cell carries weaver's own run arguments from train.log (the v1 manifests wrote
steps_per_epoch as N/512) and, for v2, the pretrained checkpoint its rule chose.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[2]
ACC_TOL = 5e-4


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FT_V2 = _load("ft_v2", "experiments/FT/ft_v2.py")

METRIC = re.compile(r"Test metric ([0-9.]+)")


def cell_metric(log: pathlib.Path) -> float:
    """The LAST Test metric in the log. weaver prints one per predict pass, and a
    retried pass appends rather than truncating, so the last is the live one."""
    hits = METRIC.findall(log.read_text(errors="replace"))
    if not hits:
        raise SystemExit(f"FATAL: no 'Test metric' in {log}. A cell that cannot "
                         "be parsed must not be silently dropped from a mean.")
    return float(hits[-1])


def read_pred_root(path: pathlib.Path):
    """(class names in file order, one-hot labels, scores) from weaver's pred.root."""
    import numpy as np
    import uproot
    t = uproot.open(path)["Events"]
    names = [k[len("label_"):] for k in t.keys() if k.startswith("label_")]
    if not names or any(f"score_label_{n}" not in t.keys() for n in names):
        raise SystemExit(f"FATAL: {path} lacks a score_label_ branch for every label_ branch")
    arr = t.arrays([f"label_{n}" for n in names] + [f"score_label_{n}" for n in names], library="np")
    return (names, np.stack([arr[f"label_{n}"] for n in names], axis=1),
            np.stack([arr[f"score_label_{n}"] for n in names], axis=1))


def pred_root_metrics(path: pathlib.Path, eval_arm, stride: int) -> dict:
    """Accuracy and macro OvR AUC from one cell's pred.root (see the docstring)."""
    import numpy as np
    names, onehot, scores = read_pred_root(path)
    onehot = onehot.astype(np.float64)
    if not np.all(onehot.sum(axis=1) == 1):
        raise SystemExit(f"FATAL: {path} labels are not one-hot")
    truth = onehot.argmax(1)
    probs = scores.astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)
    idx = np.arange(0, truth.shape[0], stride)
    m = eval_arm.metrics(probs[idx], truth[idx], len(names), names.index("QCD") if "QCD" in names else 0)
    return {"accuracy_pred_root": float((probs.argmax(1) == truth).mean()),
            "macro_auc_ovr": m["macro_auc_ovr"], "n_classes_present": m["n_classes_present"],
            "classes": names, "n_jets": int(truth.shape[0]), "n_jets_auc": int(idx.size),
            "auc_stride": stride, "label_sha256": hashlib.sha256(truth.astype(np.int64).tobytes()).hexdigest()}


def discover(root: pathlib.Path):
    """(init, N, seed, predict.log). Reference runs sit one level shallower."""
    out = []
    for log in sorted(root.rglob("predict.log")):
        rel = log.relative_to(root).parts[:-1]
        if any(".partial." in p for p in rel):
            continue
        if len(rel) == 3:
            if diverged(log.parent):
                raise SystemExit(f"FATAL: {log.parent} trained to a NaN loss; not a result")
            out.append((rel[0], rel[1], rel[2], log))
        elif len(rel) == 1:
            out.append((rel[0], "ref", "s1", log))
    return out


def ref_cells(init_dir: pathlib.Path, table: dict):
    """(init, N, seed, predict.log) of one reused init directory, each checked
    against the sha256 record; .partial.* attempts skipped as in discover()."""
    out = []
    for log in sorted(init_dir.glob("N*/s*/predict.log")):
        if ".partial." in log.parent.name:
            continue
        if diverged(log.parent):
            raise SystemExit(f"FATAL: {log.parent} trained to a NaN loss; not a result")
        bad = FT_V2.ref_cell_problem(log.parent, table)
        if bad:
            raise SystemExit(f"FATAL: reused cell refused: {bad}")
        out.append((init_dir.name, log.parent.parent.name, log.parent.name, log))
    if not out:
        raise SystemExit(f"FATAL: no cells under {init_dir}")
    return out


def run_record(cell: pathlib.Path) -> dict:
    """weaver's own run arguments and (v2) the pretrained checkpoint chosen."""
    rec = {}
    if (cell / "train.log").exists():
        rec["run"] = FT_V2.weaver_args(cell / "train.log")
    if (cell / "init_checkpoint.json").exists():
        rec["init_checkpoint"] = json.loads((cell / "init_checkpoint.json").read_text())
    return rec


def diverged(cell: pathlib.Path) -> bool:
    """A cell whose training loss went to NaN at any epoch, read off weaver's
    train.log. Such a run still ends with a best-epoch checkpoint and, before the
    retry logic of 2026-09-29, could still be marked DONE; it is not a result."""
    log = cell / "train.log"
    return log.exists() and "AvgLoss: nan" in log.read_text(errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--macro-auc", action="store_true",
                    help="also compute macro OvR AUC from each cell's pred.root (PRESPEC 2.3)")
    ap.add_argument("--auc-stride", type=int, default=4, help="as leg1_metrics.py")
    ap.add_argument("--ref-init", type=pathlib.Path, action="append", default=[],
                    help="a reused init directory (<tree>/<init>), checked against --sha-table")
    ap.add_argument("--sha-table", type=pathlib.Path,
                    help="the sha256 record of the subsets (experiments/FT/data/ft_v2_subsets_sha256.json)")
    a = ap.parse_args(argv)
    if a.ref_init and not a.sha_table:
        raise SystemExit("FATAL: --ref-init needs --sha-table: a reused cell is only as good "
                         "as the proof that its subset is unchanged")
    eval_arm = None
    if a.macro_auc:
        spec = importlib.util.spec_from_file_location("eval_arm", REPO / "experiments/EVAL/eval_arm.py")
        eval_arm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_arm)

    cells = discover(a.root)
    if not cells:
        raise SystemExit(f"FATAL: no predict.log under {a.root}")
    table = FT_V2.load_table(a.sha_table) if a.ref_init else {}
    for d in a.ref_init:
        if any(c[0] == d.name for c in cells):
            raise SystemExit(f"FATAL: init {d.name} is under both {a.root} and {d}")
        cells += ref_cells(d, table)
    res, shas = {}, {}
    for init, n, seed, log in cells:
        c = {"accuracy": cell_metric(log), **run_record(log.parent)}
        if a.macro_auc:
            m = pred_root_metrics(log.parent / "pred.root", eval_arm, a.auc_stride)
            if abs(m["accuracy_pred_root"] - c["accuracy"]) > ACC_TOL:
                raise SystemExit(f"FATAL: {log.parent}: pred.root accuracy {m['accuracy_pred_root']:.5f} "
                                 f"differs from the log's {c['accuracy']:.5f}; not the same pass")
            c.update(m)
            shas.setdefault(m["label_sha256"], []).append(f"{init}/{n}/{seed}")
        res.setdefault(init, {}).setdefault(n, {})[seed] = c
        print(f"  {init:16} {n:10} {seed:4} acc={c['accuracy']:.5f}"
              + (f" macroAUC={c['macro_auc_ovr']:.5f}" if a.macro_auc else ""), flush=True)
    if len(shas) > 1:
        raise SystemExit(f"FATAL: cells read different test jets: {[(s[:12], len(v)) for s, v in shas.items()]}")

    summary = {}
    for init, per_n in res.items():
        for n, per_seed in per_n.items():
            v = [c["accuracy"] for c in per_seed.values()]
            summary.setdefault(init, {})[n] = {
                "n_seeds": len(v),
                "accuracy_mean": st.mean(v),
                "accuracy_sd": st.stdev(v) if len(v) > 1 else None,
            }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "leg2_metrics.json").write_text(
        json.dumps({"cells": res, "summary": summary,
                    "row_alignment_sha256": next(iter(shas), None),
                    "reused_inits": [str(d) for d in a.ref_init],
                    "auc_stride": a.auc_stride if a.macro_auc else None}, indent=1))
    print(f"\n{len(cells)} cells -> {a.out / 'leg2_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
