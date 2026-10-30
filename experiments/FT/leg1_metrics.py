#!/usr/bin/env python3
"""Leg 1 of the fine-tuning legs: the IN-DOMAIN 162-way recovery curve, and
(--num-classes 188) the 188-class JetClass-II task of leg1_188.

WHAT THIS IS AND WHY IT NEEDED WRITING. job-ft-legs produced two legs. Leg 2
fine-tunes on JetClass-I and weaver prints a "Test metric" per cell, so its
curve could be read straight out of predict.log. Leg 1 fine-tunes on
JetClass-II's own 162-way vocabulary and writes a feature cache instead --
features.npy, logits.npy, label188.npy per cell -- with no metric computed by
anything in this repo. 54 cells of finished inference sat unread. This is the
readout.

THE MEASUREMENT. For each (init, N, fine-tuning seed) cell, softmax the cached
162-way logits, map the native 188 labels down to the L162 rung, and report
accuracy and macro one-vs-rest AUC. Aggregated over fine-tuning seeds this is
the in-domain half of the transfer claim; leg 2 is the domain-shifted half.

METRICS ARE NOT REDEFINED HERE. accuracy / macro_auc_ovr / per_class_auc /
rejection-vs-QCD come from experiments/EVAL/eval_arm.py's metrics(), the same
function the E1 gate was measured with, and the rung mapping comes from
label_recovery.rung_maps(), which reads the committed
configs/labelmaps/rung_label_maps.v1.csv. Two metric definitions for one
quantity is how two papers' numbers stop being comparable.

WHY THE AUC IS COMPUTED ON A STRIDE AND THE ACCURACY IS NOT. Accuracy is an
argmax and costs nothing. Macro OvR AUC is 162 ROC curves over every row, and
eval_arm.metrics() additionally computes per-class AUC, so a full pass is
~2 x 162 sorts per cell. v1 cached all 2 M test jets; the v2 cells extract every
4th of them (500,000 jets, the linear probe's AUC jets) and keep the logits of
every 4th of those (125,000; --auc-stride 4, experiments/FT/slim_outputs.py).
The subsample is a STRIDE, `arange(0, n, step)`, not a random draw:

  - it is deterministic, so the number is reproducible without recording a seed;
  - it takes the SAME rows in every cell, so cells stay paired -- the whole
    point of the comparison is between inits at fixed N, and a per-cell random
    draw would put sampling noise inside that contrast;
  - it preserves the class mix, which taking the first n rows would not. The
    test list interleaves Res2P / Res34P / QCD by file, so a head slice is
    biased toward whichever files come first.

Pass --auc-stride 1 to compute the AUC on every row.

ROW ALIGNMENT IS VERIFIED, NOT ASSUMED. Every cell must have read out the same
test jets in the same order or the between-init comparison is not paired. The
sha256 of label188.npy is checked across cells and a mismatch is fatal, which is
the same guard probe.py's check_alignment applies.

WHAT EACH CELL TRAINED ON AND HOW (audit 2026-09-29, must-fix 6 and 14). Every
cell carries its training subset's native-class coverage -- the subset
manifest's record (make_subsets.py jc2v2), else counted from the parquet's
jet_label -- because at 1e3 jets not every class is present (v1: 161 of 188,
median 5 per class); and the steps and validation weaver actually ran, from its
own argument dump in train.log, because the v1 manifests recorded N/512 steps
per epoch (1 at 1e3) against the 19 that ran. v2 cells also carry the pretrained
checkpoint their rule selected (init_checkpoint.json). --root takes several
trees, e.g. a v2 rule's leg1/ and the from-scratch reference's.

THE 188 NATIVE CLASSES (--num-classes 188; leg1_188 in scripts/build_ft_jobs.py).
The head is the 188 native classes, so the labels are used as they are (no map,
none of the L162 checks), and each cell is scored as the linear probe scores the
jc2_188 task (experiments/EVAL/linear_probe_v2.py), on the same jets: its cells
keep the logits of all 500,000 extracted jets (--auc-stride 1), which are the
probe's AUC jets. Per cell: accuracy; each class's one-vs-rest AUC on its raw
column (eval_arm.metrics' per-class definition); the means of those over every
class present in the test jets (macro_auc_ovr), over the 161 resonant classes
(0..160) and the 27 QCD classes (161..187), and over the classes the cell's
training subset holds (macro_auc_trained; n_classes_absent_train counts the rest,
from the subset's recorded coverage); and each resonant class's rejection of ALL
QCD at eps_S = 0.5, on the discriminant p_c / (p_c + sum of the 27 QCD p), from
probe.rejection_at with its n_bkg_pass and bound flag. A single QCD index, as the
162 path uses, would here be one QCD class. ~1 CPU-min per cell. --probe-labels
names a probe test cache's label188.npy and fails unless its AUC jets' labels are
the cells', in order. Accuracy here is on these 500,000 jets: the probe's matching
key is accuracy_on_auc_subsample (its "accuracy" is on all 2M).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
L162_QCD_GROUP = 161   # asserted against the committed map at runtime
N_NATIVE, N_RESONANT = 188, 161   # docs/GROUND_TRUTH.md: 161 resonant + 27 QCD (161..187)
EPS_S = 0.5            # ParT's Rej50%, the linear probe's working point for jc2_188
PROBE_FIRST, PROBE_STRIDE = 2_000_000, 4   # the probe's AUC jets: every 4th of the first 2M test jets


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FT_V2 = _load("ft_v2", "experiments/FT/ft_v2.py")
SLIM = _load("slim_outputs", "experiments/FT/slim_outputs.py")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def discover(root: pathlib.Path, inits: set | None = None) -> list[tuple[str, str, str, pathlib.Path]]:
    """(init, N, seed, features_dir) for every complete cell, sorted; only the init
    directories in inits when it is given.

    A `.partial.<timestamp>` directory is an interrupted attempt that a later
    attempt superseded; ft-legs leaves them beside the finished cell on purpose.
    They are skipped by name rather than by checking for files, because a
    partial cell can contain a complete-looking features.npy from before the
    interruption.
    """
    out = []
    for init_dir in sorted(p for p in root.iterdir() if p.is_dir() and (inits is None or p.name in inits)):
        for n_dir in sorted(p for p in init_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in n_dir.iterdir() if p.is_dir()):
                if ".partial." in seed_dir.name:
                    continue
                fd = seed_dir / "features_v2"
                # the full cache, or the v2 slim one (experiments/FT/slim_outputs.py leg1)
                cached = (fd / "logits.npy").exists() or (fd / "slim.json").exists()
                if cached and (fd / "label188.npy").exists():
                    if diverged(seed_dir):
                        raise SystemExit(f"FATAL: {seed_dir} trained to a NaN loss; not a result")
                    out.append((init_dir.name, n_dir.name, seed_dir.name, fd))
    return out


def diverged(cell: pathlib.Path) -> bool:
    """A cell whose training loss went to NaN at any epoch, read off weaver's
    train.log. Such a run still ends with a best-epoch checkpoint and, before the
    retry logic of 2026-09-29, could still be marked DONE; it is not a result."""
    log = cell / "train.log"
    return log.exists() and "AvgLoss: nan" in log.read_text(errors="replace")


def subset_coverage(subset: str, cache: dict) -> dict:
    """Native-class coverage of a training subset: its manifest's record, else
    counted from the parquet (the v1 subsets were built before it was recorded)."""
    if subset not in cache:
        p = pathlib.Path(subset)
        man = p.parent / "manifest.json"
        cov = (json.loads(man.read_text()).get("class_coverage", {}).get(p.name)
               if man.exists() else None)
        src = str(man)
        if cov is None:
            if not p.exists():
                raise SystemExit(f"FATAL: {p} is not readable and no manifest records its coverage")
            import pyarrow.parquet as pq
            ms = _load("make_subsets", "experiments/FT/make_subsets.py")
            cov = ms.class_coverage(pq.read_table(p, columns=["jet_label"])
                                    .column("jet_label").to_numpy())
            src = str(p)
        cache[subset] = {**{k: v for k, v in cov.items() if k != "counts"}, "source": src}
    return cache[subset]


def cell_record(cell: pathlib.Path, cov_cache: dict) -> dict:
    """What the cell trained on and how: subset, its coverage, weaver's own run
    arguments, and (v2) the pretrained checkpoint its rule selected."""
    man = cell / "ft_manifest.json"
    if not man.exists():
        raise SystemExit(f"FATAL: {cell} has no ft_manifest.json; its training set is unknown")
    m = json.loads(man.read_text())
    if not m.get("subset"):
        raise SystemExit(f"FATAL: {man} records no training subset")
    run = FT_V2.weaver_args(cell / "train.log") if (cell / "train.log").exists() else {}
    rec = {"train_subset": m["subset"], "train_coverage": subset_coverage(m["subset"], cov_cache),
           "run": run}
    if "steps_per_epoch" in m and "steps_per_epoch" in run:
        rec["manifest_steps_per_epoch_wrong"] = int(m["steps_per_epoch"]) != int(run["steps_per_epoch"])
    ic = cell / "init_checkpoint.json"
    if ic.exists():
        rec["init_checkpoint"] = json.loads(ic.read_text())
    return rec


def cell_metrics(fd: pathlib.Path, l162: dict[int, int], eval_arm, auc_stride: int):
    lab188 = np.load(fd / "label188.npy")
    sha = hashlib.sha256(lab188.tobytes()).hexdigest()
    top, logits = SLIM.read_leg1(fd, auc_stride)      # every row's argmax, every stride-th row's logits
    if top.shape[0] != lab188.shape[0] or logits.shape[0] != len(range(0, lab188.shape[0], auc_stride)):
        raise SystemExit(f"FATAL: {fd} has {top.shape[0]} rows ({logits.shape[0]} at the AUC "
                         f"stride) and {lab188.shape[0]} labels")
    if logits.shape[1] != 162:
        raise SystemExit(f"FATAL: {fd} head is {logits.shape[1]}-wide, not 162. "
                         "Leg 1 fine-tunes the 162-way vocabulary; a different "
                         "width means this cell is not what it claims to be.")
    # Size the table to cover BOTH the map and what the file actually holds.
    # Sizing it to the map alone makes an out-of-range label a raw IndexError
    # from numpy rather than the named failure below -- which is the same class
    # of defect as letting it through, since neither tells the reader that the
    # cache and the committed map disagree.
    if lab188.min() < 0:
        raise SystemExit(f"FATAL: {fd} holds negative native labels "
                         f"(min {int(lab188.min())})")
    lut = np.full(max(int(max(l162)), int(lab188.max())) + 1, -1, dtype=np.int64)
    for k, v in l162.items():
        lut[k] = v
    truth = lut[lab188]
    if (truth < 0).any():
        bad = np.unique(lab188[truth < 0])[:5]
        raise SystemExit(f"FATAL: {fd} holds native labels absent from the "
                         f"committed map: {bad.tolist()}")

    acc = float((top == truth).mean())                   # full sample, cheap
    idx = np.arange(0, truth.shape[0], auc_stride)
    m = eval_arm.metrics(softmax(logits), truth[idx], 162, L162_QCD_GROUP)
    return {
        "accuracy": acc,
        "accuracy_on_auc_subsample": m["accuracy"],
        "macro_auc_ovr": m["macro_auc_ovr"],
        "n_classes_present": m["n_classes_present"],
        "n_jets": int(truth.shape[0]),
        "n_jets_auc": int(idx.size),
        "auc_stride": auc_stride,
        "label188_sha256": sha,
    }


def cell_metrics_188(fd: pathlib.Path, probe, auc_stride: int, coverage: dict) -> dict:
    """One leg1_188 cell on the native labels (the docstring's 188 section); coverage is
    its training subset's record (subset_coverage), whose classes_absent it reads."""
    from sklearn.metrics import roc_auc_score
    lab188 = np.load(fd / "label188.npy")
    sha = hashlib.sha256(lab188.tobytes()).hexdigest()
    lab = lab188.astype(np.int64)
    top, logits = SLIM.read_leg1(fd, auc_stride)
    idx = np.arange(0, lab.shape[0], auc_stride)
    if top.shape[0] != lab.shape[0] or logits.shape[0] != idx.size:
        raise SystemExit(f"FATAL: {fd} has {top.shape[0]} rows ({logits.shape[0]} at the AUC "
                         f"stride) and {lab.shape[0]} labels")
    if logits.shape[1] != N_NATIVE:
        raise SystemExit(f"FATAL: {fd} head is {logits.shape[1]}-wide, not {N_NATIVE}: not a leg1_188 cell")
    if lab.min() < 0 or lab.max() >= N_NATIVE:
        raise SystemExit(f"FATAL: {fd} holds labels outside 0..{N_NATIVE - 1}; they are not native")
    if "classes_absent" not in coverage:
        raise SystemExit(f"FATAL: the coverage record of {coverage.get('source')} lists no classes_absent")
    p, y = softmax(logits), lab[idx]
    auc = {int(c): float(roc_auc_score((y == c).astype(int), p[:, c])) for c in np.unique(y)}
    is_qcd, p_qcd = y >= N_RESONANT, p[:, N_RESONANT:].sum(axis=1)
    rej = {}
    for c in (c for c in auc if c < N_RESONANT):
        m = (y == c) | is_qcd
        # eval_arm's tie: a jet with both probabilities rounded to 0 prefers neither side
        num, den = p[m, c], p[m, c] + p_qcd[m]
        d = np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)
        r, eps_b, bound, n_pass, _ = probe.rejection_at((y[m] == c).astype(int), d, EPS_S)
        rej[c] = {"rejection": float(r), "eps_b": float(eps_b), "n_bkg_pass": int(n_pass),
                  "is_bound": bool(bound)}
    absent = set(coverage["classes_absent"])

    def mean(cs):
        return float(np.mean([auc[c] for c in cs])) if cs else None
    return {
        "accuracy": float((top == lab).mean()),
        "accuracy_on_auc_subsample": float((p.argmax(1) == y).mean()),
        "macro_auc_ovr": mean(list(auc)),
        "macro_auc_resonant": mean([c for c in auc if c < N_RESONANT]),
        "macro_auc_qcd": mean([c for c in auc if c >= N_RESONANT]),
        "macro_auc_trained": mean([c for c in auc if c not in absent]),
        "n_classes_absent_train": sum(c in absent for c in auc),
        "n_classes_present": len(auc),
        "per_class_auc": auc,
        "rejection_vs_qcd": rej,
        "eps_s": EPS_S,
        "n_jets": int(lab.shape[0]),
        "n_jets_auc": int(idx.size),
        "auc_stride": auc_stride,
        "label188_sha256": sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path, nargs="+")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--auc-stride", type=int, default=4,
                    help="compute the AUC on every k-th row (1 = every row)")
    ap.add_argument("--expect-cells", metavar="FILE:PREFIX",
                    help="fail unless every cell FILE lists under PREFIX/leg1 is read (ft_v2.expected_cells)")
    ap.add_argument("--inits", nargs="+", metavar="INIT",
                    help="read only these init directories under --root (a read-out of the runs finished so far); default every one")
    ap.add_argument("--num-classes", type=int, choices=(162, N_NATIVE), default=162,
                    help="162: the L162 vocabulary; 188: the native classes (leg1_188, the docstring's 188 section)")
    ap.add_argument("--probe-labels", type=pathlib.Path, metavar="LABEL188_NPY",
                    help="188 only: a linear-probe test cache's label188.npy; fail unless its AUC jets are the cells'")
    args = ap.parse_args()
    if args.probe_labels and args.num_classes != N_NATIVE:
        ap.error("--probe-labels needs --num-classes 188")

    if args.num_classes == N_NATIVE:
        probe = _load("probe", "experiments/EVAL/probe.py")
    else:
        lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
        eval_arm = _load("eval_arm", "experiments/EVAL/eval_arm.py")
        l162 = lr.rung_maps()["L162"]
        groups = sorted(set(l162.values()))
        if groups != list(range(162)):
            raise SystemExit(f"FATAL: L162 has {len(groups)} groups, expected 162")
        qcd_members = [k for k, v in l162.items() if v == L162_QCD_GROUP]
        if len(qcd_members) != 27:
            raise SystemExit(f"FATAL: L162 group {L162_QCD_GROUP} has "
                             f"{len(qcd_members)} members, expected the 27 QCD "
                             "classes -- the QCD group id moved")

    cells, seen = [], {}
    for root in args.root:
        for c in discover(root, set(args.inits) if args.inits else None):
            if seen.setdefault(c[0], root) != root:
                raise SystemExit(f"FATAL: init {c[0]} is under both {seen[c[0]]} and {root}")
            cells.append(c)
    if not cells:
        raise SystemExit(f"FATAL: no complete cells under {args.root}")
    print(f"{len(cells)} cells", flush=True)
    FT_V2.require_cells({f"{i}/{n}/{s}" for i, n, s, _ in cells}, args.expect_cells, "leg1")

    res, shas, cov_cache = {}, {}, {}
    for init, n, seed, fd in cells:
        if args.num_classes == N_NATIVE:
            rec = cell_record(fd.parent, cov_cache)
            m = {**cell_metrics_188(fd, probe, args.auc_stride, rec["train_coverage"]), **rec}
        else:
            m = cell_metrics(fd, l162, eval_arm, args.auc_stride)
            m.update(cell_record(fd.parent, cov_cache))
        shas.setdefault(m["label188_sha256"], []).append(f"{init}/{n}/{seed}")
        res.setdefault(init, {}).setdefault(n, {})[seed] = m
        print(f"  {init:16} {n:10} {seed:4} acc={m['accuracy']:.5f} "
              f"macroAUC={m['macro_auc_ovr']:.5f}", flush=True)

    if len(shas) != 1:
        for s, where in shas.items():
            print(f"  {s[:16]}  {len(where)} cells, e.g. {where[:3]}")
        raise SystemExit("FATAL: cells did not read out the same test jets in "
                         "the same order; the between-init contrast is not "
                         "paired and the table would be meaningless")
    if args.probe_labels:
        y = np.load(args.probe_labels)[:PROBE_FIRST:PROBE_STRIDE].astype(np.int64)
        if not np.array_equal(y, np.load(cells[0][3] / "label188.npy").astype(np.int64)):
            raise SystemExit(f"FATAL: the cells' test jets are not the linear probe's AUC jets "
                             f"(every {PROBE_STRIDE}th of the first {PROBE_FIRST} in {args.probe_labels})")

    summary = {}
    for init, per_n in res.items():
        for n, per_seed in per_n.items():
            a = np.array([v["accuracy"] for v in per_seed.values()])
            u = np.array([v["macro_auc_ovr"] for v in per_seed.values()])
            summary.setdefault(init, {})[n] = {
                "n_seeds": int(a.size),
                "accuracy_mean": float(a.mean()),
                "accuracy_sd": float(a.std(ddof=1)) if a.size > 1 else None,
                "macro_auc_mean": float(u.mean()),
                "macro_auc_sd": float(u.std(ddof=1)) if u.size > 1 else None,
                # one entry per distinct training subset behind the seeds
                "train_coverage": {v["train_subset"]: v["train_coverage"]
                                   for v in per_seed.values()},
            }
            if args.num_classes == N_NATIVE:
                for k in ("macro_auc_resonant", "macro_auc_qcd", "macro_auc_trained"):
                    x = np.array([v[k] for v in per_seed.values()])
                    summary[init][n][f"{k}_mean"] = float(x.mean())
                    summary[init][n][f"{k}_sd"] = float(x.std(ddof=1)) if x.size > 1 else None

    args.out.mkdir(parents=True, exist_ok=True)
    extra = {"num_classes": N_NATIVE} if args.num_classes == N_NATIVE else {}
    if args.probe_labels:
        extra["probe_labels"] = str(args.probe_labels)
    (args.out / "leg1_metrics.json").write_text(json.dumps(
        {"row_alignment_sha256": next(iter(shas)), "roots": [str(r) for r in args.root],
         "auc_stride": args.auc_stride, **extra,
         "cells": res, "summary": summary}, indent=1))
    print(f"\nwrote {args.out / 'leg1_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
