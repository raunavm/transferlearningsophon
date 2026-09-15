#!/usr/bin/env python3
"""The 14 LR-sweep cells restated in D7's metrics. Phase 0, part A.

WHY THIS CANNOT BE READ OFF WHAT WAS STORED. The sweep's headline -- accuracy
rising with the trunk rate and then plateauing from 3e-3 to 3e-2 at ~0.930 --
is in the WRONG METRIC for D7, which makes background rejection the headline and
log(1 - AUC) the inferential quantity. weaver's "Current validation metric" is
ACCURACY, and the probe cells kept 20 epoch checkpoints each but no pred.root
and no eval_results.json, so there is no AUC anywhere to restate (item 32,
"PHASE 0, PART A: BLOCKED"). The numbers 0.893-0.930 are accuracy-scale; ParT's
top-tagging AUC is ~0.986, and reading the former as the latter would put the
whole sweep on a scale it was never measured on.

So the AUC has to be produced, not recovered: a `weaver --predict` pass over the
top test split for each cell, then this. The job that does the predict is
experiments/FT/k8s/job-ft-phase0a-raunav.yaml; this reads what it wrote.

THE METRICS ARE IMPORTED, NOT REIMPLEMENTED. `log1m_auc` and `rejection_at` come
from experiments/EVAL/probe.py, which already owns D7's conventions -- the
resolution floor on log(1 - AUC), the 1/N_bkg cap on rejection, and the Poisson
band on the surviving background count. scripts/rand_control_stats.py exists
because a metric got hand-formed once; a second copy of these two functions is
the same mistake with a different name, and they would drift silently because
both would look right.

SIGNAL IS TOP, BACKGROUND IS QCD, from configs/finetune/TopReference.yaml's
`value: [label_QCD, label_Top]`. Index order is the data config's, not ours.

Run:  python3 experiments/FT/phase0_auc.py --grid /data/results/ft/phase0a \
          --out /data/results/ft/phase0a/phase0_auc.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re

# <probe>/<arm>_lr<rate>/pred.root -- the layout job-ft-lrprobe*.yaml wrote and
# phase0_dtheta.py already parses. Same regex shape, kept independent because
# that script matches directories and this one matches a file's parent.
CELL = re.compile(r"^(?P<arm>[a-z0-9-]+)_lr(?P<lr>[0-9.e-]+)$")

REPO = pathlib.Path(__file__).resolve().parents[2]


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments/EVAL/probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SCORE, TRUTH = "score_label_Top", "label_Top"


def check_branches(have) -> None:
    """Refuse a missing branch BY NAME. Never fall back to a position.

    configs/finetune/TopReference.yaml declares `value: [label_QCD, label_Top]`,
    so Top is index 1 today. Reading "the second score branch" instead of this
    name would keep working if that order were ever reversed, and would report
    1 - AUC with no error raised anywhere -- an exactly inverted sweep table that
    still looks entirely plausible, since every value stays in [0.5, 1].
    """
    missing = [b for b in (SCORE, TRUTH) if b not in set(have)]
    if missing:
        raise SystemExit(
            f"FATAL: pred.root is missing {missing}. Present: {sorted(have)[:12]}"
            " -- the data config's label order is what names these branches, so "
            "a rename there silently changes which class is the signal.")


def row_for(y, s, probe: str, arm: str, lr: float) -> dict:
    """One cell's D7 metrics. Pure: no I/O, so the arithmetic is testable
    without a ROOT file (the installed uproot cannot round-trip one under
    numpy 2)."""
    import numpy as np
    p = _probe()
    y = np.asarray(y).astype(int)
    s = np.asarray(s)
    n_sig, n_bkg = int((y == 1).sum()), int((y == 0).sum())
    if n_sig == 0 or n_bkg == 0:
        raise SystemExit(f"FATAL: {probe}/{arm}_lr{lr} has one class only "
                         f"({n_sig} signal, {n_bkg} background)")
    l1m, censored, auc = p.log1m_auc(y, s)
    rej, eps_b, bound, n_pass, rel = p.rejection_at(y, s)
    return {
        "probe": probe, "arm": arm, "lr": lr,
        "n_test": int(y.size), "n_sig": n_sig, "n_bkg": n_bkg,
        "auc": auc,
        "log1m_auc": l1m, "log1m_auc_censored": censored,
        "rejection_at_0.50": rej, "eps_b": eps_b,
        "rejection_is_bound": bound,
        "n_bkg_pass": n_pass, "rejection_rel_stat": rel,
    }


def read_cell(pred: pathlib.Path):
    """(y, s) for the Top-vs-QCD discriminant, from one weaver pred.root."""
    import uproot
    f = uproot.open(str(pred))
    keys = [k for k in f.keys() if not k.startswith("_")]
    if not keys:
        raise SystemExit(f"FATAL: {pred} has no tree")
    t = f[keys[0]]
    check_branches(t.keys())
    return (t[TRUTH].array(library="np").astype(int),
            t[SCORE].array(library="np"))


def grid(root: pathlib.Path) -> list[dict]:
    rows = []
    for pred in sorted(root.glob("*/*_lr*/pred.root")):
        m = CELL.match(pred.parent.name)
        if not m:
            continue
        y, s = read_cell(pred)
        rows.append(row_for(y, s, pred.parent.parent.name,
                            m["arm"], float(m["lr"])))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    rows = grid(pathlib.Path(a.grid))
    if not rows:
        raise SystemExit(f"FATAL: no */*_lr*/pred.root under {a.grid}")

    print(f"{'probe':12s} {'arm':10s} {'lr':>7s} {'AUC':>8s} "
          f"{'log(1-AUC)':>11s} {'rej@0.5':>9s} {'n_bkg_pass':>11s} {'+-%':>6s}")
    for r in sorted(rows, key=lambda r: (r["arm"], r["lr"])):
        flag = ""
        if r["log1m_auc_censored"]:
            flag += " [log1m at resolution floor]"
        if r["rejection_is_bound"]:
            flag += " [rejection at 1/N_bkg cap]"
        print(f"{r['probe']:12s} {r['arm']:10s} {r['lr']:7.0e} {r['auc']:8.5f} "
              f"{r['log1m_auc']:11.3f} {r['rejection_at_0.50']:9.1f} "
              f"{r['n_bkg_pass']:11d} {100 * r['rejection_rel_stat']:5.1f}%{flag}")

    # THE OVERLAP ANCHORS ARE A FREE REPRODUCIBILITY CHECK, AND ITEM 32 ALREADY
    # LEANS ON THEM. Each probe repeated its predecessor's top rate -- 3e-4 in
    # lrprobe/lrprobe2, 3e-3 in lrprobe2/lrprobe3 -- so 18 directories carry 14
    # distinct points. On accuracy the 3e-4 anchor reproduced to five decimals
    # across two independent jobs, which is what licensed reading the 0.001-0.002
    # arm gaps at all. That licence does NOT transfer to AUC for free: rejection
    # and log(1 - AUC) read the tail of the ROC, where far fewer jets sit, so a
    # metric can be far noisier than the accuracy measured on the same run. The
    # duplicates are therefore kept and reported, never averaged away.
    anchors = {}
    for r in rows:
        anchors.setdefault((r["arm"], r["lr"]), []).append(r)
    repeated = {k: v for k, v in anchors.items() if len(v) > 1}
    if repeated:
        print("\noverlap anchors -- the same arm and rate scored in two probes:")
        for (arm, lr), v in sorted(repeated.items()):
            probes = ", ".join(x["probe"] for x in v)
            for k in ("auc", "log1m_auc", "rejection_at_0.50"):
                vals = [x[k] for x in v]
                print(f"  {arm:10s} lr {lr:7.0e} {k:18s} "
                      f"{' vs '.join(f'{x:.5f}' for x in vals)}  "
                      f"(spread {max(vals) - min(vals):+.5f})   [{probes}]")
        print("  A spread here is run-to-run variance, not a rate effect. Any arm")
        print("  gap smaller than it is unreadable and must not be quoted.")
    else:
        print("\nWARNING: no overlap anchor found. Nothing in this table measures")
        print("run-to-run variance, so no arm gap in it can be called significant.")

    # THE POINT OF THE EXERCISE: does the plateau survive the change of metric?
    # Accuracy plateaus from 3e-3; rejection and log(1 - AUC) magnify the tail of
    # the ROC and need not. Reported per arm rather than pooled, because the two
    # arms are the contrast.
    print("\nper-arm spread over the grid (max - min):")
    for arm in sorted({r["arm"] for r in rows}):
        a_rows = [r for r in rows if r["arm"] == arm]
        for k in ("auc", "log1m_auc", "rejection_at_0.50"):
            v = [r[k] for r in a_rows]
            print(f"  {arm:10s} {k:18s} {min(v):10.5f} -> {max(v):10.5f} "
                  f"(spread {max(v) - min(v):+.5f})")

    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
