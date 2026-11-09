#!/usr/bin/env python3
"""The equal-stopping decision (PI 2026-11-08, operationalised 2026-11-09).

For each (task, size) of build_ft_budget_jobs.PRE_FULL_SIZES and each of the 188- and 17-class arms (three pretraining
runs each), the old cell (stopped by patience, budget_metrics.json) against the same cell run to its last epoch with the
best validation epoch (budget_prefull read-out), on the figure's own test metric: accuracy for JetClass and JetClass-II,
R30 for top and quark/gluon tagging, with the AUC alongside. A (task, size) moves when, for either arm, the change of
its three-run mean exceeds the old three-run SD (sample SD). A (task, size) whose six new cells are not all DONE is
reported as pending, never decided.

    python3 experiments/FT/equal_stopping.py --old budget_metrics.json --new budget_prefull_metrics.json --out es.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

SIZES = (("top", 1000), ("qg", 1000), ("top", 10000), ("qg", 10000), ("jc1", 10000), ("jc2b", 10000))
TREE = {"top": "leg_top", "qg": "leg_qg", "jc1": "leg2", "jc2b": "leg_jc2b"}
METRIC = {"top": ("r30", "auc"), "qg": ("r30", "auc"), "jc1": ("accuracy", "macro_auc_ovr"),
          "jc2b": ("accuracy", "macro_auc_ovr")}
ARMS = {"188 classes": "l188", "17 classes": "r16q1"}
RUNS = (1, 2, 3)


def _vals(cells: dict, task: str, n: int, arm: str, key: str):
    v = [cells.get(f"{TREE[task]}/{arm}-s{k}/N{n}/s1", {}).get(key) for k in RUNS]
    return None if None in v else v


def decide(old: dict, new: dict) -> list:
    rows = []
    for task, n in SIZES:
        met, auc = METRIC[task]
        arms, pending = {}, False
        for name, arm in ARMS.items():
            o, w = _vals(old, task, n, arm, met), _vals(new, task, n, arm, met)
            oa, wa = _vals(old, task, n, arm, auc), _vals(new, task, n, arm, auc)
            if o is None:
                raise SystemExit(f"FATAL: old cells of {arm} {task} N={n} incomplete")
            if w is None:
                pending = True
                arms[name] = {"old": o, "new": None}
                continue
            sd = statistics.stdev(o)
            delta = statistics.mean(w) - statistics.mean(o)
            arms[name] = {"metric": met, "old": o, "new": w, "old_mean": statistics.mean(o), "old_sd": sd,
                          "new_mean": statistics.mean(w), "new_sd": statistics.stdev(w), "delta": delta,
                          "moved": abs(delta) > sd, "auc_old_mean": statistics.mean(oa),
                          "auc_new_mean": statistics.mean(wa) if wa else None}
        rows.append({"task": task, "n": n, "metric": met, "arms": arms,
                     "moved": None if pending else any(a["moved"] for a in arms.values())})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=pathlib.Path)
    ap.add_argument("--new", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    rows = decide(json.loads(a.old.read_text())["cells"], json.loads(a.new.read_text())["cells"])
    for r in rows:
        for name, x in r["arms"].items():
            if x["new"] is None:
                print(f"{r['task']:4s} {r['n']:>6d} {name:11s} pending")
                continue
            print(f"{r['task']:4s} {r['n']:>6d} {name:11s} {r['metric']:8s} old {x['old_mean']:.4g} ± {x['old_sd']:.2g}  "
                  f"new {x['new_mean']:.4g} ± {x['new_sd']:.2g}  Δ {x['delta']:+.3g}  AUC {x['auc_old_mean']:.4f} -> "
                  f"{x['auc_new_mean']:.4f}  {'MOVED' if x['moved'] else ''}")
        print(f"{r['task']:4s} {r['n']:>6d} -> {'pending' if r['moved'] is None else ('RERUN' if r['moved'] else 'keep')}")
    if a.out:
        a.out.write_text(json.dumps({"script": "experiments/FT/equal_stopping.py", "rule": "moved if |new mean - old mean| "
                                     "> old sample SD over three runs, for the 188- or the 17-class arm", "rows": rows},
                                    indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
