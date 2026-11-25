#!/usr/bin/env python3
"""Table E1 (PI 2026-11-25, ledger 33 item 4): our fine-tuned JetClass rows in the published definition, the macro
one-versus-one AUC (sklearn roc_auc_score multi_class="ovo", average="macro", as ParT's Table 1 "AUC"), with the
accuracy, from the read-out of the one rule's cells at 10^6 training jets (experiments/FIGS/data/v2/budget/
fixed_metrics.json, computed by experiments/FT/budget_metrics.jc1_metrics on all 2,000,000 test jets). Mean and SD over
runs; printed at the published rows' precision (accuracy 3 decimals, AUC 4). On JetClass's equal class counts the macro
one-vs-one AUC equals the macro one-vs-rest AUC term by term (each one-vs-rest AUC is the mean of its K-1 pairwise AUCs),
so the two agree to rounding; both are kept.

    python3 experiments/STATS/table_e1_ovo.py   # writes experiments/FIGS/data/v2/stats/table_e1_ovo.json
"""
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/FIGS/data/v2/budget/fixed_metrics.json"
OUT = ROOT / "experiments/FIGS/data/v2/stats/table_e1_ovo.json"
ROWS = (("188 classes", "l188"), ("162 classes", "l162"), ("43 classes", "r42q1"), ("17 classes", "r16q1"),
        ("17 classes, random grouping", "rand2p"), ("162 classes + mass regression", "l162mass"),
        ("17 classes + mass regression", "r16q1mass"), ("trained from scratch", "scratch"))


def arm(cell: str) -> str:
    init = cell.split("/")[1]
    return "scratch" if init == "scratch" else "rand2p" if init.startswith("rand2p") else init.rsplit("-", 1)[0]


def main() -> int:
    cells = json.loads(SRC.read_text())["cells"]
    jc1 = {k: v for k, v in cells.items() if k.startswith("leg2/") and "/N1000000/" in k}
    out = {"source": str(SRC.relative_to(ROOT)), "definition": __doc__.split("\n")[0], "rows": []}
    for name, a in ROWS:
        rs = [v for k, v in sorted(jc1.items()) if arm(k) == a]
        if any(r["n_jets"] != 2_000_000 for r in rs):
            raise SystemExit(f"FATAL: {name}: a cell is not scored on the 2,000,000 JetClass test jets")
        ovo, ovr, acc = ([r[k] for r in rs] for k in ("macro_auc_ovo", "macro_auc_ovr", "accuracy"))
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else None  # noqa: E731
        out["rows"].append({"row": name, "runs": len(rs), "cells": [k for k, v in sorted(jc1.items()) if arm(k) == a],
                            "auc_ovo": statistics.fmean(ovo), "auc_ovo_sd": sd(ovo), "auc_ovr": statistics.fmean(ovr),
                            "accuracy": statistics.fmean(acc), "accuracy_sd": sd(acc),
                            "printed": {"acc": f"{statistics.fmean(acc):.3f}", "auc_ovo": f"{statistics.fmean(ovo):.4f}"}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["rows"]:
        print(f"{r['row']:32} n={r['runs']} acc {r['printed']['acc']} (sd {r['accuracy_sd']:.4f}) "
              f"AUC ovo {r['printed']['auc_ovo']} (sd {r['auc_ovo_sd']:.5f}) ovr {r['auc_ovr']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
