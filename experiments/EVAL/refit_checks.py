#!/usr/bin/env python3
"""The PI's rules for a refitted family (2026-11-14), checked over every file of it, so that the job's last
step and the paper's reader (experiments/FIGS/mlst_results.py) apply the same test:

  edge     no selected regularisation value at either end of its grid: a linear probe's C (a record with "C"
           and its "search"), a ridge's alpha (a record with "alpha" and its "alpha_grid")
  late     no MLP best epoch in the last LATE_FRACTION of its cap (a record with "per_seed" and "max_epochs");
           if one is, the schedule is lengthened for every network and all are refitted
  cpu      with --one-cpu, one CPU model over the family (each file's "cpu_model", or its fits' records')
  missing  with --expect, every named model has its <model>.json

    python3 experiments/EVAL/refit_checks.py --family /data/.../fits_v5 --expect mtx-l188-s1 ... \\
        [--one-cpu] --out /data/results/eval/refit_v5_checks/<family>.<stamp>.json

Prints the summary, writes it to --out (new file only), and exits 1 if any rule fails.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

LATE_FRACTION = 0.1          # experiments/EVAL/atlas_resolution.py LATE_FRACTION


def records(doc, path=()):
    """(path, record) of every fitted read-out in a document."""
    if isinstance(doc, dict):
        if ("C" in doc and isinstance(doc.get("search"), dict)) or "alpha_grid" in doc or \
                ("per_seed" in doc and "max_epochs" in doc):
            yield path, doc
        for k, v in doc.items():
            yield from records(v, path + (str(k),))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from records(v, path + (str(i),))


def edge(rec: dict) -> str | None:
    """'bottom' / 'top' when the selected value is an end of its grid, else None."""
    if "C" in rec and isinstance(rec.get("search"), dict):
        grid, v = sorted(float(c) for c in rec["search"]), float(rec["C"])
    elif "alpha_grid" in rec:
        grid, v = sorted(float(a) for a in rec["alpha_grid"]), float(rec["alpha"])
    else:
        return None
    return "bottom" if v == grid[0] else "top" if v == grid[-1] else None


def late(rec: dict) -> list:
    """The seeds whose best epoch lies in the last LATE_FRACTION of the cap."""
    if "per_seed" not in rec or "max_epochs" not in rec:
        return []
    cap = rec["max_epochs"]
    return [s["seed"] for s in rec["per_seed"] if s["best_epoch"] > (1 - LATE_FRACTION) * cap]


def cpu_models(doc) -> set:
    out = set()

    def walk(d):
        if isinstance(d, dict):
            if isinstance(d.get("cpu_model"), str):
                out.add(d["cpu_model"])
            for v in d.values():
                walk(v)
        elif isinstance(d, list):
            for v in d:
                walk(v)
    walk(doc)
    return out


def check(docs: dict, expect=(), one_cpu: bool = False) -> dict:
    """{model: document} -> the summary; summary["ok"] is False if any rule fails."""
    edges, lates, cpus, n_fits = [], [], {}, 0
    for m, doc in sorted(docs.items()):
        for path, rec in records(doc):
            n_fits += 1
            if (e := edge(rec)) is not None:
                edges.append({"model": m, "cell": "/".join(path), "edge": e,
                              "value": rec.get("C", rec.get("alpha"))})
            if (s := late(rec)):
                lates.append({"model": m, "cell": "/".join(path), "seeds": s, "max_epochs": rec["max_epochs"],
                              "best_epochs": [x["best_epoch"] for x in rec["per_seed"]]})
        for c in cpu_models(doc):
            cpus.setdefault(c, []).append(m)
    missing = sorted(set(expect) - set(docs))
    ok = not edges and not lates and not missing and (not one_cpu or len(cpus) <= 1)
    return {"ok": ok, "n_models": len(docs), "n_fits": n_fits, "edges": edges, "late_best_epoch": lates,
            "cpu_models": cpus, "one_cpu_required": one_cpu, "missing": missing, "late_fraction": LATE_FRACTION}


def load(family: pathlib.Path) -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(family.glob("*.json")) if not f.name.startswith(".")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, type=pathlib.Path, nargs="+",
                    help="the family's folder(s): every <model>.json in them")
    ap.add_argument("--expect", nargs="*", default=[])
    ap.add_argument("--one-cpu", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    docs = {}
    for fam in a.family:
        docs.update({f"{fam.name}/{m}" if len(a.family) > 1 else m: d for m, d in load(fam).items()})
    expect = a.expect if len(a.family) == 1 else [f"{f.name}/{m}" for f in a.family for m in a.expect]
    s = check(docs, expect, a.one_cpu)
    text = json.dumps(s, indent=1)
    print(text)
    if a.out is not None:
        if a.out.exists():
            raise SystemExit(f"FATAL: {a.out} exists")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print("REFIT CHECKS " + ("PASS" if s["ok"] else "FAIL"), flush=True)
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
