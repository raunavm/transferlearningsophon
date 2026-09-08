#!/usr/bin/env python3
"""The label-recovery probe (docs/DOWNSTREAM_SUITE.md Core, STATISTICS P4).

WHAT IT ADDS THAT THE BINARY PROBES DO NOT. Every other probe infers survival of
a distinction from a downstream score: b-vs-c AUC drops, therefore flavour was
lost. This asks the question directly -- given an arm's frozen 128-d features,
can the FINER label be recovered at all? A downstream score conflates "the
distinction is gone" with "the distinction is present but this particular task
does not use it well". Recovery separates them.

THE MEASUREMENT. For every arm and every rung of the contraction tree, fit a
frozen probe from the arm's features to that rung's group id and report
balanced accuracy. Read the matrix, not a single cell:

  own rung        an arm must recover its OWN rung nearly perfectly. If it does
                  not, the probe is broken or the checkpoint is, and nothing
                  else in the row means anything. This is the built-in control.
  coarser rungs   recoverable by construction -- a coarser rung is a function of
                  a finer one, so these are a floor, not a result.
  FINER rungs     the actual measurement. Can R16_Q1's features still tell
                  label_X_bb from label_X_cc, distinctions its own vocabulary
                  never named? That is "which distinctions survive compression"
                  asked directly.

D6 IS NOT OPTIONAL HERE. A linear probe lower-bounds mutual information, so a
linear null cannot distinguish "absent" from "present but not linearly
decodable". Both probes run for every cell, and the reporting code refuses to
emit a null on the linear probe alone.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = pathlib.Path(__file__).resolve().parents[2]
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
SPLIT_SEED = 20260822
MLP_SEEDS = (0, 1, 2)
# A null on the linear probe alone is uninterpretable (D6), so a cell is only
# called "not recovered" when BOTH probes are at chance.
CHANCE_MARGIN = 0.02


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rung_maps() -> dict[str, dict[int, int]]:
    with MAP.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in RUNGS:
        if r not in rows[0]:
            raise SystemExit(f"FATAL: rung {r} is not a column of {MAP}")
        out[r] = {int(x["jet_label"]): int(x[r]) for x in rows}
    return out


def fit_pair(Xtr, ytr, Xte, yte, seed_offset=0):
    """Linear and MLP probes on the same split. Both, always -- see D6."""
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    # multi_class= is deprecated in sklearn 1.5 and removed in 1.8; the default
    # is already multinomial for a multi-class target.
    lin = LogisticRegression(max_iter=2000, n_jobs=-1)
    lin.fit(a, ytr)
    out = {"linear": float(balanced_accuracy_score(yte, lin.predict(b)))}
    accs = []
    for s in MLP_SEEDS:
        m = MLPClassifier(hidden_layer_sizes=(512,), max_iter=300,
                          random_state=s + seed_offset)
        m.fit(a, ytr)
        accs.append(float(balanced_accuracy_score(yte, m.predict(b))))
    out["mlp"] = float(np.median(accs))
    out["mlp_spread"] = float(np.max(accs) - np.min(accs))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="arm=DIR ...")
    ap.add_argument("--own-rung", nargs="+", required=True,
                    help="arm=RUNG ... , the vocabulary each arm was pretrained on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200_000,
                    help="jets subsampled per cell; a 188-way fit on 2M x 128 is "
                         "not worth its wall clock and the probe saturates far below it")
    ap.add_argument("--rungs", nargs="+", default=RUNGS)
    a = ap.parse_args(argv)

    probe = _probe()
    own = dict(s.split("=", 1) for s in a.own_rung)
    arms = {}
    for s in a.features:
        name, d = s.split("=", 1)
        if name not in own:
            raise SystemExit(f"FATAL: no --own-rung entry for {name}")
        arms[name] = probe.load_arm(pathlib.Path(d))
    align = probe.check_alignment(arms)
    maps = rung_maps()

    rng = np.random.default_rng(SPLIT_SEED)
    n_all = next(iter(arms.values()))["L"].shape[0]
    take = rng.permutation(n_all)[:min(a.n, n_all)]
    cut = int(0.7 * take.size)
    tr, te = take[:cut], take[cut:]

    res = {"row_alignment_sha256": align, "n_used": int(take.size),
           "n_train": int(tr.size), "n_test": int(te.size),
           "chance_margin": CHANCE_MARGIN, "arms": {}}

    for arm, d in sorted(arms.items()):
        F, L = d["F"], d["L"]
        res["arms"][arm] = {"own_rung": own[arm], "rungs": {}}
        for rung in a.rungs:
            g = np.array([maps[rung][int(x)] for x in L])
            ytr, yte = g[tr], g[te]
            k = len(set(g.tolist()))
            if k < 2:
                res["arms"][arm]["rungs"][rung] = {"skipped": "one group"}
                continue
            r = fit_pair(F[tr], ytr, F[te], yte)
            r["n_groups"] = k
            r["chance"] = 1.0 / k        # balanced accuracy chance level
            # An arm's OWN rung is the control: it must be recovered.
            r["is_own_rung"] = rung == own[arm]
            # Finer than the arm's own vocabulary = the actual measurement.
            r["is_finer_than_own"] = RUNGS.index(rung) < RUNGS.index(own[arm])
            both_at_chance = (r["linear"] <= r["chance"] + CHANCE_MARGIN
                              and r["mlp"] <= r["chance"] + CHANCE_MARGIN)
            r["not_recovered"] = bool(both_at_chance)
            res["arms"][arm]["rungs"][rung] = r
            flag = ("  <- OWN" if r["is_own_rung"]
                    else "  <- FINER" if r["is_finer_than_own"] else "")
            print(f"  {arm:12s} {rung:8s} K={k:4d}  linear {r['linear']:.4f}  "
                  f"mlp {r['mlp']:.4f}  (chance {r['chance']:.4f})"
                  f"{'  NOT RECOVERED' if r['not_recovered'] else ''}{flag}",
                  flush=True)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_recovery.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/label_recovery.json")

    # THE BUILT-IN CONTROL. An arm that cannot recover its own vocabulary has a
    # broken checkpoint or a broken probe, and every other cell in its row is
    # then uninterpretable -- so this is checked loudly rather than left to a
    # reader of the JSON.
    bad = []
    for arm, ad in res["arms"].items():
        cell = ad["rungs"].get(ad["own_rung"], {})
        if "linear" in cell and max(cell["linear"], cell["mlp"]) < 0.5:
            bad.append(f"{arm} recovers its own rung {ad['own_rung']} at only "
                       f"{max(cell['linear'], cell['mlp']):.3f}")
    if bad:
        print("\nWARNING: an arm cannot recover its own vocabulary; its row is "
              "not interpretable:")
        for b in bad:
            print("   ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
