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
import math
import pathlib

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = pathlib.Path(__file__).resolve().parents[2]
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
SPLIT_SEED = 20260822
MLP_SEEDS = (0, 1, 2)
# A null on the linear probe alone is uninterpretable (D6), so a cell is only
# called "not recovered" when BOTH probes are at chance.
#
# "At chance" is a STATISTICAL statement, so the margin cannot be an absolute
# constant. chance = 1/k runs from 0.0053 (L188, k=188) to 0.25 (R3_VIS, k=4);
# a flat 0.02 is 3.8x chance at L188 and 0.08x at R3_VIS -- most permissive
# exactly on the FINER cells this module calls the actual measurement, where it
# would print "not recovered" for a probe running at 4.5x chance. The margin is
# therefore NSIGMA standard deviations of balanced accuracy under the null,
# which scales with both k and the per-class test counts. D6 does not repair
# this: two probes compared against a threshold 76 sigma above chance both read
# as null.
CHANCE_SIGMA = 5.0
# Ceiling only; early stopping ends the fit long before this on converged cells.
MLP_MAX_ITER = 1000
# Held out for the MLP's stopping rule, and removed from the linear probe's
# training set too so the two are compared at equal n.
MLP_VAL_FRACTION = 0.1


def chance_margin(k: int, n_per_class) -> float:
    """NSIGMA sd of balanced accuracy under random guessing among k groups.

    Balanced accuracy is the mean of k per-class recalls; under the null each
    recall is Binomial(n_c, 1/k)/n_c, independent across classes, so
    Var = (1/k^2) * sum_c (1/k)(1-1/k)/n_c.
    """
    p = 1.0 / k
    inv = sum(1.0 / n for n in n_per_class if n > 0)
    return CHANCE_SIGMA * (1.0 / k) * math.sqrt(p * (1.0 - p) * inv)


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
    """Linear and MLP probes on MATCHED TRAINING SIZE. Both, always -- see D6.

    THE COMPARISON IS ONLY MEANINGFUL IF BOTH PROBES SEE THE SAME AMOUNT OF
    DATA. Turning on early_stopping to make the MLP converge also makes sklearn
    carve `validation_fraction` off its training set internally, so the MLP
    trained on 90 % of the rows while the linear probe still had 100 %. That is
    a systematic handicap pointed at exactly the probe D6 relies on: the MLP is
    there to rule out "present but not linearly decodable", and a handicapped
    MLP that finds nothing extra is weaker evidence than a matched one.

    So the linear probe is fit on the same FRACTION. The held-out rows are not
    the identical rows sklearn picks -- MLPClassifier gives no way to hand it an
    external validation set -- but n matches, which is the part that moves a
    learning curve. Rows differ per seed, which averages over the split choice
    rather than privileging one.
    """
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    n_fit = int(round(len(a) * (1.0 - MLP_VAL_FRACTION)))
    rng = np.random.default_rng(1234 + seed_offset)
    keep = rng.permutation(len(a))[:n_fit]
    # multi_class= is deprecated in sklearn 1.5 and removed in 1.8; the default
    # is already multinomial for a multi-class target.
    lin = LogisticRegression(max_iter=2000, n_jobs=-1)
    lin.fit(a[keep], ytr[keep])
    out = {"linear": float(balanced_accuracy_score(yte, lin.predict(b))),
           "n_fit": int(n_fit), "n_train_available": int(len(a))}
    # D6 MAKES THE MLP MANDATORY BESIDE ANY LINEAR NULL, because a linear probe
    # only lower-bounds mutual information: a linear null cannot distinguish
    # "absent" from "present but not linearly decodable". That argument needs a
    # CONVERGED MLP. At max_iter=300 with no early stopping the first live run
    # hit the cap on every cell -- sklearn raised ConvergenceWarning throughout
    # and l162-s1b/L188 returned mlp 0.2813 BELOW linear 0.3454, which is
    # impossible for a converged strictly-more-expressive model and is an
    # optimisation failure, not a measurement.
    #
    # Nothing recorded that. An unconverged MLP silently reads as evidence of
    # absent nonlinear structure, which is the exact inference D6 exists to
    # prevent. So: early stopping on an internal validation split (the standard
    # remedy, and it bounds wall clock too), a much higher ceiling, and the
    # iteration count and convergence flag carried into the output so a reader
    # can see it rather than infer it.
    accs, iters, capped = [], [], []
    for s in MLP_SEEDS:
        m = MLPClassifier(hidden_layer_sizes=(512,), max_iter=MLP_MAX_ITER,
                          early_stopping=True, n_iter_no_change=15,
                          validation_fraction=MLP_VAL_FRACTION,
                          random_state=s + seed_offset)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(a, ytr)
            capped.append(any(issubclass(x.category, ConvergenceWarning) for x in w))
        iters.append(int(m.n_iter_))
        accs.append(float(balanced_accuracy_score(yte, m.predict(b))))
    out["mlp"] = float(np.median(accs))
    out["mlp_spread"] = float(np.max(accs) - np.min(accs))
    out["mlp_n_iter"] = iters
    out["mlp_converged"] = not any(capped)
    # An MLP below the linear probe cannot be read as "no nonlinear structure";
    # the MLP's hypothesis class contains the linear one. Surface it.
    out["mlp_below_linear"] = bool(out["mlp"] < out["linear"])
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
           "chance_sigma": CHANCE_SIGMA, "arms": {}}

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
            _, counts = np.unique(yte, return_counts=True)
            r["chance_margin"] = chance_margin(k, counts.tolist())
            r["chance_sigma"] = CHANCE_SIGMA
            # An arm's OWN rung is the control: it must be recovered.
            r["is_own_rung"] = rung == own[arm]
            # Finer than the arm's own vocabulary = the actual measurement.
            r["is_finer_than_own"] = RUNGS.index(rung) < RUNGS.index(own[arm])
            lim = r["chance"] + r["chance_margin"]
            both_at_chance = r["linear"] <= lim and r["mlp"] <= lim
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
