#!/usr/bin/env python3
"""Top-tagging metrics of a ParT reproduction run from weaver's pred.root (plan item 2.4).

ParT's TopLandscape config labels a jet [jet_isTop, jet_isQCD], so weaver writes the branches
jet_isTop and score_jet_isTop. Reported as ParT's Table 5 reports them: accuracy (argmax), AUC,
and the background rejection 1/eps_B at 50% and 30% signal efficiency, with the number of test
jets and the background jets passing each working point.

    python3 experiments/FT/partrepro_metrics.py pred.root --out metrics.json
"""
import argparse
import json
import sys

import numpy as np


def rejection(y: np.ndarray, s: np.ndarray, eps_s: float) -> tuple:
    """1/eps_B at the threshold that keeps eps_s of the signal (ParT's Rej_X%)."""
    thr = np.quantile(s[y == 1], 1.0 - eps_s)
    n_pass = int((s[y == 0] >= thr).sum())
    eps_b = n_pass / int((y == 0).sum())
    return (1.0 / eps_b if eps_b > 0 else float("inf")), n_pass


def metrics(y: np.ndarray, s_top: np.ndarray, s_qcd: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    score = s_top / np.clip(s_top + s_qcd, 1e-12, None)
    out = {"n_jets": int(y.size), "n_signal": int(y.sum()),
           "accuracy": float(((s_top > s_qcd).astype(int) == y).mean()),
           "auc": float(roc_auc_score(y, score))}
    for name, eps in (("r50", 0.5), ("r30", 0.3)):
        out[name], out[f"{name}_n_bkg_pass"] = rejection(y, score, eps)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import uproot
    t = uproot.open(a.pred)["Events"]
    y = t["jet_isTop"].array(library="np").astype(int)
    m = metrics(y, t["score_jet_isTop"].array(library="np"), t["score_jet_isQCD"].array(library="np"))
    m["pred"] = a.pred
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
