#!/usr/bin/env python3
"""Frozen-representation probes: the paper's headline endpoint.

WHAT THIS MEASURES
------------------
Given the 128-d representations extracted per arm by extract_features.py, fit a
probe on FROZEN features and ask whether a physical distinction survived the
pretraining vocabulary. Two tasks, chosen because each dies at a different rung
-- which is what makes the rungs load-bearing rather than decorative:

    bvc_resonant   label_X_bb (0) vs label_X_cc (1)
                   distinct at L188 / L162 / R42_Q1, COLLAPSED at R16_Q1.
                   The resonant heavy-flavour axis, erased by construction at
                   the coarse end. This is the primary probe.

    bvc_qcd        label_QCD_bb (169) vs label_QCD_cc (181)
                   distinct ONLY at L188; L162, R42_Q1 and R16_Q1 all collapse
                   the 27 QCD classes to one. This is what converts L188 from a
                   bolt-on into a rung that carries an argument.

Both tasks are defined on the NATIVE label (0..187) and are therefore identical
for every arm. A task whose definition moved with the arm would not be a
controlled contrast.

WHY BOTH A LINEAR AND AN MLP PROBE (D6)
---------------------------------------
The linear probe is primary. But a linear probe lower-bounds mutual
information, so a linear NULL cannot distinguish "the distinction is absent"
from "it is present and not linearly decodable" -- and this study's central
claim is about absence. CLAUDE.md is categorical: never report a linear-probe
null without the nonlinear probe beside it. Both are therefore always computed
and always reported together; there is no flag to skip the MLP.

METRICS (D7)
------------
Background rejection 1/eps_B at fixed signal efficiency is the headline, being
the number a referee wants. Inference runs on log(1 - AUC), because rejection's
seed noise is ~12x the AUC metric's. Rejection beyond the resolvable cap
(1/N_bkg) is reported as a BOUND, never as a value -- past the cap the number is
an artefact of sample size.

Arm differences carry a PAIRED bootstrap: the arms are scored on identical jets
in identical order, which the row-alignment check below enforces rather than
assumes.

Usage:
    python3 experiments/EVAL/probe.py \
        --features L162=/data/results/eval/mtx-l162-s1/features \
                   R16_Q1=/data/results/eval/mtx-r16q1-s1/features \
        --out /data/results/eval/probe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# native label -> the two binary tasks. Arm-independent by construction.
# THE THREE TASKS FORM A 2x2, AND THE THIRD IS WHY THE FIRST MEANS ANYTHING.
#
# bvc_resonant on its own cannot support the paper's claim. If R16_Q1 scores
# worse there, the obvious rival explanation is that 17-way pretraining simply
# yields a weaker representation than 162-way and everything is worse. The
# claim needs a distinction R16_Q1 KEEPS, measured the same way:
#
#                     topology    flavour    R16_Q1 does
#   bvc_resonant      FIXED       differs    collapse it
#   retained_topology differs     FIXED      keep it
#   bvc_qcd           --          --         collapse it, and so does L162
#
# bvc_resonant and retained_topology are complements: each holds fixed what the
# other varies, so together they separate "this axis was erased" from "this arm
# is worse at everything". bvc_qcd is the null-vs-null leg -- both arms collapse
# it, so neither should win, and a gap there would indict the method.
#
# retained_topology holds FLAVOUR fixed (both endpoints are all-b) and varies
# prong count, which is the axis every rung of the contraction tree is built
# on. Difficulty is NOT matched to bvc_resonant and cannot be -- R16_Q1 groups
# by topology, so any distinction it retains is topological. Read the arms
# against each other within a task, never across tasks, and read it in
# log(1-AUC), which stays sensitive near the ceiling where raw AUC does not.
TASKS = {
    "bvc_resonant": {"signal": [0], "background": [1],
                     "names": ["label_X_bb", "label_X_cc"],
                     "collapsed_at": ["R16_Q1"]},
    "retained_topology": {"signal": [0], "background": [15],
                          "names": ["label_X_bb", "label_X_YY_bbbb"],
                          "collapsed_at": []},
    "bvc_qcd": {"signal": [169], "background": [181],
                "names": ["label_QCD_bb", "label_QCD_cc"],
                "collapsed_at": ["L162", "R42_Q1", "R16_Q1"]},
}

# --- the two PUBLISHED physics discriminants (docs/PRD_PLAN.md 3.1b, 8.2) ----
# These are not more probes of the same kind. The three above ask whether an
# axis survived; these ask whether an arm can still build a discriminant a
# PUBLISHED analysis depends on. That is the "use case survival" argument: a
# coarse vocabulary does not merely score worse, it cannot construct the
# quantity at all, because the nodes it would sum over no longer exist.
#
#   bc_vs_rest  arXiv:2503.00118 Eq. 1, the |V_cb| discriminant
#               D_bc = g_bc / (g_bc + g_bq + g_cs + g_bqq + g_QCD)
#               measured in that paper's window, 450 < pT < 600 and
#               90 < m_SD < 140, and reported at eps_S = 60 % / 40 %.
#               "bqq" is label_X_YY_qqb (native 70) -- the 3-prong q,q,b class.
#   ee_vs_mm    the lepton-flavour split 2606.09458's background suppression
#               uses.
#
# COLLAPSE RUNGS ARE DERIVED FROM configs/labelmaps/rung_label_maps.v1.csv, not
# asserted. Doing so corrected docs/PRD_PLAN.md 3.1(b), which says the ee/mumu
# split merges "from R42_Q1 down": it actually merges one rung EARLIER, at
# R63_Q1, into 2P_LEP_LL|nb0_nc0.


def qcd_indices() -> list[int]:
    """The native labels that are QCD, from the committed map.

    Read, never hardcoded as range(161, 188) -- the same reason
    experiments/MASSREG/e1_control.py reads it: "161 resonant + 27 QCD" is a
    fact about the map, and a hardcoded range silently survives a map change.
    """
    import csv
    path = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
    with path.open() as f:
        idx = [int(r["jet_label"]) for r in csv.DictReader(f)
               if r["class_name"].startswith("label_QCD_")]
    if not idx:
        raise SystemExit(f"FATAL: no label_QCD_* rows in {path}")
    return sorted(idx)


PHYSICS_TASKS = {
    "bc_vs_rest": {"signal": [4], "background": [6, 5, 70] + qcd_indices(),
                   "names": ["label_X_bc", "{bq,cs,bqq,QCD}"],
                   "collapsed_at": ["R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"],
                   "window": {"jet_pt": [450.0, 600.0], "jet_sdmass": [90.0, 140.0]},
                   "eps_s": [0.60, 0.40]},
    "ee_vs_mm": {"signal": [10], "background": [11],
                 "names": ["label_X_ee", "label_X_mm"],
                 "collapsed_at": ["R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1",
                                  "R3_VIS", "R1_Q1"]},
}
TASKS.update(PHYSICS_TASKS)

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EPS_S = 0.5           # signal efficiency at which rejection is quoted
MLP_SEEDS = (0, 1, 2)
SPLIT_SEED = 20260822


def load_arm(d: pathlib.Path) -> dict:
    F = np.load(d / "features.npy")
    L = np.load(d / "label188.npy")
    man = json.loads((d / "extract_manifest.json").read_text())
    if F.shape[0] != L.shape[0]:
        raise SystemExit(f"FATAL: {d} has {F.shape[0]} features and {L.shape[0]} labels")
    obs = {}
    if (d / "observers.npz").exists():
        z = np.load(d / "observers.npz")
        obs = {k: z[k] for k in z.files}
        for k, v in obs.items():
            if v.shape[0] != L.shape[0]:
                raise SystemExit(f"FATAL: {d} observer {k} has {v.shape[0]} rows, "
                                 f"not {L.shape[0]}")
    return {"F": F, "L": L, "manifest": man, "obs": obs,
            "label_sha": hashlib.sha256(L.tobytes()).hexdigest()}


def check_alignment(arms: dict[str, dict]) -> str:
    """The paired bootstrap is only valid if the arms scored the SAME jets in
    the SAME order. Verified from the label vectors, not assumed from the fact
    that the same config was used."""
    shas = {a: v["label_sha"] for a, v in arms.items()}
    if len(set(shas.values())) != 1:
        print("FATAL: arms are not row-aligned; their native-label vectors differ.",
              file=sys.stderr)
        for a, s in shas.items():
            print(f"  {a:8s} n={arms[a]['L'].shape[0]:>10,}  label188 sha {s[:16]}",
                  file=sys.stderr)
        print("  Every arm must be extracted with the SAME data config and file "
              "order. Without that, a paired comparison is comparing different "
              "jets.", file=sys.stderr)
        raise SystemExit(2)
    return next(iter(shas.values()))


def make_splits(n: int, rng_seed: int = SPLIT_SEED):
    """Deterministic, arm-independent probe train/val/test split.

    Arm-independent is the point: the same jets must land in the same split for
    every arm, or the comparison is confounded by the split.
    """
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return perm[:a], perm[a:b], perm[b:]


def rejection_at(y: np.ndarray, s: np.ndarray, eps_s: float = EPS_S):
    """(rejection, eps_B, is_bound) at fixed signal efficiency.

    Linear interpolation on the ROC between adjacent thresholds bracketing
    eps_s, per docs/STATISTICS.md. Past the resolvable cap 1/N_bkg the value is
    an artefact of sample size and is flagged as a bound.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    eps_b = float(np.interp(eps_s, tpr, fpr))
    n_bkg = int((y == 0).sum())
    cap = float(n_bkg)
    if eps_b <= 0:
        return cap, 0.0, True
    r = 1.0 / eps_b
    return (min(r, cap), eps_b, r >= cap)


def fit_linear(Xtr, ytr, Xva, yva, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_auc, best_C = None, -1.0, None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
        clf.fit(Xtr_, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Xva_))
        if auc > best_auc:
            best, best_auc, best_C = clf, auc, C
    return best.decision_function(Xte_), {"C": best_C, "val_auc": float(best_auc)}


def fit_mlp(Xtr, ytr, Xva, yva, Xte, seeds=MLP_SEEDS):
    """MLPHead([256], dropout 0.1). Mandatory beside every linear null (D6)."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    tr = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    va = torch.tensor(sc.transform(Xva), dtype=torch.float32)
    te = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    scores, meta = [], []
    for sd in seeds:
        torch.manual_seed(sd)
        net = torch.nn.Sequential(
            torch.nn.Linear(tr.shape[1], 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(256, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        best_va, best_state, patience = -1.0, None, 0
        for epoch in range(60):
            net.train()
            perm = torch.randperm(tr.shape[0])
            for i in range(0, tr.shape[0], 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                lossf(net(tr[idx]), ytr_t[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = (net(va)[:, 1] - net(va)[:, 0]).numpy()
            auc = roc_auc_score(yva, v)
            if auc > best_va:
                best_va, best_state, patience = auc, \
                    {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 8:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            scores.append((net(te)[:, 1] - net(te)[:, 0]).numpy())
        meta.append({"seed": sd, "val_auc": float(best_va)})
    return np.mean(scores, axis=0), {"seeds": meta,
                                     "val_auc_mean": float(np.mean([m["val_auc"] for m in meta])),
                                     "val_auc_std": float(np.std([m["val_auc"] for m in meta]))}


def main() -> int:
    from sklearn.metrics import roc_auc_score
    from src.stats.bootstrap import ci, paired_bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True,
                    help="ARM=/path/to/features, one per arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    arms = {}
    for spec in args.features:
        if "=" not in spec:
            raise SystemExit(f"FATAL: --features wants ARM=path, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))
    if len(arms) < 1:
        raise SystemExit("FATAL: no arms given")

    align_sha = check_alignment(arms)
    L = next(iter(arms.values()))["L"]
    print(f"{len(arms)} arms, {L.shape[0]:,} jets, row-aligned "
          f"(label188 sha {align_sha[:16]})")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"n_jets_total": int(L.shape[0]), "row_alignment_sha256": align_sha,
               "eps_s": EPS_S, "tasks": {}}

    for task in args.tasks:
        spec = TASKS[task]
        sig = np.isin(L, spec["signal"])
        bkg = np.isin(L, spec["background"])
        keep = sig | bkg
        # A task may be defined only inside a published kinematic window; the
        # number is not comparable to that paper's outside it.
        win = spec.get("window")
        if win:
            obs = next(iter(arms.values()))["obs"]
            missing = [k for k in win if k not in obs]
            if missing:
                print(f"\n=== {task} === SKIPPED: needs observers {missing}, which "
                      f"this extraction did not save")
                results["tasks"][task] = {"skipped": True, "missing_observers": missing}
                continue
            for k, (lo, hi) in win.items():
                keep &= (obs[k] > lo) & (obs[k] < hi)
        rows = np.where(keep)[0]
        if rows.size < 1000:
            print(f"\n=== {task} === SKIPPED: only {rows.size} jets match "
                  f"{spec['names']}")
            results["tasks"][task] = {"skipped": True, "n": int(rows.size)}
            continue
        y = sig[rows].astype(np.int64)
        tr, va, te = make_splits(rows.size)
        print(f"\n=== {task} ===  {spec['names'][0]} vs {spec['names'][1]}")
        print(f"  {rows.size:,} jets ({y.sum():,} signal), "
              f"split {tr.size:,}/{va.size:,}/{te.size:,}; "
              f"collapsed at: {', '.join(spec['collapsed_at'])}")

        tr_res = {"n": int(rows.size), "n_signal": int(y.sum()),
                  "names": spec["names"], "collapsed_at": spec["collapsed_at"],
                  "arms": {}}
        te_scores = {}
        for arm, d in sorted(arms.items()):
            X = d["F"][rows]
            entry = {}
            eps_list = spec.get("eps_s", [EPS_S])
            for kind, fn in (("linear", fit_linear), ("mlp", fit_mlp)):
                s, meta = fn(X[tr], y[tr], X[va], y[va], X[te])
                auc = float(roc_auc_score(y[te], s))
                rejs = {}
                for e in eps_list:
                    r, eb, bd = rejection_at(y[te], s, e)
                    rejs[f"{e:.2f}"] = {"rejection": r, "eps_b": eb,
                                        "rejection_is_bound": bd}
                first = rejs[f"{eps_list[0]:.2f}"]
                entry[kind] = {"auc": auc, "log1m_auc": float(np.log(max(1 - auc, 1e-12))),
                               "rejection": first["rejection"], "eps_b": first["eps_b"],
                               "rejection_is_bound": first["rejection_is_bound"],
                               "rejection_at": rejs, "selection": meta}
                te_scores.setdefault(kind, {})[arm] = s
                txt = "  ".join(
                    f"1/eps_B@{float(e):.0%}={rejs[e]['rejection']:.1f}"
                    f"{' (BOUND)' if rejs[e]['rejection_is_bound'] else ''}"
                    for e in rejs)
                print(f"  {arm:8s} {kind:6s} AUC {auc:.5f}   {txt}")
            tr_res["arms"][arm] = entry

        # paired arm differences on log(1-AUC) -- D7's inferential metric
        tr_res["contrasts"] = {}
        names = sorted(arms)
        for kind in ("linear", "mlp"):
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    sa, sb = te_scores[kind][a], te_scores[kind][b]
                    ids = np.arange(te.size)          # one jet per event

                    def stat(r, sa=sa, sb=sb, yy=y[te]):
                        return (np.log(max(1 - roc_auc_score(yy[r], sa[r]), 1e-12))
                                - np.log(max(1 - roc_auc_score(yy[r], sb[r]), 1e-12)))

                    from src.stats.bootstrap import event_bootstrap
                    point = stat(np.arange(te.size))
                    dist = event_bootstrap(ids, stat, b=args.bootstrap)
                    lo, hi = ci(dist)
                    tr_res["contrasts"][f"{kind}:{a}-{b}"] = {
                        "delta_log1m_auc": float(point),
                        "ci95": [lo, hi],
                        "excludes_zero": bool(lo > 0 or hi < 0),
                    }
                    print(f"  [{kind}] {a} - {b}: dlog(1-AUC) = {point:+.4f} "
                          f"CI95 [{lo:+.4f}, {hi:+.4f}]"
                          f"{'  *' if (lo > 0 or hi < 0) else ''}")
        results["tasks"][task] = tr_res

    (out / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'probe_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
