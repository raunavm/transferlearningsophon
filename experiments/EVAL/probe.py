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
import math
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
                     "names": ["label_X_bb", "label_X_cc"]},
    "retained_topology": {"signal": [0], "background": [15],
                          "names": ["label_X_bb", "label_X_YY_bbbb"]},
    "bvc_qcd": {"signal": [169], "background": [181],
                "names": ["label_QCD_bb", "label_QCD_cc"]},
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


RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS",
         "R1_Q1"]


def derive_collapsed_at(signal: list[int], background: list[int]) -> list[str]:
    """Rungs where the task is unmeasurable, DERIVED from the committed map.

    A task is collapsed at a rung when any signal class shares that rung's
    group with any background class -- the multi-class form, because
    bc_vs_rest's background is 30 classes.

    Derived over ALL EIGHT rungs, not just the four-arm run matrix. Before
    this, the three original tasks enumerated the run matrix and the two
    published discriminants enumerated all eight, and both were written to the
    same JSON key, so the record said bvc_resonant collapses at exactly one
    rung when the map says four.
    """
    import csv
    path = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
    with path.open() as f:
        rows = {int(r["jet_label"]): r for r in csv.DictReader(f)}
    out = []
    for rung in RUNGS:
        if rung not in next(iter(rows.values())):
            raise SystemExit(f"FATAL: rung {rung} missing from {path}")
        sg = {rows[i][rung] for i in signal if i in rows}
        bg = {rows[i][rung] for i in background if i in rows}
        if sg & bg:
            out.append(rung)
    return out


PHYSICS_TASKS = {
    "bc_vs_rest": {"signal": [4], "background": [6, 5, 70] + qcd_indices(),
                   "names": ["label_X_bc", "{bq,cs,bqq,QCD}"],
                   # The published selection is THREE cuts, not two
                   # (arXiv:2503.00118 App. A): "450 < p_T < 600, |eta| < 2.4,
                   # and a soft-drop mass requirement of 90 < m_SD < 140".
                   # |eta| < 2.4 is written as the equivalent two-sided bound so
                   # it goes through the same machinery as the other two.
                   # It is an ACTIVE cut: JetClass-II is generated to |eta| <
                   # 2.5 and configs/data/JetClassII_base.yaml imposes no eta
                   # cut, so 2.4-2.5 is populated in the cache.
                   "window": {"jet_pt": [450.0, 600.0],
                              "jet_sdmass": [90.0, 140.0],
                              "jet_eta": [-2.4, 2.4]},
                   "eps_s": [0.60, 0.40]},
    "ee_vs_mm": {"signal": [10], "background": [11],
                 "names": ["label_X_ee", "label_X_mm"]},
}
TASKS.update(PHYSICS_TASKS)

for _name, _spec in TASKS.items():
    _spec["collapsed_at"] = derive_collapsed_at(_spec["signal"], _spec["background"])

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EPS_S = 0.5           # signal efficiency at which rejection is quoted
MIN_PER_CLASS = 1000  # per class, not on the union -- see the guard below
MLP_SEEDS = (0, 1, 2)
# PIN TORCH'S INTRA-OP THREADS. torch.manual_seed fixes the weights and the
# dropout masks, but NOT the order in which a CPU matmul reduces partial sums --
# that follows the thread count, which follows the pod's CPU allocation. Two
# runs of the SAME job over the SAME cached features, differing only in
# `cpu: 4` vs `cpu: 8`, gave 25 of 48 MLP cells different AUCs, max |delta|
# 0.0009, while all 48 LINEAR cells were bit-identical. 0.0009 is larger than
# label_recovery's 5-sigma chance margin at the finest rungs (0.00058 at L188),
# so an unpinned thread count could flip a "not recovered" call between runs.
# The value is recorded in the results so it is auditable, and 4 is what the
# original spec allocated.
MLP_THREADS = 4
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

    # label188 is a property of the DATA, so the check above passes for two
    # caches built from DIFFERENT CHECKPOINTS of the same arm -- features_v2
    # (best epoch) and features_e79 share a file list and therefore a label
    # sha. Mixing them reports an epoch difference as a vocabulary effect and
    # nothing errors. The checkpoint is recorded per arm and surfaced here.
    ckpts = {a: (v.get("manifest") or {}).get("checkpoint_sha256")
                or (v.get("manifest") or {}).get("sha256")
             for a, v in arms.items()}
    named = {a: c for a, c in ckpts.items() if c}
    if len(named) != len(arms):
        print(f"  WARNING: {sorted(set(arms) - set(named))} record no checkpoint "
              f"digest; their manifests predate the field, so a cross-checkpoint "
              f"comparison cannot be ruled out here.", file=sys.stderr)
    for a, c in sorted(named.items()):
        print(f"  {a:12s} checkpoint {c[:16]}")
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
    """(rejection, eps_B, is_bound, n_bkg_pass, rel_stat) at fixed signal eff.

    Linear interpolation on the ROC between adjacent thresholds bracketing
    eps_s, per docs/STATISTICS.md. Past the resolvable cap 1/N_bkg the value is
    an artefact of sample size and is flagged as a bound.

    n_bkg_pass and rel_stat are returned because the rejection is 1/eps_B and
    eps_B is estimated from a COUNT: with a handful of surviving background
    jets the estimator takes only a few distinct values and its spread is
    enormous, while the reported number looks as precise as any other. Poisson
    on the surviving count is the honest band, and anchors.py already reports
    exactly this for the same quantity.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    eps_b = float(np.interp(eps_s, tpr, fpr))
    n_bkg = int((y == 0).sum())
    cap = float(n_bkg)
    if eps_b <= 0:
        return cap, 0.0, True, 0, float("inf")
    n_pass = eps_b * n_bkg
    rel = float("inf") if n_pass <= 0 else 1.0 / math.sqrt(n_pass)
    r = 1.0 / eps_b
    return (min(r, cap), eps_b, r >= cap, int(round(n_pass)), rel)


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
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(MLP_THREADS)
    try:
        return _fit_mlp(Xtr, ytr, Xva, yva, Xte, seeds)
    finally:
        torch.set_num_threads(prev_threads)


def _fit_mlp(Xtr, ytr, Xva, yva, Xte, seeds=MLP_SEEDS):
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
        # `stopped_early` distinguishes "the validation AUC plateaued" from
        # "we ran out of epochs". Both end the loop and both restore the best
        # state, so the returned number looks identical either way -- but only
        # the first means the probe converged, and D6 leans on the MLP to tell
        # "absent" apart from "present but not linearly decodable". That
        # argument needs a converged fit. label_recovery.py's sklearn MLP hit
        # its cap on every cell of the first live run and returned a value
        # BELOW its own linear probe with nothing recording why.
        best_va, best_state, patience, stopped_early = -1.0, None, 0, False
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
                    stopped_early = True
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            scores.append((net(te)[:, 1] - net(te)[:, 0]).numpy())
        meta.append({"seed": sd, "val_auc": float(best_va),
                     "epochs_run": epoch + 1, "converged": stopped_early})
    return np.mean(scores, axis=0), {
        "seeds": meta,
        "val_auc_mean": float(np.mean([m["val_auc"] for m in meta])),
        "val_auc_std": float(np.std([m["val_auc"] for m in meta])),
        "all_converged": all(m["converged"] for m in meta)}


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
               "eps_s_default": EPS_S,
               # Which checkpoint each arm's features came from. Without this
               # the artefact cannot be audited after the fact: label188 is
               # identical across checkpoints of one arm, so a best-epoch cache
               # and an epoch-79 cache are indistinguishable in every other
               # field, and the contrast would read as a vocabulary effect.
               "arm_checkpoints": {
                   a: ((v.get("manifest") or {}).get("checkpoint_sha256")
                       or (v.get("manifest") or {}).get("sha256"))
                   for a, v in sorted(arms.items())},
               "min_per_class_test": MIN_PER_CLASS,
               "mlp_threads": MLP_THREADS,
               "tasks": {}}

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
        y = sig[rows].astype(np.int64)
        # Guard EACH class, not the union. bc_vs_rest is one native class
        # against 30 (27 of them QCD) inside a narrow published window, so the
        # union is entirely background-driven: it cleared 1,000 at 12,499 jets
        # while carrying only 1,370 signal. A union guard cannot see that.
        n_sig, n_bkg = int(y.sum()), int(y.size - y.sum())
        # SPLIT FIRST, then guard on the TEST split. Every reported number --
        # AUC and rejection alike -- is computed on the test split, which is
        # 20 % of the sample, so guarding the full in-window count passes tasks
        # whose test split holds a few hundred of a class. A rejection of 810
        # on 2,223 test-background jets means ~2.7 jets survive the cut: the
        # estimator can then take only a handful of distinct values and its
        # 16-84 % spread runs from roughly half the true value to the cap,
        # while the reported number looks as precise as any other.
        tr, va, te = make_splits(rows.size)
        te_sig, te_bkg = int(y[te].sum()), int(y[te].size - y[te].sum())
        if min(te_sig, te_bkg) < MIN_PER_CLASS:
            print(f"\n=== {task} === SKIPPED: test split has {te_sig:,} signal / "
                  f"{te_bkg:,} background ({n_sig:,}/{n_bkg:,} in window) "
                  f"matching {spec['names']}; need {MIN_PER_CLASS:,} of each "
                  f"IN THE TEST SPLIT")
            results["tasks"][task] = {"skipped": True, "n": int(rows.size),
                                      "n_signal": n_sig,
                                      "n_background": n_bkg,
                                      "n_signal_test": te_sig,
                                      "n_background_test": te_bkg}
            continue
        print(f"\n=== {task} ===  {spec['names'][0]} vs {spec['names'][1]}")
        print(f"  {rows.size:,} jets ({y.sum():,} signal), "
              f"split {tr.size:,}/{va.size:,}/{te.size:,}; "
              f"collapsed at: {', '.join(spec['collapsed_at'])}")

        tr_res = {"eps_s": [float(e) for e in spec.get("eps_s", [EPS_S])],
                  "n": int(rows.size), "n_signal": int(y.sum()),
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
                    r, eb, bd, npass, rel = rejection_at(y[te], s, e)
                    rejs[f"{e:.2f}"] = {"rejection": r, "eps_b": eb,
                                        "rejection_is_bound": bd,
                                        "n_bkg_pass": npass,
                                        "rel_stat_err": rel}
                first = rejs[f"{eps_list[0]:.2f}"]
                entry[kind] = {"auc": auc, "log1m_auc": float(np.log(max(1 - auc, 1e-12))),
                               # The flat fields mirror eps_list[0], which is
                               # NOT the global default for every task, so the
                               # working point they were measured at travels
                               # with them.
                               "rejection": first["rejection"], "eps_b": first["eps_b"],
                               "rejection_is_bound": first["rejection_is_bound"],
                               "n_bkg_pass": first["n_bkg_pass"],
                               "rel_stat_err": first["rel_stat_err"],
                               "rejection_eps_s": float(eps_list[0]),
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
