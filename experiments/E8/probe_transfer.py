#!/usr/bin/env python3
"""Frozen-probe JC1->JC2 transfer panel (E8): ParT vs Arm P vs Arm S.

Reads the cached JC2 embeddings (from extract_jc2_embeddings.py), remaps the 188
native labels to the g10sem grouping, and fits three probes on the frozen 128-d
embeddings of each backbone, ACROSS A DATA-SIZE SCALING LADDER:

  - linear (primary): standardized multinomial L2 logistic regression, C on val.
  - mlp    (secondary): torch MLPHead([256], dropout 0.1) — same MLP as the
                        scaling curve; 3 probe-init seeds -> mean ± std.
  - knn    (tertiary): cosine k=20 (reference capped for tractability).

Every probe is refit at each adaptation-set size {10k,30k,100k,300k,1M,full}.
All probes are evaluated on ONE fixed test subsample (N_EVAL, seed 777) that is
identical across backbones (row-aligned embeddings) -> the Arm P vs Arm S gap is
paired jet-for-jet and carries (a) backbone-seed spread and (b) an event-level
paired bootstrap (src/stats). ParT is a single-checkpoint reference line.

Output: <out>/transfer_results.json + transfer_table.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.stats.bootstrap import ci, event_bootstrap  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
KNN_K = 20
KNN_REF_CAP = 300_000            # cap kNN reference set for tractability (noted in output)
N_EVAL = 100_000                 # fixed shared test-eval subsample (per backbone-identical)
SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]  # adaptation-set scaling ladder
TEST_SEED = 777
TRAIN_SEED = 1234


def subsample(y, size, n_classes, rng):
    """Class-balanced index subsample of ~`size` jets (>=1/class ⇒ every class
    present, so probe heads always have all `n_classes` columns). Deterministic
    for a fixed rng; since y is identical across backbones, so are the indices."""
    per = max(1, size // n_classes)
    idx = []
    for c in range(n_classes):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, min(per, len(ci)), replace=False))
    return np.concatenate(idx)


def load_g10sem():
    j = json.loads((REPO / "experiments/E2/labels/label_maps.json").read_text())
    return np.asarray(j["g10sem"]["map"], dtype=np.int64)  # (188,) -> 0..9


def load_split(emb_dir, backbone, split, remap):
    emb = np.load(Path(emb_dir) / f"{backbone}_{split}_emb.npy").astype(np.float32)
    lab188 = np.load(Path(emb_dir) / f"{split}_label188.npy")
    return emb, remap[lab188]


def macro_auc(y, prob):
    present = np.unique(y)
    if len(present) < prob.shape[1]:  # a class absent in this (bootstrap/subsample)
        return roc_auc_score(y, prob[:, present], multi_class="ovr",
                             average="macro", labels=present)
    return roc_auc_score(y, prob, multi_class="ovr", average="macro")


def sig_vs_qcd_rej(y, prob, qcd_class=9, eff=0.5):
    """Background(QCD) rejection = 1/FPR at signal efficiency `eff`.
    Signal score = 1 - P(QCD); threshold passes `eff` of true signal; FPR =
    fraction of QCD above it."""
    sig_score = 1.0 - prob[:, qcd_class]
    is_sig = y != qcd_class
    if is_sig.sum() == 0 or (~is_sig).sum() == 0:
        return float("nan")
    thr = np.quantile(sig_score[is_sig], 1.0 - eff)
    fpr = float(np.mean(sig_score[~is_sig] >= thr))
    return float("inf") if fpr == 0 else 1.0 / fpr


def fit_linear(Xtr, ytr, Xva, yva):
    """L2 logistic; C selected on VAL accuracy. Returns (scaler, clf, C)."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    best, best_acc, best_C = None, -1.0, None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1).fit(Xtr_s, ytr)
        acc = accuracy_score(yva, clf.predict(Xva_s))
        if acc > best_acc:
            best, best_acc, best_C = clf, acc, C
    return sc, best, best_C


def fit_mlp(Xtr, ytr, Xva, yva, n_classes, seed, epochs=200, patience=12):
    import torch
    import torch.nn.functional as F
    from src.models.heads import MLPHead
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xva_t = torch.tensor(sc.transform(Xva), dtype=torch.float32, device=dev)
    yva_t = torch.tensor(yva, dtype=torch.long, device=dev)
    net = MLPHead(Xtr.shape[1], n_classes, hidden_dims=[256], dropout=0.1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs, n = 4096, len(Xtr_t)
    best_state, best_va, bad = None, 1e9, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = F.cross_entropy(net(Xva_t), yva_t).item()
        if va < best_va - 1e-4:
            best_va, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return sc, net


def mlp_prob(sc, net, X):
    import torch
    import torch.nn.functional as F
    dev = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        p = F.softmax(net(torch.tensor(sc.transform(X), dtype=torch.float32, device=dev)), 1)
    return p.cpu().numpy()


def evaluate_backbone(emb_dir, backbone, remap, n_classes, sizes, mlp_seeds, want_mlp):
    """Scaling curves for linear/knn/mlp on a fixed shared eval subset.

    Returns (res, full_probs, y_eval): full_probs are the largest-rung probs on
    the eval subset (for the headline bootstrap); y_eval is identical across
    backbones (fixed TEST_SEED + row-aligned embeddings)."""
    Xtr, ytr = load_split(emb_dir, backbone, "train", remap)
    Xva, yva = load_split(emb_dir, backbone, "val", remap)
    Xte, yte = load_split(emb_dir, backbone, "test", remap)

    te_idx = (subsample(yte, N_EVAL, n_classes, np.random.default_rng(TEST_SEED))
              if N_EVAL < len(yte) else np.arange(len(yte)))
    Xe, ye = Xte[te_idx], yte[te_idx]

    ladder = sorted(s for s in sizes if s < len(ytr)) + [len(ytr)]
    curves = {"linear": [], "knn": [], "mlp": []}
    full = {}
    tr_rng = np.random.default_rng(TRAIN_SEED)
    for size in ladder:
        idx = (subsample(ytr, size, n_classes, tr_rng) if size < len(ytr)
               else np.arange(len(ytr)))
        Xs, ys = Xtr[idx], ytr[idx]

        sc, clf, C = fit_linear(Xs, ys, Xva, yva)
        pl = clf.predict_proba(sc.transform(Xe))
        curves["linear"].append({"size": int(size), "n_used": int(len(idx)), "C": C,
                                 "acc": accuracy_score(ye, pl.argmax(1)),
                                 "macro_auc": macro_auc(ye, pl),
                                 "rej_sig_vs_qcd": sig_vs_qcd_rej(ye, pl)})
        full["linear"] = pl

        kref = idx if len(idx) <= KNN_REF_CAP else tr_rng.choice(idx, KNN_REF_CAP, replace=False)
        ksc = StandardScaler().fit(Xtr[kref])
        knn = KNeighborsClassifier(KNN_K, metric="cosine", n_jobs=-1)
        knn.fit(ksc.transform(Xtr[kref]), ytr[kref])
        pk = knn.predict_proba(ksc.transform(Xe))
        curves["knn"].append({"size": int(size), "ref": int(len(kref)),
                              "acc": accuracy_score(ye, pk.argmax(1)),
                              "macro_auc": macro_auc(ye, pk)})
        full["knn"] = pk

        if want_mlp:
            accs, aucs, ps = [], [], []
            for sd in mlp_seeds:
                msc, net = fit_mlp(Xs, ys, Xva, yva, n_classes, sd)
                pm = mlp_prob(msc, net, Xe)
                ps.append(pm)
                accs.append(accuracy_score(ye, pm.argmax(1)))
                aucs.append(macro_auc(ye, pm))
            curves["mlp"].append({"size": int(size), "acc_mean": float(np.mean(accs)),
                                  "macro_auc_mean": float(np.mean(aucs)),
                                  "macro_auc_std": float(np.std(aucs)), "seeds": list(mlp_seeds)})
            full["mlp"] = np.mean(ps, axis=0)
        print(f"     size={size:>8}: lin_auc={curves['linear'][-1]['macro_auc']:.4f} "
              f"knn_auc={curves['knn'][-1]['macro_auc']:.4f}"
              + (f" mlp_auc={curves['mlp'][-1]['macro_auc_mean']:.4f}" if want_mlp else ""))

    res = {"n_train": len(ytr), "n_eval": len(ye), "curves": curves,
           "linear": curves["linear"][-1], "knn": curves["knn"][-1],
           "mlp": curves["mlp"][-1] if want_mlp else None}
    return res, full, ye


def _scaling_table(results, backbones, probe, key):
    sizes = sorted({c["size"] for b in backbones for c in results[b]["curves"][probe]})
    out = [f"### {probe} macro-AUC vs adaptation jets", "",
           "| backbone | " + " | ".join(f"{s // 1000}k" if s < 1_000_000 else f"{s / 1e6:.1f}M"
                                        for s in sizes) + " |",
           "|---|" + "---|" * len(sizes)]
    for b in backbones:
        by = {c["size"]: c[key] for c in results[b]["curves"][probe]}
        out.append(f"| {b} | " + " | ".join(f"{by[s]:.4f}" if s in by else "—" for s in sizes) + " |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", default="g10sem", choices=["g10sem"])
    ap.add_argument("--armp", nargs="*", default=["armp_s1", "armp_s2", "armp_s3"])
    ap.add_argument("--arms", nargs="*", default=["arms_s1", "arms_s2", "arms_s3"])
    ap.add_argument("--mlp-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args()

    remap = load_g10sem()
    n_classes = int(remap.max()) + 1
    want_mlp = not args.no_mlp
    backbones = ["part"] + args.armp + args.arms

    results, lin_probs, yeval = {}, {}, None
    for b in backbones:
        print(f"== probing {b} ==")
        res, full, ye = evaluate_backbone(args.emb_dir, b, remap, n_classes,
                                          args.sizes, args.mlp_seeds, want_mlp)
        results[b] = res
        lin_probs[b] = full["linear"]
        yeval = ye

    # headline: Arm P vs Arm S at full data, paired on the linear probe
    seed_gaps = []
    for pp, ss in itertools.zip_longest(args.armp, args.arms):
        if pp and ss:
            seed_gaps.append(results[pp]["linear"]["macro_auc"] - results[ss]["linear"]["macro_auc"])
    head = {"metric": "linear macro_auc @ full data",
            "arm_p_minus_arm_s_per_seed": seed_gaps,
            "gap_mean": float(np.mean(seed_gaps)), "gap_std": float(np.std(seed_gaps)),
            "arm_p_mean": float(np.mean([results[p]["linear"]["macro_auc"] for p in args.armp])),
            "arm_s_mean": float(np.mean([results[s]["linear"]["macro_auc"] for s in args.arms]))}

    ev = np.arange(len(yeval))  # each JC2 jet is its own event (no multi-jet clusters here)

    def gap_stat(rows):
        yr = yeval[rows]
        ap_auc = np.mean([macro_auc(yr, lin_probs[p][rows]) for p in args.armp])
        as_auc = np.mean([macro_auc(yr, lin_probs[s][rows]) for s in args.arms])
        return ap_auc - as_auc
    dist = event_bootstrap(ev, gap_stat, b=1000, seed=7777)
    head["bootstrap_gap_ci95"] = list(ci(dist))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transfer_results.json").write_text(json.dumps(
        {"granularity": args.granularity, "n_classes": n_classes, "n_eval": len(yeval),
         "knn_ref_cap": KNN_REF_CAP, "backbones": results,
         "headline_armP_vs_armS": head}, indent=2))

    lines = [f"# JC1->JC2 transfer panel ({args.granularity}, {n_classes} classes)", "",
             f"Eval on a fixed {len(yeval):,}-jet test subset (identical across backbones). "
             f"kNN reference capped at {KNN_REF_CAP:,}.", "",
             "## Full-data panel", "",
             "| backbone | linear acc | linear AUC | kNN AUC | MLP AUC (±seed) |",
             "|---|---|---|---|---|"]
    for b in backbones:
        r = results[b]
        mlp = f"{r['mlp']['macro_auc_mean']:.4f} ± {r['mlp']['macro_auc_std']:.4f}" if want_mlp else "—"
        lines.append(f"| {b} | {r['linear']['acc']:.4f} | {r['linear']['macro_auc']:.4f} "
                     f"| {r['knn']['macro_auc']:.4f} | {mlp} |")
    lines += ["", "## Scaling curves (macro-AUC vs adaptation-set size)", ""]
    lines += _scaling_table(results, backbones, "linear", "macro_auc")
    lines += _scaling_table(results, backbones, "knn", "macro_auc")
    if want_mlp:
        lines += _scaling_table(results, backbones, "mlp", "macro_auc_mean")
    lines += ["## Headline — Arm P vs Arm S (does the E1 preprocessing tax invert out-of-domain?)",
              f"- Arm P mean linear AUC (full data): **{head['arm_p_mean']:.4f}**",
              f"- Arm S mean linear AUC (full data): **{head['arm_s_mean']:.4f}**",
              f"- Gap (P − S): **{head['gap_mean']:+.4f} ± {head['gap_std']:.4f}** (seed spread), "
              f"paired-bootstrap 95% CI [{head['bootstrap_gap_ci95'][0]:+.4f}, "
              f"{head['bootstrap_gap_ci95'][1]:+.4f}]", "",
              "In-domain (JetClass-1) Arm P beat Arm S by +0.0083 AUC (the preprocessing tax). "
              "A gap that shrinks/flips here = Sophon preprocessing buys out-of-domain transfer "
              "robustness; the scaling curves show whether that is a few-shot or a converged effect. "
              "Gap is a TOTAL JC1→JC2 domain gap (pileup + reconstruction + class defs), not a pileup term."]
    (out / "transfer_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/transfer_results.json + transfer_table.md")


if __name__ == "__main__":
    main()
