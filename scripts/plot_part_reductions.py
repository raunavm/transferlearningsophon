#!/usr/bin/env python3
"""Visualize ParT-full's 128-d JetClass latent space with UMAP, t-SNE, and PCA.

Reads the embeddings written by extract_part_embeddings.py and produces:
  results/main_plots/part_umap.{pdf,png}
  results/main_plots/part_tsne.{pdf,png}
  results/main_plots/part_pca.{pdf,png}
  results/main_plots/part_reductions_3panel.{pdf,png}   (the headline figure)
  results/main_plots/part_reduction_metrics.json
  results/main_plots/part_reduction_sample.npz

Refuses to render unless results/main_plots/part_full_metrics.json reports
macro AUC >= 0.98 (override with --force).

Optional Sophon comparison: if --sophon-dir and --sophon-ft-dir are passed and
the embedding files exist, also produce part_vs_sophon_umap.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from plots.style import apply_style, save_fig, CLASS_COLORS

CLASSES = ["ZJetsToNuNu", "HToBB", "HToCC", "HToGG",
           "HToWW4Q", "HToWW2Q1L", "ZToQQ", "WToQQ", "TTBar", "TTBarLep"]
SHORT = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def load_embeddings(emb_dir: Path):
    embs, labs = [], []
    for k, cls in enumerate(CLASSES):
        ep = emb_dir / f"{cls}_embeddings.npy"
        if not ep.exists():
            print(f"  [skip] {cls}: {ep.name} missing")
            continue
        e = np.load(ep).astype(np.float32)
        embs.append(e)
        labs.append(np.full(len(e), k, dtype=np.int64))
    if not embs:
        raise SystemExit(f"no embeddings under {emb_dir}")
    return np.concatenate(embs), np.concatenate(labs)


def class_balanced_subsample(labels: np.ndarray, n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_pieces = []
    for k in range(10):
        cls_idx = np.where(labels == k)[0]
        if len(cls_idx) == 0:
            continue
        take = min(n_per_class, len(cls_idx))
        pick = rng.choice(cls_idx, size=take, replace=False)
        idx_pieces.append(pick)
    idx = np.concatenate(idx_pieces)
    rng.shuffle(idx)
    return idx


def silhouette(emb: np.ndarray, lab: np.ndarray, max_n: int = 10000,
               metric: str = "euclidean", seed: int = 42) -> float:
    """Mean silhouette score on up to max_n points (downsampled if larger)."""
    from sklearn.metrics import silhouette_samples
    if len(lab) > max_n:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(lab), size=max_n, replace=False)
        emb, lab = emb[sub], lab[sub]
    return float(silhouette_samples(emb, lab, metric=metric).mean())


def panel(ax, coords, lab, title, subtitle="", show_legend=False):
    handles = []
    from matplotlib.lines import Line2D
    for k in sorted(np.unique(lab)):
        m = (lab == k)
        name = SHORT[k]
        color = CLASS_COLORS.get(name, "#888888")
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.40,
                   color=color, edgecolors="none", rasterized=True, label=name)
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=color, markersize=8,
                              markeredgecolor="none", label=name))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=10)
    if subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#444",
                bbox=dict(facecolor="white", edgecolor="#bbb",
                          alpha=0.9, boxstyle="round,pad=0.3"))
    if show_legend:
        ax.legend(handles=handles, loc="upper right",
                  fontsize=8, frameon=True, framealpha=0.92,
                  facecolor="white", edgecolor="#bbb",
                  borderpad=0.5, labelspacing=0.3,
                  handletextpad=0.4, ncol=1)


def fit_umap(emb, seed):
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.30, metric="cosine",
                        random_state=seed, n_components=2, verbose=False)
    return reducer.fit_transform(emb)


def fit_tsne(emb, seed, prepca_dim: int = 50, max_iter: int = 1000):
    """Reduce to 50D with PCA first (standard), then run sklearn t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if emb.shape[1] > prepca_dim:
        emb50 = PCA(n_components=prepca_dim, random_state=seed).fit_transform(emb)
    else:
        emb50 = emb
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", max_iter=max_iter,
                random_state=seed, n_jobs=-1, verbose=1)
    return tsne.fit_transform(emb50)


def fit_pca(emb, seed):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(scaled)
    return coords, pca.explained_variance_ratio_.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", default="embeddings_part_full_test")
    p.add_argument("--n-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/main_plots")
    p.add_argument("--metrics-json",
                   default="results/main_plots/part_full_metrics.json")
    p.add_argument("--force", action="store_true",
                   help="Render even if sanity-check AUC is below 0.98.")
    p.add_argument("--tsne-fallback-n-per-class", type=int, default=1500,
                   help="If the initial t-SNE is too slow, optionally rerun "
                        "with this lower n_per_class. Triggered with "
                        "--tsne-budget-seconds.")
    p.add_argument("--tsne-budget-seconds", type=float, default=900.0,
                   help="Wall-clock budget for t-SNE. If first attempt at "
                        "n_per_class jets exceeds this estimate, fall back.")
    # Optional Sophon comparison
    p.add_argument("--sophon-dir", default=None,
                   help="Sophon pretrained embeddings dir (skips if missing)")
    p.add_argument("--sophon-ft-dir", default=None,
                   help="Sophon full-FT embeddings dir (skips if missing)")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_dir)

    # Sanity gate
    sanity_path = Path(args.metrics_json)
    if sanity_path.exists():
        m = json.loads(sanity_path.read_text())
        macro = m.get("macro_auc_ovr", 0.0)
        if macro < 0.98 and not args.force:
            raise SystemExit(
                f"\nREFUSING: macro AUC {macro:.4f} < 0.98 in {sanity_path}.\n"
                f"Pass --force to render anyway."
            )
        print(f"Sanity gate: macro AUC = {macro:.4f} (>= 0.98 OK)")
    elif args.force:
        print(f"Sanity gate: {sanity_path} missing, --force given, proceeding.")
    else:
        raise SystemExit(
            f"\nREFUSING: {sanity_path} not found.\n"
            f"Run scripts/part_sanity_check.py first, or pass --force."
        )

    apply_style()

    print(f"Loading embeddings from {emb_dir}")
    emb_all, lab_all = load_embeddings(emb_dir)
    print(f"  total: {len(lab_all):,} jets, dim {emb_all.shape[1]}")

    idx = class_balanced_subsample(lab_all, args.n_per_class, args.seed)
    emb = emb_all[idx]
    lab = lab_all[idx]
    print(f"  subsampled: {len(lab):,} jets (target {args.n_per_class}/class)")

    # NOTE: coords are appended below after each reduction so re-rendering
    # the 3-panel is fast (no UMAP/t-SNE recompute).
    sample_payload = {"indices": idx, "labels": lab,
                      "n_per_class": args.n_per_class, "seed": args.seed}

    # 128-d silhouette (in original space) — one number for the whole subset
    sil_128 = silhouette(emb, lab, max_n=10000, metric="euclidean", seed=args.seed)
    print(f"Silhouette (128-d Euclidean): {sil_128:.4f}")

    # UMAP
    print("\nUMAP ...")
    t0 = time.time(); umap_coords = fit_umap(emb, args.seed)
    umap_secs = time.time() - t0
    sil_umap = silhouette(umap_coords, lab, max_n=10000,
                          metric="euclidean", seed=args.seed)
    print(f"  done in {umap_secs:.1f}s; 2D silhouette = {sil_umap:.4f}")

    # PCA
    print("\nPCA ...")
    t0 = time.time()
    pca_coords, explained = fit_pca(emb, args.seed)
    pca_secs = time.time() - t0
    sil_pca = silhouette(pca_coords, lab, max_n=10000,
                         metric="euclidean", seed=args.seed)
    print(f"  done in {pca_secs:.1f}s; explained variance = "
          f"{explained[0]:.3f}/{explained[1]:.3f}; 2D silhouette = {sil_pca:.4f}")

    # t-SNE
    print(f"\nt-SNE ({args.n_per_class}/class) ...")
    t0 = time.time()
    tsne_coords = fit_tsne(emb, args.seed)
    tsne_secs = time.time() - t0
    tsne_n_per_class = args.n_per_class
    fell_back = False
    if (tsne_secs > args.tsne_budget_seconds
            and args.tsne_fallback_n_per_class < args.n_per_class):
        print(f"  t-SNE exceeded budget ({tsne_secs:.0f}s > "
              f"{args.tsne_budget_seconds:.0f}s); falling back to "
              f"{args.tsne_fallback_n_per_class}/class")
        idx_fb = class_balanced_subsample(lab_all,
                                          args.tsne_fallback_n_per_class,
                                          args.seed)
        emb_fb = emb_all[idx_fb]
        lab_fb = lab_all[idx_fb]
        t0 = time.time()
        tsne_coords = fit_tsne(emb_fb, args.seed)
        tsne_secs = time.time() - t0
        tsne_n_per_class = args.tsne_fallback_n_per_class
        fell_back = True
        # Update subsample to match what t-SNE actually used so labels align
        emb_for_tsne, lab_for_tsne = emb_fb, lab_fb
    else:
        emb_for_tsne, lab_for_tsne = emb, lab
    sil_tsne = silhouette(tsne_coords, lab_for_tsne, max_n=10000,
                          metric="euclidean", seed=args.seed)
    print(f"  done in {tsne_secs:.1f}s; 2D silhouette = {sil_tsne:.4f}; "
          f"used {tsne_n_per_class}/class")

    # Trustworthiness (optional, costly; computed on the smaller subset only)
    trust = {}
    try:
        from sklearn.manifold import trustworthiness
        sub_n = min(2000, len(emb))
        sub_idx = np.random.default_rng(args.seed).choice(len(emb), size=sub_n,
                                                          replace=False)
        trust["umap"] = float(trustworthiness(emb[sub_idx],
                                              umap_coords[sub_idx], n_neighbors=10))
        trust["pca"]  = float(trustworthiness(emb[sub_idx],
                                              pca_coords[sub_idx],  n_neighbors=10))
        # t-SNE trust only if we didn't fall back to a different sample
        if not fell_back:
            trust["tsne"] = float(trustworthiness(emb[sub_idx],
                                                  tsne_coords[sub_idx], n_neighbors=10))
    except Exception as e:
        print(f"(trustworthiness skipped: {e})")

    # Single-panel figures
    import matplotlib.pyplot as plt

    def single(coords, lab_use, title, fname, subtitle):
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        panel(ax, coords, lab_use, title, subtitle=subtitle, show_legend=True)
        ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
        save_fig(fig, str(out_dir / fname))
        plt.close(fig)
        print(f"Saved: {out_dir / fname}.{{pdf,png}}")

    single(umap_coords, lab,
           "ParT-full JetClass latent space: UMAP",
           "part_umap",
           f"silhouette = {sil_umap:.3f}")
    single(tsne_coords, lab_for_tsne,
           "ParT-full JetClass latent space: t-SNE",
           "part_tsne",
           f"silhouette = {sil_tsne:.3f}")
    single(pca_coords, lab,
           "ParT-full JetClass latent space: PCA",
           "part_pca",
           f"silhouette = {sil_pca:.3f}")

    # 3-panel headline
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8),
                             gridspec_kw=dict(wspace=0.04))
    panel(axes[0], umap_coords, lab, "UMAP",
          subtitle=f"silhouette = {sil_umap:.3f}")
    panel(axes[1], tsne_coords, lab_for_tsne, "t-SNE",
          subtitle=f"silhouette = {sil_tsne:.3f}")
    panel(axes[2], pca_coords, lab, "PCA",
          subtitle=f"silhouette = {sil_pca:.3f}",
          show_legend=True)
    fig.suptitle("ParT-full representations on JetClass test jets",
                 y=0.99, fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.04)
    save_fig(fig, str(out_dir / "part_reductions_3panel"))
    plt.close(fig)
    print(f"Saved: {out_dir / 'part_reductions_3panel'}.{{pdf,png}}")

    # Save coords + labels alongside indices so a future render-only script
    # can recreate the figure in seconds.
    sample_payload.update({
        "umap_coords":      umap_coords,
        "pca_coords":       pca_coords,
        "tsne_coords":      tsne_coords,
        "labels_for_tsne":  lab_for_tsne,
        "explained_var":    np.asarray(explained, dtype=np.float64),
    })
    np.savez(out_dir / "part_reduction_sample.npz", **sample_payload)

    # Metrics JSON
    metrics = {
        "n_per_class":            args.n_per_class,
        "seed":                   args.seed,
        "total_jets_subsampled":  int(len(lab)),
        "silhouette_128d":        round(sil_128, 6),
        "umap": {
            "silhouette_2d": round(sil_umap, 6),
            "wall_seconds":  round(umap_secs, 1),
            "n_neighbors":   30, "min_dist": 0.30, "metric": "cosine",
        },
        "tsne": {
            "silhouette_2d":     round(sil_tsne, 6),
            "wall_seconds":      round(tsne_secs, 1),
            "n_per_class_used":  tsne_n_per_class,
            "fell_back":         fell_back,
            "perplexity":        30, "init": "pca", "max_iter": 1000,
            "prepca_dim":        50,
        },
        "pca": {
            "silhouette_2d":              round(sil_pca, 6),
            "wall_seconds":               round(pca_secs, 1),
            "explained_variance_ratio":   [round(e, 6) for e in explained],
            "standardized_inputs":        True,
        },
        "trustworthiness":  trust,
    }
    with open(out_dir / "part_reduction_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Wrote {out_dir / 'part_reduction_metrics.json'}")

    # Optional Sophon comparison (best-effort, skip if dirs missing)
    if args.sophon_dir and args.sophon_ft_dir:
        sd, sf = Path(args.sophon_dir), Path(args.sophon_ft_dir)
        if sd.is_dir() and sf.is_dir():
            print("\nSophon comparison: rendering 3-panel UMAP")
            try:
                from src.data.embedding_dataset import _load_dir
                pre_emb, pre_lab = _load_dir(str(sd))
                ft_emb,  ft_lab  = _load_dir(str(sf))
                rng = np.random.default_rng(args.seed)
                def subsample_balanced(emb, lab, n):
                    pieces = []
                    for k in range(10):
                        ci = np.where(lab == k)[0]
                        if len(ci) == 0: continue
                        take = min(n, len(ci))
                        pieces.append(rng.choice(ci, size=take, replace=False))
                    return np.concatenate(pieces)
                n = args.n_per_class
                pi = subsample_balanced(pre_emb, pre_lab, n)
                fi = subsample_balanced(ft_emb,  ft_lab,  n)
                pre_c = fit_umap(pre_emb[pi].astype(np.float32), args.seed)
                ft_c  = fit_umap(ft_emb[fi].astype(np.float32),  args.seed)

                fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8),
                                         gridspec_kw=dict(wspace=0.04))
                panel(axes[0], pre_c, pre_lab[pi], "Sophon pretrained")
                panel(axes[1], ft_c,  ft_lab[fi],  "Sophon full-FT (10M, seed 42)")
                panel(axes[2], umap_coords, lab,   "ParT-full",
                      show_legend=True)
                fig.suptitle("Jet latent spaces: UMAP comparison",
                             y=0.99, fontsize=15)
                fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.04)
                save_fig(fig, str(out_dir / "umap_sophon_vs_part"))
                plt.close(fig)
                print(f"Saved: {out_dir / 'umap_sophon_vs_part'}.{{pdf,png}}")
            except Exception as e:
                print(f"(Sophon comparison skipped: {e})")
        else:
            print(f"(Sophon comparison skipped: {sd} or {sf} missing)")


if __name__ == "__main__":
    main()
