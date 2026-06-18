#!/usr/bin/env python3
"""Sanity-check ParT extraction: accuracy, macro AUC, per-class AUC, confusion matrix.

Reads {ClassName}_logits.npy and {ClassName}_labels.npy from --emb-dir, computes
the metrics on the entire extracted subset, and writes:

  results/main_plots/part_full_metrics.json
  results/main_plots/part_confusion_matrix.{pdf,png}

If macro AUC < 0.98 it prints a clear warning so plotting can refuse to proceed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

CLASSES = ["ZJetsToNuNu", "HToBB", "HToCC", "HToGG",
           "HToWW4Q", "HToWW2Q1L", "ZToQQ", "WToQQ", "TTBar", "TTBarLep"]
SHORT = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def load_subset(emb_dir: Path):
    logits_chunks, label_chunks = [], []
    for cls in CLASSES:
        lp = emb_dir / f"{cls}_logits.npy"
        yp = emb_dir / f"{cls}_labels.npy"
        if not lp.exists():
            print(f"  [skip] {cls}: {lp.name} missing")
            continue
        logits_chunks.append(np.load(lp).astype(np.float32))
        label_chunks.append(np.load(yp).astype(np.int64))
    if not logits_chunks:
        raise SystemExit(f"no logits found under {emb_dir}")
    return np.concatenate(logits_chunks), np.concatenate(label_chunks)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", default="embeddings_part_full_test")
    p.add_argument("--output-dir", default="results/main_plots")
    args = p.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    print(f"Loading logits + labels from {emb_dir}")
    logits, labels = load_subset(emb_dir)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    acc = float(accuracy_score(labels, preds))
    macro_auc = float(roc_auc_score(labels, probs, multi_class="ovr",
                                    average="macro"))
    per_class_auc = {}
    for k in range(10):
        y_bin = (labels == k).astype(int)
        if y_bin.sum() == 0:
            continue
        per_class_auc[SHORT[k]] = float(roc_auc_score(y_bin, probs[:, k]))

    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    metrics = {
        "n_jets":            int(len(labels)),
        "accuracy":          round(acc, 6),
        "macro_auc_ovr":     round(macro_auc, 6),
        "per_class_auc":     {k: round(v, 6) for k, v in per_class_auc.items()},
        "per_class_counts":  {SHORT[k]: int((labels == k).sum()) for k in range(10)},
        "expected_part_full_macro_auc": 0.988,
        "sanity_passed":     bool(macro_auc >= 0.98),
    }

    with open(out_dir / "part_full_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"\nN jets:        {len(labels):,}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro AUC OvR: {macro_auc:.4f}  (published ParT-full ~ 0.988)")
    print("Per-class AUC:")
    for k, v in per_class_auc.items():
        print(f"  {k:>5}: {v:.4f}")

    if macro_auc < 0.98:
        print("\nSANITY CHECK FAILED: macro AUC below 0.98 threshold")
        print("Likely a preprocessing or checkpoint bug. Investigate before plotting.")
    else:
        print("\nSanity check PASSED.")

    # Confusion matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots.style import apply_style, save_fig
        apply_style()

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                v = cm_norm[i, j]
                if v > 0.01:
                    color = "white" if v > 0.5 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(SHORT, fontsize=9)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"ParT-full confusion matrix\n"
                     f"acc = {acc:.3f}, macro AUC = {macro_auc:.3f}", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(fig, str(out_dir / "part_confusion_matrix"))
        print(f"Saved: {out_dir / 'part_confusion_matrix'}.{{pdf,png}}")
    except Exception as e:
        print(f"(confusion-matrix plot skipped: {e})")

    print(f"\nWrote {out_dir / 'part_full_metrics.json'}")


if __name__ == "__main__":
    main()
