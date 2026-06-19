"""Classification metrics utility used by training sweeps.

`compute_classification_metrics(labels, probs, label_names)` returns a single
JSON-serializable dict with accuracy, macro AUC (OvR), per-class AUC, TPR @
fixed-FPR thresholds (per class and macro average), and both raw and
normalized confusion matrices.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


FPR_THRESHOLDS = (1e-1, 1e-2, 1e-3)


def _tpr_at_fpr(y_true_binary: np.ndarray, y_score: np.ndarray,
                fpr_threshold: float) -> float:
    """Max TPR achievable while FPR <= fpr_threshold (one-vs-rest)."""
    if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    mask = fpr <= fpr_threshold
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def compute_classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    label_names: Sequence[str],
) -> dict:
    """Compute classification metrics for an N-class one-vs-rest task.

    Args:
        labels: (N,) int array of ground-truth class indices in [0, C).
        probs:  (N, C) float array of softmax probabilities.
        label_names: sequence of length C with display names per class.

    Returns:
        dict with:
          - accuracy: float
          - macro_auc_ovr: float
          - per_class_auc: {name: float}
          - tpr_at_fpr: {"1e-1": {"per_class": {name: float}, "macro": float}, ...}
          - macro_tpr_at_fpr_1e-1 / _1e-2 / _1e-3: float (convenience flat keys)
          - confusion_matrix: list[list[int]]
          - normalized_confusion_matrix: list[list[float]]
    """
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    num_classes = len(label_names)

    preds = probs.argmax(axis=1)
    accuracy = float(accuracy_score(labels, preds))

    try:
        macro_auc = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_auc = float("nan")

    per_class_auc: dict[str, float] = {}
    for k in range(num_classes):
        y_bin = (labels == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auc[label_names[k]] = float("nan")
            continue
        try:
            per_class_auc[label_names[k]] = float(roc_auc_score(y_bin, probs[:, k]))
        except ValueError:
            per_class_auc[label_names[k]] = float("nan")

    tpr_at_fpr: dict[str, dict] = {}
    flat_macro: dict[str, float] = {}
    for thr in FPR_THRESHOLDS:
        key = f"{thr:.0e}"  # e.g. "1e-01"
        # Normalize to user-friendly form "1e-1"/"1e-2"/"1e-3"
        if thr == 1e-1:
            key = "1e-1"
        elif thr == 1e-2:
            key = "1e-2"
        elif thr == 1e-3:
            key = "1e-3"

        per_cls: dict[str, float] = {}
        finite = []
        for k in range(num_classes):
            y_bin = (labels == k).astype(int)
            val = _tpr_at_fpr(y_bin, probs[:, k], thr)
            per_cls[label_names[k]] = val
            if np.isfinite(val):
                finite.append(val)
        macro = float(np.mean(finite)) if finite else float("nan")
        tpr_at_fpr[key] = {"per_class": per_cls, "macro": macro}
        flat_macro[f"macro_tpr_at_fpr_{key}"] = macro

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    out: dict = {
        "accuracy": accuracy,
        "macro_auc_ovr": macro_auc,
        "per_class_auc": per_class_auc,
        "tpr_at_fpr": tpr_at_fpr,
        "confusion_matrix": cm.astype(int).tolist(),
        "normalized_confusion_matrix": cm_norm.astype(float).tolist(),
    }
    out.update(flat_macro)
    return out
