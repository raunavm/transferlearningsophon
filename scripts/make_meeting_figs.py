#!/usr/bin/env python3
"""Meeting add-on figures:
  1) figures/e1_gate_reproduction.{png,pdf} — Arm P (from-scratch, ParT recipe) val-accuracy
     vs epoch converging onto the published-ParT line; annotated with the test-set gate.
  2) figures/silhouette_comparison.{png,pdf} — 128-d silhouette: pretrained Sophon vs
     Sophon full-FT vs ParT (supervised).

Val curve is parsed from a weaver train.log (Arm P run). Pass its path as argv[1]
(default: the local pull at /tmp/armp_test/train.log).
"""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "grey": "#999999", "vermillion": "#D55E00"}

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/armp_test/train.log"


def val_curve(path):
    ep, val = [], []
    for ln in open(path, errors="ignore"):
        m = re.search(r"Epoch #(\d+): Current validation metric: ([\d.]+)", ln)
        if m:
            ep.append(int(m.group(1)))
            val.append(float(m.group(2)))
    return ep, val


def fig_gate():
    ep, val = val_curve(LOG)
    best = max(val)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ep, val, color=OKABE["orange"], lw=1.8, marker="o", ms=3,
            label="Arm P — from scratch (ParT recipe)")
    ax.axhline(0.861, ls="--", color=OKABE["grey"], lw=1.4,
               label="published ParT (accuracy 0.861)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (10-class JetClass)")
    ax.set_title("E1: from-scratch training reproduces published ParT")
    ax.set_ylim(0.82, 0.868)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.95,
            f"best val acc = {best:.4f}\ntest macro AUC = 0.9877 = published ParT (Δ +0.0000)\n"
            f"2.21 M params · 327 M MACs/jet (ParT ref 340 M)",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=OKABE["grey"], alpha=0.9))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/e1_gate_reproduction.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/e1_gate_reproduction.{png,pdf}")


def fig_silhouette():
    names = ["Pretrained\nSophon", "Sophon\nfull-FT (10M)", "ParT\n(supervised)"]
    vals = [0.077, 0.206, 0.265]
    cols = [OKABE["grey"], OKABE["green"], OKABE["blue"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(names, vals, color=cols, width=0.62, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Silhouette (native 128-d)")
    ax.set_title("Representation separability of jet classes")
    ax.set_ylim(0, 0.30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/silhouette_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/silhouette_comparison.{png,pdf}")


if __name__ == "__main__":
    fig_gate()
    fig_silhouette()
