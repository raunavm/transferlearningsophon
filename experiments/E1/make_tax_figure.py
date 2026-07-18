#!/usr/bin/env python3
"""E1 preprocessing-tax figure from eval_results.json (frozen eval records).

Two panels (macro AUC, accuracy): per-seed points + arm means for Arm P
(ParT preprocessing) vs Arm S (Sophon preprocessing), published ParT as a
dashed reference line only. Output: figures/e1_preprocessing_tax.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "eval_results.json").read_text())

arms = {"arm_p": {"label": "Arm P\n(ParT preproc)", "x": 0, "c": "#1f77b4"},
        "arm_s": {"label": "Arm S\n(Sophon preproc)", "x": 1, "c": "#d62728"}}
ref = res["reference_lines"]["published_ParT_2202.03772"]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, (key, name) in zip(axes, [("macro_auc_ovr", "Macro AUC (OvR)"),
                                  ("accuracy", "Accuracy")]):
    means = {}
    for arm, sty in arms.items():
        vals = [r[key] for r in res["runs"] if r["arm"] == arm]
        means[arm] = sum(vals) / len(vals)
        ax.scatter([sty["x"]] * len(vals), vals, s=28, color=sty["c"],
                   zorder=3, label=None)
        ax.hlines(means[arm], sty["x"] - 0.18, sty["x"] + 0.18,
                  color=sty["c"], lw=2.5, zorder=4)
    tax = means["arm_p"] - means["arm_s"]
    ax.axhline(ref[key], ls="--", lw=1, color="gray", zorder=1)
    ax.text(0.985, ref[key], "published ParT (ref. line)", ha="right",
            va="bottom", fontsize=7, color="gray", transform=ax.get_yaxis_transform())
    ymid = (means["arm_p"] + means["arm_s"]) / 2
    ax.annotate("", xy=(0.5, means["arm_s"]), xytext=(0.5, means["arm_p"]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text(0.54, ymid, f"tax = +{tax:.4f}", fontsize=9, va="center")
    ax.set_xticks([0, 1], [arms["arm_p"]["label"], arms["arm_s"]["label"]])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(name, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("E1 preprocessing tax — same Sophon architecture, recipe, and jets; "
             "only the data config differs (test_20M, n=20M)", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "figures"
out.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"e1_preprocessing_tax.{ext}", dpi=200)
print("wrote", out / "e1_preprocessing_tax.{pdf,png}")
