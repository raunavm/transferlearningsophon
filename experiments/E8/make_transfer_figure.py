#!/usr/bin/env python3
"""E8 transfer-panel figure from results_summary.json (frozen job outputs).

Panel A: linear-probe scaling curves on JetClass-II (macro AUC vs adaptation
jets) — Arm P vs Arm S seed bands, official ParT as reference line.
Panel B: the preprocessing tax in-domain (E1, JetClass-1) vs out-of-domain
(E8, JetClass-2 transfer): the tax widens, it does not invert.

Output: figures/e8_transfer_tax.{pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
try:
    from plots.style import apply_style
    apply_style()
except Exception:
    pass

res = json.loads((HERE / "results_summary.json").read_text())
sizes = np.asarray(res["sizes"])
lin = res["curves_linear_macro_auc"]

C_P, C_S, C_REF = "#0072B2", "#D55E00", "#666666"   # Okabe-Ito blue/vermillion

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                               gridspec_kw={"width_ratios": [1.5, 1.0]})

# --- Panel A: scaling curves ---
armp = np.array([lin["armp_s1"], lin["armp_s3"]])
arms = np.array([lin["arms_s1"], lin["arms_s2"], lin["arms_s3"]])
axA.plot(sizes, np.asarray(lin["part"]), "--", color=C_REF, lw=1.4,
         label="official ParT (ref.)")
for arr, c, lab in [(armp, C_P, "Arm P (ParT preproc)"),
                    (arms, C_S, "Arm S (Sophon preproc)")]:
    axA.fill_between(sizes, arr.min(0), arr.max(0), color=c, alpha=0.25, lw=0)
    axA.plot(sizes, arr.mean(0), "-o", color=c, ms=3.5, lw=1.6, label=lab)
axA.set_xscale("log")
axA.set_xlabel("adaptation jets (linear probe)")
axA.set_ylabel("macro AUC (JetClass-II, 10 classes)")
axA.set_title("Frozen JC1 backbones probed on JetClass-II")
axA.legend(frameon=False, fontsize=8, loc="lower right")
axA.grid(True, alpha=0.3)

# --- Panel B: tax in- vs out-of-domain ---
e1 = res["in_domain_reference_E1"]
hd = res["headline_armP_minus_armS_linear_top_rung"]
xs = [0, 1]
gaps = [e1["tax_auc"], hd["gap_mean"]]
seed_err = [e1["tax_std_seed"], hd["gap_std_seed"]]
axB.bar(xs, gaps, width=0.5, color=[C_P, C_S], alpha=0.85)
axB.errorbar(xs, gaps, yerr=seed_err, fmt="none", ecolor="k", capsize=4, lw=1.2)
lo, hi = hd["bootstrap_ci95"]
axB.plot([1.0, 1.0], [lo, hi], color="k", lw=0.8)
axB.annotate("bootstrap\n95% CI", xy=(1.02, hi), fontsize=7, va="bottom")
axB.annotate("", xy=(1, hd["gap_mean"]), xytext=(0, e1["tax_auc"]),
             arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0,
                             connectionstyle="arc3,rad=-0.25"))
axB.text(0.5, (e1["tax_auc"] + hd["gap_mean"]) / 2 + 0.0012,
         f"×{hd['gap_mean'] / e1['tax_auc']:.1f}", ha="center", fontsize=9, color="0.25")
axB.set_xticks(xs)
axB.set_xticklabels(["in-domain\n(JetClass-1, E1)", "transfer\n(JetClass-II, E8)"])
axB.set_ylabel("Arm P − Arm S macro AUC")
axB.set_title("Preprocessing tax widens off-domain")
axB.set_ylim(0, max(gaps) * 1.35)
axB.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
out = HERE / "figures"
out.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"e8_transfer_tax.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {out}/e8_transfer_tax.pdf/.png")
