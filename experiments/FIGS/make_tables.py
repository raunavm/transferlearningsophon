#!/usr/bin/env python3
"""Every number the paper prints, generated from the result files that hold it.

WHAT THIS REPLACES. `paper/ml4ps/results.tex` was typed by hand: a macro, a
comment naming the run it came from, and nothing that could check the two still
agreed. This script writes the same kind of file -- one `\\newcommand` per
number -- but each line carries the file, the path inside that file, and the
first 16 hex of the file's sha256, and `paper/journal/provenance.json` carries
the full hash beside the value. A number whose source file has changed shows up
as a changed hash, not as a silent disagreement with the text.

WHAT IT REFUSES TO DO.
  * It never writes a placeholder. An input that does not exist yet means the
    macros that would have come from it are NOT emitted and the input is named
    in the missing report -- so an unwritten number is a `\\pending` in the
    draft, never a plausible-looking one.
  * It reads no test. Every result is the mean and standard deviation (ddof=1)
    over pretraining seeds of the per-seed rows the result files hold, rounded
    by the Particle Data Group rule (fmt_pm); the statistics blocks those files
    also carry are not read.
  * It refuses outright if two inputs disagree on `row_alignment_sha256`, or if
    a ladder file has changed since the analysis that read it. Both mean the
    arms were not scored on the same jets in the same order, and every
    comparison in the paper assumes they were.
  * A background rejection that is a lower bound (no background jet survived
    the cut, so the number is the sample size, not a measurement) is never
    printed as a bare number or averaged. `$>$` and the 95 % lower limit when every
    seed is at the cap, `$\\geq$` with a footnote when only some are. Same for an AUC that
    saturated at 1.

Usage:
    python3 experiments/FIGS/make_tables.py            # write paper/journal/
    python3 experiments/FIGS/make_tables.py --check    # CI: non-zero on drift
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]

# Contraction order, finest first. docs/DECISIONS.md D3. Names are config keys,
# never prose: the paper says "43-class", and the size comes from the map below.
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
ARM_RUNG = {"l188": "L188", "l162": "L162", "r42q1": "R42_Q1", "r16q1": "R16_Q1"}

# Descriptive names for the probe tasks and the fine-tuning initialisations.
# Labels only -- no number lives here.
TASK_LABELS = {
    "bvc_resonant": "$b$ vs $c$, two-prong resonance",
    "bvc_qcd": "$b$ vs $c$, QCD",
    "retained_topology": "two-prong vs four-prong",
    "ee_vs_mm": "electron vs muon pair",
    "bvc_4prong": "$b$ vs $c$, four-prong",
    "visible_content": "$bbqq$ vs $cq\\tau_h\\nu$, four-prong",
    "bc_vs_rest": "$X\\to bc$ vs its backgrounds",
    # the second grid's single-pair probes (A10, 2026-10-01)
    "bc_vs_bq": "$X\\to bc$ vs $X\\to bq$",
    "bc_vs_cs": "$X\\to bc$ vs $X\\to cs$",
}
INIT_LABELS = {"scratch": "random initialisation",
               "sophon-public": "published 188-class checkpoint",
               "ref_e1arms": "pipeline-validation reference"}

# Result files that the paper needs and that no run has produced yet. Each is
# reported by name so the missing report is a to-do list, not a shrug.
PENDING_INPUTS = [
    ("community benchmarks, bench v3 readout",
     "experiments/FIGS/data/bench_v3_metrics*/bench_metrics_last.json"),
]

# THE SECOND PRETRAINING GRID ("v2", decided 2026-09-29). Every pretraining run is
# repeated with the data order and dropout derived from (seed, epoch), identical
# trunk weights across vocabularies, Sophon's family-stratified loading over disjoint
# random windows of each file's rows, a fixed validation sample and the checkpoint at
# the first maximum within epochs 70-79 (PRESPEC A14); JetClass-II fine-tuning
# moves to held-out files; more random partitions, a loss-share-matched mass arm,
# the 64- and 30-class levels and leave-one-family-out arms are added. Until a
# section's v2 inputs exist, the section keeps its first-grid text and the paper
# prints the marker below, in red, naming what it waits for. Each input is a
# directory under experiments/FIGS/data/v2/; when it appears the marker changes
# to a request to rewrite the section, never to nothing, so first-grid text
# cannot outlive its replacement silently.
V2_PENDING = {
    "Pretrain": ("pretraining rerun: data order and dropout fixed by seed and epoch, "
                 "identical initial trunk weights, each epoch reading one of five disjoint "
                 "random parts of the rows of nearly every file, fixed validation sample, "
                 "checkpoint at the first maximum within the last ten epochs", "pretraining"),
    "Probes": ("frozen probes on the rerun models, at the first maximum within the last ten "
               "epochs and on the model whose weights are the average of the last ten epochs, "
               "BatchNorm statistics recomputed, with untrained-network and pooled-embedding "
               "reference rows", "probe_ladder"),
    "Paired": ("paired ratios with run and test-sample errors on the rerun models",
               "paired_errors"),
    "Levels": ("the 64- and 30-class vocabularies", "levels_64_30"),
    "Lofo": ("the leave-one-family-out vocabularies", "leave_one_family_out"),
    "Recovery": ("label-tree recovery on the rerun models", "label_recovery"),
    "Random": ("the rerun's random partitions and flavour pair", "random_partitions"),
    "FtHeldout": ("JetClass-II fine-tuning on files held out from pretraining",
                  "finetune"),
    "FtRefs": ("fine-tuning from random initialisation and from the self-supervised "
               "model", "finetune_references"),
    "Bench": ("top tagging and quark/gluon tagging (Pythia and Herwig) fine-tuned "
              "from every rerun model", "benchmarks"),
    "Ssl": ("the second and third self-supervised pretraining runs", "self_supervised"),
    "MassLambda": ("the mass-output models with the loss weight matched in loss share",
                   "mass_lambda_matched"),
    "Anomaly": ("anomaly detection on the rerun models, per-run values",
                "anomaly"),
    "RealData": ("the open-data fits on the rerun models", "real_data"),
    "Conclusion": ("written once every input above is in", None),
}

# THE COMMITTED v2 LAYOUT. Every v2 input sits under experiments/FIGS/data/v2/<sub>/, one
# <sub> per V2_PENDING entry, as small JSONs copied from /data unchanged (no npz),
# mirroring the first grid's probe_ladder_*/, mass_resolution/, label_recovery_curve_v1err/,
# paired_v1err/<family>/ and w2b_leg*_metrics files. A <sub> is copied whole, once every run
# it holds has finished: a section is read only complete (v2_section_runs). Names:
#   <run>      the extraction's directory, mtx-<arm lower-cased, no underscores>-s<k>, or
#              init-s<k> for the untrained trunk of run index k (A14's reference)
#   <tag>      best70 (the primary, A14), wavg (robustness, A8), bestval (sensitivity),
#              best70_bn and bestval_bn (the BatchNorm twins, frozen readouts only), init
#   <readout>  features (the class token) or pooled (the pooled embedding, A14)
#
#   pretraining/<run>/best_window_epoch.json, best_epoch.json
#       from /data/results/mtx_v2/<run>/: the run's primary epoch (first maximum within
#       70-79) and its global best, for every run of every grid arm
#   probe_ladder/          the four vocabularies (L188, L162, R42_Q1, R16_Q1), the two
#                          mass-output arms (L162_MASS, R16_Q1_MASS), init-s1..5:
#       probe/<run>/<tag>/<readout>/probe_results.json
#       mass_resolution/<run>/<tag>/<readout>/mass_resolution.json
#           from /data/results/eval/v2/{probe,mass_resolution}/<run>/<tag>/<readout>/
#           (scripts/build_probe_jobs.py --v2); a tag the readouts hold as a link to
#           another (bestval at best70's epoch) stays a link, or a byte-identical copy
#       analysis*/<tag>/<readout>/{seed_level_results.json, mass_resolution_table.json,
#           label_recovery_curve_summary.json}: experiments/STATS/seed_level.py --v2 over
#           every frozen-readout section present (V2_FROZEN) and label_recovery/, into a
#           new directory each time they grow; the one whose inputs are the files present
#           is read (v2_analysis)
#   levels_64_30/          the same probe/ and mass_resolution/ trees, R63_Q1 and R29_Q1
#   leave_one_family_out/  the same, the four *_LOFO4P arms and MPM_LOFO4P
#   random_partitions/     the same, RAND2_p1..5, FLAV_F0, FLAV_F1, FLAV_F1R
#   self_supervised/       the same, MPM (the pooled readout only)
#   mass_lambda_matched/   the same, R16_Q1_MASS_LM; and loss_share.json (SLOTS, A11's realised
#                          shares): experiments/STATS/paired_errors.py a11-shares
#                          --run-dirs-root /data/results/mtx_v2
#   label_recovery/label_recovery_curve/<run>/<tag>/<readout>/label_recovery_curve.json
#       from /data/results/eval/v2/label_recovery_curve/..., every run on the tree
#   paired_errors/<family>/ratios.json, <family> probes, mass, finetune, anomaly
#       experiments/STATS/paired_errors.py ratios over the v2 replicates of every model
#       present (configs/analysis/contrasts.v2.json, with --run-dirs-root: the A7 check)
#   finetune/<rule>[_t12]_leg1_metrics.json, <rule>[_t12]_leg2_metrics.json
#       from /data/results/ft_v2/<rule>[_t12]_<leg>_metrics/ (scripts/build_ft_jobs.py,
#       v2 read-outs, whose headers say experiments/FIGS/data/ft_v2/: this layout moves
#       them here), <rule> best70 or wavg; _t12 the analysis freeze (tiers 1-2, A14),
#       the plain name every tier, read in its place once it exists (v2_ft_files)
#   finetune_references/scratch_leg1_metrics.json   the v2 from-scratch reference's standalone
#                          read-out; the rule read-outs carry the same cells beside their own
#   benchmarks/<rule>[_t12]_bench_metrics{,_herwig}{,_last}.json  (same read-outs)
#   anomaly/<set>/         <set> t12 (the freeze) or t123 (every tier; read in its place, the
#                          runs both hold equal), from scripts/build_anomaly_jobs.py --v2:
#       merged/<tag>/<readout>/anomaly_results.json   /data/results/eval/v2/anomaly_merged_<set>/
#       anomaly_heads.json                            /data/results/eval/v2/anomaly_heads_<set>/
#       summary/<tag>/<readout>/anomaly_summary.json  experiments/EVAL/anomaly_summary.py --grid
#                          configs/arms/v2_grid.json --anomaly merged/<tag>/<readout>/
#                          anomaly_results.json --heads anomaly_heads.json, run on these copies
#   real_data/t12/, t3/    from /data/results/aoj/full_v2/t12/ and t3/ (scripts/build_aoj_jobs.py
#                          --v2): fit_v6/results.json, analysis_v6/aoj_top.json (refit_from_bins.
#                          write_analysis_v2, per_checkpoint), and t12's injection/summary.json; t12
#                          (the freeze) gives the rows of tiers 1-2, t3 (at t12's shape) tier 3's
V2_DATA = ("experiments", "FIGS", "data", "v2")
# The sections whose v2 inputs this script prints, by V2_PENDING key. A section whose v2
# directory exists and is not listed keeps its first-grid numbers, and its marker says so.
V2_READ = {"Pretrain", "Probes", "Paired", "Levels", "Recovery", "FtHeldout", "Random", "Lofo", "Ssl",
           "FtRefs", "Bench", "MassLambda", "Anomaly", "RealData"}
# The frozen-readout sections: probe/ and mass_resolution/ trees, all read by one analysis.
V2_FROZEN = ("probe_ladder", "levels_64_30", "leave_one_family_out", "random_partitions",
             "self_supervised", "mass_lambda_matched")
# {analysis: (per-run file, seed_level.py --v2 output)}
V2_ANALYSES = {"probe": ("probe_results.json", "seed_level_results.json"),
               "mass_resolution": ("mass_resolution.json", "mass_resolution_table.json"),
               "label_recovery_curve": ("label_recovery_curve.json",
                                        "label_recovery_curve_summary.json")}
V2_PRIMARY, V2_TWIN, V2_INIT = "best70", "best70_bn", "init"
V2_CLASS_TOKEN, V2_POOLED = "features", "pooled"
# The readouts the frozen tables print (A14): the primary, its BatchNorm twin beside it,
# and the reference rows, the pooled embedding of every model and the untrained trunk.
V2_FROZEN_NEEDED = ((V2_PRIMARY, V2_CLASS_TOKEN), (V2_TWIN, V2_CLASS_TOKEN),
                    (V2_PRIMARY, V2_POOLED), (V2_INIT, V2_CLASS_TOKEN), (V2_INIT, V2_POOLED))
V2_FT_RULE = "best70"            # fine-tuning starts from the primary (A14, 2026-10-02)
V2_PAIRED_FAMILIES = ("probes", "mass", "finetune", "anomaly")

_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
           "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
_TEX_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


# ------------------------------------------------------------------ rendering

def texname(*parts) -> str:
    """A LaTeX-legal macro name from arbitrary keys: digits become words.

    `\\newcommand` accepts letters only, and the paper has to be able to read
    the name back, so 162 becomes `Onesixtwo` and `bvc_resonant` becomes
    `BvcResonant` rather than either being hashed to something opaque.
    """
    out = []
    for p in parts:
        s = "".join(_DIGITS.get(ch, ch) for ch in str(p))
        out += [c[:1].upper() + c[1:] for c in re.split(r"[^A-Za-z]+", s) if c]
    return "".join(out)


def appendix_module():
    """appendix_tables.py beside this file: the appendix, and the decay-name
    rendering the text shares with it."""
    spec = importlib.util.spec_from_file_location(
        "appendix_tables", pathlib.Path(__file__).resolve().parent / "appendix_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tex(s: str) -> str:
    """Escape a string that came out of a data file for LaTeX."""
    s = str(s).replace("->", "\x00").replace("\u2212", "-").replace("\u00b1", "\x01")
    s = "".join(_TEX_ESCAPE.get(ch, ch) for ch in s)
    return s.replace("\x00", r"$\to$").replace("\x01", r"$\pm$")


def fmt_int(x) -> str:
    return f"{int(round(float(x))):,}".replace(",", "{,}")


def fmt(x, nd: int, sign: bool = False) -> str:
    return f"{float(x):{'+' if sign else ''}.{nd}f}"


def fmt_p(p) -> str:
    """p as the report prints it (3 significant figures), in LaTeX maths."""
    if p is None:
        return "---"
    s = f"{float(p):.3g}"
    if "e" in s:
        m, e = s.split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"${s}$"


def words(n: int) -> str:
    """A count below ten as a word, for captions: 5 -> five."""
    return _DIGITS[str(n)] if 0 <= n < 10 else str(n)


def word_list(ns) -> str:
    """[3] -> 'three'; [1, 3] -> 'one and three'; [1, 2, 3] -> 'one, two and three'."""
    w = [words(n) for n in ns]
    if not w:
        return "none"
    return w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1]


def pdg(sd) -> tuple[float, int]:
    """sd rounded by the Particle Data Group rule, and the decimal place it ends at.

    The three highest significant digits of sd decide: 100-354 keeps two
    significant figures, 355-949 one, and 950-999 rounds up to 1000 of that
    scale and keeps two. A negative place is left of the point: 351 -> (350, -1).
    """
    sd = float(sd)
    e = math.floor(math.log10(sd))
    top = round(sd / 10 ** (e - 2))
    if top >= 1000:                       # 999.6 rounds into the next decade
        top, e = 100, e + 1
    if top <= 354:
        nd = 2
    elif top <= 949:
        nd = 1
    else:
        sd, e, nd = 10.0 ** (e + 1), e + 1, 2
    return round(sd, nd - 1 - e), nd - 1 - e


def _fixed(x, place: int) -> str:
    return f"{round(float(x), place):,.{max(place, 0)}f}".replace(",", "{,}")


def fmt_pm(mean, sd) -> str:
    """mean +- sd, sd PDG-rounded and the mean printed to the same place:
    1377 +- 351 -> 1{,}380 +- 350; 0.997684 +- 0.0000745 -> 0.99768 +- 0.00007.
    Seeds that agree exactly (a spread of zero) print the common value alone."""
    if sd == 0:
        return fmt_one(mean)
    sd, place = pdg(sd)
    return f"{_fixed(mean, place)}\\,$\\pm$\\,{_fixed(sd, place)}"


def fmt_pm_sci(mean, sd) -> str:
    """fmt_pm with a common power of ten taken from the mean: $(2.32 \\pm 0.07)\\times10^{-3}$."""
    if sd == 0:
        return fmt_one_sci(mean)
    e = math.floor(math.log10(abs(float(mean))))
    sd, place = pdg(float(sd) / 10 ** e)
    return f"$({_fixed(float(mean) / 10 ** e, place)} \\pm {_fixed(sd, place)})\\times10^{{{e}}}$"


def fmt_one(x, place: int | None = None) -> str:
    """A single run: three significant figures, or the decimal place given."""
    if place is None:
        place = 2 - math.floor(math.log10(abs(float(x))))
    return _fixed(x, place)


def fmt_one_sci(x) -> str:
    """A single run in scientific form, three significant figures."""
    e = math.floor(math.log10(abs(float(x))))
    return f"${float(x) / 10 ** e:.2f}\\times10^{{{e}}}$"


def _ratio_place(x) -> int:
    e = math.floor(math.log10(abs(float(x))))
    return (2 if f"{float(x):e}"[0] == "1" else 1) - e


def fmt_ratio(x) -> str:
    """A ratio to two significant figures, three when the leading digit is 1, so
    3.7 but 1.02 rather than a 1.0 that hides the difference."""
    return fmt_one(x, _ratio_place(x))


def fmt_paired(ratio, lo, hi) -> str:
    """A paired ratio and its 95 % interval, `3.7~[3.0, 4.5]`: the point to the
    places of fmt_ratio and the interval to the same place, plus one more place
    while rounding would move a bound onto or across 1 -- an interval that
    excludes 1 must not print as if it reached it, nor the reverse -- or onto the
    point, which would print an interval of no width on that side."""
    side = lambda v, p: (round(float(v), p) > 1) - (round(float(v), p) < 1)
    on_point = lambda v, p: float(v) != float(ratio) and round(float(v), p) == round(float(ratio), p)
    place = _ratio_place(ratio)
    while place < 6 and any(side(v, place) != side(v, 15) or on_point(v, place) for v in (lo, hi)):
        place += 1
    return f"{fmt_one(ratio, place)}~[{fmt_one(lo, place)}, {fmt_one(hi, place)}]"


# 95 % confidence upper limit on a Poisson mean given k observed events,
# 0.5 * chi2.ppf(0.95, 2 (k + 1)). Background jets passing a tight cut are few.
POISSON_UP95 = {0: 2.996, 1: 4.744, 2: 6.296, 3: 7.754}


def fmt_rejection(values, bounds, n_pass=None) -> str:
    """Per-seed background rejections as mean +- SD; a bound is never averaged in.

    `rejection_is_bound` means at most one background jet passed the cut (the
    interpolated efficiency is at or below 1/N_B), and probe.py then stores the
    cap N_B, the number of background test jets, in place of a value. Every seed
    bound -> the 95 % confidence lower limit N_B / mu95(k), with k the largest
    number of passing background jets among those seeds (N_B/3 when none passed).
    Only some -> the median over seeds, those seeds at N_B, rounded down, with the
    weaker sign and a footnote.
    """
    if all(bounds):
        k = max((n for n, b in zip(n_pass or [], bounds) if b and n is not None), default=0)
        return "$>$" + fmt_int(math.floor(values[0] / POISSON_UP95[round(k)]))
    if any(bounds):
        return f"$\\geq${fmt_int(math.floor(np.median(values)))}$^{{\\ast}}$"
    return fmt_pm(np.mean(values), np.std(values, ddof=1)) if len(values) > 1 else fmt_one(values[0])


def fmt_auc_pm(aucs, censored) -> str:
    """AUC as mean +- SD over seeds, daggered when it saturated at 1 in any seed."""
    if all(censored):
        return "$1^{\\dagger}$"
    s = fmt_pm(np.mean(aucs), np.std(aucs, ddof=1))
    return s + "$^{\\dagger}$" if any(censored) else s


def fmt_auc(mean, n_censored: int, n_seeds: int = 0) -> str:
    """AUC, daggered when it saturated at 1 in any seed (then 1-AUC is a bound).

    A cell that saturated in EVERY seed prints as $1$, not as 1.00000: five
    decimals on a number that is exactly 1 at the sample's resolution reads as a
    precision the measurement does not have.
    """
    if n_censored and n_censored == n_seeds:
        return "$1^{\\dagger}$"
    return f"{fmt(mean, 5)}$^{{\\dagger}}$" if n_censored else fmt(mean, 5)


def fmt_n_jets(key: str) -> str:
    """A fine-tuning column heading from its summary key: `N10000` -> $10^4$."""
    if not key.startswith("N") or not key[1:].isdigit():
        return tex(key)
    n = int(key[1:])
    p = len(str(n)) - 1
    return f"$10^{{{p}}}$" if n == 10 ** p else f"${fmt_int(n)}$"


def n_tag(key: str) -> str:
    """Macro-name part for a fine-tuning set size: `N10000` -> `Efour` ($10^4$)."""
    if key.startswith("N") and key[1:].isdigit():
        n = int(key[1:])
        p = len(str(n)) - 1
        if n == 10 ** p:
            return texname("E", p)
    return texname(key)


def init_tag(init: str, sizes: dict) -> str:
    """Macro-name part for an initialisation, by vocabulary size where it has one.

    `r16q1-s4` spelled digit by digit is unreadable in the paper source, and the
    arm key is internal anyway: the 17-class model of seed 4 is `OnesevenSfour`.
    """
    m = re.match(r"^([a-z0-9_]+)-s(\d+[a-z]?)$", init)
    if m and m.group(1) in ARM_RUNG:
        return texname(sizes[ARM_RUNG[m.group(1)]], "s" + m.group(2))
    return texname(init)


def init_label(init: str, sizes: dict) -> str:
    """A fine-tuning initialisation, named by vocabulary size rather than arm key."""
    if init in INIT_LABELS:
        return INIT_LABELS[init]
    m = re.match(r"^([a-z0-9_]+)-s(\d+)([a-z]?)$", init)
    if not m:
        return tex(init)
    stem, seed = m.group(1), m.group(2)
    if stem in ARM_RUNG:
        return f"{sizes[ARM_RUNG[stem]]}-class, seed {seed}"
    return f"{INIT_LABELS.get(stem, tex(stem))}, seed {seed}"


# ------------------------------------------------------------------ provenance

_TEXT_MINUS = re.compile(r"(?<![\w{\\\-])-(?=\d)")


def math_safe(body: str) -> str:
    """`$x$` -> `\\ensuremath{x}`, so a macro reads the same in text and in maths.

    The formatters write `$...$` because the tables are text. A macro is also
    used inside the prose's own maths -- `$p=\\TrendPCone$` -- where a bare `$`
    closes the formula and LaTeX stops with "Missing $ inserted". A leading
    minus becomes a maths minus too; in text an ASCII '-' prints as a hyphen.
    """
    parts = body.split("$")
    return "".join(_TEXT_MINUS.sub(lambda m: "\\ensuremath{-}", x) if i % 2 == 0
                   else "\\ensuremath{" + x + "}" for i, x in enumerate(parts))


def text_minus(line: str) -> str:
    """A table line with every minus outside `$...$` typeset as a maths minus."""
    parts = line.split("$")
    return "$".join(_TEXT_MINUS.sub("$-$", x) if i % 2 == 0 else x for i, x in enumerate(parts))


class Emitter:
    """The only way a number is allowed to reach the paper.

    Every call records the file, the path inside it and the file's hash, so the
    generated .tex and provenance.json cannot drift apart: they are two views of
    this one list.
    """

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.macros: list[tuple[str, str, str]] = []     # name, body, comment
        self.provenance: dict[str, dict] = {}
        self._sha: dict[str, str] = {}

    def sha256(self, path: pathlib.Path) -> str:
        key = str(path)
        if key not in self._sha:
            self._sha[key] = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        return self._sha[key]

    def macro(self, name: str, body: str, source: pathlib.Path, json_path: str,
              note: str = "") -> str:
        if name in self.provenance:
            raise SystemExit(f"FATAL: macro \\{name} emitted twice; the second value would "
                             f"silently win in LaTeX")
        body = math_safe(body)
        sha = self.sha256(source)
        rel = str(pathlib.Path(source).relative_to(self.root))
        comment = f"{rel} :: {json_path} :: sha256 {sha[:16]}" + (f" :: {note}" if note else "")
        self.macros.append((name, body, comment))
        self.provenance[name] = {"source_file": rel, "json_path": json_path,
                                 "sha256": sha, "value": body}
        return body


# ------------------------------------------------------------------ inputs

# The paired-ratio files (experiments/FIGS/data/paired_v1err/<family>/ratios.json)
# and the metric of the entries the text can quote from each family.
PAIRED_FAMILIES = ("probes", "finetune", "mass")
PAIRED_METRIC = {"probe": "1-auc", "ft": "1-macro_auc", "mass": "sigma_eff"}
# The fine-tuning legs as the paired file names them: leg 1 is the JetClass-II
# task (w2b_leg1_metrics_v2), leg 2 JetClass (w2b_leg2_metrics_v2); checked
# against the class counts the metrics files record (emit_paired).
PAIRED_FT_DATASET = {"leg1": "Jcii", "leg2": "Jc"}


def input_paths(root: pathlib.Path) -> dict:
    # THE PROBE RE-RUN IS CANONICAL, not the original run beside it. It is a
    # strict superset -- the same four vocabularies, the same five seeds, the
    # same tasks and the same jets, plus the 70 % and 90 % working points that
    # docs/PRESPEC_2026-09.md fixed once 50 % proved censored -- and it
    # reproduces the original's confirmatory p-value to every digit printed.
    # The paper's headline background rejection exists only here. The original
    # run stays on disk, frozen, as the provenance record of what ran first.
    data = root / "experiments" / "FIGS" / "data"
    maps = root / "configs" / "labelmaps"
    # The *_mlp2 files are those runs with the MLP probe refitted to convergence
    # (probe.py --mlp-rerun-of): their linear blocks are the originals' byte for
    # byte, and their MLP blocks replace fits that stopped at a 60-epoch cap.
    return {"ladder": sorted((data / "probe_ladder_v2_mlp2").glob("s*.json"))
                      + sorted((data / "probe_ladder_mass2x2_mlp2").glob("s*.json")),
            # analysis_with_c5, not analysis: the same run of the same script over
            # the same five ladder files, plus the mass-output 2x2 that makes C5
            # measurable. Verified a strict superset before it was adopted -- C1's
            # p reproduces to all sixteen digits and every other field is
            # identical; only the C5 block, the Holm family it joins and the
            # provenance differ. The earlier directory is kept as the record of
            # what the tables said while C5 was still pending.
            # analysis_family_of_four supersedes it (amendment of 2026-09-22): the
            # same inputs, with C4 outside the Holm count and S2's p the
            # intersection-union maximum. Verified field by field: only the Holm
            # families, the S2 p and bound fields and C1's family-size wording differ.
            "analysis": data / "probe_ladder_v2_mlp2" / "analysis" / "seed_level_results.json",
            "leg1": data / "leg1_metrics.json",
            "leg2": data / "leg2_metrics.json",
            # The learning curve, not the single-size fit beside it: a class-weighted
            # logistic regression on nested training sets up to every jet not held out
            # (experiments/EVAL/label_recovery_curve.py; it supersedes the unweighted fit
            # on 126,000 jets of label_recovery_ladder_v1, kept as the record).
            "recovery": data / "label_recovery_curve_v1err" / "analysis" / "label_recovery_curve_summary.json",
            # Each of these is the latest pass of its analysis. Every earlier pass
            # beside it is kept and differs only where its PRESPEC entry says:
            # analysis_v2 of C4 adds the post-hoc grouping cost; analysis_v2 of
            # S3/S4 and of section 5 fix wording; analysis_holm of S7 applies the
            # Holm correction amendment A4 requires; analysis_labelled of the real
            # data fixes one verdict's wording.
            "random_control": data / "probe_ladder_randcontrol_mlp2" / "analysis" / "c4_random_control.json",
            # The +mass arms' probe files alone, for the mass-output cells.
            "mass2x2": sorted((data / "probe_ladder_mass2x2_mlp2").glob("s*.json")),
            # The per-cell fine-tuning metrics themselves (fine-tuning wave 2b); which
            # file is which dataset is read from the class count its cells record.
            # v2 rereads every v1 cell plus the random-label draws 2 and 3.
            "ft_legs": [data / "w2b_leg1_metrics_v2.json", data / "w2b_leg2_metrics_v2.json"],
            # anomaly_v1err: the same per-model results as anomaly_merged_v4/analysis_v4
            # (its epoch-79 cells reproduce them, checkpoint_rule.epoch79_reproduces_committed),
            # plus the output-layer score of every epoch 70-79 (commit 54854e9), which
            # tells a single epoch's output layer apart from the vocabulary.
            "anomaly": data / "anomaly_v1err" / "analysis" / "anomaly_summary.json",
            "mass_resolution": data / "mass_resolution" / "analysis_holm" / "s7_mass_resolution.json",
            # analysis_v6: one peak shape shared by the pretrained models and the tops
            # that fail the cut (fit_v6, experiments/AOJ/fit_v6.py; DECISIONS_PENDING,
            # 2026-10-01). Its provenance names the fit by the path it ran at
            # (/scratch/fit_v6/...); committed_path maps it to the committed copy by hash.
            "real_data": data / "aoj_full_v1" / "analysis_v6" / "aoj_top.json",
            "aoj_injection": data / "aoj_injection_v1" / "summary.json",
            "aoj_closure": data / "aoj_checks_v2" / "closure_all_shards.json",
            # The per-seed cells of the |V_cb| window probe, the file its analysis
            # read: they carry the surviving background counts the analysis drops.
            "vcb": data / "probe_ladder_vcbwindow_mlp2" / "sall.json",
            "probe_code": root / "experiments" / "EVAL" / "probe.py",
            "design_spec": root / "experiments" / "MTX" / "k8s" / "job-mtx-l188-s1-raunav.yaml",
            "design_arch": root / "experiments" / "E1" / "ParT_sophon_arch_10c.py",
            "design_arm": root / "configs" / "arms" / "L188.yaml",
            "literature": data / "literature_facts.json",
            # collected 2026-09-29 (collect_recipes.py over w2b, bench_v2, bench_v3)
            "ft_recipes": data / "ft_recipes" / "recipes_w2b_bench_v2_v3.json",
            "ft_leg_specs": sorted((root / "experiments" / "FT" / "k8s").glob("job-ft-legs-w[23]*-raunav.yaml")),
            "mass_specs": sorted((root / "experiments" / "MTX" / "k8s").glob("job-mtx-*_mass-s[1-5]-raunav.yaml")),
            "survival": maps / "usecase_survival.v1.json",
            "rung_map": maps / "rung_label_maps.v1.csv",
            # Paired ratios (experiments/STATS/paired_errors.py ratios): run k with run
            # k, one bootstrap of the test jets shared by both models and every run.
            # Every ratio the text quotes comes from here, not from a ratio of means.
            "paired": [data / "paired_v1err" / k / "ratios.json" for k in PAIRED_FAMILIES],
            # The rerun's design, read where it is fixed: the grid, the partition and
            # flavour-pair maps and the matched mass weight.
            "v2_grid": root / "configs" / "arms" / "v2_grid.json",
            # the rerun's statistics: its contrasts, labels and untrained-trunk references
            "contrasts_v2": root / "configs" / "analysis" / "contrasts.v2.json",
            "mass_lambda": root / "configs" / "arms" / "v2" / "mass_lambda.v2.json",
            "rand_v1_map": maps / "rand_label_map.v1.csv",
            "rand_v2_map": maps / "rand_label_map.v2.csv",
            "rand_v2_rule": maps / "rand_v2_selection.json",
            "flavour_pair": maps / "flavour_pair.v2.json",
            "flavour_map": maps / "flavour_pair_map.v2.csv",
            "realised_shares": maps / "realised_native_shares.v2.json",
            "design_spec_v2": root / "experiments" / "MTX" / "k8s" / "v2" / "grid"
                              / "job-mtx2-l188-s1-raunav.yaml",
            "lofo_spec_v2": root / "experiments" / "MTX" / "k8s" / "v2" / "grid"
                            / "job-mtx2-l188lofo4p-s1-raunav.yaml",
            "pretrain_v2": root / "experiments" / "MTX" / "pretrain_v2.py",
            # The smallest effect of interest, fixed in the plan: the band within which a
            # result is called robust to the checkpoint.
            "mei_code": root / "src" / "stats" / "mde.py",
            # The loader dry runs that fixed the rerun's reading (committed with their
            # hashes, experiments/FIGS/data/v2_loader/provenance.json): the final loader
            # and the one that leaves a family out.
            "v2_dryrun": data / "v2_loader" / "loader_dryrun" / "dryrun_s176_seed1.json",
            "v2_dryrun_lofo": data / "v2_loader" / "loader_dryrun" / "dryrun_lofo4p_seed1.json",
            # Which GPU product each run index of the rerun trains on: the grid's job specs.
            "v2_grid_specs": sorted((root / "experiments" / "MTX" / "k8s" / "v2" / "grid")
                                    .glob("job-mtx2-*-s*-raunav.yaml"))}


def check_alignment(paths: dict, root: pathlib.Path) -> str:
    """Refuse unless every input that names a row alignment names the same one.

    `row_alignment_sha256` is the hash of the label vector the arms were scored
    against. Two different values mean two different jet orders, and a paired
    contrast across them is comparing different jets -- an error that produces
    perfectly plausible numbers, so it has to stop the run.
    """
    seen: dict[str, list[str]] = {}
    for p in paths["ladder"]:
        h = json.loads(p.read_text()).get("row_alignment_sha256")
        if h:
            seen.setdefault(h, []).append(str(p.relative_to(root)))
    for key in ("analysis", "leg1", "recovery", "random_control", "mass_resolution", "anomaly"):
        p = paths[key]
        if not pathlib.Path(p).exists():
            continue
        d = json.loads(pathlib.Path(p).read_text())
        h = d.get("row_alignment_sha256") or d.get("provenance", {}).get("row_alignment_sha256")
        if h:
            seen.setdefault(h, []).append(str(pathlib.Path(p).relative_to(root)))
    if len(seen) > 1:
        detail = "; ".join(f"{h[:16]}: {', '.join(f)}" for h, f in sorted(seen.items()))
        raise SystemExit(f"FATAL: inputs disagree on row_alignment_sha256 ({detail}). The arms "
                         f"were not scored on the same jets in the same order; no table from "
                         f"these files is a paired contrast.")
    if not seen:
        raise SystemExit("FATAL: no input carries a row_alignment_sha256; refusing to build a "
                         "paired table whose alignment nothing asserts")
    return next(iter(seen))


def check_analysis_is_current(analysis: dict, root: pathlib.Path) -> None:
    """The analysis file records the hash of every ladder file it read. Hold it.

    If a ladder file has been rewritten since, the tests in the analysis file
    describe data that is no longer on disk, and the per-seed points a figure
    draws would come from the newer file than the p-value beside them.
    """
    for rec in analysis.get("provenance", {}).get("inputs", []):
        p = root / rec["path"]
        if not p.exists():
            raise SystemExit(f"FATAL: {rec['path']} was read by the analysis and is now missing")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != rec["sha256"]:
            raise SystemExit(f"FATAL: {rec['path']} changed since the analysis ran "
                             f"({rec['sha256'][:16]} -> {got[:16]}). Re-run "
                             f"experiments/STATS/seed_level.py before regenerating the tables.")


def committed_path(root: pathlib.Path, rel: str, sha: str | None = None) -> pathlib.Path:
    """The file an analysis names, in the repository. The real-data readouts ran on a
    scratch copy of their fit (/scratch/fit_vN/<file>); the committed copy is
    experiments/FIGS/data/aoj_full_v1/fit_vN/<file>, accepted only if its hash is the one
    the analysis recorded. Any other path is taken relative to the root as it stands."""
    p = root / rel
    m = re.fullmatch(r"/scratch/(fit_v\d+)/([^/]+)", rel)
    if p.exists() or not m:
        return p
    q = root / "experiments" / "FIGS" / "data" / "aoj_full_v1" / m.group(1) / m.group(2)
    if q.exists() and (sha is None or hashlib.sha256(q.read_bytes()).hexdigest() == sha):
        return q
    return p


def check_inputs_unchanged(analysis_path: pathlib.Path, root: pathlib.Path) -> None:
    """The later analysis files each record the hash of what they read, in one of
    two shapes ({path, sha256} anywhere, or input + input_sha256). Hold them all:
    a result file rewritten after its analysis ran would put a table and the test
    beside it on different data."""
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str):
                yield x["path"], x["sha256"]
            if isinstance(x.get("input"), str) and isinstance(x.get("input_sha256"), str):
                yield x["input"], x["input_sha256"]
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
    d = json.loads(pathlib.Path(analysis_path).read_text())
    seen = list(walk(d))
    if not seen:
        raise SystemExit(f"FATAL: {analysis_path} records no input hash; refusing to trust it")
    for rel, sha in seen:
        p = committed_path(root, rel, sha)
        if not p.exists():
            raise SystemExit(f"FATAL: {rel}, read by {analysis_path}, is missing")
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"FATAL: {rel} changed since {analysis_path} was written; re-run "
                             f"experiments/STATS/seed_level.py")


def vocabulary_sizes(rung_map: pathlib.Path) -> dict:
    """Classes per rung, counted from the label map -- the sizes the paper quotes.

    The config keys are named for their resonant nodes only (`R42_Q1` has 42
    resonant groups plus one QCD group = 43 classes), so the number in the name
    is NOT the number of classes. Counting the column is the only safe source.
    """
    with pathlib.Path(rung_map).open() as f:
        rows = list(csv.DictReader(f))
    missing = [r for r in RUNGS if r not in rows[0]]
    if missing:
        raise SystemExit(f"FATAL: rungs {missing} missing from {rung_map}")
    return {r: len({row[r] for row in rows}) for r in RUNGS}


def ordered_tasks(keys) -> list:
    """Tasks in the paper's reading order, with anything unknown appended.

    The order of `levels` in the analysis file is the order seed_level.py
    happened to iterate; the table should not inherit it.
    """
    keys = set(keys)
    return [t for t in TASK_LABELS if t in keys] + sorted(keys - set(TASK_LABELS))


def seed_rows(table: list[dict], task: str, probe: str, level: int) -> list:
    """Per-seed cells straight from the analysis table, in seed order."""
    rows = [r for r in table
            if r["task"] == task and r["probe"] == probe and r["level"] == level
            and not r.get("dropped_pair")]
    return sorted(rows, key=lambda r: r["seed"])


# ------------------------------------------------------------------ macros

def headline_rejection(row: dict) -> tuple:
    """(key, signal efficiency) of the working point the paper quotes, the one
    docs/PRESPEC_2026-09.md fixed blind: 90 % signal efficiency.

    The flat `rejection_*` fields sit at the probe's DEFAULT working point,
    50 %, where no background jet survives at the three finer vocabularies and
    the number is a statement about the size of the test sample rather than
    about the models. An analysis file written before the working points were
    recorded has no `rejection_points`; the key is then None, the flat fields
    are read, and the efficiency says which point the reader is looking at.
    """
    eps = row.get("headline_eps_s")
    if eps in (row.get("rejection_points") or {}):
        return eps, float(eps)
    return None, float(row["rejection_eps_s"])


def seed_rejections(table: list[dict], task: str, probe: str, level: int, key) -> tuple:
    """Per-seed rejections and their is-bound flags at working point `key`."""
    rows = seed_rows(table, task, probe, level)
    if key is None:
        return ([r["rejection"] for r in rows], [r["rejection_is_bound"] for r in rows],
                [r.get("n_bkg_pass") for r in rows])
    pts = [r["rejection_points"][key] for r in rows]
    return [p["rejection"] for p in pts], [p["is_bound"] for p in pts], [p.get("n_bkg_pass") for p in pts]


def emit_design(em: Emitter, A: dict, src: pathlib.Path) -> None:
    p = A["provenance"]
    em.macro("ProbeNJets", fmt_int(p["n_jets_total"]), src, "provenance.n_jets_total")
    em.macro("ProbeNSeeds", str(len(A["seeds_used"])), src, "seeds_used (length)")
    em.macro("ProbeNSeedsWord", words(len(A["seeds_used"])), src, "seeds_used (length)",
             "pretraining runs per vocabulary, as a word for the prose")
    em.macro("ProbeRowAlign", f"\\texttt{{{p['row_alignment_sha256'][:16]}}}", src,
             "provenance.row_alignment_sha256 (first 16)")
    t0 = sorted(A["levels"])[0]
    _, eps = headline_rejection(A["levels"][t0]["linear"][0])
    em.macro("ProbeEpsS", f"{eps * 100:.0f}\\%", src,
             f"levels.{t0}.linear[0].headline_eps_s")


def emit_levels(em: Emitter, A: dict, src: pathlib.Path) -> None:
    """One macro per measured number in the per-granularity summary.

    AUC, 1-AUC and the headline rejection are the mean and SD of the per-seed
    table rows. The mean AUC is cross-checked against the summary block: a
    disagreement means the table rows and the summary came from different
    reads, which is worth a crash rather than a rounding argument.
    """
    for task in sorted(A["levels"]):
        for probe in sorted(A["levels"][task]):
            oma = {}
            for i, row in enumerate(A["levels"][task][probe]):
                if not row["n_seeds"]:
                    continue
                lv, key = row["level"], texname(task, probe, row["level"])
                jp, tp = f"levels.{task}.{probe}[{i}]", f"table[{task},{probe},{lv}]"
                rows = seed_rows(A["table"], task, probe, lv)
                aucs = [r["auc"] for r in rows]
                if abs(float(np.mean(aucs)) - row["mean_auc"]) > 1e-12:
                    raise SystemExit(f"FATAL: {jp}.mean_auc disagrees with the mean of the "
                                     f"per-seed table rows for {task}/{probe}/{lv}")
                em.macro("ProbeAuc" + key,
                         fmt_auc(row["mean_auc"], row["n_censored"], row["n_seeds"]), src,
                         jp + ".mean_auc",
                         f"{row['n_seeds']} seeds, {row['n_censored']} saturated at AUC=1")
                em.macro("ProbeAucSd" + key, fmt(np.std(aucs, ddof=1), 5) if len(aucs) > 1
                         else "---", src, tp + ".auc (SD over seeds, ddof=1)")
                em.macro("ProbeLogOneMinusAuc" + key, fmt(row["mean"], 4, sign=True), src,
                         jp + ".mean", "natural log of 1-AUC, lower is better")
                em.macro("ProbeLogOneMinusAucSd" + key,
                         fmt(row["seed_sd"], 4) if row["seed_sd"] is not None else "---", src,
                         jp + ".seed_sd")
                if any(r["censored"] for r in rows):
                    em.macro("ProbeOma" + key, "---", src, tp + ".auc",
                             "not quoted: the AUC reached 1 in a seed, so 1-AUC is only a bound")
                else:
                    x = [1.0 - a for a in aucs]
                    oma[lv] = float(np.mean(x))
                    em.macro("ProbeOma" + key, fmt_pm_sci(np.mean(x), np.std(x, ddof=1)), src,
                             tp + ".auc", f"1-AUC, mean +- SD over {len(x)} seeds")
                pt = (row.get("rejection_points") or {}).get(row.get("headline_eps_s"))
                if pt and "mean_n_bkg_pass" in pt:
                    em.macro("ProbeBkgLeft" + key, fmt(pt["mean_n_bkg_pass"], 1), src,
                             jp + f".rejection_points['{row['headline_eps_s']}'].mean_n_bkg_pass",
                             "background jets passing the cut, mean over seeds")
                k_eps, eps = headline_rejection(row)
                vals, bounds, n_pass = seed_rejections(A["table"], task, probe, lv, k_eps)
                em.macro("ProbeRej" + key, fmt_rejection(vals, bounds, n_pass), src,
                         tp + (f".rejection_points['{k_eps}'].rejection" if k_eps else ".rejection"),
                         f"at {eps:.0%} signal efficiency, mean +- SD over seeds; "
                         f"{sum(bounds)} of {len(bounds)} seeds at the cap")
            for fine in oma:
                for coarse in oma:
                    if coarse < fine:
                        em.macro("ProbeOmaRatio" + texname(task, probe, coarse) + "Over"
                                 + texname(fine), fmt_ratio(oma[coarse] / oma[fine]), src,
                                 f"table[{task},{probe},{coarse}/{fine}].auc",
                                 "ratio of the seed means of 1-AUC, coarser over finer")


def emit_probe_settings(em: Emitter, path: pathlib.Path) -> None:
    """The probe split and fit settings, read from experiments/EVAL/probe.py, which sets them."""
    s = path.read_text()

    def one(pat):
        m = re.findall(pat, s, re.M)
        if len(m) != 1:
            raise SystemExit(f"FATAL: {path} matches {pat} {len(m)} times; expected once")
        return m[0]
    a, v = (float(x) for x in one(r"^SPLIT_FRACTIONS = \(([0-9.]+), ([0-9.]+)\)"))
    for name, frac in (("Train", a), ("Val", v), ("Test", 1 - a - v)):
        em.macro("ProbeSplit" + name, f"{100 * frac:.0f}\\%", path, "SPLIT_FRACTIONS",
                 "share of the probe sample; the same jets for every model")
    em.macro("ProbeMlpHidden", one(r"torch\.nn\.Linear\(tr\.shape\[1\], (\d+)\)"), path,
             "_fit_mlp", "hidden width of the MLP probe")
    seeds = [x for x in one(r"^MLP_SEEDS = \(([^)]*)\)").split(",") if x.strip()]
    em.macro("ProbeMlpInits", str(len(seeds)), path, "MLP_SEEDS",
             "MLP initialisations whose scores are averaged per cell")
    factor = float(one(r'"lr_factor": ([0-9.]+)'))
    lr_pat, stop = int(one(r'"lr_patience": (\d+)')), int(one(r'"stop_patience": (\d+)'))
    if stop % (lr_pat + 1):
        raise SystemExit(f"FATAL: {path} stop_patience {stop} is not a whole number of "
                         f"learning-rate patience windows ({lr_pat} + 1)")
    em.macro("ProbeMlpLrDrop", f"{1 / factor:g}", path, "MLP_SCHEDULE['lr_factor']",
             "factor the MLP's learning rate drops by on a plateau")
    em.macro("ProbeMlpPlateaus", words(stop // (lr_pat + 1)), path,
             "MLP_SCHEDULE: stop_patience / (lr_patience + 1)",
             "patience windows without gain after which the MLP stops")
    grid = [float(x) for x in one(r"^C_GRID = \[([^\]]*)\]").split(",")]
    for name, c in (("ProbeCMin", min(grid)), ("ProbeCMax", max(grid))):
        e = round(math.log10(c))
        if not math.isclose(c, 10.0 ** e):
            raise SystemExit(f"FATAL: {path} C_GRID end {c} is not a power of ten")
        em.macro(name, f"$10^{{{e}}}$", path, "C_GRID",
                 "end of the logistic-regression inverse-regularisation grid")


def emit_vocabulary(em: Emitter, sizes: dict, src: pathlib.Path) -> None:
    for rung, n in sizes.items():
        em.macro("VocabSize" + texname(rung), str(n), src,
                 f"column {rung} (distinct group count)")


# Table 1: the published discriminants it prints, with the label each is given,
# and the ones in configs/labelmaps/usecase_survival.v1.json it leaves out and why.
# A row in neither dict stops the build, so nothing is dropped silently.
USECASE_ROWS = {
    "sophon_eq4": "$X\\to b\\bar b$ vs QCD",
    "sophon_eq6_a1": "$X\\to c\\bar s$ vs QCD (Sophon's $A_1$)",
    "sophon_eq6_a3": "three quarks, exactly one $b$, vs QCD (Sophon's $A_3$)",
    "sophon_eq7_9": "$W'\\to W\\phi\\to WWW$ event discriminant",
    "vcb_eq1": "$D_{bc}$, boosted $|V_{cb}|$ measurement",
}
USECASE_LEFT_OUT = {
    # ATL-PHYS-PUB-2026-013 p.10 Eq. (1): D_s = log[p_s / (sum_b f_b p_b + ...)]
    # with f_QCD "applied to all QCD classes". It needs only the QCD sum, so over
    # these labels it is the X->bb row again; the per-subclass rejection the row
    # encoded is an evaluation of the tagger, not a discriminant built from it.
    "gn3x_qcd_subclass": "reads as the X->bb row: one QCD fraction for every subclass",
    # Not a published discriminant. W-vs-Z is never expressible over labels that
    # name decay products; the W-like score of Sophon Eq. (9) is, and the text
    # gives where it survives (UseWlike*, from wlike_survival below).
    "w_vs_qcd_resonance": "not published; stated in the text",
}
# Sophon Eq. (9): g_W(2) = g_{X->cs} + g_{X->qq}, the W-like two-prong score.
WLIKE_CLASSES = ("label_X_cs", "label_X_qq")
USECASE_CITE = {"arXiv:2405.12972": "sophon", "arXiv:2503.00118": "vcbboosted"}


def usecase_rows(surv: dict) -> list:
    unknown = set(surv) - set(USECASE_ROWS) - set(USECASE_LEFT_OUT)
    if unknown:
        raise SystemExit(f"FATAL: Table 1 has no decision for {sorted(unknown)}: add each to "
                         f"USECASE_ROWS or USECASE_LEFT_OUT with its reason")
    return [d for d in USECASE_ROWS if d in surv]


def wlike_survival(rung_map: pathlib.Path) -> dict | None:
    """{rung: constructible} for Sophon's W-like score g_W(2) / (g_W(2) + sum g_QCD).

    Sophon's class-division property (arXiv:2405.12972 Eq. 2): a coarse class's
    score is the sum of the scores it merged, so the ratio is buildable at a level
    iff its numerator set and its denominator set are each a union of groups there.
    None when the map does not hold its classes (a synthetic map).
    """
    with pathlib.Path(rung_map).open() as f:
        rows = list(csv.DictReader(f))
    if not set(WLIKE_CLASSES) <= {r["class_name"] for r in rows}:
        return None
    num = set(WLIKE_CLASSES)
    den = num | {r["class_name"] for r in rows if r["class_name"].startswith("label_QCD_")}
    out = {}
    for rung in RUNGS:
        groups = collections.defaultdict(set)
        for r in rows:
            groups[r[rung]].add(r["class_name"])
        out[rung] = all(g <= s or not (g & s) for s in (num, den) for g in groups.values())
    return out


def emit_usecase(em: Emitter, surv: dict, sizes: dict, src: pathlib.Path,
                 rung_map: pathlib.Path) -> None:
    for disc in usecase_rows(surv):
        row, key = surv[disc], texname(disc)
        dies = row["dies_at"]
        em.macro("UseDiesAt" + key, "survives" if dies is None else str(sizes[dies]), src,
                 f"{disc}.dies_at", "first vocabulary size at which it is not constructible")
        alive = row["last_rung_alive"]
        em.macro("UseLastAlive" + key, "none" if alive is None else str(sizes[alive]), src,
                 f"{disc}.last_rung_alive")
    w = wlike_survival(rung_map)
    if w is None:
        return
    alive = [r for r in RUNGS if w[r]]
    if not alive or alive != RUNGS[:len(alive)]:
        raise SystemExit(f"FATAL: the W-like score is constructible at {alive}, not at a run of "
                         f"the finest levels; the text's reading of it no longer holds")
    em.macro("UseWlikeLastAlive", str(sizes[alive[-1]]), rung_map,
             "groups at each level vs {label_X_cs, label_X_qq} and that plus the QCD classes",
             "last level at which Sophon's W-like score g_W(2) is constructible")
    em.macro("UseWlikeDiesAt", str(sizes[RUNGS[len(alive)]]) if len(alive) < len(RUNGS)
             else "survives", rung_map,
             "groups at each level vs {label_X_cs, label_X_qq} and that plus the QCD classes",
             "first level at which it is not")


def emit_legs(em: Emitter, legs: dict, src_by_leg: dict, sizes: dict) -> None:
    """Fine-tuning accuracies. WAVE 1, SUPERSEDED -- said on every line."""
    for leg, d in legs.items():
        src = src_by_leg[leg]
        for init in sorted(d["summary"]):
            for n in sorted(d["summary"][init]):
                cell = d["summary"][init][n]
                key = texname("Leg", leg) + init_tag(init, sizes) + n_tag(n)
                em.macro("Acc" + key, fmt(cell["accuracy_mean"], 5), src,
                         f"summary.{init}.{n}.accuracy_mean",
                         f"wave 1, superseded; {cell['n_seeds']} fine-tuning seeds")
                sd = cell.get("accuracy_sd")
                em.macro("AccSd" + key, fmt(sd, 5) if sd else "---", src,
                         f"summary.{init}.{n}.accuracy_sd", "wave 1, superseded")


# Numbers the text has a slot for and another piece of work supplies. Each slot
# is a macro: the value once its JSON exists, a red marker naming the input until
# then. (macro, JSON glob relative to the root, key inside it, formatter, what the
# marker says).
#   The open-data selection chain (the real-data checks against fit_v6,
#   aoj_checks_v2): the dataset, the staged jets, the rho window, the jets in the
#   mass fit and its pT categories.
#   The REALISED mass-loss share of the rerun's mass-output runs (the check that
#   the matched weight did its job, PRESPEC A11 correction): per run, x/(1+x) with
#   x = lambda * L_reg / L_cls averaged over the run's epochs -- the definition the
#   first-grid shares and the matched weight use (emit_mass_lambda_matched).
#   The native-class coverage of the 10^3-jet JetClass-II fine-tuning subset (the
#   seed-1 draw): of the rerun's held-out subsets, from the manifest make_subsets.py
#   jc2v2 wrote (committed at experiments/FT/data/jc2_v2_manifest.json); and of the
#   first set's subsets, drawn from the pretraining files, which make_subsets.py jc2
#   wrote without a coverage record -- counted with make_subsets.class_coverage from
#   /data/finetune/jc2/train_N1000_s1.parquet, in the same schema, when it is.
SLOTS = [
    ("AojChainDataset", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "n_dataset", "millions",
     "jets in the open-data sample"),
    ("AojChainStaged", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "n_staged", "int",
     "jets after the kinematic selection"),
    ("AojChainRhoWindow", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "n_rho_window", "int",
     "jets inside the rho window"),
    ("AojNJetsFit", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "n_fit", "int",
     "jets in the mass fit"),
    ("AojChainPtBins", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "n_pt_bins", "int",
     "pT categories of the fit"),
    ("AojChainEtaMax", "experiments/FIGS/data/aoj_checks_v2/selection_chain_v2.json", "abs_eta_max", "one",
     "the pseudorapidity cut"),
    ("MassLossShareVtwoOnesixtwoMass", "experiments/FIGS/data/v2/mass_lambda_matched/loss_share.json",
     "shares.162+mass", "percent_pm", "realised mass-loss share in the rerun, 162 classes"),
    ("MassLossShareVtwoOnesevenMass", "experiments/FIGS/data/v2/mass_lambda_matched/loss_share.json",
     "shares.17+mass", "percent_pm", "realised mass-loss share in the rerun, 17 classes"),
    ("MassLossShareVtwoOnesevenMassMatched",
     "experiments/FIGS/data/v2/mass_lambda_matched/loss_share.json",
     "shares.17+mass_matched", "percent_pm",
     "realised mass-loss share in the rerun, 17 classes, matched weight"),
    ("FtCoverageEThreeNClasses", "experiments/FT/data/jc2_v2_manifest.json",
     "class_coverage['train_N1000_s1.parquet'].n_classes_present", "int",
     "native classes present in the smallest held-out fine-tuning subset"),
    ("FtCoverageEThreeMedian", "experiments/FT/data/jc2_v2_manifest.json",
     "class_coverage['train_N1000_s1.parquet'].median_per_present_class", "g",
     "median jets per class in the smallest held-out fine-tuning subset"),
    ("FtCoverageEThreeOneJet", "experiments/FT/data/jc2_v2_manifest.json",
     "class_coverage['train_N1000_s1.parquet'].n_classes_with_one_jet", "int",
     "classes with a single jet in the smallest held-out fine-tuning subset"),
    ("FtCoverageVoneEThreeNClasses", "experiments/FT/data/jc2_v1_coverage.json",
     "class_coverage['train_N1000_s1.parquet'].n_classes_present", "int",
     "native classes present in the first set's smallest fine-tuning subset"),
    ("FtCoverageVoneEThreeMedian", "experiments/FT/data/jc2_v1_coverage.json",
     "class_coverage['train_N1000_s1.parquet'].median_per_present_class", "g",
     "median jets per class in the first set's smallest fine-tuning subset"),
    ("FtCoverageVoneEThreeOneJet", "experiments/FT/data/jc2_v1_coverage.json",
     "class_coverage['train_N1000_s1.parquet'].n_classes_with_one_jet", "int",
     "classes with a single jet in the first set's smallest fine-tuning subset"),
]


def emit_slots(em: Emitter, missing: list) -> None:
    """Fill each SLOTS macro from its JSON, or mark it pending (never a guess)."""
    me = pathlib.Path(__file__).resolve()
    for name, pattern, key, form, what in SLOTS:
        hits = sorted(em.root.glob(pattern))
        if len(hits) > 1:
            raise SystemExit(f"FATAL: {pattern} matches {len(hits)} files; the slot {name} "
                             f"needs exactly one")
        if not hits:
            if me.is_relative_to(em.root.resolve()):
                em.macro(name, f"\\pending{{pending: {what}}}", me, f"SLOTS[{name}]",
                         f"a marker, not a number: {pattern} :: {key} does not exist yet")
            missing.append(f"{name} -- {pattern} :: {key}")
            continue
        v = pick(json.loads(hits[0].read_text()), key)
        body = {"int": lambda x: fmt_int(x),
                "millions": lambda x: fmt_int(float(x) / 1e6),
                "one": lambda x: fmt_one(x),
                "g": lambda x: f"{float(x):g}",
                "percent_pm": lambda x: fmt_pm(100 * np.mean(x), 100 * np.std(x, ddof=1)) + "\\%"}[form](v)
        em.macro(name, body, hits[0], key)


def v2_present(root: pathlib.Path, sub: str) -> bool:
    d = root / "experiments" / "FIGS" / "data" / "v2" / sub
    return d.is_dir() and any(d.iterdir())


# A section V2_READ lists whose readout lives in another section's files waits for it: the
# models that leave a family out are read in the anomaly study (A13), which is merged after
# their frozen readouts are copied.
V2_READ_WITH = {"Lofo": "Anomaly"}


def v2_printed(root: pathlib.Path) -> set:
    """The V2_PENDING keys this script prints from v2 now: read (V2_READ), present, and not
    waiting for the section V2_READ_WITH names."""
    present = {k for k, (_, sub) in V2_PENDING.items() if sub and v2_present(root, sub)}
    return {k for k in present & V2_READ if V2_READ_WITH.get(k, k) in present}


def emit_pending(em: Emitter, missing: list) -> None:
    """One red marker per v2 input the text waits for, from V2_PENDING (this file).

    The source recorded for each marker is this script, where the registry lives.
    A fixture root that does not contain this script gets no markers: it has no
    manuscript that could print them.
    """
    me = pathlib.Path(__file__).resolve()
    if not me.is_relative_to(em.root.resolve()):
        return
    printed = v2_printed(em.root)
    for key, (what, sub) in V2_PENDING.items():
        here = (v2_present(em.root, sub) if sub else
                all(v2_present(em.root, x) for _, x in V2_PENDING.values() if x))
        text = ((f"v2 numbers in place ({sub}): rewrite the text around them" if key in printed
                 else f"v2 input present ({sub}) but not printed yet: the numbers here are "
                      "the first grid's") if here and sub
                else "v2 input present: rewrite this from it" if here
                else f"pending v2: {what}")
        if not here and sub:
            missing.append(f"v2 {what} -- experiments/FIGS/data/v2/{sub}/")
        em.macro("Pending" + key, f"\\pending{{{text}}}", me, f"V2_PENDING['{key}']",
                 "a marker, not a number: red in the draft until the input exists")


# ------------------------------------------------------------------ later results
#
# Everything below reads a result file written after the probe ladder and
# applies the same rules: the mean and SD over the per-seed rows it stores, never
# a test. The note on each macro says what it aggregates.

def pick(d, path: str):
    """The value at a dotted path, `[i]` for list indices and `['k']` for a key
    that itself holds dots: "a.b[0].c", "cov['train_N1000_s1.parquet'].n"."""
    for part in re.findall(r"\[\d+\]|\['[^']*'\]|[^.\[\]]+", path):
        if part.startswith("['"):
            d = d[part[2:-2]]
        else:
            d = d[int(part[1:-1])] if part.startswith("[") else d[part]
    return d


def fmt_sci(x) -> str:
    """5e-4 -> $5\\times10^{-4}$; 819200000 -> $8.192\\times10^{8}$."""
    m, e = f"{float(x):.6e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_factor(log_value, nd: int = 2) -> str:
    """A difference of natural logs as the factor it is in the underlying quantity."""
    return fmt(np.exp(float(log_value)), nd)


def of(n, m) -> str:
    return f"{n} of {m}"


def emit_literature(em: Emitter, path: pathlib.Path) -> None:
    """Numbers quoted from other papers, each with the verbatim passage it rests on
    (experiments/FIGS/data/literature_facts.json). Not results, but typed by hand
    they would be the one kind of number in the manuscript that nothing traces."""
    facts = json.loads(path.read_text())["facts"]
    for i, f in enumerate(facts):
        if not f["macro"].startswith("Lit") or not f["quote"] or f["value"] not in f["quote"]:
            raise SystemExit(f"FATAL: {path} fact {i} ({f['macro']}): a literature macro must "
                             f"start with Lit and its value must appear in its quote")
        # 'tex' is only the typesetting of the same value (10−5 as $10^{-5}$): its
        # digits must be the value's, so a display form cannot carry another number.
        shown = f.get("tex", f["value"])
        if re.sub(r"\D", "", shown) != re.sub(r"\D", "", f["value"]):
            raise SystemExit(f"FATAL: {path} fact {i} ({f['macro']}): tex {shown!r} is not "
                             f"the value {f['value']!r} typeset")
        em.macro(f["macro"], shown, path, f"facts[{i}].value",
                 f"{f['source']} {f['location']}")


def emit_mass_lambda(em: Emitter, specs: list) -> None:
    """The mass-loss weight, read from every mass-output pretraining job, which must agree."""
    vals = {m for p in specs for m in re.findall(r"--mass-lambda\s+([0-9.]+)", p.read_text())}
    if len(specs) != 10 or len(vals) != 1:
        raise SystemExit(f"FATAL: expected one --mass-lambda over ten mass-output specs, found "
                         f"{sorted(vals)} over {len(specs)}")
    em.macro("DesignMassLambda", fmt(float(vals.pop()), 1), specs[0], "--mass-lambda",
             "identical in all ten mass-output pretraining specs")


def emit_mass_lambda_matched(em: Emitter, path: pathlib.Path, grid_path: pathlib.Path) -> None:
    """The matched mass-loss weight and the first-grid shares it was matched on
    (PRESPEC A11 and its correction): per first-grid run, x = lambda * L_reg / L_cls
    averaged over its epochs, share = x / (1 + x), and lambda_m = lambda * mean x_162
    / mean x_17. The weight printed must be the one the rerun grid trains with."""
    L = json.loads(path.read_text())
    grid = {a["name"]: a for a in json.loads(grid_path.read_text())["arms"]}
    v = L["variants"]["mean_over_epochs_0_79"]
    lam, x162, x17 = L["lambda_m"], v["x_162_per_run"], v["x_17_per_run"]
    lam1 = grid["R16_Q1_MASS"]["mass_lambda"]
    if grid["R16_Q1_MASS_LM"]["mass_lambda"] != lam or grid["L162_MASS"]["mass_lambda"] != lam1:
        raise SystemExit(f"FATAL: {grid_path} does not train the matched weight {lam} of {path}")
    if round(lam1 * float(np.mean(x162)) / float(np.mean(x17)), 2) != lam:
        raise SystemExit(f"FATAL: {path} lambda_m {lam} does not follow from its own per-run x")
    em.macro("MassLambdaMatched", f"{lam:g}", path, "lambda_m",
             "the weight the rerun's R16_Q1_MASS_LM runs train with (v2_grid.json), "
             "lambda * mean x_162 / mean x_17 to two decimals")
    for key, x, k in (("MassLossShareOnesixtwoMass", x162, "x_162_per_run"),
                      ("MassLossShareOnesevenMass", x17, "x_17_per_run")):
        s = [100 * xi / (1 + xi) for xi in x]
        em.macro(key, fmt_pm(np.mean(s), np.std(s, ddof=1)) + "\\%", path,
                 f"variants.mean_over_epochs_0_79.{k}",
                 f"x/(1+x) per first-grid run, mean +- SD over {len(s)} runs")


def emit_paired(em: Emitter, files: list, ft: dict | None) -> None:
    """Every ratio the text can quote, as the paired geometric mean over runs of
    coarse/fine with its 95 % interval (experiments/STATS/paired_errors.py: run k
    with run k, one bootstrap of the test jets shared by both models and every run,
    run and test-sample terms combined). Only `ratio` and `ci95` are read, with the
    keys that say what was compared. A ratio that a saturated AUC floors is not a
    measurement and gets no macro."""
    for leg, ds in PAIRED_FT_DATASET.items():
        if ft and ds in ft and leg not in ft[ds]["path"].name:
            raise SystemExit(f"FATAL: the paired file's {leg} is not {ft[ds]['path'].name}")
    name = lambda x: texname(str(x).replace("+mass", " mass"))
    for path in files:
        for i, r in enumerate(json.loads(path.read_text())["ratios"]):
            fam = r["family"]
            if (r["metric"] != PAIRED_METRIC.get(fam) or r.get("censored_models")
                    or r.get("ratio") is None or r.get("ci95") is None):
                continue
            head = {"probe": lambda: "PairedProbe" + texname(r["task"], r["kind"]),
                    "ft": lambda: "PairedFt" + PAIRED_FT_DATASET[r["task"]] + n_tag(r["kind"]),
                    "mass": lambda: "PairedMassRes" + texname(r["kind"])}[fam]()
            lo, hi = r["ci95"]
            em.macro(head + name(r["coarse"]) + "Over" + name(r["fine"]),
                     fmt_paired(r["ratio"], lo, hi), path, f"ratios[{i}].ratio, ratios[{i}].ci95",
                     f"{r['coarse']} over {r['fine']}, {r['task']} {r['kind']} {r['metric']}: "
                     "paired geometric mean [95% interval]")


def emit_rand_design(em: Emitter, paths: dict, C: dict, missing: list | None = None) -> None:
    """What the random partitions and the flavour pair merge, read from the maps
    that define them. First grid: which partitions merge each control task's pair.
    Rerun: the number of partitions and runs and the balance rule, checked on the
    map itself; the flavour pair's cut and the orbits moved to keep shares exact,
    and the random-cut control F1r once its map column exists."""
    rows = {r["class_name"]: r for r in csv.DictReader(paths["rand_v1_map"].open())}
    ladder = json.loads((em.root / C["provenance"]["inputs"][0]["path"]).read_text())
    draws = sorted({r["draw"] for r in C["table"]})
    for task in ordered_tasks({r["task"] for r in C["table"]}):
        a, b = ladder["tasks"][task]["names"]
        merged = [d for d in draws if rows[a][f"RAND_d{d}"] == rows[b][f"RAND_d{d}"]]
        em.macro("RandMergingDraws" + texname(task), word_list(merged), paths["rand_v1_map"],
                 f"RAND_d* of {a} and {b}", "first-grid partitions that merge the task's pair")
        em.macro("RandSplittingDraws" + texname(task),
                 word_list([d for d in draws if d not in merged]), paths["rand_v1_map"],
                 f"RAND_d* of {a} and {b}", "first-grid partitions that keep the pair apart")

    grid = json.loads(paths["v2_grid"].read_text())["arms"]
    rand = [a for a in grid if a["name"].startswith("RAND2_")]
    flav = [a for a in grid if a["name"].startswith("FLAV_")]
    runs = {a["runs"] for a in rand}
    frun = {a["runs"] for a in flav}
    if len(runs) != 1 or len(frun) != 1:
        raise SystemExit(f"FATAL: {paths['v2_grid']} gives the partitions {runs} and the flavour "
                         f"pair {frun} runs; the text states one number for each")
    rule = json.loads(paths["rand_v2_rule"].read_text())
    lo, hi = rule["merged_in"]
    v2 = list(csv.DictReader(paths["rand_v2_map"].open()))
    cols = sorted(c for c in v2[0] if re.fullmatch(r"RAND2_p\d+", c))
    if cols != sorted(a["name"] for a in rand):
        raise SystemExit(f"FATAL: {paths['rand_v2_map']} holds {cols}, the grid trains "
                         f"{sorted(a['name'] for a in rand)}")
    g = {r["class_name"]: r for r in v2}
    for pair, (a, b) in rule["pairs"].items():
        k = sum(g[a][c] == g[b][c] for c in cols)
        if not lo <= k <= hi:
            raise SystemExit(f"FATAL: {pair} is merged in {k} of the partitions in "
                             f"{paths['rand_v2_map']}, outside the rule's {lo}-{hi}")
    src = paths["v2_grid"]
    em.macro("RandVtwoNPartitions", words(len(rand)), src, "arms[RAND2_*] (count)",
             "random partitions in the rerun")
    em.macro("RandVtwoRuns", words(runs.pop()), src, "arms[RAND2_*].runs", "runs per partition")
    em.macro("RandFlavRuns", words(frun.pop()), src, "arms[FLAV_*].runs", "runs per flavour-pair model")
    lofo = {a["runs"] for a in grid if a.get("extra_selection")}
    if len(lofo) != 1:
        raise SystemExit(f"FATAL: {paths['v2_grid']} gives the models trained without a family {lofo} runs")
    em.macro("DesignLofoRuns", words(lofo.pop()), src, "arms[extra_selection set].runs",
             "runs per model trained without the left-out family")
    ssl = [a["runs"] for a in grid if a.get("num_classes") is None and not a.get("extra_selection")]
    if len(ssl) != 1:
        raise SystemExit(f"FATAL: {paths['v2_grid']} has {len(ssl)} self-supervised configurations "
                         f"trained on the full stream; the text describes one")
    em.macro("DesignSslRuns", words(ssl[0]), src, "arms[num_classes null, no extra_selection].runs",
             "self-supervised pretraining runs in the rerun")
    em.macro("RandVtwoNPairs", words(len(rule["pairs"])), paths["rand_v2_rule"], "pairs (count)",
             "probe pairs the balance rule covers")
    for name, n in (("RandVtwoMergedMin", lo), ("RandVtwoMergedMax", hi)):
        em.macro(name, words(n), paths["rand_v2_rule"], "merged_in",
                 "partitions merging each probe pair, checked on the map")

    # The flavour pair, every statement the text makes about it checked on the map:
    # F1's cut is exactly the b-containing classes of the split orbit, and the
    # classes F1 moves are that cut plus whole orbits moved back. The orbits are
    # named from the definition file, so a rebuilt pair renames them in the text.
    # The move back is printed only once the file records every share-exact
    # option and the one used changes the fewest pairs of classes across orbits
    # (PRESPEC A10, PI 2026-10-01); a pair built before that rule is a red marker.
    F = json.loads(paths["flavour_pair"].read_text())
    fmap = list(csv.DictReader(paths["flavour_map"].open()))
    fp, fm = paths["flavour_pair"], paths["flavour_map"]
    me = pathlib.Path(__file__).resolve()

    def pend(name, what, why):
        if me.is_relative_to(em.root.resolve()):
            em.macro(name, f"\\pending{{pending: {what}}}", me, f"emit_rand_design[{name}]",
                     f"a marker, not a number: {why}")
        if missing is not None:
            missing.append(f"{name} -- {why}")

    has_b = lambda c: "b" in c.rpartition("_")[2]
    split = {r["class_name"] for r in fmap if r["orbit"] == F["split_orbit"]}
    moved = {r["class_name"] for r in fmap if r["FLAV_F0"] != r["FLAV_F1"]}
    cut = set(F["split_classes_moved"])
    back = {r["class_name"] for r in fmap if r["orbit"] in F["orbits_moved_B_to_A"]}
    if cut != {c for c in split if has_b(c)}:
        raise SystemExit(f"FATAL: {fp} cuts {sorted(cut)}, not the classes of {F['split_orbit']} "
                         f"that contain a b quark")
    if moved != cut | back or cut & back:
        raise SystemExit(f"FATAL: {fm} moves {len(moved)} classes from F0 to F1, "
                         f"not the {len(cut)} cut plus the {len(back)} of the orbits moved back")
    orbit_tex = appendix_module().native_tex
    em.macro("RandFlavSplitOrbit", orbit_tex(F["split_orbit"]), fp, "split_orbit",
             "the decays F1 cuts, Q any quark")
    em.macro("RandFlavSplitOrbitClasses", str(len(split)), fm,
             "classes whose orbit is split_orbit (count)", "classes of the split orbit")
    em.macro("RandFlavCutClasses", str(len(cut)), fp, "split_classes_moved (count)",
             "classes of the split orbit that F1 moves: those with a b quark")
    opts = F.get("f1_options")
    if not opts:
        for name in ("RandFlavOrbitsMoved", "RandFlavOrbitsMovedList", "RandFlavOrbitClassesMoved"):
            pend(name, "the rebuilt flavour pair",
                 f"{fp.relative_to(em.root) if fp.is_relative_to(em.root) else fp.name} records "
                 f"no f1_options; its move back predates the 2026-10-01 rule")
    else:
        cross = lambda o: (o.get("pairs_changed_from_F0") or {}).get("cross_orbit")
        used = [o for o in opts if o.get("orbits_moved_B_to_A") == F["orbits_moved_B_to_A"]]
        if (len(used) != 1 or None in map(cross, opts)
                or cross(used[0]) != min(map(cross, opts))):
            raise SystemExit(f"FATAL: {fp} moves back {F['orbits_moved_B_to_A']}, which is not "
                             f"the recorded option changing the fewest pairs across orbits")
        T = F["orbits_moved_B_to_A"]
        em.macro("RandFlavOrbitsMoved", words(len(T)), fp, "orbits_moved_B_to_A (count)",
                 "whole orbits F1 moves the other way")
        em.macro("RandFlavOrbitsMovedList",
                 ", ".join(orbit_tex(o) for o in T[:-1]) + (" and " if len(T) > 1 else "")
                 + orbit_tex(T[-1]), fp, "orbits_moved_B_to_A",
                 "the orbits F1 moves the other way, Q any quark")
        em.macro("RandFlavOrbitClassesMoved", str(len(back)), fm,
                 "classes whose orbit is in orbits_moved_B_to_A", "classes in those orbits")

    # F1r, the random-cut control: F1 with the cut drawn at random within the same
    # orbit. Until its map column exists, a red marker; once it does, the text's
    # claims are checked: same size of cut, same move back, and the pair of the
    # four-prong b-vs-c task kept together where F1 separates it.
    if "FLAV_F1R" not in fmap[0]:
        pend("RandFlavFonerCutWithB", "the F1r map", f"{fm.name} has no FLAV_F1R column yet")
        return
    cut_r = {r["class_name"] for r in fmap if r["FLAV_F0"] != r["FLAV_F1R"]} & split
    back_r = {r["class_name"] for r in fmap if r["FLAV_F0"] != r["FLAV_F1R"]} - split
    g1 = {r["class_name"]: r["FLAV_F1"] for r in fmap}
    gr = {r["class_name"]: r["FLAV_F1R"] for r in fmap}
    a, b = ladder["tasks"]["bvc_4prong"]["names"]
    if len(cut_r) != len(cut) or back_r != back or gr[a] != gr[b] or g1[a] == g1[b]:
        raise SystemExit(f"FATAL: {fm} FLAV_F1R is not F1 with a random cut of the same size "
                         f"({len(cut_r)} vs {len(cut)}), the same move back, and {a}/{b} kept "
                         f"together where F1 separates them")
    em.macro("RandFlavFonerCutWithB", str(sum(map(has_b, cut_r))), fm,
             "FLAV_F1R != FLAV_F0 within split_orbit, with a b quark (count)",
             "classes with a b quark among those the F1r cut moves")
    # The b<->c boundaries (PRESPEC A14): pairs of the split orbit that differ only by
    # exchanging b and c quarks, as the definition file lists them, counted split on the map.
    bc = [tuple(p) for p in F.get("b_c_pairs_in_split_orbit", [])]
    if bc:
        if any(x not in split or y not in split for x, y in bc):
            raise SystemExit(f"FATAL: {fp} lists b<->c pairs outside {F['split_orbit']}")
        n1 = sum(g1[x] != g1[y] for x, y in bc)
        nr = sum(gr[x] != gr[y] for x, y in bc)
        em.macro("RandFlavBcPairs", str(len(bc)), fp, "b_c_pairs_in_split_orbit (count)",
                 "pairs of the split orbit's classes that differ only by exchanging b and c")
        em.macro("RandFlavFoneBcSplit", str(n1), fm, "b_c_pairs_in_split_orbit split by FLAV_F1 (count)")
        em.macro("RandFlavFonerBcSplit", "none" if nr == 0 else str(nr), fm,
                 "b_c_pairs_in_split_orbit split by FLAV_F1R (count)", "PRESPEC A14: none")
    else:
        pend("RandFlavFonerBcSplit", "the F1r map without a b<->c boundary",
             f"{fp.name} lists no b_c_pairs_in_split_orbit")
    # F1r's realised group shares against F1's, from the realised stream (the rule fixed
    # with the F1r correction of 2026-10-01: within 1 % of F1's).
    rs = paths.get("realised_shares")
    tol = (F.get("F1R") or {}).get("realised_tolerance_vs_F1")
    if rs and pathlib.Path(rs).exists() and tol:
        R = json.loads(pathlib.Path(rs).read_text())
        share = dict(zip(R["class_name"], R["share"]))
        grp = lambda col: {g: sum(share[r["class_name"]] for r in fmap if r[col] == g) for g in {r[col] for r in fmap}}
        s1, sr = grp("FLAV_F1"), grp("FLAV_F1R")
        dev = max(abs(sr[g] / s1[g] - 1) for g in s1)
        limit = tol[0] / tol[1]
        if set(s1) != set(sr) or dev > limit + 1e-12:
            raise SystemExit(f"FATAL: {fm} F1r's realised group shares are {dev:.4f} off F1's, "
                             f"beyond the {limit:g} its rule allows")
        em.macro("RandFlavFonerShareTol", f"{100 * limit:g}\\%", fp, "F1R.realised_tolerance_vs_F1",
                 "largest allowed relative difference of a group's realised share from F1's")
        em.macro("RandFlavFonerShareDev", f"{100 * dev:.2f}\\%", rs,
                 "share summed per FLAV_F1R and FLAV_F1 group of flavour_pair_map.v2.csv (largest relative difference)",
                 "realised shares from the final loader's dry run")


def emit_v2_training(em: Emitter, spec: pathlib.Path, code: pathlib.Path) -> None:
    """The rerun's loader, validation sample and robustness model, from the grid's
    188-class run-1 job (every grid job carries the same loader and validation
    files; tests/test_mtx_v2_specs.py) and from experiments/MTX/pretrain_v2.py."""
    s = spec.read_text()

    def arg(flag):
        vals = set(re.findall(rf"{flag}\s+([^\s\\]+)", s))
        if len(vals) != 1:
            raise SystemExit(f"FATAL: {spec} gives {flag} as {sorted(vals)}; expected one value")
        return vals.pop()
    em.macro("DesignVtwoDataFraction", f"{100 * float(arg('--data-fraction')):.0f}\\%", spec,
             "--data-fraction", "share of every file read per epoch")
    val = set(re.findall(r"--data-val ((?:/\S+\.parquet\s*)+)", s))
    if len(val) != 1:
        raise SystemExit(f"FATAL: {spec} gives --data-val {len(val)} different ways")
    n_val = 0
    for f in val.pop().split():
        m = re.search(r"\{(\d+)\.\.(\d+)\}", f)
        n_val += int(m.group(2)) - int(m.group(1)) + 1 if m else 1
    em.macro("DesignVtwoNValFiles", str(n_val), spec, "--data-val (files, brace ranges expanded)",
             "validation files of the fixed sample")
    c = code.read_text()

    def const(name):
        m = re.findall(rf"^{name} = ([\d_]+)", c, re.M)
        if len(m) != 1:
            raise SystemExit(f"FATAL: {code} sets {name} {len(m)} times")
        return int(m[0].replace("_", ""))
    epochs, last = int(arg("--num-epochs")), const("WAVG_EPOCHS")
    em.macro("DesignWavgFirstEpoch", str(epochs - last), code, "--num-epochs - WAVG_EPOCHS",
             "first epoch of the weight average (epochs count from 0)")
    em.macro("DesignWavgLastEpoch", str(epochs - 1), spec, "--num-epochs - 1", "last epoch")
    em.macro("DesignBnJets", fmt_int(const("BN_JETS")), code, "BN_JETS",
             "training jets the BatchNorm statistics are recomputed on")


def gpu_name(product: str) -> str:
    """A Kubernetes GPU product label as the paper names it: NVIDIA-GeForce-RTX-3090 -> RTX 3090."""
    return product.removeprefix("NVIDIA-").removeprefix("GeForce-").replace("-", " ")


def gpu_by_run(specs: list) -> dict:
    """{run index: {GPU product: [job files]}} of the rerun, read from the nodeAffinity of
    the grid's job specs (key nvidia.com/gpu.product), the products the jobs can be
    scheduled on: the specs, not the launch builder's table, are the record of what was
    launched."""
    out = {}
    for f in map(pathlib.Path, specs):
        m = re.search(r"-s(\d+)-raunav\.yaml$", f.name)
        prods = re.findall(r"key: nvidia\.com/gpu\.product\s+operator: In\s+values: \[([^\]]*)\]",
                           f.read_text())
        if not m or len(prods) != 1:
            raise SystemExit(f"FATAL: {f} names its GPU product {len(prods)} times")
        vals = [v.strip().strip("\"'") for v in prods[0].split(",")]
        if len(vals) != 1:
            raise SystemExit(f"FATAL: {f} lets the job run on several GPU products {vals}")
        out.setdefault(int(m.group(1)), {}).setdefault(vals[0], []).append(f)
    return out


def emit_v2_recipe(em: Emitter, paths: dict, missing: list | None = None) -> None:
    """The rerun's recipe where the text states it, each value read from the code or the
    record that fixes it: the learning-rate schedule and optimizer (experiments/MTX/
    pretrain_v2.py), the share of each file an epoch reads (the grid's jobs), how much of
    its planned reading an epoch uses and what leaving a family out changes in the stream
    (the committed loader dry runs), and which GPU each run index trains on."""
    spec, lofo, code = (paths[k].read_text() for k in ("design_spec_v2", "lofo_spec_v2", "pretrain_v2"))

    def flag(text, f, src):
        vals = set(re.findall(rf"{f}\s+([^\s\\]+)", text))
        if len(vals) != 1:
            raise SystemExit(f"FATAL: {src} gives {f} as {sorted(vals)}; expected one value")
        return vals.pop()
    frac = float(flag(spec, "--data-fraction", paths["design_spec_v2"]))
    k = round(1 / frac)
    if not math.isclose(k * frac, 1.0):
        raise SystemExit(f"FATAL: --data-fraction {frac} is not one of k equal windows")
    k_lofo = int(flag(lofo, "--data-windows", paths["lofo_spec_v2"]))
    if "--data-fraction" in lofo:
        raise SystemExit(f"FATAL: {paths['lofo_spec_v2']} sets both --data-windows and --data-fraction")
    em.macro("DesignVtwoNWindows", words(k), paths["design_spec_v2"], "1 / --data-fraction",
             "disjoint windows of each file's rows, one read per epoch")
    em.macro("DesignLofoNWindows", words(k_lofo), paths["lofo_spec_v2"], "--data-windows",
             "windows of the models trained without one family")

    D, L = (json.loads(paths[k_].read_text()) for k_ in ("v2_dryrun", "v2_dryrun_lofo"))
    if not math.isclose(D["args"]["data_fraction"], frac) or L["args"].get("data_windows") != k_lofo:
        raise SystemExit("FATAL: the loader dry runs did not run the windows the grid's jobs use")
    used = [(e["max_fetch_id"] + 1) / D["args"]["data_split_num"] for e in D["epochs"]]
    em.macro("DesignVtwoPlanUsed", f"{100 * np.mean(used):.0f}\\%", paths["v2_dryrun"],
             "epochs[*].(max_fetch_id + 1) / args.data_split_num, mean",
             f"share of an epoch's planned reading used before its jets are reached, {len(used)} epochs")
    used_l = [(e["max_fetch_id"] + 1) / e["fetches_per_pass"] for e in L["epochs"]]
    em.macro("DesignLofoPlanUsed", f"{100 * np.mean(used_l):.0f}\\%", paths["v2_dryrun_lofo"],
             "epochs[*].(max_fetch_id + 1) / fetches_per_pass, mean", "the same, leaving the family out")
    # How often a row is read: the share of each file an epoch actually reads is its
    # window times the part of the window used before the epoch's jets are reached. The
    # nominal window ratio k / k_lofo ignores that the family-out epochs stop earlier.
    em.macro("DesignLofoRowFactor", fmt((np.mean(used_l) / k_lofo) / (np.mean(used) / k), 2),
             paths["v2_dryrun_lofo"],
             "(mean plan used / --data-windows) over (mean plan used / k) of the full models' dry run",
             "how much more often a row of a file is read per epoch when the family is left out")
    # Distinct jets the full models train on per cycle: the k windows of a cycle are
    # disjoint (experiments/MTX/stream_v2.py cycle_of), so their epochs' distinct jets add.
    cyc = [sum(e["distinct_jets"] for e in D["epochs"][c * k:(c + 1) * k])
           for c in range(len(D["epochs"]) // k)]
    if not cyc or [e["epoch"] for e in D["epochs"][:k * len(cyc)]] != list(range(k * len(cyc))):
        raise SystemExit(f"FATAL: {paths['v2_dryrun']} does not hold whole cycles from epoch 0")
    em.macro("DesignDistinctJetsCycle", fmt_sci(float(f"{np.mean(cyc):.2g}")), paths["v2_dryrun"],
             f"sum of epochs[c*k:(c+1)*k].distinct_jets, mean over {len(cyc)} cycles",
             "distinct jets the full models train on in one cycle of windows")
    gone = L["summary"]["labels_absent_every_epoch"]
    counts = np.sum([e["native_counts"] for e in D["epochs"]], axis=0)
    share = float(counts[gone].sum() / counts.sum())
    em.macro("DesignLofoFamilyShare", f"{100 * share:.1f}\\%", paths["v2_dryrun"],
             "epochs[*].native_counts over summary.labels_absent_every_epoch of the family-out run",
             "the left-out family's share of the full models' stream")
    em.macro("DesignLofoNClasses", str(len(gone)), paths["v2_dryrun_lofo"],
             "summary.labels_absent_every_epoch (count)", "native classes of the left-out family")
    em.macro("DesignLofoExposureFactor", fmt(1 / (1 - share), 2), paths["v2_dryrun"],
             "1 / (1 - family share)", "how much more often every other class is seen")
    for name, d, src in (("DesignLofoQcdShareFull", D, "v2_dryrun"), ("DesignLofoQcdShareLofo", L, "v2_dryrun_lofo")):
        em.macro(name, f"{100 * d['summary']['mean_qcd_share']:.1f}\\%", paths[src],
                 "summary.mean_qcd_share", "QCD share of the realised stream")

    # The learning-rate schedule and optimizer, as pretrain_v2.make_optimizer builds them.
    def one(pat, what):
        m = re.findall(pat, code)
        if len(set(m)) != 1:
            raise SystemExit(f"FATAL: {paths['pretrain_v2']} {what}: {pat} matches {m}")
        return m[0]
    decay = float(one(r"int\(num_epochs \* ([0-9.]+)\)", "decay share"))
    floor = float(one(r"gamma=([0-9.]+) \*\* \(1\.0 / n_decay\)", "final factor"))
    one(r"(Ranger\(model\.parameters\(\), lr=lr\))", "Ranger at its defaults")
    one(r"(torch\.cuda\.amp\.GradScaler)", "dynamic loss scaling")
    if "--use-amp" not in spec:
        raise SystemExit(f"FATAL: {paths['design_spec_v2']} does not train in mixed precision")
    epochs, lr = int(flag(spec, "--num-epochs", paths["design_spec_v2"])), float(flag(spec, "--start-lr", paths["design_spec_v2"]))
    em.macro("DesignLrFlatPercent", f"{100 * (1 - decay):.0f}\\%", paths["pretrain_v2"],
             "make_optimizer: 1 - the decay share", "share of the epochs at the initial rate")
    em.macro("DesignLrDecayFirstEpoch", str(epochs - max(1, int(epochs * decay))), paths["pretrain_v2"],
             "num_epochs - int(num_epochs * decay share)", "first epoch at a reduced rate (epochs from 0)")
    em.macro("DesignLrEndPercent", f"{100 * floor:g}\\%", paths["pretrain_v2"],
             "make_optimizer: gamma ** n_decay", "the last epoch's rate over the initial rate")
    em.macro("DesignEndLr", fmt_sci(lr * floor), paths["design_spec_v2"], "--start-lr x the final factor",
             "the rate of the last epoch")
    em.macro("DesignWeaverVersion", one(r"weaver (\d+\.\d+\.\d+)", "weaver version"), paths["pretrain_v2"],
             "the weaver release whose loop it reproduces")

    # The band within which a result is robust to the checkpoint: the smallest effect of
    # interest of the plan, ln 1.1 in src/stats/mde.py.
    mei = paths.get("mei_code")
    if mei and pathlib.Path(mei).exists():
        m = re.findall(r"^MEI_LOG = float\(np\.log\(([0-9.]+)\)\)", pathlib.Path(mei).read_text(), re.M)
        if len(m) != 1:
            raise SystemExit(f"FATAL: {mei} sets MEI_LOG {len(m)} times or not as ln of a factor")
        em.macro("DesignMeiFactor", m[0], mei, "MEI_LOG = ln(factor)",
                 "the smallest effect of interest, as a factor")
    elif missing is not None:
        missing.append(f"DesignMeiFactor -- {mei}")

    # Run index -> GPU, from the grid's job specs. Every configuration of one run index
    # trains on one product (I7); a run index that mixes products is not printed.
    by_run = gpu_by_run(paths["v2_grid_specs"])
    me = pathlib.Path(__file__).resolve()

    def pend(why, names=("DesignVtwoSecondGpu", "DesignVtwoSecondGpuRuns")):
        if me.is_relative_to(em.root.resolve()):
            for name in names:
                em.macro(name, "\\pending{pending: the GPU type of each run index}", me,
                         f"emit_v2_recipe[{name}]", f"a marker, not a number: {why}")
        if missing is not None:
            missing.append(f"{', '.join(names)} -- {why}")
    mixed = {r: sorted(p) for r, p in by_run.items() if len(p) > 1}
    prod = {r: next(iter(p)) for r, p in by_run.items()}
    r0 = min(by_run)
    if mixed:
        why = f"the grid's job specs mix GPU products within run indices {mixed}"
        return pend(why, ("DesignVtwoGpu", "DesignVtwoSecondGpu", "DesignVtwoSecondGpuRuns"))
    em.macro("DesignVtwoGpu", gpu_name(prod[r0]), by_run[r0][prod[r0]][0],
             "nodeAffinity nvidia.com/gpu.product", f"the product of run index {r0}, every configuration")
    other = sorted(r for r, g in prod.items() if g != prod[r0])
    if not other:
        return pend("the grid's job specs put every run index on one product")
    seconds = {prod[r] for r in other}
    if len(seconds) != 1:
        raise SystemExit(f"FATAL: the grid's job specs use more than two GPU products: {prod}")
    second = seconds.pop()
    em.macro("DesignVtwoSecondGpu", gpu_name(second), by_run[other[0]][second][0],
             "nodeAffinity nvidia.com/gpu.product", f"the product of run indices {other}")
    em.macro("DesignVtwoSecondGpuRuns", word_list(other), by_run[other[0]][second][0],
             "nodeAffinity nvidia.com/gpu.product over the grid's job specs",
             "run indices not on the first product")


def ft_recipe(R: dict, leg_specs: list, reported: set) -> dict:
    """The fine-tuning settings of the runs behind the reported fine-tuning rows,
    from what each run recorded, checked against the commands.

    The runs are the JetClass-II (leg 1) and JetClass (leg 2) cells of the models
    in `reported`, at fine-tuning seed 1; interrupted attempts (.partial.) wrote
    a manifest too and are left out. Every such cell must have exactly one
    recorded run, and every field the table quotes must take ONE value, or this
    stops. Three facts are not in the run manifests and are read from the
    committed job commands, which must all agree: the optimizer and mixed
    precision, the learning-rate schedule (no --lr-scheduler flag, so weaver's
    default) and the validation size."""
    runs = [r for r in R["runs"] if ".partial." not in r["path"] and r["leg"] in ("1", "2")
            and r["init"] in reported and r["ft_seed"] == "1"]
    cells = collections.Counter((r["leg"], r["init"], int(r["n_train"])) for r in runs)
    sizes = sorted({n for _, _, n in cells})
    want = {(leg, i, n) for leg in ("1", "2") for i in reported for n in sizes}
    if set(cells) != want or any(c != 1 for c in cells.values()):
        raise SystemExit(f"FATAL: the recorded runs do not match the reported cells one to one: "
                         f"missing {sorted(want - set(cells))[:5]}, "
                         f"repeated {sorted(k for k, c in cells.items() if c != 1)[:5]}")
    one = {}
    for r in runs:
        for key in ("lr", "head_lr_mult", "weight_decay", "batch_size", "lr_schedule"):
            one.setdefault(key, set()).add(r[key])
        one.setdefault(("epochs", int(r["n_train"])), set()).add(r["epochs"])
    bad = {str(k): sorted(map(str, v)) for k, v in one.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"FATAL: fine-tuning runs disagree within a group: {bad}")
    v = {k: next(iter(s)) for k, s in one.items()}
    # weaver takes samples_per_epoch // batch_size optimizer steps per epoch and
    # drops the last partial batch (train.py:1006); the job commands decouple
    # samples_per_epoch from the subset size at the smallest N only, where an
    # "epoch" is several passes over the subset (scripts/build_ft_jobs.py
    # SAMPLES_PER_EPOCH). The run manifests' steps_per_epoch = N/512 is wrong
    # there, so the steps are computed from the commands, not read from them.
    spe = set()
    for f in leg_specs:
        spe |= {tuple(m) for m in re.findall(r"samples_for \(\) \{ case \$1 in (\d+)\) echo (\d+);;",
                                            f.read_text())}
    if len(spe) != 1:
        raise SystemExit(f"FATAL: the fine-tuning commands disagree on samples per epoch: {sorted(spe)}")
    (n_small, spe_small), = spe
    per_epoch = {n: int(spe_small) if n == int(n_small) else n for n in sizes}
    batch = int(v["batch_size"])
    for f in leg_specs:
        t = f.read_text()
        for need in ("--use-amp", "--optimizer ranger", "LR=1e-4"):
            if need not in t:
                raise SystemExit(f"FATAL: {f} lacks {need!r}")
        if "--lr-scheduler" in t or set(re.findall(r"--samples-per-epoch-val (\S+)", t)) != {"20000"}:
            raise SystemExit(f"FATAL: {f} is not the JetClass recipe the table states")
    if v["lr_schedule"] is not None:
        raise SystemExit("FATAL: the recorded schedule is not weaver's default")
    return {"n_runs": len(runs), "lr": v["lr"], "head_mult": v["head_lr_mult"],
            "weight_decay": v["weight_decay"], "batch": v["batch_size"],
            "epochs_jetclass": {n: v[("epochs", n)] for n in sizes},
            "steps": {n: per_epoch[n] // batch for n in sizes},
            "passes": {n: per_epoch[n] // n for n in sizes}, "spec": leg_specs[0],
            "val_jetclass": str(20000 // batch * batch)}


def emit_ft_recipe(em: Emitter, rec: dict, src: pathlib.Path) -> None:
    em.macro("FtRecipeNRuns", fmt_int(rec["n_runs"]), src, "runs (count), reported cells")
    em.macro("FtRecipeLr", fmt_sci(rec["lr"]), src, "runs[*].lr, pretrained starts")
    em.macro("FtRecipeHeadMult", rec["head_mult"], src, "runs[*].head_lr_mult, pretrained starts")
    em.macro("FtRecipeLrHead", fmt_sci(float(rec["lr"]) * float(rec["head_mult"])), src,
             "lr x head_lr_mult")
    em.macro("FtRecipeWeightDecay", rec["weight_decay"], src, "runs[*].weight_decay")
    em.macro("FtRecipeBatch", rec["batch"], src, "runs[*].batch_size")
    for n, e in rec["epochs_jetclass"].items():
        p = texname(f"{len(str(n)) - 1}")
        em.macro("FtRecipeEpochsE" + p, e, src, f"runs[leg 1,2; n_train {n}].epochs")
        em.macro("FtRecipeStepsE" + p, fmt_int(rec["steps"][n]), rec["spec"],
                 f"job commands: samples per epoch at n_train {n} // batch size",
                 "optimizer steps per epoch, as weaver floors them")
        em.macro("FtRecipePassesE" + p, words(rec["passes"][n]), rec["spec"],
                 f"job commands: samples per epoch at n_train {n} // n_train",
                 "passes over the training subset per epoch")
    em.macro("FtRecipeValJets", fmt_int(rec["val_jetclass"]), rec["spec"],
             "--samples-per-epoch-val 20000 // batch x batch", "validation jets per epoch, floored to whole batches")


def table_ft_recipe(rec: dict) -> str:
    ep = rec["epochs_jetclass"]
    body = [f"optimizer & Ranger, mixed precision, batch size {rec['batch']}, "
            f"weight decay {rec['weight_decay']} \\\\",
            f"learning rate & {fmt_sci(rec['lr'])} (pretrained weights), "
            f"{fmt_sci(float(rec['lr']) * float(rec['head_mult']))} (new head) \\\\",
            "schedule & pretrained weights at a constant rate, new head flat then decayed "
            "to 1\\% over the last 30\\% of epochs (weaver default with a head multiplier) \\\\",
            "epochs & " + ", ".join(str(ep[n]) for n in sorted(ep))
            + " at " + ", ".join(fmt_n_jets(f"N{n}") for n in sorted(ep)) + " jets \\\\",
            "steps per epoch & " + ", ".join(fmt_int(rec["steps"][n]) for n in sorted(ep))
            + f" (at {fmt_n_jets(f'N{min(ep)}')} jets an epoch is {words(rec['passes'][min(ep)])} "
            "passes over the training jets) \\\\",
            f"validation jets & {fmt_int(rec['val_jetclass'])} \\\\",
            "checkpoint & best validation accuracy \\\\"]
    caption = (f"Fine-tuning settings of the {fmt_int(rec['n_runs'])} runs behind the JetClass-II and "
               "JetClass results, as each run recorded them, checked against the job commands; the "
               "steps per epoch are computed from the commands. "
               "The head is freshly initialised in every run; every other weight is loaded "
               "from the pretrained model.")
    return _table(body, caption, "tab:ftrecipe", "l p{0.7\\linewidth}", [])


def emit_training_design(em: Emitter, spec: pathlib.Path, arch: pathlib.Path,
                         arm: pathlib.Path) -> None:
    """Design numbers the Setup section quotes, read from the files that set them.

    The 188-class seed-1 pretraining job stands for all of them: the arm jobs
    differ only in the label set, the seed streams and the names
    (tests/test_arm_configs.py). These are not results, but typed by hand they
    would be the one set of numbers in the paper that nothing checks.
    """
    s = spec.read_text()

    def arg(flag):
        vals = set(re.findall(rf"{flag}\s+'?([^\s']+)", s))
        if len(vals) != 1:
            raise SystemExit(f"FATAL: {spec} gives {flag} as {sorted(vals)}; expected one value")
        return vals.pop()

    epochs, per_epoch = int(arg("--num-epochs")), int(arg("--samples-per-epoch"))
    em.macro("DesignEpochs", str(epochs), spec, "--num-epochs")
    em.macro("DesignSamplesPerEpoch", fmt_int(per_epoch), spec, "--samples-per-epoch")
    em.macro("DesignExamplesSeen", fmt_sci(epochs * per_epoch), spec,
             "--num-epochs x --samples-per-epoch", "training examples drawn over the run")
    em.macro("DesignBatchSize", arg("--batch-size"), spec, "--batch-size")
    em.macro("DesignStartLr", fmt_sci(float(arg("--start-lr"))), spec, "--start-lr")
    hidden = re.findall(r"fc_params '\[\((\d+),", s)
    if len(set(hidden)) != 1:
        raise SystemExit(f"FATAL: {spec} fc_params hidden width is {sorted(set(hidden))}")
    em.macro("DesignHeadHidden", hidden[0], spec, "-o fc_params (hidden width)")

    a = arch.read_text()
    for name, pat in (("DesignParticleBlocks", r"num_layers=(\d+)"),
                      ("DesignClassBlocks", r"num_cls_layers=(\d+)"),
                      ("DesignHeads", r"num_heads=(\d+)"),
                      ("DesignEmbedDim", r"\bembed_dims=\[\d+, \d+, (\d+)\]")):
        m = re.findall(pat, a)
        if len(m) != 1:
            raise SystemExit(f"FATAL: {arch} matches {pat} {len(m)} times")
        em.macro(name, m[0], arch, pat)

    mpm = spec.parents[3] / "experiments" / "MTX" / "mpm.py"
    if mpm.exists():
        r = re.findall(r"^DEFAULT_MASK_RATE = ([0-9.]+)", mpm.read_text(), re.M)
        if len(r) != 1:
            raise SystemExit(f"FATAL: {mpm} sets DEFAULT_MASK_RATE {len(r)} times")
        em.macro("DesignMpmMaskRate", f"{100 * float(r[0]):.0f}\\%", mpm, "DEFAULT_MASK_RATE",
                 "share of each jet's particles hidden from the encoder")
    m = re.search(r"\(jet_pt > (\d+)\) & \(jet_pt < (\d+)\) & \(jet_sdmass > (\d+)\) & "
                  r"\(jet_sdmass < (\d+)\)", arm.read_text())
    if not m:
        raise SystemExit(f"FATAL: {arm} has no selection line of the expected form")
    for name, v in zip(("DesignPtMin", "DesignPtMax", "DesignMsdMin", "DesignMsdMax"), m.groups()):
        em.macro(name, fmt_int(v), arm, "selection", "GeV")


def recovery_acc(R: dict, probe: str = "linear") -> dict:
    """{(rung, model level): {run: balanced accuracy}} at the largest training set of the
    label-recovery learning curve. A model off the tree (level None: a second-grid model
    without a vocabulary of the ladder, or the untrained trunk) has no level and no cell."""
    n = max(R["sizes"])
    out = {}
    for r in R["table"]:
        if r["probe"] == probe and r["n_train"] == n and r.get("level") is not None:
            out.setdefault((r["rung"], r["level"]), {})[r["seed"]] = r["accuracy"]
    return out


def paired_diff(x: dict, y: dict) -> tuple[float, float, float]:
    """y minus x, paired run by run: the mean difference and its 95 % Student-t interval.
    Every model is scored on the same test jets, so the paired difference keeps the
    run-to-run variation and drops what the jets share."""
    from scipy import stats
    runs = sorted(set(x) & set(y))
    d = np.array([y[r] - x[r] for r in runs])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def fmt_diff(d, lo, hi, nd: int = 4) -> str:
    """A paired difference and its 95 % interval, signed: +0.0060~[+0.0051, +0.0069].
    As fmt_paired does about 1, one more place while rounding would put a bound on 0 or
    on the other side of it, or put the point on 0."""
    side = lambda v, p: (round(float(v), p) > 0) - (round(float(v), p) < 0)
    while nd < 8 and any(side(v, nd) != side(v, 15) for v in (d, lo, hi)):
        nd += 1
    return f"{fmt(d, nd, sign=True)}~[{fmt(lo, nd, sign=True)}, {fmt(hi, nd, sign=True)}]"


def emit_recovery(em: Emitter, R: dict, src: pathlib.Path, sizes: dict, align: str | None = None) -> None:
    """Label recovery at every level of the tree from every model: balanced accuracy of a
    class-weighted logistic regression fitted on the largest training set of the
    learning curve (experiments/EVAL/label_recovery_curve.py), mean +- SD over runs, and
    the paired differences the text reads."""
    files = [em.root / x["path"] for x in R["provenance"]["inputs"]]
    per = [json.loads(p.read_text()) for p in files]
    if align and {d["row_alignment_sha256"] for d in per} != {align}:
        raise SystemExit("FATAL: the label-recovery curve was not scored on the probe ladder's jets")
    for key in ("n_test", "n_pool", "sizes"):
        if len({json.dumps(d[key]) for d in per}) != 1:
            raise SystemExit(f"FATAL: the label-recovery files disagree on {key}")
    if sorted(per[0]["sizes"]) != sorted(R["sizes"]) or max(R["sizes"]) != per[0]["n_pool"]:
        raise SystemExit("FATAL: the largest training set is not every jet the curve holds out from test")
    if R.get("unconverged"):
        raise SystemExit(f"FATAL: label-recovery fits did not converge: {R['unconverged'][:3]}")
    acc = recovery_acc(R)
    levels = sorted({lv for _, lv in acc}, reverse=True)
    for rung in RUNGS:
        for lv in levels:
            a = list(acc[(rung, lv)].values())
            em.macro("RecoveryAcc" + texname(sizes[rung]) + "By" + texname(lv),
                     fmt_pm(np.mean(a), np.std(a, ddof=1)), src,
                     f"table[rung={rung},level={lv},probe=linear,n_train=max].accuracy",
                     f"balanced accuracy, mean +- SD over {len(a)} runs")
    gap = lambda rung, fine, coarse: (float(np.mean(list(acc[(rung, fine)].values())))
                                      - float(np.mean(list(acc[(rung, coarse)].values()))))
    em.macro("Recovery" + texname(188, 162) + "MaxAbs",
             fmt(max(abs(gap(r, 188, 162)) for r in RUNGS), 4), src,
             "table[rung=*,level=188/162,probe=linear,n_train=max].accuracy",
             "max |difference of the run means| over the tree")
    # Paired differences the text reads in words: the 17-class model against the finest
    # one at the 17-class level and at the two coarser levels of the tree.
    for rung in ("R16_Q1", "R3_VIS", "R1_Q1"):
        for fine in (188, 162):
            em.macro("RecoveryPaired" + texname(sizes[rung]) + "LevelOnesevenMinus" + texname(fine),
                     fmt_diff(*paired_diff(acc[(rung, fine)], acc[(rung, 17)])), src,
                     f"table[rung={rung},level=17 minus {fine},probe=linear,n_train=max].accuracy",
                     "paired by run, mean difference [95% Student-t interval]")
    mlp = {r["level"]: r for r in R["mlp_minus_linear_at_largest"]}
    if 17 in mlp and mlp[17]["rung"] == "L188":
        m = mlp[17]
        d = {i + 1: v for i, v in enumerate(m["per_run"])}
        em.macro("RecoveryMlpGainOneseven", fmt_diff(*paired_diff({k: 0.0 for k in d}, d)), src,
                 "mlp_minus_linear_at_largest[rung=L188,level=17].per_run",
                 "nonlinear minus linear probe, 188 native labels from the 17-class models")
    for g in R["data_gain_last_step"]:
        if (g["rung"], g["level"]) in (("L188", 188), ("L188", 17)):
            em.macro("RecoveryLastGain" + texname(g["level"]), fmt_pm(g["mean"], g["sd"]), src,
                     f"data_gain_last_step[rung=L188,level={g['level']}]",
                     f"balanced accuracy gained from {g['from']} to {g['to']} training jets")
    em.macro("RecoveryNTrain", fmt_int(max(R["sizes"])), src, "sizes (max)",
             "jets the linear probe is fit on at the end of the curve")
    em.macro("RecoveryNTrainMin", fmt_int(min(R["sizes"])), src, "sizes (min)")
    em.macro("RecoveryNTrainPrev", fmt_int(sorted(R["sizes"])[-2]), src, "sizes (second largest)")
    em.macro("RecoveryNTest", fmt_int(per[0]["n_test"]), files[0], "n_test",
             f"jets the probes are scored on, recorded in all {len(files)} per-run files")
    em.macro("RecoveryMlpHidden", str(per[0]["mlp"]["hidden"]), files[0], "mlp.hidden",
             "hidden units of the nonlinear probe")


def emit_random_control(em: Emitter, C: dict, src: pathlib.Path) -> None:
    """The random-label control: 1-AUC of each draw, one run each, and over the draws."""
    n = words(len({r["draw"] for r in C["table"]}))
    em.macro("RandNDraws", n, src, "table[*].draw", "random partitions drawn")
    em.macro("RandNDrawsCap", n[:1].upper() + n[1:], src, "table[*].draw",
             "random partitions drawn, capitalised to open a sentence")
    for task in ordered_tasks({r["task"] for r in C["table"]}):
        for probe in ("linear", "mlp"):
            rows = sorted(((i, r) for i, r in enumerate(C["table"])
                           if (r["task"], r["probe"]) == (task, probe)), key=lambda x: x[1]["draw"])
            x = [float(np.exp(r["control_log1m_auc"])) for _, r in rows]
            for (i, r), v in zip(rows, x):
                em.macro("RandOma" + texname(task, "draw", r["draw"], probe),
                         "---" if r["censored"] else fmt_one_sci(v), src,
                         f"table[{i}].control_log1m_auc", "1-AUC = exp of it, one run"
                         + ("; not quoted: the AUC reached 1" if r["censored"] else ""))
            em.macro("RandOmaMean" + texname(task, probe),
                     "---" if any(r["censored"] for _, r in rows)
                     else fmt_pm_sci(np.mean(x), np.std(x, ddof=1)), src,
                     f"table[task={task},probe={probe}].control_log1m_auc",
                     f"1-AUC, mean +- SD over {len(x)} draws")
    g = C["grouping_cost_post_hoc"]
    for task, per in g["tasks"].items():
        for probe, x in per.items():
            k2 = texname(task, probe)
            path = f"grouping_cost_post_hoc.tasks.{task}.{probe}"
            em.macro("RandCostFactor" + k2, fmt(x["factor_control"], 2), src,
                     path + ".factor_control", "POST HOC: random grouping vs finer models, in 1-AUC")
            em.macro("RandSemCostFactor" + k2, fmt(x["factor_semantic"], 2), src,
                     path + ".factor_semantic", "POST HOC: semantic 17-class vs finer models")


# Fine-tuning datasets by the class count their cells record: macro key, name.
FT_DATASETS = {162: ("Jcii", "JetClass-II"), 10: ("Jc", "JetClass")}


# The random-label control's three draws, one pretraining run each; its row is
# the mean +- SD over the draws.
RAND_FT = ("rand-d1-s1b", "rand-d2-s2", "rand-d3-s3")


def ft_rows(sizes: dict) -> list:
    """Rows of the fine-tuning tables: (macro key, label, initialisations), each
    model fine-tuned once, at fine-tuning seed s1.

    Random initialisation and the self-supervised model are not rows while their
    fine-tuning recipes are corrected and rerun; each is one more entry here
    once its metrics exist.
    """
    five = lambda stem, first="1": [f"{stem}-s{first}"] + [f"{stem}-s{s}" for s in range(2, 6)]
    rows = [(texname(sizes[r]), f"{sizes[r]} classes", five(a, "1b" if a == "l162" else "1"))
            for a, r in ARM_RUNG.items()]
    rows += [(texname(sizes[ARM_RUNG[a]]) + "Mass", f"{sizes[ARM_RUNG[a]]} classes + mass output",
              five(a + "mass")) for a in ("l162", "r16q1")]
    # The random-label control has three partitions, one run each: a row per partition,
    # never a mean and spread of three values.
    return rows + [(texname("Rand", "draw", i + 1), f"random partition {i + 1}", [init])
                   for i, init in enumerate(RAND_FT)]


def ft_load(paths: list) -> dict:
    """{dataset key: its cells}, each file identified by the class count its cells
    record rather than by its name: 162 is JetClass-II's task, 10 JetClass's."""
    out = {}
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        k = {c["n_classes_present"] for i in d["cells"].values() for n in i.values()
             for c in n.values()}
        ds = FT_DATASETS.get(next(iter(k))) if len(k) == 1 else None
        if ds is None or ds[0] in out:
            raise SystemExit(f"FATAL: {p} is not one fine-tuning dataset of "
                             f"{sorted(FT_DATASETS)} classes")
        out[ds[0]] = {"path": pathlib.Path(p), "name": ds[1], "classes": next(iter(k)),
                      "cells": d["cells"]}
    return {key: out[key] for key, _ in FT_DATASETS.values() if key in out}


def ft_sizes(cells: dict, rows: list) -> list:
    return sorted(cells[rows[0][2][0]], key=lambda n: int(n[1:]))


def ft_text(cells: dict, rows: list, metric: str) -> dict:
    """{(row key, size): text}, mean +- SD over the row's models of one metric at
    fine-tuning seed s1. A single run prints to the finest decimal place of the
    mean +- SD entries in its column, so the two read alike."""
    text, place = {}, {}
    for key, _, inits in rows:
        for n in ft_sizes(cells, rows):
            v = [cells[i][n]["s1"][metric] for i in inits]
            if len(v) > 1:
                text[(key, n)] = fmt_pm(np.mean(v), np.std(v, ddof=1))
                if np.std(v) > 0:
                    place[n] = max(place.get(n, -99), pdg(np.std(v, ddof=1))[1])
    for key, _, inits in rows:
        if len(inits) == 1:
            for n in ft_sizes(cells, rows):
                text[(key, n)] = fmt_one(cells[inits[0]][n]["s1"][metric], place.get(n))
    return text


def ft_n_test(cells: dict, rows: list, field: str) -> int:
    """The test-jet count behind every cell of one dataset; it must be one number."""
    n = {cells[i][s]["s1"][field] for _, _, inits in rows for i in inits for s in cells[i]}
    if len(n) != 1:
        raise SystemExit(f"FATAL: fine-tuning cells disagree on {field}: {sorted(n)}")
    return n.pop()


def emit_finetune(em: Emitter, ft: dict, sizes: dict, rows: list | None = None,
                  keys: dict | None = None, restrict=None) -> None:
    """Fine-tuning every pretrained model on JetClass-II and JetClass: macro AUC and
    accuracy at the best-validation-accuracy epoch, fine-tuning seed s1.

    `rows` default to the first grid's (ft_rows); `keys` name the rows of the 188-, 162-
    and 17-class models ({rung: row key}, default from `sizes`); `restrict(a, b)` returns
    the initialisations of two rows a ratio of their means may use (I7, the second grid:
    v2_one_product), or None to print none."""
    rows = rows or ft_rows(sizes)
    keys = keys or {r: texname(sizes[r]) for r in ("L188", "L162", "R16_Q1")}
    for ds, F in ft.items():
        c, src, ns = F["cells"], F["path"], ft_sizes(F["cells"], rows)
        for metric, name in (("macro_auc_ovr", "FtAuc"), ("accuracy", "FtAcc")):
            text = ft_text(c, rows, metric)
            for key, _, inits in rows:
                for n in ns:
                    em.macro(name + ds + n_tag(n) + key, text[(key, n)], src,
                             f"cells.{{{','.join(inits)}}}.{n}.s1.{metric}",
                             "one run" if len(inits) == 1
                             else f"mean +- SD over the row's {len(inits)} pretrained models")
        oma = {key: {n: np.mean([1 - c[i][n]["s1"]["macro_auc_ovr"] for i in inits]) for n in ns}
               for key, _, inits in rows}
        by_key = {key: inits for key, _, inits in rows}
        for n in ns:
            for fine in (keys["L188"], keys["L162"]):
                for key in oma:
                    if key == fine:
                        continue
                    if restrict is None:
                        r = oma[key][n] / oma[fine][n]
                    else:
                        pair = restrict(by_key[key], by_key[fine])
                        if pair is None:
                            continue
                        r = (np.mean([1 - c[i][n]["s1"]["macro_auc_ovr"] for i in pair[0]])
                             / np.mean([1 - c[i][n]["s1"]["macro_auc_ovr"] for i in pair[1]]))
                    em.macro("FtOmaRatio" + ds + n_tag(n) + key + "Over" + fine,
                             fmt_ratio(r), src, f"cells.*.{n}.s1.macro_auc_ovr",
                             "ratio of the seed means of 1 - macro AUC, this row over the reference"
                             + ("" if restrict is None else
                                "; runs of one GPU product where the two rows' runs differ (I7)"))
        # The accuracy has no bootstrap interval in the paired files; the coarse-minus-fine
        # difference paired by run, with a Student-t interval over the runs, is what the
        # text may read in words.
        fine, coarse = keys["L188"], keys["R16_Q1"]
        for n in ns:
            acc = {k: {run_index(i): c[i][n]["s1"]["accuracy"] for i in by_key[k]} for k in (fine, coarse)}
            em.macro("PairedFtAcc" + ds + n_tag(n) + coarse + "Minus" + fine,
                     fmt_diff(*paired_diff(acc[fine], acc[coarse])), src,
                     f"cells.{{{','.join(by_key[coarse] + by_key[fine])}}}.{n}.s1.accuracy",
                     "accuracy, coarse minus fine, paired by run [95% Student-t interval]")
        em.macro("FtNClasses" + ds, str(F["classes"]), src, "cells.*.*.s1.n_classes_present",
                 "classes of the fine-tuning task")
        em.macro("FtNTestAuc" + ds, fmt_int(ft_n_test(c, rows, "n_jets_auc")), src,
                 "cells.*.*.s1.n_jets_auc", "test jets the macro AUC is computed on")
        em.macro("FtNTestAcc" + ds, fmt_int(ft_n_test(c, rows, "n_jets")), src,
                 "cells.*.*.s1.n_jets", "test jets the accuracy is computed on")
    F = next(iter(ft.values()))
    for n in ft_sizes(F["cells"], rows):
        em.macro("FtSize" + n_tag(n), fmt_n_jets(n), F["path"], f"cells.*.{n}",
                 "fine-tuning training jets")
    em.macro("FtNRunsWord", words(max(len(i) for *_, i in rows)), F["path"],
             "cells (pretrained models per vocabulary row)", "pretraining runs per vocabulary")


# The class sum from the model's own outputs, with the signal's 17-class group
# removed at every label set, then the two detectors on the frozen features.
ANOMALY_FAMILIES = ("class_sum_matched", "mahalanobis", "knn")


def anomaly_signal_key(sig: str) -> str:
    """`label_X_YY_bbb` -> `XYYBbb`, the macro-name part for a signal."""
    return texname(sig.removeprefix("label_"))


def anomaly_cells(S: dict, fam: str, sig: str) -> dict:
    """Per label set at the primary injection: sigma_min and max SIC per seed."""
    lv = S["families"][fam][sig][S["conventions"]["primary_injection"]]["levels"]
    return {int(k): {"sigma_min": np.exp(v["ln_sigma_min"]), "max_sic": np.array(v["max_sic"])}
            for k, v in lv.items()}


def anomaly_n_runs(S: dict) -> int:
    """Pretraining runs behind every anomaly cell; one number, or the captions lie."""
    inj = S["conventions"]["primary_injection"]
    n = {len(v["arms"]) for f in ANOMALY_FAMILIES for s in S["families"][f]
         for v in S["families"][f][s][inj]["levels"].values()}
    if len(n) != 1:
        raise SystemExit(f"FATAL: anomaly cells hold {sorted(n)} runs; the text states one number")
    return n.pop()


def emit_anomaly(em: Emitter, S: dict, src: pathlib.Path) -> None:
    """Anomaly detection with the feature-based detectors: sigma_min (the smallest
    initial significance from which a 5 sigma discovery is still reached,
    arXiv:2604.20965) and the maximum significance improvement, mean +- SD over
    seeds of each seed's median over resamplings (experiments/EVAL/anomaly_summary.py)."""
    res_path = em.root / S["provenance"]["inputs"]["anomaly"]["path"]
    R = json.loads(res_path.read_text())
    inj = S["conventions"]["primary_injection"]
    em.macro("AnomalyInjection", fmt_int(inj), src, "conventions.primary_injection",
             "injected signal jets")
    em.macro("AnomalyNResamplings", str(S["provenance"]["resamplings_per_seed"]), src,
             "provenance.resamplings_per_seed", "resamplings per model, median taken")
    em.macro("AnomalyNBkg", fmt_int(R["n_bkg"]), res_path, "n_bkg", "background jets in the data sample")
    em.macro("AnomalyNTemplate", fmt_int(R["n_template"]), res_path, "n_template",
             "jets in the background template")
    em.macro("AnomalyStatCut", f"{100 * R['stat_cut']:.0f}\\%", res_path, "stat_cut",
             "largest relative statistical error on eps_B at a usable threshold")
    em.macro("AnomalyMinBkgPass", str(int(np.ceil(1 / R["stat_cut"] ** 2))), res_path, "stat_cut",
             "background jets that must pass a threshold, 1/stat_cut^2")
    em.macro("AnomalySigmaT", fmt(R["sigma_t"], 0), res_path, "sigma_t", "target significance")
    code = em.root / "experiments" / "EVAL" / "anomaly.py"
    k = re.findall(r"^KNN_K = (\d+)", code.read_text(), re.M)
    if len(k) != 1:
        raise SystemExit(f"FATAL: {code} sets KNN_K {len(k)} times")
    em.macro("AnomalyKnnK", k[0], code, "KNN_K", "the k of the nearest-neighbour distance")
    for sig, per in S["classes_removed_by_level"]["class_sum_matched"].items():
        n = set(per.values())
        if len(n) != 1:
            raise SystemExit(f"FATAL: class_sum_matched removes {per} classes for {sig}")
        em.macro("AnomalyClassSumRemoved" + anomaly_signal_key(sig), str(n.pop()), src,
                 f"classes_removed_by_level.class_sum_matched.{sig}",
                 "native classes left out of the class sum, the same at every label set")
    nd = S["not_detected_rule"]
    em.macro("AnomalyNdThreshold", fmt(nd["threshold_max_sic"], 1), src,
             "not_detected_rule.threshold_max_sic", "max SIC below this at every label set")
    em.macro("AnomalyNRunsWord", words(anomaly_n_runs(S)), src,
             f"families.*.*.{inj}.levels.*.arms (count)", "pretraining runs per vocabulary")
    for fam in ANOMALY_FAMILIES:
        for sig in S["families"][fam]:
            for lv, c in anomaly_cells(S, fam, sig).items():
                key = texname(fam) + anomaly_signal_key(sig) + texname(lv)
                jp = f"families.{fam}.{sig}.{inj}.levels.{lv}"
                em.macro("AnomalySigmaMin" + key,
                         fmt_pm(np.mean(c["sigma_min"]), np.std(c["sigma_min"], ddof=1)), src,
                         jp + ".ln_sigma_min", "exp of each seed's value; mean +- SD over seeds")
                em.macro("AnomalyMaxSic" + key,
                         fmt_pm(np.mean(c["max_sic"]), np.std(c["max_sic"], ddof=1)), src,
                         jp + ".max_sic", "mean +- SD over seeds")
    # The output ratio again, each run's ln sigma_min averaged over its checkpoints of
    # epochs 70-79, each scored as a whole (commit 54854e9): a single epoch's output layer
    # cannot then stand in for the vocabulary. Only where the file holds it.
    rule = S.get("checkpoint_rule", {}).get("mean_ln_over_epochs_70_79", {}).get("class_sum_matched")
    if rule:
        for sig in rule:
            for lv in sorted(rule[sig][inj], key=int):
                v = np.exp(rule[sig][inj][lv]["ln_sigma_min"])
                em.macro("AnomalySigmaMinClassSumMatchedLate" + anomaly_signal_key(sig) + texname(lv),
                         fmt_pm(np.mean(v), np.std(v, ddof=1)), src,
                         f"checkpoint_rule.mean_ln_over_epochs_70_79.class_sum_matched.{sig}.{inj}.{lv}.ln_sigma_min",
                         "exp of each run's mean ln sigma_min over epochs 70-79; mean +- SD over runs")
    # Paired, run by run (sigma_min of the second over the first, the geometric mean of the
    # per-run ratios with a 95 % Student-t interval): every comparison the text puts in
    # words, between vocabularies within a score and between scores within a vocabulary.
    runs = lambda d: {run_index(a): x for a, x in zip(d["arms"], d["ln_sigma_min"])}
    cell = lambda fam, sig, lv: runs(S["families"][fam][sig][inj]["levels"][str(lv)])
    late = lambda sig, lv: runs(rule[sig][inj][str(lv)])
    exp_ = lambda d: {k: math.exp(v) for k, v in d.items()}
    levels = sorted({int(k) for f in ANOMALY_FAMILIES for s_ in S["families"][f]
                     for k in S["families"][f][s_][inj]["levels"]}, reverse=True)
    for fam in ANOMALY_FAMILIES:
        for sig in S["families"][fam]:
            if f"{fam}|{sig}" in nd["not_detected"]:
                continue
            for i, fine in enumerate(levels):
                for coarse in levels[i + 1:]:
                    for tag, get in (("", lambda lv: cell(fam, sig, lv)),
                                     ("Late", (lambda lv: late(sig, lv)) if rule and fam == "class_sum_matched" else None)):
                        if get is None:
                            continue
                        em.macro("PairedAnomaly" + texname(fam) + tag + anomaly_signal_key(sig)
                                 + texname(coarse) + "Over" + texname(fine),
                                 fmt_paired(*paired_ratio(exp_(get(fine)), exp_(get(coarse)))), src,
                                 f"families.{fam}.{sig}.{inj}.levels.{{{coarse},{fine}}}.ln_sigma_min"
                                 + (" (epochs 70-79 mean)" if tag else ""),
                                 "sigma_min ratio paired by run [95% Student-t interval]; above 1 = less sensitive")
    for sig in S["families"]["class_sum_matched"]:
        if any(f"{f}|{sig}" in nd["not_detected"] for f in ("class_sum_matched", "mahalanobis")):
            continue
        for lv in levels:
            em.macro("PairedAnomalyOutputOverMahalanobis" + anomaly_signal_key(sig) + texname(lv),
                     fmt_paired(*paired_ratio(exp_(cell("mahalanobis", sig, lv)),
                                              exp_(cell("class_sum_matched", sig, lv)))), src,
                     f"families.{{class_sum_matched,mahalanobis}}.{sig}.{inj}.levels.{lv}.ln_sigma_min",
                     "output ratio over Mahalanobis distance, paired by run [95% Student-t interval]")


def emit_mass_output(em: Emitter, A: dict, a_src: pathlib.Path, files: list, sizes: dict) -> None:
    """1-AUC on b vs c two-prong with and without the mass output. The plain
    models come from the ladder table, the +mass models from their own probe
    files, which must have scored the same jets in the same order."""
    task = "bvc_resonant"
    docs = [json.loads(p.read_text()) for p in files]
    if {d["row_alignment_sha256"] for d in docs} != {A["provenance"]["row_alignment_sha256"]}:
        raise SystemExit("FATAL: the mass-output probe files were not scored on the ladder's jets")
    em.macro("MassNRunsWord", words(len(files)), files[0], "per-seed files (count)",
             "pretraining runs of each mass-output configuration")
    for probe in ("linear", "mlp"):
        k = texname(probe)
        for rung in ("L162", "R16_Q1"):
            lv = sizes[rung]
            stem = {r: a for a, r in ARM_RUNG.items()}[rung] + "mass-"
            plain = seed_rows(A["table"], task, probe, lv)
            mass = [c[probe] for d in docs for a, c in sorted(d["tasks"][task]["arms"].items())
                    if a.startswith(stem)]
            if any(r["censored"] for r in plain) or any(c["log1m_auc_censored"] for c in mass):
                raise SystemExit(f"FATAL: a {task} cell reached AUC=1; its 1-AUC is only a bound")
            x0, x1 = [1 - r["auc"] for r in plain], [1 - c["auc"] for c in mass]
            em.macro("MassOma" + k + texname(lv), fmt_pm_sci(np.mean(x0), np.std(x0, ddof=1)),
                     a_src, f"table[{task},{probe},{lv}].auc",
                     f"1-AUC, mean +- SD over {len(x0)} seeds")
            em.macro("MassOma" + k + texname(lv) + "Mass",
                     fmt_pm_sci(np.mean(x1), np.std(x1, ddof=1)), files[0],
                     f"tasks.{task}.arms.{stem}s*.{probe}.auc",
                     f"1-AUC, mean +- SD over the {len(x1)} per-seed files")
            em.macro("MassOmaRatio" + k + texname(lv), fmt_ratio(np.mean(x1) / np.mean(x0)),
                     files[0], f"tasks.{task}.arms.{stem}s*.{probe}.auc over table[{task},{probe},{lv}].auc",
                     "seed mean of 1-AUC with the mass output over without")


MASS_CELLS = ["188", "162", "43", "17", "162+mass", "17+mass"]


def mass_values(M: dict, cell: str, probe: str, field: str = "sigma_eff") -> list:
    return [r[field] for r in M["table"] if r["cell"] == cell and r["probe"] == probe]


def mass_n_test(M: dict, root: pathlib.Path) -> tuple[int, pathlib.Path]:
    """The test jets sigma_eff is computed on: the split every per-seed file
    records, checked against each arm's own count and against probe.make_splits,
    whose test split is what is left after the first int(0.8 n)."""
    files = [root / x["path"] for x in M["provenance"]["inputs"]]
    n = set()
    for p in files:
        d = json.loads(p.read_text())
        used, te = d["centering_detail"]["n_jets_used"], d["centering_detail"]["split"][2]
        counts = {a[k]["n"] for a in d["arms"].values() for k in ("ridge", "mlp")}
        if te != used - int(0.8 * used) or counts != {te}:
            raise SystemExit(f"FATAL: {p} test split {te} disagrees with make_splits or with "
                             f"its arms' counts {sorted(counts)}")
        n.add(te)
    if len(n) != 1:
        raise SystemExit(f"FATAL: the mass-resolution files disagree on the test split: {sorted(n)}")
    return n.pop(), files[0]


def emit_mass_resolution(em: Emitter, M: dict, src: pathlib.Path, cells=MASS_CELLS) -> None:
    """Frozen-feature jet-mass regression: sigma_eff per label set and probe, over `cells`
    (the first grid's six by default)."""
    n_test, first = mass_n_test(M, em.root)
    em.macro("MassResNTest", fmt_int(n_test), first, "centering_detail.split[2]",
             "test jets sigma_eff is computed on, the same in every per-seed file and arm")
    em.macro("MassResNClasses", str(M["provenance"]["n_classes_used"]), src,
             "provenance.n_classes_used", "native classes with enough training jets to centre")
    for probe in ("ridge", "mlp"):
        k = texname(probe)
        for cell in cells:
            v = mass_values(M, cell, probe)
            em.macro("MassResSigmaEff" + k + texname(cell.replace("+mass", " mass")),
                     fmt_pm(np.mean(v), np.std(v, ddof=1)), src,
                     f"table[cell={cell},probe={probe}].sigma_eff", f"mean +- SD over {len(v)} seeds")
        gain = {lv: np.mean(mass_values(M, lv + "+mass", probe)) - np.mean(mass_values(M, lv, probe))
                for lv in ("162", "17")}
        for lv, g in gain.items():
            em.macro("MassResGain" + k + texname(lv), fmt(g, 4, sign=True), src,
                     f"table[cell={lv}+mass/{lv},probe={probe}].sigma_eff",
                     "sigma_eff with the mass output minus without, difference of the seed means; "
                     "negative is better")
        em.macro("MassResDid" + k, fmt(gain["162"] - gain["17"], 4, sign=True), src,
                 f"table[cell=162+mass/162/17+mass/17,probe={probe}].sigma_eff",
                 "gain at 162 classes minus gain at 17")
    tgt = {round(r["target_sigma_eff"], 12) for r in M["table"] if r["probe"] == "ridge"}
    if len(tgt) == 1:
        em.macro("MassResTargetSigmaEff", fmt(tgt.pop(), 4), src, "table[*].target_sigma_eff",
                 "sigma_eff of the class-centred target itself (a model that predicts the class mean)")
    # The ridge probe's validation R^2 on the four vocabularies without the mass
    # output, from the per-seed files: the share of the target's variance it explains.
    files = [em.root / x["path"] for x in M["provenance"]["inputs"]]
    r2 = [a["ridge"]["val_r2"] for p in files
          for k, a in json.loads(p.read_text())["arms"].items() if k.split("-s")[0] in ARM_RUNG]
    for name, v in (("MassResRidgeRtwoMin", min(r2)), ("MassResRidgeRtwoMax", max(r2))):
        em.macro(name, f"{100 * v:.0f}\\%", files[0], "arms.{l188,l162,r42q1,r16q1}-s*.ridge.val_r2",
                 f"validation R^2 of the ridge probe, {'smallest' if v == min(r2) else 'largest'} "
                 f"of {len(r2)} models without the mass output, all {len(files)} per-seed files")
    for cell in sorted({r["cell"] for r in M["table"]}):
        rows = [r for r in M["table"] if r["cell"] == cell and r["probe"] == "ridge"]
        k = texname(cell.replace("+mass", " mass"))
        for field, nd in (("sigma_eff", 4), ("sd", 3), ("tail_fraction", 3)):
            em.macro("MassRes" + texname(field) + k, fmt(np.mean([r[field] for r in rows]), nd),
                     src, f"table[cell={cell},probe=ridge].{field}",
                     f"mean over {len(rows)} seeds")


def emit_vcb(em: Emitter, V: dict, src: pathlib.Path, sizes: dict) -> None:
    """The |V_cb| window probe, X->bc against its backgrounds, 162 and 17 classes,
    from the per-seed cells."""
    (task, T), = V["tasks"].items()
    eps = "0.60"
    em.macro("VcbNSignal", fmt_int(T["n_signal_test"]), src, f"tasks.{task}.n_signal_test",
             "X->bc test jets in the window")
    em.macro("VcbNBackground", fmt_int(T["n_background_test"]), src,
             f"tasks.{task}.n_background_test", "bq, cs, bqq and QCD test jets in the window")
    em.macro("VcbEpsSixty", f"{100 * float(eps):.0f}", src, f"tasks.{task}.eps_s",
             "signal efficiency, percent")
    lo, hi = sizes["R16_Q1"], sizes["L162"]
    for probe in ("linear", "mlp"):
        k, oma, logs = texname(probe), {}, {}
        for lv in (hi, lo):
            stem = {sizes[r]: a for a, r in ARM_RUNG.items()}[lv]
            cells = [T["arms"][a][probe] for a in sorted(T["arms"]) if a.split("-")[0] == stem]
            jp = f"tasks.{task}.arms.{stem}-s*.{probe}"
            if any(c["log1m_auc_censored"] for c in cells):
                raise SystemExit(f"FATAL: a {task} cell reached AUC=1; its 1-AUC is only a bound")
            logs[lv] = [c["log1m_auc"] for c in cells]
            x = np.exp(logs[lv])
            oma[lv] = float(np.mean(x))
            em.macro("VcbOma" + k + texname(lv), fmt_pm_sci(np.mean(x), np.std(x, ddof=1)), src,
                     jp + ".log1m_auc", f"1-AUC = exp of it, mean +- SD over {len(x)} seeds")
            em.macro("VcbLogOma" + k + texname(lv), fmt(np.mean(logs[lv]), 3, sign=True), src,
                     jp + ".log1m_auc", "mean over seeds")
            pts = [c["rejection_at"][eps] for c in cells]
            em.macro("VcbRej" + k + "Sixty" + texname(lv),
                     fmt_rejection([p["rejection"] for p in pts], [p["rejection_is_bound"] for p in pts],
                                   [p["n_bkg_pass"] for p in pts]),
                     src, jp + f".rejection_at['{eps}'].rejection",
                     "background rejection at this signal efficiency, mean +- SD over seeds")
            em.macro("VcbBkgLeft" + k + "Sixty" + texname(lv),
                     fmt(np.mean([p["n_bkg_pass"] for p in pts]), 1), src,
                     jp + f".rejection_at['{eps}'].n_bkg_pass",
                     "background jets passing the cut, mean over seeds")
        em.macro("VcbOmaRatio" + k, fmt_ratio(oma[lo] / oma[hi]), src,
                 f"tasks.{task}.arms.*.{probe}.log1m_auc", f"seed mean of 1-AUC, {lo} classes over {hi}")
        em.macro("VcbFactor" + k, fmt(np.exp(np.mean(logs[lo]) - np.mean(logs[hi])), 2), src,
                 f"tasks.{task}.arms.*.{probe}.log1m_auc",
                 f"exp of the difference of the seed means of log(1-AUC), {lo} classes over {hi}")


AOJ_SETS = ("188", "162", "43", "17", "162+mass", "17+mass")


def aoj_n_runs(P: dict) -> int:
    """Pretraining runs behind every open-data row; one number, or the captions lie."""
    n = {P["label_sets"][lv]["signal_yield"]["n"] for lv in AOJ_SETS}
    if len(n) != 1:
        raise SystemExit(f"FATAL: the open-data rows hold {sorted(n)} runs; the text states one")
    return n.pop()


def emit_aoj_settings(em: Emitter, code: pathlib.Path) -> None:
    """The open-data selection and fit settings the text states, read from
    experiments/AOJ/peak_fit.py, which sets them."""
    s = code.read_text()

    def one(pat):
        m = re.findall(pat, s, re.M)
        if len(m) != 1:
            raise SystemExit(f"FATAL: {code} matches {pat} {len(m)} times; expected once")
        return m[0]
    lo, hi = one(r"^RHO_RANGE = \(([-0-9.]+), ([-0-9.]+)\)")
    lo_pt, hi_pt = one(r"^PT_RANGE = \(([0-9.]+), ([0-9.]+)\)")
    em.macro("AojPtMin", fmt_int(lo_pt), code, "PT_RANGE[0]", "GeV")
    em.macro("AojPtMax", fmt_int(hi_pt), code, "PT_RANGE[1]", "GeV")
    em.macro("AojRhoMin", fmt(float(lo), 1), code, "RHO_RANGE[0]")
    em.macro("AojRhoMax", fmt(float(hi), 1), code, "RHO_RANGE[1]")
    for peak, key in (("W", "AojWindowW"), ("top", "AojWindowTop")):
        w = one(rf"^\s*{peak}=dict\(window=\(([0-9.]+), ([0-9.]+)\), fit_range=\(([0-9.]+), ([0-9.]+)\)")
        em.macro(key + "Lo", fmt_int(w[0]), code, f"PEAKS[{peak}].window[0]", "GeV")
        em.macro(key + "Hi", fmt_int(w[1]), code, f"PEAKS[{peak}].window[1]", "GeV")
        if peak == "top":
            em.macro("AojFitLo", fmt_int(w[2]), code, "PEAKS[top].fit_range[0]", "GeV")
            em.macro("AojFitHi", fmt_int(w[3]), code, "PEAKS[top].fit_range[1]", "GeV")
    em.macro("AojMassBin", fmt_int(one(r"^MASS_BIN = ([0-9.]+)")), code, "MASS_BIN", "GeV")
    o = one(r"^MAP_ORDER, MAP_CELLS, MAP_MIN_PASS, MAP_ITERATIONS = \((\d+), (\d+)\)")
    em.macro("AojMapOrderRho", o[0], code, "MAP_ORDER[0]", "polynomial order in rho")
    em.macro("AojMapOrderPt", o[1], code, "MAP_ORDER[1]", "polynomial order in ln pT")


def run_index(model: str) -> int:
    """`l162-s1b` -> 1: the pretraining run a model belongs to."""
    return int(re.search(r"-s(\d+)", model).group(1))


def paired_ratio(x: dict, y: dict) -> tuple[float, float, float]:
    """y over x, paired run by run: the geometric mean of the per-run ratios and its 95 %
    Student-t interval from their spread (n - 1 degrees of freedom). Each run's value
    already carries its own statistical error, so the spread of the ratios holds it."""
    from scipy import stats
    runs = sorted(set(x) & set(y))
    if len(runs) < 3:
        raise SystemExit(f"FATAL: a paired ratio over {len(runs)} runs; the text quotes none below three")
    d = np.log([y[r] / x[r] for r in runs])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))


def ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")


def aoj_fits(root: pathlib.Path, J: dict) -> tuple[pathlib.Path, dict]:
    """The fit results an analysis_vN read, from the committed copy its hash names."""
    p = committed_path(root, J["provenance"]["input"], J["provenance"]["input_sha256"])
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != J["provenance"]["input_sha256"]:
        raise SystemExit(f"FATAL: no committed file holds {J['provenance']['input']} at the hash "
                         f"the real-data readout recorded")
    return p, json.loads(p.read_text())


def emit_real_data(em: Emitter, J: dict, src: pathlib.Path, paths: dict | None = None,
                   fits: tuple | None = None) -> None:
    """The top peak in CMS open data (AspenOpenJets) at 1% data efficiency, from fit_v6:
    one Gaussian peak shape shared by the pretrained models (pooled over them) and the
    tops that fail each cut taken from the CMS reference's fit (experiments/AOJ/fit_v6.py).
    The yield per vocabulary is the mean +- SD over the pretraining runs; each model's
    own floated shape, the shape and fail-region systematics, the working-point fit
    quality and the passing-jet residual below the top window are the checks."""
    paths = paths or {}
    res_path, res = fits or aoj_fits(em.root, J)
    em.macro("AojEffPercent", fmt_one(100 * float(res["eff"]), 0) + "\\%", res_path, "eff",
             "the score cut's pass fraction outside the mass windows")
    code = em.root / "experiments" / "AOJ" / "peak_fit.py"
    if code.exists():
        emit_aoj_settings(em, code)
    P = J["per_label_set"]
    pool = P["shapes"]["pool"]
    if [res["pooled_shape"]["mean"], res["pooled_shape"]["width"]] != list(P["shapes"]["pooled"]) \
            or sorted(res["pooled_shape"]["pool"]) != sorted(pool):
        raise SystemExit(f"FATAL: {src} and {res_path} disagree on the shared peak shape")
    em.macro("AojNRunsWord", words(aoj_n_runs(P)), src,
             "per_label_set.label_sets.*.signal_yield.n", "pretraining runs per vocabulary")
    # n_jets counts the staged jets inside the rho window, all files: the
    # population the score map is built on, NOT the jets in the mass fit (the
    # fit's 105-300 GeV range holds fewer; AojNJetsFit, from the selection chain).
    em.macro("AojNJetsRhoWindow", fmt_int(res["n_jets"]), res_path, "n_jets",
             "staged jets inside the rho window, all files; not the jets in the mass fit")
    shift = []
    for lv in AOJ_SETS:
        c, k = P["label_sets"][lv], texname(lv.replace("+mass", " mass"))
        y, own = c["signal_yield"], c["by_shape"]["own_shape"]
        em.macro("AojYield" + k, fmt_pm(y["mean"], y["sd"]), src,
                 f"per_label_set.label_sets.{lv}.signal_yield", f"mean +- SD over {y['n']} runs, shared shape")
        em.macro("AojStatErr" + k, fmt_int(c["median_stat_err"]), src,
                 f"per_label_set.label_sets.{lv}.median_stat_err", "median per-fit statistical error")
        em.macro("AojYieldOwnShape" + k, fmt_pm(own["mean"], own["sd"]), src,
                 f"per_label_set.label_sets.{lv}.by_shape.own_shape", "every fit with its own floated shape")
        shift.append(own["mean"] / y["mean"] - 1)
        em.macro("AojSpreadOverStat" + k, fmt(y["sd"] / c["median_stat_err"], 1), src,
                 f"per_label_set.label_sets.{lv}.signal_yield.sd / median_stat_err",
                 "spread between runs over the median per-fit statistical error")
    for name, v in (("AojOwnShapeShiftMin", min(shift)), ("AojOwnShapeShiftMax", max(shift))):
        em.macro(name, fmt(100 * v, 0, sign=True) + "\\%", src,
                 "per_label_set.label_sets.*.by_shape.own_shape.mean / signal_yield.mean - 1",
                 "the vocabulary means with each model's own shape, relative to the shared shape")
    for name, v in zip(("AojPooledMean", "AojPooledWidth"), P["shapes"]["pooled"]):
        em.macro(name, fmt(v, 1), src, "per_label_set.shapes.pooled", "GeV")
    em.macro("AojPoolSize", str(len(pool)), src, "per_label_set.shapes.pool (count)",
             "pretrained models the shared shape is fitted over")
    models = {m: v["top"] for m, v in res["models"].items() if m in pool}
    if sorted(models) != sorted(pool):
        raise SystemExit(f"FATAL: {res_path} lacks fits of {sorted(set(pool) - set(models))}")
    em.macro("AojNModels", str(len(models)), res_path, "models (pretrained)", "fits")
    for q, name in (("mean", "Mean"), ("width", "Width")):
        vals = [f["own_shape_fit"][q] for f in models.values()]
        em.macro(f"AojOwn{name}Min", fmt(min(vals), 1), res_path, f"models.*.top.own_shape_fit.{q} (min)", "GeV")
        em.macro(f"AojOwn{name}Max", fmt(max(vals), 1), res_path, f"models.*.top.own_shape_fit.{q} (max)", "GeV")
    step = res["pooled_shape"]["systematic_step"]
    for q in ("mean", "width"):
        Q = q.capitalize()
        em.macro(f"AojShapeStep{Q}", fmt(step[q], 1), res_path, f"pooled_shape.systematic_step.{q}",
                 "GeV: the SD of the models' own floated values, one shift for every model together")
        rel = [abs(f["shape_systematic"][f"pooled_{q}_{d}"] / f["signal_yield"] - 1)
               for f in models.values() for d in ("down", "up")]
        em.macro(f"AojShapeSyst{Q}Max", f"{100 * max(rel):.0f}\\%", res_path,
                 f"models.*.top.shape_systematic.pooled_{q}_(down|up) / signal_yield - 1 (largest |.|)",
                 "the largest yield change of any model under that shift")
    eps = {tuple(f["leak_systematic"]["eps_ref"]) for f in models.values()}
    if len(eps) != 1:
        raise SystemExit(f"FATAL: the fail-region systematic was taken at {sorted(eps)}")
    eps_ref = eps.pop()
    hi, lo = max(eps_ref), min(eps_ref)
    em.macro("AojLeakEpsHi", f"{hi:g}", res_path, "models.*.top.leak_systematic.eps_ref (max)",
             "the reference's assumed efficiency for data tops, upper variation")
    em.macro("AojLeakEpsLo", f"{lo:g}", res_path, "models.*.top.leak_systematic.eps_ref (min)",
             "the same, lower variation")
    em.macro("AojLeakShiftMin", "+" + fmt_int(min(f["leak_systematic"]["shift"][f"{hi:g}"] for f in models.values())),
             res_path, f"models.*.top.leak_systematic.shift['{hi:g}'] (min)", "tops added to a yield")
    em.macro("AojLeakShiftMax", "+" + fmt_int(max(f["leak_systematic"]["shift"][f"{lo:g}"] for f in models.values())),
             res_path, f"models.*.top.leak_systematic.shift['{lo:g}'] (max)", "tops added to a yield")
    p_fit = {m: f["asymptotic_p"] for m, f in models.items()}
    em.macro("AojFitPMin", fmt(min(p_fit.values()), 3), res_path, "models.*.top.asymptotic_p (min)",
             "goodness of fit of the working-point fit, pretrained models")
    em.macro("AojFitPMax", fmt(max(p_fit.values()), 3), res_path, "models.*.top.asymptotic_p (max)")
    weak = min(models, key=lambda m: models[m]["signal_yield"])
    em.macro("AojWeakestYield", fmt_int(models[weak]["signal_yield"]), res_path,
             f"models.{weak}.top.signal_yield", "the smallest yield of any pretrained model")
    em.macro("AojWeakestYieldErr", fmt_int(models[weak]["signal_yield_err"]), res_path,
             f"models.{weak}.top.signal_yield_err")
    em.macro("AojWeakestFitP", fmt(p_fit[weak], 2), res_path, f"models.{weak}.top.asymptotic_p")
    # The passing jets below the top window, from the fit's own histograms
    # (experiments/AOJ/sideband_residuals.py): the summed residual of every model, its
    # largest single-bin pull, and that bin read against the models' mean pull per bin.
    hists = res_path.parent / "histograms.npz"
    if hists.exists():
        spec = importlib.util.spec_from_file_location(
            "sideband_residuals", em.root / "experiments" / "AOJ" / "sideband_residuals.py")
        SR = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(SR)
        side = {m: r for m, r in SR.all_residuals(hists).items() if m in pool}
        z = [r["z"] for r in side.values()]
        em.macro("AojSidebandZMin", fmt(min(z), 2, sign=True), hists,
                 "experiments/AOJ/sideband_residuals.py: z (min over the pretrained models)",
                 "passing jets minus the fit, summed from the fit's lower edge to the top window")
        em.macro("AojSidebandZMax", fmt(max(z), 2, sign=True), hists,
                 "experiments/AOJ/sideband_residuals.py: z (max over the pretrained models)")
        zw = side[weak]["z"]
        em.macro("AojWeakestSidebandZ", fmt(zw, 2, sign=True) if round(zw, 2) else fmt(0, 2), hists,
                 f"experiments/AOJ/sideband_residuals.py: z of {weak}", "the smallest-yield model")
        big = max(side, key=lambda m: side[m]["max_abs_pull"])
        em.macro("AojSidebandMaxPull", fmt(side[big]["max_abs_pull"], 1), hists,
                 "experiments/AOJ/sideband_residuals.py: max_abs_pull (max over the pretrained models)",
                 "largest single-bin pull below the top window, any pretrained model")
        bins = collections.Counter(tuple(r["max_pull_bin_gev"]) for r in side.values())
        (blo, bhi), n_here = bins.most_common(1)[0]
        mp = SR.mean_pulls(hists, sorted(side))
        i = mp["edges"].index(blo)
        em.macro("AojSharedBinLo", fmt_int(blo), hists, "the bin holding the most models' largest pull", "GeV")
        em.macro("AojSharedBinHi", fmt_int(bhi), hists, "its upper edge", "GeV")
        em.macro("AojSharedBinNModels", str(n_here), hists, "models whose largest pull is in that bin")
        em.macro("AojSharedBinMeanPull", fmt(mp["mean_pull"][i], 1, sign=True), hists,
                 "experiments/AOJ/sideband_residuals.py mean_pulls: that bin", "mean pull over the pretrained models")
        em.macro("AojSharedBinRank", ordinal(mp["rank"][i]), hists,
                 "experiments/AOJ/sideband_residuals.py mean_pulls: rank of that bin's |mean pull|")
        em.macro("AojFitNBins", str(len(mp["mean_pull"])), hists, "mass bins of the fit range")
    # Paired, run by run: each vocabulary's yield over the 188-class yield, and each
    # mass-output configuration's over the same vocabulary without it.
    per = {lv: {run_index(m): y for m, y in zip(P["label_sets"][lv]["models"], P["label_sets"][lv]["signal_yields"])}
           for lv in AOJ_SETS}
    for coarse, fine in (("162", "188"), ("43", "188"), ("17", "188"), ("17", "162"),
                         ("162+mass", "162"), ("17+mass", "17")):
        r, a, b = paired_ratio(per[fine], per[coarse])
        em.macro("AojPairedYield" + texname(coarse.replace("+mass", " mass")) + "Over"
                 + texname(fine), fmt_paired(r, a, b), src,
                 f"per_label_set.label_sets.{{{coarse},{fine}}}.signal_yields",
                 "yield ratio paired by run, geometric mean [95% Student-t interval]")
    # The fail-region tops come from the reference's fit, common to every model. Scaled
    # up for a lower efficiency of the reference (the leak systematic), how far does each
    # paired ratio between configurations move?
    moves = []
    for e in eps_ref:
        scaled, nominal = ({lv: {run_index(m): (res["models"][m]["top"]["leak_systematic"]["signal_yield"][f"{e:g}"]
                                                if key else res["models"][m]["top"]["signal_yield"])
                                 for m in P["label_sets"][lv]["models"]} for lv in AOJ_SETS}
                           for key in (True, False))
        for coarse, fine in (("162", "188"), ("43", "188"), ("17", "188"), ("17", "162"),
                             ("162+mass", "162"), ("17+mass", "17")):
            if nominal[fine] != per[fine] or nominal[coarse] != per[coarse]:
                raise SystemExit(f"FATAL: {src} and {res_path} disagree on the {coarse}/{fine} yields")
            moves.append(abs(paired_ratio(scaled[fine], scaled[coarse])[0]
                             / paired_ratio(per[fine], per[coarse])[0] - 1))
    em.macro("AojLeakPairedShiftMax", f"{100 * max(moves):.0f}\\%", res_path,
             "models.*.top.leak_systematic.signal_yield against signal_yield, paired ratios",
             "largest change of a paired yield ratio when the fail-region tops are scaled up")
    ref, pub = res["reference"]["top"], res["models"]["sophon-public"]["top"]
    em.macro("AojRefEffPercent", fmt(100 * ref["data_efficiency"], 2) + "\\%", res_path,
             "reference.top.data_efficiency", "fraction of all data jets passing its cut")
    em.macro("AojRefFitP", fmt_p(ref["asymptotic_p"]), res_path, "reference.top.asymptotic_p",
             "goodness of fit of the reference's own top fit")
    v = ref["validation"]
    n_toys = int(v.get("n_toys", res["n_toys"]))
    worse = v.get("n_toys_worse")
    if worse is None:                     # peak_fit.toy_p_value: (worse + 1) / (n + 1)
        worse = round(v["toy_p"] * (n_toys + 1) - 1)
    if abs((worse + 1) / (n_toys + 1) - v["toy_p"]) > 1e-12:
        raise SystemExit(f"FATAL: reference.top.validation.toy_p {v['toy_p']} is not "
                         f"(k + 1) / ({n_toys} + 1) for any count k of toys")
    if worse == 0:
        # No toy reached the observed deviance: 1/(n+1) is the floor of the estimate,
        # so the p-value is only bounded, rounded up to one significant figure.
        place = -math.floor(math.log10(v["toy_p"]))
        p_txt = f"$p \\leq {math.ceil(v['toy_p'] * 10 ** place) / 10 ** place:g}$"
    else:
        p_txt = f"$p = {fmt_one(v['toy_p'])}$"
    em.macro("AojRefValidationP", f"{p_txt} ({worse} of {n_toys} toys reach the observed deviance)",
             res_path, "reference.top.validation.toy_p, n_toys",
             "toy p-value of the background validation, (k + 1) / (n + 1)")
    em.macro("AojRefBandSignalZ", fmt(v["band_signal_z"], 1), res_path,
             "reference.top.validation.band_signal_z",
             "significance of a peak of the score's own shape in the validation band")
    eff = [f["data_efficiency"] for f in models.values()]
    em.macro("AojModelEffMin", fmt(100 * min(eff), 2) + "\\%", res_path,
             "models.*.top.data_efficiency (min)", "pretrained models")
    em.macro("AojModelEffMax", fmt(100 * max(eff), 2) + "\\%", res_path,
             "models.*.top.data_efficiency (max)", "pretrained models")
    em.macro("AojModelsValidationPMin", fmt(min(f["validation"]["toy_p"] for f in models.values()), 2),
             res_path, "models.*.top.validation.toy_p (min)", "pretrained models")
    for name, f, path in (("AojRef", ref, "reference.top"), ("AojPublic", pub, "models.sophon-public.top")):
        em.macro(name + "Yield", fmt_int(f["signal_yield"]), res_path, path + ".signal_yield")
        em.macro(name + "YieldErr", fmt_int(f["signal_yield_err"]), res_path, path + ".signal_yield_err")
        em.macro(name + "Mass", fmt(f["floated_mean"], 1), res_path, path + ".floated_mean", "GeV")
        em.macro(name + "Width", fmt(f["floated_width"], 1), res_path, path + ".floated_width", "GeV")
    # Pseudo-data: tops injected into the top window of the data, fitted back.
    inj = paths.get("aoj_injection")
    if inj and pathlib.Path(inj).exists():
        G = json.loads(pathlib.Path(inj).read_text())["groups"]
        fl = [g for g in G if (g["region"], g["mode"], g["variant"]) == ("top", "bootstrap", "float")]
        en = [g for g in G if (g["region"], g["mode"], g["variant"]) == ("top", "ensemble", "ensemble")]
        if not fl or not en or {g["size"] for g in fl} != {g["size"] for g in en}:
            raise SystemExit(f"FATAL: {inj} lacks matched floated-shape and whole-procedure top-window toys")
        sizes = sorted(g["size"] for g in en)
        em.macro("AojInjSizeMin", fmt_int(sizes[0]), inj, "groups[region=top,mode=ensemble].size (min)", "injected tops")
        em.macro("AojInjSizeMax", fmt_int(sizes[-1]), inj, "groups[region=top,mode=ensemble].size (max)")
        em.macro("AojInjFloatPullSdMax", fmt(max(g["pull"]["sd"] for g in fl), 1), inj,
                 "groups[region=top,mode=bootstrap,variant=float].pull.sd (max)",
                 "pull width with each model's own floated shape")
        for name, q, nd in (("Recovery", "ratio", 3), ("PullSd", "pull", 2)):
            stat = [g[q]["mean" if q == "ratio" else "sd"] for g in en]
            em.macro(f"AojInjEnsemble{name}Min", fmt(min(stat), nd), inj,
                     f"groups[region=top,mode=ensemble].{q} (min)", "the whole procedure, shape re-derived per toy set")
            em.macro(f"AojInjEnsemble{name}Max", fmt(max(stat), nd), inj,
                     f"groups[region=top,mode=ensemble].{q} (max)")
    # The inputs the network receives on the open data against JetClass-II QCD jets,
    # pooled over every staged shard (aoj_checks_v2).
    cl = paths.get("aoj_closure")
    if cl and pathlib.Path(cl).exists():
        C = json.loads(pathlib.Path(cl).read_text())
        if C["hard_flags"]:
            raise SystemExit(f"FATAL: {cl} raises hard flags {C['hard_flags']}")
        em.macro("AojClosureNShards", words(C["n_shards"]), cl, "n_shards", "staged shards of the open data")
        em.macro("AojClosurePtLo", fmt_int(C["pt_window"][0]), cl, "pt_window[0]", "GeV")
        em.macro("AojClosurePtHi", fmt_int(C["pt_window"][1]), cl, "pt_window[1]", "GeV")
        for feat, name in (("part_d0", "AojClosureDzeroIqr"), ("part_dz", "AojClosureDzIqr")):
            em.macro(name, fmt(C["domain_shift"][feat]["iqr_ratio"], 1), cl,
                     f"domain_shift.{feat}.iqr_ratio", "open data over JetClass-II QCD, pooled over the shards")
        for feat, name in (("part_isNeutralHadron", "AojClosureNeutral"),
                           ("part_isChargedHadron", "AojClosureCharged")):
            em.macro(name, fmt(np.mean(C["features"][feat]["ratio"]["per_shard"]), 2), cl,
                     f"features.{feat}.ratio.per_shard (mean)", "open data over JetClass-II QCD")
        n = C["features"]["n_particles"]
        em.macro("AojClosureNPartAoj", fmt_int(np.mean(n["aoj"]["per_shard"])), cl,
                 "features.n_particles.aoj.per_shard (mean of the shard medians)", "median")
        em.macro("AojClosureNPartRef", fmt_int(np.mean(n["reference"]["per_shard"])), cl,
                 "features.n_particles.reference.per_shard (mean)", "median")


# ------------------------------------------------------------------ tables

def _table(body: list[str], caption: str, label: str, colspec: str,
           notes: list[str], wide: bool = False) -> str:
    env = "table*" if wide else "table"
    # adjustbox shrinks a table wider than the text and leaves a narrower one
    # alone. The notes sit below the tabular, not in a \\multicolumn of it: a
    # note row of paragraph width would itself force the table to full width.
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += [text_minus(b) for b in body]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    for note in notes:
        lines.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    lines += [f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def table_probe_ladder(A: dict, probe: str, nbkg: dict) -> str:
    """T1: granularities down the side, probe tasks across the top.

    Two rows per granularity rather than a two-line cell: the second column says
    which quantity the row holds, so the table needs nothing but booktabs.
    """
    tasks = ordered_tasks(A["levels"])
    levels = A["levels_fine_to_coarse"]
    key, eps = headline_rejection(A["levels"][tasks[0]][probe][0])
    head = ["classes & quantity & " + " & ".join(TASK_LABELS.get(t, tex(t)) for t in tasks)
            + " \\\\", "\\midrule"]
    body, auc_all, rej_all = [], [], []
    n_seeds = set()
    for i, lv in enumerate(levels):
        auc, rej = [], []
        for t in tasks:
            rows = seed_rows(A["table"], t, probe, lv)
            n_seeds.add(len(rows))
            auc.append(fmt_auc_pm([r["auc"] for r in rows], [r["censored"] for r in rows]))
            rej.append(fmt_rejection(*seed_rejections(A["table"], t, probe, lv, key)))
        auc_all += auc
        rej_all += rej
        body.append(f"{lv} & AUC & " + " & ".join(auc) + " \\\\")
        body.append(f"     & $1/\\epsilon_B$ at {eps * 100:.0f}\\% & " + " & ".join(rej) + " \\\\")
        if i < len(levels) - 1:
            body.append("\\addlinespace")
    caption = (f"Frozen {'linear' if probe == 'linear' else 'nonlinear (MLP)'} probes on the "
               f"four pretraining vocabularies: AUC and the background rejection $1/\\epsilon_B$ at "
               f"{eps * 100:.0f}\\% signal efficiency, mean {tex('±')} standard deviation over the "
               f"{words(min(n_seeds))} pretraining runs. Rows are the number of classes in the "
               f"pretraining vocabulary, finest first and coarsest last.")
    return _table(head + body, caption, f"tab:probes-{probe}",
                  "r l " + "r" * len(tasks), _probe_notes(" ".join(auc_all + rej_all), tasks, nbkg),
                  wide=True)


def _probe_notes(cells: str, tasks: list, nbkg: dict) -> list:
    """The footnotes a probe table needs, from the text of its cells: what a bound and a
    saturated AUC mean, and the background test jets per task."""
    notes = []
    if "$>$" in cells:
        notes.append("$>$ at most one background jet passed the cut in every run; the entry is "
                     "the 95\\% confidence lower limit on the rejection, $N_B/3.0$ when none passed "
                     "and $N_B/4.74$ when one did, with $N_B$ the number of background test jets.")
    if "$\\geq$" in cells:
        notes.append("$\\geq$ with $^{\\ast}$: in some runs at most one background jet passed "
                     "the cut; the entry is the median over runs with those runs at $N_B$, not a "
                     "measured value.")
    if "dagger" in cells:
        notes.append("$^{\\dagger}$ the AUC reached 1 at the resolution of the sample in at least "
                     "one run; $1-$AUC is then an upper bound and the cell is not a measurement.")
    if nbkg:
        notes.insert(0, "Background test jets $N_B$ per task: " + "; ".join(
            f"{TASK_LABELS.get(t, tex(t))}, {fmt_int(nbkg[t])}" for t in tasks if t in nbkg) + ".")
    return notes


def table_usecase(surv: dict, sizes: dict, pretrained: list) -> str:
    """Table 1: which published discriminant is still constructible at each vocabulary."""
    cols = [sizes[r] for r in RUNGS]
    head = ["discriminant & source & " + " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    rows = []
    for disc in usecase_rows(surv):
        row = surv[disc]
        marks = " & ".join("$\\bullet$" if row["constructible"][r] else "---" for r in RUNGS)
        ref, eq = row["source"].split(" ", 1)
        eq = eq.replace("Eq. ", "Eqs.~" if ")-(" in eq else "Eq.~").replace(")-(", ")--(")
        cite = f"\\cite{{{USECASE_CITE[ref]}}}" if ref in USECASE_CITE else tex(ref)
        rows.append(f"{USECASE_ROWS[disc]} & {cite}, {eq} & {marks} \\\\")
    caption = ("Published discriminants built from sums of Sophon's output scores, against the "
               "number of classes in the vocabulary (columns, finest first). $\\bullet$: the "
               "discriminant can be built exactly from that vocabulary's outputs; ---: it cannot. "
               "By Sophon's class-division property (Property~1, Eq.~(2) of "
               "Ref.~\\cite{sophon}), which holds for an ideally trained classifier, the score of a "
               "merged class is the sum of the scores of the classes it merges, so a discriminant "
               "survives a merge only if its numerator and its denominator each contain every merged "
               "class or none of them. The table states which sums a vocabulary's outputs can form, a "
               "property of the partition that needs no training and no data, not that a trained "
               "coarse model reproduces them.")
    notes = ["Pretrained in this study: the "
             + ", ".join(str(sizes[r]) for r in pretrained[:-1])
             + f" and {sizes[pretrained[-1]]}-class vocabularies."]
    return _table(head + rows, caption, "tab:usecase", "l l " + "c" * len(cols), notes,
                  wide=True)


def table_legs(legs: dict, sizes: dict) -> str:
    """The fine-tuning legs. WAVE 1, SUPERSEDED, and the caption says so first."""
    # Columns are the fine-tuning set sizes, smallest first; anything that is not
    # an N-cell (the single full-data reference point) goes last.
    ns = sorted({n for d in legs.values() for s in d["summary"].values() for n in s},
                key=lambda k: (0, int(k[1:])) if k.startswith("N") and k[1:].isdigit() else (1, 0))
    inits = sorted({i for d in legs.values() for i in d["summary"]})
    head = ["& " + " & ".join(f"\\multicolumn{{{len(ns)}}}{{c}}{{leg {leg}}}" for leg in legs)
            + " \\\\",
            "initialisation & " + " & ".join(fmt_n_jets(n) for _ in legs for n in ns)
            + " \\\\", "\\midrule"]
    rows = []
    for init in inits:
        cells = []
        for leg, d in legs.items():
            for n in ns:
                cell = d["summary"].get(init, {}).get(n)
                if cell is None:
                    cells.append("---")
                    continue
                sd = cell.get("accuracy_sd")
                cells.append(fmt(cell["accuracy_mean"], 4)
                             + (f"\\,{tex('±')}\\,{fmt(sd, 4)}" if sd else ""))
        rows.append(f"{init_label(init, sizes)} & " + " & ".join(cells) + " \\\\")
    caption = ("WAVE 1, SUPERSEDED. Fine-tuning accuracy, mean over fine-tuning seeds "
               f"{tex('±')} their standard deviation, against the number of fine-tuning jets "
               "$N$. Leg 1 is in-domain, leg 2 is the domain-shifted transfer. These runs come "
               "from the first fine-tuning wave, whose 162-class side has a single pretraining "
               "seed; they are reported for completeness and are superseded by the five-seed "
               "wave.")
    notes = ["--- is an initialisation, or a fine-tuning set size, that leg did not run."]
    return _table(head + rows, caption, "tab:finetuning-wave1",
                  "l " + " ".join("r" * len(ns) for _ in legs), notes, wide=True)


SIGNAL_LABELS = {"label_X_bb": "$X\\to b\\bar b$", "label_X_qq": "$X\\to q\\bar q$",
                 "label_X_YY_bbb": "$X\\to YY\\to bbb$", "label_X_YY_bbbb": "$X\\to YY\\to bbbb$",
                 "label_X_YY_qqq": "$X\\to YY\\to qqq$", "label_X_YY_qqqq": "$X\\to YY\\to qqqq$"}
FAMILY_LABELS = {"class_sum": "output ratio", "class_sum_matched": "output ratio",
                 "mahalanobis": "Mahalanobis",
                 "knn": "$k$-nearest neighbours", "iad_hgb": "classifier-based (HGB)"}


def table_finetune(ft: dict, sizes: dict, metric: str, rows: list | None = None,
                   v2: bool = False) -> str:
    """Fine-tuning on JetClass-II and JetClass, one metric, per pretrained model
    (rows) and fine-tuning set size (columns). `rows` default to the first grid's;
    `v2` writes the second grid's caption."""
    rows = rows or ft_rows(sizes)
    ns = ft_sizes(next(iter(ft.values()))["cells"], rows)
    head = ["training jets & " + " & ".join(fmt_n_jets(n) for n in ns) + " \\\\", "\\midrule"]
    body = []
    for j, F in enumerate(ft.values()):
        text = ft_text(F["cells"], rows, metric)
        body.append(f"\\multicolumn{{{1 + len(ns)}}}{{@{{}}l}}{{\\itshape {F['name']}, "
                    f"{F['classes']} classes}} \\\\")
        for key, label, inits in rows:
            n = len(inits)
            label += " (one run)" if n == 1 else f" ($n={n}$)" if n < 5 else ""
            body.append(f"{label} & " + " & ".join(text[(key, s)] for s in ns) + " \\\\")
        if j < len(ft) - 1:
            body.append("\\addlinespace")
    auc = metric == "macro_auc_ovr"
    n_test = " and ".join(
        f"{fmt_int(ft_n_test(F['cells'], rows, 'n_jets_auc' if auc else 'n_jets'))} ({F['name']})"
        for F in ft.values())
    caption = ("Fine-tuning every pretrained model on "
               + " and on ".join(f"{F['name']}'s {F['classes']}-class task" for F in ft.values())
               + ": " + ("macro-averaged one-vs-rest AUC" if auc else "accuracy")
               + f" on {n_test} test jets, at the epoch of best validation accuracy. Mean "
               f"{tex('±')} standard deviation over the "
               f"{words(max(len(i) for *_, i in rows))} pretraining runs, each fine-tuned once; "
               "each random partition of the random-label control is one pretraining run and is shown "
               "on its own. Rows are the pretraining vocabulary, "
               "columns the number of fine-tuning training jets.")
    if v2:
        caption = ("Fine-tuning every pretrained model of the second set of runs, from its primary "
                   "checkpoint (the first maximum of the validation accuracy within epochs 70--79), on "
                   + " and on ".join(f"{F['name']}'s {F['classes']}-class task" for F in ft.values())
                   + ", the JetClass-II jets drawn from files held out from pretraining: "
                   + ("macro-averaged one-vs-rest AUC" if auc else "accuracy")
                   + f" on {n_test} test jets, at the epoch of best validation accuracy. Mean "
                   f"{tex('±')} standard deviation over each row's pretraining runs ($n$ where it is "
                   f"not {words(max(len(i) for *_, i in rows))}), each fine-tuned once. Rows are the "
                   "pretraining vocabulary, columns the number of fine-tuning training jets.")
    return _table(head + body, caption, "tab:finetune" if auc else "tab:finetune-accuracy",
                  "l " + "r" * len(ns), [])


def table_anomaly(S: dict, root: pathlib.Path) -> str:
    """Anomaly detection: sigma_min and max SIC per signal, detector and label set.
    Signals no feature-based detector sees at any label set are listed in a note."""
    sigma_t = json.loads((root / S["provenance"]["inputs"]["anomaly"]["path"]).read_text())["sigma_t"]
    nd = set(S["not_detected_rule"]["not_detected"])
    levels = sorted({int(k) for f in ANOMALY_FAMILIES for s in S["families"][f]
                     for k in S["families"][f][s][S["conventions"]["primary_injection"]]["levels"]},
                    reverse=True)
    sigs = [g for g in SIGNAL_LABELS if any(f"{f}|{g}" not in nd for f in ANOMALY_FAMILIES)]
    head = ["& detector & " + " & ".join(f"{lv} classes" for lv in levels) + " \\\\", "\\midrule"]
    body = []
    for qty, name in (("sigma_min", "$\\sigma_{\\min}$"), ("max_sic", "max SIC")):
        body.append(f"\\multicolumn{{{2 + len(levels)}}}{{@{{}}l}}{{\\itshape {name}}} \\\\")
        for g in sigs:
            for i, f in enumerate(ANOMALY_FAMILIES):
                c = anomaly_cells(S, f, g)
                cells = [fmt_pm(np.mean(c[lv][qty]), np.std(c[lv][qty], ddof=1)) for lv in levels]
                body.append((SIGNAL_LABELS[g] if i == 0 else "") + f" & {FAMILY_LABELS[f]} & "
                            + " & ".join(cells) + " \\\\")
        if qty == "sigma_min":
            body.append("\\addlinespace")
    light = [g for g in SIGNAL_LABELS if all(f"{f}|{g}" in nd for f in ANOMALY_FAMILIES)]
    caption = (f"Anomaly detection with the resonance-to-QCD probability ratio from the model's own "
               f"outputs (output ratio) and two detectors on the frozen features, "
               f"{fmt_int(S['conventions']['primary_injection'])} signal jets injected: "
               "$\\sigma_{\\min}$, the smallest initial significance from which a "
               f"${fmt(sigma_t, 0)}\\sigma$ "
               "discovery is still reached (lower is more sensitive), and the maximum significance "
               f"improvement (max SIC). Mean {tex('±')} standard deviation over the "
               f"{words(anomaly_n_runs(S))} pretraining runs of each run's median over "
               f"{words(S['provenance']['resamplings_per_seed'])} resamplings of the background "
               "and signal samples; Table~\\ref{tab:anomaly-per-run} gives each run's value.")
    thr = fmt(S['not_detected_rule']['threshold_max_sic'], 1)
    notes = [f"A detector whose max SIC stays below {thr} at every vocabulary does not detect "
             "that signal."]
    if light:
        notes.append("Not detected by any of the three at any vocabulary, and not shown: "
                     + ", ".join(SIGNAL_LABELS[g] for g in light) + ".")
    return _table(head + body, caption, "tab:anomaly", "l l " + "r" * len(levels), notes)


def table_anomaly_per_run(S: dict) -> str:
    """sigma_min of every pretraining run, for every signal and score Table 8 shows:
    the values behind its mean +- SD, ordered by run. A single run's defective
    output layer shows here and not in a mean."""
    inj = S["conventions"]["primary_injection"]
    nd = set(S["not_detected_rule"]["not_detected"])
    first = S["families"][ANOMALY_FAMILIES[0]]
    levels = sorted({int(k) for k in first[next(iter(first))][inj]["levels"]}, reverse=True)
    sigs = [g for g in SIGNAL_LABELS if any(f"{f}|{g}" not in nd for f in ANOMALY_FAMILIES)]
    run = lambda a: int(re.search(r"-s(\d+)", a).group(1))
    body = []
    for g in sigs:
        for i, f in enumerate(ANOMALY_FAMILIES):
            cells = []
            for lv in levels:
                c = S["families"][f][g][inj]["levels"][str(lv)]
                pairs = sorted(zip(map(run, c["arms"]), np.exp(c["ln_sigma_min"])))
                if [r for r, _ in pairs] != list(range(1, len(pairs) + 1)):
                    raise SystemExit(f"FATAL: {f}/{g}/{lv} runs are {[r for r, _ in pairs]}")
                cells.append(", ".join(fmt(v, 2) for _, v in pairs))
            body.append((SIGNAL_LABELS[g] if i == 0 else "") + f" & {FAMILY_LABELS[f]} & "
                        + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    head = ["& score & " + " & ".join(f"{lv} classes" for lv in levels) + " \\\\", "\\midrule"]
    caption = (f"$\\sigma_{{\\min}}$ of each pretraining run, runs one to "
               f"{words(anomaly_n_runs(S))} in order, for the "
               "signals and scores of Table~\\ref{tab:anomaly}: each value is that run's median "
               f"over resamplings at {fmt_int(inj)} injected signal jets.")
    return _table(head + body[:-1], caption, "tab:anomaly-per-run", "l l " + "r" * len(levels), [],
                  wide=True)


def table_random_control(C: dict, A: dict, probes=("linear",)) -> str:
    """The random-label control beside the four pretrained label sets, in 1-AUC.

    Linear probe only for now: the MLP rows are one more entry in `probes` once
    their rerun lands.
    """
    levels = A["levels_fine_to_coarse"]
    draws = sorted({r["draw"] for r in C["table"]})
    tasks = [t for t in TASK_LABELS if any(r["task"] == t for r in C["table"])]
    head = [f"& \\multicolumn{{{len(levels)}}}{{c}}{{pretraining label set}} & "
            f"\\multicolumn{{{len(draws) + 1}}}{{c}}{{random partitions into {levels[-1]} classes}} \\\\",
            "task & " + " & ".join(str(lv) for lv in levels) + " & "
            + " & ".join(f"{d}" for d in draws) + f" & mean {tex('±')} SD \\\\", "\\midrule"]
    body, n_seeds = [], set()
    for probe in probes:
        for t in tasks:
            cells = []
            for lv in levels:
                rows = seed_rows(A["table"], t, probe, lv)
                n_seeds.add(len(rows))
                x = [1e3 * (1 - r["auc"]) for r in rows]
                cells.append("---" if any(r["censored"] for r in rows)
                             else fmt_pm(np.mean(x), np.std(x, ddof=1)))
            ctl = sorted((r for r in C["table"] if (r["task"], r["probe"]) == (t, probe)),
                         key=lambda r: r["draw"])
            x = [1e3 * float(np.exp(r["control_log1m_auc"])) for r in ctl]
            cells += ["---" if r["censored"] else fmt_one(v) for r, v in zip(ctl, x)]
            cells.append("---" if any(r["censored"] for r in ctl)
                         else fmt_pm(np.mean(x), np.std(x, ddof=1)))
            body.append(TASK_LABELS[t] + ("" if probe == "linear" else ", MLP probe") + " & "
                        + " & ".join(cells) + " \\\\")
    which = " and ".join({"linear": "linear", "mlp": "nonlinear (MLP)"}[p] for p in probes)
    caption = (f"The random-label control of the first set of pretraining runs, frozen {which} probe: $1-$AUC in units of $10^{{-3}}$ "
               f"(lower is better) on the two tasks it was built for. The pretrained vocabularies "
               f"(number of classes) are the mean {tex('±')} standard deviation over the "
               f"{words(min(n_seeds))} pretraining runs; each random partition (numbered) is one "
               f"run, and the last column is the mean {tex('±')} standard deviation over the "
               f"{words(len(draws))} partitions. Each partition permutes the resonant classes "
               f"within two sets, the two-prong decays and the three- and four-prong decays, and "
               f"cuts them into groups that each hold the same fraction of the reweighted training "
               f"examples as one {levels[-1]}-class group; the QCD class is kept.")
    return _table(head + body, caption, "tab:random-control",
                  "l " + "r" * (len(levels) + len(draws) + 1), [], wide=True)


def table_recovery(R: dict, sizes: dict) -> str:
    """Balanced accuracy recovering each level of the tree from each model, at the largest
    training set of the learning curve."""
    acc = recovery_acc(R)
    levels = sorted({lv for _, lv in acc}, reverse=True)
    head = ["level recovered & " + " & ".join(f"{lv}-class model" for lv in levels) + " \\\\",
            "\\midrule"]
    body = []
    for rung in RUNGS:
        cells = []
        for lv in levels:
            a = list(acc[(rung, lv)].values())
            cells.append(fmt_pm(np.mean(a), np.std(a, ddof=1)))
        body.append(f"{sizes[rung]} classes & " + " & ".join(cells) + " \\\\")
    n_runs = len(acc[(RUNGS[0], levels[0])])
    caption = ("Linear decodability of each level of the label tree: balanced accuracy of a "
               "class-weighted logistic regression on the frozen features, trained to recover that "
               "level (rows, finest first) from each pretrained model (columns) on "
               f"{fmt_int(max(R['sizes']))} jets, mean $\\pm$ standard deviation over the "
               f"{words(n_runs)} pretraining runs.")
    return _table(head + body, caption, "tab:recovery", "l " + "r" * len(levels), [])


def table_mass(M: dict, root: pathlib.Path, cells=MASS_CELLS) -> str:
    """Jet-mass resolution from frozen features, both probes, over `cells`."""
    tgt = {round(r["target_sigma_eff"], 12) for r in M["table"] if r["probe"] == "ridge"}
    tgt = fmt(tgt.pop(), 4) if len(tgt) == 1 else "---"
    head = ["& " + " & ".join(c.replace("+mass", " + mass") for c in cells)
            + " & true-class mean \\\\", "\\midrule"]
    body = []
    for probe, name in (("mlp", "nonlinear (MLP) probe"), ("ridge", "linear (ridge) probe")):
        text = [fmt_pm(np.mean(v), np.std(v, ddof=1))
                for v in (mass_values(M, c, probe) for c in cells)]
        body.append(f"{name} & " + " & ".join(text) + f" & {tgt} \\\\")
    caption = ("Jet-mass regression from frozen features: $\\sigma_{\\mathrm{eff}}$ of the residual. "
               "The residual is $\\ln(m_{\\mathrm{pred}}/m_{\\mathrm{true}})$ after removing each "
               "native class's training-set mean; $\\sigma_{\\mathrm{eff}}$ is half the smallest "
               f"interval holding 68\\% of it. Mean {tex('±')} standard deviation over the "
               f"{words(len(mass_values(M, cells[0], 'ridge')))} pretraining runs, on "
               f"{fmt_int(mass_n_test(M, root)[0])} test jets. Columns are the number of classes "
               "in the pretraining vocabulary, with or without the added mass output; the last is "
               "an oracle that knows each jet's true native class and returns that class's mean, "
               "the same for both probes.")
    return _table(head + body, caption, "tab:mass", "l " + "r" * (len(cells) + 1), [],
                  wide=True)


def table_realdata(J: dict, root: pathlib.Path, fits: tuple | None = None) -> str:
    """Fitted top-quark yield in CMS open data at the working point the fits record (fit_v6)."""
    P = J["per_label_set"]
    _, res = fits or aoj_fits(root, J)
    head = ["pretraining vocabulary & yield & statistical error per fit & yield, each model's own peak shape "
            "\\\\", "\\midrule"]
    body = []
    for lv in AOJ_SETS:
        c = P["label_sets"][lv]
        y, own = c["signal_yield"], c["by_shape"]["own_shape"]
        body.append(f"{lv.replace('+mass', ' + mass')} & {fmt_pm(y['mean'], y['sd'])} & "
                    f"{fmt_int(c['median_stat_err'])} & {fmt_pm(own['mean'], own['sd'])} \\\\")
    body.append("\\addlinespace")
    pub = res["models"]["sophon-public"]["top"]
    body.append(f"published 188-class checkpoint (one model) & {fmt_int(pub['signal_yield'])} & "
                f"{fmt_int(pub['signal_yield_err'])} & {fmt_int(pub['own_shape_fit']['signal_yield'])} \\\\")
    ref = res["reference"]["top"]
    body.append(f"CMS ParticleNet top score (shipped) & {fmt_int(ref['signal_yield'])} & "
                f"{fmt_int(ref['signal_yield_err'])} & --- \\\\")
    m, w = P["shapes"]["pooled"]
    caption = ("Top quarks found in CMS open data with no fine-tuning: the top-quark yield fitted with "
               "the score cut set, in bins of $\\rho$ and $p_{\\mathrm T}$, to pass "
               f"{fmt_one(100 * float(res['eff']), 0)}\\% of the jets "
               "in the mass sidebands; one simultaneous pass/fail fit over all files per model, with one "
               f"Gaussian peak shape ({fmt(m, 1)}~GeV, width {fmt(w, 1)}~GeV) fitted jointly to the "
               f"{len(P['shapes']['pool'])} pretrained models, and with the tops that fail the cut taken from "
               f"the CMS score's fit. Mean {tex('±')} standard deviation over the "
               f"{words(aoj_n_runs(P))} pretraining runs; the "
               "second column is the median statistical error of a single fit; the last column refits every "
               "model with its own floated peak shape. The CMS score's peak shape is its own throughout.")
    return _table(head + body, caption, "tab:realdata", "l r r r", [])


# ------------------------------------------------------------------ the second grid (v2)
#
# The paper reports v2 (PRESPEC A7-A14: "Every v1 result stays in the record. The paper
# reports v2"). A section prints from its v2 directory (the layout under V2_PENDING) when
# that exists and V2_READ lists it, under the first grid's macro names, so the text needs
# no renaming; otherwise from the first grid's files. The reporting rules are A14's:
#   * every number under a first-grid name is the primary checkpoint's (best70, the first
#     maximum within epochs 70-79; self-supervised: the first minimum of the validation
#     loss), through the class token;
#   * robustness: beside each paired ratio, the result at the weight average and the
#     paired ln(weight average / primary) with its 95 % interval, labelled 'depends on the
#     checkpoint' (excludes 0), 'robust' (within +-ln 1.1), 'depends on the checkpoint,
#     under 10%' (both) or 'inconclusive' -- re-derived here from the interval and checked
#     against paired_errors.py's label -- and the dependent results counted against 5 %;
#   * sensitivity: the same at the global best epoch (bestval);
#   * the BatchNorm twin beside the primary, for the frozen readouts only;
#   * reference rows in the frozen tables: the pooled embedding of every model and the
#     untrained trunk;
#   * the freeze (A14 item 8): a section is read once its directory holds every run of its
#     arms; tiers 1-2 come first, and the tier-3 rows (64 and 30 classes) join the same
#     tables when their directory exists;
#   * I7: a comparison not paired by run index uses runs of one GPU product only.

def v2_dir(root: pathlib.Path, sub: str) -> pathlib.Path:
    return pathlib.Path(root).joinpath(*V2_DATA, sub)


def _v2_rel(root: pathlib.Path, p) -> str:
    """A path relative to the root, not resolved: a tag held as a link to another keeps
    its own path (seed_level.py records it so)."""
    p = pathlib.Path(p)
    p = p if p.is_absolute() else pathlib.Path(root) / p
    return os.path.relpath(os.path.abspath(p), os.path.abspath(root))


def v2_frozen_files(root: pathlib.Path, analysis: str) -> dict:
    """{(tag, readout): {file relative to the root}} of one frozen readout over every
    section present that holds it (V2_FROZEN; label_recovery for the curve)."""
    name = V2_ANALYSES[analysis][0]
    subs = ("label_recovery",) if analysis == "label_recovery_curve" else V2_FROZEN
    out = {}
    for sub in subs:
        if v2_present(root, sub):
            for p in sorted(v2_dir(root, sub).glob(f"{analysis}/*/*/*/{name}")):
                out.setdefault((p.parents[1].name, p.parent.name), set()).add(_v2_rel(root, p))
    return out


def v2_analysis_dir(root: pathlib.Path) -> pathlib.Path:
    """The one probe_ladder/analysis*/ (experiments/STATS/seed_level.py --v2's output) that
    read exactly the frozen readouts present now, every analysis, tag and readout of them
    and nothing else. One that read fewer would drop the rows added since (tier 3 follows
    the freeze); one that read others describes files no longer here. None or two is fatal."""
    want = {(a, key): files for a in V2_ANALYSES for key, files in v2_frozen_files(root, a).items()}
    hits = []
    for d in sorted(p for p in v2_dir(root, "probe_ladder").glob("analysis*") if p.is_dir()):
        got = {}
        for a, (_, out_name) in V2_ANALYSES.items():
            for p in d.glob(f"*/*/{out_name}"):
                doc = json.loads(p.read_text())
                got[(a, (p.parents[1].name, p.parent.name))] = {
                    _v2_rel(root, x["path"]) for x in doc["provenance"]["inputs"]}
        if got == want:
            hits.append(d)
    if len(hits) != 1:
        raise SystemExit(f"FATAL: {len(hits)} v2 analyses under "
                         f"{v2_dir(root, 'probe_ladder').relative_to(root)}/analysis*/ read exactly "
                         "the frozen readouts present; run experiments/STATS/seed_level.py --v2 over "
                         "the v2 sections into a new directory")
    return hits[0]


def v2_analysis(root: pathlib.Path, analysis: str, grid: pathlib.Path) -> dict:
    """{(tag, readout): (path, document)} of `analysis` from v2_analysis_dir, each built on
    this grid (another has other levels and run counts) and holding every run of the arms
    it holds (a section is copied whole), its inputs unchanged since."""
    out = {}
    d = v2_analysis_dir(root)
    for p in sorted(d.glob(f"*/*/{V2_ANALYSES[analysis][1]}")):
        doc = json.loads(p.read_text())
        if doc["provenance"]["grid"]["sha256"] != hashlib.sha256(grid.read_bytes()).hexdigest():
            raise SystemExit(f"FATAL: {p} was built on another {grid.name}; rerun seed_level.py --v2")
        if doc.get("missing_runs"):
            raise SystemExit(f"FATAL: {p}: runs missing {doc['missing_runs']}; a v2 section is "
                             "copied once every run of its arms has finished")
        check_analysis_is_current(doc, root)
        out[(p.parents[1].name, p.parent.name)] = (p, doc)
    return out


# A14's labels of a result between two checkpoints, written here again rather than read
# from src/stats/paired.py: the labels paired_errors.py stored are re-derived from the
# interval each result stores and must agree.
CKPT_DEPENDS = "depends on the checkpoint"
CKPT_DEPENDS_UNDER_10 = "depends on the checkpoint, under 10%"
CKPT_DEPENDENT = (CKPT_DEPENDS, CKPT_DEPENDS_UNDER_10)
CKPT_BAND = math.log(1.1)        # the smallest effect of interest (src/stats/mde.py MEI_LOG)


def checkpoint_class(lo: float, hi: float) -> str:
    """A14's label from the 95 % interval [lo, hi] of a ratio (the result at another
    checkpoint over the result at the primary): 'depends on the checkpoint' when it
    excludes 1, 'robust' when it lies within [1/1.1, 1.1] (+-ln 1.1), both rules at once
    'depends on the checkpoint, under 10%' (counted as dependent), else 'inconclusive'."""
    a, b = math.log(lo), math.log(hi)
    dep, small = a > 0 or b < 0, -CKPT_BAND < a and b < CKPT_BAND
    if dep:
        return CKPT_DEPENDS_UNDER_10 if small else CKPT_DEPENDS
    return "robust" if small else "inconclusive"


def checkpoint_tally(rows: list, tag: str) -> dict:
    """{results, models}: per A14, the results between `tag`'s checkpoints (the contrasts;
    'models' the checkpoint contrast, one model's own metric), each {dependent, n, robust,
    inconclusive, identical}, a result whose two checkpoints are one file (ratio exactly 1,
    no error) counted apart from n. The same rules as paired_errors.checkpoint_dependence,
    applied again to the labels re-derived by checkpoint_class."""
    out = {}
    for what, sel in (("results", lambda r: r["contrast"] != "checkpoint"),
                      ("models", lambda r: r["contrast"] == "checkpoint")):
        rs = [r for r in rows if r["checkpoint"] == tag and "checkpoint_label" in r and sel(r)]
        same = [r for r in rs if r.get("ln_ratio") == 0.0 and r.get("ln_combined_se") == 0.0]
        rs = [r for r in rs if not (r.get("ln_ratio") == 0.0 and r.get("ln_combined_se") == 0.0)]
        lab = [checkpoint_class(*r["ci95"]) for r in rs]
        out[what] = {"dependent": sum(x in CKPT_DEPENDENT for x in lab), "n": len(lab),
                     "robust": lab.count("robust"), "inconclusive": lab.count("inconclusive"),
                     "identical": len(same)}
    return out


def check_checkpoint_labels(R: dict, path: pathlib.Path) -> None:
    """Refuse a ratios file whose stored A14 labels or counts the intervals beside them
    do not give."""
    for i, r in enumerate(R["ratios"]):
        if "checkpoint_label" in r and checkpoint_class(*r["ci95"]) != r["checkpoint_label"]:
            raise SystemExit(f"FATAL: {path} ratios[{i}] is labelled {r['checkpoint_label']!r}; its "
                             f"interval {r['ci95']} gives {checkpoint_class(*r['ci95'])!r}")
    for tag, dep in R.get("checkpoint_dependence", {}).items():
        mine = checkpoint_tally(R["ratios"], tag)
        for what in ("results", "models"):
            got = (dep[what]["dependent"], dep[what]["n"], dep[what]["identical_checkpoint"])
            want = (mine[what]["dependent"], mine[what]["n"], mine[what]["identical"])
            if got != want:
                raise SystemExit(f"FATAL: {path} counts {got} dependent/n/identical {what} at {tag}; "
                                 f"its labelled rows give {want}")


def v2_products(specs: list) -> dict | None:
    """{run index: GPU product} of the second grid, from its job specs (gpu_by_run); None
    without specs. A run index on two products is fatal: no comparison could be read."""
    if not specs:
        return None
    by_run = gpu_by_run(specs)
    mixed = {r: sorted(p) for r, p in by_run.items() if len(p) > 1}
    if mixed:
        raise SystemExit(f"FATAL: the grid's job specs put run indices {mixed} on several GPU "
                         "products; no v2 comparison can keep to one (I7)")
    return {r: next(iter(p)) for r, p in by_run.items()}


def v2_run(model: str) -> int:
    """'l188-s3@best70' or 'l188-s3' -> 3: the run index of a v2 model or fine-tuning init."""
    return int(re.search(r"-s(\d+)(?:@|$)", model).group(1))


def v2_one_product(products: dict):
    """I7 for a comparison that is not paired by run index: (a, b) -> the initialisations of
    two rows it may use. Two rows with the same run indices hold the same products in the
    same proportion and keep every run; otherwise both keep only the runs on the product
    of run index 1 (RTX 3090 in the grid), and a row left with none gives None."""
    first = products[min(products)]

    def restrict(a, b):
        if sorted(map(v2_run, a)) == sorted(map(v2_run, b)):
            return a, b
        a, b = ([i for i in x if products[v2_run(i)] == first] for x in (a, b))
        return (a, b) if a and b else None
    return restrict


def check_one_product(rows: list, products: dict, path: pathlib.Path) -> None:
    """I7: every contrast paired_errors.py formed unpaired (stream_pairing 'exempt': A13's
    unseen against seen, the self-supervised model against the vocabularies) reads runs
    of one GPU product."""
    for i, r in enumerate(rows):
        if not str(r.get("stream_pairing", "")).startswith("exempt"):
            continue
        runs = {v2_run(m.split("/")[0]) for m in r.get("fine_models", []) + r.get("coarse_models", [])}
        prods = {products[k] for k in runs}
        if len(prods) > 1:
            raise SystemExit(f"FATAL: {path} ratios[{i}] ({r['contrast']}) is unpaired over runs "
                             f"{sorted(runs)} on {sorted(prods)}; I7 keeps such a contrast to one "
                             "GPU product (contrasts.v2.json `runs`)")


def _v2_rows(A: dict, task: str, probe: str, arm: str) -> list:
    return sorted((r for r in A["table"] if (r["task"], r["probe"], r["arm"]) == (task, probe, arm)),
                  key=lambda r: r["seed"])


def emit_v2_probes(em: Emitter, V: dict, ssl: bool = False) -> dict:
    """The frozen probes of the second grid beyond emit_design and emit_levels (which build
    calls on the primary checkpoint's class token): the background test jets per task, and
    beside each 1-AUC its BatchNorm twin (...Twin), its pooled-embedding reference
    (...Pooled), and the untrained trunk's (ProbeOma<task><probe>Untrained,
    ...UntrainedPooled), and with `ssl` (the self-supervised model past PRESPEC 4's bar) its
    pooled embedding (...SelfSupervised). Returns {task: background test jets}."""
    src, A = V[(V2_PRIMARY, V2_CLASS_TOKEN)]
    nbkg = {}
    for task in sorted(A["tasks"]):
        n = A["tasks"][task]["n_background_test"]
        if n is not None:
            nbkg[task] = n
            em.macro("ProbeNBkgTest" + texname(task), fmt_int(n), src,
                     f"tasks.{task}.n_background_test", "test-sample background jets")

    def oma(rows):
        if not rows or any(r["censored"] for r in rows):
            return None
        x = [1.0 - r["auc"] for r in rows]
        return fmt_pm_sci(np.mean(x), np.std(x, ddof=1)) if len(x) > 1 else fmt_one_sci(x[0])
    for (tag, ro), suffix in (((V2_TWIN, V2_CLASS_TOKEN), "Twin"), ((V2_PRIMARY, V2_POOLED), "Pooled")):
        s2, B = V[(tag, ro)]
        for task in ordered_tasks(B["levels"]):
            for probe in sorted(B["levels"][task]):
                for lv in B["levels_fine_to_coarse"]:
                    text = oma(seed_rows(B["table"], task, probe, lv))
                    if text is not None:
                        em.macro("ProbeOma" + texname(task, probe, lv) + suffix, text, s2,
                                 f"table[{task},{probe},{lv}].auc",
                                 f"1-AUC at {tag}, {ro}; mean +- SD over runs")
    for ro, suffix in ((V2_CLASS_TOKEN, ""), (V2_POOLED, "Pooled")):
        s2, I = V[(V2_INIT, ro)]
        for task in sorted({r["task"] for r in I["table"]}):
            for probe in ("linear", "mlp"):
                text = oma(_v2_rows(I, task, probe, "INIT"))
                if text is not None:
                    em.macro("ProbeOma" + texname(task, probe) + "Untrained" + suffix, text, s2,
                             f"table[{task},{probe},arm=INIT].auc",
                             f"1-AUC of the untrained trunk, {ro}; mean +- SD over its initialisations")
    if ssl:
        s2, P = V[(V2_PRIMARY, V2_POOLED)]
        for task in sorted({r["task"] for r in P["table"] if r["arm"] == "MPM"}):
            for probe in ("linear", "mlp"):
                text = oma(_v2_rows(P, task, probe, "MPM"))
                if text is not None:
                    em.macro("ProbeOma" + texname(task, probe) + "SelfSupervised", text, s2,
                             f"table[{task},{probe},arm=MPM].auc",
                             "1-AUC of the self-supervised model, pooled embedding; mean +- SD over runs")
    return nbkg


def table_probe_ladder_v2(V: dict, probe: str, nbkg: dict, ssl: bool = False) -> str:
    """The frozen-probe table of the second grid: per vocabulary the primary checkpoint's
    AUC and rejection and its BatchNorm twin's AUC beside them; then the reference rows,
    each vocabulary's pooled embedding and the untrained trunk through both readouts."""
    A, T, P = (V[k][1] for k in ((V2_PRIMARY, V2_CLASS_TOKEN), (V2_TWIN, V2_CLASS_TOKEN),
                                 (V2_PRIMARY, V2_POOLED)))
    # the tasks measured at the headline working point; one that pins its own (the |V_cb|
    # window probe, 60 and 40 %) is given in the text (emit_vcb_v2)
    every = ordered_tasks(A["levels"])
    tasks = [t for t in every if headline_rejection(A["levels"][t][probe][0])[0] is not None]
    own = [t for t in every if t not in tasks]
    levels = A["levels_fine_to_coarse"]
    key, eps = headline_rejection(A["levels"][tasks[0]][probe][0])
    head = ["classes & quantity & " + " & ".join(TASK_LABELS.get(t, tex(t)) for t in tasks)
            + " \\\\", "\\midrule"]
    body, cells, n_runs = [], [], set()

    def auc(rows):
        n_runs.add(len(rows))
        return "---" if not rows else fmt_auc_pm([r["auc"] for r in rows], [r["censored"] for r in rows])
    for i, lv in enumerate(levels):
        a = [auc(seed_rows(A["table"], t, probe, lv)) for t in tasks]
        rj = [fmt_rejection(*seed_rejections(A["table"], t, probe, lv, key)) for t in tasks]
        tw = [auc(seed_rows(T["table"], t, probe, lv)) for t in tasks]
        cells += a + rj + tw
        body += [f"{lv} & AUC & " + " & ".join(a) + " \\\\",
                 f"     & $1/\\epsilon_B$ at {eps * 100:.0f}\\% & " + " & ".join(rj) + " \\\\",
                 "     & AUC, BatchNorm recomputed & " + " & ".join(tw) + " \\\\"]
        if i < len(levels) - 1:
            body.append("\\addlinespace")
    body += ["\\midrule", f"\\multicolumn{{{2 + len(tasks)}}}{{@{{}}l}}{{\\itshape reference rows}} \\\\"]
    for lv in levels:
        p = [auc(seed_rows(P["table"], t, probe, lv)) for t in tasks]
        cells += p
        body.append(f"{lv} & AUC, pooled embedding & " + " & ".join(p) + " \\\\")
    if ssl:
        x = [fmt_auc_pm([r["auc"] for r in rs], [r["censored"] for r in rs]) if rs else "---"
             for rs in (_v2_rows(P, t, probe, "MPM") for t in tasks)]
        cells += x
        body.append("self-supervised & AUC, pooled embedding & " + " & ".join(x) + " \\\\")
    n_init = set()
    for ro, what in ((V2_CLASS_TOKEN, "AUC"), (V2_POOLED, "AUC, pooled embedding")):
        I = V[(V2_INIT, ro)][1]
        u = []
        for t in tasks:
            rows = _v2_rows(I, t, probe, "INIT")
            n_init.add(len(rows))
            u.append("---" if not rows else fmt_auc_pm([r["auc"] for r in rows], [r["censored"] for r in rows]))
        cells += u
        body.append(f"untrained & {what} & " + " & ".join(u) + " \\\\")
    n_runs.discard(0)
    caption = (f"Frozen {'linear' if probe == 'linear' else 'nonlinear (MLP)'} probes on the "
               f"{words(len(levels))} pretraining vocabularies of the second set of runs, at each "
               "run's primary checkpoint (the first maximum of the validation accuracy within epochs "
               f"70--79): AUC and the background rejection $1/\\epsilon_B$ at {eps * 100:.0f}\\% signal "
               f"efficiency, mean {tex('±')} standard deviation over the {words(min(n_runs))} "
               "pretraining runs, and beside them the AUC of the same checkpoint with its BatchNorm "
               "statistics recomputed on training jets. Reference rows: the pooled embedding of each "
               "model in place of the class token, and the untrained network the runs started from "
               f"({words(max(n_init))} initialisations). Rows are the number of classes in the "
               "pretraining vocabulary, finest first and coarsest last.")
    notes = _probe_notes(" ".join(cells), tasks, nbkg)
    if own:
        notes.append("Not shown, having working points of their own: "
                     + ", ".join(TASK_LABELS.get(t, tex(t)) for t in own) + ".")
    return _table(head + body, caption, f"tab:probes-{probe}", "r l " + "r" * len(tasks),
                  notes, wide=True)


def emit_mass_output_v2(em: Emitter, A: dict, src: pathlib.Path, restrict=None) -> None:
    """MassOma* of the second grid (emit_mass_output's names): 1-AUC on b vs c two-prong
    with and without the mass output at 162 and 17 classes, and with the matched weight
    (A11, ...MassMatched) once its runs are in, at the primary checkpoint."""
    task = "bvc_resonant"
    have = A["runs_by_arm"]
    if "L162_MASS" not in have and "R16_Q1_MASS" not in have:
        return
    em.macro("MassNRunsWord", words(len(have.get("L162_MASS") or have["R16_Q1_MASS"])), src,
             "runs_by_arm.L162_MASS (count)", "pretraining runs of each mass-output configuration")
    for probe in ("linear", "mlp"):
        k = texname(probe)
        done = set()
        for plain, mass, suffix in (("L162", "L162_MASS", "Mass"), ("R16_Q1", "R16_Q1_MASS", "Mass"),
                                    ("R16_Q1", "R16_Q1_MASS_LM", "MassMatched")):
            if mass not in have or plain not in have:
                continue
            lv = next(int(x) for x, a in A["level_arms"].items() if a == plain)
            p_rows, m_rows = _v2_rows(A, task, probe, plain), _v2_rows(A, task, probe, mass)
            if any(r["censored"] for r in p_rows + m_rows):
                raise SystemExit(f"FATAL: a {task} cell reached AUC=1; its 1-AUC is only a bound")
            x0, x1 = [1 - r["auc"] for r in p_rows], [1 - r["auc"] for r in m_rows]
            if plain not in done:
                em.macro("MassOma" + k + texname(lv), fmt_pm_sci(np.mean(x0), np.std(x0, ddof=1)), src,
                         f"table[{task},{probe},arm={plain}].auc", f"1-AUC, mean +- SD over {len(x0)} runs")
                done.add(plain)
            em.macro("MassOma" + k + texname(lv) + suffix, fmt_pm_sci(np.mean(x1), np.std(x1, ddof=1)),
                     src, f"table[{task},{probe},arm={mass}].auc", f"1-AUC, mean +- SD over {len(x1)} runs")
            if restrict is not None:
                pair = restrict([r["model"] for r in m_rows], [r["model"] for r in p_rows])
                if pair is None:
                    continue
                x1 = [1 - r["auc"] for r in m_rows if r["model"] in pair[0]]
                x0 = [1 - r["auc"] for r in p_rows if r["model"] in pair[1]]
            em.macro("MassOmaRatio" + k + texname(lv) + suffix.removeprefix("Mass"),
                     fmt_ratio(np.mean(x1) / np.mean(x0)), src,
                     f"table[{task},{probe},arm={mass}/{plain}].auc",
                     "run mean of 1-AUC with the mass output over without")


def emit_vcb_v2(em: Emitter, A: dict, src: pathlib.Path) -> None:
    """The |V_cb| window probe (bc_vs_rest) of the second grid under emit_vcb's names: the
    162- and 17-class models at the primary checkpoint, from seed_level.py --v2's table."""
    task, eps = "bc_vs_rest", "0.60"
    if task not in A["tasks"]:
        return
    T = A["tasks"][task]
    em.macro("VcbNSignal", fmt_int(T["n_signal_test"]), src, f"tasks.{task}.n_signal_test",
             "X->bc test jets in the window")
    em.macro("VcbNBackground", fmt_int(T["n_background_test"]), src, f"tasks.{task}.n_background_test",
             "bq, cs, bqq and QCD test jets in the window")
    em.macro("VcbEpsSixty", f"{100 * float(eps):.0f}", src, f"tasks.{task}.eps_s", "signal efficiency, percent")
    arm_of = {int(lv): a for lv, a in A["level_arms"].items()}
    hi, lo = 162, 17
    for probe in ("linear", "mlp"):
        k, oma, logs = texname(probe), {}, {}
        for lv in (hi, lo):
            rows = _v2_rows(A, task, probe, arm_of[lv])
            jp = f"table[{task},{probe},{lv}]"
            if any(r["censored"] for r in rows):
                raise SystemExit(f"FATAL: a {task} cell reached AUC=1; its 1-AUC is only a bound")
            logs[lv] = [r["log1m_auc"] for r in rows]
            x = np.exp(logs[lv])
            oma[lv] = float(np.mean(x))
            em.macro("VcbOma" + k + texname(lv), fmt_pm_sci(np.mean(x), np.std(x, ddof=1)), src,
                     jp + ".log1m_auc", f"1-AUC = exp of it, mean +- SD over {len(x)} runs")
            em.macro("VcbLogOma" + k + texname(lv), fmt(np.mean(logs[lv]), 3, sign=True), src,
                     jp + ".log1m_auc", "mean over runs")
            pts = [r["rejection_points"][eps] for r in rows]
            em.macro("VcbRej" + k + "Sixty" + texname(lv),
                     fmt_rejection([p["rejection"] for p in pts], [p["is_bound"] for p in pts],
                                   [p["n_bkg_pass"] for p in pts]),
                     src, jp + f".rejection_points['{eps}'].rejection",
                     "background rejection at this signal efficiency, mean +- SD over runs")
            em.macro("VcbBkgLeft" + k + "Sixty" + texname(lv), fmt(np.mean([p["n_bkg_pass"] for p in pts]), 1),
                     src, jp + f".rejection_points['{eps}'].n_bkg_pass",
                     "background jets passing the cut, mean over runs")
        em.macro("VcbOmaRatio" + k, fmt_ratio(oma[lo] / oma[hi]), src, f"table[{task},{probe},*].log1m_auc",
                 f"run mean of 1-AUC, {lo} classes over {hi}")
        em.macro("VcbFactor" + k, fmt(np.exp(np.mean(logs[lo]) - np.mean(logs[hi])), 2), src,
                 f"table[{task},{probe},*].log1m_auc",
                 f"exp of the difference of the run means of log(1-AUC), {lo} classes over {hi}")


# The results paired_errors.py stores beside each primary ratio and what each is (A8, A14):
# (macro suffix, checkpoint tag, the families it exists for, labelled by A14's rule).
V2_BESIDE = (("Wavg", "wavg", None, False), ("WavgShift", "wavg/best70", None, True),
             ("Bestval", "bestval", ("probe", "mass", "anomaly"), False),
             ("BestvalShift", "bestval/best70", ("probe", "mass", "anomaly"), True),
             ("Twin", "best70_bn", ("probe", "mass", "anomaly"), False),
             ("Pooled", "best70:pooled", ("probe", "mass"), False))
# the metric each v2 family's ratios are read in: the first grid's, and for the output
# ratio of the anomaly study sigma_min at the primary injection (anomaly_summary.PRIMARY)
V2_PAIRED_METRIC = {**PAIRED_METRIC, "anomaly": "sigma_min@2000"}
CKPT_LABEL_TEX = {"robust": "robust", "inconclusive": "inconclusive", CKPT_DEPENDS: CKPT_DEPENDS,
                  CKPT_DEPENDS_UNDER_10: "depends on the checkpoint, under 10\\%"}


def emit_paired_v2(em: Emitter, files: list, ft: dict | None, products: dict | None,
                   skip_arms: set = frozenset()) -> dict:
    """The second grid's paired ratios under emit_paired's names, at the primary
    checkpoint through the class token: every ratio of two arms (pairs, all_pairs and
    unpaired contrasts; P1, P2, A11 and the joint fit are other sections'), and beside each
    the results V2_BESIDE lists -- at the weight average (...Wavg) and the paired shift to
    it (...WavgShift, labelled ...WavgLabel), at the global best (...Bestval, ...BestvalShift,
    ...BestvalLabel), at the BatchNorm twin (...Twin) and through the pooled embedding
    (...Pooled). Each file's labels and counts are re-derived first (check_checkpoint_labels)
    and its unpaired contrasts checked against I7. A ratio with an arm of `skip_arms` is
    not printed (the self-supervised model before its validity bar, A14). Returns
    {family file: its document}."""
    for leg, ds in PAIRED_FT_DATASET.items():
        if ft and ds in ft and leg not in ft[ds]["path"].name:
            raise SystemExit(f"FATAL: the paired file's {leg} is not {ft[ds]['path'].name}")
    name = lambda x: texname(str(x).replace("+mass", " mass"))
    out = {}
    for path in files:
        R = json.loads(path.read_text())
        rows = R["ratios"]
        check_checkpoint_labels(R, path)
        if products is not None:
            check_one_product(rows, products, path)
        elif any(str(r.get("stream_pairing", "")).startswith("exempt") for r in rows):
            raise SystemExit(f"FATAL: {path} holds unpaired contrasts and the grid's job specs, "
                             "which give each run's GPU product (I7), are missing")
        at = {}
        for i, r in enumerate(rows):
            if "fine_arm" in r and "coarse_arm" in r and r["fine_arm"] != r["coarse_arm"]:
                at[(r["contrast"], r["family"], r["task"], r["kind"], r["metric"], r["fine"], r["coarse"],
                    r["checkpoint"])] = i
        for key, i in at.items():
            r = rows[i]
            if (key[-1] != V2_PRIMARY or r["metric"] != V2_PAIRED_METRIC.get(r["family"])
                    or r.get("censored_models") or r.get("ratio") is None or r.get("ci95") is None
                    or {r["fine_arm"], r["coarse_arm"]} & set(skip_arms)):
                continue
            fam = r["family"]
            head = {"probe": lambda: "PairedProbe" + texname(r["task"], r["kind"]),
                    "ft": lambda: "PairedFt" + PAIRED_FT_DATASET[r["task"]] + n_tag(r["kind"]),
                    "mass": lambda: "PairedMassRes" + texname(r["kind"]),
                    "anomaly": lambda: "PairedAnomaly" + texname(r["kind"]) + anomaly_signal_key(r["task"])}[fam]()
            base = head + name(r["coarse"]) + "Over" + name(r["fine"])
            what = f"{r['coarse']} over {r['fine']}, {r['task']} {r['kind']} {r['metric']}"
            em.macro(base, fmt_paired(r["ratio"], *r["ci95"]), path, f"ratios[{i}].ratio, ratios[{i}].ci95",
                     f"{what}, primary checkpoint: paired geometric mean [95% interval]")
            for suffix, tag, fams, labelled in V2_BESIDE:
                j = at.get(key[:-1] + (tag,))
                s = None if j is None else rows[j]
                if (fams and fam not in fams) or s is None or s.get("censored_models") \
                        or s.get("ratio") is None or s.get("ci95") is None:
                    continue
                em.macro(base + suffix, fmt_paired(s["ratio"], *s["ci95"]), path,
                         f"ratios[{j}].ratio, ratios[{j}].ci95", f"{what}, at {tag} [95% interval]")
                if labelled:
                    em.macro(base + suffix.removesuffix("Shift") + "Label", CKPT_LABEL_TEX[s["checkpoint_label"]],
                             path, f"ratios[{j}].checkpoint_label",
                             "A14's label of the shift, re-derived from its 95% interval")
        out[path] = R
    return out


def emit_checkpoint_counts(em: Emitter, docs: dict, extra: dict | None = None) -> str:
    """A14: the number of results that depend on the checkpoint against its 5 % null
    expectation, at the weight average (Wavg) and the global best (Bestval), through the
    class token: per family file and summed (CkptWavgDependent, ...N, ...Expected, ...Robust,
    ...Inconclusive; ...Models... for each model's own metric), and the table of them.
    The results share models and test jets, so the count is a reference, not a test.
    `extra` ({(section, word): [(label, source)]}, CheckpointLabels.rows) adds the per-model
    labels this script derives (Ckpt<word>Models<section>...), and every per-model label of
    both into Ckpt<word>ModelsAll...; the class token only (pooled rows are references)."""
    lines, total = [], {}
    for path, R in docs.items():
        fam = path.parent.name
        for tag, word in (("wavg/best70", "Wavg"), ("bestval/best70", "Bestval")):
            if tag not in R.get("checkpoint_dependence", {}):
                continue
            t = checkpoint_tally(R["ratios"], tag)
            for what, mw in (("results", ""), ("models", "Models")):
                c = t[what]
                for k in ("dependent", "n", "robust", "inconclusive"):
                    total.setdefault((word, mw, k), [0, []])[0] += c[k]
                    total[(word, mw, k)][1].append(path)
                key = "Ckpt" + word + mw + texname(fam)
                em.macro(key + "Dependent", str(c["dependent"]), path, f"checkpoint_dependence['{tag}'].{what}.dependent",
                         "results whose 95% interval excludes 0 (re-derived from the labelled rows)")
                em.macro(key + "N", str(c["n"]), path, f"checkpoint_dependence['{tag}'].{what}.n",
                         "results counted (two checkpoints that are one file left out)")
                em.macro(key + "Expected", fmt(0.05 * c["n"], 1), path, f"0.05 x checkpoint_dependence['{tag}'].{what}.n",
                         "the dependent count expected under the null")
                lines.append((fam, "each model" if mw else "every result", word, c))
    for (word, mw, k), (v, srcs) in total.items():
        em.macro("Ckpt" + word + mw + texname(k), str(v), srcs[0],
                 f"sum over the v2 paired_errors families of checkpoint_dependence.{k}",
                 f"summed over {len(srcs)} family files")
        if k == "n":
            em.macro("Ckpt" + word + mw + "Expected", fmt(0.05 * v, 1), srcs[0],
                     "0.05 x the summed n", "the dependent count expected under the null")
    every = {}
    for (word, mw, k), (v, srcs) in total.items():
        if mw:
            every[(word, k)] = [v, srcs[0]]
    for (section, word), labs in sorted((extra or {}).items()):
        c = {"dependent": sum(x in CKPT_DEPENDENT for x, _ in labs), "n": len(labs),
             "robust": sum(x == "robust" for x, _ in labs), "inconclusive": sum(x == "inconclusive" for x, _ in labs)}
        key = "Ckpt" + word + "Models" + texname(section)
        src = labs[0][1]
        for k in ("dependent", "n"):
            em.macro(key + {"dependent": "Dependent", "n": "N"}[k], str(c[k]), src,
                     f"A14 labels of the {section} results this script derives", "per model, paired over runs")
        em.macro(key + "Expected", fmt(0.05 * c["n"], 1), src, "0.05 x n", "expected under the null")
        for k, v in c.items():
            every.setdefault((word, k), [0, src])[0] += v
        lines.append((section, "each model", word, c))
    if extra:
        for (word, k), (v, src) in every.items():
            em.macro("Ckpt" + word + "ModelsAll" + texname(k), str(v), src,
                     "the per-model labels of paired_errors.py and of this script", "summed")
            if k == "n":
                em.macro("Ckpt" + word + "ModelsAllExpected", fmt(0.05 * v, 1), src, "0.05 x the summed n",
                         "expected under the null")
    if not lines:
        return ""
    body = [f"{tex(f)} & {who} & {'weight average' if w == 'Wavg' else 'global best'} & "
            f"{c['dependent']} & {fmt(0.05 * c['n'], 1)} & {c['n']} & {c['robust']} & {c['inconclusive']} \\\\"
            for f, who, w, c in lines]
    head = ["family & compared & checkpoint & dependent & expected (5\\%) & results & robust & inconclusive \\\\",
            "\\midrule"]
    caption = ("Robustness of the second set of runs to the checkpoint (A14): every paired result at "
               "another checkpoint over the same result at the primary checkpoint, run by run on the "
               "same test resamplings. A result depends on the checkpoint when its 95\\% interval "
               "excludes 1 (the count includes those within a factor 1.1 as well), is robust when "
               "the interval lies within a factor 1.1, and is inconclusive otherwise. The expected "
               "count is 5\\% of the results; the results share models and test jets, so it is a "
               "reference rather than a test. Two checkpoints that are one file are not counted.")
    return _table(head + body, caption, "tab:checkpoint-robustness", "l l l r r r r r", [])


def v2_ft_files(root: pathlib.Path, rule: str = V2_FT_RULE) -> list:
    """The fine-tuning read-outs of `rule`: every tier's (<rule>_<leg>_metrics.json) once it
    exists, else the freeze's (<rule>_t12_<leg>_metrics.json). Both present: every cell of
    the freeze's must be the full one's, value for value (the same runs read out twice)."""
    d, out = v2_dir(root, "finetune"), []
    for leg in ("leg1", "leg2"):
        full, t12 = d / f"{rule}_{leg}_metrics.json", d / f"{rule}_t12_{leg}_metrics.json"
        if full.exists() and t12.exists():
            F, T = (json.loads(p.read_text())["cells"] for p in (full, t12))
            bad = sorted(i for i in T if T[i] != F.get(i))
            if bad:
                raise SystemExit(f"FATAL: {t12.name} and {full.name} disagree on {bad[:3]}")
        p = full if full.exists() else t12 if t12.exists() else None
        if p is None:
            raise SystemExit(f"FATAL: {d.relative_to(root)} holds no {rule} read-out of {leg}")
        out.append(p)
    return out


# The order of the second grid's fine-tuning rows after the vocabularies, and each
# row's key; the vocabularies are keyed by their class count, as the first grid's.
V2_FT_EXTRA = {"L162_MASS": "Mass", "R16_Q1_MASS": "Mass", "R16_Q1_MASS_LM": "MassMatched"}


def ft_rows_v2(grid: list, labels: dict, cells: dict) -> list:
    """(macro key, label, initialisations) of the second grid's fine-tuning rows: the
    vocabularies on the tree, fine to coarse, the mass-output models, the flavour pair and
    the random partitions, each with every run the grid gives it. The models that leave a
    family out (Lofo) and the self-supervised model (FtRefs) are their sections' rows. An arm
    with some runs in `cells` and not all is fatal; one with none is not yet read out."""
    arms = {a["name"]: a for a in grid}
    tree = sorted((a for a in grid if a["name"] in RUNGS), key=lambda a: -a["num_classes"])
    rest = ([arms[n] for n in V2_FT_EXTRA if n in arms]
            + sorted((a for a in grid if a["name"].startswith("FLAV_")), key=lambda a: a["name"])
            + sorted((a for a in grid if a["name"].startswith("RAND2_")), key=lambda a: a["name"]))
    rows = []
    for a in tree + rest:
        slug = a["name"].lower().replace("_", "")
        inits = [f"{slug}-s{k}" for k in range(1, a["runs"] + 1)]
        have = [i for i in inits if i in cells]
        if not have:
            continue
        if have != inits:
            raise SystemExit(f"FATAL: the fine-tuning read-out holds {have} of {a['name']}'s {inits}")
        n = a["num_classes"]
        if a["name"] in RUNGS:
            key, label = texname(n), f"{n} classes"
        elif a["name"] in V2_FT_EXTRA:
            base = arms[a["name"].removesuffix("_LM").removesuffix("_MASS")]["num_classes"]
            key = texname(base) + V2_FT_EXTRA[a["name"]]
            label = f"{base} classes + mass output" + (", matched weight" if a["name"].endswith("_LM") else "")
        else:
            key, label = texname(labels.get(a["name"], a["name"])), tex(labels.get(a["name"], a["name"]))
        rows.append((key, label, inits))
    return rows


def emit_v2_pretraining(em: Emitter, root: pathlib.Path, grid: list, labels: dict) -> str:
    """A14: each run's selected epochs, by vocabulary -- the primary (first maximum within
    70-79) and the global best -- from the copies of best_window_epoch.json and
    best_epoch.json, per run (PretrainEpoch<arm>Run<k>, PretrainBestvalEpoch...) and per arm
    the runs whose global best is another epoch (PretrainBestvalElsewhere<arm>); and the
    table of them. An arm with some runs and not all is fatal."""
    d = v2_dir(root, "pretraining")
    body = []
    for a in grid:
        slug = a["name"].lower().replace("_", "")
        runs = [(k, d / f"mtx-{slug}-s{k}") for k in range(1, a["runs"] + 1)]
        have = [k for k, r in runs if (r / "best_window_epoch.json").exists() and (r / "best_epoch.json").exists()]
        if not have:
            continue
        if len(have) != a["runs"]:
            raise SystemExit(f"FATAL: {d.relative_to(root)} holds runs {have} of {a['name']}'s {a['runs']}")
        key = texname(labels.get(a["name"], a["name"]).replace("+mass", " mass"))
        e70, eg = [], []
        for k, r in runs:
            for name, lst, macro in (("best_window_epoch.json", e70, "PretrainEpoch"),
                                     ("best_epoch.json", eg, "PretrainBestvalEpoch")):
                e = int(json.loads((r / name).read_text())["epoch"])
                lst.append(e)
                em.macro(macro + key + "Run" + texname(k), str(e), r / name, "epoch",
                         "epochs count from 0")
        moved = sum(x != y for x, y in zip(e70, eg))
        em.macro("PretrainBestvalElsewhere" + key, of(moved, len(e70)), runs[-1][1] / "best_epoch.json",
                 "best_epoch.json != best_window_epoch.json, over every run's own pair",
                 "runs whose global best is not the primary epoch")
        body.append(f"{tex(labels.get(a['name'], a['name']))} & {', '.join(map(str, e70))} & "
                    f"{', '.join(map(str, eg))} \\\\")
    if not body:
        return ""
    head = ["pretraining model & primary epoch per run & global best per run \\\\", "\\midrule"]
    caption = ("The checkpoint of each pretraining run of the second set (runs in order, epochs from 0): "
               "the primary, the first maximum of the reweighted validation accuracy within epochs "
               "70--79, and the global best, reported as a sensitivity check.")
    return _table(head + body, caption, "tab:selected-epochs", "l l l", [])

# ------------------------------------------------------------------ v2: the readouts of A10-A14
#
# Each emitter below prints one section's v2 numbers under new names in the first grid's
# style; every label it prints is re-derived here from the interval stored beside it and
# must agree with the label paired_errors.py stored. Units: a paired ratio as fmt_paired
# prints it (coarse, merged, unseen or control over the other; above 1 = that side worse),
# a difference of ln(1 - AUC) as fmt_diff prints it.

def _ln_bounds(est: float, se: float, dof: float, level: float = 0.95) -> tuple[float, float]:
    """The central Student-t interval est -/+ t se (normal at infinite dof), in logs."""
    from scipy import stats
    q = 0.5 + level / 2
    k = float(stats.t.ppf(q, dof)) if math.isfinite(dof) else float(stats.norm.ppf(q))
    return est - k * se, est + k * se


P1_TEX = {"merging costs": "merging costs", "merging costs nothing": "merging costs nothing",
          "merging costs, under 10%": "merging costs, under 10\\%", "inconclusive": "inconclusive"}


def p1_class(lo: float, hi: float) -> str:
    """A14 P1 from the 95 % interval [lo, hi] of merged over split: 'merging costs' when it
    lies above 1, 'merging costs nothing' when its upper end is below 1.1, both at once
    'merging costs, under 10%', else 'inconclusive'."""
    a, b = math.log(lo), math.log(hi)
    if a > 0:
        return "merging costs, under 10%" if b < CKPT_BAND else "merging costs"
    return "merging costs nothing" if b < CKPT_BAND else "inconclusive"


def threshold_class(lo: float, hi: float, threshold: float = 0.0) -> str:
    """A14 P2: 'holds' above the threshold, 'fails' below it, 'inconclusive' across."""
    return "holds" if lo > threshold else "fails" if hi < threshold else "inconclusive"


def equivalence_class(lo: float, hi: float, margin: float) -> str:
    """A14 P2 (c): 'holds' within +-margin, 'fails' wholly outside, else 'inconclusive'."""
    if not margin > 0:
        return "not evaluable"
    if -margin < lo and hi < margin:
        return "holds"
    return "fails" if lo > margin or hi < -margin else "inconclusive"


def _partition_no(arm: str) -> int:
    return int(re.fullmatch(r"RAND2_p(\d+)", arm).group(1))


def emit_v2_random(em: Emitter, R: dict, path: pathlib.Path, A: dict, a_src: pathlib.Path) -> list:
    """A10 and A14, at the primary checkpoint through the class token, every probe kind:
      P1 per probe pair: RandPone<pair><probe> merging over splitting partitions (Welch, the
        run variance from the runs replicating a partition) [95%], ...Label; RandPone<pair>
        Merging/Splitting, the partitions; RandDose<pair>P<k>, each partition's realised dose
        of the pair's axis, the pair's own classes left out (the reading beside P1);
      the joint fit, secondary: RandJoint<pair><probe>, RandJointDose<pair><probe> and
        RandJointDoseAxis<axis><probe> (exp of the per-axis term: axis merged over kept);
      random against semantic, runs 1-2 paired: RandVsSem<ref><pair><probe><group> =
        reference over the random side (above 1 = the random side beats it), <group> each
        partition, Merging, Splitting, or Cells (A14's axis-account cells), the last with
        ...AxisAccount and ...PairAccount;
      P2 restated: RandPtwoGap<probe> (17 minus 43 classes on two-prong b vs c), RandPtwo
        <A..D><probe> with ...Label, RandPtwoMargin<probe>, RandPtwoWithdrawal<probe>,
        RandPtwoRuns<probe>, every difference in ln(1 - AUC).
    Returns the tables (descriptive: each partition's and each flavour model's 1-AUC)."""
    rows = R["ratios"]
    prim = lambda r: r["checkpoint"] == V2_PRIMARY and r["family"] == "probe" and r["metric"] == "1-auc"
    once = set()
    p1 = {}
    for i, r in enumerate(rows):
        if r["contrast"] != "partition_split_vs_merged" or not prim(r):
            continue
        pk, kind = texname(*r["probe_pairs"]), texname(r["kind"])
        if pk not in once:
            once.add(pk)
            for side, key in (("Merging", "merged_partitions"), ("Splitting", "split_partitions")):
                em.macro("RandPone" + pk + side, word_list(sorted(map(_partition_no, r[key]))), path,
                         f"ratios[{i}].{key}", f"the random partitions listed as {key.split('_')[0]}")
            for pair, d in (r.get("axis_doses") or {}).items():
                for arm, dose in sorted(d["realised"].items()):
                    em.macro("RandDose" + texname(pair) + "P" + texname(_partition_no(arm)), fmt(dose, 2), path,
                             f"ratios[{i}].axis_doses['{pair}'].realised.{arm}",
                             f"realised dose of the {d['axis']} axis, the pair's classes left out")
        if "not_computed" in r or r.get("censored_models") or r.get("ci95") is None:
            continue
        lab = p1_class(*r["ci95"])
        if r.get("p1_label") != lab:
            raise SystemExit(f"FATAL: {path} ratios[{i}] P1 label {r.get('p1_label')!r}; its interval gives {lab!r}")
        p1[(tuple(r["probe_pairs"]), r["kind"])] = r
        em.macro("RandPone" + pk + kind, fmt_paired(r["ratio"], *r["ci95"]), path,
                 f"ratios[{i}].ratio, ratios[{i}].ci95",
                 "merging partitions over splitting ones, Welch [95% interval]; above 1 = merging costs")
        em.macro("RandPone" + pk + kind + "Label", P1_TEX[lab], path, f"ratios[{i}].p1_label",
                 "A14's P1 label, re-derived from the interval")
    for i, r in enumerate(rows):
        if r["contrast"] not in ("partition_joint", "partition_joint_dose") or not prim(r) \
                or "not_computed" in r or r.get("ci95") is None:
            continue
        head = "RandJoint" + ("Dose" if r["contrast"].endswith("dose") else "")
        what = texname(*r["probe_pairs"]) if r.get("task") else "Axis" + texname(r["axis"])
        em.macro(head + what + texname(r["kind"]), fmt_paired(r["ratio"], *r["ci95"]), path,
                 f"ratios[{i}].ratio, ratios[{i}].ci95",
                 "joint fit, secondary (A14): " + ("the merge effect" if r.get("task") else
                                                   "the axis merged throughout over kept throughout"))
    for i, r in enumerate(rows):
        if r["contrast"] != "random_vs_semantic" or not prim(r) or "not_computed" in r \
                or r.get("censored_models") or r.get("ci95") is None:
            continue
        if r["fine"].startswith("partitions that merge"):
            group = "Merging"
        elif r["fine"].startswith("partitions that split"):
            group = "Splitting"
        elif r["fine"].startswith("the cells of"):
            group = "Cells" + (texname(*r["account_cells"]) if len(r["probe_pairs"]) > 1 else "")
        else:
            group = "P" + texname(_partition_no(r["partitions"][0]))
        name = "RandVsSem" + texname(r["coarse"]) + texname(*r["probe_pairs"]) + texname(r["kind"]) + group
        em.macro(name, fmt_paired(r["ratio"], *r["ci95"]), path, f"ratios[{i}].ratio, ratios[{i}].ci95",
                 f"{r['coarse']}-class model over {r['fine']}, {r.get('n_runs')} runs paired; "
                 "above 1 = the random side beats it")
        if "axis_account" in r:
            lo95 = math.log(r["ci95"][0])
            lo90, hi90 = _ln_bounds(r["ln_ratio"], r["ln_combined_se"], r["dof"], 0.90)
            axis = "beats" if lo95 > 0 else "inconclusive"
            pair = "equal" if -CKPT_BAND < lo90 and hi90 < CKPT_BAND else "inconclusive"
            if (axis, pair) != (r["axis_account"], r["pair_account"]):
                raise SystemExit(f"FATAL: {path} ratios[{i}] reads {r['axis_account']}/{r['pair_account']}; "
                                 f"its intervals give {axis}/{pair}")
            em.macro(name + "AxisAccount", axis, path, f"ratios[{i}].axis_account",
                     "A14: 'beats' when the 95% lower bound is above 1")
            em.macro(name + "PairAccount", pair, path, f"ratios[{i}].pair_account",
                     "A14: 'equal' when the 90% interval lies within a factor 1.1")
    for j, b in enumerate(R.get("p2_verdict", [])):
        if b["checkpoint"] != V2_PRIMARY or "not_computed" in b:
            continue
        k = texname(b["kind"])
        src = f"p2_verdict[{j}]"
        g = b["gap"]
        em.macro("RandPtwoGap" + k, fmt_diff(g["estimate"], *g["ci95"], nd=3), path, src + ".gap",
                 "17 minus 43 classes, two-prong b vs c, ln(1-AUC) [95%]")
        m = b["margin"]["value"]
        em.macro("RandPtwoMargin" + k, fmt(m, 3), path, src + ".margin.value", "a quarter of the gap")
        em.macro("RandPtwoRuns" + k, word_list(b["runs"]), path, src + ".runs", "runs paired in P2")
        for c, cl in b["clauses"].items():
            iv = cl["ci90"] if c == "c" else cl["ci95"]
            lab = equivalence_class(*iv, m) if c == "c" else threshold_class(*iv)
            if lab != cl["label"]:
                raise SystemExit(f"FATAL: {path} {src} clause ({c}) is {cl['label']!r}; its interval gives {lab!r}")
            em.macro("RandPtwo" + texname(c) + k, fmt_diff(cl["estimate"], *iv, nd=3), path,
                     f"{src}.clauses.{c}", cl["contrast"] + (" [90%]" if c == "c" else " [95%]"))
            em.macro("RandPtwo" + texname(c) + k + "Label", lab, path, f"{src}.clauses.{c}.label",
                     "A14 P2, re-derived from the interval")
        lab_b = b["clauses"]["b"]["label"]
        w = ("not evaluable: the 43- to 17-class gap is not positive" if not m > 0
             else f"not reached: (b) is {lab_b}" if lab_b != "holds"
             else {"holds": "not withdrawn", "fails": "withdrawn", "inconclusive": "inconclusive"}[
                 threshold_class(*b["clauses"]["d"]["ci95"], m)])
        if w != b["withdrawal"]["label"]:
            raise SystemExit(f"FATAL: {path} {src} withdrawal {b['withdrawal']['label']!r}; the clauses give {w!r}")
        em.macro("RandPtwoWithdrawal" + k, w, path, src + ".withdrawal.label", "A14 P2's withdrawal rule")
    return [table_v2_random(A, p1), table_v2_flavour(A, R)]


def _mean_oma(A: dict, task: str, arm: str, probe: str = "linear", runs=None) -> list:
    return [1e3 * (1 - r["auc"]) for r in _v2_rows(A, task, probe, arm) if runs is None or r["seed"] in runs]


def table_v2_random(A: dict, p1: dict) -> str:
    """Each random partition's 1-AUC (x 10^3, the mean of its runs) on each probe pair's own
    task, merged (m) or split (s), beside the 17- and 43-class models (runs 1-2) and P1."""
    parts = sorted({r["arm"] for r in A["table"] if r["arm"].startswith("RAND2_")}, key=_partition_no)
    head = ["probe pair & task & " + " & ".join(f"p{_partition_no(a)}" for a in parts)
            + " & 17 & 43 & merging over splitting \\\\", "\\midrule"]
    body = []
    for (pairs, kind), r in sorted(p1.items()):
        if kind != "linear":
            continue
        cells = []
        for a in parts:
            x = _mean_oma(A, r["task"], a)
            cells.append(("---" if not x else fmt_one(np.mean(x))) + ("$^m$" if a in r["merged_partitions"] else ""))
        for a in ("R16_Q1", "R42_Q1"):
            x = _mean_oma(A, r["task"], a, runs=(1, 2))
            cells.append("---" if len(x) < 2 else fmt_pm(np.mean(x), np.std(x, ddof=1)))
        body.append(f"{tex(' and '.join(pairs))} & {TASK_LABELS.get(r['task'], tex(r['task']))} & "
                    + " & ".join(cells) + f" & {fmt_paired(r['ratio'], *r['ci95'])} ({P1_TEX[r['p1_label']]}) \\\\")
    caption = ("The five random partitions of the second set of runs, frozen linear probe: $1-$AUC in "
               "units of $10^{-3}$ on each probe pair's own task, the mean of each partition's two runs "
               "($^m$: the partition merges the pair), beside the 17- and 43-class models (runs one and "
               f"two, mean {tex('±')} standard deviation); last, the paired ratio of the partitions that "
               "merge the pair over those that split it, with its 95\\% interval and A14's reading.")
    return _table(head + body, caption, "tab:random-partitions", "l l " + "r" * (len(parts) + 3), [], wide=True)


def table_v2_flavour(A: dict, R: dict) -> str:
    """The flavour pair F0, F1, F1r beside the 17- and 43-class models: 1-AUC (x 10^3,
    mean +- SD over runs) on b vs c in two- and four-prong decays, and P2's clauses."""
    tasks = [t for t in ("bvc_resonant", "bvc_4prong") if t in A["tasks"]]
    labels = {"FLAV_F0": "F0", "FLAV_F1": "F1", "FLAV_F1R": "F1r", "R16_Q1": "17 classes", "R42_Q1": "43 classes"}
    head = ["model & " + " & ".join(TASK_LABELS[t] for t in tasks) + " \\\\", "\\midrule"]
    body = []
    for arm, lab in labels.items():
        cells = []
        for t in tasks:
            x = _mean_oma(A, t, arm)
            cells.append("---" if len(x) < 2 else fmt_pm(np.mean(x), np.std(x, ddof=1)))
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    notes = []
    for b in R.get("p2_verdict", []):
        if b["checkpoint"] == V2_PRIMARY and b["kind"] == "linear" and "not_computed" not in b:
            notes.append("P2, linear probe, runs " + word_list(b["runs"]) + ": " + "; ".join(
                f"({c}) {cl['label']}" for c, cl in b["clauses"].items())
                + f"; withdrawal rule: {b['withdrawal']['label']}.")
    caption = ("The flavour pair of the second set of runs, frozen linear probe: $1-$AUC in units of "
               f"$10^{{-3}}$, mean {tex('±')} standard deviation over each model's runs. F0 merges b and c "
               "everywhere; F1 adds one cut that splits b from c in four-prong decays; F1r moves a random "
               "half of the same orbit without any b-to-c boundary.")
    return _table(head + body, caption, "tab:flavour-pair", "l " + "r" * len(tasks), notes)


def emit_v2_mass_lambda(em: Emitter, R: dict, path: pathlib.Path, shares: pathlib.Path | None) -> None:
    """A11 as A14 reads it: MassLambdaFraction<probe> (and ...Wavg), the fraction of the
    excess 17-class mass-output cost on b vs c two-prong that the matched weight removes,
    and MassLambdaFractionInterval<probe>, its 95% Fieller interval, or 'unbounded' when the
    denominator's own interval holds 0. Beside it the realised gradient shares and cosines
    (MassGradShareVtwo<model>, MassGradCosineVtwo<model>) from loss_share.json, whose loss
    shares fill SLOTS."""
    for i, r in enumerate(R["ratios"]):
        if (r["contrast"] != "mass_lambda_fraction" or r["family"] != "probe" or r["task"] != "bvc_resonant"
                or r["metric"] != "1-auc" or r["checkpoint"] not in (V2_PRIMARY, "wavg")
                or "not_computed" in r or "fraction" not in r):
            continue
        k = texname(r["kind"]) + ("Wavg" if r["checkpoint"] == "wavg" else "")
        em.macro("MassLambdaFraction" + k, fmt(r["fraction"], 2), path, f"ratios[{i}].fraction",
                 "the fraction of the excess 17-class mass-output cost the matched weight removes (A11)")
        iv = r.get("ci95")
        em.macro("MassLambdaFractionInterval" + k, "unbounded" if iv is None else f"[{fmt(iv[0], 2)}, {fmt(iv[1], 2)}]",
                 path, f"ratios[{i}].ci95", "95% Fieller interval; unbounded when the denominator's interval holds 0")
    if shares is None:
        return
    S = json.loads(shares.read_text())
    for key, per in S.get("grad_shares", {}).items():
        k = texname(key.replace("+mass", " mass").replace("_", " "))
        em.macro("MassGradShareVtwo" + k, fmt_pm(100 * np.mean(per), 100 * np.std(per, ddof=1)) + "\\%"
                 if len(per) > 1 else fmt_one(100 * per[0]) + "\\%", shares, f"grad_shares['{key}']",
                 "realised trunk-gradient share of the mass term, rho/(1+rho), mean +- SD over runs")
        c = S["grad_cosine"][key]
        em.macro("MassGradCosineVtwo" + k, fmt_pm(np.mean(c), np.std(c, ddof=1)) if len(c) > 1 else fmt_one(c[0]),
                 shares, f"grad_cosine['{key}']", "cosine of the two trunk gradients, mean +- SD over runs")


def welch_ratio(x: list, y: list) -> tuple[float, float, float]:
    """y's runs over x's, unpaired: exp(mean ln y - mean ln x) with Welch's 95 % interval
    (each side its own spread over runs, Student t at the Welch-Satterthwaite degrees of
    freedom): A13's unseen against seen, A14's self-supervised against a vocabulary."""
    from scipy import stats
    lx, ly = np.log(x), np.log(y)
    vx, vy = np.var(lx, ddof=1) / len(lx), np.var(ly, ddof=1) / len(ly)
    d, se = float(ly.mean() - lx.mean()), math.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx ** 2 / (len(lx) - 1) + vy ** 2 / (len(ly) - 1))
    h = float(stats.t.ppf(0.975, dof)) * se
    return math.exp(d), math.exp(d - h), math.exp(d + h)


class CheckpointLabels:
    """The A14 robustness and sensitivity labels this script derives itself, where
    paired_errors.py has no replicates (the anomaly detectors on the features, the
    benchmarks, the real-data yields): the paired ratio over runs of the result at the
    other checkpoint over the primary, Student t at n - 1 (paired_ratio), labelled by
    checkpoint_class. Kept per section, then counted (emit_checkpoint_counts)."""

    def __init__(self):
        self.rows = {}                  # (section, word) -> [(label, source file)]

    def emit(self, em, name, primary: dict, other: dict, word: str, section: str, src, json_path):
        runs = sorted(set(primary) & set(other))
        if len(runs) < 3:
            return
        r, lo, hi = paired_ratio({k: primary[k] for k in runs}, {k: other[k] for k in runs})
        lab = checkpoint_class(lo, hi)
        identical = all(primary[k] == other[k] for k in runs)
        em.macro(name + word + "Shift", fmt_paired(r, lo, hi), src, json_path,
                 f"paired over runs {runs}, Student t: the result at {word.lower()} over the primary")
        em.macro(name + word + "Label", CKPT_LABEL_TEX[lab], src, json_path,
                 "A14's label from that interval" + ("; one checkpoint file, not counted" if identical else ""))
        if not identical:
            self.rows.setdefault((section, word), []).append((lab, src))


V2_SCRATCH = "scratch-v2"


def v2_ft_refs(root: pathlib.Path, ft: dict) -> dict:
    """{dataset key: (file, {N: {fine-tuning seed: cell}})} of the from-scratch reference,
    read where the v2 read-outs put it beside the rule's cells (both legs); the standalone
    JetClass-II readout in finetune_references/ must be the same cells."""
    out = {ds: (F["path"], F["cells"][V2_SCRATCH]) for ds, F in ft.items() if V2_SCRATCH in F["cells"]}
    alone = v2_dir(root, "finetune_references") / "scratch_leg1_metrics.json"
    if alone.exists():
        cells = json.loads(alone.read_text())["cells"][V2_SCRATCH]
        if "Jcii" in out and out["Jcii"][1] != cells:
            raise SystemExit(f"FATAL: {alone.name} and {out['Jcii'][0].name} hold different from-scratch cells")
        out.setdefault("Jcii", (alone, cells))
    return out


def ssl_validity(em: Emitter, ft: dict, refs: dict, grid: list) -> bool:
    """PRESPEC 4 re-evaluated on the second grid (A14), on the inferential endpoint 2.3 fixes
    for these tasks, ln(1 - macro AUC), as its outcome of 2026-09-24 read it:
      (1) at 10^3 and 10^4 training jets, on JetClass-II and JetClass, the self-supervised
          runs beat training from scratch by more than the spread: SslMargin<ds><N> = the
          scratch mean over its fine-tuning seeds minus the self-supervised mean over its runs
          (fine-tuning seed 1), SslSpread<ds><N> = the larger of the two standard deviations
          (SslScratchSd, SslRunSd; SslSpreadSource names which), SslClauseOne<ds><N> 'holds'
          when the margin exceeds it; SslClauseOne over the four;
      (2) at 10^6 jets on JetClass-II its macro AUC is within 0.02 of the supervised-
          pretrained models': SslGap (the supervised mean minus the self-supervised mean over
          every run on the tree), SslClauseTwo.
    SslValid: 'valid' when both hold; the model then enters the tables."""
    mpm = next(a for a in grid if a["name"] == "MPM")
    inits = [f"mpm-v2-s{k}" for k in range(1, mpm["runs"] + 1)]
    sup = [f"{a['name'].lower().replace('_', '')}-s{k}" for a in grid if a["name"] in RUNGS
           for k in range(1, a["runs"] + 1)]
    lnoma = lambda c: math.log(1 - c["macro_auc_ovr"])
    if ({"Jcii", "Jc"} - set(refs) or {"Jcii", "Jc"} - set(ft)
            or any(i not in ft[ds]["cells"] for ds in ("Jcii", "Jc") for i in inits)):
        raise SystemExit("FATAL: PRESPEC 4's bar needs the self-supervised and the from-scratch fine-tuning "
                         "on JetClass-II and JetClass")
    ok1 = True
    for ds in ("Jcii", "Jc"):
        F, (rsrc, ref) = ft[ds], refs[ds]
        for n in ("N1000", "N10000"):
            s = [lnoma(F["cells"][i][n]["s1"]) for i in inits]
            z = [lnoma(c) for c in ref[n].values()]
            sd_z, sd_s = float(np.std(z, ddof=1)), float(np.std(s, ddof=1))
            margin, spread = float(np.mean(z) - np.mean(s)), max(sd_z, sd_s)
            key = ds + n_tag(n)
            em.macro("SslScratchSd" + key, fmt(sd_z, 3), F["path"], f"cells.scratch-v2.{n}.*.macro_auc_ovr",
                     "SD of ln(1-macro AUC) over the from-scratch fine-tuning seeds")
            em.macro("SslRunSd" + key, fmt(sd_s, 3), F["path"], f"cells.mpm-v2-s*.{n}.s1.macro_auc_ovr",
                     "SD of ln(1-macro AUC) over the self-supervised runs")
            em.macro("SslSpreadSource" + key, "the from-scratch fine-tuning seeds" if sd_z >= sd_s
                     else "the self-supervised runs", F["path"], "max(SslScratchSd, SslRunSd)",
                     "which spread clause 1 is judged against (the larger)")
            em.macro("SslMargin" + key, fmt(margin, 3), F["path"], f"cells.{{scratch-v2,mpm-v2-s*}}.{n}.macro_auc_ovr",
                     "ln(1-macro AUC): from scratch (mean over fine-tuning seeds) minus self-supervised (mean over runs)")
            em.macro("SslSpread" + key, fmt(spread, 3), F["path"], f"cells.{{scratch-v2,mpm-v2-s*}}.{n}.macro_auc_ovr",
                     "the larger standard deviation: scratch over fine-tuning seeds, self-supervised over runs")
            em.macro("SslClauseOne" + key, "holds" if margin > spread else "fails", F["path"],
                     f"cells.*.{n}", "PRESPEC 4 clause 1 at this size and task")
            ok1 &= margin > spread
    J = ft["Jcii"]
    sup_auc = [J["cells"][i]["N1000000"]["s1"]["macro_auc_ovr"] for i in sup if i in J["cells"]]
    gap = float(np.mean(sup_auc) - np.mean([J["cells"][i]["N1000000"]["s1"]["macro_auc_ovr"] for i in inits]))
    ok2 = abs(gap) <= 0.02
    em.macro("SslClauseOne", "holds" if ok1 else "fails", J["path"], "the four cells of clause 1")
    em.macro("SslGap", fmt(gap, 4), J["path"], "cells.*.N1000000.s1.macro_auc_ovr",
             f"supervised mean over {len(sup_auc)} runs minus self-supervised mean, macro AUC")
    em.macro("SslClauseTwo", "holds" if ok2 else "fails", J["path"], "|SslGap| <= 0.02", "PRESPEC 4 clause 2")
    em.macro("SslValid", "valid" if ok1 and ok2 else "not valid", J["path"], "clauses 1 and 2", "PRESPEC 4")
    return ok1 and ok2


def ft_with_refs(ft: dict, refs: dict, ssl_inits: list | None) -> tuple[dict, list]:
    """The fine-tuning files with the from-scratch reference as one pseudo-model per
    fine-tuning seed ('scratch-v2#s<k>', each at s1), and the reference rows: random
    initialisation, and the self-supervised model when it is valid."""
    out = {}
    seeds = None
    for ds, F in ft.items():
        cells = dict(F["cells"])
        if ds in refs:
            ref = refs[ds][1]
            seeds = sorted({s for per in ref.values() for s in per})
            for s in seeds:
                cells[f"{V2_SCRATCH}#{s}"] = {n: {"s1": per[s]} for n, per in ref.items() if s in per}
        out[ds] = {**F, "cells": cells}
    rows = []
    if seeds and all(ds in refs for ds in ft):
        rows.append(("Scratch", "random initialisation, over fine-tuning seeds",
                     [f"{V2_SCRATCH}#{s}" for s in seeds]))
    if ssl_inits:
        rows.append(("SelfSupervised", "self-supervised", ssl_inits))
    return out, rows


def emit_v2_ft_refs(em: Emitter, ft: dict, rows: list, ref_rows: list) -> None:
    """FtAuc<ds><N><row>, FtAcc<ds><N><row> of the reference rows (Scratch, SelfSupervised),
    mean +- SD over the from-scratch fine-tuning seeds or the self-supervised runs, printed
    to the places the vocabulary rows fix (ft_text)."""
    for ds, F in ft.items():
        for metric, name in (("macro_auc_ovr", "FtAuc"), ("accuracy", "FtAcc")):
            text = ft_text(F["cells"], rows + ref_rows, metric)
            for key, _, inits in ref_rows:
                for n in ft_sizes(F["cells"], rows):
                    em.macro(name + ds + n_tag(n) + key, text[(key, n)], F["path"],
                             f"cells.{{{','.join(inits)}}}.{n}.s1.{metric}",
                             f"mean +- SD over {len(inits)} " + ("fine-tuning seeds" if key == "Scratch" else "runs"))


# The benchmark read-outs (scripts/build_ft_jobs.py, v2 read-outs): {macro part: (file kind,
# set)}, best validation at every size and the last epoch at the full set (A6).
V2_BENCH = {"Top": ("bench_metrics", "top"), "Qg": ("bench_metrics", "qg"),
            "QgHerwig": ("bench_metrics_herwig", "qg")}
BENCH_METRICS = (("Rfifty", "r50"), ("Rthirty", "r30"), ("Auc", "auc"), ("Acc", "accuracy"))


def v2_bench_files(root: pathlib.Path, rule: str) -> dict:
    """{file kind: path} of `rule`'s benchmark read-outs, every tier's in place of the
    freeze's once it exists, each freeze cell then the same in both; None for a rule not
    read out."""
    d, out = v2_dir(root, "benchmarks"), {}
    for kind in ("bench_metrics", "bench_metrics_herwig", "bench_metrics_last", "bench_metrics_herwig_last"):
        full, t12 = d / f"{rule}_{kind}.json", d / f"{rule}_t12_{kind}.json"
        if full.exists() and t12.exists():
            F, T = (json.loads(p.read_text())["cells"] for p in (full, t12))
            bad = [(s, i) for s in T for i in T[s] if T[s][i] != F.get(s, {}).get(i)]
            if bad:
                raise SystemExit(f"FATAL: {t12.name} and {full.name} disagree on {bad[:3]}")
        p = full if full.exists() else t12 if t12.exists() else None
        if p is not None:
            out[kind] = p
    return out


def _bench_values(cells: dict, inits: list, n: str, metric: str) -> tuple:
    """(values, bound flags, passing counts) of one metric over a row's models or the
    from-scratch fine-tuning seeds (an init 'scratch-v2#s<k>')."""
    cs = [cells[i.split("#")[0]][n][i.split("#")[1] if "#" in i else "s1"] for i in inits]
    if metric in ("r50", "r30"):
        return [c[metric] for c in cs], [c[f"{metric}_is_bound"] for c in cs], [c[f"{metric}_n_bkg_pass"] for c in cs]
    if metric == "auc":
        return [c["auc"] for c in cs], [c["log1m_auc_censored"] for c in cs], None
    return [c[metric] for c in cs], None, None


def _bench_text(vals, bounds, n_pass, metric) -> str:
    if metric in ("r50", "r30"):
        return fmt_rejection(vals, bounds, n_pass)
    if metric == "auc":
        return fmt_auc_pm(vals, bounds) if len(vals) > 1 else fmt_one(vals[0])
    return fmt_pm(np.mean(vals), np.std(vals, ddof=1)) if len(vals) > 1 else fmt_one(vals[0])


def emit_v2_bench(em: Emitter, files: dict, rows: list, labels: CheckpointLabels) -> list:
    """The benchmarks (top tagging, quark/gluon on Pythia and on Herwig) fine-tuned from every
    model, PRESPEC 2.3's metrics, mean +- SD over each row's runs (the from-scratch row over
    its fine-tuning seeds):
      Bench<set><metric><N><row>       best validation epoch, every training size (A6);
      BenchLast<set><metric><row>      the last epoch at the full training set (A6's rule for
                                       C2, C3 and S5);
      ...Wavg                          the same from the weight average (A8);
      BenchLast<set>Rfifty<row>WavgShift, ...WavgLabel   paired over runs, A14's label.
    metric Rfifty, Rthirty (rejection at 50 and 30 % signal efficiency), Auc, Acc."""
    out = []
    for rule, suffix in (("best70", ""), ("wavg", "Wavg")):
        for set_key, (kind, s) in V2_BENCH.items():
            for last in (False, True):
                p = files.get(rule, {}).get(kind + ("_last" if last else ""))
                if p is None:
                    continue
                cells = json.loads(p.read_text())["cells"][s]
                use = [(k, lab, i) for k, lab, i in rows if all(x.split("#")[0] in cells for x in i)]
                for key, _, inits in use:
                    for n in sorted(cells[inits[0].split("#")[0]], key=lambda x: int(x[1:])):
                        for mk, metric in BENCH_METRICS:
                            v, b, k = _bench_values(cells, inits, n, metric)
                            size = n_tag(n) if re.fullmatch(r"N10*", n) else "Full"   # 1.2M, 1.6M: the full set
                            name = ("BenchLast" + texname(set_key) + mk if last
                                    else "Bench" + texname(set_key) + mk + size) + key + suffix
                            em.macro(name, _bench_text(v, b, k, metric), p,
                                     f"cells.{s}.{{{','.join(inits)}}}.{n}.{metric}",
                                     ("last epoch" if last else "best validation epoch") + f", {rule}, mean +- SD")
    # A14's label of the C2/C3/S5 quantity, the last-epoch rejection at 50 % on the full set
    for set_key, (kind, s) in V2_BENCH.items():
        pb, pw = (files.get(r, {}).get(kind + "_last") for r in ("best70", "wavg"))
        if pb is None or pw is None:
            continue
        cb, cw = (json.loads(p.read_text())["cells"][s] for p in (pb, pw))
        for key, _, inits in rows:
            if any("#" in i for i in inits) or not all(i in cb and i in cw for i in inits):
                continue
            n = next(iter(cb[inits[0]]))
            get = lambda c: {v2_run(i): c[i][n]["s1"]["r50"] for i in inits if not c[i][n]["s1"]["r50_is_bound"]}
            labels.emit(em, "BenchLast" + texname(set_key) + "Rfifty" + key, get(cb), get(cw), "Wavg", "benchmarks",
                        pw, f"cells.{s}.*.{n}.s1.r50 over {pb.name}")
    for rule in ("best70",):
        p = files.get(rule, {}).get("bench_metrics_last")
        if p is not None:
            out.append(table_v2_bench_last(files[rule], rows))
    return out


def table_v2_bench_last(files: dict, rows: list) -> str:
    """The benchmarks at the last epoch on the full training sets (A6), primary checkpoint."""
    sets = [(k, kind, s) for k, (kind, s) in V2_BENCH.items() if kind + "_last" in files]
    head = ["& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in
                              ("top tagging" if k == "Top" else "quark/gluon, Herwig" if k == "QgHerwig"
                               else "quark/gluon" for k, _, _ in sets)) + " \\\\",
            "pretraining & " + " & ".join("$1/\\epsilon_B$ at 50\\% & AUC" for _ in sets) + " \\\\", "\\midrule"]
    body, n_used, scratch_row = [], set(), False
    docs = {k: json.loads(files[kind + "_last"].read_text())["cells"][s] for k, kind, s in sets}
    for key, label, inits in rows:
        cells = []
        for k, _, _ in sets:
            c = docs[k]
            if not all(i.split("#")[0] in c for i in inits):
                cells += ["---", "---"]
                continue
            n = next(iter(c[inits[0].split("#")[0]]))
            n_used.add(n)
            for metric in ("r50", "auc"):
                cells.append(_bench_text(*_bench_values(c, inits, n, metric), metric))
        if set(cells) != {"---"}:          # a row read out at no set: the from-scratch reference
            body.append(f"{label} & " + " & ".join(cells) + " \\\\")
            scratch_row |= any("#" in i for i in inits)
    caption = ("Top tagging and quark/gluon tagging (trained on Pythia, tested on Pythia and on Herwig) "
               "fine-tuned from every model of the second set of runs, from its primary checkpoint, on the "
               "full training sets at the last epoch (amendment A6): background rejection at 50\\% signal "
               f"efficiency and AUC, mean {tex('±')} standard deviation over each row's pretraining runs"
               + (" (the random initialisation over its fine-tuning seeds)." if scratch_row else "."))
    return _table(head + body, caption, "tab:benchmarks-last", "l " + "rr" * len(sets), [], wide=True)


V2_ANOMALY_FEATURE_FAMILIES = ("knn", "mahalanobis")   # anomaly_summary.V2_FAMILIES


def v2_anomaly_inputs(root: pathlib.Path) -> dict:
    """{(tag, readout): (path, summary)}: anomaly_summary.py --grid's output for every tag and
    readout of the anomaly set read, v2/anomaly/<set>/summary/<tag>/<readout>/, the set of
    every tier (t123) in place of the freeze's (t12) once it exists; then each run's value
    common to both must be the same, and each summary's inputs unchanged since."""
    d = v2_dir(root, "anomaly")
    sets = sorted((p for p in d.iterdir() if p.is_dir() and (p / "summary").is_dir()), key=lambda p: len(p.name))
    if not sets:
        raise SystemExit(f"FATAL: {d.relative_to(root)} holds no <set>/summary/")
    load = lambda s: {(p.parents[1].name, p.parent.name): (p, json.loads(p.read_text()))
                      for p in sorted((s / "summary").glob("*/*/anomaly_summary.json"))}
    out = load(sets[-1])
    for s in sets[:-1]:
        for key, (p, S) in load(s).items():
            T = out.get(key, (None, None))[1]
            for fam, blk in S["families"].items():
                for sig, per_n in blk.items():
                    for arm, e in per_n.get("2000", {}).get("levels", {}).items():
                        other = (((T or {}).get("families", {}).get(fam, {}).get(sig, {}).get("2000", {})
                                  .get("levels", {}).get(arm)))
                        if other is None or other["ln_sigma_min"] != e["ln_sigma_min"]:
                            raise SystemExit(f"FATAL: {p} and {sets[-1].name}'s summary disagree on {fam}/{sig}/{arm}")
    for p, _ in out.values():
        check_inputs_unchanged(p, root)
    return out


def _anomaly_runs(e: dict) -> dict:
    """{run index: sigma_min} of one anomaly cell (a summary level or a checkpoint_rule entry)."""
    return {v2_run(a): math.exp(x) for a, x in zip(e["arms"], e["ln_sigma_min"])}


def lofo_signals(root: pathlib.Path, grid: list, v1_summary: dict | None, fam: str) -> list:
    """A14's A13 signals: X->YY->bbbb, and every other anomaly signal of the left-out family
    (a native class the models that leave the family out never see: the grid's
    extra_selection on its jet_label) that met the first grid's detection rule for the
    detector `fam` when seen (its not_detected list)."""
    sel = {a["extra_selection"] for a in grid if a.get("extra_selection")}
    if len(sel) != 1:
        raise SystemExit(f"FATAL: the grid's family-out arms leave out {len(sel)} different selections")
    expr = sel.pop()
    pairs = re.findall(r"\(jet_label >= (\d+)\) & \(jet_label < (\d+)\)", expr)
    singles = re.findall(r"\(jet_label == (\d+)\)", expr)
    ranges = [(int(a), int(b)) for a, b in pairs] + [(int(a), int(a) + 1) for a in singles]
    if not expr.startswith("~(") or expr.count("jet_label") != 2 * len(pairs) + len(singles):
        raise SystemExit(f"FATAL: the family-out selection {expr!r} is not ~(ranges of jet_label)")
    rows = {r["class_name"]: int(r["jet_label"]) for r in
            csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    out_of = lambda jl: any(a <= jl < b for a, b in ranges)
    nd = set((v1_summary or {}).get("not_detected_rule", {}).get("not_detected", []))
    return [s for s in SIGNAL_LABELS if s in rows and out_of(rows[s])
            and (s == "label_X_YY_bbbb" or f"{fam}|{s}" not in nd)]


def emit_v2_anomaly(em: Emitter, V: dict, grid: list, labels_of: dict, root: pathlib.Path,
                    products: dict | None, v1_summary: dict | None, ck: CheckpointLabels) -> list:
    """Section 5 on the second grid under the first grid's names, at the primary checkpoint
    through the class token: AnomalySigmaMin<fam><signal><level> and AnomalyMaxSic... for the
    k-nearest-neighbour and Mahalanobis detectors (anomaly_summary.py --grid), and
    AnomalySigmaMinClassSumMatched<signal><level>, the output ratio, from the output layers at
    best70 (its checkpoint_rule); the settings; PairedAnomaly<fam><signal><coarse>Over<fine>,
    paired by run, Student t (the output ratio's are paired_errors.py's), and
    PairedAnomalyOutputOverMahalanobis<signal><level>. Beside the detectors: ...Wavg,
    ...Bestval, ...Twin and ...Pooled, the shifts to the weight average and the global best
    with A14's labels, and the reference rows ...SelfSupervised (pooled) and ...Untrained
    [Pooled]. A13 (Lofo): LofoUnseen<fam><signal><level> = unseen over seen, Welch on runs
    1-3 of both (one GPU product, I7), LofoSigmaMin<fam><signal><level> the family-out runs,
    and PairedAnomaly<fam>Unseen<signal><coarse>Over<fine>, the ladder among them, paired,
    descriptive. Returns the tables."""
    src, S = V[(V2_PRIMARY, V2_CLASS_TOKEN)]
    inj = S["conventions"]["primary_injection"]
    R = json.loads((em.root / S["provenance"]["inputs"]["anomaly"]["path"]).read_text())
    res_path = em.root / S["provenance"]["inputs"]["anomaly"]["path"]
    em.macro("AnomalyInjection", fmt_int(inj), src, "conventions.primary_injection", "injected signal jets")
    em.macro("AnomalyNResamplings", str(S["provenance"]["resamplings_per_seed"]), src,
             "provenance.resamplings_per_seed", "resamplings per model, median taken")
    em.macro("AnomalyNBkg", fmt_int(R["n_bkg"]), res_path, "n_bkg", "background jets in the data sample")
    em.macro("AnomalyNTemplate", fmt_int(R["n_template"]), res_path, "n_template", "jets in the background template")
    em.macro("AnomalyStatCut", f"{100 * R['stat_cut']:.0f}\\%", res_path, "stat_cut",
             "largest relative statistical error on eps_B at a usable threshold")
    em.macro("AnomalyMinBkgPass", str(int(np.ceil(1 / R["stat_cut"] ** 2))), res_path, "stat_cut",
             "background jets that must pass a threshold, 1/stat_cut^2")
    em.macro("AnomalySigmaT", fmt(R["sigma_t"], 0), res_path, "sigma_t", "target significance")
    code = em.root / "experiments" / "EVAL" / "anomaly.py"
    if code.exists():
        k = re.findall(r"^KNN_K = (\d+)", code.read_text(), re.M)
        if len(k) != 1:
            raise SystemExit(f"FATAL: {code} sets KNN_K {len(k)} times")
        em.macro("AnomalyKnnK", k[0], code, "KNN_K", "the k of the nearest-neighbour distance")
    nd = set(S["not_detected_rule"]["not_detected"])
    em.macro("AnomalyNdThreshold", fmt(S["not_detected_rule"]["threshold_max_sic"], 1), src,
             "not_detected_rule.threshold_max_sic", "max SIC below this at every model")
    level_of = {a["name"]: a["num_classes"] for a in grid if a["name"] in RUNGS}
    tree = sorted((a for a in S["families"]["knn"][next(iter(S["families"]["knn"]))][inj]["levels"]
                   if a in level_of), key=lambda a: -level_of[a])
    cell = lambda T, fam, sig, arm: T["families"][fam][sig][inj]["levels"].get(arm)
    n_runs = {len(cell(S, "knn", next(iter(S["families"]["knn"])), a)["arms"]) for a in tree}
    if len(n_runs) != 1:
        raise SystemExit(f"FATAL: the v2 anomaly levels hold {sorted(n_runs)} runs; the text states one")
    em.macro("AnomalyNRunsWord", words(n_runs.pop()), src, f"families.knn.*.{inj}.levels.*.arms (count)",
             "pretraining runs per vocabulary")
    heads = S.get("checkpoint_rule", {}).get("by_checkpoint", {})
    cs = lambda tag, sig, arm: heads.get(tag, {}).get("class_sum_matched", {}).get(sig, {}).get(inj, {}).get(arm)
    pm = lambda v: fmt_pm(np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else fmt_one(v[0])
    for fam in V2_ANOMALY_FEATURE_FAMILIES:
        for sig in S["families"][fam]:
            for arm in tree:
                c = cell(S, fam, sig, arm)
                key = texname(fam) + anomaly_signal_key(sig) + texname(level_of[arm])
                jp = f"families.{fam}.{sig}.{inj}.levels.{arm}"
                em.macro("AnomalySigmaMin" + key, pm(c["sigma_min"]), src, jp + ".sigma_min", "mean +- SD over runs")
                em.macro("AnomalyMaxSic" + key, pm(c["max_sic"]), src, jp + ".max_sic", "mean +- SD over runs")
                for (tag, ro), word in (((V2_TWIN, V2_CLASS_TOKEN), "Twin"), ((V2_PRIMARY, V2_POOLED), "Pooled"),
                                        (("wavg", V2_CLASS_TOKEN), "Wavg"), (("bestval", V2_CLASS_TOKEN), "Bestval")):
                    if (tag, ro) not in V:
                        continue
                    s2, T = V[(tag, ro)]
                    o = cell(T, fam, sig, arm)
                    if o is None:
                        continue
                    em.macro("AnomalySigmaMin" + key + word, pm(o["sigma_min"]), s2, jp + ".sigma_min",
                             f"at {tag}, {ro}; mean +- SD over runs")
                    if word in ("Wavg", "Bestval"):
                        ck.emit(em, "AnomalySigmaMin" + key, _anomaly_runs(c), _anomaly_runs(o), word, "anomaly detectors",
                                s2, jp + ".ln_sigma_min")
            for (tag, ro), word in (((V2_INIT, V2_CLASS_TOKEN), "Untrained"), ((V2_INIT, V2_POOLED), "UntrainedPooled"),
                                    ((V2_PRIMARY, V2_POOLED), "SelfSupervised")):
                arm = "init" if word.startswith("Untrained") else "MPM"
                if (tag, ro) in V and (o := cell(V[(tag, ro)][1], fam, sig, arm)) is not None:
                    em.macro("AnomalySigmaMin" + texname(fam) + anomaly_signal_key(sig) + word, pm(o["sigma_min"]),
                             V[(tag, ro)][0], f"families.{fam}.{sig}.{inj}.levels.{arm}.sigma_min",
                             "reference row, mean +- SD over runs")
    for sig in SIGNAL_LABELS:
        for arm in tree:
            c = cs(V2_PRIMARY, sig, arm)
            if c is not None:
                em.macro("AnomalySigmaMinClassSumMatched" + anomaly_signal_key(sig) + texname(level_of[arm]),
                         pm([math.exp(x) for x in c["ln_sigma_min"]]), src,
                         f"checkpoint_rule.by_checkpoint.best70.class_sum_matched.{sig}.{inj}.{arm}",
                         "the output ratio at the primary checkpoint; mean +- SD over runs")
    # paired by run between vocabularies, as the first grid's (Student t over runs)
    for fam in V2_ANOMALY_FEATURE_FAMILIES:
        for sig in S["families"][fam]:
            if f"{fam}|{sig}" in nd:
                continue
            for i, fine in enumerate(tree):
                for coarse in tree[i + 1:]:
                    em.macro("PairedAnomaly" + texname(fam) + anomaly_signal_key(sig) + texname(level_of[coarse])
                             + "Over" + texname(level_of[fine]),
                             fmt_paired(*paired_ratio(_anomaly_runs(cell(S, fam, sig, fine)),
                                                      _anomaly_runs(cell(S, fam, sig, coarse)))), src,
                             f"families.{fam}.{sig}.{inj}.levels.{{{coarse},{fine}}}.ln_sigma_min",
                             "sigma_min ratio paired by run [95% Student-t interval]; above 1 = less sensitive")
    for sig in S["families"]["mahalanobis"]:
        for arm in tree:
            c, m = cs(V2_PRIMARY, sig, arm), cell(S, "mahalanobis", sig, arm)
            if c is None or m is None or {f"class_sum_matched|{sig}", f"mahalanobis|{sig}"} & nd:
                continue
            em.macro("PairedAnomalyOutputOverMahalanobis" + anomaly_signal_key(sig) + texname(level_of[arm]),
                     fmt_paired(*paired_ratio(_anomaly_runs(m), _anomaly_runs(c))), src,
                     f"checkpoint_rule.by_checkpoint.best70.class_sum_matched.{sig}.{inj}.{arm} over "
                     f"families.mahalanobis.{sig}.{inj}.levels.{arm}",
                     "output ratio over Mahalanobis distance, paired by run [95% Student-t interval]")
    tables = [table_v2_anomaly(S, tree, level_of, cs, inj, nd), table_v2_anomaly_per_run(S, tree, level_of, cs, inj, nd)]
    # A13: the family left out, against the same vocabulary seen (Welch, runs 1-3)
    lofo = {a["parent"]: a for a in grid if a.get("extra_selection") and a.get("parent")}
    body = []
    for fam in V2_ANOMALY_FEATURE_FAMILIES:
        for sig in lofo_signals(root, grid, v1_summary, fam):
            if sig not in S["families"][fam]:
                continue
            for parent, a in lofo.items():
                (s2, T) = V[(V2_PRIMARY, V2_POOLED if parent == "MPM" else V2_CLASS_TOKEN)] \
                    if (V2_PRIMARY, V2_POOLED if parent == "MPM" else V2_CLASS_TOKEN) in V else (None, None)
                if T is None or cell(T, fam, sig, a["name"]) is None or cell(T, fam, sig, parent) is None:
                    continue
                runs = list(range(1, a["runs"] + 1))
                seen = {k: v for k, v in _anomaly_runs(cell(T, fam, sig, parent)).items() if k in runs}
                unseen = {k: v for k, v in _anomaly_runs(cell(T, fam, sig, a["name"])).items() if k in runs}
                if products is not None and len({products[k] for k in set(seen) | set(unseen)}) > 1:
                    raise SystemExit(f"FATAL: A13 {a['name']} against {parent} spans two GPU products (I7)")
                lv = "SelfSupervised" if parent == "MPM" else texname(level_of[parent])
                key = texname(fam) + anomaly_signal_key(sig) + lv
                r, lo, hi = welch_ratio(list(seen.values()), list(unseen.values()))
                em.macro("LofoUnseen" + key, fmt_paired(r, lo, hi), s2,
                         f"families.{fam}.{sig}.{inj}.levels.{{{a['name']},{parent}}}.ln_sigma_min, runs {runs}",
                         "sigma_min unseen over seen, Welch [95% interval]; above 1 = less sensitive unseen")
                em.macro("LofoSigmaMin" + key, pm(list(unseen.values())), s2,
                         f"families.{fam}.{sig}.{inj}.levels.{a['name']}.sigma_min", "mean +- SD over its runs")
                body.append(f"{SIGNAL_LABELS[sig]} & {FAMILY_LABELS[fam]} & {tex(labels_of.get(parent, parent))} & "
                            f"{pm(list(seen.values()))} & {pm(list(unseen.values()))} & {fmt_paired(r, lo, hi)} \\\\")
            ladder = sorted((a["name"] for p, a in lofo.items() if p in level_of), key=lambda n: -level_of[lofo_parent(n, lofo)])
            for i, fine in enumerate(ladder):
                for coarse in ladder[i + 1:]:
                    f, c = cell(S, fam, sig, fine), cell(S, fam, sig, coarse)
                    if f is None or c is None:
                        continue
                    em.macro("PairedAnomaly" + texname(fam) + "Unseen" + anomaly_signal_key(sig)
                             + texname(level_of[lofo_parent(coarse, lofo)]) + "Over" + texname(level_of[lofo_parent(fine, lofo)]),
                             fmt_paired(*paired_ratio(_anomaly_runs(f), _anomaly_runs(c))), src,
                             f"families.{fam}.{sig}.{inj}.levels.{{{coarse},{fine}}}.ln_sigma_min",
                             "family-out models, paired by run [95% Student-t interval]; descriptive (A14)")
    if body:
        head = ["signal & detector & vocabulary & seen & unseen & unseen over seen \\\\", "\\midrule"]
        caption = ("The family of $X\\to YY\\to bbbb$ left out of pretraining (amendment A13): "
                   "$\\sigma_{\\min}$ of the models that never saw it against the same vocabulary seen, "
                   f"mean {tex('±')} standard deviation over runs one to three, and their ratio with Welch's 95\\% "
                   "interval. Leaving the family out changes the exposure to every other class as stated in the "
                   "text.")
        tables.append(_table(head + body, caption, "tab:lofo", "l l l r r r", []))
    return tables


def lofo_parent(name: str, lofo: dict) -> str:
    return next(p for p, a in lofo.items() if a["name"] == name)


def table_v2_anomaly(S, tree, level_of, cs, inj, nd) -> str:
    """sigma_min and max SIC per signal, detector and vocabulary of the second grid."""
    sigs = [g for g in SIGNAL_LABELS if all(g in S["families"][f] for f in V2_ANOMALY_FEATURE_FAMILIES)
            and any(f"{f}|{g}" not in nd for f in V2_ANOMALY_FEATURE_FAMILIES)]
    head = ["& detector & " + " & ".join(f"{level_of[a]} classes" for a in tree) + " \\\\", "\\midrule"]
    body = []
    for qty, name in (("sigma_min", "$\\sigma_{\\min}$"), ("max_sic", "max SIC")):
        body.append(f"\\multicolumn{{{2 + len(tree)}}}{{@{{}}l}}{{\\itshape {name}}} \\\\")
        for g in sigs:
            fams = list(V2_ANOMALY_FEATURE_FAMILIES) + (["class_sum_matched"] if qty == "sigma_min" else [])
            for i, f in enumerate(fams):
                cells = []
                for a in tree:
                    if f == "class_sum_matched":
                        c = cs(V2_PRIMARY, g, a)
                        v = None if c is None else [math.exp(x) for x in c["ln_sigma_min"]]
                    else:
                        v = S["families"][f][g][inj]["levels"][a][qty]
                    cells.append("---" if not v else fmt_pm(np.mean(v), np.std(v, ddof=1)))
                body.append((SIGNAL_LABELS[g] if i == 0 else "") + f" & {FAMILY_LABELS[f]} & " + " & ".join(cells) + " \\\\")
        if qty == "sigma_min":
            body.append("\\addlinespace")
    caption = ("Anomaly detection with the models of the second set of runs at their primary checkpoint: "
               "two detectors on the frozen features and the resonance-to-QCD probability ratio from the "
               f"model's own outputs (output ratio), {fmt_int(inj)} signal jets injected. $\\sigma_{{\\min}}$ "
               "is the smallest initial significance from which a discovery is still reached (lower is more "
               f"sensitive). Mean {tex('±')} standard deviation over each vocabulary's runs of each run's median "
               "over resamplings.")
    return _table(head + body, caption, "tab:anomaly", "l l " + "r" * len(tree), [])


def table_v2_anomaly_per_run(S: dict, tree: list, level_of: dict, cs, inj: str, nd: set) -> str:
    """sigma_min of every run of the second grid, for the signals and scores of the anomaly
    table, runs in order: the values behind its mean +- SD."""
    sigs = [g for g in SIGNAL_LABELS if all(g in S["families"][f] for f in V2_ANOMALY_FEATURE_FAMILIES)
            and any(f"{f}|{g}" not in nd for f in V2_ANOMALY_FEATURE_FAMILIES)]
    body = []
    for g in sigs:
        for i, f in enumerate([*V2_ANOMALY_FEATURE_FAMILIES, "class_sum_matched"]):
            cells = []
            for a in tree:
                e = cs(V2_PRIMARY, g, a) if f == "class_sum_matched" else S["families"][f][g][inj]["levels"][a]
                cells.append("---" if e is None else ", ".join(fmt(v, 2) for _, v in sorted(_anomaly_runs(e).items())))
            body.append((SIGNAL_LABELS[g] if i == 0 else "") + f" & {FAMILY_LABELS[f]} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    head = ["& score & " + " & ".join(f"{level_of[a]} classes" for a in tree) + " \\\\", "\\midrule"]
    caption = ("$\\sigma_{\\min}$ of each pretraining run of the second set, runs in order, for the signals and "
               "scores of Table~\\ref{tab:anomaly}: each value is that run's median over resamplings at "
               f"{fmt_int(inj)} injected signal jets.")
    return _table(head + body[:-1], caption, "tab:anomaly-per-run", "l l " + "r" * len(tree), [], wide=True)


V2_AOJ_FREEZE, V2_AOJ_LATER = "t12", "t3"     # scripts/build_aoj_jobs.py v2_label of tiers (1, 2) and (3,)


def _v2_aoj_run(root: pathlib.Path, d: pathlib.Path) -> tuple:
    """(analysis, (fit file, fit)) of one v2 real-data run; the fit must be the file of the
    hash its analysis recorded."""
    src = d / "analysis_v6" / "aoj_top.json"
    if not src.exists():
        raise SystemExit(f"FATAL: {src.relative_to(root)} does not exist")
    J = json.loads(src.read_text())
    res_path = d / "fit_v6" / "results.json"
    if not res_path.exists() or hashlib.sha256(res_path.read_bytes()).hexdigest() != J["provenance"]["input_sha256"]:
        raise SystemExit(f"FATAL: {res_path.relative_to(root)} is not the fit {d.name}'s analysis read")
    return J, (res_path, json.loads(res_path.read_text()))


def v2_real_data(root: pathlib.Path) -> dict:
    """The real-data fits of the second grid, as scripts/build_aoj_jobs.py --v2 runs them:
    real_data/t12/ (the freeze, tiers 1-2, which derives the pooled peak shape) for every row
    of tiers 1-2, and real_data/t3/ (tier 3, fitted at t12's shape: experiments/AOJ/fit_v6.py
    --v2 --shape-from) only for the rows of tier-3 arms. Each: analysis_v6/aoj_top.json, the
    fit it read (fit_v6/results.json) and, for t12, injection/summary.json. Refused: no t12, a
    t12 that holds another fit's shape, a t3 whose fit does not hold t12's (its pooled_shape's
    held_from and held_from_sha256), and any other directory."""
    d = v2_dir(root, "real_data")
    dirs = {p.name for p in d.iterdir() if p.is_dir()}
    if V2_AOJ_FREEZE not in dirs or dirs - {V2_AOJ_FREEZE, V2_AOJ_LATER}:
        raise SystemExit(f"FATAL: {d.relative_to(root)} holds {sorted(dirs)}; the second grid's real data "
                         f"is {V2_AOJ_FREEZE}/ and, for tier 3, {V2_AOJ_LATER}/")
    J, fits = _v2_aoj_run(root, d / V2_AOJ_FREEZE)
    if "held_from" in fits[1]["pooled_shape"]:
        raise SystemExit(f"FATAL: the freeze fit holds the shape of {fits[1]['pooled_shape']['held_from']}; "
                         "it derives its own")
    inj = d / V2_AOJ_FREEZE / "injection" / "summary.json"
    out = {"src": d / V2_AOJ_FREEZE / "analysis_v6" / "aoj_top.json", "J": J, "fits": fits,
           "injection": inj if inj.exists() else None, "t3": None}
    if V2_AOJ_LATER in dirs:
        J3, fits3 = _v2_aoj_run(root, d / V2_AOJ_LATER)
        ps = fits3[1]["pooled_shape"]
        if (ps.get("held_from_sha256") != hashlib.sha256(fits[0].read_bytes()).hexdigest()
                or not str(ps.get("held_from", "")).endswith(f"/{V2_AOJ_FREEZE}/fit_v6/results.json")):
            raise SystemExit(f"FATAL: {V2_AOJ_LATER}/'s fit holds the shape of {ps.get('held_from')!r} "
                             f"({str(ps.get('held_from_sha256'))[:16]}), not the freeze fit's")
        out["t3"] = {"src": d / V2_AOJ_LATER / "analysis_v6" / "aoj_top.json", "J": J3, "fits": fits3}
    return out


def emit_v2_real_data(em: Emitter, RD: dict, labels_of: dict, grid: list, paths: dict,
                      ck: CheckpointLabels) -> str:
    """Section 6 on the second grid under the first grid's names (emit_real_data) at the
    primary checkpoint, from the freeze fit (t12): the per-checkpoint label sets relabelled
    by the contrasts file's labels, the injection test from its own summary; beside each yield
    AojYield<set>Wavg, ...Bestval, ...Twin and the shifts to the weight average and the global
    best with A14's labels; and the levels the first grid lacks as AojYield<set> and
    AojStatErr<set>: the matched mass weight from t12, the 64- and 30-class levels from t3,
    fitted at t12's shape. A run that holds an arm of the other's tiers is refused. Returns
    the table."""
    tier = {labels_of.get(a["name"], a["name"]): int(a["tier"]) for a in grid}
    runs_of = {V2_AOJ_FREEZE: RD, V2_AOJ_LATER: RD["t3"]}
    for name, R in runs_of.items():
        for tag, blk in ((R or {}).get("J", {}).get("per_checkpoint") or {}).items():
            wrong = [labels_of.get(a, a) for a in blk["label_sets"]
                     if (tier.get(labels_of.get(a, a), 0) == 3) != (name == V2_AOJ_LATER)]
            if wrong:
                raise SystemExit(f"FATAL: real_data/{name}/ holds {wrong} at {tag}, rows of the other run's tiers")
    rel = lambda per, tag: {**per[tag], "label_sets": {labels_of.get(a, a): v for a, v in per[tag]["label_sets"].items()}}
    per12 = RD["J"]["per_checkpoint"]
    J = {"provenance": RD["J"]["provenance"], "per_label_set": rel(per12, V2_PRIMARY)}
    emit_real_data(em, J, RD["src"], {**paths, "aoj_injection": RD["injection"]}, RD["fits"])
    extra = [labels_of.get(a["name"], a["name"]) for a in grid
             if (a["name"] in RUNGS or a["name"] == "R16_Q1_MASS_LM")
             and labels_of.get(a["name"], a["name"]) not in AOJ_SETS]
    runs = lambda c: {run_index(m): y for m, y in zip(c["models"], c["signal_yields"])}
    for lv in [*AOJ_SETS, *extra]:
        R = RD if tier.get(lv, 0) < 3 else RD["t3"]
        if R is None or lv not in rel(R["J"]["per_checkpoint"], V2_PRIMARY)["label_sets"]:
            continue
        per, src = R["J"]["per_checkpoint"], R["src"]
        c = rel(per, V2_PRIMARY)["label_sets"][lv]
        k = texname(lv.replace("+mass", " mass"))
        if lv in extra:
            em.macro("AojYield" + k, fmt_pm(c["signal_yield"]["mean"], c["signal_yield"]["sd"]), src,
                     f"per_checkpoint.best70.label_sets.{lv}.signal_yield", f"mean +- SD over {c['signal_yield']['n']} runs")
            em.macro("AojStatErr" + k, fmt_int(c["median_stat_err"]), src,
                     f"per_checkpoint.best70.label_sets.{lv}.median_stat_err", "median per-fit statistical error")
        for tag, word in (("wavg", "Wavg"), ("bestval", "Bestval"), (V2_TWIN, "Twin")):
            o = rel(per, tag)["label_sets"].get(lv) if tag in per else None
            if o is None:
                continue
            em.macro("AojYield" + k + word, fmt_pm(o["signal_yield"]["mean"], o["signal_yield"]["sd"]), src,
                     f"per_checkpoint.{tag}.label_sets.{lv}.signal_yield", f"at {tag}; mean +- SD over runs")
            if word in ("Wavg", "Bestval"):
                ck.emit(em, "AojYield" + k, runs(c), runs(o), word, "real data", src,
                        f"per_checkpoint.{{best70,{tag}}}.label_sets.{lv}.signal_yields")
    return table_realdata(J, em.root, RD["fits"])


def v2_inputs(root: pathlib.Path, paths: dict, have: dict) -> dict:
    """The second grid's inputs of every section V2_READ lists whose directory exists:
    {"probe", "mass_resolution", "label_recovery_curve": {(tag, readout): (path, doc)},
     "ft": [leg files], "paired": [family files], "pretraining": True, "grid": [arms],
     "labels": {arm: label}, "products": {run: GPU product} or None}. Nothing for a root
    without the grid (a first-grid fixture)."""
    present = {k for k, (_, sub) in V2_PENDING.items() if sub and v2_present(root, sub)}
    if not present & V2_READ:
        return {}
    for k in ("v2_grid", "contrasts_v2"):
        if k not in have:
            raise SystemExit(f"FATAL: v2 inputs exist and {paths[k]} does not; it names their arms")
    out = {"grid": json.loads(paths["v2_grid"].read_text())["arms"],
           "labels": json.loads(paths["contrasts_v2"].read_text()).get("labels", {}),
           "products": v2_products(paths.get("v2_grid_specs") or [])}
    frozen = [s for s in V2_FROZEN if v2_present(root, s)]
    if frozen and "probe_ladder" not in frozen:
        raise SystemExit(f"FATAL: v2 frozen readouts {frozen} without probe_ladder, the tiers that "
                         "come first (A14) and the directory that holds their analysis")
    if "Probes" in present:
        for analysis in ("probe", "mass_resolution"):
            out[analysis] = v2_analysis(root, analysis, paths["v2_grid"])
        lack = [k for k in V2_FROZEN_NEEDED if k not in out["probe"]]
        if lack:
            raise SystemExit(f"FATAL: the v2 frozen probes lack {lack}: A14 prints the primary, its "
                             "BatchNorm twin and the reference rows together")
    if "Recovery" in present:
        out["label_recovery_curve"] = v2_analysis(root, "label_recovery_curve", paths["v2_grid"])
        if (V2_PRIMARY, V2_CLASS_TOKEN) not in out["label_recovery_curve"]:
            raise SystemExit("FATAL: the v2 label-recovery curves lack the primary checkpoint's class token")
    if "FtHeldout" in present:
        out["ft"] = v2_ft_files(root)
    if "Paired" in present:
        out["paired"] = [p for p in (v2_dir(root, "paired_errors") / f / "ratios.json"
                                     for f in V2_PAIRED_FAMILIES) if p.exists()]
    if "Pretrain" in present:
        out["pretraining"] = True
    probes_ratios = v2_dir(root, "paired_errors") / "probes" / "ratios.json"
    for key, need in (("Random", ("Probes", "Paired")), ("MassLambda", ("Probes", "Paired")),
                      ("Ssl", ("FtHeldout",)), ("FtRefs", ("FtHeldout",)), ("Bench", ("FtHeldout",))):
        if key in present and (set(need) - present or ("Paired" in need and not probes_ratios.exists())):
            raise SystemExit(f"FATAL: v2 {V2_PENDING[key][1]}/ is present and its readout needs "
                             f"{sorted(V2_PENDING[k][1] for k in need)}" + (" with paired_errors/probes/ratios.json"
                                                                          if "Paired" in need else ""))
    if "Random" in present:
        out["random"] = probes_ratios
    if "MassLambda" in present:
        shares = v2_dir(root, "mass_lambda_matched") / "loss_share.json"
        out["mass_lambda"] = (probes_ratios, shares if shares.exists() else None)
    out["ssl"] = "Ssl" in present
    out["ft_refs"] = "FtRefs" in present
    if "Bench" in present:
        out["bench"] = {rule: v2_bench_files(root, rule) for rule in ("best70", "wavg")}
        if not out["bench"]["best70"]:
            raise SystemExit("FATAL: v2 benchmarks/ holds no best70 read-out")
    if "Anomaly" in present:
        out["anomaly"] = v2_anomaly_inputs(root)
        if (V2_PRIMARY, V2_CLASS_TOKEN) not in out["anomaly"]:
            raise SystemExit("FATAL: the v2 anomaly summaries lack the primary checkpoint's class token")
    if "RealData" in present:
        out["real_data"] = v2_real_data(root)
    return out


def v2_not_computed(docs: dict, root: pathlib.Path) -> list:
    """Every result the v2 paired files hold without a value, the joint fit's included, with
    paired_errors.py's reason: none is skipped silently. Written beside provenance.json as
    v2_not_computed.json and printed on every run."""
    out = []
    for path, R in docs.items():
        rel = _v2_rel(root, path)
        for i, r in enumerate(R["ratios"]):
            if "not_computed" in r:
                out.append({"file": rel, "where": f"ratios[{i}]", "contrast": r["contrast"],
                            **{k: r.get(k) for k in ("family", "task", "axis", "kind", "metric", "checkpoint",
                                                     "fine", "coarse")}, "reason": r["not_computed"]})
        for j, b in enumerate(R.get("p2_verdict", [])):
            if "not_computed" in b:
                out.append({"file": rel, "where": f"p2_verdict[{j}]", "contrast": b["contrast"],
                            "kind": b.get("kind"), "checkpoint": b.get("checkpoint"), "reason": b["not_computed"]})
    return out


def emit_unprinted(em: Emitter, root: pathlib.Path, missing: list) -> None:
    """A macro the text uses, which the paper's last generation defined
    (paper/journal/provenance.json) and this one does not: a section now printed from the
    second grid has no such number (the first grid's output-layer mean over epochs 70-79,
    its random-label draws, ...). It becomes a red marker, so the draft still compiles and
    shows each sentence to rewrite; never the first grid's number under the second grid's
    text. Nothing when every name is defined, as with the first grid alone."""
    tex_, prov = root / "paper" / "journal" / "main.tex", root / "paper" / "journal" / "provenance.json"
    me = pathlib.Path(__file__).resolve()
    if not (tex_.exists() and prov.exists() and me.is_relative_to(em.root.resolve())):
        return
    used = set(re.findall(r"\\([A-Za-z]+)", tex_.read_text()))
    for name in sorted(used & set(json.loads(prov.read_text())) - set(em.provenance)):
        em.macro(name, "\\pending{no v2 number: rewrite this}", tex_, "used in the text",
                 "a marker, not a number: its section reads the second grid, which has no such number")
        missing.append(f"{name} -- used in the text, no v2 number")


# ------------------------------------------------------------------ assembly

def render_macros(em: Emitter, missing: list[str], skipped: list[str]) -> str:
    head = [
        "% GENERATED by experiments/FIGS/make_tables.py -- do not edit.",
        "% Every line carries: source file :: path inside that file :: first 16 hex of its",
        "% sha256. paper/journal/provenance.json carries the full hash beside the value.",
        "% Regenerate with:  python3 experiments/FIGS/make_tables.py",
        "% Check in CI with: python3 experiments/FIGS/make_tables.py --check",
        "%",
        "% Macro names spell digits as words (162 -> Onesixtwo) because \\newcommand takes",
        "% letters only. A number whose source file does not exist yet is NOT emitted --",
        "% see the missing list below, and leave a \\pending{} in the text for it.",
    ]
    if missing:
        head += ["%", "% MISSING INPUTS -- no macro was emitted for these:"]
        head += [f"%   {m}" for m in missing]
    if skipped:
        head += ["%", "% SKIPPED TABLES:"]
        head += [f"%   {s}" for s in skipped]
    head.append("")
    body = [f"\\newcommand{{\\{n}}}{{{b}}}  % {c}" for n, b, c in em.macros]
    return "\n".join(head + body) + "\n"


def build(root: pathlib.Path) -> tuple[dict, list, list]:
    """Everything the paper gets, as {relative path: text}, plus the two reports."""
    paths = input_paths(root)
    missing, skipped = [], []
    for name, pattern in PENDING_INPUTS:
        if not list(root.glob(pattern)):
            missing.append(f"{name} -- {pattern}")

    have = {k: v for k, v in paths.items()
            if (v if isinstance(v, list) else [v]) and
            all(pathlib.Path(p).exists() for p in (v if isinstance(v, list) else [v]))}
    for key in ("ladder", "analysis", "rung_map"):
        if key not in have:
            raise SystemExit(f"FATAL: {key} is required to build any table and is missing "
                             f"({paths[key]})")
    check_alignment(paths, root)
    A = json.loads(pathlib.Path(paths["analysis"]).read_text())
    check_analysis_is_current(A, root)
    sizes = vocabulary_sizes(paths["rung_map"])

    # The second grid (v2_inputs): each section it holds replaces the first grid's below.
    V2 = v2_inputs(root, paths, have)
    restrict = v2_one_product(V2["products"]) if V2.get("products") else None
    em = Emitter(root)
    emit_pending(em, missing)
    emit_slots(em, missing)
    ck = CheckpointLabels()
    # The self-supervised model enters a table only once PRESPEC 4's bar passes on v2 (A14).
    ft_v2 = ft_load(V2["ft"]) if "ft" in V2 else None
    refs = v2_ft_refs(root, ft_v2) if ft_v2 and (V2["ssl"] or V2["ft_refs"]) else {}
    ssl_ok = bool(V2.get("ssl")) and ssl_validity(em, ft_v2, refs, V2["grid"])
    if "probe" in V2:
        src2, A2 = V2["probe"][(V2_PRIMARY, V2_CLASS_TOKEN)]
        emit_design(em, A2, src2)
        emit_levels(em, A2, src2)
    else:
        emit_design(em, A, paths["analysis"])
        emit_levels(em, A, paths["analysis"])
    emit_vocabulary(em, sizes, paths["rung_map"])
    if "probe_code" in have:
        emit_probe_settings(em, paths["probe_code"])
    if "probe" in V2:
        nbkg = emit_v2_probes(em, V2["probe"], ssl_ok)
        out = {f"tables/probes_{p}.tex": table_probe_ladder_v2(V2["probe"], p, nbkg, ssl_ok)
               for p in ("linear", "mlp")}
    else:
        # The test-sample background count, which bounds every rejection: the same
        # jets in every ladder file, or the files were not scored on the same sample.
        ladders = [json.loads(pathlib.Path(f).read_text()) for f in paths["ladder"]]
        nbkg = {}
        for task in sorted(ladders[0]["tasks"]):
            counts = {d["tasks"][task].get("n_background_test") for d in ladders if task in d["tasks"]}
            if len(counts) == 1 and None not in counts:
                nbkg[task] = counts.pop()
                em.macro("ProbeNBkgTest" + texname(task), fmt_int(nbkg[task]), paths["ladder"][0],
                         f"tasks.{task}.n_background_test", "test-sample background jets")

        out = {"tables/probes_linear.tex": table_probe_ladder(A, "linear", nbkg),
               "tables/probes_mlp.tex": table_probe_ladder(A, "mlp", nbkg)}
    if V2.get("pretraining"):
        t = emit_v2_pretraining(em, root, V2["grid"], V2["labels"])
        if t:
            out["tables/v2_selected_epochs.tex"] = t

    if "survival" in have:
        surv = json.loads(pathlib.Path(paths["survival"]).read_text())
        emit_usecase(em, surv, sizes, paths["survival"], paths["rung_map"])
        pretrained = [r for r in RUNGS if r in ARM_RUNG.values()
                      or (r in ("R63_Q1", "R29_Q1") and v2_present(root, "levels_64_30"))]
        out["tables/usecase_survival.tex"] = table_usecase(surv, sizes, pretrained)
    else:
        missing.append(f"use-case survival -- {paths['survival']}")
        skipped.append("T3 use-case survival: configs/labelmaps/usecase_survival.v1.json missing")

    if all(k in have for k in ("design_spec", "design_arch", "design_arm")):
        emit_training_design(em, paths["design_spec"], paths["design_arch"], paths["design_arm"])
    if "literature" in have:
        emit_literature(em, paths["literature"])
    if "mass_specs" in have and paths["mass_specs"]:
        emit_mass_lambda(em, paths["mass_specs"])
    if all(k in have for k in ("ft_recipes", "ft_leg_specs")) and paths["ft_leg_specs"]:
        reported = {i for _, _, inits in ft_rows(sizes) for i in inits}
        rec = ft_recipe(json.loads(paths["ft_recipes"].read_text()), paths["ft_leg_specs"], reported)
        emit_ft_recipe(em, rec, paths["ft_recipes"])
        out["tables/finetune_recipe.tex"] = table_ft_recipe(rec)
    later = (("recovery", lambda em, d, src: emit_recovery(em, d, src, sizes,
                                                          A["provenance"].get("row_alignment_sha256")),
              lambda d: table_recovery(d, sizes), "label_recovery"),
             ("random_control", emit_random_control, lambda d: table_random_control(d, A, probes=("linear", "mlp")),
              "random_control"),
             ("anomaly", emit_anomaly, lambda d: table_anomaly(d, root), "anomaly"),
             ("mass_resolution", emit_mass_resolution, lambda d: table_mass(d, root), "mass"),
             ("real_data", lambda em, d, src: emit_real_data(em, d, src, paths),
              lambda d: table_realdata(d, root), "realdata"))
    v2_later = {}
    if "label_recovery_curve" in V2:
        v2_later["recovery"] = (V2["label_recovery_curve"][(V2_PRIMARY, V2_CLASS_TOKEN)],
                                lambda em, d, src: emit_recovery(em, d, src, sizes),
                                lambda d: table_recovery(d, sizes))
    if "mass_resolution" in V2:
        M2 = V2["mass_resolution"][(V2_PRIMARY, V2_CLASS_TOKEN)][1]
        order = ([c for c, _ in sorted({(r["cell"], r["level"]) for r in M2["table"] if r["level"]},
                                       key=lambda x: -x[1])]
                 + [c for c in ("162+mass", "17+mass", "17+mass, matched lambda")
                    if c in {r["cell"] for r in M2["table"]}])
        v2_later["mass_resolution"] = (V2["mass_resolution"][(V2_PRIMARY, V2_CLASS_TOKEN)],
                                       lambda em, d, src: emit_mass_resolution(em, d, src, order),
                                       lambda d: table_mass(d, root, order))
    v1_anomaly = json.loads(paths["anomaly"].read_text()) if "anomaly" in have else None
    for key, emit, table, name in later:
        if key == "anomaly" and "anomaly" in V2:
            t = emit_v2_anomaly(em, V2["anomaly"], V2["grid"], V2["labels"], root, V2["products"],
                                v1_anomaly, ck)
            out.update(zip(("tables/anomaly.tex", "tables/anomaly_per_run.tex", "tables/v2_lofo.tex"), t))
            continue
        if key == "real_data" and "real_data" in V2:
            out["tables/realdata.tex"] = emit_v2_real_data(em, V2["real_data"], V2["labels"], V2["grid"],
                                                           paths, ck)
            continue
        if key in v2_later:
            (src, d), emit2, table2 = v2_later[key]
            emit2(em, d, src)
            out[f"tables/{name}.tex"] = table2(d)
            continue
        if key not in have:
            missing.append(f"{key} -- {paths[key]}")
            continue
        check_inputs_unchanged(paths[key], root)
        d = json.loads(pathlib.Path(paths[key]).read_text())
        emit(em, d, paths[key])
        out[f"tables/{name}.tex"] = table(d)
        if key == "anomaly":
            out["tables/anomaly_per_run.tex"] = table_anomaly_per_run(d)

    rand_maps = ("rand_v1_map", "v2_grid", "rand_v2_map", "rand_v2_rule", "flavour_pair", "flavour_map")
    if "random_control" in have and all(k in have for k in rand_maps):
        emit_rand_design(em, paths, json.loads(pathlib.Path(paths["random_control"]).read_text()),
                         missing)
    if all(k in have for k in ("mass_lambda", "v2_grid")):
        emit_mass_lambda_matched(em, paths["mass_lambda"], paths["v2_grid"])
    if all(k in have for k in ("design_spec_v2", "pretrain_v2")):
        emit_v2_training(em, paths["design_spec_v2"], paths["pretrain_v2"])
    if all(k in have for k in ("design_spec_v2", "lofo_spec_v2", "pretrain_v2", "v2_dryrun",
                               "v2_dryrun_lofo", "v2_grid_specs")):
        emit_v2_recipe(em, paths, missing)
    if "probe" in V2:
        emit_mass_output_v2(em, A2, src2, restrict)
    elif "mass2x2" in have:
        emit_mass_output(em, A, paths["analysis"], paths["mass2x2"], sizes)
    else:
        missing.append("mass-output probes -- experiments/FIGS/data/probe_ladder_mass2x2/s*.json")
    ft = None
    if "ft" in V2:
        ft = ft_v2
        cells = next(iter(ft.values()))["cells"]
        rows = ft_rows_v2(V2["grid"], V2["labels"], cells)
        for F in ft.values():
            if {i for *_, inits in rows for i in inits} - set(F["cells"]):
                raise SystemExit(f"FATAL: {F['path'].name} lacks rows the other fine-tuning file has")
        lv = {a["name"]: texname(a["num_classes"]) for a in V2["grid"] if a["name"] in RUNGS}
        emit_finetune(em, ft, sizes, rows, {r: lv[r] for r in ("L188", "L162", "R16_Q1")}, restrict)
        mpm = next((a for a in V2["grid"] if a["name"] == "MPM"), None)
        ft2, ref_rows = ft_with_refs(ft, refs if V2["ft_refs"] else {},
                                     [f"mpm-v2-s{k}" for k in range(1, mpm["runs"] + 1)] if ssl_ok else None)
        if ref_rows:
            emit_v2_ft_refs(em, ft2, rows, ref_rows)
        out["tables/finetune.tex"] = table_finetune(ft2, sizes, "macro_auc_ovr", rows + ref_rows, v2=True)
        out["tables/finetune_accuracy.tex"] = table_finetune(ft2, sizes, "accuracy", rows + ref_rows, v2=True)
        if "bench" in V2:
            out.update(zip(("tables/v2_benchmarks_last.tex",), emit_v2_bench(em, V2["bench"], rows + ref_rows, ck)))
        skipped.append("finetune_recipe: the settings of the first grid's fine-tuning runs; the "
                       "second grid's recipe is not tabulated yet")
    elif "ft_legs" in have:
        ft = ft_load(paths["ft_legs"])
        emit_finetune(em, ft, sizes)
        out["tables/finetune.tex"] = table_finetune(ft, sizes, "macro_auc_ovr")
        out["tables/finetune_accuracy.tex"] = table_finetune(ft, sizes, "accuracy")
    else:
        missing.append(f"fine-tuning metrics -- {', '.join(map(str, paths['ft_legs']))}")
    if "paired" in V2:
        docs = emit_paired_v2(em, V2["paired"], ft, V2["products"],
                              set() if ssl_ok else {"MPM", "MPM_LOFO4P"})
        t = emit_checkpoint_counts(em, docs, ck.rows)
        if t:
            out["tables/checkpoint_robustness.tex"] = t
        out["v2_not_computed.json"] = json.dumps(v2_not_computed(docs, root), indent=1) + "\n"
        if "random" in V2:
            out.update(zip(("tables/v2_random_partitions.tex", "tables/v2_flavour_pair.tex"),
                           emit_v2_random(em, json.loads(V2["random"].read_text()), V2["random"], A2, src2)))
        if "mass_lambda" in V2:
            emit_v2_mass_lambda(em, json.loads(V2["mass_lambda"][0].read_text()), *V2["mass_lambda"])
    elif "paired" in have:
        emit_paired(em, paths["paired"], ft)
    else:
        missing.append(f"paired ratios -- {', '.join(map(str, paths['paired']))}")
    if "probe" in V2:
        emit_vcb_v2(em, A2, src2)
    elif "vcb" in have:
        emit_vcb(em, json.loads(pathlib.Path(paths["vcb"]).read_text()), paths["vcb"], sizes)
    else:
        missing.append(f"vcb window probe -- {paths['vcb']}")

    legs = {leg: json.loads(pathlib.Path(paths[f"leg{leg}"]).read_text())
            for leg in (1, 2) if f"leg{leg}" in have}
    if legs:
        emit_legs(em, legs, {leg: paths[f"leg{leg}"] for leg in legs}, sizes)
        out["tables/finetuning_wave1.tex"] = table_legs(legs, sizes)
    else:
        missing.append("fine-tuning legs -- experiments/FIGS/data/leg{1,2}_metrics.json")

    # T4. docs/RECORD.md 2.1 is the only place the per-arm parameter and MAC
    # counts are written down, and it is a document, not a result file: the
    # measurement it reports lives at /data/results/mtx/flops/flops_by_arm.json
    # on the cluster and nothing in this repository holds it. Copying the numbers
    # out of the .md would make this script the second hand-typed source it
    # exists to remove, so the table is skipped until that JSON is committed.
    skipped.append("T4 compute/parameters: no machine-readable source in the repository; "
                   "docs/RECORD.md 2.1 names /data/results/mtx/flops/flops_by_arm.json, "
                   "which is not committed")
    missing.append("per-arm parameters and MACs -- flops_by_arm.json (see docs/RECORD.md 2.1)")

    # The appendix: vocabularies, random partitions, and the classes of every task.
    out.update(appendix_module().build(root, paths["ladder"][0], paths["vcb"] if "vcb" in have else None,
                              paths["anomaly"] if "anomaly" in have else None,
                              TASK_LABELS, v2_present(root, "levels_64_30")))

    emit_unprinted(em, root, missing)
    out["results_generated.tex"] = render_macros(em, missing, skipped)
    out["provenance.json"] = json.dumps(em.provenance, indent=2, sort_keys=True) + "\n"
    return out, missing, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=REPO,
                    help="repository root holding experiments/FIGS/data and configs/labelmaps")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="destination directory (default <root>/paper/journal)")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit non-zero if regenerating would change anything")
    a = ap.parse_args(argv)
    out_dir = a.out or (a.root / "paper" / "journal")

    built, missing, skipped = build(a.root)
    nc = json.loads(built.get("v2_not_computed.json", "[]"))
    for r in nc:
        print(f"NOT COMPUTED  {r['file']} {r['where']} {r['contrast']} "
              f"{r.get('task') or r.get('axis') or ''} {r.get('kind') or ''} {r.get('checkpoint')}: {r['reason']}")
    if nc:
        print(f"{len(nc)} v2 result(s) paired_errors.py did not compute; listed in v2_not_computed.json\n")

    if a.check:
        drift = []
        for rel, text in sorted(built.items()):
            p = out_dir / rel
            if not p.exists():
                drift.append(f"{rel}: not generated yet")
            elif p.read_text() != text:
                drift.append(f"{rel}: differs from what the inputs now say")
        for line in drift:
            print(f"DRIFT  {line}")
        for line in missing:
            print(f"MISSING  {line}")
        if drift:
            print(f"\n{len(drift)} generated file(s) are stale. Run "
                  f"python3 experiments/FIGS/make_tables.py")
            return 1
        print(f"up to date: {len(built)} generated file(s) match the inputs")
        return 0

    for rel, text in sorted(built.items()):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p.relative_to(a.root) if p.is_relative_to(a.root) else p}")
    n_macros = len(json.loads(built["provenance.json"]))
    print(f"\n{n_macros} macros, each traceable to a file, a path inside it and its sha256")
    if skipped:
        print("\nSKIPPED:")
        for s in skipped:
            print(f"  {s}")
    if missing:
        print("\nMISSING INPUTS (no number was written for any of these; a slot the text "
              "uses prints a red pending marker instead):")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
