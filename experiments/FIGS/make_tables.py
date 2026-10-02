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


def emit_pending(em: Emitter, missing: list) -> None:
    """One red marker per v2 input the text waits for, from V2_PENDING (this file).

    The source recorded for each marker is this script, where the registry lives.
    A fixture root that does not contain this script gets no markers: it has no
    manuscript that could print them.
    """
    me = pathlib.Path(__file__).resolve()
    if not me.is_relative_to(em.root.resolve()):
        return
    for key, (what, sub) in V2_PENDING.items():
        here = (v2_present(em.root, sub) if sub else
                all(v2_present(em.root, x) for _, x in V2_PENDING.values() if x))
        text = (f"v2 input present ({sub}): rewrite this from it" if here
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
    label-recovery learning curve."""
    n = max(R["sizes"])
    out = {}
    for r in R["table"]:
        if r["probe"] == probe and r["n_train"] == n:
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


def emit_finetune(em: Emitter, ft: dict, sizes: dict) -> None:
    """Fine-tuning every pretrained model on JetClass-II and JetClass: macro AUC and
    accuracy at the best-validation-accuracy epoch, fine-tuning seed s1."""
    rows = ft_rows(sizes)
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
        for n in ns:
            for fine in (texname(sizes["L188"]), texname(sizes["L162"])):
                for key in oma:
                    if key == fine:
                        continue
                    em.macro("FtOmaRatio" + ds + n_tag(n) + key + "Over" + fine,
                             fmt_ratio(oma[key][n] / oma[fine][n]), src,
                             f"cells.*.{n}.s1.macro_auc_ovr",
                             "ratio of the seed means of 1 - macro AUC, this row over the reference")
        # The accuracy has no bootstrap interval in the paired files; the coarse-minus-fine
        # difference paired by run, with a Student-t interval over the runs, is what the
        # text may read in words.
        by_key = {key: inits for key, _, inits in rows}
        fine, coarse = texname(sizes["L188"]), texname(sizes["R16_Q1"])
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


def emit_mass_resolution(em: Emitter, M: dict, src: pathlib.Path) -> None:
    """Frozen-feature jet-mass regression: sigma_eff per label set and probe."""
    n_test, first = mass_n_test(M, em.root)
    em.macro("MassResNTest", fmt_int(n_test), first, "centering_detail.split[2]",
             "test jets sigma_eff is computed on, the same in every per-seed file and arm")
    em.macro("MassResNClasses", str(M["provenance"]["n_classes_used"]), src,
             "provenance.n_classes_used", "native classes with enough training jets to centre")
    for probe in ("ridge", "mlp"):
        k = texname(probe)
        for cell in MASS_CELLS:
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


def emit_real_data(em: Emitter, J: dict, src: pathlib.Path, paths: dict | None = None) -> None:
    """The top peak in CMS open data (AspenOpenJets) at 1% data efficiency, from fit_v6:
    one Gaussian peak shape shared by the pretrained models (pooled over them) and the
    tops that fail each cut taken from the CMS reference's fit (experiments/AOJ/fit_v6.py).
    The yield per vocabulary is the mean +- SD over the pretraining runs; each model's
    own floated shape, the shape and fail-region systematics, the working-point fit
    quality and the passing-jet residual below the top window are the checks."""
    paths = paths or {}
    res_path, res = aoj_fits(em.root, J)
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
    cells = " ".join(auc_all + rej_all)
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
    return _table(head + body, caption, f"tab:probes-{probe}",
                  "r l " + "r" * len(tasks), notes, wide=True)


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


def table_finetune(ft: dict, sizes: dict, metric: str) -> str:
    """Fine-tuning on JetClass-II and JetClass, one metric, per pretrained model
    (rows) and fine-tuning set size (columns)."""
    rows = ft_rows(sizes)
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


def table_mass(M: dict, root: pathlib.Path) -> str:
    """Jet-mass resolution from frozen features, both probes."""
    tgt = {round(r["target_sigma_eff"], 12) for r in M["table"] if r["probe"] == "ridge"}
    tgt = fmt(tgt.pop(), 4) if len(tgt) == 1 else "---"
    head = ["& " + " & ".join(c.replace("+mass", " + mass") for c in MASS_CELLS)
            + " & true-class mean \\\\", "\\midrule"]
    body = []
    for probe, name in (("mlp", "nonlinear (MLP) probe"), ("ridge", "linear (ridge) probe")):
        cells = [fmt_pm(np.mean(v), np.std(v, ddof=1))
                 for v in (mass_values(M, c, probe) for c in MASS_CELLS)]
        body.append(f"{name} & " + " & ".join(cells) + f" & {tgt} \\\\")
    caption = ("Jet-mass regression from frozen features: $\\sigma_{\\mathrm{eff}}$ of the residual. "
               "The residual is $\\ln(m_{\\mathrm{pred}}/m_{\\mathrm{true}})$ after removing each "
               "native class's training-set mean; $\\sigma_{\\mathrm{eff}}$ is half the smallest "
               f"interval holding 68\\% of it. Mean {tex('±')} standard deviation over the "
               f"{words(len(mass_values(M, MASS_CELLS[0], 'ridge')))} pretraining runs, on "
               f"{fmt_int(mass_n_test(M, root)[0])} test jets. Columns are the number of classes "
               "in the pretraining vocabulary, with or without the added mass output; the last is "
               "an oracle that knows each jet's true native class and returns that class's mean, "
               "the same for both probes.")
    return _table(head + body, caption, "tab:mass", "l " + "r" * (len(MASS_CELLS) + 1), [],
                  wide=True)


def table_realdata(J: dict, root: pathlib.Path) -> str:
    """Fitted top-quark yield in CMS open data at the working point the fits record (fit_v6)."""
    P = J["per_label_set"]
    _, res = aoj_fits(root, J)
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

    em = Emitter(root)
    emit_pending(em, missing)
    emit_slots(em, missing)
    emit_design(em, A, paths["analysis"])
    emit_levels(em, A, paths["analysis"])
    emit_vocabulary(em, sizes, paths["rung_map"])
    if "probe_code" in have:
        emit_probe_settings(em, paths["probe_code"])
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
    for key, emit, table, name in later:
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
    if "mass2x2" in have:
        emit_mass_output(em, A, paths["analysis"], paths["mass2x2"], sizes)
    else:
        missing.append("mass-output probes -- experiments/FIGS/data/probe_ladder_mass2x2/s*.json")
    ft = None
    if "ft_legs" in have:
        ft = ft_load(paths["ft_legs"])
        emit_finetune(em, ft, sizes)
        out["tables/finetune.tex"] = table_finetune(ft, sizes, "macro_auc_ovr")
        out["tables/finetune_accuracy.tex"] = table_finetune(ft, sizes, "accuracy")
    else:
        missing.append(f"fine-tuning metrics -- {', '.join(map(str, paths['ft_legs']))}")
    if "paired" in have:
        emit_paired(em, paths["paired"], ft)
    else:
        missing.append(f"paired ratios -- {', '.join(map(str, paths['paired']))}")
    if "vcb" in have:
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
