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
import csv
import hashlib
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
    "visible_content": "visible decay content",
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


def fmt_ratio(x) -> str:
    """A ratio to two significant figures, three when the leading digit is 1, so
    3.7 but 1.02 rather than a 1.0 that hides the difference."""
    e = math.floor(math.log10(abs(float(x))))
    return fmt_one(x, (2 if f"{float(x):e}"[0] == "1" else 1) - e)


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
    return {"ladder": sorted((data / "probe_ladder_v2").glob("s*.json"))
                      + sorted((data / "probe_ladder_mass2x2").glob("s*.json")),
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
            "analysis": data / "probe_ladder_v2" / "analysis_family_of_four" / "seed_level_results.json",
            "leg1": data / "leg1_metrics.json",
            "leg2": data / "leg2_metrics.json",
            "recovery": data / "label_recovery_ladder_v1" / "analysis" / "s9_label_recovery.json",
            # Each of these is the latest pass of its analysis. Every earlier pass
            # beside it is kept and differs only where its PRESPEC entry says:
            # analysis_v2 of C4 adds the post-hoc grouping cost; analysis_v2 of
            # S3/S4 and of section 5 fix wording; analysis_holm of S7 applies the
            # Holm correction amendment A4 requires; analysis_labelled of the real
            # data fixes one verdict's wording.
            "random_control": data / "probe_ladder_randcontrol" / "analysis_v2" / "c4_random_control.json",
            # The +mass arms' probe files alone, for the mass-output cells.
            "mass2x2": sorted((data / "probe_ladder_mass2x2").glob("s*.json")),
            # The per-cell fine-tuning metrics themselves (fine-tuning wave 2b); which
            # file is which dataset is read from the class count its cells record.
            "ft_legs": [data / "w2b_leg1_metrics.json", data / "w2b_leg2_metrics.json"],
            "anomaly": data / "anomaly_merged_v4" / "analysis_v3" / "anomaly_summary.json",
            "mass_resolution": data / "mass_resolution" / "analysis_holm" / "s7_mass_resolution.json",
            # analysis_v3 supersedes analysis_labelled (2026-09-28): the same registered
            # analysis on the fits redone in an orthonormal basis (fit_v3), whose own
            # diagnostic puts every fit at its minimum. The first run's fits stopped short
            # (fit_convergence_check/, fit_v2_diagnostic/); the verdicts are unchanged.
            "real_data": data / "aoj_full_v1" / "analysis_v3" / "aoj_top.json",
            # The per-seed cells of the |V_cb| window probe, the file its analysis
            # read: they carry the surviving background counts the analysis drops.
            "vcb": data / "probe_ladder_vcbwindow" / "sall.json",
            "probe_code": root / "experiments" / "EVAL" / "probe.py",
            "design_spec": root / "experiments" / "MTX" / "k8s" / "job-mtx-l188-s1-raunav.yaml",
            "design_arch": root / "experiments" / "E1" / "ParT_sophon_arch_10c.py",
            "design_arm": root / "configs" / "arms" / "L188.yaml",
            "literature": data / "literature_facts.json",
            "ft_recipes": data / "ft_recipes" / "recipes_w2b_bench_v2.json",
            "ft_leg_specs": sorted((root / "experiments" / "FT" / "k8s").glob("job-ft-legs-w[23]*-raunav.yaml")),
            "ft_bench_specs": sorted((root / "experiments" / "FT" / "k8s").glob("job-ft-legs-bench-v2-*-raunav.yaml")),
            "mass_specs": sorted((root / "experiments" / "MTX" / "k8s").glob("job-mtx-*_mass-s[1-5]-raunav.yaml")),
            "survival": maps / "usecase_survival.v1.json",
            "rung_map": maps / "rung_label_maps.v1.csv"}


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
        p = root / rel
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
    a, b = (float(x) for x in one(r"a, b = int\(([0-9.]+) \* n\), int\(([0-9.]+) \* n\)"))
    for name, frac in (("Train", a), ("Val", b - a), ("Test", 1 - b)):
        em.macro("ProbeSplit" + name, f"{100 * frac:.0f}\\%", path, "make_splits",
                 "share of the probe sample; the same jets for every model")
    em.macro("ProbeMlpHidden", one(r"torch\.nn\.Linear\(tr\.shape\[1\], (\d+)\)"), path,
             "_fit_mlp", "hidden width of the MLP probe")
    seeds = [x for x in one(r"^MLP_SEEDS = \(([^)]*)\)").split(",") if x.strip()]
    em.macro("ProbeMlpInits", str(len(seeds)), path, "MLP_SEEDS",
             "MLP initialisations whose scores are averaged per cell")
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


def emit_usecase(em: Emitter, surv: dict, sizes: dict, src: pathlib.Path) -> None:
    for disc, row in surv.items():
        key = texname(disc)
        dies = row["dies_at"]
        em.macro("UseDiesAt" + key, "survives" if dies is None else str(sizes[dies]), src,
                 f"{disc}.dies_at", "first vocabulary size at which it is not constructible")
        alive = row["last_rung_alive"]
        em.macro("UseLastAlive" + key, "none" if alive is None else str(sizes[alive]), src,
                 f"{disc}.last_rung_alive")
        em.macro("UseVectors" + key, str(row["n_coefficient_vectors"]), src,
                 f"{disc}.n_coefficient_vectors")


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


# ------------------------------------------------------------------ later results
#
# Everything below reads a result file written after the probe ladder and
# applies the same rules: the mean and SD over the per-seed rows it stores, never
# a test. The note on each macro says what it aggregates.

def pick(d, path: str):
    """The value at a dotted path, `[i]` for list indices: 'a.b[0].c'."""
    for part in re.findall(r"\[\d+\]|[^.\[\]]+", path):
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
        em.macro(f["macro"], f["value"], path, f"facts[{i}].value",
                 f"{f['source']} {f['location']}")


def emit_mass_lambda(em: Emitter, specs: list) -> None:
    """The mass-loss weight, read from every mass-output pretraining job, which must agree."""
    vals = {m for p in specs for m in re.findall(r"--mass-lambda\s+([0-9.]+)", p.read_text())}
    if len(specs) != 10 or len(vals) != 1:
        raise SystemExit(f"FATAL: expected one --mass-lambda over ten mass-output specs, found "
                         f"{sorted(vals)} over {len(specs)}")
    em.macro("DesignMassLambda", fmt(float(vals.pop()), 1), specs[0], "--mass-lambda",
             "identical in all ten mass-output pretraining specs")


def ft_recipe(R: dict, leg_specs: list, bench_specs: list) -> dict:
    """The fine-tuning settings, from what each run recorded, checked against the commands.

    Every field the table quotes must take ONE value within its group, or this
    stops. Three facts are not in the run manifests and are read from the
    committed job commands instead, which must all agree: the optimizer and
    mixed precision, the learning-rate schedule of the JetClass-II/JetClass waves
    (no --lr-scheduler flag, so weaver's default), and their validation size.
    The manifests also record head_lr_mult=50 for the from-scratch runs, but
    their commands pass no multiplier (LR=5e-4; MULT=()) and weaver's logs show
    none was applied, so the from-scratch row is taken from the commands."""
    one = {}

    def put(key, v):
        one.setdefault(key, set()).add(v)
    for r in R["runs"]:
        fam = "jetclass" if r["leg"] in ("1", "2") else "bench"
        kind = "scratch" if r["init"] == "scratch" else "pretrained"
        put((fam, kind, "lr"), r["lr"])
        put((fam, "weight_decay"), r["weight_decay"])
        put((fam, "batch_size"), r["batch_size"])
        put((fam, "lr_schedule"), r["lr_schedule"])
        put((fam, "epochs", int(r["n_train"]) if fam == "jetclass" else "all"), r["epochs"])
        if kind == "pretrained":
            put((fam, "head_lr_mult"), r["head_lr_mult"])
        if fam == "bench":
            put(("bench", "val", int(r["n_train"]) >= 100_000), r["samples_per_epoch_val"])
    bad = {k: sorted(map(str, v)) for k, v in one.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"FATAL: fine-tuning runs disagree within a group: {bad}")
    v = {k: next(iter(s)) for k, s in one.items()}
    for f in leg_specs + bench_specs:
        t = f.read_text()
        for need in ("--use-amp", "--optimizer ranger", "LR=5e-4; MULT=()", "LR=1e-4"):
            if need not in t:
                raise SystemExit(f"FATAL: {f} lacks {need!r}")
    for f in leg_specs:
        t = f.read_text()
        if "--lr-scheduler" in t or set(re.findall(r"--samples-per-epoch-val (\S+)", t)) != {"20000"}:
            raise SystemExit(f"FATAL: {f} is not the JetClass recipe the table states")
    for f in bench_specs:
        if "--lr-scheduler none" not in f.read_text():
            raise SystemExit(f"FATAL: {f} does not run the benchmarks at a constant rate")
    if (v[("jetclass", "lr_schedule")], v[("bench", "lr_schedule")]) != (None, "constant"):
        raise SystemExit("FATAL: recorded schedules are not (weaver default, constant)")
    for fam in ("jetclass", "bench"):
        if v[(fam, "pretrained", "lr")] != v[("jetclass", "pretrained", "lr")] or \
                v[(fam, "scratch", "lr")] != v[("jetclass", "scratch", "lr")] or \
                v[(fam, "head_lr_mult")] != v[("jetclass", "head_lr_mult")] or \
                v[(fam, "weight_decay")] != v[("jetclass", "weight_decay")] or \
                v[(fam, "batch_size")] != v[("jetclass", "batch_size")]:
            raise SystemExit("FATAL: the two families of fine-tuning differ in a shared setting")
    return {"n_runs": len(R["runs"]), "lr": v[("jetclass", "pretrained", "lr")],
            "head_mult": v[("jetclass", "head_lr_mult")], "lr_scratch": v[("jetclass", "scratch", "lr")],
            "weight_decay": v[("jetclass", "weight_decay")], "batch": v[("jetclass", "batch_size")],
            "epochs_jetclass": {n: v[("jetclass", "epochs", n)] for n in (1_000, 10_000, 100_000, 1_000_000)},
            "epochs_bench": v[("bench", "epochs", "all")],
            "val_bench_small": v[("bench", "val", False)], "val_bench_large": v[("bench", "val", True)],
            "val_jetclass": "20000"}


def emit_ft_recipe(em: Emitter, rec: dict, src: pathlib.Path) -> None:
    em.macro("FtRecipeNRuns", fmt_int(rec["n_runs"]), src, "runs (count)")
    em.macro("FtRecipeLr", fmt_sci(rec["lr"]), src, "runs[*].lr, pretrained starts")
    em.macro("FtRecipeHeadMult", rec["head_mult"], src, "runs[*].head_lr_mult, pretrained starts")
    em.macro("FtRecipeLrHead", fmt_sci(float(rec["lr"]) * float(rec["head_mult"])), src,
             "lr x head_lr_mult")
    em.macro("FtRecipeLrScratch", fmt_sci(rec["lr_scratch"]), src, "runs[*].lr, from scratch")
    em.macro("FtRecipeWeightDecay", rec["weight_decay"], src, "runs[*].weight_decay")
    em.macro("FtRecipeBatch", rec["batch"], src, "runs[*].batch_size")
    for n, e in rec["epochs_jetclass"].items():
        em.macro("FtRecipeEpochsE" + texname(f"{len(str(n)) - 1}"), e, src,
                 f"runs[leg 1,2; n_train {n}].epochs")
    em.macro("FtRecipeEpochsBench", rec["epochs_bench"], src, "runs[bench].epochs")


def table_ft_recipe(rec: dict) -> str:
    ep = rec["epochs_jetclass"]
    body = ["& JetClass-II and JetClass & top tagging and quark/gluon \\\\", "\\midrule",
            f"optimizer & \\multicolumn{{2}}{{l}}{{Ranger, mixed precision, batch size {rec['batch']}, "
            f"weight decay {rec['weight_decay']}}} \\\\",
            f"learning rate, pretrained start & \\multicolumn{{2}}{{l}}{{{fmt_sci(rec['lr'])} "
            f"(trunk), {fmt_sci(float(rec['lr']) * float(rec['head_mult']))} (new head)}} \\\\",
            f"learning rate, from scratch & \\multicolumn{{2}}{{l}}{{{fmt_sci(rec['lr_scratch'])} "
            f"(all parameters)}} \\\\",
            "schedule & pretrained start: trunk at a constant rate, new head flat then decayed "
            "to 1\\% over the last 30\\% of epochs (weaver default with a head multiplier); "
            "from scratch: all parameters flat then decayed & constant \\\\",
            "epochs & " + ", ".join(str(ep[n]) for n in sorted(ep))
            + " at " + ", ".join(fmt_n_jets(f"N{n}") for n in sorted(ep))
            + f" jets & {rec['epochs_bench']} at every size \\\\",
            f"validation jets & {fmt_int(rec['val_jetclass'])} & {fmt_int(rec['val_bench_small'])} "
            f"below {fmt_n_jets('N100000')}, {fmt_int(rec['val_bench_large'])} from it \\\\",
            "checkpoint & \\multicolumn{2}{l}{best validation accuracy} \\\\"]
    caption = (f"Fine-tuning settings, as the {fmt_int(rec['n_runs'])} fine-tuning runs behind the "
               "reported results recorded them, checked against their job commands. The head is "
               "freshly initialised in every run; a pretrained start loads every other weight.")
    return _table(body, caption, "tab:ftrecipe", "l p{0.42\\linewidth} l", [])


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

    m = re.search(r"\(jet_pt > (\d+)\) & \(jet_pt < (\d+)\) & \(jet_sdmass > (\d+)\) & "
                  r"\(jet_sdmass < (\d+)\)", arm.read_text())
    if not m:
        raise SystemExit(f"FATAL: {arm} has no selection line of the expected form")
    for name, v in zip(("DesignPtMin", "DesignPtMax", "DesignMsdMin", "DesignMsdMax"), m.groups()):
        em.macro(name, fmt_int(v), arm, "selection", "GeV")


def recovery_acc(R: dict) -> dict:
    """{(rung, model level): per-seed balanced accuracies of the linear probe}."""
    out = {}
    for r in sorted(R["table"], key=lambda r: r["seed"]):
        if r["probe"] == "linear":
            out.setdefault((r["rung"], r["level"]), []).append(r["accuracy"])
    return out


def emit_recovery(em: Emitter, R: dict, src: pathlib.Path, sizes: dict) -> None:
    """Label recovery at every level of the tree, linear probe, from the per-seed rows."""
    acc = recovery_acc(R)
    levels = sorted({lv for _, lv in acc}, reverse=True)
    for rung in RUNGS:
        for lv in levels:
            a = acc[(rung, lv)]
            em.macro("RecoveryAcc" + texname(sizes[rung]) + "By" + texname(lv),
                     fmt_pm(np.mean(a), np.std(a, ddof=1)), src,
                     f"table[rung={rung},level={lv},probe=linear].accuracy",
                     f"balanced accuracy, mean +- SD over {len(a)} seeds")
    # The pairs the text quotes: 188 against 162, which never differ, and 162
    # against 17, whose advantage decays to zero at the 17-class level.
    gap = lambda rung, fine, coarse: np.mean(acc[(rung, fine)]) - np.mean(acc[(rung, coarse)])
    for fine, coarse in ((188, 162), (162, 17)):
        for rung in RUNGS:
            em.macro("RecoveryAdv" + texname(f"{fine} over {coarse}", "at", rung),
                     fmt(gap(rung, fine, coarse), 4, sign=True), src,
                     f"table[rung={rung},level={fine}/{coarse},probe=linear].accuracy",
                     "finer minus coarser model, difference of the seed means")
    em.macro("Recovery" + texname(188, 162) + "MaxAbs",
             fmt(max(abs(gap(r, 188, 162)) for r in RUNGS), 4), src,
             "table[rung=*,level=188/162,probe=linear].accuracy",
             "max |difference of the seed means| over the tree")
    # The jets the linear probe is fit and scored on, as label_recovery.py
    # recorded them in every cell: it fits on a random subset of the training
    # split (n_fit) to match the MLP's training fraction, not on all of it.
    files = [em.root / x["path"] for x in R["provenance"]["inputs"]]
    per = [json.loads(p.read_text()) for p in files]
    n_fit = {c["n_fit"] for d in per for a in d["arms"].values() for c in a["rungs"].values()
             if "n_fit" in c}
    n_test = {d["n_test"] for d in per}
    if len(n_fit) != 1 or len(n_test) != 1:
        raise SystemExit(f"FATAL: the label-recovery files disagree on the probe's jets: "
                         f"fit {sorted(n_fit)}, test {sorted(n_test)}")
    em.macro("RecoveryNTrain", fmt_int(n_fit.pop()), files[0], "arms.*.rungs.*.n_fit",
             f"jets the linear probe is fit on, recorded in every cell of all {len(files)} "
             "per-seed files")
    em.macro("RecoveryNTest", fmt_int(n_test.pop()), files[0], "n_test",
             f"jets the probes are scored on, recorded in all {len(files)} per-seed files")


def emit_random_control(em: Emitter, C: dict, src: pathlib.Path) -> None:
    """The random-label control: 1-AUC of each draw, one run each, and over the draws."""
    em.macro("RandNDraws", words(len({r["draw"] for r in C["table"]})), src, "table[*].draw",
             "random partitions drawn")
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
    return rows + [("RandDrawOne", "random-label control, draw 1", ["rand-d1-s1b"])]


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
                             else f"mean +- SD over {len(inits)} pretraining seeds")
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
        em.macro("FtNTestAuc" + ds, fmt_int(ft_n_test(c, rows, "n_jets_auc")), src,
                 "cells.*.*.s1.n_jets_auc", "test jets the macro AUC is computed on")
        em.macro("FtNTestAcc" + ds, fmt_int(ft_n_test(c, rows, "n_jets")), src,
                 "cells.*.*.s1.n_jets", "test jets the accuracy is computed on")
    F = next(iter(ft.values()))
    for n in ft_sizes(F["cells"], rows):
        em.macro("FtSize" + n_tag(n), fmt_n_jets(n), F["path"], f"cells.*.{n}",
                 "fine-tuning training jets")


ANOMALY_FAMILIES = ("mahalanobis", "knn")     # feature-based; the class sum waits for its rerun


def anomaly_signal_key(sig: str) -> str:
    """`label_X_YY_bbb` -> `XYYBbb`, the macro-name part for a signal."""
    return texname(sig.removeprefix("label_"))


def anomaly_cells(S: dict, fam: str, sig: str) -> dict:
    """Per label set at the primary injection: sigma_min and max SIC per seed."""
    lv = S["families"][fam][sig][S["conventions"]["primary_injection"]]["levels"]
    return {int(k): {"sigma_min": np.exp(v["ln_sigma_min"]), "max_sic": np.array(v["max_sic"])}
            for k, v in lv.items()}


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
    nd = S["not_detected_rule"]
    em.macro("AnomalyNdThreshold", fmt(nd["threshold_max_sic"], 1), src,
             "not_detected_rule.threshold_max_sic", "max SIC below this at every label set")
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


def emit_mass_output(em: Emitter, A: dict, a_src: pathlib.Path, files: list, sizes: dict) -> None:
    """1-AUC on b vs c two-prong with and without the mass output. The plain
    models come from the ladder table, the +mass models from their own probe
    files, which must have scored the same jets in the same order."""
    task = "bvc_resonant"
    docs = [json.loads(p.read_text()) for p in files]
    if {d["row_alignment_sha256"] for d in docs} != {A["provenance"]["row_alignment_sha256"]}:
        raise SystemExit("FATAL: the mass-output probe files were not scored on the ladder's jets")
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


def emit_real_data(em: Emitter, J: dict, src: pathlib.Path) -> None:
    """Section 6, the top peak in CMS open data (AspenOpenJets), 1% data efficiency."""
    r = J["secondary"]["real_data_top"]
    base = "secondary.real_data_top"
    res_path = em.root / J["provenance"]["input"]
    res = json.loads(res_path.read_text())
    em.macro("AojNJetsFit", fmt_int(res["n_jets"]), res_path, "n_jets",
             "jets in the fit region, all files")
    em.macro("AojNToys", str(res["n_toys"]), res_path, "n_toys")
    fits = {"reference": res["reference"]["top"], **{m: v["top"] for m, v in res["models"].items()}}
    em.macro("AojNConverged", of(sum(bool(f["converged"]) for f in fits.values()), len(fits)),
             res_path, "reference.top.converged, models.*.top.converged",
             "fits whose minimiser reported success (scipy L-BFGS-B), reference included")
    em.macro("AojRefValidationToyP", fmt_p(res["reference"]["top"]["validation"]["toy_p"]), res_path,
             "reference.top.validation.toy_p", "background-only fit in the fail-region band")
    worst = min(res["models"], key=lambda m: res["models"][m]["top"]["signal_yield"])
    w = res["models"][worst]["top"]
    em.macro("AojWorstYield", fmt_int(w["signal_yield"]), res_path, f"models.{worst}.top.signal_yield")
    em.macro("AojWorstYieldErr", fmt_int(w["signal_yield_err"]), res_path,
             f"models.{worst}.top.signal_yield_err")
    em.macro("AojWorstMean", fmt(w["floated_mean"], 0), res_path, f"models.{worst}.top.floated_mean",
             "GeV, the peak position fitted with the shape floating")
    em.macro("AojWorstZ", fmt(w["signal_yield"] / w["signal_yield_err"], 1), res_path,
             f"models.{worst}.top.signal_yield / signal_yield_err", "standard deviations from zero")
    diag_path = res_path.parent / "diagnostic.json"
    if diag_path.exists():
        D = json.loads(diag_path.read_text())
        s = D["summary"]
        if pathlib.Path(D["results"]).name != "results.json" or not s["n_as_run_reproduced"] == s["n_fits"]:
            raise SystemExit(f"FATAL: {diag_path} does not reproduce every fit it checks")
        em.macro("AojFitNChecked", str(s["n_fits"]), diag_path, "summary.n_fits",
                 "fits redone from their bins by fit_minimum_diagnostic.py")
        em.macro("AojFitMaxNewtonShift", fmt_sci(float(f'{s["max_abs_newton_yield_shift_over_err"]:.1g}')), diag_path,
                 "summary.max_abs_newton_yield_shift_over_err",
                 "largest yield change, in units of its error, when Newton steps continue from the fit")
        em.macro("AojFitProfileAgree", fmt_sci(float(f'{max(abs(x - 1) for x in s["profile_over_as_run_err_range"]):.1g}')),
                 diag_path, "summary.profile_over_as_run_err_range",
                 "largest relative difference between a quoted error and its profile-likelihood error")
    em.macro("AojTrendP", fmt_p(r["trend"]["p"]), src, f"{base}.trend.p", "Holm table of two")
    em.macro("AojEquivFactor", fmt_factor(r["clause3_equivalence"]["smallest_bound_passed_by_all"], 2),
             src, f"{base}.clause3_equivalence.smallest_bound_passed_by_all", "a factor in yield")
    for i, cl in enumerate(r["clauses"]):
        em.macro("AojClause" + texname(cl["n"]) + "Verdict", tex(cl["verdict"]), src,
                 f"{base}.clauses[{i}].verdict", cl["text"])
    for lv in ("162", "17"):
        g = r["mass_output_2x2"][f"gain_{lv}"]
        em.macro("AojMassGain" + texname(lv), fmt(g["mean_diff"], 3, sign=True), src,
                 f"{base}.mass_output_2x2.gain_{lv}.mean_diff", "with minus without, -ln(yield)")
        em.macro("AojMassGainP" + texname(lv), fmt_p(g["p"]), src,
                 f"{base}.mass_output_2x2.gain_{lv}.p")
    d = r["mass_output_2x2"]["difference_in_differences"]
    em.macro("AojMassDid", fmt(d["mean_diff"], 3, sign=True), src,
             f"{base}.mass_output_2x2.difference_in_differences.mean_diff")
    em.macro("AojMassDidP", fmt_p(d["p"]), src, f"{base}.mass_output_2x2.difference_in_differences.p")
    for i, p in enumerate(r["pairwise_exploratory"]):
        if (p["fine"], p["coarse"]) == (188, 17):
            em.macro("Aoj" + texname(188, "vs", 17) + "Diff", fmt(p["mean_diff"], 3, sign=True), src,
                     f"{base}.pairwise_exploratory[{i}].mean_diff", "-ln(yield), 17 minus 188")
            em.macro("Aoj" + texname(188, "vs", 17) + "P", fmt_p(p["p"]), src,
                     f"{base}.pairwise_exploratory[{i}].p")
    for lv in ("188", "162", "43", "17", "162+mass", "17+mass"):
        rows = [x for x in J["table"] if str(x["level"]) == lv]
        y = np.array([x["signal_yield"] for x in rows])
        k = texname(lv.replace("+mass", " mass"))
        em.macro("AojYield" + k, fmt_int(np.exp(np.mean(np.log(y)))), src,
                 f"table[level={lv}].signal_yield", f"geometric mean over {len(y)} seeds")
        em.macro("AojYieldMin" + k, fmt_int(y.min()), src, f"table[level={lv}].signal_yield (min)")
        em.macro("AojYieldMax" + k, fmt_int(y.max()), src, f"table[level={lv}].signal_yield (max)")
    ref = J["reference"]
    em.macro("AojRefYield", fmt_int(ref["signal_yield"]), src, "reference.signal_yield",
             "shipped CMS ParticleNet top score through the same fit")
    em.macro("AojRefYieldErr", fmt_int(ref["signal_yield_err"]), src, "reference.signal_yield_err")
    em.macro("AojRefMass", fmt(ref["mean"], 1), src, "reference.mean", "GeV")
    em.macro("AojRefZ", fmt(ref["z_wald"], 0), src, "reference.z_wald")
    pub = J["reference_models"]["sophon-public"]
    em.macro("AojPublicYield", fmt_int(pub["signal_yield"]), src,
             "reference_models.sophon-public.signal_yield")
    ok = sum(1 for x in J["table"] if all(x["criteria"].values()))
    em.macro("AojNPass", of(ok, len(J["table"])), src, "table[*].criteria",
             "models passing all four peak criteria")
    rel = np.median([x["signal_yield_err"] / x["signal_yield"] for x in J["table"]])
    eff = np.median([x["efficiency_relative_to_reference"] for x in J["table"]])
    em.macro("AojMedianRelEff", f"{eff * 100:.0f}\\%", src,
             "table[*].efficiency_relative_to_reference", "median over models: yield / CMS tagger's yield")
    em.macro("AojMedianFitErr", f"{rel * 100:.0f}\\%", src,
             "table[*].signal_yield_err / signal_yield", "median over models")


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
               f"{words(min(n_seeds))} pretraining seeds. Rows are the pretraining label-set size, "
               f"finest first; lower granularity is further down.")
    cells = " ".join(auc_all + rej_all)
    notes = []
    if "$>$" in cells:
        notes.append("$>$ at most one background jet passed the cut in every seed; the entry is "
                     "the 95\\% confidence lower limit on the rejection, $N_B/3.0$ when none passed "
                     "and $N_B/4.74$ when one did, with $N_B$ the number of background test jets.")
    if "$\\geq$" in cells:
        notes.append("$\\geq$ with $^{\\ast}$: in some seeds at most one background jet passed "
                     "the cut; the entry is the median over seeds with those seeds at $N_B$, not a "
                     "measured value.")
    if "dagger" in cells:
        notes.append("$^{\\dagger}$ the AUC reached 1 at the resolution of the sample in at least "
                     "one seed; $1-$AUC is then an upper bound and the cell is not a measurement.")
    if nbkg:
        notes.insert(0, "Background test jets $N_B$ per task: " + "; ".join(
            f"{TASK_LABELS.get(t, tex(t))}, {fmt_int(nbkg[t])}" for t in tasks if t in nbkg) + ".")
    return _table(head + body, caption, f"tab:probes-{probe}",
                  "r l " + "r" * len(tasks), notes, wide=True)


def table_usecase(surv: dict, sizes: dict) -> str:
    """T3: which published discriminant is still constructible at each vocabulary."""
    cols = [sizes[r] for r in RUNGS]
    head = ["discriminant & source & " + " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    rows = []
    # A row absent at EVERY vocabulary is not a coarsening result, and the table
    # is actively misleading if it renders identically to one: a reader would
    # conclude that a finer vocabulary would have bought the discriminant. It
    # would not -- the quantity is not expressible over the native classes at
    # all. Marked with a dagger and explained in the notes.
    never = [d for d, r in surv.items()
             if not r.get("expressible_in_native_vocabulary", True)]
    for disc, row in surv.items():
        marks = " & ".join("$\\bullet$" if row["constructible"][r] else "---" for r in RUNGS)
        title = tex(row["title"]) + ("$^{\\dagger}$" if disc in never else "")
        rows.append(f"{title} & {tex(row['source'])} & {marks} \\\\")
    caption = ("Published Sophon-family discriminants against pretraining vocabulary size "
               "(columns, finest first). $\\bullet$ = the discriminant is exactly constructible "
               "from that vocabulary's output nodes; --- = it is not. The criterion is Sophon's "
               "class-division property: a discriminant survives a merge only if both of its "
               "coefficient vectors are constant on every merged group, so this is a statement "
               "about the partition and needs no training and no data.")
    notes = ["Column headings are the number of classes in the vocabulary, counted from "
             "\\texttt{configs/labelmaps/rung\\_label\\_maps.v1.csv}; they are not the numbers "
             "in the internal rung names, which count resonant groups only."]
    if never:
        notes.append(
            "$^{\\dagger}$ Absent for a different reason from every other row: not lost to "
            "coarsening, but never expressible. The vocabulary labels jets by their decay "
            "products and not by the parent resonance, so wherever the $W$ and the $Z$ share a "
            "decay mode they share a class, and no sum of a head's outputs separates them --- at "
            "the finest vocabulary as much as at the coarsest. A finer label set of this kind "
            "would not buy this discriminant; a differently organised one would.")
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
FAMILY_LABELS = {"class_sum": "class sum", "mahalanobis": "Mahalanobis",
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
               f"{words(max(len(i) for *_, i in rows))} pretraining seeds, each fine-tuned once; "
               "a row with fewer models gives their number. Rows are the pretraining label set, "
               "columns the number of fine-tuning training jets.")
    return _table(head + body, caption, "tab:finetune" if auc else "tab:finetune-accuracy",
                  "l " + "r" * len(ns), [])


def table_anomaly(S: dict) -> str:
    """Anomaly detection: sigma_min and max SIC per signal, detector and label set.
    Signals no feature-based detector sees at any label set are listed in a note."""
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
    caption = (f"Anomaly detection with detectors on the frozen features, "
               f"{fmt_int(S['conventions']['primary_injection'])} signal jets injected: "
               "$\\sigma_{\\min}$, the smallest initial significance from which a $5\\sigma$ "
               "discovery is still reached (lower is more sensitive), and the maximum significance "
               f"improvement (max SIC). Mean {tex('±')} standard deviation over the five pretraining "
               "seeds of each seed's median over "
               f"{words(S['provenance']['resamplings_per_seed'])} resamplings of the background "
               "and signal samples.")
    notes = ["Not detected by either detector at any label set (max SIC below "
             f"{fmt(S['not_detected_rule']['threshold_max_sic'], 1)} at every label set): "
             + ", ".join(SIGNAL_LABELS[g] for g in light) + "."] if light else []
    return _table(head + body, caption, "tab:anomaly", "l l " + "r" * len(levels), notes)


def table_random_control(C: dict, A: dict, probes=("linear",)) -> str:
    """The random-label control beside the four pretrained label sets, in 1-AUC.

    Linear probe only for now: the MLP rows are one more entry in `probes` once
    their rerun lands.
    """
    levels = A["levels_fine_to_coarse"]
    draws = sorted({r["draw"] for r in C["table"]})
    tasks = [t for t in TASK_LABELS if any(r["task"] == t for r in C["table"])]
    head = [f"& \\multicolumn{{{len(levels)}}}{{c}}{{pretraining label set}} & "
            f"\\multicolumn{{{len(draws) + 1}}}{{c}}{{random {levels[-1]}-group control}} \\\\",
            "task & " + " & ".join(str(lv) for lv in levels) + " & "
            + " & ".join(f"draw {d}" for d in draws) + f" & mean {tex('±')} SD \\\\", "\\midrule"]
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
    caption = (f"The random-label control, frozen {which} probe: $1-$AUC in units of $10^{{-3}}$ "
               f"(lower is better) on the two tasks it was built for. The pretrained label sets are "
               f"the mean {tex('±')} standard deviation over the {words(min(n_seeds))} pretraining "
               f"seeds; each draw of the control is one run, and the last column is the mean "
               f"{tex('±')} standard deviation over the {words(len(draws))} draws. Each draw permutes "
               f"the resonant classes within the two-prong and the three-/four-prong strata and cuts "
               f"them into groups matching the training-stream share of each {levels[-1]}-class "
               f"group; the QCD class is kept.")
    return _table(head + body, caption, "tab:random-control",
                  "l " + "r" * (len(levels) + len(draws) + 1), [], wide=True)


def table_recovery(R: dict, sizes: dict) -> str:
    """Balanced accuracy recovering each level of the tree from each model."""
    acc = recovery_acc(R)
    levels = sorted({lv for _, lv in acc}, reverse=True)
    head = ["read out at & " + " & ".join(f"{lv}-class model" for lv in levels) + " \\\\",
            "\\midrule"]
    body = []
    for rung in RUNGS:
        cells = [fmt_pm(np.mean(acc[(rung, lv)]), np.std(acc[(rung, lv)], ddof=1)) for lv in levels]
        body.append(f"{sizes[rung]} classes & " + " & ".join(cells) + " \\\\")
    caption = ("Label recovery: balanced accuracy of a frozen linear probe trained to recover "
               "each level of the label tree (rows, finest first) from each pretrained model "
               f"(columns), mean $\\pm$ standard deviation over the "
               f"{words(len(acc[(RUNGS[0], levels[0])]))} pretraining seeds.")
    return _table(head + body, caption, "tab:recovery", "l " + "r" * len(levels), [])


def table_mass(M: dict, root: pathlib.Path) -> str:
    """Jet-mass resolution from frozen features, both probes."""
    tgt = {round(r["target_sigma_eff"], 12) for r in M["table"] if r["probe"] == "ridge"}
    tgt = fmt(tgt.pop(), 4) if len(tgt) == 1 else "---"
    head = ["& " + " & ".join(c.replace("+mass", " + mass") for c in MASS_CELLS)
            + " & class mean only \\\\", "\\midrule"]
    body = []
    for probe, name in (("mlp", "nonlinear (MLP) probe"), ("ridge", "linear (ridge) probe")):
        cells = [fmt_pm(np.mean(v), np.std(v, ddof=1))
                 for v in (mass_values(M, c, probe) for c in MASS_CELLS)]
        body.append(f"{name} & " + " & ".join(cells) + f" & {tgt} \\\\")
    caption = ("Jet-mass regression from frozen features: $\\sigma_{\\mathrm{eff}}$ of the residual. "
               "The residual is $\\ln(m_{\\mathrm{pred}}/m_{\\mathrm{true}})$ after removing each "
               "native class's training-set mean; $\\sigma_{\\mathrm{eff}}$ is half the smallest "
               f"interval holding 68\\% of it. Mean {tex('±')} standard deviation over the "
               f"{words(len(mass_values(M, MASS_CELLS[0], 'ridge')))} pretraining seeds, on "
               f"{fmt_int(mass_n_test(M, root)[0])} test jets. Columns are the pretraining label set, "
               "with or without the added mass output; the last is a predictor that returns the "
               "class mean alone, the same for both probes.")
    return _table(head + body, caption, "tab:mass", "l " + "r" * (len(MASS_CELLS) + 1), [],
                  wide=True)


def table_realdata(J: dict) -> str:
    """Section 6: fitted top yield at 1% data efficiency in CMS open data."""
    groups = ["188", "162", "43", "17", "162+mass", "17+mass"]
    head = ["label set & geometric-mean yield & range over seeds \\\\", "\\midrule"]
    body = []
    for g in groups:
        y = np.array([x["signal_yield"] for x in J["table"] if str(x["level"]) == g])
        body.append(f"{g.replace('+mass', ' + mass')} & {fmt_int(np.exp(np.mean(np.log(y))))} & "
                    f"{fmt_int(y.min())}--{fmt_int(y.max())} \\\\")
    pub = J["reference_models"]["sophon-public"]
    body.append("\\addlinespace")
    body.append(f"published 188-class checkpoint & {fmt_int(pub['signal_yield'])} "
                f"$\\pm$ {fmt_int(pub['signal_yield_err'])} & --- \\\\")
    ref = J["reference"]
    body.append(f"CMS ParticleNet (shipped) & {fmt_int(ref['signal_yield'])} $\\pm$ "
                f"{fmt_int(ref['signal_yield_err'])} & --- \\\\")
    caption = ("Top quarks found in CMS open data with no fine-tuning: fitted top-peak yield at 1\\% "
               "data efficiency, from one simultaneous pass/fail fit over all files, per pretrained "
               "model. Geometric mean and range over five pretraining seeds.")
    return _table(head + body, caption, "tab:realdata", "l r r", [])


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
        emit_usecase(em, surv, sizes, paths["survival"])
        out["tables/usecase_survival.tex"] = table_usecase(surv, sizes)
    else:
        missing.append(f"use-case survival -- {paths['survival']}")
        skipped.append("T3 use-case survival: configs/labelmaps/usecase_survival.v1.json missing")

    if all(k in have for k in ("design_spec", "design_arch", "design_arm")):
        emit_training_design(em, paths["design_spec"], paths["design_arch"], paths["design_arm"])
    if "literature" in have:
        emit_literature(em, paths["literature"])
    if "mass_specs" in have and paths["mass_specs"]:
        emit_mass_lambda(em, paths["mass_specs"])
    if all(k in have for k in ("ft_recipes", "ft_leg_specs", "ft_bench_specs")) \
            and paths["ft_leg_specs"] and paths["ft_bench_specs"]:
        rec = ft_recipe(json.loads(paths["ft_recipes"].read_text()),
                        paths["ft_leg_specs"], paths["ft_bench_specs"])
        emit_ft_recipe(em, rec, paths["ft_recipes"])
        out["tables/finetune_recipe.tex"] = table_ft_recipe(rec)
    later = (("recovery", lambda em, d, src: emit_recovery(em, d, src, sizes),
              lambda d: table_recovery(d, sizes), "label_recovery"),
             ("random_control", emit_random_control, lambda d: table_random_control(d, A),
              "random_control"),
             ("anomaly", emit_anomaly, table_anomaly, "anomaly"),
             ("mass_resolution", emit_mass_resolution, lambda d: table_mass(d, root), "mass"),
             ("real_data", emit_real_data, table_realdata, "realdata"))
    for key, emit, table, name in later:
        if key not in have:
            missing.append(f"{key} -- {paths[key]}")
            continue
        check_inputs_unchanged(paths[key], root)
        d = json.loads(pathlib.Path(paths[key]).read_text())
        emit(em, d, paths[key])
        out[f"tables/{name}.tex"] = table(d)

    if "mass2x2" in have:
        emit_mass_output(em, A, paths["analysis"], paths["mass2x2"], sizes)
    else:
        missing.append("mass-output probes -- experiments/FIGS/data/probe_ladder_mass2x2/s*.json")
    if "ft_legs" in have:
        ft = ft_load(paths["ft_legs"])
        emit_finetune(em, ft, sizes)
        out["tables/finetune.tex"] = table_finetune(ft, sizes, "macro_auc_ovr")
        out["tables/finetune_accuracy.tex"] = table_finetune(ft, sizes, "accuracy")
    else:
        missing.append(f"fine-tuning metrics -- {', '.join(map(str, paths['ft_legs']))}")
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
        print("\nMISSING INPUTS (no placeholder was written for any of these):")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
