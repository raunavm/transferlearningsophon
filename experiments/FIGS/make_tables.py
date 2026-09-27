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
  * It never recomputes a test. Statistics come out of
    `seed_level_results.json` exactly as `experiments/STATS/seed_level.py`
    wrote them; this script formats them and stops.
  * It refuses outright if two inputs disagree on `row_alignment_sha256`, or if
    a ladder file has changed since the analysis that read it. Both mean the
    arms were not scored on the same jets in the same order, and every contrast
    in the paper assumes they were.
  * A background rejection that is a lower bound (no background jet survived
    the cut, so the number is the sample size, not a measurement) is never
    printed as a bare number. `$>$` when every seed is at the cap, `$\\geq$`
    with a footnote when only some are. Same for an AUC that saturated at 1.

Usage:
    python3 experiments/FIGS/make_tables.py            # write paper/journal/
    python3 experiments/FIGS/make_tables.py --check    # CI: non-zero on drift
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    ("C2 and C3 as registered: community benchmarks at the last epoch (amendment A6)",
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


def fmt_rejection(median, n_bound: int, n_seeds: int) -> str:
    """A rejection that is a bound never prints as a bare number.

    `rejection_is_bound` means no background jet passed the cut, so the value is
    1/(0 background) floored at the sample size: a lower bound on the true
    rejection. All seeds at the cap -> the median is a bound too. Only some ->
    the median may or may not be, so it gets the weaker sign and a footnote.
    """
    v = f"{float(median):,.0f}" if float(median).is_integer() else f"{float(median):,.1f}"
    v = v.replace(",", "{,}")
    if n_bound == n_seeds and n_seeds:
        return f"$>${v}"
    if n_bound:
        return f"$\\geq${v}$^{{\\ast}}$"
    return v


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
            "finetune": data / "finetune_s3_s4" / "analysis_v2" / "s3_s4_finetune.json",
            "anomaly": data / "anomaly_merged_v4" / "analysis_v2" / "anomaly_s5.json",
            "mass_resolution": data / "mass_resolution" / "analysis_holm" / "s7_mass_resolution.json",
            "real_data": data / "aoj_full_v1" / "analysis_labelled" / "aoj_top.json",
            "design_spec": root / "experiments" / "MTX" / "k8s" / "job-mtx-l188-s1-raunav.yaml",
            "design_arch": root / "experiments" / "E1" / "ParT_sophon_arch_10c.py",
            "design_arm": root / "configs" / "arms" / "L188.yaml",
            "literature": data / "literature_facts.json",
            "trend_sim": data / "trend_size_sim" / "trend_size_sim.json",
            "trend_holm": data / "trend_size_sim" / "holm_under_simulated_nulls.json",
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


def bound_mark(is_bound: bool) -> str:
    """A contrast between cells where one reached AUC=1 is a bound, not a value."""
    return "$^{\\ast}$" if is_bound else ""


def seed_values(table: list[dict], task: str, probe: str, level: int, field: str) -> list:
    """Per-seed cells straight from the analysis table, in seed order."""
    rows = [r for r in table
            if r["task"] == task and r["probe"] == probe and r["level"] == level
            and not r.get("dropped_pair")]
    return [r[field] for r in sorted(rows, key=lambda r: r["seed"])]


# ------------------------------------------------------------------ macros

def headline_rejection(row: dict) -> dict:
    """The background rejection the paper quotes, at the working point
    docs/PRESPEC_2026-09.md fixed blind: 90 % signal efficiency.

    The flat `rejection_*` fields sit at the probe's DEFAULT working point,
    50 %, where no background jet survives at the three finer vocabularies and
    the number is a statement about the size of the test sample rather than
    about the models. Reading them here would print that censored number as the
    paper's headline. An analysis file written before the working points were
    recorded has no `rejection_points`; it then falls back to the flat fields,
    and `eps` says which point the reader is actually looking at.
    """
    pts = row.get("rejection_points") or {}
    eps = row.get("headline_eps_s")
    if eps in pts:
        p = pts[eps]
        return {"eps": float(eps), "median": p["median"], "range": p["range"],
                "n_bound": p["n_bound"], "n_seeds": p["n_seeds"],
                "path": f".rejection_points['{eps}'].median"}
    return {"eps": float(row["rejection_eps_s"]), "median": row["rejection_median"],
            "range": row["rejection_range"], "n_bound": row["n_rejection_bound"],
            "n_seeds": row["n_seeds"], "path": ".rejection_median"}


def emit_design(em: Emitter, A: dict, src: pathlib.Path) -> None:
    p = A["provenance"]
    em.macro("ProbeNJets", fmt_int(p["n_jets_total"]), src, "provenance.n_jets_total")
    em.macro("ProbeNSeeds", str(len(A["seeds_used"])), src, "seeds_used (length)")
    em.macro("ProbeRowAlign", f"\\texttt{{{p['row_alignment_sha256'][:16]}}}", src,
             "provenance.row_alignment_sha256 (first 16)")
    t0 = sorted(A["levels"])[0]
    h = headline_rejection(A["levels"][t0]["linear"][0])
    em.macro("ProbeEpsS", f"{h['eps'] * 100:.0f}\\%", src,
             f"levels.{t0}.linear[0].headline_eps_s")


def emit_levels(em: Emitter, A: dict, src: pathlib.Path) -> None:
    """One macro per measured number in the per-granularity summary.

    The AUC seed spread is the one quantity the analysis file does not store
    (it stores the spread of the endpoint, log(1-AUC)), so it is computed here
    from the same per-seed rows and cross-checked against the stored mean. A
    disagreement means the table rows and the summary block came from different
    reads, which is worth a crash rather than a rounding argument.
    """
    for task in sorted(A["levels"]):
        for probe in sorted(A["levels"][task]):
            for i, row in enumerate(A["levels"][task][probe]):
                if not row["n_seeds"]:
                    continue
                lv, key = row["level"], texname(task, probe, row["level"])
                jp = f"levels.{task}.{probe}[{i}]"
                aucs = seed_values(A["table"], task, probe, lv, "auc")
                if abs(float(np.mean(aucs)) - row["mean_auc"]) > 1e-12:
                    raise SystemExit(f"FATAL: {jp}.mean_auc disagrees with the mean of the "
                                     f"per-seed table rows for {task}/{probe}/{lv}")
                em.macro("ProbeAuc" + key,
                         fmt_auc(row["mean_auc"], row["n_censored"], row["n_seeds"]), src,
                         jp + ".mean_auc",
                         f"{row['n_seeds']} seeds, {row['n_censored']} saturated at AUC=1")
                em.macro("ProbeAucSd" + key, fmt(np.std(aucs, ddof=1), 5) if len(aucs) > 1
                         else "---", src,
                         f"table[{task},{probe},{lv}].auc (SD over seeds, ddof=1)")
                em.macro("ProbeLogOneMinusAuc" + key, fmt(row["mean"], 4, sign=True), src,
                         jp + ".mean", "natural log of 1-AUC, lower is better")
                em.macro("ProbeLogOneMinusAucSd" + key,
                         fmt(row["seed_sd"], 4) if row["seed_sd"] is not None else "---", src,
                         jp + ".seed_sd")
                h = headline_rejection(row)
                pt = (row.get("rejection_points") or {}).get(row.get("headline_eps_s"))
                if pt and "mean_n_bkg_pass" in pt:
                    em.macro("ProbeBkgLeft" + key, fmt(pt["mean_n_bkg_pass"], 1), src,
                             jp + f".rejection_points['{row['headline_eps_s']}'].mean_n_bkg_pass",
                             "background jets passing the cut, mean over seeds")
                em.macro("ProbeRej" + key,
                         fmt_rejection(h["median"], h["n_bound"], h["n_seeds"]),
                         src, jp + h["path"],
                         f"at {h['eps']:.0%} signal efficiency; "
                         f"{h['n_bound']} of {h['n_seeds']} seeds at the cap")


def emit_mde(em: Emitter, A: dict, src: pathlib.Path) -> None:
    for i, m in enumerate(A.get("mde", [])):
        if not m.get("estimable"):
            continue
        key = texname(m["task"], m["probe"])
        em.macro("Mde" + key, fmt(m["mde"], 4), src, f"mde[{i}].mde",
                 f"{m['n_pairs']} seed pairs, 80% power")
        em.macro("MdeRatio" + key, fmt(m["mde_as_ratio_of_1m_auc"], 3), src,
                 f"mde[{i}].mde_as_ratio_of_1m_auc", "as a factor in 1-AUC")


def holm_verdict(entry: dict | None) -> str:
    """The Holm column, in the words seed_level.py uses for it."""
    if entry is None or entry["status"] == "pending":
        return "pending"
    if entry["reject_whatever_pending"]:
        return "rejected"
    if entry["reject_possible"]:
        return "depends on pending"
    return "not rejected"


def holm_lookup(A: dict) -> dict:
    """Holm rows keyed by their test name, confirmatory and secondary together."""
    out = {}
    for block in ("confirmatory", "secondary"):
        for h in A.get(block, {}).get("holm_family", []):
            out[h["test"]] = h
    return out


def emit_tests(em: Emitter, A: dict, src: pathlib.Path) -> None:
    """Trend, equivalence and sign-agreement numbers, copied, never recomputed."""
    holm = holm_lookup(A)
    for name, path in (("C1", "confirmatory.C1"), ("S1", "secondary.S1")):
        r = (A.get("confirmatory") or {}).get(name) or (A.get("secondary") or {}).get(name)
        if not r or not r.get("run"):
            continue
        key = texname(name)
        em.macro("TrendStat" + key, fmt(r["stat"], 3), src, path + ".stat",
                 f"max-T, {r['method']}, {r['n_blocks']} seed blocks")
        em.macro("TrendP" + key, fmt_p(r["p"]), src, path + ".p")
        em.macro("TrendPMin" + key, fmt_p(r["p_min"]), src, path + ".p_min",
                 "smallest p the design can attain")
        em.macro("TrendBlocks" + key, str(r["n_blocks"]), src, path + ".n_blocks")
        lo, hi = r["argmax_step"]
        em.macro("TrendStep" + key,
                 "$\\{" + ",".join(str(x) for x in lo) + "\\}$ vs $\\{"
                 + ",".join(str(x) for x in hi) + "\\}$", src, path + ".argmax_step",
                 "localisation only")
        em.macro("TrendIsoP" + key, fmt_p(r["isotonic"]["p"]), src, path + ".isotonic.p")
        entry = holm.get(name) or holm.get(f"{name} {r['task']}")
        em.macro("TrendHolm" + key, holm_verdict(entry), src,
                 f"{path.split('.')[0]}.holm_family[{name}]",
                 f"family of {entry['family_size']}" if entry else "")

    # C5, the mass-output x granularity interaction. Both probes, because D6
    # forbids a linear result standing alone, and the two one-sided gains,
    # because the interaction is their difference and a reader cannot see which
    # side moved from the difference alone.
    c5 = (A.get("confirmatory") or {}).get("C5")
    if c5:
        base = "confirmatory.C5.confirmatory.probes"
        for probe in ("linear", "mlp"):
            d = (c5.get("confirmatory", {}).get("probes", {}).get(probe, {}).get("did") or {})
            if not d.get("estimable"):
                continue
            key, path = texname("C5", probe), f"{base}.{probe}.did"
            em.macro("MassDid" + key, fmt(d["mean_diff"], 4, sign=True), src,
                     path + ".mean_diff",
                     "(162+mass - 162) - (17+mass - 17) in log(1-AUC); "
                     "positive = the mass output helps more at 17 classes")
            em.macro("MassDidT" + key, fmt(d["t"], 2, sign=True), src, path + ".t")
            em.macro("MassDidP" + key, fmt_p(d["p"]), src, path + ".p")
            em.macro("MassDidDf" + key, str(d["df"]), src, path + ".df")
            em.macro("MassDidCI" + key,
                     f"$[{fmt(d['ci95'][0], 4, sign=True)},\\,{fmt(d['ci95'][1], 4, sign=True)}]$",
                     src, path + ".ci95", "95% interval")
            em.macro("MassDidN" + key, str(d["n_pairs"]), src, path + ".n_pairs",
                     "pretraining seeds with all four corners of the 2x2")
        for lv in ("162", "17"):
            g = (c5.get("confirmatory", {}).get("probes", {}).get("linear", {})
                 .get("gain_by_level", {}).get(lv) or {})
            if not g.get("estimable"):
                continue
            key = texname("C5", "gain", lv)
            path = f"{base}.linear.gain_by_level.{lv}"
            em.macro("MassGain" + key, fmt(g["mean_diff"], 4, sign=True), src,
                     path + ".mean_diff",
                     f"with the mass output minus without, at {lv} classes; negative is better")
            em.macro("MassGainP" + key, fmt_p(g["p"]), src, path + ".p")

    for task, per_probe in (A.get("secondary", {}).get("S2") or {}).items():
        for probe, e in per_probe.items():
            if not e.get("run"):
                continue
            key, path = texname("S2", task, probe), f"secondary.S2.{task}.{probe}"
            top = e["largest_pair"]
            em.macro("TostBound" + key, fmt(e["target_bound"], 5), src, path + ".target_bound",
                     "+-ln(1.1), the pre-specified equivalence bound")
            em.macro("TostDiff" + key,
                     fmt(top["mean_diff"], 4, sign=True) + bound_mark(top["is_bound"]), src,
                     path + ".largest_pair.mean_diff",
                     f"{top['coarse']}-class minus {top['fine']}-class"
                     + (", a bound: a cell reached AUC=1" if top["is_bound"] else ""))
            em.macro("TostCI" + key,
                     f"$[{fmt(top['ci90'][0], 4, sign=True)},\\,{fmt(top['ci90'][1], 4, sign=True)}]$",
                     src, path + ".largest_pair.ci90", "90% interval, the TOST interval")
            em.macro("TostP" + key, fmt_p(e["p"]), src, path + ".p",
                     "intersection-union: the largest TOST p over all six pairs")
            em.macro("TostSmallestBound" + key, fmt(e["smallest_bound_passed_by_all"], 4), src,
                     path + ".smallest_bound_passed_by_all")

    for task, g in (A.get("secondary", {}).get("S6") or {}).items():
        if not g["n_pairs"]:
            continue
        em.macro("SignAgree" + texname(task), f"{g['n_agree']} of {g['n_pairs']}", src,
                 f"secondary.S6.{task}.n_agree", "level pairs the MLP orders as the linear probe")


def emit_pairwise(em: Emitter, A: dict, src: pathlib.Path, reference: int) -> None:
    """The paired contrasts against the reference vocabulary, which the paper quotes.

    Only the pairs that involve the reference are emitted -- the other three of
    the six are in the analysis file and nothing in the text cites them.
    """
    for task in sorted(A.get("pairwise_exploratory", {})):
        for probe in sorted(A["pairwise_exploratory"][task]):
            for i, r in enumerate(A["pairwise_exploratory"][task][probe]):
                if not r["estimable"]:
                    continue
                if reference not in (r["fine"], r["coarse"]):
                    # A step between two non-reference vocabularies, e.g. 43 -> 17:
                    # the size of the step itself, not its sum with 162 -> 43.
                    k2 = texname(task, probe, r["coarse"], "vs", r["fine"])
                    jp = f"pairwise_exploratory.{task}.{probe}[{i}]"
                    em.macro("PairFactor" + k2, fmt_factor(r["mean_diff"], 3) + bound_mark(r["is_bound"]),
                             src, jp + ".mean_diff",
                             f"{r['coarse']}-class over {r['fine']}-class, a factor in 1-AUC")
                    em.macro("PairP" + k2, fmt_p(r["p"]), src, jp + ".p", f"df={r['df']}")
                    continue
                other = r["coarse"] if r["fine"] == reference else r["fine"]
                key = texname(task, probe, other)
                jp = f"pairwise_exploratory.{task}.{probe}[{i}]"
                note = (f"{r['coarse']}-class minus {r['fine']}-class, paired by seed"
                        + (", a bound: a cell reached AUC=1" if r["is_bound"] else ""))
                em.macro("PairDiff" + key,
                         fmt(r["mean_diff"], 4, sign=True) + bound_mark(r["is_bound"]), src,
                         jp + ".mean_diff", note)
                em.macro("PairCI" + key,
                         f"$[{fmt(r['ci95'][0], 4, sign=True)},\\,{fmt(r['ci95'][1], 4, sign=True)}]$",
                         src, jp + ".ci95", "95% paired-t interval over seeds")
                em.macro("PairP" + key, fmt_p(r["p"]), src, jp + ".p", f"df={r['df']}")
                em.macro("PairFactor" + key, fmt_factor(r["mean_diff"], 3) + bound_mark(r["is_bound"]),
                         src, jp + ".mean_diff",
                         "exp of the difference: coarser over finer, a factor in 1-AUC")
                em.macro("PairHolm" + key, "yes" if r["holm_reject"] else "no", src,
                         jp + ".holm_reject", "Holm within this table of six")


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
# Everything below reads an analysis file that experiments/STATS/seed_level.py
# wrote after the probe ladder, and applies the same rules: copy, format, never
# re-test. Where a number is an aggregate of stored per-seed rows (a mean over
# seeds, a count of cells) the note on the macro says so.

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
            "schedule & flat, then decay (weaver default) & constant \\\\",
            "epochs & " + ", ".join(str(ep[n]) for n in sorted(ep))
            + " at " + ", ".join(fmt_n_jets(f"N{n}") for n in sorted(ep))
            + f" jets & {rec['epochs_bench']} at every size \\\\",
            f"validation jets & {fmt_int(rec['val_jetclass'])} & {fmt_int(rec['val_bench_small'])} "
            f"below {fmt_n_jets('N100000')}, {fmt_int(rec['val_bench_large'])} from it \\\\",
            "checkpoint & best validation & last epoch (C2, C3, S5); best validation otherwise \\\\"]
    caption = (f"Fine-tuning settings, as the {fmt_int(rec['n_runs'])} fine-tuning runs behind the "
               "reported results recorded them, checked against their job commands. The head is "
               "freshly initialised in every run; a pretrained start loads every other weight.")
    return _table(body, caption, "tab:ftrecipe", "l l l", [])


def emit_trend_sim(em: Emitter, T: dict, src: pathlib.Path) -> None:
    """The trend test under a null with equal means and the observed seed spreads
    (experiments/STATS/trend_size_sim.py): C1 in full, every other test in summary."""
    names = {"cov": "Cov", "indep": "Indep", "indep_upper": "IndepUpper"}
    prov = T["provenance"]
    em.macro("TrendSimNDraws", fmt_int(prov["n_stat_draws"]), src, "provenance.n_stat_draws")
    em.macro("TrendSimNSizeDraws", fmt_int(prov["n_size_draws"]), src, "provenance.n_size_draws")
    c1 = [i for i, r in enumerate(T["tests"]) if r["json_path"] == "confirmatory.C1"]
    if len(c1) != 1:
        raise SystemExit(f"FATAL: {src} holds {len(c1)} C1 rows")
    i = c1[0]
    for k, n in names.items():
        r = T["tests"][i]["nulls"][k]
        base = f"tests[{i}].nulls.{k}"
        em.macro(f"TrendSimSizeCone{n}", fmt(100 * r["size"]["size"], 1) + "\\%", src,
                 f"{base}.size.size", "percent, nominal 5%")
        em.macro(f"TrendSimNGeCone{n}", fmt_int(r["n_ge"]), src, f"{base}.n_ge")
        em.macro(f"TrendSimPCone{n}", fmt_p(r["p_sim"]), src, f"{base}.p_sim")
        if k == "indep_upper":
            em.macro("TrendSimSdFactor", fmt(r["sd_factor"], 2), src, f"{base}.sd_factor")
    S = T["summary"]
    em.macro("TrendSimNTests", str(S["n_tests"]), src, "summary.n_tests")
    em.macro("TrendSimNRejected", str(S["n_rejected"]), src, "summary.n_rejected")
    em.macro("TrendSimNRejectedSurviving", str(S["n_rejected_surviving_every_null"]), src,
             "summary.n_rejected_surviving_every_null", "p_sim <= 0.05 under all three nulls")
    for k, n in names.items():
        em.macro(f"TrendSimMaxSize{n}", fmt(100 * S["max_size"][k], 1) + "\\%", src,
                 f"summary.max_size.{k}", "largest size over every trend test")


def emit_trend_holm(em: Emitter, H: dict, src: pathlib.Path) -> None:
    """Each family's Holm correction applied to the simulated p-values."""
    names = {"cov": "Cov", "indep": "Indep", "indep_upper": "IndepUpper"}
    F = H["families"]
    a = F["anomaly (all tests)"]
    base = "families.anomaly (all tests)"
    for k, n in names.items():
        em.macro(f"TrendSimAnomalyNRejected{n}", str(a[f"n_rejected_{k}"]), src, f"{base}.n_rejected_{k}")
    lost = sorted({x for v in a["lost"].values() for x in v})
    em.macro("TrendSimAnomalyNLostAny", str(len(lost)), src, f"{base}.lost (union over nulls)")
    ft = [f for name, f in F.items() if "fine-tuning" in name]
    n_lost = sum(len(v) for f in ft for v in f["lost"].values())
    em.macro("TrendSimFtNLost", str(n_lost), src, "families.*fine-tuning*.lost (count)")


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


def emit_c1_detail(em: Emitter, A: dict, src: pathlib.Path) -> None:
    """C1's clause verdicts, its equivalence bound and the per-level seed spread.
    The trend numbers themselves are emitted by emit_tests."""
    c1 = A["confirmatory"]["C1"]
    for i, cl in enumerate(c1["clauses"]):
        em.macro("TestConeClause" + texname(cl["n"]) + "Verdict", tex(cl["verdict"]), src,
                 f"confirmatory.C1.clauses[{i}].verdict", cl["text"])
    eq = c1["clause3_equivalence"]
    em.macro("TestConeEquivP", fmt_p(eq["p"]), src, "confirmatory.C1.clause3_equivalence.p",
             "intersection-union over the pairs inside {188, 162, 43}")
    em.macro("TestConeEquivBound", fmt(eq["smallest_bound_passed_by_all"], 4), src,
             "confirmatory.C1.clause3_equivalence.smallest_bound_passed_by_all")
    em.macro("TestConeEquivFactor", fmt_factor(eq["smallest_bound_passed_by_all"], 3), src,
             "confirmatory.C1.clause3_equivalence.smallest_bound_passed_by_all",
             "exp of the bound: a factor in 1-AUC")
    em.macro("TestConeEquivTarget", fmt_factor(eq["target_bound"], 1), src,
             "confirmatory.C1.clause3_equivalence.target_bound", "exp(ln 1.1)")
    for lv, sd in zip(c1["levels_fine_to_coarse"], c1["seed_sd_per_level"]):
        em.macro("TestConeSeedSd" + texname(lv), fmt(sd, 3), src,
                 "confirmatory.C1.seed_sd_per_level", f"{lv} classes, log(1-AUC)")


def emit_recovery(em: Emitter, R: dict, src: pathlib.Path) -> None:
    """S9, label recovery at every level of the tree."""
    S = R["secondary"]["S9"]
    base = "secondary.S9"
    for field in ("n_train", "n_test"):
        em.macro("RecoveryN" + texname(field.split("_")[1]), fmt_int(R["provenance"][field]), src,
                 f"provenance.{field}", "jets")
    for i, cl in enumerate(S["clauses"]):
        em.macro("RecoveryClause" + texname(cl["n"]) + "Verdict", tex(cl["verdict"]), src,
                 f"{base}.clauses[{i}].verdict", cl["text"])
    worst = {"linear": 0.0, "mlp": 0.0}
    for lv, per in S["wins"].items():
        for probe, w in per.items():
            em.macro("RecoveryBeaten" + texname(lv, probe), of(w["n_beaten"], w["n_estimable"]),
                     src, f"{base}.wins.{lv}.{probe}.n_beaten",
                     "cells at or below its own vocabulary where another model is ahead")
            worst[probe] = max([worst[probe]] + [abs(b["mean_diff"]) for b in w["beaten_by"]])
    for probe, v in worst.items():
        em.macro("RecoveryWorstLoss" + texname(probe), fmt(v, 4), src,
                 f"{base}.wins.*.{probe}.beaten_by[*].mean_diff",
                 "largest |difference| in balanced accuracy among this probe's losses")
    for probe in ("linear", "mlp"):
        cr = [S["pairs"][p][probe]["crossover"] for p in S["pairs"]]
        em.macro("RecoveryCrossAtOwn" + texname(probe),
                 of(sum(c["crossover_at_coarser_own_rung"] for c in cr), len(cr)), src,
                 f"{base}.pairs.*.{probe}.crossover.crossover_at_coarser_own_rung",
                 "model pairs crossing exactly at the coarser model's own level")
        em.macro("RecoveryCrossInBracket" + texname(probe),
                 of(sum(c["coarser_own_rung_in_bracket"] for c in cr), len(cr)), src,
                 f"{base}.pairs.*.{probe}.crossover.coarser_own_rung_in_bracket")
    # The pairs the text quotes: 188 against 162, which never differ, and 162
    # against 17, whose advantage decays to zero at the 17-class level.
    for pair in ("188_vs_162", "162_vs_17"):
        rungs = S["pairs"][pair]["linear"]["rungs"]
        for j, r in enumerate(rungs):
            key = texname(pair.replace("_vs_", " over "), "at", r["rung"])
            path = f"{base}.pairs.{pair}.linear.rungs[{j}]"
            em.macro("RecoveryAdv" + key, fmt(r["mean_diff"], 4, sign=True), src,
                     path + ".mean_diff", "finer minus coarser, balanced accuracy")
            em.macro("RecoveryAdvP" + key, fmt_p(r["p"]), src, path + ".p", f"df={r['df']}")
    rungs = S["pairs"]["188_vs_162"]["linear"]["rungs"]
    em.macro("Recovery" + texname(188, 162) + "MaxAbs", fmt(max(abs(r["mean_diff"]) for r in rungs), 4),
             src, f"{base}.pairs.188_vs_162.linear.rungs[*].mean_diff", "max |difference| over the tree")
    em.macro("Recovery" + texname(188, 162) + "NDistinct",
             of(sum(r["holm_reject"] for r in rungs), len(rungs)), src,
             f"{base}.pairs.188_vs_162.linear.rungs[*].holm_reject")


def emit_random_control(em: Emitter, C: dict, src: pathlib.Path) -> None:
    """C4, descriptive (A1), and the post-hoc grouping cost beside it."""
    for probe in ("linear", "mlp"):
        r = C["C4"][probe]
        key = texname(probe)
        em.macro("RandMatch" + key, of(r["n_match"], r["n_signed_cells"]), src,
                 f"C4.{probe}.n_match", "signed cells matching the predicted sign")
        em.macro("RandP" + key, fmt_p(r["p_at_least"]), src, f"C4.{probe}.p_at_least",
                 "exact binomial, descriptive")
        em.macro("RandPFloor" + key, fmt_p(r["p_floor"]), src, f"C4.{probe}.p_floor")
        for i, c in enumerate(r["cells"]):
            k2 = texname(c["task"], "draw", c["draw"], probe)
            path = f"C4.{probe}.cells[{i}]"
            em.macro("RandDiff" + k2, fmt(c["diff"], 3, sign=True), src, path + ".diff",
                     "control minus 17-class model, log(1-AUC)")
            em.macro("RandCI" + k2, f"$[{fmt(c['ci'][0], 3, sign=True)},\\,"
                     f"{fmt(c['ci'][1], 3, sign=True)}]$", src, path + ".ci",
                     "paired bootstrap 95% interval")
    same = sum(1 for a, b in zip(C["C4"]["linear"]["cells"], C["C4"]["mlp"]["cells"])
               if np.sign(a["diff"]) == np.sign(b["diff"]))
    em.macro("RandMlpSameSign", of(same, len(C["C4"]["linear"]["cells"])), src,
             "C4.{linear,mlp}.cells[*].diff", "cells where the MLP probe has the linear sign")
    for i, row in enumerate(C["table"]):
        if row["probe"] != "linear":
            continue
        em.macro("RandLevel" + texname(row["task"], "draw", row["draw"]),
                 fmt(row["control_log1m_auc"], 2, sign=True), src,
                 f"table[{i}].control_log1m_auc", "log(1-AUC), linear probe")
    g = C["grouping_cost_post_hoc"]
    for task, per in g["tasks"].items():
        for probe, x in per.items():
            k2 = texname(task, probe)
            path = f"grouping_cost_post_hoc.tasks.{task}.{probe}"
            em.macro("RandCostFactor" + k2, fmt(x["factor_control"], 2), src,
                     path + ".factor_control", "POST HOC: random grouping vs finer models, in 1-AUC")
            em.macro("RandSemCostFactor" + k2, fmt(x["factor_semantic"], 2), src,
                     path + ".factor_semantic", "POST HOC: semantic 17-class vs finer models")


def ft_levels(per_size: dict, n: str) -> dict:
    return {r["level"]: r["mean"] for r in per_size[n]["levels"]}


def emit_finetune(em: Emitter, F: dict, src: pathlib.Path) -> None:
    """S3 and S4: fine-tuning on JetClass and JetClass-II, fine-tuning seed 1."""
    for key in ("S4", "S3"):
        S = F["secondary"][key]
        base = f"secondary.{key}"
        k = texname(key)
        for i, cl in enumerate(S["clauses"]):
            em.macro("FtClause" + k + texname(cl["n"]) + "Verdict", tex(cl["verdict"]), src,
                     f"{base}.clauses[{i}].verdict", cl["text"])
        for n, d in S["per_size"].items():
            k2 = k + n_tag(n)
            t = d["trend"]
            em.macro("FtTrendP" + k2, fmt_p(t["p"]), src, f"{base}.per_size.{n}.trend.p")
            em.macro("FtTrendHolm" + k2, "rejected" if t.get("holm_reject_within_table")
                     else "not rejected", src,
                     f"{base}.per_size.{n}.trend.holm_reject_within_table", "Holm within the table")
            gap = S["gap_17_minus_188"][n]
            em.macro("FtGap" + k2, fmt(gap, 3, sign=True), src, f"{base}.gap_17_minus_188.{n}",
                     "17-class minus 188-class, log(1 - macro AUC)")
            em.macro("FtGapFactor" + k2, fmt_factor(gap, 2), src, f"{base}.gap_17_minus_188.{n}",
                     "exp of the gap: a factor in 1 - macro AUC")
            em.macro("FtSize" + k2, fmt_n_jets(n), src, f"{base}.per_size.{n}", "training jets")
            for j, r in enumerate(d["pairwise"]):
                if not r["estimable"]:
                    continue
                path = f"{base}.per_size.{n}.pairwise[{j}]"
                k3 = k2 + texname(r["coarse"], "vs", r["fine"])
                em.macro("FtPairDiff" + k3, fmt(r["mean_diff"], 3, sign=True), src,
                         path + ".mean_diff", f"{r['coarse']}-class minus {r['fine']}-class, exploratory")
                em.macro("FtPairCI" + k3,
                         f"$[{fmt(r['ci95'][0], 3, sign=True)},\\,{fmt(r['ci95'][1], 3, sign=True)}]$",
                         src, path + ".ci95", "95% paired-t interval")
                em.macro("FtPairHolm" + k3, "yes" if r["holm_reject"] else "no", src,
                         path + ".holm_reject", "Holm within the six pairs at this size")
            em.macro("FtPairNExcludeZero" + k2,
                     of(sum(1 for r in d["pairwise"] if r["estimable"] and (r["ci95"][0] > 0 or r["ci95"][1] < 0)),
                        sum(1 for r in d["pairwise"] if r["estimable"])), src,
                     f"{base}.per_size.{n}.pairwise[*].ci95", "pairs whose 95% interval excludes zero")
            em.macro("FtPairNHolm" + k2,
                     of(sum(1 for r in d["pairwise"] if r.get("holm_reject")),
                        sum(1 for r in d["pairwise"] if r["estimable"])), src,
                     f"{base}.per_size.{n}.pairwise[*].holm_reject", "pairs rejected after Holm")
            refs = S["reference_rows"][n]
            for arm in ("scratch", "mpm-s1"):
                if arm in refs:
                    em.macro("FtRefAuc" + k2 + texname(arm), fmt(refs[arm]["macro_auc"], 4), src,
                             f"{base}.reference_rows.{n}.{arm}.macro_auc",
                             "fine-tuning seed 1, macro AUC")


def emit_anomaly(em: Emitter, S5: dict, src: pathlib.Path) -> None:
    """Section 5, anomaly detection: one trend test per (signal, detector family)."""
    s = S5["section5"]
    run = [v for v in s["tests"].values() if v["trend"].get("run")]
    em.macro("AnomalyNTested", str(len(run)), src, "section5.tests[*].trend.run",
             "trend tests that could run at the fixed injection")
    em.macro("AnomalyNRejected", str(sum(1 for v in run if v.get("holm_reject"))), src,
             "section5.tests[*].holm_reject", "Holm within the table")
    em.macro("AnomalyNPlanned", str(len(s["tests"])), src, "section5.tests (count)")
    em.macro("AnomalyInjection", fmt_int(s["injection"]), src, "section5.injection",
             "injected signal jets")
    em.macro("AnomalyNTrainings", str(S5["provenance"]["detector_trainings"]), src,
             "provenance.detector_trainings", "detector trainings per model")
    em.macro("AnomalyClauseOneVerdict", tex(s["clause1"]), src, "section5.clause1")
    em.macro("AnomalyClauseTwoVerdict", tex(s["clause2"]), src, "section5.clause2")
    for fam, r in s["clause1_per_family"].items():
        n_sig = len({k.split("|")[1] for k in s["tests"]})
        em.macro("AnomalyRejected" + texname(fam), of(r["n_signals_rejected"], n_sig), src,
                 f"section5.clause1_per_family.{fam}.n_signals_rejected")
    for fam, r in s["clause2_on_testable_signals"].items():
        em.macro("AnomalyGapB" + texname(fam), fmt(r["mean_gap_b"], 3, sign=True), src,
                 f"section5.clause2_on_testable_signals.{fam}.mean_gap_b",
                 "mean 17 minus 188 in ln sigma_min, b-quark signals")
        em.macro("AnomalyGapLight" + texname(fam), fmt(r["mean_gap_light"], 3, sign=True), src,
                 f"section5.clause2_on_testable_signals.{fam}.mean_gap_light")
    sigs = sorted({k.split("|")[1] for k in s["tests"]})
    testable = sorted({k.split("|")[1] for k, v in s["tests"].items() if v["trend"].get("run")})
    em.macro("AnomalyNSignals", str(len(sigs)), src, "section5.tests (distinct signals)")
    em.macro("AnomalyNTestableSignals", str(len(testable)), src, "section5.tests[*].trend.run",
             "signals testable at the fixed injection")
    for k, v in s["tests"].items():
        if not v["trend"].get("run"):
            continue
        k2 = texname(*k.split("|"))
        em.macro("AnomalyGap" + k2, fmt(v["gap_17_minus_188"], 3, sign=True), src,
                 f"section5.tests.{k}.gap_17_minus_188", "17 minus 188 in ln sigma_min")
        em.macro("AnomalyTrendP" + k2, fmt_p(v["trend"]["p"]), src, f"section5.tests.{k}.trend.p")
    bl = s["clause2_on_testable_signals"]
    em.macro("AnomalyBLarger", of(sum(bool(r["b_larger"]) for r in bl.values()), len(bl)), src,
             "section5.clause2_on_testable_signals.*.b_larger",
             "families where the b-quark signals gain more, testable signals only; descriptive")
    below = sum(1 for v in run if v.get("holm_reject")
                and {188, 162} <= set(v["trend"]["argmax_step"][0]))
    em.macro("AnomalyStepBelowOnesixtwo", of(below, sum(1 for v in run if v.get("holm_reject"))),
             src, "section5.tests[*].trend.argmax_step",
             "rejected cells whose arg-max contrast has 188 and 162 on the same side")


def emit_mass_resolution(em: Emitter, M: dict, src: pathlib.Path) -> None:
    """S7, frozen-feature jet-mass regression (Holm within its table, A4)."""
    S = M["secondary"]["S7"]
    base = "secondary.S7"
    em.macro("MassResNJets", fmt_int(M["provenance"]["n_jets_valid"]), src,
             "provenance.n_jets_valid", "test jets with a matched generator-level groomed mass")
    em.macro("MassResNClasses", str(M["provenance"]["n_classes_used"]), src,
             "provenance.n_classes_used", "native classes with enough training jets to centre")
    for i, cl in enumerate(S["clauses"]):
        em.macro("MassResClause" + texname(cl["n"]) + "Verdict", tex(cl["verdict"]), src,
                 f"{base}.clauses[{i}].verdict", cl["text"])
    for probe in ("ridge", "mlp"):
        P = S["probes"][probe]
        k = texname(probe)
        for lv, g in P["gain_by_level"].items():
            em.macro("MassResGain" + k + texname(lv), fmt(g["mean_diff"], 4, sign=True), src,
                     f"{base}.probes.{probe}.gain_by_level.{lv}.mean_diff",
                     "sigma_eff with the mass output minus without; negative is better")
            em.macro("MassResGainP" + k + texname(lv), fmt_p(g["p"]), src,
                     f"{base}.probes.{probe}.gain_by_level.{lv}.p")
        em.macro("MassResDid" + k, fmt(P["did"]["mean_diff"], 4, sign=True), src,
                 f"{base}.probes.{probe}.did.mean_diff")
        em.macro("MassResDidP" + k, fmt_p(P["did"]["p"]), src, f"{base}.probes.{probe}.did.p")
        em.macro("MassResLadderHolm" + k,
                 of(sum(p["holm_reject"] for p in P["ladder_pairs"]), len(P["ladder_pairs"])), src,
                 f"{base}.probes.{probe}.ladder_pairs[*].holm_reject",
                 "coarser-better pairs surviving Holm within the table")
        em.macro("MassResLadderSign" + k,
                 of(sum(p["mean_diff"] < 0 for p in P["ladder_pairs"]), len(P["ladder_pairs"])),
                 src, f"{base}.probes.{probe}.ladder_pairs[*].mean_diff",
                 "pairs where the coarser model is better on the point estimate")
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


def table_probe_ladder(A: dict, probe: str) -> str:
    """T1: granularities down the side, probe tasks across the top.

    Two rows per granularity rather than a two-line cell: the second column says
    which quantity the row holds, so the table needs nothing but booktabs.
    """
    tasks = ordered_tasks(A["levels"])
    levels = A["levels_fine_to_coarse"]
    eps = headline_rejection(A["levels"][tasks[0]][probe][0])["eps"]
    head = ["classes & quantity & " + " & ".join(TASK_LABELS.get(t, tex(t)) for t in tasks)
            + " \\\\", "\\midrule"]
    body = []
    n_seeds = set()
    for i, lv in enumerate(levels):
        auc, rej = [], []
        for t in tasks:
            row = next(r for r in A["levels"][t][probe] if r["level"] == lv)
            n_seeds.add(row["n_seeds"])
            sd = seed_values(A["table"], t, probe, lv, "auc")
            # A cell that saturated in every seed has a spread of exactly zero.
            # Printing "+- 0.00000" would read as a measured agreement.
            spread = (f"\\,{tex('±')}\\,{fmt(np.std(sd, ddof=1), 5)}"
                      if len(sd) > 1 and row["n_censored"] < row["n_seeds"] else "")
            auc.append(fmt_auc(row["mean_auc"], row["n_censored"], row["n_seeds"]) + spread)
            h = headline_rejection(row)
            rej.append(fmt_rejection(h["median"], h["n_bound"], h["n_seeds"])
                       + f" [{fmt_rejection(h['range'][0], 0, 0)}, "
                         f"{fmt_rejection(h['range'][1], 0, 0)}]")
        body.append(f"{lv} & AUC & " + " & ".join(auc) + " \\\\")
        body.append("     & $1/\\epsilon_B$ & " + " & ".join(rej) + " \\\\")
        if i < len(levels) - 1:
            body.append("\\addlinespace")
    seeds = min(n_seeds)
    caption = (f"Frozen {'linear' if probe == 'linear' else 'nonlinear (MLP)'} probes on the "
               f"four pretraining vocabularies. AUC is the mean over {seeds} pretraining seeds "
               f"{tex('±')} the seed standard deviation; $1/\\epsilon_B$ is the background "
               f"rejection at {float(eps) * 100:.0f}\\% signal efficiency, as the median over "
               f"seeds with the [min, max] range beside it. Rows are the pretraining label-set "
               f"size, finest first; lower granularity is further down.")
    notes = [
        "$>$ a lower bound: no background jet survived the cut in any seed, so the value is the "
        "sample size, not a measured rejection. $^{\\ast}$ only some seeds are at that cap, so "
        "the median may itself be a bound.",
        "$^{\\dagger}$ the AUC reached 1 at the resolution of the sample in at least one seed; "
        "$1-$AUC is then an upper bound and the cell is not a measurement.",
    ]
    return _table(head + body, caption, f"tab:probes-{probe}",
                  "r l " + "r" * len(tasks), notes, wide=True)


def table_tests(A: dict) -> str:
    """T2: the pre-specified tests, read out of the analysis file verbatim."""
    holm = holm_lookup(A)
    rows = []

    def trend_row(name, r):
        pred = r["alternative"].split("= ", 1)[-1] if "= " in r["alternative"] else r["alternative"]
        entry = holm.get(name) or holm.get(f"{name} {r['task']}")
        rows.append(" & ".join([
            f"{name}: {TASK_LABELS.get(r['task'], tex(r['task']))}",
            tex(pred),
            f"max-$T$ trend, {r['method']}, {r['n_blocks']} seed blocks",
            f"${fmt(r['stat'], 3)}$",
            fmt_p(r["p"]),
            holm_verdict(entry)]) + " \\\\")

    c1 = (A.get("confirmatory") or {}).get("C1")
    if c1 and c1.get("run"):
        trend_row("C1", c1)
        # C1's prediction has three clauses and the trend test answers only two
        # of them. The third -- that the three finer vocabularies perform alike
        # -- is a predicted null, so PRESPEC 2.5 requires an equivalence test,
        # and one of its three pairs fails at the declared bound. Printing the
        # trend row alone would read as a clean confirmation of a prediction
        # that is only partly confirmed, which is the failure a pre-registration
        # exists to prevent. The clause rows come straight out of the analysis;
        # nothing here recomputes a verdict.
        for cl in c1.get("clauses", []):
            rows.append(" & ".join([
                f"\\quad clause {cl['n']}",
                tex(cl["text"]),
                tex(cl["test"]),
                "---" if cl.get("detail") is None else tex(str(cl["detail"])),
                "---" if cl.get("p") is None else fmt_p(cl["p"]),
                tex(cl["verdict"])]) + " \\\\")
        if c1.get("composite_verdict"):
            rows.append("\\multicolumn{6}{@{}l@{}}{\\itshape C1 overall: "
                        + tex(c1["composite_verdict"]) + "} \\\\")
    # C5, the mass-output x granularity interaction. It is the first member of
    # the confirmatory family that can become available WITHOUT being a trend
    # test, so it needs a renderer of its own.
    rendered = {"C1"} if (c1 and c1.get("run")) else set()
    c5 = (A.get("confirmatory") or {}).get("C5")
    lin = ((c5 or {}).get("confirmatory", {}).get("probes", {})
           .get("linear", {}).get("did") or {})
    if c5 and not lin.get("estimable"):
        # It RAN. Holm keeps it pending because there is no p, but the table must
        # not say "not yet measured" about a test that was measured and could not
        # be estimated -- those are different facts about the study.
        rendered.add("C5")
        rows.append(f"C5: {TASK_LABELS.get(c5.get('task', ''), tex(c5.get('task', '')))} & "
                    + tex(c5.get("prediction", "")) + " & paired difference-in-differences & "
                    "--- & --- & measured, not estimable ("
                    + tex(str(c5.get("not_estimable_reason") or "unknown")) + ") \\\\")
    elif c5:
        rendered.add("C5")
        rows.append(" & ".join([
            f"C5: {TASK_LABELS.get(c5['task'], tex(c5['task']))}",
            tex(c5["prediction"]),
            "paired difference-in-differences, " + tex(c5["did_definition"]),
            f"${fmt(lin['mean_diff'], 4, sign=True)}$ ($t={fmt(lin['t'], 2, sign=True)}$, "
            f"{lin['df']} df)",
            fmt_p(lin["p"]),
            holm_verdict(holm.get("C5"))]) + " \\\\")
        # D6: never a linear probe alone. The nonlinear probe goes beside it,
        # and it is not a second test -- it has no Holm entry and no verdict.
        mlp = (c5["confirmatory"]["probes"].get("mlp", {}).get("did") or {})
        if mlp.get("estimable"):
            rows.append(" & ".join([
                "\\quad nonlinear probe",
                "the nonlinear probe gives the same interaction",
                "paired difference-in-differences",
                f"${fmt(mlp['mean_diff'], 4, sign=True)}$ ($t={fmt(mlp['t'], 2, sign=True)}$, "
                f"{mlp['df']} df)",
                fmt_p(mlp["p"]), "descriptive"]) + " \\\\")
        # The interaction is a difference of two gains; printing only the
        # difference leaves a reader unable to see which side moved.
        for lv in ("162", "17"):
            g = c5["confirmatory"]["probes"]["linear"]["gain_by_level"].get(lv) or {}
            if g.get("estimable"):
                rows.append(" & ".join([
                    f"\\quad gain at {lv} classes",
                    "effect of adding the mass output at this granularity alone",
                    "paired difference",
                    f"${fmt(g['mean_diff'], 4, sign=True)}$",
                    fmt_p(g["p"]), "descriptive"]) + " \\\\")
    for h in A.get("confirmatory", {}).get("holm_family", []):
        if h["status"] == "pending":
            rows.append(f"{h['test']} & not yet measured & --- & --- & --- & pending \\\\")
        elif h["test"] not in rendered:
            # A measured confirmatory test with no renderer of its own. Printing
            # nothing is the one thing this table must never do: the caption
            # claims a family of five, and a member that has been measured and
            # Holm-judged would vanish while the caption went on asserting it.
            # This row is deliberately bare -- it exists so the omission is
            # visible in the manuscript rather than silent.
            rows.append(f"{h['test']} & measured; no row generator & --- & --- & "
                        f"{fmt_p(h['p_raw'])} & {holm_verdict(h)} \\\\")
    # C4 is pre-specified but outside the Holm count (PRESPEC amendment 2026-09-22):
    # its smallest attainable p, 1/16, is above alpha. It keeps a row so it cannot vanish.
    rows.append("C4 & random-label control: a pair is separated better when the labels "
                "split it & sign pattern over six cells, exact binomial over the four signed "
                "cells (smallest attainable $p=1/16$) & --- & --- & descriptive \\\\")
    rows.append("\\addlinespace")
    s1 = (A.get("secondary") or {}).get("S1")
    if s1 and s1.get("run"):
        trend_row("S1", s1)
    s2 = A.get("secondary", {}).get("S2") or {}
    for task in ordered_tasks(s2):
        e = s2[task].get("linear")
        if not e or not e.get("run"):
            continue
        top = e["largest_pair"]
        entry = holm.get(f"S2 {task}")
        rows.append(" & ".join([
            f"S2: {TASK_LABELS.get(task, tex(task))}",
            "equivalent within $\\pm\\ln(1.1)$ of $1-$AUC",
            f"TOST on all {e['n_pairs_total']} pairs, intersection-union; largest pair "
            f"{top['coarse']} vs {top['fine']} classes, {top['n_pairs']} seed pairs",
            f"${fmt(top['mean_diff'], 4, sign=True)}"
            + ("^{\\ast}$" if top["is_bound"] else "$"),
            fmt_p(e["p"]),
            holm_verdict(entry)]) + " \\\\")
    s6 = A.get("secondary", {}).get("S6") or {}
    for task in ordered_tasks(s6):
        g = s6[task]
        if not g["n_pairs"]:
            continue
        rows.append(" & ".join([
            f"S6: {TASK_LABELS.get(task, tex(task))}",
            "the nonlinear probe orders the vocabularies as the linear probe does",
            "sign agreement over the estimable level pairs",
            f"{g['n_agree']} of {g['n_pairs']}", "---", "descriptive"]) + " \\\\")

    head = ["test & prediction & test used & statistic & $p$ & Holm \\\\", "\\midrule"]
    fam = A.get("confirmatory", {}).get("holm_family", [{}])[0].get("family_size", "?")
    caption = ("The pre-specified tests, exactly as "
               "\\texttt{experiments/STATS/seed\\_level.py} wrote them; nothing in this table is "
               f"recomputed. ``Holm'' is the verdict at the full confirmatory family size of "
               f"{fam} with the unmeasured members still pending, so a rejection here holds "
               "whatever those turn out to be. Differences run coarser minus finer in "
               "$\\log(1-$AUC$)$, paired by pretraining seed: positive means the coarser "
               "vocabulary is worse.")
    notes = ["S6 is descriptive and carries no $p$: it is a count of level pairs, not a test.",
             "$^{\\ast}$ the contrast involves a cell whose AUC reached 1 at the resolution of "
             "the sample, so the difference is a bound rather than a measured value."]
    return _table(head + rows, caption, "tab:tests", "l p{0.20\\linewidth} p{0.20\\linewidth} r r l",
                  notes, wide=True)


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


def table_finetune(F: dict) -> str:
    """S4 and S3: mean log(1 - macro AUC) per vocabulary and training-set size."""
    sizes = list(F["secondary"]["S4"]["per_size"])
    ncol = 1 + len(sizes)
    head = ["training jets & " + " & ".join(fmt_n_jets(n) for n in sizes) + " \\\\", "\\midrule"]
    body = []
    for key, name in (("S4", "JetClass-II, 162 classes"), ("S3", "JetClass, 10 classes")):
        S = F["secondary"][key]
        body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\itshape {name}}} \\\\")
        for lv in S["per_size"][sizes[0]]["trend"]["levels_fine_to_coarse"]:
            body.append(f"{lv} classes & " + " & ".join(
                fmt(ft_levels(S["per_size"], n)[lv], 3, sign=True) for n in sizes) + " \\\\")
        for arm, label in (("scratch", "random initialisation"),
                           ("mpm-s1", "self-supervised, seed 1"),
                           ("rand-d1-s1b", "random-label control, draw 1")):
            body.append(f"{label} & " + " & ".join(
                fmt(np.log1p(-S["reference_rows"][n][arm]["macro_auc"]), 3, sign=True)
                if arm in S["reference_rows"][n] else "---" for n in sizes) + " \\\\")
        body.append("trend $p$ & " + " & ".join(
            fmt_p(S["per_size"][n]["trend"]["p"])
            + ("" if S["per_size"][n]["trend"].get("holm_reject_within_table") else "$^{\\circ}$")
            for n in sizes) + " \\\\")
        if key == "S4":
            body.append("\\addlinespace")
    caption = ("Fine-tuning every pretrained model on the pretraining dataset's own 162-class task "
               "and on JetClass's 10-class task: natural log of $1-$macro AUC (lower is better), "
               "mean over the five pretraining seeds, fine-tuning seed 1, best-validation epoch. The "
               "reference rows are single models. The trend test is the pre-specified max-$T$ test "
               "with pretraining seed as the block.")
    notes = ["$^{\\circ}$ not rejected after Holm within the four sizes of its dataset."]
    return _table(head + body, caption, "tab:finetune", "l " + "r" * len(sizes), notes)


def table_anomaly(S5: dict) -> str:
    """Section 5: 17-class minus 188-class gap in ln sigma_min, per signal and family."""
    s = S5["section5"]
    fams = [f for f in FAMILY_LABELS if any(k.startswith(f + "|") for k in s["tests"])]
    sigs = [g for g in SIGNAL_LABELS if any(k.endswith("|" + g) for k in s["tests"])]
    head = ["signal & " + " & ".join(FAMILY_LABELS[f] for f in fams) + " \\\\", "\\midrule"]
    body = []
    for g in sigs:
        cells = []
        for f in fams:
            t = s["tests"][f"{f}|{g}"]
            if not t["trend"].get("run"):
                cells.append("---")
                continue
            v = fmt(t["gap_17_minus_188"], 2, sign=True)
            cells.append(f"\\textbf{{{v}}}" if t.get("holm_reject") else v)
        body.append(SIGNAL_LABELS[g] + " & " + " & ".join(cells) + " \\\\")
    caption = (f"Anomaly-detection sensitivity at {fmt_int(s['injection'])} injected signal jets: "
               "the 17-class minus 188-class difference in $\\ln\\sigma_{\\min}$, the smallest "
               "initial significance from which the signal is still discovered at $5\\sigma$ "
               "(positive: the coarser vocabulary needs more signal). Mean over paired pretraining "
               "seeds of the median over ten detector trainings. Bold: the four-level trend test "
               "rejects after Holm over the table.")
    notes = ["--- the test sample holds too few such jets at this injection; the trend test was "
             "not run.",
             "Only the class-sum score uses the pretraining labels; the other three families work "
             "on the frozen features alone."]
    return _table(head + body, caption, "tab:anomaly", "l " + "r" * len(fams), notes)


def table_random_control(C: dict) -> str:
    """C4 cells and the post-hoc grouping cost, linear probe."""
    cells = C["C4"]["linear"]["cells"]
    draws = sorted({c["draw"] for c in cells})
    tasks = [t for t in TASK_LABELS if any(c["task"] == t for c in cells)]
    sign = {-1: "$-$", 0: "$0$", 1: "$+$"}
    head = ["pair & " + " & ".join(f"draw {d}" for d in draws) + " \\\\", "\\midrule"]
    body = []
    for t in tasks:
        row = []
        for d in draws:
            c = next(x for x in cells if x["task"] == t and x["draw"] == d)
            row.append(f"{fmt(c['diff'], 2, sign=True)} [{fmt(c['ci'][0], 2, sign=True)}, "
                       f"{fmt(c['ci'][1], 2, sign=True)}] ({sign[c['predicted_sign']]})")
        body.append(TASK_LABELS[t] + " & " + " & ".join(row) + " \\\\")
    caption = ("The random-label control against the 17-class model of the same seed index: "
               "difference in the natural log of $1-$AUC on the two dedicated pairs, frozen linear "
               "probe, with its paired bootstrap 95\\% interval and, in brackets, the sign that was "
               "predicted (negative: the control is better). Each draw is one random partition "
               "with the class-size structure of the 17-class label set.")
    notes = ["The MLP probe agrees in sign in every cell."]
    return _table(head + body, caption, "tab:random-control", "l " + "c" * len(draws), notes,
                  wide=True)


def table_recovery(R: dict, sizes: dict) -> str:
    """S9: balanced accuracy recovering each level of the tree from each model."""
    S = R["secondary"]["S9"]
    levels = S["levels_fine_to_coarse"]
    head = ["read out at & " + " & ".join(f"{lv}-class model" for lv in levels) + " \\\\",
            "\\midrule"]
    body = []
    for rung in S["rungs_fine_to_coarse"]:
        cells = []
        for lv in levels:
            acc = [r["accuracy"] for r in R["table"]
                   if r["rung"] == rung and r["level"] == lv and r["probe"] == "linear"]
            cells.append(f"{fmt(np.mean(acc), 3)}\\,{tex('±')}\\,{fmt(np.std(acc, ddof=1), 3)}")
        body.append(f"{sizes[rung]} classes & " + " & ".join(cells) + " \\\\")
    caption = ("Label recovery: balanced accuracy of a frozen linear probe trained to recover "
               "each level of the label tree (rows, finest first) from each pretrained model "
               "(columns), mean over five pretraining seeds $\\pm$ their standard deviation.")
    return _table(head + body, caption, "tab:recovery", "l " + "r" * len(levels), [])


def table_mass(M: dict) -> str:
    """S7: jet-mass resolution from frozen features, ridge probe."""
    cells = ["188", "162", "43", "17", "162+mass", "17+mass"]
    label = {c: (c.replace("+mass", " + mass")) for c in cells}
    head = ["& " + " & ".join(label[c] for c in cells) + " \\\\", "\\midrule"]
    body = []
    for field, name, nd in (("sigma_eff", "$\\sigma_{\\mathrm{eff}}$", 4),
                            ("sd", "standard deviation", 3),
                            ("tail_fraction", "tail fraction", 3)):
        vals = []
        for c in cells:
            rows = [r[field] for r in M["table"] if r["cell"] == c and r["probe"] == "ridge"]
            vals.append(fmt(np.mean(rows), nd))
        body.append(f"{name} & " + " & ".join(vals) + " \\\\")
    caption = ("Jet-mass regression from frozen features. The residual is "
               "$\\ln(m_{\\mathrm{pred}}/m_{\\mathrm{true}})$ after removing each native class's "
               "training-set mean; $\\sigma_{\\mathrm{eff}}$ is half the smallest interval holding "
               "68\\% of it, the tail fraction is the share with $|$residual$|>1$. Ridge probe, mean "
               "over five pretraining seeds; columns are the label set, with or without the added "
               "mass output.")
    return _table(head + body, caption, "tab:mass", "l " + "r" * len(cells), [], wide=True)


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
    emit_mde(em, A, paths["analysis"])
    emit_tests(em, A, paths["analysis"])
    # The 162-class vocabulary is the reference the paper contrasts against: it is
    # the largest set that drops nothing but the 26 QCD subclasses.
    reference = A["levels_fine_to_coarse"][1]
    emit_pairwise(em, A, paths["analysis"], reference)
    emit_vocabulary(em, sizes, paths["rung_map"])
    # The test-sample background count, which bounds every rejection: the same
    # jets in every ladder file, or the files were not scored on the same sample.
    ladders = [json.loads(pathlib.Path(f).read_text()) for f in paths["ladder"]]
    for task in sorted(ladders[0]["tasks"]):
        counts = {d["tasks"][task].get("n_background_test") for d in ladders if task in d["tasks"]}
        if len(counts) == 1 and None not in counts:
            em.macro("ProbeNBkgTest" + texname(task), fmt_int(counts.pop()), paths["ladder"][0],
                     f"tasks.{task}.n_background_test", "test-sample background jets")

    out = {"tables/probes_linear.tex": table_probe_ladder(A, "linear"),
           "tables/probes_mlp.tex": table_probe_ladder(A, "mlp"),
           "tables/tests.tex": table_tests(A)}

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
    if "trend_sim" in have:
        check_inputs_unchanged(paths["trend_sim"], root)
        emit_trend_sim(em, json.loads(paths["trend_sim"].read_text()), paths["trend_sim"])
        if "trend_holm" not in have:
            raise SystemExit("FATAL: the trend simulation has no Holm summary; run "
                             "trend_size_sim.py --holm-from")
        check_inputs_unchanged(paths["trend_holm"], root)
        emit_trend_holm(em, json.loads(paths["trend_holm"].read_text()), paths["trend_holm"])
    else:
        missing.append(f"trend-test size simulation -- {paths['trend_sim']}")
    if "mass_specs" in have and paths["mass_specs"]:
        emit_mass_lambda(em, paths["mass_specs"])
    if all(k in have for k in ("ft_recipes", "ft_leg_specs", "ft_bench_specs")) \
            and paths["ft_leg_specs"] and paths["ft_bench_specs"]:
        rec = ft_recipe(json.loads(paths["ft_recipes"].read_text()),
                        paths["ft_leg_specs"], paths["ft_bench_specs"])
        emit_ft_recipe(em, rec, paths["ft_recipes"])
        out["tables/finetune_recipe.tex"] = table_ft_recipe(rec)
    if (A.get("confirmatory") or {}).get("C1", {}).get("clause3_equivalence"):
        emit_c1_detail(em, A, paths["analysis"])
    later = (("recovery", emit_recovery, lambda d: table_recovery(d, sizes), "label_recovery"),
             ("random_control", emit_random_control, table_random_control, "random_control"),
             ("finetune", emit_finetune, table_finetune, "finetune"),
             ("anomaly", emit_anomaly, table_anomaly, "anomaly"),
             ("mass_resolution", emit_mass_resolution, table_mass, "mass"),
             ("real_data", emit_real_data, table_realdata, "realdata"))
    for key, emit, table, name in later:
        if key not in have:
            missing.append(f"{key} -- {paths[key]}")
            continue
        check_inputs_unchanged(paths[key], root)
        d = json.loads(pathlib.Path(paths[key]).read_text())
        emit(em, d, paths[key])
        out[f"tables/{name}.tex"] = table(d)

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
