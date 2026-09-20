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
    ("label recovery across the contraction tree, five seeds",
     "experiments/FIGS/data/label_recovery_ladder_v1/s*/label_recovery.json"),
    ("anomaly detection, merged over signal models",
     "experiments/FIGS/data/anomaly_merged_v*.json"),
    ("community benchmarks (top tagging, quark/gluon), from experiments/FT/bench_metrics.py",
     "experiments/FIGS/data/bench_metrics*.json"),
    ("real data: CMS open-data jets",
     "experiments/FIGS/data/aoj_*.json"),
    ("mass-output models",
     "experiments/FIGS/data/mass_*.json"),
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
    return {"ladder": sorted((data / "probe_ladder_v2").glob("s*.json")),
            # analysis_with_c5, not analysis: the same run of the same script over
            # the same five ladder files, plus the mass-output 2x2 that makes C5
            # measurable. Verified a strict superset before it was adopted -- C1's
            # p reproduces to all sixteen digits and every other field is
            # identical; only the C5 block, the Holm family it joins and the
            # provenance differ. The earlier directory is kept as the record of
            # what the tables said while C5 was still pending.
            "analysis": data / "probe_ladder_v2" / "analysis_with_c5" / "seed_level_results.json",
            "leg1": data / "leg1_metrics.json",
            "leg2": data / "leg2_metrics.json",
            "recovery": data / "label_recovery_v3.json",
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
    for key in ("analysis", "leg1", "recovery"):
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
            em.macro("TostP" + key, fmt_p(top["p"]), src, path + ".largest_pair.p")
            em.macro("TostSmallestBound" + key, fmt(top["smallest_bound_passed"], 4), src,
                     path + ".largest_pair.smallest_bound_passed")

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
                if not r["estimable"] or reference not in (r["fine"], r["coarse"]):
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


# ------------------------------------------------------------------ tables

def _table(body: list[str], caption: str, label: str, colspec: str,
           notes: list[str], ncol: int, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{4pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += body
    lines.append("\\bottomrule")
    for note in notes:
        lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}p{{\\linewidth}}@{{}}}}"
                     f"{{\\footnotesize {note}}} \\\\")
    lines += ["\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def table_probe_ladder(A: dict, probe: str) -> str:
    """T1: granularities down the side, probe tasks across the top.

    Two rows per granularity rather than a two-line cell: the second column says
    which quantity the row holds, so the table needs nothing but booktabs.
    """
    tasks = ordered_tasks(A["levels"])
    levels = A["levels_fine_to_coarse"]
    eps = headline_rejection(A["levels"][tasks[0]][probe][0])["eps"]
    ncol = len(tasks) + 2
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
                  "r l " + "r" * len(tasks), notes, ncol, wide=True)


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
            f"TOST, largest pair ({top['coarse']} vs {top['fine']} classes), "
            f"{top['n_pairs']} seed pairs",
            f"${fmt(top['mean_diff'], 4, sign=True)}"
            + ("^{\\ast}$" if top["is_bound"] else "$"),
            fmt_p(top["p"]),
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
                  notes, 6, wide=True)


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
                  len(cols) + 2, wide=True)


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
    ncol = 1 + len(ns) * len(legs)
    return _table(head + rows, caption, "tab:finetuning-wave1",
                  "l " + " ".join("r" * len(ns) for _ in legs), notes, ncol, wide=True)


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
