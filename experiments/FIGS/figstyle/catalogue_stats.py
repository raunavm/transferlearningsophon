#!/usr/bin/env python3
"""Statistics of the figure-style catalogue, read from docs/figstyle/catalogue_rows/*.json.

    python3 experiments/FIGS/figstyle/catalogue_stats.py [tallies] [limits] [full] [open]

tallies: counts behind the one-page summary. limits: the PI's complexity limits, with verdicts computed
here. full: the one-row-per-figure tables and the precedent index. open: rows with estimates and internal
inconsistencies. No argument prints all four. Every judgement this script makes from free text is in a
named table or regex below, so that it can be audited against the rows.
"""
import collections
import glob
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROWS = ROOT / "docs/figstyle/catalogue_rows"
PNG = "docs/figstyle/precedents/{}/fig{}.png"

GROUPS = {
    "Sophon": {"2405.12972": "Sophon"},
    "FM": {"2401.13537": "Golling (MPM)", "2401.13536": "Vigl (fine-tuning)", "2403.05618": "OmniJet-alpha",
           "2403.07066": "RS3L", "2404.16091": "OmniLearn Letter", "2502.14652": "OmniLearn",
           "2510.24066": "OmniLearned", "2606.14813": "JetParticle-JEPA", "2606.14870": "Elsharkawy (objectives)",
           "2607.23377": "Uslu (scaling laws)", "2610.36121": "SuperJEPA"},
    "Tagger": {"1902.08570": "ParticleNet", "1902.09914": "Landscape", "2202.03772": "ParT"},
    "Experiment": {"2505.19689": "ATLAS GN2", "ATL-PHYS-PUB-2024-015": "ATLAS b-jet regression",
                   "2604.09809": "CMS JME-25-001"},
    "Vision": {"2410.23129": "Hong (label granularity)"},
    "Ours": {"map2026_poster": "MAP 2026 poster", "tl_deck_b": "Transfer-learning deck",
             "tl_deck_a": "Group poster (Reyes Casillas et al.)"},
}
NAME = {p: n for g in GROUPS.values() for p, n in g.items()}
GROUP = {p: g for g, papers in GROUPS.items() for p in papers}
ORDER = list(NAME)

# Figures from this number on are in an appendix or supplement (page of the figure against the heading).
APPENDIX_FROM = {"2401.13537": 7, "2401.13536": 8, "2403.05618": 8, "2606.14813": 5, "2606.14870": 13,
                 "ATL-PHYS-PUB-2024-015": 12, "2410.23129": 6}

# max_series_per_panel given as prose: the count of series in the fullest panel, reference lines not counted.
SERIES_TEXT = {
    "14 (8 step curves + 6 stacked": 14, "6 stacked filled histograms": 6, "4 lines (Sophon discr.": 4,
    "5 (filled target": 5, "8 + 1 horizontal reference": 8, "main 4 threshold series": 4,
    "4 bars per group": 4, "3 bars per group": 3, "main 4 (3 step lines": 4, "22 points in 4 groups": 4,
    "main 15 (5 models": 15, "left 2; right 2 + ~60": 62, "left 27 curves": 27, "136 circles + 5 squares": 2,
}

# Figures whose only non-solid line is one reference, baseline or diagonal, read from line_styles.
SINGLE_REFERENCE = {
    ("2405.12972", 5): "5 sigma limit",
    ("2510.24066", 4): "zero line in the ratio", ("2510.24066", 5): "zero line in the ratio",
    ("2410.23129", 2): "baseline", ("ATL-PHYS-PUB-2024-015", 1): "vertical line at 1",
    ("map2026_poster", 5): "y=x diagonal", ("tl_deck_b", "fig4_slide"): "y=x diagonal",
    **{("2606.14870", f): "from-scratch baseline" for f in (3, 6, 7, 8, 11, 15, 16, 17, 18, 19, 22, 23)},
}
NO_LINES = re.compile(r"^(n/a|no lines|no connecting|no centre line|no data lines|filled contours|marker outlines)")

# Candidate lists for the two precedent kinds the rows have no kind value for.
MASS_RESPONSE = {
    ("ATL-PHYS-PUB-2024-015", 2): "b-jet pT response median and resolution vs pT",
    ("ATL-PHYS-PUB-2024-015", 3): "b-jet pT response median and resolution vs muon count",
    ("ATL-PHYS-PUB-2024-015", 4): "b-jet pT response median and resolution vs track fraction",
    ("ATL-PHYS-PUB-2024-015", 7): "large-R mass and pT response distributions, ratio below",
    ("ATL-PHYS-PUB-2024-015", 8): "large-R mass response median and resolution vs pT",
    ("ATL-PHYS-PUB-2024-015", 9): "large-R pT response median and resolution vs pT",
    ("ATL-PHYS-PUB-2024-015", 10): "regressed mass spectra by process",
    ("2604.09809", 4): "reconstructed mass by algorithm, peak and FWHM in the legend",
    ("2403.05618", 4): "not a mass response: reconstruction closure in jet mass (OmniJet-alpha, closest FM figure)",
}
SCULPTING = {
    ("2604.09809", 8): "QCD mSD at background working points, ratio to inclusive",
    ("2604.09809", 9): "Jensen-Shannon distance of QCD mSD vs background efficiency and vs pT",
    ("2405.12972", 3): "mSD after discriminant cuts from 1 to 1e-4 (a peak-revealing figure, closest in Sophon)",
    ("2405.12972", 4): "mSD after the SALAD selection",
    ("2510.24066", 4): "mSD at anomaly-score thresholds with a fit and a ratio (closest FM figure)",
    ("2510.24066", 5): "same, score from the pretraining classes",
}
EXTRA_INDEX = {
    "roc": {("2401.13536", 1): "bottom panel is a ROC"},
    "metric vs size": {("2606.14813", 2): "panel (e) only", ("2606.14813", 15): "map of metric vs size and steps",
                       ("2410.23129", 2): "metric vs number of pretraining classes, the nearest x axis to ours",
                       ("2410.23129", 5): "metric vs number of pretraining classes"},
    "heatmap or confusion matrix": {("2606.14813", 2): "panels (f,g) only",
                                    ("2610.36121", 2): "corner plot of densities"},
}


def load():
    rows = []
    for fn in glob.glob(str(ROWS / "*.json")):
        rows += json.load(open(fn, encoding="utf-8"))
    missing = {r["paper"] for r in rows} - set(ORDER)
    assert not missing, missing
    rows.sort(key=lambda r: (ORDER.index(r["paper"]), fig_order(r["fig"])))
    return rows


def fig_order(f):
    return int(re.search(r"\d+", str(f)).group())


def key(r):
    return (r["paper"], r["fig"])


def is_data(r):
    return r["kind"] not in ("schematic", "table_like")


def in_appendix(r):
    return fig_order(r["fig"]) >= APPENDIX_FROM.get(r["paper"], 10 ** 6)


def num(v):
    """First number in a value, or None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return next((n for n in map(num, v.values()) if n is not None), None)
    m = re.search(r"\d+\.\d+|\d+", str(v))
    return float(m.group()) if m else None


def main_panels(r):
    """Data panels without ratio axes. A few rows count the ratio axes in n_panels; undo that here."""
    m = re.search(r"1 main(?: panel)? over (\d)", r["arrangement"])
    return r["n_panels"] // (1 + int(m.group(1))) if m else r["n_panels"]


def series(r):
    v = r["max_series_per_panel"]
    if isinstance(v, int):
        return v
    if v.startswith("n/a"):
        return None
    for start, n in SERIES_TEXT.items():
        if v.startswith(start):
            return n
    raise ValueError((key(r), v))


def line_class(r):
    s = str(r["line_styles"]).lower()
    if NO_LINES.match(s) or r["kind"] == "bar" and "bar edge" in s:
        return "no lines"
    if not re.search(r"dash|dott", s):
        return "solid only"
    return "single reference" if key(r) in SINGLE_REFERENCE else "non-solid series"


def uncert(r):
    return r["uncertainty"]["type"]


def ratio(r):
    return bool(r["ratio_panel"]["present"])


def ratio_height(r):
    """Height of one ratio axes over the main axes; RS3L Fig. 3 gives it as '6 strips each 0.28'."""
    h = r["ratio_panel"]["height_relative_to_main"]
    m = re.search(r"each (\d\.\d+)", str(h))
    return float(m.group(1)) if m else num(h)


def legend_place(r):
    s = str(r["legend"]["location"])
    if s.startswith("inside") and "outside" in s:
        return "mixed"
    return s.split()[0]


def legend_corner(r):
    s = str(r["legend"]["corner"]).lower()
    v = re.search(r"upper|top|lower|bottom|centre", s)
    h = re.search(r"left|right|centre", s)
    if not v and not h:
        return "other"
    return " ".join(dict.fromkeys(x for x in (v and v.group().replace("top", "upper").replace("bottom", "lower"),
                                              h and h.group()) if x))


def legend_numbers(r):
    v = r["legend"]["numbers_in_entries"]
    return v is True or (isinstance(v, str) and v.lower().startswith("yes"))


def legend_results(r):
    """Numbers in the entries that are results, not sample sizes or labels."""
    t = json.dumps(r["legend"], ensure_ascii=False)
    return legend_numbers(r) and bool(re.search(r"AUC|significance|entropy|mu=|chi|\u03c7|FWHM", t))


def mark(r):
    """(kind of mark, text). Kinds: beyond-range mark, limit line, projection mark, or None."""
    t = r["truncation_or_limit_marks"]
    what = r.get("truncation_what", "")
    if isinstance(t, dict):
        present, what = t["present"], t["what"]
    elif isinstance(t, str):
        present, what = True, t
    else:
        present = t
    if not present:
        return None, what
    if "triangle" in what.lower():
        return "beyond-range mark", what
    if "projection" in what.lower():
        return "projection mark", what
    return "limit line", what


def silent_clip(r):
    """Data clipped by the axes with no mark, from the truncation fields or the notes of a data figure."""
    if not is_data(r) or mark(r)[0]:
        return False
    t = f"{mark(r)[1]} {r['notes']}".lower()
    return bool(re.search(r"clipped(?! from)|no truncation mark|run off|runs off|leave[^.]*no mark|silently|no extend arrow", t))


def font_family(r):
    s = str(r["fonts"]["family"])
    for pat, name in (("DejaVu Serif", "DejaVu Serif"), ("Heros|Nimbus|Helvetica", "Helvetica-like"),
                      ("Palatino", "Palatino"), ("STIX", "STIX"), ("DejaVu", "DejaVu Sans"),
                      ("Computer Modern|CMR", "Computer Modern")):
        if re.search(pat, s):
            return name
    return "unknown"


def grid_class(r):
    g = str(r["axis"].get("grid")).lower()
    if g.startswith(("none", "n/a")):
        return "none"
    extent = "major+minor" if "minor" in g and "major" in g else "major"
    style = "dashed or dotted" if re.search(r"dash|dott", g) else "solid"
    return f"{extent}, {style}"


def minor_ticks(r):
    v = r["axis"].get("minor_ticks")
    return not (v in (False, None) or str(v).lower() in ("false", "n/a", "none"))


def tick_dir(r):
    s = str(r["axis"].get("tick_direction")).lower()
    m = re.match(r"(?:\(a-c\) no tick marks; \(d\) |colourbar )?(out|in)\b", s)
    return m.group(1) if m else "n/a"


def four_sided(r):
    return "four sides" in str(r["axis"].get("tick_direction")).lower()


def y_scale(r):
    s = str(r["axis"].get("y_scale")).lower()
    return "log" if s.startswith("log") or ": log" in s[:12] else "linear" if "linear" in s else "other"


def label_at_end(r):
    t = (str(r["axis"].get("x_label")) + " " + r["notes"]).replace("sits right-aligned", "")
    return bool(re.search(r"right end|right-aligned|axis end|end of their axes|ha='right'", t))


def units_style(r):
    t = str(r["axis"].get("x_label")) + str(r["axis"].get("y_label"))
    if re.search(r"\[(GeV|mm|%)\]", t):
        return "[GeV]"
    if "(GeV)" in t:
        return "(GeV)"
    return "no unit"


def panel_label(r):
    s = str(r["panel_labels"])
    if re.search(r"\(a\)|\ba\) ", s) and "no (a)" not in s and "caption says" not in s:
        return "letters"
    if re.search(r"title|strip label|inside", s):
        return "titles or in-panel text"
    return "none"


def ticks_xy(r):
    d = r["axis"].get("tick_density")
    if isinstance(d, dict):
        return num(d.get("x_major", d.get("x"))), num(d.get("y_major", d.get("y", d.get("y_main"))))
    s = str(d)
    both = re.search(r"(\d+) major (?:per|each) axis", s)
    if both:
        return float(both.group(1)), float(both.group(1))
    x, y = re.search(r"\bx\s*(\d+)", s), re.search(r"\by\s*(\d+)", s)
    return (float(x.group(1)) if x else None), (float(y.group(1)) if y else None)


def tab10(r):
    hexes = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}", json.dumps(r["palette_hex"]))}
    return len(hexes & {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                        "#bcbd22", "#17becf"}) >= 2


def markers(r):
    return not re.match(r"(none\b(?! on the line)|n/a)", str(r["markers"]))


# ---------------------------------------------------------------------- helpers for printing


def figs_str(rows, val=None):
    """'Paper: 8 (10), 9 (10); Other: 3 (6)', one entry per paper, value in brackets when val is given."""
    out = collections.OrderedDict()
    for r in rows:
        out.setdefault(NAME[r["paper"]], []).append(f"{r['fig']}" + (f" ({val(r)})" if val else ""))
    return "; ".join(f"{p} Fig. {', '.join(v)}" for p, v in out.items()).replace("Fig. fig", "fig") or "none"


def share(k, n):
    return f"{k} of {n}" + (f" ({100 * k / n:.0f}%)" if n else "")


def med_range(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    return f"{statistics.median(vals):g} ({min(vals):g}-{max(vals):g})"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    return "\n".join(out)


def by_group(rows, group):
    return [r for r in rows if GROUP[r["paper"]] == group]


def verdict_exceed(rows, val, limit):
    """Largest value, the figure that has it, and the figures above the limit."""
    vals = [(val(r), r) for r in rows if val(r) is not None]
    if not vals:
        return None, None, []
    v, r = max(vals, key=lambda t: t[0])
    return v, r, [x for n, x in vals if n > limit]


# ---------------------------------------------------------------------- tallies for the summary


def categorical(rows, name, fn):
    data = {}
    for label, part in (("Sophon", by_group(rows, "Sophon")), ("FM", by_group(rows, "FM"))):
        c = collections.Counter(str(fn(r)) for r in part)
        papers = collections.defaultdict(list)
        for r in part:
            papers[r["paper"]].append(str(fn(r)))
        n = len(part)
        items = []
        for v, k in c.most_common():
            used = sum(v in x for x in papers.values())
            major = sum(x.count(v) > len(x) / 2 for x in papers.values())
            items.append(f"{v}: {k}/{n} figs, in {used} papers, majority in {major}")
        data[label] = "; ".join(items)
    print(f"- **{name}**\n  - Sophon: {data['Sophon']}\n  - FM ({len(by_group(rows, 'FM'))} figs): {data['FM']}")


def per_paper(rows, name, fn):
    print(f"- **{name}** median (range) per paper:")
    for g, papers in GROUPS.items():
        parts = [f"{n} {med_range([fn(r) for r in rows if r['paper'] == p])}" for p, n in papers.items()
                 if any(r["paper"] == p for r in rows)]
        print(f"  - {g}: all {med_range([fn(r) for r in by_group(rows, g)])}. " + "; ".join(parts))


def tallies(rows):
    data = [r for r in rows if is_data(r)]
    print("## Tallies behind the summary (data figures only unless stated)\n")
    print(f"Rows {len(rows)}; data {len(data)}; schematic or table-like {len(rows) - len(data)}.")
    for g, papers in GROUPS.items():
        print(f"- {g}: {len(papers)} papers, {len(by_group(rows, g))} figures, {len(by_group(data, g))} data figures")
    print("- kinds in the Sophon rows:", [(r["fig"], r["kind"]) for r in by_group(rows, "Sophon")])
    print()
    per_paper(data, "width fraction", lambda r: num(r["width_fraction_of_text_block"]))
    per_paper(data, "aspect", lambda r: num(r["aspect_w_over_h"]))
    per_paper(data, "main panels", main_panels)
    per_paper(data, "series per panel", series)
    per_paper(data, "line width pt", lambda r: num(r["line_width_pt"]))
    per_paper(data, "label pt", lambda r: num(r["fonts"]["label_pt"]))
    per_paper(data, "tick pt", lambda r: num(r["fonts"]["tick_pt"]))
    per_paper(data, "legend pt", lambda r: num(r["fonts"]["legend_pt"]))

    def rel(a, b):
        return lambda r: (num(r["fonts"][a]) / num(r["fonts"][b])
                          if num(r["fonts"][a]) and num(r["fonts"][b]) else None)
    per_paper(data, "label/tick", rel("label_pt", "tick_pt"))
    per_paper(data, "legend/tick", rel("legend_pt", "tick_pt"))
    per_paper(data, "legend/tick as in the row", lambda r: num(r["legend"]["font_ratio_to_ticks"]))
    per_paper(data, "x major ticks", lambda r: ticks_xy(r)[0])
    per_paper(data, "y major ticks", lambda r: ticks_xy(r)[1])
    per_paper([r for r in data if ratio(r)], "ratio height / main", ratio_height)
    print()
    categorical(data, "width class", lambda r: r["width_class"])
    categorical(data, "width at most 0.6", lambda r: num(r["width_fraction_of_text_block"]) <= 0.6)
    categorical(data, "main panels", lambda r: main_panels(r) if main_panels(r) <= 4 else "above 4")
    categorical(data, "line style", line_class)
    categorical(data, "markers on data", markers)
    categorical(data, "colour alone separates", lambda r: r["colour_alone_distinguishes"])
    categorical(data, "tab10 hexes (2 or more)", tab10)
    categorical(data, "legend place", legend_place)
    categorical(data, "legend corner", lambda r: legend_corner(r) if legend_place(r) not in ("none", "n/a") else "no legend")
    categorical(data, "legend frame", lambda r: r["legend"]["frame"] if legend_place(r) not in ("none", "n/a") else "no legend")
    categorical(data, "numbers in legend entries", legend_numbers)
    categorical(data, "result numbers in legend entries", legend_results)
    categorical(data, "y scale", y_scale)
    categorical(data, "unit style", units_style)
    categorical(data, "axis label at the axis end", label_at_end)
    categorical(data, "grid", grid_class)
    categorical(data, "minor ticks", minor_ticks)
    categorical(data, "tick direction", tick_dir)
    categorical(data, "ticks on four sides", four_sided)
    categorical(data, "font family", font_family)
    categorical(data, "uncertainty", uncert)
    categorical(data, "ratio panel", ratio)
    categorical(data, "panel labels", panel_label)
    categorical(data, "hatching", lambda r: r["hatching"])
    x = collections.Counter(str(r["axis"].get("x_label")) for r in data)
    print("- commonest x labels:", x.most_common(8))
    y = collections.Counter(str(r["axis"].get("y_label")) for r in data)
    print("- commonest y labels:", y.most_common(8))
    print()


# ---------------------------------------------------------------------- the limits


def limit_panels(rows):
    print("### (i) At most four panels per figure\n")
    print("Counted on data figures, main axes only: ratio and side axes are not panels. Rows that counted "
          "their ratio axes in n_panels are corrected here (section 4 lists them). Corner plots and small multiples count every cell.\n")
    table = []
    for g in GROUPS:
        data = [r for r in by_group(rows, g) if is_data(r)]
        v, top, over = verdict_exceed(data, main_panels, 4)
        mt = [r for r in data if not in_appendix(r)]
        vm, topm, overm = verdict_exceed(mt, main_panels, 4)
        table.append([g, len(data), f"{v} ({NAME[top['paper']]} Fig. {top['fig']})",
                      share(len(over), len(data)), f"{vm} ({NAME[topm['paper']]} Fig. {topm['fig']})",
                      share(len(overm), len(mt)), "holds" if not over else "**refuted**"])
    print(md_table(["group", "data figures", "maximum", "above 4", "maximum outside appendices",
                    "above 4 outside appendices", "limit"], table))
    print()
    for g in ("FM", "Tagger", "Experiment", "Ours"):
        over = [r for r in by_group(rows, g) if is_data(r) and main_panels(r) > 4]
        print(f"- {g} above four: {figs_str(over, main_panels)}")
    print(f"- Schematics, not counted above: maximum {max(main_panels(r) for r in rows if not is_data(r))} "
          f"({figs_str([r for r in rows if not is_data(r) and main_panels(r) >= 5])}).\n")
    print("Largest panel and series counts in the foundation-model papers, by kind of figure:\n")
    table = []
    for k in sorted({r["kind"] for r in rows if is_data(r)}):
        part = [r for r in by_group(rows, "FM") if r["kind"] == k]
        if part:
            a, ra, _ = verdict_exceed(part, main_panels, 4)
            b, rb, _ = verdict_exceed(part, series, 6)
            table.append([k, len(part), f"{a} ({NAME[ra['paper']]} Fig. {ra['fig']})",
                          f"{b} ({NAME[rb['paper']]} Fig. {rb['fig']})" if rb else "n/a"])
    print(md_table(["kind", "figures", "most panels", "most series"], table))
    print()


def limit_series(rows):
    print("### (ii) At most six series per panel\n")
    print("Series in the fullest panel, reference lines not counted. Figures with no series (heatmaps, jet displays, "
          "schematics) are left out.\n")
    table = []
    for g in GROUPS:
        data = [r for r in by_group(rows, g) if is_data(r) and series(r) is not None]
        v, top, over = verdict_exceed(data, series, 6)
        table.append([g, len(data), f"{v} ({NAME[top['paper']]} Fig. {top['fig']})", share(len(over), len(data)),
                      "holds" if not over else "**refuted**"])
    print(md_table(["group", "figures with series", "maximum", "above 6", "limit"], table))
    print()
    for g in ("Sophon", "FM", "Tagger", "Experiment", "Ours"):
        over = [r for r in by_group(rows, g) if is_data(r) and (series(r) or 0) > 6]
        print(f"- {g} above six: {figs_str(over, series)}")
    print("- Prose entries in the rows were mapped by SERIES_TEXT in the script. Uslu Fig. 2 (136 circles + 5 squares, "
          "one point per run) counts as 2 marker families; Elsharkawy Fig. 5 counts its ~60 faint trajectories.\n")


def limit_lines(rows):
    print("### (iii) Solid lines, one colour per series, a dashed line only for a single reference\n")
    data = [r for r in rows if is_data(r)]
    table = []
    for g in GROUPS:
        part = by_group(data, g)
        c = collections.Counter(line_class(r) for r in part)
        lines = [r for r in part if line_class(r) != "no lines"]
        ok = [r for r in lines if line_class(r) in ("solid only", "single reference")
              and r["colour_alone_distinguishes"] is True]
        table.append([g, len(lines), c["solid only"], c["single reference"], c["non-solid series"],
                      share(len(ok), len(lines)), "holds" if not c["non-solid series"] else "**refuted**"])
    print(md_table(["group", "figures with lines", "solid only", "one non-solid reference",
                    "non-solid data series", "solid or one reference, and colour alone separates series", "limit"], table))
    print()
    for g in ("Sophon", "FM", "Tagger", "Experiment", "Vision", "Ours"):
        print(f"- {g}, a non-solid line carries data: {figs_str([r for r in by_group(data, g) if line_class(r) == 'non-solid series'])}")
    print(f"- One non-solid reference: {figs_str([r for r in data if line_class(r) == 'single reference'])}")
    s = [r for r in by_group(data, "Sophon") if line_class(r) == "non-solid series"]
    print(f"- Sophon itself breaks this limit in {figs_str(s)}: {s[0]['line_styles']}\n")


def is_roc(r):
    return r["kind"] == "roc" or key(r) == ("2401.13536", 1)


def limit_ratio(rows):
    print("### (iv) No ratio panels unless the figure of that kind has one\n")
    print("Every ROC figure in the catalogue. Sophon Fig. 2 is filed as kind roc in its row, but its axes are "
          "significance against background efficiency, so it is the nearest thing to a ROC and not a ROC.\n")
    table = []
    for r in rows:
        if not is_roc(r):
            continue
        rp = r["ratio_panel"]
        h = rp["height_relative_to_main"] if rp["present"] else "-"
        m = re.search(r"over (\d|three) stacked", r["arrangement"])
        each = rp.get("per_panel", (m.group(1).replace("three", "3") if m else 1)) if rp["present"] else "-"
        table.append([GROUP[r["paper"]], NAME[r["paper"]], r["fig"], "yes" if rp["present"] else "no", each, h,
                      str(r["axis"].get("x_label"))[:40], str(r["axis"].get("y_label"))[:40]])
    print(md_table(["group", "paper", "fig", "ratio panel", "ratio axes per main", "height / main", "x", "y"], table))
    print()
    for g in GROUPS:
        part = [r for r in by_group(rows, g) if is_roc(r)]
        if part:
            print(f"- {g}: {share(sum(ratio(r) for r in part), len(part))} ROC figures have a ratio panel.")
    print("\nRatio panels on all data figures, by kind:\n")
    table = []
    kinds = sorted({r["kind"] for r in rows if is_data(r)})
    for k in kinds:
        cells = []
        for g in GROUPS:
            part = [r for r in by_group(rows, g) if r["kind"] == k]
            cells.append(f"{sum(ratio(r) for r in part)}/{len(part)}" if part else "-")
        table.append([k] + cells)
    print(md_table(["kind"] + list(GROUPS), table))
    print()


def limit_bands(rows):
    print("### (v) No uncertainty bands by default\n")
    table = []
    for g in GROUPS:
        data = [r for r in by_group(rows, g) if is_data(r)]
        c = collections.Counter(uncert(r) for r in data)
        table.append([g, len(data), c["band"], c["bars"], c["none"] + c["n/a"],
                      "holds" if not c["band"] else "**refuted**"])
    print(md_table(["group", "data figures", "band", "bars", "none", "limit"], table))
    print("\nBy kind of figure (band / bars / none):\n")
    table = []
    for k in sorted({r["kind"] for r in rows if is_data(r)}):
        cells = []
        for g in GROUPS:
            part = [r for r in by_group(rows, g) if r["kind"] == k]
            c = collections.Counter(uncert(r) for r in part)
            cells.append(f"{c['band']}/{c['bars']}/{c['none'] + c['n/a']}" if part else "-")
        table.append([k] + cells)
    print(md_table(["kind"] + list(GROUPS), table))
    print("\nWhere the bands are:\n")
    undefined = re.compile(r"not stated|not defined|does not say|meaning is unknown|not described|says nothing|not resolved")
    table = []
    for r in rows:
        if uncert(r) == "band":
            rep = r["uncertainty"]["represents"]
            table.append([GROUP[r["paper"]], NAME[r["paper"]], r["fig"], r["kind"],
                          "no" if undefined.search(rep) else "yes", rep[:110]])
    print(md_table(["group", "paper", "fig", "kind", "defined by the paper", "what the row says (first words)"], table))
    print(f"\nBands the paper does not define: {sum(t[4] == 'no' for t in table)} of {len(table)}.\n")


def limit_marks(rows):
    print("### (vi) No hatching, no offset markers, no truncation lines\n")
    data = [r for r in rows if is_data(r)]
    table = []
    for g in GROUPS:
        part = by_group(rows, g)
        marks = [r for r in part if mark(r)[0]]
        hatch = [r for r in part if r["hatching"]]
        off = [r for r in part if re.search(r"(?<!not )(?<!no x )offset", str(r["markers"]))]
        table.append([g, len(part), f"{sum(is_data(r) for r in hatch)} + {sum(not is_data(r) for r in hatch)}", len(off),
                      sum(mark(r)[0] == "beyond-range mark" for r in marks),
                      sum(mark(r)[0] == "limit line" for r in marks), sum(mark(r)[0] == "projection mark" for r in marks),
                      sum(silent_clip(r) for r in part)])
    print(md_table(["group", "figures", "hatching (data + schematic)", "offset markers", "beyond-range marks", "limit lines",
                    "projection marks", "clipped with no mark (field or notes)"], table))
    print()
    print(f"- Hatching: {figs_str([r for r in rows if r['hatching']])}. Uses: Elsharkawy, hatch carries model or "
          "dataset size on bars; JetParticle-JEPA Figs. 16-20, a hatch marks a wrong prediction in an inset bar, Fig. 4 "
          "token glyphs in a schematic; CMS Fig. 11, a hatched uncertainty band.")
    stated = sum("offset" in str(r["markers"]) for r in rows)
    silent = [r for r in data if markers(r) and "offset" not in str(r["markers"])]
    print(f"- Offset markers: no row records one. {stated} rows say not offset in the markers field; "
          f"{len(silent)} data figures with markers do not mention it: {figs_str(silent)}.")
    for kind in ("beyond-range mark", "limit line", "projection mark"):
        part = [r for r in rows if mark(r)[0] == kind]
        print(f"- {kind}: {figs_str(part)}. {mark(part[0])[1][:150]}")
    print(f"- Clipped with no mark: {figs_str([r for r in rows if silent_clip(r)])}. A lower bound: it comes from "
          "free-text fields, which the row authors did not fill to one rule.\n")


def limit_legend(rows):
    print("### (vii) Legend inside the axes\n")
    table = []
    for g in GROUPS:
        data = [r for r in by_group(rows, g) if is_data(r)]
        c = collections.Counter(legend_place(r) for r in data)
        with_leg = c["inside"] + c["outside"] + c["mixed"]
        table.append([g, len(data), c["inside"], c["outside"], c["mixed"], c["none"] + c["n/a"],
                      "holds" if not c["outside"] + c["mixed"] else "**refuted**"])
    print(md_table(["group", "data figures", "inside", "outside", "mixed", "no legend", "limit"], table))
    print()
    print(f"- Outside or mixed: {figs_str([r for r in rows if is_data(r) and legend_place(r) in ('outside', 'mixed')])}\n")


def limit_captions(rows):
    print("### (viii) Captions of 40-80 words saying what is plotted\n")
    print("Word counts over rows with a caption (the deck has none). Data figures and schematics apart. \"Method\" is the "
          "share of captions whose row says also_method.\n")
    table = []

    def line(label, part):
        out = []
        for sel in (is_data, lambda r: not is_data(r)):
            p = [r for r in part if sel(r) and r["caption_words"] > 0]
            if not p:
                out += ["-", "-", "-", "-"]
                continue
            w = [r["caption_words"] for r in p]
            out += [len(p), med_range(w), share(sum(40 <= x <= 80 for x in w), len(p)),
                    share(sum(r["caption_describes"] == "also_method" for r in p), len(p))]
        return [label] + out
    for g, papers in GROUPS.items():
        for p, n in papers.items():
            table.append(line(n, [r for r in rows if r["paper"] == p]))
        table.append(line(f"**{g} all**", by_group(rows, g)))
    print(md_table(["paper", "data n", "words, median (range)", "within 40-80", "method",
                    "schematic n", "words, median (range)", "within 40-80", "method"], table))
    print()
    for g in GROUPS:
        p = [r for r in by_group(rows, g) if is_data(r) and r["caption_words"] > 0]
        v, top, over = verdict_exceed(p, lambda r: r["caption_words"], 80)
        lo = min(p, key=lambda r: r["caption_words"])
        under = [r for r in p if r["caption_words"] < 40]
        print(f"- {g}: longest data caption {v} ({NAME[top['paper']]} Fig. {top['fig']}), shortest {lo['caption_words']} "
              f"({NAME[lo['paper']]} Fig. {lo['fig']}); {len(over)} above 80, {len(under)} below 40.")
    print()


def limit_summary(rows):
    """One line per limit: the verdict for each group and the largest value, computed from the rows."""
    def cell(g, fn):
        return fn([r for r in by_group(rows, g) if is_data(r)])

    def maxim(val, limit):
        def f(part):
            v, top, over = verdict_exceed(part, val, limit)
            if top is None:
                return "no instance"
            where = f"Fig. {top['fig']}" if GROUP[top["paper"]] == "Sophon" else f"{NAME[top['paper']]} Fig. {top['fig']}"
            n = sum(val(r) is not None for r in part)
            return f"{'holds' if not over else 'REFUTED'}: max {v} ({where}), {len(over)} of {n} above"
        return f

    def count(test, what):
        def f(part):
            hits = [r for r in part if test(r)]
            return f"{'holds' if not hits else 'REFUTED'}: {what} in {len(hits)} of {len(part)}"
        return f

    def caption(part):
        p = [r for r in part if r["caption_words"] > 0]
        w = [r["caption_words"] for r in p]
        bad = sum(not 40 <= x <= 80 for x in w)
        return (f"{'holds' if not bad else 'REFUTED'}: median {statistics.median(w):g}, {min(w)}-{max(w)}, "
                f"{bad} of {len(p)} outside; method {sum(r['caption_describes'] == 'also_method' for r in p)} of {len(p)}")

    def roc(part):
        p = [r for r in part if is_roc(r)]
        if not p:
            return "no ROC figure"
        return f"ratio panel on {sum(ratio(r) for r in p)} of {len(p)} ROC figures"

    def marks_cell(part):
        c = collections.Counter(mark(r)[0] for r in part)
        return (f"beyond-range triangles in {c['beyond-range mark']}, limit reference line in {c['limit line']}, "
                f"clipped silently in at least {sum(silent_clip(r) for r in part)}, of {len(part)}")

    limits_ = [
        ("(i) at most 4 panels", maxim(main_panels, 4)),
        ("(ii) at most 6 series per panel", maxim(series, 6)),
        ("(iii) solid; dashed only for one reference", count(lambda r: line_class(r) == "non-solid series", "non-solid data series")),
        ("(iv) ratio panel only if the kind has one: ROC", roc),
        ("(v) no bands by default", count(lambda r: uncert(r) == "band", "band")),
        ("(vi) no hatching", count(lambda r: r["hatching"], "hatching")),
        ("(vi) no offset markers", count(lambda r: re.search(r"(?<!not )(?<!no x )offset", str(r["markers"])), "offset")),
        ("(vi) no truncation lines: no row has a line marking a cut range; nearest marks", marks_cell),
        ("(vii) legend inside", count(lambda r: legend_place(r) in ("outside", "mixed"), "outside or mixed")),
        ("(viii) captions 40-80 words, data figures", caption),
    ]
    print(md_table(["limit", "Sophon", "FM papers", "taggers", "experiments"],
                   [[name] + [cell(g, fn) for g in ("Sophon", "FM", "Tagger", "Experiment")] for name, fn in limits_]))
    print()


def limits(rows):
    limit_summary(rows)
    limit_panels(rows)
    limit_series(rows)
    limit_lines(rows)
    limit_ratio(rows)
    limit_bands(rows)
    limit_marks(rows)
    limit_legend(rows)
    limit_captions(rows)


# ---------------------------------------------------------------------- the full table


def cell(v, n=None):
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    s = str(v).replace("\n", " ").strip()
    return s if n is None or len(s) <= n else s[:n] + "..."


def full_row(r):
    a, f, l, u, rp = r["axis"], r["fonts"], r["legend"], r["uncertainty"], r["ratio_panel"]
    numbers = next((l[k] for k in ("numbers_which", "numbers_note", "numbers") if l.get(k) not in (None, "", "n/a")),
                   l["numbers_in_entries"])
    what = mark(r)
    ratio_cell = "no" if not rp["present"] else f"{rp['height_relative_to_main']}" + (
        f", {rp['per_panel']} per main" if "per_panel" in rp else "")
    return [r["fig"], r["kind"], f"{r['width_fraction_of_text_block']} {r['width_class']}", r["aspect_w_over_h"],
            f"{main_panels(r)}; {r['arrangement']}", cell(r["max_series_per_panel"]), cell(r["line_styles"]),
            cell(r["line_width_pt"]), cell(r["markers"]), cell(", ".join(map(str, r["palette_hex"]))
                                                              if isinstance(r["palette_hex"], list) else r["palette_hex"]),
            r["colour_alone_distinguishes"], l["location"], cell(l["corner"]), l["frame"], l["font_ratio_to_ticks"],
            cell(numbers), f"{cell(a.get('x_label'))}; {cell(a.get('x_scale'))}",
            f"{cell(a.get('y_label'))}; {cell(a.get('y_scale'))}", cell(a.get("units")), cell(a.get("tick_density")),
            cell(a.get("grid")), cell(a.get("minor_ticks")), cell(a.get("tick_direction")), cell(f["family"]),
            f"{cell(f['title_pt'])} / {cell(f['label_pt'])} / {cell(f['tick_pt'])} / {cell(f['legend_pt'])}",
            f"{u['type']}: {cell(u['represents'], 140)}", ratio_cell, cell(r["panel_labels"]),
            r["caption_words"], r["caption_describes"], f"{what[0]}: {cell(what[1], 80)}" if what[0] else "no",
            r["hatching"], PNG.format(r["paper"], r["fig"]).replace("figfig", "fig")]


HEADER = ["fig", "kind", "width (of text block), class", "aspect w/h", "main panels; arrangement", "series per panel",
          "line styles", "line width pt", "markers", "palette", "colour alone", "legend place", "legend corner", "frame",
          "legend/tick", "numbers in entries", "x label; scale", "y label; scale", "units", "tick density", "grid",
          "minor ticks", "tick direction", "font", "title/label/tick/legend pt", "uncertainty", "ratio height",
          "panel labels", "caption words", "caption", "marks", "hatching", "image"]


def full(rows):
    print("### 3.1 Every figure, by paper in authority order\n")
    for g, papers in GROUPS.items():
        print(f"#### {g}\n")
        for p, n in papers.items():
            part = [r for r in rows if r["paper"] == p]
            print(f"**{n}** ({p}), {len(part)} figures\n")
            print(md_table(HEADER, [full_row(r) for r in part]))
            print()
    print("### 3.2 Precedent index by figure kind\n")
    index = {
        "roc": ([r for r in rows if r["kind"] == "roc"], lambda r: "ratio panel" if ratio(r) else "no ratio panel"),
        "metric vs size": ([r for r in rows if r["kind"] == "metric_vs_size"], lambda r: uncert(r)),
        "heatmap or confusion matrix": ([r for r in rows if r["kind"] in ("heatmap", "confusion_matrix")], lambda r: r["kind"]),
        "bar": ([r for r in rows if r["kind"] == "bar"], lambda r: "hatched" if r["hatching"] else "no hatching"),
    }
    by_key = {key(r): r for r in rows}
    for name, (part, note) in index.items():
        print(f"**{name}**\n")
        table = [[GROUP[r["paper"]], NAME[r["paper"]], r["fig"], note(r), r["width_fraction_of_text_block"],
                  r["aspect_w_over_h"], main_panels(r), PNG.format(r["paper"], r["fig"]).replace("figfig", "fig")] for r in part]
        for k, why in EXTRA_INDEX.get(name, {}).items():
            r = by_key[k]
            table.append([GROUP[r["paper"]], NAME[r["paper"]], r["fig"], why, r["width_fraction_of_text_block"],
                          r["aspect_w_over_h"], main_panels(r), PNG.format(r["paper"], r["fig"]).replace("figfig", "fig")])
        table.sort(key=lambda t: (list(GROUPS).index(t[0]), ORDER.index(next(p for p, n in NAME.items() if n == t[1]))))
        print(md_table(["group", "paper", "fig", "note", "width", "aspect", "panels", "image"], table))
        print()
    for name, chosen in (("mass response", MASS_RESPONSE), ("sculpting", SCULPTING)):
        print(f"**{name}**\n")
        table = [[GROUP[k[0]], NAME[k[0]], k[1], why, by_key[k]["width_fraction_of_text_block"],
                  by_key[k]["aspect_w_over_h"], main_panels(by_key[k]), PNG.format(k[0], k[1])]
                 for k, why in chosen.items()]
        print(md_table(["group", "paper", "fig", "note", "width", "aspect", "panels", "image"], table))
        print()


# ---------------------------------------------------------------------- what is open


def walk(v, path=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from walk(x, f"{path}.{k}" if path else k)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x, path)
    elif isinstance(v, str):
        yield path, v


def open_items(rows):
    print("### Rows with \"~\" values, by paper\n")
    table = []
    for p in ORDER:
        fields = collections.defaultdict(list)
        for r in rows:
            if r["paper"] != p:
                continue
            for path, s in walk({k: v for k, v in r.items() if k != "notes"}):
                if "~" in s:
                    fields[path.split(".")[0] if path.startswith(("fonts", "legend")) else path].append(r["fig"])
        if fields:
            table.append([NAME[p], "; ".join(f"{k} (Fig. {', '.join(map(str, dict.fromkeys(v)))})" for k, v in fields.items())])
    print(md_table(["paper", "field (figures)"], table))
    print("\n### Notes of uncertainty in the rows, by paper\n")
    pats = {"not stated or not defined by the paper": r"not stated|not defined|does not define|meaning is unknown|undefined|not explained|never explain|does not say",
            "estimated or by eye": r"by eye|estimate|approximate|pixel estimate",
            "inferred": r"inferred|\[I\]|\[K\]",
            "cannot be checked or not resolved": r"cannot be|not resolved|unresolved|not identifiable|cannot"}
    table = []
    for p in ORDER:
        cells = [NAME[p]]
        for pat in pats.values():
            hit = [r["fig"] for r in rows if r["paper"] == p and re.search(pat, json.dumps(r, ensure_ascii=False))]
            cells.append(", ".join(map(str, hit)) or "-")
        table.append(cells)
    print(md_table(["paper"] + list(pats), table))
    print("\n### Inconsistencies found by this script\n")
    flagged = [r for r in rows if r["colour_alone_distinguishes"] is True and "same hue" in str(r["line_styles"])]
    for r in flagged:
        print(f"- {NAME[r['paper']]} Fig. {r['fig']}: colour_alone_distinguishes is true, but non-solid lines carry "
              f"data ({r['line_styles'][:100]}).")
    fixed = [r for r in rows if main_panels(r) != r["n_panels"]]
    print(f"- n_panels counts ratio axes in {len(fixed)} rows, divided out by main_panels(): {figs_str(fixed)}.")
    kinds = collections.Counter(type(r["truncation_or_limit_marks"]).__name__ for r in rows)
    print(f"- truncation_or_limit_marks has mixed types across the row files: {dict(kinds)}.")
    types = collections.Counter(type(r["legend"]["numbers_in_entries"]).__name__ for r in rows)
    print(f"- legend.numbers_in_entries has mixed types: {dict(types)}.")
    small = [r for r in rows if is_data(r) and (num(r["fonts"]["tick_pt"]) or 99) < 3.5]
    over = [r for r in rows if num(r["width_fraction_of_text_block"]) > 1]
    print(f"- Width above 1 of the text block (crop slack in the extractor; JetParticle-JEPA rows round the same case to 1.0): "
          f"{figs_str(over, lambda r: r['width_fraction_of_text_block'])}.")
    print(f"- Printed tick text under 3.5 pt, a precedent not to copy: {figs_str(small, lambda r: num(r['fonts']['tick_pt']))}.")
    print()


if __name__ == "__main__":
    rows = load()
    want = sys.argv[1:] or ["tallies", "limits", "full", "open"]
    for w in want:
        {"tallies": tallies, "limits": limits, "full": full, "open": open_items}[w](rows)
