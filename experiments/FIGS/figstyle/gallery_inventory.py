#!/usr/bin/env python3
"""The gallery's one-page map (PI, 2026-11-08, the story-based cut): every figure and table of the paper by tier (main
text, appendix, supplement), its number within the tier and the draft sentence it backs with the items that sentence
also rests on (gallery_placement), the sculpting test that decides F10b's tier (head/sculpting_boot.json), the PI's book
numbers against the gallery ids, and what the placement drops; with them the fine-tuned results at 10^6 jets, from
scratch on the full schedule beside the 188- and 17-class networks (PI 2026-11-08). Drawn at the gallery page's width
and at a height that leaves the page's notes room under it, so it prints at 100% and its 8 pt is 8 pt."""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

import gallery_common as C
import gallery_placement as G
import style_v5 as S

W, H = 27.7, 15.7           # cm: the landscape gallery page's width; its height less the change, caption and notes
COLUMNS = 3
GAP = 0.4                   # cm between columns
PT = 6.0                    # the map only (gallery, not the paper): the 2026-11-13 families, tables, plain sentences and the UMAP grids of L19 fit one page
SPACING = 1.25              # matplotlib linespacing: a line pitch of 0.293 cm (8.3 pt) for STIX at PT, measured
LINE = 0.293 * PT / 8       # cm per line (0.293 at 8 pt, scaled)
INDENT = 0.3                # cm: a row's continuation lines, so rows part without a gap
BOOT = C.P.DATA / "head/sculpting_boot.json"     # experiments/EVAL/sculpting_boot.py
# the bootstrap's differences (sculpting_boot.DIFFS) and working points, as the map names them
BOOT_DIFFS = (("17-188", "17 − 188"), ("43-188", "43 − 188"), ("17mass-17", "17 + mass − 17"),
              ("162mass-162", "162 + mass − 162"), ("flavf0-188", "no flavor − 188"),
              ("flavf1-188", "b-quark − 188"), ("flavf1r-188", "rand. flavor − 188"))
BOOT_WPS = (("eps_b=0.01", "QCD 1%"), ("eps_b=0.1", "QCD 10%"), ("eps_s=0.5", "res. 50%"))
# cm within a map column: the left edge of the difference, then per working point the right edges of Δ, σ and Δ/σ
BOOT_X = (0.0, 2.95, 3.54, 4.33, 5.24, 5.83, 6.62, 7.53, 8.12, 8.91)
# fine-tuning at 10^6 jets (PI 2026-11-08): the rows, and per task (JetClass-II class-balanced, JetClass) accuracy and
# macro one-vs-rest AUC; cm: the row label's left edge, then each column's right edge
FT_N = 1_000_000
FT_ARMS = ("L188", "R16_Q1", "SCRATCH")
FT_X = (0.0, 3.75, 5.0, 7.15, 8.4)


def _rows(items: tuple) -> list:
    """(number, ids, sentence) per row, consecutive items of one tier and one sentence merged ("Fig. S1–S4"); the
    sentence ends with what it also rests on (gallery_placement.EVIDENCE)."""
    rows = []
    for key, tier, sentence in items:
        num, ids = G.number(key), key + (f"; {G.PROVISIONAL[key]}" if key in G.PROVISIONAL else "")
        if also := G.evidence(key)[1:]:
            sentence += f" [also {', '.join(map(G.number, also))}]"
        if rows and rows[-1][2] == sentence and rows[-1][3] == tier:
            rows[-1] = (f"{rows[-1][0].split('–')[0]}–{num.split()[-1]}", f"{rows[-1][1]}, {ids}", sentence, tier)
        else:
            rows.append((num, ids, sentence, tier))
    return [r[:3] for r in rows]


def _boot_rows(path: pathlib.Path) -> list:
    """The sculpting test, one table row per difference: at each working point the Jensen-Shannon distance difference
    Δ and its combined uncertainty σ (in units of 0.01) and Δ/σ, starred where it passes, as sculpting_boot.json gives
    them (the run and bootstrap SDs are in tab:sculptboot); then F10b's tier by gallery_placement.f10b_main, with the
    rule and the rule's Δ/σ. Or one line saying the file is pending."""
    if not path.exists():
        return [("", f"Pending {path.name}: the paired bootstrap over the test jets (experiments/EVAL/sculpting_boot.py).", "")]
    diffs = json.loads(path.read_text())["differences"]
    num = lambda v, fmt: format(v, fmt).replace("-", "−")     # noqa: E731
    rows = [("", ("", *(c for _, wp_label in BOOT_WPS for c in ("", "", wp_label))), BOOT_X),     # over each Δ/σ
            ("", ("Difference", *("Δ", "σ", "Δ/σ") * len(BOOT_WPS)), BOOT_X)]
    for name, label in BOOT_DIFFS:
        cells = [label]
        for wp, _ in BOOT_WPS:
            d = diffs.get(name, {}).get(wp, {}).get("distance")
            cells += ["–"] * 3 if d is None else [
                num(100 * d["delta"], ".2f"), "–" if d["sigma"] is None else num(100 * d["sigma"], ".2f"),
                ("–" if d["z"] is None else num(d["z"], ".1f")) + ("*" if d["pass"] else "")]
        rows.append(("", tuple(cells), BOOT_X))
    label, wp_label = dict(BOOT_DIFFS), dict(BOOT_WPS)[G.F10B_WP]
    z = lambda name: diffs.get(name, {}).get(G.F10B_WP, {}).get("distance", {}).get("z")     # noqa: E731
    rule = " and ".join(f"{label[n]} {'>' if sign > 0 else '<'} {'+' if sign > 0 else '−'}3σ" for n, sign in G.F10B_RULE)
    here = " and ".join("–" if z(n) is None else num(z(n), "+.2f") + "σ" for n, _ in G.F10B_RULE)
    rows.append(("", f"F10b: {'main text' if G.f10b_main(diffs) else 'appendix'} (main text only if at {wp_label} "
                     f"{rule}; here {here}).", ""))
    return rows


def _pm(values) -> str:
    """mean(SD in the last digit) over the runs, the SD PDG-rounded and never past the mean's fourth significant
    figure (make_tables.fmt_pm's places): 0.4263(5), 0.9854(<1); "–" when there are no values."""
    if not values:
        return "–"
    m, sd = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    place = 3 - math.floor(math.log10(abs(m)))
    if sd > 0:
        place = min(C.M._pdg_place(sd), place)
    return f"{m:.{place}f}" + (f"({round(sd * 10 ** place) or '<1'})" if sd > 0 else "")


def ft_values() -> dict:
    """{arm of FT_ARMS: [JetClass-II accuracy, JetClass-II macro AUC, JetClass accuracy, JetClass macro AUC], each the
    list over runs or None} at FT_N, through the tables' own loaders: JetClass-II pretrained from the fine-tuning read-out
    (mlst_results.load_ft, tab:ft), JetClass from ft_curves (mlst_results.jc1_full, tab:jetclass). From scratch only the
    full-schedule cells (mlst_results.SCRATCH_FULL / mlst_v4.SCRATCH_FULL): JetClass from
    curves/full_budget_long_fulldecay_leg2.json, the JetClass-II AUC the mean over the 188 classes' AUCs of
    curves_v2/jc2_ft_ovr_fulldecay (as tab:ft's macro AUC is), its accuracy only once the full-schedule budget read-outs
    are pulled; None where a value is not pulled."""
    data, per, M = C.P.DATA, C.I["per"], C.M
    jc1 = M.jc1_full(data, C.RUNS)
    pre, ref = (jc1[0], jc1[1]) if jc1 else ({}, {})
    out = {}
    for arm in FT_ARMS[:2]:
        ms = C.RUNS[arm]
        jc2 = [[M._ft(per, "jc2", m, FT_N, k) for m in ms] for k in ("accuracy", "macro_auc_ovr")]
        b = [M._jc1_block(pre, m) for m in ms]
        out[arm] = [None if None in v else v for v in
                    (*jc2, *([None if x is None else x[k] for x in b] for k in ("accuracy", "macro_auc_ovr")))]
    seeds = M.V4.JC1_LONG_SCRATCH
    full_jc2 = data / M.V4.ROC_DIR / f"jc2_ft_ovr{M.V4.SCRATCH_FULL}"
    docs = C.ROC.get("jc2_ft_ovr", {}) if full_jc2.is_dir() else {}
    jc2_auc = [float(np.mean([c["auc"] for c in docs[f"scratch-{s}"]["curves"].values()])) for s in seeds
               if f"scratch-{s}" in docs]
    jc2_acc = ([r["accuracy"] for r in per["jc2"].get(M.SCRATCH, {}).get(FT_N, [])]
               if all((data / "budget" / f).exists() for f in M.SCRATCH_FULL) else [])
    jc1s = ([b["jc1"] for s in seeds if (b := ref.get(M.SCRATCH, {}).get(M.JC1_N, {}).get(s))]
            if (data / "curves/full_budget_long_fulldecay_leg2.json").exists() else [])
    out["SCRATCH"] = [v if len(v) == len(seeds) else None for v in
                      (jc2_acc, jc2_auc, [b["accuracy"] for b in jc1s], [b["macro_auc_ovr"] for b in jc1s])]
    return out


def _ft_rows(values: dict) -> list:
    """The 10^6 block: two header rows, one row per FT_ARMS with _pm per column, and the recipe line, naming what is
    pending (not read out or not pulled)."""
    rows = [("", ("", "", "JetClass-II, balanced", "", "JetClass"), FT_X),
            ("", ("", "accuracy", "AUC", "accuracy", "AUC"), FT_X)]
    rows += [("", (S.LEGEND[a] + ("" if a == "SCRATCH" else f" {S.FINE_TUNED}"), *map(_pm, values[a])), FT_X)
             for a in FT_ARMS]
    names = ("JetClass-II accuracy", "JetClass-II AUC", "JetClass accuracy", "JetClass AUC")
    gaps = [f"{n}, {S.LEGEND[a]}" for a in FT_ARMS for n, v in zip(names, values[a]) if not v]
    rows.append(("", "From scratch: ParT's full flat+decay schedule, the best validation epoch kept."
                     + (f" –: pending: {'; '.join(gaps)}." if gaps else ""), ""))
    return rows


def blocks() -> list:
    """(heading, rows) in the map's order; a row is (number, ids, sentence), ("", text, "") for running text, or
    ("", cells, x positions) for a table row (the sculpting test, the 10^6 fine-tuning)."""
    out = []
    for tier in G.TIERS:
        for kind, items in (("figures", G.FIGURES), ("tables", G.TABLES)):
            out.append((f"{G.TIER_NAME[tier]}: {kind}", _rows(tuple(i for i in items if i[1] == tier))))
    out[0] = ("Main text: figures (every sentence a DRAFT, worded to today's numbers; [also ...]: what else it rests on)",
              out[0][1])
    out.append(("Sculpting, bootstrap: Jensen-Shannon distance Δ, σ (0.01); *: |Δ| > 3σ",
                _boot_rows(BOOT)))
    out.append(("Fine-tuned on a million jets: mean (SD in the last digit) of three runs", _ft_rows(ft_values())))
    out.append(("Results-book figure numbers, as gallery ids", [("", "; ".join(f"{b} = {g}" for b, g in G.BOOK_NAMES) + ".", "")]))
    lost = f" NEITHER PLACED NOR DROPPED: {', '.join(unplaced())}." if unplaced() else ""
    out.append(("Not placed", [("", f"Dropped figures: {', '.join(G.DROPPED)}. Tables dropped (nothing cites them): "
                                    f"{', '.join(G.DROPPED_TABLES)}.{lost}", "")]))
    return out


def unplaced() -> list:
    """The table labels of mlst_results.CAPTIONS that no tier holds and the placement does not drop (none, when the
    placement is whole; check_citations.py --gallery fails on any)."""
    placed = {k for t, *_ in G.TABLES for k in t.split(" + ")} | set(G.DROPPED_TABLES)
    return sorted(label for label, *_ in C.M.CAPTIONS.values() if label not in placed)


def draw_inventory(path: pathlib.Path) -> None:
    """The map is the gallery's index, not a figure of the paper: it keeps STIX, for which LINE was measured."""
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        _draw_inventory(path)


def _draw_inventory(path: pathlib.Path) -> None:
    fig = S.figure_cm(W, H)
    ax = S.axes_cm(fig, 0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    col_w = (W - (COLUMNS - 1) * GAP) / COLUMNS
    col, y = 0, 0.0

    def room(lines: int) -> float:
        nonlocal col, y
        if y + lines * LINE > H:
            col, y = col + 1, 0.0
            if col >= COLUMNS:
                raise SystemExit("FATAL: the map does not fit one page; shorten the sentences")
        return col * (col_w + GAP)

    def wrap(text: str, bold: bool = False, indent: float = INDENT) -> list:
        """Words filled to the column's width as drawn (TextToPath), the first line across it, the rest indent in."""
        prop = FontProperties(size=PT, weight="bold" if bold else "normal")
        width = lambda t: TextToPath().get_text_width_height_descent(t, prop, ismath="$" in t)[0] / 72 * 2.54
        lines = [""]
        for word in text.split():
            line = f"{lines[-1]} {word}".strip()
            if lines[-1] and width(line) > col_w - (indent if len(lines) > 1 else 0):
                lines.append(word)
            else:
                lines[-1] = line
        return lines

    for heading, rows in blocks():
        lines = wrap(heading, bold=True, indent=0)
        x = room(len(lines) + 2)            # never a heading alone at the foot of a column
        ax.text(x, y, "\n".join(lines), fontsize=PT, fontweight="bold", va="top", linespacing=SPACING)
        y += LINE * (len(lines) + 0.15)
        for num, ids, sentence in rows:
            if isinstance(ids, tuple):          # a table row: one line, cells at its x positions
                x = room(1)
                for k, (dx, cell) in enumerate(zip(sentence, ids)):
                    ax.text(x + dx, y, cell, fontsize=PT, va="top", ha="left" if k == 0 else "right")
                y += LINE
                continue
            lines = wrap(f"{num} ({ids}): {sentence}" if num else ids)
            x = room(len(lines))
            ax.text(x, y, lines[0], fontsize=PT, va="top")
            if lines[1:]:
                ax.text(x + INDENT, y + LINE, "\n".join(lines[1:]), fontsize=PT, va="top", linespacing=SPACING)
            y += len(lines) * LINE
        y += LINE * 0.4
    S.save(fig, path)


PAGES = [
    C.Page(
        id="INVENTORY",
        title="The one-page map: placement of every figure and table",
        precedent=[],
        precedent_label="None: the map the PI's story-based cut asks for",
        caption="Every figure and table of the paper by tier: main text, then appendix, then supplement, each with its "
                "number within the tier, its gallery id or table label and the draft sentence it backs, of the abstract "
                "or conclusions for the main text and of the main text for the appendix. Then the sculpting test, the "
                "fine-tuned results at a million jets with and without pretraining, the results book's numbers against "
                "the gallery ids and the figures and tables dropped.",
        differs="Not a paper figure. Drawn across the whole gallery page, at the size it prints.",
        draw=draw_inventory,
        wide=True),
]
