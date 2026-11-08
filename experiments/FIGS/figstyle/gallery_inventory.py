#!/usr/bin/env python3
"""The gallery's one-page map (PI, 2026-11-08, the story-based cut): every figure and table of the paper by tier (main
text, appendix, supplement), its number within the tier and the draft sentence it backs with the items that sentence
also rests on (gallery_placement), the sculpting test that decides F10b's tier (head/sculpting_boot.json), the PI's book
numbers against the gallery ids, and what the placement drops. Drawn at the gallery page's width and at a
height that leaves the page's notes room under it, so it prints at 100% and its 8 pt is 8 pt."""
from __future__ import annotations

import json
import pathlib

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

import gallery_common as C
import gallery_placement as G
import style_v5 as S

W, H = 27.7, 15.7           # cm: the landscape gallery page's width; its height less the change, caption and notes
COLUMNS = 3
GAP = 0.6                   # cm between columns
PT = 8.0
SPACING = 1.25              # matplotlib linespacing: a line pitch of 0.293 cm (8.3 pt) for STIX at PT, measured
LINE = 0.293                # cm per line
INDENT = 0.3                # cm: a row's continuation lines, so rows part without a gap
BOOT = C.P.DATA / "head/sculpting_boot.json"     # experiments/EVAL/sculpting_boot.py
# the bootstrap's differences (sculpting_boot.DIFFS) and working points, as the map names them
BOOT_DIFFS = (("17-188", "17 − 188"), ("43-188", "43 − 188"), ("17mass-17", "17 + mass − 17"),
              ("162mass-162", "162 + mass − 162"), ("flavf0-188", "no flavor − 188"),
              ("flavf1-188", "b-quark − 188"), ("flavf1r-188", "random flavor − 188"))
BOOT_WPS = (("eps_b=0.01", "QCD 1%"), ("eps_b=0.1", "QCD 10%"), ("eps_s=0.5", "res. 50%"))
# cm within a map column: the left edge of the difference, then per working point the right edges of Δ and Δ/σ
BOOT_X = (0.0, 3.4, 4.3, 5.5, 6.4, 7.6, 8.5)


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
    (in units of 0.01) and Δ/σ, starred where it passes, as sculpting_boot.json gives them (the run and bootstrap SDs
    are in tab:sculptboot); then F10b's tier by gallery_placement.f10b_main. Or one line saying the file is pending."""
    if not path.exists():
        return [("", f"Pending {path.name}: the paired bootstrap over the test jets (experiments/EVAL/sculpting_boot.py).", "")]
    diffs = json.loads(path.read_text())["differences"]
    num = lambda v, fmt: format(v, fmt).replace("-", "−")     # noqa: E731
    rows = [("", ("", *(c for _, wp_label in BOOT_WPS for c in ("", wp_label))), ""),     # over each Δ/σ
            ("", ("Difference", *("Δ", "Δ/σ") * len(BOOT_WPS)), "")]
    for name, label in BOOT_DIFFS:
        cells = [label]
        for wp, _ in BOOT_WPS:
            d = diffs.get(name, {}).get(wp, {}).get("distance")
            cells += ["–", "–"] if d is None else [
                num(100 * d["delta"], ".2f"), ("–" if d["z"] is None else num(d["z"], ".1f")) + ("*" if d["pass"] else "")]
        rows.append(("", tuple(cells), ""))
    rows.append(("", f"F10b: {'main text' if G.f10b_main(diffs) else 'appendix'}.", ""))
    return rows


def blocks() -> list:
    """(heading, rows) in the map's order; a row is (number, ids, sentence), ("", text, "") for running text, or
    ("", cells, "") for a table row (the sculpting test)."""
    out = []
    for tier in G.TIERS:
        for kind, items in (("figures", G.FIGURES), ("tables", G.TABLES)):
            out.append((f"{G.TIER_NAME[tier]}: {kind}", _rows(tuple(i for i in items if i[1] == tier))))
    out[0] = ("Main text: figures (every sentence a DRAFT, worded to today's numbers; [also ...]: what else it rests on)",
              out[0][1])
    out.append(("Sculpting test: Δ of the distance (0.01) at cuts keeping 1% or 10% of QCD or 50% of resonant jets; "
                "*: |Δ| > 3σ. F10b main text if at QCD 1% 17 − 188 > 0 and 17 + mass − 17 < 0, both passing",
                _boot_rows(BOOT)))
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
            if isinstance(ids, tuple):          # a table row: one line, cells at BOOT_X
                x = room(1)
                for k, (dx, cell) in enumerate(zip(BOOT_X, ids)):
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
                "results book's numbers against the gallery ids and the figures and tables dropped.",
        differs="Not a paper figure. Drawn across the whole gallery page, at the size it prints.",
        draw=draw_inventory,
        wide=True),
]
