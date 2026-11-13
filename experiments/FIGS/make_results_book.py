#!/usr/bin/env python3
"""Plain LaTeX documents of every result of the paper and nothing else: each figure with its caption and each results
table, in the paper's placement (plan 2026-11-08: main text, appendix, supplement), with the gallery's figures and
captions.

    python3 experiments/FIGS/make_results_book.py <figures-dir with pages.json> [--tier paper|supplement|all]
        paper       main text and appendix   -> paper/journal/results_book.tex (the default)
        supplement  the supplement           -> paper/journal/results_supplement.tex
        all         all three, for review    -> paper/journal/results_book_all.tex
    cd paper/journal && tectonic -X compile --keep-intermediates results_book.tex && tectonic -X compile results_supplement.tex

The supplement's references into the paper resolve through results_book.aux (xr-hyper), so the paper book is compiled
first, keeping its intermediates; the supplement is not written without it. The figures are copied to figures/round4/; the paper's own figures
(figures/mlst/gallery) stay as approved. Tables are the paper's generated files (tables/*.tex) with its macros
(results_generated.tex), copied in with their citations and equation references taken out, since the book has no
bibliography and no text (PI, 2026-11-07). Each tier starts a page with its figures, then its tables; appendix items
are numbered A1, A2, ..., the supplement's S1, S2, ..., as on the gallery's map. Everything is upright: the wide tables are split into stacked blocks by the generator
(mlst_results.split_wide).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "paper" / "journal"
FIGS = ROOT / "figures" / "round4"

# The placement is the gallery's (figstyle/gallery_placement.py, PI 2026-11-08), the one source: per tier its figures
# (gallery ids) in its order, then its tables (labels; "tab:a + tab:b" one number, its tables a and b), so that every
# item has here the number the gallery's map gives it. A table label is its generated file through the CAPTIONS key
# (mlst_results). A table with \pending cells is fatal unless the placement marks it PROVISIONAL: then it is set with
# its pending cells shown. A figure the render lacks is fatal unless PROVISIONAL: then a box keeps its number. A
# PROVISIONAL item that is complete is set as placed.
_spec = importlib.util.spec_from_file_location("gallery_placement", HERE / "figstyle" / "gallery_placement.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
_spec = importlib.util.spec_from_file_location("mlst_results", HERE / "mlst_results.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
TABLE_FILE = {M.CAPTIONS[k][0]: n for k, n, *_ in M.PLACEHOLDERS + M.LONGTABLES}
TIERS = {t: [k for k, tier, _ in G.FIGURES if tier == t] + [k for k, tier, _ in G.TABLES if tier == t] for t in G.TIERS}
OUTPUTS = {"paper": ("results_book", (G.MAIN, G.APPENDIX)), "supplement": ("results_supplement", (G.SUPPLEMENT,)),
           "all": ("results_book_all", G.TIERS)}
PREFIX = {G.MAIN: "", G.APPENDIX: "A", G.SUPPLEMENT: "S"}

# the gallery captions are plain text; typeset their physics notation
TEX = [("X→YY→bb̄qq", r"$X\to YY\to b\bar bqq$"), ("cc̄qq", r"$c\bar cqq$"), ("X→bc", r"$X\to bc$"),
       ("Jensen-Shannon", "Jensen--Shannon"), ("X→bb̄", r"$X\to b\bar b$"), ("X→cc̄", r"$X\to c\bar c$"), ("X→qq̄", r"$X\to q\bar q$"),
       ("H→bb̄", r"$H\to b\bar b$"), ("Z→qq̄", r"$Z\to q\bar q$"), ("W→qq′", r"$W\to qq'$"), ("X→bs", r"$X\to bs$"),
       ("X→ee", r"$X\to ee$"), ("X→μμ", r"$X\to\mu\mu$"), ("μμ", r"$\mu\mu$"), ("X→bb", r"$X\to b\bar b$"),
       ("cc̄", r"$c\bar c$"), ("bb̄", r"$b\bar b$"),
       ("r = m/m_gen", r"$r = m/m_{\mathrm{gen}}$"), (" pT ", r" $p_{\mathrm{T}}$ "), ("sigma_r", r"$\sigma_r$"), ("eps_B", r"$\varepsilon_B$"),
       ("1 - g_QCD", r"$1-g_{\mathrm{QCD}}$"), ("16-84%", r"16--84\%"), ("20-500 GeV", r"20--500~GeV"),
       ("R30", r"$R_{30}$"), ("10^4", r"$10^{4}$"), ("10^3", r"$10^{3}$"), ("10^6", r"$10^{6}$"),
       (" (a)-(c)", r" (a)--(c)"), ("(b)-(d)", r"(b)--(d)"), ("15.9-84.1%", r"15.9--84.1\%"),
       ("b-quark", r"$b$-quark"), ("response r (", r"response $r$ ("), ("range of r over", r"range of $r$ over"),
       # the gallery's captions name a figure of the paper in words; here it has a number
       ("Sub-panels as in the main-text mass figure", r"Sub-panels as in figure~\ref{fig:F9b}"),
       ("as in the main-text figure", r"as in figure~\ref{fig:F9b}"),
       ("The mass sculpting figure for", r"As figure~\ref{fig:F10a}, for")]


def tex_caption(text: str) -> str:
    for a, b in TEX:
        text = text.replace(a, b)
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    left = sorted({c for c in text if ord(c) > 127})
    if left:
        raise SystemExit(f"FATAL: no LaTeX for {left} in: {text}")
    return text


MAX_H_CM = 18.5         # three quarters of the paper's text height, so a figure and its caption fit on one page


NEW_PAGE_CM = 12.0      # a figure taller than this starts a page: [H] does not move it past a table above it


def height_cm(pdf: pathlib.Path) -> float:
    import fitz
    with fitz.open(pdf) as d:
        return d[0].rect.height / 72 * 2.54


def size(pdf: pathlib.Path) -> str:
    """Every figure at the size it is drawn: across the text block (PI 2026-11-11), a one-panel figure narrower; the height cap
    for a figure that would then be too tall for a page."""
    import fitz
    with fitz.open(pdf) as d:
        r = d[0].rect
    if r.width / 72 * 2.54 < TEXT_W_CM - 1.0:        # a one-panel figure (PI 2026-11-12): at the size it is drawn
        return ""
    return f"[height={MAX_H_CM}cm]" if r.height / r.width * TEXT_W_CM > MAX_H_CM else "[width=\\linewidth]"


TEXT_W_CM = 15.0        # the paper's text width


def figure(pid: str, rec: dict, src: pathlib.Path) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{pid}.pdf", FIGS / f"{pid}.pdf")
    return (("\\clearpage\n" if height_cm(src / f"{pid}.pdf") > NEW_PAGE_CM else "") + "\\begin{figure}[H]\n\\centering\n"
            f"\\includegraphics{size(src / f'{pid}.pdf')}{{../../figures/round4/{pid}.pdf}}\n"
            f"\\caption{{{tex_caption(rec['caption'])}}}\\label{{fig:{pid}}}\n\\end{{figure}}\n\\FloatBarrier\n")


# references into the paper's text, which the book does not have: the definition stays in words
NO_TEXT = [(", equation~(\\ref{eq:ratio}), to", " to"), (" of equation~(\\ref{eq:ratio}),", ","),
           (" of equation~(\\ref{eq:ratio}) at", " at"), (" of equation~(\\ref{eq:resolution}),", ","),
           (" of equation~(\\ref{eq:discr}).", "."), (" of equation~(\\ref{eq:discr});", ";"), (" (equation~(\\ref{eq:cut}))", ""),
           ("$\\nmin$", "\\MlstMinBkgPass{}"), ("Published rows from the tables cited", "Published rows from the original papers"),
           ("Published rows (exp inputs) from the tables cited", "Published rows (exp inputs) from the original papers")]
CITE = re.compile(r"~?\\cite(\[[^\]]*\])?\{[^}]*\}")


def table(name: str, provisional: bool = False) -> str:
    """A generated table set for the book; \\pending cells only in a PROVISIONAL one."""
    body = (PAPER / "tables" / f"{name}.tex").read_text()
    for a, b in NO_TEXT:
        body = body.replace(a, b)
    body = CITE.sub("", body)
    # not a float: a table split into blocks may run over a page, and breaks between its blocks
    body = re.sub(r"\\begin\{table\*?\}\[\w*\]", r"\\begin{center}", body)
    body = re.sub(r"\\end\{table\*?\}", r"\\end{center}", body)
    if "longtable" not in body and "\\captionof" not in body:   # a longtable or a table set to break: as it is
        body = body.replace("\\caption{", "\\captionof{table}{")
        # a table short enough to be a float stays whole on one page
        i, j = body.index("\\begin{center}") + len("\\begin{center}"), body.rindex("\\end{center}")
        body = body[:i] + "\n\\begin{minipage}{\\linewidth}\\centering" + body[i:j] + "\\end{minipage}\n" + body[j:]
    left = re.findall(r"\\(?:cite|eqref)\b|\\ref\{(?:eq|sec|app|fig):[^}]*\}" + ("" if provisional else r"|\\pending\{"),
                      body)
    if left:
        raise SystemExit(f"FATAL: {name} still has {left}")
    return f"% tables/{name}.tex\n{body}\\FloatBarrier\n"


def table_item(key: str, tier: str) -> str:
    """A placed table by its label; labels joined by " + " share one number, their tables lettered a, b, ... A
    PROVISIONAL table is set with the pending cells its generator leaves."""
    labels = key.split(" + ")
    unknown = [lab for lab in labels if lab not in TABLE_FILE]
    if unknown:
        raise SystemExit(f"FATAL: no generated table has the label(s) {unknown}")
    if len(labels) == 1:
        if key in G.PROVISIONAL and "\\pending{" in (PAPER / "tables" / f"{TABLE_FILE[key]}.tex").read_text():
            print(f"note: {key} is provisional ({G.PROVISIONAL[key]}); set with its pending cells")
        return table(TABLE_FILE[key], key in G.PROVISIONAL)
    return "".join(("\\addtocounter{table}{-1}" if k else "") + "\\begingroup"
                   f"\\renewcommand{{\\thetable}}{{{PREFIX[tier]}\\arabic{{table}}{chr(97 + k)}}}"
                   f"\\renewcommand{{\\theHtable}}{{{tier}.\\arabic{{table}}{chr(97 + k)}}}\n"
                   + table(TABLE_FILE[lab]) + "\\endgroup\n" for k, lab in enumerate(labels))


def pending(key: str) -> str:
    """A box in place of a PROVISIONAL figure not drawn yet, so that the figures after it keep their numbers."""
    box = "\\fbox{\\parbox[c][3cm]{0.6\\linewidth}{\\centering pending}}\n"
    cap = f"Pending ({tex_caption(G.PROVISIONAL[key])}). {tex_caption(G.BACKS[key])}"
    return f"\\begin{{figure}}[H]\n\\centering\n{box}\\caption{{{cap}}}\\label{{fig:{key}}}\n\\end{{figure}}\n\\FloatBarrier\n"


def preamble(external: str | None = None) -> str:
    """The paper's preamble up to \\begin{document}, without its title block, plus float (capt-of is in it); with
    external, xr-hyper reading that document's labels. Both packages are loaded before hyperref, as they must be;
    the labels are read at the end of the preamble, where the macros of their captions exist."""
    main = (PAPER / "main.tex").read_text()
    head = main[:main.index("\\title{")]
    m = re.search(r"\\usepackage(\[[^]]*\])?\{hyperref\}", head)
    if not m:
        raise SystemExit("FATAL: the paper's preamble loads no hyperref, which float and xr-hyper must precede")
    add = "\\usepackage{float}\n" + ("\\usepackage{xr-hyper}\n" if external else "")
    return (head[:m.start()] + add + head[m.start():] + "\\pagestyle{plain}\n"
            + (f"\\externaldocument{{{external}}}\n" if external else ""))


def tier_start(tier: str) -> str:
    """Each tier starts a page and numbers its figures and tables with its prefix (none in the main text)."""
    p = PREFIX[tier]
    return ("\\clearpage\n\\setcounter{figure}{0}\\setcounter{table}{0}\n" + "".join(
        f"\\renewcommand{{\\the{c}}}{{{p}\\arabic{{{c}}}}}\\renewcommand{{\\theH{c}}}{{{tier}.\\arabic{{{c}}}}}\n"
        for c in ("figure", "table")))


def check_spec() -> None:
    seen = [i for items in TIERS.values() for i in items]
    twice = sorted({i for i in seen if seen.count(i) > 1})
    if twice:
        raise SystemExit(f"FATAL: placed twice: {twice}")


def labels_of(tiers) -> set:
    """The figure and table labels the placement puts in these tiers."""
    return {lab for t in tiers for k in TIERS[t] for lab in (k.split(" + ") if k.startswith("tab:") else [f"fig:{k}"])}


def build(src: pathlib.Path, which: str) -> str:
    """Writes the document of one output (OUTPUTS) from the gallery render in src; returns its name."""
    check_spec()
    recs = {r["id"]: r for r in json.loads((src / "pages.json").read_text())}
    name, tiers = OUTPUTS[which]
    external = None
    if which == "supplement":        # its references into the paper resolve through the paper book's labels
        external = OUTPUTS["paper"][0]
        if not (PAPER / f"{external}.aux").exists():
            raise SystemExit(f"FATAL: no {PAPER / external}.aux for the supplement's references into the paper: compile "
                             f"{external}.tex first (tectonic -X compile --keep-intermediates {external}.tex)")
    out = [preamble(external), "\\begin{document}\n"]    # no title or headings: the first page starts with Figure 1 (PI)
    n_fig = n_tab = n_box = 0
    for tier in tiers:
        out.append(tier_start(tier))
        for item in TIERS[tier]:
            if item.startswith("tab:"):
                out.append(table_item(item, tier))
                n_tab += 1
            elif recs.get(item, {}).get("drawn"):
                if item in G.PROVISIONAL:
                    print(f"note: {item} is provisional ({G.PROVISIONAL[item]}); set as placed")
                out.append(figure(item, recs[item], src))
                n_fig += 1
            elif item in G.PROVISIONAL:
                print(f"note: {item} is provisional ({G.PROVISIONAL[item]}) and not drawn in {src}; a box keeps its number")
                out.append(pending(item))
                n_box += 1
            else:
                raise SystemExit(f"FATAL: {item} was not drawn in {src}")
    out.append("\\end{document}\n")
    doc = "".join(out)
    known = labels_of(tiers) | (labels_of(OUTPUTS["paper"][1]) if external else set())
    for lab in sorted({r for r in re.findall(r"\\ref\{((?:fig|tab):[^}]+)\}", doc) if r not in known}):
        print(f"note: {lab} is referred to but placed in no tier of this book: it prints as ??")
    (PAPER / f"{name}.tex").write_text(doc)
    print(f"wrote {PAPER / name}.tex ({', '.join(tiers)}): {n_fig} figures, {n_box} pending boxes, {n_tab} tables")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", type=pathlib.Path, help="the gallery render: a directory with pages.json")
    ap.add_argument("--tier", choices=tuple(OUTPUTS), default="paper")
    a = ap.parse_args()
    build(a.figures, a.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
