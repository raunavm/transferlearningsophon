#!/usr/bin/env python3
"""A plain LaTeX document of every result of the paper and nothing else: each figure with its caption and each results
table, in the paper's order (main text, then appendix), with the round-4 figures and captions of the gallery.

    python3 experiments/FIGS/make_results_book.py <figures-dir with pages.json>     # writes paper/journal/results_book.tex
    cd paper/journal && tectonic -X compile results_book.tex

The figures are copied to figures/round4/; the paper's own figures (figures/mlst/gallery) stay as approved. Tables are
the paper's generated files (tables/*.tex) with its macros (results_generated.tex), copied in with their citations
and equation references taken out, since the book has no bibliography and no text (PI, 2026-11-07). Left out: the
design figure and the tables that describe the setup, not results (the discriminants each label set can build, the
class hierarchy and vocabulary, the input features, the compute), and X->bb/X->bs, which waits for its samples.
Everything is upright: the wide tables are split into stacked blocks by the generator (mlst_results.split_wide).
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "journal"
FIGS = ROOT / "figures" / "round4"

MAIN = ["F2a", "F2b", "F3", "F4", "T:mlst_jetclass", "FJ", "FJ-ctrl", "PT",
        "T:mlst_bench_top", "F6", "T:mlst_fill_main", "T:mlst_bench_qg", "F8",
        "F9a", "F9b", "T:mlst_mass", "F10a", "F10b", "F10c"]
APPENDIX = ["VAL", "T:mlst_lrsweep", "T:mlst_fill", "T:mlst_mlp", "T:mlst_bn", "T:mlst_boot", "T:mlst_balance",
            "T:mlst_partrepro", "T:mlst_probes", "T:mlst_probes_n", "A10", "A8", "F6-app", "T:mlst_ft", "T:mlst_ft_rej",
            "T:mlst_per_run", "T:mlst_jetclass_all", "T:mlst_jetclass_probe", "T:mlst_jc1_class_auc", "FJ-app",
            "FJ-ctrl-app", "FJ-s", "CONF", "T:mlst_jetclass_nbkg", "T:mlst_rand_prong", "T:mlst_pairs", "T:mlst_qcdhf", "T:mlst_coarse", "F4-app",
            "F3-scatter", "F3-app", "F3-app2", "F3-app3", "F3-ft", "F3-ft2", "A3R", "A3R-ft", "UMAP-jc1", "UMAP-jc2-had",
            "UMAP-jc2-lep", "UMAP-top", "UMAP-qg", "T:mlst_mass_detail", "A17-res", "A17-qcd", "A17-probe",
            "F10a-mass", "F10a-flav", "T:mlst_sculpt"]

# the gallery captions are plain text; typeset their physics notation
TEX = [("X→YY→bb̄qq", r"$X\to YY\to b\bar bqq$"), ("cc̄qq", r"$c\bar cqq$"), ("X→bc", r"$X\to bc$"),
       ("Jensen-Shannon", "Jensen--Shannon"), ("X→bb̄", r"$X\to b\bar b$"), ("X→cc̄", r"$X\to c\bar c$"), ("X→qq̄", r"$X\to q\bar q$"),
       ("H→bb̄", r"$H\to b\bar b$"), ("Z→qq̄", r"$Z\to q\bar q$"), ("W→qq′", r"$W\to qq'$"), ("X→bs", r"$X\to bs$"),
       ("X→ee", r"$X\to ee$"), ("X→μμ", r"$X\to\mu\mu$"), ("μμ", r"$\mu\mu$"), ("X→bb", r"$X\to b\bar b$"),
       ("cc̄", r"$c\bar c$"), ("bb̄", r"$b\bar b$"),
       ("r = m/m_gen", r"$r = m/m_{\mathrm{gen}}$"), ("sigma_r", r"$\sigma_r$"), ("eps_B", r"$\varepsilon_B$"),
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
    """Printed size (the figures are saved at it), or the height cap for a figure too tall for a page."""
    return f"[height={MAX_H_CM}cm]" if height_cm(pdf) > MAX_H_CM else ""


def figure(pid: str, rec: dict, src: pathlib.Path) -> str:
    FIGS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{pid}.pdf", FIGS / f"{pid}.pdf")
    return (("\\clearpage\n" if height_cm(src / f"{pid}.pdf") > NEW_PAGE_CM else "") + "\\begin{figure}[H]\n\\centering\n"
            f"\\includegraphics{size(src / f'{pid}.pdf')}{{../../figures/round4/{pid}.pdf}}\n"
            f"\\caption{{{tex_caption(rec['caption'])}}}\\label{{fig:{pid}}}\n\\end{{figure}}\n\\FloatBarrier\n")


# references into the paper's text, which the book does not have: the definition stays in words
NO_TEXT = [(", equation~(\\ref{eq:ratio}), to", " to"), (" of equation~(\\ref{eq:ratio}),", ","),
           (" of equation~(\\ref{eq:ratio}) at", " at"), (" of equation~(\\ref{eq:resolution}),", ","),
           (" of equation~(\\ref{eq:discr}).", "."), (" (equation~(\\ref{eq:cut}))", ""),
           ("$\\nmin$", "\\MlstMinBkgPass{}"), ("Published rows from the tables cited", "Published rows from the original papers"),
           ("Published rows (exp inputs) from the tables cited", "Published rows (exp inputs) from the original papers")]
CITE = re.compile(r"~?\\cite(\[[^\]]*\])?\{[^}]*\}")


def table(name: str) -> str:
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
    left = re.findall(r"\\(?:cite|eqref)\b|\\ref\{(?:eq|sec|app|fig):[^}]*\}|\\pending\{", body)
    if left:
        raise SystemExit(f"FATAL: {name} still has {left}")
    return f"% tables/{name}.tex\n{body}\\FloatBarrier\n"


def preamble() -> str:
    """The paper's preamble up to \\begin{document}, without its title block, plus float (capt-of is in it)."""
    main = (PAPER / "main.tex").read_text()
    head = main[:main.index("\\title{")]
    return head + "\\usepackage{float}\n\\pagestyle{plain}\n"


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = pathlib.Path(sys.argv[1])
    recs = {r["id"]: r for r in json.loads((src / "pages.json").read_text())}
    out = [preamble(), "\\begin{document}\n"]       # no title or headings: the first page starts with Figure 1 (PI)
    for items in (MAIN, APPENDIX):
        for item in items:
            kind, _, name = item.rpartition(":")
            if kind:
                out.append(table(name))
            else:
                if not recs.get(item, {}).get("drawn"):
                    raise SystemExit(f"FATAL: {item} was not drawn in {src}")
                out.append(figure(item, recs[item], src))
    out.append("\\end{document}\n")
    (PAPER / "results_book.tex").write_text("".join(out))
    n_fig = sum(":" not in i for i in MAIN + APPENDIX)
    print(f"wrote {PAPER / 'results_book.tex'}: {n_fig} figures, {len(MAIN) + len(APPENDIX) - n_fig} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
