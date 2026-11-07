#!/usr/bin/env python3
"""A plain LaTeX document of every result of the paper and nothing else: each figure with its caption and each results
table, in the paper's order (main text, then appendix), with the round-4 figures and captions of the gallery.

    python3 experiments/FIGS/make_results_book.py <figures-dir with pages.json>     # writes paper/journal/results_book.tex
    cd paper/journal && tectonic -X compile results_book.tex

The figures are copied to figures/round4/; the paper's own figures (figures/mlst/gallery) stay as approved. Tables are
the paper's generated files (tables/*.tex) with its macros (results_generated.tex). Left out: the tables that
describe the setup, not results (the class hierarchy and vocabulary, the input features, the compute), and X->bb/X->bs,
which waits for its samples.
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

MAIN = ["F1", "T:usecase_survival", "F2a", "F2b", "F3", "F4", "T:mlst_jetclass", "FJ", "FJ-ctrl", "PT",
        "T:mlst_bench_top", "F6", "T:mlst_fill_main", "T:mlst_bench_qg", "F8",
        "F9a", "F9b", "T:mlst_mass", "F10a", "F10b"]
APPENDIX = ["VAL", "T:mlst_lrsweep", "T:mlst_fill", "T:mlst_mlp", "T:mlst_bn", "T:mlst_boot", "T:mlst_balance",
            "T:mlst_partrepro", "T:mlst_probes", "T:mlst_probes_n", "A10", "A8", "F6-app", "T:mlst_ft", "T:mlst_ft_rej",
            "L:mlst_per_run", "T:mlst_jetclass_all", "T:mlst_jetclass_probe", "T:mlst_jc1_class_auc", "FJ-app",
            "FJ-ctrl-app", "FJ-s", "CONF", "T:mlst_jetclass_nbkg", "T:mlst_rand_prong", "T:mlst_pairs", "F4-app",
            "F3-scatter", "F3-app", "F3-app2", "F3-ft", "F3-ft2", "A3R", "A3R-ft", "UMAP-jc1", "UMAP-jc2-had",
            "UMAP-jc2-lep", "UMAP-top", "UMAP-qg", "T:mlst_mass_detail", "A17-res", "A17-qcd", "A17-probe",
            "F10a-mass", "T:mlst_sculpt"]

# the gallery captions are plain text; typeset their physics notation
TEX = [("X→bb̄", r"$X\to b\bar b$"), ("X→cc̄", r"$X\to c\bar c$"), ("X→qq̄", r"$X\to q\bar q$"),
       ("H→bb̄", r"$H\to b\bar b$"), ("Z→qq̄", r"$Z\to q\bar q$"), ("W→qq′", r"$W\to qq'$"), ("X→bs", r"$X\to bs$"),
       ("X→ee", r"$X\to ee$"), ("X→μμ", r"$X\to\mu\mu$"), ("X→bb", r"$X\to b\bar b$"),
       ("cc̄", r"$c\bar c$"), ("bb̄", r"$b\bar b$"),
       ("r = m/m_gen", r"$r = m/m_{\mathrm{gen}}$"), ("sigma_r", r"$\sigma_r$"), ("eps_B", r"$\varepsilon_B$"),
       ("1 - g_QCD", r"$1-g_{\mathrm{QCD}}$"), ("16-84%", r"16--84\%"), ("20-500 GeV", r"20--500~GeV"),
       ("R30", r"$R_{30}$"), ("10^4", r"$10^{4}$"), ("10^3", r"$10^{3}$"), ("10^6", r"$10^{6}$"),
       (" (a)-(c)", r" (a)--(c)"), ("(b)-(d)", r"(b)--(d)")]


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
            f"\\caption{{{tex_caption(rec['caption'])}}}\n\\end{{figure}}\n\\FloatBarrier\n")


def table(name: str, landscape: bool) -> str:
    body = f"\\input{{tables/{name}}}\n"
    return (f"\\begin{{landscape}}\n{body}\\end{{landscape}}\n" if landscape else body) + "\\FloatBarrier\n"


def preamble() -> str:
    """The paper's preamble up to \\begin{document}, without its title block, plus the float package."""
    main = (PAPER / "main.tex").read_text()
    head = main[:main.index("\\title{")]
    return head + "\\usepackage{float}\n\\title{Results: figures and tables}\n\\date{}\n"


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = pathlib.Path(sys.argv[1])
    recs = {r["id"]: r for r in json.loads((src / "pages.json").read_text())}
    out = [preamble(), "\\begin{document}\n\\maketitle\n"]
    for heading, items in (("Main text", MAIN), ("Appendix", APPENDIX)):
        out.append(f"\\section*{{{heading}}}\n")
        for item in items:
            kind, _, name = item.rpartition(":")
            if kind:
                out.append(table(name, kind == "L"))
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
