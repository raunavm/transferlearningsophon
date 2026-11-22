#!/usr/bin/env python3
"""What the manuscript holds, against the placement (PI 2026-11-22T23:11:36Z, LEDGER SP-1): "keep it only if the
manuscript's figures and tables are exactly gallery_placement's MAIN and APPENDIX lists ... There is one source of truth
for what the paper contains."

The manuscript's items are read from paper/journal/main.tex in \\input order: every \\paperfig{<width>}{<id>} (through
main.tex's \\setfigfile{<id>}{gallery/<gallery id>} to its gallery id) and every \\input{tables/<file>} (through the
results book's map of generated files to labels, mlst_results CAPTIONS). Sections input after \\appendix are the
appendix; the rest is the main text. Commented-out lines do not count.
    python3 experiments/FIGS/manuscript_items.py     # prints every mismatch; exit 1 if any
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
JOURNAL = HERE.parents[1] / "paper" / "journal"
sys.path.insert(0, str(HERE))
import make_results_book as B  # noqa: E402  (its TABLE_FILE and the placement G)

G = B.G


def _lines(path: pathlib.Path) -> list:
    return [re.sub(r"(?<!\\)%.*", "", ln) for ln in path.read_text().splitlines()]


def _file_label(path: pathlib.Path) -> str | None:
    """The \\label of a table file that make_tables writes (not in mlst_results' map), if the file is there."""
    m = path.is_file() and re.search(r"\\label\{(tab:[^}]*)\}", path.read_text())
    return m.group(1) if m else None


def manuscript(journal: pathlib.Path = JOURNAL) -> dict:
    """{tier: [item, ...]} in manuscript order: gallery ids for figures, labels for tables."""
    main = _lines(journal / "main.tex")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", "\n".join(main)))
    expand = lambda s: re.sub(r"\\(\w+)", lambda m: macros.get(m.group(1), m.group(0)), s)
    figfile = {expand(a): expand(b).removeprefix("gallery/")
               for a, b in re.findall(r"\\setfigfile\{(.+?)\}\{(.+?)\}\s*$", "\n".join(main), re.M)}
    label = {f: lab for lab, f in B.TABLE_FILE.items()}
    out = {G.MAIN: [], G.APPENDIX: []}
    tier = G.MAIN
    for ln in main:
        if re.search(r"\\appendix\b", ln):
            tier = G.APPENDIX
        for sec in re.findall(r"\\input\{(sections/[\w-]+)\}", ln):
            for sl in _lines(journal / f"{sec}.tex"):
                for fid in re.findall(r"\\paperfig\{[^{}]*\}\{(.+?)\}", sl):
                    fid = expand(fid)
                    out[tier].append(figfile.get(fid, f"UNMAPPED figure id {fid}"))
                for f in re.findall(r"\\input\{tables/([\w-]+)\}", sl):
                    out[tier].append(label.get(f) or _file_label(journal / "tables" / f"{f}.tex")
                                     or f"UNMAPPED table file {f}")
    return out


def placement() -> dict:
    """{tier: [item, ...]} the placement's main-text and appendix figures, then its tables, in its order."""
    return {t: [k for k, tier, _ in G.FIGURES if tier == t] + [k for k, tier, _ in G.TABLES if tier == t]
            for t in (G.MAIN, G.APPENDIX)}


def mismatch(journal: pathlib.Path = JOURNAL) -> list:
    """Every difference between the manuscript and the placement, one line each; empty when they agree."""
    have, want = manuscript(journal), placement()
    where = {k: t for t, ks in want.items() for k in ks}
    tier_of = {k: t for t, ks in have.items() for k in ks}
    out = []
    for t in (G.MAIN, G.APPENDIX):
        for k in have[t]:
            if k not in where:
                out.append(f"{t}: {k} is in the manuscript but not placed in the main text or appendix "
                           f"(placement: {G.TIER.get(k, 'not in the placement')})")
            elif where[k] != t:
                out.append(f"{t}: {k} is in the manuscript's {t}, placed in the {where[k]}")
        for k in want[t]:
            if k not in tier_of:
                out.append(f"{t}: {k} is placed here but not in the manuscript")
        dup = sorted({k for k in have[t] if have[t].count(k) > 1})
        if dup:
            out.append(f"{t}: input more than once: {', '.join(dup)}")
    return out


if __name__ == "__main__":
    lines = mismatch()
    print("\n".join(lines) or "the manuscript holds exactly the placement's main-text and appendix items")
    raise SystemExit(1 if lines else 0)
