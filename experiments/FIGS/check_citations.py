#!/usr/bin/env python3
"""Every figure and table is cited (PI 2026-11-08: anything nothing cites is dropped).

    python3 experiments/FIGS/check_citations.py paper/journal/main.tex    # the paper: fails on an uncited figure or table
    python3 experiments/FIGS/check_citations.py --gallery                 # the placement: each item's planned citing sentence

The paper mode follows \\input / \\include from the root and lists every \\label{fig:...} and \\label{tab:...} with the
files that \\ref it from outside its own float (a generated table that is not a float: outside its file); it exits 1
if any is cited nowhere. The gallery mode reads the placement of make_results_book.py (main, appendix, supplement;
keyed by gallery id and table label, as the gallery's) and the "backs" sentence and TIER of each item from the gallery
code (whichever figstyle module defines BACKS), and exits 1 if a placed item has no sentence, if TIER places it in
another tier, if the book places an item the gallery's TIER does not hold, if the gallery places an item the book
does not, if a table of mlst_results.CAPTIONS is neither placed nor in the placement's DROPPED_TABLES, or if a page of
gallery_figures.ORDER is neither placed nor in its DROPPED (figures) or GALLERY_ONLY (the map, the palette).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
INPUT = re.compile(r"^[^%\n]*?\\(?:input|include)\{([^}]+)\}", re.M)
LABEL = re.compile(r"\\label\{((?:fig|tab):[^}#]+)\}")
REF = re.compile(r"\\(?:[cC]ref|ref|autoref|pageref)\{([^}]+)\}")
FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?|longtable)\}.*?\\end\{\1\}", re.S)


def files(root: pathlib.Path) -> list[pathlib.Path]:
    """root and every file it inputs, in order (paths relative to root's directory, .tex added)."""
    out, todo = [], [root]
    while todo:
        f = todo.pop(0)
        if f in out or not f.exists():
            continue
        out.append(f)
        todo += [root.parent / (n if n.endswith(".tex") else n + ".tex") for n in INPUT.findall(f.read_text())]
    return out


def strip_comments(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def citations(root: pathlib.Path) -> dict:
    """{label: [files citing it]} of every figure and table label; a reference inside the label's own float (or, for a
    label in no float, its own file) does not count."""
    text = {f: strip_comments(f.read_text()) for f in files(root)}
    where = {}                                  # label -> (file, start, end) of the span its self-references lie in
    for f, t in text.items():
        floats = [m.span() for m in FLOAT.finditer(t)]
        for m in LABEL.finditer(t):
            a, b = next(((a, b) for a, b in floats if a <= m.start() < b), (0, len(t)))
            where[m.group(1)] = (f, a, b)
    cited = {lab: [] for lab in where}
    for f, t in text.items():
        for m in REF.finditer(t):
            for lab in (x.strip() for x in m.group(1).split(",")):
                if lab not in cited:
                    continue
                g, a, b = where[lab]
                if not (f == g and a <= m.start() < b) and f not in cited[lab]:
                    cited[lab].append(f)
    return cited


def placement(where: pathlib.Path = HERE / "figstyle"):
    """The module in where that defines BACKS (gallery_placement), or None if none does."""
    for f in sorted(where.glob("*.py")):
        if re.search(r"^BACKS\s*[:=]", f.read_text(), re.M):
            sys.path.insert(0, str(f.parent))
            spec = importlib.util.spec_from_file_location(f.stem, f)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


def order(where: pathlib.Path = HERE / "figstyle") -> tuple:
    """gallery_figures.ORDER, read without importing the gallery (which loads its data)."""
    tree = ast.parse((where / "gallery_figures.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "ORDER")


def _book():
    spec = importlib.util.spec_from_file_location("make_results_book", HERE / "make_results_book.py")
    book = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(book)
    return book


def gallery(where: pathlib.Path = HERE / "figstyle", book=None) -> int:
    """The gallery mode; book: an object with TIERS ({tier: [key]}), PREFIX, check_spec and M (mlst_results, for
    CAPTIONS): make_results_book."""
    book = book or _book()
    book.check_spec()
    g = placement(where)
    b, tiers = getattr(g, "BACKS", {}), getattr(g, "TIER", {})
    bad, moved, placed = [], [], []
    for tier, items in book.TIERS.items():
        print(f"== {tier}")
        n = {"Fig.": 0, "Table": 0}
        for item in items:
            kind = "Table" if item.startswith("tab:") else "Fig."
            n[kind] += 1
            s = b.get(item)
            print(f"{kind} {book.PREFIX[tier]}{n[kind]:<3} {item:<28} {s or 'NO CITING SENTENCE'}")
            bad += [] if s else [item]
            moved += [f"{item} ({tier} here, {tiers[item]} in the gallery)"] if item in tiers and tiers[item] != tier else []
            placed.append(item)
    absent = [k for k in placed if k not in tiers]
    unplaced = [k for k in tiers if k not in placed]
    labels = {lab for k in placed for lab in k.split(" + ")}
    tables = sorted({lab for lab, *_ in book.M.CAPTIONS.values()} - labels - set(getattr(g, "DROPPED_TABLES", ())))
    pages = [k for k in order(where) if k not in labels
             and k not in set(getattr(g, "DROPPED", ())) | set(getattr(g, "GALLERY_ONLY", ()))]
    for what, keys, sep in (("placed item(s) with no planned citing sentence", bad, ", "),
                            ("placed differently from the gallery's map", moved, "; "),
                            ("placed in the book but in no tier of the gallery's map", absent, ", "),
                            ("in the gallery's map but placed nowhere in the book", unplaced, ", "),
                            ("generated table(s) neither placed nor dropped", tables, ", "),
                            ("gallery page(s) neither placed nor dropped", pages, ", ")):
        if keys:
            print(f"\n{len(keys)} {what}: {sep.join(keys)}")
    return 1 if bad or moved or absent or unplaced or tables or pages else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", type=pathlib.Path, help="the .tex root, e.g. paper/journal/main.tex")
    ap.add_argument("--gallery", action="store_true", help="report the placement's planned citing sentences")
    a = ap.parse_args()
    if a.gallery:
        return gallery()
    if a.root is None:
        ap.error("give a .tex root or --gallery")
    cited = citations(a.root)
    unc = sorted(lab for lab, fs in cited.items() if not fs)
    for lab, fs in sorted(cited.items()):
        print(f"{lab:<28} {', '.join(str(f.relative_to(a.root.parent)) for f in fs) or 'NOT CITED'}")
    print(f"\n{len(cited)} figure and table labels; {len(unc)} not cited" + (f": {', '.join(unc)}" if unc else ""))
    return 1 if unc else 0


if __name__ == "__main__":
    raise SystemExit(main())
