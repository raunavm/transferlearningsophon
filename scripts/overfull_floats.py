#!/usr/bin/env python3
"""The paper build's stop on overfull floats (PI 2026-11-23, LEDGER FG-25: "make overfull boxes over 10 pt in any
float stop the paper build").

Reads the compile log (tectonic's output, scripts/build_render_job.py's PAPER mode) for every
"Overfull \\hbox (N pt too wide)" warning, attributes each to its source, and lists them all. A warning in a generated
table (tables/*.tex, every one a table) or in a line range inside a figure or table environment of another source
(sections/*.tex) is in a float. Exit 4 with a FATAL line when one in a float is more than LIMIT_PT too wide; any other
overfull box is listed only.

    python3 scripts/overfull_floats.py <tectonic log> <directory of main.tex>
"""
from __future__ import annotations

import pathlib
import re
import sys

LIMIT_PT = 10.0
# tectonic: "warning: tables/mlst_sets:26: Overfull \hbox (121.6568pt too wide) in alignment at lines 5--26"
WARN = re.compile(r"^warning: (?:(?P<file>[^:\s]+):(?P<line>\d+): )?Overfull \\hbox \((?P<pt>[\d.]+)pt too wide\)"
                  r"(?:.*? at lines? (?P<a>\d+)(?:--(?P<b>\d+))?)?", re.M)
FLOAT = re.compile(r"\\begin\{(figure|table)(\*?)\}.*?\\end\{\1\2\}", re.S)


def float_lines(tex: str) -> list:
    """[(first line, last line)] of the figure and table environments of a source, 1-based."""
    return [(tex.count("\n", 0, m.start()) + 1, tex.count("\n", 0, m.end()) + 1) for m in FLOAT.finditer(tex)]


def overfull(log: str, texdir: pathlib.Path) -> list:
    """Each distinct overfull box of the log (the compile's passes repeat them): its source, its lines, how far too
    wide (pt) and the float it lies in (None outside one)."""
    out, seen = [], set()
    for m in WARN.finditer(log):
        lo = int(m["a"] or m["line"] or 0)
        hi = int(m["b"] or lo)
        key = (m["file"], lo, hi, m["pt"])
        if key in seen:
            continue
        seen.add(key)
        where, name = None, m["file"]
        if name:
            name = name if name.endswith(".tex") else name + ".tex"
            path = texdir / name
            if pathlib.PurePath(name).parts[0] == "tables":
                where = "a generated table"
            elif path.is_file():
                where = next((f"the float at lines {a}--{b}" for a, b in float_lines(path.read_text(errors="replace"))
                              if a <= hi and lo <= b), None)
        out.append({"source": name or "unknown source", "lines": (lo, hi), "pt": float(m["pt"]), "float": where})
    return out


def main(argv: list | None = None) -> int:
    log, texdir = argv if argv is not None else sys.argv[1:]
    rows = overfull(pathlib.Path(log).read_text(errors="replace"), pathlib.Path(texdir))
    for r in rows:
        print(f"overfull {r['pt']:.1f} pt: {r['source']} lines {r['lines'][0]}--{r['lines'][1]}"
              + (f", in {r['float']}" if r["float"] else ""))
    bad = [r for r in rows if r["float"] and r["pt"] > LIMIT_PT]
    if bad:
        print(f"FATAL: {len(bad)} overfull boxes over {LIMIT_PT:g} pt in floats (PI 2026-11-23, LEDGER FG-25): "
              + "; ".join(f"{r['source']} lines {r['lines'][0]}--{r['lines'][1]}, {r['pt']:.1f} pt" for r in bad))
        return 4
    print(f"overfull boxes: {len(rows)}, none over {LIMIT_PT:g} pt in a float")
    return 0


if __name__ == "__main__":
    sys.exit(main())
