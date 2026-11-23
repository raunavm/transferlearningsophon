#!/usr/bin/env python3
"""The paper's data release (PI 2026-11-22; MLST asks for a data availability statement): the values behind every
figure and table that figstyle/gallery_placement.py lists (main text, appendix and supplement), for every network,
run, bin and class, as CSV files with a README.txt, from the same files the tables and figures read.

    python3 experiments/FIGS/data_release.py --out DIR                 # the whole release
    python3 experiments/FIGS/data_release.py --out DIR --no-figures    # without the figures (no TeX needed)

One source: nothing here selects, averages or rounds a result.
  * Tables and macros. make_tables.build() installs the recorder (install, install_mlst): every table cell is caught
    where its token is printed (mlst_results.uniform_columns and _render), with the mean and SD or the values at full
    precision and the runs' values that _cell, mlst_v4._value_cell, mlst_v4._jc_cell and TAB.two averaged; every
    macro number where a formatter writes it (make_tables.fmt_pm, fmt_int, ...; mlst_results._sig2, _two, ...).
    finish() matches every number of the generated tables (by row and column) and of results_generated.tex to its
    value and stops the build on one that is not its value rounded at its printed precision (the check runs where
    the data is read; tests/test_data_release.py repeats it against the CSV files).
  * Figures: matplotlib's Axes plotting calls are wrapped while gallery_figures' pages draw (FigureRecorder), so a
    figure's CSV holds the series as plotted: figure id, panel, series, x, y, error bars and bands.
  * Per-run values: the data files the build and the figures read (pathlib.Path.read_text while they run), each
    flattened to CSV (per_run/), so every mean and SD can be recomputed from the runs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "experiments/FIGS/data"
PLACEMENT = REPO / "experiments/FIGS/figstyle/gallery_placement.py"
JOURNAL = REPO / "paper/journal"
# /data holds 2 TiB at 88% with a hard line at 90% (PI 2026-11-22): the release stays well under 200 MB; above BUDGET_MB
# the per-run arrays longer than THIN_MIN (the ROC and sculpting curves on their efficiency grids) are thinned
BUDGET_MB = 150
THIN_MIN = 100
MIN_POINTS = 50                 # a thinned array keeps at least this many points
MAX_STRIDE = 8
SKIP_KEYS = ("search",)         # a probe fit's scan over the regularization grid: the model selection, behind no value
# the embeddings (one run each) are the UMAP figures' series; their per-jet files are not repeated in per_run/
SKIP_DIRS = ("umap",)
MAX_LAYOUTS = 30                # per folder; the rarer layouts go one number a row into one long file
WIDE = 300                      # an object with more scalars than this is a map: one row per key
TOK_OPEN, TOK_CLOSE = "\x02", "\x03"
TOKEN = re.compile(TOK_OPEN + r"(\w) ([^" + TOK_CLOSE + r"]*)" + TOK_CLOSE)
LABEL_RE = re.compile(r"\\label\{(tab:[^}:]+)\}")

# ------------------------------------------------------------------ numbers as printed


Num = collections.namedtuple("Num", "s value dec q")      # printed string, value, decimals, rounding quantum


def _skip_group(t: str, i: int, open_: str = "{", close: str = "}") -> int:
    """The index after the balanced group that opens at t[i] (t[i] == open_)."""
    depth, j = 0, i
    while j < len(t):
        c = t[j]
        if c == "\\":
            j += 2
            continue
        depth += (c == open_) - (c == close)
        j += 1
        if depth == 0:
            return j
    return len(t)


# a command and how many braced arguments go with it; the text after them stays (\multicolumn's content)
_DROP = {"pending": 1, "ref": 1, "eqref": 1, "cite": 1, "citep": 1, "label": 1, "hspace": 1, "vspace": 1, "rule": 2,
         "cmidrule": 1, "cline": 1, "phantom": 1, "hphantom": 1, "vphantom": 1, "setlength": 2, "addlinespace": 0,
         "multicolumn": 2, "makebox": 0, "raisebox": 1, "textcolor": 1, "color": 1, "rowcolor": 1, "caption": 1,
         "captionof": 2, "TableContinued": 2, "fontseries": 1}
_DROP_RE = re.compile(r"\\(" + "|".join(sorted(_DROP, key=len, reverse=True)) + r")(?![A-Za-z])\*?")


def _drop(t: str) -> str:
    out, i = [], 0
    while (m := _DROP_RE.search(t, i)):
        out.append(t[i:m.start()])
        j = m.end()
        for _ in range(_DROP[m.group(1)]):
            while j < len(t) and t[j] in " \n":
                j += 1
            while j < len(t) and t[j] in "[(":          # optional arguments: [..] and booktabs' (lr)
                j = _skip_group(t, j, t[j], "]" if t[j] == "[" else ")")
            if j < len(t) and t[j] == "{":
                j = _skip_group(t, j)
        while j < len(t) and t[j] == "[":
            j = _skip_group(t, j, "[", "]")
        i = j
    out.append(t[i:])
    return "".join(out)


_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\\times|×)\s*10\s*\^?\s*\{?\s*([-+]?\s*\d+)\s*\}?")
_POW = re.compile(r"(?<![\d.,])10\s*\^\s*\{?\s*([-+]?\s*\d+)\s*\}?")
_PLAIN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])(?!\.\d)")


def _quantum(digits: str, dec: int) -> float:
    """The printed precision: the last decimal shown; for an integer, its last nonzero digit (1,800 may be a rounded
    1,803)."""
    if dec:
        return 10.0 ** -dec
    d = digits.replace(",", "")
    return 10.0 ** (len(d) - len(d.rstrip("0"))) if d.strip("0") else 1.0


def numbers(text: str) -> list:
    """Every number a TeX string prints, in order, as Num: thousands separators ({,}), a leading minus in any of its
    forms, and m x 10^e read as one number; subscripts and superscripts (R_{30}, $^{\\S}$), \\pending{...} markers,
    references and lengths are not numbers."""
    t = _drop(text)
    for a, b in ((r"\ensuremath{-}", "-"), ("$-$", "-"), ("\u2212", "-"), ("\u2013", "--"), ("{,}", ",")):
        t = t.replace(a, b)
    found = []
    for rx, mant in ((_SCI, True), (_POW, False)):
        def take(m, mant=mant):
            e = int(m.group(2 if mant else 1).replace(" ", ""))
            ms, dec = (m.group(1), len(m.group(1).partition(".")[2])) if mant else ("1", 0)
            found.append((m.start(), Num(m.group(0).strip(), float(ms) * 10.0 ** e, dec, 10.0 ** (e - dec))))
            return " " * len(m.group(0))
        t = rx.sub(take, t)
    t = re.sub(r"[_^]\s*\{[^{}]*\}", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"[_^]\s*(?:\\[A-Za-z]+|[A-Za-z0-9*])", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"\\[A-Za-z]+\*?|\\.", lambda m: " " * len(m.group(0)), t)
    for m in _PLAIN.finditer(t):
        i = m.start()
        sign = ""
        if i and t[i - 1] in "-+" and (i < 2 or (t[i - 2] not in "-+" and not t[i - 2].isalnum())):
            sign = t[i - 1]
        frac = m.group(2) or ""
        v = float(m.group(1).replace(",", "") + ("." + frac if frac else ""))
        found.append((i - len(sign), Num(sign + m.group(0), -v if sign == "-" else v, len(frac),
                                         _quantum(m.group(1), len(frac)))))
    return [n for _, n in sorted(found, key=lambda x: x[0])]


def close(value, num: Num, rule: str = "round") -> bool:
    """value printed as num under rule: round (to the printed precision), at_most (value <= printed: "equal within
    3%"), exact (the printed number is the value)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    p = num.value
    eps = 1e-9 * max(abs(p), abs(v), num.q)
    if rule == "at_most":
        return v <= p + eps
    if rule == "exact":
        return abs(v - p) <= eps
    return abs(v - p) <= num.q / 2 + eps


# ------------------------------------------------------------------ table cells as printed

Cell = collections.namedtuple("Cell", "part block row column text nums")
_ENV = re.compile(r"\\(begin|end)\{(tabular\*?|longtable|tabularx)\}")
_NARGS = {"tabular": 1, "tabular*": 2, "tabularx": 2, "longtable": 1}
_LEAD = re.compile(r"\s*(?:\\(?:toprule|midrule|bottomrule|hline|newpage|pagebreak|nopagebreak|endfirsthead|endhead|"
                   r"endfoot|endlastfoot|addlinespace|cmidrule|cline|noalign|specialrule|penalty)(?![A-Za-z])\*?)")
K_SIZES = ((r"\Kfull", "188"), (r"\Kres", "162"), (r"\Kmid", "43"), (r"\Kcoarse", "17"))
SUBHEADINGS = ("17-class partitions",)      # mlst_results.PARTITIONS: under the heading above it, in italics or not


def _split0(s: str, sep: str) -> list:
    """s split at sep outside braces (\\& is no separator)."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and not s.startswith(sep, i):
            cur.append(s[i:i + 2])
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        if depth == 0 and s.startswith(sep, i):
            out.append("".join(cur))
            cur, i = [], i + len(sep)
            if sep == "\\\\":       # a row end: \\*, \\[2pt]
                while i < len(s) and s[i] == "*":
                    i += 1
                if i < len(s) and s[i] == "[":
                    i = _skip_group(s, i, "[", "]")
            continue
        cur.append(c)
        i += 1
    return out + ["".join(cur)]


def detex(t: str) -> str:
    """A row label or heading as plain text: the label-set sizes, bold and box wrappers, maths and spacing removed."""
    for k, v in K_SIZES:
        t = t.replace(k + "{}", v).replace(k, v)
    t = t.replace(r"{\fontseries{b}\selectfont ", "{").replace(r"\quad", " ").replace(r"\,", " ").replace("{,}", ",")
    for a, b in ((r"\pm", "±"), (r"\to", "→"), (r"\%", "%"), (r"\bar", ""), (r"\mu", "μ"), (r"\tau", "τ"), (r"\nu", "ν"),
                 (r"\epsB", "eps_B"), (r"\epsS", "eps_S"), ("~", " "), ("$", "")):
        t = t.replace(a, b)
    t = _drop(t)
    t = re.sub(r"\^\{(?![^{}]*\d)[^{}]*\}|\^(?:\\[A-Za-z]+|\*)", "", t)     # marks ($^{\dagger}$), not 10^{4}
    t = re.sub(r"\\(?:mbox|textit|textbf|emph|text|mathrm|ensuremath|textsc|textrm|mathit)\s*\{", "{", t)
    t = re.sub(r"\\([A-Za-z]+)\s*", r"\1", t).replace("{", "").replace("}", "").replace("\\", "")    # \rho: rho
    return " ".join(t.split())


def _tabulars(text: str) -> list:
    """The bodies of a file's top-level tabular, tabular* and longtable environments (column specification removed)."""
    out, i = [], 0
    while True:
        m = _ENV.search(text, i)
        while m and m.group(1) != "begin":
            m = _ENV.search(text, m.end())
        if not m:
            return out
        j = m.end()
        while j < len(text) and text[j] in " \n":
            j += 1
        if j < len(text) and text[j] == "[":
            j = _skip_group(text, j, "[", "]")
        for _ in range(_NARGS[m.group(2)]):
            while j < len(text) and text[j] in " \n":
                j += 1
            if j < len(text) and text[j] == "{":
                j = _skip_group(text, j)
        depth, k, end = 1, j, len(text)
        for mm in _ENV.finditer(text, j):
            depth += 1 if mm.group(1) == "begin" else -1
            if depth == 0:
                end, k = mm.start(), mm.end()
                break
        else:
            k = len(text)
        out.append(text[j:end])
        i = k


def _lead(row: str) -> tuple:
    """(row without its leading rules and page commands, whether a \\midrule was among them)."""
    mid = False
    while True:
        m = _LEAD.match(row)
        if not m:
            return row, mid
        mid |= m.group(0).strip().startswith(r"\midrule")       # not \cmidrule
        j = m.end()
        while j < len(row) and row[j] in " \n":
            j += 1
        while j < len(row) and row[j] in "[({":
            if row[j] == "{":
                j = _skip_group(row, j)
            else:
                j = _skip_group(row, j, row[j], "]" if row[j] == "[" else ")")
            while j < len(row) and row[j] in " \n":
                j += 1
        row = row[j:]


def _span(cell: str) -> tuple:
    m = re.match(r"\s*\\multicolumn\{(\d+)\}", cell)
    return (int(m.group(1)) if m else 1), (_drop(cell).strip().removeprefix("{").removesuffix("}") if m else cell)


def cells(text: str) -> list:
    """Every value cell of a generated table file, in reading order: (tabular index, the italic heading above it, the
    row label, the column heading, the cell's TeX, the numbers it prints). The first column is the row label; the
    heading rows are those above the first \\midrule (a longtable's, above \\endfirsthead)."""
    out = []
    for part, body in enumerate(_tabulars(re.sub(r"(?<!\\)%[^\n]*", "", text))):
        if r"\endfirsthead" in body:
            head, rest = body.split(r"\endfirsthead", 1)
            last = max(rest.rfind(k) for k in (r"\endhead", r"\endfoot", r"\endlastfoot"))
            rest = rest[last:] if last >= 0 else rest
            head_rows, body_rows, inbody = _split0(head, "\\\\"), _split0(rest, "\\\\"), False
            rows = [(r, False) for r in head_rows] + [(r, True) for r in body_rows]
        else:
            rows, inbody = [], r"\midrule" not in body
            for r in _split0(body, "\\\\"):
                r2, mid = _lead(r)
                inbody |= mid
                rows.append((r, inbody))
        header = collections.defaultdict(list)
        top = sub = ""          # an italic heading, and a plain one under it ("17-class partitions")
        for raw, isbody in rows:
            r, _ = _lead(raw)
            if not r.strip() or r"\caption" in r:
                continue
            cs = _split0(r, "&")
            if not isbody:
                col = 0
                for c in cs:
                    n, t = _span(c)
                    t = detex(t)
                    for k in range(col, col + n):
                        if t and (not header[k] or header[k][-1] != t):
                            header[k].append(t)
                    col += n
                continue
            if len(cs) == 1:
                if r.strip().startswith(r"\multicolumn"):
                    h = _span(r)[1]
                    if re.search(r"\\(?:textit|itshape)", h) and detex(h) not in SUBHEADINGS:
                        top, sub = detex(h), ""
                    else:
                        sub = detex(h)
                continue
            block = f"{top} / {sub}" if top and sub else top or sub
            row, col = detex(cs[0]), _span(cs[0])[0]
            # the label column's heading names the quantity of a part (tab:massdetail: resolution, median response)
            q = " / ".join(header.get(0, []))
            for c in cs[1:]:
                n, t = _span(c)
                t = t.strip()
                head = " / ".join(header.get(col, [])) or f"#{col}"
                out.append(Cell(part, block, row, f"{q}: {head}" if q else head, t, numbers(t)))
                col += n
    return out


def table_label(text: str) -> str | None:
    m = LABEL_RE.search(text)
    return m.group(1) if m else None


# ------------------------------------------------------------------ the recorder


def _floats(xs) -> list:
    return [float(x) for x in xs]


def _pm(out, mean, sd):
    if sd == 0:
        return [float(mean)], ["round"]
    return [float(mean), float(sd)], ["round", "at_most" if "<" in out else "round"]


_ONE = lambda out, x, *a, **k: ([float(x)], ["round"])       # noqa: E731
# what each formatter prints, in order: (values, rules)
MT_SPECS = {"fmt_pm": _pm, "fmt_int": _ONE, "fmt": _ONE, "fmt_one": _ONE, "fmt_ratio": _ONE, "fmt_sci": _ONE,
            "fmt_one_sci": _ONE, "fmt_auc": _ONE,
            "fmt_p": lambda out, p: None if p is None else ([float(p)], ["round"]),
            "fmt_paired": lambda out, r, lo, hi: (_floats((r, lo, hi)), ["round"] * 3),
            "words": lambda out, n: ([float(n)], ["round"]) if str(out).isdigit() else None}
M_SPECS = {"_sig2": _ONE, "_lone": _ONE, "_tab1": _ONE, "_at": _ONE,
           "_two": lambda out, a, b, *r, **k: (_floats((a, b)), ["round"] * 2),
           "_within": lambda out, dev: ([100 * float(dev)], ["at_most"]),
           "_pctsd": lambda out, x: None if x is None else ([100 * float(x)], ["round"])}


class Recorder:
    """What one build printed: formatter calls (log), table tokens (renders: as printed; pos: by row and column, per
    table label), the runs behind each mean and SD (runs), and the macros with the stretch of the log that wrote
    them. finish() matches them to the generated files (result)."""

    def __init__(self):
        self.log = []           # [formatter, output, values, rules, nums, claimed]
        self.renders = []       # [kind, values, output, runs, nums, claimed, the table label that claimed it]
        self.rmark = 0
        self.pos = collections.defaultdict(list)    # label -> [entry]
        self.pending = []
        self.macros = []        # (name, body, log start, log end)
        self.mark = 0
        self.runs = {}          # (repr mean, repr sd) -> the runs' values
        self.depth = 0
        self.result = None

    def note_runs(self, out, vals) -> None:
        """The runs behind each mean +- SD token of out, when they reproduce it."""
        v = [float(x) for x in vals if x is not None]
        if len(v) < 2 or not isinstance(out, str):
            return
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
        for kind, body in TOKEN.findall(out):
            parts = body.split()
            if kind.lower() == "p" and len(parts) >= 2:
                m, s = float(parts[0]), float(parts[1])
                if abs(m - mean) <= 1e-9 * max(1.0, abs(m)) and abs(s - sd) <= 1e-9 * max(1.0, abs(s)):
                    self.runs[tuple(parts[:2])] = v

    def _runs_of(self, kind: str, parts: list):
        if kind == "p":
            return self.runs.get(tuple(parts[:2]))
        return [float(parts[0])] if kind == "s" and len(parts) == 1 else None

    def positional(self, tab: str, out: str) -> None:
        """uniform_columns(tab) -> out: each token cell by its row and column, with its values."""
        if TOK_OPEN not in tab:
            return
        a, b = cells(tab), cells(out)
        if len(a) != len(b):
            return
        ents = []
        for ca, cb in zip(a, b):
            toks = TOKEN.findall(ca.text)
            if len(toks) == 1:
                kind, parts = toks[0][0].lower(), toks[0][1].split()
                vals = _floats(parts[:2] if kind == "p" else parts)
                ents.append({"row": ca.row, "column": ca.column, "block": cb.block, "kind": kind, "values": vals,
                             "runs": self._runs_of(kind, parts)})
            elif ca.nums or toks:
                ents.append({"row": ca.row, "column": ca.column, "block": cb.block, "kind": "literal"})
        lab = table_label(tab)
        if lab:
            self.pos[lab].extend(ents)
        else:
            self.pending.extend(ents)

    def claim(self, label: str) -> None:
        """The end of a table (_float, _longtable): the cells printed since the last end are its own."""
        self.pos[label].extend(self.pending)
        self.pending = []
        for e in self.renders[self.rmark:]:
            e[6] = label
        self.rmark = len(self.renders)

    # ---------------------------------------------------------------- matching (finish)
    def _index(self, items, nums_at: int, claimed_at: int, limit=None):
        idx = collections.defaultdict(collections.deque)
        for k, e in enumerate(items if limit is None else items[:limit]):
            if e[claimed_at] or e[2] is None:
                continue
            if e[nums_at] is None:
                e[nums_at] = tuple(n.s for n in numbers(e[1]))
            if e[nums_at]:
                idx[e[nums_at]].append(e)
        return idx

    def match(self, out: dict) -> list:
        """Every number of the generated tables and macros with its value: result = {"tables": {file: [match]},
        "macros": [match], "notes": [...]}. Returns the mismatches."""
        bad, res = [], {"tables": {}, "macros": [], "notes": []}
        for e in self.renders:
            e[4] = tuple(n.s for n in numbers(e[2]))
        rpool = collections.defaultdict(collections.deque)     # by printed numbers: the table's own first, any
        for e in self.renders:
            if e[4]:
                rpool[(e[6], e[4])].append(e)
                rpool[e[4]].append(e)
        lpool = self._index(self.log, 4, 5)
        for rel, text in out.items():
            if not (rel.startswith("tables/") and rel.endswith(".tex")):
                continue
            try:
                cs = cells(text)
            except Exception as exc:          # a file the parser cannot read is listed, never a stopped build
                res["notes"].append(f"{rel}: not parsed ({type(exc).__name__}: {exc})")
                continue
            label = table_label(text)
            # a cell's place: its heading, row and column (and which of equal ones); a row and column without the
            # heading only where the pair is one cell on both sides (a heading the table adds after printing)
            pos, alone, occ = {}, collections.defaultdict(list), collections.Counter()
            for e in self.pos.get(label, []) if label else []:
                k = (e["block"], e["row"], e["column"])
                pos[k + (occ[k],)] = e
                occ[k] += 1
                alone[(e["row"], e["column"])].append(e)
            once = collections.Counter((c.row, c.column) for c in cs if c.nums)
            occ = collections.Counter()
            rows = []
            for n, c in enumerate(cs):
                if not c.nums:
                    continue
                k = (c.block, c.row, c.column)
                e = pos.get(k + (occ[k],))
                occ[k] += 1
                if e is None and once[(c.row, c.column)] == 1 and len(alone[(c.row, c.column)]) == 1:
                    e = alone[(c.row, c.column)][0]
                m = {"cell": len(rows), "part": c.part, "block": c.block, "row": c.row, "column": c.column,
                     "printed": c.text, "nums": c.nums, "kind": "printed", "basis": "printed",
                     "values": [x.value for x in c.nums], "rules": ["exact"] * len(c.nums)}
                if e is not None and e["kind"] != "literal":
                    vals = e["values"]
                    if e["kind"] == "p":
                        vals = vals[:len(c.nums)] if len(c.nums) in (1, 2) else None
                    elif len(vals) != len(c.nums):
                        vals = None
                    m.update(kind=e["kind"], basis="position", runs=e["runs"], token=e["values"])
                    if vals is None or not all(close(v, x) for v, x in zip(vals, c.nums)):
                        bad.append(f"{rel}: under {c.block!r}, row {c.row!r}, column {c.column!r} prints {c.text!r}, "
                                   f"its value is {e['values']}")
                        m.update(values=e["values"], rules=["round"] * len(e["values"]))
                    else:
                        m.update(values=vals, rules=["round"] * len(vals))
                    rows.append(m)
                    continue
                key = tuple(x.s for x in c.nums)
                for r in (rpool.get((label, key)), rpool.get(key)):
                    while r and r[0][5]:
                        r.popleft()
                    if r:
                        break
                if r:
                    e2 = r.popleft()
                    e2[5] = True
                    kind = e2[0]
                    vals = e2[1][:len(c.nums)] if kind == "p" else e2[1]
                    if len(vals) == len(c.nums) and all(close(v, x) for v, x in zip(vals, c.nums)):
                        m.update(kind=kind, basis="text", runs=e2[3], token=e2[1], values=vals,
                                 rules=["round"] * len(vals))
                        rows.append(m)
                        continue
                got = self._take(lpool, c.nums)
                if got:
                    m.update(kind="formatter", basis="text", values=[g[0] for g in got], rules=[g[1] for g in got],
                             formatter=";".join(sorted({g[2] for g in got})))
                rows.append(m)
            res["tables"][rel] = {"label": label, "rows": rows}
        self._match_macros(res, bad)
        self.result = res
        return bad

    def _take(self, pool, nums) -> list | None:
        """The values of nums from formatter calls of the same output (FIFO), whole or number by number."""
        key = tuple(x.s for x in nums)
        q = pool.get(key)
        while q and q[0][5]:
            q.popleft()
        if q:
            e = q.popleft()
            e[5] = True
            if len(e[2]) == len(nums) and all(close(v, x, r) for v, x, r in zip(e[2], nums, e[3])):
                return [(v, r, e[0]) for v, r in zip(e[2], e[3])]
        got = []
        for x in nums:
            q = pool.get((x.s,))
            while q and q[0][5]:
                q.popleft()
            if not q:
                return None
            got.append(q[0])
        if all(close(e[2][0], x, e[3][0]) for e, x in zip(got, nums)):
            for e in got:
                e[5] = True
            return [(e[2][0], e[3][0], e[0]) for e in got]
        return None

    def _match_macros(self, res: dict, bad: list) -> None:
        for name, body, start, end in self.macros:
            nums = numbers(body)
            vals = [None] * len(nums)
            for e in reversed(self.log[start:end]):
                if None not in vals:
                    break
                if e[5] or e[2] is None:
                    continue
                if e[4] is None:
                    e[4] = tuple(n.s for n in numbers(e[1]))
                en = e[4]
                if not en or len(en) != len(e[2]):
                    continue
                for i in range(len(nums) - len(en) + 1):
                    if all(vals[i + j] is None and nums[i + j].s == en[j] for j in range(len(en))):
                        for j in range(len(en)):
                            vals[i + j] = (e[2][j], e[3][j], e[0])
                        break
            for i, x in enumerate(nums):
                v = vals[i]
                if v is not None and not close(v[0], x, v[1]):
                    bad.append(f"\\{name}: prints {x.s} in {body!r}, its value is {v[0]!r} ({v[2]}, {v[1]})")
                res["macros"].append({"macro": name, "body": body, "number": i, "printed": x.s,
                                      "value": x.value if v is None else v[0], "rule": "exact" if v is None else v[1],
                                      "basis": "printed" if v is None else "formatter", "formatter": "" if v is None else v[2]})
            if not nums:
                res["macros"].append({"macro": name, "body": body, "number": "", "printed": "", "value": "",
                                      "rule": "", "basis": "", "formatter": ""})


# ------------------------------------------------------------------ hooks (make_tables.build calls these)

def _unwrapped(f):
    """f without the wrappers of a build that stopped before finish() (they are marked _release)."""
    while getattr(f, "_release", False):
        f = f.__wrapped__
    return f


def _wrap(g: dict, name: str, spec, rec: Recorder, orig) -> None:
    f = orig.get(name)
    if f is None:
        return

    @functools.wraps(f)
    def w(*a, **k):
        out = f(*a, **k)
        if rec.depth == 0 and isinstance(out, str):
            try:
                vr = spec(out, *a, **k)
            except Exception:       # a value the spec cannot read is a printed number, never a stopped build
                vr = None
            rec.log.append([name, out, vr[0] if vr else None, vr[1] if vr else None, None, False])
        return out
    w._release = True
    g[name] = w


def install(g: dict) -> Recorder:
    """make_tables.build(): wrap its formatters and Emitter.macro (a stopped build's wrappers are taken off first, so
    nothing is wrapped twice; finish() puts the originals back); the recorder is g["_RELEASE"]."""
    rec = Recorder()
    orig = {n: _unwrapped(g[n]) for n in MT_SPECS if n in g}
    orig["Emitter.macro"] = _unwrapped(g["Emitter"].__dict__["macro"])
    g["_RELEASE_ORIG"] = orig
    for name, spec in MT_SPECS.items():
        _wrap(g, name, spec, rec, orig)
    macro = orig["Emitter.macro"]

    @functools.wraps(macro)
    def emit(self, name, body, *a, **k):
        out = macro(self, name, body, *a, **k)
        rec.macros.append((name, out, rec.mark, len(rec.log)))
        rec.mark = len(rec.log)
        return out
    emit._release = True
    g["Emitter"].macro = emit
    g["_RELEASE"] = rec
    return rec


def install_mlst(m, g: dict) -> None:
    """The freshly loaded mlst_results (and its mlst_v4): its formatters, the token printing and the tables' ends."""
    rec = g["_RELEASE"]
    d, V = m.__dict__, m.V4.__dict__
    orig = {n: d[n] for n in M_SPECS if n in d}
    for name, spec in M_SPECS.items():
        _wrap(d, name, spec, rec, orig)
    uc, render, fl, lt = d["uniform_columns"], d["_render"], d["_float"], d["_longtable"]

    @functools.wraps(uc)
    def uniform_columns(tab, *a, **k):
        out = uc(tab, *a, **k)
        try:
            rec.positional(tab, out)
        except Exception:
            pass
        return out

    @functools.wraps(render)
    def _render(kind, v, *a, **k):
        rec.depth += 1
        try:
            out = render(kind, v, *a, **k)
        finally:
            rec.depth -= 1
        kk = kind.lower()
        parts = [repr(float(x)) for x in v]
        rec.renders.append([kk, _floats(v[:2] if kk == "p" else v), out, rec._runs_of(kk, parts), None, False, None])
        return out

    def ends(f):
        @functools.wraps(f)
        def w(key, *a, **k):
            out = f(key, *a, **k)
            rec.claim(m.CAPTIONS[key][0] if key in m.CAPTIONS else key)
            return out
        return w
    d.update(uniform_columns=uniform_columns, _render=_render, _float=ends(fl), _longtable=ends(lt))

    def runs_hook(ns: dict, name: str, values_of):
        f = ns.get(name)
        if f is None:
            return

        @functools.wraps(f)
        def w(*a, **k):
            out = f(*a, **k)
            try:
                rec.note_runs(out, values_of(a, k))
            except Exception:
                pass
            return out
        ns[name] = w
    runs_hook(d, "_cell", lambda a, k: [x * (a[3] if len(a) > 3 else k.get("scale", 1.0))
                                        for x in (a[0] if a else k["values"]) if x is not None])
    runs_hook(V, "_value_cell", lambda a, k: a[0] if a else k["vals"])
    runs_hook(V, "_jc_cell", lambda a, k: a[0] if a else k["vals"])
    two = m._TableFmt.__dict__["two"].__func__

    def two_hook(a, b, *r, **k):
        out = two(a, b, *r, **k)
        rec.note_runs(out, [a, b])
        return out
    m._TableFmt.two = staticmethod(two_hook)


def finish(out: dict, g: dict) -> None:
    """The end of make_tables.build(): the originals back, every printed number matched to its value, and the build
    stopped on one that is not its value rounded at its printed precision."""
    rec = g.get("_RELEASE")
    for name, f in g.pop("_RELEASE_ORIG", {}).items():
        if name == "Emitter.macro":
            g["Emitter"].macro = f
        else:
            g[name] = f
    if rec is None:
        return
    bad = rec.match(out)
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} printed number(s) are not their value rounded at the printed precision "
                         "(experiments/FIGS/data_release.py):\n  " + "\n  ".join(bad[:30]))


# ------------------------------------------------------------------ figures


class FigureRecorder:
    """matplotlib's Axes plotting calls wrapped while a gallery page draws: each call's series as plotted."""
    CALLS = ("plot", "step", "errorbar", "fill_between", "fill_betweenx", "scatter", "stairs", "bar", "barh", "hist",
             "imshow", "pcolormesh", "axhline", "axvline", "hlines", "vlines", "axhspan", "axvspan", "text")

    def __init__(self):
        self.calls, self.depth, self.current, self.orig = [], 0, None, {}

    def install(self) -> None:
        import matplotlib.axes
        A = matplotlib.axes.Axes
        for n in self.CALLS:
            self.orig[n] = getattr(A, n)
            setattr(A, n, self._wrap(n, self.orig[n]))

    def uninstall(self) -> None:
        import matplotlib.axes
        for n, f in self.orig.items():
            setattr(matplotlib.axes.Axes, n, f)

    def _wrap(self, name, f):
        rec = self

        @functools.wraps(f)
        def w(ax, *a, **k):
            rec.depth += 1
            try:
                res = f(ax, *a, **k)
            finally:
                rec.depth -= 1
            if rec.depth == 0 and rec.current is not None:
                rec.calls.append((ax, name, a, k, res))
            return res
        return w

    def take(self) -> list:
        """The rows of the page just drawn (one per point), and a fresh start."""
        calls, self.calls = self.calls, []
        return _figure_rows(calls)


def _arr(x):
    import numpy as np
    try:
        return np.ma.filled(np.ma.asarray(x, dtype=float), np.nan).ravel()
    except (TypeError, ValueError):
        return None


def _hex(c) -> str:
    try:
        import matplotlib.colors as mc
        return mc.to_hex(c, keep_alpha=False)
    except (ValueError, TypeError):
        return ""


def _sig(art) -> tuple:
    import matplotlib.lines as ml
    if isinstance(art, ml.Line2D):
        return (_hex(art.get_color()), str(art.get_linestyle()), str(art.get_marker()))
    try:
        fc = art.get_facecolor()
        fc = fc[0] if hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__") else fc
        return (_hex(fc),)
    except Exception:
        return ("",)


def _legends(fig) -> dict:
    """signature -> legend text, over the figure's and its axes' legends."""
    out = {}
    legs = list(getattr(fig, "legends", [])) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for leg in legs:
        hs = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for h, t in zip(hs, leg.get_texts()):
            if h is None:
                continue
            h0 = h[0] if isinstance(h, tuple) else getattr(h, "lines", [h])[0] if hasattr(h, "lines") and h.lines else h
            s = _sig(h0)
            out.setdefault(s, t.get_text())
            out.setdefault(s[:1], t.get_text())
    return out


def _arg(a, k, i, name, default=None):
    return a[i] if len(a) > i else k.get(name, default)


def _series(ax, name, a, k, res):
    """(kind, label, artist, columns) of one plotting call."""
    import numpy as np
    lab = k.get("label", "")
    if name in ("plot", "step"):
        for ln in res:
            yield "line", ln.get_label(), ln, {"x": _arr(ln.get_xdata()), "y": _arr(ln.get_ydata())}
    elif name == "errorbar":
        x, y = _arr(_arg(a, k, 0, "x")), _arr(_arg(a, k, 1, "y"))
        cols = {"x": x, "y": y}
        for key, i, axis in (("yerr", 2, "y"), ("xerr", 3, "x")):
            e = _arg(a, k, i, key)
            if e is None:
                continue
            e = np.asarray(e, dtype=float)
            lo, hi = (e[0], e[1]) if e.ndim == 2 else (e, e)
            base = cols[axis]
            cols[f"{axis}_lo"] = base - np.broadcast_to(lo, base.shape)
            cols[f"{axis}_hi"] = base + np.broadcast_to(hi, base.shape)
        yield "errorbar", lab or res.get_label(), res.lines[0] if res.lines and res.lines[0] is not None else res, cols
    elif name in ("fill_between", "fill_betweenx"):
        v, w = ("x", "y") if name == "fill_between" else ("y", "x")
        x = _arr(_arg(a, k, 0, v))
        y1 = np.broadcast_to(_arr(_arg(a, k, 1, f"{w}1")), x.shape)
        y2 = np.broadcast_to(_arr(np.atleast_1d(_arg(a, k, 2, f"{w}2", 0))), x.shape)
        where = k.get("where")
        if where is not None:
            keep = np.broadcast_to(np.asarray(where, bool), x.shape)
            x, y1, y2 = x[keep], y1[keep], y2[keep]
        yield "band", lab or res.get_label(), res, {v: x, f"{w}_lo": y1, f"{w}_hi": y2}
    elif name == "stairs":
        vals, edges, _ = res.get_data()
        yield "hist", lab or res.get_label(), res, {"x_lo": _arr(edges[:-1]), "x_hi": _arr(edges[1:]), "y": _arr(vals)}
    elif name == "hist":
        n, bins = res[0], _arr(res[1])
        for j, nn in enumerate(np.atleast_2d(np.asarray(n, float))):
            yield "hist", f"{lab} [{j}]" if np.ndim(n) > 1 else lab, None, {"x_lo": bins[:-1], "x_hi": bins[1:], "y": _arr(nn)}
    elif name == "scatter":
        off = np.asarray(res.get_offsets(), float)
        cols = {"x": off[:, 0], "y": off[:, 1]}
        c = res.get_array()
        if c is not None and np.size(c) == len(off):
            cols["z"] = _arr(c)
        yield "scatter", lab or res.get_label(), res, cols
    elif name in ("bar", "barh"):
        ps = list(res.patches)
        x0 = np.array([p.get_x() for p in ps]); w_ = np.array([p.get_width() for p in ps])
        y0 = np.array([p.get_y() for p in ps]); h = np.array([p.get_height() for p in ps])
        cols = ({"x": x0 + w_ / 2, "y_lo": y0, "y": y0 + h} if name == "bar" else {"y": y0 + h / 2, "x_lo": x0, "x": x0 + w_})
        yield "bar", lab or res.get_label(), ps[0] if ps else None, cols
    elif name in ("imshow", "pcolormesh"):
        z = np.ma.filled(np.ma.asarray(res.get_array(), dtype=float), np.nan)
        if z.ndim == 2:
            i, j = np.indices(z.shape)
            yield "image", lab, None, {"y": _arr(i), "x": _arr(j), "z": _arr(z)}
        elif z.ndim == 1:
            yield "image", lab, None, {"x": _arr(np.arange(len(z))), "z": _arr(z)}
    elif name == "axhline":
        yield "hline", lab or res.get_label(), res, {"y": _arr(np.atleast_1d(_arg(a, k, 0, "y", 0)))}
    elif name == "axvline":
        yield "vline", lab or res.get_label(), res, {"x": _arr(np.atleast_1d(_arg(a, k, 0, "x", 0)))}
    elif name in ("hlines", "vlines"):
        v, w = ("y", "x") if name == "hlines" else ("x", "y")
        p = _arr(np.atleast_1d(_arg(a, k, 0, v)))
        cols = {v: p, f"{w}_lo": np.broadcast_to(_arr(np.atleast_1d(_arg(a, k, 1, f"{w}min"))), p.shape),
                f"{w}_hi": np.broadcast_to(_arr(np.atleast_1d(_arg(a, k, 2, f"{w}max"))), p.shape)}
        yield "segments", lab, res, cols
    elif name in ("axhspan", "axvspan"):
        v = "y" if name == "axhspan" else "x"
        yield "span", lab, res, {f"{v}_lo": _arr([_arg(a, k, 0, f"{v}min")]), f"{v}_hi": _arr([_arg(a, k, 1, f"{v}max")])}
    elif name == "text":
        s = str(_arg(a, k, 2, "s", ""))
        if re.search(r"\d", s):
            yield "text", s, None, {"x": _arr([_arg(a, k, 0, "x")]), "y": _arr([_arg(a, k, 1, "y")])}


SERIES_COLS = ("panel", "series_index", "panel_title", "x_label", "y_label", "series", "color", "kind", "coords",
               "n_points", "note")
POINT_COLS = ("panel", "series_index", "i", "x", "y", "x_lo", "x_hi", "y_lo", "y_hi", "z")


def _figure_rows(calls) -> list:
    """[(panel, title, x label, y label, series index, label, colour, kind, coords, {column: array})]"""
    out, figs, count = [], {}, collections.Counter()
    for ax, name, a, k, res in calls:
        fig = ax.figure
        if id(fig) not in figs:
            figs[id(fig)] = _legends(fig)
        leg = figs[id(fig)]
        panel = fig.axes.index(ax) if ax in fig.axes else -1
        title = ax.get_title() or ax.get_title("left") or ax.get_title("right")
        tr = k.get("transform")
        coords = "data" if tr is None or tr == ax.transData else "axes"
        try:
            for kind, label, art, cols in _series(ax, name, a, k, res):
                label = "" if not label or str(label).startswith("_") else str(label)
                sig = _sig(art) if art is not None else ("",)
                if not label and kind != "text":
                    label = leg.get(sig) or leg.get(sig[:1]) or ""
                    if label and kind == "band":
                        label += " (band)"
                cols = {c: v for c, v in cols.items() if v is not None}
                if not cols or not max(len(v) for v in cols.values()):
                    continue
                out.append((panel, title, ax.get_xlabel(), ax.get_ylabel(), count[panel], label, sig[0], kind, coords,
                            cols))
                count[panel] += 1
        except Exception as exc:    # one call the reader does not know is listed in the figure's file, never a stop
            out.append((panel, title, ax.get_xlabel(), ax.get_ylabel(), count[panel], f"{name}: not read ({exc})", "",
                        "unread", coords, {}))
            count[panel] += 1
    return out


def _g(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(float(x), ".8g")


def figure_files(fid: str, rows: list, seen: dict) -> tuple:
    """(figures/<id>_series.csv, figures/<id>.csv): one row per series (panel, its title and axis labels, the legend
    text, colour, kind, the number of points), and one per point. A series another figure already holds point for point
    gets no points; its note names that figure (same as)."""
    series, points = [], []
    for panel, title, xl, yl, si, label, color, kind, coords, cols in rows:
        if kind == "unread":
            series.append([panel, si, title, xl, yl, "", "", kind, coords, 0, label])
            continue
        n = max(len(v) for v in cols.values())
        vals = {c: [_g(x) for x in v] for c, v in cols.items()}
        h = hashlib.sha256(repr((kind, sorted((c, tuple(v)) for c, v in vals.items()))).encode()).hexdigest()
        note = f"text {label}" if kind == "text" else ""
        if h in seen and seen[h][0] != fid:
            series.append([panel, si, title, xl, yl, "" if kind == "text" else label, color, kind, coords, n,
                           f"same as figures/{seen[h][0]}.csv panel {seen[h][1]} series {seen[h][2]}"])
            continue
        seen.setdefault(h, (fid, panel, si))
        series.append([panel, si, title, xl, yl, "" if kind == "text" else label, color, kind, coords, n, note])
        for i in range(n):
            points.append([panel, si, i] + [vals[c][i] if c in vals and i < len(vals[c]) else "" for c in POINT_COLS[3:]])
    return _csv(SERIES_COLS, series), _csv(POINT_COLS, points)


def record_figures(ids) -> tuple:
    """Draw the gallery pages of ids (gallery_figures' modules, its style per page) into a scratch folder with the
    plotting calls recorded: ({id: rows}, {id: note})."""
    import tempfile
    here = REPO / "experiments/FIGS/figstyle"
    sys.path[:0] = [str(here), str(here.parent)]
    import gallery_figures as GF        # noqa: E402  imports the modules, which read the data (gallery_common)
    import gallery_common as C          # noqa: E402
    pages = {p.id: p for mod in GF.MODULES for p in mod.PAGES}
    fr, rows, notes = FigureRecorder(), {}, {}
    fr.install()
    try:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="release_figs_"))
        for fid in ids:
            page = pages.get(fid)
            if page is None:
                notes[fid] = "no gallery page draws it"
                continue
            GF.S.use_style5(simplify=fid != "SOPHON-FIG2")
            fr.current = fid
            try:
                page.draw(tmp / fid)
                rows[fid] = fr.take()
                print(f"{fid}: {len(rows[fid])} series")
            except C.Absent as e:
                fr.take()
                notes[fid] = f"not drawn: {e}"
                print(f"{fid}: absent: {e}")
            except (Exception, SystemExit) as e:
                fr.take()
                notes[fid] = f"not drawn: {type(e).__name__}: {e}"
                print(f"{fid}: FAILED {type(e).__name__}: {e}")
            finally:
                fr.current = None
                import matplotlib.pyplot as plt
                plt.close("all")
    finally:
        fr.uninstall()
    return rows, notes


# ------------------------------------------------------------------ the files read: per-run values


class ReadLog:
    """Every data file (under experiments/FIGS/data/) read through pathlib.Path.read_text while installed."""

    def __init__(self):
        self.paths, self.orig = [], None

    def install(self) -> None:
        self.orig = pathlib.Path.read_text
        log, orig = self, self.orig

        def read_text(p, *a, **k):
            out = orig(p, *a, **k)
            try:
                q = pathlib.Path(p).resolve()
                if q.suffix == ".json" and q.is_relative_to(DATA) and q not in log.seen:
                    log.seen.add(q)
                    log.paths.append(q)
            except Exception:
                pass
            return out
        self.seen = set()
        pathlib.Path.read_text = read_text

    def uninstall(self) -> None:
        if self.orig is not None:
            pathlib.Path.read_text = self.orig


def _scalar(v) -> bool:
    return isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and len(v) <= 80)


def _numlist(v) -> bool:
    return (isinstance(v, list) and bool(v) and all(isinstance(x, (int, float)) and not isinstance(x, bool) or x is None
                                                     for x in v))


def _val(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return "" if v is None else str(v)


def _array(file: str, path: str, key: str, v: list, stride: int) -> list:
    """One numeric list as one row: file, path, key, its length, the stride kept and its values (ten significant
    figures, ";" between them). A list longer than THIN_MIN keeps every stride-th point and its last, never fewer than
    MIN_POINTS."""
    n = len(v)
    s = 1 if n <= THIN_MIN else max(1, min(stride, n // MIN_POINTS))
    idx = range(n) if s == 1 else sorted(set(range(0, n, s)) | {n - 1})
    return [file, path, key, n, s, ";".join("" if v[i] is None else str(v[i]) if isinstance(v[i], int)
                                             else format(v[i], ".10g") for i in idx)]


def _flatten(node, file: str, path: str, out: dict, stride: int) -> None:
    """node's numbers into out: (kind, keys) -> rows. A dict's scalar children are one record (file, path, a column
    each); each numeric list one row (_array); deeper dicts and lists of dicts recursively; SKIP_KEYS left out."""
    if isinstance(node, dict):
        scal = {k: v for k, v in node.items() if k not in SKIP_KEYS and _scalar(v)}
        if len(scal) > WIDE:            # a map keyed by name (a file index): a row per key
            out[("records", ("key", "value"))] += [[file, path, k, _val(v)] for k, v in scal.items()]
        elif scal:
            out[("records", tuple(scal))].append([file, path] + [_val(v) for v in scal.values()])
        for k, v in node.items():
            if k in SKIP_KEYS or _scalar(v):
                continue
            if _numlist(v):
                out[("arrays", ())].append(_array(file, path, str(k), v, stride))
            else:
                _flatten(v, file, f"{path}.{k}" if path else str(k), out, stride)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if _numlist(v):
                out[("arrays", ())].append(_array(file, f"{path}[{i}]", "", v, stride))
            elif isinstance(v, (dict, list)):
                _flatten(v, file, f"{path}[{i}]", out, stride)


ARRAY_COLS = ("file", "path", "key", "n", "stride", "values")


def per_run_files(paths, stride: int = 1) -> tuple:
    """({per_run/<folder>__<layout>.csv: text}, [(file, folder, kind, columns, example path, rows, files)]): the data
    files read, one folder at a time, every number flattened (_flatten): the numeric lists in <folder>__arrays.csv, the
    records one file per layout (records<k>), the rarer layouts one number a row."""
    groups = collections.defaultdict(list)
    for p in paths:
        rel = p.relative_to(DATA)
        if len(rel.parts) > 1 and rel.parts[0] != "v2":
            continue            # the first grid's files: macros of the earlier draft, no figure or table of this paper
        if any(rel.parts[1].startswith(s) for s in SKIP_DIRS if len(rel.parts) > 2):
            continue
        groups[rel.parent.as_posix()].append(p)
    files, index = {}, []
    for folder in sorted(groups):
        out = collections.defaultdict(list)
        for p in sorted(groups[folder]):
            try:
                doc = json.loads(p.read_text())
            except (ValueError, OSError):
                continue
            _flatten(doc, p.name, "", out, stride)
        stem = "per_run/" + (folder.replace("/", "__") if folder != "." else "data")
        arrays = out.pop(("arrays", ()), [])
        layouts = sorted(out.items(), key=lambda kv: -len(kv[1]))
        # the MAX_LAYOUTS largest record layouts a file each, the rest one row per number (key and value columns)
        long = [r[:2] + [k, v] for (_, keys), rows in layouts[MAX_LAYOUTS:] for r in rows
                for k, v in zip(keys, r[2:]) if v != ""]
        own = [(f"{stem}__records{k}.csv", ("file", "path") + keys, rows)
               for k, ((_, keys), rows) in enumerate(layouts[:MAX_LAYOUTS], 1)]
        if long:
            own.append((f"{stem}__records_long.csv", ("file", "path", "key", "value"), long))
        if arrays:
            own.append((f"{stem}__arrays.csv", ARRAY_COLS, arrays))
        for name, cols, rows in own:
            files[name] = _csv(cols, rows)
            index.append((name, folder, "arrays" if name.endswith("arrays.csv") else "records", cols[2:], rows[0][1],
                          len(rows), sorted({r[0] for r in rows})))
    return files, index


# ------------------------------------------------------------------ the release


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _csv(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _j(xs) -> str:
    return ";".join(_val(float(x)) if isinstance(x, (int, float)) else str(x) for x in xs)


TABLE_COLS = ("table", "file", "cell", "part", "block", "row", "column", "printed", "numbers", "values", "rules",
              "kind", "mean", "sd", "n_runs", "run_values", "basis")


def table_csv_names(labels: dict) -> dict:
    """{generated table file: its CSV in the release}, from {file: its label}: tables/<label>.csv (tab:x -> tab_x.csv);
    a second file with the same label (an older generator's: mass.tex beside mlst_mass.tex) and a file with none by
    its own name."""
    by = collections.defaultdict(list)
    for rel, label in labels.items():
        by[label].append(rel)
    out = {}
    for label, rels in by.items():
        main = next((r for r in rels if pathlib.Path(r).name.startswith("mlst_")), rels[0]) if label else None
        for r in rels:
            out[r] = "tables/" + (label.replace(":", "_") if r == main else pathlib.Path(r).stem) + ".csv"
    return out


def _names(rec: Recorder) -> dict:
    return table_csv_names({rel: t["label"] for rel, t in rec.result["tables"].items()})


def table_files(rec: Recorder) -> dict:
    """{tables/<label>.csv: text}: every numeric cell of every generated table, in reading order (cells())."""
    out, names = {}, _names(rec)
    for rel, t in rec.result["tables"].items():
        rows = []
        for m in t["rows"]:
            runs = m.get("runs")
            mean = sd = ""
            if m["kind"] == "p":
                mean, sd = (_val(float(x)) for x in m["token"][:2])
            rows.append([t["label"] or "", "paper/journal/" + rel, m["cell"], m["part"], m["block"], m["row"],
                         m["column"], m["printed"], ";".join(x.s for x in m["nums"]), _j(m["values"]),
                         ";".join(m["rules"]), "formatter:" + m["formatter"] if m["kind"] == "formatter" else m["kind"],
                         mean, sd, len(runs) if runs else "", _j(runs) if runs else "", m["basis"]])
        out[names[rel]] = _csv(TABLE_COLS, rows)
    return out


def used_macros(built: dict) -> set:
    texts = [p.read_text() for p in [JOURNAL / "main.tex", *sorted((JOURNAL / "sections").glob("*.tex"))] if p.exists()]
    texts += [t for rel, t in built.items() if rel.startswith("tables/") and rel.endswith(".tex")]
    return {m for t in texts for m in re.findall(r"\\([A-Za-z]+)", t)}


MACRO_COLS = ("macro", "used_in_paper", "body", "number", "printed", "value", "rule", "basis", "formatter",
              "source_file", "json_path")


def macro_file(rec: Recorder, built: dict) -> str:
    prov = json.loads(built.get("provenance.json", "{}"))
    used = used_macros(built)
    rows = []
    for m in rec.result["macros"]:
        p = prov.get(m["macro"], {})
        rows.append([m["macro"], "yes" if m["macro"] in used else "no", m["body"], m["number"], m["printed"],
                     _val(m["value"]) if m["value"] != "" else "", m["rule"], m["basis"], m["formatter"],
                     p.get("source_file", ""), p.get("json_path", "")])
    return _csv(MACRO_COLS, rows)


def runs_file() -> str:
    M = _module("experiments/FIGS/mlst_results.py", "mlst_results_release")
    runs = M._paper(REPO)
    rows = [[arm, detex(M.LABEL.get(arm, arm)), m] for arm, ms in runs.items() for m in ms]
    return _csv(("configuration", "network", "run"), rows)


def items(built: dict, rec: Recorder, figs: dict, fig_notes: dict) -> list:
    """[(id, kind, tier, number, sentence, files, note)] for every figure and table of gallery_placement, in its order,
    then the generated tables it places nowhere."""
    GP = _module("experiments/FIGS/figstyle/gallery_placement.py", "gallery_placement_release")
    names = _names(rec)
    by_label = collections.defaultdict(list)
    for rel, t in rec.result["tables"].items():
        if t["label"] and names[rel] == "tables/" + t["label"].replace(":", "_") + ".csv":
            by_label[t["label"]].append(rel)
    out = []
    for key, tier, sentence in GP.FIGURES:
        files = [f"figures/{key}_series.csv", f"figures/{key}.csv"] if key in figs else []
        note = fig_notes.get(key, "") if not files else ""
        if key == "PT":
            files.append("build_csv/jet_pt_bins.csv")
        if key == "PC-jc2":
            files.append("build_csv/jetclass2_per_class.csv")
        if key in GP.PROVISIONAL:
            note = (note + "; " if note else "") + GP.PROVISIONAL[key]
        out.append((key, "figure", tier, GP.number(key), sentence, files, note))
    placed = set()
    for key, tier, sentence in GP.TABLES:
        rels = by_label.get(key, [])
        placed.update(rels)
        files = [names[r] for r in rels]
        note = "" if rels else "no generated file carries this label"
        for r in rels:
            body = built.get(r, "")
            if r"\pending{no result yet}" in body and not rec.result["tables"][r]["rows"]:
                note = "pending: no result yet"
        out.append((key, "table", tier, GP.number(key), sentence, files, note))
    for rel, t in rec.result["tables"].items():
        if rel not in placed:
            key = t["label"] if t["label"] and t["label"] not in GP.TIER else pathlib.Path(rel).stem
            out.append((key, "table", "not placed", "", "", [names[rel]],
                        f"generated ({rel}), placed nowhere in the paper"))
    return out


README_HEAD = """DATA RELEASE
Label granularity in supervised pretraining of jet foundation models

Version: commit {commit} of the analysis repository; written {date} by experiments/FIGS/data_release.py.{preview}

This release holds the values behind every figure and table of the paper, its appendix and its
supplementary material: every network, pretraining run, bin and class, at full precision, as CSV
(comma-separated, UTF-8, one header row). Nothing in it is typed by hand. The script that writes it
records each number where the paper's own table and figure code prints or plots it, from the same
result files, and checks that every number printed in the paper's tables and text is the released
value rounded at its printed precision (tests/test_data_release.py repeats the check against these
files). Section ITEMS below maps each figure (by its number and gallery id) and each table (by its
number and LaTeX label) to its files.

FILES
  README.txt          this file
  items.csv           every figure and table: id or label, kind, placement (main text, appendix,
                      supplement), number, the sentence it backs, its files, a note
  runs.csv            the pretraining runs of each configuration: configuration key, network, run
  tables/<label>.csv  one file per table (label tab:x -> tab_x.csv), one row per printed cell
  macros.csv          every number written into the text (the generated LaTeX macros)
  figures/<id>*.csv   two files per figure: its series (<id>_series.csv) and their points
  per_run/*.csv       the result files the tables and figures read, every number of every run
  build_csv/*.csv     two value tables the table build writes beside the paper: jet_pt_bins.csv
                      (every point of the jet-pT figures, mean, SD and run count) and
                      jetclass2_per_class.csv (each JetClass-II class's AUC for every network)

COLUMNS
tables/<label>.csv
  table, file        the table's label and the generated .tex file it is printed from
  cell               the cell's index among the table's numeric cells, in reading order
  part               the index of the tabular within the file (a table too wide or long for one
                     page is set as several)
  block              the italic heading above the row (the downstream mode, a task, a working point)
  row, column        the row label and the column heading(s), " / " between heading rows
  printed            the cell as typeset (LaTeX)
  numbers            the numbers the cell prints, ";" between them
  values             the value behind each printed number at full precision (Python repr), ";"
  rules              how each value is printed: round (rounded at the printed precision), exact
                     (the printed number is the value: a count, a setting or a published value)
  kind               p: mean +- SD over runs; s: a single run's value; o: a number that is not a
                     run's (a reference such as the soft-drop mass); formatter: printed by the
                     table code's own formatter; printed: no value beyond the printed number
  mean, sd           for kind p, the mean and the sample SD (ddof = 1) over the runs
  n_runs, run_values the runs averaged into the cell and each run's value, ";" between them,
                     where the table code averaged them (mean and SD recompute from them)
  basis              position: matched by row and column to the value the table code printed there;
                     text: matched to the value whose printed form the cell carries; printed: none
macros.csv
  macro, used_in_paper  the LaTeX macro and whether main.tex, its sections or tables use it
  body               the macro as written into results_generated.tex
  number, printed    the index and text of each number in the body
  value, rule, basis the value behind it (rule as above; at_most: the value is at most the printed
                     number, for "equal within X%"), and whether a formatter call wrote it
                     (formatter, named) or no recorded value goes beyond the printed text (printed)
  source_file, json_path  the result file and the path inside it the number came from
                     (paper/journal/provenance.json; the sha256 of every input is there and in
                     results_generated.tex)
figures/<id>_series.csv and figures/<id>.csv
  <id>_series.csv has one row per plotted series: panel (its index in drawing order; the paper's
  panel letters follow it), panel_title, x_label, y_label as drawn, series_index (the call's index
  within the panel), series (its legend text where it has one), color (the legend's swatch), kind,
  coords, n_points and a note. kind is line, errorbar, band (filled between y_lo and y_hi), scatter,
  hist (bins x_lo to x_hi), bar, image (a heatmap: row y, column x, value z), hline, vline,
  segments, span, or text (a number printed on the figure, given in the note); coords is data, or
  axes where the call drew in the panel's own coordinates. A series another figure holds point for
  point has no points here; its note names that figure, panel and series.
  <id>.csv has one row per point: panel, series_index, i, x, y, x_lo, x_hi, y_lo, y_hi, z (an
  error bar spans y_lo to y_hi, or x_lo to x_hi, at x), to eight significant figures.
per_run/<folder>__records<k>.csv, __records_long.csv, __arrays.csv
  One folder of result files (its path under experiments/FIGS/data, "/" as "__"). records<k>: one
  row per JSON object of scalars of one layout (file, path inside the file, a column per key);
  records_long: the rarer layouts, one row per number (file, path, key, value); arrays: one row
  per numeric list (file, path, key, n its length, stride, values: ten significant figures, ";"
  between them; a list thinned to every stride-th point keeps its last as well). Section PER-RUN
  FILES lists each file with its columns and an example path. A probe fit's scan over its
  regularization grid ("search") is left out: it chose the fit and is behind no value. The
  embeddings of the UMAP figures are their figures' series.

HOW THE VALUES WERE COMPUTED
Every value is the mean over the independently pretrained networks (runs) of a configuration,
each followed by one linear probe or one fine-tuning, with the sample standard deviation (ddof =
1) over those runs as its uncertainty. Three runs are printed as mean +- SD; two runs as mean +- SD
in tables (the SD of two values is their difference over sqrt 2) and as both values in the text; a
single run as its value, marked. The five random partitions, one run each, are averaged as one
configuration (mean +- SD over the five) where a table pools them, and listed one by one in the
supplement's full tables. Ratios are formed run by run between networks of the same pretraining
seed and then averaged. A ROC curve of a configuration is the mean over its runs of the rejection
at each signal efficiency, with a band of one SD (the range for two runs), drawn where at least the
stated number of background jets pass. The SD is rounded following the Particle Data Group (to two
significant figures when its leading digits are 100 to 354, to one from 355 to 949) and the mean
to the same decimal place; a table column is printed at one place.
Where a table or sentence states a difference beyond three standard deviations, sigma^2 is the sum
of each configuration's variance over its runs and the variance of the difference over bootstrap
replicas of the test jets, shared by every network (each network scored on the same replica); the
bootstrap SDs printed in the paper (tab:sculptboot and the sentences that cite the bootstrap) are
in their tables' and macros' rows, and the bootstrap result files in per_run/ (folders bootstrap,
l3_rerun). Mass sculpting is the Jensen-Shannon distance with logarithms to base 2; the mass
resolution is half the 15.9-84.1% inter-quantile range of the response over its median.

ROUNDING AND THE CHECK
A printed number matches its value when |value - printed| is at most half a unit of its last
printed digit (for an integer, of its last nonzero digit: 1,800 stands for 1,750 to 1,850). The
table build stops when a number in a generated table or macro does not (make_tables.build, via
data_release.finish), and tests/test_data_release.py repeats the check against these files.
{coverage}

WHAT IS NOT IN THIS RELEASE
The simulated jets (JetClass-II, JetClass, the top-tagging reference sample, the Pythia and
Herwig quark/gluon samples and the X->bb and X->bs samples) are public from their own authors and
are not copied here; the per-jet network scores, the pretrained and fine-tuned networks and the
training logs are not in this release (see the paper's data availability statement).
{size}
ITEMS (paper order: main text, appendix, supplement; then the generated tables placed nowhere)
"""


def readme(commit: str, its: list, cover: str, size: str, pindex: list, fig_rows: dict) -> str:
    preview = ("\nPREVIEW: built from the partial fine-tuning results (MLST_FT_PREVIEW=1) while reruns are running; "
               "not the\nfinal release." if os.environ.get("MLST_FT_PREVIEW") == "1" else "")
    out = [README_HEAD.format(commit=commit, date=datetime.date.today().isoformat(), coverage=cover, size=size,
                              preview=preview)]
    for key, kind, tier, num, sentence, files, note in its:
        out.append(f"  {num or '-':<11} {key} ({kind}, {tier})")
        if sentence:
            out.append("      backs: " + sentence)
        out.append("      files: " + (", ".join(files) if files else "none"))
        if kind == "figure" and key in fig_rows:
            panels = sorted({r[0] for r in fig_rows[key]})
            out.append(f"      {len(fig_rows[key])} series in {len(panels)} panel(s)")
        if note:
            out.append("      note: " + note)
    out.append("\nPER-RUN FILES (file: source folder, layout, rows; columns; an example path; the files flattened)")
    for name, folder, kind, keys, path, n, srcs in pindex:
        cols = ", ".join(keys[:12]) + (f", ... ({len(keys)} columns)" if len(keys) > 12 else "")
        out.append(f"  {name}: experiments/FIGS/data/{folder}, {kind}, {n} rows")
        out.append(f"      columns: {cols}")
        out.append(f"      e.g. {srcs[0]} at {path or '(top level)'}; {len(srcs)} file(s)")
    return "\n".join(out) + "\n"


def coverage(rec: Recorder) -> str:
    """How many printed numbers have a value beyond their printed text, by table, and in the macros."""
    tab = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for rel, t in rec.result["tables"].items():
        for m in t["rows"]:
            n = len(m["nums"])
            tab[m["basis"]] += n
            by[t["label"] or rel][m["basis"]] += n
    mac = collections.Counter(m["basis"] for m in rec.result["macros"] if m["printed"])
    lines = [f"Tables: {sum(tab.values())} printed numbers; {tab['position']} matched by row and column to the value "
             f"printed there, {tab['text']} matched by their printed form, {tab['printed']} with no value beyond the "
             f"printed number (counts, settings, published values and numbers the table code prints itself; their "
             f"runs' values are in per_run/)."]
    few = [f"{k} {v['printed']}" for k, v in sorted(by.items()) if v["printed"]]
    if few:
        lines.append("  Numbers with no value beyond the printed one, by table: " + "; ".join(few) + ".")
    lines.append(f"Macros: {sum(mac.values())} printed numbers; {mac['formatter']} written by a formatter from their "
                 f"value, {mac['printed']} with none beyond the printed text (counts, settings and literature values "
                 f"read as text).")
    if rec.result["notes"]:
        lines.append("Not read: " + "; ".join(rec.result["notes"]))
    return "\n".join(_wrap_text(x) for x in lines)


def _wrap_text(s: str, width: int = 100) -> str:
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = "  " + w if out or s.startswith("  ") else w
        else:
            cur = f"{cur} {w}" if cur else w
    return "\n".join(out + [cur])


def release_files(built: dict, rec: Recorder, figs: dict | None = None, fig_notes: dict | None = None,
                  reads=(), commit: str = "", budget_mb: float = BUDGET_MB) -> tuple:
    """({path in the release: text}, report). The per-run arrays are thinned (THIN_MIN) until the whole is within
    budget_mb; the README says by how much."""
    figs, fig_notes = figs or {}, fig_notes or {}
    files = {**table_files(rec), "macros.csv": macro_file(rec, built), "runs.csv": runs_file()}
    files.update({"build_csv/" + pathlib.Path(rel).name: t for rel, t in built.items() if rel.startswith("data_release/")})
    seen = {}
    for fid in figs:
        files[f"figures/{fid}_series.csv"], files[f"figures/{fid}.csv"] = figure_files(fid, figs[fid], seen)
    fixed = sum(len(t.encode()) for t in files.values())
    stride = 1
    while True:
        pr, pindex = per_run_files(reads, stride)
        total = fixed + sum(len(t.encode()) for t in pr.values())
        print(f"release size estimate at per-run stride {stride}: {total / 1e6:.1f} MB "
              f"(tables, macros and figures {fixed / 1e6:.1f} MB)")
        if total <= budget_mb * 1e6 or stride >= MAX_STRIDE or not pr:
            break
        stride *= 2
    files.update(pr)
    size = (f"\nSIZE\nAbout {total / 1e6:.0f} MB in {len(files) + 2} files. "
            + ("Every point of every per-run array is kept.\n" if stride == 1 else
               f"To stay well under 200 MB, every per-run array longer than {THIN_MIN} points (the ROC and sculpting "
               f"curves of each run on their efficiency grids) keeps every {stride}th point and its last, and never "
               f"fewer than {MIN_POINTS} points (its stride column says which); the figures' series keep every "
               f"plotted point.\n"))
    big = sorted(files.items(), key=lambda kv: -len(kv[1]))[:25]
    print("largest files: " + ", ".join(f"{k} {len(v) / 1e6:.1f} MB" for k, v in big))
    by = collections.Counter()
    for k, v in files.items():
        by[k.split("/")[0] if "/" in k else "."] += len(v)
    print("by folder: " + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in by.most_common()))
    its = items(built, rec, figs, fig_notes)
    files["items.csv"] = _csv(("id", "kind", "placement", "number", "sentence", "files", "note"),
                              [(k, kind, t, n, s, " ".join(f), note) for k, kind, t, n, s, f, note in its])
    files["README.txt"] = readme(commit, its, coverage(rec), _wrap_text(size.strip()).replace("SIZE ", "SIZE\n", 1) + "\n",
                                 pindex, figs)
    return files, {"bytes": sum(len(t.encode()) for t in files.values()), "files": len(files), "stride": stride}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the release folder (created)")
    ap.add_argument("--no-figures", action="store_true", help="without the figures (they need TeX)")
    ap.add_argument("--budget-mb", type=float, default=BUDGET_MB)
    a = ap.parse_args(argv)
    reads = ReadLog()
    reads.install()
    try:
        MT = _module("experiments/FIGS/make_tables.py", "make_tables_release")
        built, _, _ = MT.build(REPO)
        rec = MT._RELEASE
        GP = _module("experiments/FIGS/figstyle/gallery_placement.py", "gallery_placement_ids")
        figs, notes = ({}, {}) if a.no_figures else record_figures([k for k, _, _ in GP.FIGURES])
    finally:
        reads.uninstall()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    files, rep = release_files(built, rec, figs, notes, reads.paths, commit or "unknown", a.budget_mb)
    for rel, text in sorted(files.items()):
        p = a.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(f"RELEASE: {rep['files']} files, {rep['bytes'] / 1e6:.1f} MB, per-run stride {rep['stride']}, into {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
