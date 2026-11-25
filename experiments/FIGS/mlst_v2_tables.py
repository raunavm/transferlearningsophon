#!/usr/bin/env python3
"""Tables 1 (versions A and B), 2 and 3 of the rewritten paper, the abstract's numbers and the tests of its clauses
(PI 2026-11-24, LEDGER RW-4, RW-5, RW-9; docs/STYLE.md sections 2-4).

Every value is read through the paper's build: make_tables.build runs once, and the arguments it gives
mlst_results.table_summary and table_tests (the probe fits, the fine-tuning cells, the runs, each tested difference)
are kept, so a cell here is the run values the old tables printed, formatted for the new ones. A fine-tuned cell is
the one-rule cell at its row's recorded rate (mlst_results.merge_fixed) or the word "pending". Table 1 is read from the
class maps (configs/labelmaps). Nothing is typed by hand.

    python3 experiments/FIGS/mlst_v2_tables.py      # writes paper/journal/v2/tables/
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_tables as MT  # noqa: E402

OUT = "paper/journal/v2/tables"
MAPS, RAND_MAPS = "configs/labelmaps/rung_label_maps.v1.csv", "configs/labelmaps/rand_label_map.v2.csv"
NESTED = ("L188", "L162", "R42_Q1", "R16_Q1")
RAND, MASS = "RAND", (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
PENDING = "pending"
SIGMA = 3.0     # the meaning of "significant" (STYLE section 4)

# A distinction of Table 1: (column heading, two lines; what a label set that merges it is said to merge; pairs of
# JetClass-II classes that differ in it). A label set keeps it when it keeps every pair apart and merges it when it
# merges every pair; anything else is "mixed", which stops the build for a nested label set.
DISTINCTIONS = (
    (("QCD", "flavor"), "the QCD classes into one", (("QCD_bb", "QCD_light"), ("QCD_cc", "QCD_b"), ("QCD_s", "QCD_light"))),
    (("$b$, $c$", "number"), "the number of $b$ and $c$ quarks",
     (("X_bb", "X_bq"), ("X_cc", "X_cq"), ("X_YY_bbqq", "X_YY_qqqb"), ("X_YY_bbb", "X_YY_qqb"))),
    (("$s$ vs", "$u$, $d$"), "$s$ with $u$, $d$", (("X_ss", "X_qq"), ("X_cs", "X_cq"), ("X_YY_ssqq", "X_YY_qqqq"))),
    (("$g$ vs", "$u$, $d$, $s$"), "gluons with $u$, $d$, $s$",
     (("X_gg", "X_qq"), ("X_YY_qqgg", "X_YY_qqqq"), ("X_YY_bbg", "X_YY_bbq"))),
    (("$e$ vs", r"$\mu$"), r"$e$ with $\mu$", (("X_ee", "X_mm"), ("X_YY_bbee", "X_YY_bbmm"), ("X_tauhtaue", "X_tauhtaum"))),
    (("$b$, $c$ or", "neither"), "$b$, $c$ and light flavor",
     (("X_bb", "X_cc"), ("X_cc", "X_qq"), ("X_YY_bbqq", "X_YY_ccqq"), ("X_YY_bbe", "X_YY_qqe"))),
    (("prongs:", "number, type"), "prong number and type",
     (("X_qq", "X_YY_qqqq"), ("X_qq", "X_ee"), ("X_YY_qqq", "X_YY_qqqq"), ("X_tauhtauh", "X_tauhtaue"))),
)
DECAY = {"bb": r"b\bar b", "bc": "bc", "bq": "bq"}     # the two-prong decays Table 1 names beside X->bb
KEEPS, MERGES = r"\checkmark", "--"


def class_maps(root: pathlib.Path) -> dict:
    """{label set: {JetClass-II class: its class in the label set}}, the four nested sets and the five random
    groupings; class names without their "label_" prefix."""
    out: dict = {}
    for f, cols in ((MAPS, NESTED), (RAND_MAPS, None)):
        rows = list(csv.DictReader((root / f).open()))
        for c in cols or [c for c in rows[0] if re.fullmatch(r"RAND2_p\d", c)]:
            out[c] = {r["class_name"].removeprefix("label_"): r[c] for r in rows}
    return out


def mark(m: dict, pairs) -> bool | None:
    """True: the map keeps every pair apart; False: it merges every pair; None: mixed."""
    kept = {m[a] != m[b] for a, b in pairs}
    return kept.pop() if len(kept) == 1 else None


def marks(maps: dict, arm: str) -> list:
    """A nested label set's mark for each distinction; the build stops on a mixed one."""
    out = [mark(maps[arm], pairs) for _, _, pairs in DISTINCTIONS]
    if None in out:
        raise SystemExit(f"FATAL: {arm} keeps some and merges other class pairs of "
                         f"{[d[1] for d, x in zip(DISTINCTIONS, out) if x is None]}; Table 1 has no mark for that")
    return out


def counts(m: dict) -> tuple:
    """(classes that hold resonant decays, classes that hold QCD jets) of a map."""
    qcd = {v for k, v in m.items() if k.startswith("QCD")}
    res = {v for k, v in m.items() if not k.startswith("QCD")}
    if qcd & res:
        raise SystemExit("FATAL: a class holds resonant and QCD jets")
    return len(res), len(qcd)


def bb_class(m: dict) -> str:
    """The class of a map that contains X->bb, in plain words (PI 2026-11-24, GO-6)."""
    with_bb = [k.removeprefix("X_") for k, v in m.items() if v == m["X_bb"]]
    if with_bb == ["bb"]:
        return r"$X\to b\bar b$ alone"
    if all(k in DECAY for k in with_bb):
        d = [DECAY[k] for k in with_bb]
        return "decays to $" + "$, $".join(d[:-1]) + "$ or $" + d[-1] + "$"
    if sorted(with_bb) == sorted(k.removeprefix("X_") for k in m if re.fullmatch(r"X_[bcsqg]{2}", k)):
        return "every decay to two quarks or gluons"
    raise SystemExit(f"FATAL: no name for the class of X->bb: {with_bb}")


def merged_words(maps: dict, arm: str) -> str:
    """Version B's phrase: the distinctions the label set merges and the next finer one keeps."""
    i = NESTED.index(arm)
    if i == 0:
        return "nothing (Sophon's classes)"
    now, before = marks(maps, arm), marks(maps, NESTED[i - 1])
    new = [d[1] for d, a, b in zip(DISTINCTIONS, now, before) if b and not a]
    if any(a and not b for a, b in zip(now, before)):
        raise SystemExit(f"FATAL: {arm} keeps a distinction {NESTED[i - 1]} merges; the label sets are not nested")
    return ("" if i == 1 else "also ") + "; ".join(new)


def name(maps: dict, arm: str) -> str:
    """A row's name (STYLE section 3), the class count read from the map."""
    for mass, base in MASS:
        if arm == mass:
            return f"{len(set(maps[base].values()))} classes + mass regression"
    if arm == RAND:
        n = {len(set(maps[c].values())) for c in maps if c.startswith("RAND2_p")}
        if len(n) != 1:
            raise SystemExit("FATAL: the random groupings differ in their number of classes")
        return f"{n.pop()} classes, random grouping"
    return f"{len(set(maps[arm].values()))} classes"


TABLE1_CAPTION = ("The label sets, what each merges from the finer set above it, and the number of pretraining runs; every "
                  "set keeps the number of decay products and which of four kinds each is: a quark or gluon, an electron "
                  "or muon, a tau lepton, or a neutrino.")


def short(maps: dict, arm: str) -> str:
    """A row's name in Tables 2 and 3 (PI 2026-11-24, GO-6: shorter row labels so that six columns fit); Table 1
    gives the full names."""
    return name(maps, arm).replace(", random grouping", ", random").replace(" + mass regression", " + mass")


def table1(maps: dict, runs: dict) -> str:
    """Table 1 (PI 2026-11-24, GO-6: version B with its fixes). Below the rule the random groupings and the mass
    regression."""
    rand = sorted(c for c in maps if c.startswith("RAND2_p"))
    n_runs = {a: str(len(runs[a])) for a in NESTED + tuple(m for m, _ in MASS)}
    if {len(runs[c]) for c in rand} != {1}:
        raise SystemExit("FATAL: a random grouping without exactly one run")
    n_runs[RAND] = f"{len(rand)} groupings, 1 each"
    cls = lambda m: "{} + {}".format(*counts(m))                    # noqa: E731
    rand_counts = {counts(maps[c]) for c in rand}
    if len(rand_counts) != 1:
        raise SystemExit("FATAL: the random groupings differ in their resonant and QCD classes")
    # the label column wraps too: with it unwrapped the table was 17% wider than the text block and was set at 7.7 pt
    b = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}l>{\raggedright\arraybackslash}p{4.3cm}"
         r">{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{1.7cm}@{}}",
         r"\toprule", r"Label set & Classes & Merges & Class containing & Runs \\",
         r" & (resonant + QCD) & & $X\to b\bar b$ & \\", r"\midrule"]
    for arm in NESTED:
        m = maps[arm]
        b.append(f"{name(maps, arm)} & {cls(m)} & {merged_words(maps, arm)} & {bb_class(m)} & {n_runs[arm]}" + r" \\")
    b.append(r"\midrule")
    res, qcd = rand_counts.pop()
    n_res = len([k for k in maps["L188"] if not k.startswith("QCD")])
    b.append(f"{name(maps, RAND)} & {res} + {qcd} & the {n_res} resonant classes into {res} random groups, "
             f"the QCD classes into one & differs between groupings & {n_runs[RAND]}" + r" \\")
    for mass, base in MASS:
        m = maps[base]
        b.append(f"{name(maps, mass)} & {cls(m)} & as {name(maps, base)} & {bb_class(m)} & {n_runs[mass]}" + r" \\")
    return _float(TABLE1_CAPTION, "tab:sets", b + [r"\bottomrule", r"\end{tabular}"])


def _float(caption: str, label: str, tabular: list, numbered: bool = True) -> str:
    """A portrait table float on one page: the caption above, the tabular at 9 pt, shrunk only if wider than the text."""
    cap = rf"\caption{{{caption}}}\label{{{label}}}" if numbered else rf"\caption*{{{caption}}}\label{{{label}}}"
    return "\n".join([r"\begin{table}[htbp]", r"\centering", cap, r"\fontsize{9}{11}\selectfont\setlength{\tabcolsep}{3pt}",
                      r"\begin{adjustbox}{max width=\linewidth}", *tabular, r"\end{adjustbox}", r"\end{table}", ""])


PM = re.compile(r"^(?P<m>[\d{},.]+)\\,\$\\pm\$\\,(?P<s>[\d{},.]+)$")


def paren(pm: str) -> str:
    """make_tables.fmt_pm's "mean +- sd" in the concise form (STYLE section 2): 0.4348 +- 0.0004 -> 0.4348(4),
    72.8 +- 2.4 -> 72.8(2.4), 1,800 +- 60 -> 1,800(60). Anything else (a lone value, a bound) is returned as it is."""
    m = PM.match(pm)
    if not m:
        return pm
    s = m["s"].replace("{,}", "")
    if "." in s and float(s) < 1:
        s = str(int(s.replace(".", "")))
    return f"{m['m']}({s})"


def _place(mean: float, sd: float) -> int | None:
    """The decimals make_tables.fmt_pm prints a row at (PDG rounding of its SD); None for a lone or bounded value."""
    m = PM.match(MT.fmt_pm(mean, sd))
    return None if not m else len(m["m"].split(".")[1]) if "." in m["m"] else 0


def value_table(columns, rows, empty: str) -> tuple:
    """(the tabular's body rows, the cells with no value) of a model-per-row table. columns: (name, f(arm) -> its run
    values); rows: (name, arm, runs expected) or None for a rule. PI 2026-11-24 (GO-6): mean +- SD, one number of
    decimals per column (the most common PDG place of its rows, the larger on a tie), the SD to the same decimals; no
    bold. A cell is printed only when every one of its runs has a value. PI 2026-11-25 (LEDGER 32.3): a column whose
    name contains "accuracy" is printed in percent, the values scaled before the decimals are chosen."""
    stats, missing = {}, []
    for r in rows:
        if r is None:
            continue
        for c, (cname, get) in enumerate(columns):
            v = [float(x) * (100.0 if "accuracy" in cname.lower() else 1.0) for x in get(r[1]) if x is not None]
            if len(v) < r[2] or not v:
                missing.append((r[0], cname))
                continue
            stats[r[1], c] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None)
    places = {}
    for c in range(len(columns)):
        got = [_place(m, s) for (a, k), (m, s) in stats.items() if k == c and s]
        got = [p for p in got if p is not None]
        places[c] = max(set(got), key=lambda p: (got.count(p), p)) if got else 4
    fx = lambda x, c: MT._fixed(x, places[c])  # noqa: E731
    # PI 2026-11-25 (LEDGER §31 3): a spread that rounds to zero at the column's decimals is printed as a bound,
    # "< 0.0001", never "0.0000" (the runs behind such cells are distinct: experiments/EVAL/dup_check_v2.py)
    sd = lambda s, c: (f"$<${MT._fixed(10.0 ** -places[c], places[c])}"  # noqa: E731
                       if round(s, places[c]) == 0 else fx(s, c))
    cells = {k: fx(m, k[1]) if s is None else f"{fx(m, k[1])}\\,$\\pm$\\,{sd(s, k[1])}" for k, (m, s) in stats.items()}
    body = [r"\midrule" if r is None else
            r[0] + " & " + " & ".join(cells.get((r[1], c)) or empty for c in range(len(columns))) + r" \\" for r in rows]
    return body, missing


def capture(root: pathlib.Path) -> dict:
    """One run of the paper's build, keeping the loaded mlst_results ("R"), the arguments of its first table_summary
    call ("summary": fits, fill, chosen, per, runs, ...) and table_tests' rows_out ("tested")."""
    got: dict = {}
    load = MT.data_release_module

    def release_module():
        rel = load()
        install = rel.install_mlst

        def install_mlst(mlst, g):
            # PI 2026-11-25 (LEDGER 32.3): one number of decimals per column in every table; every token at its
            # column's place (uniform_columns), not at its own SD's place. Only this run of the build, for v2.
            mlst._printed_place = lambda kind, v, place: place
            install(mlst, g)
            got["R"] = mlst
            summary, tests = mlst.table_summary, mlst.table_tests

            def table_summary(*a, **k):
                got.setdefault("summary", a)
                got.setdefault("summary_kw", k)
                return summary(*a, **k)

            def table_tests(*a, **k):
                got["tested"] = k["rows_out"]
                return tests(*a, **k)
            mlst.table_summary, mlst.table_tests = table_summary, table_tests
        rel.install_mlst = install_mlst
        return rel
    MT.data_release_module = release_module
    try:
        got["built"] = MT.build(root)[0]
    finally:
        MT.data_release_module = load
    if {"R", "summary", "tested"} - set(got):
        raise SystemExit(f"FATAL: the build did not reach {sorted({'R', 'summary', 'tested'} - set(got))}")
    return got


# PI 2026-11-24 (GO-6) and 2026-11-25 (section 28, 1.3): the same six columns in Tables 2 and 3
HEAD = [r" & JetClass-II & JetClass & \multicolumn{2}{c}{Top tagging} & \multicolumn{2}{c}{Quark/gluon} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & Accuracy (\%) & Accuracy (\%) & Accuracy (\%) & Rej$_{30\%}$ & Accuracy (\%) & Rej$_{30\%}$ \\", r"\midrule"]
LP_HEAD = FT_HEAD = HEAD


def tables23(got: dict, maps: dict) -> tuple:
    """(Table 2, Table 3, the fine-tuned cells that are pending as [(row, column)], Table 2's columns, Table 3's)."""
    R = got["R"]
    fits, fill, chosen, per, runs = got["summary"][:5]
    runs = R._pooled(runs)
    pre = [(short(maps, a), a, len(runs[a])) for a in NESTED + (RAND,) + tuple(m for m, _ in MASS)]

    def probe(ds, met):
        g = R.summary_probe(fits, fill, chosen, ds, met)
        # the released checkpoint on top and quark/gluon: only with the substitution probes ours use (mlst_results)
        return lambda a: [] if g is None or (a == R.SOPHON and ds in R.SUMMARY_NO_SOPHON and R.SOPHON_MODEL not in fill) \
            else [g(m) for m in runs[a]]

    lp_cols = [("JetClass-II accuracy", probe("jc2_188", "accuracy")), ("JetClass accuracy", probe("jc1", "accuracy")),
               ("top accuracy", probe("top", "accuracy")), ("top Rej30%", probe("top", "r30")),
               ("quark/gluon accuracy", probe("qg", "accuracy")), ("quark/gluon Rej30%", probe("qg", "r30"))]
    lp_rows = pre + [None, ("untrained network", R.INIT_BN, len(runs[R.INIT_BN])), ("Sophon (released)", R.SOPHON, 1)]
    body, miss = value_table(lp_cols, lp_rows, "--")
    if miss:
        raise SystemExit(f"FATAL: Table 2 has cells without a linear probe: {miss}")
    t2 = _float(r"Linear probes on the frozen features: accuracy and background rejection at a signal efficiency of "
                r"30\% (Rej$_{30\%}$); mean $\pm$ standard deviation over runs.", "tab:probes",
                [r"\begin{tabular}{@{}l" + "c" * len(lp_cols) + "@{}}", r"\toprule", *LP_HEAD, *body, r"\bottomrule",
                 r"\end{tabular}"])

    scratch = "trained from scratch"

    def ft(key, met):
        n = {k: n for k, n, *_ in R.SUMMARY_PANELS}[key]
        g = R.summary_ft(per, key, n, met)
        return lambda a: R.summary_scratch(per, key, n, met) if a == scratch else [] if g is None else [g(m) for m in runs[a]]

    ft_cols = [("JetClass-II accuracy", ft("jc2", "accuracy")), ("JetClass accuracy", ft("jc1", "accuracy")),
               ("top accuracy", ft("top", "accuracy")), ("top Rej30%", ft("top", "r30")),
               ("quark/gluon accuracy", ft("qg", "accuracy")), ("quark/gluon Rej30%", ft("qg", "r30"))]
    ft_rows = pre + [None, (scratch, scratch, len(R.FT_SCRATCH_MODELS))]
    body, pending = value_table(ft_cols, ft_rows, PENDING)
    t3 = _float(r"Fine-tuned models: accuracy and background rejection at a signal efficiency of 30\% (Rej$_{30\%}$); "
                r"mean $\pm$ standard deviation over runs.", "tab:finetuned",
                [r"\begin{tabular}{@{}l" + "c" * len(ft_cols) + "@{}}", r"\toprule", *FT_HEAD, *body, r"\bottomrule",
                 r"\end{tabular}"])
    return t2, t3, pending, lp_cols, ft_cols


# ------------------------------------------------------------------ the tested rows the text rests on
# PI 2026-11-25 (LEDGER sections 28-31): the abstract and introduction of paper/journal/v2/pi_prose_1125.tex, and the
# Methods, Results and Discussion written to them. Each clause names the tested differences it rests on: (clause,
# heading of the build's table of tests, first network, second network, what must hold: "below" / "above" (first minus
# second beyond 3 sigma) or "null" (within 3 sigma); "report": printed, never a stop). A missing or failing row stops
# the build; it is reported to the PI, never worded around.
BINARY = (("top tagging", "top"), ("quark/gluon tagging (Pythia)", "qg"), ("quark/gluon tagging (Herwig)", "qg_herwig"))
METRICS = (("accuracy", "accuracy"), ("AUC", "auc"), ("$R_{50}$", "r50"), ("$R_{30}$", "r30"))
AUC_PASS = ("quark/gluon tagging (Pythia)", "AUC", "ft", "R16_Q1", "L162")     # the one pass (LEDGER 30.1)


class Rows:
    """The headings of the build's table of tests, and its rows."""

    def __init__(self, R, tested: dict):
        self.R, self.tested = R, tested
        lp, ft = R.HEADING["lp"], R.HEADING["ft"]
        self.acc, self.auc = f"JetClass-II, accuracy, {lp}", f"JetClass-II, AUC, {lp}"
        self.auc_qcd = f"JetClass-II, AUC over the QCD classes, {lp}"
        self.auc_res = f"JetClass-II, AUC over the resonant classes, {lp}"
        self.jc1_lp = f"JetClass, accuracy, {lp}"
        self.jc1, self.jc2 = f"JetClass, accuracy, {ft}", f"JetClass-II, accuracy, {ft}"
        six = R._power("1e+06") + " training jets, mass regression and from scratch"
        self.jc1_six, self.jc2_six = f"{self.jc1}, {six}", f"{self.jc2}, {six}"
        self.mass = "jet mass, resolution, multilayer-perceptron probe on the frozen features"
        self.mlp = "JetClass-II, AUC, multilayer-perceptron probe on the frozen features"
        self.xbx = {sig: rf"{h}, signal efficiency (\%), " + R.XBX_HEAD for sig, h in R.SUMMARY_XBX}

    def pair(self, k):
        return f"{self.R._PAIR[k]}, AUC, {self.R.HEADING['lp']}"

    def full(self, task, metric, mode):
        return rf"{task}, {metric}, {self.R.HEADING[mode]}, full training set, label sets and random grouping"

    def sizes(self):
        """The headings of fine-tuned JetClass-II below the largest training set."""
        return sorted(h for h, a, b in self.tested if h.startswith(self.jc2 + ", ") and "training jets" in h
                      and "mass regression" not in h and (a, b) == ("R16_Q1", "L188"))

    def get(self, head, a, b):
        v = self.tested.get((head, a, b))
        if not isinstance(v, tuple):
            raise SystemExit(f"FATAL: no tested difference {a} - {b} under '{head}' ({v})")
        return v

    def z(self, head, a, b):
        v = self.get(head, a, b)
        return v[0] / v[4]


def claims(R, tested: dict) -> tuple:
    r = Rows(R, tested)
    pairs = list(__import__("itertools").combinations(NESTED, 2))
    out = []
    add = lambda clause, *rows: out.extend((clause, *x) for x in rows)       # noqa: E731
    add("frozen: the 188 classes are identified less accurately with coarser labels",
        (r.acc, "R16_Q1", "L188", "below"), (r.acc, "R42_Q1", "L188", "below"), (r.acc, "L162", "L188", "null"),
        (r.auc, "L162", "L188", "below"), (r.auc_qcd, "L162", "L188", "below"), (r.auc_res, "L162", "L188", "null"))
    add("random groupings still reach 42%: the loss depends on which classes are merged",
        (r.acc, RAND, "R16_Q1", "above"), (r.auc, RAND, "R16_Q1", "above"), (r.acc, RAND, "L188", "below"),
        (r.acc, RAND, "R42_Q1", "above"))
    add("pairs of classes that a label set merges are separated less well",
        (r.pair("bb_vs_cc"), "R16_Q1", "L188", "below"), (r.pair("bbqq_vs_ccqq"), "R16_Q1", "L188", "below"),
        (r.pair("bc_vs_bq_cs"), "R42_Q1", "L188", "below"))
    add("at 43 classes some pairs the labels keep apart lose separation as well",
        (r.pair("bb_vs_cc"), "R42_Q1", "L188", "below"), (r.pair("bbqq_vs_ccqq"), "R42_Q1", "L188", "below"))
    add("with the 17-class model the resonance benchmark loses signal efficiency for both decays",
        (r.xbx["xbb"], "R16_Q1", "L188", "below"), (r.xbx["xbs"], "R16_Q1", "L188", "below"))
    add("43 classes on the resonance benchmark: X->bs lower, X->bb not significantly different",
        (r.xbx["xbs"], "R42_Q1", "L188", "below"), (r.xbx["xbb"], "R42_Q1", "L188", "null"))
    add("transfer learning on the frozen features against the same classifier trained from scratch, X->bs",
        (r.xbx["xbs"], "L188", "SCRATCH", "above"))
    add("fine-tuning narrows the gap without closing it; the 43-class model is also behind, the 162-class model not",
        (r.jc2, "R16_Q1", "L188", "below"), (r.jc1, "R16_Q1", "L188", "below"), (r.jc2, "R42_Q1", "L188", "below"),
        (r.jc1, "R42_Q1", "L188", "below"), (r.jc2, "L162", "L188", "null"), (r.jc1, "L162", "L188", "null"),
        *[(h, "R16_Q1", "L188", "below") for h in r.sizes()])
    add("gains from pretraining over training from scratch on JetClass-II and JetClass",
        (r.jc2_six, "L188", "SCRATCH", "above"), (r.jc1_six, "L188", "SCRATCH", "above"))
    # LEDGER 30.1: no difference among the four label sets in accuracy or background rejection at the full sets,
    # frozen or fine-tuned; in AUC exactly one passes, which Results and Discussion state
    for task, _ in BINARY:
        for mode in ("lp", "ft"):
            for metric, _ in METRICS:
                for b, a in pairs:
                    one = (task, metric, mode, a, b) == AUC_PASS
                    want = "below" if one else "report" if metric == "AUC" else "null"
                    add("the one AUC difference among the label sets" if one else
                        "binary benchmarks: no significant difference in accuracy or background rejection",
                        (r.full(task, metric, mode), a, b, want))
    add("mass regression sharpens the mass resolution of the frozen representation",
        (r.mass, "L162_MASS", "L162", "below"), (r.mass, "R16_Q1_MASS", "R16_Q1", "below"),
        (r.mass, "R16_Q1", "L188", "below"),
        ("jet mass, resolution, mass output of the pretraining head", "L162_MASS_HEAD", "SDMASS", "below"))
    add("mass regression: no significant cost in the accuracy of the frozen representation",
        *[(h, m, b, "null") for h in (r.acc, r.auc, r.mlp) for m, b in MASS])
    add("after fine-tuning: about equal cost at both label sets, significant only at 17 classes",
        (r.jc2_six, "R16_Q1_MASS", "R16_Q1", "below"), (r.jc1_six, "R16_Q1_MASS", "R16_Q1", "below"),
        (r.jc2_six, "L162_MASS", "L162", "null"), (r.jc1_six, "L162_MASS", "L162", "null"))
    add("random groupings offer no advantage over the 17-class set on frozen JetClass and are below 188 classes",
        (r.jc1_lp, RAND, "R16_Q1", "null"), (r.jc1_lp, RAND, "L188", "below"))
    lpset = lambda m: f"JetClass, {m}, {R.HEADING['lp']}, label sets"                 # noqa: E731
    add("frozen JetClass: 17 - 188 not significant in accuracy, significant in AUC; 43 and 162 nulls",
        (lpset("accuracy"), "R16_Q1", "L188", "null"), (lpset("AUC"), "R16_Q1", "L188", "below"),
        *[(lpset(m), a, "L188", "null") for m in ("accuracy", "AUC") for a in ("R42_Q1", "L162")])
    add("on top tagging the random groupings' frozen features are significantly below those of the 17-class set",
        *[(r.full("top tagging", m, "lp"), RAND, "R16_Q1", "below") for m, _ in METRICS])
    add("quark/gluon, frozen: random grouping not significantly different from 17 classes",
        *[(r.full("quark/gluon tagging (Pythia)", m, "lp"), RAND, "R16_Q1", "null") for m, _ in METRICS])
    add("fine-tuned on JetClass the random grouping is above the 17-class model", (r.jc1, RAND, "R16_Q1", "above"))
    add("appendix E: kinematic-only pretraining reaches the same fine-tuned top Rej30% as zero-filled inputs",
        (f"top tagging, $R_{{30}}$, {R.HEADING['ft']}", "KIN", "L188", "null"))
    add("pretrained against from scratch: accuracy and AUC on top tagging and quark/gluon (Pythia)",
        *[(r.full(t, m, "ft"), "L188", "SCRATCH", "above") for t in ("top tagging", "quark/gluon tagging (Pythia)")
          for m in ("accuracy", "AUC")])
    add("pretrained against from scratch: Rej30% passes on neither benchmark, nothing on Herwig",
        *[(r.full(t, "$R_{30}$", "ft"), "L188", "SCRATCH", "null") for t in ("top tagging", "quark/gluon tagging (Pythia)")],
        *[(r.full("quark/gluon tagging (Herwig)", m, "ft"), "L188", "SCRATCH", "null") for m, _ in METRICS])
    add("appendix: the flavor-boundary groupings between the 17-class model and the random groupings",
        (r.acc, "FLAV_F0", "R16_Q1", "above"), (r.acc, "FLAV_F1", "FLAV_F0", "above"), (r.acc, "FLAV_F1R", "FLAV_F0", "above"),
        (r.acc, RAND, "FLAV_F1", "above"), (r.acc, RAND, "FLAV_F1R", "above"))
    mlp_tail = "multilayer-perceptron probe on the frozen features"
    groups = "JetClass-II, accuracy on the 17 merged groups, "
    top_mlp = rf"top tagging, $R_{{30}}$, {mlp_tail}, missing inputs as for the linear probe"
    add("appendix: where the MLP probe changes a conclusion",
        (groups + mlp_tail, "R16_Q1_17", "L188_17", "below"),
        *[(groups + R.HEADING["lp"], a, "L188_17", "null") for a in ("L162_17", "R42_Q1_17", "R16_Q1_17")],
        (top_mlp, "R16_Q1_MASS", "L188", "above"),
        (rf"top tagging, $R_{{30}}$, {R.HEADING['lp']}", "R16_Q1_MASS", "L188", "null"))
    return tuple(out)


def check_claims(R, tested: dict) -> tuple:
    """(one line per tested difference, the lines whose test is missing or fails, AUC passes among the label sets)."""
    lines, bad, auc_pass = [], [], []
    for clause, head, a, b, want in claims(R, tested):
        v = tested.get((head, a, b))
        if not isinstance(v, tuple):
            verdict, nums = "MISSING" if v is None else "PENDING (rerun bootstrap)", ""
        else:
            z = v[0] / v[4]
            ok = z < -SIGMA if want == "below" else z > SIGMA if want == "above" else abs(z) < SIGMA
            verdict = "passes" if ok else "FAILS"
            if want == "report":
                verdict = "significant" if abs(z) > SIGMA else "not significant"
                if abs(z) > SIGMA:
                    auc_pass.append(f"{head} | {a} - {b} | {z:+.2f}")
            nums = f"delta {v[0]:.5g}  sigma {v[4]:.5g}  delta/sigma {z:+.2f}  "
        line = f"{clause} | {head} | {a} - {b} | wants {want} | {nums}{verdict}"
        lines.append(line)
        if verdict != "passes" and (want != "report" or verdict.startswith(("MISSING", "PENDING"))):
            bad.append(line)
    # the text says one AUC difference passes: a second one is a failed sentence
    bad += [f"a second AUC difference among the label sets passes | {x}" for x in auc_pass]
    return lines, bad


# ------------------------------------------------------------------ the numbers of the text
def numbers(got: dict, lp_cols, ft_cols, maps: dict, runs: dict) -> str:
    """Every number the v2 text prints that is not a macro of the build's results_generated.tex (which the draft also
    inputs): class counts, accuracies in percent, differences in percentage points, |Delta/sigma| of the central
    results, the output layer's accuracies, the resonance benchmark's efficiencies and the class-pair AUCs."""
    R, tested = got["R"], got["tested"]
    r = Rows(R, tested)
    lp, ft = dict(lp_cols), dict(ft_cols)
    got["old"] = dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(?:%.*)?$", got["built"]["results_generated.tex"], re.M))
    _fits, _fill, _chosen, _per = got["summary"][:4]
    _g = R.summary_ft(_per, "qg", {k: n for k, n, *_ in R.SUMMARY_PANELS}["qg"], "auc")
    got["ft_auc_qg"] = lambda arm: [_g(m) for m in runs[arm]]
    data = REPO / "experiments/FIGS/data/v2"

    def vals(cols, col, arm):
        v = [float(x) for x in cols[col](arm) if x is not None]
        n = len(runs[arm]) if arm in runs else len(R.FT_SCRATCH_MODELS)
        if len(v) != n:
            raise SystemExit(f"FATAL: {col}, {arm}: {len(v)} of {n} runs")
        return v
    mean = lambda cols, col, arm: float(np.mean(vals(cols, col, arm)))                     # noqa: E731
    pct = lambda cols, col, arm, nd=1: f"{100 * mean(cols, col, arm):.{nd}f}"              # noqa: E731
    pts = lambda head, a, b, nd=1: f"{100 * abs(r.get(head, a, b)[0]):.{nd}f}"             # noqa: E731
    zz = lambda head, a, b: f"{abs(r.z(head, a, b)):.1f}"                                  # noqa: E731
    one = lambda xs: MT.words(xs.pop()) if len(xs) == 1 else None                          # noqa: E731
    S = "trained from scratch"
    J2, J1, TA, TR, QA, QR = ("JetClass-II accuracy", "JetClass accuracy", "top accuracy", "top Rej30%",
                              "quark/gluon accuracy", "quark/gluon Rej30%")
    arms = {"Full": "L188", "Res": "L162", "Mid": "R42_Q1", "Coarse": "R16_Q1", "Rand": RAND,
            "ResMass": "L162_MASS", "CoarseMass": "R16_Q1_MASS"}
    out = {"Kfull": len(set(maps["L188"].values())), "Kres": len(set(maps["L162"].values())),
           "Kmid": len(set(maps["R42_Q1"].values())), "Kcoarse": len(set(maps["R16_Q1"].values())),
           "VRunsWord": one({len(runs[a]) for a in NESTED}), "VMassRunsWord": one({len(runs[m]) for m, _ in MASS}),
           "VRandWord": MT.words(len([c for c in maps if c.startswith("RAND2_p")])),
           "VScratchRunsWord": MT.words(len(R.FT_SCRATCH_MODELS)),
           # the abstract's whole percents
           "VAccFullPct": pct(lp, J2, "L188", 0), "VAccCoarsePct": pct(lp, J2, "R16_Q1", 0),
           "VAccRandPct": pct(lp, J2, RAND, 0), "VDFtJcTwoCoarseRound": pts(r.jc2, "R16_Q1", "L188", 0)}
    if None in out.values():
        raise SystemExit(f"FATAL: label sets with different run counts: {out}")
    for w, a in arms.items():
        out[f"VLpJcTwo{w}"], out[f"VLpJcOne{w}"] = pct(lp, J2, a), pct(lp, J1, a)
        out[f"VFtJcTwo{w}"], out[f"VFtJcOne{w}"] = pct(ft, J2, a), pct(ft, J1, a)
        out[f"VLpTopRej{w}"] = MT.fmt_int(mean(lp, TR, a))
    out.update({"VLpJcTwoUntrained": pct(lp, J2, R.INIT_BN), "VLpJcTwoSophon": pct(lp, J2, R.SOPHON),
                "VFtJcTwoScratch": pct(ft, J2, S), "VFtJcOneScratch": pct(ft, J1, S),
                "VFtTopAccFull": pct(ft, TA, "L188", 2), "VFtTopAccScratch": pct(ft, TA, S, 2),
                "VFtQgAccFull": pct(ft, QA, "L188", 2), "VFtQgAccScratch": pct(ft, QA, S, 2)})
    lo, hi = sorted(vals(lp, J2, "R16_Q1_MASS"))[0], sorted(vals(lp, J2, "R16_Q1_MASS"))[-1]
    out.update({"VLpJcTwoCoarseMassLo": f"{100 * lo:.1f}", "VLpJcTwoCoarseMassHi": f"{100 * hi:.1f}"})
    # differences in percentage points, from the tested rows
    out.update({"VDLpJcTwoCoarse": pts(r.acc, "R16_Q1", "L188"), "VDLpJcTwoMid": pts(r.acc, "R42_Q1", "L188"),
                "VDLpJcTwoRes": pts(r.acc, "L162", "L188"), "VDLpJcTwoRand": pts(r.acc, RAND, "L188"), "VDLpJcOneRand": pts(r.jc1_lp, RAND, "L188"),
                "VDFtJcTwoCoarse": pts(r.jc2, "R16_Q1", "L188"), "VDFtJcOneCoarse": pts(r.jc1, "R16_Q1", "L188"),
                "VDFtJcTwoMid": pts(r.jc2, "R42_Q1", "L188"), "VDFtJcOneMid": pts(r.jc1, "R42_Q1", "L188"),
                "VDFtJcTwoGain": pts(r.jc2_six, "L188", "SCRATCH"), "VDFtJcOneGain": pts(r.jc1_six, "L188", "SCRATCH"),
                "VDFtJcOneMassRes": pts(r.jc1_six, "L162_MASS", "L162", 2),
                "VDFtJcOneMassCoarse": pts(r.jc1_six, "R16_Q1_MASS", "R16_Q1", 2),
                "VDFtJcTwoMassRes": pts(r.jc2_six, "L162_MASS", "L162", 2),
                "VDFtJcTwoMassCoarse": pts(r.jc2_six, "R16_Q1_MASS", "R16_Q1", 2),
                "VDFtJcOneRand": pts(r.jc1, RAND, "R16_Q1"),
                "VDFtTopAccGain": pts(r.full("top tagging", "accuracy", "ft"), "L188", "SCRATCH", 2),
                "VDFtQgAccGain": pts(r.full("quark/gluon tagging (Pythia)", "accuracy", "ft"), "L188", "SCRATCH", 2)})
    sizes = r.sizes()
    out.update({"VDFtJcTwoCoarseSmallest": pts(sizes[0], "R16_Q1", "L188"),
                "VDFtJcTwoCoarseSmaller": pts(sizes[-1], "R16_Q1", "L188"), "VFtSizesWord": MT.words(len(sizes))})
    # |Delta / sigma| of the central results
    out.update({"VZLpJcTwoCoarse": zz(r.acc, "R16_Q1", "L188"), "VZLpJcTwoRand": zz(r.acc, RAND, "R16_Q1"),
                "VZFtJcTwoCoarse": zz(r.jc2, "R16_Q1", "L188"), "VZFtJcOneCoarse": zz(r.jc1, "R16_Q1", "L188"),
                "VZFtJcTwoMassCoarse": zz(r.jc2_six, "R16_Q1_MASS", "R16_Q1"),
                "VZQgAuc": zz(r.full(AUC_PASS[0], AUC_PASS[1], AUC_PASS[2]), AUC_PASS[3], AUC_PASS[4]),
                "VZMassRes": zz(r.mass, "L162_MASS", "L162"), "VZMassCoarse": zz(r.mass, "R16_Q1_MASS", "R16_Q1"),
                "VZXbsCoarse": zz(r.xbx["xbs"], "R16_Q1", "L188"), "VZXbbCoarse": zz(r.xbx["xbb"], "R16_Q1", "L188")})
    # cold readers (draft e6e3b22): the step from 43 to 17 classes; the full uncertainty (run spreads and test-sample
    # term) of a difference between two label sets in fine-tuned Rej30%; 188 classes against from scratch in Rej30%
    out["VDLpJcTwoStep"] = pts(r.acc, "R16_Q1", "R42_Q1")
    out["VDLpJcTwoMidStep"] = pts(r.acc, "R42_Q1", "L162")
    # the fine-tuned and linear-probe JetClass-II accuracies of the two finest label sets, to two decimals (LEDGER 33.1)
    for w, arm in (("Full", "L188"), ("Res", "L162")):
        out[f"VFtJcTwo{w}Two"] = pct(ft, J2, arm, 2)
        out[f"VLpJcTwo{w}Two"] = pct(lp, J2, arm, 2)
    for w, task, nd in (("Top", "top tagging", 0), ("Qg", "quark/gluon tagging (Pythia)", 1)):
        h = r.full(task, "$R_{30}$", "ft")
        sig = [r.get(h, a, b)[4] for b, a in __import__("itertools").combinations(NESTED, 2)]
        out[f"VFt{w}RejSigLo"], out[f"VFt{w}RejSigHi"] = f"{min(sig):.{nd}f}", f"{max(sig):.{nd}f}"
        out[f"VZFt{w}RejScratch"] = zz(h, "L188", "SCRATCH")
    auc5 = lambda arm: f"{float(np.mean([float(x) for x in got['ft_auc_qg'](arm)])):.5f}"      # noqa: E731
    out["VFtQgAucCoarse"], out["VFtQgAucRes"] = auc5("R16_Q1"), auc5("L162")
    n_par = int(str(got["old"]["MlstNParamsJc"]).replace("{,}", ""))
    out["VParamsMillion"] = f"{n_par / 1e6:.1f}"
    for w, m in (("Acc", "accuracy"), ("Auc", "AUC")):
        head = f"JetClass, {m}, {R.HEADING['lp']}, label sets"
        out[f"VZLpJcOneCoarse{w}"] = f"{abs(r.z(head, 'R16_Q1', 'L188')):.2f}"
    # PI 2026-11-25 (LEDGER 32.2): frozen JetClass 17 - 188 in points; the top-tagging working point: background jets
    # that pass at 30% signal efficiency (of the test set's), and the test-sample term of 188 - scratch in Rej30%
    out["VDLpJcOneCoarse"] = pts(f"JetClass, accuracy, {R.HEADING['lp']}, label sets", "R16_Q1", "L188")
    nkey = {k: n for k, n, *_ in R.SUMMARY_PANELS}["top"]
    _pass = R.summary_ft(got["summary"][3], "top", nkey, "r30_n_bkg_pass")
    _nb = R.summary_ft(got["summary"][3], "top", nkey, "n_background")
    out["VFtTopBkgPassFull"] = MT.fmt_int(np.mean([_pass(m) for m in runs["L188"]]))
    out["VFtTopBkgPassScratch"] = MT.fmt_int(np.mean(R.summary_scratch(got["summary"][3], "top", nkey, "r30_n_bkg_pass")))
    out["VFtTopNBkg"] = MT.fmt_int(one_value({_nb(m) for m in runs["L188"]}))
    out["VFtTopRejScratchBoot"] = MT.fmt_int(r.get(r.full("top tagging", "$R_{30}$", "ft"), "L188", "SCRATCH")[3])
    # Fig. 3 (a): the random groupings that merge X->bb with X->cc and those that keep them apart (LEDGER 32.4)
    _fits = got["summary"][0]
    merged, apart = [], []
    for m in runs[RAND]:
        g = "RAND2_p" + re.search(r"rand2p(\d)", m).group(1)
        (merged if maps[g]["X_bb"] == maps[g]["X_cc"] else apart).append(R._pair(_fits, m, "bb_vs_cc"))
    out["VAucBbRandMerged"] = f"{float(np.mean(merged)):.4f}"
    out["VAucBbRandApart"] = f"{float(np.mean(apart)):.4f}"
    out["VDQgAuc"] = f"{abs(r.get(r.full(AUC_PASS[0], AUC_PASS[1], AUC_PASS[2]), AUC_PASS[3], AUC_PASS[4])[0]):.4f}"
    n_pairs = len(list(__import__("itertools").combinations(NESTED, 2)))
    out["VNBinaryCmp"] = str(n_pairs * len(BINARY) * len(METRICS) * 2)
    # the run-to-run spread of the fine-tuned background rejection among the four label sets
    for w, col, nd in (("Top", TR, 0), ("Qg", QR, 1)):
        sd = [float(np.std(vals(ft, col, a), ddof=1)) for a in NESTED]
        out[f"VFt{w}RejSdLo"], out[f"VFt{w}RejSdHi"] = (f"{min(sd):.{nd}f}", f"{max(sd):.{nd}f}")
        out[f"VFt{w}RejFull"] = MT.fmt_int(mean(ft, col, "L188")) if nd == 0 else f"{mean(ft, col, 'L188'):.1f}"
        out[f"VFt{w}RejScratch"] = MT.fmt_int(mean(ft, col, S)) if nd == 0 else f"{mean(ft, col, S):.1f}"
    # the pretrained output layer (experiments/EVAL/head_accuracy.py; LEDGER 28 1.2, 31.2)
    f = sorted((data / "head_acc").glob("head_acc_bias_*.json"))
    if len(f) != 1:
        raise SystemExit(f"FATAL: {len(f)} output-layer files in {data / 'head_acc'}")
    doc = json.loads(f[0].read_text())
    ours = [doc["models"][m] for m in runs["L188"]]
    before = float(np.mean([m["accuracy"] for m in ours]))
    after = float(np.mean([m["bias_refit"]["accuracy"] for m in ours]))
    probe = mean(lp, J2, "L188")
    out.update({"VHeadAcc": f"{100 * before:.1f}", "VHeadRefit": f"{100 * after:.1f}",
                "VHeadClosed": f"{100 * (after - before) / (probe - before):.0f}",
                "VMajority": f"{100 * doc['majority']['accuracy']:.1f}", "VChance": f"{100 / out['Kfull']:.1f}",
                "VProbeNTrain": MT.fmt_int(ours[0]["bias_refit"]["n_train"]),
                "VProbePerClassMax": MT.fmt_int(ours[0]["bias_refit"]["train_class_count_max"]),
                "VProbePerClassMin": MT.fmt_int(ours[0]["bias_refit"]["train_class_count_min"])})
    # the resonance benchmark: signal efficiency in percent at the benchmark's background efficiency
    xbx = got["summary_kw"].get("xbx") or {}
    if not xbx.get("fits"):
        raise SystemExit("FATAL: no X->bb / X->bs fits in the build")
    XR, xfits, scr = xbx["XR"], xbx["fits"], xbx.get("scr") or {}
    stars = set()
    for sig, w in (("xbb", "Xbb"), ("xbs", "Xbs")):
        for a, arm in (("Full", "L188"), ("Mid", "R42_Q1"), ("Coarse", "R16_Q1")):
            recs = [XR.readout(xfits.get(m, {}).get(sig, {}), "head") for m in runs[arm]]
            if not all(recs):
                raise SystemExit(f"FATAL: {sig}, {arm}: a run without its fit")
            stars |= {x["eps_b_star"] for x in recs}
            out[f"V{w}{a}"] = f"{100 * float(np.mean([x['eps_s_at_star'] for x in recs])):.1f}"
        v = [x["eps_s_at_star"] for x in scr.get(sig, [])]
        if not v:
            raise SystemExit(f"FATAL: {sig}: no classifier trained from scratch")
        out[f"V{w}Scratch"] = f"{100 * float(np.mean(v)):.1f}"
    out["VXbxEpsB"] = f"{100 * one_value(stars):g}"
    # PI 2026-11-25 (LEDGER 32.3): the output-score discriminant of Fig. 5 (a) at the benchmark's background efficiency,
    # Sophon's direct X->bb readout, for the two label sets that keep X->bb as a class
    for a, arm in (("Full", "L188"), ("Res", "L162")):
        recs = [XR.readout(xfits.get(m, {}).get("xbb", {}), "zero_shot_xbb") for m in runs[arm]]
        if all(recs) and {x["eps_b_star"] for x in recs} <= stars:
            v = [100 * x["eps_s_at_star"] for x in recs]
            out[f"VXbbDirect{a}"] = MT.fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
        else:
            out[f"VXbbDirect{a}"] = r"\pending{no output-score value at this background efficiency in the build}"
    # class pairs: the AUC of a binary linear probe
    fits = got["summary"][0]
    for k, w in (("bb_vs_cc", "Bb"), ("bbqq_vs_ccqq", "Bbqq"), ("bc_vs_bq_cs", "Bc")):
        for a, arm in (("Full", "L188"), ("Mid", "R42_Q1"), ("Coarse", "R16_Q1")):
            v = [R._pair(fits, m, k) for m in runs[arm]]
            if None in v:
                raise SystemExit(f"FATAL: class pair {k}, {arm}: a run without its probe")
            out[f"VAuc{w}{a}"] = f"{float(np.mean(v)):.4f}"
    return "% written by experiments/FIGS/mlst_v2_tables.py\n" + "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in out.items())


def one_value(xs: set):
    if len(xs) != 1:
        raise SystemExit(f"FATAL: expected one value, found {sorted(xs)}")
    return next(iter(xs))


# ------------------------------------------------------------------ appendix tables
TOKEN = re.compile(r"tauh|taue|taum|tau|light|[bcsqgemv]")
TOKEN_TEX = {"tauh": r"\tau_h", "taue": r"\tau_e", "taum": r"\tau_\mu", "tau": r"\tau", "m": r"\mu", "v": r"\nu",
             "light": r"\text{light}"}


def class_tex(k: str) -> str:
    """A JetClass-II class in Sophon's notation: X_YY_bbtauhtaue -> $bb\tau_h\tau_e$, QCD_bbc -> $bbc$."""
    body = k.removeprefix("QCD_").removeprefix("X_YY_").removeprefix("X_")
    toks = TOKEN.findall(body)
    if "".join(toks) != body:
        raise SystemExit(f"FATAL: class name {k} not read")
    return "$" + " ".join(TOKEN_TEX.get(t, t) for t in toks).replace(" ", "") .replace("\\", " \\").strip() + "$"


def class_name(k: str) -> str:
    """A JetClass-II class with its decay structure: X_bb -> $X\to bb$, X_YY_bbcc -> $X\to YY\to bbcc$, QCD_bb -> QCD $bb$."""
    body = class_tex(k)
    if k.startswith("QCD_"):
        return "QCD light" if k == "QCD_light" else "QCD " + body
    return "$X\\to " + ("YY\\to " if k.startswith("X_YY_") else "") + body[1:]


CLASSMAP_ROWS = 60      # rows per column of Table A1 (two columns a page at 8 pt)


def classmap_rows(maps: dict) -> list:
    """Table A1's rows (PI's request, 2026-11-25): (number of the class among the 188, its name, its class number in
    the 162-, 43- and 17-class sets), ordered so the nesting reads (by 17, then 43, then 162, then 188)."""
    m188, m162, m43, m17 = (maps[a] for a in NESTED)
    order = list(m188)
    for fine, coarse, what in ((m162, m43, "162 in 43"), (m43, m17, "43 in 17"), (m188, m162, "188 in 162")):
        for c in set(fine.values()):
            if len({coarse[k] for k in order if fine[k] == c}) != 1:
                raise SystemExit(f"FATAL: the label sets are not nested ({what})")
    rows = [(int(m188[k]) + 1, class_name(k), int(m162[k]) + 1, int(m43[k]) + 1, int(m17[k]) + 1) for k in order]
    rows.sort(key=lambda r: (r[4], r[3], r[2], r[0]))
    if len({r[1] for r in rows}) != len(rows):
        raise SystemExit("FATAL: two JetClass-II classes print the same name")
    return rows


def table_classmap(maps: dict) -> str:
    """Appendix Table A1: every JetClass-II class on its own row with its class in each coarser set, two columns of
    rows a page at 8 pt; the second page continues the same float number."""
    rows = classmap_rows(maps)
    line = lambda r: " & ".join(str(x) for x in r)                                      # noqa: E731
    head = [r"\begin{tabular}[t]{@{}r>{\raggedright\arraybackslash}p{3.0cm}rrr@{}}", r"\toprule",
            r"\Kfull & Class & \Kres & \Kmid & \Kcoarse \\", r"\midrule"]
    col = lambda rs: "\n".join(head + [line(r) + r" \\" for r in rs] + [r"\bottomrule", r"\end{tabular}"])  # noqa: E731
    n = CLASSMAP_ROWS
    pages = [rows[k:k + 2 * n] for k in range(0, len(rows), 2 * n)]
    out = []
    for p, chunk in enumerate(pages):
        half = (len(chunk) + 1) // 2
        cap = (r"\caption{Every JetClass-II class with its number among the \Kfull, ordered by its class in the "
               r"\Kcoarse-, \Kmid- and \Kres-class sets, whose numbers the last three columns give. The \Kres-class "
               r"set merges the \LitJetClassTwoNQcdClasses{} QCD classes into one. Resonant classes are named by "
               r"their decay, QCD classes by their quark content.}\label{tab:classmap}" if p == 0
               else r"\ContinuedFloat\caption{(continued)}")
        out += [r"\begin{table}[p]", r"\centering", cap, r"\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{3pt}",
                col(chunk[:half]), r"\hspace{0.6cm}", col(chunk[half:]), r"\end{table}", ""]
    return "\n".join(out)


def strip_bold(t: str) -> str:
    """Old generated tabulars without their bold (PI 2026-11-24, GO-6: no bold where one model is compared under labels)."""
    key = r"{\fontseries{b}\selectfont "
    while key in t:
        i = t.index(key)
        depth, j = 1, i + len(key)
        while depth:
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        t = t[:i] + t[i + len(key):j - 1] + t[j:]
    return t


RENAME = ((r" classes + mass ", " classes + mass regression "), ("released Sophon checkpoint", "Sophon (released)"),
          ("from scratch (this work)", "trained from scratch (this work)"), ("random partitions", "random grouping"),
          (r"\textit{17-class partitions}", r"\textit{17 classes, other groupings}"), ("17-class partitions", "17 classes, other groupings"),
          (r"$^{\S}$", ""), ("multilayer-perceptron probe", "MLP probe"), ("$R_{30}$", r"Rej$_{30\%}$"), ("$R_{50}$", r"Rej$_{50\%}$"))
DROP_ROWS = ("random network", r"\quad BatchNorm recomputed")


AGREE_MARK = r"$^{\P}$"
NUM = re.compile(r"(?<![\w.$^{])(\d+\.\d+)")


def pct(cell: str) -> str:
    """Every decimal number of a cell times 100, exactly: the decimal point moves two places (0.71658 -> 71.658,
    0.00022 -> 0.022, 0.940 -> 94.0), so no digit is added, dropped or rounded."""
    from decimal import Decimal
    return NUM.sub(lambda m: format(Decimal(m.group(1)).scaleb(2), "f"), cell)


def percent_columns(tab: str) -> str:
    """A rendered tabular with each column headed "accuracy" printed in percent (PI 2026-11-25, LEDGER 32.3): the
    header gains "(%)" and every cell below it is scaled by pct."""
    lines, col = tab.split("\n"), None
    for i, ln in enumerate(lines):
        if r"\toprule" in ln:          # a new block of the table: its own header decides
            col = None
        parts = [x.strip() for x in ln.rstrip(" \\").split("&")] if "&" in ln else None
        if parts and any(x.lower() == "accuracy" for x in parts):
            col = [j for j, x in enumerate(parts) if x.lower() == "accuracy"][0]
            lines[i] = ln.replace("accuracy", r"accuracy (\%)", 1)
            continue
        if col is not None and parts and len(parts) > col and "multicolumn" not in ln:
            cells = ln.split("&")
            cells[col] = pct(cells[col])
            lines[i] = "&".join(cells)
    return "\n".join(lines)


OURS = ("(this work)", "kinematic inputs only")
CELL = re.compile(r"^\s*(?P<m>-?[\d{},]*\.?\d+)(?:\\,\$\\pm\$\\,(?P<s>[\d{},]*\.?\d+))?\s*$")


def _decimals(cell: str) -> int | None:
    m = CELL.match(cell)
    if not m:
        return None
    t = m["m"].replace("{,}", "")
    return len(t.split(".")[1]) if "." in t else 0


def published_precision(tab: str, ovo: dict | None = None) -> str:
    """PI 2026-11-25 (LEDGER 33.4): in the comparison with published taggers our rows follow the published precision
    of each column of each block (the decimals of the published rows), and on JetClass our AUC is the one-versus-one
    average of the published definition (stats/table_e1_ovo.json), at that precision."""
    from decimal import Decimal, ROUND_HALF_EVEN
    blocks, cur = [], []
    for ln in tab.split("\n"):
        if r"\toprule" in ln and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    blocks.append(cur)
    out = []
    for b in blocks:
        head = next((ln for ln in b if "&" in ln and "AUC" in ln), "")
        auc_col = [x.strip() for x in head.split("&")].index("AUC") if "AUC" in head else None
        task = None
        rows = []
        for ln in b:
            mc = re.match(r"\s*\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}", ln)
            if mc:
                task = mc.group(1)
            rows.append((task, ln))
        data = [(t, ln) for t, ln in rows if "&" in ln and ln.rstrip().endswith("\\\\") and "multicolumn" not in ln]
        places = {}
        for t, ln in data:
            if any(o in ln for o in OURS):
                continue
            for j, c in enumerate(ln.rstrip(" \\").split("&")[1:], 1):
                d = _decimals(c)
                if d is not None:
                    places[(t, j)] = max(places.get((t, j), 0), d)
        new = []
        for t, ln in rows:
            if (t, ln) in data and any(o in ln for o in OURS):
                cells = ln.rstrip().removesuffix("\\\\").split("&")
                for j in range(1, len(cells)):
                    mark = re.search(r"\$\^\{?\\[A-Za-z]+\}?\$\s*$", cells[j])
                    m = CELL.match(cells[j][:mark.start()] if mark else cells[j])
                    if not m or (t, j) not in places:
                        continue
                    q = Decimal(1).scaleb(-places[(t, j)])
                    val = Decimal(m["m"].replace("{,}", ""))
                    if ovo and t == "JetClass" and j == auc_col:
                        name = "trained from scratch" if "scratch" in cells[0] else "188 classes"
                        val = Decimal(repr(ovo[name]["auc_ovo"]))
                    mean = format(val.quantize(q, ROUND_HALF_EVEN), ",f").replace(",", "{,}")
                    txt = mean
                    if m["s"] and not (ovo and t == "JetClass" and j == auc_col):
                        sd = Decimal(m["s"].replace("{,}", "")).quantize(q, ROUND_HALF_EVEN)
                        txt += r"\,$\pm$\," + (format(sd, ",f").replace(",", "{,}") if sd else f"$<${format(q, 'f')}")
                    elif ovo and t == "JetClass" and j == auc_col:
                        sd = Decimal(repr(ovo[name]["auc_ovo_sd"])).quantize(q, ROUND_HALF_EVEN)
                        txt += r"\,$\pm$\," + (format(sd, "f") if sd else f"$<${format(q, 'f')}")
                    cells[j] = " " + txt + (mark.group(0).strip() if mark else "") + " "
                ln = "&".join(cells) + "\\\\"
            new.append(ln)
        out.append("\n".join(new))
    return "\n".join(out)


def reuse(built: dict, key: str, caption: str, label: str, size: str = r"\fontsize{9}{11}\selectfont",
          ovo: dict | None = None) -> str:
    """A table the paper's build writes (its tabulars as they are: every cell is the build's), under the rewrite's
    caption and row names, without bold and without the rows of the network the text does not use."""
    old = built.get(key)
    if old is None:
        raise SystemExit(f"FATAL: the build wrote no {key}")
    tabs = re.findall(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", old, re.S)
    if not tabs:
        raise SystemExit(f"FATAL: no tabular in {key}")
    body = []
    for t in tabs:
        t = strip_bold(t)
        if ovo is not None:
            t = published_precision(t, ovo)
        t = percent_columns(t)
        for a, b in RENAME:
            t = t.replace(a, b)
        body.append("\n".join(ln for ln in t.split("\n") if not ln.lstrip().startswith(DROP_ROWS)))
    if AGREE_MARK in "".join(body):
        caption += r" $^{\P}$ marks a run spread below the last digit shown."
    return "\n".join([r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}", size,
                      r"\setlength{\tabcolsep}{4pt}", "\n\\par\\medskip\n".join(body), r"\end{table}", ""])


def appendix_tables(got: dict, maps: dict) -> dict:
    """The appendix's value tables in Tables 2 and 3's form: the AUC columns, the flavor-boundary groupings, Herwig."""
    R = got["R"]
    fits, fill, chosen, per, runs = got["summary"][:5]
    runs = R._pooled(runs)
    S = "trained from scratch"
    pre = [(name(maps, a), a, len(runs[a])) for a in NESTED + (RAND,) + tuple(m for m, _ in MASS)]
    refs = [None, ("untrained network", R.INIT_BN, len(runs[R.INIT_BN])), ("Sophon (released)", R.SOPHON, 1)]
    scr = [None, (S, S, len(R.FT_SCRATCH_MODELS))]
    sizes = {k: n for k, n, *_ in R.SUMMARY_PANELS}

    def probe(ds, met):
        g = R.summary_probe(fits, fill, chosen, ds, met)
        return lambda a: [] if g is None or (a == R.SOPHON and ds in R.SUMMARY_NO_SOPHON and R.SOPHON_MODEL not in fill) \
            else [g(m) for m in runs[a]]

    def ft(key, met):
        g = R.summary_ft(per, key, sizes[key], met)
        return lambda a: (R.summary_scratch(per, key, sizes[key], met) if a == S else [] if g is None
                          else [g(m) for m in runs[a]])

    def tab(cols, rows, head):
        body, _ = value_table(cols, rows, "--")
        return [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
                r"\end{tabular}"]

    def two(caption, label, a, b):
        return "\n".join([r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}",
                          r"\fontsize{9}{11}\selectfont\setlength{\tabcolsep}{4pt}", r"\textit{Linear probes}\par\smallskip",
                          *a, r"\par\medskip", r"\textit{Fine-tuned}\par\smallskip", *b, r"\end{table}", ""])
    auc_head = [r" & JetClass-II & JetClass & Top tagging & Quark/gluon \\"]
    auc = two(r"AUC of the models of Tables~\ref{tab:probes} and~\ref{tab:finetuned}, averaged over classes on "
              r"JetClass-II and JetClass; mean $\pm$ standard deviation over runs.", "tab:auc",
              tab([("JetClass-II AUC", probe("jc2_188", "macro_auc_ovr")), ("JetClass AUC", probe("jc1", "macro_auc_ovr")),
                   ("top AUC", probe("top", "auc")), ("quark/gluon AUC", probe("qg", "auc"))], pre + refs, auc_head),
              tab([("JetClass-II AUC", ft("jc2", "macro_auc_ovr")), ("JetClass AUC", ft("jc1", "macro_auc_ovr")),
                   ("top AUC", ft("top", "auc")), ("quark/gluon AUC", ft("qg", "auc"))], pre + scr, auc_head))
    # short row names: with "17 classes, " before each the six columns were 82-98 pt wider than the text block
    flav = [(w, a, len(runs[a])) for a, w in
            (("FLAV_F0", "no flavor boundary"), ("FLAV_F1", "$b$-quark boundary"), ("FLAV_F1R", "random flavor boundary"))]
    keep = [("random grouping" if x[1] == RAND else x[0], *x[1:]) for x in pre if x[1] in ("R16_Q1", RAND)]
    # PI RW-4: a table too wide has too many columns. Six columns were 37-54 pt wider than the text block, so this
    # table keeps the accuracy on the two multi-class tasks and the background rejection on the two binary ones
    four = lambda f: [("JetClass-II accuracy", f("jc2_188" if f is probe else "jc2", "accuracy")),    # noqa: E731
                      ("JetClass accuracy", f("jc1", "accuracy")), ("top Rej30%", f("top", "r30")),
                      ("quark/gluon Rej30%", f("qg", "r30"))]
    four_head = [r" & JetClass-II & JetClass & Top tagging & Quark/gluon \\",
                 r" & Accuracy (\%) & Accuracy (\%) & Rej$_{30\%}$ & Rej$_{30\%}$ \\"]
    flavor = two(r"The flavor-boundary groupings, each with 17 classes, beside the 17-class model and the random "
                 r"grouping: accuracy and background rejection Rej$_{30\%}$; mean $\pm$ standard deviation over runs.",
                 "tab:flavor", tab(four(probe), keep[:1] + flav + keep[1:], four_head),
                 tab(four(ft), keep[:1] + flav + keep[1:], four_head))
    hw_head = [r" & Accuracy (\%) & Rej$_{50\%}$ & Rej$_{30\%}$ \\"]
    hw = lambda f: [("accuracy", f("qg_herwig", "accuracy")), ("Rej50%", f("qg_herwig", "r50")),     # noqa: E731
                    ("Rej30%", f("qg_herwig", "r30"))]
    herwig = two(r"Quark--gluon discrimination scored on Herwig jets, for models trained on Pythia jets; mean $\pm$ "
                 r"standard deviation over runs.", "tab:herwig",
                 tab(hw(probe), pre + refs[:2], hw_head), tab(hw(ft), pre + scr, hw_head))
    # cold readers (draft e6e3b22): every number the text quotes is in a table. The linear probes' AUC over the
    # resonant and the QCD classes and of the class pairs; fine-tuned JetClass-II below the largest training set; the
    # resonance benchmark; the pretrained output layer
    four = [(n, a, k) for n, a, k in pre if a in NESTED]
    pair = lambda key: (lambda a: [R._pair(fits, m, key) for m in runs[a]])                           # noqa: E731
    split = one_tab(r"Linear probes on JetClass-II in more detail: AUC averaged over the resonant and over the QCD "
                    r"classes, and AUC of binary probes on three pairs of classes; mean $\pm$ standard deviation over runs.",
                    "tab:pairs", tab([("AUC resonant", probe("jc2_188", "macro_auc_resonant")),
                                      ("AUC QCD", probe("jc2_188", "macro_auc_qcd")), ("bb vs cc", pair("bb_vs_cc")),
                                      ("bbqq vs ccqq", pair("bbqq_vs_ccqq")), ("bc vs bq, cs", pair("bc_vs_bq_cs"))], four,
                                     [r" & \multicolumn{2}{c}{All classes, AUC} & \multicolumn{3}{c}{Pairs of classes, AUC} \\",
                                      r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
                                      r" & Resonant & QCD & $b\bar b$ / $c\bar c$ & $b\bar bqq$ / $c\bar cqq$ & $bc$ / $bq$, $cs$ \\"]))
    small = sorted(n for n in R._sizes(per, "jc2") if n < sizes["jc2"] and all(
        R._ft(per, "jc2", m, n, "accuracy") is not None for a in NESTED for m in runs[a]))
    ftn = lambda n: (lambda a: (R.summary_scratch(per, "jc2", n, "accuracy") if a == S                # noqa: E731
                                else [R._ft(per, "jc2", m, n, "accuracy") for m in runs[a]]))
    size = one_tab(r"Fine-tuned accuracy (\%) on JetClass-II against the number of training jets; mean $\pm$ "
                   r"standard deviation over runs.", "tab:size",
                   tab([(f"accuracy {n}", ftn(n)) for n in small + [sizes["jc2"]]], four + scr,
                       [" & " + " & ".join(R._power(f"{n:e}") for n in small + [sizes["jc2"]]) + r" \\"]))
    xbx = got["summary_kw"].get("xbx") or {}
    XR, xfits, xscr = xbx.get("XR"), xbx.get("fits") or {}, xbx.get("scr") or {}
    eff = lambda sig: (lambda a: ([100 * x["eps_s_at_star"] for x in xscr.get(sig, [])] if a == S else   # noqa: E731
                                  [100 * rec["eps_s_at_star"] for m in runs[a]
                                   if (rec := XR.readout(xfits.get(m, {}).get(sig, {}), "head"))]))
    res = one_tab(r"The resonance benchmark: signal efficiency in percent at the background efficiency of the text, by "
                  r"transfer learning on the frozen features and for the same network trained from scratch; mean "
                  r"$\pm$ standard deviation over runs.", "tab:resonance",
                  tab([("X->bb", eff("xbb")), ("X->bs", eff("xbs"))], four + scr, [r" & $X\to b\bar b$ & $X\to bs$ \\"]))
    return {"table_auc.tex": auc, "table_flavor.tex": flavor, "table_herwig.tex": herwig, "table_pairs.tex": split,
            "table_size.tex": size, "table_resonance.tex": res}


def one_tab(caption: str, label: str, tabular: list) -> str:
    return "\n".join([r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}",
                      r"\fontsize{9}{11}\selectfont\setlength{\tabcolsep}{4pt}", *tabular, r"\end{table}", ""])


def table_head(got: dict, runs: dict) -> str:
    """The pretrained output layer on the probes' test jets, as it is and with only its class biases refitted
    (experiments/EVAL/head_accuracy.py), beside the linear probe: each run of the 188-class model and Sophon (released)."""
    R = got["R"]
    f = sorted((REPO / "experiments/FIGS/data/v2/head_acc").glob("head_acc_bias_*.json"))
    doc = json.loads(f[0].read_text())["models"]
    g = R.summary_probe(*got["summary"][:3], "jc2_188", "accuracy")
    rows = [(f"188 classes, run {k}", m) for k, m in enumerate(runs["L188"], 1)] + [("Sophon (released)", R.SOPHON_MODEL)]
    if any(m not in doc for _, m in rows):
        raise SystemExit(f"FATAL: the output-layer file lacks {[m for _, m in rows if m not in doc]}")
    body = [f"{n} & {100 * doc[m]['accuracy']:.2f} & {100 * doc[m]['bias_refit']['accuracy']:.2f} & "
            f"{100 * float(g(m)):.2f}" + r" \\" for n, m in rows]
    return one_tab("Accuracy (\\%) on the class-balanced JetClass-II test jets of the pretrained output layer, of the same "
                   "layer with only its class biases refitted on the balanced training jets, and of the linear probe.",
                   "tab:head", [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                                r" & Output layer & Biases refitted & Linear probe \\", r"\midrule", *body,
                                r"\bottomrule", r"\end{tabular}"])



def main() -> int:
    maps = class_maps(REPO)
    got = capture(REPO)
    R, built = got["R"], got["built"]
    t2, t3, pending, lp_cols, ft_cols = tables23(got, maps)
    lines, bad = check_claims(R, got["tested"])
    d = REPO / OUT
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(ln for ln in lines if "binary benchmarks: no significant" not in ln or "passes" not in ln))
    print(f"fine-tuning read-out: {R.ft_source(REPO / 'experiments/FIGS/data/v2')}")
    print(f"pending cells: {len(pending)}")
    for r, c in pending:
        print(f"  PENDING  Table 3 | {r} | {c}")
    if bad:      # STYLE section 4: the build stops when a sentence's test is missing or fails
        raise SystemExit("FATAL: a clause of the text without a passing test:\n" + "\n".join(bad))
    runs = R._pooled(got["summary"][4])
    out = {"table1.tex": table1(maps, runs), "table2.tex": t2, "table3.tex": t3,
           "numbers.tex": numbers(got, lp_cols, ft_cols, maps, runs),
           "pending.txt": "".join(f"Table 3 | {r} | {c}\n" for r, c in pending),
           "results_generated.tex": built["results_generated.tex"],
           "table_classmap.tex": table_classmap(maps), **appendix_tables(got, maps), "table_head.tex": table_head(got, runs),
           "table4.tex": reuse(built, "tables/mlst_mass.tex", TABLE4_CAPTION, "tab:mass"),
           "table_inputs.tex": reuse(built, "tables/mlst_inputs.tex", INPUTS_CAPTION, "tab:inputs"),
           # 7.5 pt: at 8 pt the four blocks overran the page number (draft da28a31, p. 25)
           "table_published.tex": reuse(built, "tables/mlst_published.tex", PUBLISHED_CAPTION, "tab:published",
                                        r"\fontsize{7.5}{9}\selectfont", ovo=load_ovo()),
           "table_mlp.tex": reuse(built, "tables/mlst_mlp.tex", MLP_CAPTION, "tab:mlp", r"\fontsize{8}{10}\selectfont"),
           "table_groups.tex": reuse(built, "tables/mlst_coarse.tex", GROUPS_CAPTION, "tab:groups")}
    for f, text in out.items():
        (d / f).write_text(text)
    print("V2 TABLES COMPLETE")
    return 0


TABLE4_CAPTION = (r"Jet-mass resolution and median response of the estimated mass over the generator-level mass, on "
                  r"JetClass-II test jets with a generator-level match; mean $\pm$ standard deviation over runs and, for "
                  r"the soft-drop mass, over bootstrap resamples of the test jets.")
INPUTS_CAPTION = (r"The per-particle inputs, with a bullet where the top-tagging or quark--gluon data provide an input "
                  r"and a dash where they lack it: momenta and their logarithms, distances $\Delta\eta$, $\Delta\phi$ "
                  r"and $\Delta R$ from the jet axis, charge and particle type, and the transverse and longitudinal "
                  r"displacements $d_0$, $d_z$ of a track with their uncertainties. The last column gives the value "
                  r"zero takes after the input normalization.")
def load_ovo() -> dict:
    doc = json.loads((REPO / "experiments/FIGS/data/v2/stats/table_e1_ovo.json").read_text())
    return {r["row"]: r for r in doc["rows"]}


PUBLISHED_CAPTION = (r"Published models beside ours on JetClass, top tagging and quark--gluon discrimination, as each "
                     r"paper reports them; ours are the mean $\pm$ standard deviation over runs, at the published precision. AUC on "
                     r"JetClass averages over pairs of classes, as published; with equal class counts it equals the "
                     r"average over classes against the rest. The quark--gluon rows use the reduced particle-type inputs.")
GROUPS_CAPTION = (r"Probes asked only for the group of the 17-class label set to which a jet belongs, on the frozen "
                  r"features of each model; mean $\pm$ standard deviation over runs.")
MLP_CAPTION = (r"MLP probes on the frozen features, on JetClass-II, the benchmarks and pairs of JetClass-II classes; "
               r"mean $\pm$ standard deviation over runs.")


if __name__ == "__main__":
    sys.exit(main())
