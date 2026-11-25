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
    """The class of a map that contains X->bb, as Table 1 names it."""
    with_bb = [k.removeprefix("X_") for k, v in m.items() if v == m["X_bb"]]
    if with_bb == ["bb"]:
        return r"$X\to b\bar b$ only"
    if all(k in DECAY for k in with_bb):
        return r"$X\to " + "$, $".join(DECAY[k] for k in with_bb) + "$"
    if sorted(with_bb) == sorted(k.removeprefix("X_") for k in m if re.fullmatch(r"X_[bcsqg]{2}", k)):
        return "all two-parton"
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


def table1(maps: dict, runs: dict) -> str:
    """Versions A and B, one float each, for one page. Below the rule the random groupings and the mass regression."""
    rand = sorted(c for c in maps if c.startswith("RAND2_p"))
    n_runs = {a: str(len(runs[a])) for a in NESTED + tuple(m for m, _ in MASS)}
    if {len(runs[c]) for c in rand} != {1}:
        raise SystemExit("FATAL: a random grouping without exactly one run")
    n_runs[RAND] = rf"{len(rand)} $\times$ 1"
    cls = lambda m: "{} + {}".format(*counts(m))                    # noqa: E731
    sym = lambda x: KEEPS if x else MERGES                          # noqa: E731
    rand_counts = {counts(maps[c]) for c in rand}
    if len(rand_counts) != 1:
        raise SystemExit("FATAL: the random groupings differ in their resonant and QCD classes")
    rand_marks = [{mark(maps[c], pairs) for c in rand} for _, _, pairs in DISTINCTIONS]
    nd = len(DISTINCTIONS)
    # version A: the row names wrap (two lines below the rule) and the marks sit close, to keep the type size
    a = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.7cm}l" + r"@{\hspace{6pt}}c" * nd + "lc@{}}", r"\toprule",
         "Label set & Classes & " + " & ".join(h for (h, _), _, _ in DISTINCTIONS) + r" & Class with & Runs \\",
         r" & res.\ + QCD & " + " & ".join(h for (_, h), _, _ in DISTINCTIONS) + r" & $X\to b\bar b$ & \\",
         r"\midrule"]
    b = [r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{5.6cm}>{\raggedright\arraybackslash}p{3.1cm}c@{}}",
         r"\toprule", r"Label set & Classes & Merges & Class with & Runs \\",
         r" & res.\ + QCD & & $X\to b\bar b$ & \\", r"\midrule"]
    for arm in NESTED:
        m = maps[arm]
        a.append(f"{name(maps, arm)} & {cls(m)} & " + " & ".join(sym(x) for x in marks(maps, arm))
                 + f" & {bb_class(m)} & {n_runs[arm]}" + r" \\")
        b.append(f"{name(maps, arm)} & {cls(m)} & {merged_words(maps, arm)} & {bb_class(m)} & {n_runs[arm]}" + r" \\")
    a.append(r"\midrule")
    b.append(r"\midrule")
    # a random grouping keeps some class pairs of a distinction and merges others, and the five differ: one note over
    # the columns no single mark describes
    clean = [next(iter(s)) if len(s) == 1 and None not in s else None for s in rand_marks]
    first = next(i for i, x in enumerate(clean) if x is None)
    if any(x is not None for x in clean[first:]):
        raise SystemExit("FATAL: Table 1's note for the random groupings spans a column with a single mark")
    res, qcd = rand_counts.pop()
    a.append(f"{name(maps, RAND)} & {res} + {qcd} & " + " & ".join(sym(x) for x in clean[:first])
             + rf" & \multicolumn{{{nd - first}}}{{c}}{{mixed: kept for some class pairs only}} & varies & "
             + n_runs[RAND] + r" \\")
    n_res = len([k for k in maps["L188"] if not k.startswith("QCD")])
    b.append(f"{name(maps, RAND)} & {res} + {qcd} & the {n_res} resonant classes into {res} random groups, "
             f"QCD into one & varies & {n_runs[RAND]}" + r" \\")
    for mass, base in MASS:
        m = maps[base]
        a.append(f"{name(maps, mass)} & {cls(m)} & " + " & ".join(sym(x) for x in marks(maps, base))
                 + f" & {bb_class(m)} & {n_runs[mass]}" + r" \\")
        b.append(f"{name(maps, mass)} & {cls(m)} & as {name(maps, base)} & {bb_class(m)} & {n_runs[mass]}" + r" \\")
    end = [r"\bottomrule", r"\end{tabular}"]
    return (_float("Table 1 (version A). The label sets, the distinctions between JetClass-II classes that each keeps "
                   r"(\checkmark) or merges (--), and the pretraining runs.", "tab:setsA", a + end, numbered=False)
            + "\n" + _float("Table 1 (version B). The label sets, what each merges, and the pretraining runs.",
                            "tab:setsB", b + end, numbered=False))


def _float(caption: str, label: str, tabular: list, numbered: bool = True) -> str:
    """A portrait table float on one page: the caption above, the tabular at 9 pt, shrunk only if wider than the text."""
    cap = rf"\caption{{{caption}}}\label{{{label}}}" if numbered else rf"\caption*{{{caption}}}\label{{{label}}}"
    return "\n".join([r"\begin{table}[tbp]", r"\centering", cap, r"\fontsize{9}{11}\selectfont\setlength{\tabcolsep}{4pt}",
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
    bold. A cell is printed only when every one of its runs has a value."""
    stats, missing = {}, []
    for r in rows:
        if r is None:
            continue
        for c, (cname, get) in enumerate(columns):
            v = [float(x) for x in get(r[1]) if x is not None]
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
            install(mlst, g)
            got["R"] = mlst
            summary, tests = mlst.table_summary, mlst.table_tests

            def table_summary(*a, **k):
                got.setdefault("summary", a)
                return summary(*a, **k)

            def table_tests(*a, **k):
                got["tested"] = k["rows_out"]
                return tests(*a, **k)
            mlst.table_summary, mlst.table_tests = table_summary, table_tests
        rel.install_mlst = install_mlst
        return rel
    MT.data_release_module = release_module
    try:
        MT.build(root)
    finally:
        MT.data_release_module = load
    if {"R", "summary", "tested"} - set(got):
        raise SystemExit(f"FATAL: the build did not reach {sorted({'R', 'summary', 'tested'} - set(got))}")
    return got


# PI 2026-11-24 (GO-6) and 2026-11-25 (section 28, 1.3): the same six columns in Tables 2 and 3
HEAD = [r" & JetClass-II & JetClass & \multicolumn{2}{c}{Top tagging} & \multicolumn{2}{c}{Quark/gluon} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & Accuracy & Accuracy & Accuracy & Rej$_{30\%}$ & Accuracy & Rej$_{30\%}$ \\", r"\midrule"]
LP_HEAD = FT_HEAD = HEAD


def tables23(got: dict, maps: dict) -> tuple:
    """(Table 2, Table 3, the fine-tuned cells that are pending as [(row, column)], Table 2's columns, Table 3's)."""
    R = got["R"]
    fits, fill, chosen, per, runs = got["summary"][:5]
    runs = R._pooled(runs)
    pre = [(name(maps, a), a, len(runs[a])) for a in NESTED + (RAND,) + tuple(m for m, _ in MASS)]

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


# The clauses of the abstract, the introduction and the outline (LEDGER RW-9) and the tested differences each rests on:
# (clause, heading of the build's table of tests, first network, second network, what must hold: "below" / "above"
# (first minus second beyond 3 sigma) or "null" (within 3 sigma); "report": printed, never a stop (not stated)).
def claims(R, tested: dict) -> tuple:
    lp, ft = R.HEADING["lp"], R.HEADING["ft"]
    acc, auc = f"JetClass-II, accuracy, {lp}", f"JetClass-II, AUC, {lp}"
    pair = lambda k: f"{R._PAIR[k]}, AUC, {lp}"                    # noqa: E731
    top, qg = f"top tagging, $R_{{30}}$, {ft}", f"quark/gluon tagging (Pythia), $R_{{30}}$, {ft}"
    jc1, jc2 = f"JetClass, accuracy, {ft}", f"JetClass-II, accuracy, {ft}"
    sizes = sorted(h for h, a, b in tested if h.startswith(jc2 + ", ") and (a, b) == ("R16_Q1", "L188"))
    if not sizes:
        raise SystemExit("FATAL: no tested difference of fine-tuned JetClass-II below the largest training set")
    mass = "jet mass, resolution, multilayer-perceptron probe on the frozen features"
    mlp = "JetClass-II, AUC, multilayer-perceptron probe on the frozen features"
    less = "coarser labels: the 188 classes identified less accurately"
    pairs = "decays that differ only in a merged flavor are harder to separate"
    rand = "random groupings into 17 classes lose far less"
    jet = "after fine-tuning the coarsest set remains worse on JetClass"
    out = "outline 4.3: JetClass significant at 43 and 17 classes, not at 162; JetClass-II likewise; every training size"
    null = "no significant difference on top tagging and quark-gluon after fine-tuning"
    sharp = "mass regression sharpens the mass resolution of the frozen features"
    cost = "mass regression: no significant cost in frozen-feature classification"
    ftcost = "outline 4.4: fine-tuned, the 17-class model with mass regression is less accurate on JetClass"
    notsaid = "not stated (the RW-9 draft's unscoped 'no significant cost in classification')"
    return ((less, acc, "L162", "L188", "report"), (less, auc, "L162", "L188", "below"), (less, acc, "R42_Q1", "L188", "below"),
            (less, acc, "R16_Q1", "L188", "below"),
            (pairs, pair("bb_vs_cc"), "R16_Q1", "L188", "below"), (pairs, pair("bbqq_vs_ccqq"), "R16_Q1", "L188", "below"),
            (pairs, pair("bc_vs_bq_cs"), "R42_Q1", "L188", "below"),
            (rand, acc, RAND, "R16_Q1", "above"), (rand, auc, RAND, "R16_Q1", "above"),
            (jet, jc1, "R16_Q1", "L188", "below"), (jet, jc2, "R16_Q1", "L188", "below"),
            (out, jc1, "R42_Q1", "L188", "below"), (out, jc1, "L162", "L188", "null"),
            (out, jc2, "R42_Q1", "L188", "below"), (out, jc2, "L162", "L188", "null"),
            *[(out, h, "R16_Q1", "L188", "below") for h in sizes],
            *[(null, h, a, "L188", "null") for h in (top, qg) for a in ("L162", "R42_Q1", "R16_Q1")],
            (sharp, mass, "L162_MASS", "L162", "below"), (sharp, mass, "R16_Q1_MASS", "R16_Q1", "below"),
            *[(cost, h, m, b, "null") for h in (acc, auc, mlp) for m, b in MASS],
            (ftcost, jc1, "R16_Q1_MASS", "R16_Q1", "below"), (notsaid, jc1, "L162_MASS", "L162", "report"),
            *[(notsaid, top, m, b, "report") for m, b in MASS], *[(notsaid, qg, m, "L188", "report") for m, _ in MASS])


def check_claims(R, tested: dict) -> tuple:
    """(one line per tested difference, the lines whose test is missing or fails)."""
    lines, bad = [], []
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
            nums = f"delta {v[0]:.5g}  sigma {v[4]:.5g}  delta/sigma {z:+.2f}  "
        line = f"{clause} | {head} | {a} - {b} | wants {want} | {nums}{verdict}"
        lines.append(line)
        if verdict != "passes" and want != "report":
            bad.append(line)
    return lines, bad


def numbers(lp_cols, ft_cols, maps: dict, runs: dict) -> str:
    """The numbers of the abstract, the introduction and the outline: the class counts of the label sets, the run
    counts in words, and the JetClass-II linear-probe accuracy of the 17- and the 188-class model in percent."""
    get = dict(lp_cols)["JetClass-II accuracy"]
    pct = lambda a: f"{100 * float(np.mean([float(x) for x in get(a)])):.0f}"      # noqa: E731
    ftv = dict(ft_cols)

    def ft_pct(col, arm):
        """A fine-tuned accuracy in percent to one decimal; a red bracket while a run of it is missing."""
        v = [x for x in ftv[col](arm) if x is not None]
        return f"{100 * float(np.mean([float(x) for x in v])):.1f}" if len(v) == len(runs[arm]) else r"\pending{pending}"
    one = lambda xs: MT.words(xs.pop()) if len(xs) == 1 else None                  # noqa: E731
    out = {"Kfull": len(set(maps["L188"].values())), "Kres": len(set(maps["L162"].values())),
           "Kmid": len(set(maps["R42_Q1"].values())), "Kcoarse": len(set(maps["R16_Q1"].values())),
           "VtwoAccCoarsePct": pct("R16_Q1"), "VtwoAccFullPct": pct("L188"),
           "VtwoFtJcOneCoarsePct": ft_pct("JetClass accuracy", "R16_Q1"), "VtwoFtJcOneFullPct": ft_pct("JetClass accuracy", "L188"),
           "VtwoFtJcTwoCoarsePct": ft_pct("JetClass-II accuracy", "R16_Q1"), "VtwoFtJcTwoFullPct": ft_pct("JetClass-II accuracy", "L188"),
           "VtwoRunsWord": one({len(runs[a]) for a in NESTED}), "VtwoMassRunsWord": one({len(runs[m]) for m, _ in MASS}),
           "VtwoRandWord": MT.words(len([c for c in maps if c.startswith("RAND2_p")]))}
    if None in out.values():
        raise SystemExit(f"FATAL: label sets with different run counts: {out}")
    return "% written by experiments/FIGS/mlst_v2_tables.py\n" + "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in out.items())


def main() -> int:
    maps = class_maps(REPO)
    got = capture(REPO)
    R = got["R"]
    t2, t3, pending, lp_cols, ft_cols = tables23(got, maps)
    lines, bad = check_claims(R, got["tested"])
    out = {"table1.tex": table1(maps, R._pooled(got["summary"][4])), "table2.tex": t2, "table3.tex": t3,
           "numbers.tex": numbers(lp_cols, ft_cols, maps, R._pooled(got["summary"][4])), "claims.txt": "\n".join(lines) + "\n",
           "pending.txt": "".join(f"Table 3 | {r} | {c}\n" for r, c in pending)}
    d = REPO / OUT
    d.mkdir(parents=True, exist_ok=True)
    for f, text in out.items():
        (d / f).write_text(text)
    print("\n".join(lines))
    print(f"fine-tuning read-out: {R.ft_source(REPO / 'experiments/FIGS/data/v2')}")
    print(f"pending cells: {len(pending)}")
    for r, c in pending:
        print(f"  PENDING  Table 3 | {r} | {c}")
    if bad:      # STYLE section 4: the build stops when a sentence's test is missing or fails
        raise SystemExit("FATAL: a clause of the abstract without a passing test:\n" + "\n".join(bad))
    print("V2 TABLES COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
