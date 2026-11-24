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


def bold(cell: str) -> str:
    m = PM.match(cell)
    return rf"\textbf{{{m['m']}}}\,$\boldsymbol{{\pm}}$\,\textbf{{{m['s']}}}" if m else rf"\textbf{{{cell}}}"


def cell(values: list, n: int, frac: bool, concise: bool, lone) -> tuple:
    """(the printed cell, its mean) of a row's run values; (None, None) unless every one of its n runs has a value."""
    v = [float(x) for x in values if x is not None]
    if len(v) < n or not v:
        return None, None
    if len(v) == 1:
        return (f"{v[0]:.4f}" if frac else lone(v[0])), v[0]
    pm = MT.fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    return (paren(pm) if concise else pm), float(np.mean(v))


def value_table(columns, rows, best_of, concise: bool, lone, empty: str) -> tuple:
    """(the tabular's body rows, the cells with no value) of a model-per-row table. columns: (name, frac, f(arm) ->
    its run values); rows: (name, arm, runs expected) or None for a rule; best_of: the arms bold is chosen among."""
    cells, means, missing = {}, {}, []
    for r in rows:
        if r is None:
            continue
        for c, (cname, frac, get) in enumerate(columns):
            cells[r[1], c], means[r[1], c] = cell(get(r[1]), r[2], frac, concise, lone)
            if cells[r[1], c] is None:
                missing.append((r[0], cname))
    for c in range(len(columns)):
        have = [a for a in best_of if means.get((a, c)) is not None]
        if have:
            top = max(have, key=lambda a: means[a, c])
            cells[top, c] = bold(cells[top, c])
    body = [r"\midrule" if r is None else
            r[0] + " & " + " & ".join(cells[r[1], c] or empty for c in range(len(columns))) + r" \\" for r in rows]
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


LP_HEAD = [r" & \multicolumn{2}{c}{JetClass-II} & \multicolumn{2}{c}{JetClass} & Top tagging & Quark/gluon \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r" & Accuracy & AUC & Accuracy & AUC & Rej$_{30\%}$ & Rej$_{30\%}$ \\", r"\midrule"]
FT_HEAD = [r" & JetClass-II & JetClass & Top tagging & Quark/gluon \\",
           r" & Accuracy & Accuracy & Rej$_{30\%}$ & Rej$_{30\%}$ \\", r"\midrule"]


def tables23(got: dict, maps: dict) -> tuple:
    """(Table 2, Table 3, the fine-tuned cells that are pending as [(row, column)], Table 2's columns)."""
    R = got["R"]
    fits, fill, chosen, per, runs = got["summary"][:5]
    runs = R._pooled(runs)
    pre = [(name(maps, a), a, len(runs[a])) for a in NESTED + (RAND,) + tuple(m for m, _ in MASS)]
    best = [a for _, a, _ in pre]

    def probe(ds, met):
        g = R.summary_probe(fits, fill, chosen, ds, met)
        # the released checkpoint on top and quark/gluon: only with the substitution probes ours use (mlst_results)
        return lambda a: [] if g is None or (a == R.SOPHON and ds in R.SUMMARY_NO_SOPHON and R.SOPHON_MODEL not in fill) \
            else [g(m) for m in runs[a]]

    lp_cols = [("JetClass-II accuracy", True, probe("jc2_188", "accuracy")), ("JetClass-II AUC", True, probe("jc2_188", "macro_auc_ovr")),
               ("JetClass accuracy", True, probe("jc1", "accuracy")), ("JetClass AUC", True, probe("jc1", "macro_auc_ovr")),
               ("top Rej30%", False, probe("top", "r30")), ("quark/gluon Rej30%", False, probe("qg", "r30"))]
    lp_rows = pre + [None, ("untrained network", R.INIT_BN, len(runs[R.INIT_BN])), ("Sophon (released)", R.SOPHON, 1)]
    body, miss = value_table(lp_cols, lp_rows, best, True, R._tab1, "--")
    if miss:
        raise SystemExit(f"FATAL: Table 2 has cells without a linear probe: {miss}")
    t2 = _float(r"Linear probes on the frozen features: accuracy, AUC and background rejection at a signal efficiency of "
                r"30\% (Rej$_{30\%}$). Bold: best pretrained model.", "tab:probes",
                [r"\begin{tabular}{@{}l" + "c" * len(lp_cols) + "@{}}", r"\toprule", *LP_HEAD, *body, r"\bottomrule",
                 r"\end{tabular}"])

    scratch = "trained from scratch"

    def ft(key, met):
        n = {k: n for k, n, *_ in R.SUMMARY_PANELS}[key]
        g = R.summary_ft(per, key, n, met)
        return lambda a: R.summary_scratch(per, key, n, met) if a == scratch else [] if g is None else [g(m) for m in runs[a]]

    ft_cols = [("JetClass-II accuracy", True, ft("jc2", "accuracy")), ("JetClass accuracy", True, ft("jc1", "accuracy")),
               ("top Rej30%", False, ft("top", "r30")), ("quark/gluon Rej30%", False, ft("qg", "r30"))]
    ft_rows = pre + [None, (scratch, scratch, len(R.FT_SCRATCH_MODELS))]
    body, pending = value_table(ft_cols, ft_rows, best, False, R._tab1, PENDING)
    t3 = _float(r"Fine-tuned models: accuracy and background rejection at a signal efficiency of 30\% (Rej$_{30\%}$). "
                "Bold: best pretrained model.", "tab:finetuned",
                [r"\begin{tabular}{@{}l" + "c" * len(ft_cols) + "@{}}", r"\toprule", *FT_HEAD, *body, r"\bottomrule",
                 r"\end{tabular}"])
    return t2, t3, pending, lp_cols


# The abstract's clauses (LEDGER RW-9) and the tested differences behind each: (clause, heading of the build's table of
# tests, first network, second network, what must hold: "below" / "above" (first minus second beyond 3 sigma) or
# "null" (within 3 sigma); "report": printed, never a stop (the abstract does not state it)).
def claims(R) -> tuple:
    lp, ft = R.HEADING["lp"], R.HEADING["ft"]
    acc, auc = f"JetClass-II, accuracy, {lp}", f"JetClass-II, AUC, {lp}"
    pair = lambda k: f"{R._PAIR[k]}, AUC, {lp}"                    # noqa: E731
    top, qg = f"top tagging, $R_{{30}}$, {ft}", f"quark/gluon tagging (Pythia), $R_{{30}}$, {ft}"
    mass = "jet mass, resolution, multilayer-perceptron probe on the frozen features"
    mlp = "JetClass-II, AUC, multilayer-perceptron probe on the frozen features"
    less = "coarser labels: the 188 classes identified less accurately"
    pairs = "decays that differ only in a merged flavor are harder to separate"
    rand = "random groupings into 17 classes lose far less"
    null = "no significant difference on top tagging and quark-gluon after fine-tuning"
    sharp = "mass regression sharpens the mass resolution of the frozen features"
    cost = "mass regression: no significant cost in classification"
    return ((less, acc, "L162", "L188", "report"), (less, auc, "L162", "L188", "below"), (less, acc, "R42_Q1", "L188", "below"), (less, acc, "R16_Q1", "L188", "below"),
            (pairs, pair("bb_vs_cc"), "R16_Q1", "L188", "below"), (pairs, pair("bbqq_vs_ccqq"), "R16_Q1", "L188", "below"),
            (pairs, pair("bc_vs_bq_cs"), "R42_Q1", "L188", "below"),
            (rand, auc, RAND, "R16_Q1", "above"), (rand, acc, RAND, "R42_Q1", "above"), (rand, acc, "R16_Q1", "R42_Q1", "below"),
            *[(null, h, a, "L188", "null") for h in (top, qg) for a in ("L162", "R42_Q1", "R16_Q1")],
            (sharp, mass, "L162_MASS", "L162", "below"), (sharp, mass, "R16_Q1_MASS", "R16_Q1", "below"),
            *[(cost, h, m, b, "null") for h in (acc, auc, mlp, top) for m, b in MASS],
            *[(cost, qg, m, "L188", "null") for m, _ in MASS])


def check_claims(R, tested: dict) -> tuple:
    """(one line per tested difference, the lines whose test is missing or fails)."""
    lines, bad = [], []
    for clause, head, a, b, want in claims(R):
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


def numbers(lp_cols, maps: dict, runs: dict) -> str:
    """The numbers of the abstract, the introduction and the outline: the class counts of the label sets, the run
    counts in words, and the JetClass-II linear-probe accuracy of the 17- and the 188-class model in percent."""
    get = dict((n, g) for n, _, g in lp_cols)["JetClass-II accuracy"]
    pct = lambda a: f"{100 * float(np.mean([float(x) for x in get(a)])):.0f}"      # noqa: E731
    one = lambda xs: MT.words(xs.pop()) if len(xs) == 1 else None                  # noqa: E731
    out = {"Kfull": len(set(maps["L188"].values())), "Kres": len(set(maps["L162"].values())),
           "Kmid": len(set(maps["R42_Q1"].values())), "Kcoarse": len(set(maps["R16_Q1"].values())),
           "VtwoAccCoarsePct": pct("R16_Q1"), "VtwoAccFullPct": pct("L188"),
           "VtwoRunsWord": one({len(runs[a]) for a in NESTED}), "VtwoMassRunsWord": one({len(runs[m]) for m, _ in MASS}),
           "VtwoRandWord": MT.words(len([c for c in maps if c.startswith("RAND2_p")]))}
    if None in out.values():
        raise SystemExit(f"FATAL: label sets with different run counts: {out}")
    return "% written by experiments/FIGS/mlst_v2_tables.py\n" + "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in out.items())


def main() -> int:
    maps = class_maps(REPO)
    got = capture(REPO)
    R = got["R"]
    t2, t3, pending, lp_cols = tables23(got, maps)
    lines, bad = check_claims(R, got["tested"])
    out = {"table1.tex": table1(maps, R._pooled(got["summary"][4])), "table2.tex": t2, "table3.tex": t3,
           "numbers.tex": numbers(lp_cols, maps, got["summary"][4]), "claims.txt": "\n".join(lines) + "\n",
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
