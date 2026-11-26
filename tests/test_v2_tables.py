"""Tables 1-3 of the rewritten paper (experiments/FIGS/mlst_v2_tables.py; PI 2026-11-24, LEDGER RW-4, RW-9)."""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "FIGS"))
import mlst_v2_tables as V  # noqa: E402

MAPS = V.class_maps(ROOT)


@pytest.mark.parametrize("pm, want", [(r"0.4348\,$\pm$\,0.0004", "0.4348(4)"), (r"0.4328\,$\pm$\,0.0010", "0.4328(10)"),
                                      (r"72.8\,$\pm$\,2.4", "72.8(2.4)"), (r"1{,}800\,$\pm$\,60", "1{,}800(60)"),
                                      ("0.4340", "0.4340"), ("$>$0.9999", "$>$0.9999")])
def test_paren(pm, want):
    assert V.paren(pm) == want


def test_marks_of_the_nested_label_sets():
    """Each coarser label set merges what the finer one merges and more; the coarsest keeps prong number and type."""
    n = len(V.DISTINCTIONS)
    assert V.marks(MAPS, "L188") == [True] * n
    assert V.marks(MAPS, "L162") == [False] + [True] * (n - 1)
    assert V.marks(MAPS, "R42_Q1") == [False] * (n - 2) + [True, True]
    assert V.marks(MAPS, "R16_Q1") == [False] * (n - 1) + [True]


def test_classes_and_the_class_of_xbb():
    assert [V.counts(MAPS[a]) for a in V.NESTED] == [(161, 27), (161, 1), (42, 1), (16, 1)]
    assert V.bb_class(MAPS["L162"]) == r"$X\to b\bar b$ alone"
    assert V.bb_class(MAPS["R42_Q1"]) == r"decays to $b\bar b$, $bc$ or $bq$"
    assert V.bb_class(MAPS["R16_Q1"]) == "every decay to two quarks or gluons"


def test_table1_rows():
    """PI 2026-11-24 (GO-6): version B, one table, with its fixes."""
    runs = {**{a: ["a", "b", "c"] for a in V.NESTED}, **{m: ["a", "b"] for m, _ in V.MASS},
            **{c: ["a"] for c in MAPS if c.startswith("RAND2_p")}, **{c: ["a", "b"] for c in V.FLAV}}
    tex = V.table1(MAPS, runs)
    assert tex.count(r"\begin{table}") == 1 and "sideways" not in tex and "landscape" not in tex
    rows = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and ln.split(" & ")[0].endswith(("classes", "grouping", "groupings", "regression"))]
    assert [ln.split(" & ")[0] for ln in rows] == [
        "188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
        "17 classes, flavor-boundary groupings", "162 classes + mass regression", "17 classes + mass regression"]
    assert "5 groupings, 1 each" in tex and "3 groupings, 2 each" in tex and "(resonant + QCD)" in tex
    assert "Class containing" in tex and "two-parton" not in tex   # the caption no longer repeats section 3.3 (editor, 25)


def test_a_bound_in_a_percent_column_is_scaled():
    """A spread printed as a bound (below 0.001 of a fraction) is in percent too (draft 8f085fe, Table E1)."""
    assert V.pct(r"0.843\,$\pm$\,$<$0.001") == r"84.3\,$\pm$\,$<$0.1"


def test_table_cell_reads_one_cell_or_stops():
    tex = "\n".join([r"\begin{tabular}{lcc}", r"\multicolumn{3}{l}{top tagging} \\", r"ours & 1 & 2{,}799\,$\pm$\,56 \\",
                     r"\multicolumn{3}{l}{quark/gluon tagging} \\", r"ours & 3 & 84.3\,$\pm$\,0.1 \\", r"\end{tabular}"])
    assert V.table_cell(tex, 0, "top tagging", "ours", 2) == r"2{,}799\,$\pm$\,56"
    assert V.table_cell(tex, 0, "quark/gluon tagging", "ours", 2, False) == "84.3"
    with pytest.raises(SystemExit):
        V.table_cell(tex, 0, None, "ours", 2)


def test_class_map_has_every_class_once_with_its_coarser_classes():
    """Table A1 (2026-11-25): 188 rows, unique names, each 162/43/17 number the map's, every row printed."""
    rows = V.classmap_rows(MAPS)
    assert len(rows) == 188 and len({r[1] for r in rows}) == 188 and sorted(r[0] for r in rows) == list(range(1, 189))
    by_number = {int(MAPS["L188"][k]) + 1: k for k in MAPS["L188"]}
    for n188, name, n162, n43, n17 in rows:
        k = by_number[n188]
        assert (n162, n43, n17) == tuple(int(MAPS[a][k]) + 1 for a in ("L162", "R42_Q1", "R16_Q1"))
        assert name == V.class_name(k)
    tex = V.table_classmap(MAPS)
    assert all(" & ".join(str(x) for x in r) + r" \\" in tex for r in rows)
    assert V.class_name("X_bb") == r"$X\to bb$" and V.class_name("QCD_bb") == "QCD $bb$"
    assert V.class_name("X_YY_bbcc") == r"$X\to YY\to bbcc$"


def test_strip_bold_keeps_the_cell():
    assert V.strip_bold(r"a & {\fontseries{b}\selectfont 1{,}800\,$\pm$\,60} & b") == r"a & 1{,}800\,$\pm$\,60 & b"


def test_value_table_pending_one_place_per_column_no_bold():
    """GO-6 (PI 2026-11-24): mean +- SD, one number of decimals per column, the SD to the same decimals, no bold."""
    vals = {"p": [0.4348, 0.4351, 0.4344], "q": [0.331, 0.312, 0.350], "r": [0.4215, 0.4230, 0.4200], "s": [None, 0.1],
            "o": [0.4340]}
    cols = [("x", lambda a: vals[a])]
    body, missing = V.value_table(cols, [("P", "p", 3), ("Q", "q", 3), ("R", "r", 3), None, ("S", "s", 2), ("O", "o", 1)],
                                  V.PENDING)
    assert missing == [("S", "x")] and body[3] == r"\midrule" and body[4] == r"S & pending \\"
    assert not any(r"\textbf" in b for b in body)
    assert body[0] == r"P & 0.4348\,$\pm$\,0.0004 \\" and body[1] == r"Q & 0.3310\,$\pm$\,0.0190 \\"
    assert body[5] == r"O & 0.4340 \\"


def test_a_spread_that_rounds_to_zero_is_printed_as_a_bound():
    """PI 2026-11-25 (LEDGER §31 3): "< 0.0001" for the spread at the column's decimals, never "0.0000"."""
    vals = {"p": [0.8432, 0.8436, 0.8429], "p2": [0.8402, 0.8406, 0.8399], "q": [0.843175, 0.843180, 0.843180],
            "r": [0.8425, 0.8425]}
    body, missing = V.value_table([("acc", lambda a: vals[a])], [("P", "p", 3), ("P2", "p2", 3), ("Q", "q", 3), ("R", "r", 2)],
                                  V.PENDING)
    assert not missing
    assert body[0] == r"P & 0.8432\,$\pm$\,0.0004 \\"
    assert body[2] == r"Q & 0.8432\,$\pm$\,$<$0.0001 \\" and body[3] == r"R & 0.8425\,$\pm$\,$<$0.0001 \\"
    assert "0.0000" not in "".join(body)


def test_accuracy_in_percent_one_place_per_column():
    """PI 2026-11-25 (LEDGER 32.3): accuracy columns in percent, scaled before the decimals are chosen; a reused table's
    accuracy column moves its decimal point exactly, and a new block's header resets the column."""
    vals = {"p": [0.4348, 0.4351, 0.4344], "q": [0.331, 0.312, 0.350]}
    body, _ = V.value_table([("JetClass-II accuracy", lambda a: vals[a])], [("P", "p", 3), ("Q", "q", 3)], V.PENDING)
    assert body[0] == r"P & 43.48\,$\pm$\,0.04 \\" and body[1] == r"Q & 33.10\,$\pm$\,1.90 \\"
    assert V.pct(r"0.71658\,$\pm$\,0.00022") == r"71.658\,$\pm$\,0.022" and V.pct("0.940") == "94.0"
    tab = "\n".join([r"\toprule", r" & parameters & accuracy \\", r"\midrule", r"ParT & 2.14\,M & 0.836 \\",
                      r"\toprule", r" & AUC & Rej \\", r"\midrule", r"ParT & 0.9858 & 0.5 \\"])
    out = V.percent_columns(tab)
    assert r"accuracy (\%)" in out and r"ParT & 2.14\,M & 83.6 \\" in out and r"ParT & 0.9858 & 0.5 \\" in out


def test_our_rows_follow_the_published_precision():
    """PI 2026-11-25 (LEDGER 33.4): our rows at the published rows' decimals; JetClass AUC one-versus-one."""
    tab = "\n".join([r"\toprule", r" & AUC & Rej \\", r"\midrule", r"\multicolumn{3}{@{}l}{JetClass} \\",
                      r"ParT & 0.9834 & -- \\", r"fine-tuned from 188 classes (this work) & 0.985907\,$\pm$\,0.000012 & -- \\",
                      r"\multicolumn{3}{@{}l}{top tagging} \\", r"ParT & 0.9858 & 413\,$\pm$\,16 \\",
                      r"trained from scratch (this work) & 0.98524\,$\pm$\,0.00008 & 358.4\,$\pm$\,11.2 \\"])
    ovo = {"188 classes": {"auc_ovo": 0.98590661, "auc_ovo_sd": 1.2e-05}, "trained from scratch": {"auc_ovo": 0.9822, "auc_ovo_sd": 0.0003}}
    out = V.published_precision(tab, ovo)
    assert r"(this work) & 0.9859\,$\pm$\,$<$0.0001 & --" in out
    assert r"(this work) & 0.9852\,$\pm$\,0.0001 & 358\,$\pm$\,11" in out


def test_table4_resolution_in_percent_one_decimal_response_three():
    """PI 2026-11-26 (LEDGER 34.4): resolution in percent with one decimal, response with three, from the unrounded
    values (a printed 0.0055 cannot be rounded to three decimals); a spread that rounds to zero is a bound."""
    tex = "\n".join([r"\begin{tabular}{lcc}", r"\toprule", r" & resolution & median response \\", r"\midrule",
                     r"\multicolumn{3}{l}{\textit{MLP probe on the frozen features}} \\",
                     r"\Kfull{} classes & 0.21806\,$\pm$\,0.00189 & 0.9919\,$\pm$\,0.0021 \\",
                     r"\Kres{} classes + mass regression & 0.15726\,$\pm$\,0.00003 & 0.9992\,$\pm$\,0.0055 \\",
                     r"soft-drop mass & 0.24485\,$\pm$\,0.00062 & 0.9294\,$\pm$\,0.0004 \\", r"\bottomrule"])
    vals = {("mlp", r"\Kfull{} classes"): (0.218061, 0.001891, 0.99191, 0.00209),
            ("mlp", r"\Kres{} classes + mass regression"): (0.157263, 0.000031, 0.999204, 0.005549),
            ("sd", "soft-drop mass"): (0.244851, 0.000622, 0.929402, 0.000398)}
    out = V.mass_columns(tex, vals).split("\n")
    assert out[2] == r" & resolution (\%) & median response \\"
    assert out[4].startswith(r"\multicolumn")
    assert out[5] == r"\Kfull{} classes & 21.8\,$\pm$\,0.2 & 0.992\,$\pm$\,0.002 \\"
    assert out[6] == r"\Kres{} classes + mass regression & 15.7\,$\pm$\,$<$0.1 & 0.999\,$\pm$\,0.006 \\"
    assert out[7] == r"soft-drop mass & 24.5\,$\pm$\,0.1 & 0.929\,$\pm$\,$<$0.001 \\"
    with pytest.raises(SystemExit):
        V.mass_columns(tex, {})


def test_generated_tables_float_to_the_top():
    """PI 2026-11-26 (LEDGER 34.4): tables float to the top of a page."""
    for f in ("experiments/FIGS/mlst_v2_tables.py", "experiments/FIGS/perclass_jc1_table.py"):
        assert "[htbp]" not in (ROOT / f).read_text()
