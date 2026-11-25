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
            **{c: ["a"] for c in MAPS if c.startswith("RAND2_p")}}
    tex = V.table1(MAPS, runs)
    assert tex.count(r"\begin{table}") == 1 and "sideways" not in tex and "landscape" not in tex
    rows = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and ln.split(" & ")[0].endswith(("classes", "grouping", "regression"))]
    assert [ln.split(" & ")[0] for ln in rows] == [
        "188 classes", "162 classes", "43 classes", "17 classes", "17 classes, random grouping",
        "162 classes + mass regression", "17 classes + mass regression"]
    assert "5 groupings, 1 each" in tex and "(resonant + QCD)" in tex and "Class containing" in tex
    assert "two-parton" not in tex and "number of decay products" in tex  # caption reworded after the cold readers (parton, not "kind")


def test_class_map_is_nested_and_complete():
    tex = V.table_classmap(MAPS)
    rows = [ln for ln in tex.splitlines() if re.match(r"^\d* & \d+ & ", ln) and "JetClass-II classes" not in ln]
    assert len(rows) == 43 and sum(bool(re.match(r"^\d+ & ", ln)) for ln in rows) == 17
    assert sum(ln.count("$") // 2 for ln in rows) == 188
    assert V.class_tex("X_YY_bbtauhtaue") == r"$bb \tau_h \tau_e$"


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
