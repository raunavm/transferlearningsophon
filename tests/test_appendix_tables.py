"""The appendix that defines every vocabulary: nothing the rerun trains on may be
missing from it, and nothing it says about a level may disagree with the map.

  * every partition map of the paper is read, the flavour pair's included (its
    file is flavour_pair_map.v2.csv, which the old *label_map* glob never matched);
    the first grid's random partitions (v1) are not; the class map prints none of them
    (PI 2026-11-23, LEDGER SP-13 step 1), and the build checks the prong mixing that
    methods.tex states of them;
  * every classification arm of configs/arms/v2_grid.json trains on a tree level
    or on a printed partition, or the build stops;
  * each level's description is checked against the group names of the label
    map, and the 4-class level is described from them: it counts VISIBLE prongs.
"""
import csv
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MAPS = REPO / "configs" / "labelmaps"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod("appendix_tables", "experiments/FIGS/appendix_tables.py")
ROWS = A.read_map(MAPS / "rung_label_maps.v1.csv")


def test_the_papers_partitions_are_read_flavour_controls_included_and_not_printed():
    """The rerun's random partitions, numbered as in the results tables, then the flavor
    control label sets; the first grid's partitions (v1) are not part of the paper. PI 2026-11-23 (LEDGER SP-13, step 1):
    at one row per 43-class group the partitions' per-class maps no longer fit the class map; by R4 they go to the
    release (the code release's label maps), and the class map prints no partition column."""
    parts = A.partitions(MAPS, ROWS)
    cols = [c for _, c, _ in parts]
    with (MAPS / "flavour_pair_map.v2.csv").open() as f:
        flav = [c for c in next(csv.reader(f)) if c.startswith("FLAV") and not c.endswith("_name")]
    with (MAPS / "rand_label_map.v2.csv").open() as f:
        rand = [c for c in next(csv.reader(f)) if c.startswith("RAND") and not c.endswith("_name")]
    assert flav and rand and cols == rand + flav
    heads = [h for h, _, _ in parts]
    assert heads == [str(k) for k in range(1, len(rand) + 1)] + [A.FLAV_HEAD[c] for c in flav]
    table = A.table_vocabulary(ROWS)
    assert "random partition" not in table and "flavor boundary" not in table
    assert "RAND_d" not in table and "rerun" not in table


def test_every_vocabulary_the_rerun_trains_is_defined_in_the_appendix(tmp_path):
    parts = A.partitions(MAPS, ROWS)
    A.check_grid_covered(REPO / "configs" / "arms" / "v2_grid.json", parts)
    grid = json.loads((REPO / "configs" / "arms" / "v2_grid.json").read_text())
    grid["arms"].append({"name": "RAND3_p1", "config": "configs/arms/v3/RAND3_p1.yaml",
                         "num_classes": 17})
    g = tmp_path / "v2_grid.json"
    g.write_text(json.dumps(grid))
    with pytest.raises(SystemExit, match="RAND3_p1"):
        A.check_grid_covered(g, parts)


def test_the_four_class_level_is_described_by_visible_prongs_from_its_group_names():
    rules = A.level_rules(ROWS)
    assert rules["R3_VIS"].startswith("two, three or four visible prongs")
    # the reason it says "visible": a four-body decay with a neutrino sits with the
    # three-prong decays, so the visible-content pair merges only at 2 classes
    vis = {r["class_name"]: r["R3_VIS_name"] for r in ROWS}
    assert vis["label_X_YY_cqtauhv"] == "3P_VIS" and vis["label_X_YY_bbqq"] == "4P_VIS"
    levels = A.table_levels(ROWS, False)
    assert "visible prongs" in levels
    # the 64- and 30-class levels were cut from the grid: not pretrained in the paper
    assert "rerun" not in levels and "64 & each decay topology by its numbers of $b$ and $c$ quarks & no" in levels


def test_a_level_description_that_no_longer_matches_the_map_stops_the_build():
    rows = [dict(r) for r in ROWS]
    for r in rows:
        if "|" in r["R42_Q1_name"]:
            r["R42_Q1_name"] = r["R42_Q1_name"].split("|")[0] + "|HF"   # the 30-class tag
    with pytest.raises(SystemExit, match="R42_Q1"):
        A.level_rules(rows)
    rows = [dict(r) for r in ROWS]
    for r in rows:
        r["R3_VIS_name"] = r["R3_VIS_name"].replace("_VIS", "_PRONG")
    with pytest.raises(SystemExit, match="visible-content groups"):
        A.level_rules(rows)


def _expand(row: str) -> list:
    """A class-map row's classes in full, the decay chain written once put back on each."""
    row = row.removesuffix(" \\\\")
    if row.startswith("QCD, "):
        return ["QCD, " + c for c in row[len("QCD, "):].split(", ")]
    first, *rest = row.split(", ")
    chain = first[:first.rindex("\\to ") + 4]
    return [first] + [chain + c[1:] for c in rest]


def test_the_class_map_has_one_row_per_43_class_group_in_blocks_of_the_17_class_groups():
    """PI 2026-11-23, LEDGER SP-9 (merge 3): tab:app-levels folds into the class map, the 17 group names as block
    headings (with their numbers and sizes), the levels never pretrained (64, 30, 4 and 2 classes) out, and its caption
    names no table that left the paper (tab:pairs, tab:app-levels) and no section. PI 2026-11-23, LEDGER SP-13 (step 1,
    the 19 Nov item): each 43-class group a row listing its JetClass-II classes, its 17-class group the block heading;
    headings without a colon (CA-10)."""
    table = A.table_vocabulary(ROWS)
    assert A.ROW_LEVEL == "R42_Q1" and "43 classes & JetClass-II classes" in table
    groups, rows43 = {}, {}
    for r in ROWS:
        groups.setdefault(r["R16_Q1"], []).append(r)
        rows43.setdefault(r["R42_Q1"], []).append(r["class_name"])
    body = table.split("\\endlastfoot")[1].splitlines()
    heads = [ln for ln in body if ln.startswith("\\multicolumn{")]
    assert len(heads) == len(groups) == 17 and not any(":" in h for h in heads)
    for gid, members in groups.items():
        n = len(members)
        assert any(f"17-class group {gid}, {A.group_tex(members[0]['R16_Q1_name'])} "
                   f"({n} {'class' if n == 1 else 'classes'})" in h for h in heads), gid
    assert A.group_tex("4P_SEMILEP_NU_2PARTON_L") == r"four-prong decays to two quarks or gluons, $\ell$ and $\nu$"
    head = table.split("\\endfirsthead")[0]
    assert not any(f"{k} classes" in head for k in ("64", "30", "4 ", "2 ")) and "level}" not in head
    assert "tab:pairs" not in table and "tab:app-levels" not in table and "\\ref{sec:" not in table
    # one row per 43-class group, each holding its classes, every class once
    rows = {ln.split(" & ")[0]: ln.split(" & ", 1)[1] for ln in body if " & " in ln and not ln.startswith("\\multicolumn")}
    assert sorted(rows, key=int) == sorted(rows43, key=int) and len(rows) == 43
    for g, names in rows43.items():
        assert sorted(_expand(rows[g])) == sorted(A.native_tex(n) for n in names), g
    assert rows["0"].startswith(r"$X\to bb$, $bc$, $bq$")
    assert rows["42"].startswith("QCD, $bbccss$, ") and rows["42"].endswith(", light \\\\")
    assert r"$X\to YY\to bbbb$, $bbcc$" in rows["21"]


def test_a_partition_that_keeps_the_prong_boundaries_stops_the_build():
    """methods.tex says the partitions cut across the 17-class groups and mix decays with different numbers of prongs;
    with no partition column printed, the build checks it where the maps are read (RR-7)."""
    parts = A.partitions(MAPS, ROWS)
    A.check_partitions_mix_prongs(ROWS, parts)
    by_prong = {r["class_name"]: r["R16_Q1"] for r in ROWS}            # the 17-class grouping keeps every prong count apart
    with pytest.raises(SystemExit, match="RAND_x"):
        A.check_partitions_mix_prongs(ROWS, parts + [("x", "RAND_x", by_prong)])
