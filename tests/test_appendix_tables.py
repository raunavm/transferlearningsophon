"""The appendix that defines every vocabulary: nothing the rerun trains on may be
missing from it, and nothing it says about a level may disagree with the map.

  * every partition map is printed, the flavour pair's included (its file is
    flavour_pair_map.v2.csv, which the old *label_map* glob never matched);
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


def test_every_partition_map_is_printed_flavour_pair_included():
    parts = A.partitions(MAPS, ROWS)
    cols = [c for _, c, _ in parts]
    with (MAPS / "flavour_pair_map.v2.csv").open() as f:
        flav = [c for c in next(csv.reader(f)) if c.startswith("FLAV") and not c.endswith("_name")]
    assert flav and set(flav) <= set(cols)
    heads = [h for h, _, _ in parts]
    # first-grid random partitions first, then the rerun's, then the flavour pair
    assert heads.index("1") < heads.index("R1") < heads.index("F0")
    table = A.table_vocabulary(ROWS, parts)
    assert "& F0 & F1" in table and "flavour pair" in table


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
    assert "visible prongs" in A.table_levels(ROWS, False)


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
