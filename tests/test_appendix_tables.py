"""The appendix that defines every vocabulary: nothing the rerun trains on may be
missing from it, and nothing it says about a level may disagree with the map.

  * every partition map of the paper is printed, the flavour pair's included (its
    file is flavour_pair_map.v2.csv, which the old *label_map* glob never matched);
    the first grid's random partitions (v1) are not;
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


def test_the_papers_partitions_are_printed_flavour_controls_included():
    """The rerun's random partitions, numbered as in the results tables, then the flavor
    control label sets; the first grid's partitions (v1) are not part of the paper."""
    parts = A.partitions(MAPS, ROWS)
    cols = [c for _, c, _ in parts]
    with (MAPS / "flavour_pair_map.v2.csv").open() as f:
        flav = [c for c in next(csv.reader(f)) if c.startswith("FLAV") and not c.endswith("_name")]
    with (MAPS / "rand_label_map.v2.csv").open() as f:
        rand = [c for c in next(csv.reader(f)) if c.startswith("RAND") and not c.endswith("_name")]
    assert flav and rand and cols == rand + flav
    heads = [h for h, _, _ in parts]
    assert heads == [str(k) for k in range(1, len(rand) + 1)] + [A.FLAV_HEAD[c] for c in flav]
    table = A.table_vocabulary(ROWS, parts)
    assert "{random partition}" in table and "{flavor boundary}" in table
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
