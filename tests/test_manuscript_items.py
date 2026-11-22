"""The manuscript's figures and tables are exactly the placement's main-text and appendix lists (PI 2026-11-22T23:11:36Z,
LEDGER SP-1): experiments/FIGS/manuscript_items.py reads main.tex's \\paperfig and \\input{tables/...} lists."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "FIGS"))
import manuscript_items as MI  # noqa: E402


def test_every_manuscript_item_maps_to_a_gallery_id_or_label():
    have = MI.manuscript()
    assert not [k for ks in have.values() for k in ks if k.startswith("UNMAPPED")]


def test_manuscript_holds_exactly_the_placement():
    assert MI.mismatch() == []
