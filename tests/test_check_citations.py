"""experiments/FIGS/check_citations.py --gallery: every generated table and every gallery page is placed or dropped,
or the check exits 1 (PI 2026-11-08: anything nothing cites is dropped)."""
import importlib.util
import pathlib
import shutil
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments/FIGS/figstyle"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CC = _load("check_citations", ROOT / "experiments/FIGS/check_citations.py")
BOOK = CC._book()


def _where(tmp_path, order) -> pathlib.Path:
    """A figstyle directory with the real placement and a gallery_figures.py holding only ORDER."""
    shutil.copy(FIG / "gallery_placement.py", tmp_path / "gallery_placement.py")
    (tmp_path / "gallery_figures.py").write_text(f"ORDER = {tuple(order)!r}\n")
    return tmp_path


def _book(extra_captions=None):
    captions = {**BOOK.M.CAPTIONS, **(extra_captions or {})}
    return types.SimpleNamespace(TIERS=BOOK.TIERS, PREFIX=BOOK.PREFIX, check_spec=BOOK.check_spec,
                                 M=types.SimpleNamespace(CAPTIONS=captions))


def test_the_placement_holds_every_table_and_page():
    assert CC.gallery() == 0
    g = CC.placement()
    assert set(CC.order()) <= set(g.TIER) | set(g.DROPPED) | set(g.GALLERY_ONLY)


def test_a_table_neither_placed_nor_dropped_fails(tmp_path, capsys):
    where = _where(tmp_path, CC.order())
    assert CC.gallery(where, _book()) == 0
    assert CC.gallery(where, _book({"new": ("tab:new", "A table nothing places.")})) == 1
    assert "1 generated table(s) neither placed nor dropped: tab:new" in capsys.readouterr().out


def test_a_page_neither_placed_nor_dropped_fails(tmp_path, capsys):
    # a dropped figure or a gallery-only page in ORDER is fine
    assert CC.gallery(_where(tmp_path, (*CC.order(), "F3")), _book()) == 0
    assert CC.gallery(_where(tmp_path, (*CC.order(), "F99")), _book()) == 1
    assert "1 gallery page(s) neither placed nor dropped: F99" in capsys.readouterr().out
