"""The manuscript's figures and tables are exactly the placement's main-text and appendix lists (PI 2026-11-22T23:11:36Z,
LEDGER SP-1): experiments/FIGS/manuscript_items.py reads main.tex's \\paperfig and \\input{tables/...} lists."""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "FIGS"))
import manuscript_items as MI  # noqa: E402


def test_every_manuscript_item_maps_to_a_gallery_id_or_label():
    have = MI.manuscript()
    assert not [k for ks in have.values() for k in ks if k.startswith("UNMAPPED")]


def test_manuscript_holds_exactly_the_placement():
    assert MI.mismatch() == []


def _journal(tmp_path, drop=()):
    """A manuscript holding exactly the placement's main-text and appendix items (less drop): main.tex maps each
    figure to its gallery file and inputs one section a tier, each \\paperfig and \\input{tables/...} in it."""
    want, files = MI.placement(), MI.B.TABLE_FILE
    (tmp_path / "sections").mkdir(parents=True)
    (tmp_path / "tables").mkdir()
    head = []
    for tier, name in ((MI.G.MAIN, "body"), (MI.G.APPENDIX, "app")):
        lines = []
        for k in want[tier]:
            if k in drop:
                continue
            if k.startswith("tab:"):
                lines.append(f"\\input{{tables/{files[k]}}}")
            else:
                head.append(f"\\setfigfile{{{k}}}{{gallery/{k}}}")
                lines.append(f"\\paperfig{{\\linewidth}}{{{k}}}")
        (tmp_path / "sections" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    (tmp_path / "main.tex").write_text("\n".join(head + ["\\input{sections/body}", "\\appendix", "\\input{sections/app}"]) + "\n")
    return tmp_path


def _make_tables():
    spec = importlib.util.spec_from_file_location("make_tables_stop_t", pathlib.Path(MI.HERE) / "make_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_paper_build_stops_on_every_mismatch_line_and_passes_the_placement(tmp_path):
    """PI 2026-11-23, LEDGER SP-12 (step 1, the placement build stop): make_tables --paper stops (SystemExit) with every
    line of manuscript_items.mismatch() when main.tex's \\paperfig and \\input lists are not the placement's MAIN and
    APPENDIX lists, and goes on when they are; the results book's build does not pass --paper."""
    import pytest
    MT = _make_tables()
    ok = _journal(tmp_path / "ok")
    assert MI.mismatch(ok) == []
    MT.placement_stop(ok)                                    # no stop: the manuscript is the placement
    bad = _journal(tmp_path / "bad", drop=("F4-mech", "tab:tests"))
    lines = MI.mismatch(bad)
    assert lines == ["main: F4-mech is placed here but not in the manuscript",
                     "appendix: tab:tests is placed here but not in the manuscript"]
    with pytest.raises(SystemExit) as e:
        MT.placement_stop(bad)
    assert str(e.value).startswith(MT.PLACEMENT_STOP) and all(ln in str(e.value) for ln in lines)
    # the render job: the paper's table build passes --paper and nothing is compiled after the stop; the book's does not
    spec = importlib.util.spec_from_file_location("build_render_job_stop_t", MI.HERE.parents[1] / "scripts/build_render_job.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    draw = R.PAPER["draw"]
    assert "python3 experiments/FIGS/make_tables.py --paper > ${OUT}/tables.txt 2>&1" in draw
    stop = draw.index('grep -q "^FATAL: the manuscript\'s figures and tables"')
    assert draw.index("--paper") < stop < draw.index("paper_figures.py") and "exit 3" in draw[stop:draw.index("paper_figures.py")]
    assert MT.PLACEMENT_STOP.startswith("FATAL: the manuscript's figures and tables")
    assert "--paper" not in R.BOOK["draw"] and "--paper" not in R.RELEASE["draw"]
