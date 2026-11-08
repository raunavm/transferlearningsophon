"""The paper's placement (experiments/FIGS/figstyle/gallery_placement.py, PI 2026-11-08): the gallery's ORDER follows it
tier by tier, every page has a place and a sentence, the dropped pages are gone, and the new pages are marked NEW."""
import ast
import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments/FIGS/figstyle"
spec = importlib.util.spec_from_file_location("gallery_placement", FIG / "gallery_placement.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def _assigned(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def _page_ids() -> set:
    """Every page id the gallery modules define: Page(id="...") or Page("...", ...), and the UMAP pages."""
    ids = set(_assigned(FIG / "gallery_labelset.py", "UMAP_PAGES"))
    for name in ("gallery_labelset", "gallery_roc", "gallery_physics", "gallery_palette", "gallery_inventory"):
        for n in ast.walk(ast.parse((FIG / f"{name}.py").read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "Page":
                first = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "id"), None)
                if isinstance(first, ast.Constant):
                    ids.add(first.value)
    return ids


def test_the_gallery_order_is_the_placement_tier_by_tier():
    order = _assigned(FIG / "gallery_figures.py", "ORDER")
    assert list(order) == ["INVENTORY", *(k for k, _, _ in G.FIGURES), "SOPHON-FIG2", "PALETTE"]
    assert G.GALLERY_ONLY == ("INVENTORY", "SOPHON-FIG2", "PALETTE")
    assert set(order) == _page_ids()
    for items in (G.FIGURES, G.TABLES):
        tiers = [G.TIERS.index(t) for _, t, _ in items]
        assert tiers == sorted(tiers)


def test_the_main_text_is_the_pis_list_in_order():
    main = [k for k, t, _ in G.FIGURES if t == G.MAIN]
    assert main == ["F1", "F2a", "F2b", "F3-main", "F4-mech", "F8", "A8", "F9b", "F11"]
    assert [k for k, t, _ in G.TABLES if t == G.MAIN] == ["tab:jetclass", "tab:benchtop + tab:benchqg", "tab:mass",
                                                          "tab:xbx", "tab:xbxrej"]
    ft = "waits for the fine-tuned read-out"
    assert G.PROVISIONAL == {"F10b": "main text if at 1% QCD efficiency 17 classes sculpt more than 188 and 17 + mass "
                                     "less than 17, each by more than 3 sigma",
                             "F11": "waits for the samples", "F4-mech-ft": ft, "F10-ft": ft,
                             "tab:xbx": "waits for the samples", "tab:xbxrej": "waits for the samples",
                             "tab:pairs": "its fine-tuned columns wait for the fine-tuned read-out",
                             "tab:coarse": "a cell waits for its run"}
    assert set(G.PROVISIONAL) <= set(G.TIER) and all(v.isascii() for v in G.PROVISIONAL.values())    # the book's boxes
    # F10b in the appendix between F10a and F10c, F10b-flav beside it, until the bootstrap decides
    appendix = [k for k, t, _ in G.FIGURES if t == G.APPENDIX]
    assert appendix[appendix.index("F10a"):appendix.index("F10c") + 1] == ["F10a", "F10b", "F10b-flav", "F10c"]
    assert G.TIER["tab:sculpt"] == G.TIER["tab:compute"] == G.APPENDIX and G.TIER["tab:massdetail"] == G.SUPPLEMENT
    # the bootstrap's table right after the sculpting table, and what the sculpting sentences rest on
    tables = [k for k, t, _ in G.TABLES if t == G.APPENDIX]
    assert tables[tables.index("tab:sculpt") + 1] == "tab:sculptboot"
    assert G.BACKS["tab:sculptboot"] == (f"The sculpting differences and their combined uncertainties that decide "
                                         f"{G.number('F10b')}'s tier.")
    for k in ("F10b", "F10b-flav", "F10a-flav"):
        assert G.evidence(k) == (k, "tab:sculpt", "tab:sculptboot")


def test_every_item_has_one_tier_and_a_sentence():
    keys = [k for k, _, _ in G.FIGURES + G.TABLES]
    assert len(keys) == len(set(keys)) and set(G.TIER.values()) <= set(G.TIERS)
    assert all(s and s.endswith(".") for s in G.BACKS.values())


def test_the_dropped_pages_are_gone():
    order, paper = _assigned(FIG / "gallery_figures.py", "ORDER"), _assigned(FIG / "gallery_figures.py", "PAPER")
    assert not set(G.DROPPED) & (set(order) | set(paper) | set(G.TIER)) and set(paper) <= set(order)
    # the tables the plan places nowhere; its "missing inputs (fill)" and "fine-tuning vs size" are placed
    assert G.DROPPED_TABLES == ("tab:balance", "tab:fill_main", "tab:ftrej", "tab:perclass", "tab:probesn")
    assert not set(G.DROPPED_TABLES) & set(G.TIER) and G.TIER["tab:fill"] == G.TIER["tab:ft"] == G.APPENDIX


def test_numbers_count_within_each_tier_and_kind():
    assert G.number("F1") == "Fig. 1" and G.number("F3-main") == "Fig. 4" and G.number("FJ") == "Fig. A1"
    assert G.number("tab:jetclass") == "Table 1" and G.number("tab:probes") == "Table A1"
    assert G.number("FJ-ctrl") == "Fig. S1" and G.number("tab:perrun") == "Table S1"
    assert G.place("F10b") == f"Appendix, Fig. A10; {G.PROVISIONAL['F10b']}"
    assert G.number("tab:sculptboot") == "Table A14" and G.number("tab:boot") == "Table A15"


def test_f10b_goes_to_the_main_text_only_with_the_signs_of_the_rule():
    """17 - 188 > 0 and 17 + mass - 17 < 0 at 1% QCD efficiency, each beyond 3 sigma; |delta| > 3 sigma of the
    wrong sign, or at another working point, is not enough."""
    def doc(a, b, sigma=0.003, wp="eps_b=0.01"):
        return {"17-188": {wp: {"distance": {"delta": a, "sigma": sigma}}},
                "17mass-17": {wp: {"distance": {"delta": b, "sigma": sigma}}}}
    assert G.F10B_RULE == (("17-188", 1), ("17mass-17", -1)) and G.F10B_WP == "eps_b=0.01"
    assert G.f10b_main(doc(0.0122, -0.0122))
    assert not G.f10b_main(doc(0.0122, 0.0122)) and not G.f10b_main(doc(-0.0122, -0.0122))
    assert not G.f10b_main(doc(0.0122, -0.0085))                      # 2.8 sigma
    assert not G.f10b_main(doc(0.0122, -0.0122, sigma=None)) and not G.f10b_main({})
    assert not G.f10b_main(doc(0.0122, -0.0122, wp="eps_b=0.1"))


def test_every_number_in_a_sentence_comes_from_number():
    """No sentence writes a figure or table number by hand: each names the item as [[key]], filled by number()."""
    raw = {k: s for k, _, s in G._FIGURES + G._TABLES}
    for key, sentence in raw.items():
        assert not re.search(r"\b(Fig\.|Table)\s*[AS]?\d", sentence), key
        refs = G.REF.findall(sentence)
        assert set(refs) <= set(G.TIER), key
        filled = G.BACKS[key]
        assert "[[" not in filled and re.findall(r"(?:Fig\.|Table) [AS]?\d+", filled) == [G.number(r) for r in refs]
    assert G.BACKS["FJ"].startswith(G.number("tab:jetclass")) and G.number("tab:jetclass") == "Table 1"
    assert G.BACKS["F3-app"] == f"Cited from {G.number('F3-main')}: the other scatter panels."


def test_every_sentence_names_its_evidence():
    assert all(set(v) <= set(G.TIER) and k in G.TIER for k, v in G.EVIDENCE.items())
    assert G.evidence("F2a") == ("F2a", "tab:probes") and G.evidence("F1") == ("F1",)


def test_the_new_pages_are_marked_new():
    spec = importlib.util.spec_from_file_location("gallery_review", FIG / "gallery_review.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    for page in ("F3-main", "F4-mech", "F4-mech-ft", "F4-other", "F10b-flav", "F10-ft"):
        assert review.CHANGES[page].startswith("NEW: ")
    assert not set(G.DROPPED) & set(review.CHANGES) and not set(G.DROPPED) & set(review.REVIEW)


def _book(monkeypatch, tmp_path):
    """make_results_book with its paper directory in tmp_path (tables/, main.tex) and its figure copies there."""
    spec = importlib.util.spec_from_file_location("make_results_book", ROOT / "experiments/FIGS/make_results_book.py")
    book = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(book)
    monkeypatch.setattr(book, "PAPER", tmp_path)
    monkeypatch.setattr(book, "FIGS", tmp_path / "figs")
    (tmp_path / "tables").mkdir()
    return book


def test_the_book_sets_pending_cells_only_in_a_provisional_table(monkeypatch, tmp_path):
    book = _book(monkeypatch, tmp_path)
    tex = "\\begin{table}[t]\n\\caption{A.}\\label{%s}\n1 & \\pending{no run yet} \\\\\n\\end{table}\n"
    for key in ("tab:pairs", "tab:probes"):
        (tmp_path / "tables" / f"{book.TABLE_FILE[key]}.tex").write_text(tex % key)
    assert "tab:pairs" in G.PROVISIONAL and "tab:probes" not in G.PROVISIONAL
    out = book.table_item("tab:pairs", G.APPENDIX)
    assert "\\pending{no run yet}" in out and "\\label{tab:pairs}" in out and "fbox" not in out
    with pytest.raises(SystemExit, match="still has"):
        book.table_item("tab:probes", G.APPENDIX)


def test_the_book_boxes_only_a_provisional_figure_the_render_lacks(monkeypatch, tmp_path):
    book = _book(monkeypatch, tmp_path)
    monkeypatch.setattr(book, "preamble", lambda external=None: "")
    monkeypatch.setattr(book, "table_item", lambda key, tier: "")
    monkeypatch.setattr(book, "figure", lambda pid, rec, src: f"FIGURE {pid}\n")
    main = [k for k, t, _ in G.FIGURES if t == G.MAIN]
    pages = [{"id": k, "drawn": k != "F11"} for k, _, _ in G.FIGURES]
    (tmp_path / "pages.json").write_text(json.dumps(pages))
    monkeypatch.setattr(book, "OUTPUTS", {**book.OUTPUTS, "main": ("main_only", (G.MAIN,))})
    book.build(tmp_path, "main")
    doc = (tmp_path / "main_only.tex").read_text()
    assert "\\label{fig:F11}" in doc and "fbox" in doc and all(f"FIGURE {k}\n" in doc for k in main if k != "F11")
    pages[1]["drawn"] = False                                         # F2a: not provisional
    (tmp_path / "pages.json").write_text(json.dumps(pages))
    with pytest.raises(SystemExit, match="F2a was not drawn"):
        book.build(tmp_path, "main")
