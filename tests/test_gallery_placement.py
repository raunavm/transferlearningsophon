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
    for name in ("gallery_labelset", "gallery_roc", "gallery_physics", "gallery_palette", "gallery_inventory", "gallery_compare"):
        for n in ast.walk(ast.parse((FIG / f"{name}.py").read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "Page":
                first = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "id"), None)
                if isinstance(first, ast.Constant):
                    ids.add(first.value)
    return ids


def test_the_gallery_order_is_the_placement_tier_by_tier():
    order = _assigned(FIG / "gallery_figures.py", "ORDER")
    assert list(order) == ["INVENTORY", *(k for k, _, _ in G.FIGURES), "SOPHON-FIG2", "PALETTE", *G.COMPARE]
    assert G.GALLERY_ONLY == ("INVENTORY", "SOPHON-FIG2", "PALETTE", *G.COMPARE)
    assert set(order) == _page_ids()
    for items in (G.FIGURES, G.TABLES):
        tiers = [G.TIERS.index(t) for _, t, _ in items]
        assert tiers == sorted(tiers)


def test_the_main_text_is_the_pis_list_in_order():
    main = [k for k, t, _ in G.FIGURES if t == G.MAIN]
    assert main == ["F1", "F2a", "F2b", "F3-main", "F4-mech", "F8", "A8", "F9b", "F11"]
    # PI 2026-11-12: one table beside the published models (tab:published) in place of the JetClass, top and
    # quark/gluon tables; the other label sets' fine-tuned rows in one appendix table (tab:labelsets), which also takes
    # in the Herwig table
    assert [k for k, t, _ in G.TABLES if t == G.MAIN] == ["tab:summary", "tab:published", "tab:mass", "tab:xbx"]
    assert [k for k, t, _ in G.TABLES if t == G.APPENDIX] == [
        "tab:probes", "tab:coarse", "tab:qcdhf", "tab:fill", "tab:inputs", "tab:partrepro", "tab:lrsweep", "tab:jetclassrej",
        "tab:labelsets", "tab:pairs", "tab:sculpt", "tab:sculptboot", "tab:xbxrej"]
    assert not {"tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig"} & set(G.TIER)
    # the fine-tuned read-out (2026-11-09) and X->bb/X->bs on Sophon's samples (2026-11-11) are in: nothing waits
    assert G.PROVISIONAL == {}
    assert set(G.PROVISIONAL) <= set(G.TIER) and all(v.isascii() for v in G.PROVISIONAL.values())    # the book's boxes
    # PI 2026-11-12: the appendix keeps these figures, everything else is in the supplement
    appendix = [k for k, t, _ in G.FIGURES if t == G.APPENDIX]
    assert appendix == ["F4-other", "F4-mech-ft", "PT", "F6", "A10", "F10b", "UMAP-jc2-had", "VAL"]
    assert G.TIER["tab:sculpt"] == G.APPENDIX and G.TIER["tab:compute"] == G.TIER["tab:massdetail"] == G.SUPPLEMENT
    # the bootstrap's table right after the sculpting table, and what the sculpting sentences rest on
    tables = [k for k, t, _ in G.TABLES if t == G.APPENDIX]
    assert tables[tables.index("tab:sculpt") + 1] == "tab:sculptboot"
    assert G.BACKS["tab:sculptboot"] == (f"The sculpting differences and their combined uncertainties that decide "
                                         f"{G.number('F10b')}'s tier.")
    for k in ("F10b", "F10b-flav", "F10c", "F10a-flav"):
        assert G.evidence(k) == (k, "tab:sculpt", "tab:sculptboot")
    # the trimmed appendix tables whole in the supplement; tab:boot nowhere
    for k in ("tab:probes", "tab:coarse", "tab:qcdhf", "tab:pairs", "tab:fill"):
        assert G.TIER[k] == G.APPENDIX and G.TIER[k + "full"] == G.SUPPLEMENT
    assert "tab:boot" not in G.TIER and "tab:boot" in G.DROPPED_TABLES


def test_every_item_has_one_tier_and_a_sentence():
    keys = [k for k, _, _ in G.FIGURES + G.TABLES]
    assert len(keys) == len(set(keys)) and set(G.TIER.values()) <= set(G.TIERS)
    assert all(s and s.endswith(".") for s in G.BACKS.values())


def test_the_dropped_pages_are_gone():
    order, paper = _assigned(FIG / "gallery_figures.py", "ORDER"), _assigned(FIG / "gallery_figures.py", "PAPER")
    assert not set(G.DROPPED) & (set(order) | set(paper) | set(G.TIER)) and set(paper) <= set(order)
    # the tables the plan places nowhere; its "missing inputs (fill)" and "fine-tuning vs size" are placed
    # and the task tables whose rows are now in tab:published and tab:labelsets (PI 2026-11-12)
    assert G.DROPPED_TABLES == ("tab:balance", "tab:fill_main", "tab:ftrej", "tab:perclass", "tab:probesn", "tab:boot",
                                "tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig")
    assert not set(G.DROPPED_TABLES) & set(G.TIER) and G.TIER["tab:fill"] == G.APPENDIX and G.TIER["tab:ft"] == G.SUPPLEMENT


def test_numbers_count_within_each_tier_and_kind():
    assert G.number("F1") == "Fig. 1" and G.number("F3-main") == "Fig. 4" and G.number("F4-other") == "Fig. A1"
    assert G.number("tab:summary") == "Table 1" and G.number("tab:probes") == "Table A1"
    assert G.number("FJ") == "Fig. S1" and G.number("tab:perrun") == "Table S1"
    assert G.place("F10b") == "Appendix, Fig. A6" and G.place("F11") == "Main text, Fig. 9"
    assert G.number("tab:sculptboot") == "Table A12" and G.number("tab:xbxrej") == "Table A13"
    # four main-text tables (PI 2026-11-12); tab:labelsets has the appendix number the Herwig table had
    assert [G.number(k) for k in ("tab:published", "tab:mass", "tab:xbx")] == ["Table 2", "Table 3", "Table 4"]
    assert G.number("tab:labelsets") == "Table A9" and G.number("tab:jetclassrej") == "Table A8" and G.number("tab:pairs") == "Table A10"


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
    assert G.BACKS["FJ"] == f"Cited beside {G.number('tab:jetclassrej')}: the per-class ROC curves its rejections come from."
    assert G.BACKS["F3-app"] == f"Cited from {G.number('F3-main')}: the other scatter panels."


def test_every_sentence_names_its_evidence():
    assert all(set(v) <= set(G.TIER) and k in G.TIER for k, v in G.EVIDENCE.items())
    assert G.evidence("F2a") == ("F2a", "tab:summary", "tab:probes") and G.evidence("F1") == ("F1",)
    # the fine-tuned values of Figs. 3 and A4 are in the comparison table and the label-set table (PI 2026-11-12)
    assert G.evidence("F2b") == ("F2b", "tab:summary", "tab:published", "tab:labelsets")
    assert G.evidence("F6") == ("F6", "tab:published", "tab:labelsets")
    # the binned mass figure's quantity has a table over all jets (tab:mass's head block), the bins in the supplement
    assert G.evidence("F9b") == ("F9b", "tab:mass", "tab:massdetail") and "[[F9b]]" in dict((k, s) for k, _, s in G._TABLES)["tab:mass"]
    # the two tables that rest on the comparison table: the 188-class and from-scratch rows, ParT's published row
    assert G.evidence("tab:labelsets") == ("tab:labelsets", "tab:published")
    assert G.evidence("tab:partrepro") == ("tab:partrepro", "tab:published")
    assert G.BACKS["CONF"] == f"Cited beside {G.number('tab:jetclassrej')}: JetClass confusion matrices."


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
    for key in ("tab:xbx", "tab:probes"):
        (tmp_path / "tables" / f"{book.TABLE_FILE[key]}.tex").write_text(tex % key)
    # nothing is provisional now (2026-11-11): the rule is tested with an entry of its own
    monkeypatch.setattr(book.G, "PROVISIONAL", {"tab:xbx": "waits for the samples"})
    out = book.table_item("tab:xbx", G.MAIN)
    assert "\\pending{no run yet}" in out and "\\label{tab:xbx}" in out and "fbox" not in out
    with pytest.raises(SystemExit, match="still has"):
        book.table_item("tab:probes", G.APPENDIX)


def test_the_book_boxes_only_a_provisional_figure_the_render_lacks(monkeypatch, tmp_path):
    book = _book(monkeypatch, tmp_path)
    monkeypatch.setattr(book, "preamble", lambda external=None: "")
    monkeypatch.setattr(book, "table_item", lambda key, tier: "")
    monkeypatch.setattr(book, "figure", lambda pid, rec, src: f"FIGURE {pid}\n")
    main = [k for k, t, _ in G.FIGURES if t == G.MAIN]
    monkeypatch.setattr(book.G, "PROVISIONAL", {"F11": "waits for the samples"})
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


def test_the_sculpting_sentence_names_exactly_the_differences_beyond_three_sigma():
    """PI 2026-11-09: the main text reports the sculpting differences that pass |delta| > 3 sigma and nothing else."""
    import json
    doc = json.loads((ROOT / "experiments/FIGS/data/v2/head/sculpting_boot.json").read_text())["differences"]
    passing = {(name, wp) for name, d in doc.items() for wp, v in d.items()
               if isinstance(v, dict) and v.get("distance", {}).get("pass")}
    assert passing == set(G.F10B_STATED)
    sentence = G.BACKS["F10b"]
    for words in ("17 classes", "1% QCD", "43 at 10%", "162 classes at 10%", "every other difference", "17 + mass",
                  "flavor boundaries"):
        assert words in sentence
    assert "sit with" not in sentence and not G.f10b_main(doc)


def test_the_scaling_figures_in_auc_sit_in_the_supplement_as_one_row():
    """PI 2026-11-11: the scaling figures need AUC too; 2026-11-12: in the supplement, one sentence."""
    sup = [k for k, t, _ in G.FIGURES if t == G.SUPPLEMENT]
    i = sup.index("F8-auc")
    assert sup[i:i + 3] == ["F8-auc", "A8-auc", "A10-auc"]
    assert len({s for k, _, s in G.FIGURES if k.endswith("-auc")}) == 1
