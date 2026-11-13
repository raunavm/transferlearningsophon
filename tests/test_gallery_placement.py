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


def _load_results():
    spec = importlib.util.spec_from_file_location("mlst_results_placement_t", ROOT / "experiments/FIGS/mlst_results.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _assigned(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def _umap_more() -> set:
    """The supplement's further UMAP grids (gallery_labelset.UMAP_MORE, round 3: the X->bb/X->bs grid and a mass-network
    grid per task), a dict built by a comprehension, so read from the module."""
    import sys
    sys.path[:0] = [str(FIG), str(FIG.parent)]
    import gallery_labelset
    return set(gallery_labelset.UMAP_MORE)


def _page_ids() -> set:
    """Every page id the gallery modules define: Page(id="...") or Page("...", ...), the per-class family's
    _page("...", ...), and the UMAP pages."""
    ids = set(_assigned(FIG / "gallery_labelset.py", "UMAP_PAGES")) | _umap_more()
    for name in ("gallery_labelset", "gallery_roc", "gallery_physics", "gallery_perclass", "gallery_palette",
                 "gallery_inventory", "gallery_compare"):
        for n in ast.walk(ast.parse((FIG / f"{name}.py").read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in ("Page", "_page"):
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
    # PI 2026-11-13 (L1, L9, L13): at most one figure a family in the main text, the slice with the approved story;
    # L20 (panels at least 4.5 cm) split the headline figure into one figure a mode, F2 (linear probes) and F2-ft
    # (fine-tuned), both in the main text (round 3, package 1b; whether L1's one figure may be two is the lead's)
    assert main == ["F1", "F2", "F2-ft", "PC-jc2", "F4-mech", "A8", "F11"]
    # PI 2026-11-12: one table beside the published models (tab:published) in place of the JetClass, top and
    # quark/gluon tables; the other label sets' fine-tuned rows in one appendix table (tab:labelsets), which also takes
    # in the Herwig table
    assert [k for k, t, _ in G.TABLES if t == G.MAIN] == ["tab:summary", "tab:published", "tab:mass", "tab:xbx"]
    # PI 2026-11-13 (L2, L13): the AUC of the headline family first; the training-size points and the mass-effect
    # summary in; the coarse targets and the b-tagging discriminant in the supplement (no sentence cites them)
    assert [k for k, t, _ in G.TABLES if t == G.APPENDIX] == [
        "tab:summaryauc", "tab:probes", "tab:probesn", "tab:fill", "tab:inputs", "tab:partrepro", "tab:lrsweep",
        "tab:jetclassrej", "tab:jc1_class_auc", "tab:labelsets", "tab:ft", "tab:pairs", "tab:sculpt", "tab:sculptboot",
        "tab:masseffect", "tab:xbxrej"]
    assert not {"tab:jetclass", "tab:benchtop", "tab:benchqg", "tab:benchqgherwig"} & set(G.TIER)
    # the fine-tuned read-out (2026-11-09) and X->bb/X->bs on Sophon's samples (2026-11-11) are in: nothing waits
    assert G.PROVISIONAL == {}
    assert set(G.PROVISIONAL) <= set(G.TIER) and all(v.isascii() for v in G.PROVISIONAL.values())    # the book's boxes
    # PI 2026-11-13: the appendix holds the rest of a family where it supports a main-text claim, everything else is
    # in the supplement
    appendix = [k for k, t, _ in G.FIGURES if t == G.APPENDIX]
    # L20 (two curve panels a row): the four other flavor pairs are F4-other and F4-light in each mode, and the ROC
    # figure is one mode a figure (F6, F6-ft)
    assert appendix == ["F2a", "F2b", "PC-jc1", "F4-mech-ft", "F4-other", "F4-light", "F4-other-ft", "F4-light-ft", "A10",
                        "F6", "F6-ft", "PT", "F9b", "F10b", "F11-part", "UMAP-jc2-had", "VAL"]
    assert G.TIER["tab:sculpt"] == G.APPENDIX and G.TIER["tab:compute"] == G.TIER["tab:massdetail"] == G.SUPPLEMENT
    # the bootstrap's table right after the sculpting table, and what the sculpting sentences rest on
    tables = [k for k, t, _ in G.TABLES if t == G.APPENDIX]
    assert tables[tables.index("tab:sculpt") + 1] == "tab:sculptboot"
    assert G.BACKS["tab:sculptboot"] == (f"{G.number('tab:sculptboot')} gives the sculpting differences behind "
                                         f"{G.number('F10b')} with their uncertainties over runs and test jets.")
    for k in ("F10b", "F10b-flav", "F10c", "F10a-flav"):
        assert G.evidence(k) == (k, "tab:sculpt", "tab:sculptboot")
    # the trimmed appendix tables whole in the supplement; the coarse targets and the b-tagging discriminant in the
    # supplement with their whole copies (L13); tab:boot nowhere
    for k in ("tab:probes", "tab:pairs", "tab:fill"):
        assert G.TIER[k] == G.APPENDIX and G.TIER[k + "full"] == G.SUPPLEMENT
    for k in ("tab:coarse", "tab:qcdhf"):
        assert G.TIER[k] == G.TIER[k + "full"] == G.SUPPLEMENT
    # restored to the supplement in round 3 (package 1), after the per-class tables, with their pooled random partitions
    assert [G.number(k) for k in ("tab:jc1conf", "tab:coarse", "tab:qcdhf")] == ["Table S11", "Table S12", "Table S13"]
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
    # (2026-11-13: tab:perclass restored to the supplement, tab:probesn and tab:ft to the appendix)
    assert G.DROPPED_TABLES == ("tab:balance", "tab:fill_main", "tab:ftrej", "tab:boot", "tab:jetclass", "tab:benchtop",
                                "tab:benchqg", "tab:benchqgherwig")
    assert not set(G.DROPPED_TABLES) & set(G.TIER) and G.TIER["tab:fill"] == G.APPENDIX and G.TIER["tab:ft"] == G.APPENDIX
    assert G.TIER["tab:perclass"] == G.TIER["tab:ftfull"] == G.TIER["tab:probesnfull"] == G.SUPPLEMENT
    # a page a family replaced (delete nothing, L13): dropped, and its successor placed
    assert set(G.SUPERSEDED) <= set(G.DROPPED) and set(G.SUPERSEDED.values()) <= set(G.TIER)
    assert G.SUPERSEDED["F3-main"] == "PC-jc2" and G.SUPERSEDED["CONF"] == "PC-conf" and G.SUPERSEDED["F6-app"] == "F6"


def test_numbers_count_within_each_tier_and_kind():
    # round 3 (L20 splits, the UMAP grids of package 1): F2-ft is Fig. 3, F4-light Fig. A6, F6-ft Fig. A11
    assert G.number("F1") == "Fig. 1" and G.number("F2-ft") == "Fig. 3" and G.number("PC-jc2") == "Fig. 4"
    assert G.number("F4-other") == "Fig. A5" and G.number("F4-light") == "Fig. A6" and G.number("F6-ft") == "Fig. A11"
    assert G.number("tab:summary") == "Table 1" and G.number("tab:summaryauc") == "Table A1" and G.number("tab:probes") == "Table A2"
    assert G.number("FJ") == "Fig. S23" and G.number("tab:perrun") == "Table S1"
    assert G.place("F10b") == "Appendix, Fig. A14" and G.place("F11") == "Main text, Fig. 7"
    assert G.number("tab:sculptboot") == "Table A14" and G.number("tab:masseffect") == "Table A15"
    assert G.number("tab:xbxrej") == "Table A16"
    # four main-text tables (PI 2026-11-12)
    assert [G.number(k) for k in ("tab:published", "tab:mass", "tab:xbx")] == ["Table 2", "Table 3", "Table 4"]
    assert G.number("tab:labelsets") == "Table A10" and G.number("tab:jetclassrej") == "Table A8" and G.number("tab:pairs") == "Table A12"
    # round 2 (2026-11-13): the jet-pT family (PT in the appendix, its partition and X->bb vs cc slices in the
    # supplement after the JetClass-II ROC curves) and the confusion percentages as a supplement table before the
    # coarse targets; round 3 (L18): the per-bin values are the data release's jet_pt_bins.csv, no table
    assert G.number("PT") == "Fig. A12" and G.number("F9b") == "Fig. A13"
    assert [G.number(k) for k in ("A3R-ft", "A3R-s", "PT-part", "PT-bbcc", "PT-bbcc-part", "F9a")] == [
        "Fig. S31", "Fig. S33", "Fig. S34", "Fig. S35", "Fig. S36", "Fig. S37"]
    assert [G.number(k) for k in ("tab:perclass", "tab:jc1conf", "tab:coarse", "tab:qcdhf")] == [
        "Table S10", "Table S11", "Table S12", "Table S13"]
    assert "tab:ptbins" not in G.TIER and G.number("tab:compute") == "Table S16"
    # the supplement's UMAP grids (round 3, package 1), then the pretraining curves of the partitions last
    sup = [k for k, t, _ in G.FIGURES if t == G.SUPPLEMENT]
    assert sup[-12:] == ["UMAP-jc1", "UMAP-jc2-lep", "UMAP-top", "UMAP-qg", "UMAP-xbx", "UMAP-mass-jc2-had", "UMAP-mass-jc1",
                         "UMAP-mass-jc2-lep", "UMAP-mass-top", "UMAP-mass-qg", "UMAP-mass-xbx", "VAL-part"]
    assert G.number("UMAP-xbx") == "Fig. S54" and G.number("VAL-part") == "Fig. S61" and len(sup) == 61


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
    # plain sentences that name their items (2026-11-13: no "Cited beside ...:" form, no colon)
    assert G.BACKS["FJ"] == f"{G.number('tab:jc1_class_auc')} gives the areas under the ROC curves of {G.number('FJ')}."
    assert G.BACKS["PC-conf"] == (f"The JetClass confusion matrices of every pretrained network are in {G.number('PC-conf')}, "
                                  f"{G.number('PC-conf-part')} and {G.number('PC-conf-mass')}.")
    assert not any(s.startswith("Cited") or ": " in s for s in G.BACKS.values())


def test_every_sentence_names_its_evidence():
    assert all(set(v) <= set(G.TIER) and k in G.TIER for k, v in G.EVIDENCE.items())
    # the headline family (L2): Table 1 holds the main figure's values, tab:summaryauc the AUC of the full versions
    assert G.evidence("F2") == ("F2", "tab:summary") and G.evidence("F2-ft") == ("F2-ft", "tab:summary")
    assert G.evidence("F1") == ("F1",)
    # F2a and F2b draw the AUC only (L20): the AUC table first
    assert G.evidence("F2a") == ("F2a", "tab:summaryauc", "tab:summary")
    assert G.evidence("F2b") == ("F2b", "tab:summaryauc", "tab:summary")
    # the ROC family: R30 and AUC in Table 1 and tab:summaryauc, accuracy and R50 in tab:probes and tab:labelsets
    assert G.evidence("F6") == ("F6", "tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets")
    assert G.evidence("F6-ft") == ("F6-ft", "tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets")
    # the binned mass figure's quantity has a table over all jets (tab:mass's head block), the bins in the supplement
    assert G.evidence("F9b") == ("F9b", "tab:mass", "tab:massdetail") and "[[F9b]]" in dict((k, s) for k, _, s in G._TABLES)["tab:mass"]
    # the two tables that rest on the comparison table: the 188-class and from-scratch rows, ParT's published row
    assert G.evidence("tab:labelsets") == ("tab:labelsets", "tab:summary", "tab:summaryauc")
    assert G.evidence("tab:partrepro") == ("tab:partrepro", "tab:published")
    # the per-class and training-size families point at their tables
    assert G.evidence("PC-jc2") == ("PC-jc2", "tab:summaryauc", "tab:perclass")
    assert G.evidence("A8") == ("A8", "tab:ft", "tab:summary") and G.evidence("A10") == ("A10", "tab:probesn", "tab:summary")
    # the jet-pT family: its values are the data release's jet_pt_bins.csv (L18), no table, so the figure is its own
    # evidence; the three supplement slices share one sentence
    assert G.evidence("PT") == ("PT",) and "tab:ptbins" not in G.TIER
    assert G.BACKS["PT-part"] == G.BACKS["PT-bbcc"] == G.BACKS["PT-bbcc-part"] == (
        f"{G.number('PT-part')} to {G.number('PT-bbcc-part')} repeat {G.number('PT')} for the partitions and X→bb̄ vs cc̄.")
    assert G.BACKS["tab:jc1conf"] == (f"The percentages of {G.number('PC-conf')} to {G.number('PC-conf-mass')} are in "
                                      f"{G.number('tab:jc1conf')}.")


def test_the_new_pages_are_marked_new():
    spec = importlib.util.spec_from_file_location("gallery_review", FIG / "gallery_review.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    for page in ("F2", "PC-jc2", "PC-jc1", "F4-other-ft", "F4-ee", "F4-rand", "F6-part", "A8-part", "A10-part", "F10-ft",
                 "F11-part", "F11-probe", "A17-ridge", "F10-ft-part", "VAL-part"):
        assert review.CHANGES[page].startswith("NEW: "), page
    for page in ("F2a", "F2b", "PC-conf", "F4-mech", "F4-mech-ft", "F4-other", "F6", "A8", "A10", "F10b-flav", "F11", "VAL"):
        assert review.CHANGES[page].startswith("CHANGED: "), page
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
    pages[1]["drawn"] = False                                         # F2: not provisional
    (tmp_path / "pages.json").write_text(json.dumps(pages))
    with pytest.raises(SystemExit, match="F2 was not drawn"):
        book.build(tmp_path, "main")


def test_the_sculpting_sentence_names_exactly_the_differences_beyond_three_sigma():
    """PI 2026-11-09, 2026-11-13 (L3): the text reports a sculpting difference only where |delta| > 3 sigma with
    sigma^2 the SD^2 over the runs of each network plus the bootstrap SD^2 over test jets (the paired-by-run verdict
    of the bootstrap file is never the sole basis). Recomputed from the per-run distances (sculpting_v3) over every run
    of each side, with the bootstrap SD of bootstrap/l3/sculpting_boot_l3.json ("l3" block, Table A14): no difference
    passes, so F10b's sentence states none and quotes no closest value (round 2, package 3: the "closest are ..."
    clause is gone)."""
    import math
    import statistics
    M = _load_results()
    data = ROOT / "experiments/FIGS/data/v2"
    boot, v3 = data / M.SCULPT_BOOT_L3, data / "head/sculpting_v3"
    if not boot.exists() or not v3.is_dir():
        pytest.skip("the sculpting files are not pulled")
    doc = json.loads(boot.read_text())["differences"]
    sculpt = {k.removesuffix("_sculpting"): v for k, v in M._load(v3).items()}
    runs = M._paper(ROOT)
    arm = {m.removeprefix("mtx-").rsplit("-s", 1)[0]: a for a, ms in runs.items() for m in ms}
    z = {}
    for name, d in doc.items():
        for wp, _ in M.SCULPT_BOOT_WPS:
            x = d["l3"][wp]["distance"]
            # every run of each side, not only the seed pairs
            assert x["runs_config"] == runs[arm[d["configuration"]]] and x["runs_reference"] == runs[arm[d["reference"]]]
            a, b = ([M._sculpt_distance(sculpt, m, wp) for m in x[side]] for side in ("runs_config", "runs_reference"))
            sigma = math.sqrt(statistics.stdev(a) ** 2 + statistics.stdev(b) ** 2 + x["boot_sd"] ** 2)
            z[name, wp] = (statistics.mean(a) - statistics.mean(b)) / sigma
            assert z[name, wp] == pytest.approx(x["z"], rel=1e-9) and x["pass"] == (abs(z[name, wp]) > 3)
    assert {k for k, v in z.items() if abs(v) > 3} == set(G.F10B_STATED) == set()
    assert not G.f10b_main({name: d["l3"] for name, d in doc.items()})
    # the closest stay below three standard deviations (the placement's comment quotes them)
    assert max(abs(v) for v in z.values()) == abs(z["162mass-162", "eps_b=0.1"]) < 3
    assert f"{abs(z['162mass-162', 'eps_b=0.1']):.2f}" == "2.76" and f"{z['17-188', 'eps_b=0.01']:.2f}" == "2.55"
    assert G.BACKS["F10b"] == ("Used directly, no label set or mass network sculpts the QCD mass differently from 188 "
                               "classes, or a mass network from its own label set, by more than three standard "
                               "deviations at 10% or 1% QCD efficiency or 50% signal efficiency.")
    assert not re.search(r"\d\.\d", G.BACKS["F10b"]) and "closest" not in G.BACKS["F10b"]


def test_the_scaling_figures_in_auc_sit_in_the_supplement_as_one_row():
    """PI 2026-11-11: the scaling figures need AUC too; 2026-11-12: in the supplement, one sentence."""
    sup = [k for k, t, _ in G.FIGURES if t == G.SUPPLEMENT]
    i = sup.index("F8-auc")
    assert sup[i:i + 3] == ["F8-auc", "A8-auc", "A10-auc"]
    assert len({s for k, _, s in G.FIGURES if k.endswith("-auc")}) == 1


def test_the_round_two_sentences_claim_only_what_passes_the_three_sigma_rule():
    """Round 2 (2026-11-13, L3): the sentences rewritten on the L3 arithmetic that gallery_placement.py's comments
    record: the jet-pT finding (probes on JetClass-II, at least 9.5 sigma in every bin; fine-tuned 17 classes in every
    JetClass-II bin and the two lowest JetClass bins), the comparison with ParT (JetClass accuracy above ParT trained
    from scratch; top and quark/gluon R30 consistent with it, no longer "between" it and ParT fine-tuned), the
    missing-input fills (the b-quark boundary's quark/gluon fills, 3.2-3.6 sigma on two runs, not claimed) and the
    single rescaled reproduction run (2.7 sigma, not a claim)."""
    assert G.BACKS["PT"] == ("Per jet-pT bin, 43 and 17 classes trail 188 on JetClass-II probes; fine-tuned, 17 classes "
                             "trail in every JetClass-II bin and the two lowest JetClass bins.")
    pub = G.BACKS["tab:published"]
    assert "consistent with ParT trained from scratch" in pub and "between" not in pub
    assert G.BACKS["tab:fill"].startswith("Random draws for the missing top inputs beat zeros for every pretrained network;")
    assert "but with the b-quark boundary, are consistent with zeros" in G.BACKS["tab:fill"]
    assert G.BACKS["tab:partrepro"].endswith("a single run with Sophon's jet-pT rescaling is 28% lower, within three "
                                             "standard deviations.")
