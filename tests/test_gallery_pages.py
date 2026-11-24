"""The gallery's map and the pages drawn from shared parts (experiments/FIGS/figstyle): the map's rows and blocks
(gallery_inventory), the class-pair family's panels and legend (gallery_roc._pair_grid, _roc_grid), the per-class
family's one axis range and colours (gallery_perclass._draw_grid), the line and colour rule (style_v5, L6), the
figure-text rules (L7), the one row order of figure families and tables (L5) and Table 1 against the headline
family's value functions (L2). Imports the gallery modules, which read the committed figure data (about a second)."""
import collections
import inspect
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments/FIGS/figstyle"), str(ROOT / "experiments/FIGS")]
import gallery_inventory as V  # noqa: E402
import gallery_labelset as L  # noqa: E402
import gallery_perclass as PC  # noqa: E402
import gallery_physics as H  # noqa: E402
import gallery_placement as G  # noqa: E402
import gallery_roc as R  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402


def test_rows_merge_consecutive_items_of_one_sentence_and_carry_their_evidence():
    supp = [i for i in G.FIGURES if i[2] == G.BACKS["FJ-ctrl"]]
    # PI 2026-11-18: the class-against-rest JetClass curves, FJ-app and FJ first, then their slices, one sentence
    span = f"{G.number('FJ-app')}–{G.number('FJ-s').split()[-1]}"
    assert int(G.number("FJ").split("S")[-1]) == int(G.number("FJ-app").split("S")[-1]) + 1 and V._rows(tuple(supp)) == [
        (span, "FJ-app, FJ, FJ-ctrl-app, FJ-mass-app, FJ-ctrl, FJ-mass, FJ-s", G.BACKS["FJ-ctrl"])]
    # with the tables it also rests on
    flav = tuple(i for i in G.FIGURES if i[0] in ("F10b-flav", "F10c", "F10a-flav"))
    rows = V._rows(flav)
    # PI 2026-11-23, LEDGER SP-9: tab:sculptboot is the tested-differences table's
    also = f" [also {G.number('tab:sculpt')}, {G.number('tab:tests')}]"
    # one sentence for the supplement's sculpting figures (PI 2026-11-12 cut): consecutive, so one row
    assert [r[0] for r in rows] == [f"{G.number('F10b-flav')}–{G.number('F10a-flav').split()[-1]}"]
    assert rows[0][2].endswith(also)
    assert V._rows(tuple(i for i in G.FIGURES if i[0] == "F10b"))[0][1] == "F10b"


def test_the_map_places_every_table_once_and_lists_the_rest():
    placed = [k for t, *_ in G.TABLES for k in t.split(" + ")]
    labels = {label for label, *_ in V.C.M.CAPTIONS.values()}
    # PI 2026-11-23, LEDGER SP-9: the class map (appendix_tables.py) is placed, the level table it takes in and the
    # use-case table (make_tables.py) dropped; neither is an mlst_results table
    other = {"tab:app-vocabulary", "tab:app-levels", "tab:usecase"}
    assert len(placed) == len(set(placed)) and set(placed) <= labels | other     # every placed label is a table the book makes
    assert set(V.unplaced()) == labels - set(placed) - set(G.DROPPED_TABLES) == set()
    assert set(G.DROPPED_TABLES) <= labels | other
    heading, rows = V.blocks()[-1]
    assert heading == "Not placed" and rows[0][1] == (f"Dropped figures: {', '.join(G.DROPPED)}. Tables dropped "
                                                      f"(nothing cites them): {', '.join(G.DROPPED_TABLES)}.")


def test_the_sculpting_test_reads_the_bootstrap_or_says_it_is_pending(tmp_path):
    missing = V._boot_rows(tmp_path / "sculpting_boot.json")
    assert len(missing) == 1 and missing[0][1].startswith("Pending sculpting_boot.json")
    q = {"n_pairs": 3, "per_pair": [0.012] * 3, "delta": 0.0122, "run_sd": 0.0029, "run_se": 0.0017, "boot_sd": 0.0005,
         "sigma": 0.00294, "z": 4.15, "pass": True}
    doc = {"differences": {name: {wp: {"distance": q, "jsd": q} for wp, _ in V.BOOT_WPS} for name, _ in V.BOOT_DIFFS}}
    doc["differences"]["43-188"]["eps_s=0.5"]["distance"] = {**q, "delta": -0.002, "z": None, "pass": False}
    (tmp_path / "sculpting_boot.json").write_text(json.dumps(doc))
    rows = V._boot_rows(tmp_path / "sculpting_boot.json")
    assert len(rows) == 2 + len(V.BOOT_DIFFS) + 1 and len(V.BOOT_X) == 10
    assert all(r[2] == V.BOOT_X and len(r[1]) == len(V.BOOT_X) for r in rows[:-1])
    assert rows[0][1] == ("", "", "", "QCD 1%", "", "", "QCD 10%", "", "", "res. 50%")
    assert rows[1][1] == ("Difference", "Δ", "σ", "Δ/σ", "Δ", "σ", "Δ/σ", "Δ", "σ", "Δ/σ")
    assert rows[2][1] == ("17 − 188", "1.22", "0.29", "4.2*", "1.22", "0.29", "4.2*", "1.22", "0.29", "4.2*")
    assert rows[3][1] == ("43 − 188", "1.22", "0.29", "4.2*", "1.22", "0.29", "4.2*", "−0.20", "0.29", "–")
    # F10b's verdict needs the signs: 17 + mass - 17 is +1.22 here, so not the main text but the placement's tier (PI
    # 2026-11-23, LEDGER SP-9: the data release, the supplement tier); the rule and its two z follow
    assert rows[-1][1] == ("F10b: supplement (main text only if at QCD 1% 17 − 188 > +3σ and 17 + mass − 17 < −3σ; "
                           "here +4.15σ and +4.15σ).")
    doc["differences"]["17mass-17"]["eps_b=0.01"]["distance"] = {**q, "delta": -0.0122, "z": -4.15}
    (tmp_path / "sculpting_boot.json").write_text(json.dumps(doc))
    rows = V._boot_rows(tmp_path / "sculpting_boot.json")
    assert rows[4][1][1:4] == ("−1.22", "0.29", "−4.2*") and rows[-1][1].startswith("F10b: main text (")


def test_the_map_shows_every_pulled_sculpting_difference():
    """On the pulled bootstrap: every difference at every working point with its Δ, σ and Δ/σ from the file, a star
    exactly where the file passes it, and F10b's tier by f10b_main with the rule's own Δ/σ."""
    if not V.BOOT.exists():
        pytest.skip(f"{V.BOOT} not pulled")
    diffs = json.loads(V.BOOT.read_text())["differences"]
    rows = V._boot_rows(V.BOOT)
    fmt = lambda v, f: format(v, f).replace("-", "−")     # noqa: E731
    for (name, label), (_, cells, _) in zip(V.BOOT_DIFFS, rows[2:-1]):
        assert cells[0] == label
        for k, (wp, _) in enumerate(V.BOOT_WPS):
            d = diffs[name][wp]["distance"]
            assert cells[1 + 3 * k:4 + 3 * k] == (fmt(100 * d["delta"], ".2f"), fmt(100 * d["sigma"], ".2f"),
                                                  fmt(d["z"], ".1f") + ("*" if d["pass"] else ""))
            assert d["pass"] == (abs(d["delta"]) > 3 * d["sigma"])
    z = [fmt(diffs[n][G.F10B_WP]["distance"]["z"], "+.2f") + "σ" for n, _ in G.F10B_RULE]
    # PI 2026-11-23, LEDGER SP-9: F10b went to the data release (the supplement tier); the main text only on the rule
    tier = "main text" if G.f10b_main(diffs) else G.TIER_NAME[G.TIER["F10b"]].lower()
    assert rows[-1][1].startswith(f"F10b: {tier} (") and rows[-1][1].endswith(f"here {z[0]} and {z[1]}).")
    assert G.f10b_main(diffs) or G.TIER["F10b"] == G.SUPPLEMENT          # the placement follows the test


def test_the_million_jet_block_formats_mean_and_sd_in_the_last_digit():
    assert V._pm([0.4267, 0.4263, 0.4257]) == "0.4262(5)"
    assert V._pm([0.40757, 0.40483, 0.40369]) == "0.4054(20)"       # two digits of the SD below 355
    assert V._pm([0.98548, 0.98535, 0.98544]) == "0.9854(1)"        # never past the fourth significant figure
    assert V._pm([0.98541, 0.98542, 0.98541]) == "0.9854(<1)" and V._pm(None) == "–" and V._pm([0.5, 0.5]) == "0.5000"
    vals = {"L188": [[0.4, 0.41]] * 4, "R16_Q1": [[0.4, 0.41]] * 4, "SCRATCH": [None, [0.4, 0.41], [0.4, 0.41], [0.4, 0.41]]}
    rows = V._ft_rows(vals)
    # L7: the use in words, never "f.t."
    assert [r[1][0] for r in rows[2:5]] == ["188 classes (fine-tuned)", "17 classes (fine-tuned)", "from scratch"]
    assert rows[4][1] == ("from scratch", "–", "0.405(7)", "0.405(7)", "0.405(7)")
    assert all(r[2] == V.FT_X for r in rows[:5]) and rows[-1][1].startswith("From scratch: ParT's full flat+decay")
    assert rows[-1][1].endswith("–: pending: JetClass-II accuracy, from scratch.")
    assert not V._ft_rows({a: [[0.4, 0.41]] * 4 for a in V.FT_ARMS})[-1][1].count("pending")


def test_the_million_jet_block_reads_the_tables_files():
    """ft_values against the pulled files themselves: JetClass from ft_curves' read-outs (tab:jetclass), the
    from-scratch cells from the full-schedule one; JetClass-II pretrained from the fine-tuning read-outs (tab:ft) and
    from scratch the mean over the 188 class AUCs of the full-schedule curves."""
    data = V.C.P.DATA
    need = [data / "curves/full_budget_long_fulldecay_leg2.json", data / "curves/full_budget_long_leg2.json",
            data / "curves_v2/jc2_ft_ovr_fulldecay", data / "budget/budget_metrics.json"]
    if not all(p.exists() for p in need):
        pytest.skip("the 10^6 fine-tuning files are not pulled")
    v = V.ft_values()
    n, seeds = "N1000000", ("s1", "s2", "s3")
    full = json.loads(need[0].read_text())["cells"]["scratch"][n]
    assert v["SCRATCH"][2] == [full[s]["jc1"]["accuracy"] for s in seeds]
    assert v["SCRATCH"][3] == [full[s]["jc1"]["macro_auc_ovr"] for s in seeds]
    long = json.loads(need[1].read_text())["cells"]
    jc2 = {**json.loads(need[3].read_text())["cells"], **json.loads((data / "budget/budget_long_metrics.json").read_text())["cells"]}
    for arm in ("L188", "R16_Q1"):
        ms = [m.removeprefix("mtx-") for m in V.C.RUNS[arm]]
        assert v[arm][2:] == [[long[m][n]["s1"]["jc1"][k] for m in ms] for k in ("accuracy", "macro_auc_ovr")]
        assert v[arm][:2] == [[jc2[f"leg_jc2b/{m}/{n}/s1"][k] for m in ms] for k in ("accuracy", "macro_auc_ovr")]
        curves = [json.loads((data / f"curves_v2/jc2_ft_ovr/mtx-{m}.json").read_text())["curves"] for m in ms]
        assert v[arm][1] == pytest.approx([np.mean([c["auc"] for c in cs.values()]) for cs in curves], abs=1e-12)
    scratch = [json.loads((need[2] / f"scratch-{s}.json").read_text())["curves"] for s in seeds]
    assert all(len(cs) == 188 for cs in scratch)
    assert v["SCRATCH"][1] == pytest.approx([np.mean([c["auc"] for c in cs.values()]) for cs in scratch], abs=1e-15)
    assert v["SCRATCH"][0] is None or len(v["SCRATCH"][0]) == 3     # accuracy: only with the full-schedule budget files


def test_the_class_pair_family_has_one_form_for_both_modes_and_every_pair(monkeypatch):
    """Plan "Class pairs" (PI 2026-11-13): columns the pairs in one order, the upper row the label sets with each mass
    network under its own (fine-tuned: and from scratch), the lower row the flavor-boundary partitions with 188 and 17
    classes; the same pairs and panel order for linear probes (F4-mech, F4-other, F4-light) and fine-tuned (F4-mech-ft,
    F4-other-ft, F4-light-ft). L20 (two 6.3 cm panels a row): the four other pairs are two figures, the resonance pair
    and QCD bb vs cc (F4-other), QCD with heavy quarks against light (F4-light)."""
    calls = []
    monkeypatch.setattr(R, "ROC", {"pairs": {"x": {}}, "pairs_ft": {"x": {}}})
    monkeypatch.setattr(R, "_roc_grid", lambda path, grid, **kw: calls.append((grid, kw)))
    # PI 2026-11-23, LEDGER SP-9 (merge 8): F4-light merged into F4-other (its own test below); the fine-tuned pages keep
    # the two-figure form in the data release
    draws = (R.draw_f4_mech, R.draw_f4_mech_ft, R.draw_f4_other_ft, R.draw_f4_light_ft)
    for draw in draws:
        draw(None)
    assert R.PAIR_ORDER[:2] == R.MECHANISM == ("bb_vs_cc", "bc_vs_bq_cs") and R.OTHER == R.PAIR_ORDER[2:6]
    assert R.LABEL_SETS == L.M.SETS and R.PARTITIONS == L.M.FLAV + ("L188", "R16_Q1")
    assert all(kw["w"] == R.PAIR_W[2] >= 6.1 for _, kw in calls)          # L20: at least 6.1 cm, two a row
    pairs_of = (R.MECHANISM, R.MECHANISM, R.OTHER[:2], R.OTHER[2:])
    for (grid, kw), pairs, ft in zip(calls, pairs_of, (False, True, True, True), strict=True):
        assert [[(key, nets) for _, key, nets, _ in row] for row in grid] == [
            [(k, R.LABEL_SETS + (("SCRATCH",) if ft else ())) for k in pairs], [(k, R.PAIR_PARTITIONS) for k in pairs]]
        # PI 2026-11-19 (list item C5.2, LEDGER FG-24): the random partitions join the partitions row; PI 2026-11-21
        # (LEDGER PS-5): those that merge the pair and those that keep it apart, separately
        assert R.PAIR_PARTITIONS == R.PARTITIONS[:3] + ("RAND_MERGED", "RAND_APART") + R.PARTITIONS[3:]
        assert kw["row_names"] == ("Label sets", "17-class partitions")
        assert all(":" not in title for row in grid for *_, title in row)            # L7: no colon in a panel title
    assert [k for k, *_ in calls[0][0][0]] == [k for k, *_ in calls[1][0][0]]   # one pair order in both modes


def test_the_other_flavor_pairs_are_the_resonance_pair_with_its_partitions_and_the_qcd_pairs(monkeypatch):
    """PI 2026-11-23, LEDGER SP-9 (merge 8): F4-light merged into F4-other, the partitions on the resonance pair only,
    as every partition merges the QCD classes. Review of render dfe5942 (C; main session 2026-11-23, an exception to
    merge 8 for the PI): that figure, 24 cm high, could not share a page with its caption, so it is two, X->YY->bbqq vs
    ccqq with the label sets beside the partitions (F4-other) and the three QCD pairs with the label sets, the legend
    in the free slot (F4-qcd); every panel its own y numbers (FG-8)."""
    calls = []
    monkeypatch.setattr(R, "ROC", {"pairs": {"x": {}}})
    monkeypatch.setattr(R, "_roc_grid", lambda path, grid, **kw: calls.append((grid, kw)))
    R.draw_f4_other(None)
    R.draw_f4_qcd(None)
    (other, kw_o), (qcd, kw_q) = calls
    assert [[(p[1], p[2]) for p in row] for row in other] == [[(R.OTHER[0], R.LABEL_SETS), (R.OTHER[0], R.PAIR_PARTITIONS)]]
    assert [[None if p is None else (p[1], p[2]) for p in row] for row in qcd] == [
        [(k, R.LABEL_SETS) for k in R.QCD_PAIRS[:2]], [(R.QCD_PAIRS[2], R.LABEL_SETS), None]]
    assert R.OTHER[0] == "bbqq_vs_ccqq" and R.QCD_PAIRS == R.OTHER[1:] and all(k.startswith("qcd_") for k in R.QCD_PAIRS)
    assert kw_q["legend_slot"] == (1, 1) and "legend_slot" not in kw_o
    assert kw_o["one_x"] and kw_q["one_x"]          # FG-2 (review of render 5f4e1b8, C8): one x range for the panels
    assert all(kw["w"] == R.PAIR_W[2] >= 6.1 and kw["ratio_scale"] == "log" and kw["own_y"] for kw in (kw_o, kw_q))
    assert not hasattr(R, "draw_f4_light") and "F4-light" not in {p.id for p in R.PAGES}
    assert {"F4-other", "F4-qcd"} <= {p.id for p in R.PAGES}


def test_a_grid_of_different_pairs_puts_every_panel_on_one_x_range(monkeypatch):
    """FG-2 (review of render 5f4e1b8, C8): F4-qcd's three pairs, one a panel in a two-column grid, share one x range
    from the lowest efficiency any curve reaches, where the columns used to be fitted one by one ((b) from 0.2, (a) and
    (c) from 0)."""
    def curve(docs, models, key):
        eps = np.linspace(0.25 if key == R.QCD_PAIRS[1] else 0.03, 1.0, 10)
        return eps, 1000 * (1 - eps) + 1, 0.9
    monkeypatch.setattr(R, "ROC", {"pairs": {"any": {}}})
    monkeypatch.setattr(R, "POOLED", {n: [n] for n in R.NETWORKS})
    monkeypatch.setattr(R, "mean_over_runs", curve)
    figs = []
    monkeypatch.setattr(R.S, "save", lambda fig, path: figs.append(fig))
    R.draw_f4_qcd(None)
    panels = [ax for ax in figs[0].axes if ax.get_title()[:1] == "("]
    assert len(panels) == 3 and len({ax.get_xlim() for ax in panels}) == 1
    assert panels[0].get_xlim()[0] < 0.0
    matplotlib.pyplot.close(figs[0])


def test_the_headline_panels_of_one_metric_share_one_relative_range_below_and_above_188_classes(monkeypatch,
                                                                                                tmp_path):
    """FG-2 (PI 2026-11-20: one relative range per metric about the 188-class mean, ticks in absolute units); review of
    render 5f4e1b8 (C6): the range runs from the largest deviation below 188 classes to the largest above, so a side no
    network reaches is not drawn and a small loss (JetClass) takes twice the width it had; CA-10 (B12): no "SD" in the
    legend."""
    drop = {"jc2_188": 0.3, "jc1": 0.025, "top": 0.4, "qg": 0.05, "qg_herwig": 0.05}
    base = {"jc2_188": 0.5, "jc1": 0.8, "top": 200.0, "qg": 80.0, "qg_herwig": 40.0}

    def value(key, n, ds, met):
        return lambda m: base[ds] * (1 if m in L.RUNS["L188"] else 1 - drop[ds])
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    L._draw_hv(tmp_path / "F2.pdf", value, (), L._head(), "none", "relative")
    titles = {key: f"({x}) {t}" for x, (key, *_, t) in zip("abcde", L.HV_PANELS)}
    axes = {key: next(ax for ax in figs[0].axes if ax.get_title() == t) for key, t in titles.items()}
    rel = {key: tuple(x / base[ds] - 1 for x in axes[key].get_xlim()) for key, _, ds, *_ in L.HV_PANELS}
    assert rel["jc2"] == pytest.approx(rel["jc1"]) and rel["top"] == pytest.approx(rel["qg"]) == rel["qg_herwig"]
    assert rel["jc2"][0] < -0.3 and 0 < rel["jc2"][1] < 0.1 and rel["top"][0] < -0.4 and 0 < rel["top"][1] < 0.1
    legend = [t.get_text() for leg in figs[0].legends for t in leg.get_texts()]
    assert legend and not [t for t in legend if "SD" in t.split()] and any("standard deviation" in t for t in legend)
    assert L.PUBLISHED_LABEL == "ParT, published"           # CA-6 (B15): the paper's name in the training-size legend
    matplotlib.pyplot.close(figs[0])


def test_every_training_size_point_reads_the_fixed_cell_at_its_rows_recorded_rate(monkeypatch):
    """RR-7 (PI 2026-11-23): every fine-tuned training-size point (Fig. 6, A8) reads the fixed-schedule cell at its
    row's recorded rate, or the build stops, also in a preview: the full top and quark/gluon sets keep ParT's recipe, a
    row without a recorded rate waits, a cell trained at another rate or a value not its cell's stops the figure
    (Herwig at the rate of the quark/gluon cells it re-scores), and so does a read-out other than the one rule's."""
    import types
    rec = lambda lr, v: {"recipe": {"lr": lr}, "r30": v}      # noqa: E731
    cells = {("qg", "l188-s1", 10_000): [rec("1e-2", 50.0)], ("qg", "l188-s1", 100_000): [rec("1e-2", 60.0)],
             ("qg_herwig", "l188-s1", 100_000): [{"r30": 30.0}]}
    original = L._fixed_cells
    monkeypatch.setattr(L, "_fixed_cells", lambda: cells)
    rates = {("qg", 10_000): "1e-2", ("qg", 100_000): "3e-4"}
    monkeypatch.setattr(L.M, "_rate_builder", lambda: types.SimpleNamespace(row_rate=lambda t, n, k: rates.get((t, n))))
    full = L.M.FULL_N["qg"]
    assert L._settled({10_000: [50.0], 1_000: [40.0], full: [100.0]}, "qg", "pre", "L188", ["l188-s1"], "r30") == {
        10_000: [50.0], full: [100.0]}
    for key, v in (("qg", 60.0), ("qg_herwig", 30.0)):           # trained at 1e-2, the row's recorded rate 3e-4
        with pytest.raises(SystemExit, match="RR-7"):
            L._settled({100_000: [v]}, key, "pre", "L188", ["l188-s1"], "r30")
    with pytest.raises(SystemExit, match="RR-7"):                 # a value that is not its cell's
        L._settled({10_000: [51.0]}, "qg", "pre", "L188", ["l188-s1"], "r30")
    monkeypatch.setattr(L.M, "ft_source", lambda data: "budget, budget_long, fulldecay, prefull (the trees before ...)")
    with pytest.raises(SystemExit, match="RR-7"):                 # no silent fallback to the older trees
        original.__wrapped__()


def test_the_pair_legend_holds_only_the_curves_drawn(monkeypatch):
    """A model with no curve in any panel, or only a complete separation, has no legend entry."""
    eps = np.linspace(0.1, 1.0, 10)
    curve = {"L188": (eps, 1000 * (1 - eps) + 1, 0.9), "L162": None, "R42_Q1": (eps[:1], eps[:1], 1.0)}
    monkeypatch.setattr(R, "ROC", {"pairs": {"any": {}}})
    monkeypatch.setattr(R, "POOLED", {n: [n] for n in R.NETWORKS})
    monkeypatch.setattr(R, "mean_over_runs", lambda docs, models, key: curve.get(models[0], (eps, 50 * (1 - eps) + 1, 0.8)))
    shown = []
    monkeypatch.setattr(R.S, "fig_legend", lambda fig, handles, **kw: shown.extend(kw["labels"]))
    monkeypatch.setattr(R.S, "save", lambda fig, path: matplotlib.pyplot.close(fig))
    R.draw_f4_mech(None)
    assert R.S.LEGEND["L162"] not in shown and R.S.LEGEND["R42_Q1"] not in shown
    assert {R.S.LEGEND[n] for n in ("L188", "L162_MASS", "R16_Q1", "R16_Q1_MASS", "FLAV_F0", "FLAV_F1", "FLAV_F1R")} <= set(shown)
    # PS-5: the random partitions that merge the pair and those that keep it apart, each under its own name
    assert {"random partitions, pair merged", "random partitions, pair apart"} <= set(shown)
    # one legend column a group of networks (L5 order), blank entries below a short column
    named = [t for t in shown if t]
    assert named == sorted(named, key=lambda t: [R._name(n) for n in R.NETWORKS].index(t))


def test_the_pair_figures_draw_the_random_partitions_that_merge_the_pair_apart_from_those_that_keep_it_apart():
    """PI 2026-11-21 (LEDGER PS-5): "show the random partitions that merge the pair and those that keep it apart
    separately, with the count of each": two curves in the partitions row, each over its groupings from the label maps
    (merge_status), the captions printing the counts the build writes (MlstPairRand{Merged,Apart}<pair>N, in F4-mech's
    caption as words, review of render 5f4e1b8); a grouping that merges only part of the pair is in neither, and the
    caption leaves it out by its count in words (MlstPairRandPartly<pair>Word)."""
    import re
    M = L.M
    st = M.merge_status(ROOT)
    assert R.PAIR_RAND == {"RAND_MERGED": "merged", "RAND_APART": "apart"} and R.PAIR_STATUS == st
    gen = (ROOT / "paper/journal/results_generated.tex").read_text()
    tex = {f: (ROOT / "paper/journal/sections" / f).read_text() for f in ("results_jc.tex", "appendix.tex")}
    partly = {}
    for f, keys in (("results_jc.tex", R.MECHANISM), ("appendix.tex", R.OTHER[:1])):
        for key in keys:
            runs = {s: R._models({}, net, key) for net, s in R.PAIR_RAND.items()}
            assert runs == {s: [m for a in M.RAND if st[a][key] == s for m in R.RUNS[a]] for s in ("merged", "apart")}
            assert runs["merged"] and runs["apart"] and not set(runs["merged"]) & set(runs["apart"]), key
            for s in ("merged", "apart"):
                n = sum(st[a][key] == s for a in M.RAND)
                name = rf"MlstPairRand{s.title()}{M.PAIR_WORD[key]}N"
                assert re.search(r"\\newcommand\{\\" + name + r"\}\{" + str(n) + r"\}", gen), name
                word = rf"MlstPairRand{s.title()}{M.PAIR_WORD[key]}Word"
                assert re.search(r"\\newcommand\{\\" + word + r"\}\{" + ("zero one two three four five".split()[n])
                                 + r"\}", gen), word
                used = {"merged": [name, word], "apart": [name, word, "MlstPairRandApartMechWord"]}[s]
                assert any("\\" + u + "{}" in tex[f] for u in used), (f, used)
            partly[key] = sum(st[a][key] == "partly" for a in M.RAND)
    assert partly == {"bb_vs_cc": 0, "bc_vs_bq_cs": 1, "bbqq_vs_ccqq": 0}
    # fourth review (C3): the partial merger said in words
    assert r"\MlstPairRandPartlyBcWord{}" + "\nmerging $X\\to bc$ with $bq$ or $cs$ alone left out" in tex["results_jc.tex"]
    assert to_hex(R.S.SERIES[R._style("RAND_APART")]) == to_hex(R.S.SERIES["RAND"]) != to_hex(R.S.SERIES["RAND_MERGED"])


@pytest.mark.parametrize("pid", ["F4-mech", "F4-other", "F4-qcd", "F9b", "A8", "A10", "F11", "PC-jc2", "CONF-share"])
def test_every_panel_has_its_own_x_and_y_labels(pid, monkeypatch, tmp_path):
    """FG-8 (PI 2026-11-20): "every panel has its own x and y labels naming exactly what it shows", in the figures the
    reviews of renders 5c40d69 and dfe5942 named (Figs. 4, 5, 6, 7, 9, 10, 11): every axes has a y label, and every
    panel an x label, its own or that of the ratio panel under it, which shares its x axis; and (review of dfe5942, B9)
    every panel with data shows its own y tick numbers, shared ranges included."""
    import gallery_figures as GF
    pages = {p.id: p for mod in GF.MODULES for p in mod.PAGES}
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    try:
        pages[pid].draw(tmp_path / f"{pid}.pdf")
    except L.C.Absent as e:
        pytest.skip(f"{pid}: {e}")
    except SystemExit as e:         # RR-7 (PI 2026-11-23): no training-size figure without the one rule's cells
        if "(RR-7)" not in str(e):
            raise
        pytest.skip(f"{pid}: {e}")
    (fig,) = figs
    assert fig.axes and all(ax.get_ylabel() for ax in fig.axes), pid
    fig.canvas.draw()
    for ax in fig.axes:
        if ax.get_title():
            assert any(a.get_xlabel() for a in ax.get_shared_x_axes().get_siblings(ax)), (pid, ax.get_title())
        if pid not in ("A8", "A10") and (ax.lines or ax.collections):
            assert [t for t in ax.get_yticklabels() if t.get_visible() and t.get_text()], (pid, ax.get_title())
    matplotlib.pyplot.close(fig)


def test_the_per_class_figure_marks_the_classes_its_sentence_counts(monkeypatch, tmp_path):
    """Review of render dfe5942 (B2): Fig. 4 backs "164 of the 188 classes ... beyond three standard deviations" in the
    PDF: a class beyond three standard deviations from the 188-class recall (perclass_recall.py's diffs, the test whose
    count mlst_results.jc2_recall_macros prints) is a filled point, one within them open, with a legend entry."""
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    try:
        PC.draw_jc2(tmp_path / "PC-jc2.pdf")
    except L.C.Absent as e:
        pytest.skip(str(e))
    (fig,) = figs
    for i, (a, _) in enumerate(PC.FIG4):
        d = PC._doc("jc2", 0)["diffs"][f"{a}-L188"]
        # CA-12 (review of render 5f4e1b8, B13): the mode heads its column, the linear probes the left one
        ax = next(x for x in fig.axes if x.get_title() == f"({'abcdef'[2 * i]}) {L.S.LEGEND[a]}")
        pts = [ln for ln in ax.lines if ln.get_linestyle() in ("None", "none", "")]
        filled = [ln for ln in pts if to_hex(ln.get_markerfacecolor()) != "#ffffff"]
        assert len(pts) == len(d["delta"]) and len(filled) == len(d["lower"]) + len(d["higher"]), a
    assert PC.WITHIN in [t.get_text().replace("\n", " ") for leg in fig.legends for t in leg.get_texts()]
    matplotlib.pyplot.close(fig)


def test_the_confusion_test_figure_draws_the_nine_label_sets_at_their_share_and_accuracy(monkeypatch, tmp_path):
    """DL-8b (PI 2026-11-23): the pre-registered confusion test (CS-10) is an appendix figure, one point per 17-group
    label set (the 17-class level, three flavor-boundary partitions, five random partitions) at the share of the
    188-class probe's resonant confusions it keeps apart (x) and its mean JetClass-II linear-probe accuracy (y), a
    single run open (FG-6); one panel of a single-panel figure's size at the text width, labelled, under the heading
    "Linear probe" (CA-13), and drawn by the paper's render (PAPER)."""
    import json
    import gallery_figures as GF
    figs = []
    monkeypatch.setattr(PC.S, "save", lambda fig, path: figs.append(fig))
    try:
        PC.draw_conf_share(tmp_path / "CONF-share.pdf")
    except L.C.Absent as e:
        pytest.skip(str(e))
    groups = json.loads(PC.SHARE.read_text())["groupings"]
    (fig,) = figs
    (ax,) = fig.axes
    pts = sorted((ln for ln in ax.lines if ln.get_linestyle() in ("None", "none", "")),
                 key=lambda ln: ln.get_xdata()[0])
    want = sorted((g["share_apart"], float(np.mean(list(g["accuracy_runs"].values())))) for g in groups.values())
    assert len(groups) == len(pts) == 9
    assert [ln.get_xdata()[0] for ln in pts] == pytest.approx([x for x, _ in want])
    assert [ln.get_ydata()[0] for ln in pts] == pytest.approx([y for _, y in want])
    single = [ln for ln in pts if to_hex(ln.get_markerfacecolor()) == "#ffffff"]
    assert len(single) == sum(len(g["accuracy_runs"]) == 1 for g in groups.values()) == 5
    assert fig.get_figwidth() * 2.54 == pytest.approx(PC.S.PRINT_W) and ax.get_position().width * PC.S.PRINT_W >= 6.1
    assert ax.get_xlabel() and ax.get_ylabel() and not ax.get_title() and fig.legends
    assert "CONF-share" in GF.PAPER and GF.MODE_HEADINGS["CONF-share"] == "Linear probe"
    matplotlib.pyplot.close(fig)


def test_the_validation_figure_is_set_at_the_text_width_with_a_single_panel(monkeypatch, tmp_path):
    """FG-14, FG-19 (review of render dfe5942, B15): the paper sets every figure at the text width, so VAL is drawn 15 cm
    wide (its text printed at 8-9 pt), its one panel the size of a single-panel figure's, the legend beside it."""
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    try:
        L.draw_val(tmp_path / "VAL.pdf")
    except L.C.Absent as e:
        pytest.skip(str(e))
    (fig,) = figs
    assert fig.get_figwidth() * 2.54 == pytest.approx(L.S.PRINT_W)
    (ax,) = fig.axes
    assert ax.get_position().width * L.S.PRINT_W == pytest.approx(L.S.SINGLE_W - 1.75) and fig.legends
    matplotlib.pyplot.close(fig)


def test_the_binned_mass_figure_is_points_with_bars_and_no_connecting_lines(monkeypatch, tmp_path):
    """LEDGER FG-18 (PI 2026-11-15): binned figures are "points with error bars and horizontal bin bars; no connecting
    lines" (F9b, and the A17 family that shares its panels): every line drawn is a series of markers or the dotted unity
    reference, every panel holds its bars, and no step line is left."""
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    try:
        H.draw_f9b(tmp_path / "F9b.pdf")
    except L.C.Absent as e:
        pytest.skip(str(e))
    (fig,) = figs
    for ax in fig.axes:
        for ln in ax.lines:
            if ln.get_linestyle() in ("None", "none", " ", ""):
                assert ln.get_marker() not in (None, "None", "none", ""), ln.get_label()
            else:
                assert ln.get_linestyle() == ":" and len(set(ln.get_ydata())) == 1, ln.get_label()
        assert ax.collections and not [p for p in ax.patches if type(p).__name__ == "StepPatch"]
    assert not hasattr(H.L, "steps")
    matplotlib.pyplot.close(fig)


def test_the_linear_probe_training_size_figures_draw_no_published_full_training(monkeypatch, tmp_path):
    """CS-8 (compare like with like): ParT's published values are full trainings, drawn beside fine-tuned networks
    (A8), never beside linear probes (A10, A10-part)."""
    for draw in (L.draw_a10, L.draw_a10_part):
        figs = []
        monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
        try:
            draw(tmp_path / "x.pdf")
        except L.C.Absent:
            continue
        (fig,) = figs
        assert not [ln for ax in fig.axes for ln in ax.lines if ln.get_label() == L.PUBLISHED_LABEL], draw.__name__
        matplotlib.pyplot.close(fig)


def test_the_training_size_figures_draw_no_superseded_fine_tuning(monkeypatch):
    """LEDGER RR-3, FT-8, SP-11 (review of render 5c40d69, B11): below ParT's full top and quark/gluon sets a fine-tuned
    point is drawn only from the one-rule read-out and only where its row has a recorded rate; a row still in its rate
    search (quark/gluon 1e5 pretrained on 23 Nov) waits for its rerun. RR-7 (PI 2026-11-23): the trees before the one
    rule stop the figure instead of leaving only the full sets drawn."""
    M = L.M
    series = {1_000: [1.0], 10_000: [1.0], 100_000: [1.0], 1_600_000: [1.0]}
    rates = {("qg", 10_000, "pre"): "1e-2", ("qg", 100_000, "scratch"): "1e-3"}
    builder = type("B", (), {"row_rate": staticmethod(lambda task, n, kind: rates.get((task, n, kind)))})
    monkeypatch.setattr(M, "_rate_builder", lambda: builder)
    cell = lambda lr: [{"recipe": {"lr": lr}, "r30": 1.0}]     # noqa: E731
    cells = {("qg", "l188-s1", 10_000): cell("1e-2"), ("qg", M.SCRATCH, 100_000): cell("1e-3"),
             ("qg_herwig", M.SCRATCH, 100_000): [{"r30": 1.0}]}
    monkeypatch.setattr(M, "_budget_cells", lambda path: cells)
    monkeypatch.setattr(L, "_fixed_cells", L._fixed_cells.__wrapped__)       # the read-out each call, not cached
    monkeypatch.setattr(M, "ft_source", lambda data: M.FIXED_PARTIAL + " (preview)")
    pre, scratch = ("pre", "L188", ["l188-s1"], "r30"), ("scratch", "SCRATCH", [M.SCRATCH], "r30")
    assert sorted(L._settled(series, "qg", *pre)) == [10_000, 1_600_000]
    assert sorted(L._settled(series, "qg_herwig", *scratch)) == [100_000, 1_600_000]
    assert L._settled({10_000: [1.0], 1_000_000: [1.0]}, "jc1", *pre) == {}      # JetClass's 1e6 is a one-rule cell
    monkeypatch.setattr(M, "ft_source", lambda data: M.FIXED)
    assert sorted(L._settled(series, "qg", *pre)) == [10_000, 1_600_000]
    monkeypatch.setattr(M, "ft_source", lambda data: "budget, budget_long, fulldecay, prefull (the trees before the one rule)")
    with pytest.raises(SystemExit, match="RR-7"):
        L._settled(series, "qg", *pre)


def test_the_per_class_grid_puts_every_panel_on_one_range_and_colours_classes_not_networks(monkeypatch):
    """Per-class family (PI 2026-11-12/13; L20: networks as rows; PI 2026-11-18: per-class recall, Fig. 4's rows 162, 43,
    17 classes, 162 classes + mass and 17 classes + mass against 188): linear probe beside fine-tuned, one linear axis
    range for the grid from the lowest and highest recall drawn; a point's colour is what its own row's labels merge its
    class with (review of render 5f4e1b8, C16; gallery_perclass.MERGE_COLOURS, from CLASS10), never a colour that means
    a network (L6); the diagonal is the black dotted reference; the mode once, an italic heading above each column, and
    the panel titles name the network only (CA-12, B13)."""
    classes = range(188)
    monkeypatch.setattr(PC, "SOURCES", {"jc2": (lambda m: {c: 0.30 + 0.003 * c for c in classes},
                                                lambda m: {c: 0.25 + 0.0035 * c for c in classes})})
    monkeypatch.setattr(PC, "_doc", lambda test, mode: {})          # no test-jet bootstrap: bars of the run SD only
    figs = []
    monkeypatch.setattr(PC.S, "save", lambda fig, path: figs.append(fig))
    PC.draw_jc2(None)
    letters = "abcdef"
    panels = [ax for ax in figs[0].axes if ax.get_title()[:3] in {f"({x})" for x in letters}]
    lo, hi = 0.25, 0.25 + 0.0035 * 187
    pad = 0.04 * (hi - lo)
    assert len(panels) == 6 and all(ax.get_xlim() == ax.get_ylim() == panels[0].get_xlim() for ax in panels)
    assert panels[0].get_xlim() == pytest.approx((lo - pad, hi + pad))
    assert all(ax.get_xscale() == ax.get_yscale() == "linear" for ax in panels)
    networks = {to_hex(c) for c in PC.S.SERIES.values()}
    for ax in panels:
        diagonal, *points = ax.lines
        assert diagonal.get_linestyle() == ":" and to_hex(diagonal.get_color()) == "#000000"
        assert len(points) == 188 and not {to_hex(ln.get_color()) for ln in points} & networks
        assert {to_hex(ln.get_color()) for ln in points} <= {to_hex(c) for c in PC.MERGE_COLOURS.values()}
    assert len(set(PC.MERGE_COLOURS.values())) == len(PC.MERGE_KINDS)
    # the merges of each label set: 27 QCD classes, merged into one by every coarser set; of the 161 resonant ones, at
    # 17 classes 77 share their group with a decay differing only in quark flavor, 14 only in lepton flavor, 64 with
    # both, 6 with neither; 162 classes merge only the QCD classes; the five random partitions agree on 61
    counts = {a: collections.Counter(PC.merge_kind(a).values()) for a in ("L162", "R42_Q1", "R16_Q1", "RAND")}
    assert {a: {k: c[k] for k in PC.MERGE_KINDS if c[k]} for a, c in counts.items()} == {
        "L162": {"qcd": 27, "apart": 161}, "R42_Q1": {"qcd": 27, "quark": 67, "lepton": 34, "both": 44, "apart": 16},
        "R16_Q1": {"qcd": 27, "quark": 77, "lepton": 14, "both": 64, "apart": 6},
        "RAND": {"qcd": 27, "quark": 44, "apart": 17, "mixed": 100}}
    assert PC.merge_kind() == PC.merge_kind("R16_Q1")
    # each panel coloured by its own row's merges, both modes
    for ax, (a, _) in zip(panels, [r for r in PC.FIG4 for _ in (0, 1)]):
        assert collections.Counter(to_hex(ln.get_color()) for ln in ax.lines[1:]) == collections.Counter(
            to_hex(PC.MERGE_COLOURS[k]) for k in PC.merge_kind(a).values()), (a, ax.get_title())
    assert [ax.get_title() for ax in panels] == [f"({x}) {PC.S.LEGEND[a]}" for x, (a, _) in
                                                  zip(letters, [r for r in PC.FIG4 for _ in (0, 1)])]
    heads = [t for t in figs[0].texts if t.get_gid() == PC.S.MODE_HEADING]
    assert [t.get_text() for t in heads] == list(PC.MODES) and all(t.get_fontstyle() == "italic" for t in heads)
    figs[0].canvas.draw()
    for t, ax in zip(heads, panels[:2]):           # centred over its column, above its first panel's title
        tb, ab = t.get_window_extent(), ax.get_window_extent()
        assert tb.x0 + tb.x1 == pytest.approx(ab.x0 + ab.x1, abs=1.0) and tb.y0 > ax.title.get_window_extent().y1
    # one gallery_roc colour per decay family (the class-pair family), all from CLASS10
    assert len(R.FAMILY_COLOURS) == len(R.FAMILIES) == len(set(R.FAMILY_COLOURS))
    assert R.FAMILY_COLOURS == tuple(PC.S.CLASS10[k] for k in (0, 1, 4, 2, 3, 5))
    matplotlib.pyplot.close(figs[0])


def test_the_paper_captions_replace_the_page_captions_and_name_tables_that_are_placed():
    """gallery_captions (PI 2026-11-12): one caption for each figure of the main text and the appendix, set on the
    gallery's pages; a table is named by its label and printed with its number of the day, so every label must be a
    placed table (2026-11-13: the ROC figure points to Table 1 and its AUC table for R30 and the AUC, to tab:probes and
    tab:labelsets for the accuracy and R50)."""
    import re
    import gallery_captions as K
    paper = [k for k, t, _ in G.FIGURES if t in (G.MAIN, G.APPENDIX)]
    # PI 2026-11-14 (step 5): the mass probe joins the appendix, two task grids replace the JetClass-II UMAP grid
    # PI 2026-11-18: 9 main-text and 15 appendix figures; the 19 Nov placement: 7 and 16, PC-jc1 to the supplement
    # PI 2026-11-23, LEDGER SP-9: 7 main-text and 4 appendix figures; the others' captions stay on their release pages
    # PI 2026-11-23, LEDGER DL-8b: the confusion test joins the appendix (7 main-text and 6 appendix figures)
    assert sorted(K.CAPTIONS) == sorted(paper) and len(paper) == 13
    assert set(K.RELEASE_CAPTIONS) == set(K.RELEASE) and all(G.TIER[k] == G.SUPPLEMENT for k in K.RELEASE)
    for pid, raw in K.CAPTIONS.items():
        labels = re.findall(r"\{(tab:[a-z_0-9]+)\}", raw)
        assert set(labels) <= set(G.TIER) and not set(labels) & set(G.DROPPED_TABLES), pid
        text = K.text(pid)
        # a table by its number of the day, "table N" in a sentence, "Table N" only where a sentence starts with it
        assert "{tab:" not in text and len(text.split()) >= 40, pid
        assert all(m.start() == 0 or text[m.start() - 2:m.start()] == ". " for m in re.finditer("Table ", text)), pid
    assert re.findall(r"\{(tab:[a-z_0-9]+)\}", K.CAPTIONS["F6"]) == ["tab:summary", "tab:summaryauc", "tab:probes"]
    number = lambda k: G.number(k).removeprefix("Table ")      # noqa: E731
    # L21 (band for 188 classes only): the other standard deviations are in the tables with the values
    assert K.text("F6").endswith(f"values and the other standard deviations are in tables {number('tab:summary')}, "
                                 f"{number('tab:summaryauc')} and {number('tab:probes')}.")
    assert [number(k) for k in ("tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets")] == ["1", "A1", "A7", "A8"]
    # PI 2026-11-14: a paper caption names no supplement item by number, only "the supplementary material"
    for pid, raw in K.CAPTIONS.items():
        refs = re.findall(r"\{((?:tab|fig):[A-Za-z0-9_-]+)\}", raw)
        assert not [r for r in refs if G.TIER.get(r.removeprefix("fig:")) == G.SUPPLEMENT or G.TIER.get(r) == G.SUPPLEMENT], pid
        assert not re.search(r"(figure|table)s? S\d", K.text(pid)) and not re.search(r"cap of|epoch cap|at most \d+ epochs",
                                                                                     K.text(pid)), pid
        assert "not shown" not in K.text(pid) and "not claimed" not in K.text(pid), pid
    # the headline family's captions name Table 1 and the AUC table (L2)
    assert all(K.text(k).endswith(f"tables {number('tab:summary')} and {number('tab:summaryauc')}.") for k in ("F2a", "F2b"))
    # the PI's sentence on JetClass, in the main headline figure of the fine-tuned networks (L20 split it by mode)
    assert "17 classes stay below 188 classes on JetClass-II and make about 3% more errors on JetClass" in K.text("F2-ft")
    # the jet-pT figure (L4): why only JetClass-II and JetClass are binned, and its values in the data release (L18)
    # PI 2026-11-23, LEDGER SP-9: PT is a figure of the data release, its caption on its page
    pt = " ".join(K.fill(K.RELEASE_CAPTIONS["PT"]).split())
    assert re.findall(r"\{(tab:[a-z_0-9]+)\}", K.RELEASE_CAPTIONS["PT"]) == [] and "tab:ptbins" not in G.TIER
    assert "JetClass-II and JetClass only, as top and quark/gluon jet pT span about 50 to 100 GeV" in pt
    # PI 2026-11-20 (PS-1): one PDF, no supplementary material, "remove every 'in the supplementary material' from
    # every caption"; the file is the data release's (L18)
    assert pt.endswith("Values are in the data release file jet_pt_bins.csv.")
    # the pages of the gallery carry these captions (apply at the end of each module)
    for mod in (L, R, PC, H):
        for page in mod.PAGES:
            if page.id in K.CAPTIONS:
                assert page.caption == K.text(page.id), page.id
    # no text is drawn in a pair panel for a network that separates the pair: the caption says it (X->ee vs mumu, the
    # supplement's own page since 2026-11-13)
    assert "no curve" in R.SUPPLEMENT_CAPTIONS["F4-ee"]


def test_every_paper_caption_is_within_the_length_figcheck_allows():
    """figcheck.caption_findings: 40 to 145 words (PI 2026-11-12)."""
    import gallery_captions as K
    words = {pid: len(K.text(pid).split()) for pid in K.CAPTIONS}
    assert {pid: n for pid, n in words.items() if not 40 <= n <= 145} == {}


def test_every_main_and_appendix_figure_caption_is_at_most_135_words():
    """L16 (PI 2026-11-13): the test limit is 135 words for every main-text and appendix figure caption as the book
    prints it (gallery_captions.text, the numbers of the day filled in); supplement captions are left as they are."""
    import gallery_captions as K
    paper = [k for k, t, _ in G.FIGURES if t in (G.MAIN, G.APPENDIX)]
    assert sorted(K.CAPTIONS) == sorted(paper)
    words = {pid: len(K.text(pid).split()) for pid in paper}
    assert {pid: n for pid, n in words.items() if n > 135} == {}


def _unbold(tex: str) -> str:
    """A printed table with every bold value light (mlst_results._b: BOLD_ON, the text, a closing brace)."""
    on, out = L.M.BOLD_ON, ""
    while on in tex:
        before, rest = tex.split(on, 1)
        depth, i = 1, 0
        while depth:
            depth += (rest[i] == "{") - (rest[i] == "}")
            i += 1
        out, tex = out + before + rest[:i - 1], rest[i:]
    return out + tex


def test_table_one_and_its_auc_table_print_the_values_the_headline_figures_draw():
    """tab:summary and tab:summaryauc (PI 2026-11-13, L1, L2): Table 1 holds the headline metric of every row of the
    main headline figure (F2) on its five tasks, Herwig included, the released Sophon checkpoint's of its full version
    (F2a) on JetClass-II and JetClass; the appendix table the AUC beside each, the lower rows of F2a and F2b. Every cell
    is the figures' own value function (_probe_value, _ft_value, _scratch_value) over the row's runs (_hv_values), in
    the figures' row order (L5), printed at the table's one precision per column."""
    M, I = L.M, L.I
    assert tuple((key, n, ds, met) for key, n, ds, met, _ in L.HV_PANELS) == M.SUMMARY_PANELS == M.SUMMARY_GRID
    assert L.F2_GRID == [g for pair in zip(M.SUMMARY_GRID, M.SUMMARY_AUC_GRID) for g in pair] and len(L.F2_GRID) == 10
    assert L.F2_AUC == M.SUMMARY_AUC and [k for k, *_ in M.SUMMARY_PANELS] == ["jc2", "jc1", "top", "qg", "qg_herwig"]
    assert L.HV_ROWS == M.SETS and L.HV_PARTITIONS == M.SUMMARY_CONTROLS and L.HV_HEAD == M.PARTITIONS
    if any(f(*g) is None for f in (L._probe_value, L._ft_value) for g in L.F2_GRID):
        pytest.skip("the linear probes or the fine-tuning read-outs are not pulled")
    fits = L._fits()
    runs = {**L.RUNS, **({M.SOPHON: [M.SOPHON_MODEL]} if M.SOPHON_MODEL in fits else {})}
    pooled = M._pooled(runs)
    name = {**{a: M.LABEL[a] for a in M.SETS + (M.INIT, M.SOPHON)}, **{a: r"\quad " + M.LABEL[a] for a in L.HV_PARTITIONS},
            "SCRATCH": M.SCRATCH_LABEL}
    for grid, key in ((M.SUMMARY_GRID, "summary"), (M.SUMMARY_AUC_GRID, "summary_auc")):
        parts = M.table_summary(fits, I["fill"], L._chosen(), I["per"], runs, M.TAB, grid, key).split("%SPLIT%")
        assert len(parts) == 2                     # one turned page a mode: linear probes, then fine-tuned
        modes = []
        for value_of, refs in ((L._probe_value, L.HV_REFS["probe_all"]), (L._ft_value, L.HV_REFS["ft"])):
            get = [value_of(*cell) for cell in grid]
            rows = []
            for row in L.HV_ROWS + L.HV_PARTITIONS + refs:
                if row not in pooled and row != "SCRATCH":
                    continue
                cells = []
                for g, cell in zip(get, grid):
                    v = L._hv_values(row, g, cell)
                    n = len(v) if row == "SCRATCH" else len(pooled[row])
                    cells.append(M._cell(v, n, M.TAB, frac=cell[3] != "r30") if v else "--")
                rows.append(f"{name[row]} & " + " & ".join(cells) + r" \\")
            modes.append("\n".join(rows))
        # one precision per column across both modes (D1, round-1 fix 2026-11-13): a column's linear-probe and
        # fine-tuned cells print at the same place
        wants = M.uniform_columns("\n%MODEBREAK%\n".join(modes)).split("\n%MODEBREAK%\n")
        for part, want in zip(parts, wants, strict=True):
            got = [_unbold(ln) for ln in part.splitlines() if " & " in ln and not ln.startswith(" & ")]
            assert got == want.split("\n"), key
            assert M.V4.TOK_OPEN not in part and r"\pending" not in part and all(ln.count(" & ") == 5 for ln in got)
            # the header: the task over its column, quark/gluon over Pythia and Herwig, then the metric
            head = part.splitlines()
            assert head[2] == r" & JetClass-II & JetClass & top tagging & \multicolumn{2}{c}{quark/gluon tagging} \\"
            assert head[3] == r"\cmidrule(lr){5-6}"
            assert head[4] == " & " + " & ".join(M.SUMMARY_HEAD[met] + {"qg": ", Pythia", "qg_herwig": ", Herwig"}.get(k, "")
                                                 for k, _, _, met in grid) + r" \\"


def test_the_line_and_colour_rule():
    """L6 (PI 2026-11-13): colour identifies the network; solid is fine-tuned (or a figure's only mode), dashed a linear
    probe beside fine-tuned curves, dash-dot the direct discriminant; from scratch grey and solid in every use; the mass
    networks a lighter shade of their label set's colour, solid; the kinematic-inputs network and the released Sophon
    checkpoint their own colours; the random network a dark grey apart from from scratch's; dotted only for reference
    lines; class and decay-family colours reuse no network colour."""
    S = L.S
    assert S.MODE_LINE == {"ft": "-", "probe": "--", "direct": "-."}
    assert S.REFERENCE["linestyle"] == ":" and to_hex(S.REFERENCE["color"]) == "#000000"
    assert all(S.look(k, m)["linestyle"] != ":" for k in S.SERIES for m in S.MODE_LINE)       # no dotted network
    colours = [to_hex(c) for c in S.SERIES.values()]
    assert len(set(colours)) == len(colours) and "#000000" not in colours                     # one colour a network
    for mass, own in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")):
        lighter = S.tint(S.SERIES[own], 0.4)
        assert S.SERIES[mass] == lighter and to_hex(lighter) != to_hex(S.SERIES[own])
        assert all(int(lighter[i:i + 2], 16) >= int(S.SERIES[own][i:i + 2], 16) for i in (1, 3, 5))     # towards white
        assert S.look(mass)["linestyle"] == "-" and S.marker(mass)["markerfacecolor"] == S.SERIES[mass]
    assert (S.SERIES["L162_MASS"], S.SERIES["R16_Q1_MASS"]) == ("#CC8FC2", "#8FCCC2")
    grey = lambda c: len({to_hex(c)[1:3], to_hex(c)[3:5], to_hex(c)[5:7]}) == 1            # noqa: E731
    assert grey(S.SERIES["SCRATCH"]) and grey(S.SERIES["INIT_BN"]) and to_hex(S.SERIES["INIT_BN"]) < to_hex(S.SERIES["SCRATCH"])
    assert all(S.look("SCRATCH", m)["linestyle"] == "-" and S.marker("SCRATCH", m)["markerfacecolor"] == S.SERIES["SCRATCH"]
               for m in S.MODE_LINE)
    assert S.look("KIN", "ft")["linestyle"] == "-" and S.LEGEND["SOPHON"] == "released Sophon checkpoint"
    # a hollow marker only for a linear probe beside fine-tuned networks
    assert S.marker("L188", "probe")["markerfacecolor"] == "white" and S.marker("L188", "ft")["markerfacecolor"] == S.SERIES["L188"]
    assert not {to_hex(c) for c in S.CLASS10} & set(colours)
    assert (S.PROBE, S.MLP_PROBE, S.FINE_TUNED) == ("(linear probe)", "(MLP probe)", "(fine-tuned)")
    assert not hasattr(S, "TINT_188")


def test_no_figure_has_a_heading_and_no_panel_title_a_colon():
    """L7: the caption names the figure, so nothing is written above the legend (style_v5 has no heading any more and
    no gallery module calls one); figcheck's "title" check finds a heading, a colon in a panel title and a text inside
    a panel, and nothing in a figure without them."""
    pytest.importorskip("fitz")                 # figcheck rasterises with PyMuPDF
    import figcheck
    assert not hasattr(L.S, "heading")
    for name in ("gallery_labelset", "gallery_roc", "gallery_physics", "gallery_perclass", "gallery_palette"):
        assert "heading(" not in (ROOT / "experiments/FIGS/figstyle" / f"{name}.py").read_text().replace("_heading(", ""), name

    def fig_with(heading=None, title="(a) Top tagging", inside=None):
        fig = L.S.figure_cm(8.0, 7.0)
        ax = L.S.axes_cm(fig, 1.5, 2.0, 5.0, 4.0)
        ax.plot([0, 1], [0, 1], label="188 classes")
        L.S.title(ax, title)
        L.S.fig_legend(fig, ax.lines, top_cm=L.S.LEGEND_TOP, ncol=1)
        if heading:
            fig.text(0.5, 0.995, heading, ha="center", va="top")
        if inside:
            ax.text(0.5, 0.5, inside)
        found = figcheck.check(fig, "test", 8.0, 1.0)["findings"]["title"]
        matplotlib.pyplot.close(fig)
        return found
    assert fig_with() == []
    assert any(f.startswith("figure heading") for f in fig_with(heading="Linear probes, mean over runs"))
    assert any(f.startswith("colon in the panel title") for f in fig_with(title="(a) X->bb: label sets"))
    assert any(f.startswith("text inside the panel") for f in fig_with(inside="0.98"))


@pytest.mark.parametrize("pid", ["F2", "F2-ft", "PC-jc2", "F4-mech", "A8", "F11", "CONF-share", "F6", "F10b", "PT",
                                 "PT-part", "PT-bbcc", "PT-bbcc-part"])
def test_the_drawn_figures_follow_the_figure_rules(pid, monkeypatch, tmp_path):
    """The main-text figures, three appendix ones and the jet-pT family (round 2) as the gallery draws them: no heading, no colon in a panel title,
    no text in a panel (figcheck "title", L7); every dotted line black (a reference line, L6); from scratch solid; a mass
    network solid, or dash-dot where it is the direct discriminant (F11), never dashed or dotted."""
    pytest.importorskip("fitz")                 # figcheck rasterises with PyMuPDF
    import figcheck
    import gallery_figures as GF
    pages = {p.id: p for mod in GF.MODULES for p in mod.PAGES}
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    try:
        pages[pid].draw(tmp_path / f"{pid}.pdf")
    except L.C.Absent as e:
        pytest.skip(f"{pid}: {e}")
    except SystemExit as e:         # RR-7 (PI 2026-11-23): no training-size figure without the one rule's cells
        if "(RR-7)" not in str(e):
            raise
        pytest.skip(f"{pid}: {e}")
    assert figs, pid
    allowed = {**{to_hex(L.S.SERIES[k]): ("-", "-.") for k in ("L162_MASS", "R16_Q1_MASS")},
               to_hex(L.S.SERIES["SCRATCH"]): ("-",)}
    for fig in figs:
        assert figcheck.check(fig, pid, 15.0, 1.0)["findings"]["title"] == [], pid
        for ax in fig.axes:
            for ln in ax.lines:
                style, colour = ln.get_linestyle(), to_hex(ln.get_color())
                if style == ":":
                    assert colour == "#000000", (pid, ln.get_label())
                if colour in allowed and style not in ("None", " ", ""):
                    assert style in allowed[colour], (pid, ln.get_label(), colour, style)
        matplotlib.pyplot.close(fig)


def test_every_figure_family_and_table_lists_the_networks_in_one_order():
    """L5: 188, 162, 162 + mass, 43, 17, 17 + mass, then the 17-class partitions (no flavor boundary, b-quark boundary,
    random flavor boundary, random partitions), then the references, with the figure legends' names, in every figure
    family and every table that lists networks."""
    import importlib.util
    M = L.M
    spec = importlib.util.spec_from_file_location("xbx_report_order_t", ROOT / "experiments/XBS/xbx_report.py")
    XR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(XR)
    sets = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
    order = sets + ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND")
    # the figure families
    # PI 2026-11-21 (LEDGER PS-5): on a class pair the random partitions split into those that merge it and keep it apart
    assert R.NETWORKS == order + ("RAND_MERGED", "RAND_APART", "INIT", "SCRATCH") and sum(R.LEGEND_GROUPS, ()) == R.NETWORKS
    assert R.LABEL_SETS == L.HV_ROWS == L.SIZE_ROWS == L.VAL_ROWS == H.XBX_LABEL_SETS == H.SCULPT_ROWS == H.PT_ROWS == sets
    # the jet-pT family's partition figure (round 2): 188 and 17 classes for reference, then the partitions
    assert H.PT_PARTITIONS == H.SCULPT_PARTITIONS == ("L188", "R16_Q1") + order[len(sets):]
    assert L.HV_PARTITIONS == order[len(sets):] and H.PROBE_NETS == sets + ("INIT",)
    # PI 2026-11-14: the random partitions drawn once their head outputs exist; F11-part shows every network
    assert H.F10C_ROWS == sets + ("FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND") and H.XBX_PARTITIONS == order
    assert [a for a, _ in PC.LABEL_SETS + PC.PARTITIONS] == ["L162", "R42_Q1", "R16_Q1", *order[len(sets):]]
    assert [a for a, _ in PC.MASS] == ["L162_MASS", "R16_Q1_MASS"]
    # the tables
    assert M.SETS == sets and M.ALL_ROWS == sets + M.FLAV + M.RAND and M._paper_rows(M.ALL_ROWS) == order
    assert M.XBX_ROWS == tuple(XR.ROWS) == order + (M.INIT,) and M.SIZE_ROWS == order and M.MASS_ROWS == sets + (M.INIT,)
    assert M.SUMMARY_CONTROLS == order[len(sets):] and M.RAND_POOLED == "RAND"
    # round 2: the per-bin values (the data release's jet_pt_bins.csv since L18), the confusion percentages
    # (tab:jc1conf, each network against its reference as the confusion figures draw it) and the pair tables' columns
    # in the class-pair figures' order
    pooled = M._pooled(L.RUNS)
    one = {"pt_edges": [200, 300], "bins": [{"accuracy": 0.5, "bb_vs_cc": {"auc": 0.6}}]}
    perf = {sub: {m: one for ms in pooled.values() for m in ms} for sub in M.PT_DIRS}
    blocks = {}
    for ln in M.pt_bins_csv(perf, pooled).splitlines()[1:]:
        name, *key = ln.split(",")[:4]
        blocks.setdefault(tuple(key), []).append(name)
    refs = {M._plain(M.INIT), M._plain(M.INIT_BN), M.SCRATCH_LABEL, "from scratch", "random network"}
    assert len(blocks) == 6 and all(names[:len(order)] == [M._plain(a) for a in order] and set(names[len(order):]) <= refs
                                    for names in blocks.values()), blocks
    assert [a for a, _ in M.CONF_DIFF] == list(order[1:]) and dict(M.CONF_DIFF) == dict(PC.LABEL_SETS + PC.PARTITIONS + PC.MASS)
    assert M.PAIR_ORDER == R.PAIR_ORDER and [k for k, *_ in M.PAIRS_FIG] == list(R.PAIR_ORDER)
    assert sorted(M.PAIRS_FIG) == sorted(M.PAIRS) and M.CONF_CLASSES == L.S.JC1_CLASSES
    # one name for each: the table rows are the legends (the label sets as macros)
    plain = lambda t: t.replace(r"\Kfull{}", "188").replace(r"\Kres{}", "162").replace(r"\Kmid{}", "43").replace(r"\Kcoarse{}", "17")   # noqa: E731
    assert [plain(M.LABEL[a]) for a in order] == [L.S.LEGEND[a] for a in order] == [
        "188 classes", "162 classes", "162 classes + mass", "43 classes", "17 classes", "17 classes + mass",
        "no flavor boundary", "$b$-quark boundary", "random flavor boundary", "random partitions"]
    assert (M.LABEL[M.INIT], M.SCRATCH_LABEL, M.LABEL[M.SOPHON]) == tuple(L.S.LEGEND[k] for k in ("INIT_BN", "SCRATCH", "SOPHON"))


# ------------------------------------------------------------------ round 2 (2026-11-13): the jet-pT family and the tables
# that hold what the figures draw
def _pt_docs() -> dict:
    """The perf_vs_pt folders the jet-pT family draws (gallery_physics.PERF_DIR: the v5 probes once in use, the one
    rule's fine-tuned cells), {folder: {model: doc}}; {} if one is not pulled (the from-scratch JetClass folder,
    jc1_fulldecay, only jet_pt_bins.csv reads)."""
    docs = {sub: L.M._load(H.PERF_DIR[sub] / sub) for sub in L.M.PT_DIRS if sub != "jc1_fulldecay"}
    return docs if all(docs.values()) else {}


def test_the_jet_pt_family_is_two_by_two_with_one_mode_a_row_and_its_slices_in_the_same_form(monkeypatch):
    """PI 2026-11-13 (L4-L7): PT is linear probes over fine-tuned networks, JetClass-II (ten bins, 200-2500 GeV) beside
    JetClass (five bins, 500-1000 GeV), for the label sets and the mass networks; PT-part the same panels for the
    17-class partitions with 188 and 17 classes for reference (the random partitions one point); PT-bbcc the X->bb vs
    X->cc AUC per JetClass-II bin in both modes, PT-bbcc-part its partitions. Every panel has its own labelled y axis,
    no colon in a title, the x range a little wider than the bin edges (_pad_x, L7), the legend in the L5 order,
    nothing else drawn (from scratch and the random network are not in the family)."""
    M = L.M
    assert H.PT_PANELS == (("jc2", "JetClass-II, linear probe"), ("jc1_probe", "JetClass, linear probe"),
                           ("jc2_ft", "JetClass-II, fine-tuned"), ("jc1", "JetClass, fine-tuned"))
    assert [s for s, _ in H.PT_BBCC] == ["jc2", "jc2_ft"]
    assert set(M.PT_DIRS) == {s for s, _ in H.PT_PANELS + H.PT_BBCC} | {"jc1_fulldecay"}     # + from scratch for the CSV
    assert not any(":" in t or "LP" in t or "f.t." in t for _, t in H.PT_PANELS + H.PT_BBCC)
    if not _pt_docs():
        pytest.skip("the perf_vs_pt folders are not pulled")
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    for draw in (H.draw_pt, H.draw_pt_part, H.draw_pt_bbcc, H.draw_pt_bbcc_part):
        draw(None)
    pad = lambda lo, hi: (lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))      # noqa: E731  _pad_x
    pt = {"jc2": pad(200.0, 2500.0), "jc2_ft": pad(200.0, 2500.0), "jc1_probe": pad(500.0, 1000.0), "jc1": pad(500.0, 1000.0)}
    assert len(figs) == 4
    for fig, panels, rows, ylabel in zip(figs, (H.PT_PANELS, H.PT_PANELS, H.PT_BBCC, H.PT_BBCC),
                                         (H.PT_ROWS, H.PT_PARTITIONS, H.PT_ROWS, H.PT_PARTITIONS),
                                         ("Accuracy", "Accuracy", "AUC", "AUC"), strict=True):
        axes = [ax for ax in fig.axes if ax.get_xlabel()]
        assert [ax.get_title() for ax in axes] == [f"({'abcd'[k]}) {t}" for k, (_, t) in enumerate(panels)]
        assert all(ax.get_ylabel() == ylabel and ax.get_xlabel() == H.X_PT for ax in axes)
        assert [v for ax in axes for v in ax.get_xlim()] == pytest.approx([v for s, _ in panels for v in pt[s]])
        shown = [t.get_text() for t in fig.legends[0].get_texts() if t.get_text()]
        assert shown == [L.S.LEGEND[r] for r in rows]
        assert not ({L.S.LEGEND[k] for k in ("SCRATCH", "INIT_BN")} & set(shown))
        matplotlib.pyplot.close(fig)


def test_the_jet_pt_files_hold_every_run_on_one_binning_and_the_jets_of_their_mode():
    """The four folders the family draws: every run of the figure's rows, one pT binning per test set, the fine-tuned
    JetClass-II networks binned on the probes' own jets, the JetClass probes and fine-tuned networks on as many jets per
    bin; the fine-tuned cells the one rule's at 1e6 (ft_v2/fixed, PI 2026-11-15), from scratch among them."""
    docs = _pt_docs()
    if not docs:
        pytest.skip("the perf_vs_pt folders are not pulled")
    M = L.M
    pooled = M._pooled(L.RUNS)
    need = {m for r in H.PT_ROWS + H.PT_PARTITIONS for m in pooled[r]}
    assert len(need) == 27 and all(need <= set(d) for d in docs.values())
    edges = {sub: {tuple(d["pt_edges"]) for d in docs[sub].values()} for sub in docs}
    assert edges["jc2"] == edges["jc2_ft"] == {(200, 300, 400, 550, 700, 850, 1000, 1250, 1500, 2000, 2500)}
    assert edges["jc1_probe"] == edges["jc1"] == {(500, 600, 700, 800, 900, 1000)}
    probe = {d["jets_sha256"] for d in docs["jc2"].values()}
    assert len(probe) == 1 and {d["probe_jets_sha256"] for d in docs["jc2_ft"].values()} == probe
    assert all(d["same_jets_as_probe"] for d in docs["jc2_ft"].values())
    n = lambda sub: {tuple(b["n"] for b in d["bins"]) for m, d in docs[sub].items() if m in need}      # noqa: E731
    assert n("jc2") == n("jc2_ft") and len(n("jc2")) == 1 and n("jc1_probe") == n("jc1") and len(n("jc1")) == 1
    assert {d["mode"] for d in docs["jc1_probe"].values()} == {"linear probe"}
    # the fine-tuning cells: the one rule's at 1e6, every pretrained one and the three from scratch
    scratch = {f"scratch-s{k}" for k in (1, 2, 3)}
    for sub, leg in (("jc2_ft", "leg_jc2b"), ("jc1", "leg2")):
        assert need | scratch <= set(docs[sub]), sub
        assert all(f"/ft_v2/fixed/{leg}/" in d["cell"] and "/N1000000/s" in d["cell"] for d in docs[sub].values()), sub
    assert "pt_bins" not in M.CAPTIONS       # L18: no typeset table, no caption
    # PI 2026-11-14 (step 8): no caption sentence about epoch caps
    assert "epoch" not in PC.SUPPLEMENT_CAPTIONS["PC-jc2-part"]


def test_the_jet_pt_csv_holds_every_point_of_the_jet_pt_figures():
    """L18 (PI 2026-11-13): the per-bin values are the data release's jet_pt_bins.csv, no typeset table: every row of the
    jet-pT figures (PT, PT-part, PT-bbcc, PT-bbcc-part) in every bin and mode, the mean over the runs of exactly the
    values the figures draw (gallery_physics._accuracy, _bb_cc over _runs_in) to six decimals, the sample SD and the
    run count; the generated file is the builder's output on the pulled folders. The captions of the family point to
    the file."""
    import csv
    import io
    import gallery_captions as K
    docs = _pt_docs()
    if not docs:
        pytest.skip("the perf_vs_pt folders are not pulled")
    M = L.M
    pooled = M._pooled(L.RUNS)
    full = {**docs, "jc1_fulldecay": M._load(H.PERF / "jc1_fulldecay")}
    text = M.pt_bins_csv(full, pooled)
    path = ROOT / "paper/journal" / M.PT_CSV
    assert path.read_text() == text                # the released file is this build's
    rows = {(r["network"], r["mode"], r["test_set"], r["quantity"], int(r["pt_low_GeV"])): r
            for r in csv.DictReader(io.StringIO(text))}
    figure = {"accuracy": H._accuracy, "AUC of X->bb against X->cc": H._bb_cc}
    n = 0
    for test_set, quantity, dirs, _ in M.PT_BLOCKS:
        for mode, sub in dirs:
            e = H._edges(docs[sub])
            for a in M.SETS + M.FLAV + (M.RAND_POOLED,):
                drawn = np.array([figure[quantity](d) for d in H._runs_in(docs[sub], a)])
                for i in range(len(e) - 1):
                    r = rows[M._plain(a), M.PT_MODE[mode], test_set, quantity, int(e[i])]
                    assert int(r["pt_high_GeV"]) == int(e[i + 1]) and int(r["n_runs"]) == len(pooled[a]) == len(drawn)
                    assert float(r["mean"]) == pytest.approx(float(np.mean(drawn[:, i])), abs=5e-7), (a, sub, i)
                    assert float(r["sd"]) == pytest.approx(float(np.std(drawn[:, i], ddof=1)), abs=5e-7), (a, sub, i)
                    n += 1
    assert n == 10 * (10 + 10) * 2 + 10 * 5 * 2           # ten networks: JetClass-II two quantities, JetClass one
    assert K.fill(K.RELEASE_CAPTIONS["PT"]).endswith("jet_pt_bins.csv.")      # PI 2026-11-23, LEDGER SP-9: a release page
    assert all("jet_pt_bins.csv" in H.SUPPLEMENT_CAPTIONS[k] for k in ("PT-part", "PT-bbcc", "PT-bbcc-part"))


def test_the_uncertainty_display_of_l21():
    """L21 (PI 2026-11-13: overlapping bands "make some of them very confusing and look bad"): where x takes a few
    values, points with thin vertical +-1 SD bars and no caps, dodged within the bin by one step in every bin, joined by
    a thin line, no bar across the bin; where x is continuous, every band in a panel of at most three series, else the
    bands of 188 classes and the modes' references only (lines_v6.keeps_band; the random network of a probe figure,
    INIT_BN, a reference too). PI 2026-11-13, later: every point of a discrete-x figure keeps its bar (no bars
    argument)."""
    import lines_v6 as LV
    assert LV.BAND_ALL == 3 and LV.BAND_REFS == ("L188", "SCRATCH", "INIT", "INIT_BN")
    assert all(LV.keeps_band(net, 3) for net in ("L162", "R16_Q1_MASS", "FLAV_F1"))
    assert [LV.keeps_band(net, 7) for net in ("L188", "SCRATCH", "INIT", "INIT_BN", "L162", "R16_Q1_MASS")] == \
        [True] * 4 + [False] * 2
    assert "bars" not in inspect.signature(LV.scan).parameters and "bars" not in inspect.signature(LV.binned).parameters
    fig, ax = matplotlib.pyplot.subplots()
    e = np.array([200.0, 300.0, 400.0, 550.0])
    for k in range(3):
        LV.binned(ax, e, np.array([0.5, 0.6, 0.7]), np.full(3, 0.01), np.full(3, 0.02), color="#000000", probe=False,
                  label=str(k), k=k, n=3)
    xs = np.array([ln.get_xdata() for ln in ax.lines])          # one series of points each, no joining line
    centres = 0.5 * (e[1:] + e[:-1])
    step = LV.BIN_SPREAD / 2 * np.diff(e).min()                # the narrowest bin's width shared by the three
    assert np.allclose(xs - centres, np.array([[-step], [0.0], [step]]))           # the same step in every bin
    assert np.all((xs > e[:-1]) & (xs < e[1:]))                                     # inside the bin
    # PI 2026-11-14 (step 8): no connecting lines; a vertical bar per point and a horizontal bar across its bin
    assert all(ln.get_linestyle() in ("None", "none", "") for ln in ax.lines)
    bars = [c for c in ax.collections]
    assert len(bars) == 6 and all(len(c.get_segments()) == 3 for c in bars)
    vertical = [c for c in bars if all(np.allclose(seg[:, 0], seg[0, 0]) for seg in c.get_segments())]
    across = [c for c in bars if c not in vertical]
    assert len(vertical) == len(across) == 3
    assert all(np.allclose([seg[0][0] for seg in c.get_segments()], e[:-1]) and
               np.allclose([seg[1][0] for seg in c.get_segments()], e[1:]) for c in across)
    assert not [c for c in ax.get_children() if type(c).__name__ == "Line2D" and c.get_marker() in ("|", "_")]   # no caps
    matplotlib.pyplot.close(fig)


def test_the_supplement_tables_hold_the_points_their_figures_draw():
    """Round 2 (2026-11-13, one home per plotted number): tab:compute's last block is the last point of the
    pretraining-curve figures (VAL, VAL-part), tab:sculpt's 1/JSD column F10c's vertical axis, and tab:jc1conf every
    matrix of the confusion figures (PC-conf, PC-conf-part, PC-conf-mass): each from the same files by the same
    reduction."""
    M = L.M
    pooled = M._pooled(L.RUNS)
    comp = M._load(L.P.DATA / "compute")
    if comp:
        for row in L.VAL_ROWS + L.VAL_PARTITIONS:
            ms = [m for m in pooled[row] if m in comp and "val_acc" in comp[m]["epochs"][0]]
            last = np.array([[e["val_acc"] for e in comp[m]["epochs"]] for m in ms])[:, -1]
            assert [M._val_acc(comp, m) for m in ms] == list(last), row
        tex = (ROOT / "paper/journal/tables/compute.tex").read_text()
        # the summary: a two-column table of its own after the longtable (page review round 3), its rows below its rule
        block = tex[tex.index(r"\end{longtable}"):]
        block = block[block.index("validation accuracy at the last pretraining epoch"):]
        names = [ln.split(" & ")[0].removeprefix(r"\quad ") for ln in block.splitlines() if " & " in ln]
        assert names == [M.LABEL[a] for a in M.SETS + M.FLAV + (M.RAND_POOLED,)]
    # the per-run sculpting files as the build reads them (head/sculpting_v3), the figure's own inputs on its runs
    sc = {k.removesuffix("_sculpting"): v for k, v in M._load(L.P.DATA / "head/sculpting_v3").items()}
    rows = tuple(r for r in H.F10C_ROWS if r in L.RUNS)      # the random partitions once their head outputs exist
    if sc and all(H.F10C_KEY in sc.get(m, {}) for r in rows for m in L.RUNS[r]):
        assert all(H.I["sculpt"][m][H.F10C_KEY] == sc[m][H.F10C_KEY] for r in rows for m in L.RUNS[r])
        t = M.table_sculpt(sc, L.RUNS, M.TAB, rows)
        for row in rows:
            ln = next(x for x in t.splitlines() if x.startswith(M.LABEL[row] + " & ")).removesuffix(r" \\")
            y = [1 / sc[m][H.F10C_KEY]["jsd"] for m in L.RUNS[row]]
            assert ln.split(" & ")[-1] == M._cell(y, len(L.RUNS[row]), M.TAB), row
    conf = M._load(PC.CONF)
    if conf:
        assert M.CONF_CLASSES == L.S.JC1_CLASSES
        for doc in conf.values():
            assert np.array_equal(M._conf_matrix(doc), 100 * PC._matrix(doc), equal_nan=True)
        seed = lambda m: m.rsplit("-s", 1)[1]      # noqa: E731
        for a, b in M.CONF_DIFF:
            ref = {seed(m): M._conf_matrix(conf[m]) for m in pooled[b]}
            mine = np.mean([M._conf_matrix(conf[m]) - ref[seed(m)] for m in pooled[a]], axis=0)
            assert np.allclose(mine, PC._diff(conf, a, b), equal_nan=True), (a, b)


def test_a_figure_of_one_mode_names_it_in_a_heading_above_its_panels(tmp_path):
    """PI 2026-11-14: a figure that holds one mode (linear probe, fine-tuned, ...) says which in a heading above the
    panels; style_v3.save grows the figure at the top by HEADING_CM and leaves everything drawn where it was."""
    import gallery_figures as F
    import style_v3
    import style_v5
    ids = {p.id for m in F.MODULES for p in m.PAGES}
    assert set(F.MODE_HEADINGS) <= ids and style_v3.HEADINGS == F.MODE_HEADINGS
    assert F.MODE_HEADINGS["F2"] == "Linear probe" and F.MODE_HEADINGS["F2-ft"] == "Fine-tuned"
    # the figures that hold both modes name them in their panels instead
    assert not {"PT", "PC-jc2", "PC-jc1", "F8"} & set(F.MODE_HEADINGS)
    fig = style_v5.figure_cm(15.0, 6.0)
    ax = style_v5.axes_cm(fig, 1.0, 1.0, 5.0, 4.0)
    leg = style_v5.fig_legend(fig, [matplotlib.lines.Line2D([], [], label="x")], top_cm=0.15, ncol=1)
    fig.canvas.draw()
    before = (ax.get_window_extent().y1, leg.get_window_extent().y1)
    style_v3._add_heading(fig, "Linear probe")
    fig.canvas.draw()
    cm = 1 / 2.54 * fig.dpi
    assert fig.get_figheight() * 2.54 == pytest.approx(6.0 + style_v3.HEADING_CM)
    # distances from the bottom unchanged, so from the top each grew by HEADING_CM
    assert (ax.get_window_extent().y1, leg.get_window_extent().y1) == pytest.approx(before, abs=0.01 * cm)
    assert [t.get_text() for t in fig.texts] == ["Linear probe"]
    assert fig.texts[0].get_fontstyle() == "italic"            # CA-12: the mode as an italic heading
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("draw", ["draw_roc_main", "draw_roc_main_ft"])
def test_the_main_roc_figures_hold_every_jetclass_class_against_qcd_and_fit_the_page(monkeypatch, draw):
    """PI 2026-11-18: one figure a mode, the nine JetClass signal classes against QCD jets, then top and quark/gluon,
    in a 4 x 3 grid with the legend in the free slot; every panel the label sets and mass networks."""
    figs = []
    monkeypatch.setattr(R.S, "save", lambda fig, path: figs.append(fig))
    getattr(R, draw)(None)
    fig = figs[0]
    panels = [ax for ax in fig.axes if ax.get_title()[:1] == "("]
    assert len(panels) == 11 and len(R.MAIN_JC1) == 9 and "label_QCD" not in R.MAIN_JC1
    assert [ax.get_title()[4:].split(", pending")[0] for ax in panels[:9]] == [R.S.JC1_TEX[c] for c in R.MAIN_JC1]
    assert len({ax.get_ylim() for ax in panels}) == 1 and len(fig.legends) == 1
    import make_results_book as B
    assert fig.get_figheight() / R.S.CM <= B.MAX_H_CM


def test_fig_9s_widening_clause_appears_only_when_its_3_sigma_test_passes():
    """PI 2026-11-21: "the gap widens below 0.1% for X->bs" only if the widening of the 188 - 17 figure-of-merit gap
    over its value at 1% QCD efficiency passes 3 sigma (run spreads, the QCD-jet and signal counting terms) at 1e-3 and
    1e-4 (gallery_physics.xbs_widening)."""
    import gallery_physics as G
    import gallery_captions as K
    w = G._XBS_WIDEN
    text = K.fill(K.CAPTIONS["F11"])
    assert ("widens below 0.1% for X→bs" in text) == bool(w["pass"])
    if "widening" in w:
        assert w["pass"] == all(d > 3 * s for d, s in w["widening"].values())
