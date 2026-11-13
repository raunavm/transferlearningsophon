"""The gallery's map and the pages drawn from shared parts (experiments/FIGS/figstyle): the map's rows and blocks
(gallery_inventory), the class-pair family's panels and legend (gallery_roc._pair_grid, _roc_grid), the per-class
family's one axis range and colours (gallery_perclass._draw_grid), the line and colour rule (style_v5, L6), the
figure-text rules (L7), the one row order of figure families and tables (L5) and Table 1 against the headline
family's value functions (L2). Imports the gallery modules, which read the committed figure data (about a second)."""
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
    span = f"{G.number('FJ-ctrl')}–{G.number('FJ-s').split()[-1]}"
    assert span == "Fig. S19–S22" and V._rows(tuple(supp)) == [(span, "FJ-ctrl, FJ-app, FJ-ctrl-app, FJ-s", G.BACKS["FJ-ctrl"])]
    # with the tables it also rests on
    flav = tuple(i for i in G.FIGURES if i[0] in ("F10b-flav", "F10c", "F10a-flav"))
    rows = V._rows(flav)
    also = f" [also {G.number('tab:sculpt')}, {G.number('tab:sculptboot')}]"
    # one sentence for the supplement's sculpting figures (PI 2026-11-12 cut): consecutive, so one row
    assert [r[0] for r in rows] == [f"{G.number('F10b-flav')}–{G.number('F10a-flav').split()[-1]}"]
    assert rows[0][2].endswith(also)
    assert V._rows(tuple(i for i in G.FIGURES if i[0] == "F10b"))[0][1] == "F10b"


def test_the_map_places_every_table_once_and_lists_the_rest():
    placed = [k for t, *_ in G.TABLES for k in t.split(" + ")]
    labels = {label for label, *_ in V.C.M.CAPTIONS.values()}
    assert len(placed) == len(set(placed)) and set(placed) <= labels      # every placed label is a table the book makes
    assert set(V.unplaced()) == labels - set(placed) - set(G.DROPPED_TABLES) == set()
    assert set(G.DROPPED_TABLES) <= labels
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
    # F10b's verdict needs the signs: 17 + mass - 17 is +1.22 here, so the appendix; the rule and its two z follow
    assert rows[-1][1] == ("F10b: appendix (main text only if at QCD 1% 17 − 188 > +3σ and 17 + mass − 17 < −3σ; "
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
    tier = "main text" if G.f10b_main(diffs) else "appendix"
    assert rows[-1][1].startswith(f"F10b: {tier} (") and rows[-1][1].endswith(f"here {z[0]} and {z[1]}).")
    assert tier == G.TIER_NAME[G.TIER["F10b"]].lower()          # the placement follows the test


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
    classes; the same pairs and panel order for linear probes (F4-mech, F4-other) and fine-tuned (F4-mech-ft,
    F4-other-ft)."""
    calls = []
    monkeypatch.setattr(R, "ROC", {"pairs": {"x": {}}, "pairs_ft": {"x": {}}})
    monkeypatch.setattr(R, "_roc_grid", lambda path, grid, **kw: calls.append((grid, kw)))
    for draw in (R.draw_f4_mech, R.draw_f4_mech_ft, R.draw_f4_other, R.draw_f4_other_ft):
        draw(None)
    assert R.PAIR_ORDER[:2] == R.MECHANISM == ("bb_vs_cc", "bc_vs_bq_cs") and R.OTHER == R.PAIR_ORDER[2:6]
    assert R.LABEL_SETS == L.M.SETS and R.PARTITIONS == L.M.FLAV + ("L188", "R16_Q1")
    for (grid, kw), pairs, ft in zip(calls, (R.MECHANISM, R.MECHANISM, R.OTHER, R.OTHER), (False, True, False, True)):
        assert [[(key, nets) for _, key, nets, _ in row] for row in grid] == [
            [(k, R.LABEL_SETS + (("SCRATCH",) if ft else ())) for k in pairs], [(k, R.PARTITIONS) for k in pairs]]
        assert kw["row_names"] == ("Label sets", "17-class partitions")
        assert all(":" not in title for row in grid for *_, title in row)            # L7: no colon in a panel title
    assert [k for k, *_ in calls[0][0][0]] == [k for k, *_ in calls[1][0][0]]   # one pair order in both modes


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
    # one legend column a group of networks (L5 order), blank entries below a short column
    named = [t for t in shown if t]
    assert named == sorted(named, key=lambda t: [R.S.LEGEND[R._style(n)] for n in R.NETWORKS].index(t))


def test_the_per_class_grid_puts_every_panel_on_one_range_and_colours_classes_not_networks(monkeypatch):
    """Per-class family (PI 2026-11-12/13): linear probe over fine-tuned, a column per compared network, one axis range
    for the grid from the lowest AUC drawn; a point's colour is its decay family's, never a colour that means a
    network (L6); the diagonal is the black dotted reference."""
    classes = range(188)
    monkeypatch.setattr(PC, "SOURCES", {"jc2": (lambda m: {c: 0.81 + 0.001 * c for c in classes},
                                                lambda m: {c: 0.90 + 0.0005 * c for c in classes})})
    figs = []
    monkeypatch.setattr(PC.S, "save", lambda fig, path: figs.append(fig))
    PC.draw_jc2(None)
    panels = [ax for ax in figs[0].axes if ax.get_title()[:3] in {f"({x})" for x in "abcdef"}]
    lo = 0.81 - 0.04 * (1 - 0.81)          # the lowest AUC drawn, 4% of its distance to 1 below it
    assert len(panels) == 6 and all(ax.get_xlim() == ax.get_ylim() == panels[0].get_xlim() for ax in panels)
    assert panels[0].get_xlim()[0] == pytest.approx(lo)
    networks = {to_hex(c) for c in PC.S.SERIES.values()}
    for ax in panels:
        diagonal, *points = ax.lines
        assert diagonal.get_linestyle() == ":" and to_hex(diagonal.get_color()) == "#000000"
        assert len(points) == 188 and not {to_hex(ln.get_color()) for ln in points} & networks
        assert {to_hex(ln.get_color()) for ln in points} <= {to_hex(c) for c in PC.S.CLASS10[:len(R.FAMILIES)]}
    matplotlib.pyplot.close(figs[0])


def test_the_paper_captions_replace_the_page_captions_and_name_tables_that_are_placed():
    """gallery_captions (PI 2026-11-12): one caption for each figure of the main text and the appendix, set on the
    gallery's pages; a table is named by its label and printed with its number of the day, so every label must be a
    placed table (2026-11-13: the ROC figure points to Table 1 and its AUC table for R30 and the AUC, to tab:probes and
    tab:labelsets for the accuracy and R50)."""
    import re
    import gallery_captions as K
    paper = [k for k, t, _ in G.FIGURES if t in (G.MAIN, G.APPENDIX)]
    assert sorted(K.CAPTIONS) == sorted(paper) and len(paper) == 20
    for pid, raw in K.CAPTIONS.items():
        labels = re.findall(r"\{(tab:[a-z_0-9]+)\}", raw)
        assert set(labels) <= set(G.TIER) and not set(labels) & set(G.DROPPED_TABLES), pid
        text = K.text(pid)
        assert "{tab:" not in text and "Table " not in text and len(text.split()) >= 40, pid
    assert re.findall(r"\{(tab:[a-z_0-9]+)\}", K.CAPTIONS["F6"]) == ["tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets"]
    number = lambda k: G.number(k).removeprefix("Table ")      # noqa: E731
    assert K.text("F6").endswith(f"R30 and the area under the ROC curve are in tables {number('tab:summary')} and "
                                 f"{number('tab:summaryauc')}, the rejection at 50% signal efficiency in tables "
                                 f"{number('tab:probes')} and {number('tab:labelsets')}.")
    assert [number(k) for k in ("tab:summary", "tab:summaryauc", "tab:probes", "tab:labelsets")] == ["1", "A1", "A2", "A10"]
    # the headline family's captions name Table 1 and the AUC table (L2)
    assert all(K.text(k).endswith(f"tables {number('tab:summary')} and {number('tab:summaryauc')}.") for k in ("F2a", "F2b"))
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
        for part, (value_of, refs) in zip(parts, ((L._probe_value, L.HV_REFS["probe_all"]), (L._ft_value, L.HV_REFS["ft"]))):
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
            want = M.uniform_columns("\n".join(rows)).split("\n")
            got = [_unbold(ln) for ln in part.splitlines() if " & " in ln and not ln.startswith(" & ")]
            assert got == want, key
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


@pytest.mark.parametrize("pid", ["F2", "PC-jc2", "F4-mech", "A8", "F11", "F6", "F10b"])
def test_the_drawn_figures_follow_the_figure_rules(pid, monkeypatch, tmp_path):
    """The main-text figures and three appendix ones as the gallery draws them: no heading, no colon in a panel title,
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
    assert R.NETWORKS == order + ("INIT", "SCRATCH") and sum(R.LEGEND_GROUPS, ()) == R.NETWORKS
    assert R.LABEL_SETS == L.HV_ROWS == L.SIZE_ROWS == L.VAL_ROWS == H.XBX_LABEL_SETS == H.SCULPT_ROWS == sets
    assert L.HV_PARTITIONS == order[len(sets):] and H.PROBE_NETS == sets + ("INIT",)
    assert H.F10C_ROWS == sets + ("FLAV_F0", "FLAV_F1", "FLAV_F1R")
    assert [a for a, _ in PC.LABEL_SETS + PC.PARTITIONS] == ["L162", "R42_Q1", "R16_Q1", *order[len(sets):]]
    assert [a for a, _ in PC.MASS] == ["L162_MASS", "R16_Q1_MASS"]
    # the tables
    assert M.SETS == sets and M.ALL_ROWS == sets + M.FLAV + M.RAND and M._paper_rows(M.ALL_ROWS) == order
    assert M.XBX_ROWS == tuple(XR.ROWS) == order + (M.INIT,) and M.SIZE_ROWS == order and M.MASS_ROWS == sets + (M.INIT,)
    assert M.SUMMARY_CONTROLS == order[len(sets):] and M.RAND_POOLED == "RAND"
    # one name for each: the table rows are the legends (the label sets as macros)
    plain = lambda t: t.replace(r"\Kfull{}", "188").replace(r"\Kres{}", "162").replace(r"\Kmid{}", "43").replace(r"\Kcoarse{}", "17")   # noqa: E731
    assert [plain(M.LABEL[a]) for a in order] == [L.S.LEGEND[a] for a in order] == [
        "188 classes", "162 classes", "162 classes + mass", "43 classes", "17 classes", "17 classes + mass",
        "no flavor boundary", "$b$-quark boundary", "random flavor boundary", "random partitions"]
    assert (M.LABEL[M.INIT], M.SCRATCH_LABEL, M.LABEL[M.SOPHON]) == tuple(L.S.LEGEND[k] for k in ("INIT_BN", "SCRATCH", "SOPHON"))
