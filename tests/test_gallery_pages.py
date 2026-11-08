"""The gallery's map and the pages drawn from shared parts (experiments/FIGS/figstyle): the map's rows and blocks
(gallery_inventory), F4-mech's panels and legend (gallery_roc._mechanism, _pairs) and F3-main's one axis range
(gallery_labelset._draw_scatter). Imports the gallery modules, which read the committed figure data (about a second)."""
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
import gallery_placement as G  # noqa: E402
import gallery_roc as R  # noqa: E402


def test_rows_merge_consecutive_items_of_one_sentence_and_carry_their_evidence():
    supp = [i for i in G.FIGURES if i[2] == G.BACKS["FJ-ctrl"]]
    assert V._rows(tuple(supp)) == [("Fig. S1–S4", "FJ-ctrl, FJ-app, FJ-ctrl-app, FJ-s", G.BACKS["FJ-ctrl"])]
    # one sentence, not consecutive: two rows, each with the table it also rests on
    flav = tuple(i for i in G.FIGURES if i[0] in ("F10b-flav", "F10c", "F10a-flav"))
    rows = V._rows(flav)
    also = f" [also {G.number('tab:sculpt')}, {G.number('tab:sculptboot')}]"
    assert [r[0] for r in rows] == ["Fig. A11", "Fig. A12", "Fig. A13"] and rows[0][2] == rows[2][2] == G.SCULPT_FLAV + also
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
    assert [r[1][0] for r in rows[2:5]] == ["188 classes (f.t.)", "17 classes (f.t.)", "no pretraining (f.t.)"]
    assert rows[4][1] == ("no pretraining (f.t.)", "–", "0.405(7)", "0.405(7)", "0.405(7)")
    assert all(r[2] == V.FT_X for r in rows[:5]) and rows[-1][1].startswith("No pretraining: ParT's full flat+decay")
    assert rows[-1][1].endswith("–: pending: JetClass-II accuracy, no pretraining.")
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


def test_the_mechanism_figure_is_the_two_pairs_by_label_sets_and_flavor_boundaries(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "_pairs", lambda path, family, panels, heading: calls.append((family, panels)))
    R.draw_f4_mech(None)
    R.draw_f4_mech_ft(None)
    assert [c[0] for c in calls] == ["pairs", "pairs_ft"] and calls[0][1] == calls[1][1]
    panels = calls[0][1]
    assert [(rows, key) for rows, key, _ in panels] == [(R.PAIR_LABEL_SETS, "bb_vs_cc"), (R.PAIR_LABEL_SETS, "bc_vs_bq_cs"),
                                                        (R.PAIR_BOUNDARIES, "bb_vs_cc"), (R.PAIR_BOUNDARIES, "bc_vs_bq_cs")]
    assert set(R.PAIR_LABEL_SETS) == set(R.S.LEVELS) and R.PAIR_BOUNDARIES == tuple(R.S.CONTROLS) + ("L188",)


def test_the_pair_legend_holds_only_the_curves_drawn(monkeypatch):
    """A model with no curve in any panel, or only a complete separation, has no legend entry."""
    eps = np.linspace(0.1, 1.0, 10)
    curve = {"L188": (eps, 1000 * (1 - eps) + 1, 0.9), "L162": None, "R42_Q1": (eps[:1], eps[:1], 1.0)}
    monkeypatch.setattr(R, "ROC", {"pairs": {"any": {}}})
    monkeypatch.setattr(R, "RUNS", {a: [a] for a in R.S.LEVELS + tuple(R.S.CONTROLS)})
    monkeypatch.setattr(R, "mean_over_runs", lambda docs, models, key: curve.get(models[0], (eps, 50 * (1 - eps) + 1, 0.8)))
    shown = []
    monkeypatch.setattr(R.S, "fig_legend", lambda fig, handles, **kw: shown.extend(h.get_label() for h in handles))
    monkeypatch.setattr(R.S, "save", lambda fig, path: matplotlib.pyplot.close(fig))
    R.draw_f4_mech(None)
    assert R.S.LEGEND["L162"] not in shown and R.S.LEGEND["R42_Q1"] not in shown
    assert R.S.LEGEND["L188"] in shown and R.S.LEGEND["R16_Q1"] in shown


def test_f3_main_puts_both_panels_on_one_range(monkeypatch):
    classes = range(188)
    probe = lambda models: {c: 0.80 + 0.001 * c for c in classes}
    ft = lambda models: {c: 0.90 + 0.0005 * c for c in classes}
    monkeypatch.setattr(L, "_class_auc", probe)
    monkeypatch.setattr(L, "_ft_class_auc", ft)
    figs = []
    monkeypatch.setattr(L.S, "save", lambda fig, path: figs.append(fig))
    L.draw_f3_main(None)
    a, b = [ax for ax in figs[0].axes if ax.get_title().startswith(("(a)", "(b)"))]
    lo = 0.80 - 0.04 * (1 - 0.80)
    assert a.get_xlim() == b.get_xlim() == a.get_ylim() == b.get_ylim() and a.get_xlim()[0] == pytest.approx(lo)
    matplotlib.pyplot.close(figs[0])
