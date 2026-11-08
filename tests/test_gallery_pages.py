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
    assert V._rows(tuple(i for i in G.FIGURES if i[0] == "F10b"))[0][1] == f"F10b; {G.PROVISIONAL['F10b']}"


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
    assert len(rows) == 2 + len(V.BOOT_DIFFS) + 1 and len(V.BOOT_X) == 7
    assert rows[0][1] == ("", "", "QCD 1%", "", "QCD 10%", "", "res. 50%")
    assert rows[1][1] == ("Difference", "Δ", "Δ/σ", "Δ", "Δ/σ", "Δ", "Δ/σ")
    assert rows[2][1] == ("17 − 188", "1.22", "4.2*", "1.22", "4.2*", "1.22", "4.2*")
    assert rows[3][1] == ("43 − 188", "1.22", "4.2*", "1.22", "4.2*", "−0.20", "–")
    # F10b's verdict needs the signs: 17 + mass - 17 is +1.22 here, so the appendix
    assert rows[-1][1] == "F10b: appendix."
    doc["differences"]["17mass-17"]["eps_b=0.01"]["distance"] = {**q, "delta": -0.0122, "z": -4.15}
    (tmp_path / "sculpting_boot.json").write_text(json.dumps(doc))
    rows = V._boot_rows(tmp_path / "sculpting_boot.json")
    assert rows[4][1][1:3] == ("−1.22", "−4.2*") and rows[-1][1] == "F10b: main text."


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
