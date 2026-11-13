"""experiments/XBS/xbx_report.py: the resonance-tagging figure, table and macros from fit_xbx.py's
per-model files (synthetic here, in fit_xbx.py's layout)."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


XR = _mod("xbx_report", "experiments/XBS/xbx_report.py")
M = _mod("mlst_results", "experiments/FIGS/mlst_results.py")
MT = _mod("make_tables", "experiments/FIGS/make_tables.py")
EPS_B = [1e-4, 1e-3, 1e-2, 1e-1]


def _end(eff, n_s=1000, n_b=500_000):
    return {"n_s": n_s, "n_b": n_b, "auc": 0.9, "eps_b_star": 1e-3, "eps_s_at_star": eff,
            "curve": {"eps_b": EPS_B, "eps_s_over_sqrt_eps_b": [eff / e ** 0.5 for e in EPS_B]}}


def _sig(eff, own=False, window=(93.75, 143.75)):
    s = {"n_train_per_class": 277_156, "repetition": 4.51, "window": list(window),
         "linear": _end(eff / 2), "head": [_end(eff - 0.01), _end(eff), _end(eff + 0.01)]}
    if own:
        s["zero_shot_xbb"] = _end(eff / 3)
    s["zero_shot_xbq"] = _end(eff / 4)
    return s


@pytest.fixture
def fits_dir(tmp_path):
    runs = M._paper(REPO)
    d = tmp_path / "fits"
    d.mkdir()
    for i, m in enumerate([m for ms in runs.values() for m in ms] + list(M.INIT_MODELS)):
        eff = 0.2 + 0.001 * i
        # every model carries the key, as the stand-in fits did with a merged node below 162
        sigs = {"xbb_jc2": _sig(eff, own=True)}
        if m.startswith("init-"):
            sigs = {"xbb_jc2": {k: v for k, v in sigs["xbb_jc2"].items() if k != "head"}}
        (d / f"{m}.json").write_text(json.dumps({"model": m, "signals": sigs}))
    return d


def test_the_transfer_seeds_are_averaged_and_the_untrained_network_uses_its_probe(fits_dir):
    runs = M._paper(REPO)
    fits = XR.load(fits_dir, {"xbb_jc2": "xbb"})
    assert set(next(iter(fits.values()))) == {"xbb"}
    m = runs["L188"][0]
    eff = fits[m]["xbb"]["head"][1]["eps_s_at_star"]
    assert XR.model_readout(fits, m, "xbb")["eps_s_at_star"] == pytest.approx(eff)
    init = XR.model_readout(fits, "init-s1", "xbb")
    assert init["eps_s_at_star"] == fits["init-s1"]["xbb"]["linear"]["eps_s_at_star"]
    e = XR.efficiencies(M, runs, fits, "xbb")
    assert len(e["RAND"]) == sum(len(runs[a]) for a in M.RAND) and len(e["INIT"]) == len(M.INIT_MODELS)
    direct = XR.efficiencies(M, runs, fits, "xbb", "zero_shot_xbb")
    assert direct["R16_Q1"] == [] and len(direct["L188"]) == len(runs["L188"])


def test_the_table_and_macros(fits_dir, tmp_path):
    runs = M._paper(REPO)
    fits = XR.load(fits_dir, {"xbb_jc2": "xbb"})
    tex = XR.table(M, MT, runs, fits)
    # the signal over its read-outs, the working point once across the columns (PI 2026-11-11); since 2026-11-13 the
    # classifier on the frozen features, the linear probe on them and the direct discriminant
    assert r"\multicolumn{3}{c}{$X\to b\bar b$}" in tex and "pretraining & classifier & linear probe & direct" in tex
    assert "bs" not in tex and "transfer" not in tex
    row17 = next(line for line in tex.splitlines() if line.startswith(M.LABEL["R16_Q1"] + " &"))
    assert row17.endswith(r"-- \\")             # no direct discriminant at 17 classes
    em = MT.Emitter(pathlib.Path("/"))
    XR.macros(em, MT, fits, next(fits_dir.glob("*.json")).resolve())
    got = {n: b for n, b, _ in em.macros}
    assert got["XbxMx"] == "125" and got["XbxWindowLo"] == "93.75" and got["XbbRepetition"] == "4.5"
    assert got["XbbNEvalQcd"] == MT.fmt_int(500_000)
    paths = XR.figure(M, _mod("mlst_style", "experiments/FIGS/mlst_style.py"), runs, fits, tmp_path / "out")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_mixed_windows_are_refused(fits_dir):
    f = next(fits_dir.glob("mtx-l188-s1.json"))
    d = json.loads(f.read_text())
    d["signals"]["xbb_jc2"]["window"] = [150.0, 230.0]
    f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="mass windows"):
        XR.macros(MT.Emitter(pathlib.Path("/")), MT, XR.load(fits_dir), f.resolve())


def test_the_curve_stops_where_too_few_qcd_jets_pass():
    # 10 QCD jets pass at 1e-4 of 100,000, fewer than MIN_BKG_PASS (20); 100 at 1e-3
    assert XR.MIN_BKG_PASS == 20
    x, keep = XR._kept({"n_b": 100_000, "curve": {"eps_b": EPS_B}})
    assert list(x) == EPS_B[1:] and keep.tolist() == [False, True, True, True]
    assert XR._kept({"n_b": 500_000, "curve": {"eps_b": EPS_B}})[1].tolist() == [True] * 4


def test_the_rows_follow_the_tables_one_order_and_a_single_from_scratch_seed_is_marked(fits_dir):
    """PI 2026-11-13 (L5): the efficiency table in the row order of every table (the label sets with each network with
    a mass output under its own, the 17-class partitions, the random network, from scratch last, in the classifier's
    column); with the tables' formatter a from-scratch signal with one seed is a single-run cell, printed with the
    single-run mark, and three seeds are not."""
    import types
    runs = M._paper(REPO)
    fits = XR.load(fits_dir, {"xbb_jc2": "xbb"})
    assert tuple(XR.ROWS) == M.XBX_ROWS == M.SETS + M.FLAV + (M.RAND_POOLED,) + (M.INIT,)
    tab = types.SimpleNamespace(fmt_pm=M.TAB)
    tex = XR.table(M, tab, runs, fits, {"xbb": [_end(0.477)]})
    lines = tex.splitlines()
    assert [ln.split(" & ")[0] for ln in lines[lines.index(r"\midrule") + 1:-2]] == [M.LABEL[r] for r in XR.ROWS] + [M.SCRATCH_LABEL]
    assert lines[-3] == f"{M.SCRATCH_LABEL} & " + M.V4.tok("s", 0.477 * 100.0) + r" & -- & -- \\"
    printed = M.uniform_columns(tex).splitlines()
    assert M.V4.TOK_OPEN not in "".join(printed) and printed[-3].split(" & ")[1].endswith(M.SINGLE)
    assert sum(M.SINGLE in ln for ln in printed) == 1 and "$<$" not in "".join(printed)
    three = M.uniform_columns(XR.table(M, tab, runs, fits, {"xbb": [_end(0.47), _end(0.48), _end(0.49)]}))
    assert M.SINGLE not in three and three.splitlines()[-3].startswith(M.SCRATCH_LABEL + r" & 48.")
    assert "pretraining & classifier & linear probe & direct" in tex       # the header is unchanged
    # the random network has only the linear probe: a dash in the classifier's and the direct column
    init = next(ln for ln in printed if ln.startswith(M.LABEL[M.INIT] + " & ")).removesuffix(r" \\").split(" & ")
    assert init[1] == init[3] == "--" and init[2] != "--"
