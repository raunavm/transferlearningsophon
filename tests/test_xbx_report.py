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
        sigs = {"xbb_jc2": _sig(eff, own=m in runs["L188"] + runs["L162"])}
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
    assert r"$X\to b\bar b$, transfer" in tex and r"$X\to b\bar b$, direct" in tex and "bs" not in tex
    row17 = next(line for line in tex.splitlines() if line.startswith(M.LABEL["R16_Q1"] + " &"))
    assert row17.endswith(r"--- \\")            # no direct discriminant at 17 classes
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
