"""experiments/FIGS/paper_figures.py, the paper's final figures: every figure draws from a small
synthetic set of inputs in the layouts the generator reads (the loaders are replaced, so the tests
do not depend on how much has been pulled), a figure whose input is absent is skipped without
failing, the style is Sophon's at the printed size with nothing but data drawn, every ROC panel
follows the curves_v2 files' cut and pairs its ratio run by run, and the render job runs the paper
build while its mock-up spec stays as it was."""
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
N6 = "1000000"
# probe sizes per data set (v4 layout)
SIZES = {"jc2bal": ("1000", N6), "jc1": ("1000", N6), "top": ("1000", "1200000"), "qg": ("1000", "1600000"),
         "qg_herwig": ("1000", "1600000")}
FT_SIZES = {"jc2": (10_000, 1_000_000), "jc1": (10_000, 1_000_000), "top": (1_000, 1_200_000),
            "qg": (1_000, 1_600_000), "qg_herwig": (1_000, 1_600_000)}
# the commit the mock-up spec is compared with (the last before the paper build was added)
BASE = "62fdbb5"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def P():
    return _mod("paper_figures", "experiments/FIGS/paper_figures.py")


@pytest.fixture(scope="module")
def M(P):
    return P.F._mlst()


@pytest.fixture(scope="module")
def runs(M):
    return M._paper(REPO)


def _pretrained(M, runs) -> list:
    return [m for a, ms in runs.items() if a not in (M.INIT, M.INIT_BN) for m in ms]


def _rec(i: int, k: int) -> dict:
    """A probe cell: k orders the training sizes (and the substitutions), i the models."""
    return {"C": 1.0, "converged": True, "accuracy": max(0.05, 0.4 + 0.01 * k - 1e-3 * i),
            "macro_auc_ovr": 0.97 + 2e-3 * k - 1e-4 * i, "auc": 0.9 + 1e-3 * k, "r50": 10.0 + k,
            "r30": max(5.0, 30.0 + 10 * k + i),
            "per_class_auc": {str(c): 0.95 - 1e-4 * i - 1e-3 * (c % 5) for c in range(188)},
            "search": {"1.0": {"val_cross_entropy": 0.5 + 0.01 * k + 1e-3 * i}}}


def _fits(M, runs, root: pathlib.Path) -> tuple:
    """fits_v4 files of every paper run, the randomly initialized networks (init and init_bn) and the
    kinematic-only model, read back through the generator's loader (mlst_v4.load_fits)."""
    d = root / "linprobe/fits_v4"
    d.mkdir(parents=True)
    docs = {**{m: ("best70",) for m in _pretrained(M, runs) + [M.KIN_MODEL]},
            **{m: ("init", "init_bn") for m in M.INIT_MODELS}}
    for i, (m, cks) in enumerate(docs.items()):
        cells = {}
        for ck in cks:
            c = {ds: {n: _rec(i, k) for k, n in enumerate(ns)} for ds, ns in SIZES.items()}
            c["jc2bal.mtop"] = {N6: _rec(i, -30)}
            c["jc2bal.mqg"] = {N6: _rec(i, -10)}
            c["jc2bal_pairs"] = {key: {"auc": 0.99 - 1e-3 * j - 1e-4 * i, "converged": True}
                                 for j, (key, *_) in enumerate(M.PAIRS)}
            cells[ck] = c
        (d / f"{m}.json").write_text(json.dumps({"cells": cells, "c_grid": [1.0]}))
    return M.V4.load_fits(root)


def _fill(M, runs) -> dict:
    """fits_v4_fill in its own layout: the three substitutions of top, quark/gluon and the JetClass-II slots."""
    out = {}
    models = {**{m: "best70" for m in _pretrained(M, runs)}, **{m: "init_bn" for m in M.INIT_MODELS}}
    for i, (m, ck) in enumerate(models.items()):
        c = {}
        for task in ("top", "qg", "qg_herwig"):
            for k, key in enumerate(M.SUBST_KEY[task].values()):
                c[key] = {n: _rec(i, k + j) for j, n in enumerate(SIZES[task])}
        c["jc2bal"] = {N6: _rec(i, 0)}
        for task, keys in M.JC2_SUBST.items():
            for k, key in enumerate(keys.values()):
                c[key] = {N6: _rec(i, -30 + 5 * k)}
        out[m] = {"cells": {ck: c}}
    return out


def _mlp(M, runs) -> dict:
    out = {m: {"cells": {"best70": {"top": {"r30": 41.0, "auc": 0.91}, "qg": {"r30": 21.0, "auc": 0.8},
                                     "bc_vs_bq_cs": {"auc": 0.96 - 1e-3 * i}}}}
           for i, m in enumerate(_pretrained(M, runs))}
    out.update({m: {"cells": {"init_bn": {"top": {"r30": 30.0}, "qg": {"r30": 15.0}}}} for m in M.INIT_BN_MODELS})
    return out


def _per(M, runs) -> dict:
    rec = lambda k: {"accuracy": 0.5 + 0.01 * k, "macro_auc_ovr": 0.98 + 1e-3 * k, "r30": 100.0 + 10 * k}
    per = {}
    for key, ns in FT_SIZES.items():
        cells = {m[len("mtx-"):]: {n: [rec(i % 7)] for n in ns} for i, m in enumerate(_pretrained(M, runs))}
        cells[M.SCRATCH] = {n: [rec(-1), rec(-2), rec(-3)] for n in ns}
        per[key] = cells
    return per


def _summary(r: float) -> dict:
    return {"median": 1.0, "q159": 1 - r, "q841": 1 + r, "resolution": r, "n": 1000, "median_sd": 1e-3,
            "resolution_sd": 2e-3}


def _binned(r: float, edges: list) -> dict:
    """A binned block with its last bin below the jet minimum (null values)."""
    nb = len(edges) - 1
    return {"edges": edges, **{k: [v] * (nb - 1) + [None] for k, v in _summary(r).items() if k != "n"},
            "n": [1000] * (nb - 1) + [50]}


def _group(r: float) -> dict:
    e = np.linspace(0.0, 2.0, 101)
    mid = (e[:-1] + e[1:]) / 2
    counts = (1e4 * np.exp(-0.5 * ((mid - 1) / r) ** 2)).astype(int).tolist()
    return {"inclusive": _summary(r), "vs_pt": _binned(r, [200, 400, 700, 1000, 2500]),
            "vs_mgen": _binned(r, [20, 50, 100, 150, 250, 500]),
            "hist": {"edges": e.tolist(), "counts": counts, "underflow": 5, "overflow": 7}}


def _atlas(M, runs) -> dict:
    out = {}
    for a in M.MAIN + M.MASS:
        for i, m in enumerate(runs[a]):
            est = {"groomed": {g: _group(0.2) for g in ("resonant", "qcd", "all")},
                   "mlp": {g: _group(0.15) for g in ("resonant", "qcd", "all")}}
            if a in M.MASS:
                est["regressed"] = {g: _group(0.11 + 0.01 * i) for g in ("resonant", "qcd", "all")}
            out[m] = {"model": m, "estimators": est}
    return out


def _sculpt(P, runs) -> dict:
    before = [100.0 + 10 * k for k in range(24)]
    out = {}
    for row in P.F.SCULPT_ROWS:
        for i, m in enumerate(runs[row]):
            curve = [{"eff": e, "n_pass": int(4e5 * e), "js_distance": (1 + 0.1 * i) * 1e-2 / e ** 0.25}
                     for e in (0.001, 0.01, 0.1, 0.5)]
            wps = {wp: {"hist_after": before[::-1], "n_pass": int(4e5 * float(wp)), "jsd": 1e-4}
                   for wp in ("0.1", "0.01")}
            out[m] = {"bins_jet_sdmass": list(range(20, 501, 20)), "hist_before": before, "n_qcd": 400_000,
                      "working_points": wps, "curve": curve}
    return out


def _endpoint(k: float) -> dict:
    eps_b = np.logspace(-5, 0, 31)
    eps_s = np.clip(eps_b ** (0.3 + 0.01 * k), 0, 1)
    return {"n_b": 400_000, "eps_s_at_star": float(eps_s[6]), "eps_b_star": 1e-3, "auc": 0.95, "r30": 300.0,
            "r50": 100.0, "curve": {"eps_b": eps_b.tolist(), "eps_s": eps_s.tolist(),
                                    "eps_s_over_sqrt_eps_b": (eps_s / np.sqrt(eps_b)).tolist()}}


def _xbx(P, runs) -> dict:
    out = {}
    for row in P.LEVELS:
        for i, m in enumerate(runs[row]):
            out[m] = {"xbb": {"head": [_endpoint(i), _endpoint(i + 1)], "zero_shot_xbb": _endpoint(i + 2)},
                      "xbs": {"head": [_endpoint(i)], "zero_shot_xbq": _endpoint(i + 3)}}
    return out


# the curves_v2 grid (ROC interface 1.5): 0.001 ... 0.010, then 0.015 ... 0.995
EPS = [round(0.001 * k, 3) for k in range(1, 11)] + [round(0.005 * k, 3) for k in range(3, 200)]
MIN_BKG = 20


def _curve(k: float, n_bkg: int, apart: bool = False) -> dict:
    """One curve of a curves_v2 file: rejection (1 + k/100) / eps_S^2, cut at the first grid point with
    MIN_BKG background jets passing; apart: the test set separated completely (cut null). The first two
    grid points pass no background jet (eps_B = 0): rejection null, as roc_points.py writes it."""
    eps = np.array(EPS)
    rej = (1 + 0.01 * k) / eps ** 2
    nb = np.rint(n_bkg / rej).astype(int)
    nb[:2] = 0
    i = int(np.argmax(nb >= MIN_BKG))
    assert i >= 2
    cut = None if apart else {"eps_s": EPS[i], "eps_b": 1 / rej[i], "rejection": rej[i], "n_bkg_pass": int(nb[i])}
    return {"signal": ["s"], "background": ["b"], "discriminant": "synthetic", "n_signal": 1000,
            "n_background": n_bkg, "n_train": 1000, "auc": 0.99, "rejection": [None, None] + rej[2:].tolist(),
            "n_bkg_pass": nb.tolist(), "cut": cut, "n_bkg_pass_full_eff": n_bkg, "label_sha256": "0" * 64,
            "source": {"file": "synthetic", "sha256": "0" * 64}}


def _roc_docs(M, runs) -> dict:
    """I["roc"] (mlst_v4.load_roc's {family: {model: doc}}) for the six families, with the pair X->ee vs
    X->mumu separated completely (cut null) by every configuration that keeps it apart."""
    pre = _pretrained(M, runs)
    scratch = [f"{M.SCRATCH}-s{k}" for k in (1, 2, 3)]
    apart = {m for a in ("L188", "L162") + M.FLAV for m in runs[a]}
    jc1 = [c for c, _ in M.V4.JC1_COLS]
    fams = {"jc1_ft": (pre + scratch, {c: 200_000 for c in jc1}),
            "jc1_probe": (pre + list(M.INIT_MODELS) + list(M.INIT_BN_MODELS), {c: 200_000 for c in jc1}),
            "pairs": (pre, {**{k: 200_000 for k, *_ in M.PAIRS}, "ee_vs_mm": 5_000}),
            "top_ft": (pre + scratch + [M.KIN_MODEL], {"top": 201_914}),
            "qg_ft": (pre + scratch, {"qg": 100_000, "qg_herwig": 100_000}),
            "jc2_ovr": ([m for a in M.MAIN for m in runs[a]], {str(c): 900_000 for c in range(188)})}
    out = {}
    for fam, (models, keys) in fams.items():
        for i, m in enumerate(models):
            curves = {key: _curve(i, nb, apart=key == "ee_vs_mm" and m in apart) for key, nb in keys.items()}
            out.setdefault(fam, {})[m] = {"family": fam, "model": m, "seed": int(m.rsplit("-s", 1)[1]),
                                          "method": "synthetic", "checkpoint": None, "ft_manifest": None,
                                          "eps_grid": EPS, "min_bkg_pass": MIN_BKG, "curves": curves}
    return out


@pytest.fixture
def inputs(P, M, runs, tmp_path):
    fits, raw = _fits(M, runs, tmp_path)
    census = {"counts": {"ft_test": [5000] * 184 + [224, 600, 900, 3000]}}
    return {"fits": fits, "raw": raw, "mlp": _mlp(M, runs), "fill": _fill(M, runs), "per": _per(M, runs),
            "atlas": _atlas(M, runs), "sculpt": _sculpt(P, runs), "census": census, "xbx": _xbx(P, runs),
            "xbx_scratch": {"xbs": [_endpoint(-k) for k in (1, 2, 3)]}, "roc": _roc_docs(M, runs)}


def _empty() -> dict:
    return {"fits": {}, "raw": {}, "mlp": {}, "fill": {}, "per": {}, "atlas": {}, "sculpt": {}, "census": None,
            "xbx": {}, "xbx_scratch": {}, "roc": {}}


@pytest.fixture
def seen(P, monkeypatch):
    """Every figure saved: its size and the strings and sizes of its visible texts, recorded before the
    real save."""
    from matplotlib.text import Text
    got = {}
    real = P.style.save

    def save(fig, name, outdir=None, tight=True, dpi=150):
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        got[name] = {"width": fig.get_figwidth(), "tight": tight,
                     "texts": [(t.get_text(), t.get_fontsize()) for t in texts]}
        return real(fig, name, outdir, tight=tight, dpi=dpi)
    monkeypatch.setattr(P.style, "save", save)
    return got


def test_every_figure_draws_from_synthetic_inputs(P, M, runs, inputs, seen, tmp_path):
    out = tmp_path / "out"
    assert set(P.FILES) == set(P.FIGURES)
    for f in P.FIGURES:
        before = set(seen)
        paths = f(M, runs, inputs, out)
        # FILES (what main removes on a skip) names exactly what the figure saves
        assert set(seen) - before == set(P.FILES[f]), f.__name__
        assert sorted(p.suffix for p in paths) == [".pdf", ".png"], f.__name__
        assert all(p.stat().st_size > 0 for p in paths), f.__name__
        # vector PDF with the fonts embedded as TrueType, never as Type 3
        assert b"/Type3" not in next(p for p in paths if p.suffix == ".pdf").read_bytes(), f.__name__
    assert len(seen) == len(P.FIGURES)
    assert {"F6J_roc_jetclass", "F6_roc_benchmarks", "A6J_roc_jetclass_probe", "A3R_roc_jetclass2"} <= set(seen)
    assert "A_roc_benchmarks" not in seen
    for name, rec in seen.items():
        # MLST: 15 cm wide (every figure of the paper), saved at that size, text between 8 and 12 pt
        assert rec["width"] == pytest.approx(P.FULL), name
        assert rec["tight"] is False, name
        for text, size in rec["texts"]:
            assert 8.0 <= size <= 12.0, (name, text, size)
            # nothing but data: no watermark or stand-in note
            assert "preliminary" not in text.lower() and "stand-in" not in text.lower(), (name, text)


def test_absent_inputs_skip_every_figure_without_failing(P, M, runs, seen, tmp_path):
    for f in P.FIGURES:
        with pytest.raises(P.Skip):
            f(M, runs, _empty(), tmp_path / "out")
    assert not seen


def test_a_partial_input_drops_its_panel_or_series_and_draws_the_rest(P, M, runs, inputs, seen, tmp_path, capsys):
    part = dict(inputs, per={}, fill={}, xbx_scratch={}, census=None)
    for f in (P.figure_f2, P.figure_f3, P.figure_f5, P.figure_f7, P.figure_f9, P.figure_a10):
        assert f(M, runs, part, tmp_path / "out"), f.__name__
    log = capsys.readouterr().out
    assert "absent: F2: fine-tuning on class-balanced JetClass-II" in log
    assert "absent: F3 hatching" in log and "absent: F5: the training-mean" in log
    assert "absent: F9 xbs: training from scratch" in log
    # no regression output of the mass models: the groomed mass alone, logged
    atlas = {m: {"estimators": {k: v for k, v in d["estimators"].items() if k != "regressed"}}
             for m, d in inputs["atlas"].items()}
    assert P.figure_f8a(M, runs, dict(inputs, atlas=atlas), tmp_path / "out")
    assert "regression output of L162_MASS" in capsys.readouterr().out


def test_main_prints_what_it_drew_and_exits_zero_only_when_something_was_drawn(P, inputs, seen, monkeypatch,
                                                                               tmp_path, capsys):
    monkeypatch.setattr(P, "inputs", lambda M: _empty())
    # an older render of a skipped figure is removed, so main.tex prints its pending marker
    stale = tmp_path / "none"
    stale.mkdir()
    old = [stale / f"F4_flavor_pairs.{ext}" for ext in ("pdf", "png")]
    for p in old + [stale / "unrelated.pdf"]:
        p.write_bytes(b"old")
    assert P.main(["--outdir", str(stale)]) == 1
    out = capsys.readouterr().out
    assert out.count("skipped, no input") == len(P.FIGURES) and f"drawn 0, skipped {len(P.FIGURES)}" in out
    assert not any(p.exists() for p in old) and (stale / "unrelated.pdf").exists()
    monkeypatch.setattr(P, "inputs", lambda M: inputs)
    assert P.main(["--outdir", str(tmp_path / "all")]) == 0
    assert f"drawn {len(P.FIGURES)}, skipped 0, failed 0" in capsys.readouterr().out
    assert len(list((tmp_path / "all").glob("*.pdf"))) == len(P.FIGURES)


def test_the_label_set_figures_draw_the_generators_values(P, M, runs, inputs):
    fits, fill, per = inputs["fits"], inputs["fill"], inputs["per"]
    chosen, _ = M.chosen_substitutions(fill, runs)
    # F2: absolute values, the probe at the largest training set as the tables select it
    n = M._largest(fits, "jc2_188")
    probe = P._f2_probe(M, fits, fill, chosen, "jc2_188", "accuracy")
    for arm in ("L188", "R16_Q1", "RAND2_p1", "FLAV_F1"):
        want = [M._probe_rec(fits, m, "jc2_188", n)["accuracy"] for m in runs[arm]]
        assert want and P._vals(runs, arm, probe) == pytest.approx(want)
    top = P._f2_probe(M, fits, fill, chosen, "top", "r30")
    m = runs["L188"][0]
    nt = M._bench_sizes(fits, fill, chosen, "top")[-1]
    assert top(m) == M._rej(M._bench_rec(fits, fill, chosen, m, "top", nt), "r30")
    ft = P._f2_ft(M, per, "jc1", 1_000_000, "accuracy")
    assert ft(m) == M._ft(per, "jc1", m, 1_000_000, "accuracy")
    assert P._f2_ft(M, {}, "jc2", 1_000_000, "accuracy") is None
    # F3: the mean over indices of the per-class AUC difference (percentage points)
    rows, R = P.heat(M, fits, runs, REPO)
    assert len(rows) == 188 and rows[-1]["class_name"].startswith("label_QCD")
    j = [a for a, _ in P.MK.HEAT].index("R16_Q1")
    diff = M.per_class_auc_diff(fits, runs, "R16_Q1")
    c = int(rows[0]["jet_label"])
    assert R[0, j] == pytest.approx(np.mean(diff[c]))
    merges = P.flavor_merges(M, REPO)
    assert merges["R16_Q1"]["label_X_bb"][0] and not merges["L188"]["label_X_bb"][0]
    assert P._nice_up(0.31) == pytest.approx(0.5) and P._nice_up(2.0) == pytest.approx(2.0)


def test_the_jetclass2_roc_figure_draws_the_generators_selection(P, M, runs, inputs, monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(P, "roc_figure", lambda M, fig, panels, *a: got.setdefault("keys", [p[1] for p in panels]))
    P.figure_a3r(M, runs, inputs, tmp_path)
    sel = M.jc2_roc_classes(inputs["fits"], runs, REPO, inputs["census"])
    want = [str(c) for c in list(sel["hurt"]) + list(sel["stable"])]
    assert got["keys"] == want and len(want) == M.NHURT + M.NSTABLE


def test_a_roc_panel_follows_the_cut_and_pairs_its_ratio_by_index(P, M, runs, inputs):
    import matplotlib.pyplot as plt
    fam = inputs["roc"]["pairs"]
    configs = [(row, list(runs[row])) for row in P.LEVELS + M.FLAV]
    fig, (ax, rx) = plt.subplots(2, 1)
    ratio, rows = P.roc_panel(ax, rx, M, fam, "bb_vs_cc", configs)
    assert ratio and rows == [r for r, _ in configs]
    # the 162-class ratio: mean over indices of each run over the 188-class run of the same index,
    # where every run survives the cut
    s = lambda m: M.V4.roc_series(fam[m], "bb_vs_cc")[1]
    ref = {M._idx(m): s(m) for m in runs["L188"]}
    q = np.array([s(m) / ref[M._idx(m)] for m in runs["L162"]])
    ok = np.isfinite(q).all(axis=0) & np.isfinite(np.array(list(ref.values()))).all(axis=0)
    line = rx.lines[1]          # [0]: the 188-class reference flat at 1
    assert np.asarray(line.get_ydata()) == pytest.approx(q[:, ok].mean(axis=0))
    # the limit line N_B / 20 is drawn
    assert any(np.allclose(ln.get_ydata(), M.V4.roc_limit(fam[runs["L188"][0]], "bb_vs_cc")) for ln in ax.lines)
    plt.close(fig)
    # X->ee vs X->mumu: the configurations that keep the pair apart separate the test set completely
    # (cut null), so the 188-class reference has no point: no ratio panel, every configuration kept
    fig, (ax, rx) = plt.subplots(2, 1)
    ratio, rows = P.roc_panel(ax, rx, M, fam, "ee_vs_mm", configs)
    assert not ratio and "L188" in rows and not rx.lines
    plt.close(fig)


def _scaled(doc: dict, key: str, f: float) -> dict:
    """A copy of a curves_v2 doc with the rejection of curve key multiplied by f (cut unchanged)."""
    d = json.loads(json.dumps(doc))
    c = d["curves"][key]
    c["rejection"] = [None if v is None else v * f for v in c["rejection"]]
    return d


def test_the_randomly_initialized_network_pairs_by_index_and_from_scratch_takes_the_mean(P, M, runs, inputs):
    """The ratio of init(bn)-sK is to the 188-class run sK (its initial weights); training from scratch
    is to the 188-class mean (seed k trains on subset k). The 188-class run of index k and the init
    network of index k are scaled by the same factor, so pairing and the mean differ."""
    import matplotlib.pyplot as plt
    key = M.V4.JC1_COLS[0][0]
    f = {"1": 1.0, "2": 2.0, "3": 4.0}
    scale = lambda fam, models: {**fam, **{m: _scaled(fam[m], key, f.get(M._idx(m), 1.0)) for m in models}}
    s = lambda fam, m: M.V4.roc_series(fam[m], key)[1]
    for family, row, models in (("jc1_probe", P.INIT_BN, list(M.INIT_BN_MODELS)),
                                ("jc1_ft", P.SCRATCH, P._scratch_models(inputs["roc"]["jc1_ft"]))):
        fam = scale(inputs["roc"][family], list(runs["L188"]) + (models if row == P.INIT_BN else []))
        configs = [(r, list(runs[r])) for r in P.LEVELS] + [(row, models)]
        fig, (ax, rx) = plt.subplots(2, 1)
        ratio, _ = P.roc_panel(ax, rx, M, fam, key, configs)
        assert ratio
        R = np.array([s(fam, m) for m in runs["L188"]])
        ref = {M._idx(m): s(fam, m) for m in runs["L188"]}
        paired = np.array([s(fam, m) / ref[M._idx(m)] for m in models])
        mean = np.array([s(fam, m) / R.mean(axis=0) for m in models])
        want, other = (paired, mean) if row == P.INIT_BN else (mean, paired)
        ok = np.isfinite(want).all(axis=0) & np.isfinite(R).all(axis=0)
        line = np.asarray(rx.lines[-1].get_ydata())        # the last configuration's ratio line
        assert line == pytest.approx(want[:, ok].mean(axis=0)), row
        assert not np.allclose(line, other[:, ok].mean(axis=0)), row
        plt.close(fig)


def test_the_mass_ratio_to_groomed_is_drawn_only_where_the_groomed_mass_has_enough_jets(P, M, runs, inputs):
    import matplotlib.pyplot as plt
    fig, (ax, rx) = plt.subplots(2, 1)
    assert P.mass_hist(ax, rx, M, runs, inputs["atlas"], "all", "test")
    counts = np.asarray(P._groomed(inputs["atlas"], "all")["hist"]["counts"])
    low = counts < M.MASS_MIN_JETS
    assert low.any() and (~low).any()
    assert len(rx.patches) == len(M.MASS)                   # one ratio histogram per mass model
    for patch in rx.patches:
        v = np.asarray(patch.get_data().values, float)
        assert np.isnan(v[low]).all() and np.isfinite(v[~low]).all()
    plt.close(fig)


def test_the_substitution_chosen_on_validation_is_the_generators(P, M, runs, inputs):
    chosen, tots = M.chosen_substitutions(inputs["fill"], runs)
    # the synthetic validation cross-entropy grows with the substitution's index: zero is chosen
    assert chosen["top"] == "zero" and set(tots["top"]) == set(P.SUBST)
    m = runs["L188"][0]
    fits, fill = inputs["fits"], inputs["fill"]
    assert P._bench_r30(M, fits, {}, m, "top", "zero") == M._rej(M._probe_rec(fits, m, "top"), "r30")
    assert P._bench_r30(M, fits, fill, m, "top", "mean") == M._fill_r30(fill, m, "top", "mean")


def test_the_sculpting_floor_is_the_generators_with_its_spread(P, M, inputs, runs):
    d = inputs["sculpt"][runs["L188"][0]]
    eff, mean, sd = P.sculpt_floor(M, d)
    k = int(np.count_nonzero(d["hist_before"]))
    assert mean == pytest.approx([M._floor_distance(k, p["n_pass"], d["n_qcd"]) for p in d["curve"]])
    assert np.all(sd > 0) and np.all(sd < mean)


def test_the_final_style_is_sophons_at_the_printed_size(P, M):
    import matplotlib
    import matplotlib.pyplot as plt
    S = P.style
    rc = plt.rcParams
    with plt.rc_context():
        S.use_style(final=True)
        # plain matplotlib (no mplhep, no LaTeX): STIX text and mathematics, tab10 in order
        assert not rc["text.usetex"] and rc["font.family"] == ["STIXGeneral"] and rc["mathtext.fontset"] == "stix"
        tab10 = matplotlib.rcParamsDefault["axes.prop_cycle"].by_key()["color"]
        assert rc["axes.prop_cycle"].by_key()["color"] == S.TAB10 == tab10
        # dashed light-grey grid on major and minor ticks, outward ticks bottom and left, framed legend
        assert rc["axes.grid"] and rc["axes.grid.which"] == "both" and rc["grid.linestyle"] == "--"
        assert rc["xtick.direction"] == "out" and not rc["xtick.top"] and not rc["ytick.right"]
        assert rc["legend.frameon"] and rc["xaxis.labellocation"] == "right" and rc["axes.titlelocation"] == "center"
        # TrueType fonts, text between 8 and 12 pt at the printed size
        assert rc["pdf.fonttype"] == 42
        assert all(8.0 <= rc[k] <= 12.0 for k in ("font.size", "axes.labelsize", "xtick.labelsize", "legend.fontsize"))
        # the scaled layout of the mock-ups is not left at the printed size by an earlier final figure
        S.use_style()
        assert rc["font.size"] > 20
    assert not hasattr(S, "hep") and not hasattr(S, "latex_helvetica")
    assert S.ONE_COLUMN == pytest.approx(8.5 / 2.54) and S.FULL_WIDTH == pytest.approx(15 / 2.54)
    # one encoding for every configuration a figure draws; the TeX colours of the text are the same
    for row in M.MAIN + M.MASS + M.FLAV + ("RAND", M.SCRATCH, M.INIT, M.INIT_BN, "KIN"):
        assert {"color", "linestyle", "marker"} <= set(S.CONFIG[row]), row
    assert S.CONFIG["L188"]["color"] == tab10[0] and S.CONFIG[M.SCRATCH]["color"] == "black"
    assert "\\definecolor{c188}{HTML}{1F77B4}" in S.tex_colour_definitions()


def test_the_render_job_builds_the_paper_and_keeps_the_mockup_spec(tmp_path):
    R = _mod("build_render_job", "scripts/build_render_job.py")
    draw, collect = R.PAPER["draw"], R.PAPER["collect"]
    assert "python3 experiments/FIGS/make_tables.py > ${OUT}/tables.txt 2>&1; echo \"tables rc=$?\"" in draw
    assert "python3 experiments/FIGS/paper_figures.py > ${OUT}/figures.txt 2>&1; echo \"figures rc=$?\"" in draw
    assert "deck_figures" not in draw and "mlst_figures" not in draw
    assert draw.index("make_tables.py") < draw.index("paper_figures.py")
    for path in ("figures/mlst/*.pdf", "paper/journal/results_generated.tex", "paper/journal/provenance.json",
                 "paper/journal/tables", "paper/journal/main.pdf"):
        assert path in collect, path
    old = subprocess.run(["git", "show", f"{BASE}:scripts/build_render_job.py"], cwd=REPO, capture_output=True,
                         text=True)
    if old.returncode:
        pytest.skip(f"{BASE} is not in this clone")
    f = tmp_path / "build_render_job_base.py"
    f.write_text(old.stdout)
    O = _mod("build_render_job_base", f)       # REPO / an absolute path is that path
    commit = "abcdef1" + "0" * 33
    spec = lambda X: X.SPEC.format(short=commit[:7], commit=commit, image=X.IMAGE, pins=X.PINS,
                                   tectonic=X.TECTONIC, **X.MOCKUPS)
    assert spec(R) == spec(O)


def test_configurations_drawn_together_differ_in_line_style_and_marker(P, M):
    # the figures must read without colour (colour-vision deficiency, greyscale print; PI 2026-11-02):
    # every set of configurations one figure draws together has distinct line styles and markers
    C = P.style.CONFIG
    together = {"label sets with from scratch and the kinematic-only model (F6, F6J)": M.MAIN + (M.SCRATCH, "KIN"),
                "label sets with the controlled-flavor partitions (F4)": M.MAIN + M.FLAV,
                "label sets with the randomly initialized network (A6J)": M.MAIN + (M.INIT_BN,),
                "label sets with the mass regression (F8b)": M.MAIN + M.MASS}
    for what, rows in together.items():
        styles = [str(C[r]["linestyle"]) for r in rows]
        assert len(set(styles)) == len(rows), what
    for rows in (M.MAIN + M.MASS + M.FLAV + ("KIN",),):
        assert len({C[r]["marker"] for r in rows}) == len(rows)


def test_a_curve_only_some_panels_show_gets_its_own_legend_in_those_panels(P, M, runs, inputs, tmp_path):
    import matplotlib.pyplot as plt
    seen = {}
    real = P._save
    P._save = lambda fig, name, outdir: seen.setdefault("fig", fig) and []
    try:
        P.figure_f6(M, runs, inputs, tmp_path)
    finally:
        P._save = real
    fig = seen["fig"]
    shared = [t.get_text() for t in fig.legends[0].get_texts()]
    kin = P.NAME[P.KIN].replace("\n", " ")
    assert kin not in shared and all(P.NAME[r].replace("\n", " ") in shared for r in P.LEVELS)
    top = fig.axes[0].get_legend()
    assert top is not None and [t.get_text() for t in top.get_texts()] == [kin]
    assert fig.axes[2].get_legend() is None and fig.axes[4].get_legend() is None
    plt.close(fig)


def test_the_two_mass_models_have_their_own_marker_shapes(P, M, runs, inputs):
    # F8a and A17 draw each mass model with its marker in style.CONFIG (filled for 162 classes, open for
    # 17), in the binned panels and in the shared legend, so the two read apart without colour
    import matplotlib.pyplot as plt
    from matplotlib.colors import same_color
    fig, axs = plt.subplots(3, 1)
    assert P.mass_bins(axs, M, runs, inputs["atlas"], "all", "vs_pt", "pT", "test")
    got = {}
    for a in M.MASS:
        lk = P.look(M, a)
        lines = [ln for ln in axs[0].lines if same_color(ln.get_color(), lk["color"]) and ln.get_marker() != "None"]
        assert len({ln.get_marker() for ln in lines}) == 1, a
        got[a] = (lines[0].get_marker(), same_color(lines[0].get_markerfacecolor(), "white"))
        assert got[a][0] == P.style.CONFIG[a]["marker"], a
    assert len({m for m, _ in got.values()}) == len(M.MASS)
    assert not got["L162_MASS"][1] and got["R16_Q1_MASS"][1]
    leg = plt.figure()
    P._mass_legend(leg, M)
    handles = leg.legends[0].legend_handles[1:]
    assert [h.get_marker() for h in handles] == [P.style.CONFIG[a]["marker"] for a in M.MASS]
    plt.close(fig)
    plt.close(leg)


def _shape(ls) -> tuple:
    """The look of a line style without colour: solid, or each on-segment of its dash pattern a dot
    (at most 2 points) or a dash, as matplotlib draws the named styles."""
    named = {"-": (), "solid": (), "--": ("dash",), ":": ("dot",), "-.": ("dash", "dot")}
    if isinstance(ls, str):
        return named[ls]
    return tuple("dot" if on <= 2 else "dash" for on in ls[1][::2])


def test_no_two_f9_curves_of_a_row_share_their_look_without_colour(P, M, runs, inputs, monkeypatch, tmp_path):
    # every series of one F9 row (transfer heads, direct discriminant, output-node control, from scratch)
    # must differ in line-style shape or marker, not by colour only (the 17-class head is dotted, so the
    # output-node control may not be)
    drawn = {}
    real = P._mean_band

    def record(ax, x, v, kw, label):
        if label is not None:       # the right panels: one call per series
            drawn.setdefault(id(ax), []).append((label, _shape(kw.get("linestyle", "-")), kw.get("marker")))
        return real(ax, x, v, kw, label)
    monkeypatch.setattr(P, "_mean_band", record)
    assert P.figure_f9(M, runs, inputs, tmp_path)
    assert len(drawn) == 2
    for row in drawn.values():
        looks = [(shape, marker) for _, shape, marker in row]
        assert len(set(looks)) == len(looks), row
    assert any("output node" in label for row in drawn.values() for label, *_ in row)


def test_a_roc_panel_says_why_it_has_no_curve_and_zooms_when_all_curves_start_high(P, M, runs, inputs):
    import matplotlib.pyplot as plt
    configs = [(row, list(runs[row])) for row in P.LEVELS]
    def panel(cut):
        fam = json.loads(json.dumps(inputs["roc"]["pairs"]))
        for row, models in configs:
            for m in models:
                fam[m]["curves"]["bb_vs_cc"]["cut"]["eps_s"] = cut
        fig, (ax, rx) = plt.subplots(2, 1)
        ratio, rows = P.roc_panel(ax, rx, M, fam, "bb_vs_cc", configs)
        return fig, ax, ratio, rows
    fig, ax, ratio, rows = panel(0.993)          # one grid point left for every configuration
    assert not ratio and rows == [r for r, _ in configs]
    assert [t.get_text() for t in ax.texts][0].startswith("No curve: fewer than 20 background jets")
    assert ax.get_xlim() == (0, 1) and ax.get_ylim()[0] < 1e3 * ax.get_ylim()[1] / 1e3 / 10
    plt.close(fig)
    fig, ax, ratio, rows = panel(0.97)
    assert ax.get_xlim() == (0.95, 1) and not ax.texts
    plt.close(fig)
    fig, ax, ratio, rows = panel(0.5)
    assert ax.get_xlim() == (0, 1) and not ax.texts
    plt.close(fig)


def test_a_roc_curve_with_one_surviving_point_is_not_drawn(P, M, runs, inputs):
    # a configuration whose cut leaves a single grid point (here 0.995, cut at 0.993) draws no visible
    # line, so neither the ROC nor the ratio panel draws it, and the generator calls it 'not reached'
    import matplotlib.pyplot as plt
    from matplotlib.colors import same_color
    fam = json.loads(json.dumps(inputs["roc"]["pairs"]))
    for m in runs["R16_Q1"]:
        fam[m]["curves"]["bb_vs_cc"]["cut"]["eps_s"] = 0.993
    assert sum(e >= 0.993 for e in EPS) == 1
    configs = [(row, list(runs[row])) for row in P.LEVELS]
    fig, (ax, rx) = plt.subplots(2, 1)
    ratio, rows = P.roc_panel(ax, rx, M, fam, "bb_vs_cc", configs)
    red = P.look(M, "R16_Q1")["color"]
    assert ratio and "R16_Q1" in rows
    assert not any(same_color(ln.get_color(), red) for ln in list(ax.lines) + list(rx.lines))
    assert any(same_color(ln.get_color(), P.look(M, "R42_Q1")["color"]) for ln in ax.lines)
    assert M._roc_cut([fam[m]["curves"]["bb_vs_cc"] for m in runs["R16_Q1"]], EPS) == "not reached"
    plt.close(fig)
