"""experiments/FIGS/paper_figures.py, the paper's final figures: every figure draws from a small
synthetic set of inputs in the layouts the generator reads (the loaders are replaced, so the tests
do not depend on how much has been pulled), a figure whose input is absent is skipped without
failing, the style is Helvetica at the printed size with nothing but data drawn, and the render
job runs the paper build while its mock-up spec stays as it was."""
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


def _roc(k: float) -> dict:
    eps = [0.05 * i for i in range(1, 20)]
    return {"roc": {"eps_s": eps, "rejection": [(1 + 0.01 * k) / e ** 2 for e in eps]}}


def _ft_curves(M, runs) -> dict:
    dss = ("top", "qg", "qg_herwig")
    out = {m: {"best70": {ds: _roc(i) for ds in dss}} for i, m in enumerate(m for a in M.MAIN for m in runs[a])}
    out.update({f"{M.SCRATCH}-s{k}": {"best70": {ds: _roc(-k) for ds in dss}} for k in (1, 2, 3)})
    out[M.KIN_MODEL] = {"best70": {"top": _roc(5)}}
    return out


@pytest.fixture
def inputs(P, M, runs, tmp_path):
    fits, raw = _fits(M, runs, tmp_path)
    census = {"counts": {"ft_test": [5000] * 184 + [224, 600, 900, 3000]}}
    return {"fits": fits, "raw": raw, "mlp": _mlp(M, runs), "fill": _fill(M, runs), "per": _per(M, runs),
            "atlas": _atlas(M, runs), "sculpt": _sculpt(P, runs), "census": census, "xbx": _xbx(P, runs),
            "xbx_scratch": {"xbs": [_endpoint(-k) for k in (1, 2, 3)]}, "ft_curves": _ft_curves(M, runs)}


def _empty() -> dict:
    return {"fits": {}, "raw": {}, "mlp": {}, "fill": {}, "per": {}, "atlas": {}, "sculpt": {}, "census": None,
            "xbx": {}, "xbx_scratch": {}, "ft_curves": {}}


@pytest.fixture
def seen(P, monkeypatch):
    """Every figure saved: its size and the strings and sizes of its visible texts, recorded before the
    real save. LaTeX is switched off, so the tests do not depend on a TeX installation."""
    from matplotlib.text import Text
    monkeypatch.setattr(P.style, "latex_helvetica", lambda: False)
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
    for f in P.FIGURES:
        paths = f(M, runs, inputs, out)
        assert sorted(p.suffix for p in paths) == [".pdf", ".png"], f.__name__
        assert all(p.stat().st_size > 0 for p in paths), f.__name__
        # vector PDF with the fonts embedded as TrueType, never as Type 3
        assert b"/Type3" not in next(p for p in paths if p.suffix == ".pdf").read_bytes(), f.__name__
    assert len(seen) == len(P.FIGURES)
    for name, rec in seen.items():
        # MLST: 8.5 cm or 15 cm wide, saved at that size, text between 8 and 12 pt
        assert rec["width"] in (pytest.approx(P.ONE), pytest.approx(P.FULL)), name
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
    assert P.main(["--outdir", str(tmp_path / "none")]) == 1
    out = capsys.readouterr().out
    assert out.count("skipped, no input") == len(P.FIGURES) and f"drawn 0, skipped {len(P.FIGURES)}" in out
    monkeypatch.setattr(P, "inputs", lambda M: inputs)
    assert P.main(["--outdir", str(tmp_path / "all")]) == 0
    assert f"drawn {len(P.FIGURES)}, skipped 0, failed 0" in capsys.readouterr().out
    assert len(list((tmp_path / "all").glob("*.pdf"))) == len(P.FIGURES)


def test_the_label_set_figures_draw_the_generators_ratios(P, M, runs, inputs):
    fits = inputs["fits"]
    probe = P._jc2_probe(M, fits)
    for arm in ("R16_Q1", "RAND2_p1", "FLAV_F1"):
        want = [x for x in M._ratio_vals(runs, arm, "L188", probe)[0] if x is not None]
        assert want and P._over_full(M, runs, arm, probe) == pytest.approx(want)
    rows, R = P.heat(M, fits, runs, REPO)
    assert len(rows) == 188 and rows[-1]["class_name"].startswith("label_QCD")
    j = [a for a, _ in P.MK.HEAT].index("R16_Q1")
    rat = M.per_class_ratios(fits, runs, "R16_Q1")
    c = int(rows[0]["jet_label"])
    assert R[0, j] == pytest.approx(np.mean(rat[c]))
    merges = P.flavor_merges(M, REPO)
    assert merges["R16_Q1"]["label_X_bb"][0] and not merges["L188"]["label_X_bb"][0]


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


def test_the_final_style_is_helvetica_at_the_printed_size(P, monkeypatch):
    import matplotlib.pyplot as plt
    S = P.style
    rc = plt.rcParams
    # rc_context: the LaTeX setting must not reach the figures of later tests
    with plt.rc_context():
        monkeypatch.setattr(S, "latex_helvetica", lambda: True)
        S.use_style(final=True)
        assert rc["text.usetex"] and rc["font.family"] == ["sans-serif"]
        assert all(p in rc["text.latex.preamble"]
                   for p in (r"\usepackage{helvet}", r"\usepackage{sansmath}", r"\sansmath"))
        assert rc["pdf.fonttype"] == 42 and rc["font.size"] == 10.0
        monkeypatch.setattr(S, "latex_helvetica", lambda: False)
        S.use_style(final=True)
        assert not rc["text.usetex"] and rc["font.sans-serif"][0] in S.SANS and rc["mathtext.fontset"] == "custom"
        # the scaled layout of the mock-ups is not left at the printed size by an earlier final figure
        S.use_style()
        assert rc["font.size"] > 20
    assert S.ONE_COLUMN == pytest.approx(8.5 / 2.54) and S.FULL_WIDTH == pytest.approx(15 / 2.54)
    # the packages the LaTeX preamble needs come with the render job's TeX Live install
    job = (REPO / "scripts/build_render_job.py").read_text()
    assert all(p in job for p in ("texlive-latex-base", "texlive-latex-extra", "texlive-fonts-recommended", "dvipng"))


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
