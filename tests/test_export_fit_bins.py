"""experiments/AOJ/export_fit_bins.py: the exported bins must be the ones the run
fitted -- a fit rebuilt from them reproduces the stored yield -- and the export
refuses when they are not."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _mod("export_fit_bins", "experiments/AOJ/export_fit_bins.py")
T = _mod("test_aoj_peak_fit_helpers", "tests/test_aoj_peak_fit.py")
P = E.P


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(91)
    parts = [T.sample(92, n_bkg=250_000), T.sample(93, n_bkg=0, n_sig=5000, peak="top")]
    mass, pt = (np.concatenate([p[k] for p in parts]) for k in (0, 1))
    kind = np.repeat([0, 1], [len(p[0]) for p in parts])
    tag = rng.normal(np.where(kind == 1, 3.0, -3.0), 2.0)
    np.savez(tmp / "jets.npz", jet_sdmass=mass.astype(np.float32), aoj_jet_pt=pt.astype(np.float32),
             aoj_pn_TvsQCD=(1 / (1 + np.exp(-tag))).astype(np.float16))
    np.savez(tmp / "m.npz", three_prong_logodds=tag.astype(np.float16))
    argv = sys.argv
    try:
        sys.argv = ["peak_fit.py", "--jets", str(tmp / "jets.npz"), "--toys", "0", "--out", str(tmp),
                    "--peaks", "top", "--scores", f"m={tmp/'m.npz'}"]
        assert P.main() == 0
    finally:
        sys.argv = argv
    return tmp


def _export(run, out, histograms=None):
    return E.main(["--jets", str(run / "jets.npz"), "--results", str(run / "results.json"),
                   "--histograms", str(histograms or run / "histograms.npz"),
                   "--scores", f"m={run/'m.npz'}", "--out", str(out)])


def test_a_fit_rebuilt_from_the_exported_bins_reproduces_the_stored_yield(run, tmp_path):
    assert _export(run, tmp_path / "bins.npz") == 0
    z = np.load(tmp_path / "bins.npz")
    res = json.loads((run / "results.json").read_text())
    # the scores' fits take the tops failing their cut, the reference's fitted signal (peak_fit.main)
    ref_bins = {k: z[f"reference|main|{k}"] for k in E.KEYS}
    tops = P.tops_from_reference(ref_bins, res["reference"]["top"]) if res.get("fail_tops") else None
    for name, stored in (("reference", res["reference"]["top"]), ("m", res["models"]["m"]["top"])):
        b = {k: z[f"{name}|main|{k}"] for k in E.KEYS}
        side = ~P.in_windows(P._bin_centres(b), [P.PEAKS["top"]["window"]])
        tf_norm = b["n_pass"][side].sum() / max(b["n_fail"][side].sum(), 1.0)
        model = P._Model(b, tuple(stored["tf_order"]), tf_norm, stored["mean"], stored["width"],
                         None if name == "reference" else tops)
        x, _ = model.fit()
        y = float(model.G.sum(axis=0) @ x[model.n_tf:])
        assert abs(y - stored["signal_yield"]) <= 1e-9 * abs(stored["signal_yield"])
        assert len(z[f"{name}|validation|n_pass"]) > 0


def test_the_export_refuses_bins_that_differ_from_the_runs_histograms(run, tmp_path):
    h = dict(np.load(run / "histograms.npz"))
    h["m_top_validation_n_fail"] = h["m_top_validation_n_fail"] + 1
    np.savez(tmp_path / "other.npz", **h)
    with pytest.raises(SystemExit, match="not the bins that run fitted"):
        _export(run, tmp_path / "bins.npz", tmp_path / "other.npz")
    assert not (tmp_path / "bins.npz").exists()


D = _mod("fit_minimum_diagnostic", "experiments/AOJ/fit_minimum_diagnostic.py")


def test_the_orthonormal_basis_is_the_same_polynomial_and_reaches_the_same_minimum(run, tmp_path):
    assert _export(run, tmp_path / "bins.npz") == 0
    z = np.load(tmp_path / "bins.npz")
    res = json.loads((run / "results.json").read_text())
    stored = res["models"]["m"]["top"]
    b = {k: z[f"m|main|{k}"] for k in E.KEYS}
    side = ~P.in_windows(P._bin_centres(b), [P.PEAKS["top"]["window"]])
    tf_norm = b["n_pass"][side].sum() / max(b["n_fail"][side].sum(), 1.0)
    args = (b, tuple(stored["tf_order"]), tf_norm, stored["mean"], stored["width"])
    ortho, mono = D.Orthonormal(*args), P._Model(*args)
    u = np.random.default_rng(0).normal(size=len(ortho.x0))
    t = ortho.transform()
    assert np.allclose(mono.loss(t @ u)[0], ortho.loss_u(t)(u)[0])
    assert np.allclose(ortho.X @ (t @ u)[:ortho.n_tf], np.linalg.qr(ortho.X)[0] @ u[:ortho.n_tf])
    (x1, f1), (x2, f2) = mono.fit(), ortho.fit()
    assert abs(f1 - f2) < 1e-6
    xn, fn, edm, _ = D.newton(ortho, x2)
    assert fn <= f2 + 1e-9 and edm < 1e-6


def test_the_diagnostic_reproduces_the_run_and_finds_nothing_on_a_well_posed_fit(run, tmp_path):
    assert _export(run, tmp_path / "bins.npz") == 0
    assert D.main(["--bins", str(tmp_path / "bins.npz"), "--results", str(run / "results.json"),
                   "--out", str(tmp_path / "d.json"), "--workers", "1"]) == 0
    s = json.loads((tmp_path / "d.json").read_text())["summary"]
    assert s["n_fits"] == 2 and s["n_as_run_reproduced"] == 2 and s["n_same_order"] == 2
    assert s["max_abs_yield_shift_over_err"] < 0.01 and s["max_newton_edm"] < 1e-6


def test_the_profile_likelihood_error_matches_the_orthonormal_hessian_error(run, tmp_path):
    assert _export(run, tmp_path / "bins.npz") == 0
    assert D.main(["--bins", str(tmp_path / "bins.npz"), "--results", str(run / "results.json"),
                   "--out", str(tmp_path / "d.json"), "--workers", "1", "--only", "m"]) == 0
    r = json.loads((tmp_path / "d.json").read_text())["fits"]["m"]
    assert r["profile_err_at_minimum"] == pytest.approx(r["orthonormal_err_from_orthonormal_hessian"], rel=0.02)
