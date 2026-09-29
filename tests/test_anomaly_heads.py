"""experiments/EVAL/anomaly_heads.py: the class-sum anomaly result from stored
per-jet head scores equals anomaly.py's own result from the logits, draw for
draw; the head diagnostics expose a head that never predicts QCD."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ah = _load("anomaly_heads", "experiments/EVAL/anomaly_heads.py")
xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
an = ah.an

RUNG = "R42_Q1"


def _world(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    qcd = an._probe().qcd_indices()
    _, signals = xv.anomaly_classes()
    lab = np.where(rng.random(n) < 0.5, rng.choice(qcd, n),
                   rng.choice(list(signals.values()) + [5, 6, 30], n))
    k = len({r[RUNG] for r in an.read_map()})
    node_of = an.node_roles(RUNG)[0]
    z = rng.normal(size=(n, k)).astype(np.float32)
    z[np.arange(n), [node_of[x] for x in lab]] += 2.0
    return lab.astype(np.int64), z, signals


def test_stored_head_scores_reproduce_anomaly_py_draw_for_draw():
    L, z, signals = _world()
    rows = np.flatnonzero(np.isin(L, an._probe().qcd_indices() + list(signals.values()))
                          | (np.arange(L.size) % 40 == 0))
    h = {"rows": rows, "label188": L[rows], **xv.head_score_columns(z[rows], RUNG, signals)}
    sigs = ["label_X_bb", "label_X_YY_bbbb"]
    cells = ah.anomaly_cells("r42q1-s5", RUNG, L, h, sigs, [150], 3, 1000, 1000)
    node_of = an.node_roles(RUNG)[0]
    for sig in sigs:
        lab = ah.an_label(sig)
        reps = []
        for t in range(3):
            seed = an.cell_seed("r42q1-s5", sig, 150, t)
            r = an.run_one("r42q1-s5", RUNG, None, L, z, {}, lab, node_of[lab], 150,
                           np.random.default_rng(seed), 1000, 1000, seed=t,
                           families=("class_sum", "class_sum_matched"))
            r["rng_seed"] = int(seed)
            reps.append(r)
        ref = an.aggregate(reps, "r42q1-s5", sig, 150)
        got = cells[sig]["150"]
        assert got["rng_seeds"] == ref["rng_seeds"]
        for fam in ("class_sum", "class_sum_matched"):
            assert got[fam]["sigma_min"] == ref[fam]["sigma_min"]
            assert got[fam]["max_sic"] == ref[fam]["max_sic"]


def test_a_diagnostics_only_extraction_skips_the_anomaly_cells():
    L, z, signals = _world()
    rows = np.flatnonzero(np.arange(L.size) % 40 == 0)
    h = {"rows": rows, "label188": L[rows], **xv.head_score_columns(z[rows], RUNG, signals)}
    assert ah.anomaly_cells("a", RUNG, L, h, ["label_X_bb"], [150], 2, 500, 500) is None


def test_head_diagnostics_expose_a_head_that_never_predicts_qcd():
    L, z, signals = _world()
    rows = np.arange(L.size)
    good = {"rows": rows, "label188": L, **xv.head_score_columns(z, RUNG, signals)}
    _, _, qcd = an.node_roles(RUNG)
    zb = z.copy()
    zb[:, sorted(qcd)] -= 20.0
    bad = {"rows": rows, "label188": L, **xv.head_score_columns(zb, RUNG, signals)}
    g, b = ah.head_diagnostics(good, RUNG, 1), ah.head_diagnostics(bad, RUNG, 1)
    assert b["top1_accuracy"] < g["top1_accuracy"] - 0.2
    assert b["mean_p_qcd_qcd"] < 1e-6 < g["mean_p_qcd_qcd"]
    assert b["median_logodds_res_qcd_on_qcd"] > g["median_logodds_res_qcd_on_qcd"] + 10


def test_vectorised_sigma_min_equals_the_scalar_asimov_scan():
    """anomaly.sigma_min_asimov scans thresholds with numpy; it must give the
    number the committed per-threshold loop gave."""
    import math
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 400
        eps_b = np.sort(rng.random(n))[::-1] * 0.5 + 1e-4
        eps_s = np.clip(eps_b ** rng.uniform(0.2, 0.9), 0, 1)
        B = 200_000.0

        def best_z(sig0):
            S = sig0 * math.sqrt(B)
            return max(an.asimov(S * float(a), B * float(b)) for a, b in zip(eps_s, eps_b))

        lo, hi = 1e-6, 1.0
        while best_z(hi) < an.SIGMA_T:
            lo, hi = hi, hi * 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if best_z(mid) >= an.SIGMA_T:
                hi = mid
            else:
                lo = mid
        ref = 0.5 * (lo + hi)
        assert an.sigma_min_asimov(eps_s, eps_b, int(B)) == np.float64(ref) or \
            abs(an.sigma_min_asimov(eps_s, eps_b, int(B)) / ref - 1) < 1e-12


def test_main_reads_extract_v2_directories_in_parallel(tmp_path, monkeypatch):
    import json
    import sys
    # worker processes import the module by name, as they do when it runs as a script
    monkeypatch.syspath_prepend(str(ROOT / "experiments" / "EVAL"))
    monkeypatch.setitem(sys.modules, "anomaly_heads", ah)
    L, z, signals = _world()
    np.save(tmp_path / "labels.npy", L.astype(np.int16))
    rows = np.flatnonzero(np.isin(L, an._probe().qcd_indices() + list(signals.values()))
                          | (np.arange(L.size) % 100 == 0))
    res = {"n_stream": int(L.size), "label188_sha256": "x", "labels": L, "checkpoints": {}}
    for tag, shift in (("e078", 0.0), ("e079", 0.5)):
        zz = z + shift
        res["checkpoints"][tag] = {
            "features": np.zeros((0, 128), np.float16), "rows": np.zeros(0, np.int64),
            "label188": np.zeros(0, np.int16), "head_rows": rows,
            "head_label188": L[rows].astype(np.int16),
            "head": xv.head_score_columns(zz[rows], RUNG, signals)}
    meta = {"rung": RUNG, "diag_stride": 100,
            "checkpoints": {"e078": {"checkpoint_sha256": "a"}, "e079": {"checkpoint_sha256": "b"}}}
    xv.write(tmp_path / "m", res, meta)
    out = tmp_path / "o.json"
    assert ah.main(["--models", f"r42q1-s5={RUNG}={tmp_path / 'm'}", "--labels",
                    str(tmp_path / "labels.npy"), "--signals", "label_X_bb", "--n-sig", "150",
                    "--trainings", "2", "--n-bkg", "1000", "--n-template", "1000",
                    "--procs", "2", "--out", str(out)]) == 0
    got = json.loads(out.read_text())["models"]["r42q1-s5"]
    assert set(got["checkpoints"]) == {"e078", "e079"}
    assert "anomaly" in got["checkpoints"]["e079"] and "head_over_70_79" not in got


def test_the_v1err_analysis_spec_reads_all_thirty_models_with_the_retry_policy():
    import yaml
    ba = _load("build_anomaly_jobs", "scripts/build_anomaly_jobs.py")
    text = ba.v1err_spec()
    d = yaml.safe_load(text)
    assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
    line = next(l for l in text.splitlines() if "for spec in" in l)
    models = line.split("for spec in ")[1].rstrip("; do").split()
    assert len(models) == 30 and "mtx-l162-s1b:L162" in models and "mtx-r16q1mass-s4:R16_Q1" in models
    assert (ROOT / "experiments/EVAL/k8s/job-anomaly-heads-v1err-raunav.yaml").read_text() == text
