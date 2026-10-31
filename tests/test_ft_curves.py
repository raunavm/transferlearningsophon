"""experiments/FT/ft_curves.py (ROC points and per-class tables from stored scores) and the
curves jobs of scripts/build_curve_jobs.py."""
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest
import yaml

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FC = _load("ft_curves", "experiments/FT/ft_curves.py")
PROBE = _load("probe", "experiments/EVAL/probe.py")


def _binary(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    return y, rng.normal(size=n) + 1.2 * y


def test_working_points_equal_probe_rejection_at():
    """The convention bench_metrics' R50 and R30 use: interpolated ROC, capped, flagged."""
    y, s = _binary()
    for eps in (0.5, 0.3):
        rej, eps_b, bound, n_pass, _ = PROBE.rejection_at(y, s, eps)
        wp = FC.working_point(y, s, eps)
        assert wp["rejection"] == pytest.approx(rej) and wp["eps_b"] == pytest.approx(eps_b)
        assert wp["is_bound"] == bound and wp["n_bkg_pass"] == n_pass


def test_a_working_point_with_no_background_passing_is_a_bound_at_the_cap():
    y = np.array([0] * 50 + [1] * 50)
    s = np.r_[np.zeros(50), np.ones(50)]
    r = FC.roc_points(y, s)
    assert all(r["is_bound"]) and set(r["rejection"]) == {50.0} and set(r["n_bkg_pass"]) == {0}
    assert r["eps_s"] == list(FC.EPS_GRID) and r["n_background"] == 50


def test_the_grid_reaches_part_working_points():
    assert {0.5, 0.99, 0.995} <= set(FC.EPS_GRID) and FC.EPS_GRID[0] == 0.05


def _probs(y, k, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(y.size, k))
    z[np.arange(y.size), y] += 1.5
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def test_jetclass_block_is_part_table_1():
    """ParT's working points per class, OvO beside OvR, each class against QCD alone."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 10, 6000)
    p = _probs(y, 10)
    b = FC.jc1_block(p, y, list(FC.LP.JC1_CLASSES))
    assert b["macro_auc_ovo"] == pytest.approx(roc_auc_score(y, p, multi_class="ovo"))
    assert b["macro_auc_ovr"] == pytest.approx(roc_auc_score(y, p, multi_class="ovr"))
    assert "label_QCD" not in b["per_class"] and len(b["per_class"]) == 9
    assert b["per_class"]["label_Hqql"]["eps_s"] == 0.99 and b["per_class"]["label_Tbl"]["eps_s"] == 0.995
    assert b["per_class"]["label_Hbb"]["rejection"]["eps_s"] == 0.5
    m = (y == 1) | (y == 0)
    d = p[m, 1] / (p[m, 1] + p[m, 0])
    assert b["per_class"]["label_Hbb"]["auc"] == pytest.approx(roc_auc_score(y[m] == 1, d))
    with pytest.raises(SystemExit, match="are not"):
        FC.jc1_block(p, y, list(FC.LP.JC1_CLASSES)[::-1])


def test_jetclass2_classes_are_scored_against_the_summed_qcd_classes():
    rng = np.random.default_rng(3)
    y = np.r_[rng.integers(0, 161, 3000), rng.integers(161, 188, 3000)]
    p = _probs(y, 188)
    b = FC.jc2_188_block(p, y)
    assert set(b["per_class"]) == {str(c) for c in np.unique(y) if c < 161}
    c = int(y[0])
    m = (y == c) | (y >= 161)
    d = p[m, c] / (p[m, c] + p[m, 161:].sum(1))
    assert b["per_class"][str(c)]["auc"] == pytest.approx(roc_auc_score(y[m] == c, d))
    with pytest.raises(SystemExit, match="188-class"):
        FC.jc2_188_block(p[:, :162], y)


def test_a_jet_with_both_probabilities_zero_scores_the_tie():
    p = np.array([[0.0, 0.0, 1.0], [0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])
    y = np.array([1, 1, 0])
    t, d = FC.versus(p, y, 1, FC.summed(p, y, (0,)))
    assert t.tolist() == [1, 1, 0] and d[0] == 0.5


def _cell(tree, init, n, seed, done=True, logits=None, y=None):
    c = tree / init / f"N{n}" / f"s{seed}"
    (c / "features").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    y = y if y is not None else (rng.random(500) < 0.5).astype(np.int16)
    logits = logits if logits is not None else np.c_[np.zeros(500), rng.normal(size=500) + y].astype(np.float32)
    np.save(c / "features" / "logits.npy", logits)
    np.save(c / "features" / "label188.npy", y)
    if done:
        (c / "DONE").touch()
    return c, y, logits


def test_cells_read_every_done_cell_and_list_the_rest(tmp_path):
    tree = tmp_path / "leg_top"
    c, y, z = _cell(tree, "l188-s1", 1000, 1)
    _cell(tree, "l188-s1", 10000, 1, done=False)
    _cell(tree, "l188-s1", 1000, 2)
    (tree / "l188-s1" / "N1000" / "s2").rename(tree / "l188-s1" / "N1000" / "s2.partial.1")
    out = tmp_path / "curves" / "top.json"
    FC.main(["cells", "--tree", str(tree), "--task", "top", "--out", str(out)])
    d = json.loads(out.read_text())
    cell = d["cells"]["l188-s1"]["N1000"]["s1"]["top"]
    s = z[:, 1].astype(np.float64) - z[:, 0]          # the softmax's order
    assert cell["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert set(cell) >= {"accuracy", "auc", "r50", "r30", "roc"} and cell["r50"]["eps_s"] == 0.5
    assert list(d["cells"]["l188-s1"]) == ["N1000"] and d["skipped"] == [str(tree / "l188-s1/N10000/s1")]
    with pytest.raises(SystemExit, match="exists"):
        FC.main(["cells", "--tree", str(tree), "--task", "top", "--out", str(out)])


def test_qg_cells_also_give_herwig(tmp_path):
    tree = tmp_path / "leg_qg"
    c, y, z = _cell(tree, "scratch-v2", 1000, 1)
    np.save(c / "features" / "logits_herwig.npy", z[::-1].copy())
    np.save(c / "features" / "label_herwig.npy", y)
    blocks = FC.cell_blocks(c, "qg")
    assert set(blocks) == {"qg", "qg_herwig"} and blocks["qg"]["auc"] != blocks["qg_herwig"]["auc"]


def test_a_cell_that_trained_to_nan_is_fatal(tmp_path):
    tree = tmp_path / "leg_top"
    c, *_ = _cell(tree, "l188-s1", 1000, 1)
    (c / "train.log").write_text("Epoch 3 AvgLoss: nan\n")
    with pytest.raises(SystemExit, match="NaN"):
        FC.main(["cells", "--tree", str(tree), "--task", "top", "--out", str(tmp_path / "o.json")])


def test_jetclass2_cells_read_the_slim_cache_with_the_probe_rows(tmp_path):
    """Stride 1 slim (every extracted jet) of an extraction of every 4th test jet."""
    fd = tmp_path / "l188-s1" / "N1000" / "s1" / FC.FEATURES_188
    fd.mkdir(parents=True)
    y = np.r_[np.arange(188), np.arange(188)].astype(np.int16)
    z = np.random.default_rng(0).normal(size=(y.size, 188)).astype(np.float32)
    np.save(fd / "logits_auc.npy", z)
    np.save(fd / "argmax.npy", z.argmax(1).astype(np.int16))
    np.save(fd / "label188.npy", y)
    (fd / "slim.json").write_text(json.dumps({"auc_stride": 1, "n_rows": y.size, "n_classes": 188}))
    (fd / "extract_manifest.json").write_text(json.dumps({"stride": 4}))
    rows, yy, p, _ = FC.cell_scores(fd.parent, "jc2_188")
    assert rows.tolist() == (4 * np.arange(y.size)).tolist() and np.array_equal(yy, y)
    assert p.shape == (y.size, 188) and np.allclose(p.sum(1), 1)


def test_jetclass_cells_read_pred_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    y = rng.integers(0, 10, 3000)
    p = _probs(y, 10)
    names = [n.removeprefix("label_") for n in FC.LP.JC1_CLASSES]
    monkeypatch.setattr(FC.LEG2, "read_pred_root", lambda path: (names, np.eye(10)[y].astype(np.int32), p))
    b = FC.cell_blocks(tmp_path, "jc1")["jc1"]
    assert b["accuracy"] == pytest.approx((p.argmax(1) == y).mean()) and b["n_jets"] == 3000
    # the macro AUCs on leg2_metrics' jets (every 4th), the per-class AUC on every jet
    assert b["auc_stride"] == 4 and b["n_jets_auc"] == 750
    assert b["macro_auc_ovr"] == pytest.approx(roc_auc_score(y[::4], p[::4], multi_class="ovr"))
    assert b["macro_auc_ovo"] == pytest.approx(roc_auc_score(y[::4], p[::4], multi_class="ovo"))
    m = (y == 1) | (y == 0)
    assert b["per_class"]["label_Hbb"]["auc"] == pytest.approx(
        roc_auc_score(y[m] == 1, p[m, 1] / (p[m, 1] + p[m, 0])))
    bad = np.eye(10)[y].astype(np.int32)
    bad[0, (y[0] + 1) % 10] = 1
    monkeypatch.setattr(FC.LEG2, "read_pred_root", lambda path: (names, bad, p))
    with pytest.raises(SystemExit, match="one-hot"):
        FC.cell_blocks(tmp_path, "jc1")


def test_probes_are_read_at_their_largest_training_set(tmp_path):
    d = tmp_path / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True)
    y, s = _binary(800)
    np.savez(d / "top_N1000.npz", rows=np.arange(y.size), y=y.astype(np.int16), s=s[::-1].astype(np.float32))
    np.savez(d / "top_N1200000.npz", rows=np.arange(y.size), y=y.astype(np.int16), s=s.astype(np.float32))
    np.savez(d / "jc2_188_pairs_qcd_bb_vs_cc_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16),
             s=s.astype(np.float32))
    out = tmp_path / "probes.json"
    FC.main(["probes", "--probs-root", str(tmp_path), "--models", "mtx-l188-s1", "--out", str(out)])
    cell = json.loads(out.read_text())["cells"]["mtx-l188-s1"]["best70"]
    assert list(cell) == ["top"] and cell["top"]["n_train"] == 1_200_000
    assert cell["top"]["auc"] == pytest.approx(roc_auc_score(y, s.astype(np.float32)))
    with pytest.raises(SystemExit, match="no saved outputs"):
        FC.main(["probes", "--probs-root", str(tmp_path), "--models", "init-s1", "--out", str(tmp_path / "x.json")])


def test_the_v4_pairs_and_balanced_set_are_read(tmp_path):
    d = tmp_path / "mtx-l188-s1" / "best70"
    d.mkdir(parents=True)
    y, s = _binary(800)
    np.savez(d / "jc2bal_pairs_X_bb_vs_X_cc.npz", rows=np.arange(y.size), y=y.astype(np.int16),
             s=s.astype(np.float32))
    out = tmp_path / "probes.json"
    FC.main(["probes", "--probs-root", str(tmp_path), "--out", str(out)])
    cell = json.loads(out.read_text())["cells"]["mtx-l188-s1"]["best70"]
    assert list(cell) == ["jc2bal_pairs.X_bb_vs_X_cc"]
    assert cell["jc2bal_pairs.X_bb_vs_X_cc"]["auc"] == pytest.approx(roc_auc_score(y, s.astype(np.float32)))
    assert len(cell["jc2bal_pairs.X_bb_vs_X_cc"]["roc"]["eps_s"]) == len(FC.EPS_GRID)
    assert FC.PROBE_FILE.match("jc2bal_N1000000.npz").group(1) == "jc2bal"


# ------------------------------------------------------------------ the curves jobs
BC = _load("build_curve_jobs", "scripts/build_curve_jobs.py")


@pytest.fixture(scope="module")
def specs():
    return BC.build(BC.PIN)


def test_the_specs_on_disk_are_the_generators(specs):
    for p, text in specs.items():
        assert p.read_text() == text, f"{p} differs from scripts/build_curve_jobs.py; regenerate"


def test_every_job_is_ours_cpu_only_pinned_and_tried_once(specs):
    jobs = {p: yaml.safe_load(t) for p, t in specs.items() if p.parent == BC.OUT_DIR}
    assert {d["metadata"]["name"] for d in jobs.values()} == {
        f"{w}-{x}-raunav" for w in ("curves", "boot") for x in ("ft", "jc2", "probe")}
    for p, d in jobs.items():
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert d["spec"]["backoffLimit"] == 1 and "nvidia.com/gpu" not in c["resources"]["limits"]
        assert f'--branch "{BC.PIN}"' in c["args"][0] and "configmap" not in specs[p].lower()


def test_the_curves_jobs_wait_for_the_reported_runs_and_write_one_file_per_tree(specs):
    text = specs[BC.OUT_DIR / "job-curves-ft-raunav.yaml"]
    assert "/best70/leg2/{" + ",".join(BC.FT_RUNS) + "}/N{1000,10000,100000,1000000}/s1/DONE" in text
    assert "/ref_v100/bench/leg_qg/scratch-v2/N{1000,10000,100000,1600000}/s{1,2,3}/DONE" in text
    for n in ("best70_leg2", "best70_leg_top", "best70_leg_qg", "ref_v100_jc1_leg2"):
        assert f"--out {BC.CURVES}/{n}.json" in text
    jc2 = specs[BC.OUT_DIR / "job-curves-jc2-raunav.yaml"]
    assert "--task jc2_188" in jc2 and f"/best70/{BC.LEG1_188}/" in jc2
    probe = specs[BC.OUT_DIR / "job-curves-probe-raunav.yaml"]
    assert f"{BC.FITS_V3}/{{mtx-l188-s1," in probe and "init-s3}.json" in probe


def test_the_sixteen_runs_are_paper_runs():
    assert len(BC.FT_RUNS) == 16 and {f"mtx-{r}" for r in BC.FT_RUNS} <= set(BC.BF.v2_paper_runs())
    assert len(BC.PROBE_MODELS) == 30


def _run_guard(tmp_path, outs, inputs):
    script = "set -euo pipefail\n" + BC._guard(outs, inputs) + "echo READY\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_wait_refuses_an_existing_output_and_gives_up_with_exit_42(tmp_path, monkeypatch):
    monkeypatch.setattr(BC, "GIVE_UP_S", 0)
    monkeypatch.setattr(BC, "POLL_S", 0)
    inp, out = tmp_path / "a" / "DONE", tmp_path / "out.json"
    r = _run_guard(tmp_path, [str(out)], [str(inp)])
    assert r.returncode == 42 and "still missing" in r.stdout
    inp.parent.mkdir()
    inp.touch()
    r = _run_guard(tmp_path, [str(out)], [str(tmp_path) + "/{a,a}/DONE"])
    assert r.returncode == 0 and "READY" in r.stdout
    out.touch()
    r = _run_guard(tmp_path, [str(out)], [str(inp)])
    assert r.returncode == 42 and "exists" in r.stdout
    r = _run_guard(tmp_path, [], [str(inp)])                    # no outputs to refuse
    assert r.returncode == 0 and "READY" in r.stdout


def test_a_retried_curves_pod_skips_the_trees_already_written(tmp_path, monkeypatch):
    """An eviction after the first trees: the retry writes only the rest."""
    names = ["best70_leg2", "best70_leg_top"]
    monkeypatch.setattr(BC, "CURVES", str(tmp_path))
    monkeypatch.setattr(BC, "done_markers", lambda n: str(tmp_path / "DONE"))
    (tmp_path / "DONE").touch()
    (tmp_path / "best70_leg2.json").write_text("{}")
    log = tmp_path / "calls"
    script = ("set -euo pipefail\nhalt () { exit 42; }\n"
              f"python3 () {{ echo \"$*\" >> {log}; }}\n" + BC.curves_body(names))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and f"skip {tmp_path}/best70_leg2.json" in r.stdout
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and calls[0].endswith(f"--out {tmp_path}/best70_leg_top.json")
