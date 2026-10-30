"""experiments/STATS/paired_bootstrap.py (paired bootstrap over test jets for ratios of 1 - AUC)
and the bootstrap jobs of scripts/build_curve_jobs.py."""
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PB = _load("paired_bootstrap", "experiments/STATS/paired_bootstrap.py")


def _binary(n=3000, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return y, rng.normal(size=n) + sep * y


def test_a_resample_is_scored_as_the_repeated_jets_ties_included():
    """src/stats/paired.AucScorer, tested in src/stats/tests/test_paired.py, as used here."""
    y, s = _binary(800)
    s = np.round(s, 1)                               # many ties
    c = np.random.default_rng(1).integers(0, 4, (3, y.size)).astype(np.uint8)
    full, per = PB.one_minus_auc(s, y == 1, c)
    assert full == pytest.approx(1 - roc_auc_score(y, s), abs=1e-12)
    for k in range(3):
        rep = np.repeat(np.arange(y.size), c[k])
        assert per[k] == pytest.approx(1 - roc_auc_score(y[rep], s[rep]), abs=1e-12)


def test_resamples_keep_the_class_counts_and_depend_on_seed_and_index_only():
    y = np.r_[np.zeros(700, int), np.ones(200, int), np.full(100, 2)]
    c = PB.counts(y, 5, 7)
    for k in range(5):
        assert [int(c[k][y == j].sum()) for j in range(3)] == [700, 200, 100]
    assert np.array_equal(c, PB.counts(y, 5, 7)) and np.array_equal(c[:3], PB.counts(y, 3, 7))
    assert not np.array_equal(c[0], c[1]) and not np.array_equal(c, PB.counts(y, 5, 8))


def _npz(path, y, rows=None, **kw):
    np.savez(path, rows=np.arange(y.size) if rows is None else rows, y=y.astype(np.int16), **kw)
    return f"{path.stem}={path}"


def test_ratios_of_paired_models(tmp_path):
    y, s = _binary()
    weak = s + np.random.default_rng(2).normal(size=y.size)
    a = _npz(tmp_path / "a.npz", y, s=s.astype(np.float32))
    b = _npz(tmp_path / "b.npz", y, s=weak.astype(np.float32))
    a2 = _npz(tmp_path / "a2.npz", y, s=s.astype(np.float32))
    out = tmp_path / "boot.json"
    PB.main(["--model", a, "--model", b, "--model", a2, "--pair", "b/a", "--pair", "a2/a", "--b", "50",
             "--out", str(out)])
    d = json.loads(out.read_text())
    m = d["models"]
    assert m["a"]["auc"]["one_minus_auc"] == pytest.approx(1 - roc_auc_score(y, s.astype(np.float32)))
    assert m["a"]["auc"]["sd"] > 0 and m["b"]["auc"]["sd"] > 0
    r = d["pairs"]["b/a"]["auc"]
    assert r["ratio"] == pytest.approx(m["b"]["auc"]["one_minus_auc"] / m["a"]["auc"]["one_minus_auc"])
    assert r["ratio"] > 1 and r["sd"] > 0 and r["ln_ratio_sd"] > 0
    # the same scores on the same resamples: the ratio is exactly one in every resample
    assert d["pairs"]["a2/a"]["auc"] == {"ratio": 1.0, "sd": 0.0, "ln_ratio_sd": 0.0}
    assert set(d["pairs"]) == {"b/a", "a2/a"} and d["n_jets"] == y.size and d["b"] == 50
    with pytest.raises(SystemExit, match="exists"):
        PB.main(["--model", a, "--model", b, "--out", str(out)])


def test_models_on_other_jets_are_refused_unless_reduced_to_one_models_rows(tmp_path):
    y, s = _binary(400)
    a = _npz(tmp_path / "a.npz", y, s=s)
    flipped = y.copy()
    flipped[0] = 1 - flipped[0]
    b = _npz(tmp_path / "b.npz", flipped, s=s)
    with pytest.raises(SystemExit, match="same jets"):
        PB.main(["--model", a, "--model", b, "--b", "5", "--out", str(tmp_path / "o.json")])
    # a fine-tune on every jet, a probe on every 4th: reduced to the probe's rows
    every = _npz(tmp_path / "every.npz", np.repeat(y, 4), rows=np.arange(4 * y.size), s=np.repeat(s, 4))
    fourth = _npz(tmp_path / "fourth.npz", np.repeat(y, 4)[::4], rows=np.arange(0, 4 * y.size, 4),
                  s=np.repeat(s, 4)[::4])
    with pytest.raises(SystemExit, match="same jets"):
        PB.main(["--model", every, "--model", fourth, "--b", "5", "--out", str(tmp_path / "o.json")])
    out = tmp_path / "ok.json"
    PB.main(["--model", every, "--model", fourth, "--take-rows-of", "fourth", "--pair", "every/fourth",
             "--b", "5", "--out", str(out)])
    assert json.loads(out.read_text())["pairs"]["every/fourth"]["auc"]["ratio"] == 1.0
    with pytest.raises(SystemExit, match="not models"):
        PB.main(["--model", a, "--pair", "a/zz", "--out", str(tmp_path / "x.json")])


def test_class_probabilities_give_the_macro_and_per_class_auc(tmp_path):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 2000)
    z = rng.normal(size=(y.size, 4))
    z[np.arange(y.size), y] += 1.0
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    a = _npz(tmp_path / "a.npz", y, p=p.astype(np.float16))
    out = tmp_path / "boot.json"
    PB.main(["--model", a, "--per-class", "--b", "10", "--out", str(out)])
    m = json.loads(out.read_text())["models"]["a"]
    q = p.astype(np.float16).astype(np.float64)
    per = [roc_auc_score(y == k, q[:, k]) for k in range(4)]
    assert m["macro"]["one_minus_auc"] == pytest.approx(1 - np.mean(per), abs=1e-12)
    for k in range(4):
        assert m[f"class_{k}"]["one_minus_auc"] == pytest.approx(1 - per[k], abs=1e-12)


def _top_cell(tmp_path, y, logits):
    fd = tmp_path / "N1000" / "s1" / "features"
    fd.mkdir(parents=True)
    np.save(fd / "logits.npy", logits.astype(np.float32))
    np.save(fd / "label188.npy", y.astype(np.int16))
    return fd.parent


def test_fine_tuning_cells_are_checked_as_ft_curves_checks_them(tmp_path, monkeypatch):
    y, s = _binary(500)
    cell = _top_cell(tmp_path, y, np.c_[np.zeros(y.size), s])
    rows, yy, score = PB.load(cell, "top")
    assert np.array_equal(rows, np.arange(y.size)) and np.array_equal(yy, y)
    assert roc_auc_score(y, score) == pytest.approx(roc_auc_score(y, s))
    ft = PB._ft_curves()
    seen = []
    monkeypatch.setattr(ft.BM, "check_label_order", seen.append)
    PB.load(cell, "top")
    assert seen == ["top"]
    (cell / "train.log").write_text("Epoch 3 AvgLoss: nan\n")
    with pytest.raises(SystemExit, match="NaN"):
        PB.load(cell, "top")


def test_jetclass_cells_are_bootstrapped_on_the_tables_jets(tmp_path, monkeypatch):
    """leg2_metrics' macro AUC reads every 4th jet of pred.root; so does the bootstrap."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 10, 4000)
    p = rng.dirichlet(np.ones(10), y.size)
    ft = PB._ft_curves()
    names = [n.removeprefix("label_") for n in ft.LP.JC1_CLASSES]
    monkeypatch.setattr(ft.LEG2, "read_pred_root", lambda path: (names, np.eye(10)[y].astype(np.int32), p))
    rows, yy, pp = PB.load(tmp_path, "jc1")
    assert ft.JC1_AUC_STRIDE == 4 and np.array_equal(rows, np.arange(0, y.size, 4))
    assert np.array_equal(yy, y[::4]) and np.allclose(pp, p[::4])


# ------------------------------------------------------------------ the bootstrap jobs
BC = _load("build_curve_jobs_boot", "scripts/build_curve_jobs.py")


def test_ratio_pairs():
    runs = ["l188-s1", "l162-s1", "l162mass-s1", "l188-s2", "l162-s2", "rand2p2-s2"]
    got = BC.ratio_pairs(runs, "mtx-", lambda k: f"init-s{k}")
    assert got == ["mtx-l188-s1/init-s1", "mtx-l162-s1/mtx-l188-s1", "mtx-l162-s1/init-s1",
                   "mtx-l162mass-s1/mtx-l188-s1", "mtx-l162mass-s1/mtx-l162-s1", "mtx-l162mass-s1/init-s1",
                   "mtx-l188-s2/init-s2", "mtx-l162-s2/mtx-l188-s2", "mtx-l162-s2/init-s2",
                   "mtx-rand2p2-s2/mtx-l188-s2", "mtx-rand2p2-s2/init-s2"]


def _boot_call(tmp_path, fn: str, call: str) -> list[str]:
    """Run the job's bash boot function with python3 stubbed; the arguments it passed."""
    log = tmp_path / "argv"
    script = ("set -euo pipefail\nhalt () { exit 42; }\n"
              f"python3 () {{ printf '%s\\n' \"$@\" > {log}; }}\n"
              f"TMP={tmp_path}\nPAIRS=(--pair x/y)\n" + fn + call + "\n")
    subprocess.run(["bash", "-c", script], check=True)
    return log.read_text().split("\n")[:-1]


def test_the_fine_tuning_bootstrap_reads_the_reported_runs_and_the_references(tmp_path):
    fn = BC.BOOT_FT.format(runs=" ".join(BC.FT_RUNS), seeds="2 3", scratch=BC.SCRATCH, base=BC.BASELINE)
    argv = _boot_call(tmp_path, fn, "boot top /T /R 1000 /S")
    models = [argv[i + 1] for i, a in enumerate(argv) if a == "--model"]
    assert models[0] == "l188-s1=/T/l188-s1/N1000/s1" and len(models) == 19
    # seed 1 from the reference's tree, seeds 2-3 from the subset-s1 tree
    assert models[-3:] == [f"{BC.BASELINE}-1=/R/{BC.SCRATCH}/N1000/s1",
                           f"{BC.BASELINE}-2=/S/{BC.SCRATCH}/N1000/s2",
                           f"{BC.BASELINE}-3=/S/{BC.SCRATCH}/N1000/s3"]
    assert BC.BASELINE == "scratch-s1"
    assert argv[:3] == ["experiments/STATS/paired_bootstrap.py", "--task", "top"]
    assert argv[-2:] == ["--out", f"{tmp_path}/top_N1000.json"] and "x/y" in argv


def test_the_probe_bootstrap_reads_every_probe_at_its_checkpoint(tmp_path):
    fn = BC.BOOT_PROBE.format(models="mtx-l188-s1:best70 init-s1:init", probs="/P")
    argv = _boot_call(tmp_path, fn, "boot jc2_188 1000000")
    models = [argv[i + 1] for i, a in enumerate(argv) if a == "--model"]
    assert models == ["mtx-l188-s1=/P/mtx-l188-s1/best70/jc2_188_N1000000.npz",
                      "init-s1=/P/init-s1/init/jc2_188_N1000000.npz"]
    assert "npz" in argv and argv[-1] == f"{tmp_path}/jc2_188_N1000000.json"


def test_the_bootstrap_specs_cover_every_task_and_size():
    ft, jc2, probe = (BC.boot_ft_body("ft"), BC.boot_ft_body("jc2"), BC.boot_probe_body())
    for task, n in [("jc1", 1000), ("top", 1200000), ("qg", 1600000), ("qg_herwig", 1000)]:
        assert f"boot {task} " in ft and f" {n}\n" in ft
    calls = [ln for ln in ft.splitlines() if ln.startswith("          boot ") and "()" not in ln]
    assert len(calls) == 16 and jc2.count("\n          boot jc2_188 ") == 4
    assert "--pair l162mass-s1/l162-s1" in ft and "--pair l188-s1/l188-s1" not in ft
    assert f"--pair r16q1-s3/{BC.BASELINE}-3 " in ft
    assert "boot jc2_188_pairs_qcd_bb_vs_cc 1000000" in probe and "boot qg_herwig 100000" in probe
    assert "--pair mtx-flavf1r-s2/init-s2" in probe
    for body, out in ((ft, "ft"), (jc2, "jc2"), (probe, "probe")):
        assert f"mv ${{TMP}} {BC.BOOT}/{out}" in body
