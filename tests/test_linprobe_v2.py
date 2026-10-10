"""The downstream linear probes of the v2 models: experiments/EVAL/linear_probe_v2.py on synthetic
features laid out as extract_v2.py writes them, and the jobs scripts/build_linprobe_jobs.py emits."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess

import numpy as np
import pytest
import yaml

pytest.importorskip("sklearn")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


LP = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
L = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")


def _split(d: pathlib.Path, x: np.ndarray, y: np.ndarray, ckpt_sha="abc", rows=None):
    d.mkdir(parents=True)
    np.save(d / "features.npy", x.astype(np.float16))
    np.save(d / "label188.npy", y.astype(np.int16))
    np.save(d / "rows.npy", np.arange(y.size) if rows is None else rows)
    (d / "manifest.json").write_text(json.dumps({"n_feature_rows": int(y.size), "n_stream": int(y.size),
                                                 "checkpoint_sha256": ckpt_sha}))


def _jets(rng, y, k, sep=2.0):
    """Features whose class means differ: a linear probe separates them."""
    means = rng.normal(size=(k, 128))
    return sep * means[y] + rng.normal(size=(y.size, 128))


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(7)
    m = tmp_path / "mtx-l188-s1"
    l162 = _load("label_recovery", "experiments/EVAL/label_recovery.py").rung_maps()["L162"]
    native = np.array(sorted(l162))
    for ds, k in (("top", 2), ("qg", 2), ("jc1", 10), ("jc2", 162)):
        means = rng.normal(size=(k, 128))
        mk = lambda y: 2.0 * means[y] + rng.normal(size=(y.size, 128))
        for n in (300, 3000):
            y = rng.integers(0, k, n)
            _split(m / ds / f"train_N{n}" / "best70", mk(y), y)
        y = rng.integers(0, k, 2000)
        _split(m / ds / "val" / "best70", mk(y), y)
        if ds == "jc2":                       # the test stream carries native labels
            nat = rng.choice(native, 5000)
            groups = np.array([l162[a] for a in nat])
            _split(m / ds / "test" / "best70", mk(groups), nat)
        else:
            y = rng.integers(0, k, 4000)
            _split(m / ds / "test" / "best70", mk(y), y)
        if ds == "qg":
            y = rng.integers(0, 2, 3000)
            _split(m / ds / "herwig" / "best70", mk(y) + 0.3, y)
    return tmp_path


def test_every_dataset_and_size_is_probed_and_scored_with_fine_tuning_metrics(root, monkeypatch):
    monkeypatch.setattr(LP, "TEST_FIRST", {"jc2": 4000})
    out = root / "fits"
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                    "--out", str(out)]) == 0
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    cells = doc["cells"]["best70"]
    assert set(cells) == {"top", "qg", "qg_herwig", "jc1", "jc2", "jc2_pairs"}
    pairs = cells.pop("jc2_pairs")
    assert set(pairs) == {"3000"} and set(pairs["3000"]) == set(LP.PAIRS)      # the largest subset only
    for c in pairs["3000"].values():
        assert c["auc"] > 0.8 and c["C"] in LP.C_GRID and 10 <= c["n_train"] < 3000
    for ds in cells:
        assert set(cells[ds]) == {"300", "3000"}
        for n, c in cells[ds].items():
            assert c["C"] in LP.C_GRID and set(c["search"]) == {str(v) for v in LP.C_GRID}
            assert c["n_train"] == int(n) and c["accuracy"] > 0.5
    for ds in ("top", "qg", "qg_herwig"):
        c = cells[ds]["3000"]
        assert c["auc"] > 0.9 and c["r50"] > 1 and {"r30", "log1m_auc", "r50_is_bound"} <= set(c)
    for ds, k in (("jc1", 10), ("jc2", 162)):
        c = cells[ds]["3000"]
        assert c["macro_auc_ovr"] > 0.9 and c["auc_stride"] == 4
    assert cells["jc2"]["3000"]["n_jets"] == 4000          # the first TEST_FIRST jets only
    assert cells["jc2"]["3000"]["n_jets_auc"] == 1000
    # more training jets do not make a separable task worse
    assert cells["jc1"]["3000"]["accuracy"] >= cells["jc1"]["300"]["accuracy"] - 0.02
    # a second call finds the fit and does nothing
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                    "--out", str(out)]) == 0


def test_an_unfinished_or_reordered_extraction_is_fatal(root, tmp_path):
    (root / "mtx-l188-s1/top/val/best70/manifest.json").unlink()
    with pytest.raises(SystemExit, match="did not finish"):
        LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--datasets", "top",
                 "--checkpoints", "best70", "--out", str(tmp_path / "f")])
    d = root / "mtx-l188-s1/qg/val/best70"
    np.save(d / "rows.npy", np.arange(2000)[::-1].copy())
    with pytest.raises(SystemExit, match="not the stream's"):
        LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--datasets", "qg",
                 "--checkpoints", "best70", "--out", str(tmp_path / "f")])


def test_a_class_missing_from_a_small_training_subset_gets_probability_zero():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    clf = LogisticRegression().fit(rng.normal(size=(200, 4)) + y[:, None], y)
    p = LP.full_proba(clf, rng.normal(size=(50, 4)), 5)
    assert p.shape == (50, 5) and np.allclose(p.sum(1), 1) and (p[:, 3:] == 0).all()
    assert LP.cross_entropy(p, np.full(50, 4)) == pytest.approx(-np.log(LP.P_FLOOR))


# ------------------------------------------------------------------ the jobs
SPECS = L.build()
GRID_RUNS = L.BX.v2_runs()


def _args(spec: str) -> str:
    return yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_two_jobs_per_model_for_every_run_and_untrained_trunk():
    models = [m for m, *_ in L.models()]
    assert len(models) == len(GRID_RUNS) + 3 == 40
    assert sorted(SPECS) == sorted(f"job-linprobe-{kind}-{m.removeprefix('mtx-')}-raunav.yaml"
                                   for m in models for kind in ("x", "fit"))
    for name, spec in SPECS.items():
        d = yaml.safe_load(spec)
        assert d["metadata"]["name"].endswith("-raunav") and name == f"job-{d['metadata']['name']}.yaml"
        a = _args(spec)
        assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0, name
        pin = L.LP_PIN if "-x-" in name else L.FIT_PIN
        assert f'--branch "{pin}"' in a and '[ "${USED}" -lt 85 ]' in a
        assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]


def test_extraction_reads_every_split_fine_tuning_reads_and_never_takes_a_3090():
    for model, run, rung, k, reg, ckpts in L.models():
        spec = SPECS[f"job-linprobe-x-{model.removeprefix('mtx-')}-raunav.yaml"]
        a = _args(spec)
        ex = {e["key"]: e for e in yaml.safe_load(spec)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]}
        assert ex["nvidia.com/gpu.product"] == {"key": "nvidia.com/gpu.product", "operator": "NotIn",
                                                "values": ["NVIDIA-GeForce-RTX-3090"]}
        assert set(L.BX.V2_GPU_FAULT_NODES) <= set(ex["kubernetes.io/hostname"]["values"])
        assert f"--run-dir {L.BX.V2_ROOT}/{run} --rung {rung} --num-classes {k} --num-reg {reg}" in a
        assert f"--checkpoints {' '.join(ckpts)} " in a
        assert ckpts == (("init",) if model.startswith("init-") else ("best70", "best70_bn"))
        assert "--no-pooled" in a and "--prefix-features 2000000000" in a and "--head-prefix 0" in a
        res = yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"]["memory"] == res["limits"]["memory"] == "88Gi"   # fine-tuning's, same files
        assert f"OUT={L.LP_ROOT}/{model}/$1/$2" in a
        got = re.findall(r"^extract (\w+) (\w+) (\S+) (\d+) (.+)$", a, re.M)
        want = {"jc2": L.BF.SIZES, "jc1": L.BF.SIZES, "top": L.BF.BENCH_SIZES["top"], "qg": L.BF.BENCH_SIZES["qg"]}
        for ds, sizes in want.items():
            assert [s for d, s, *_ in got if d == ds] == [f"train_N{n}" for n in sizes] + (
                ["val", "test", "herwig"] if ds == "qg" else ["val", "test"])
        calls = {(d, s): (c, int(m), f) for d, s, c, m, f in got}
        assert calls[("jc2", "test")] == ("configs/data/JetClassII_base.yaml", 2_000_000, L.BF.test2m_list())
        assert calls[("jc1", "test")][2] == "${TEST1}" and calls[("top", "test")][2] == "/data/finetune/top/top_test.parquet"
        assert all(m == 0 for (d, s), (c, m, f) in calls.items() if (d, s) != ("jc2", "test"))
        assert calls[("jc2", "train_N1000000")] == ("configs/finetune/JetClassII_L162_noweight.yaml", 0,
                                                     "/data/finetune/jc2_v2/train_N1000000_s1.parquet")
        assert a.count("python3 experiments/EVAL/extract_v2.py") == 1      # one function, every call


def test_the_fit_job_reads_the_features_the_extraction_writes():
    for model, *_, ckpts in L.models():
        spec = SPECS[f"job-linprobe-fit-{model.removeprefix('mtx-')}-raunav.yaml"]
        a = _args(spec)
        assert (f"linear_probe_v2.py --root {L.LP_ROOT} --model {model}" in a
                and f"--checkpoints {' '.join(ckpts)} --out {L.LP_ROOT}/{L.FITS}" in a)
        feats = " ".join(f"{c}={L.LP_ROOT}/{model}/jc2/test/{c}" for c in ckpts)
        assert (f"mass_resolution.py --observers {L.MASS_OBS}" in a and f"--features {feats} " in a
                and f"--out {L.LP_ROOT}/mass/{model} || halt" in a)


def test_the_extraction_can_leave_out_the_pooled_embedding():
    src = (ROOT / "experiments/EVAL/extract_v2.py").read_text()
    assert '"--no-pooled", action="store_true"' in src
    assert "pooled_factory=None if a.no_pooled else PooledTap" in src
    for path, text in {**L.NEEDED, **L.FIT_NEEDED}.items():
        assert text in (ROOT / path).read_text(), path


def test_the_summary_is_in_the_fine_tuning_read_outs_format(root, monkeypatch, tmp_path):
    S = _load("linear_probe_summary", "experiments/EVAL/linear_probe_summary.py")
    monkeypatch.setattr(LP, "TEST_FIRST", {"jc2": 4000})
    fits = root / "fits"
    for c in ("best70",):
        assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", c, "--out", str(fits)]) == 0
    doc = json.loads((fits / "mtx-l188-s1.json").read_text())
    doc["cells"]["best70_bn"] = doc["cells"]["best70"]          # the twin, as a second rule
    (fits / "mtx-l188-s1.json").write_text(json.dumps(doc))
    monkeypatch.setattr(S, "expected_models", lambda: ["mtx-l188-s1"])
    assert S.main(["--fits", str(fits), "--out", str(tmp_path / "v2")]) == 0
    leg1 = json.loads((tmp_path / "v2/finetune/linprobe_leg1_metrics.json").read_text())
    assert set(leg1["cells"]) == {"l188-s1"} and set(leg1["cells"]["l188-s1"]) == {"300", "3000"}
    assert leg1["cells"]["l188-s1"]["3000"]["s1"]["macro_auc_ovr"] == doc["cells"]["best70"]["jc2"]["3000"]["macro_auc_ovr"]
    bench = json.loads((tmp_path / "v2/benchmarks/linprobe_bn_bench_metrics.json").read_text())
    assert set(bench["cells"]) == {"top", "qg"} and bench["checkpoint"] == "best70_bn"
    pairs = json.loads((tmp_path / "v2/finetune/linprobe_pair_metrics.json").read_text())
    assert set(pairs["cells"]) == {"bc_vs_bq_cs", "bb_vs_cc", "bbqq_vs_ccqq"}
    assert set(pairs["cells"]["bb_vs_cc"]["l188-s1"]) == {"3000"}
    her = json.loads((tmp_path / "v2/benchmarks/linprobe_bench_metrics_herwig.json").read_text())
    assert set(her["cells"]) == {"qg"} and "r50" in her["cells"]["qg"]["l188-s1"]["3000"]["s1"]
    monkeypatch.setattr(S, "expected_models", lambda: ["mtx-l188-s1", "mtx-l188-s2"])
    with pytest.raises(SystemExit, match="no linear-probe fit"):
        S.main(["--fits", str(fits), "--out", str(tmp_path / "v2")])


@pytest.mark.parametrize("k, absent", [(12, (3, 7)), (2, ())])
def test_the_torch_fit_reaches_scikit_learns_minimum(k, absent):
    """The probe minimises scikit-learn's LogisticRegression objective (lbfgs, L2, unpenalised
    intercept; softmax over the classes present, one sigmoid logit for two); at a C where both
    converge, the probabilities agree, including a class absent from training."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    y = rng.integers(0, k, 4000)
    y = y[~np.isin(y, absent)]
    x = _jets(rng, y, k, sep=0.3)
    xt = _jets(rng, rng.integers(0, k, 1000), k, sep=0.3)
    sk = LogisticRegression(C=0.1, max_iter=5000, tol=1e-10).fit(x, y)
    ours = LP.Logit(np.unique(y), "cpu")
    assert ours.fit(x, y, 0.1, tol=1e-10) < LP.MAX_ITER
    idx = np.searchsorted(np.unique(y), y)

    def objective(m):     # scikit-learn's, over N: mean log-loss + ||W||^2 / (2 C N)
        p = np.clip(m.predict_proba(x)[np.arange(y.size), idx], 1e-300, None)
        return -np.log(p).mean() + (m.coef_ ** 2).sum() / (2 * 0.1 * y.size)
    # the same minimum (it is flat: 1e-11 in the objective moves probabilities by ~1e-5)
    assert abs(objective(ours) - objective(sk)) < 1e-9
    np.testing.assert_allclose(ours.predict_proba(xt), sk.predict_proba(xt), atol=1e-4)
    p = LP.full_proba(ours, xt, k)
    assert np.allclose(p.sum(1), 1) and (p[:, list(absent)] == 0).all()


def test_the_fit_job_runs_on_a_gpu_that_is_not_the_grids():
    for model, *_ in L.models():
        spec = SPECS[f"job-linprobe-fit-{model.removeprefix('mtx-')}-raunav.yaml"]
        c = yaml.safe_load(spec)["spec"]["template"]["spec"]
        assert c["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
        ex = {e["key"]: e for e in c["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"][0]["matchExpressions"]}
        assert ex["nvidia.com/gpu.product"]["operator"] == "NotIn"
        assert ex["nvidia.com/gpu.product"]["values"] == list(L.NOT_ON)
