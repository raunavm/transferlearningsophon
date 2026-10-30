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


# ------------------------------------------------------------------ the 188-class task (v3)
@pytest.fixture
def root188(tmp_path):
    """jc2 features whose cached training/validation labels are the 162 groups (as the L162
    extraction wrote them), native labels in the subsets' parquet files, a native test stream,
    and top/qg for the nonlinear probe."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rng = np.random.default_rng(11)
    m = tmp_path / "mtx-l188-s1"
    pq_dir = tmp_path / "jc2_v2"
    pq_dir.mkdir()
    l162 = _load("label_recovery", "experiments/EVAL/label_recovery.py").rung_maps()["L162"]
    means = rng.normal(size=(188, 128))
    mk = lambda y: 2.0 * means[y] + rng.normal(size=(y.size, 128))
    grp = lambda y: np.array([l162[a] for a in y])
    for split, n in (("train_N300", 300), ("train_N4000", 4000), ("val", 2000)):
        nat = rng.permutation(np.resize(np.arange(188), n))
        _split(m / "jc2" / split / "best70", mk(nat), grp(nat))
        name = "val.parquet" if split == "val" else f"{split}_s1.parquet"
        pq.write_table(pa.table({"jet_label": nat.astype(np.int64)}), pq_dir / name)
    nat = rng.permutation(np.resize(np.arange(188), 6000))
    _split(m / "jc2" / "test" / "best70", mk(nat), nat)
    for ds in ("top", "qg"):
        mb = rng.normal(size=(2, 128))
        mkb = lambda y: 1.0 * mb[y] + rng.normal(size=(y.size, 128))
        for split, n in (("train_N300", 300), ("train_N3000", 3000), ("val", 2000), ("test", 3000)):
            y = rng.integers(0, 2, n)
            _split(m / ds / split / "best70", mkb(y), y)
        if ds == "qg":
            y = rng.integers(0, 2, 2000)
            _split(m / ds / "herwig" / "best70", mkb(y) + 0.3, y)
    return tmp_path, pq_dir


def test_the_188_class_task_reads_native_labels_and_reports_resonant_and_qcd_parts(root188, monkeypatch):
    root, pq_dir = root188
    monkeypatch.setattr(LP, "TEST_FIRST", {"jc2": 6000})
    out, probs = root / "fits", root / "probs"
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                    "--datasets", "jc2_188", "top", "--out", str(out), "--probs-out", str(probs),
                    "--jc2-parquet", str(pq_dir)]) == 0
    cells = json.loads((out / "mtx-l188-s1.json").read_text())["cells"]["best70"]
    assert set(cells) == {"jc2_188", "jc2_188_pairs", "top"}
    c = cells["jc2_188"]["4000"]
    assert c["n_classes_in_train"] == 188 and c["n_classes_absent_train"] == 0
    assert c["macro_auc_ovr"] > 0.9 and c["macro_auc_qcd"] > 0.9 and c["macro_auc_resonant"] > 0.9
    assert len(c["per_class_auc"]) == 188 and len(c["rejection_vs_qcd"]) == 161
    assert c["macro_auc_ovr"] == pytest.approx(np.mean(list(c["per_class_auc"].values())))
    assert c["n_jets_auc"] == 1500 and c["auc_stride"] == 4
    small = cells["jc2_188"]["300"]                       # every class seen at 300 jets here
    assert small["macro_auc_trained"] == small["macro_auc_ovr"]
    assert set(cells["jc2_188_pairs"]) == {"4000"}
    assert set(cells["jc2_188_pairs"]["4000"]) == set(LP.PAIRS) | set(LP.QCD_PAIRS)
    for name in ("jc2_188_N4000", "top_N300", "top_N3000", "jc2_188_pairs_qcd_bb_vs_cc_N4000"):
        z = np.load(probs / "mtx-l188-s1" / "best70" / f"{name}.npz")
        assert {"rows", "y", "coef", "intercept", "scaler_mean", "scaler_scale", "C"} <= set(z.files)
    z = np.load(probs / "mtx-l188-s1" / "best70" / "jc2_188_N4000.npz")
    assert z["p"].shape == (1500, 188) and z["p"].dtype == np.float16
    assert not (probs / "mtx-l188-s1" / "best70" / "jc2_188_N300.npz").exists()   # largest N only
    # every search records the final gradient and calls a fit converged only below it
    for s in c["search"].values():
        assert s["converged"] == (s["n_iter"] < LP.MAX_ITER and s["grad_max"] <= LP.CONVERGED_GRAD)


def test_a_parquet_that_does_not_match_the_cached_groups_is_fatal(root188, monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, pq_dir = root188
    y = pq.read_table(pq_dir / "val.parquet").column(0).to_numpy()
    pq.write_table(pa.table({"jet_label": np.roll(y, 1)}), pq_dir / "val.parquet")
    with pytest.raises(SystemExit, match="row for row"):
        LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                 "--datasets", "jc2_188", "--out", str(tmp_path / "f"), "--jc2-parquet", str(pq_dir)])


def test_the_qcd_flavour_pairs_are_defined_on_quark_content():
    lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
    import csv
    with lr.MAP.open() as f:
        names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader(f)}
    g = LP.native_pair_groups(names)
    assert g["qcd_bb_vs_cc"] == ((169,), (181,))
    assert g["qcd_b_vs_light"] == (tuple(range(161, 179)), (187,))
    assert g["qcd_c_vs_light"] == (tuple(range(179, 185)), (187,))
    assert LP.QCD_NATIVE == tuple(range(161, 188))


def test_the_nonlinear_probe_scores_the_binary_tasks_at_their_largest_training_set(root188, monkeypatch):
    pytest.importorskip("torch")
    root, pq_dir = root188
    monkeypatch.setattr(LP, "TEST_FIRST", {"jc2": 6000})
    out = root / "mlp"
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70", "--mlp",
                    "--out", str(out), "--jc2-parquet", str(pq_dir)]) == 0
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    cells = doc["cells"]["best70"]
    assert doc["probe"] == "mlp"
    assert set(cells) == set(LP.PAIRS) | set(LP.QCD_PAIRS) | {"top", "qg", "qg_herwig"}
    assert cells["top"]["train_size"] == 3000 and cells["top"]["auc"] > 0.8
    assert cells["qcd_bb_vs_cc"]["train_size"] == 4000


def test_the_v3_jobs_cover_the_paper_runs_and_the_untrained_trunks():
    specs = L.build_v3()
    assert set(specs) == {"job-linprobe-fit3-raunav.yaml", "job-linprobe-mlp3-raunav.yaml"}
    fit = yaml.safe_load(specs["job-linprobe-fit3-raunav.yaml"])
    mlp = yaml.safe_load(specs["job-linprobe-mlp3-raunav.yaml"])
    paper = L.BF.v2_paper_runs()
    for doc, n in ((fit, 27 + 3), (mlp, 12 + 3)):
        sp = doc["spec"]
        assert sp["completionMode"] == "Indexed" and sp["completions"] == n
        assert "backoffLimit" not in sp and sp["parallelism"] <= 6
        assert sp["podFailurePolicy"]["rules"][0]["action"] == "FailIndex"
    a = fit["spec"]["template"]["spec"]["containers"][0]["args"][0]
    models = re.search(r"MODELS=\(([^)]*)\)", a).group(1).split()
    assert models == paper + ["init-s1", "init-s2", "init-s3"]
    assert "--datasets jc2_188 jc1 top qg" in a and f"--probs-out {L.LP_ROOT}/{L.PROBS_V3}" in a
    m = mlp["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in m["resources"]["limits"] and "--mlp" in m["args"][0]
    assert all(x.split("-")[1] in ("l188", "l162", "r42q1", "r16q1", "s1", "s2", "s3")
               for x in re.search(r"MODELS=\(([^)]*)\)", m["args"][0]).group(1).split())


# ------------------------------------------------------------------ the nonlinear probe, v4 (2026-10-30)
def _old_binary_fit_mlp(P, Xtr, ytr, Xva, yva, Xte, seeds):
    """probe._fit_mlp's scores, and each seed's (best validation AUC, epochs run), as they were
    before n_classes (mtx-s2.18), for the binary check."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    S = P.MLP_SCHEDULE
    sc = StandardScaler().fit(Xtr)
    tr, va, te = (torch.tensor(sc.transform(x), dtype=torch.float32) for x in (Xtr, Xva, Xte))
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    scores, per_seed = [], []
    for sd in seeds:
        torch.manual_seed(sd)
        net = torch.nn.Sequential(torch.nn.Linear(tr.shape[1], 256), torch.nn.ReLU(),
                                  torch.nn.Dropout(0.1), torch.nn.Linear(256, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=S["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=S["lr_factor"], patience=S["lr_patience"],
            threshold=S["min_rel_gain"], threshold_mode="rel")
        lossf = torch.nn.CrossEntropyLoss()
        best_va, best_state, patience = -1.0, None, 0
        for epoch in range(S["max_epochs"]):
            net.train()
            perm = torch.randperm(tr.shape[0])
            for i in range(0, tr.shape[0], 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                lossf(net(tr[idx]), ytr_t[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = (net(va)[:, 1] - net(va)[:, 0]).numpy()
            auc = roc_auc_score(yva, v)
            sched.step(1.0 - auc)
            if 1.0 - auc < (1.0 - best_va) * (1.0 - S["min_rel_gain"]):
                best_va, best_state, patience = auc, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= S["stop_patience"]:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            scores.append((net(te)[:, 1] - net(te)[:, 0]).numpy())
        per_seed.append((float(best_va), epoch + 1))
    return np.mean(scores, axis=0), per_seed


def test_the_binary_mlp_is_bit_identical_to_the_v3_probe(monkeypatch):
    pytest.importorskip("torch")
    P = _load("probe", "experiments/EVAL/probe.py")
    monkeypatch.setitem(P.MLP_SCHEDULE, "max_epochs", 25)
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 3000)
    x = _jets(rng, y, 2, sep=0.3)
    s, meta = P.fit_mlp(x[:2000], y[:2000], x[2000:2500], y[2000:2500], x[2500:], seeds=(0, 1))
    assert s.shape == (500,) and len(meta["seeds"]) == 2
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(P.MLP_THREADS)          # fit_mlp's thread count fixes the sums' order
    try:
        old, per_seed = _old_binary_fit_mlp(P, x[:2000], y[:2000], x[2000:2500], y[2000:2500], x[2500:], (0, 1))
    finally:
        torch.set_num_threads(threads)
    np.testing.assert_array_equal(s, old)
    assert [(m["seed"], m["val_auc"], m["epochs_run"]) for m in meta["seeds"]] == \
        [(sd, va, ep) for sd, (va, ep) in zip((0, 1), per_seed)]


def test_the_mlp_fits_k_classes_with_softmax_and_averages_probabilities(monkeypatch):
    pytest.importorskip("torch")
    P = _load("probe", "experiments/EVAL/probe.py")
    monkeypatch.setitem(P.MLP_SCHEDULE, "max_epochs", 100)
    rng = np.random.default_rng(6)
    y = rng.integers(0, 5, 4000)
    x = _jets(rng, y, 5, sep=1.0)
    p, meta = P.fit_mlp(x[:2500], y[:2500], x[2500:3000], y[2500:3000], x[3000:], n_classes=5)
    assert p.shape == (1000, 5) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert P.macro_auc_ovr(y[3000:], p) > 0.9
    assert len(meta["seeds"]) == len(P.MLP_SEEDS)
    for m in meta["seeds"]:          # the plateau rule reads the validation macro AUC
        assert 0.8 < m["val_auc"] <= 1 and m["epochs_run"] <= 100 and isinstance(m["converged"], bool)
        assert min(abs(v - m["val_auc"]) for v in m["val_auc_curve"]) < 1e-9


def test_the_v4_nonlinear_probe_adds_jetclass_and_the_188_class_task(root188, monkeypatch):
    pytest.importorskip("torch")
    root, pq_dir = root188
    rng = np.random.default_rng(12)
    means = rng.normal(size=(10, 128))
    for split, n in (("train_N300", 300), ("train_N3000", 3000), ("val", 2000), ("test", 4000)):
        y = rng.integers(0, 10, n)
        _split(root / "mtx-l188-s1" / "jc1" / split / "best70", 2.0 * means[y] + rng.normal(size=(n, 128)), y)
    monkeypatch.setattr(LP, "TEST_FIRST", {"jc2": 6000})
    load = LP._load

    def short(name, rel):            # a short schedule: this checks the cells, not the optimum
        m = load(name, rel)
        if name == "probe":
            m.MLP_SCHEDULE["max_epochs"] = 100
        return m
    monkeypatch.setattr(LP, "_load", short)
    out = root / "mlp4"
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                    "--mlp", "v4", "--out", str(out), "--jc2-parquet", str(pq_dir)]) == 0
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    cells = doc["cells"]["best70"]
    assert doc["probe"] == "mlp" and doc["mlp_set"] == "v4"
    assert set(cells) == set(LP.PAIRS) | set(LP.QCD_PAIRS) | {"top", "qg", "qg_herwig", "jc1", "jc2_188"}
    for c in cells.values():
        assert isinstance(c["converged"], bool) and len(c["epochs"]) == 3
        assert c["converged"] == c["mlp"]["all_converged"]
    c = cells["jc2_188"]                   # native_metrics' fields, on the linear probe's jets
    assert c["train_size"] == 4000 and c["n_classes_in_train"] == 188 and c["n_jets_auc"] == 1500
    assert {"macro_auc_ovr", "macro_auc_resonant", "macro_auc_qcd", "macro_auc_trained"} <= set(c)
    assert c["macro_auc_ovr"] > 0.8 and len(c["per_class_auc"]) == 188
    j = cells["jc1"]                       # multiclass_metrics' fields
    assert j["train_size"] == 3000 and j["n_jets"] == 4000 and j["auc_stride"] == 4
    assert j["macro_auc_ovr"] > 0.8
    assert len(j["per_class_auc"]) == 10 and set(j["rejection_vs_qcd"]) == {str(c) for c in range(1, 10)}
    assert set(j["eps_s"]) == set(j["rejection_vs_qcd"])        # jc1_per_class, as the linear jc1 cell
    # the v3 set is unchanged in its cells
    assert LP.main(["--root", str(root), "--model", "mtx-l188-s1", "--checkpoints", "best70",
                    "--mlp", "--out", str(root / "mlp3"), "--jc2-parquet", str(pq_dir)]) == 0
    v3 = json.loads((root / "mlp3" / "mtx-l188-s1.json").read_text())
    assert v3["mlp_set"] == "v3" and set(v3["cells"]["best70"]) == set(cells) - {"jc1", "jc2_188"}


def test_the_v4_jobs_cover_every_model_and_the_init_bn_checkpoint():
    specs = L.build_v4()
    assert set(specs) == {"job-linprobe-mlp4-raunav.yaml", "job-linprobe-initbn-raunav.yaml",
                          "job-linprobe-mlp4bn-raunav.yaml"}
    paper, inits = L.BF.v2_paper_runs(), ["init-s1", "init-s2", "init-s3"]
    want = {"job-linprobe-mlp4-raunav.yaml": (paper + inits, ["best70"] * 27 + ["init"] * 3),
            "job-linprobe-initbn-raunav.yaml": (inits, ["init_bn"] * 3),
            "job-linprobe-mlp4bn-raunav.yaml": (inits, ["init_bn"] * 3)}
    for name, spec in specs.items():
        d = yaml.safe_load(spec)
        sp, a = d["spec"], _args(spec)
        models, ckpts = want[name]
        assert d["metadata"]["name"].endswith("-raunav") and name == f"job-{d['metadata']['name']}.yaml"
        assert sp["completionMode"] == "Indexed" and sp["completions"] == len(models)
        assert sp["parallelism"] == L.V4_PARALLEL[d["metadata"]["name"].split("-")[1]] <= 2
        assert sp["podFailurePolicy"]["rules"][0]["action"] == "FailIndex"
        assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0, name
        assert f'--branch "{L.V4_PIN}"' in a and L.V4_PIN == "mtx-s2.20"
        assert re.search(r"MODELS=\(([^)]*)\)", a).group(1).split() == models
        assert re.findall(r'"(\w+)"', re.search(r"CKPTS=\(([^)]*)\)", a).group(1)) == ckpts
        # no pod waits for its features: one storage check, then the work
        assert "sleep" not in a and "manifest.json" not in a and a.count("df --output=pcent /data") == 1
    for name, out in (("mlp4", "fits_v3_mlp2"), ("mlp4bn", "fits_v3_mlp2/init_bn")):
        spec = specs[f"job-linprobe-{name}-raunav.yaml"]
        res = yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "nvidia.com/gpu" not in res["limits"]
        assert f"--mlp v4 --out {L.LP_ROOT}/{out} || halt" in _args(spec)
    mlp = _args(specs["job-linprobe-mlp4-raunav.yaml"]) + _args(specs["job-linprobe-mlp4bn-raunav.yaml"])
    assert "only after the init_bn extraction jobs init-bn-s1..s3 have completed" in L.__doc__
    lin = specs["job-linprobe-initbn-raunav.yaml"]
    assert yaml.safe_load(lin)["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert ("--datasets jc2_188 jc1 top qg" in _args(lin)
            and f"--out {L.LP_ROOT}/fits_v3_initbn --probs-out {L.LP_ROOT}/fits_v3_initbn_probs" in _args(lin))
    for path, text in L.V4_NEEDED.items():       # what the tag must carry
        assert text in (ROOT / path).read_text(), path
    # new directories only: none of the earlier fits' outputs
    for old in (L.FITS, L.FITS_V3, L.PROBS_V3, L.MLP_V3):
        assert f"{L.LP_ROOT}/{old} " not in mlp + _args(lin) and f"{L.LP_ROOT}/{old}\n" not in mlp + _args(lin)

