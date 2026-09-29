"""experiments/EVAL/extract_v2.py: row selection, checkpoint resolution, the
output-layer scores (anomaly.py's own class sums), and one pass serving several
checkpoints on identical rows -- without weaver's data files."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
an = _load("anomaly", "experiments/EVAL/anomaly.py")


def test_probe_classes_cover_every_probe_task():
    probe = _load("probe", "experiments/EVAL/probe.py")
    got = set(xv.probe_classes())
    for spec in probe.TASKS.values():
        assert set(spec["signal"]) | set(spec["background"]) <= got
    assert {0, 1, 15, 169, 181, 10, 11, 18, 34, 158, 4} <= got


def test_selector_keeps_feature_classes_everywhere_and_head_rows_inside_the_prefix():
    s = xv.Selector([0, 1], prefix_features=10, head_classes=[170], head_prefix=100, diag_stride=7)
    rows = np.arange(95, 110)
    lab = np.array([0, 5, 170, 1, 170, 5, 5, 5, 5, 0, 170, 5, 5, 5, 5])
    fm, hm = s.feature_mask(rows, lab), s.head_mask(rows, lab)
    assert fm.tolist() == np.isin(lab, [0, 1]).tolist()
    assert hm.tolist() == [((r < 100) and (l == 170 or r % 7 == 0)) for r, l in zip(rows, lab)]
    assert s.feature_mask(np.array([3]), np.array([99]))[0]            # inside the prefix


def _write_v2_run(d: pathlib.Path, vals: dict, best: int):
    (d / "metrics").mkdir(parents=True)
    for e, v in vals.items():
        (d / "metrics" / f"epoch-{e:03d}.json").write_text(json.dumps(
            {"epoch": e, "selection": {"metric": "val.acc", "value": v}}))
        (d / f"net_epoch-{e}_state.pt").write_text("x")
    (d / "best_epoch.json").write_text(json.dumps({"epoch": best, "metric": "val.acc"}))


def test_best_checkpoint_is_the_fixed_sample_argmax_ties_to_the_earlier(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5, 71: 0.7, 72: 0.7, 73: 0.6}, best=71)
    got = xv.resolve_checkpoints(tmp_path, ["best", "72-73"])
    assert [t for t, _ in got] == ["best", "e072", "e073"]
    assert got[0][1].name == "net_epoch-71_state.pt"


def test_a_stale_best_record_is_refused(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5, 71: 0.7}, best=70)
    with pytest.raises(SystemExit, match="per-epoch records say 71"):
        xv.resolve_checkpoints(tmp_path, ["best"])


def test_missing_checkpoint_is_refused(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5}, best=70)
    with pytest.raises(SystemExit, match="missing"):
        xv.resolve_checkpoints(tmp_path, ["70-71"])


@pytest.mark.parametrize("rung", ["L188", "R42_Q1", "R16_Q1"])
def test_head_scores_are_anomaly_class_sums_and_consistent_probabilities(rung):
    k = len({r[rung] for r in an.read_map()})
    rng = np.random.default_rng(0)
    z = (rng.normal(size=(300, k)) * 4).astype(np.float32)
    _, signals = xv.anomaly_classes()
    h = xv.head_score_columns(z, rung, signals)
    node_of, res, qcd = an.node_roles(rung)
    for sig, lab in signals.items():
        assert np.array_equal(h[f"class_sum|{sig}"],
                              an.class_sum_without(z, rung, {node_of[lab]}).astype(np.float32))
        assert np.array_equal(h[f"class_sum_matched|{sig}"],
                              an.class_sum_without(z, rung, an.matched_nodes(rung, lab))
                              .astype(np.float32))
    p = np.exp(z.astype(np.float64))
    p /= p.sum(1, keepdims=True)
    assert np.allclose(h["p_qcd"], p[:, sorted(qcd)].sum(1), rtol=1e-5)
    lo = np.log(p[:, sorted(res)].sum(1) / p[:, sorted(qcd)].sum(1))
    assert np.allclose(h["logodds_res_qcd"], lo, rtol=1e-4, atol=1e-4)
    assert np.array_equal(h["argmax"], z.argmax(1))


def test_one_pass_serves_every_checkpoint_on_the_same_rows(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self, k, seed):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.trunk = torch.nn.Linear(3, 128)
            self.fc = torch.nn.Linear(128, k)
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(torch.randn(p.shape, generator=g))

        def forward(self, x):
            return self.fc(self.trunk(x))

    class Tap:
        def __init__(self, m):
            self.buf = None
            self.h = m.fc.register_forward_pre_hook(lambda _m, i: setattr(self, "buf", i[0]))

        def close(self):
            self.h.remove()

    k, rung = 17, "R16_Q1"
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 188, 1000)
    X = rng.normal(size=(1000, 3)).astype(np.float32)
    batches = [({"x": torch.from_numpy(X[i:i + 128])}, labels[i:i + 128]) for i in range(0, 1000, 128)]
    models = {"e078": Toy(k, 1).eval(), "e079": Toy(k, 2).eval()}
    sel = xv.Selector([0, 1], prefix_features=0, head_classes=[170, 171], head_prefix=600,
                      diag_stride=50)
    _, signals = xv.anomaly_classes()
    to_in = lambda Xb, need: [Xb["x"][torch.from_numpy(np.flatnonzero(need))]]
    res = xv.run(iter(batches), models, sel, rung, k, signals, Tap, to_in)
    only = xv.run(iter(batches), models, sel, rung, k, signals, Tap, to_in, features_at={"e079"})
    assert only["checkpoints"]["e078"]["rows"].size == 0
    assert np.array_equal(only["checkpoints"]["e078"]["head_rows"],
                          res["checkpoints"]["e078"]["head_rows"])
    assert np.array_equal(only["checkpoints"]["e079"]["features"],
                          res["checkpoints"]["e079"]["features"])
    assert res["n_stream"] == 1000 and np.array_equal(res["labels"], labels.astype(np.int16))
    a, b = res["checkpoints"]["e078"], res["checkpoints"]["e079"]
    want_f = np.flatnonzero(np.isin(labels, [0, 1]))
    want_h = np.flatnonzero((np.arange(1000) < 600)
                            & (np.isin(labels, [170, 171]) | (np.arange(1000) % 50 == 0)))
    for c in (a, b):
        assert np.array_equal(c["rows"], want_f) and np.array_equal(c["head_rows"], want_h)
        assert c["features"].dtype == np.float16 and c["features"].shape == (want_f.size, 128)
    with torch.no_grad():
        zb = models["e079"](torch.from_numpy(X[want_h])).numpy()
    assert np.array_equal(b["head"]["argmax"], zb.argmax(1))
    assert not np.array_equal(a["head"]["argmax"], b["head"]["argmax"])
    sizes = xv.write(tmp_path, res, {"checkpoints": {"e078": {"checkpoint_sha256": "a"},
                                                     "e079": {"checkpoint_sha256": "b"}}})
    man = json.loads((tmp_path / "e079" / "manifest.json").read_text())
    assert man["checkpoint_sha256"] == "b" and man["n_head_rows"] == want_h.size
    assert set(sizes) == {"e078", "e079"}


def test_v1err_head_specs_cover_every_model_with_the_retry_policy():
    import yaml
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    jobs = bx.build_v1err()
    diag = {k for k in jobs if k.startswith("job-heads-diag-")}
    anom = {k for k in jobs if k.startswith("job-heads-anomaly-")}
    assert len(diag) == 30 and len(anom) == 20 and "job-test-class-counts-raunav.yaml" in jobs
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"] and fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
        assert d["spec"]["template"]["spec"]["containers"][0]["name"] == "main"
        assert f'--branch "{bx.V1ERR_PIN}"' in text and "|| halt" in text
        assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"
        if "heads-" in fname:
            run = "mtx-" + fname.split("-v1err-")[1].removesuffix("-raunav.yaml")
            assert f"--align-with /data/results/eval/{run}/features_e79" in text
            assert "--checkpoints 70-79" in text and "--max-jets 2000000" in text
            assert ("--num-reg 1" in text) == ("mass" in run)
            gpu = "nvidia.com/gpu" in text
            assert gpu == fname.startswith("job-heads-anomaly-")
            assert ("--no-anomaly-rows" in text) == (not gpu)


def test_v2_specs_one_per_classification_run_primary_features_only():
    import yaml
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    jobs = bx.build_v2()
    assert len(jobs) == len(bx.v2_runs()) and not any("mpm" in k for k in jobs)
    assert bx.v2_rung("L162_MASS") == "L162" and bx.v2_rung("R16_Q1_MASS_LM") == "R16_Q1"
    assert bx.v2_rung("R42_Q1_LOFO4P") == "R42_Q1" and bx.v2_rung("RAND2_p1") == "none"
    for fname, text in jobs.items():
        yaml.safe_load(text)
        assert "--checkpoints best 70-79 --features-at best" in text
        assert "--feature-classes probe --prefix-features 2000000" in text
        assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"


def test_head_scores_outside_the_tree_keep_only_what_is_defined():
    z = np.random.default_rng(0).normal(size=(10, 17)).astype(np.float32)
    h = xv.head_score_columns(z, "none", {})
    assert set(h) == {"argmax", "logsumexp"}
