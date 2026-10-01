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
    (tmp_path / xv.WAVG_FILE).write_text("x")
    got = xv.resolve_checkpoints(tmp_path, ["bestval", "wavg", "72-73"])
    assert [t for t, _ in got] == ["bestval", "wavg", "e072", "e073"]
    assert got[1][1].name == "net_wavg70-79_state.pt"
    assert got[0][1].name == "net_epoch-71_state.pt"


def test_a_stale_best_record_is_refused(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5, 71: 0.7}, best=70)
    with pytest.raises(SystemExit, match="per-epoch records say 71"):
        xv.resolve_checkpoints(tmp_path, ["bestval"])


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
    diag = {k for k in jobs if k.startswith("job-heads-diag-") and "diag-mass" not in k}
    anom = {k for k in jobs if k.startswith("job-heads-anomaly-")}
    assert len(diag) == 20 and len(anom) == 20 and not any("mass" in k for k in diag | anom)
    batch = jobs["job-heads-diag-mass-v1err-raunav.yaml"]
    assert batch.count("run_heads mtx-") == 10 and "mass" in batch
    for fname, text in jobs.items():
        d = yaml.safe_load(text)
        assert "raunav" in d["metadata"]["name"] and fname == f"job-{d['metadata']['name']}.yaml"
        assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
        assert d["spec"]["template"]["spec"]["containers"][0]["name"] == "main"
        pin = bx.V1ERR_PIN if fname[4:-5] in bx.HEADS_APPLIED_AT_V1ERR_PIN else bx.HEADS_PIN
        assert f'--branch "{pin}"' in text and "|| halt" in text
        assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"
        if "heads-" in fname and "diag-mass" not in fname:
            run = "mtx-" + fname.split("-v1err-")[1].removesuffix("-raunav.yaml")
            assert f"--align-with /data/results/eval/{run}/features_e79" in text
            assert "--checkpoints 70-79" in text and "--max-jets 2000000" in text
            assert ("--num-reg 1" in text) == ("mass" in run)
            gpu = "nvidia.com/gpu" in text
            assert gpu == fname.startswith("job-heads-anomaly-")
            assert ("--no-anomaly-rows" in text) == (not gpu)
            # a GPU pod that sees no device must not fall back to the CPU
            check = "gpu_ok ()" if fname[4:-5] in bx.HEADS_RECREATED_AFTER_GPU_FAULT \
                else "torch.cuda.is_available()"
            assert (check in text and "exit 137" in text) == gpu
            assert all((n in text) == gpu for n in bx.BAD_GPU_NODES)
    # the re-created job carries the GPU-fault rule; every job that ran keeps its text
    for name in bx.HEADS_RECREATED_AFTER_GPU_FAULT:
        text = jobs[f"job-{name}.yaml"]
        assert bx.HALT_GPU in text and "tee -a ${LOG} || halt" in text
        assert all(n in text for n in bx.GPU_FAULT_NODES)
    assert all(bx.HALT in t for f, t in jobs.items()
               if f[4:-5] not in bx.HEADS_RECREATED_AFTER_GPU_FAULT)


def _halt_exit(halt: str, gpu_works: bool, cmd: str, tmp_path) -> int:
    """The exit status of a spec's error path: `cmd | tee -a LOG || halt` under the
    spec's shell options, with gpu_ok stubbed."""
    import subprocess
    script = (f"set -euo pipefail\n{halt}\ngpu_ok () {{ return {0 if gpu_works else 1}; }}\n"
              f"LOG={tmp_path}/a.log\n{cmd} 2>&1 | tee -a ${{LOG}} || halt\necho ok\n")
    return subprocess.run(["bash", "-c", script], capture_output=True).returncode


def test_a_failure_on_a_dead_gpu_is_retried_and_a_failure_on_a_working_one_halts(tmp_path):
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    # heads-anomaly-v1err-r16q1-s1, 2026-09-30: a CUDA error is exit 1 from Python
    assert _halt_exit(bx.HALT_GPU, False, "(exit 1)", tmp_path) == 137     # retried elsewhere
    assert _halt_exit(bx.HALT_GPU, True, "(exit 1)", tmp_path) == 42       # a code failure halts
    assert _halt_exit(bx.HALT_GPU, True, "(exit 137)", tmp_path) == 137    # a signal is retried
    assert _halt_exit(bx.HALT_GPU, True, "true", tmp_path) == 0
    assert _halt_exit(bx.HALT, False, "(exit 1)", tmp_path) == 42          # the old rule did not ask
    # the log keeps the failing command's output: tee does not mask its status
    _halt_exit(bx.HALT_GPU, True, "(echo traceback; exit 1)", tmp_path)
    assert "traceback" in (tmp_path / "a.log").read_text()


def test_the_gpu_fault_rule_refuses_a_template_it_does_not_match():
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    with pytest.raises(SystemExit):
        bx.gpu_fault_aware("no halt here")


def test_v2_specs_one_per_classification_run_primary_features_only():
    import yaml
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    jobs = bx.build_v2()
    assert len(jobs) == len(bx.v2_runs()) and not any("mpm" in k for k in jobs)
    assert bx.v2_rung("L162_MASS") == "L162" and bx.v2_rung("R16_Q1_MASS_LM") == "R16_Q1"
    assert bx.v2_rung("R42_Q1_LOFO4P") == "R42_Q1" and bx.v2_rung("RAND2_p1") == "none"
    for fname, text in jobs.items():
        yaml.safe_load(text)
        assert "--checkpoints bestval wavg --features-at bestval wavg" in text
        assert "--feature-classes probe --prefix-features 2000000" in text
        assert bx.HALT_GPU in text and "gpu_ok ()" in text and "tee -a ${LOG} || halt" in text
        assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"


def test_head_scores_outside_the_tree_keep_only_what_is_defined():
    z = np.random.default_rng(0).normal(size=(10, 17)).astype(np.float32)
    h = xv.head_score_columns(z, "none", {})
    assert set(h) == {"argmax", "logsumexp"}


def test_storage_estimate_counts_each_row_once():
    cc = _load("class_counts", "experiments/EVAL/class_counts.py")
    sel = np.zeros(188, int)
    sel[[0, 1]] = [1000, 2000]
    prefix = np.zeros(188, int)
    prefix[[0, 5, 170]] = [100, 50, 300]
    s = cc.storage({"selected_per_class": sel.tolist()}, prefix, [0, 1], [170], n_models=2,
                   n_ckpt_features=1, n_ckpt_heads=11, diag=10)
    assert s["feature_rows_per_checkpoint"] == 3000 + 50 + 300     # class 0 counted once
    assert s["head_rows_per_checkpoint"] == 310
    assert s["bytes_total"] == 2 * (3350 * cc.FEATURE_ROW_BYTES + 11 * 310 * cc.HEAD_ROW_BYTES)


def test_sizing_reads_the_measured_counts_and_the_largest_v1_rejection(tmp_path):
    sz = _load("extraction_v2_sizing", "experiments/EVAL/extraction_v2_sizing.py")
    sel = np.full(188, 400_000)
    (tmp_path / "c.json").write_text(json.dumps({"selected_per_class": sel.tolist(),
                                                 "in_window_per_class": (sel // 50).tolist()}))
    np.save(tmp_path / "L.npy", np.arange(2000) % 188)
    probes = sorted((ROOT / "experiments/FIGS/data/probe_ladder_v2_mlp2").glob("s*.json"))
    out = tmp_path / "o.json"
    assert sz.main(["--counts", str(tmp_path / "c.json"), "--probe-files", *map(str, probes),
                    "--prefix-labels", str(tmp_path / "L.npy"), "--free-bytes", "2.3e11",
                    "--size-bytes", "1e12", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    t = r["tasks_by_test_fraction"]["0.6"]["bvc_resonant"]
    assert t["n_background_split"] == 400_000 and t["n_background_test"] == 240_000
    assert t["max_v1_rejection_at_90"] == pytest.approx(11876 / 6)     # 188-class run 3: 6 pass
    assert t["meets_min_pass"] is (240_000 / (11876 / 6) >= 100)
    assert len(r["storage"]) == 2 * len(sz.PLANS)
    anywhere, windowed = xv.probe_feature_rules()
    s = next(iter(r["storage"].values()))
    # windowed-only classes enter with their in-window counts, not the whole split
    want = (400_000 * len(anywhere) + 8_000 * sum(len(c) for c, _ in windowed)
            + 2000 - np.isin(np.arange(2000) % 188, anywhere).sum())
    assert s["feature_rows_per_checkpoint"] == want


def test_classes_only_a_windowed_task_reads_are_kept_inside_its_window():
    anywhere, windowed = xv.probe_feature_rules()
    qcd, _ = xv.anomaly_classes()
    assert 169 in anywhere and 181 in anywhere            # b vs c in QCD is unwindowed
    assert not (set(qcd) - {169, 181}) & set(anywhere)
    (cls, win), = windowed
    assert {4, 5, 6, 70} <= set(cls) and set(win) == {"jet_pt", "jet_sdmass", "jet_eta"}
    s = xv.Selector(anywhere, 0, [], 0, 0, windowed)
    lab = np.array([4, 4, 170, 0, 170])
    obs = {"jet_pt": np.array([500., 300., 500., 300., 700.]),
           "jet_sdmass": np.array([100., 100., 120., 50., 100.]),
           "jet_eta": np.zeros(5)}
    assert s.feature_mask(np.arange(5), lab, obs).tolist() == [True, False, True, True, False]
    with pytest.raises(SystemExit, match="observers"):
        s.feature_mask(np.arange(5), lab, None)
