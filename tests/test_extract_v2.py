"""experiments/EVAL/extract_v2.py: row selection, checkpoint resolution, the
output-layer scores (anomaly.py's own class sums), and one pass serving several
checkpoints on identical rows -- without weaver's data files."""
import importlib.util
import json
import pathlib
import sys

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


def _write_80(d: pathlib.Path, vals: dict):
    """A finished v2 run: every epoch's record, the global and the window best as the
    driver writes them, and the state files of the epochs given."""
    pv = _load("pretrain_v2_for_xv", "experiments/MTX/pretrain_v2.py")
    full = {e: vals.get(e, 0.1) for e in range(80)}
    top = max(full.values())
    _write_v2_run(d, full, best=min(e for e, v in full.items() if v == top))
    pv.write_window_best(d, 80, {"driver": "test"})


def test_best70_is_the_primary_and_the_global_best_its_sensitivity_check(tmp_path):
    _write_80(tmp_path, {40: 0.9, 73: 0.8, 76: 0.8})
    (tmp_path / xv.INIT_FILE).write_text("x")
    got = dict(xv.resolve_checkpoints(tmp_path, ["best70", "bestval", "init"]))
    assert got["best70"].name == "net_epoch-73_state.pt"       # the first maximum within 70-79
    assert got["bestval"].name == "net_epoch-40_state.pt"
    assert got["init"].name == "init_trunk.pt"
    assert xv.aliases(list(got.items())) == {"best70": "best70", "bestval": "bestval", "init": "init"}
    (tmp_path / "best_window_epoch.json").write_text(json.dumps(
        {**json.loads((tmp_path / "best_window_epoch.json").read_text()), "epoch": 76}))
    with pytest.raises(SystemExit, match="says epoch 76, the per-epoch records say 73"):
        xv.resolve_checkpoints(tmp_path, ["best70"])


def test_a_checkpoint_two_rules_select_is_extracted_once_and_both_tags_point_at_it(tmp_path):
    _write_80(tmp_path / "run", {74: 0.9})
    ckpts = xv.resolve_checkpoints(tmp_path / "run", ["best70", "bestval"])
    assert ckpts[0][1] == ckpts[1][1]
    alias = xv.aliases(ckpts)
    assert alias == {"best70": "best70", "bestval": "best70"}
    c = {"features": np.zeros((2, 128), np.float16), "pooled": np.ones((2, 128), np.float16),
         "rows": np.arange(2), "label188": np.zeros(2, np.int16), "has_head": False,
         "head_rows": np.zeros(0, np.int64), "head_label188": np.zeros(0, np.int16), "head": {}}
    res = {"n_stream": 2, "label188_sha256": "s", "observers": {}, "checkpoints": {"best70": c}}
    out = tmp_path / "out"
    xv.write(out, res, {"checkpoints": {"best70": {"checkpoint_sha256": "a"}}}, alias)
    assert (out / "bestval").is_symlink() and (out / "bestval").resolve() == (out / "best70").resolve()
    man = json.loads((out / "bestval" / "manifest.json").read_text())
    assert man["tag"] == "best70" and man["tags"] == ["best70", "bestval"]
    assert np.array_equal(np.load(out / "bestval" / "pooled.npy"), c["pooled"])
    assert not (out / "best70" / "head_scores.npz").exists() and man["has_head"] is False
    xv.write(out, res, {"checkpoints": {"best70": {"checkpoint_sha256": "a"}}}, alias)   # a rerun
    assert (out / "bestval").is_symlink()
    (out / "bestval").unlink()
    (out / "bestval").mkdir()
    with pytest.raises(SystemExit, match="holds an extraction of its own"):
        xv.write(out, res, {"checkpoints": {"best70": {"checkpoint_sha256": "a"}}}, alias)


def _write_twin(d: pathlib.Path, e: int, tags: list):
    """A BatchNorm twin as experiments/MTX/bn_twins_v2.py writes it."""
    import hashlib
    src, state = d / f"net_epoch-{e}_state.pt", d / f"net_epoch-{e}_bn_state.pt"
    state.write_text(f"bn{e}")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (d / f"net_epoch-{e}_bn.json").write_text(json.dumps(
        {"epoch": e, "tags": tags, "inputs": {str(e): sha(src)}, "sha256": sha(state)}))


def test_the_batchnorm_twins_resolve_to_the_twin_of_each_rules_epoch(tmp_path):
    _write_80(tmp_path, {40: 0.9, 73: 0.8, 76: 0.8})
    _write_twin(tmp_path, 73, ["best70"])
    _write_twin(tmp_path, 40, ["bestval"])
    got = dict(xv.resolve_checkpoints(tmp_path, ["best70", "bestval", "best70_bn", "bestval_bn"]))
    assert got["best70_bn"].name == "net_epoch-73_bn_state.pt"
    assert got["bestval_bn"].name == "net_epoch-40_bn_state.pt"
    assert xv.aliases(list(got.items())) == {t: t for t in got}


def test_one_epoch_both_rules_select_has_one_twin_both_tags_point_at(tmp_path):
    _write_80(tmp_path, {74: 0.9})
    _write_twin(tmp_path, 74, ["best70", "bestval"])
    ckpts = xv.resolve_checkpoints(tmp_path, ["best70_bn", "bestval_bn"])
    assert xv.aliases(ckpts) == {"best70_bn": "best70_bn", "bestval_bn": "best70_bn"}


def test_a_missing_or_mismatched_twin_is_refused(tmp_path):
    _write_80(tmp_path, {74: 0.9})
    with pytest.raises(SystemExit, match="no BatchNorm twin of epoch 74: run experiments/MTX/bn_twins_v2.py"):
        xv.resolve_checkpoints(tmp_path, ["best70_bn"])
    _write_twin(tmp_path, 74, ["best70", "bestval"])
    (tmp_path / "net_epoch-74_state.pt").write_text("another state")
    with pytest.raises(SystemExit, match="does not record net_epoch-74_bn_state.pt as the twin"):
        xv.resolve_checkpoints(tmp_path, ["best70_bn"])


def test_a_stale_best_record_is_refused(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5, 71: 0.7}, best=70)
    with pytest.raises(SystemExit, match="per-epoch records say 71"):
        xv.resolve_checkpoints(tmp_path, ["bestval"])


def test_missing_checkpoint_is_refused(tmp_path):
    _write_v2_run(tmp_path, {70: 0.5}, best=70)
    with pytest.raises(SystemExit, match="missing"):
        xv.resolve_checkpoints(tmp_path, ["70-71"])


# ------------------------------------------------- the pooled readout (amendment A14)

def test_the_pooled_mean_skips_padded_particles_in_either_layout():
    import torch
    g = torch.Generator().manual_seed(0)
    n, p, c = 4, 4, 128                       # N == P: the layout cannot be read off x alone
    real = torch.tensor([3, 4, 1, 2])
    pad = torch.arange(p)[None, :] >= real[:, None]           # (N, P), True on padded
    x = torch.randn(n, p, c, generator=g)
    want = torch.stack([x[i, :real[i]].mean(0) for i in range(n)])
    garbage = x.masked_fill(pad.unsqueeze(-1), 1e6)            # what the blocks leave there
    got = xv.pooled_mean(garbage, pad, torch.zeros(n, 1, c))  # batch-first
    torch.testing.assert_close(got, want)
    seq = xv.pooled_mean(garbage.transpose(0, 1), pad, torch.zeros(1, n, c))   # weaver 0.4.17
    torch.testing.assert_close(seq, want)
    one = xv.pooled_mean(garbage[1:2].transpose(0, 1), pad[1:2], torch.zeros(1, 1, c))
    torch.testing.assert_close(one, want[1:2])
    with pytest.raises(SystemExit, match="neither layout"):
        xv.pooled_mean(garbage[:, :3], pad, torch.zeros(n, 1, c))


def _part(k=5):
    """The extraction's model (experiments/EVAL/extract_features.build_model's arch and
    head) on a seven-feature toy input, in evaluation mode."""
    pytest.importorskip("weaver")
    import types
    import torch
    arch = _load("arch_for_pooled", "experiments/MTX/ParT_sophon_arch_mtx.py")
    dc = types.SimpleNamespace(input_dicts={"pf_features": list(range(7))}, input_names=[
        "pf_points", "pf_features", "pf_vectors", "pf_mask"], input_shapes={})
    torch.manual_seed(0)
    return arch.get_model(dc, num_classes=k, fc_params=[(512, 0.1)])[0].eval()


def _jets(n_real, p=16, seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    n = len(n_real)
    mask = (torch.arange(p)[None, :] < torch.tensor(n_real)[:, None]).float()[:, None, :]
    p3 = torch.randn(n, 3, p, generator=g)
    vec = torch.cat([p3, p3.pow(2).sum(1, keepdim=True).sqrt() + 1], 1)
    return [torch.randn(n, 2, p, generator=g) * mask, torch.randn(n, 7, p, generator=g) * mask,
            vec * mask, mask]


def test_padded_particles_do_not_change_the_pooled_embedding_of_the_real_network():
    import torch
    model = _part()
    tap, inner = xv.PooledTap(model), []
    h = model.mod.cls_blocks[0].register_forward_pre_hook(lambda _m, a: inner.append(a[0]))
    n_real = [3, 16, 1, 8, 5, 12]
    pts, feat, vec, mask = _jets(n_real)
    with torch.no_grad():
        for _ in range(6):                    # past the trimmer's warm-up
            model(pts, feat, vec, mask)
        model(pts, feat, vec, mask)
    a = tap.buf.clone()
    assert a.shape == (6, 128) and torch.isfinite(a).all()
    # other values in every padded slot, features and four-vectors alike: nothing moves
    junk = _jets([16] * 6, seed=2)
    pad = 1 - mask
    with torch.no_grad():
        model(pts + junk[0] * pad, feat + junk[1] * pad, vec + junk[2] * pad, mask)
    assert torch.equal(tap.buf, a)
    # each jet alone (trimmed to its own length) gives its pooled embedding
    for i in range(6):
        with torch.no_grad():
            model(pts[i:i + 1], feat[i:i + 1], vec[i:i + 1], mask[i:i + 1])
        torch.testing.assert_close(tap.buf[0], a[i], atol=1e-4, rtol=1e-4)
    # not vacuous: the blocks write the padded slots, so a plain mean would differ
    x = inner[6]
    x = x if x.shape[0] == 6 else x.transpose(0, 1)
    assert not torch.allclose(x.float().mean(1), a, atol=1e-3)
    h.remove()
    tap.close()


def test_a_trunk_without_an_output_layer_loads_whole_or_not_at_all(tmp_path):
    import torch
    src = _part(k=17)
    trunk = {k[len("mod."):]: v for k, v in src.state_dict().items()
             if k.startswith("mod.") and not k.startswith("mod.fc.")}
    torch.save({"trunk": trunk}, tmp_path / "init_trunk.pt")         # pretrain_v2.trunk_state
    torch.save({**{f"trunk.mod.{k}": v for k, v in trunk.items()},     # an MPMNet state
                "decoder.w": torch.zeros(3)}, tmp_path / "mpm.pt")
    for f, kind in (("init_trunk.pt", "init_trunk"), ("mpm.pt", "mpm")):
        m = _part(k=1)
        with torch.no_grad():
            for p_ in m.parameters():
                p_.zero_()
        prov = xv.load_headless(m, tmp_path / f)
        assert prov["format"] == kind and prov["trunk_tensors_loaded"] == len(trunk)
        got = m.state_dict()
        assert all(torch.equal(got[f"mod.{k}"], v) for k, v in trunk.items())
    torch.save({"trunk": {**trunk, "extra.w": torch.zeros(1)}}, tmp_path / "bad.pt")
    with pytest.raises(SystemExit, match="did not load cleanly"):
        xv.load_headless(_part(k=1), tmp_path / "bad.pt")
    torch.save({"trunk": {k: v for k, v in trunk.items() if not k.startswith("blocks.7.")}},
               tmp_path / "short.pt")
    with pytest.raises(SystemExit, match="did not load cleanly"):
        xv.load_headless(_part(k=1), tmp_path / "short.pt")
    torch.save(src.state_dict(), tmp_path / "classifier.pt")
    with pytest.raises(SystemExit, match="needs its --num-classes"):
        xv.load_headless(_part(k=1), tmp_path / "classifier.pt")


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
    assert set(sizes) == {"e078", "e079"} and man["pooled"] is None
    assert not (tmp_path / "e079" / "pooled.npy").exists()

    # the pooled readout beside the features, and a checkpoint without an output layer
    class Pooled:
        def __init__(self, m):
            self.buf = None
            self.h = m.trunk.register_forward_hook(lambda _m, _i, o: setattr(self, "buf", -o))

        def close(self):
            self.h.remove()

    both = xv.run(iter(batches), models, sel, rung, k, signals, Tap, to_in, pooled_factory=Pooled,
                  heads_at={"e079"})
    a, b = both["checkpoints"]["e078"], both["checkpoints"]["e079"]
    assert a["has_head"] is False and a["head_rows"].size == 0 and a["head"] == {}
    assert np.array_equal(b["head_rows"], want_h) and np.array_equal(b["head"]["argmax"], zb.argmax(1))
    for c, t in ((a, "e078"), (b, "e079")):
        assert np.array_equal(c["rows"], want_f) and c["pooled"].dtype == np.float16
        assert np.array_equal(c["pooled"], -res["checkpoints"][t]["features"])
    xv.write(tmp_path / "p", both, {"checkpoints": {"e078": {"checkpoint_sha256": "a"},
                                                    "e079": {"checkpoint_sha256": "b"}}})
    assert np.array_equal(np.load(tmp_path / "p" / "e078" / "pooled.npy"), a["pooled"])
    assert not (tmp_path / "p" / "e078" / "head_scores.npz").exists()
    assert (tmp_path / "p" / "e079" / "head_scores.npz").exists()


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
        pin = (bx.V1ERR_PIN if fname[4:-5] in bx.HEADS_APPLIED_AT_V1ERR_PIN
               else bx.HEADS_DIAG_PIN if fname.startswith("job-heads-diag-v1err-")
               and "diag-mass" not in fname else bx.HEADS_PIN)
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


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
def test_v2_specs_one_per_run_and_init_reference_and_only_committed_when_the_plan_fits():
    import yaml
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    jobs = bx.build_v2()
    grid = json.loads(bx.V2_GRID.read_text())["arms"]
    assert len(bx.v2_runs()) == sum(a["runs"] for a in grid)       # the self-supervised runs too
    assert len(jobs) == len(bx.v2_runs()) + 5
    assert bx.v2_rung("L162_MASS") == "L162" and bx.v2_rung("R16_Q1_MASS_LM") == "R16_Q1"
    assert bx.v2_rung("R42_Q1_LOFO4P") == "R42_Q1" and bx.v2_rung("RAND2_p1") == "none"
    assert bx.v2_rung("MPM") == bx.v2_rung("MPM_LOFO4P") == "none"
    twins, bn_why = bx.v2_bn_twins_needed()
    # committed: the specs of every tier whose plan fits with the fine-tuning (A14: "sized and
    # emitted tier by tier"), once A14's BatchNorm rule has fired
    tiers = sorted({int(a["tier"]) for a in grid})
    fitting = [t for t in tiers if bx.v2_plan_fits(tier=t)[0]] if twins is True else []
    why = {t: bx.v2_plan_fits(tier=t)[1] for t in tiers}
    expected = sorted(f for t in fitting for f in bx.build_v2(t))
    committed = sorted(p.name for p in bx.OUT_DIR.glob("job-extract-v2-*.yaml"))
    for run, arm, k, reg, s in bx.v2_runs():
        text = jobs[f"job-extract-v2-{run.removeprefix('mtx-')}-raunav.yaml"]
        # amendment A14: the primary, the robustness check and the sensitivity check, and the
        # BatchNorm twins of the primary and the sensitivity check (A14's BatchNorm rule fired),
        # all with features (and the pooled readout); --num-classes 0 = no output layer
        assert (f"--run-dir {bx.V2_ROOT}/{run} --rung {bx.v2_rung(arm)} --num-classes {k} "
                f"--num-reg {reg}") in text and f"OUT={bx.V2_OUT}/{run}\n" in text
        assert ("--checkpoints best70 bestval wavg best70_bn bestval_bn \\\n" in text
                and "--features-at" not in text)
        assert (k == 0) == (arm in ("MPM", "MPM_LOFO4P"))
    for s in range(1, 6):
        text = jobs[f"job-extract-v2-init-s{s}-raunav.yaml"]
        assert (f"--run-dir {bx.V2_ROOT}/mtx-l188-s{s} --rung none --num-classes 0 --num-reg 0"
                in text and "--checkpoints init \\\n" in text and f"OUT={bx.V2_OUT}/init-s{s}\n" in text)
    for fname, text in jobs.items():
        yaml.safe_load(text)
        assert "--feature-classes probe --prefix-features 2000000" in text
        assert bx.HALT_GPU in text and "gpu_ok ()" in text and "tee -a ${LOG} || halt" in text
        assert f'--branch "{bx.V2_PIN}"' in text and bx.V2_PIN == "mtx-s2.00"
        assert '[ "${USED}" -lt 90 ]' in text and '-lt 85 ]' not in text      # the PI's line
        for node in ("ry-gpu-04.sdsc.optiputer.net", "ry-gpu-09.sdsc.optiputer.net", "nrp-01.laccd.edu",
                     "patternlab.calit2.optiputer.net"):
            assert f'"{node}"' in text
        if fname in expected:
            assert (bx.OUT_DIR / fname).read_text() == text, f"{fname} not committed as built"
    assert committed == expected, (why, bn_why)
    assert fitting == [1]       # the PI's 90 % line (2026-10-07) fits tier 1 only, so far


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
def test_v2_extraction_is_emitted_and_sized_tier_by_tier(tmp_path, monkeypatch):
    # amendment A14: the extraction is "sized and emitted tier by tier"
    import json
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    grid = json.loads(bx.V2_GRID.read_text())["arms"]
    tiers = sorted({int(a["tier"]) for a in grid})
    assert sum(len(bx.v2_runs(t)) for t in tiers) == len(bx.v2_runs())
    every = bx.build_v2()
    for t in tiers:
        jobs = bx.build_v2(t)
        assert set(jobs) <= set(every) and all(jobs[f] == every[f] for f in jobs)
        inits = [f for f in jobs if "-init-s" in f]
        assert len(inits) == (5 if t == 1 else 0)     # the init references go with tier 1
        assert len(jobs) == len(bx.v2_runs(t)) + len(inits)
    assert sum(len(bx.build_v2(t)) for t in tiers) == len(every)
    assert bx.v2_plan_key(1) == bx.V2_PLAN.replace("every run", "tier-1 runs")
    s = json.loads(bx.V2_SIZING.read_text())
    assert s["storage"][bx.v2_plan_key(1)]["n_models"] == len(bx.build_v2(1))
    fits, why = bx.v2_plan_fits(tier=2)
    assert fits is False and "does not size the plan" in why     # no tier-2 sizing committed
    e = s["storage"][bx.v2_plan_key(1)]
    e.update(fits_under_85pc=True, headroom_to_85pc_bytes=e["bytes_total"] + 1)
    f = tmp_path / "sizing.json"
    f.write_text(json.dumps(s))
    monkeypatch.setattr(bx, "V2_SIZING", f)
    assert bx.v2_plan_fits(v2_pretraining_done=True, tier=1)[0] is True
    assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is False     # the whole grid still does not


def test_the_v2_extraction_pin_carries_the_batchnorm_twins():
    import subprocess
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    r = subprocess.run(["git", "show", f"{bx.V2_PIN}:experiments/EVAL/extract_v2.py"],
                       cwd=bx.ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"tag {bx.V2_PIN} not in this clone")
    assert "def bn_twin" in r.stdout and '"best70_bn"' in r.stdout


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
def test_the_v2_plan_is_refused_when_sizing_says_it_does_not_fit(tmp_path, monkeypatch):
    import json
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    s = json.loads(bx.V2_SIZING.read_text())
    e = s["storage"][bx.V2_PLAN]
    assert e["bytes_total"] == (e["extraction_bytes"] + e["pretraining_checkpoint_bytes"]
                                + e["fine_tuning_bytes"]) > 0
    e["n_models"] = len(bx.v2_runs()) + len(bx.v2_init_refs())   # the flag, whatever grid it was run on
    extra = (bx.V2_LINE_PC - 85) / 100 * s["volume"]["size_bytes"]   # the PI's line, 2026-10-07
    for fit in (False, True):
        e["headroom_to_85pc_bytes"] = e["bytes_total"] - extra + (1 if fit else -1)
        f = tmp_path / f"sizing_{fit}.json"
        f.write_text(json.dumps(s))
        monkeypatch.setattr(bx, "V2_SIZING", f)
        assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is fit
    e["n_models"] -= 1
    f.write_text(json.dumps(s))
    assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is False      # sized for another grid
    e["n_models"] += 1
    e["fine_tuning_bytes"] -= 1e9
    f.write_text(json.dumps(s))
    fits, why = bx.v2_plan_fits(v2_pretraining_done=True)
    assert fits is False and "now emits" in why  # sized for another set of fine-tuning specs
    s["fine_tuning_bytes_from"] = {"source": "--fine-tuning-bytes"}
    f.write_text(json.dumps(s))
    assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is True       # given, not derived
    e.pop("fine_tuning_bytes")
    f.write_text(json.dumps(s))
    assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is False      # one budget or no answer
    e["fine_tuning_bytes"] = 0.0
    e.pop("pretraining_checkpoint_bytes")
    f.write_text(json.dumps(s))
    assert bx.v2_plan_fits()[0] is False          # a sizing without the checkpoints is no answer
    monkeypatch.setattr(bx, "V2_SIZING", tmp_path / "sizing_False.json")
    monkeypatch.setattr(sys, "argv", ["build_extract_jobs.py", "--v2"])
    written = []
    monkeypatch.setattr(bx.pathlib.Path, "write_text", lambda self, t: written.append(self))
    with pytest.raises(SystemExit):
        bx.main()
    assert not written


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
def test_the_v2_plan_holds_back_the_pretraining_peak_until_pretraining_is_done(tmp_path, monkeypatch):
    import json
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    launch = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    peak = launch.V2_GRID_RESERVE_GIB * 2**30
    s = json.loads(bx.V2_SIZING.read_text())
    e = s["storage"][bx.V2_PLAN]
    assert peak > e["pretraining_checkpoint_bytes"]  # the peak covers what the runs keep at the end
    # fits with room to spare, but not once the peak beyond the kept checkpoints is held back
    spare = (peak - e["pretraining_checkpoint_bytes"]) / 2
    extra = (bx.V2_LINE_PC - 85) / 100 * s["volume"]["size_bytes"]   # the PI's line, 2026-10-07
    e.update(n_models=len(bx.v2_runs()) + len(bx.v2_init_refs()),
             headroom_to_85pc_bytes=e["bytes_total"] + spare - extra)
    f = tmp_path / "sizing.json"
    f.write_text(json.dumps(s))
    monkeypatch.setattr(bx, "V2_SIZING", f)
    assert bx.v2_plan_fits(v2_pretraining_done=True)[0] is True
    fits, why = bx.v2_plan_fits()
    assert fits is False and "held for v2 pretraining" in why
    e["headroom_to_85pc_bytes"] = (e["extraction_bytes"] + e["fine_tuning_bytes"]
                                   + peak + 1 - extra)    # the extraction, the fine-tuning and the peak
    f.write_text(json.dumps(s))
    assert bx.v2_plan_fits()[0] is True


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


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
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
    anywhere, windowed = xv.probe_feature_rules()
    s = next(iter(r["storage"].values()))
    # windowed-only classes enter with their in-window counts, not the whole split
    want = (400_000 * len(anywhere) + 8_000 * sum(len(c) for c, _ in windowed)
            + 2000 - np.isin(np.arange(2000) % 188, anywhere).sum())
    assert s["feature_rows_per_checkpoint"] == want
    # amendment A14: every run at five checkpoints (best70, wavg, bestval and the BatchNorm
    # twins of best70 and bestval), features with their pooled rows and observers; head
    # scores only where there is an output layer; five init references at one checkpoint.
    # No plan drops the weight average.
    cc = _load("class_counts", "experiments/EVAL/class_counts.py")
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    feat = want * (cc.FEATURE_ROW_BYTES + 128 * 2 + 4 * len(xv.V2_OBSERVERS))
    head = s["head_rows_per_checkpoint"] * cc.HEAD_ROW_BYTES
    runs = bx.v2_runs()
    n_cls, n_ssl = sum(1 for r_ in runs if r_[2]), sum(1 for r_ in runs if not r_[2])
    e = r["storage"][bx.V2_PLAN]
    assert (e["n_classification_runs"], e["n_self_supervised_runs"], e["n_init_references"]) == (n_cls, n_ssl, 5)
    assert e["n_models"] == len(runs) + 5 and n_ssl == 6
    assert e["extraction_bytes"] == 5 * n_cls * (feat + head) + 5 * n_ssl * feat + 5 * feat
    # one budget with the v2 fine-tuning (verification 2026-10-02): every spec it emits
    bf = _load("build_ft_jobs", "scripts/build_ft_jobs.py")
    assert e["fine_tuning_bytes"] == sum(bf.v2_need().values()) > 0
    assert r["fine_tuning_bytes_from"] == {"source": "v2_need", "specs": len(bf.v2_need())}
    assert e["bytes_total"] == e["extraction_bytes"] + e["pretraining_checkpoint_bytes"] + e["fine_tuning_bytes"]
    # A14's BatchNorm rule fired: the twins of best70 and bestval, inside the total
    assert e["batchnorm_twins_bytes"] == 2 * (n_cls * (feat + head) + n_ssl * feat)
    assert set(r["storage"]) == {bx.V2_PLAN, bx.V2_PLAN.replace("every run", "tier-1 runs")}
    assert all("BatchNorm twins of best70 and bestval" in k and v["checkpoints_per_run"] == 5
               for k, v in r["storage"].items())
    # the pretraining checkpoints: what a finished run keeps (pretrain_v2 prune), all runs
    ml = _load("build_mtx_launch", "scripts/build_mtx_launch.py")
    per_run = ((21 + 3 + 2) * ml.V2_STATE_MIB + ml.V2_RESUME_MIB + ml.V2_RECORDS_MIB
               + ml.V2_DIAG_BATCH_MIB) * 2**20                    # + the two BatchNorm twins
    assert r["checkpoint_bytes_per_run"] == pytest.approx(per_run)
    assert e["pretraining_checkpoint_bytes"] == pytest.approx(per_run * len(runs))
    # the early states amendment A14 added: about 98 MiB a run
    assert 11 * ml.V2_STATE_MIB == pytest.approx(98, abs=1)


@pytest.mark.skip(reason="the JetClass-II extraction plan for the probe battery, anomaly and label recovery is superseded by the downstream linear probes (PI scope 2026-10-09; scripts/build_linprobe_jobs.py)")
def test_the_committed_sizing_is_the_plan_the_generator_emits_on_this_grid():
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    s = json.loads(bx.V2_SIZING.read_text())
    e = s["storage"][bx.V2_PLAN]
    assert e["n_models"] == len(bx.build_v2()) and s["n_runs_v2_grid"] == len(bx.v2_runs())
    # df of /data, 2026-10-02T01:05Z
    assert s["volume"] == {"size_bytes": 2199023255552, "free_bytes": 1321495166976}
    assert e["headroom_to_85pc_bytes"] == pytest.approx(0.85 * 2199023255552 - 877528088576)
    # ONE BUDGET: with the BatchNorm twins (A14's rule fired, 2026-10-03) the extraction
    # does not fit even on its own, and the tier-1 plan not beside the v2 fine-tuning, so the
    # plan waits on the PI's storage decision
    bf = _load("build_ft_jobs", "scripts/build_ft_jobs.py")
    assert e["fine_tuning_bytes"] == sum(bf.v2_need().values())
    assert e["fits_under_85pc"] is (e["bytes_total"] < e["headroom_to_85pc_bytes"]) is False
    assert e["bytes_total"] - e["fine_tuning_bytes"] > e["headroom_to_85pc_bytes"]
    t1 = s["storage"][bx.V2_PLAN.replace("every run", "tier-1 runs")]
    assert t1["fits_under_85pc"] is False
    assert t1["bytes_total"] - t1["fine_tuning_bytes"] < t1["headroom_to_85pc_bytes"]
    assert bx.v2_plan_fits()[0] is False
    # the PI's 90 % line (2026-10-07): tier 1 with the v2 fine-tuning fits, the whole grid not
    assert bx.V2_LINE_PC == 90
    assert bx.v2_plan_fits(tier=1)[0] is True and bx.v2_plan_fits(v2_pretraining_done=True)[0] is False
    assert s["checkpoint_bytes_per_run_from"]["early_keep"] == [0, 2, 4, 9, 19, 29, 39, 49, 55, 62, 69]


def test_the_v2_plan_is_not_final_until_the_batchnorm_rule_has_read_out(tmp_path, monkeypatch):
    """Amendment A14, "Batch normalisation, decomposition on v1": if recomputing BatchNorm
    alone repairs at least half of the defective stored epochs (primary rule, eight runs),
    every reported v2 checkpoint gets a BatchNorm-recomputed twin. The rule fired
    (2026-10-03), so the plan extracts the twins; a readout that did not fire would make
    them superfluous, and no readout leaves the plan unfinal."""
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    monkeypatch.setattr(bx, "V2_BN_DIAG", tmp_path / "head_bn_diag.json")
    monkeypatch.setattr(bx, "ROOT", tmp_path)
    assert bx.v2_bn_twins_needed()[0] is None                       # no readout yet

    def readout(repaired, defective, runs=8):
        c = {"runs": runs, "repaired_by_bn": repaired, "defective_stored_epochs": defective}
        bx.V2_BN_DIAG.write_text(json.dumps({"primary_rule": "top1_10pct_vs_median",
                                             "counts": {"top1_10pct_vs_median": c,
                                                        "another_rule": {**c, "repaired_by_bn": 0}}}))
        return bx.v2_bn_twins_needed()[0]
    assert readout(5, 10) is True and readout(6, 11) is True        # at least half: twins
    assert readout(4, 9) is False and readout(0, 12) is False
    assert readout(10, 10, runs=7) is None                          # not over the eight runs
    # --v2 writes nothing while the rule is unread or does not fire, even when the plan fits
    monkeypatch.setattr(bx, "v2_plan_fits", lambda done=False, tier=None: (True, "fits"))
    monkeypatch.setattr(sys, "argv", ["build_extract_jobs.py", "--v2"])
    written = []
    monkeypatch.setattr(bx.pathlib.Path, "write_text", lambda self, t: written.append(self))
    for state in (None, False):
        monkeypatch.setattr(bx, "v2_bn_twins_needed", lambda s=state: (s, "why"))
        with pytest.raises(SystemExit, match="not final" if state is None else "does not require"):
            bx.main()
    assert not written


def test_the_committed_batchnorm_readout_fires_the_rule():
    bx = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
    fires, why = bx.v2_bn_twins_needed()
    assert fires is True and why.startswith("recomputing BatchNorm alone repairs 21 of the 22 ")
    assert {"best70_bn", "bestval_bn"} <= set(bx.V2_CHECKPOINTS)


def test_classes_only_a_windowed_task_reads_are_kept_inside_its_window():
    anywhere, windowed = xv.probe_feature_rules()
    qcd, _ = xv.anomaly_classes()
    assert 169 in anywhere and 181 in anywhere            # b vs c in QCD is unwindowed
    assert not (set(qcd) - {169, 181}) & set(anywhere)
    (cls, win), = windowed
    assert 70 in cls and set(win) == {"jet_pt", "jet_sdmass", "jet_eta"}
    # X->bc, X->cs, X->bq everywhere: the single-pair b-vs-c tasks have no window
    assert set(xv.full_range_classes()) == {4, 5, 6} <= set(anywhere) and not {4, 5, 6} & set(cls)
    s = xv.Selector(anywhere, 0, [], 0, 0, windowed)
    lab = np.array([4, 4, 170, 0, 170, 70, 70])
    obs = {"jet_pt": np.array([500., 300., 500., 300., 700., 500., 300.]),
           "jet_sdmass": np.array([100., 100., 120., 50., 100., 100., 100.]),
           "jet_eta": np.zeros(7)}
    assert s.feature_mask(np.arange(7), lab, obs).tolist() == [True, True, True, True, False,
                                                              True, False]
    with pytest.raises(SystemExit, match="observers"):
        s.feature_mask(np.arange(7), lab, None)
