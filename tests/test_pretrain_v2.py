"""v2 pretraining streams, loader, seeds, validation and resume
(experiments/MTX/stream_v2.py, experiments/MTX/pretrain_v2.py).

Everything here runs on CPU against small synthetic JetClass-II-shaped parquet
files and the real arm configs, with reweighting histograms written into a
sidecar exactly as the make_weight job would.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MTX = ROOT / "experiments" / "MTX"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MTX))

torch = pytest.importorskip("torch")
ak = pytest.importorskip("awkward")
pytest.importorskip("weaver")
yaml = pytest.importorskip("yaml")

import stream_v2 as sv  # noqa: E402
import pretrain_v2 as pv  # noqa: E402

# The laptop's weaver (dev branch) moved trunc_normal_ out of ParticleTransformer,
# where the image's 0.4.17 has it and mpm.Decoder imports it from.
# Its Block also lacks 0.4.17's add_bias_kv, so the self-supervised decoder
# cannot be built here; those checks run in the image (the smoke job runs this file).
import inspect  # noqa: E402
import weaver.nn.model.ParticleTransformer as _part  # noqa: E402
if not hasattr(_part, "trunc_normal_"):
    _part.trunc_normal_ = torch.nn.init.trunc_normal_
needs_0417 = pytest.mark.skipif(
    "add_bias_kv" not in inspect.signature(_part.Block.__init__).parameters,
    reason="the installed weaver is not the image's 0.4.17 (no Block.add_bias_kv)")

FAMILIES = {"Res2P": (3, 0, 15), "Res34P": (4, 15, 161), "QCD": (3, 161, 188)}
ROWS = 300


def _write_file(path, lab_lo, lab_hi, rng):
    n = ROWS
    npart = rng.integers(4, 24, n)
    tot = int(npart.sum())
    jet_pt = rng.uniform(150, 2600, n)
    jet_eta = rng.uniform(-2, 2, n)
    frac = rng.uniform(0.01, 0.2, tot)
    phi = rng.uniform(-np.pi, np.pi, tot)
    pt = np.repeat(jet_pt, npart) * frac
    pz = pt * np.sinh(np.repeat(jet_eta, npart) + rng.normal(0, 0.1, tot))
    ptype = rng.integers(0, 5, tot)

    def jag(x, dt="float32"):
        return ak.unflatten(np.asarray(x, dtype=dt), npart)

    sdm = rng.uniform(10, 510, n)
    arr = {
        "part_px": jag(pt * np.cos(phi)), "part_py": jag(pt * np.sin(phi)), "part_pz": jag(pz),
        "part_energy": jag(np.hypot(pt, pz) * 1.0001),
        "part_deta": jag(rng.normal(0, 0.2, tot)), "part_dphi": jag(rng.normal(0, 0.2, tot)),
        "part_d0val": jag(rng.normal(0, 0.1, tot)), "part_d0err": jag(rng.uniform(0, 0.1, tot)),
        "part_dzval": jag(rng.normal(0, 0.1, tot)), "part_dzerr": jag(rng.uniform(0, 0.1, tot)),
        "part_charge": jag(rng.integers(-1, 2, tot)),
        "part_isChargedHadron": jag(ptype == 0), "part_isNeutralHadron": jag(ptype == 1),
        "part_isPhoton": jag(ptype == 2), "part_isElectron": jag(ptype == 3), "part_isMuon": jag(ptype == 4),
        "jet_pt": jet_pt.astype("float32"), "jet_eta": jet_eta.astype("float32"),
        "jet_energy": (jet_pt * np.cosh(jet_eta) * 1.05).astype("float32"),
        "jet_sdmass": sdm.astype("float32"),
        "genjet_sdmass": np.where(rng.uniform(size=n) < 0.05, 0.0, sdm * rng.uniform(0.9, 1.1, n)).astype("float32"),
        "jet_label": rng.integers(lab_lo, lab_hi, n).astype("int32"),
    }
    ak.to_parquet(ak.Array(arr), str(path))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from weaver.utils.data.config import _md5
    root = tmp_path_factory.mktemp("jc2")
    rng = np.random.default_rng(7)
    files = {}
    for fam, (n, lo, hi) in FAMILIES.items():
        files[fam] = []
        for i in range(n + 1):          # the last file of each family is validation
            p = root / f"{fam}_{i:04d}.parquet"
            _write_file(p, lo, hi, rng)
            files[fam].append(str(p))
    cfgs = {}
    for arm in ("R16_Q1", "L188", "L162_MASS"):
        src = (ROOT / "configs" / "arms" / f"{arm}.yaml").read_text()
        cfg = root / f"{arm}.yaml"
        cfg.write_text(src)
        opts = yaml.safe_load(src)
        classes = opts["weights"]["reweight_classes"]
        nx = len(opts["weights"]["reweight_vars"]["jet_pt"]) - 1
        ny = len(opts["weights"]["reweight_vars"]["jet_sdmass"]) - 1
        opts["weights"]["reweight_hists"] = {
            c: (np.full((nx, ny), 0.3 if "QCD" in c else 0.6)).tolist() for c in classes}
        (root / f"{arm}.{_md5(str(cfg))}.auto.yaml").write_text(yaml.safe_dump(opts, sort_keys=False))
        cfgs[arm] = str(cfg)
    train = {f: v[:-1] for f, v in files.items()}
    val = sorted(v[-1] for v in files.values())
    return {"root": root, "train": train, "val": val, "cfg": cfgs}


def _dc(data, arm):
    return pv.sidecar(data["cfg"][arm])


def _rows(ds, epoch, workers=0):
    ds.set_epoch(epoch)
    kw = {"multiprocessing_context": "fork"} if workers and sys.platform != "win32" else {}
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, **kw)
    out = []
    for i, (X, y, Z) in enumerate(loader):
        out.append(Z["_rowid"].numpy())
        if ds.mode == "train" and i >= 5:
            break
    return np.concatenate(out)


# ------------------------------------------------------------------ schedule
def test_n_div_d_sep_is_sophons_documented_example():
    np.testing.assert_allclose(sv.n_div_d_sep(5, 3),
                               [[0, 0, 0, 0, 0], [1, 2 / 3, 0, 0, 0], [1, 1, 1, 1 / 3, 0], [1, 1, 1, 1, 1]])


def test_every_split_reads_every_family_and_every_file_is_read_once():
    fd = {"A": [f"a{i}" for i in range(40)], "B": [f"b{i}" for i in range(172)], "Q": [f"q{i}" for i in range(56)]}
    plan = sv.sophon_splits(fd, 200)
    assert len(plan) == 200
    cover = {}
    for files, ranges in plan:
        assert {f[0] for f in files} == {"a", "b", "q"}
        for f, (lo, hi) in zip(files, ranges):
            cover.setdefault(f, []).append((lo, hi))
    for f, rs in cover.items():           # the pieces tile [0, 1) exactly
        rs.sort()
        assert rs[0][0] == 0 and rs[-1][1] == 1
        assert all(a[1] == b[0] for a, b in zip(rs, rs[1:]))
    rows = {f: set() for f in cover}      # and as row slices, each row once
    for f, rs in cover.items():
        for lo, hi in rs:
            a, b = sv._slice_bounds(100_000, lo, hi)
            assert not rows[f] & set(range(a, b))
            rows[f] |= set(range(a, b))
        assert rows[f] == set(range(100_000))


def test_training_load_ranges_take_random_rows_that_still_tile_each_file():
    n, pieces = 1000, [(0.0, 0.3), (0.3, 2 / 3), (2 / 3, 1.0)]
    got = [sv.file_rows(n, lo, hi, 5, 2, 1, 0, 7) for lo, hi in pieces]
    allrows = np.concatenate(got)
    assert sorted(allrows.tolist()) == list(range(n))          # each row exactly once per pass
    assert [len(g) for g in got] == [sv._slice_bounds(n, lo, hi)[1] - sv._slice_bounds(n, lo, hi)[0] for lo, hi in pieces]
    assert not np.array_equal(got[0], np.arange(300))          # not the contiguous slice
    assert got[0].mean() == pytest.approx(n / 2, abs=60)       # spread over the file
    np.testing.assert_array_equal(got[1], sv.file_rows(n, 0.3, 2 / 3, 5, 2, 1, 0, 7))
    assert not np.array_equal(got[1], sv.file_rows(n, 0.3, 2 / 3, 5, 3, 1, 0, 7))


def test_reweighting_draws_equal_weavers_given_the_same_generator():
    from weaver.utils.dataset import _get_reweight_indices
    w = np.random.default_rng(3).uniform(0, 1, 5000) ** 3
    np.random.seed(11)
    ref = _get_reweight_indices(w, up_sample=True, weight_scale=1, max_resample=10)
    got = sv.reweight_indices(w, np.random.RandomState(11), max_resample=10)
    np.testing.assert_array_equal(ref, got)


# ------------------------------------------------------------------ streams
def test_epoch_stream_is_a_function_of_seed_epoch_and_worker_only(data):
    a = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=5, split_num=4)
    b = sv.StreamDataset(data["train"], _dc(data, "L188"), mode="train", batch_size=64, seed=5, split_num=4)
    e2_after = (_rows(a, 0, 2), _rows(a, 1, 2), _rows(a, 2, 2))[2]
    e2_direct = _rows(b, 2, 2)                 # other vocabulary, no epochs before
    np.testing.assert_array_equal(e2_after, e2_direct)
    assert not np.array_equal(_rows(a, 1, 2), e2_direct)
    c = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=6, split_num=4)
    assert not np.array_equal(_rows(c, 2, 2), e2_direct)


def test_labels_only_projection_draws_the_same_rows(data):
    full = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=5, split_num=4)
    lab = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=5,
                           split_num=4, labels_only=True)
    assert "part_px" not in lab.load_branches
    np.testing.assert_array_equal(_rows(full, 3, 2), _rows(lab, 3, 2))


def test_an_extra_selection_removes_its_jets_from_both_streams(data):
    sel = "(jet_label < 100) | (jet_label >= 161)"
    tr = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=5,
                          split_num=4, extra_selection=sel)
    va = sv.StreamDataset({"_": data["val"]}, _dc(data, "R16_Q1"), mode="val", batch_size=64,
                          extra_selection=sel)
    for ds in (tr, va):
        ds.set_epoch(0)
        import itertools
        batches = itertools.islice(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0), 6)
        labels = np.concatenate([Z["_jet_label"].numpy() for _, _, Z in batches])
        assert len(labels) and not ((labels >= 100) & (labels < 161)).any()


def test_validation_is_every_selected_row_in_a_fixed_order(data):
    dc = _dc(data, "R16_Q1")
    ds = sv.StreamDataset({"_": data["val"]}, dc, mode="val", batch_size=64)
    r0, r1 = _rows(ds, 0, 2), _rows(ds, 7, 2)
    np.testing.assert_array_equal(r0, r1)
    n_sel = 0
    for f in data["val"]:
        t = ak.from_parquet(f)
        n_sel += int(ak.sum((t.jet_pt > 200) & (t.jet_pt < 2500) & (t.jet_sdmass > 20) & (t.jet_sdmass < 500)))
    assert len(r0) == len(set(r0.tolist())) == n_sel


def test_plan_hash_moves_with_seed_epoch_and_worker_count():
    fd = {"A": [f"/x/A_{i:04d}.parquet" for i in range(40)], "Q": [f"/x/Q_{i:04d}.parquet" for i in range(30)]}
    h = sv.plan_sha256(fd, 5, 0, 5, 20, 1.0)
    assert h == sv.plan_sha256(fd, 5, 0, 5, 20, 1.0)
    assert h != sv.plan_sha256(fd, 5, 1, 5, 20, 1.0)
    assert h != sv.plan_sha256(fd, 6, 0, 5, 20, 1.0)
    assert h != sv.plan_sha256(fd, 5, 0, 2, 20, 1.0)


def test_native_to_class_finds_the_qcd_group():
    from weaver.utils.data.config import DataConfig
    m = sv.native_to_class(DataConfig.load(str(ROOT / "configs/arms/R16_Q1.yaml"), load_observers=False))
    assert set(m[161:].tolist()) == {16} and 16 not in set(m[:161].tolist())
    m = sv.native_to_class(DataConfig.load(str(ROOT / "configs/arms/L188.yaml"), load_observers=False))
    np.testing.assert_array_equal(m, np.arange(188))


# ------------------------------------------------------------------ seeds
def _build(data, arm, k, seeds, arch="ParT_sophon_arch_mtx.py", extra=None):
    opts = {"num_classes": k, "fc_params": [(512, 0.1)]}
    opts.update(extra or {})
    model, _ = pv.build_model(str(MTX / arch), pv.load_data_config(data["cfg"][arm]), opts, seeds)
    return model


def _eq(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_trunk_init_seeds_the_whole_trunk_and_nothing_else(data):
    s = {"trunk_init": 1, "head_init": 2}
    m17, m188 = _build(data, "R16_Q1", 17, s), _build(data, "L188", 188, s)
    assert _eq(pv.trunk_state(m17), pv.trunk_state(m188))       # class token included
    assert "cls_token" in pv.trunk_state(m17)
    t2 = _build(data, "R16_Q1", 17, {"trunk_init": 9, "head_init": 2})
    assert not torch.equal(pv.trunk_state(t2)["cls_token"], pv.trunk_state(m17)["cls_token"])
    assert _eq({k: v for k, v in t2.state_dict().items() if "fc." in k},
               {k: v for k, v in m17.state_dict().items() if "fc." in k})
    h2 = _build(data, "R16_Q1", 17, {"trunk_init": 1, "head_init": 9})
    assert _eq(pv.trunk_state(h2), pv.trunk_state(m17))
    assert not torch.equal(h2.state_dict()["mod.fc.1.weight"], m17.state_dict()["mod.fc.1.weight"])


def test_the_mass_model_shares_the_trunk_initialisation(data, monkeypatch):
    monkeypatch.setenv("HYBRID_MASS_INSTALLED", "1")
    s = {"trunk_init": 1, "head_init": 2}
    ref = pv.trunk_state(_build(data, "R16_Q1", 17, s))
    assert _eq(pv.trunk_state(_build(data, "L162_MASS", 162, s, "ParT_sophon_arch_mass.py")), ref)


@needs_0417
def test_the_self_supervised_model_shares_the_trunk_initialisation(data, monkeypatch):
    monkeypatch.setenv("MPM_INSTALLED", "1")
    s = {"trunk_init": 1, "head_init": 2}
    ref = pv.trunk_state(_build(data, "R16_Q1", 17, s))
    mpm = _build(data, "L188", 188, s, "ParT_sophon_arch_mpm.py", {"mask_rate": 0.4})
    assert _eq(pv.trunk_state(mpm), ref)


def test_lr_schedule_is_weavers_flat_plus_decay():
    m = torch.nn.Linear(2, 2)
    opt, sched = pv.make_optimizer(m, 5e-4, 80)
    lrs = []
    for _ in range(80):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    g = 0.01 ** (1 / 24)
    np.testing.assert_allclose(lrs, [5e-4 * g ** max(0, e - 55) for e in range(80)], rtol=1e-12)


def test_optimizer_state_keeps_the_lookahead_slow_weights():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    opt, _ = pv.make_optimizer(m, 1e-2, 10)
    for _ in range(9):                       # 9 steps: counter 3, slow != fast
        opt.zero_grad()
        m(torch.randn(8, 4)).pow(2).sum().backward()
        opt.step()
    import io
    buf = io.BytesIO()                       # through bytes, as through the resume file
    torch.save(pv.optimizer_state(opt), buf)
    buf.seek(0)
    st = torch.load(buf, weights_only=False)
    slow = [t.clone() for t in st["slow"]]
    assert opt.step_counter == 3 and not torch.equal(slow[0], m.weight)
    m2 = torch.nn.Linear(4, 3)
    m2.load_state_dict(m.state_dict())
    opt2, _ = pv.make_optimizer(m2, 1e-2, 10)
    pv.load_optimizer_state(opt2, st)
    assert opt2.step_counter == 3
    for p2, s in zip(m2.parameters(), slow):
        assert torch.equal(opt2.state[p2]["cached_params"], s)
    x = torch.randn(8, 4)
    for mm, oo in ((m, opt), (m2, opt2)):
        for _ in range(4):
            oo.zero_grad()
            mm(x).pow(2).sum().backward()
            oo.step()
    assert torch.equal(m.weight, m2.weight)


# ------------------------------------------------------------------ the driver end to end
def _args(data, out, arm="R16_Q1", k=17, epochs=3, extra=()):
    head = ["-o", "num_classes", str(k), "-o", "fc_params", "[(512,0.1)]"] if k else []
    a = ["--seed", "3", "--out", str(out),
         "--data-train", *[f"{fam}:{p}" for fam, ps in data["train"].items() for p in ps],
         "--data-val", *data["val"], "--data-config", data["cfg"][arm],
         "--network-config", str(MTX / "ParT_sophon_arch_mtx.py"), *head,
         "--batch-size", "32", "--samples-per-epoch", "96", "--num-epochs", str(epochs),
         "--num-workers", "2", "--data-split-num", "4", "--log-every", "0", "--device", "cpu"]
    return a + list(extra)


def _epochs(out, what):
    return {int(p.stem.split("-")[1]): json.loads(p.read_text()) for p in sorted((out / what).glob("epoch-*.json"))}


@pytest.fixture(scope="module")
def run_a(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("A") / "a"
    assert pv.main(_args(data, out)) == 0
    return out


def test_a_resumed_run_repeats_the_uninterrupted_run_exactly(data, run_a, tmp_path, monkeypatch):
    out = tmp_path / "b"
    real = pv.torch_save

    def dies_after_epoch_1(obj, path):
        real(obj, path)
        if path.name == "net_epoch-1_resume.pt":
            raise KeyboardInterrupt("killed")
    monkeypatch.setattr(pv, "torch_save", dies_after_epoch_1)
    with pytest.raises(KeyboardInterrupt):
        pv.main(_args(data, out))
    monkeypatch.setattr(pv, "torch_save", real)
    assert pv.latest_complete_epoch(out) == 1
    assert pv.main(_args(data, out)) == 0
    sa, sb = _epochs(run_a, "stream"), _epochs(out, "stream")
    assert {e: r["sha256"] for e, r in sa.items()} == {e: r["sha256"] for e, r in sb.items()}
    # To float32 tolerance, the acceptance criterion: CPU kernels on a loaded laptop
    # have differed in the last bits between two identical fresh runs (1e-6 relative
    # in the validation loss of epoch 0, which no resume touches).
    ma, mb = _epochs(run_a, "metrics"), _epochs(out, "metrics")
    for e in range(3):
        assert ma[e]["train"]["n_jets"] == mb[e]["train"]["n_jets"]
        for part in ("train", "val"):
            for k in ("loss", "acc", "head_top1_acc", "p_qcd_resonant", "p_qcd_qcd"):
                if k in ma[e][part]:
                    assert ma[e][part][k] == pytest.approx(mb[e][part][k], rel=1e-5), (e, part, k)
    sa2, sb2 = torch.load(run_a / "net_epoch-2_state.pt"), torch.load(out / "net_epoch-2_state.pt")
    assert sa2.keys() == sb2.keys()
    for k in sa2:
        torch.testing.assert_close(sa2[k], sb2[k], rtol=1e-5, atol=1e-6)


def test_another_output_width_sees_the_same_stream_and_trunk(data, run_a, tmp_path):
    out = tmp_path / "c"
    assert pv.main(_args(data, out, arm="L188", k=188, epochs=1)) == 0
    assert _epochs(out, "stream")[0]["sha256"] == _epochs(run_a, "stream")[0]["sha256"]
    assert _eq(torch.load(out / "init_trunk.pt")["trunk"], torch.load(run_a / "init_trunk.pt")["trunk"])


def test_records_are_written_in_the_agreed_format(run_a):
    rec = _epochs(run_a, "stream")[1]
    assert set(rec) == {"run", "epoch", "seed_data", "seed_dropout", "files_sha256", "rows_sha256",
                        "sha256", "n_jets"}
    assert rec["sha256"] == hashlib.sha256((rec["files_sha256"] + rec["rows_sha256"]).encode()).hexdigest()
    assert rec["n_jets"] == 96 and rec["epoch"] == 1 and rec["run"] == "a"
    m = _epochs(run_a, "metrics")[2]
    for k in ("acc", "loss", "head_top1_acc", "p_qcd_resonant", "p_qcd_qcd"):
        assert math.isfinite(m["val"][k])
    assert 0 < m["train"]["qcd_share"] < 1 and m["stream_sha256"] == _epochs(run_a, "stream")[2]["sha256"]
    best = json.loads((run_a / "best_epoch.json").read_text())
    vals = {e: r["val"]["acc"] for e, r in _epochs(run_a, "metrics").items()}
    assert best["epoch"] == max(vals, key=vals.get)
    assert _eq(torch.load(run_a / "net_best_epoch_state.pt"), torch.load(run_a / f"net_epoch-{best['epoch']}_state.pt"))
    assert sorted(p.name for p in run_a.glob("net_epoch-*_state.pt")) == [f"net_epoch-{e}_state.pt" for e in range(3)]


def test_a_different_recipe_in_the_same_directory_halts(data, run_a, tmp_path):
    out = tmp_path / "r"
    shutil.copytree(run_a, out)
    assert pv.main(_args(data, out, extra=["--start-lr", "1e-3"])) == pv.EXIT_HALT


def test_window_retention_keeps_best_last_ten_and_newest_resume(tmp_path):
    for e in range(15):
        (tmp_path / f"net_epoch-{e}_state.pt").write_text("s")
        (tmp_path / f"net_epoch-{e}_resume.pt").write_text("r")
    pv.prune(tmp_path, {2} | set(range(5, 15)), 14)
    assert sorted(int(p.name.split("-")[1].split("_")[0]) for p in tmp_path.glob("*_state.pt")) == [2] + list(range(5, 15))
    assert [p.name for p in tmp_path.glob("*_resume.pt")] == ["net_epoch-14_resume.pt"]


def test_states_retention_keeps_every_state_and_the_newest_resume(tmp_path):
    for e in range(5):
        (tmp_path / f"net_epoch-{e}_state.pt").write_text("s")
        (tmp_path / f"net_epoch-{e}_resume.pt").write_text("r")
    pv.prune(tmp_path, None, 4)
    assert len(list(tmp_path.glob("*_state.pt"))) == 5
    assert [p.name for p in tmp_path.glob("*_resume.pt")] == ["net_epoch-4_resume.pt"]


def test_the_mass_path_runs(data, tmp_path):
    mass = ["--network-config", str(MTX / "ParT_sophon_arch_mass.py"), "--mass-lambda", "5.0"]
    out = tmp_path / "m"
    assert pv.main(_args(data, out, arm="L162_MASS", k=162, epochs=2, extra=mass)) == 0
    m = _epochs(out, "metrics")[1]
    assert math.isfinite(m["val"]["loss_reg"]) and math.isfinite(m["train"]["loss_reg"])
    assert m["selection"]["metric"] == "val.acc"


@needs_0417
def test_the_self_supervised_path_runs(data, tmp_path):
    ssl = ["--network-config", str(MTX / "ParT_sophon_arch_mpm.py"), "--mpm"]
    out2 = tmp_path / "s"
    assert pv.main(_args(data, out2, arm="L188", k=0, epochs=2, extra=ssl)) == 0
    s = _epochs(out2, "metrics")
    assert s[1]["selection"]["metric"] == "-val.loss" and math.isfinite(s[1]["val"]["loss"])
    # the self-supervised stream is the supervised stream
    assert _epochs(out2, "stream")[0]["n_jets"] == 96


def test_the_loader_dry_run_draws_the_rows_training_draws(data, run_a, tmp_path):
    import loader_dryrun
    out = tmp_path / "dry.json"
    assert loader_dryrun.main([
        "--seed", "3", "--epochs", "3", "--samples-per-epoch", "96", "--batch-size", "32",
        "--num-workers", "2", "--data-split-num", "4", "--check-jets", "64",
        "--data-config", data["cfg"]["R16_Q1"], "--out", str(out),
        "--data-train", *[f"{fam}:{p}" for fam, ps in data["train"].items() for p in ps]]) == 0
    dry = json.loads(out.read_text())
    assert [e["sha256"] for e in dry["epochs"]] == [r["sha256"] for r in _epochs(run_a, "stream").values()]
    m = _epochs(run_a, "metrics")
    assert [e["qcd_share"] for e in dry["epochs"]] == [m[e]["train"]["qcd_share"] for e in range(3)]
    assert dry["summary"]["jets_per_epoch"] == 96 and sum(c[2] for c in dry["epochs"][0]["per_fetch"]) == 96
