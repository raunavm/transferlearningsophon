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
    got = [sv.file_rows(n, lo, hi, 5, 2, 0, 7) for lo, hi in pieces]
    allrows = np.concatenate(got)
    assert sorted(allrows.tolist()) == list(range(n))          # each row exactly once per pass
    assert [len(g) for g in got] == [sv._slice_bounds(n, lo, hi)[1] - sv._slice_bounds(n, lo, hi)[0] for lo, hi in pieces]
    assert not np.array_equal(got[0], np.arange(300))          # not the contiguous slice
    assert got[0].mean() == pytest.approx(n / 2, abs=60)       # spread over the file
    np.testing.assert_array_equal(got[1], sv.file_rows(n, 0.3, 2 / 3, 5, 2, 0, 7))
    assert not np.array_equal(got[1], sv.file_rows(n, 0.3, 2 / 3, 5, 3, 0, 7))


def test_a_data_fraction_reads_every_file_every_epoch_and_every_row_once_per_cycle():
    fd = {"A": [f"/x/A_{i:04d}.parquet" for i in range(40)], "Q": [f"/x/Q_{i:04d}.parquet" for i in range(14)]}
    k, n = 5, 1000
    seen = {}
    for e in range(k):                          # one cycle
        cycle, (wlo, whi) = sv.cycle_of(e, 1 / k)
        assert cycle == 0 and (wlo, whi) == (pytest.approx(e / k), pytest.approx((e + 1) / k))
        plan = sv.train_plan(fd, 5, e, 0, 2, 20, 1.0, 0, 1 / k)
        files = {f for fs, _ in plan for f in fs}
        assert files == set(sv.worker_files(fd, 0, 2)["A"] + sv.worker_files(fd, 0, 2)["Q"])
        for fs, rs in plan:
            for f, (lo, hi) in zip(fs, rs):
                seen.setdefault(f, []).append(sv.file_rows(n, lo, hi, 5, cycle, 0, 3))
    for f, parts in seen.items():               # file index fixed at 3: same permutation
        rows = np.concatenate(parts)
        assert sorted(rows.tolist()) == list(range(n)), f
    assert sv.cycle_of(5, 0.2)[0] == 1 and sv.cycle_of(7, 0.2)[1] == (pytest.approx(0.4), pytest.approx(0.6))
    with pytest.raises(ValueError):
        sv.cycle_of(0, 0.3)


def test_an_integer_window_count_tiles_every_file_exactly():
    fd = {"A": [f"/x/A_{i:04d}.parquet" for i in range(40)], "Q": [f"/x/Q_{i:04d}.parquet" for i in range(14)]}
    for n in (300, 999, 1000):                  # divisible by three and not
        seen = {}
        for e in range(3):                      # one cycle
            cycle, window = sv.window_of(e, 3)
            assert cycle == 0 and window == (e, 3)
            per_epoch = {}
            for fs, rs in sv.train_plan(fd, 5, e, 0, 2, 20, 1.0, 0, windows=3):
                for f, (lo, hi) in zip(fs, rs):
                    r = sv.file_rows(n, lo, hi, 5, cycle, 0, 3, window)
                    per_epoch.setdefault(f, []).append(r)
                    seen.setdefault(f, []).append(r)
            for f, parts in per_epoch.items():  # no row twice within an epoch
                rows = np.concatenate(parts)
                assert len(rows) == len(set(rows.tolist()))
        for f, parts in seen.items():           # every row once per cycle
            assert sorted(np.concatenate(parts).tolist()) == list(range(n)), (n, f)
    assert sv.window_of(4, 3) == (1, (1, 3))
    for bad in (0, 2.5):
        with pytest.raises(ValueError):
            sv.window_of(0, bad)
    assert sv.plan_sha256(fd, 5, 0, 2, 20, 1.0, 1 / 3, 3) != sv.plan_sha256(fd, 5, 1, 2, 20, 1.0, 1 / 3, 3)


def test_the_fraction_path_is_unchanged_by_integer_windows():
    """Hashes from the code before integer windows (commit c96fc60): the arms at
    --data-fraction 0.2 draw the same schedule and rows."""
    fd = {"Res2P": [f"/jc2/jet_data/Res2P_{i:04d}.parquet" for i in range(200)],
          "Res34P": [f"/jc2/jet_data/Res34P_{i:04d}.parquet" for i in range(860)],
          "QCD": [f"/jc2/jet_data/QCD_{i:04d}.parquet" for i in range(280)]}
    assert sv.plan_sha256(fd, 12345, 0, 5, 200, 1.0, 0.2) == \
        "6e3d463b7496dfcc8dbaeb43c57bf8f842e4add985bd12539ca250de84d47926"
    assert sv.plan_sha256(fd, 12345, 7, 5, 200, 1.0, 0.2) == \
        "2d031b7e28fbc67d1165d2a6d65955393c161120225c054739a41d4ae6b810a7"
    h = hashlib.sha256()
    for e in range(5):
        cyc, (lo, hi) = sv.cycle_of(e, 0.2)
        for a, b in ((lo, lo + (hi - lo) * 0.37), (lo + (hi - lo) * 0.37, hi)):
            h.update(sv.file_rows(100000, a, b, 777, cyc, 0, 11).tobytes())
    assert h.hexdigest() == "9ea34abcef983a671fec5708a8af7dd7c25652ac90e99eba1f8645225212d336"


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


def test_consecutive_epochs_of_a_cycle_read_disjoint_rows(data):
    ds = sv.StreamDataset(data["train"], _dc(data, "R16_Q1"), mode="train", batch_size=64, seed=5,
                          split_num=4, data_fraction=0.5)
    r0, r1 = set(_rows(ds, 0, 2).tolist()), set(_rows(ds, 1, 2).tolist())
    assert r0 and r1 and not r0 & r1


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
    # Streams exactly; numbers to 1e-4. On the laptop two identical FRESH runs have
    # differed by 1e-6 relative in the epoch-0 validation loss, which no resume touches,
    # and by more under heavy load (threaded BLAS). The bitwise check is the GPU smoke
    # (RUNS.csv mtx2-smoke-3090: A and A2 bitwise equal; the resume state bitwise equal).
    ma, mb = _epochs(run_a, "metrics"), _epochs(out, "metrics")
    for e in range(3):
        assert ma[e]["train"]["n_jets"] == mb[e]["train"]["n_jets"]
        for part in ("train", "val"):
            for k in ("loss", "acc", "head_top1_acc", "p_qcd_resonant", "p_qcd_qcd"):
                if k in ma[e][part]:
                    rel = abs(ma[e][part][k] - mb[e][part][k]) / abs(ma[e][part][k])
                    assert rel < 1e-4, (e, part, k, ma[e][part][k], mb[e][part][k], rel)
    sa2, sb2 = torch.load(run_a / "net_epoch-2_state.pt"), torch.load(out / "net_epoch-2_state.pt")
    assert sa2.keys() == sb2.keys()
    for k in sa2:
        torch.testing.assert_close(sa2[k], sb2[k], rtol=1e-4, atol=1e-5)


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
    assert best["epoch"] == max(vals, key=vals.get) and best["metric"] == "val.acc"
    assert all(r["selection"]["value"] == r["val"]["acc"] for r in _epochs(run_a, "metrics").values())
    assert json.loads((run_a / "recipe.json").read_text())["select_on"] == "acc"
    assert _eq(torch.load(run_a / "net_best_epoch_state.pt"), torch.load(run_a / f"net_epoch-{best['epoch']}_state.pt"))
    assert sorted(p.name for p in run_a.glob("net_epoch-*_state.pt")) == [f"net_epoch-{e}_state.pt" for e in range(3)]


def test_a_different_recipe_in_the_same_directory_halts(data, run_a, tmp_path):
    out = tmp_path / "r"
    shutil.copytree(run_a, out)
    assert pv.main(_args(data, out, extra=["--start-lr", "1e-3"])) == pv.EXIT_HALT


def test_a_resume_under_other_code_halts(data, run_a, tmp_path, monkeypatch):
    rec = json.loads((run_a / "recipe.json").read_text())["code"]
    assert set(rec) == {"repo_ref", "commit"} and rec["commit"]
    out = tmp_path / "t"
    shutil.copytree(run_a, out)
    monkeypatch.setenv("REPO_REF", "mtx-s9.99" if rec["repo_ref"] != "mtx-s9.99" else "mtx-s9.98")
    assert pv.main(_args(data, out)) == pv.EXIT_HALT


def test_a_sidecar_that_is_not_its_config_plus_histograms_halts(data, tmp_path):
    """Another config's sidecar under this config's md5 (stale or renamed) never trains;
    the arm's own passes."""
    from weaver.utils.data.config import _md5
    assert pv.sidecar(data["cfg"]["R16_Q1"]) == sv.sidecar_path(data["cfg"]["R16_Q1"])
    cfg = tmp_path / "R16_Q1.yaml"
    shutil.copy(data["cfg"]["R16_Q1"], cfg)
    shutil.copy(pv.sidecar(data["cfg"]["L188"]), tmp_path / f"R16_Q1.{_md5(str(cfg))}.auto.yaml")
    assert "labels" in sv.sidecar_mismatch(str(cfg), sv.sidecar_path(str(cfg)))
    with pytest.raises(SystemExit) as e:
        pv.sidecar(str(cfg))
    assert e.value.code == pv.EXIT_HALT
    lone = tmp_path / "lone" / "R16_Q1.yaml"                  # a config with no sidecar
    lone.parent.mkdir()
    shutil.copy(data["cfg"]["R16_Q1"], lone)
    with pytest.raises(SystemExit) as e:
        pv.sidecar(str(lone))
    assert e.value.code == pv.EXIT_HALT


def test_a_sidecar_written_as_weavers_make_weight_writes_it_passes(tmp_path):
    """weaver's training load drops the observers (load_observers=False) and
    WeightMaker.produce adds weights.reweight_hists before dumping the options."""
    from weaver.utils.data.config import DataConfig
    for arm in ("R16_Q1", "L162_MASS"):
        cfg = tmp_path / f"{arm}.yaml"
        shutil.copy(ROOT / "configs" / "arms" / f"{arm}.yaml", cfg)
        dc = DataConfig.load(str(cfg), load_observers=False)
        dc.options["weights"]["reweight_hists"] = {c: [[0.5] * 3] * 3 for c in dc.reweight_classes}
        dc.dump(sv.sidecar_path(str(cfg)))
        assert DataConfig.load(sv.sidecar_path(str(cfg))).options["observers"] == []
        assert sv.sidecar_mismatch(str(cfg), sv.sidecar_path(str(cfg))) == []


def test_window_retention_keeps_best_last_ten_and_newest_resume(tmp_path):
    for e in range(15):
        (tmp_path / f"net_epoch-{e}_state.pt").write_text("s")
        (tmp_path / f"net_epoch-{e}_resume.pt").write_text("r")
    pv.prune(tmp_path, {2} | set(range(5, 15)), 14)
    assert sorted(int(p.name.split("-")[1].split("_")[0]) for p in tmp_path.glob("*_state.pt")) == [2] + list(range(5, 15))
    assert [p.name for p in tmp_path.glob("*_resume.pt")] == ["net_epoch-14_resume.pt"]


def _kept(path, kind="state"):
    return sorted(int(p.name.split("-")[1].split("_")[0]) for p in path.glob(f"net_epoch-*_{kind}.pt"))


def test_window_retention_keeps_the_early_states_the_last_ten_and_nothing_else():
    assert pv.window_epochs(80) == [0, 2, 4, 9, 19, 29, 39, 49, 55, 62, 69] + list(range(70, 80))
    assert pv.window_epochs(16) == [0, 2, 4] + list(range(6, 16))     # 9 falls inside the last ten
    assert pv.window_epochs(3) == [0, 1, 2]


def test_window_retention_keeps_the_newest_resume_epochs_state_outside_the_window(tmp_path):
    """The driver's order (state, resume, prune) through epoch 44 of 80 with the best at
    40: the early states so far, the best, and the restart point at epoch 44."""
    keep = lambda best: {best} | set(pv.window_epochs(80))
    for e in range(45):
        (tmp_path / f"net_epoch-{e}_state.pt").write_text("s")
        (tmp_path / f"net_epoch-{e}_resume.pt").write_text("r")
        pv.prune(tmp_path, keep(min(e, 40)), e)
    assert _kept(tmp_path) == [0, 2, 4, 9, 19, 29, 39, 40, 44]
    assert _kept(tmp_path, "resume") == [44]
    assert pv.latest_complete_epoch(tmp_path) == 44


def test_a_window_run_killed_outside_the_window_resumes_there(data, tmp_path, monkeypatch, capsys):
    """Killed during epoch 4 of 16 (window: best epoch 0, early epochs 0, 2, 4 and epochs
    6-15), after epoch 3's prune: epoch 2's state is kept, epoch 1's is not, and the
    restart resumes after epoch 3, not from scratch. The kept set is in the recipe, so a
    restart under another one halts."""
    monkeypatch.setattr(pv.Objective, "selection", lambda self, val: ("val.acc", 0.0))  # best stays 0
    monkeypatch.setattr(pv, "_GRAD_DIAG", False)
    real = pv.torch_save

    def killed(obj, path):
        if path.name == "net_epoch-4_state.pt":
            raise KeyboardInterrupt("killed during epoch 4")
        real(obj, path)
    out = tmp_path / "w"
    args = _args(data, out, epochs=16, extra=["--keep-checkpoints", "window"])
    monkeypatch.setattr(pv, "torch_save", killed)
    with pytest.raises(KeyboardInterrupt):
        pv.main(args)
    assert _kept(out) == [0, 2, 3] and _kept(out, "resume") == [3]
    assert pv.latest_complete_epoch(out) == 3
    rec = json.loads((out / "recipe.json").read_text())
    assert rec["keep_checkpoints"] == "window" and rec["keep_epochs"] == pv.window_epochs(16)

    early = pv.EARLY_KEEP
    monkeypatch.setattr(pv, "torch_save", real)
    monkeypatch.setattr(pv, "EARLY_KEEP", (0, 1))
    assert pv.main(args) == pv.EXIT_HALT
    monkeypatch.setattr(pv, "EARLY_KEEP", early)

    def stop_after_epoch_4(obj, path):
        real(obj, path)
        if path.name == "net_epoch-4_resume.pt":
            raise KeyboardInterrupt("stop")
    monkeypatch.setattr(pv, "torch_save", stop_after_epoch_4)
    capsys.readouterr()
    with pytest.raises(KeyboardInterrupt):
        pv.main(args)
    log = capsys.readouterr().out
    assert "resumed after epoch 3" in log and "fresh start" not in log
    assert pv.latest_complete_epoch(out) == 4


def test_states_retention_keeps_every_state_and_the_newest_resume(tmp_path):
    for e in range(5):
        (tmp_path / f"net_epoch-{e}_state.pt").write_text("s")
        (tmp_path / f"net_epoch-{e}_resume.pt").write_text("r")
    pv.prune(tmp_path, None, 4)
    assert len(list(tmp_path.glob("*_state.pt"))) == 5
    assert [p.name for p in tmp_path.glob("*_resume.pt")] == ["net_epoch-4_resume.pt"]


def test_the_mass_path_runs(data, tmp_path, monkeypatch):
    """The run, then its epoch-1 gradient diagnostic recomputed by hand: lambda = 5 times
    the mean log-cosh of the mass node against its target over the matched jets, and the
    norm of the summed loss's gradient, |g1 + g2|."""
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)               # two pieces of the batch size
    monkeypatch.setattr(pv, "DIAG_JETS", 64)
    mass = ["--network-config", str(MTX / "ParT_sophon_arch_mass.py"), "--mass-lambda", "5.0"]
    out = tmp_path / "m"
    assert pv.main(_args(data, out, arm="L162_MASS", k=162, epochs=2, extra=mass)) == 0
    m = _epochs(out, "metrics")[1]
    assert math.isfinite(m["val"]["loss_reg"]) and math.isfinite(m["train"]["loss_reg"])
    assert m["selection"]["metric"] == "val.acc"
    g = m["grad_diag"]
    assert set(g["grad_norm"]) == {"loss_cls", "lambda_loss_reg", "loss"} and -1 <= g["cosine"] <= 1
    assert g["value"]["loss"] == pytest.approx(g["value"]["loss_cls"] + g["value"]["lambda_loss_reg"])
    assert all(v > 0 and math.isfinite(v) for v in g["grad_norm"].values())

    def terms(o, y):
        d = o[:, 162].double() - y["mass_target"].double()
        valid = y["mass_valid"].bool()
        return {"loss_cls": torch.nn.functional.cross_entropy(o[:, :162], y["truth_label"].long()),
                "lambda_loss_reg": 5.0 * (torch.log(torch.cosh(d)) * valid).sum() / valid.sum().clamp(min=1)}
    monkeypatch.setenv("HYBRID_MASS_INSTALLED", "1")
    model = _build(data, "L162_MASS", 162, {"trunk_init": 1, "head_init": 2}, "ParT_sophon_arch_mass.py")
    pieces = pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, "L162_MASS"), mode="val",
                                            batch_size=32), 32, 0)["pieces"]
    vals, grads = _by_hand(model, out, 1, pieces, pv.load_data_config(data["cfg"]["L162_MASS"]).input_names, terms)
    for name in ("loss_cls", "lambda_loss_reg"):
        assert g["value"][name] == pytest.approx(vals[name], rel=1e-5), name
        assert g["grad_norm"][name] == pytest.approx(grads[name].norm().item(), rel=1e-5), name
    g1, g2 = grads.values()
    assert len(pieces) == 2 and g["batch"]["n_jets"] == 64
    assert g["cosine"] == pytest.approx((g1 @ g2 / (g1.norm() * g2.norm())).item(), abs=1e-5)
    assert g["grad_norm"]["loss"] == pytest.approx((g1 + g2).norm().item(), rel=1e-5)
    assert g["value"]["loss"] == pytest.approx(vals["loss_cls"] + vals["lambda_loss_reg"], rel=1e-5)


@needs_0417
def test_the_self_supervised_path_runs(data, tmp_path, monkeypatch):
    """The run, then its epoch-1 gradient diagnostic recomputed by hand: the masks drawn by
    torch seeded once with the recorded fixed seed (the run's dropout seed, tag grad-diag,
    epoch 0) before the pieces, the L1 and identity terms, and the norm of the summed loss's
    gradient, |g1 + g2|."""
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)               # two pieces of the batch size
    monkeypatch.setattr(pv, "DIAG_JETS", 64)
    ssl = ["--network-config", str(MTX / "ParT_sophon_arch_mpm.py"), "--mpm"]
    out2 = tmp_path / "s"
    assert pv.main(_args(data, out2, arm="L188", k=0, epochs=2, extra=ssl)) == 0
    s = _epochs(out2, "metrics")
    assert s[1]["selection"]["metric"] == "-val.loss" and math.isfinite(s[1]["val"]["loss"])
    # the self-supervised stream is the supervised stream
    assert _epochs(out2, "stream")[0]["n_jets"] == 96
    g = s[1]["grad_diag"]
    assert set(g["grad_norm"]) == {"loss_l1", "id_weight_loss_ce", "loss"} and -1 <= g["cosine"] <= 1
    win = json.loads((out2 / "best_window_epoch.json").read_text())
    losses = [s[e]["val"]["loss"] for e in (0, 1)]
    assert win["metric"] == "-val.loss" and win["epoch"] == losses.index(min(losses))
    assert (out2 / "net_wavg0-1_state.pt").exists() and (out2 / "DONE").exists()

    from src.utils.reproducibility import derive_all
    seed = pv.epoch_seed(derive_all(3)["dropout"], "grad-diag", 0)
    assert g["batch"]["torch_seed"] == s[0]["grad_diag"]["batch"]["torch_seed"] == seed
    monkeypatch.setenv("MPM_INSTALLED", "1")
    from mpm import DEFAULT_ID_WEIGHT, DEFAULT_MASK_RATE
    model = _build(data, "L188", 188, {"trunk_init": 1, "head_init": 2}, "ParT_sophon_arch_mpm.py",
                   {"mask_rate": DEFAULT_MASK_RATE})
    pieces = pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, "L188"), mode="val",
                                            batch_size=32), 32, seed)["pieces"]
    assert len(pieces) == 2 and g["batch"]["n_jets"] == 64

    def terms(o, y):
        pred_cont, pred_id, tgt_cont, tgt_id = o
        return {"loss_l1": (pred_cont.float() - tgt_cont.float()).abs().mean(),
                "id_weight_loss_ce": DEFAULT_ID_WEIGHT * torch.nn.functional.cross_entropy(pred_id.float(), tgt_id)}
    vals, grads = _by_hand(model, out2, 1, pieces, pv.load_data_config(data["cfg"]["L188"]).input_names, terms,
                           skip="decoder.", torch_seed=seed)
    for name in ("loss_l1", "id_weight_loss_ce"):
        assert g["value"][name] == pytest.approx(vals[name], rel=1e-5), name
        assert g["grad_norm"][name] == pytest.approx(grads[name].norm().item(), rel=1e-5), name
    g1, g2 = grads.values()
    assert g["grad_norm"]["loss"] == pytest.approx((g1 + g2).norm().item(), rel=1e-5)
    # the two gradients are close to orthogonal here: the check above still tells |g1 + g2| from |g1 - g2|
    assert abs((g1 + g2).norm() - (g1 - g2).norm()) > 1e-4 * (g1 + g2).norm()
    assert g["cosine"] == pytest.approx((g1 @ g2 / (g1.norm() * g2.norm())).item(), abs=1e-5)


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
    e0 = dry["epochs"][0]
    assert sum(sum(c[3:6]) for c in e0["per_fetch"]) == 96                      # family counts per fetch
    assert e0["last20"]["native_counts"] == m[0]["train"]["native_counts_last20"]
    assert e0["distinct_jets"] <= 96 and e0["copy_factor"] >= 1
    for key in ("epoch", "last20"):
        assert set(dry["summary"][key]) == {"two-prong", "three/four-prong", "QCD"}
    assert dry["summary"]["per_fetch"]["fetches"] >= 0


def test_a_held_out_family_run_reads_a_third_and_the_dry_run_draws_its_rows(data, tmp_path):
    """A LOFO arm: --extra-selection with --data-windows 3; recorded as fraction 1/3,
    and the loader dry run (same flags) draws the training rows, none from two
    fetches, none of the excluded labels."""
    import loader_dryrun
    sel = "~((jet_label >= 100) & (jet_label < 161))"
    out = tmp_path / "lofo"
    assert pv.main(_args(data, out, epochs=3, extra=["--extra-selection", sel, "--data-windows", "3"])) == 0
    rec = json.loads((out / "recipe.json").read_text())
    assert rec["data_windows"] == 3 and rec["data_fraction"] == 1 / 3 and rec["extra_selection"] == sel
    dry = tmp_path / "dry.json"
    assert loader_dryrun.main([
        "--seed", "3", "--epochs", "3", "--samples-per-epoch", "96", "--batch-size", "32",
        "--num-workers", "2", "--data-split-num", "4", "--check-jets", "64", "--data-windows", "3",
        "--extra-selection", sel, "--data-config", data["cfg"]["R16_Q1"], "--out", str(dry),
        "--data-train", *[f"{fam}:{p}" for fam, ps in data["train"].items() for p in ps]]) == 0
    d = json.loads(dry.read_text())
    assert [e["sha256"] for e in d["epochs"]] == [r["sha256"] for r in _epochs(out, "stream").values()]
    assert d["summary"]["rows_in_two_fetches"] == 0
    assert set(range(100, 161)) <= set(d["summary"]["labels_absent_every_epoch"])
    assert all(e["max_fetch_id"] < e["fetches_per_pass"] for e in d["epochs"])


def test_the_run_ends_with_the_weight_average_in_the_fine_tuning_format(run_a):
    rec = json.loads((run_a / "net_wavg0-2.json").read_text())
    state = run_a / "net_wavg0-2_state.pt"
    assert rec["inputs"] == {str(e): pv.sha256_file(run_a / f"net_epoch-{e}_state.pt") for e in range(3)}
    assert rec["sha256"] == pv.sha256_file(state) and (run_a / "DONE").exists()
    bn = rec["bn_recompute"]
    assert bn["n_jets"] == min(pv.BN_JETS, bn["n_jets"]) > 0 and bn["batchnorm_layers"] > 0 and bn["files"]
    w = torch.load(state)
    ins = [torch.load(run_a / f"net_epoch-{e}_state.pt") for e in range(3)]
    k = "mod.fc.1.weight"
    torch.testing.assert_close(w[k], sum(s[k] for s in ins) / 3)
    bnk = [k for k in w if k.endswith("running_mean")]
    assert bnk and not all(torch.equal(w[k], sum(s[k] for s in ins) / 3) for k in bnk)   # recomputed


def test_the_fine_tuning_reader_accepts_the_weight_average(tmp_path):
    """Round trip through experiments/FT/ft_v2.py resolve_wavg with an 80-epoch run."""
    sys.path.insert(0, str(ROOT / "experiments" / "FT"))
    ft = pytest.importorskip("ft_v2")
    m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    for e in range(80):
        with torch.no_grad():
            for p_ in m.parameters():
                p_.add_(0.01)
        torch.save(m.state_dict(), tmp_path / f"net_epoch-{e}_state.pt")

    class A:
        num_epochs = 80

    class DS:
        file_index = {"f.parquet": 0}

        def set_epoch(self, e):
            pass

    import pretrain_v2
    batches = [({"x": torch.randn(64, 3)}, {}, {"_rowid": torch.arange(64), "_jet_label": torch.zeros(64, dtype=torch.long)})] * 4

    class Loader:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter(batches)
    import torch.utils.data as tud
    real = tud.DataLoader
    tud.DataLoader = Loader
    try:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
        pretrain_v2.write_weight_average(tmp_path, A, model, DS(), {"dropout": 1, "data_sampling": 2},
                                         torch.device("cpu"), False, {}, ["x"])
    finally:
        tud.DataLoader = real
    (tmp_path / "DONE").write_text("{}")
    rec = ft.resolve_wavg(tmp_path, {})
    assert rec["epoch"] == "wavg70-79" and rec["sha256"] == pretrain_v2.sha256_file(tmp_path / "net_wavg70-79_state.pt")
    assert sorted(rec["inputs"]) == [str(e) for e in range(70, 80)]


def test_the_fine_tuning_reader_finds_the_best_epoch_on_the_reweighted_accuracy(run_a):
    sys.path.insert(0, str(ROOT / "experiments" / "FT"))
    ft = pytest.importorskip("ft_v2")
    rec = ft.resolve(run_a, "bestval", n_epochs=3)
    vals = {e: r["val"]["acc"] for e, r in _epochs(run_a, "metrics").items()}
    assert rec["epoch"] == max(vals, key=vals.get) and rec["metric"] == "val.acc"
    assert rec["sha256"] == pv.sha256_file(
        run_a / f"net_epoch-{rec['epoch']}_state.pt")


# ------------------------------------------------------------------ failures, the window checkpoint, the gradient diagnostic
def test_a_non_finite_loss_fails_the_attempt_for_a_retry_before_its_epoch_writes(data, tmp_path, monkeypatch, capsys):
    """A NaN loss returns EXIT_RETRY naming the epoch and step, caught mid-epoch at a
    --log-every boundary or at the epoch end, before any file of that epoch is written;
    the retry resumes after the last complete epoch."""
    monkeypatch.setattr(pv, "_GRAD_DIAG", False)
    monkeypatch.setattr(pv, "BN_JETS", 64)
    real = pv.Objective.train_loss

    def nan_at(call):
        calls = []

        def train_loss(self, model, inputs, y, dev):
            loss, parts, corr = real(self, model, inputs, y, dev)
            calls.append(1)
            if len(calls) == call:
                loss = loss * float("nan")
                parts = {k: v * float("nan") for k, v in parts.items()}
            return loss, parts, corr
        return train_loss

    out = tmp_path / "n1"
    monkeypatch.setattr(pv.Objective, "train_loss", nan_at(2))          # epoch 0, step 2
    assert pv.main(_args(data, out, epochs=2, extra=["--log-every", "1"])) == pv.EXIT_RETRY
    assert "FATAL: non-finite training loss at epoch 0 step 2:" in capsys.readouterr().out
    assert not list(out.glob("net_epoch-*")) and not (out / "metrics").exists() and not (out / "stream").exists()

    out = tmp_path / "n2"
    monkeypatch.setattr(pv.Objective, "train_loss", nan_at(5))          # epoch 1, step 2: seen at its end
    args = _args(data, out, epochs=2)
    assert pv.main(args) == pv.EXIT_RETRY
    assert "FATAL: non-finite training loss at epoch 1 step 3:" in capsys.readouterr().out
    assert sorted(p.name for p in (out / "metrics").iterdir()) == ["epoch-000.json"]
    assert pv.latest_complete_epoch(out) == 0 and not (out / "net_epoch-1_state.pt").exists()

    monkeypatch.setattr(pv.Objective, "train_loss", real)                # the retry
    assert pv.main(args) == 0
    assert "resumed after epoch 0" in capsys.readouterr().out and (out / "DONE").exists()


def test_a_run_asked_for_cuda_without_a_usable_gpu_fails_before_writing(data, tmp_path, monkeypatch, capsys):
    """--device cuda never falls back to the CPU: no GPU, or a device that fails to
    start, returns EXIT_NO_GPU (not EXIT_HALT) with nothing written to the run directory."""
    out = tmp_path / "g"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pv.main(_args(data, out, extra=["--device", "cuda"])) == pv.EXIT_NO_GPU != pv.EXIT_HALT
    assert "FATAL: --device cuda and no usable GPU" in capsys.readouterr().out and not out.exists()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)       # the device itself fails to start
    assert pv.main(_args(data, out, extra=["--device", "cuda:999"])) == pv.EXIT_NO_GPU
    assert not out.exists()


def test_the_window_checkpoint_is_the_first_maximum_within_the_last_ten_epochs(tmp_path):
    vals = {e: 0.5 + 0.001 * e for e in range(80)}
    vals[40] = 0.99                         # the global best lies before the window
    vals[72] = vals[75] = 0.9               # a tie inside it: the first
    for e, v in vals.items():
        pv.write_json(tmp_path / "metrics" / f"epoch-{e:03d}.json",
                      {"epoch": e, "selection": {"metric": "val.acc", "value": v}})
    rec = pv.write_window_best(tmp_path, 80, {"driver": "x"})
    assert json.loads((tmp_path / "best_window_epoch.json").read_text()) == rec
    assert rec["epoch"] == 72 and rec["value"] == 0.9 and rec["window"] == [70, 79] and rec["metric"] == "val.acc"
    assert rec["values"] == {str(e): vals[e] for e in range(70, 80)} and rec["code"] == {"driver": "x"}


def _diag_direct(data, arm, extra_selection=None):
    """The diagnostic batch computed here from the validation sample: the rows at positions
    0, DIAG_STRIDE, 2 DIAG_STRIDE, ... of the sample in its fixed order, accepted by
    stream_v2.reweight_indices with a generator seeded DIAG_SEED, which then permutes them,
    and the first DIAG_JETS."""
    ds = sv.StreamDataset({"_": data["val"]}, _dc(data, arm), mode="val", batch_size=32,
                          extra_selection=extra_selection)
    batches = list(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0))
    X = {k: torch.cat([b[0][k] for b in batches]) for k in batches[0][0]}
    Z = {k: np.concatenate([b[2][k].numpy() for b in batches]) for k in ("_rowid", "_jet_label", "_weight")}
    cand = np.arange(0, len(Z["_rowid"]), pv.DIAG_STRIDE)
    rng = np.random.default_rng(pv.DIAG_SEED)
    acc = sv.reweight_indices(Z["_weight"][cand], rng)
    rng.shuffle(acc)
    rows = cand[acc[:pv.DIAG_JETS]]
    return {"rowid": Z["_rowid"][rows], "label": Z["_jet_label"][rows], "X": {k: v[rows] for k, v in X.items()},
            "candidates": len(cand), "accepted": len(acc), "first_labels": Z["_jet_label"][:pv.DIAG_JETS]}


def test_the_run_records_its_window_checkpoint_and_gradient_diagnostic(data, run_a):
    """A 3-epoch run: the window is its last ten epochs, all three. The diagnostic batch
    is the one computed here from the validation sample (_diag_direct; at the default stride
    of 64 a few jets of this small sample); with --log-every 0 epoch 0's file holds the
    initial point only."""
    win = json.loads((run_a / "best_window_epoch.json").read_text())
    m = _epochs(run_a, "metrics")
    sel = {e: r["selection"]["value"] for e, r in m.items()}
    assert win["window"] == [0, 2] and win["epoch"] == max(sel, key=sel.get)
    assert win["code"]["driver"] == "experiments/MTX/pretrain_v2.py" and set(win["code"]) >= {"repo_ref", "commit"}
    ref = _diag_direct(data, "R16_Q1")
    for e in range(3):
        g = m[e]["grad_diag"]
        assert set(g["grad_norm"]) == {"loss_cls"} and g["grad_norm"]["loss_cls"] > 0 and "cosine" not in g
        assert g["batch"]["n_jets"] == len(ref["rowid"]) > 0 and g["batch"]["piece_jets"] == 32
        assert g["batch"]["rows_sha256"] == hashlib.sha256(ref["rowid"].astype("<i8").tobytes()).hexdigest()
        assert (g["batch"]["stride"], g["batch"]["seed"]) == (64, pv.DIAG_SEED)
    d0 = json.loads((run_a / "metrics" / "grad_diag-000.json").read_text())
    assert [p["step"] for p in d0["points"]] == [0] and d0["batch"] == m[0]["grad_diag"]["batch"]
    assert not (run_a / "metrics" / "grad_diag-001.json").exists()


def test_the_diagnostic_batch_is_built_once_and_reloaded_on_a_restart(tmp_path, monkeypatch):
    """The batch is saved at the first start, so a restart does not read the whole
    validation sample again; a saved batch built otherwise halts the start."""
    calls = []

    def build(ds, piece, torch_seed):
        calls.append(piece)
        return {"pieces": [({"x": torch.arange(6.0).reshape(2, 3)}, {"y": torch.zeros(2)})],
                "definition": {"stride": pv.DIAG_STRIDE, "seed": pv.DIAG_SEED, "piece_jets": piece,
                               "torch_seed": torch_seed, "extra_selection": ds.extra_selection,
                               "rows_sha256": "r"}}
    monkeypatch.setattr(pv, "diag_batch", build)
    ds = type("DS", (), {"extra_selection": None})()
    first = pv.load_or_build_diag(tmp_path, ds, 32, 7)
    again = pv.load_or_build_diag(tmp_path, ds, 32, 7)
    assert calls == [32] and (tmp_path / pv.DIAG_FILE).exists() and _same(first, again)
    for piece, seed in ((64, 7), (32, 8)):          # another piece size or torch seed
        with pytest.raises(SystemExit) as e:
            pv.load_or_build_diag(tmp_path, ds, piece, seed)
        assert e.value.code == pv.EXIT_HALT
    assert calls == [32]


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


@pytest.mark.parametrize("objective", ["classification", pytest.param("mpm", marks=needs_0417)])
def test_the_gradient_diagnostic_leaves_the_run_bitwise_unchanged(data, tmp_path, monkeypatch, objective):
    """Two short epochs with the diagnostic at initialisation, after every step and at
    each epoch end, against the same run without it: the same weights, optimizer state
    (slow weights and counter included), scheduler, trimmer counters and streams, bit
    for bit. One CPU thread, so the arithmetic itself is reproducible. The recorded
    classification value is then recomputed by hand from the saved epoch."""
    monkeypatch.setattr(pv, "DIAG_JETS", 64)                # two pieces of the batch size
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)
    monkeypatch.setattr(pv, "BN_JETS", 64)
    arm, k, extra = "R16_Q1", 17, ["--log-every", "1"]
    if objective == "mpm":
        arm, k = "L188", 0
        extra += ["--network-config", str(MTX / "ParT_sophon_arch_mpm.py"), "--mpm"]
    runs, threads = {}, torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for on in (True, False):
            monkeypatch.setattr(pv, "_GRAD_DIAG", on)
            runs[on] = tmp_path / ("on" if on else "off")
            assert pv.main(_args(data, runs[on], arm=arm, k=k, epochs=2, extra=extra)) == 0
    finally:
        torch.set_num_threads(threads)
    on, off = runs[True], runs[False]
    for e in (0, 1):
        assert _same(torch.load(on / f"net_epoch-{e}_state.pt"), torch.load(off / f"net_epoch-{e}_state.pt"))
        assert _same(*(torch.load(r / f"net_epoch-{e}_resume.pt", weights_only=False) for r in (on, off)))
    assert {e: r["sha256"] for e, r in _epochs(on, "stream").items()} == \
        {e: r["sha256"] for e, r in _epochs(off, "stream").items()}
    steps = {e: [p["step"] for p in json.loads((on / "metrics" / f"grad_diag-{e:03d}.json").read_text())["points"]]
             for e in (0, 1)}
    assert steps == {0: [0, 1, 2, 3], 1: [1, 2, 3]}
    assert "grad_diag" in _epochs(on, "metrics")[1] and "grad_diag" not in _epochs(off, "metrics")[1]
    assert not list((off / "metrics").glob("grad_diag-*"))
    if objective != "classification":
        return
    rec = _epochs(on, "metrics")[1]["grad_diag"]
    dc = pv.load_data_config(data["cfg"][arm])
    batch = pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, arm), mode="val", batch_size=32), 32, 0)
    assert batch["definition"]["rows_sha256"] == rec["batch"]["rows_sha256"] and len(batch["pieces"]) == 2
    model = _build(data, arm, k, {"trunk_init": 1, "head_init": 2})
    vals, grads = _by_hand(model, on, 1, batch["pieces"], dc.input_names, lambda out, y: {
        "loss_cls": torch.nn.functional.cross_entropy(out, y[dc.label_names[0]].long())})
    assert rec["grad_norm"]["loss_cls"] == pytest.approx(grads["loss_cls"].norm().item(), rel=1e-6)
    assert rec["value"]["loss_cls"] == pytest.approx(vals["loss_cls"], rel=1e-6)


def _by_hand(model, out, epoch, pieces, input_names, terms, skip="mod.fc.1.", torch_seed=None):
    """The diagnostic recomputed here from a run's saved epoch: each term's value and its
    gradient over every parameter not named skip* (default the output layer mod.fc.1: the
    class nodes and the mass node; the hidden layer mod.fc.0 is included), averaged over the
    pieces. torch_seed: torch seeded once before the pieces (the self-supervised masks)."""
    model.load_state_dict(torch.load(out / f"net_epoch-{epoch}_state.pt"))
    resume = torch.load(out / f"net_epoch-{epoch}_resume.pt", weights_only=False)
    pv.set_trimmer_counters(model, resume["trimmer_counters"])
    model.eval()
    trunk = [p for n, p in model.named_parameters() if not n.startswith(skip)]
    assert len(trunk) == len(list(model.parameters())) - (2 if skip == "mod.fc.1." else
                                                         len(list(model.get_submodule(skip[:-1]).parameters())))
    vals, grads = {}, {}
    with torch.random.fork_rng(devices=[]):
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        for X, y in pieces:
            for name, t in terms(model(*[X[n] for n in input_names]), y).items():
                g = torch.cat([(x if x is not None else torch.zeros_like(p)).double().flatten()
                               for x, p in zip(torch.autograd.grad(t, trunk, retain_graph=True, allow_unused=True),
                                               trunk)])
                grads[name] = grads.get(name, 0.0) + g / len(pieces)
                vals[name] = vals.get(name, 0.0) + t.item() / len(pieces)
    return vals, grads


@pytest.mark.parametrize("arch", ["classification", "mass", pytest.param("mpm", marks=needs_0417)])
def test_the_diagnostic_differentiates_all_but_the_output_layer_and_the_decoder(data, monkeypatch, arch):
    monkeypatch.setenv("HYBRID_MASS_INSTALLED", "1")
    monkeypatch.setenv("MPM_INSTALLED", "1")
    s = {"trunk_init": 1, "head_init": 2}
    model = {"classification": lambda: _build(data, "R16_Q1", 17, s),
             "mass": lambda: _build(data, "L162_MASS", 162, s, "ParT_sophon_arch_mass.py"),
             "mpm": lambda: _build(data, "L188", 188, s, "ParT_sophon_arch_mpm.py", {"mask_rate": 0.4})}[arch]()
    kept = {id(p) for p in pv.diag_params(model)}
    left_out = {n for n, p in model.named_parameters() if id(p) not in kept}
    if arch == "mpm":
        assert left_out == {n for n, _ in model.named_parameters() if n.startswith("decoder.")} != set()
    else:
        assert left_out == {"mod.fc.1.weight", "mod.fc.1.bias"}
        assert {"mod.fc.0.0.weight", "mod.fc.0.0.bias", "mod.cls_token"} <= \
            {n for n, p in model.named_parameters() if id(p) in kept}


def test_the_diagnostic_batch_has_the_training_class_mix(data, monkeypatch):
    """The sorted validation files start with an all-QCD file, so the first jets of the sample
    (the batch as first built) are all QCD. The batch is instead the one computed here: every
    second row a candidate, weaver's reweighting draw with its own seed, the first 64 of the
    accepted rows permuted. All three families are in it, in the mix that draw gives."""
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)
    monkeypatch.setattr(pv, "DIAG_JETS", 64)
    ref = _diag_direct(data, "R16_Q1")
    assert pathlib.Path(sorted(data["val"])[0]).name.startswith("QCD_") and (ref["first_labels"] >= 161).all()
    b = pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, "R16_Q1"), mode="val", batch_size=32), 32, 0)
    d = b["definition"]
    fam = np.bincount(np.searchsorted((0, 15, 161), ref["label"], side="right") - 1, minlength=3)
    assert d["family_counts"] == {"two-prong": fam[0], "three/four-prong": fam[1], "QCD": fam[2]}
    assert d["qcd_share"] == fam[2] / 64 and 0 < d["qcd_share"] < 1 and (fam > 0).all()
    assert d["rows_sha256"] == hashlib.sha256(ref["rowid"].astype("<i8").tobytes()).hexdigest()
    assert (d["n_jets"], d["candidates"], d["accepted"], d["stride"], d["seed"]) == \
        (64, ref["candidates"], ref["accepted"], 2, pv.DIAG_SEED)
    assert d["files"] == sorted(pathlib.Path(f).name for f in data["val"]) and d["extra_selection"] is None
    assert len(b["pieces"]) == 2
    for k, v in ref["X"].items():
        assert torch.equal(torch.cat([X[k] for X, _ in b["pieces"]]), v), k


def test_the_diagnostic_batch_is_the_same_for_every_vocabulary_and_run_index(data, monkeypatch):
    """Two vocabularies at one run index, another run index and a mass-output model give the
    same jets, inputs and definition (only torch_seed, the masks' seed, follows the run), and
    building the batch draws from none of the global generators. A held-out-family run's
    sample lacks the family: its batch differs and records the selection."""
    import random
    from src.utils.reproducibility import derive_all
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)
    monkeypatch.setattr(pv, "DIAG_JETS", 64)
    states = torch.get_rng_state(), np.random.get_state(), random.getstate()
    b = {(arm, run): pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, arm), mode="val", batch_size=32),
                                   32, pv.epoch_seed(derive_all(run)["dropout"], "grad-diag", 0))
         for arm, run in (("R16_Q1", 1), ("L188", 1), ("R16_Q1", 2), ("L162_MASS", 4))}
    assert torch.equal(states[0], torch.get_rng_state()) and random.getstate() == states[2]
    np_now = np.random.get_state()
    assert np_now[0] == states[1][0] and np.array_equal(np_now[1], states[1][1]) and np_now[2:] == states[1][2:]
    ref = b["R16_Q1", 1]
    for other in b.values():
        assert {k: v for k, v in other["definition"].items() if k != "torch_seed"} == \
            {k: v for k, v in ref["definition"].items() if k != "torch_seed"}
        assert len(other["pieces"]) == len(ref["pieces"]) == 2
        assert all(_same(xa, xb) for (xa, _), (xb, _) in zip(ref["pieces"], other["pieces"]))
    assert ref["definition"]["torch_seed"] != b["R16_Q1", 2]["definition"]["torch_seed"]
    sel = "~((jet_label >= 100) & (jet_label < 161))"
    lofo = pv.diag_batch(sv.StreamDataset({"_": data["val"]}, _dc(data, "R16_Q1"), mode="val", batch_size=32,
                                          extra_selection=sel), 32, 0)["definition"]
    assert lofo["extra_selection"] == sel and lofo["rows_sha256"] != ref["definition"]["rows_sha256"]
    assert lofo["rows_sha256"] == hashlib.sha256(_diag_direct(data, "R16_Q1", sel)["rowid"].astype("<i8")
                                                 .tobytes()).hexdigest()


def test_within_epoch_points_are_recorded_in_epochs_0_to_4_only(data, tmp_path, monkeypatch):
    """Six epochs of three steps with --log-every 3: a point at step 3 of epochs 0-4 (and the
    initial point in epoch 0), none in epoch 5, whose end-of-epoch value is still recorded."""
    monkeypatch.setattr(pv, "DIAG_JETS", 32)
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)
    monkeypatch.setattr(pv, "BN_JETS", 64)
    out = tmp_path / "p"
    assert pv.main(_args(data, out, epochs=6, extra=["--log-every", "3"])) == 0
    steps = {int(p.stem.split("-")[1]): [q["step"] for q in json.loads(p.read_text())["points"]]
             for p in (out / "metrics").glob("grad_diag-*.json")}
    assert steps == {0: [0, 3], 1: [3], 2: [3], 3: [3], 4: [3]}
    assert all("grad_diag" in r for r in _epochs(out, "metrics").values())


def test_the_diagnostic_runs_in_float32_without_autocast_and_restores_use_amp(data, tmp_path, monkeypatch):
    """--use-amp without a GPU: torch's CUDA autocast, which the model opens itself when its
    use_amp flag is on, is replaced by CPU bfloat16 autocast, so the flag acts on the CPU.
    Every training step then runs with the flag on and a bfloat16 loss; every diagnostic call
    in evaluation mode with the flag off, no autocast open and float32 terms; and the step
    after a diagnostic call has the flag on again."""
    real = torch.autocast

    def cpu_bf16(*args, enabled=True, **kw):
        return real("cpu", dtype=torch.bfloat16, enabled=enabled)
    monkeypatch.setattr(torch, "autocast", cpu_bf16)
    monkeypatch.setattr(torch.cuda.amp, "autocast", cpu_bf16)
    monkeypatch.setattr(pv, "DIAG_JETS", 32)
    monkeypatch.setattr(pv, "DIAG_STRIDE", 2)
    monkeypatch.setattr(pv, "BN_JETS", 64)
    train_loss, grad_terms, seen = pv.Objective.train_loss, pv.Objective.grad_terms, []

    def state(where, model, losses):
        seen.append({"where": where, "use_amp": pv.part_of(model).use_amp, "training": model.training,
                     "autocast": torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"),
                     "dtypes": {t.dtype for t in losses}})

    def tl(self, model, inputs, y, dev):
        r = train_loss(self, model, inputs, y, dev)
        state("train", model, [r[0]])
        return r

    def gt(self, model, inputs, y, dev):
        r = grad_terms(self, model, inputs, y, dev)
        state("diag", model, r.values())
        return r
    monkeypatch.setattr(pv.Objective, "train_loss", tl)
    monkeypatch.setattr(pv.Objective, "grad_terms", gt)
    out = tmp_path / "amp"
    assert pv.main(_args(data, out, epochs=1, extra=["--use-amp", "--log-every", "1"])) == 0
    assert json.loads((out / "recipe.json").read_text())["use_amp"] is True
    # one piece per diagnostic call: initial, after each of the three steps, at the epoch end
    assert [r["where"] for r in seen] == ["diag", "train", "diag", "train", "diag", "train", "diag", "diag"]
    for r in seen:
        on = r.pop("where") == "train"
        assert r == {"use_amp": on, "training": on, "autocast": False,
                     "dtypes": {torch.bfloat16 if on else torch.float32}}, (on, r)
