"""X->bb / X->bs fits per model (experiments/XBS/fit_xbx.py) on synthetic extractions laid out as
experiments/XBS/extract_xbx.py writes them."""
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit_xbx_t", ROOT / "experiments/XBS/fit_xbx.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def _split(d, x, y, msd, ckpt, head=None):
    d.mkdir(parents=True)
    np.save(d / "features.npy", x.astype(np.float16))
    np.save(d / "label188.npy", y.astype(np.int16))
    np.save(d / "rows.npy", np.arange(y.size))
    np.savez(d / "observers.npz", jet_pt=np.full(y.size, 500.0), jet_eta=np.zeros(y.size), jet_sdmass=msd)
    if head is not None:
        np.savez_compressed(d / "head_scores.npz", rows=np.arange(y.size), **head)
    (d / "manifest.json").write_text(json.dumps({
        "n_feature_rows": int(y.size), "n_stream": int(y.size), "has_head": head is not None,
        "checkpoint": str(ckpt), "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}))


W = (150.0, 230.0)


def _model(tmp_path, with_head=True, arm="R16_Q1"):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    ckpt = tmp_path / "net_epoch-75_state.pt"
    torch.save({"mod.fc.0.0.weight": torch.randn(512, 128) * 0.05, "mod.fc.0.0.bias": torch.zeros(512),
                "mod.fc.1.weight": torch.randn(17, 512)}, ckpt)
    md = tmp_path / "xbx" / "mtx-r16q1-s1"
    shift = np.r_[np.full(8, 1.0), np.zeros(120)]

    def jets(n, cls):
        x = rng.normal(0, 1, (n, 128)) + cls * shift
        head = None
        if with_head:
            lq = rng.normal(-1, 0.5, n)                  # the class is in the X->bb node only
            head = {"logp_xbb": (rng.normal(-2, 0.5, n) + 2 * cls).astype(np.float32),
                    "logp_xbq": rng.normal(-2, 0.5, n).astype(np.float32), "logp_qcd": lq.astype(np.float32)}
        return x, head

    for part, n in (("train", 1000), ("val", 400)):
        xs, hs = jets(n, 1)
        xb, hb = jets(n, 0)
        y = np.r_[np.ones(n), np.zeros(n)]
        o = rng.permutation(2 * n)                   # staging shuffles every shard
        h = None if hs is None else {k: np.r_[hs[k], hb[k]][o] for k in hs}
        _split(md / "xbb_jc2" / part / "best70", np.r_[xs, xb][o], y[o], rng.uniform(50, 300, 2 * n), ckpt, h)
    xs, hs = jets(3000, 1)
    _split(md / "xbb_jc2" / "eval" / "best70", xs, np.ones(3000), rng.uniform(50, 300, 3000), ckpt, hs)
    (md / "xbb_jc2" / "eval" / "xbx.json").write_text(json.dumps({"arm": arm}))     # as extract_xbx.py writes it
    xb, hb = jets(20000, 0)
    _split(md / "qcd" / "eval" / "best70", xb, np.zeros(20000), rng.uniform(50, 300, 20000), ckpt, hb)
    return md


def test_every_readout_is_scored_inside_the_window(tmp_path, monkeypatch):
    md = _model(tmp_path)
    monkeypatch.setattr(F.X, "HEAD_SAMPLES", 102_400)
    r = F.fit_signal(md, "xbb_jc2", "best70", W, (1, 2))
    msd_s = np.load(md / "xbb_jc2/eval/best70/observers.npz")["jet_sdmass"]
    msd_b = np.load(md / "qcd/eval/best70/observers.npz")["jet_sdmass"]
    n_s, n_b = int(((msd_s > 150) & (msd_s < 230)).sum()), int(((msd_b > 150) & (msd_b < 230)).sum())
    for key in ("linear", "group_node_xbb", "zero_shot_xbq"):
        assert r[key]["n_s"] == n_s and r[key]["n_b"] == n_b and r[key]["eps_b_star"] == 1e-2
    assert r["n_sig_in_window"] == n_s and r["n_qcd_in_window"] == n_b and r["eps_b_star"] == 1e-2
    assert [h["seed"] for h in r["head"]] == [1, 2] and all(h["n_s"] == n_s for h in r["head"])
    # the stand-in signal is separable in the features and in the X->bb node, not in the X->bq node
    assert r["linear"]["auc"] > 0.9 and r["head"][0]["auc"] > 0.75 and r["group_node_xbb"]["auc"] > 0.9
    assert 0.45 < r["zero_shot_xbq"]["auc"] < 0.55
    assert r["linear"]["converged"] and r["n_train_per_class"] == 1000
    # the head's training length: N per class, its samples and the repetition factor 2.5 M / 2N
    assert r["head_samples"] == 102_400 and r["repetition"] == pytest.approx(102_400 / 2000)


def test_eq4_is_formed_only_where_xbb_is_a_class_of_its_own(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    merged = F.fit_signal(_model(tmp_path / "a"), "xbb_jc2", "best70", W, ())
    assert merged["zero_shot_xbb"] is None and merged["zero_shot_xbb_reason"].startswith("cannot be formed")
    assert "group_node_xbb" in merged
    # at 17 classes X->bq sits in X->bb's node, so the X->bq control scores that same node
    assert merged["nodes"] == {"xbb": 0, "xbq": 0, "xbb_own": False}
    own = F.fit_signal(_model(tmp_path / "b", arm="L188"), "xbb_jc2", "best70", W, ())
    assert own["zero_shot_xbb"]["auc"] > 0.9 and "group_node_xbb" not in own and "zero_shot_xbb_reason" not in own
    assert own["nodes"]["xbb_own"] and own["nodes"]["xbq"] != own["nodes"]["xbb"]


def test_an_untrained_network_has_the_linear_probe_only(tmp_path):
    md = _model(tmp_path, with_head=False)
    r = F.fit_signal(md, "xbb_jc2", "best70", W, (1,))
    assert "linear" in r and "head" not in r and "zero_shot_xbb" not in r


def test_a_checkpoint_that_is_not_the_extractions_is_refused(tmp_path, monkeypatch):
    import torch
    md = _model(tmp_path)
    torch.save({"mod.fc.0.0.weight": torch.zeros(512, 128), "mod.fc.0.0.bias": torch.zeros(512)},
               tmp_path / "net_epoch-75_state.pt")
    with pytest.raises(SystemExit, match="not the checkpoint"):
        F.fit_signal(md, "xbb_jc2", "best70", W, (1,))


def test_a_split_of_the_wrong_class_is_refused(tmp_path):
    md = _model(tmp_path)
    np.save(md / "qcd/eval/best70/label188.npy", np.ones(20000, np.int16))
    with pytest.raises(SystemExit, match="qcd: classes"):
        F.fit_signal(md, "xbb_jc2", "best70", W, (1,))


def test_a_fit_records_window_and_mass_and_never_overwrites(tmp_path, monkeypatch):
    md = _model(tmp_path, with_head=False)
    out = tmp_path / "fits_ihep"
    args = ["model", "--root", str(md.parent), "--model", md.name, "--signals", "xbb_jc2", "--checkpoint", "best70",
            "--seeds", "1", "--window", "93.75", "143.75", "--mx", "125", "--out", str(out)]
    assert F.main(args) == 0
    r = json.loads((out / f"{md.name}.json").read_text())
    assert r["window"] == [93.75, 143.75] and r["mx"] == 125.0 and r["signals"]["xbb_jc2"]["eps_b_star"] == 1e-2
    before = (out / f"{md.name}.json").read_bytes()
    with pytest.raises(SystemExit, match="never overwritten"):
        F.main(args)
    assert (out / f"{md.name}.json").read_bytes() == before
    # the window and the mass are always given, and agree (Sophon's window scaled by mX/200)
    with pytest.raises(SystemExit):
        F.main([a for a in args if a not in ("--window", "93.75", "143.75")])
    mismatch = [a.replace("93.75", "150").replace("143.75", "230") for a in args]
    with pytest.raises(SystemExit, match="Sophon's scaled to mX = 125"):
        F.main([a.replace(str(out), str(tmp_path / "other")) for a in mismatch])
    assert not (tmp_path / "other").exists()


def test_the_from_scratch_model_is_scored_on_its_own_probability_and_records_its_training(tmp_path):
    rng = np.random.default_rng(1)
    for name, cls, n in (("features_sig", 1, 4000), ("features_qcd", 0, 30000)):
        d = tmp_path / name
        d.mkdir()
        z = rng.normal(0, 1, (n, 2))
        z[:, 1] += 2.0 * cls
        np.save(d / "logits.npy", z.astype(np.float32))
        np.save(d / "label188.npy", np.full(n, cls, np.int16))
        np.savez(d / "observers.npz", jet_sdmass=rng.uniform(50, 300, n))
    (tmp_path / "ft_manifest.json").write_text(json.dumps(        # as smoke_checks.py manifest writes it
        {"n_train": "2500000", "samples_per_epoch": "2500000", "epochs": "50"}))
    out = tmp_path / "endpoints.json"
    args = ["scratch", "--sig", str(tmp_path / "features_sig"), "--qcd", str(tmp_path / "features_qcd"),
            "--window", "150", "230", "--mx", "200", "--ft-manifest", str(tmp_path / "ft_manifest.json"),
            "--out", str(out)]
    assert F.main(args) == 0
    r = json.loads(out.read_text())
    assert r["window"] == [150.0, 230.0] and r["mx"] == 200.0 and r["n_b"] < 30000
    from scipy.stats import norm
    assert r["auc"] == pytest.approx(norm.cdf(1.0), abs=0.02)     # logit gap 2, unit noise on both logits
    # Sophon App. B.3's budget: 50 x 2.5 M examples, 1.25 M jets per class
    assert r["training"] == {"n_train_per_class": 1_250_000, "samples_per_epoch": 2_500_000, "epochs": 50,
                             "examples_seen": 125_000_000, "passes": 50.0}
    with pytest.raises(SystemExit, match="never overwritten"):
        F.main(args)
