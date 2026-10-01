"""A v2 cache (experiments/EVAL/extract_v2.py) is usable end to end, without weaver
or a GPU: probe.py runs the windowed |V_cb| task (bc_vs_rest) on it rather than
recording it SKIPPED, and mass_resolution.py regresses the jet mass from it.

Verification 2026-10-01 found write() saved no observers: bc_vs_rest's window could
not be re-applied, and mass_resolution.py had no target and wanted a v1 manifest.
The stream here is synthetic; the selection, storage and both consumers are the
committed code."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
PREFIX = 6_000


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.trunk = torch.nn.Linear(3, 128)
            self.fc = torch.nn.Linear(128, 17)
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

    rng = np.random.default_rng(3)
    qcd = xv.anomaly_classes()[0]
    # the prefix: a uniform mix; after it, the b/c classes the |V_cb| task reads
    # (X->bc signal; X->cs, X->bq, X->bqq, QCD background), most inside its window
    head = rng.choice([0, 1, 4, 5, 6, 70, qcd[0]], PREFIX)
    tail = np.concatenate([np.full(6_000, 4), np.repeat([5, 6, 70], 1_500), np.full(1_500, qcd[3]),
                           rng.choice([0, 1, 2, 3], 3_000)])
    labels = np.concatenate([head, rng.permutation(tail)]).astype(np.int64)
    n = labels.size
    inside = rng.random(n) < 0.9
    obs = {"jet_pt": np.where(inside, 500.0, 300.0) + rng.normal(0, 10, n),
           "jet_sdmass": np.where(inside, 115.0, 60.0) + rng.normal(0, 5, n),
           "jet_eta": rng.uniform(-2.0, 2.0, n),
           "jet_phi": rng.uniform(-3, 3, n)}
    shift = rng.normal(0, 0.1, n)
    obs["genjet_sdmass"] = obs["jet_sdmass"] * np.exp(shift)
    X = np.stack([(labels == 4) + rng.normal(0, 0.7, n), shift + rng.normal(0, 0.05, n),
                  rng.normal(size=n)], axis=1).astype(np.float32)
    batches = [({"x": torch.from_numpy(X[i:i + 500])}, labels[i:i + 500],
                {k: v[i:i + 500] for k, v in obs.items()}) for i in range(0, n, 500)]
    anywhere, windowed = xv.probe_feature_rules()
    sel = xv.Selector(anywhere, PREFIX, [], 0, 0, windowed)
    to_in = lambda Xb, need: [Xb["x"][torch.from_numpy(np.flatnonzero(need))]]
    res = xv.run(iter(batches), {"e079": Toy().eval()}, sel, "R16_Q1", 17, {}, Tap, to_in,
                 observers=xv.V2_OBSERVERS)
    out = tmp_path_factory.mktemp("v2") / "mtx-r16q1-s9"
    xv.write(out, res, {"prefix_features": PREFIX, "checkpoints": {"e079": {"checkpoint_sha256": "c" * 64}}})
    return out / "e079", labels, obs


def test_the_cache_keeps_its_rows_observers_and_the_full_range_bc_classes(cache):
    d, labels, obs = cache
    rows = np.load(d / "rows.npy")
    z = np.load(d / "observers.npz")
    man = json.loads((d / "manifest.json").read_text())
    assert sorted(z.files) == sorted(xv.V2_OBSERVERS) == man["observers"]
    for k in z.files:
        assert np.allclose(z[k], obs[k][rows].astype(np.float32))
    assert man["observers_sha256"] and np.array_equal(rows[:PREFIX], np.arange(PREFIX))
    # X->bc, X->cs, X->bq outside the |V_cb| window too (the single-pair tasks)
    outside = ~((obs["jet_pt"] > 450) & (obs["jet_pt"] < 600))
    for c in xv.full_range_classes():
        want = np.flatnonzero((labels == c) & outside)
        assert want.size and np.isin(want, rows).all()
    # QCD beyond the prefix is kept only inside the window
    q = xv.anomaly_classes()[0][3]
    far = np.flatnonzero((labels == q) & outside & (np.arange(labels.size) >= PREFIX))
    assert far.size and not np.isin(far, rows).any()


def test_probe_runs_the_windowed_vcb_task_on_the_cache(cache, tmp_path):
    d, _, _ = cache
    probe = _load("probe", "experiments/EVAL/probe.py")
    old = sys.argv
    sys.argv = ["probe.py", "--features", f"A={d}", "--out", str(tmp_path / "p"),
                "--tasks", "bc_vs_rest", "--bootstrap", "20"]
    try:
        assert probe.main() == 0
    finally:
        sys.argv = old
    task = json.loads((tmp_path / "p" / "probe_results.json").read_text())["tasks"]["bc_vs_rest"]
    assert not task.get("skipped"), task


def test_mass_resolution_reads_the_cache_and_its_prefix(cache, tmp_path):
    d, labels, _ = cache
    mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
    assert mr.main(["--features", f"A={d}", "--observers", str(d), "--out", str(tmp_path / "m")]) == 0
    res = json.loads((tmp_path / "m" / "mass_resolution.json").read_text())
    assert res["n_jets_total"] == PREFIX
    assert res["row_alignment_sha256"] == res["arms"]["A"]["provenance"]["label188_sha256"]
    assert res["arms"]["A"]["provenance"]["cache"] == "v2"
    assert res["arms"]["A"]["ridge"]["sigma_eff"] < res["arms"]["A"]["target"]["sigma_eff"]


def test_a_cache_without_its_whole_prefix_is_refused(cache, tmp_path):
    d, _, _ = cache
    mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
    bad = tmp_path / "bad"
    bad.mkdir()
    for f in ("features.npy", "label188.npy", "observers.npz"):
        (bad / f).write_bytes((d / f).read_bytes())
    rows = np.load(d / "rows.npy")
    np.save(bad / "rows.npy", np.where(rows == 7, 10 ** 9, rows))
    (bad / "manifest.json").write_text((d / "manifest.json").read_text())
    with pytest.raises(SystemExit):
        mr.v2_prefix(bad)
