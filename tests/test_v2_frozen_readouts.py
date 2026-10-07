"""The frozen readouts on a v2 cache (experiments/EVAL/extract_v2.py): probe.py and
mass_resolution.py read the pooled embedding (pooled.npy, amendment A14) when asked,
and both label-recovery scripts read only the cache's uniform prefix, as on a v1
cache, never the rows the extraction selected by label.

The cache is written by extract_v2's own run() and write(), from a synthetic stream:
a uniform prefix, then rows of two probe classes (X->bc, X->bq) beyond it, which a
v2 cache keeps over the whole split and a label-recovery fit must not see."""
import importlib.util
import json
import pathlib
import shutil
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
probe = _load("probe", "experiments/EVAL/probe.py")
mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
lr = _load("label_recovery", "experiments/EVAL/label_recovery.py")
lc = _load("label_recovery_curve", "experiments/EVAL/label_recovery_curve.py")
PREFIX = 3_000


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

    class Tap:              # the class token's stand-in: the output layer's input
        def __init__(self, m):
            self.buf = None
            self.h = m.fc.register_forward_pre_hook(lambda _m, i: setattr(self, "buf", i[0]))

        def close(self):
            self.h.remove()

    class Pooled:           # the pooled embedding's: another function of the same jets
        def __init__(self, m):
            self.buf = None
            self.h = m.trunk.register_forward_hook(
                lambda _m, _i, o: setattr(self, "buf", torch.tanh(0.3 * o)))

        def close(self):
            self.h.remove()

    rng = np.random.default_rng(5)
    head = rng.choice([0, 1, 4, 6, 10, 11, 18, 34, 2, 3, 100], PREFIX)
    tail = rng.permutation(np.concatenate([np.full(2_000, 4), np.full(2_000, 6),
                                           rng.choice([2, 3, 100], 2_000)]))
    labels = np.concatenate([head, tail]).astype(np.int64)
    n = labels.size
    obs = {"jet_pt": rng.uniform(300, 700, n), "jet_sdmass": rng.uniform(40, 200, n),
           "jet_eta": rng.uniform(-2, 2, n)}
    shift = rng.normal(0, 0.1, n)
    obs["genjet_sdmass"] = obs["jet_sdmass"] * np.exp(shift)
    X = np.stack([(labels == 4) + (labels % 7) / 7 + rng.normal(0, 0.7, n),
                  shift + rng.normal(0, 0.05, n), rng.normal(size=n)], axis=1).astype(np.float32)
    batches = [({"x": torch.from_numpy(X[i:i + 500])}, labels[i:i + 500],
                {k: v[i:i + 500] for k, v in obs.items()}) for i in range(0, n, 500)]
    anywhere, windowed = xv.probe_feature_rules()
    sel = xv.Selector(anywhere, PREFIX, [], 0, 0, windowed)
    to_in = lambda Xb, need: [Xb["x"][torch.from_numpy(np.flatnonzero(need))]]  # noqa: E731
    res = xv.run(iter(batches), {"best70": Toy().eval()}, sel, "R16_Q1", 17, {}, Tap, to_in,
                 observers=xv.V2_OBSERVERS, pooled_factory=Pooled)
    out = tmp_path_factory.mktemp("v2") / "mtx-r16q1-s9"
    xv.write(out, res, {"prefix_features": PREFIX,
                        "checkpoints": {"best70": {"checkpoint_sha256": "c" * 64}}})
    d = out / "best70"
    assert (d / "pooled.npy").exists() and np.load(d / "rows.npy").size > PREFIX
    return d


def _with_features(src: pathlib.Path, dst: pathlib.Path, F) -> pathlib.Path:
    """`src` with features.npy replaced by F."""
    shutil.copytree(src, dst)
    np.save(dst / "features.npy", F)
    return dst


def _v1_prefix_copy(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """The v1 cache of the same jets: the uniform prefix, float32, extract_manifest.json."""
    dst.mkdir()
    rows = np.load(src / "rows.npy")
    keep = rows < PREFIX
    np.save(dst / "features.npy", np.load(src / "features.npy")[keep].astype(np.float32))
    np.save(dst / "label188.npy", np.load(src / "label188.npy")[keep])
    (dst / "extract_manifest.json").write_text('{"checkpoint_sha256": "%s"}' % ("c" * 64))
    return dst


def _probe(argv):
    old = sys.argv
    sys.argv = ["probe.py"] + argv
    try:
        return probe.main()
    finally:
        sys.argv = old


def _probe_run(d, out, extra=()):
    assert _probe(["--features", f"A={d}", "--out", str(out), "--tasks", "bc_vs_bq",
                   "--split-fractions", "0.2", "0.1", "--bootstrap", "20", *extra]) == 0
    return json.loads((out / "probe_results.json").read_text())


def test_probe_reads_the_pooled_embedding_and_records_the_readout(cache, tmp_path):
    pooled = _probe_run(cache, tmp_path / "pooled", ["--readout", "pooled"])
    same = _probe_run(_with_features(cache, tmp_path / "swapped", np.load(cache / "pooled.npy")),
                      tmp_path / "swapped_out")
    cls = _probe_run(cache, tmp_path / "cls")
    assert pooled["readout"] == "pooled" and same["readout"] == cls["readout"] == "features"
    assert not pooled["tasks"]["bc_vs_bq"].get("skipped")
    # the pooled call is the class-token call on pooled.npy, and not the class token
    assert pooled["tasks"] == same["tasks"]
    a = lambda r: r["tasks"]["bc_vs_bq"]["arms"]["A"]["linear"]["auc"]  # noqa: E731
    assert a(pooled) != a(cls)
    assert pooled["arm_checkpoints"] == cls["arm_checkpoints"] == {"A": "c" * 64}


def test_an_mlp_rerun_never_copies_another_readouts_linear_numbers(cache, tmp_path):
    _probe_run(cache, tmp_path / "cls")
    with pytest.raises(SystemExit, match="readout"):
        _probe(["--features", f"A={cache}", "--out", str(tmp_path / "rerun"), "--readout", "pooled",
                "--mlp-rerun-of", str(tmp_path / "cls" / "probe_results.json")])


def test_mass_resolution_reads_the_pooled_embedding(cache, tmp_path):
    def run(d, out, extra=()):
        assert mr.main(["--features", f"A={d}", "--observers", str(cache), "--out", str(out),
                        *extra]) == 0
        return json.loads((out / "mass_resolution.json").read_text())
    pooled = run(cache, tmp_path / "pooled", ["--readout", "pooled"])
    same = run(_with_features(cache, tmp_path / "swapped", np.load(cache / "pooled.npy")),
               tmp_path / "swapped_out")
    cls = run(cache, tmp_path / "cls")
    assert pooled["readout"] == "pooled" and cls["readout"] == "features"
    for p in ("ridge", "mlp"):
        assert pooled["arms"]["A"][p]["sigma_eff"] == same["arms"]["A"][p]["sigma_eff"]
    assert pooled["arms"]["A"]["ridge"]["sigma_eff"] != cls["arms"]["A"]["ridge"]["sigma_eff"]
    assert pooled["n_jets_total"] == PREFIX


def test_label_recovery_reads_only_the_prefix_of_a_v2_cache(cache, tmp_path):
    def run(d, out, extra=()):
        assert lr.main(["--features", f"A={d}", "--own-rung", "A=R16_Q1", "--out", str(out),
                        "--n", "100000", "--rungs", "R16_Q1", "L188", *extra]) == 0
        return json.loads((out / "label_recovery.json").read_text())
    v2 = run(cache, tmp_path / "v2")
    v1 = run(_v1_prefix_copy(cache, tmp_path / "v1cache"), tmp_path / "v1")
    assert v2["n_used"] == v1["n_used"] == PREFIX < np.load(cache / "rows.npy").size
    assert v2["v2_prefix"] == PREFIX and v1["v2_prefix"] is None and v2["readout"] == "features"
    for rung in ("R16_Q1", "L188"):
        assert v2["arms"]["A"]["rungs"][rung]["linear"] == v1["arms"]["A"]["rungs"][rung]["linear"]
    pooled = run(cache, tmp_path / "pooled", ["--readout", "pooled"])
    assert pooled["readout"] == "pooled" and pooled["n_used"] == PREFIX


def test_label_recovery_curve_reads_only_the_prefix_and_resumes_one_readout(cache, tmp_path):
    def run(d, out, extra=()):
        assert lc.main(["--features", f"A={d}", "--own-rung", "A=none", "--out", str(out),
                        "--sizes", "500", "0", "--rungs", "R16_Q1", "--mlp-rungs", *extra]) == 0
        return json.loads((out / "label_recovery_curve.json").read_text())
    v2 = run(cache, tmp_path / "v2")
    v1 = run(_v1_prefix_copy(cache, tmp_path / "v1cache"), tmp_path / "v1")
    assert v2["n_jets"] == v1["n_jets"] == PREFIX
    assert v2["v2_prefix"] == PREFIX and v1["v2_prefix"] is None and v2["readout"] == "features"
    strip = lambda c: [{k: v for k, v in x.items() if k != "seconds"} for x in c]  # noqa: E731
    assert strip(v2["arms"]["A"]["rungs"]["R16_Q1"]["curve"]) == strip(
        v1["arms"]["A"]["rungs"]["R16_Q1"]["curve"])
    with pytest.raises(SystemExit, match="readout"):
        run(cache, tmp_path / "v2", ["--readout", "pooled"])
    assert run(cache, tmp_path / "pooled", ["--readout", "pooled"])["readout"] == "pooled"
