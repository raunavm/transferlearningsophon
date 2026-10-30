"""extract_v2.py --keep-head-outputs (every row's logits, mass output and observers), the
observer config it reads, and the head jobs scripts/build_head_jobs.py emits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


xv = _load("extract_v2", "experiments/EVAL/extract_v2.py")
HB = _load("build_head_jobs", "scripts/build_head_jobs.py")
SPECS = HB.build()


# ------------------------------------------------------------- extract_v2.py

def _toy_stream(n=1000, k=17, reg=1, observers=xv.HEAD_OBSERVERS):
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self, width, seed):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.trunk = torch.nn.Linear(3, 128)
            self.fc = torch.nn.Linear(128, width)
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
    labels = rng.integers(0, 188, n)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    obs = {o: rng.uniform(20, 500, n).astype(np.float32) for o in observers}
    batches = [({"x": torch.from_numpy(X[i:i + 128])}, labels[i:i + 128],
                {o: v[i:i + 128] for o, v in obs.items()}) for i in range(0, n, 128)]
    models = {"best70": Toy(k + reg, 1).eval(), "init": Toy(k + reg, 2).eval()}
    to_in = lambda Xb, need: [Xb["x"][torch.from_numpy(np.flatnonzero(need))]]  # noqa: E731
    with torch.no_grad():
        out = models["best70"](torch.from_numpy(X)).numpy()
    return batches, models, Tap, to_in, labels, obs, out


def test_keep_head_outputs_keeps_every_row_of_the_prefix(tmp_path):
    k = 17
    batches, models, Tap, to_in, labels, obs, out = _toy_stream(k=k)
    sel = xv.Selector([0, 1], 0, [], 0, 0)
    res = xv.run(iter(batches), models, sel, "R16_Q1", k, {}, Tap, to_in,
                 heads_at={"best70"}, keep_head=True, head_max=900, num_reg=1)
    ho = res["checkpoints"]["best70"]["head_outputs"]
    assert np.array_equal(ho["rows"], np.arange(900))
    assert np.array_equal(ho["label188"], labels[:900].astype(np.int16))
    assert ho["logits"].dtype == np.float16 and ho["logits"].shape == (900, k)
    # float16 keeps about three significant figures; the batches differ from one full pass
    assert np.allclose(ho["logits"].astype(np.float32), out[:900, :k], rtol=2e-3, atol=1e-3)
    assert ho["mass"].dtype == np.float32 and np.allclose(ho["mass"], out[:900, k], rtol=1e-5, atol=1e-5)
    assert res["checkpoints"]["init"]["head_outputs"] is None            # no output layer
    for o in xv.HEAD_OBSERVERS:
        assert np.array_equal(res["head_observers"][o], obs[o][:900])
    # the feature rows are what they are without the option
    plain = xv.run(iter(batches), models, sel, "R16_Q1", k, {}, Tap, to_in, heads_at={"best70"})
    assert plain["checkpoints"]["best70"]["head_outputs"] is None
    for t in models:
        assert np.array_equal(plain["checkpoints"][t]["features"], res["checkpoints"][t]["features"])
        assert np.array_equal(plain["checkpoints"][t]["rows"], np.flatnonzero(np.isin(labels, [0, 1])))

    meta = {"checkpoints": {"best70": {"checkpoint_sha256": "a"}, "init": {"checkpoint_sha256": "b"}}}
    xv.write(tmp_path / "keep", res, meta)
    z = np.load(tmp_path / "keep" / "best70" / "head_outputs.npz")
    assert set(z.files) == {"rows", "label188", "logits", "mass", *xv.HEAD_OBSERVERS}
    assert np.array_equal(z["logits"], ho["logits"]) and np.array_equal(z["genjet_pt"], obs["genjet_pt"][:900])
    man = json.loads((tmp_path / "keep" / "best70" / "manifest.json").read_text())
    assert man["head_outputs"]["n"] == 900 and man["head_outputs"]["mass"] is True
    assert man["head_outputs"]["label188_sha256"] == hashlib.sha256(ho["label188"].tobytes()).hexdigest()
    assert not (tmp_path / "keep" / "init" / "head_outputs.npz").exists()
    assert json.loads((tmp_path / "keep" / "init" / "manifest.json").read_text())["head_outputs"] is None
    xv.write(tmp_path / "plain", plain, meta)                            # the default writes none
    assert not list((tmp_path / "plain").rglob("head_outputs.npz"))


def test_without_a_mass_output_no_mass_column_is_written():
    batches, models, Tap, to_in, _, _, out = _toy_stream(reg=0)
    res = xv.run(iter(batches), models, xv.Selector([], 0, [], 0, 0), "L162", 17, {}, Tap, to_in,
                 heads_at={"best70"}, keep_head=True)
    ho = res["checkpoints"]["best70"]["head_outputs"]
    assert "mass" not in ho and ho["rows"].size == 1000                 # no head_max: every row
    assert np.allclose(ho["logits"].astype(np.float32), out[:, :17], rtol=2e-3, atol=1e-3)


def test_a_stream_without_the_head_observers_or_the_declared_width_is_refused():
    batches, models, Tap, to_in, *_ = _toy_stream(observers=xv.V2_OBSERVERS)
    with pytest.raises(SystemExit, match="keep-head-outputs needs"):
        xv.run(iter(batches), models, xv.Selector([], 0, [], 0, 0), "R16_Q1", 17, {}, Tap, to_in,
               heads_at={"best70"}, keep_head=True, num_reg=1)
    batches, models, Tap, to_in, *_ = _toy_stream(reg=0)
    with pytest.raises(SystemExit, match="outputs, not 17 classes"):
        xv.run(iter(batches), models, xv.Selector([], 0, [], 0, 0), "R16_Q1", 17, {}, Tap, to_in,
               heads_at={"best70"}, keep_head=True, num_reg=1)


def test_the_option_is_wired_into_the_command_line():
    src = (ROOT / "experiments/EVAL/extract_v2.py").read_text()
    assert '"--keep-head-outputs", action="store_true"' in src
    assert "keep_head=a.keep_head_outputs" in src and "head_max=a.max_jets, num_reg=a.num_reg" in src
    # the refusals main() makes before and after the stream (they need weaver to run)
    assert "*(HEAD_OBSERVERS if a.keep_head_outputs else ())]" in src
    assert 'raise SystemExit(f"FATAL: {a.data_config} does not list the observers {absent}")' in src
    assert ("if a.keep_head_outputs and a.max_jets and res[\"n_stream\"] < a.max_jets:\n"
            "        raise SystemExit(") in src


# ------------------------------------------------------------- observer config

def _weights_sha(text: str) -> str:
    return hashlib.sha256(text[re.search(r"^weights:", text, re.M).start():].encode()).hexdigest()


def test_the_observer_config_is_the_mass_config_plus_genjet_pt():
    """Same selection, inputs and labels => same rows; the weights: block byte for byte."""
    mreg = (ROOT / "configs/data/JetClassII_massreg.yaml").read_text()
    obs = (ROOT / HB.CONFIG).read_text()
    a = [l for l in mreg.splitlines() if not l.startswith("#")]
    b = [l for l in obs.splitlines() if not l.startswith("#")]
    assert [l for l in b if l not in a] == ["   - genjet_pt"] and [l for l in a if l not in b] == []
    assert b.index("   - genjet_pt") == b.index("   - genjet_sdmass") + 1
    assert _weights_sha(obs) == _weights_sha(mreg)
    assert hashlib.sha256(mreg.encode()).hexdigest() in obs              # the header names its source
    assert set(xv.HEAD_OBSERVERS) <= set(yaml.safe_load(obs)["observers"])


# ------------------------------------------------------------- the head jobs

def _args(spec: str) -> str:
    return yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_one_job_per_run_the_paper_fine_tunes():
    assert len(HB.RUNS) == 16 and [m for m, *_ in HB.models()] == list(HB.RUNS)
    assert HB.init_models() == ["init-s1", "init-s2", "init-s3"]
    assert sorted(SPECS) == sorted([f"job-head-x-{m.removeprefix('mtx-')}-raunav.yaml" for m in HB.RUNS]
                                   + [f"job-head-mass-{m}-raunav.yaml" for m in HB.init_models()])
    for name, spec in SPECS.items():
        d = yaml.safe_load(spec)
        assert d["metadata"]["name"].endswith("-raunav") and name == f"job-{d['metadata']['name']}.yaml"
        a = _args(spec)
        assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0, name
        assert f'--branch "{HB.PIN}"' in a and '[ "${USED}" -lt 85 ]' in a
        assert d["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [42]
        assert (HB.OUT_DIR / name).read_text() == spec, f"{name} not written as built"


def test_the_extraction_reads_the_probe_test_jets_and_the_analyses_read_its_output():
    lp_test = HB.LP.DATASETS["jc2"][2]["test"]
    for model, run, rung, k, reg in HB.models():
        spec = SPECS[f"job-head-x-{model.removeprefix('mtx-')}-raunav.yaml"]
        a = _args(spec)
        assert rung in ("L188", "L162", "R42_Q1", "R16_Q1") and reg == ("mass" in model)
        assert f"--run-dir {HB.BX.V2_ROOT}/{run} --rung {rung} --num-classes {k} --num-reg {reg}" in a
        assert (f"--checkpoints best70 --data-config {HB.CONFIG} --max-jets {lp_test[0]}" in a
                and "--keep-head-outputs" in a and "--prefix-features 0" in a)
        assert re.search(r"--data-test (.*?) \\\n", a).group(1) == lp_test[2] == HB.BF.test2m_list()
        assert f"OUT={HB.HEAD_ROOT}/{model}/test\n" in a and a.count("--out ${OUT} 2>&1 | tee") == 1
        assert f"FEAT={HB.LP.LP_ROOT}/{model}/jc2/test/best70\n" in a
        assert (f"mass_resolution.py --observers ${{OUT}}/best70 \\\n  --features best70=${{FEAT}}"
                in a and f"--out {HB.HEAD_ROOT}/{model} || halt" in a)
        assert ("--head best70=${OUT}/best70" in a) == bool(reg)
        assert f"mass_sculpting.py --head ${{OUT}}/best70 \\\n  --out {HB.HEAD_ROOT}/{model}/sculpting.json" in a
        # sculpting needs only the head outputs: a mass_resolution failure must not lose it
        assert a.index("mass_sculpting.py") < a.index("mass_resolution.py")
        ex = {e["key"]: e for e in yaml.safe_load(spec)["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]}
        assert ex["nvidia.com/gpu.product"]["values"] == list(HB.LP.NOT_ON)
        assert set(HB.BX.V2_GPU_FAULT_NODES) <= set(ex["kubernetes.io/hostname"]["values"])


def test_the_untrained_trunks_get_the_frozen_feature_mass_regression_on_the_cpu():
    for model in HB.init_models():
        spec = SPECS[f"job-head-mass-{model}-raunav.yaml"]
        d, a = yaml.safe_load(spec), _args(spec)
        assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0, model
        assert f'--branch "{HB.PIN}"' in a and '[ "${USED}" -lt 85 ]' in a
        assert "nvidia.com/gpu" not in spec and "extract_v2.py" not in a and "mass_sculpting" not in a
        assert f"FEAT={HB.LP.LP_ROOT}/{model}/jc2/test/init\n" in a
        assert f"OBS={HB.HEAD_ROOT}/{HB.OBS_MODEL}/test/best70\n" in a and HB.OBS_MODEL in HB.RUNS
        # applied before the observers' head job finished, it stops rather than retrying
        assert '[ -f ${OBS}/manifest.json ] || { echo "FATAL' in a and "exit 42; }" in a
        assert (f"mass_resolution.py --observers ${{OBS}} \\\n  --features init=${{FEAT}} "
                f"--out {HB.HEAD_ROOT}/{model} || halt") in a
        assert (HB.OUT_DIR / f"job-head-mass-{model}-raunav.yaml").read_text() == spec


def test_the_storage_plan_counts_every_kept_column():
    want = sum(2_000_000 * (2 * k + 4 * reg + 8 + 2 + 4 * len(xv.HEAD_OBSERVERS))
               for _m, _r, _g, k, reg in HB.models())
    assert HB.planned_bytes() == want and 5e9 < want < 1e10


def test_the_tag_needs_what_the_working_tree_has():
    for path, text in HB.NEEDED.items():
        assert text in (ROOT / path).read_text(), path
