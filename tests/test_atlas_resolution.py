"""experiments/EVAL/atlas_resolution.py (the ATLAS jet-mass resolution, PI decision P7) and the job
scripts/build_atlas_res_job.py writes: the statistic on responses of known quantiles, the null
rule, the output schema on synthetic head outputs and features, the check against mass_v3.json,
and the spec."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
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


ar = _load("atlas_resolution", "experiments/EVAL/atlas_resolution.py")
mr = _load("mass_resolution", "experiments/EVAL/mass_resolution.py")
BJ = _load("build_atlas_res_job", "scripts/build_atlas_res_job.py")

Z841 = 0.998575       # the standard normal's 84.1 % quantile: Phi^-1(0.841)


# ------------------------------------------------------------- the statistic

def test_a_gaussian_response_gives_sigma_over_median():
    r = np.random.default_rng(0).normal(2.0, 0.2, 400_000)
    s = ar.summary(r)
    assert s["n"] == r.size and s["median"] == pytest.approx(2.0, abs=2e-3)
    assert s["q159"] == pytest.approx(2.0 - 0.2 * Z841, abs=3e-3)
    assert s["q841"] == pytest.approx(2.0 + 0.2 * Z841, abs=3e-3)
    assert s["resolution"] == pytest.approx(0.1 * Z841, rel=1e-2)
    assert s["resolution"] == pytest.approx(0.5 * (s["q841"] - s["q159"]) / s["median"])


def test_quantiles_are_numpys_linear_ones_at_15_9_and_84_1_percent():
    r = np.arange(1.0, 201.0)                        # 200 jets: linear interpolation is exact here
    s = ar.summary(r)
    assert s["q159"] == pytest.approx(1 + 0.159 * 199) and s["q841"] == pytest.approx(1 + 0.841 * 199)
    assert s["median"] == 100.5
    assert s["resolution"] == pytest.approx(0.5 * (0.841 - 0.159) * 199 / 100.5)
    # the median, not the mean, divides: a skewed sample tells them apart
    sk = np.concatenate([np.ones(150), np.full(50, 10.0)])
    assert ar.summary(sk)["median"] == 1.0


def test_the_bootstrap_sds_match_a_gaussians_asymptotic_quantile_errors():
    # a sample quantile's variance is p(1-p) / (n f(x_p)^2); for a Gaussian the 15.9 / 84.1 %
    # quantiles are uncorrelated with the median, so to first order
    #   var(res) = var(q841 - q159) / (4 mu^2) + res^2 var(median) / mu^2
    mu, sig, n, p = 2.0, 0.2, 20_000, 0.159
    f = math.exp(-Z841 ** 2 / 2) / math.sqrt(2 * math.pi) / sig
    var_d = 2 * p * (1 - 2 * p) / (n * f ** 2)
    var_med = math.pi / 2 * sig ** 2 / n
    res = sig * Z841 / mu
    s = ar.summary(np.random.default_rng(5).normal(mu, sig, n))
    assert s["median_sd"] == pytest.approx(math.sqrt(var_med), rel=0.2)
    assert s["resolution_sd"] == pytest.approx(math.sqrt(var_d / (4 * mu ** 2) + res ** 2 * var_med / mu ** 2),
                                               rel=0.2)
    # seeded: the same jets give the same errors
    r = np.random.default_rng(6).normal(1, 0.1, 500)
    assert ar.bootstrap_sd(r) == ar.bootstrap_sd(r.copy())


def test_the_histogram_counts_every_jet_once():
    r = np.array([-0.1, 0.0, 0.01, 0.999, 1.005, 1.99, 2.0, 2.5, 3.0])
    h = ar.histogram(r)
    assert h["edges"] == ar.HIST_EDGES.tolist() and len(h["counts"]) == len(h["edges"]) - 1 == 100
    assert h["underflow"] == 1 and h["overflow"] == 2
    assert sum(h["counts"]) + h["underflow"] + h["overflow"] == r.size
    assert h["counts"][0] == 2 and h["counts"][49] == 1 and h["counts"][50] == 1 and h["counts"][-1] == 2


def test_fewer_than_100_jets_carry_their_count_and_nulls():
    rng = np.random.default_rng(1)
    assert ar.MIN_JETS == 100
    few = ar.summary(rng.normal(1, 0.1, 99))
    assert few == {"median": None, "q159": None, "q841": None, "resolution": None, "n": 99,
                   "median_sd": None, "resolution_sd": None}
    assert ar.summary(rng.normal(1, 0.1, 100))["resolution"] is not None
    # one bin with 50 jets, one with 300, and 20 jets outside the edges in no bin
    x = np.concatenate([np.full(50, 250.0), np.full(300, 500.0), np.full(20, 3000.0)])
    b = ar.binned(rng.normal(1, 0.1, x.size), x, [200, 400, 700])
    assert b["edges"] == [200, 400, 700] and b["n"] == [50, 300]
    assert b["median"][0] is None and b["resolution"][0] is None and b["resolution"][1] is not None
    assert b["resolution_sd"][0] is None and b["resolution_sd"][1] > 0
    assert set(b) == {"edges", "median", "q159", "q841", "resolution", "n", "median_sd", "resolution_sd"}


def test_the_bins_are_the_earlier_mass_files_bins():
    assert ar.PT_EDGES == [200, 400, 700, 1000, 2500]
    assert ar.MGEN_EDGES == [20, 50, 100, 150, 250, 500]
    assert [(lo, hi) for lo, hi in zip(ar.PT_EDGES, ar.PT_EDGES[1:])] == list(mr.PT_BINS)
    assert [(lo, hi) for lo, hi in zip(ar.MGEN_EDGES, ar.MGEN_EDGES[1:])] == list(mr.MGEN_BINS)
    with pytest.raises(SystemExit, match="not contiguous"):
        ar.edges(((0, 1), (2, 3)))


def test_groups_split_at_the_first_qcd_label():
    rng = np.random.default_rng(2)
    lab = rng.choice([0, 160, 161, 187], 4000)
    r = np.where(lab >= 161, rng.normal(1, 0.3, lab.size), rng.normal(1, 0.1, lab.size))
    blk = ar.response_block(r, lab, rng.uniform(200, 2500, lab.size), rng.uniform(20, 500, lab.size))
    assert set(blk) == {"resonant", "qcd", "all"}
    assert blk["resonant"]["inclusive"]["n"] == int((lab < 161).sum())
    assert blk["qcd"]["inclusive"]["n"] + blk["resonant"]["inclusive"]["n"] == blk["all"]["inclusive"]["n"]
    assert blk["qcd"]["inclusive"]["resolution"] == pytest.approx(0.3 * Z841, rel=0.1)
    assert blk["resonant"]["inclusive"]["resolution"] == pytest.approx(0.1 * Z841, rel=0.1)
    assert sum(blk["all"]["vs_pt"]["n"]) == lab.size


# ------------------------------------------------------------- the script on synthetic data

def _head_dir(d: pathlib.Path, n=6000, seed=3):
    """An extract_v2.py --keep-head-outputs checkpoint: a log mass-ratio per class, 1 % unmatched
    jets (generator mass 0.0), and a mass output that sees through half of it."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    lab = rng.choice([0, 5, 161, 170], n).astype(np.int16)
    sd = rng.uniform(30, 400, n)
    y = rng.normal(np.where(lab >= 161, -0.2, 0.05), 0.1) + rng.exponential(0.3, n) - 0.3
    gen = sd * np.exp(y)
    gen[:60] = 0.0
    np.savez(d / mr.HEAD_FILE, rows=np.arange(n), label188=lab,
             logits=rng.normal(size=(n, 17)).astype(np.float16), mass=(0.5 * y).astype(np.float32),
             jet_pt=rng.uniform(200, 2500, n).astype(np.float32), jet_eta=np.zeros(n, np.float32),
             jet_sdmass=sd.astype(np.float32), genjet_sdmass=gen.astype(np.float32),
             genjet_pt=np.ones(n, np.float32))
    (d / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "ab" * 32}))
    return lab, y


def _feature_cache(d: pathlib.Path, lab: np.ndarray, y: np.ndarray, extra=300):
    """A linear-probe cache: every row of a stream a few jets longer than the observers', with
    features that carry part of the target."""
    d.mkdir(parents=True)
    n = lab.size + extra
    rng = np.random.default_rng(4)
    F = rng.normal(size=(n, 8))
    F[:lab.size, 0] += 3 * y
    np.save(d / "features.npy", F.astype(np.float16))
    np.save(d / "rows.npy", np.arange(n))
    np.save(d / "label188.npy", np.concatenate([lab, np.zeros(extra, np.int16)]))
    (d / "manifest.json").write_text(json.dumps({"prefix_features": 2_000_000_000, "n_stream": n,
                                                 "checkpoint_sha256": "cd" * 32}))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """mass_resolution.py's mass_v3.json (with its residuals) and this script's output on it."""
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    t = tmp_path_factory.mktemp("atlas")
    lab, y = _head_dir(t / "head")
    _feature_cache(t / "feat", lab, y)
    assert mr.main(["--observers", str(t / "head"), "--features", f"best70={t / 'feat'}",
                    "--head", f"best70={t / 'head'}", "--out", str(t / "v3"), "--save-residuals"]) == 0
    argv = ["--model", "mtx-x-s1", "--observers", str(t / "head"), "--features", str(t / "feat"),
            "--head", str(t / "head"), "--mass-v3", str(t / "v3" / mr.OUT_JSON), "--out-dir", str(t / "out")]
    assert ar.main(argv) == 0
    return t, argv, json.loads((t / "out" / "mtx-x-s1.json").read_text())


def test_the_output_has_the_interface_schema(run):
    t, _argv, doc = run
    assert doc["model"] == "mtx-x-s1" and doc["definition"] == ar.DEFINITION
    assert set(doc["estimators"]) == {"groomed", "regressed", "ridge", "mlp"}
    keys = {"median", "q159", "q841", "resolution", "n", "median_sd", "resolution_sd"}
    for est, blk in doc["estimators"].items():
        assert {"resonant", "qcd"} <= set(blk)
        for g, b in blk.items():
            assert set(b) == {"inclusive", "vs_pt", "vs_mgen", "hist"} and set(b["inclusive"]) == keys
            assert b["inclusive"]["n"] >= ar.MIN_JETS and b["inclusive"]["resolution"] > 0
            assert b["inclusive"]["resolution_sd"] > 0 and b["inclusive"]["median_sd"] > 0
            h = b["hist"]
            assert sum(h["counts"]) + h["underflow"] + h["overflow"] == b["inclusive"]["n"]
            for var, e in (("vs_pt", ar.PT_EDGES), ("vs_mgen", ar.MGEN_EDGES)):
                c = b[var]
                assert set(c) == keys | {"edges"} and c["edges"] == e
                assert all(len(c[k]) == len(e) - 1 for k in keys)
                for i, n in enumerate(c["n"]):
                    if n < ar.MIN_JETS:
                        assert all(c[k][i] is None for k in keys - {"n"})
                    else:
                        assert c["resolution"][i] == pytest.approx(
                            0.5 * (c["q841"][i] - c["q159"][i]) / c["median"][i])
    src = pathlib.Path(doc["source"]["file"])
    assert src == t / "out" / "mtx-x-s1_perjet.npz"
    assert doc["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()


def test_every_estimator_is_scored_on_mass_v3s_test_jets_against_the_generator_mass(run):
    t, _argv, doc = run
    z = np.load(t / "out" / "mtx-x-s1_perjet.npz")
    res = np.load(t / "v3" / mr.OUT_RESIDUALS)
    assert np.array_equal(z["rows"], res["rows"]) and np.array_equal(z["label188"], res["label188"])
    h = np.load(t / "head" / mr.HEAD_FILE)
    rows = z["rows"]
    gen, sd = h["genjet_sdmass"][rows].astype(np.float64), h["jet_sdmass"][rows].astype(np.float64)
    assert (gen > 0).all() and np.allclose(z["genjet_sdmass"], gen)
    assert np.allclose(z["m_groomed"], sd)
    assert np.allclose(z["m_regressed"], sd * np.exp(h["mass"][rows].astype(np.float64)))
    # the probes: m_SD exp(prediction), the residual mass_resolution.py stored for them
    for p in ("ridge", "mlp"):
        assert np.allclose(np.log(z[f"m_{p}"] / gen), res[f"{p}|best70"], atol=1e-9)
    # the JSON is the summary of exactly these per-jet values
    qcd = z["label188"] >= 161
    for est in ("groomed", "regressed", "ridge", "mlp"):
        r = z[f"m_{est}"] / z["genjet_sdmass"]
        assert doc["estimators"][est]["qcd"]["inclusive"] == pytest.approx(ar.summary(r[qcd]))
        assert doc["estimators"][est]["resonant"]["inclusive"] == pytest.approx(ar.summary(r[~qcd]))
        assert doc["estimators"][est]["qcd"]["hist"]["counts"] == ar.histogram(r[qcd])["counts"]
    # the mass output sees through half of the target, so it narrows the response
    g, m = doc["estimators"]["groomed"], doc["estimators"]["regressed"]
    assert m["all"]["inclusive"]["resolution"] < g["all"]["inclusive"]["resolution"]


def test_the_overall_values_agree_with_mass_v3(run):
    _t, _argv, doc = run
    chk = doc["check_against_mass_v3"]
    for est in ("groomed", "regressed", "ridge", "mlp"):
        assert chk[est]["stored"] is not None and chk[est]["abs_diff"] <= 1e-6, est
    assert chk["groomed"]["abs_diff"] <= ar.CHECK_TOL and chk["regressed"]["abs_diff"] <= ar.CHECK_TOL
    assert doc["jets"]["row_alignment_sha256"] == doc["inputs"]["features"]["label188_sha256"]


def test_an_existing_output_is_kept_and_other_jets_stop_the_job(run):
    t, argv, _doc = run
    before = (t / "out" / "mtx-x-s1.json").read_bytes()
    assert ar.main(argv) == 0
    assert (t / "out" / "mtx-x-s1.json").read_bytes() == before
    v3 = json.loads((t / "v3" / mr.OUT_JSON).read_text())
    v3["uncorrected"]["sigma68_central"] += 1e-3
    (t / "bad_v3.json").write_text(json.dumps(v3))
    bad = list(argv)
    bad[bad.index("--mass-v3") + 1] = str(t / "bad_v3.json")
    bad[bad.index("--out-dir") + 1] = str(t / "bad_out")
    with pytest.raises(SystemExit) as e:
        ar.main(bad)
    assert e.value.code == 42 and not (t / "bad_out" / "mtx-x-s1.json").exists()


# ------------------------------------------------------------- the job

SPEC = BJ.spec("mtx-s9.99", 5)


def _args(spec: str) -> str:
    return yaml.safe_load(spec)["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_the_models_are_the_mass_tables():
    ms = BJ.models()
    hb = _load("build_head_jobs", "scripts/build_head_jobs.py")
    assert [m for m, *_ in ms] == list(hb.RUNS) + hb.init_models() and len(ms) == 19
    for m, obs, feat, arm, head in ms:
        if m.startswith("init-"):
            assert obs == f"{hb.HEAD_ROOT}/{hb.OBS_MODEL}/test/best70" and arm == "init" and head == ""
            assert feat == f"{hb.LP.LP_ROOT}/{m}/jc2/test/init"
        else:
            assert obs == f"{hb.HEAD_ROOT}/{m}/test/best70" and arm == "best70"
            assert feat == f"{hb.LP.LP_ROOT}/{m}/jc2/test/best70"
            assert (head == obs) == ("mass" in m) and (head == "") == ("mass" not in m)


def test_the_spec_is_one_indexed_cpu_job():
    d = yaml.safe_load(SPEC)
    assert d["kind"] == "Job" and d["metadata"]["name"] == BJ.NAME and BJ.NAME.endswith("-raunav")
    assert "configmap" not in SPEC.lower()
    s = d["spec"]
    assert s["completionMode"] == "Indexed" and s["completions"] == 19 and s["parallelism"] == 5
    assert s["backoffLimitPerIndex"] == 6
    rules = s["podFailurePolicy"]["rules"]
    assert rules[0]["action"] == "FailIndex" and rules[0]["onExitCodes"]["values"] == [42]
    pod = s["template"]["spec"]
    c = pod["containers"][0]
    assert "nvidia.com/gpu" not in SPEC and c["image"] == BJ.LAUNCH.V2_IMAGE
    assert c["resources"]["limits"]["memory"] == BJ.MEM and c["resources"]["requests"]["cpu"] == BJ.CPU
    ex = {e["key"]: e for e in pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"][0]["matchExpressions"]}
    assert ex["topology.kubernetes.io/region"]["values"] == ["us-west"]
    assert ex["kubernetes.io/hostname"]["operator"] == "NotIn"
    assert ex["kubernetes.io/hostname"]["values"] == list(BJ.LAUNCH.V2_BAD_NODES + BJ.LAUNCH.V2_LATER_BAD_NODES)
    assert [v["persistentVolumeClaim"]["claimName"] for v in pod["volumes"]] == ["transfer-learning-vol"]


def test_the_script_is_valid_bash_and_runs_each_model_once():
    a = _args(SPEC)
    assert subprocess.run(["bash", "-n"], input=a, text=True).returncode == 0
    assert "--branch mtx-s9.99 " in a and f"python3 {BJ.SCRIPT} " in a
    for i, (m, obs, feat, arm, head) in enumerate(BJ.models()):
        assert f'  {i}) MODEL={m}; OBS={obs}; FEAT={feat}; ARM={arm}; HEAD="{head}";;\n' in a
    assert "--mass-v3 ${MASSV3} --out-dir ${OUT} || halt" in a and "${HEAD:+--head ${HEAD}}" in a
    assert f"OUT={BJ.OUT_ROOT}\n" in a and "touch ${OUT}/DONE" in a
    # missing inputs stop the index rather than retrying it
    assert '[ -f ${F} ] || { echo "FATAL: no ${F}"; exit 42; }' in a
    assert a.index("touch ${OUT}/DONE") > a.index(BJ.SCRIPT)


def test_the_tag_needs_what_the_working_tree_has():
    for path, text in BJ.NEEDED.items():
        assert text in (ROOT / path).read_text(), path
