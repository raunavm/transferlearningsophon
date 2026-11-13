"""experiments/EVAL/sculpting_boot.py: the paired bootstrap over the test jets of the pretraining heads' mass
sculpting, and the 3-sigma test of each difference (plan 2026-11-08, step 2)."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SB = _load("sculpting_boot", "experiments/EVAL/sculpting_boot.py")
MS = SB.MS


def test_the_bins_are_np_histograms_including_the_edges_and_outside_them():
    msd = np.r_[np.random.default_rng(0).uniform(0, 600, 5000), 20.0, 40.0, 500.0, 19.99, 500.01]
    h = np.bincount(SB.bin_index(msd), minlength=SB.NB + 1)[:SB.NB]
    assert np.array_equal(h, np.histogram(msd, bins=MS.BINS)[0])


def _sample(n_q=20_000, n_r=20_000, seed=1):
    rng = np.random.default_rng(seed)
    lab = np.r_[rng.integers(161, 188, n_q), rng.integers(0, 161, n_r)]
    msd = rng.uniform(20, 500, lab.size)
    return lab, msd, rng


def test_the_fast_path_is_mass_sculpting_on_the_test_set():
    lab, msd, rng = _sample()
    q, r = lab >= 161, lab < 161
    s = 0.5 * rng.uniform(size=lab.size) + 0.5 * msd / 500
    got = SB.point_values(s[q], s[r], SB.bin_index(msd[q]))
    ref = MS.sculpting(s[q], msd[q], s[r])
    assert got["eps_b=0.01"] == (ref["working_points"]["0.01"]["jsd"], ref["working_points"]["0.01"]["n_pass"])
    assert got["eps_b=0.1"] == (ref["working_points"]["0.1"]["jsd"], ref["working_points"]["0.1"]["n_pass"])
    assert got["eps_s=0.5"] == (ref["sig_eff_50"]["jsd"], ref["sig_eff_50"]["n_pass"])


def _stored(runs, msd, lab):
    """What the sculpting_v3 documents hold for these runs (mass_sculpting.sculpting on the test set)."""
    q, r = (lab >= 161) & (lab < 188), lab < 161
    out = {}
    for m, run in runs.items():
        ref = MS.sculpting(run["s"][q], msd[q], run["s"][r])
        out[m] = {"eps_b=0.01": ref["working_points"]["0.01"]["jsd"], "eps_b=0.1": ref["working_points"]["0.1"]["jsd"],
                  "eps_s=0.5": ref["sig_eff_50"]["jsd"]}
    return out


def test_identical_models_differ_by_zero_with_zero_sigma():
    lab, msd, rng = _sample()
    runs = {}
    for k in (1, 2):
        s = rng.uniform(size=lab.size) + 0.2 * msd / 500
        runs[f"mtx-r16q1-s{k}"], runs[f"mtx-l188-s{k}"] = {"s": s}, {"s": s.copy()}
    nominal, _, boot = SB.bootstrap(runs, msd, lab, 30, 0, _stored(runs, msd, lab))
    pairs = SB.pairs_of(list(runs), "r16q1", "l188")
    assert pairs == [("mtx-r16q1-s1", "mtx-l188-s1"), ("mtx-r16q1-s2", "mtx-l188-s2")]
    for wp, *_ in SB.WPS:
        assert np.array_equal(boot["mtx-r16q1-s1"][wp], boot["mtx-l188-s1"][wp])     # the same jets every replica
        assert boot["mtx-r16q1-s1"][wp].std() > 0                                   # but not the same each time
        for q, f in SB.QTY.items():
            d = SB.difference(nominal, boot, pairs, wp, f)
            assert d["delta"] == d["run_sd"] == d["boot_sd"] == d["sigma"] == d["sigma_run_se"] == 0.0
            assert d["z"] is None and d["pass"] is False and d["pass_run_se"] is False and d["n_pairs"] == 2


def test_a_known_shift_is_detected_and_two_alike_configurations_are_not():
    lab, msd, rng = _sample(40_000, 40_000, seed=2)
    runs = {}
    for k in (1, 2, 3):
        runs[f"mtx-r16q1-s{k}"] = {"s": rng.uniform(size=lab.size) + 0.6 * msd / 500}    # sculpts
        runs[f"mtx-l188-s{k}"] = {"s": rng.uniform(size=lab.size) + 0.1 * msd / 500}
        runs[f"mtx-r42q1-s{k}"] = {"s": rng.uniform(size=lab.size) + 0.1 * msd / 500}    # as l188
    nominal, _, boot = SB.bootstrap(runs, msd, lab, 60, 3, _stored(runs, msd, lab))
    for wp in ("eps_b=0.01", "eps_b=0.1"):
        big = SB.difference(nominal, boot, SB.pairs_of(list(runs), "r16q1", "l188"), wp, np.sqrt)
        assert big["delta"] > 0 and big["pass"] and big["z"] > 3 and big["sigma"] >= big["boot_sd"] > 0
        assert big["sigma"] == pytest.approx(np.hypot(big["run_sd"], big["boot_sd"]))      # the run-to-run SD
        assert big["sigma_run_se"] == pytest.approx(np.hypot(big["run_sd"] / np.sqrt(3), big["boot_sd"]))
        assert big["sigma_run_se"] <= big["sigma"] and big["pass_run_se"]
        same = SB.difference(nominal, boot, SB.pairs_of(list(runs), "r42q1", "l188"), wp, np.sqrt)
        assert not same["pass"] and abs(same["z"]) < 3
        assert same["boot_mean"] == pytest.approx(same["delta"], abs=3 * same["boot_sd"])


def test_decision_l3_reads_the_sides_unpaired_over_all_their_runs():
    lab, msd, rng = _sample(20_000, 20_000, seed=4)
    runs = {f"mtx-r16q1-s{k}": {"s": rng.uniform(size=lab.size) + 0.6 * msd / 500} for k in (1, 2, 3)}
    runs |= {f"mtx-r16q1mass-s{k}": {"s": rng.uniform(size=lab.size) + 0.1 * msd / 500} for k in (1, 2)}
    nominal, _, boot = SB.bootstrap(runs, msd, lab, 40, 5, _stored(runs, msd, lab))
    a, b = SB.runs_of(list(runs), "r16q1mass"), SB.runs_of(list(runs), "r16q1")
    assert a == ["mtx-r16q1mass-s1", "mtx-r16q1mass-s2"] and len(b) == 3
    wp = "eps_b=0.1"
    u = SB.unpaired(nominal, boot, a, b, wp, np.sqrt)
    va, vb = (np.sqrt([nominal[m][wp] for m in r]) for r in (a, b))
    assert u["delta"] == pytest.approx(va.mean() - vb.mean()) and u["delta"] < 0
    assert u["sd_config"] == pytest.approx(np.std(va, ddof=1)) and u["sd_reference"] == pytest.approx(np.std(vb, ddof=1))
    assert u["sigma"] == pytest.approx(np.sqrt(u["sd_config"] ** 2 + u["sd_reference"] ** 2 + u["boot_sd"] ** 2))
    assert u["boot_sd"] > 0 and u["pass"]
    one = SB.unpaired(nominal, boot, a[:1], b, wp, np.sqrt)    # one run a side: no SD, no claim
    assert one["sigma"] is None and one["pass"] is False
    assert SB.DIFFS["162-188"] == ("l162", "l188")


def test_replica_zero_must_reproduce_the_stored_documents():
    lab, msd, rng = _sample(5_000, 5_000)
    runs = {"mtx-l188-s1": {"s": rng.uniform(size=lab.size) + 0.2 * msd / 500}}
    stored = _stored(runs, msd, lab)
    SB.bootstrap(runs, msd, lab, 2, 0, stored)
    stored["mtx-l188-s1"]["eps_b=0.1"] += 1e-9
    with pytest.raises(SystemExit, match="FATAL: mtx-l188-s1 eps_b=0.1: replica 0 gives JSD"):
        SB.bootstrap(runs, msd, lab, 2, 0, stored)


def test_a_stored_document_of_other_head_outputs_is_refused(tmp_path):
    f = tmp_path / "mtx-l188-s1_sculpting.json"
    f.write_text(json.dumps({"head_outputs_sha256": "ab" * 32, "working_points": {"0.01": {"jsd": 0.1},
                                                                                 "0.1": {"jsd": 0.2}},
                             "sig_eff_50": {"jsd": 0.3}}))
    assert SB.stored_jsd(f, "ab" * 32) == {"eps_b=0.01": 0.1, "eps_b=0.1": 0.2, "eps_s=0.5": 0.3}
    with pytest.raises(SystemExit, match="is of other head outputs"):
        SB.stored_jsd(f, "cd" * 32)


def _head(d: pathlib.Path, lab, msd, logits, rung="R16_Q1"):
    d.mkdir(parents=True)
    np.savez(d / MS.HEAD_FILE, rows=np.arange(lab.size), label188=lab.astype(np.int16),
             logits=logits.astype(np.float16), jet_sdmass=msd.astype(np.float32))
    (d / "manifest.json").write_text(json.dumps({"rung": rung, "num_classes": logits.shape[1],
                                                 "head_outputs": {"sha256": "01" * 32}}))


def test_runs_on_other_jets_too_few_replicas_and_a_missing_label_set_are_refused(tmp_path):
    lab, msd, rng = _sample(500, 500)
    _head(tmp_path / "mtx-r16q1-s1" / "test" / "best70", lab, msd, rng.normal(size=(lab.size, 17)))
    _head(tmp_path / "mtx-l188-s1" / "test" / "best70", lab[::-1], msd, rng.normal(size=(lab.size, 188)), "L188")
    _head(tmp_path / "mtx-flavf0-s1" / "test" / "best70", lab, msd, rng.normal(size=(lab.size, 17)), "none")
    base = ["--head-root", str(tmp_path), "--sculpting-dir", str(tmp_path), "--out", str(tmp_path / "b.json")]
    with pytest.raises(SystemExit, match="is not scored on the jets"):
        SB.main(base + ["--models", "mtx-r16q1-s1", "mtx-l188-s1"])
    with pytest.raises(SystemExit, match="at least 200"):
        SB.main(base + ["--models", "mtx-r16q1-s1", "--replicas", "50"])
    with pytest.raises(SystemExit, match="needs --label-set"):
        SB.main(base + ["--models", "mtx-flavf0-s1"])
    assert SB.load_run(tmp_path / "mtx-flavf0-s1" / "test" / "best70", "FLAV_F0")["qcd_nodes"] == [16]
    assert not (tmp_path / "b.json").exists()


def test_the_job_runs_every_difference_on_the_sculpting_rerun_runs():
    GD = _load("build_gallery_data_job_sb", "scripts/build_gallery_data_job.py")
    HB = _load("build_head_jobs_sb", "scripts/build_head_jobs.py")
    fn, name, dirs = GD.MODES["sculpt-boot"]
    lines = [ln.strip() for ln in fn().splitlines()]
    assert name == "sculpt-boot-raunav" and len(lines) == 2
    heads = lines[0].removeprefix("for f in ").split(";")[0].split()
    assert heads == ([f"{HB.HEAD_ROOT}/{m}/test/best70/head_outputs.npz" for m in HB.SCULPT_V3]
                     + [f"{HB.SCULPT_V3_DIR}/{m}_sculpting.json" for m in HB.SCULPT_V3]) and "exit 42" in lines[0]
    args = lines[1].split()
    assert args[:2] == ["python3", "experiments/EVAL/sculpting_boot.py"] and GD.SCULPT_BOOT == f"{HB.HEAD_ROOT}/sculpting_boot.json"
    models = args[args.index("--models") + 1:args.index("--label-set")]
    sets = dict(x.split("=") for x in args[args.index("--label-set") + 1:args.index("--replicas")])
    assert '--branch "mtx-s2.86"' in (ROOT / f"experiments/FT/k8s/curves/job-{name}.yaml").read_text()
    assert models == list(HB.SCULPT_V3) and len(models) == 22
    assert sets == {m: a for m, _r, a, _k, _g in HB.flav_models()} and args[args.index("--replicas") + 1] == "200"
    assert args[args.index("--sculpting-dir") + 1] == HB.SCULPT_V3_DIR and args[-2:] == ["--out", GD.SCULPT_BOOT]
    for diff, (a, b) in SB.DIFFS.items():
        assert len(SB.pairs_of(models, a, b)) >= 2, diff
    spec = (ROOT / f"experiments/FT/k8s/curves/job-{name}.yaml").read_text()
    assert f"name: {name}" in spec and "backoffLimit: 1" in spec and '[ "${USED}" -le 85 ]' in spec
    assert "configmap" not in spec.lower() and "nvidia.com/gpu" not in spec and all(ln in spec for ln in lines)
