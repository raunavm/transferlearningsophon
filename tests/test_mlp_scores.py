"""experiments/EVAL/mlp_scores.py and scripts/build_mlp_scores_jobs.py (PI 2026-11-23, LEDGER DL-9, SC-2; PF-1).

The refit calls the MLP exactly as the family it repeats called it, keeps every test jet's score in the npz the
bootstrap reads, and the jobs pin one CPU model with a request equal to the MLP's threads."""
import importlib.util
import json
import pathlib
import types

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MS = _load("mlp_scores", "experiments/EVAL/mlp_scores.py")
BJ = _load("build_mlp_scores_jobs", "scripts/build_mlp_scores_jobs.py")
BOOTDATA = pathlib.Path(__file__).resolve().parents[1] / "experiments/FIGS/data/v2/bootstrap"


def _split(n, k, seed):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, 4)).astype(np.float32), r.integers(0, k, n), {}


class Probe:
    MLP_THREADS, MLP_SEEDS, MLP_SCHEDULE = 4, (0, 1, 2), {"max_epochs": 500}

    def __init__(self):
        self.calls = []

    def fit_mlp(self, xtr, ytr, xva, yva, xte, n_classes=2):
        self.calls.append((xtr, ytr, xva, yva, xte, n_classes))
        meta = {"seeds": [{"epochs_run": 3}], "all_converged": True}
        if n_classes == 2:
            return np.linspace(-1, 1, len(xte)).astype(np.float32), meta
        p = np.full((len(xte), n_classes), 1.0 / n_classes, dtype=np.float32)
        p[np.arange(len(xte)), np.arange(len(xte)) % n_classes] += 0.5
        return p / p.sum(1, keepdims=True), meta


def _pb(tmp, k=188):
    splits = {"train_N1000000": _split(50, k, 1), "val": _split(20, k, 2), "test": _split(30, k, 3)}
    lp = types.SimpleNamespace(
        train_sizes=lambda root, ds: [1000, 1000000],
        native_metrics=lambda p, y, present: {"accuracy": float((p.argmax(1) == y).mean())},
        mlp_fit_record=lambda meta: {"converged": meta["all_converged"]},
        binary_score_metrics=lambda s, y, probe, threshold: {"auc": 0.5, "r30": 1.0, "n_jets": int(y.size)},
        load_split=lambda d: _split({"test": 30, "val": 20, "herwig": 25}.get(d.parent.name, 50), 2,
                                    len(str(d))))
    lut = np.arange(188) % 17
    return types.SimpleNamespace(
        lp=lp, NEW_ROOT=tmp, OLD_ROOT=tmp, N_NATIVE=188, load=lambda m, ds, split, ck: splits[split],
        coarse_lut=lambda t: lut, coarse_metrics=lambda p, g: {"accuracy": float((p.argmax(1) == g).mean())}), splits


def test_jc2bal_fits_as_fits_v5_mlp_and_keeps_every_test_jet(tmp_path):
    pb, sp = _pb(tmp_path)
    probe = Probe()
    cells = MS.jc2bal(pb, probe, "mtx-l188-s1", "best70", tmp_path, coarse=False)
    xtr, ytr, xva, yva, xte, k = probe.calls[0]
    assert k == 188 and xtr is sp["train_N1000000"][0] and ytr is sp["train_N1000000"][1] and xte is sp["test"][0]
    z = np.load(cells["jc2bal"]["scores"])
    assert z["p"].dtype == np.float32 and z["p"].shape == (30, 188)
    assert np.array_equal(z["rows"], np.arange(30)) and np.array_equal(z["y"], sp["test"][1])
    assert cells["jc2bal"]["accuracy"] == float((z["p"].argmax(1) == z["y"]).mean())
    assert pathlib.Path(cells["jc2bal"]["scores"]).name == BJ.FILES["jc2bal"][0] + ".npz"


def test_coarse_fits_the_17_merged_groups(tmp_path):
    pb, sp = _pb(tmp_path)
    probe = Probe()
    cells = MS.jc2bal(pb, probe, "mtx-l162-s1", "best70", tmp_path, coarse=True)
    _xtr, ytr, _xva, yva, _xte, k = probe.calls[0]
    lut = np.arange(188) % 17
    assert k == 17 and np.array_equal(ytr, lut[sp["train_N1000000"][1]]) and np.array_equal(yva, lut[sp["val"][1]])
    z = np.load(cells["jc2bal_coarse"]["scores"])
    assert z["p"].shape == (30, 17) and np.array_equal(z["y"], lut[sp["test"][1]])
    assert pathlib.Path(cells["jc2bal_coarse"]["scores"]).name == BJ.FILES["coarse"][0] + ".npz"


def test_bench_scores_herwig_with_the_quark_gluon_probe(tmp_path):
    pb, _ = _pb(tmp_path)
    pb.lp.train_sizes = lambda root, ds: [1000, 1200000] if ds == "top" else [1000, 1600000]
    probe = Probe()
    cells = MS.bench(pb, probe, "mtx-l188-s1", "best70", tmp_path)
    assert [len(c[4]) for c in probe.calls] == [30, 55]          # top test; quark/gluon test + Herwig
    assert sorted(cells) == ["qg", "qg_herwig", "top"]
    assert {pathlib.Path(c["scores"]).name for c in cells.values()} == {f + ".npz" for f in BJ.FILES["bench"]}
    # the linear probes' substitution (CS-8; review of 5f4e1b8, B3(c)), in the folder the bootstrap reads
    assert (cells["top"]["fill"], cells["qg"]["fill"], cells["qg_herwig"]["fill"]) == ("itop4", "nqg", "nqg")
    for k, c in cells.items():
        name = pathlib.Path(c["scores"]).stem
        assert c["scores"].endswith(BJ.npz("bench", "mtx-l188-s1", name).split("/bench/", 1)[1])
    zq, zh = (np.load(cells[k]["scores"]) for k in ("qg", "qg_herwig"))
    s = probe.fit_mlp(*probe.calls[1][:5])[0]
    assert np.array_equal(zq["s"], s[:30]) and np.array_equal(zh["s"], s[30:]) and zq["s"].dtype == np.float32


def test_stored_records_each_metric_beside_the_stored_fit(tmp_path):
    (tmp_path / "mtx-l188-s1.json").write_text(json.dumps({"cells": {"best70": {"jc2bal": {"accuracy": 0.4, "macro_auc_ovr": 0.9}}}}))
    got = MS.stored("jc2bal", "mtx-l188-s1", "best70", {"jc2bal": {"accuracy": 0.4, "macro_auc_ovr": 0.8}}, tmp_path)
    assert got["jc2bal"]["accuracy"] == [0.4, 0.4] and got["jc2bal"]["macro_auc_ovr"] == [0.9, 0.8]
    assert MS.stored("coarse", "mtx-l188-s1", "best70", {}) == {}


def _boot_name(m, ck):
    return m.replace("init-", "initbn-") if ck == "init_bn" else m


def test_rows_are_the_runs_of_the_linear_probe_rows():
    assert len(BJ.ROWS) == len(set(BJ.ROWS)) == 13 + 12 + 33 + 33 + 1 + 6 + 6
    paper = {f"mtx-{r}" for a, rs in BJ.ARMS.items() if a not in BJ.INIT and a != "kin" for r in rs}
    assert len(paper) == 27 and {m for f, m, ck in BJ.ROWS if f == "bench" and ck == "best70"} == paper
    assert {(m, ck) for f, m, ck in BJ.ROWS if f == "bench"} == {(m, ck) for f, m, ck in BJ.ROWS if f == "bench0"}
    assert {(m, ck) for f, m, ck in BJ.ROWS if f == "bench0"} == {(m, "best70") for m in paper} | {
        (f"init-s{k}", ck) for k in (1, 2, 3) for ck in ("init", "init_bn")}
    for fam, row in BJ.SMOKES.items():
        assert row in BJ.fit_rows(fam, skip_smoke=False) and row not in BJ.fit_rows(fam, skip_smoke=True)
    assert sum(len(BJ.fit_rows(f, True)) for f in BJ.FAMILY_ARMS) == len(BJ.ROWS) - len(BJ.SMOKES)


@pytest.mark.parametrize("which", ["fit", "smoke", "boot"])
def test_specs_pin_one_cpu_model_and_request_the_mlp_threads(which):
    text = {"fit": lambda: BJ.fit_spec("mtx-s9.99", "bench", "12Gi", 8, True), "smoke": lambda: BJ.smoke_spec("mtx-s9.99", "12Gi"),
            "boot": lambda: BJ.boot_spec("mtx-s9.99", "4Gi", "jc2")}[which]()
    assert BJ.CPU_MODEL == "AMD EPYC 7402P 24-Core Processor"
    assert f'[ "${{CPU}}" = "{BJ.CPU_MODEL}" ]' in text and "igrok-la.cenic.net" in text
    assert "--branch mtx-s9.99 " in text and f'-le {BJ.LINE} ]' in text and "raunav" in text
    cpu = "1" if which == "boot" else str(Probe.MLP_THREADS)
    assert f'cpu: "{cpu}"' in text
    assert "ConfigMap" not in text and "CronJob" not in text


def test_bootstrap_reads_every_npz_the_fits_write():
    want = {BJ.npz(f, _boot_name(m, ck), n) for f, m, ck in BJ.ROWS if f != "kin" for n in BJ.FILES[f]}
    assert set(BJ.boot_inputs()) == want
    assert BJ.B == 200
    assert set().union(*(BJ.boot_inputs(p) for p in BJ.BOOT_PARTS)) == want
    assert BJ.npz("bench0", "initbn-s2", "top_N1200000").endswith("/bench0/probs/init-s2/init_bn/top_N1200000.npz")
    assert BJ.npz("bench0", "init-s2", "qg_N1600000").endswith("/bench0/probs/init-s2/init/qg_N1600000.npz")
    assert BJ.npz("bench", "mtx-l188-s1", "qg_N1600000").endswith("/bench/probs/mtx-l188-s1/best70.nqg/qg_N1600000.npz")
    assert sorted(o for p in BJ.BOOT_PARTS for o in BJ.boot_outs(p)) == sorted(BJ.BOOT_OUT)
    for part in [p for p in BJ.BOOT_PARTS if p not in ("jc2mass", "bcpair", "eepair")]:
        text = BJ.boot_spec("mtx-s9.99", "4Gi", part)
        assert f"name: bs3-boot-{part}-raunav" in text
        for out in BJ.boot_outs(part):
            assert f"[ -e {out} ] ||" in text
        assert all(f"[ -e {o} ] ||" not in text for o in set(BJ.BOOT_OUT) - set(BJ.boot_outs(part)))
        assert "--diff l162-188=mtx-l162-s1,mtx-l162-s2,mtx-l162-s3/mtx-l188-s1,mtx-l188-s2,mtx-l188-s3" in text


def test_the_analytic_check_reads_the_linear_probes_files():
    text = BJ.analytic_spec("mtx-s9.99", "3Gi")
    assert "experiments/STATS/analytic_check.py" in text and 'cpu: "1"' in text
    assert "fits_v5_fill_probs/mtx-l188-s1/best70.itop4/top_N1200000.npz" in text
    assert "fits_v5_probs/mtx-r16q1-s3/best70/jc2bal_pairs_bc_vs_bq_cs.npz" in text
    for out in BJ.ANALYTIC_OUT:
        assert f"[ -e {out} ] ||" in text
    # the sides of a --diff split at the colon (the paths hold slashes)
    d = text.split("--diff ", 1)[1].split(" ", 1)[0]
    assert d.count(":") == 1 and all(x.endswith(".npz") for side in d.split("=", 1)[1].split(":") for x in side.split(","))


def test_one_job_a_family_and_the_smokes_rows_left_out():
    for fam in BJ.FAMILY_ARMS:
        text = BJ.fit_spec("mtx-s9.99", fam, "6Gi", 8, True)
        assert f"name: bs3-mlp-{fam}-raunav" in text and f"completions: {len(BJ.fit_rows(fam, True))}" in text
        assert f"--family {fam} " in text and all(f"--family {g} " not in text for g in BJ.FAMILY_ARMS if g != fam)
    assert "name: bs3-mlp-smoke-bench-raunav" in BJ.smoke_spec("mtx-s9.99", "8Gi", "bench")


def test_bench0_is_tab_mlps_zero_fill_with_its_checkpoint(tmp_path):
    pb, _ = _pb(tmp_path)
    pb.lp.train_sizes = lambda root, ds: [1000, 1200000] if ds == "top" else [1000, 1600000]
    seen = []
    load = pb.lp.load_split
    pb.lp.load_split = lambda d: seen.append(d) or load(d)
    cells = MS.bench(pb, Probe(), "init-s1", "init_bn", tmp_path / "out", zero=True)
    assert seen and all(d.name == "init_bn" and str(d).startswith(str(tmp_path / "init-s1")) for d in seen)
    assert {c["fill"] for c in cells.values()} == {"zero"}
    assert all("/probs/init-s1/init_bn/" in c["scores"] for c in cells.values())
    assert MS.out_name("init-s1", "init_bn") == "init-s1.init_bn.json"
    assert MS.out_name("mtx-l188-s1", "best70") == "mtx-l188-s1.json"
    assert MS.STORED["bench0"].name == "fits_v3_mlp2" and MS.STORED_BN.name == "fits_v3_mlp2_initbn"


def test_bench0_bootstraps_every_pair_of_table_10_rows():
    outs = BJ.boot_outs("bench0")
    assert len(outs) == 3
    for fam, n, d, flags in outs.values():
        assert fam == "bench0" and len(d) == len(set(d)) == 12 * 11 // 2 and flags == "--headline"
        assert all(a != "l188" for a, b in d)
    text = BJ.boot_spec("mtx-s9.99", "6Gi", "bench0")
    assert "--diff init-initbn=init-s1,init-s2,init-s3/initbn-s1,initbn-s2,initbn-s3" in text
    assert "--diff flavf0-flavf1=" in text


def test_the_17_class_lepton_rows_read_the_stored_sibling_scores():
    text = BJ.ee_spec("mtx-s9.99", "2Gi")
    assert "fits_v5_sib2_probs/mtx-r16q1-s1/best70/jc2bal_pairs_ee_vs_mm.npz" in text
    assert "fits_v5_sib2_probs/init-s3/init_bn/jc2bal_pairs_ee_vs_mm.npz" in text
    assert "--diff r16q1-188=" in text and "--diff r16q1-initbn=" in text and "experiments/EVAL/mlp_scores.py" not in text


def test_the_mass_rows_fit_the_mass_networks_and_bootstrap_every_printed_auc():
    rows = BJ.fit_rows("jc2bal", True, ("l162mass", "r16q1", "r16q1mass"))
    assert sorted(m for _f, m, _c in rows) == sorted(["mtx-l162mass-s1", "mtx-l162mass-s2", "mtx-r16q1-s1", "mtx-r16q1-s2",
                                                      "mtx-r16q1-s3", "mtx-r16q1mass-s1", "mtx-r16q1mass-s2"])
    text = BJ.fit_spec("mtx-s9.99", "jc2bal", "12Gi", 7, True, ("l162mass", "r16q1", "r16q1mass"), "2")
    assert "name: bs3-mlp-jc2bal2-raunav" in text and "completions: 7" in text and "mtx-l188-s1" not in text
    text = BJ.boot_mass_spec("mtx-s9.99", "8Gi")
    assert "name: bs3-boot-jc2mass-raunav" in text and "completions: 4" in text and "parallelism: 3" in text
    assert text.count("--cache-only") == 3 and "--headline --native-groups" in text and BJ.MASS_CACHE in text
    assert "--diff l162mass-l162=mtx-l162mass-s1,mtx-l162mass-s2/mtx-l162-s1,mtx-l162-s2,mtx-l162-s3" in text
    assert "--diff r16q1mass-r16q1=" in text and f"--out {BJ.MASS_OUT}" in text
    assert BJ.MASS_CACHE.startswith("/data/results/bootstrap_cache/")


def test_the_mass_helper_takes_the_cache_pods_later_networks():
    text = BJ.boot_mass_helper_spec("mtx-s9.99", "8Gi")
    assert "name: bs3-boot-jc2mass-helper-raunav" in text and "completions: 7" in text and "parallelism: 7" in text
    assert text.count("--cache-only") == 7 and f"--cache {BJ.MASS_CACHE}" in text and "--diff" not in text
    assert "--model mtx-l162mass-s1=" not in text and "--model mtx-l162-s2=" in text and "--model mtx-r16q1-s3=" in text


def test_the_kinematic_row_fits_top_alone_and_pairs_with_188_draws(tmp_path):
    assert [r for r in BJ.ROWS if r[0] == "kin"] == [("kin", "mtx-l188kin-s1", "best70")]
    pb, _ = _pb(tmp_path)
    pb.lp.train_sizes = lambda root, ds: [1000, 1200000]
    probe = Probe()
    cells = MS.bench(pb, probe, "mtx-l188kin-s1", "best70", tmp_path / "out", zero=True, datasets=("top",))
    assert sorted(cells) == ["top"] and len(probe.calls) == 1 and cells["top"]["fill"] == "zero"
    assert MS.STORED["kin"].name == "fits_v5_mlp"
    text = BJ.boot_kin_spec("mtx-s9.99", "2Gi")
    assert "/bench/probs/mtx-l188-s2/best70.itop4/top_N1200000.npz" in text
    assert "/kin/probs/mtx-l188kin-s1/best70/top_N1200000.npz" in text
    assert "--diff 188draws-kin=mtx-l188-s1,mtx-l188-s2,mtx-l188-s3/mtx-l188kin-s1" in text


def test_cs15b_pairs_each_networks_cell_with_its_probe():
    ins = BJ.cs15b_inputs()
    assert sorted(ins) == sorted(BJ.CS15B_ARMS) and len(BJ.CS15B_ARMS) == 10
    for a, v in ins.items():
        ft, lp = [p for n, p in v if n.startswith("ft-")], [p for n, p in v if n.startswith("lp-")]
        assert len(ft) == len(lp) == len(BJ.ARMS[a])
        assert all("/leg_jc2b/" in p and p.endswith(("/s1", "/s2", "/s3")) or "/N1000000/" in p for p in ft)
        assert all(p.endswith("/best70/jc2bal_N1000000.npz") for p in lp)
    text = BJ.cs15b_spec("mtx-s9.99", "8Gi")
    assert "--task jc2b " in text and "--headline-only" in text and "--diff ft-lp-l188=ft-mtx-l188-s1," in text
    assert "name: bs3-boot-cs15b-raunav" in text


def test_cs8b_and_sc2c_rows():
    assert BJ.npz("bench", "initbn-s2", "top_N1200000").endswith("/bench/probs/init-s2/init_bn.itop4/top_N1200000.npz")
    assert BJ.npz("bench", "init-s1", "qg_herwig_N1600000").endswith("/bench/probs/init-s1/init.nqg/qg_herwig_N1600000.npz")
    for fam, n, d, flags in BJ.boot_outs("benchf").values():
        assert fam == "bench" and len(d) == 66 and all(a != "l188" for a, b in d)
    assert [r for r in BJ.ROWS if r[0] == "bcpair"] == [("bcpair", f"mtx-{a}-s{k}", "best70") for a in ("l188", "r16q1") for k in (1, 2, 3)]
    text = BJ.boot_spec("mtx-s9.99", "4Gi", "bcpair")
    assert "--diff r16q1-188=" in text and "/bcpair/probs/mtx-r16q1-s3/best70/jc2bal_pairs_bc_vs_bq_cs.npz" in text
    assert "--family bench " in BJ.fit_spec("mtx-s9.99", "bench", "7Gi", 6, True, ("init", "initbn"), "2")


def test_bcpair_fits_the_one_pair_as_mlp_cells_does(tmp_path):
    pb, _ = _pb(tmp_path)
    r = np.random.default_rng(5)
    sides = {"signal": np.arange(0, 20), "background": np.arange(20, 50)}
    te = {"signal": np.arange(0, 10), "background": np.arange(10, 30)}
    pb.pair_index = lambda: {"train": {MS.BC_PAIR: sides}, "test": {MS.BC_PAIR: te}}
    pb.load = lambda m, ds, split, ck: (r.normal(size=(60, 4)).astype(np.float32), None, None)
    pb.PAIR_SEED, pb.PAIR_VAL_FRACTION = 20261031, 0.1
    probe = Probe()
    cells = MS.bcpair(pb, probe, "mtx-l188-s1", "best70", tmp_path)
    assert len(probe.calls) == 1 and len(probe.calls[0][4]) == 30
    z = np.load(cells[f"pair_{MS.BC_PAIR}"]["scores"])
    assert np.array_equal(z["y"], np.r_[np.ones(10), np.zeros(20)]) and z["s"].shape == (30,)


def test_ee_pair_and_jetclass_partition_rows():
    assert [r[1] for r in BJ.ROWS if r[0] == "eepair"] == [f"mtx-{a}-s{k}" for a in ("l188", "r16q1") for k in (1, 2, 3)]
    text = BJ.boot_spec("mtx-s9.99", "4Gi", "eepair")
    assert "/eepair/probs/mtx-r16q1-s1/best70/jc2bal_pairs_ee_vs_mm.npz" in text and "--diff r16q1-188=" in text
    text = BJ.jc1_parts_spec("mtx-s9.99", "6Gi")
    assert "fits_v5_probs/mtx-rand2p5-s2/best70/jc1_N1000000.npz" in text and "--diff rand-r16q1=" in text
    assert "--diff flavf0-188=" in text and "--diff rand-188=" in text and "mlp_scores" not in text.split("GENERATED")[0]


def test_go7_reads_the_inputs_of_the_tested_full_set_files():
    rows = BJ.go7_rows()
    assert len(rows) == 7 and len(BJ.GO7_PAIRS) == 10
    lp, ft = json.loads((BOOTDATA / "l3_v5/probe_top_N1200000.itop4.all.json").read_text()), \
        json.loads((BOOTDATA / "l3/ft_qg_N1600000.all.json").read_text())
    for m in BJ.models("rand") + BJ.models("r16q1"):
        assert f"--model {m}={lp['models'][m]['input']} " in rows[0]
        assert f"--model {m}={ft['models'][m]['input']} " in rows[4]
    sc = json.loads((BOOTDATA / "l3/ft_top_N1200000.scratch.json").read_text())
    assert all(f"--model {m}={sc['models'][m]['input']} " in rows[3] for m in BJ.BT.SCRATCH)
    assert "--diff rand-r16q1=" in rows[0] and "--diff r16q1-r42q1=" in rows[0] and "--diff l162-188=" in rows[0]
    assert "--diff rand-scratch=" in rows[3] and "--task qg_herwig " in rows[5]
    assert "ft_jc1parts_N100000.json" in rows[6] and "--diff rand-r16q1=" in rows[6]
    assert "name: bs3-boot-go7-raunav" in BJ.go7_spec("mtx-s9.99", "10Gi")
