"""The v5 refits of the frozen read-outs (PI 2026-11-14): the wider grids and their plateau rule, the mass MLP's epoch
record, the refit rules (experiments/EVAL/refit_checks.py), the job specs (scripts/build_refit_v5_jobs.py) and the
reader's refusal to mix a v5 family with its predecessor (experiments/FIGS/mlst_v4.pick)."""
from __future__ import annotations

import importlib.util
import json
import re
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load("linear_probe_v2_t5", "experiments/EVAL/linear_probe_v2.py")
RC = _load("refit_checks_t5", "experiments/EVAL/refit_checks.py")


def _search(ces):
    return {str(c): {"val_cross_entropy": v} for c, v in ces}


def test_the_v5_grid_spans_twelve_decades_and_its_keys_round_trip():
    g = LP.C_GRID_V5
    assert len(g) == 13 and g[0] == 1e-6 and g[-1] == 1e6 and 0.001 in g and 10000.0 in g
    assert all(float(str(c)) == c for c in g)


def test_the_argmin_is_kept_without_a_tolerance_and_the_plateau_rule_takes_the_smallest_tied_c():
    # mtx-r16q1-s1 qg.nqg at 1.6e6 (fits_v4_fill): the argmin is the top of the v4 grid, 1e-9 nats below 1e3
    s = _search([(0.001, 0.438545553), (0.01, 0.437610240), (0.1, 0.437558831), (1.0, 0.437557066),
                 (10.0, 0.437557063), (100.0, 0.437556979), (1000.0, 0.437556976), (10000.0, 0.437556976)])
    assert LP.select(s, None) == 1000.0          # the first of the equal minima, as the old loop
    assert LP.select(s, LP.PLATEAU_TOL) == 1.0
    # a strict interior minimum stays where it is
    t = _search([(0.01, 0.5), (0.1, 0.4), (1.0, 0.3), (10.0, 0.35)])
    assert LP.select(t, None) == LP.select(t, LP.PLATEAU_TOL) == 1.0


def test_fit_returns_the_probe_of_the_selected_c():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 128)).astype(np.float32)
    y = (x[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(np.int64)
    old = LP.C_GRID, LP.SELECT_TOL
    try:
        LP.C_GRID, LP.SELECT_TOL = (0.01, 1.0, 100.0), LP.PLATEAU_TOL
        sc, clf, rec = LP.fit(x[:300], y[:300], x[300:], y[300:], 2)
        assert rec["C"] == LP.select(rec["search"], LP.PLATEAU_TOL) and rec["selection_tol"] == LP.PLATEAU_TOL
        ref = LP.Logit(np.unique(y[:300]), "cpu")
        ref.fit(sc.transform(x[:300]), y[:300], rec["C"])
        assert np.allclose(clf.coef_, ref.coef_, atol=1e-4)
    finally:
        LP.C_GRID, LP.SELECT_TOL = old


def test_probe_bal_switches_grid_rule_and_slim_saving_together():
    pb = _load("probe_bal_t5", "experiments/EVAL/probe_bal.py")
    pb.set_grid("v5")
    assert pb.lp.C_GRID == pb.lp.C_GRID_V5 and pb.lp.SELECT_TOL == pb.lp.PLATEAU_TOL and pb.SLIM
    assert pb.grid_record("v5")["c_grid"][0] == 1e-6 and "smallest C" in pb.grid_record("v5")["selection"]
    pb.set_grid("v4")
    assert pb.lp.C_GRID == pb.C_GRID_V4 and pb.lp.SELECT_TOL is None and not pb.SLIM
    assert pb.grid_record("v4")["selection"] == "validation cross-entropy"


def test_the_ridge_default_is_unchanged_and_v2_takes_the_largest_alpha_within_the_tolerance():
    lsp = _load("latent_scale_probe_t5", "experiments/EVAL/latent_scale_probe.py")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 8))
    y = x @ rng.normal(size=8) + rng.normal(size=3000)
    _, rec = lsp.fit_ridge(x[:2000], y[:2000], x[2000:2500], y[2000:2500], x[2500:])
    assert set(rec) == {"alpha", "val_r2"} and rec["alpha"] in lsp.ALPHAS
    _, r2 = lsp.fit_ridge(x[:2000], y[:2000], x[2000:2500], y[2000:2500], x[2500:], lsp.ALPHAS_V2, lsp.ALPHA_TOL)
    best = max(r2["search"].values())
    assert r2["alpha"] == max(float(a) for a, v in r2["search"].items() if v >= best - lsp.ALPHA_TOL)
    assert r2["alpha_grid"] == lsp.ALPHAS_V2 and r2["alpha"] not in (lsp.ALPHAS_V2[0], lsp.ALPHAS_V2[-1])


def test_the_mass_mlp_records_its_best_epoch_epochs_run_and_cap():
    pytest.importorskip("torch")
    lsp = _load("latent_scale_probe_t5b", "experiments/EVAL/latent_scale_probe.py")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = x[:, 0] + 0.1 * rng.normal(size=600)
    _, meta = lsp.fit_mlp(x[:400], y[:400], x[400:500], y[400:500], x[500:], seeds=(0,), max_epochs=3)
    s = meta["per_seed"][0]
    assert meta["max_epochs"] == 3 and 1 <= s["best_epoch"] <= s["epochs_run"] <= 3
    assert RC.late({"per_seed": meta["per_seed"], "max_epochs": 3}) == ([0] if s["best_epoch"] > 2.7 else [])


def test_the_refit_rules_find_both_edges_late_epochs_several_cpus_and_missing_models():
    good = {"cells": {"a": {"C": 1.0, "search": _search([(0.1, 1), (1.0, 0.5), (10.0, 0.6)])}}, "cpu_model": "X"}
    top = {"cells": {"a": {"C": 10.0, "search": _search([(0.1, 1), (1.0, 0.5), (10.0, 0.4)])}}, "cpu_model": "X"}
    bottom = {"fits": {"ridge": {"alpha": 0.01, "alpha_grid": [0.01, 1.0, 100.0]}}, "cpu_model": "Y"}
    late = {"fits": {"mlp": {"per_seed": [{"seed": 0, "best_epoch": 58}], "max_epochs": 60}}}
    assert RC.check({"g": good}, ["g"])["ok"]
    r = RC.check({"g": good, "t": top, "b": bottom, "l": late}, ["g", "t", "b", "l", "m"], one_cpu=True)
    assert not r["ok"] and {(e["model"], e["edge"]) for e in r["edges"]} == {("t", "top"), ("b", "bottom")}
    assert [x["model"] for x in r["late_best_epoch"]] == ["l"] and r["missing"] == ["m"]
    assert set(r["cpu_models"]) == {"X", "Y"}


def test_the_stored_v4_records_fail_the_rules_where_the_audit_found_edges():
    data = ROOT / "experiments/FIGS/data/v2/xbx/xbb"
    if not data.is_dir():
        pytest.skip("X->bb fits not pulled")
    r = RC.check(RC.load(data))
    assert any(e["edge"] == "bottom" for e in r["edges"])


@pytest.fixture(scope="module")
def specs():
    b = _load("build_refit_v5_jobs_t5", "scripts/build_refit_v5_jobs.py")
    return b, b.build("mtx-s2.98", 5)


def test_every_spec_is_ours_cpu_only_but_the_one_extraction_and_guards_at_the_finishing_line(specs):
    b, s = specs
    assert set(s) == {"fit", "fill", "sib", "coarse", "qcdhf", "nat", "sophonx", "sophon", "xbx", "mlp", "atlas",
                      "atlas-part", "rebin", "boot", "bootl3", "bootall", "checks", "mlp-cpu", "mlp-init-cpu",
                      "atlas-cpu", "atlas-part-cpu", "cpu-after", "bootall-ft"}
    for kind, text in s.items():
        name = next(ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.strip().startswith("name: ")
                    and "raunav" in ln)
        assert "raunav" in name and "ConfigMap" not in text and "kind: Secret" not in text, kind
        assert ("nvidia.com/gpu" in text) == (kind == "sophonx"), kind
        assert "backoffLimitPerIndex: 6" not in text and "backoffLimit: 6" not in text, kind
        if kind not in ("rebin", "checks"):
            assert '-le 90 ]' in text and "-lt 85" not in text and '-le 85 ]' not in text, kind
        # the one-CPU refit clones the tag its family's majority ran at; every other kind the tag build() was given
        tag = b.CPU_REFIT_TAG[kind.removesuffix("-cpu")] if kind.endswith("-cpu") else "mtx-s2.98"
        assert f"--branch {tag}" in text, kind


def test_the_refits_write_new_folders_only_on_the_v5_grid(specs):
    b, s = specs
    for kind in ("fit", "fill", "sib", "coarse", "nat", "xbx"):
        assert "--grid v5" in s[kind], kind
    for old in ("fits_v4 ", "fits_v4_probs ", "fits_v4_fill ", "atlas_res ", "fits_ihep/xbb "):
        assert all(f"--out /data/results/eval/v2_linprobe_bal/{old}" not in t for t in s.values())
    assert "--out-dir ${OUT} --grid v2 --centre global --mlp-max-epochs 200" in s["atlas"] and "OUT=/data/results/eval/atlas_res_v3" in s["atlas"]
    for k, v in b.CPU_PIN.items():
        for kind in ("atlas", "mlp"):
            assert f"key: {k}" in s[kind] and f'values: ["{v}"]' in s[kind], kind
    assert "--linear-only --grid v5 --out /data/results/eval/xbx/fits_ihep_v5/xbb" in s["xbx"]
    assert "--fill-root /data/results/eval/v2_linprobe_sophon_fill" in s["sophon"]
    assert "best70.${FILL}" not in s["boot"] and "${m##*:}.${FILL}/$1.npz" in s["boot"]
    assert "FILL=itop4; bootfill top_N1200000" in s["boot"] and "FILL=nqg; bootfill qg_herwig_N1600000" in s["boot"]
    assert "touch /data/results/eval/refit_v5_checks/PASS" in s["checks"]
    for m in ("mtx-l162mass-s1", "mtx-r16q1mass-s2", "mtx-rand2p5-s2", "mtx-l188-s1", "init-s3", "mtx-l188kin-s1"):
        assert f"MODEL={m};" in s["mlp"], m
    assert "--mlp --mlp-datasets ${DS} --out /data/results/eval/v2_linprobe_bal/fits_v5_mlp" in s["mlp"]


def _family(d: pathlib.Path, docs: dict) -> None:
    d.mkdir(parents=True)
    for m, doc in docs.items():
        (d / f"{m}.json").write_text(json.dumps(doc))


def test_the_reader_takes_a_v5_family_whole_and_refuses_a_part_or_an_edge(tmp_path):
    V4 = _load("mlst_v4_t5", "experiments/FIGS/mlst_v4.py")
    good = {"cells": {"best70": {"top": {"1000": {"C": 1.0, "search": _search([(0.1, 1), (1.0, 0.5), (10.0, 0.6)])}}}}}
    _family(tmp_path / "linprobe/fits_v4_fill", {"a": good, "b": good})
    assert V4.pick(tmp_path, "linprobe/fits_v4_fill") == "linprobe/fits_v4_fill"
    _family(tmp_path / "linprobe/fits_v5_fill", {"a": good})
    with pytest.raises(SystemExit, match="lacks 1"):
        V4.pick(tmp_path, "linprobe/fits_v4_fill")
    edge = {"cells": {"best70": {"top": {"1000": {"C": 10.0, "search": _search([(0.1, 1), (1.0, 0.5), (10.0, 0.4)])}}}}}
    (tmp_path / "linprobe/fits_v5_fill/b.json").write_text(json.dumps(edge))
    with pytest.raises(SystemExit, match="refit rules"):
        V4.pick(tmp_path, "linprobe/fits_v4_fill")
    (tmp_path / "linprobe/fits_v5_fill/b.json").write_text(json.dumps(good))
    assert V4.pick(tmp_path, "linprobe/fits_v4_fill") == "linprobe/fits_v5_fill"
    # one grid for every linear probe: fits_v5_fill without the other v5 families stops the build
    with pytest.raises(SystemExit, match="one grid"):
        V4.check_v5(tmp_path)



def test_the_v5_bootstraps_write_l3_v5_and_cover_every_printed_network(specs):
    """2026-11-21: bootl3 wrote to l3/ (build_boot_l3_jobs.pb's default root bound at import) and skipped; bootall adds
    every printed top and quark/gluon network against 188 classes and each mass network against its own label set."""
    b, s = specs
    assert "/data/results/bootstrap/l3/" not in s["bootl3"] and s["bootl3"].count(b.L3_V5 + "/probe_") >= 10
    for d in ("rand-188", "flavf0-188", "r16q1mass-188", "l162mass-188", "162mass-162", "17mass-17"):
        assert f"--diff {d}=" in s["bootall"], d
    assert s["bootall"].count("--model ") == 3 * 27
    assert s["bootall-ft"].count("--model ") == 3 * 27 and s["bootall-ft"].count("--headline") == 3
    for d in ("rand-188", "17mass-17", "162mass-162"):
        assert s["bootall-ft"].count(f"--diff {d}=") == 3, d
    assert "best70/leg_top/rand2p5-s2/N1200000/s1" in s["bootall-ft"] and "--task qg_herwig" in s["bootall-ft"]
    assert "/data/results/bootstrap/l3/ft_top_N1200000.all.json" in s["bootall-ft"]


def test_every_cpu_fit_is_pinned_to_the_nodes_of_one_cpu_model_and_checks_it(specs):
    """PI 2026-11-21: fits_v5_mlp ran on three EPYC models and atlas_res_v3 on two, all family 23 model 49 (CPU_PIN);
    every CPU fit of a family is pinned by hostname to one model's nodes, and the pod stops before fitting on another."""
    b, s = specs
    assert set(b.PINNED) <= set(s) and {"boot", "bootl3", "bootall", "rebin", "checks", "sophonx"}.isdisjoint(b.PINNED)
    assert b.FAMILY_CPU["mlp"] == "AMD EPYC 7402P 24-Core Processor" and b.FAMILY_CPU["atlas"] == "AMD EPYC 7742 64-Core Processor"
    for kind, fam in b.PINNED.items():
        cpu, text = b.FAMILY_CPU[fam], s[kind]
        for other, nodes in b.CPU_NODES.items():
            assert all((f'"{n}"' in text) == (other == cpu) for n in nodes), (kind, other)
        assert text.count("key: kubernetes.io/hostname\n                operator: In\n") == 1, kind
        assert text.count(f'[ "${{CPU}}" = "{cpu}" ] || {{') == 1 and "exit 42; }" in text, kind
    assert not set(b.CPU_NODES["AMD EPYC 7402P 24-Core Processor"]) & set(b.CPU_NODES["AMD EPYC 7742 64-Core Processor"])


def test_the_minority_refit_changes_nothing_but_the_node(specs):
    """The refit of the minority networks (CPU_MINORITY) clones the tag the majority ran at, each network's row is the
    launched spec's row character for character, and only those networks are refitted."""
    b, s = specs
    launched = {"mlp-cpu": "mlp", "mlp-init-cpu": "mlp-init", "atlas-cpu": "atlas", "atlas-part-cpu": "atlas-part"}
    for kind, old in launched.items():
        old_text = (b.OUT_DIR / f"job-refit5-{old}-raunav.yaml").read_text()
        rows = [ln.split(") ", 1)[1] for ln in s[kind].splitlines() if re.match(r"^ +\d+\) MODEL=", ln)]
        assert sorted(r.split(";")[0].removeprefix("MODEL=") for r in rows) == sorted(b.CPU_MINORITY[old]), kind
        assert all(r in old_text for r in rows), kind
        assert f"--branch {b.CPU_REFIT_TAG[old]}" in old_text and f"--branch {b.CPU_REFIT_TAG[old]}" in s[kind], kind
        assert "aside () {" in s[kind] and len(re.findall(r"^ +aside /data/", s[kind], re.M)) == 1, kind   # defined, called once
    assert 'memory: "24Gi"' in s["mlp-init-cpu"] and 'memory: "12Gi"' in s["mlp-cpu"]
    assert "--out /data/results/eval/mass_sd_boot/mass_sd_boot_v3.json" in s["cpu-after"]
    assert s["cpu-after"].index("refit_v5_checks/PASS") < s["cpu-after"].index("atlas_rebin.py") < s["cpu-after"].index("mass_sd_boot.py")


def test_aside_moves_only_a_fit_from_another_cpu_and_keeps_it(specs, tmp_path):
    b, _ = specs
    fam, fine = tmp_path / "fam", tmp_path / "fine"
    fam.mkdir(); fine.mkdir()
    (fam / "a.json").write_text(json.dumps({"cpu_model": "AMD EPYC 7402P 24-Core Processor"}))
    (fam / "a_perjet.npz").write_bytes(b"x")
    (fine / "a.json").write_text("{}")
    (fam / "b.json").write_text(json.dumps({"cpu_model": "AMD EPYC 7742 64-Core Processor"}))
    import subprocess
    sh = b.aside_fn("AMD EPYC 7742 64-Core Processor") + f"aside {fam}/a.json {fam}/a_perjet.npz {fine}/a.json\naside {fam}/b.json\naside {fam}/c.json\n"
    subprocess.run(["bash", "-euo", "pipefail", "-c", sh], check=True)
    assert sorted(p.name for p in fam.iterdir()) == ["b.json"]
    assert sorted(p.name for p in (tmp_path / "fam.cpu_aside").iterdir()) == ["MANIFEST.tsv", "a.json", "a_perjet.npz"]
    assert (tmp_path / "fine.cpu_aside" / "a.json").exists() and not (fine / "a.json").exists()
    rows = [ln.split("\t") for ln in (tmp_path / "fam.cpu_aside" / "MANIFEST.tsv").read_text().splitlines()]
    assert [r[0] for r in rows] == ["a.json", "a_perjet.npz"] and rows[0][2] == "AMD EPYC 7402P 24-Core Processor"
