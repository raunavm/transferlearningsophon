"""The second grid through the table generator, end to end on a synthetic one
(tests/v2_fixture.py): v2 files -> experiments/STATS/seed_level.py --v2 and
experiments/STATS/paired_errors.py ratios -> experiments/FIGS/make_tables.py macros under
the first grid's names -> the red v2 markers switched to the request to rewrite the text.

The generator is copied into the fixture root, as it sits in the repository, so the
markers it prints only inside its own repository are printed here too. The first grid's
fixture (tests/test_make_tables.py) is laid beside the second's: the generator always
needs it, and a section without v2 files must keep reading it."""
import importlib.util
import json
import math
import pathlib
import re
import shutil

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


FX = _mod("v2_fixture_mt", REPO / "tests/v2_fixture.py")
T1 = _mod("test_make_tables_v1fx", REPO / "tests/test_make_tables.py")


def _generator(root):
    for f in ("make_tables.py", "appendix_tables.py"):
        dest = root / "experiments/FIGS" / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "experiments/FIGS" / f, dest)
    return _mod(f"make_tables_in_{root.name}", root / "experiments/FIGS/make_tables.py")


REAL_SIZES = {"L188": 188, "L162": 162, "R63_Q1": 64, "R42_Q1": 43, "R29_Q1": 30, "R16_Q1": 17,
              "R3_VIS": 4, "R1_Q1": 2}


def _rung_map(root):
    """A 188-class map with the tree's real group counts: the first grid's synthetic map has
    two rungs of two groups, which label recovery, read at every rung, would name alike."""
    names = {v: k for k, v in T1.CLASS_ROW.items()}
    # the anomaly signals at their real jet_label, which A13's family-out selection reads
    names.update({3: "label_X_qq", 15: "label_X_YY_bbbb", 25: "label_X_YY_bbb", 63: "label_X_YY_qqqq",
                  73: "label_X_YY_qqq"})
    head = ["jet_label", "class_name"] + [c for r in T1.M.RUNGS for c in (r, f"{r}_name")]
    lines = [",".join(head)]
    for i in range(188):
        row = [str(i), names.get(i, f"c{i:03d}")]
        for r in T1.M.RUNGS:
            g = i * REAL_SIZES[r] // 188
            row += [f"g{g}", f"{r}_g{g}"]
        lines.append(",".join(row))
    (root / "configs/labelmaps/rung_label_maps.v1.csv").write_text("\n".join(lines) + "\n")


def _v1(root):
    T1.write_analysis(root, T1.write_ladder(root))
    T1.write_rung_map(root)
    _rung_map(root)
    T1.write_extras(root)


@pytest.fixture
def v1_only(tmp_path):
    _v1(tmp_path)
    return tmp_path, _generator(tmp_path)


@pytest.fixture
def v2_root(tmp_path):
    _v1(tmp_path)
    FX.write_design(tmp_path)
    FX.write_frozen(tmp_path, tier3=False)
    FX.write_ft(tmp_path)
    FX.write_pretraining(tmp_path)
    FX.write_paired(tmp_path, wavg_shift={"R16_Q1": 0.2})
    FX.seed_level_v2(tmp_path)
    return tmp_path, _generator(tmp_path)


def macros(built):
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  %", built["results_generated.tex"]))


def test_without_v2_files_every_marker_waits_and_the_first_grid_is_read(v1_only):
    root, M = v1_only
    built, missing, _ = M.build(root)
    m = macros(built)
    assert m["PendingProbes"].startswith("\\pending{pending v2:")
    assert any("v2/probe_ladder/" in x for x in missing)
    assert "ProbeAucAlphaLinearOnetwo" in m              # the first grid's synthetic ladder


def test_the_v2_files_reach_the_paper_under_the_first_grid_names(v2_root):
    root, M = v2_root
    built, missing, skipped = M.build(root)
    m = macros(built)
    # the markers: read sections ask for the text around their numbers; a section whose
    # files exist but are not printed says so; the rest still wait
    assert m["PendingProbes"] == "\\pending{v2 numbers in place (probe_ladder): rewrite the text around them}"
    for key in ("PendingPaired", "PendingFtHeldout", "PendingPretrain", "PendingSsl", "PendingRandom",
                "PendingMassLambda", "PendingFtRefs"):
        assert m[key].startswith("\\pending{v2 numbers in place"), key
    assert m["PendingAnomaly"].startswith("\\pending{pending v2:")
    assert not any(x.startswith("v2 ") and "probe_ladder" in x for x in missing)
    assert any("v2/anomaly/" in x for x in missing)
    # the frozen probes at the primary checkpoint, class token, under the first grid's names:
    # the 17-class model's 1 - AUC is exp(BASE + 4 STEP + 0.01 run) over runs 1-3
    x = [math.exp(FX.log1m("R16_Q1", k, "best70", "features", "bvc_resonant")) for k in (1, 2, 3)]
    assert m["ProbeOmaBvcResonantLinearOneseven"] == M.math_safe(M.fmt_pm_sci(np.mean(x), np.std(x, ddof=1)))
    assert m["ProbeNSeedsWord"] == "three" and m["ProbeNBkgTestBvcResonant"] == "1{,}000"
    assert "ProbeAucBvcResonantLinearOneeighteight" in m and "ProbeAucAlphaLinearOnetwo" not in m
    # beside it, the twin, the pooled embedding and the untrained trunk
    tw = [math.exp(FX.log1m("R16_Q1", k, "best70_bn", "features", "bvc_resonant")) for k in (1, 2, 3)]
    assert m["ProbeOmaBvcResonantLinearOnesevenTwin"] == M.math_safe(M.fmt_pm_sci(np.mean(tw), np.std(tw, ddof=1)))
    assert "ProbeOmaBvcResonantLinearOnesevenPooled" in m
    assert "ProbeOmaBvcResonantLinearUntrained" in m and "ProbeOmaBvcResonantMlpUntrainedPooled" in m
    table = built["tables/probes_linear.tex"]
    assert "second set of runs" in table and "BatchNorm recomputed" in table
    assert "reference rows" in table and "untrained & AUC, pooled embedding" in table
    for lv in (188, 162, 43, 17):
        assert f"\n{lv} & AUC & " in table
    # the |V_cb| window probe and the mass output, from the same table
    v = [math.exp(FX.log1m(a, k, "best70", "features", "bc_vs_rest")) for a in ("R16_Q1", "L162") for k in (1, 2, 3)]
    assert m["VcbOmaRatioLinear"] == M.fmt_ratio(np.mean(v[:3]) / np.mean(v[3:]))
    assert m["VcbNSignal"] == "200" and m["MassNRunsWord"] == "three"
    x1 = [math.exp(FX.log1m("L162_MASS", k, "best70", "features", "bvc_resonant")) for k in (1, 2, 3)]
    x0 = [math.exp(FX.log1m("L162", k, "best70", "features", "bvc_resonant")) for k in (1, 2, 3)]
    assert m["MassOmaRatioLinearOnesixtwo"] == M.fmt_ratio(np.mean(x1) / np.mean(x0))
    # mass regression and label recovery, the v2 columns
    s = [0.15 + 0.01 * FX.RANK["R16_Q1_MASS"] + 0.001 * k - 0.02 for k in (1, 2, 3)]
    assert m["MassResSigmaEffMlpOnesevenMass"] == M.math_safe(M.fmt_pm(np.mean(s), np.std(s, ddof=1)))
    assert "17 + mass" in built["tables/mass.tex"]
    acc = [FX.recovery_acc("R42_Q1", k, "L188", 1000) for k in (1, 2, 3)]
    key = "RecoveryAccOneeighteightByFourthree"
    assert m[key] == M.math_safe(M.fmt_pm(np.mean(acc), np.std(acc, ddof=1)))
    # fine-tuning, from the freeze's read-out (tiers 1-2): the rows the grid gives
    ft = [1 - FX.ft_value("R16_Q1", k, "N1000", 1) for k in (1, 2, 3)]
    assert m["FtAucJciiEThreeOneseven"] == M.math_safe(M.fmt_pm(np.mean(ft), np.std(ft, ddof=1)))
    assert "FtAucJciiEThreeOnesevenMass" in m and "FtAucJciiEThreeSelfSupervised" in m
    assert "second set of runs" in built["tables/finetune.tex"]
    assert any(s.startswith("finetune_recipe") for s in skipped)
    # the selected epochs by vocabulary
    assert m["PretrainEpochOnesevenRunTwo"] == "74" and m["PretrainBestvalElsewhereOnesixtwo"] == "3 of 3"
    assert "tables/v2_selected_epochs.tex" in built


def test_the_paired_ratios_carry_the_checkpoint_rules_beside_them(v2_root):
    # The fixture moves the 17-class model by +0.2 in ln(1 - AUC) at the weight average:
    # 17 over 188 shifts by exp(0.2) there and depends on the checkpoint; 162 over 188 is robust.
    root, M = v2_root
    built, _, _ = M.build(root)
    m = macros(built)
    R = json.loads((FX.v2(root, "paired_errors") / "probes/ratios.json").read_text())
    row = next(r for r in R["ratios"] if (r["contrast"], r["task"], r["kind"], r["fine"], r["coarse"],
                                          r["checkpoint"]) == ("ladder", "bvc_resonant", "linear", "188", "17", "best70"))
    base = "PairedProbeBvcResonantLinearOnesevenOverOneeighteight"
    assert m[base] == M.math_safe(M.fmt_paired(row["ratio"], *row["ci95"]))
    assert row["ratio"] == pytest.approx(math.exp(4 * FX.STEP), rel=1e-3)
    assert m[base + "WavgLabel"] == "depends on the checkpoint"
    assert float(m[base + "WavgShift"].split("~")[0]) == pytest.approx(math.exp(0.2), abs=0.01)
    assert m["PairedProbeBvcResonantLinearOnesixtwoOverOneeighteightWavgLabel"] == "robust"
    for suffix in ("Wavg", "Bestval", "BestvalShift", "BestvalLabel", "Twin", "Pooled"):
        assert base + suffix in m, suffix
    # the dependent count against 5 %, re-derived from the labelled rows
    dep = R["checkpoint_dependence"]["wavg/best70"]["results"]
    assert m["CkptWavgDependent"] == str(dep["dependent"]) and m["CkptWavgN"] == str(dep["n"])
    assert m["CkptWavgExpected"] == M.fmt(0.05 * dep["n"], 1)
    assert m["CkptWavgModelsProbesDependent"] == str(R["checkpoint_dependence"]["wavg/best70"]["models"]["dependent"])
    assert int(m["CkptWavgModelsProbesDependent"]) >= 2          # the 17-class model, both probes
    assert "tab:checkpoint-robustness" in built["tables/checkpoint_robustness.tex"]


@pytest.mark.parametrize("lo, hi, want", [
    (1.02, 1.05, "depends on the checkpoint, under 10%"),   # excludes 1, inside [1/1.1, 1.1]
    (0.92, 0.98, "depends on the checkpoint, under 10%"),
    (1.12, 1.30, "depends on the checkpoint"),               # excludes 1, beyond 1.1
    (0.70, 0.95, "depends on the checkpoint"),               # excludes 1, lower end below 1/1.1
    (0.95, 1.08, "robust"),                                  # holds 1, inside the band
    (0.85, 1.05, "inconclusive"),                            # holds 1, lower end below 1/1.1
    (0.95, 1.20, "inconclusive"),
])
def test_the_checkpoint_label_by_hand(v1_only, lo, hi, want):
    _, M = v1_only
    assert M.checkpoint_class(lo, hi) == want


def test_the_dependent_count_by_hand(v1_only):
    _, M = v1_only
    rows = [{"contrast": "ladder", "checkpoint": "wavg/best70", "checkpoint_label": "x", "ci95": ci,
             "ln_ratio": 0.1, "ln_combined_se": 0.01}
            for ci in ([1.02, 1.05], [1.12, 1.3], [0.95, 1.08], [0.85, 1.05], [0.95, 1.08])]
    rows.append({"contrast": "ladder", "checkpoint": "wavg/best70", "checkpoint_label": "robust",
                 "ci95": [1.0, 1.0], "ln_ratio": 0.0, "ln_combined_se": 0.0})       # one file: apart
    rows.append({"contrast": "checkpoint", "checkpoint": "wavg/best70", "checkpoint_label": "x",
                 "ci95": [1.2, 1.4], "ln_ratio": 0.3, "ln_combined_se": 0.05})
    t = M.checkpoint_tally(rows, "wavg/best70")
    assert t["results"] == {"dependent": 2, "n": 5, "robust": 2, "inconclusive": 1, "identical": 1}
    assert t["models"] == {"dependent": 1, "n": 1, "robust": 0, "inconclusive": 0, "identical": 0}


def test_a_label_or_count_the_intervals_do_not_give_is_refused(v2_root):
    root, M = v2_root
    p = FX.v2(root, "paired_errors") / "probes/ratios.json"
    good = p.read_text()
    R = json.loads(good)
    i = next(i for i, r in enumerate(R["ratios"]) if r.get("checkpoint_label") == "robust")
    R["ratios"][i]["checkpoint_label"] = "depends on the checkpoint"
    p.write_text(json.dumps(R))
    with pytest.raises(SystemExit, match="its interval"):
        M.build(root)
    R = json.loads(good)
    R["checkpoint_dependence"]["wavg/best70"]["results"]["dependent"] += 1
    p.write_text(json.dumps(R))
    with pytest.raises(SystemExit, match="labelled rows give"):
        M.build(root)


def test_an_unpaired_contrast_keeps_to_one_gpu_product(v2_root):
    # runs 1-3 train on an RTX 3090 in the fixture's job specs, as the grid's do; then run 3 moves
    root, M = v2_root
    p = FX.v2(root, "paired_errors") / "probes/ratios.json"
    R = json.loads(p.read_text())
    row = {"contrast": "family_unseen_vs_seen", "family": "probe", "task": "bvc_resonant", "kind": "linear",
           "metric": "1-auc", "checkpoint": "best70", "fine": "188", "coarse": "188, family unseen",
           "fine_arm": "L188", "coarse_arm": "L188_LOFO4P", "stream_pairing": "exempt: unpaired",
           "fine_models": ["l188-s1@best70", "l188-s2@best70"],
           "coarse_models": ["l188lofo4p-s1@best70", "l188lofo4p-s2@best70"], "ratio": 1.1, "ci95": [1.0, 1.2]}
    R["ratios"].append(row)
    p.write_text(json.dumps(R))
    m = macros(M.build(root)[0])
    assert "PairedProbeBvcResonantLinearOneeighteightFamilyUnseenOverOneeighteight" in m
    row["fine_models"].append("l188-s3@best70")
    p.write_text(json.dumps(R))
    assert "PairedProbeBvcResonantLinearOneeighteightFamilyUnseenOverOneeighteight" in macros(M.build(root)[0])
    FX.write_specs(root, {1: "NVIDIA-GeForce-RTX-3090", 2: "NVIDIA-GeForce-RTX-3090", 3: "NVIDIA-L40"})
    with pytest.raises(SystemExit, match="I7"):
        M.build(root)


def test_the_one_product_rule_by_hand(v1_only):
    _, M = v1_only
    r = M.v2_one_product({1: "RTX", 2: "RTX", 3: "RTX", 4: "L40", 5: "L40"})
    five = [f"l188-s{k}" for k in range(1, 6)]
    assert r(five, [f"r16q1-s{k}" for k in range(1, 6)]) == (five, [f"r16q1-s{k}" for k in range(1, 6)])
    # three self-supervised runs against five: both sides keep runs 1-3
    assert r(["mpm-v2-s1", "mpm-v2-s2", "mpm-v2-s3"], five) == (["mpm-v2-s1", "mpm-v2-s2", "mpm-v2-s3"], five[:3])
    assert r(["x-s4", "x-s5"], five) is None
    assert M.v2_run("mpm-v2-s3@best70:pooled") == 3 and M.v2_run("rand2p1-s2") == 2


def test_tier_three_rows_join_when_their_directory_arrives(v2_root):
    root, M = v2_root
    m = macros(M.build(root)[0])
    assert "ProbeAucBvcResonantLinearSixfour" not in m
    assert m["PendingLevels"].startswith("\\pending{pending v2:")
    # the 64-class runs arrive: the analysis that read fewer files is refused ...
    FX.write_frozen(root)
    with pytest.raises(SystemExit, match="seed_level.py --v2"):
        M.build(root)
    # ... and one over every file present is read, the old one beside it untouched
    FX.seed_level_v2(root, "analysis_t123")
    built = M.build(root)[0]
    m = macros(built)
    assert "ProbeAucBvcResonantLinearSixfour" in m
    assert m["PendingLevels"].startswith("\\pending{v2 numbers in place (levels_64_30)")
    assert "\n64 & AUC & " in built["tables/probes_linear.tex"]


def test_an_incomplete_section_is_refused(v2_root):
    root, M = v2_root
    shutil.rmtree(FX.v2(root, "probe_ladder") / "probe/mtx-r42q1-s2")
    shutil.rmtree(FX.v2(root, "probe_ladder") / "mass_resolution/mtx-r42q1-s2")
    FX.seed_level_v2(root, "analysis_partial")
    with pytest.raises(SystemExit, match="runs missing"):
        M.build(root)


def test_the_full_fine_tuning_read_out_replaces_the_freeze_only_if_it_agrees(v2_root):
    root, M = v2_root
    FX.write_ft(root, scope="")
    assert "FtAucJciiEThreeOneeighteight" in macros(M.build(root)[0])
    p = FX.v2(root, "finetune") / "best70_leg1_metrics.json"
    d = json.loads(p.read_text())
    d["cells"]["l188-s1"]["N1000"]["s1"]["accuracy"] += 0.01
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="disagree"):
        M.build(root)


def test_every_v2_macro_is_traceable_and_legal(v2_root):
    root, M = v2_root
    built = M.build(root)[0]
    prov = json.loads(built["provenance.json"])
    assert set(prov) == set(macros(built))
    for name, entry in prov.items():
        assert re.fullmatch(r"[A-Za-z]+", name), name
        assert (root / entry["source_file"]).exists(), name
