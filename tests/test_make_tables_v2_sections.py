"""The second grid's sections beyond the frozen probes, end to end on the synthetic grid of
tests/v2_fixture.py: the random partitions and the flavour pair (A10, A14 P1/P2, random
against semantic), the matched mass weight (A11), the self-supervised model's validity bar
(PRESPEC 4, A14) and the fine-tuning references, the benchmarks (A6), anomaly detection
and the family left out (A13), and the real data. Each label printed is re-derived from its
interval; a stored label the interval does not give stops the build."""
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


FX = _mod("v2_fixture_sec", REPO / "tests/v2_fixture.py")
T2 = _mod("test_make_tables_v2_sec", REPO / "tests/test_make_tables_v2.py")


def _full(root, scratch_rank=FX.SCRATCH_RANK):
    T2._v1(root)
    FX.write_design(root)
    FX.write_frozen(root, tier3=True)
    FX.write_ft(root, scratch_rank=scratch_rank)
    FX.write_pretraining(root)
    FX.write_paired(root)
    FX.write_paired_ft(root)
    FX.write_loss_share(root)
    FX.write_bench(root)
    FX.write_anomaly(root, sets=("t12", "t123"))
    FX.write_paired_anomaly(root)
    FX.write_real_data(root)
    FX.seed_level_v2(root)
    return root, T2._generator(root)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root, M = _full(tmp_path_factory.mktemp("full"))
    built, missing, skipped = M.build(root)
    return root, M, built, T2.macros(built)


def _ratios(root, fam="probes"):
    return json.loads((FX.v2(root, "paired_errors") / fam / "ratios.json").read_text())


def _iv(text):
    """'1.35~[1.33, 1.37]' or '+0.300~[+0.295, +0.305]' -> (1.35, 1.33, 1.37)."""
    t = text.replace("\\ensuremath{-}", "-")
    return tuple(float(x) for x in re.fullmatch(r"([-+\d.]+)~\[([-+\d.]+), ([-+\d.]+)\]", t).groups())


# ------------------------------------------------------------------ Random (A10, A14)
def test_p1_reads_the_merge_cost_on_each_pair_with_its_doses(full):
    root, M, built, m = full
    R = _ratios(root)
    row = next(r for r in R["ratios"] if r["contrast"] == "partition_split_vs_merged" and r["checkpoint"] == "best70"
               and r["task"] == "bvc_resonant" and r["kind"] == "linear")
    assert _iv(m["RandPoneBbCcLinear"])[0] == pytest.approx(math.exp(FX.MERGE), abs=0.01)
    assert m["RandPoneBbCcLinear"] == M.math_safe(M.fmt_paired(row["ratio"], *row["ci95"]))
    assert m["RandPoneBbCcLinearLabel"] == "merging costs"
    merging = sorted(int(a[-1]) for a in row["merged_partitions"])
    assert m["RandPoneBbCcMerging"] == M.word_list(merging)
    assert m["RandPoneBbCcSplitting"] == M.word_list(sorted({1, 2, 3, 4, 5} - set(merging)))
    for arm, dose in row["axis_doses"]["bb/cc"]["realised"].items():
        assert m["RandDoseBbCcP" + M.texname(int(arm[-1]))] == M.fmt(dose, 2)
    # against the 17-class model, runs 1-2: a partition that splits the pair ties it, one
    # that merges it is the planted cost worse (the reference over the random side)
    assert _iv(m[f"RandVsSemOnesevenBbCcLinearP{M.texname(merging[0])}"])[0] == pytest.approx(math.exp(-FX.MERGE), abs=0.01)
    split = min({1, 2, 3, 4, 5} - set(merging))
    assert _iv(m[f"RandVsSemOnesevenBbCcLinearP{M.texname(split)}"])[0] == pytest.approx(1.0, abs=0.01)
    assert m["RandVsSemOnesevenBbCcLinearCellsAxisAccount"] == "inconclusive"     # merging costs, not beats
    assert "tab:random-partitions" in built["tables/v2_random_partitions.tex"]


def test_p2_every_clause_holds_on_the_planted_flavour_pair(full):
    root, M, built, m = full
    for probe in ("Linear", "Mlp"):
        assert _iv(m["RandPtwoGap" + probe])[0] == pytest.approx(FX.STEP, abs=0.01)     # 17 - 43 classes
        assert m["RandPtwoMargin" + probe] == M.fmt(FX.STEP / 4, 3)
        assert _iv(m["RandPtwoA" + probe])[0] == pytest.approx(0.75 * FX.STEP, abs=0.01)   # F0 - F1, four-prong
        assert _iv(m["RandPtwoB" + probe])[0] == pytest.approx(0.75 * FX.STEP - FX.STEP / 2, abs=0.01)
        assert _iv(m["RandPtwoC" + probe])[0] == pytest.approx(0.0, abs=0.01)          # F0 - 17 classes
        assert _iv(m["RandPtwoD" + probe])[0] == pytest.approx(0.75 * FX.STEP, abs=0.01)   # F1r - F1
        for c in "ABCD":
            assert m[f"RandPtwo{c}{probe}Label"] == "holds"
        assert m["RandPtwoWithdrawal" + probe] == "not withdrawn"
    flav = built["tables/v2_flavour_pair.tex"]
    assert "(a) holds; (b) holds; (c) holds; (d) holds; withdrawal rule: not withdrawn" in flav


@pytest.mark.parametrize("where, edit, msg", [
    ("p1", lambda R: next(r for r in R["ratios"] if r.get("p1_label")).update(p1_label="merging costs nothing"),
     "P1 label"),
    ("p2", lambda R: R["p2_verdict"][0]["clauses"]["b"].update(label="fails"), "clause"),
    ("p2", lambda R: R["p2_verdict"][0]["withdrawal"].update(label="withdrawn"), "withdrawal"),
    ("acc", lambda R: next(r for r in R["ratios"] if "axis_account" in r).update(axis_account="beats"), "intervals give"),
])
def test_a_stored_reading_its_interval_does_not_give_is_refused(tmp_path, where, edit, msg):
    root, M = _full(tmp_path)
    p = FX.v2(root, "paired_errors") / "probes" / "ratios.json"
    R = json.loads(p.read_text())
    R["p2_verdict"] = [b for b in R["p2_verdict"] if b["checkpoint"] == "best70"]
    edit(R)
    p.write_text(json.dumps(R))
    with pytest.raises(SystemExit, match=msg):
        M.build(root)


def test_p1_and_p2_labels_by_hand(tmp_path):
    M = T2._generator(tmp_path)
    assert M.p1_class(1.02, 1.08) == "merging costs, under 10%"
    assert M.p1_class(1.02, 1.30) == "merging costs"
    assert M.p1_class(0.95, 1.05) == "merging costs nothing"
    assert M.p1_class(0.95, 1.20) == "inconclusive"
    assert M.threshold_class(0.1, 0.3) == "holds" and M.threshold_class(-0.3, -0.1) == "fails"
    assert M.threshold_class(-0.1, 0.1) == "inconclusive" and M.threshold_class(0.2, 0.4, 0.1) == "holds"
    assert M.equivalence_class(-0.05, 0.05, 0.1) == "holds" and M.equivalence_class(0.2, 0.3, 0.1) == "fails"
    assert M.equivalence_class(-0.05, 0.15, 0.1) == "inconclusive" and M.equivalence_class(0, 1, -1) == "not evaluable"


# ------------------------------------------------------------------ A11
def test_the_matched_weight_fraction_and_the_realised_shares(full, tmp_path):
    root, M, built, m = full
    # 17 classes: mass output costs 0.4 STEP-units, the matched weight 0.3; 162 classes 0.2: half removed
    assert m["MassLambdaFractionLinear"] == "0.50" and m["MassLambdaFractionLinearWavg"] == "0.50"
    lo, hi = (float(x) for x in m["MassLambdaFractionIntervalLinear"].strip("[]").split(", "))
    assert lo < 0.5 < hi
    assert m["MassLossShareVtwoOnesevenMass"] == M.math_safe(M.fmt_pm(33.2, 0.1)) + "\\%"     # the SLOT, filled
    assert m["MassGradShareVtwoOnesevenMassMatched"].startswith("7.70")
    assert m["MassGradCosineVtwoOnesixtwoMass"] == M.math_safe(M.fmt_pm(0.2, 0.1))
    # a denominator whose interval holds 0 leaves the Fieller set unbounded: said, not printed as a number
    M2 = T2._generator(tmp_path)
    em = M2.Emitter(tmp_path)
    src = tmp_path / "r.json"
    src.write_text("{}")
    M2.emit_v2_mass_lambda(em, {"ratios": [{"contrast": "mass_lambda_fraction", "family": "probe",
                                            "task": "bvc_resonant", "metric": "1-auc", "checkpoint": "best70",
                                            "kind": "linear", "fraction": 0.7, "ci95": None}]}, src, None)
    assert dict((n, b) for n, b, _ in em.macros)["MassLambdaFractionIntervalLinear"] == "unbounded"


def test_the_share_file_paired_errors_writes(tmp_path):
    pe = FX._pe()
    spec = pe.load_spec(pe.CONTRASTS["v2"])
    for arm in ("L162_MASS", "R16_Q1_MASS", "R16_Q1_MASS_LM"):
        for k in range(1, 6):
            d = tmp_path / f"mtx-{arm.lower().replace('_', '')}-s{k}"
            (d / "metrics").mkdir(parents=True)
            lam = spec["grid_arms"][arm]["mass_lambda"]
            (d / "metrics" / "epoch-000.json").write_text(json.dumps(
                {"train": {"loss_cls": 1.0, "loss_reg": 0.02 * k / lam},
                 "grad_diag": {"grad_norm": {"loss_cls": 1.0, "lambda_loss_reg": 0.1}, "cosine": 0.5}}))
            (d / "DONE").write_text("{}")
    S = pe.a11_share_file(spec, tmp_path)
    x = [0.02 * k for k in range(1, 6)]
    assert S["shares"]["17+mass_matched"] == pytest.approx([v / (1 + v) for v in x])
    assert S["grad_shares"]["162+mass"] == pytest.approx([0.1 / 1.1] * 5) and S["grad_cosine"]["17+mass"] == [0.5] * 5
    (tmp_path / "mtx-r16q1mass-s5" / "DONE").unlink()
    with pytest.raises(SystemExit, match="finished runs"):
        pe.a11_share_file(spec, tmp_path)


# ------------------------------------------------------------------ Ssl and FtRefs
def test_the_self_supervised_model_enters_once_its_bar_passes(full):
    root, M, built, m = full
    assert m["SslValid"] == "valid" and m["SslClauseOne"] == "holds" and m["SslClauseTwo"] == "holds"
    # clause 1 at 10^3 jets on JetClass-II: scratch's mean ln(1 - macro AUC) over its fine-tuning seeds
    # minus the self-supervised runs' (fine-tuning seed 1)
    z = [math.log(FX.ft_value("INIT", s, "N1000", 1, FX.SCRATCH_RANK)) for s in (1, 2, 3)]
    s = [math.log(FX.ft_value("MPM", k, "N1000", 1)) for k in (1, 2)]
    assert m["SslMarginJciiEThree"] == M.fmt(np.mean(z) - np.mean(s), 3)
    assert m["SslSpreadJciiEThree"] == M.fmt(max(np.std(z, ddof=1), np.std(s, ddof=1)), 3)
    for key in ("FtAucJciiEThreeSelfSupervised", "ProbeOmaBvcResonantLinearSelfSupervised",
                "PairedFtJciiEThreeSelfSupervisedOverOneeighteight"):
        assert key in m, key
    a, lo, hi = _iv(m["PairedFtJciiEThreeSelfSupervisedOverOneeighteight"])     # Welch, runs 1-3 against 1-2
    assert a == pytest.approx(FX.ft_value("MPM", 1.5, "N1000", 1) / FX.ft_value("L188", 2, "N1000", 1), rel=0.01)
    assert "self-supervised & AUC, pooled embedding" in built["tables/probes_linear.tex"]


def test_a_self_supervised_model_that_fails_the_bar_enters_no_table(tmp_path):
    root, M = _full(tmp_path, scratch_rank=3)            # from scratch now beats it
    m = T2.macros(M.build(root)[0])
    assert m["SslValid"] == "not valid" and m["SslClauseOne"] == "fails"
    assert not [k for k in m if "SelfSupervised" in k and not k.startswith(("Pending", "Pretrain", "Anomaly", "Lofo",
                                                                              "BenchLast", "Bench"))
                and k.startswith(("FtAuc", "FtAcc", "ProbeOma", "PairedFt"))]


def test_the_from_scratch_reference_is_a_row_over_its_fine_tuning_seeds(full, tmp_path):
    root, M, built, m = full
    v = [1 - FX.ft_value("INIT", s, "N1000", 1, FX.SCRATCH_RANK) for s in (1, 2, 3)]
    assert m["FtAucJciiEThreeScratch"] == M.math_safe(M.fmt_pm(np.mean(v), np.std(v, ddof=1)))
    assert "random initialisation, over fine-tuning seeds ($n=3$)" in built["tables/finetune.tex"]
    # the standalone JetClass-II readout must hold the same cells as the read-out beside the rule's
    root2, M2 = _full(tmp_path)
    p = FX.v2(root2, "finetune_references") / "scratch_leg1_metrics.json"
    d = json.loads(p.read_text())
    d["cells"]["scratch-v2"]["N1000"]["s1"]["macro_auc_ovr"] -= 0.01
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="different from-scratch cells"):
        M2.build(root2)


# ------------------------------------------------------------------ Bench (A6)
def test_the_benchmarks_at_best_validation_and_at_the_last_epoch(full, tmp_path):
    root, M, built, m = full
    r = [FX.bench_cell("L188", k, "best70", "top")["r50"] for k in (1, 2, 3)]
    assert m["BenchTopRfiftyEThreeOneeighteight"] == M.math_safe(M.fmt_rejection(r, [False] * 3, [0] * 3))
    assert m["BenchLastQgHerwigRfiftyOneeighteight"] == m["BenchTopRfiftyEThreeOneeighteight"]
    assert "BenchTopRfiftyEThreeScratch" in m and "BenchLastTopRfiftyScratch" not in m
    assert m["BenchLastTopRfiftyOneeighteightWavgLabel"] == "depends on the checkpoint, under 10\\%"   # x1.02
    assert _iv(m["BenchLastTopRfiftyOneeighteightWavgShift"])[0] == pytest.approx(1.02)
    assert "random initialisation" not in built["tables/v2_benchmarks_last.tex"]     # no last epoch kept
    root2, M2 = _full(tmp_path)
    full_file = FX.v2(root2, "benchmarks") / "best70_bench_metrics.json"
    d = json.loads((FX.v2(root2, "benchmarks") / "best70_t12_bench_metrics.json").read_text())
    d["cells"]["top"]["l188-s1"]["N1000"]["s1"]["r50"] += 1
    full_file.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="disagree"):
        M2.build(root2)


# ------------------------------------------------------------------ Anomaly, Lofo (A13)
def test_anomaly_detection_on_the_second_grid(full):
    root, M, built, m = full
    v = [math.exp(FX.ln_sigma_min("L188", k, "best70", "features", "label_X_YY_bbbb")) for k in (1, 2, 3)]
    assert m["AnomalySigmaMinKnnXYYBbbbOneeighteight"] == M.math_safe(M.fmt_pm(np.mean(v), np.std(v, ddof=1)))
    assert m["AnomalyNRunsWord"] == "three" and m["AnomalyNBkg"] == "100{,}000"
    x = lambda a: {k: math.exp(FX.ln_sigma_min(a, k, "best70", "features", "label_X_YY_bbbb")) for k in (1, 2, 3)}
    assert m["PairedAnomalyKnnXYYBbbbOnesevenOverOneeighteight"] == M.math_safe(
        M.fmt_paired(*M.paired_ratio(x("L188"), x("R16_Q1"))))
    assert "PairedAnomalyClassSumMatchedXYYBbbbOnesevenOverOneeighteight" in m       # paired_errors.py's
    assert "PairedAnomalyOutputOverMahalanobisXYYBbbbOneseven" in m
    assert m["AnomalySigmaMinKnnXYYBbbbOneeighteightWavgLabel"] == "depends on the checkpoint, under 10\\%"
    assert "AnomalySigmaMinKnnXYYBbbbSelfSupervised" in m and "AnomalySigmaMinKnnXYYBbbbOneeighteightTwin" in m
    assert "second set of runs" in built["tables/anomaly.tex"] and "tab:anomaly-per-run" in built["tables/anomaly_per_run.tex"]


def test_the_family_left_out_against_the_same_vocabulary_seen(full):
    root, M, built, m = full
    seen = [math.exp(FX.ln_sigma_min("L188", k, "best70", "features", "label_X_YY_bbbb")) for k in (1, 2, 3)]
    unseen = [math.exp(FX.ln_sigma_min("L188_LOFO4P", k, "best70", "features", "label_X_YY_bbbb")) for k in (1, 2, 3)]
    assert m["LofoUnseenKnnXYYBbbbOneeighteight"] == M.math_safe(M.fmt_paired(*M.welch_ratio(seen, unseen)))
    assert _iv(m["LofoUnseenKnnXYYBbbbOneeighteight"])[0] == pytest.approx(math.exp(FX.LOFO_EFFECT), abs=0.01)
    assert "LofoUnseenMahalanobisXYYBbbbSelfSupervised" in m                        # pooled readout
    assert "PairedAnomalyKnnUnseenXYYBbbbOnesevenOverOneeighteight" in m            # the ladder among them
    assert "tab:lofo" in built["tables/v2_lofo.tex"]


def test_the_lofo_signals_follow_the_first_grids_detection(tmp_path):
    root, M = _full(tmp_path)
    grid = json.loads((root / "configs/arms/v2_grid.json").read_text())["arms"]
    assert M.lofo_signals(root, grid, None, "knn") == ["label_X_YY_bbbb", "label_X_YY_qqqq"]
    v1 = {"not_detected_rule": {"not_detected": ["knn|label_X_YY_qqqq"]}}
    assert M.lofo_signals(root, grid, v1, "knn") == ["label_X_YY_bbbb"]
    assert M.lofo_signals(root, grid, v1, "mahalanobis") == ["label_X_YY_bbbb", "label_X_YY_qqqq"]


def test_the_lofo_section_waits_for_the_anomaly_study_and_keeps_to_one_gpu(tmp_path):
    root, M = _full(tmp_path)
    shutil.rmtree(FX.v2(root, "anomaly"))
    shutil.rmtree(FX.v2(root, "paired_errors") / "anomaly")
    m = T2.macros(M.build(root)[0])
    assert m["PendingLofo"].startswith("\\pending{v2 input present (leave_one_family_out) but not printed yet")
    root2, M2 = _full(tmp_path / "b")
    FX.write_specs(root2, {1: "NVIDIA-GeForce-RTX-3090", 2: "NVIDIA-GeForce-RTX-3090", 3: "NVIDIA-L40"})
    with pytest.raises(SystemExit, match="I7"):
        M2.build(root2)


def test_two_anomaly_sets_must_agree_on_every_run_they_share(tmp_path):
    root, M = _full(tmp_path)
    p = FX.v2(root, "anomaly") / "t12" / "summary" / "best70" / "features" / "anomaly_summary.json"
    d = json.loads(p.read_text())
    d["families"]["knn"]["label_X_bb"]["2000"]["levels"]["L188"]["ln_sigma_min"][0] += 0.1
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="disagree"):
        M.build(root)


# ------------------------------------------------------------------ RealData
def test_the_real_data_section_reads_the_second_grids_fit(full, tmp_path):
    root, M, built, m = full
    # the fixture's best70 fits ARE the first grid's: every first-grid number comes back
    v1 = json.loads((REPO / "paper/journal/provenance.json").read_text())
    for name in ("AojYieldOneeighteight", "AojYieldOnesevenMass", "AojStatErrOneseven", "AojPairedYieldOnesevenOverOneeighteight",
                 "AojPooledMean", "AojWeakestYield", "AojRefYield", "AojPublicYield", "AojInjEnsembleRecoveryMin",
                 "AojLeakPairedShiftMax", "AojNRunsWord", "AojFitPMin"):
        assert m[name] == v1[name]["value"], name
    assert _iv(m["AojYieldOneeighteightWavgShift"])[0] == pytest.approx(1.05)
    assert m["AojYieldOneeighteightWavgLabel"] == "depends on the checkpoint, under 10\\%"
    root2, M2 = _full(tmp_path)
    p = FX.v2(root2, "real_data") / "t12" / "fit_v6" / "results.json"
    p.write_text(p.read_text().replace('"eff"', '"eff" ', 1))
    with pytest.raises(SystemExit, match="is not the fit"):
        M2.build(root2)


# ------------------------------------------------------------------ the A14 count
def test_the_dependent_count_takes_this_scripts_labels_with_paired_errors(full):
    root, M, built, m = full
    per_model = sum(_ratios(root, f)["checkpoint_dependence"]["wavg/best70"]["models"]["n"]
                    for f in ("probes", "finetune", "anomaly"))
    derived = sum(int(m[f"CkptWavgModels{s}N"]) for s in ("AnomalyDetectors", "Benchmarks", "RealData"))
    assert int(m["CkptWavgModelsAllN"]) == per_model + derived
    assert m["CkptWavgModelsAllExpected"] == M.fmt(0.05 * (per_model + derived), 1)
    assert "anomaly detectors" in built["tables/checkpoint_robustness.tex"]


# ------------------------------------------------------------------ what the text still needs
def test_a_number_the_text_uses_and_the_second_grid_lacks_becomes_a_marker(tmp_path):
    # the text was written against the first grid's macros; a section now read from the
    # second grid defines every name it can, and the rest are red markers, never v1 numbers
    root, M = _full(tmp_path)
    j = root / "paper" / "journal"
    j.mkdir(parents=True)
    (j / "main.tex").write_text("\\FtAucJciiEThreeOneseven, \\AnomalyClassSumRemovedXBb and \\textbf{x}\n")
    (j / "provenance.json").write_text(json.dumps({n: {} for n in ("FtAucJciiEThreeOneseven",
                                                                  "AnomalyClassSumRemovedXBb")}))
    built, missing, _ = M.build(root)
    m = T2.macros(built)
    assert m["AnomalyClassSumRemovedXBb"] == "\\pending{no v2 number: rewrite this}"
    assert not m["FtAucJciiEThreeOneseven"].startswith("\\pending")
    assert "textbf" not in m and any(x.startswith("AnomalyClassSumRemovedXBb --") for x in missing)
    # a benchmark's full training set, not a power of ten, is named Full
    M2 = T2._generator(tmp_path / "x")
    assert M2.texname("Full") == "Full"
    assert any(k.startswith("BenchTopRfiftyEThree") for k in m) and not any("BenchTopRfiftyN" in k for k in m)


# ------------------------------------------------------------------ round 3: t3, not-computed, SSL spread
def test_real_data_reads_t12_alone_and_tier_three_only_from_t3(tmp_path):
    root, M = _full(tmp_path)
    m = T2.macros(M.build(root)[0])                         # t12 alone
    assert "AojYieldSixfour" not in m and m["AojYieldOneeighteight"]
    FX.write_real_data_t3(root)                              # t12 + t3, t3 at t12's shape
    built = M.build(root)[0]
    m3, prov = T2.macros(built), json.loads(built["provenance.json"])
    assert prov["AojYieldSixfour"]["source_file"].endswith("real_data/t3/analysis_v6/aoj_top.json")
    assert prov["AojYieldOneeighteight"]["source_file"].endswith("real_data/t12/analysis_v6/aoj_top.json")
    assert m3["AojYieldOneeighteight"] == m["AojYieldOneeighteight"] and "AojYieldSixfourWavgLabel" in m3


@pytest.mark.parametrize("break_it, msg", [
    (lambda root: FX.write_real_data_t3(root, held_sha="0" * 64), "not the freeze fit's"),
    (lambda root: shutil.copytree(FX.v2(root, "real_data") / "t12", FX.v2(root, "real_data") / "t123"), "holds"),
])
def test_a_tier_three_fit_not_at_the_freeze_shape_or_another_directory_is_refused(tmp_path, break_it, msg):
    root, M = _full(tmp_path)
    break_it(root)
    with pytest.raises(SystemExit, match=msg):
        M.build(root)


def test_every_result_paired_errors_did_not_compute_is_listed_and_printed(full, capsys):
    root, M, built, m = full
    nc = json.loads(built["v2_not_computed.json"])
    want = [(R_i, r) for R_i, r in enumerate(_ratios(root)["ratios"]) if "not_computed" in r]
    assert want and len([x for x in nc if x["file"].endswith("probes/ratios.json")]) == len(want)
    joint = [x for x in nc if x["contrast"].startswith("partition_joint")]
    assert joint and all(x["reason"] for x in joint)
    # a joint-fit row with no value gets no macro, and is listed instead
    rows = _ratios(root)["ratios"]
    for x in joint:
        r = rows[int(x["where"][7:-1])]
        if x["checkpoint"] == "best70" and r.get("task"):
            name = (("RandJointDose" if r["contrast"].endswith("dose") else "RandJoint")
                    + M.texname(*r["probe_pairs"]) + M.texname(r["kind"]))
            assert name not in m, name
    assert M.main(["--root", str(root), "--out", str(root / "out")]) == 0
    out = capsys.readouterr().out
    assert f"{len(nc)} v2 result(s) paired_errors.py did not compute" in out
    assert any(line.startswith("NOT COMPUTED") and "partition_joint" in line for line in out.splitlines())
    assert json.loads((root / "out" / "v2_not_computed.json").read_text()) == nc


def test_the_self_supervised_spread_states_which_sd_it_is(full):
    root, M, built, m = full
    z = [math.log(FX.ft_value("INIT", s, "N1000", 1, FX.SCRATCH_RANK)) for s in (1, 2, 3)]
    s = [math.log(FX.ft_value("MPM", k, "N1000", 1)) for k in (1, 2)]
    assert m["SslScratchSdJciiEThree"] == M.fmt(np.std(z, ddof=1), 3)
    assert m["SslRunSdJciiEThree"] == M.fmt(np.std(s, ddof=1), 3)
    larger = "the from-scratch fine-tuning seeds" if np.std(z, ddof=1) >= np.std(s, ddof=1) else "the self-supervised runs"
    assert m["SslSpreadSourceJciiEThree"] == larger
    assert m["SslSpreadJciiEThree"] == M.fmt(max(np.std(z, ddof=1), np.std(s, ddof=1)), 3)
