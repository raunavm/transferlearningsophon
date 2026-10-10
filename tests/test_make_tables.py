"""The generated-number pipeline: what it must refuse, and what it must not invent.

Every fixture here is synthetic and written to tmp_path in the schema the real
files use, so no real result is pinned in this file -- the two tests that touch
the committed inputs check a property (every macro is traceable, the checked-in
outputs match the inputs), never a value.

What each test guards, in the order the generator can get it wrong:
  * a rejection or an AUC that is a bound never prints as a bare number;
  * an input that does not exist yet produces no macro and one line of report;
  * two inputs that disagree on the row alignment stop the run;
  * --check fails when a committed output no longer follows from the inputs;
  * the .tex and provenance.json are two views of one list, not two lists.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

LEVELS = [12, 8, 4, 2]                       # the synthetic ladder, fine -> coarse
SEEDS = [1, 2, 3]
RUNG_SIZES = {"L188": 12, "L162": 8, "R63_Q1": 6, "R42_Q1": 4,
              "R29_Q1": 3, "R16_Q1": 2, "R3_VIS": 2, "R1_Q1": 1}
ARM_OF_LEVEL = {12: "l188", 8: "l162", 4: "r42q1", 2: "r16q1"}
TASKS = {"alpha": ["sig_a", "sig_b"], "beta": ["sig_c", "sig_d"]}
# Row of each named class in the synthetic label map. alpha's pair falls inside
# one R42_Q1 group (so it merges at a level the ladder measures); beta's pair
# survives to R1_Q1 (so it merges off the measured range).
CLASS_ROW = {"sig_a": 1, "sig_b": 2, "sig_c": 0, "sig_d": 6}
ALIGN = "a" * 64


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod("make_tables", "experiments/FIGS/make_tables.py")


# ------------------------------------------------------------------ fixture

def cell(task, probe, level, seed):
    """One probe cell, in probe.py's schema. Censored and bound cases are built in."""
    i = LEVELS.index(level)
    if task == "beta" and level in LEVELS[:2]:
        # saturated: the AUC hit 1 at the sample's resolution in every seed
        log1m, auc, censored = -14.0, 1.0, True
    else:
        log1m = -6.0 + 0.5 * i + 0.01 * seed + (0.05 if probe == "mlp" else 0.0)
        auc, censored = 1.0 - math.exp(log1m), False
    if task == "alpha" and probe == "linear" and level == LEVELS[0]:
        rej, bound = 1000.0, True                      # every seed at the cap
    elif task == "alpha" and probe == "linear" and level == LEVELS[1]:
        rej, bound = (900.0, True) if seed < 3 else (450.0, False)   # some seeds
    else:
        rej, bound = 500.0 - 100.0 * i + seed, False
    return {"auc": auc, "log1m_auc": log1m, "log1m_auc_censored": censored,
            "rejection": rej, "rejection_is_bound": bound, "rejection_eps_s": 0.5,
            "rejection_at": {"0.50": {"rejection": rej, "rejection_is_bound": bound}}}


def write_ladder(root):
    out = []
    for seed in SEEDS:
        doc = {"n_jets_total": 1000, "row_alignment_sha256": ALIGN, "eps_s_default": 0.5,
               "arm_checkpoints": {f"{ARM_OF_LEVEL[lv]}-s{seed}": "0" * 64 for lv in LEVELS},
               "tasks": {}}
        for task, names in TASKS.items():
            merged = [r for r in M.RUNGS
                      if len({CLASS_ROW[n] * RUNG_SIZES[r] // 12 for n in names}) == 1]
            doc["tasks"][task] = {
                "eps_s": [0.5], "n": 100, "n_signal": 50, "names": names,
                "collapsed_at": merged,
                "arms": {f"{ARM_OF_LEVEL[lv]}-s{seed}":
                         {p: cell(task, p, lv, seed) for p in ("linear", "mlp")}
                         for lv in LEVELS}}
        p = root / "experiments/FIGS/data/probe_ladder_v2_mlp2" / f"s{seed}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc))
        out.append(p)
    return out


def _rows(root):
    return [{"task": t, "probe": p, "level": lv, "seed": s,
             "arm": f"{ARM_OF_LEVEL[lv]}-s{s}", "file": f"s{s}.json", "dropped_pair": False,
             "log1m_auc": c["log1m_auc"], "censored": c["log1m_auc_censored"],
             "auc": c["auc"], "rejection": c["rejection"],
             "rejection_is_bound": c["rejection_is_bound"], "rejection_eps_s": 0.5}
            for t in TASKS for p in ("linear", "mlp") for lv in LEVELS for s in SEEDS
            for c in [cell(t, p, lv, s)]]


def _level_block(rows, task, probe):
    out = []
    for lv in LEVELS:
        rs = [r for r in rows if (r["task"], r["probe"], r["level"]) == (task, probe, lv)]
        y = [r["log1m_auc"] for r in rs]
        rej = [r["rejection"] for r in rs]
        mean = sum(y) / len(y)
        var = sum((v - mean) ** 2 for v in y) / (len(y) - 1)
        out.append({"level": lv, "n_seeds": len(rs), "mean": mean, "seed_sd": math.sqrt(var),
                    "mean_auc": sum(r["auc"] for r in rs) / len(rs),
                    "rejection_median": sorted(rej)[len(rej) // 2],
                    "rejection_range": [min(rej), max(rej)], "rejection_eps_s": 0.5,
                    "n_rejection_bound": sum(r["rejection_is_bound"] for r in rs),
                    "n_censored": sum(r["censored"] for r in rs)})
    return out


def _pairwise(rows, task, probe):
    out = []
    for i, fine in enumerate(LEVELS):
        for coarse in LEVELS[i + 1:]:
            d = [next(r["log1m_auc"] for r in rows
                      if (r["task"], r["probe"], r["level"], r["seed"]) == (task, probe, coarse, s))
                 - next(r["log1m_auc"] for r in rows
                        if (r["task"], r["probe"], r["level"], r["seed"]) == (task, probe, fine, s))
                 for s in SEEDS]
            mean = sum(d) / len(d)
            bound = any(r["censored"] for r in rows
                        if (r["task"], r["probe"]) == (task, probe)
                        and r["level"] in (fine, coarse))
            out.append({"fine": fine, "coarse": coarse, "n_pairs": len(d), "seeds": list(SEEDS),
                        "is_bound": bound, "estimable": True, "mean_diff": mean,
                        "sd_diff": 0.01, "t": 3.0, "df": len(d) - 1, "p": 0.02,
                        "ci95": [mean - 0.05, mean + 0.05],
                        "sign_flip": {"p": 0.25, "floor": 0.25, "n_arrangements": 8},
                        "holm_reject": coarse == LEVELS[-1]})
    return out


def write_analysis(root, ladder):
    rows = _rows(root)
    trend = {"task": "alpha", "probe": "linear", "levels_fine_to_coarse": list(LEVELS),
             "alternative": "log1m_auc increasing along levels = performance falls with "
                            "coarser labels",
             "blocks_used": list(SEEDS), "blocks_dropped_incomplete": {}, "run": True,
             "family": "marcus", "method": "exact", "n_blocks": 3, "n_arrangements": 13824,
             "stat": 9.5, "p": 0.0004, "p_min": 1e-5, "end_step_p_bound": 0.015625,
             "argmax_step": [LEVELS[:3], LEVELS[3:]], "predicted_step": None,
             "argmax_is_predicted_step": None, "contrasts_localisation_only": [],
             "n_censored_cells": 0, "seed_sd_per_level": [0.01] * len(LEVELS),
             "isotonic": {"stat": 2.0, "p": 0.001, "fitted": [], "pooled": [], "means": []}}
    tost = {"task": "beta", "probe": "linear", "target_bound": 0.0953, "log_base": "e",
            "n_pairs_estimable": 6, "n_pairs_total": 6, "run": True,
            "largest_pair": {"fine": LEVELS[1], "coarse": LEVELS[-1], "n_pairs": 3,
                             "is_bound": True, "mean_diff": 8.5, "p": 0.99,
                             "ci90": [8.0, 9.0], "equivalent_at_target": False,
                             "smallest_bound_passed": 9.0},
            "p": 0.99, "verdict": "inconclusive", "smallest_bound_passed_by_all": 9.0,
            "pairs": []}
    doc = {
        "provenance": {
            "inputs": [{"path": str(p.relative_to(root)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in ladder],
            "script_sha256": "0" * 64, "prespec_sha256": None, "stats_modules_sha256": {},
            "row_alignment_sha256": ALIGN, "n_jets_total": 1000,
            "arm_checkpoints": {}, "argv": []},
        "endpoint": {"field": "log1m_auc", "log_base": "e (natural log, probe.py np.log)",
                     "lower_is_better": True, "difference": "coarser − finer, by seed index"},
        "levels_fine_to_coarse": list(LEVELS), "seeds_used": list(SEEDS),
        "dropped_pairs": {"seeds": [], "reason": None}, "missing_cells": {},
        "skipped_tasks": [], "table": rows,
        "mde": [{"task": t, "probe": p, "n_pairs": 3, "seeds": list(SEEDS), "estimable": True,
                 "sd_paired_diff": 0.02, "multiplier": 2.5, "mde": 0.05,
                 "mde_as_ratio_of_1m_auc": 1.05} for t in TASKS for p in ("linear", "mlp")],
        "levels": {t: {p: _level_block(rows, t, p) for p in ("linear", "mlp")} for t in TASKS},
        "confirmatory": {"C1": trend,
                         "holm_family": [
                             {"test": "C1", "status": "available", "p_raw": 0.0004,
                              "family_size": 2, "threshold_if_smallest": 0.025,
                              "reject_whatever_pending": True, "reject_possible": True},
                             {"test": "C2", "status": "pending", "p_raw": None,
                              "family_size": 2, "threshold_if_smallest": 0.025,
                              "reject_whatever_pending": None, "reject_possible": None}]},
        "secondary": {"S1": {**trend, "task": "beta"},
                      "S2": {"beta": {"linear": tost, "mlp": {**tost, "probe": "mlp"}}},
                      "S6": {"alpha": {"n_agree": 5, "n_pairs": 6, "pairs": []}},
                      "holm_family": [{"test": "S1 beta", "status": "available", "p_raw": 0.0004,
                                       "family_size": 2, "threshold_if_smallest": 0.025,
                                       "reject_whatever_pending": True, "reject_possible": True},
                                      {"test": "S2 beta", "status": "available", "p_raw": 0.99,
                                       "family_size": 2, "threshold_if_smallest": 0.025,
                                       "reject_whatever_pending": False,
                                       "reject_possible": False}]},
        "pairwise_exploratory": {t: {p: _pairwise(rows, t, p) for p in ("linear", "mlp")}
                                 for t in TASKS}}
    p = root / "experiments/FIGS/data/probe_ladder_v2_mlp2/analysis/seed_level_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def write_rung_map(root):
    """A 12-class label map whose group columns have the sizes RUNG_SIZES names."""
    names = {v: k for k, v in CLASS_ROW.items()}
    head = ["jet_label", "class_name"] + [c for r in M.RUNGS for c in (r, f"{r}_name")]
    lines = [",".join(head)]
    for i in range(12):
        row = [str(i), names.get(i, f"c{i:02d}")]
        for r in M.RUNGS:
            g = i * RUNG_SIZES[r] // 12
            row += [f"g{g}", f"{r}_g{g}"]
        lines.append(",".join(row))
    p = root / "configs/labelmaps/rung_label_maps.v1.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def write_extras(root):
    surv = {"sophon_eq4": {"title": "X->bb vs QCD", "source": "arXiv:0000.00000 Eq. (1)",
                         "constructible": {r: r in M.RUNGS[:2] for r in M.RUNGS},
                         "n_coefficient_vectors": 2, "dies_at": M.RUNGS[2],
                         "last_rung_alive": M.RUNGS[1]}}
    p = root / "configs/labelmaps/usecase_survival.v1.json"
    p.write_text(json.dumps(surv))
    legs = []
    for leg in (1, 2):
        d = {"row_alignment_sha256": ALIGN,
             "summary": {"l162-s1b": {"N10000": {"n_seeds": 3, "accuracy_mean": 0.5 + 0.01 * leg,
                                                 "accuracy_sd": 0.002}},
                         "scratch": {"N10000": {"n_seeds": 1, "accuracy_mean": 0.4,
                                                "accuracy_sd": None}}}}
        q = root / f"experiments/FIGS/data/leg{leg}_metrics.json"
        q.write_text(json.dumps(d))
        legs.append(q)
    return p, legs


@pytest.fixture
def root(tmp_path):
    ladder = write_ladder(tmp_path)
    write_analysis(tmp_path, ladder)
    write_rung_map(tmp_path)
    write_extras(tmp_path)
    return tmp_path


# ------------------------------------------------------------------ tests

def macros_in(text):
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  %", text))


def test_a_rejection_that_is_a_bound_never_prints_as_a_bare_number(root):
    """The JSON flags it; a bare number in the paper would be a claim we cannot make.

    No background jet passed the cut, so the value is the sample size. All seeds
    at the cap makes the median a bound too ($>$); only some makes it uncertain
    ($\\geq$ plus a footnote); none leaves it a measurement.
    """
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeRejAlphaLinearOnetwo"] == "\\ensuremath{>}333"     # N_B/3, N_B = 1000
    assert m["ProbeRejAlphaLinearEight"].startswith("\\ensuremath{\\geq}")
    assert "^{\\ast}" in m["ProbeRejAlphaLinearEight"]
    assert m["ProbeRejAlphaLinearFour"] == "302.0\\,\\ensuremath{\\pm}\\,1.0"   # 301, 302, 303
    table = built["tables/probes_linear.tex"]
    assert "$>$333" in table and "$\\geq$900$^{\\ast}$" in table
    assert "95\\% confidence lower limit" in table        # the footnote is not optional


def test_a_saturated_auc_is_marked_rather_than_printed_to_five_decimals(root):
    """AUC = 1 at the sample's resolution is a bound on 1-AUC, not a measurement."""
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeAucBetaLinearOnetwo"] == "\\ensuremath{1^{\\dagger}}"
    assert m["ProbeAucAlphaLinearOnetwo"].startswith("0.")
    table = built["tables/probes_linear.tex"]
    assert "$1^{\\dagger}$" in table and "1.00000" not in table


def test_a_missing_input_is_reported_not_invented(root):
    """No placeholder, no macro, one line in the report naming the file."""
    (root / "experiments/FIGS/data/leg1_metrics.json").unlink()
    (root / "experiments/FIGS/data/leg2_metrics.json").unlink()
    (root / "configs/labelmaps/usecase_survival.v1.json").unlink()
    built, missing, skipped = M.build(root)
    assert "tables/finetuning_wave1.tex" not in built
    assert "tables/usecase_survival.tex" not in built
    tex = built["results_generated.tex"]
    assert not [n for n in macros_in(tex) if n.startswith(("AccLeg", "UseDiesAt"))]
    assert any("leg" in x for x in missing) and any("usecase_survival" in x for x in missing)
    for entry in missing:                    # the report reaches the .tex, not just stdout
        assert entry in tex
    # the inputs that never existed are named too, and T4 says why it is absent
    assert any("label_recovery_curve_v1err" in x for x in missing)
    assert any("bench_metrics" in x for x in missing)
    assert any("anomaly_v1err" in x for x in missing)
    assert any("T4" in s for s in skipped)


def test_inputs_that_disagree_on_the_row_alignment_stop_the_run(root):
    """Different alignments means different jets: a paired contrast across them
    is not a paired contrast, and every number here assumes it is one."""
    p = root / "experiments/FIGS/data/probe_ladder_v2_mlp2/s2.json"
    d = json.loads(p.read_text())
    d["row_alignment_sha256"] = "b" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        M.build(root)
    assert "row_alignment_sha256" in str(e.value)


def test_a_ladder_file_rewritten_since_the_analysis_stops_the_run(root):
    """Then the p-values describe data that is no longer on disk."""
    p = root / "experiments/FIGS/data/probe_ladder_v2_mlp2/s2.json"
    d = json.loads(p.read_text())
    d["n_jets_total"] = 1001
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        M.build(root)
    assert "changed since the analysis ran" in str(e.value)


def test_check_mode_is_clean_after_a_write_and_fails_after_drift(root, tmp_path):
    out = tmp_path / "journal"
    assert M.main(["--root", str(root), "--out", str(out)]) == 0
    assert M.main(["--root", str(root), "--out", str(out), "--check"]) == 0
    # an input changes -> the committed output no longer follows from it
    p = root / "experiments/FIGS/data/leg1_metrics.json"
    d = json.loads(p.read_text())
    d["summary"]["scratch"]["N10000"]["accuracy_mean"] = 0.41
    p.write_text(json.dumps(d))
    assert M.main(["--root", str(root), "--out", str(out), "--check"]) == 1
    # and so does a hand-edit of a generated file
    assert M.main(["--root", str(root), "--out", str(out)]) == 0
    (out / "tables/probes_linear.tex").write_text("hand edited\n")
    assert M.main(["--root", str(root), "--out", str(out), "--check"]) == 1


def test_check_mode_reports_a_file_that_was_never_generated(root, tmp_path):
    assert M.main(["--root", str(root), "--out", str(tmp_path / "empty"), "--check"]) == 1


@pytest.mark.parametrize("where", ["fixture", "repo"])
def test_every_macro_is_in_provenance_and_every_entry_is_a_macro(root, where):
    """The .tex and provenance.json are two views of one list. If a macro can be
    in one and not the other, the hash beside a number stops meaning anything."""
    built = M.build(root if where == "fixture" else REPO)[0]
    tex, prov = built["results_generated.tex"], json.loads(built["provenance.json"])
    names = macros_in(tex)
    assert set(names) == set(prov)
    assert len(names) == len(re.findall(r"(?m)^\\newcommand", tex))
    for name, body in names.items():
        entry = prov[name]
        assert entry["value"] == body
        assert set(entry) == {"source_file", "json_path", "sha256", "value"}
        src = (root if where == "fixture" else REPO) / entry["source_file"]
        assert hashlib.sha256(src.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["sha256"][:16] in tex.split(f"\\{name}}}")[1].split("\n")[0]


def test_macro_names_are_legal_latex(root):
    """\\newcommand takes letters only, so digits are spelled out. A name with a
    digit in it compiles to a confusing error, at the reviewer's end."""
    prov = json.loads(M.build(root)[0]["provenance.json"])
    assert prov, "the fixture produced no macros at all"
    for name in prov:
        assert re.fullmatch(r"[A-Za-z]+", name), name
    assert "ProbeAucAlphaLinearOnetwo" in prov       # 12 -> Onetwo


def test_the_committed_outputs_still_follow_from_the_committed_inputs():
    """CI's job: the paper's numbers are the numbers in the result files today."""
    assert M.main(["--check"]) == 0, "run python3 experiments/FIGS/make_tables.py"


def _with_points(root, points, per_seed):
    """Give the fixture analysis the working-point block the real one carries,
    in the per-level summary and in each per-seed row: `per_seed` maps a seed to
    its (rejection, is_bound) at 90 %."""
    p = root / "experiments/FIGS/data/probe_ladder_v2_mlp2/analysis/seed_level_results.json"
    A = json.loads(p.read_text())
    for r in A["table"]:
        rej, bound = per_seed[r["seed"]]
        r["rejection_points"] = {
            "0.50": {"rejection": r["rejection"], "is_bound": r["rejection_is_bound"]},
            "0.90": {"rejection": rej, "is_bound": bound}}
    for task in A["levels"]:
        for probe in A["levels"][task]:
            for row in A["levels"][task][probe]:
                row["headline_eps_s"] = "0.90"
                row["rejection_points"] = {
                    "0.50": {"n_seeds": row["n_seeds"], "median": row["rejection_median"],
                             "range": row["rejection_range"],
                             "n_bound": row["n_rejection_bound"],
                             "mean_n_bkg_pass": 0.0, "is_headline": False},
                    "0.90": {"n_seeds": row["n_seeds"], **points,
                             "mean_n_bkg_pass": 40.0, "is_headline": True}}
    p.write_text(json.dumps(A))


def test_the_paper_quotes_rejection_at_the_headline_working_point(root):
    """docs/PRESPEC_2026-09.md fixed 90 % signal efficiency as the headline,
    blind, because at 50 % no background jet survives at the three finer
    vocabularies and the number is then the size of the test sample. The flat
    `rejection_*` fields sit at 50 %, so reading them would put the censored
    number in the paper under a caption claiming it is a measurement."""
    _with_points(root, {"median": 321.0, "range": [300.0, 350.0], "n_bound": 0},
                 {1: (300.0, False), 2: (321.0, False), 3: (350.0, False)})
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeEpsS"] == "90\\%"
    assert m["ProbeRejAlphaLinearOnetwo"] == "324\\,\\ensuremath{\\pm}\\,25", (
        "the mean +- SD of the 0.90 values, not the 1,000.0 bound the flat fields hold at 0.50")
    prov = json.loads(built["provenance.json"])
    assert "rejection_points['0.90']" in prov["ProbeRejAlphaLinearOnetwo"]["json_path"]
    assert "90\\% signal efficiency" in built["tables/probes_linear.tex"]


def test_a_censored_headline_cell_still_never_prints_as_a_bare_number(root):
    """If the headline point turns out to be censored too, that is the finding
    and the table must say so -- it must not silently fall back to a number."""
    _with_points(root, {"median": 11876.0, "range": [11876.0, 11876.0],
                        "n_bound": len(SEEDS)}, {s: (11876.0, True) for s in SEEDS})
    m = macros_in(M.build(root)[0]["results_generated.tex"])
    assert m["ProbeRejAlphaLinearOnetwo"] == "\\ensuremath{>}3{,}963"   # 11876 / 2.996


def test_an_analysis_without_working_points_falls_back_and_says_which_point(root):
    """Analysis files written before the working points were recorded must keep
    building, at the point they actually hold."""
    built, _, _ = M.build(root)          # fixture has no rejection_points
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeEpsS"] == "50\\%"
    assert json.loads(built["provenance.json"])[
        "ProbeRejAlphaLinearFour"]["json_path"].endswith(".rejection")


# ---------------------------------------------------------------- later results

def _repo_macros():
    built = M.build(REPO)[0]
    return macros_in(built["results_generated.tex"]), json.loads(built["provenance.json"])


def _load(rel):
    return json.loads((REPO / rel).read_text())


def test_later_results_are_the_stored_values_re_read_independently():
    """Each check reads the result file itself, without the generator's helpers
    for reading it, and compares with the macro the paper would print."""
    got, prov = _repo_macros()
    import numpy as np
    ft = _load(prov["FtAucJciiEFourOneeighteight"]["source_file"])["cells"]
    assert {c["s1"]["n_classes_present"] for c in ft["l188-s1"].values()} == {162}
    v = [ft[f"l188-s{s}"]["N10000"]["s1"]["macro_auc_ovr"] for s in range(1, 6)]
    assert got["FtAucJciiEFourOneeighteight"] == M.math_safe(M.fmt_pm(np.mean(v), np.std(v, ddof=1)))
    c4 = _load(prov["RandOmaBvcFourprongDrawThreeLinear"]["source_file"])
    row = next(r for r in c4["table"]
               if (r["task"], r["probe"], r["draw"]) == ("bvc_4prong", "linear", 3))
    assert got["RandOmaBvcFourprongDrawThreeLinear"] == M.math_safe(
        M.fmt_one_sci(math.exp(row["control_log1m_auc"])))
    # sigma_min straight from the per-model results, not from the summary file.
    raw = _load(_load(prov["AnomalySigmaMinMahalanobisXYYBbbbOneseven"]["source_file"])
                ["provenance"]["inputs"]["anomaly"]["path"])
    v = [raw["arms"][f"r16q1-s{s}"]["signals"]["label_X_YY_bbbb"]["2000"]["mahalanobis"]["sigma_min"]
         for s in range(1, 6)]
    assert got["AnomalySigmaMinMahalanobisXYYBbbbOneseven"] == M.math_safe(
        M.fmt_pm(np.mean(v), np.std(v, ddof=1)))
    s7 = _load(prov["MassResSigmaEffOneseven"]["source_file"])["table"]
    v = [r["sigma_eff"] for r in s7 if r["cell"] == "17" and r["probe"] == "ridge"]
    assert len(v) == 5 and got["MassResSigmaEffOneseven"] == f"{sum(v) / 5:.4f}"
    d = _load(prov["MassResNTest"]["source_file"])["centering_detail"]
    n = d["n_jets_used"]
    assert got["MassResNTest"] == M.fmt_int(n - int(0.8 * n)) == M.fmt_int(d["split"][2])
    # the yield from the fits themselves, not from the analysis file's summary; the
    # readout names the fit by the scratch path it ran at, the committed copy by hash
    J = _load(prov["AojYieldOneeighteight"]["source_file"])
    fits = json.loads(M.committed_path(REPO, J["provenance"]["input"],
                                       J["provenance"]["input_sha256"]).read_text())["models"]
    y = [fits[f"l188-s{s}"]["top"]["signal_yield"] for s in range(1, 6)]
    assert got["AojYieldOneeighteight"] == M.math_safe(M.fmt_pm(np.mean(y), np.std(y, ddof=1)))


def test_the_design_numbers_are_the_ones_the_pretraining_job_ran_with():
    got, prov = _repo_macros()
    spec = (REPO / prov["DesignEpochs"]["source_file"]).read_text()
    assert f"--num-epochs {got['DesignEpochs']}" in spec
    assert f"--batch-size {got['DesignBatchSize']}" in spec
    assert got["DesignExamplesSeen"] == "\\ensuremath{8.192\\times10^{8}}"   # docs/GROUND_TRUTH.md
    assert (got["DesignParticleBlocks"], got["DesignClassBlocks"], got["DesignEmbedDim"]) == \
        ("8", "2", "128")                                          # docs/GROUND_TRUTH.md


def test_the_design_parser_refuses_a_flag_given_two_values(tmp_path):
    spec = tmp_path / "job.yaml"
    spec.write_text("--num-epochs 80 --num-epochs 60 --samples-per-epoch 10 --batch-size 1 "
                    "--start-lr 5e-4 -o fc_params '[(512,0.1)]'")
    em = M.Emitter(tmp_path)
    with pytest.raises(SystemExit, match="--num-epochs"):
        M.emit_training_design(em, spec, spec, spec)


def test_a_later_analysis_whose_input_changed_stops_the_run(tmp_path):
    (tmp_path / "in.json").write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    a = tmp_path / "analysis.json"
    a.write_text(json.dumps({"provenance": {"inputs": [{"path": "in.json", "sha256": sha}]}}))
    M.check_inputs_unchanged(a, tmp_path)
    (tmp_path / "in.json").write_text("{ }")
    with pytest.raises(SystemExit, match="changed since"):
        M.check_inputs_unchanged(a, tmp_path)
    a.write_text("{}")
    with pytest.raises(SystemExit, match="records no input hash"):
        M.check_inputs_unchanged(a, tmp_path)


def test_every_literature_number_is_in_its_quoted_passage():
    got, prov = _repo_macros()
    facts = _load(prov["LitJetClassTwoNJetsMillion"]["source_file"])["facts"]
    assert facts, "no literature facts"
    for f in facts:
        assert got[f["macro"]] == M.math_safe(f.get("tex", f["value"])) and f["value"] in f["quote"]
    assert got["LitJetClassTwoNResonantClasses"] == "161"      # docs/GROUND_TRUTH.md
    assert got["LitJetClassTwoNQcdClasses"] == "27"            # docs/GROUND_TRUTH.md


def test_a_literature_number_missing_from_its_quote_is_refused(tmp_path):
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"facts": [{"macro": "LitX", "value": "140", "source": "s",
                                        "location": "l", "quote": "around 139 M"}]}))
    with pytest.raises(SystemExit, match="must appear in its quote"):
        M.emit_literature(M.Emitter(tmp_path), p)


def test_the_mass_loss_weight_is_read_from_all_ten_specs_and_they_must_agree(tmp_path):
    got, _ = _repo_macros()
    assert got["DesignMassLambda"] == "5.0"                    # docs/DECISIONS.md D2
    specs = []
    for i in range(10):
        s = tmp_path / f"job-mtx-x_mass-s{i}-raunav.yaml"
        s.write_text(f"--mass-lambda {'5.0' if i else '0.05'}")
        specs.append(s)
    with pytest.raises(SystemExit, match="one --mass-lambda"):
        M.emit_mass_lambda(M.Emitter(tmp_path), specs)


REPORTED = {"l162-s2", "r16q1-s2"}
EPOCHS = {1_000: "50", 10_000: "50", 100_000: "30", 1_000_000: "10"}


def _recipes(tmp_path):
    """One recorded run per reported cell (two legs, four sizes), plus what the
    table must leave out: an interrupted attempt, another fine-tuning seed, a
    model the paper does not report, a benchmark run."""
    run = {"path": "/data/results/ft/w2b/x/ft_manifest.json", "leg": "1", "init": "l162-s2",
           "n_train": "1000", "ft_seed": "1", "lr": "1e-4", "head_lr_mult": "50", "epochs": "50",
           "lr_schedule": None, "weight_decay": "0.01", "batch_size": "512"}
    runs = [{**run, "leg": leg, "init": i, "n_train": str(n), "epochs": e}
            for leg in ("1", "2") for i in sorted(REPORTED) for n, e in EPOCHS.items()]
    runs += [{**run, "path": "/data/results/ft/w2b/leg1/l162-s2/N1000/s1.partial.17/ft_manifest.json",
              "lr": "3e-4"},
             {**run, "ft_seed": "2", "lr": "3e-4"},
             {**run, "init": "scratch-v2", "lr": "1e-3", "head_lr_mult": "1"},
             {**run, "leg": "top", "lr_schedule": "constant", "epochs": "20"}]
    leg = tmp_path / "leg.yaml"
    leg.write_text("--use-amp --optimizer ranger LR=1e-4 --samples-per-epoch-val 20000\n"
                   "samples_for () { case $1 in 1000) echo 10000;; *) echo $1;; esac; }")
    return {"runs": runs}, [leg]


def test_the_fine_tuning_table_states_the_settings_of_the_reported_runs_only(tmp_path):
    R, legs = _recipes(tmp_path)
    rec = M.ft_recipe(R, legs, REPORTED)
    assert (rec["lr"], rec["head_mult"], rec["n_runs"]) == ("1e-4", "50", 16)
    assert rec["epochs_jetclass"] == EPOCHS
    # weaver's steps are samples_per_epoch // batch; the smallest set is cycled
    assert rec["steps"][1000] == 10000 // 512 and rec["passes"][1000] == 10
    assert rec["val_jetclass"] == str(20000 // 512 * 512)
    got, prov = _repo_macros()
    assert got["FtRecipeLrHead"] == "\\ensuremath{5\\times10^{-3}}"
    assert "FtRecipeLrScratch" not in got          # no from-scratch row is reported yet


def test_a_run_that_departs_from_its_group_stops_the_table(tmp_path):
    R, legs = _recipes(tmp_path)
    R["runs"][0]["lr"] = "3e-4"
    with pytest.raises(SystemExit, match="disagree within a group"):
        M.ft_recipe(R, legs, REPORTED)


def test_a_reported_cell_without_exactly_one_recorded_run_stops_the_table(tmp_path):
    R, legs = _recipes(tmp_path)
    with pytest.raises(SystemExit, match="one to one"):
        M.ft_recipe({"runs": R["runs"][1:]}, legs, REPORTED)
    with pytest.raises(SystemExit, match="one to one"):
        M.ft_recipe({"runs": R["runs"] + [R["runs"][0]]}, legs, REPORTED)


def test_a_command_with_a_schedule_flag_contradicts_the_jetclass_row(tmp_path):
    R, legs = _recipes(tmp_path)
    legs[0].write_text(legs[0].read_text() + " --lr-scheduler none")
    with pytest.raises(SystemExit, match="not the JetClass recipe"):
        M.ft_recipe(R, legs, REPORTED)


def test_a_vcb_rejection_that_is_a_bound_in_any_seed_is_printed_as_a_bound(tmp_path):
    sizes = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}

    def cell(log1m, rej, bound):
        return {"log1m_auc": log1m, "log1m_auc_censored": False,
                "rejection_at": {"0.60": {"rejection": rej, "rejection_is_bound": bound,
                                          "n_bkg_pass": 0 if bound else 3}}}
    arms = {"l162-s1b": cell(-3.0, 1000.0, False), "l162-s2": cell(-3.1, 1500.0, True),
            "l162-s3": cell(-3.2, 1500.0, True),
            "r16q1-s1": cell(-2.5, 1500.0, True), "r16q1-s2": cell(-2.5, 1500.0, True),
            "r16q1-s3": cell(-2.4, 1500.0, True)}
    V = {"tasks": {"bc_vs_rest": {"n_signal_test": 100, "n_background_test": 1500, "eps_s": [0.6],
                                  "arms": {a: {"linear": c, "mlp": c} for a, c in arms.items()}}}}
    src = tmp_path / "sall.json"
    src.write_text(json.dumps(V))
    em = M.Emitter(tmp_path)
    M.emit_vcb(em, V, src, sizes)
    got = {name: body for name, body, _ in em.macros}
    assert got["VcbRejLinearSixtyOnesixtwo"] == M.math_safe("$\\geq$1{,}500$^{\\ast}$")  # two of three bound
    assert got["VcbRejLinearSixtyOneseven"] == M.math_safe("$>$500")        # every seed bound: N_B/3
    assert got["VcbEpsSixty"] == "60" and "VcbEpsForty" not in got
    oma = lambda xs: sum(math.exp(x) for x in xs) / len(xs)
    assert got["VcbOmaRatioLinear"] == M.fmt_ratio(oma([-2.5, -2.5, -2.4]) / oma([-3.0, -3.1, -3.2]))


# ---------------------------------------------------------------- descriptive reporting

@pytest.mark.parametrize("sd, want", [(351, "350"), (0.071, "0.07"), (0.97, "1.0"),
                                      (0.0000745, "0.00007"), (0.09996, "0.10"), (25.1, "25")])
def test_the_spread_is_rounded_by_the_pdg_rule(sd, want):
    """100-354: two significant figures; 355-949: one; 950-999: up to 1000, two."""
    got, place = M.pdg(sd)
    assert M._fixed(got, place) == want


def test_the_mean_is_printed_to_the_place_of_its_rounded_spread():
    assert M.fmt_pm(1377, 351) == "1{,}380\\,$\\pm$\\,350"
    assert M.fmt_pm(0.997684, 0.0000745) == "0.99768\\,$\\pm$\\,0.00007"
    assert M.fmt_pm(0.5, 0.97) == "0.5\\,$\\pm$\\,1.0"
    assert M.fmt_pm_sci(2.3213e-3, 0.0712e-3) == "$(2.32 \\pm 0.07)\\times10^{-3}$"
    assert M.fmt_pm(851.3, 0.0) == "851"                 # seeds that agree exactly
    assert M.fmt_one(0.910295) == "0.910" and M.fmt_one(1319.6) == "1{,}320"
    assert M.fmt_ratio(3.72) == "3.7" and M.fmt_ratio(1.0213) == "1.02"


def test_a_bound_is_never_averaged_into_a_mean():
    """All seeds bound: the 95% lower limit N_B/3. Some: the median, marked. None: mean +- SD."""
    assert M.fmt_rejection([2554.0] * 5, [True] * 5) == "$>$852"                     # N_B/2.996
    assert M.fmt_rejection([2554.0] * 5, [True] * 5, [0, 1, 0, 0, 0]) == "$>$538"      # one passed: N_B/4.744
    assert M.fmt_rejection([900.0, 900.0, 450.0], [True, True, False]) == "$\\geq$900$^{\\ast}$"
    assert M.fmt_rejection([301.0, 302.0, 303.0], [False] * 3) == "302.0\\,$\\pm$\\,1.0"


FORBIDDEN = re.compile(r"Holm|\$p\$|p=|trend|registered|clause|verdict|\b(?:C[1-5]|S(?:10|[1-9]))\b")


def test_no_generated_file_carries_test_language_or_prediction_labels():
    built = M.build(REPO)[0]
    bad = []
    for rel, text in built.items():
        lines = ([f"{k} {json.dumps(v)}" for k, v in json.loads(text).items()]
                 if rel == "provenance.json" else text.splitlines())
        bad += [f"{rel}: {m.group()!r} in {ln[:120]!r}" for ln in lines
                for m in [FORBIDDEN.search(ln)] if m]
    assert not bad, "\n".join(bad)
    assert "tables/tests.tex" not in built and "tables/s8.tex" not in built


# ---------------------------------------------------------------- paired ratios (audit must-fix 5)

@pytest.mark.parametrize("r, lo, hi, want", [
    (3.6639, 2.9676, 4.5236, "3.7~[3.0, 4.5]"),
    (1.0681, 1.0204, 1.1180, "1.07~[1.02, 1.12]"),
    (0.8393, 0.7713, 0.9134, "0.84~[0.77, 0.91]"),
    # 0.998 would round to 1.00 and read as touching 1: one more place
    (0.9832, 0.9686, 0.9980, "0.983~[0.969, 0.998]"),
    (1.0016, 0.9978, 1.0055, "1.002~[0.998, 1.006]"),
    # 1.0547 would round onto the point 1.05: one more place
    (1.0472, 1.0431, 1.0547, "1.047~[1.043, 1.055]")])
def test_a_paired_ratio_prints_its_point_and_interval_and_never_moves_a_bound_across_1(r, lo, hi, want):
    assert M.fmt_paired(r, lo, hi) == want


def _ratios(tmp_path, entries):
    p = tmp_path / "ratios.json"
    p.write_text(json.dumps({"ratios": entries}))
    return p


def _entry(**kw):
    e = {"family": "probe", "task": "bvc_resonant", "kind": "linear", "metric": "1-auc",
         "fine": "43", "coarse": "17", "n_runs": 5, "ratio": 3.66, "ci95": [2.97, 4.52],
         "ln_combined_se": 0.097, "z": 13.4, "dof": 40.0}
    return {**e, **kw}


def test_paired_macros_are_named_by_what_they_compare_and_skip_what_is_not_a_measurement(tmp_path):
    p = _ratios(tmp_path, [
        _entry(),
        _entry(coarse="162+mass", fine="162", ratio=1.11, ci95=[1.06, 1.17]),
        _entry(coarse="random draw 3", fine="random draw 1", n_runs=1, ratio=0.95, ci95=[0.85, 1.06]),
        _entry(family="ft", task="leg1", kind="N1000", metric="1-macro_auc", fine="188",
               coarse="random", ratio=0.82, ci95=[0.73, 0.93]),
        _entry(family="mass", task="resolution", kind="mlp", metric="sigma_eff", fine="17",
               coarse="17+mass", ratio=0.83, ci95=[0.82, 0.84]),
        _entry(task="ee_vs_mm", coarse="17", fine="188", ratio=9e5, ci95=[7e5, 1.3e6],
               censored_models=["l188-s1"]),                       # floored: not a measurement
        _entry(metric="eps_b@0.90")])                               # a metric the text never quotes
    em = M.Emitter(tmp_path)
    M.emit_paired(em, [p], None)
    got = {n: b for n, b, _ in em.macros}
    assert got == {"PairedProbeBvcResonantLinearOnesevenOverFourthree": "3.7~[3.0, 4.5]",
                   "PairedProbeBvcResonantLinearOnesixtwoMassOverOnesixtwo": "1.11~[1.06, 1.17]",
                   "PairedProbeBvcResonantLinearRandomDrawThreeOverRandomDrawOne": "0.95~[0.85, 1.06]",
                   "PairedFtJciiEThreeRandomOverOneeighteight": "0.82~[0.73, 0.93]",
                   "PairedMassResMlpOnesevenMassOverOneseven": "0.83~[0.82, 0.84]"}


def test_every_paired_macro_is_the_stored_ratio_and_interval_re_read_independently():
    got, prov = _repo_macros()
    # the ratios of experiments/STATS/paired_errors.py; the run-by-run intervals the
    # generator forms itself are re-derived in their own test below
    paired = {n: e for n, e in prov.items() if n.startswith("Paired") and e["json_path"].startswith("ratios[")}
    assert paired, "no paired ratio reached the paper"
    for name, e in paired.items():
        i = int(re.match(r"ratios\[(\d+)\]", e["json_path"]).group(1))
        r = _load(e["source_file"])["ratios"][i]
        assert got[name] == M.fmt_paired(r["ratio"], *r["ci95"]), name
    # the audit's cell: b vs c two-prong, 43 over 162 classes, linear probe
    assert got["PairedProbeBvcResonantLinearFourthreeOverOnesixtwo"].startswith("1.07~[1.02,")


# ---------------------------------------------------------------- the matched mass weight (PRESPEC A11)

def _lambda_files(tmp_path, lam_grid=1.74):
    x162, x17 = [0.1731, 0.1728, 0.1731, 0.1719, 0.1726], [0.4965, 0.4948, 0.4961, 0.4956, 0.4953]
    L = tmp_path / "mass_lambda.v2.json"
    L.write_text(json.dumps({"lambda_m": 1.74, "variants": {"mean_over_epochs_0_79": {
        "x_162_per_run": x162, "x_17_per_run": x17}}}))
    G = tmp_path / "v2_grid.json"
    G.write_text(json.dumps({"arms": [{"name": "L162_MASS", "mass_lambda": 5.0},
                                      {"name": "R16_Q1_MASS", "mass_lambda": 5.0},
                                      {"name": "R16_Q1_MASS_LM", "mass_lambda": lam_grid}]}))
    return L, G, x162, x17


def test_the_matched_weight_is_the_one_the_grid_trains_and_the_shares_are_x_over_1_plus_x(tmp_path):
    import numpy as np
    L, G, x162, x17 = _lambda_files(tmp_path)
    em = M.Emitter(tmp_path)
    M.emit_mass_lambda_matched(em, L, G)
    got = {n: b for n, b, _ in em.macros}
    assert got["MassLambdaMatched"] == "1.74"
    s = [100 * x / (1 + x) for x in x162]
    assert got["MassLossShareOnesixtwoMass"] == M.math_safe(
        M.fmt_pm(np.mean(s), np.std(s, ddof=1)) + "\\%")
    L, G, *_ = _lambda_files(tmp_path, lam_grid=1.75)
    with pytest.raises(SystemExit, match="does not train the matched weight"):
        M.emit_mass_lambda_matched(M.Emitter(tmp_path), L, G)


@pytest.mark.skip(reason="cut from the grid and the paper on 2026-10-09 (configs/arms/v2_grid.launched.json keeps the design)")
def test_the_paper_prints_the_trained_weight_from_its_committed_source():
    got, prov = _repo_macros()
    grid = {a["name"]: a for a in _load("configs/arms/v2_grid.json")["arms"]}
    assert got["MassLambdaMatched"] == f"{grid['R16_Q1_MASS_LM']['mass_lambda']:g}"
    assert prov["MassLambdaMatched"]["source_file"] == "configs/arms/v2/mass_lambda.v2.json"


# ---------------------------------------------------------------- slots and their keys

def test_a_slot_key_may_hold_dots_and_fills_from_its_file(tmp_path):
    assert M.pick({"c": {"a.b": {"n": 3}}}, "c['a.b'].n") == 3
    d = tmp_path / "experiments/FT/data"
    d.mkdir(parents=True)
    (d / "jc2_v2_manifest.json").write_text(json.dumps({"class_coverage": {"train_N1000_s1.parquet": {
        "n_classes_present": 150, "median_per_present_class": 4.5, "n_classes_with_one_jet": 30}}}))
    em, missing = M.Emitter(tmp_path), []
    M.emit_slots(em, missing)
    got = {n: b for n, b, _ in em.macros}
    assert (got["FtCoverageEThreeNClasses"], got["FtCoverageEThreeMedian"],
            got["FtCoverageEThreeOneJet"]) == ("150", "4.5", "30")
    assert not [m for m in missing if m.startswith("FtCoverageEThree")]
    # the first set's subsets have no coverage record: listed missing, never filled
    assert [m for m in missing if m.startswith("FtCoverageVone")]


def test_the_held_out_subset_coverage_is_read_from_the_committed_manifest():
    got, prov = _repo_macros()
    cov = _load("experiments/FT/data/jc2_v2_manifest.json")["class_coverage"]["train_N1000_s1.parquet"]
    assert got["FtCoverageEThreeNClasses"] == str(cov["n_classes_present"])
    assert got["FtCoverageEThreeOneJet"] == str(cov["n_classes_with_one_jet"])
    assert prov["FtCoverageEThreeNClasses"]["source_file"] == "experiments/FT/data/jc2_v2_manifest.json"


# ---------------------------------------------------------------- real data: a toy p-value at its floor

def test_a_toy_p_value_at_its_floor_prints_as_a_bound_with_the_toy_count():
    got, _ = _repo_macros()
    J = _load("experiments/FIGS/data/aoj_full_v1/analysis_v6/aoj_top.json")
    res = M.aoj_fits(REPO, J)[1]
    v, n = res["reference"]["top"]["validation"], res["n_toys"]
    worse = round(v["toy_p"] * (n + 1) - 1)
    if worse == 0:
        assert got["AojRefValidationP"].startswith("\\ensuremath{p \\leq ")
        assert f"0 of {n} toys" in got["AojRefValidationP"]
    else:
        assert f"{worse} of {n} toys" in got["AojRefValidationP"]


def test_captions_state_the_run_counts_and_working_point_the_data_hold():
    built = M.build(REPO)[0]
    J = _load("experiments/FIGS/data/aoj_full_v1/analysis_v6/aoj_top.json")
    res = M.aoj_fits(REPO, J)[1]
    cap = built["tables/realdata.tex"]
    assert f"pass {M.fmt_one(100 * res['eff'], 0)}\\% of the jets" in cap
    assert f"fitted jointly to the {len(J['per_label_set']['shapes']['pool'])} pretrained models" in cap
    S = _load("experiments/FIGS/data/anomaly_v1err/analysis/anomaly_summary.json")
    assert f"over the {M.words(M.anomaly_n_runs(S))} pretraining runs" in built["tables/anomaly.tex"]
    S["families"]["knn"][next(iter(S["families"]["knn"]))][S["conventions"]["primary_injection"]][
        "levels"]["17"]["arms"].pop()
    with pytest.raises(SystemExit, match="runs; the text states one number"):
        M.anomaly_n_runs(S)


# ---------------------------------------------------------------- the rerun's partitions, from their maps

def test_the_partition_counts_come_from_the_maps_and_the_balance_rule_is_checked(tmp_path):
    got, _ = _repo_macros()
    import csv as _csv
    v1 = {r["class_name"]: r for r in _csv.DictReader((REPO / "configs/labelmaps/rand_label_map.v1.csv").open())}
    merged = [d for d in (1, 2, 3) if v1["label_X_YY_bbqq"][f"RAND_d{d}"] == v1["label_X_YY_ccqq"][f"RAND_d{d}"]]
    assert got["RandMergingDrawsBvcFourprong"] == M.word_list(merged)
    paths = M.input_paths(REPO)
    rows = list(_csv.DictReader(paths["rand_v2_map"].open()))
    for r in rows:                   # merge the b vs c two-prong pair in every partition
        if r["class_name"] == "label_X_cc":
            bb = next(x for x in rows if x["class_name"] == "label_X_bb")
            for c in r:
                if c.startswith("RAND2_p") and not c.endswith("_name"):
                    r[c] = bb[c]
    bad = tmp_path / "rand_label_map.v2.csv"
    with bad.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    C = _load("experiments/FIGS/data/probe_ladder_randcontrol_mlp2/analysis/c4_random_control.json")
    with pytest.raises(SystemExit, match="outside the rule"):
        M.emit_rand_design(M.Emitter(REPO), {**paths, "rand_v2_map": bad}, C)


def _write_csv(path, rows):
    import csv as _csv
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_the_flavour_pair_text_is_read_from_its_definition_and_checked_on_the_map(tmp_path):
    """The orbits F1 cuts and moves back are named from flavour_pair.v2.json, so a
    rebuilt pair renames them in the text. A cut other than the b-containing
    classes of the split orbit, which is what the text says it is, stops the build;
    so does a move back that is not the recorded option changing the fewest pairs
    across orbits; and with no options recorded, the move back is a red marker."""
    import csv as _csv
    got, _ = _repo_macros()
    F = _load("configs/labelmaps/flavour_pair.v2.json")
    tex = M.appendix_module().native_tex
    assert tex("X_YY_QQmm") == r"$X\to YY\to QQ\mu\mu$" and tex("X_mm") == r"$X\to \mu\mu$"
    assert got["RandFlavSplitOrbit"] == M.math_safe(tex(F["split_orbit"]))
    if F.get("f1_options"):
        assert all(M.math_safe(tex(o)) in got["RandFlavOrbitsMovedList"]
                   for o in F["orbits_moved_B_to_A"])
    else:
        assert got["RandFlavOrbitsMovedList"].startswith("\\pending")

    # a root of links to the committed inputs, so every macro's source lies under it;
    # the flavour pair's two files are written there per case
    paths = M.input_paths(REPO)
    C = _load("experiments/FIGS/data/probe_ladder_randcontrol_mlp2/analysis/c4_random_control.json")
    root = tmp_path / "root"
    linked = {k: v for k, v in paths.items() if k in (
        "rand_v1_map", "v2_grid", "rand_v2_map", "rand_v2_rule", "realised_shares")}
    linked["ladder"] = REPO / C["provenance"]["inputs"][0]["path"]
    for src in linked.values():
        dst = root / pathlib.Path(src).relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
    mine = {**paths, **{k: root / pathlib.Path(v).relative_to(REPO) for k, v in linked.items()},
            "flavour_pair": root / "configs/labelmaps/flavour_pair.v2.json",
            "flavour_map": root / "configs/labelmaps/flavour_pair_map.v2.csv"}
    rows = list(_csv.DictReader(paths["flavour_map"].open()))
    T = F["orbits_moved_B_to_A"]
    opt = lambda t, n: {"orbits_moved_B_to_A": t, "pairs_changed_from_F0": {"cross_orbit": n}}

    def run(Fx, rows_x, missing=None):
        mine["flavour_pair"].write_text(json.dumps(Fx))
        _write_csv(mine["flavour_map"], rows_x)
        em = M.Emitter(root)
        M.emit_rand_design(em, mine, C, missing)
        return {n_: b_ for n_, b_, _ in em.macros}

    with pytest.raises(SystemExit, match="contain a b quark"):
        run({**F, "split_classes_moved": F["split_classes_moved"][:-1]}, rows)
    with pytest.raises(SystemExit, match="fewest pairs across orbits"):
        run({**F, "f1_options": [opt(T, 504), opt(["X_mm"], 479)]}, rows)
    out = run({**F, "f1_options": [opt(T, 479), opt(["X_mm"], 504)]}, rows)
    assert out["RandFlavOrbitsMovedList"].count("\\ensuremath") == len(T)
    missing = []
    run({k: v for k, v in F.items() if k != "f1_options"}, rows, missing)
    assert any(m.startswith("RandFlavOrbitsMovedList") for m in missing)

    # F1r on the committed map: its realised group shares within the rule's tolerance of
    # F1's, and none of the orbit's b<->c pairs separated (PRESPEC A14)
    out = run(F, rows)
    if "FLAV_F1R" in rows[0] and F.get("b_c_pairs_in_split_orbit"):
        assert out["RandFlavFonerBcSplit"] == "none"
        assert float(out["RandFlavFonerShareDev"].rstrip("\\%")) <= float(out["RandFlavFonerShareTol"].rstrip("\\%"))
        # a share file that triples the share of a class F1r groups apart from F1 breaks
        # the rule and stops the build
        R = json.loads(paths["realised_shares"].read_text())
        moved = next(r["class_name"] for r in rows if r["FLAV_F1R"] != r["FLAV_F1"])
        R["share"][R["class_name"].index(moved)] *= 3
        bad = root / "configs/labelmaps/realised_native_shares.v2.json"
        bad.unlink()
        bad.write_text(json.dumps(R))
        with pytest.raises(SystemExit, match="realised group shares"):
            run(F, rows)
    # The synthetic cuts below are not drawn under the share rule; it is checked above.
    mine["realised_shares"] = None

    # F1r: until the map has its column, listed missing; a column that is F1 with a
    # random cut of the same size, the same move back and the four-prong b-vs-c pair
    # kept together is counted; one that splits the pair stops the build
    rows = [{k: v for k, v in r.items() if not k.startswith("FLAV_F1R")} for r in rows]
    missing = []
    run(F, rows, missing)
    assert any(m.startswith("RandFlavFonerCutWithB") for m in missing)
    a, b = "label_X_YY_bbqq", "label_X_YY_ccqq"
    split = [r for r in rows if r["orbit"] == F["split_orbit"]]
    with_b = [r for r in split if "b" in r["class_name"].rpartition("_")[2] and r["class_name"] != a]
    without = [r for r in split if "b" not in r["class_name"].rpartition("_")[2] and r["class_name"] != b]
    n = len(F["split_classes_moved"])
    cut_r = {r["class_name"] for r in with_b[:n // 2] + without[:n - n // 2]}
    group_b = next(r["FLAV_F1"] for r in split if r["class_name"] in F["split_classes_moved"])
    for r in rows:
        r["FLAV_F1R"] = (group_b if r["class_name"] in cut_r else
                         r["FLAV_F0"] if r["orbit"] == F["split_orbit"] else r["FLAV_F1"])
    assert run(F, rows)["RandFlavFonerCutWithB"] == str(n // 2)
    swap = with_b[0]["class_name"]          # same size of cut, but bbqq cut and ccqq not
    for r in rows:
        if r["class_name"] in (a, swap):
            r["FLAV_F1R"] = group_b if r["class_name"] == a else r["FLAV_F0"]
    with pytest.raises(SystemExit, match="kept together"):
        run(F, rows)


def test_what_the_text_says_about_fine_tuning_holds_on_the_paired_intervals():
    """Sec. 4.4 and the Discussion read these intervals in words; a regeneration
    that changes which side of 1 they fall on must stop here, not in review."""
    got, _ = _repo_macros()

    def iv(name):
        m = re.fullmatch(r"([\d.]+)~\[([\d.]+), ([\d.]+)\]", got[name])
        return tuple(map(float, m.groups()))

    has_1 = lambda name: iv(name)[1] <= 1 <= iv(name)[2]
    above = lambda name: iv(name)[1] > 1
    below = lambda name: iv(name)[2] < 1
    # JetClass-II: 188 and 162 indistinguishable from 10^4 jets; the 10^3 inversion
    assert all(has_1(f"PairedFtJciiE{n}OnesixtwoOverOneeighteight") for n in ("Four", "Five", "Six"))
    assert below("PairedFtJciiEThreeFourthreeOverOneeighteight")
    assert below("PairedFtJciiEThreeRandomOverOneeighteight")
    # JetClass: 188 and 162 agree except at 10^5, where 162 is ahead (Discussion);
    # flavour-keeping 43 trails slightly, removing flavour costs more
    assert all(has_1(f"PairedFtJcE{n}OnesixtwoOverOneeighteight") for n in ("Three", "Four", "Six"))
    assert below("PairedFtJcEFiveOnesixtwoOverOneeighteight")
    assert above("PairedFtJcESixFourthreeOverOneeighteight")
    assert iv("PairedFtJcESixOnesevenOverOneeighteight")[1] > iv("PairedFtJcESixFourthreeOverOneeighteight")[2]


# ---------------------------------------------------------------- the readings the text puts in words

def _iv(got, name, pattern=r"([\d.+-]+)~\[([\d.+-]+), ([\d.+-]+)\]"):
    text = got[name].replace("\\ensuremath{-}", "-")
    m = re.fullmatch(pattern, text)
    assert m, (name, got[name])
    return tuple(map(float, m.groups()))


def test_what_the_text_says_about_anomaly_holds_on_the_paired_intervals():
    """Sec. 4.7: the frozen-feature detectors lose multi-b sensitivity from 43 classes
    down, disagree on X->bb, and separate 188 from 162 only on bbbb; for the output
    ratio no paired difference between vocabularies is resolved, at epoch 79 or averaged
    over epochs 70-79; the output ratio is less sensitive than the Mahalanobis distance
    on every b signal except bbb at 17 classes."""
    got, _ = _repo_macros()
    above = lambda n: _iv(got, n)[1] > 1
    below = lambda n: _iv(got, n)[2] < 1
    has_1 = lambda n: _iv(got, n)[1] <= 1 <= _iv(got, n)[2]
    for det in ("Mahalanobis", "Knn"):
        assert above(f"PairedAnomaly{det}XYYBbbbFourthreeOverOnesixtwo")
        assert above(f"PairedAnomaly{det}XYYBbbbOnesevenOverFourthree")
        assert below(f"PairedAnomaly{det}XYYBbbbOnesixtwoOverOneeighteight")
        assert all(has_1(f"PairedAnomaly{det}{s}OnesixtwoOverOneeighteight") for s in ("XBb", "XYYBbb"))
    for n in ("PairedAnomalyMahalanobisXYYBbbFourthreeOverOnesixtwo",
              "PairedAnomalyMahalanobisXYYBbbOnesevenOverFourthree"):
        assert above(n)
    assert above("PairedAnomalyMahalanobisXBbOnesevenOverOneeighteight")
    assert below("PairedAnomalyKnnXBbOnesevenOverOneeighteight")
    output = [n for n in got if n.startswith("PairedAnomalyClassSumMatched")]
    assert any("Late" in n for n in output) and any("Late" not in n for n in output)
    assert all(has_1(n) for n in output), [n for n in output if not has_1(n)]
    ratio = [n for n in got if n.startswith("PairedAnomalyOutputOverMahalanobis")]
    assert ratio and all(above(n) for n in ratio if n != "PairedAnomalyOutputOverMahalanobisXYYBbbOneseven")
    assert has_1("PairedAnomalyOutputOverMahalanobisXYYBbbOneseven")


def test_what_the_text_says_about_label_recovery_holds_on_the_paired_intervals():
    """Sec. 4.2: at the 17-class level the 17-class models are not resolved from the
    188-class ones, at the 4- and 2-class levels they are ahead, and the MLP adds a
    small but resolved amount at the 188-class level."""
    got, _ = _repo_macros()
    d = lambda n: _iv(got, n)
    lo, hi = d("RecoveryPairedOnesevenLevelOnesevenMinusOneeighteight")[1:]
    assert lo <= 0 <= hi
    assert d("RecoveryPairedFourLevelOnesevenMinusOneeighteight")[1] > 0
    assert d("RecoveryPairedTwoLevelOnesevenMinusOneeighteight")[1] > 0
    assert d("RecoveryMlpGainOneseven")[1] > 0


def test_what_the_text_says_about_the_open_data_holds():
    """Sec. 4.8: no paired yield ratio between vocabularies, or with and without the mass
    output, is resolved; the summed passing-jet residual below the top window is small
    in every pretrained model."""
    got, _ = _repo_macros()
    for n in [n for n in got if n.startswith("AojPairedYield")]:
        lo, hi = _iv(got, n)[1:]
        assert lo <= 1 <= hi, n
    z = [float(got[k].replace("\\ensuremath{-}", "-")) for k in ("AojSidebandZMin", "AojSidebandZMax")]
    assert -2 < z[0] <= z[1] < 2


def test_what_the_text_says_about_fine_tuning_in_words_beyond_the_ratios_to_188():
    """Sec. 4.4: at 10^3 jets the 17-class models are not resolved from the 188-class
    ones in macro AUC but trail them in accuracy; at 10^6 jets the random partitions lie
    between the 43- and 17-class models on both datasets; the mass output costs a
    resolved amount after fine-tuning."""
    got, _ = _repo_macros()
    lo, hi = _iv(got, "PairedFtJciiEThreeOnesevenOverOneeighteight")[1:]
    assert lo <= 1 <= hi
    assert _iv(got, "PairedFtAccJciiEThreeOnesevenMinusOneeighteight")[2] < 0
    for ds in ("Jcii", "Jc"):
        assert _iv(got, f"PairedFt{ds}ESixRandomOverFourthree")[1] > 1
        assert _iv(got, f"PairedFt{ds}ESixRandomOverOneseven")[2] < 1
    for n in ("PairedFtJciiESixOnesixtwoMassOverOnesixtwo", "PairedFtJciiESixOnesevenMassOverOneseven"):
        assert _iv(got, n)[1] > 1


def test_run_by_run_intervals_are_re_derived_independently():
    """The intervals the generator forms itself (anomaly, open data, label recovery),
    recomputed here from the stored per-run values with scipy, not with its helpers."""
    import numpy as np
    from scipy import stats
    got, prov = _repo_macros()

    def t_interval(d):
        d = np.asarray(d, float)
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), d.mean() - h, d.mean() + h

    S = _load(prov["PairedAnomalyMahalanobisXYYBbbbFourthreeOverOnesixtwo"]["source_file"])
    cell = lambda lv: S["families"]["mahalanobis"]["label_X_YY_bbbb"]["2000"]["levels"][lv]
    run = lambda a: int(re.search(r"-s(\d+)", a).group(1))
    a, b = (dict(zip(map(run, cell(lv)["arms"]), cell(lv)["ln_sigma_min"])) for lv in ("162", "43"))
    m, lo, hi = t_interval([b[k] - a[k] for k in sorted(a)])
    assert got["PairedAnomalyMahalanobisXYYBbbbFourthreeOverOnesixtwo"] == M.fmt_paired(*np.exp([m, lo, hi]))

    J = _load(prov["AojPairedYieldOnesevenOverOneeighteight"]["source_file"])
    L = J["per_label_set"]["label_sets"]
    y = {lv: dict(zip(map(run, L[lv]["models"]), L[lv]["signal_yields"])) for lv in ("188", "17")}
    m, lo, hi = t_interval(np.log([y["17"][k] / y["188"][k] for k in sorted(y["188"])]))
    assert got["AojPairedYieldOnesevenOverOneeighteight"] == M.fmt_paired(*np.exp([m, lo, hi]))

    R = _load(prov["RecoveryPairedFourLevelOnesevenMinusOneeighteight"]["source_file"])
    n = max(R["sizes"])
    acc = {lv: {r["seed"]: r["accuracy"] for r in R["table"] if (r["rung"], r["level"], r["probe"], r["n_train"])
                == ("R3_VIS", lv, "linear", n)} for lv in (188, 17)}
    m, lo, hi = t_interval([acc[17][k] - acc[188][k] for k in sorted(acc[188])])
    assert got["RecoveryPairedFourLevelOnesevenMinusOneeighteight"] == M.math_safe(M.fmt_diff(m, lo, hi))


@pytest.mark.parametrize("d,lo,hi,want", [
    (0.0060, 0.0051, 0.0069, "+0.0060~[+0.0051, +0.0069]"),
    (0.00101, 0.00003, 0.00198, "+0.00101~[+0.00003, +0.00198]"),     # a bound may not round onto 0
    (0.0006, -0.0004, 0.0015, "+0.0006~[-0.0004, +0.0015]"),
])
def test_a_paired_difference_never_rounds_a_bound_onto_zero(d, lo, hi, want):
    assert M.fmt_diff(d, lo, hi) == want


def test_a_scratch_path_maps_to_the_committed_fit_only_at_its_hash(tmp_path):
    q = tmp_path / "experiments/FIGS/data/aoj_full_v1/fit_v9/results.json"
    q.parent.mkdir(parents=True)
    q.write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    assert M.committed_path(tmp_path, "/scratch/fit_v9/results.json", sha) == q
    assert M.committed_path(tmp_path, "/scratch/fit_v9/results.json", "0" * 64) != q
    assert M.committed_path(tmp_path, "experiments/x.json") == tmp_path / "experiments/x.json"


def test_a_literature_display_form_may_not_carry_another_number(tmp_path):
    p = tmp_path / "facts.json"
    fact = {"macro": "LitX", "value": "10−5", "source": "s", "location": "l", "quote": "eps = 10−5."}
    p.write_text(json.dumps({"facts": [{**fact, "tex": "$10^{-5}$"}]}))
    em = M.Emitter(tmp_path)
    M.emit_literature(em, p)
    assert em.macros[0][1] == "\\ensuremath{10^{-5}}"
    p.write_text(json.dumps({"facts": [{**fact, "tex": "$10^{-6}$"}]}))
    with pytest.raises(SystemExit, match="typeset"):
        M.emit_literature(M.Emitter(tmp_path), p)


def test_the_rerun_recipe_is_read_from_the_jobs_the_code_and_the_dry_runs():
    """The loader windows from the grid's jobs, the schedule from pretrain_v2.py, and what
    leaving a family out does to the stream from the committed dry runs."""
    got, prov = _repo_macros()
    spec = (REPO / prov["DesignVtwoNWindows"]["source_file"]).read_text()
    f = float(re.search(r"--data-fraction ([0-9.]+)", spec).group(1))
    assert got["DesignVtwoNWindows"] == M.words(round(1 / f))
    D = _load(prov["DesignLofoFamilyShare"]["source_file"])
    L = _load("experiments/FIGS/data/v2_loader/loader_dryrun/dryrun_lofo4p_seed1.json")
    import numpy as np
    counts = np.sum([e["native_counts"] for e in D["epochs"]], axis=0)
    share = counts[L["summary"]["labels_absent_every_epoch"]].sum() / counts.sum()
    assert got["DesignLofoExposureFactor"] == f"{1 / (1 - share):.2f}"
    assert got["DesignLofoQcdShareLofo"] == f"{100 * L['summary']['mean_qcd_share']:.1f}\\%"
    code = (REPO / "experiments/MTX/pretrain_v2.py").read_text()
    assert "int(num_epochs * 0.3)" in code and got["DesignLrFlatPercent"] == "70\\%"


def test_what_the_text_says_about_mass_resolution_holds_on_the_paired_intervals():
    """Sec. 4.6: the 43- and 17-class models resolve the jet mass slightly better than
    the 188-class models, the 162-class models are not resolved from them, and the mass
    output improves the resolution at both vocabularies."""
    got, _ = _repo_macros()
    for n in ("PairedMassResMlpFourthreeOverOneeighteight", "PairedMassResMlpOnesevenOverOneeighteight",
              "PairedMassResMlpOnesixtwoMassOverOnesixtwo", "PairedMassResMlpOnesevenMassOverOneseven"):
        assert _iv(got, n)[2] < 1, n
    lo, hi = _iv(got, "PairedMassResMlpOnesixtwoOverOneeighteight")[1:]
    assert lo <= 1 <= hi


def test_a_family_out_row_is_read_as_often_as_the_epochs_actually_read():
    """Sec. 2.3: the factor by which a row is read more often without the family counts
    how much of its window each epoch reads before its jets are reached, not only the
    nominal window ratio (5/3 = 1.67 would contradict the 1.29 exposure factor)."""
    got, prov = _repo_macros()
    D = _load("experiments/FIGS/data/v2_loader/loader_dryrun/dryrun_s176_seed1.json")
    L = _load(prov["DesignLofoRowFactor"]["source_file"])
    import numpy as np
    k, k_l = round(1 / D["args"]["data_fraction"]), L["args"]["data_windows"]
    read = np.mean([(e["max_fetch_id"] + 1) / D["args"]["data_split_num"] for e in D["epochs"]]) / k
    read_l = np.mean([(e["max_fetch_id"] + 1) / e["fetches_per_pass"] for e in L["epochs"]]) / k_l
    assert got["DesignLofoRowFactor"] == f"{read_l / read:.2f}"
    assert got["DesignLofoRowFactor"] != f"{k / k_l:.2f}"


def test_a_cycle_trains_on_an_order_of_magnitude_more_distinct_jets_than_parameters():
    """Sec. 5 places the runs on Pirovano et al.'s axis: one cycle of windows holds
    more than ten distinct jets per parameter of the 188-class network."""
    got, prov = _repo_macros()
    D = _load(prov["DesignDistinctJetsCycle"]["source_file"])
    k = round(1 / D["args"]["data_fraction"])
    import numpy as np
    cyc = [sum(e["distinct_jets"] for e in D["epochs"][c * k:(c + 1) * k]) for c in range(len(D["epochs"]) // k)]
    assert got["DesignDistinctJetsCycle"] == M.math_safe(M.fmt_sci(float(f"{np.mean(cyc):.2g}")))
    assert min(cyc) > 10 * float(got["LitSophonParamsMillion"]) * 1e6


def test_scaling_the_fail_region_tops_moves_the_paired_yields_as_the_text_says():
    """Sec. 4.8: the tops in the failing jets come from the reference's fit, common to
    every model; scaled up, no paired yield ratio moves by more than the quoted share,
    and none becomes resolved."""
    from scipy import stats
    import numpy as np
    got, prov = _repo_macros()
    res = _load(prov["AojLeakPairedShiftMax"]["source_file"])["models"]
    runs = lambda pre: {int(re.search(r"-s(\d)", m).group(1)): f["top"] for m, f in res.items()
                        if re.fullmatch(pre + r"-s\db?", m)}
    sets = {lv: runs(pre) for lv, pre in (("188", "l188"), ("162", "l162"), ("43", "r42q1"),
                                          ("17", "r16q1"), ("162m", "l162mass"), ("17m", "r16q1mass"))}

    def ratio(fine, coarse, y):
        d = np.log([y(sets[coarse][k]) / y(sets[fine][k]) for k in sorted(sets[fine])])
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return np.exp([d.mean(), d.mean() - h, d.mean() + h])
    moves = []
    for e in ("0.6", "0.4"):
        for fine, coarse in (("188", "162"), ("188", "43"), ("188", "17"), ("162", "17"),
                             ("162", "162m"), ("17", "17m")):
            r0 = ratio(fine, coarse, lambda f: f["signal_yield"])
            r1 = ratio(fine, coarse, lambda f: f["leak_systematic"]["signal_yield"][e])
            assert r1[1] <= 1 <= r1[2], (e, fine, coarse)
            moves.append(abs(r1[0] / r0[0] - 1))
    assert got["AojLeakPairedShiftMax"] == f"{100 * max(moves):.0f}\\%"


def test_the_gpu_of_each_run_index_is_read_from_the_job_specs(tmp_path):
    """The rerun's GPU per run index comes from the jobs' nodeAffinity, and a run index
    whose configurations differ in GPU is reported, never printed."""
    def spec(name, product):
        p = tmp_path / f"job-mtx2-{name}-raunav.yaml"
        p.write_text("              - key: nvidia.com/gpu.product\n                operator: In\n"
                     f'                values: ["{product}"]\n')
        return p
    a = [spec("l188-s1", "NVIDIA-GeForce-RTX-3090"), spec("r16q1-s1", "NVIDIA-GeForce-RTX-3090"),
         spec("l188-s4", "NVIDIA-L40"), spec("r16q1-s4", "NVIDIA-L40")]
    by = M.gpu_by_run(a)
    assert {r: sorted(p) for r, p in by.items()} == {1: ["NVIDIA-GeForce-RTX-3090"], 4: ["NVIDIA-L40"]}
    assert M.gpu_name("NVIDIA-L40") == "L40"
    mixed = M.gpu_by_run(a[:3] + [spec("r16q1-s4", "NVIDIA-GeForce-RTX-3090")])
    assert len(mixed[4]) == 2
    (tmp_path / "job-mtx2-x-s2-raunav.yaml").write_text('values: ["NVIDIA-L40"]\n')
    with pytest.raises(SystemExit, match="GPU product"):
        M.gpu_by_run([tmp_path / "job-mtx2-x-s2-raunav.yaml"])


@pytest.mark.skip(reason="cut from the grid and the paper on 2026-10-09 (configs/arms/v2_grid.launched.json keeps the design)")
def test_the_self_supervised_runs_and_the_robustness_band_are_read_where_they_are_fixed():
    got, _ = _repo_macros()
    grid = _load("configs/arms/v2_grid.json")["arms"]
    ssl = [a["runs"] for a in grid if a["num_classes"] is None and not a["extra_selection"]]
    assert got["DesignSslRuns"] == M.words(ssl[0])
    mde = _mod("mde", "src/stats/mde.py")
    assert math.isclose(math.exp(mde.MEI_LOG), float(got["DesignMeiFactor"]))
