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
        p = root / "experiments/FIGS/data/probe_ladder_v2" / f"s{seed}.json"
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
    p = root / "experiments/FIGS/data/probe_ladder_v2/analysis_family_of_four/seed_level_results.json"
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
    surv = {"disc_one": {"title": "X->bb vs QCD", "source": "arXiv:0000.00000 Eq. (1)",
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
    assert any("label_recovery_ladder_v1" in x for x in missing)
    assert any("bench_metrics" in x for x in missing)
    assert any("anomaly_merged" in x for x in missing)
    assert any("T4" in s for s in skipped)


def test_inputs_that_disagree_on_the_row_alignment_stop_the_run(root):
    """Different alignments means different jets: a paired contrast across them
    is not a paired contrast, and every number here assumes it is one."""
    p = root / "experiments/FIGS/data/probe_ladder_v2/s2.json"
    d = json.loads(p.read_text())
    d["row_alignment_sha256"] = "b" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        M.build(root)
    assert "row_alignment_sha256" in str(e.value)


def test_a_ladder_file_rewritten_since_the_analysis_stops_the_run(root):
    """Then the p-values describe data that is no longer on disk."""
    p = root / "experiments/FIGS/data/probe_ladder_v2/s2.json"
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
    p = root / "experiments/FIGS/data/probe_ladder_v2/analysis_family_of_four/seed_level_results.json"
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
    # the yield from the fits themselves, not from the analysis file's summary
    fits = _load(_load(prov["AojYieldOneeighteight"]["source_file"])["provenance"]["input"])["models"]
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
        assert got[f["macro"]] == f["value"] and f["value"] in f["quote"]
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


def _recipes(tmp_path, **override):
    run = {"leg": "1", "init": "l162-s2", "n_train": "1000", "lr": "1e-4", "head_lr_mult": "50",
           "epochs": "50", "lr_schedule": None, "weight_decay": "0.01", "batch_size": "512",
           "samples_per_epoch_val": None}
    runs = [{**run, **override},
            {**run, "init": "scratch", "lr": "5e-4"},
            {**run, "leg": "top", "lr_schedule": "constant", "epochs": "20", "samples_per_epoch_val": "20000"},
            {**run, "leg": "top", "n_train": "100000", "lr_schedule": "constant", "epochs": "20",
             "samples_per_epoch_val": "200000"},
            {**run, "leg": "top", "init": "scratch", "lr": "5e-4", "lr_schedule": "constant",
             "epochs": "20", "samples_per_epoch_val": "20000"}]
    for n, e in ((10_000, "50"), (100_000, "30"), (1_000_000, "10")):
        runs.append({**run, "n_train": str(n), "epochs": e})
    leg = tmp_path / "leg.yaml"
    leg.write_text("--use-amp --optimizer ranger LR=5e-4; MULT=() LR=1e-4 --samples-per-epoch-val 20000")
    bench = tmp_path / "bench.yaml"
    bench.write_text("--use-amp --optimizer ranger LR=5e-4; MULT=() LR=1e-4 --lr-scheduler none")
    return {"runs": runs}, [leg], [bench]


def test_the_fine_tuning_table_states_one_recorded_value_per_setting(tmp_path):
    R, legs, bench = _recipes(tmp_path)
    rec = M.ft_recipe(R, legs, bench)
    assert (rec["lr"], rec["lr_scratch"], rec["head_mult"]) == ("1e-4", "5e-4", "50")
    assert rec["epochs_jetclass"] == {1_000: "50", 10_000: "50", 100_000: "30", 1_000_000: "10"}
    assert (rec["val_bench_small"], rec["val_bench_large"]) == ("20000", "200000")
    got, prov = _repo_macros()
    assert got["FtRecipeLrHead"] == "\\ensuremath{5\\times10^{-3}}"


def test_a_run_that_departs_from_its_group_stops_the_table(tmp_path):
    R, legs, bench = _recipes(tmp_path, lr="3e-4")
    R["runs"].append({**R["runs"][0], "lr": "1e-4"})
    with pytest.raises(SystemExit, match="disagree within a group"):
        M.ft_recipe(R, legs, bench)


def test_a_command_with_a_schedule_flag_contradicts_the_jetclass_row(tmp_path):
    R, legs, bench = _recipes(tmp_path)
    legs[0].write_text(legs[0].read_text() + " --lr-scheduler none")
    with pytest.raises(SystemExit, match="not the JetClass recipe"):
        M.ft_recipe(R, legs, bench)


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
