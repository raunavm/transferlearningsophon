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
            "p": 0.99, "verdict": "inconclusive",
            "all_pairs_companion": {"max_p": 0.99, "smallest_bound_passed_by_all": 9.0},
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
    p = root / "experiments/FIGS/data/probe_ladder_v2/analysis/seed_level_results.json"
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
    assert m["ProbeRejAlphaLinearOnetwo"].startswith("$>$")
    assert m["ProbeRejAlphaLinearEight"].startswith("$\\geq$")
    assert "^{\\ast}" in m["ProbeRejAlphaLinearEight"]
    assert re.fullmatch(r"[0-9{},.]+", m["ProbeRejAlphaLinearFour"])
    table = built["tables/probes_linear.tex"]
    assert "$>$1{,}000" in table and "$\\geq$900$^{\\ast}$" in table
    assert "a lower bound" in table              # the footnote is not optional


def test_a_saturated_auc_is_marked_rather_than_printed_to_five_decimals(root):
    """AUC = 1 at the sample's resolution is a bound on 1-AUC, not a measurement."""
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeAucBetaLinearOnetwo"] == "$1^{\\dagger}$"
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
    (out / "tables/tests.tex").write_text("hand edited\n")
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


def _with_points(root, points):
    """Give the fixture analysis the working-point block the real one carries."""
    p = root / "experiments/FIGS/data/probe_ladder_v2/analysis/seed_level_results.json"
    A = json.loads(p.read_text())
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
    _with_points(root, {"median": 321.0, "range": [300.0, 350.0], "n_bound": 0})
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeEpsS"] == "90\\%"
    assert m["ProbeRejAlphaLinearOnetwo"] == "321", (
        "the 0.90 value, not the 1,000.0 bound the flat fields hold at 0.50")
    prov = json.loads(built["provenance.json"])
    assert "rejection_points['0.90']" in prov["ProbeRejAlphaLinearOnetwo"]["json_path"]
    assert "90\\% signal efficiency" in built["tables/probes_linear.tex"]


def test_a_censored_headline_cell_still_never_prints_as_a_bare_number(root):
    """If the headline point turns out to be censored too, that is the finding
    and the table must say so -- it must not silently fall back to a number."""
    _with_points(root, {"median": 11876.0, "range": [11876.0, 11876.0],
                        "n_bound": len(SEEDS)})
    m = macros_in(M.build(root)[0]["results_generated.tex"])
    assert m["ProbeRejAlphaLinearOnetwo"].startswith("$>$")


def test_an_analysis_without_working_points_falls_back_and_says_which_point(root):
    """Analysis files written before the working points were recorded must keep
    building, at the point they actually hold."""
    built, _, _ = M.build(root)          # fixture has no rejection_points
    m = macros_in(built["results_generated.tex"])
    assert m["ProbeEpsS"] == "50\\%"
    assert json.loads(built["provenance.json"])[
        "ProbeRejAlphaLinearFour"]["json_path"].endswith(".rejection_median")


# ------------------------------- no confirmatory test may vanish from T2

def _measured_c5(p=2.0e-07):
    """A C5 block in the shape experiments/STATS/seed_level.py writes it."""
    def did(mean, t, pv):
        return {"estimable": True, "n_pairs": 5, "seeds": [1, 2, 3, 4, 5],
                "is_bound": False, "mean_diff": mean, "sd_diff": 0.02, "t": t, "df": 4,
                "p": pv, "ci95": [mean - 0.04, mean + 0.04],
                "sign_flip": {"p": 0.0625, "floor": 0.0625, "n_arrangements": 32}}
    probes = {k: {"did": did(0.5044, 73.96, p),
                  "gain_by_level": {"162": did(-0.0490, -4.09, 0.0149),
                                    "17": did(-0.5078, -47.10, 1.22e-06)}}
              for k in ("linear", "mlp")}
    return {"prediction": "two-sided; the written expectation is that the mass output "
                          "helps more at 17 classes than at 162",
            "task": "alpha", "endpoint": "log1m_auc",
            "did_definition": "(162+mass - 162) - (17+mass - 17), paired by seed index",
            "expected_sign_if_written_expectation_holds": "+",
            "confirmatory": {"task": "alpha", "probes": probes},
            "p": p, "exploratory": [], "exploratory_note": "no verdict"}


def _analysis_of(root):
    p = root / "experiments/FIGS/data/probe_ladder_v2/analysis/seed_level_results.json"
    return p, json.loads(p.read_text())


def test_every_confirmatory_family_member_reaches_the_tests_table(root):
    """The defect this pins: table_tests rendered a confirmatory member ONLY while
    its status was 'pending'. C5 is the first member that can become 'available'
    without being a trend test, so the moment it was measured it rendered as
    nothing, while the caption went on claiming the full family. A measured,
    Holm-judged confirmatory result must never be silently absent."""
    _, A = _analysis_of(root)
    A["confirmatory"]["C5"] = _measured_c5()
    A["confirmatory"]["holm_family"].append(
        {"test": "C5", "status": "available", "p_raw": 2.0e-07, "family_size": 3,
         "threshold_if_smallest": 0.05 / 3,
         "reject_whatever_pending": True, "reject_possible": True})
    tex = M.table_tests(A)
    for h in A["confirmatory"]["holm_family"]:
        assert h["test"] in tex, (
            f"{h['test']} is in the confirmatory family and has no row in the tests table")
    assert "C5: " in tex, "a measured C5 needs its own row, not the bare fallback"
    assert "no row generator" not in tex, "C5 has a renderer; it must not hit the fallback"


def test_the_c5_row_carries_the_nonlinear_probe_and_both_one_sided_gains(root):
    """D6: never a linear probe alone. And the interaction is a DIFFERENCE of two
    gains, so a reader who sees only the difference cannot tell which side moved."""
    _, A = _analysis_of(root)
    A["confirmatory"]["C5"] = _measured_c5()
    A["confirmatory"]["holm_family"].append(
        {"test": "C5", "status": "available", "p_raw": 2.0e-07, "family_size": 3,
         "threshold_if_smallest": 0.05 / 3,
         "reject_whatever_pending": True, "reject_possible": True})
    tex = M.table_tests(A)
    assert "nonlinear probe" in tex
    assert "gain at 162 classes" in tex and "gain at 17 classes" in tex
    # the interaction is Holm-judged; the companions never are
    assert tex.count("rejected") >= 1
    for line in tex.splitlines():
        if "gain at" in line or "nonlinear probe" in line:
            assert "descriptive" in line, line


def test_a_measured_confirmatory_test_with_no_renderer_still_prints_a_row(root):
    """Belt and braces for the same failure: if a future confirmatory member
    becomes available and nobody writes it a renderer, the table must show that it
    exists rather than drop it."""
    _, A = _analysis_of(root)
    for h in A["confirmatory"]["holm_family"]:
        if h["test"] == "C2":
            h.update({"status": "available", "p_raw": 0.031,
                      "reject_whatever_pending": False, "reject_possible": True})
    tex = M.table_tests(A)
    assert "C2" in tex, "an available member with no renderer vanished from the table"
    assert "no row generator" in tex


def test_c5_emits_macros_for_both_probes_and_both_gains(root):
    """PRESPEC: every number in the manuscript comes from this generator. If C5 has
    no macros, its numbers cannot be written into the paper at all."""
    path, A = _analysis_of(root)
    A["confirmatory"]["C5"] = _measured_c5()
    path.write_text(json.dumps(A))
    built, _, _ = M.build(root)
    m = macros_in(built["results_generated.tex"])
    names = set(m)
    did = [n for n in names if n.startswith("MassDid")]
    gain = [n for n in names if n.startswith("MassGain")]
    assert did, "no C5 difference-in-differences macro; it cannot reach the manuscript"
    assert gain, "no per-granularity gain macro"
    assert any("Linear" in n for n in did) and any("Mlp" in n for n in did), \
        "D6: the nonlinear probe travels with the linear one"
    assert len([n for n in gain if not n.startswith("MassGainP")]) == 2, \
        "one gain macro per granularity"
