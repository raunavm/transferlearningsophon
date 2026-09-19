"""Seed-level inference on the four-granularity probe ladder (PRESPEC_2026-09 §2-3).

Every fixture is synthetic and written to tmp_path in the schema probe.py emits,
so no real result is read or pinned here. Each test guards one thing the
pre-specification fixes: the direction of the trend test, where the step is
located, the order in which the report is printed, the equivalence wording and
bound, the family size of the Holm step, and what the loader refuses.

Set PROBE_LADDER_EXAMPLE to a real probe_results.json to check that the fixture
builder still mimics the true schema.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import pathlib

import numpy as np
import pytest
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _mod("seed_level", "experiments/STATS/seed_level.py")
L = _mod("leg_stats", "experiments/FT/leg_stats.py")

from src.stats.inference import paired_t      # noqa: E402
from src.stats.mde import MEI_LOG              # noqa: E402
from src.stats.trend import max_t_trend        # noqa: E402

TASKS = ("bvc_resonant", "retained_topology", "bvc_qcd", "ee_vs_mm", "bvc_4prong",
         "visible_content")
PREFIX = {188: "l188", 162: "l162", 43: "r42q1", 17: "r16q1"}
SHA = "ab" * 32


# ---------------------------------------------------------------- fixtures

def ladder_values(step=0.0, noise=0.05, rng_seed=7, step_tasks=("bvc_resonant",), mlp_sign=1.0):
    """value(task, probe, level, seed) -> log(1-AUC): seed effect + noise + a step AT 17."""
    rng = np.random.default_rng(rng_seed)
    block = rng.normal(0, 0.3, 6)
    eps = rng.normal(0, noise, (len(TASKS), 2, 4, 6))

    def value(task, kind, level, seed):
        i = (TASKS.index(task), S.PROBES.index(kind), S.LEVELS.index(level), seed)
        planted = step * (mlp_sign if kind == "mlp" else 1.0) \
            if task in step_tasks and level == 17 else 0.0
        return float(-4.0 + block[seed] + eps[i] + planted)
    return value


FLOOR = -17.0      # a censored cell: AUC == 1, endpoint pinned at the sample's resolution


def _entry(l1m, kind, censored=False):
    l1m = FLOOR if censored else l1m
    auc = 1.0 if censored else 1.0 - math.exp(l1m)
    rej = {"rejection": 50.0, "eps_b": 0.02, "rejection_is_bound": False, "n_bkg_pass": 60,
           "rel_stat_err": 0.129}
    sel = {"C": 1.0, "val_auc": auc} if kind == "linear" else {
        "seeds": [{"seed": i, "val_auc": auc, "epochs_run": 20, "converged": True}
                  for i in range(3)],
        "val_auc_mean": auc, "val_auc_std": 0.0, "all_converged": True}
    return {"auc": auc, "log1m_auc": l1m, "log1m_auc_censored": censored, **rej,
            "rejection_eps_s": 0.5, "rejection_at": {"0.50": dict(rej)}, "selection": sel}


def seed_doc(value, seed, omit=(), sha=SHA, n_jets=2_000_000, extra_arms=(), censor=()):
    arms = {f"{PREFIX[lv]}-s{seed}" + ("b" if (lv, seed) == (162, 1) else ""): lv
            for lv in S.LEVELS if (lv, seed) not in omit}
    arms.update(extra_arms)
    return {"n_jets_total": n_jets, "row_alignment_sha256": sha, "eps_s_default": 0.5,
            "arm_checkpoints": {a: "cd" * 32 for a in arms}, "min_per_class_test": 1000,
            "mlp_threads": 4,
            "tasks": {t: {"eps_s": [0.5], "n": 30000, "n_signal": 15000, "names": ["a", "b"],
                          "collapsed_at": ["R16_Q1"],
                          "arms": {a: {k: _entry(value(t, k, lv, seed), k, (t, lv) in censor)
                                       for k in S.PROBES} for a, lv in arms.items()},
                          "contrasts": {}} for t in TASKS}}


def write_ladder(root, value, seeds=(1, 2, 3, 4, 5), **kw):
    for s in seeds:
        d = root / f"s{s}"
        d.mkdir(parents=True)
        (d / "probe_results.json").write_text(json.dumps(seed_doc(value, s, **kw)))
    return root


def cells_of(root, drop=()):
    return S.index_cells(S.load_ladder(root)["rows"], drop)


def run(root, out, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert S.main([str(root), "--out", str(out), *extra]) == 0
    return json.loads((out / "seed_level_results.json").read_text()), buf.getvalue()


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """A +1.0 step in log(1-AUC) at the 17-class level of bvc_resonant, both probes."""
    root = write_ladder(tmp_path_factory.mktemp("planted"), ladder_values(step=1.0))
    return run(root, root / "out")


# ---------------------------------------------------------------- loader

def test_arm_names_parse_and_l162_s1b_is_seed_1():
    assert S.parse_arm("l162-s1b") == (162, 1)
    assert S.parse_arm("l188-s3") == (188, 3)
    assert S.parse_arm("r42q1-s4") == (43, 4)
    assert S.parse_arm("r16q1-s5") == (17, 5)
    for bad in ("l162-s2b", "l162-s0", "L162-s1", "rand-d1", "r16q1_mass-s1", "l162-s1-x"):
        with pytest.raises(SystemExit, match="does not parse"):
            S.parse_arm(bad)


def test_tidy_table_has_one_row_per_cell_at_full_precision(tmp_path):
    value = ladder_values(step=0.3)
    rows = S.load_ladder(write_ladder(tmp_path, value))["rows"]
    assert len(rows) == len(TASKS) * 2 * 4 * 5
    assert len({(r["task"], r["probe"], r["level"], r["seed"]) for r in rows}) == len(rows)
    for r in rows:
        assert r["log1m_auc"] == value(r["task"], r["probe"], r["level"], r["seed"])
        assert r["rejection_eps_s"] == 0.5 and r["censored"] is False
    s1b = [r for r in rows if r["arm"] == "l162-s1b"]
    assert len(s1b) == len(TASKS) * 2 and {(r["level"], r["seed"]) for r in s1b} == {(162, 1)}


def test_accepts_a_list_of_files_as_well_as_a_directory(tmp_path):
    root = write_ladder(tmp_path, ladder_values())
    files = sorted(root.glob("s*/probe_results.json"))
    assert S.load_ladder(files)["rows"] == S.load_ladder(root)["rows"]
    assert [f["sha256"] for f in S.load_ladder(files)["files"]] == [S._sha(f) for f in files]


@pytest.mark.parametrize("field, kw", [("row_alignment_sha256", {"sha": "ef" * 32}),
                                       ("n_jets_total", {"n_jets": 1_999_999})])
def test_mismatched_alignment_is_refused(tmp_path, field, kw):
    value = ladder_values()
    write_ladder(tmp_path, value, seeds=(1, 2, 3, 4))
    write_ladder(tmp_path, value, seeds=(5,), **kw)
    with pytest.raises(SystemExit, match=field) as e:
        S.main([str(tmp_path), "--out", str(tmp_path / "out")])
    assert e.value.code not in (0, None)
    assert not (tmp_path / "out").exists()


def test_duplicated_cell_is_refused(tmp_path):
    # seed index 1 of the 162-class model under BOTH of its names
    write_ladder(tmp_path, ladder_values(), seeds=(1,), extra_arms={"l162-s1": 162})
    with pytest.raises(SystemExit, match="duplicated cell"):
        S.load_ladder(tmp_path)


def test_unparseable_arm_in_a_file_is_refused(tmp_path):
    write_ladder(tmp_path, ladder_values(), seeds=(1,), extra_arms={"rand-d1": 17})
    with pytest.raises(SystemExit, match="does not parse"):
        S.load_ladder(tmp_path)


@pytest.mark.skipif(not os.environ.get("PROBE_LADDER_EXAMPLE"),
                    reason="set PROBE_LADDER_EXAMPLE to a real probe_results.json")
def test_fixture_mimics_the_real_schema():
    real = json.loads(pathlib.Path(os.environ["PROBE_LADDER_EXAMPLE"]).read_text())
    fake = seed_doc(ladder_values(), 3)
    assert set(fake) == set(real)
    rt = next(t for t in real["tasks"].values() if not t.get("skipped"))
    ft = fake["tasks"]["bvc_resonant"]
    assert set(ft) == set(rt)
    ra, fa = next(iter(rt["arms"].values())), next(iter(ft["arms"].values()))
    for k in S.PROBES:
        assert set(fa[k]) == set(ra[k])
        assert set(fa[k]["selection"]) == set(ra[k]["selection"])
        assert set(next(iter(fa[k]["rejection_at"].values()))) == \
            set(next(iter(ra[k]["rejection_at"].values())))
    assert S.load_ladder([os.environ["PROBE_LADDER_EXAMPLE"]])["rows"]


# ---------------------------------------------------------------- C1: direction and location

def test_planted_step_is_detected_at_43_to_17_in_the_predicted_direction(planted):
    res, out = planted
    c1 = res["confirmatory"]["C1"]
    assert c1["run"] and c1["method"] == "exact" and c1["n_blocks"] == 5
    assert c1["n_arrangements"] == 24 ** 5 and c1["family"] == "marcus"
    assert c1["p"] <= 0.25 ** 5                       # the end-step bound, and below 0.05/5
    assert c1["argmax_step"] == [[188, 162, 43], [17]]
    assert c1["argmax_is_predicted_step"] is True
    top = c1["contrasts_localisation_only"][int(np.argmin(
        [c["p_adj"] for c in c1["contrasts_localisation_only"]]))]
    assert top["difference"] > 0.9                    # log(1-AUC) ROSE: performance fell
    assert c1["isotonic"]["p"] < 0.01 and c1["isotonic"]["pooled"] == [[188, 162, 43], [17]]
    assert "localisation only" in out and "predicted step [188, 162, 43] | [17]: YES" in out


def test_sign_convention_matches_the_class_count_reading(tmp_path):
    """fine->coarse + "increasing" IS "log(1-AUC) decreases as classes are added"."""
    cells = cells_of(write_ladder(tmp_path, ladder_values(step=0.15)))
    ours = S.trend_test(cells, "bvc_resonant", "linear", [1, 2, 3, 4, 5])
    keys = [k for k in cells if k[:2] == ("bvc_resonant", "linear")]
    ref = max_t_trend([cells[k]["log1m_auc"] for k in keys], [k[2] for k in keys],
                      [k[3] for k in keys], alternative="decreasing", family="marcus",
                      exact=True)             # default order: ascending class count
    assert ref["levels"] == [17, 43, 162, 188]
    assert ours["p"] == pytest.approx(ref["p"], rel=1e-12) and 0 < ours["p"] < 1


def test_step_in_the_wrong_direction_gives_a_one_sided_p_near_one(tmp_path):
    cells = cells_of(write_ladder(tmp_path, ladder_values(step=-1.0)))
    r = S.trend_test(cells, "bvc_resonant", "linear", [1, 2, 3, 4, 5], S.C1["predicted_step"])
    assert r["p"] > 0.95 and r["isotonic"]["p"] > 0.5


def test_null_is_not_rejected(tmp_path):
    cells = cells_of(write_ladder(tmp_path, ladder_values(step=0.0)))
    r = S.trend_test(cells, "bvc_resonant", "linear", [1, 2, 3, 4, 5], S.C1["predicted_step"])
    assert r["p"] > 0.2 and r["isotonic"]["p"] > 0.2


def test_missing_cell_drops_the_seed_block_and_says_so(tmp_path):
    root = write_ladder(tmp_path, ladder_values(step=1.0), omit={(43, 2)})
    res, out = run(root, tmp_path / "out")
    c1 = res["confirmatory"]["C1"]
    assert c1["blocks_dropped_incomplete"] == {"2": [43]} and c1["blocks_used"] == [1, 3, 4, 5]
    assert c1["n_blocks"] == 4 and c1["n_arrangements"] == 24 ** 4
    assert set(res["missing_cells"]) == {f"{t}/{k}" for t in TASKS for k in S.PROBES}
    assert all(v == [[43, 2]] for v in res["missing_cells"].values())
    assert "seed blocks dropped whole (missing levels): seed 2: [43]" in out
    assert "MISSING cells, every task, both probes: seed 2: levels [43]" in out
    # the pairwise contrasts keep seed 2 wherever both of its cells exist
    pw = {(r["fine"], r["coarse"]): r for r in res["pairwise_exploratory"]["bvc_resonant"]["linear"]}
    assert pw[(162, 17)]["n_pairs"] == 5 and pw[(43, 17)]["n_pairs"] == 4


def test_single_file_is_handled_without_running_any_test(tmp_path):
    root = write_ladder(tmp_path, ladder_values(), seeds=(3,))
    res, out = run(root, tmp_path / "out")
    assert res["confirmatory"]["C1"]["run"] is False
    assert "fewer than 2 complete seed blocks — not run" in out
    assert all(not m["estimable"] for m in res["mde"])
    assert all(not r["estimable"] for t in res["pairwise_exploratory"].values()
               for k in t.values() for r in k)
    assert [h["status"] for h in res["confirmatory"]["holm_family"]] == ["pending"] * 5


# ---------------------------------------------------------------- order of operations

def test_mde_is_printed_before_any_contrast(planted):
    res, out = planted
    i_mde, i_sd = out.index("MINIMUM DETECTABLE EFFECT"), out.index("per-level standard deviation")
    i_con = out.index("== 2. CONTRASTS")
    assert i_mde < i_sd < i_con
    for marker in ("C1:", "S1:", "S2:", "S6:", "Holm", "PAIRWISE", " p=", "t=", "diff=", "mean "):
        assert marker in out and out.index(marker) > i_con, marker
    assert [(m["task"], m["probe"]) for m in res["mde"]] == \
        [(t, k) for t in S.SEEN_TASKS for k in S.PROBES]          # already-seen tasks only
    for m in res["mde"]:
        assert m["n_pairs"] == 5 and m["multiplier"] == pytest.approx(1.66, abs=0.005)
        assert m["mde"] == pytest.approx(m["multiplier"] * m["sd_paired_diff"])
    assert set(res["levels"]) == set(TASKS)                       # spreads for EVERY task


def test_mde_uses_the_17_minus_162_paired_differences(tmp_path):
    value = ladder_values(step=0.4)
    cells = cells_of(write_ladder(tmp_path, value))
    d = [value("bvc_resonant", "linear", 17, s) - value("bvc_resonant", "linear", 162, s)
         for s in range(1, 6)]
    m = S.mde_row(cells, "bvc_resonant", "linear", [1, 2, 3, 4, 5])
    assert m["sd_paired_diff"] == pytest.approx(np.std(d, ddof=1), rel=1e-12)


def test_report_never_prints_a_bare_sigma_or_no_effect(planted):
    _, out = planted
    low = out.lower()
    assert "sigma" not in low and "σ" not in out and "no effect" not in low


# ---------------------------------------------------------------- pairwise contrasts

def test_pairwise_is_a_paired_t_on_n_minus_1_df_with_the_sign_flip_beside_it(planted):
    res, out = planted
    rows = {(r["fine"], r["coarse"]): r for r in res["pairwise_exploratory"]["bvc_resonant"]["linear"]}
    assert set(rows) == set(map(tuple, S.PAIRS)) and len(rows) == 6
    table = {(r["level"], r["seed"]): r["log1m_auc"] for r in res["table"]
             if (r["task"], r["probe"]) == ("bvc_resonant", "linear")}
    r = rows[(43, 17)]
    a, b = [table[(17, s)] for s in range(1, 6)], [table[(43, s)] for s in range(1, 6)]
    ref = stats.ttest_rel(a, b)
    assert r["df"] == 4 and r["n_pairs"] == 5 and r["mean_diff"] > 0.9   # coarser − finer
    assert r["t"] == pytest.approx(ref.statistic, rel=1e-12)
    assert r["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert r["sign_flip"] == {"p": 2 / 2 ** 5, "floor": 2 / 2 ** 5, "n_arrangements": 32}
    assert r["holm_reject"] is True
    assert "sign-flip p=0.0625 (floor 0.0625)" in out and "df=4" in out


def test_sign_flip_p_is_exact():
    assert S.sign_flip_p([1.0, 2.0, 3.0])["p"] == 2 / 8
    # |sum| >= 2 among the 8 sign patterns of (1, 2, -1): +-(1+2-1), +-(1+2+1), +-(-1+2+1)
    assert S.sign_flip_p([1.0, 2.0, -1.0])["p"] == 6 / 8


def test_drop_pairs_needs_a_reason_and_is_recorded(tmp_path):
    root = write_ladder(tmp_path, ladder_values(step=1.0))
    with pytest.raises(SystemExit, match="--drop-reason"):
        S.main([str(root), "--out", str(tmp_path / "out"), "--drop-pairs", "2"])
    why = "162-class seed 2 ran on a different GPU model"
    res, out = run(root, tmp_path / "out", "--drop-pairs", "2", "--drop-reason", why)
    assert res["dropped_pairs"] == {"seeds": [2], "reason": why}
    assert res["seeds_used"] == [1, 3, 4, 5] and res["missing_cells"] == {}
    assert res["confirmatory"]["C1"]["n_blocks"] == 4
    assert all(r["n_pairs"] == 4 and r["df"] == 3
               for r in res["pairwise_exploratory"]["bvc_resonant"]["linear"])
    assert {r["seed"] for r in res["table"] if r["dropped_pair"]} == {2}
    assert f"DROPPED seed indices [2] from every contrast. Reason: {why}" in out
    with pytest.raises(SystemExit, match="Refusing to overwrite"):      # an earlier look is a record
        S.main([str(root), "--out", str(tmp_path / "out")])


# ---------------------------------------------------------------- S2: equivalence

def test_tost_bound_is_ln_1p1_in_the_endpoints_log_base():
    assert S.LOG_BASE == math.e
    assert S.tost_bound() == pytest.approx(math.log(1.1), rel=1e-15) == pytest.approx(MEI_LOG)
    assert S.tost_bound(10.0) == pytest.approx(math.log10(1.1), rel=1e-15)
    # ... and the endpoint really is a natural log: probe.py on a toy with AUC = 0.75
    P = _mod("probe", "experiments/EVAL/probe.py")
    value, censored, auc = P.log1m_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.5, 0.9]))
    assert auc == 0.75 and not censored
    assert value == pytest.approx(math.log(0.25) / math.log(S.LOG_BASE), rel=1e-15)


def test_equivalence_wording_and_smallest_bound(tmp_path):
    tight = cells_of(write_ladder(tmp_path / "tight", ladder_values(noise=0.01)))
    e = S.equivalence(tight, "ee_vs_mm", "linear", [1, 2, 3, 4, 5])
    assert e["target_bound"] == pytest.approx(math.log(1.1))
    assert e["largest_pair"]["equivalent_at_target"] and e["p"] < 0.05
    assert e["verdict"].startswith("equivalent within ±0.0953")
    assert abs(e["largest_pair"]["mean_diff"]) == max(abs(r["mean_diff"]) for r in e["pairs"])

    loose = cells_of(write_ladder(tmp_path / "loose", ladder_values(noise=0.3)))
    e = S.equivalence(loose, "ee_vs_mm", "linear", [1, 2, 3, 4, 5])
    top = e["largest_pair"]
    x = max(abs(top["ci90"][0]), abs(top["ci90"][1]))
    assert not top["equivalent_at_target"] and x > e["target_bound"]
    assert top["smallest_bound_passed"] == x
    assert e["verdict"].startswith(f"inconclusive at ±10%; equivalent within ±{x:.4f}")
    assert e["all_pairs_companion"]["smallest_bound_passed_by_all"] >= x
    assert "no effect" not in e["verdict"]


def test_censored_cells_are_bounds_and_never_produce_a_nan(tmp_path):
    """Two levels at AUC == 1 in every seed: their paired differences are all exactly 0."""
    root = write_ladder(tmp_path, ladder_values(),
                        censor={("ee_vs_mm", 188), ("ee_vs_mm", 162)})
    res, out = run(root, tmp_path / "out")

    def no_nan(name):
        raise AssertionError(f"{name} in seed_level_results.json")
    json.loads((tmp_path / "out" / "seed_level_results.json").read_text(), parse_constant=no_nan)
    pw = {(r["fine"], r["coarse"]): r for r in res["pairwise_exploratory"]["ee_vs_mm"]["linear"]}
    assert not pw[(188, 162)]["estimable"] and "zero variance" in pw[(188, 162)]["reason"]
    assert pw[(162, 17)]["estimable"] and pw[(162, 17)]["is_bound"] and pw[(162, 17)]["mean_diff"] > 10
    e = res["secondary"]["S2"]["ee_vs_mm"]["linear"]
    assert e["n_pairs_estimable"] == 5 and e["largest_pair"]["is_bound"]
    assert e["verdict"].startswith("inconclusive at ±10%; equivalent within ±")
    assert "[BOUND: a cell reached AUC=1]" in out and "censored in 5" in out
    assert res["secondary"]["S6"]["ee_vs_mm"]["n_pairs"] == 5


# ---------------------------------------------------------------- S6, C4, Holm

def test_mlp_sign_agreement(tmp_path, planted):
    assert planted[0]["secondary"]["S6"]["bvc_resonant"]["n_pairs"] == 6
    # the three pairs that contain the 17-class level carry the planted step
    cells = cells_of(write_ladder(tmp_path, ladder_values(step=1.0, mlp_sign=-1.0)))
    pw = {k: S.pairwise_table(cells, "bvc_resonant", k, [1, 2, 3, 4, 5]) for k in S.PROBES}
    g = S.sign_agreement(pw["linear"], pw["mlp"])
    flipped = [p for p in g["pairs"] if p["coarse"] == 17]
    assert len(flipped) == 3 and all(p["linear"] == 1 and p["mlp"] == -1 for p in flipped)
    assert g["n_agree"] <= 3 and g["n_pairs"] == 6


def test_c4_sign_pattern_on_a_synthetic_fixture():
    assert S.C4_PREDICTED == {"bvc_4prong": (-1, -1, 0), "visible_content": (1, 0, 1)}
    hit = S.c4_sign_pattern({"bvc_4prong": [-0.4, -0.3, 0.01], "visible_content": [0.5, -0.02, 0.2]},
                            ci={"bvc_4prong": [(-0.5, -0.3), (-0.4, -0.2), (-0.05, 0.07)],
                                "visible_content": [(0.4, 0.6), (-0.10, -0.01), (0.1, 0.3)]})
    assert hit["n_signed_cells"] == 4 and hit["n_match"] == 4
    assert hit["p_at_least"] == 1 / 16 == hit["p_floor"]
    assert hit["zero_cells_consistent"] == [True, False]     # second zero cell excludes 0
    assert len(hit["cells"]) == 6 and "descriptive" in hit["inference_level"]
    assert [(c["task"], c["draw"], c["seed"]) for c in hit["cells"]][:3] == \
        [("bvc_4prong", 1, 1), ("bvc_4prong", 2, 2), ("bvc_4prong", 3, 3)]

    miss = S.c4_sign_pattern({"bvc_4prong": [0.4, 0.3, 0.0], "visible_content": [-0.5, 0.0, -0.2]})
    assert miss["n_match"] == 0 and miss["p_at_least"] == 1.0
    assert miss["zero_cells_consistent"] == [None, None]     # no interval, not scored
    half = S.c4_sign_pattern({"bvc_4prong": [-0.4, 0.3, 0.0], "visible_content": [0.5, 0.0, -0.2]})
    assert half["n_match"] == 2 and half["p_at_least"] == 11 / 16
    assert "DESCRIPTIVE" in S.c4_sign_pattern.__doc__ and "ONE run per draw" in S.c4_sign_pattern.__doc__


def test_holm_family_is_five_with_pending_members(planted):
    fam = planted[0]["confirmatory"]["holm_family"]
    assert [h["test"] for h in fam] == ["C1", "C2", "C3", "C4", "C5"]
    assert [h["status"] for h in fam] == ["available"] + ["pending"] * 4
    assert all(h["family_size"] == 5 and h["threshold_if_smallest"] == 0.05 / 5 for h in fam)
    assert fam[0]["p_raw"] == planted[0]["confirmatory"]["C1"]["p"]
    assert fam[0]["reject_whatever_pending"] is True

    def c1(p):
        return S.holm_family([("C1", p)] + [(c, None) for c in S.CONFIRMATORY[1:]])[0]
    assert c1(0.0099)["reject_whatever_pending"] and c1(0.0099)["reject_possible"]
    assert not c1(0.03)["reject_whatever_pending"] and c1(0.03)["reject_possible"]
    assert not c1(0.2)["reject_possible"]
    # nothing pending: it is src.stats.inference.holm, step-down stop included
    full = S.holm_family([("a", 0.001), ("b", 0.04), ("c", 0.03)])
    assert [h["reject_whatever_pending"] for h in full] == [True, False, False]
    assert [h["reject_possible"] for h in full] == [True, False, False]
    sec = planted[0]["secondary"]["holm_family"]
    assert [h["test"] for h in sec] == ["S1 bvc_qcd", "S2 retained_topology", "S2 ee_vs_mm"]
    assert all(h["family_size"] == 3 for h in sec)


def test_provenance_is_recorded(planted):
    prov = planted[0]["provenance"]
    assert prov["script_sha256"] == S._sha(REPO / "experiments/STATS/seed_level.py")
    assert prov["row_alignment_sha256"] == SHA and len(prov["inputs"]) == 5
    assert all(len(f["sha256"]) == 64 for f in prov["inputs"])
    if S.PRESPEC.exists():
        assert prov["prespec_sha256"] == S._sha(S.PRESPEC)
    assert "l162-s1b" in prov["arm_checkpoints"]


# ---------------------------------------------------------------- two implementations, reconciled

def test_leg_stats_welch_agrees_with_scipy_and_src_stats():
    """experiments/FT/leg_stats.py hand-rolls its t tail; nothing else reconciles it."""
    rng = np.random.default_rng(11)
    for nx, ny, shift, scale in ((3, 3, 0.5, 1.0), (5, 4, 2.0, 3.0), (9, 3, -0.2, 0.3), (2, 6, 5.0, 1.0)):
        x, y = rng.normal(shift, scale, nx).tolist(), rng.normal(0, 1, ny).tolist()
        ours, ref = L.welch(x, y), stats.ttest_ind(x, y, equal_var=False)
        assert ours["t"] == pytest.approx(ref.statistic, abs=1e-9)
        assert ours["df"] == pytest.approx(ref.df, abs=1e-9)
        assert ours["p_two_sided"] == pytest.approx(ref.pvalue, abs=1e-6)
        lo, hi = ref.confidence_interval(0.95)
        assert ours["ci95"] == pytest.approx([lo, hi], abs=1e-6)
    # the same hand-rolled tail against the paired t this module uses as primary
    for n in (3, 5):
        d = rng.normal(0.3, 0.5, n)
        t = paired_t(d)
        assert 2 * L._t_sf(abs(t["t"]), n - 1) == pytest.approx(t["p"], abs=1e-6)
