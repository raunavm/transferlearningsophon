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

def ladder_values(step=0.0, noise=0.05, rng_seed=7, step_tasks=("bvc_resonant",), mlp_sign=1.0,
                  offsets=None):
    """value(task, probe, level, seed) -> log(1-AUC): seed effect + noise + a step AT 17.

    `offsets` = {level: shift} adds a fixed shift at named levels of `step_tasks`
    on top of the step. It exists to break C1's third clause, "188 ~ 162 ~ 43",
    without touching the first two.
    """
    rng = np.random.default_rng(rng_seed)
    block = rng.normal(0, 0.3, 6)
    eps = rng.normal(0, noise, (len(TASKS), 2, 4, 6))

    def value(task, kind, level, seed):
        i = (TASKS.index(task), S.PROBES.index(kind), S.LEVELS.index(level), seed)
        planted = step * (mlp_sign if kind == "mlp" else 1.0) \
            if task in step_tasks and level == 17 else 0.0
        if task in step_tasks:
            planted += (offsets or {}).get(level, 0.0)
        return float(-4.0 + block[seed] + eps[i] + planted)
    return value


def identical_fine_levels(step=1.0, jitter=1e-4, rng_seed=3):
    """188, 162 and 43 identical to within `jitter`, with the step at 17.

    NOT bit-identical. Paired differences with zero variance are not estimable --
    the same rule pair_contrast uses for two cells censored at one resolution
    floor -- so an exactly flat fixture would exercise the "not run" path rather
    than equivalence. `jitter` is four orders below the ±ln(1.1) bound, so every
    pair inside the set passes with room to spare.
    """
    rng = np.random.default_rng(rng_seed)
    block = rng.normal(0, 0.3, 6)
    eps = rng.normal(0, jitter, (len(TASKS), 2, 4, 6))

    def value(task, kind, level, seed):
        i = (TASKS.index(task), S.PROBES.index(kind), S.LEVELS.index(level), seed)
        return float(-4.0 + block[seed] + eps[i] + (step if level == 17 else 0.0))
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


# ------------------------------------------------- C5: the mass-output 2x2

def mass_values(gain_162=0.0, gain_17=0.0, noise=0.05, rng_seed=11):
    """value(task, probe, corner, seed) for the four corners of the 2x2.

    `gain_162` / `gain_17` are the planted effects of ADDING the mass output at
    each granularity, in log(1-AUC), so the planted difference-in-differences is
    gain_162 - gain_17. Negative is an improvement.
    """
    rng = np.random.default_rng(rng_seed)
    block = rng.normal(0, 0.3, 6)
    eps = rng.normal(0, noise, (len(TASKS), 2, 4, 6))
    corner = {"162": 0, "162+mass": 1, "17": 2, "17+mass": 3}

    def value(task, kind, cell, seed):
        i = (TASKS.index(task), S.PROBES.index(kind), corner[cell], seed)
        base = 0.0 if cell.startswith("162") else 0.4      # 17 is worse to begin with
        planted = (gain_162 if cell == "162+mass" else
                   gain_17 if cell == "17+mass" else 0.0)
        return float(-4.0 + block[seed] + eps[i] + base + planted)
    return value


def mass_doc(value, seed, omit=(), sha=SHA, n_jets=2_000_000, extra_arms=()):
    arms = {f"l162-s{seed}" + ("b" if seed == 1 else ""): "162",
            f"l162mass-s{seed}": "162+mass",
            f"r16q1-s{seed}": "17",
            f"r16q1mass-s{seed}": "17+mass"}
    arms = {a: c for a, c in arms.items() if (c, seed) not in omit}
    arms.update(extra_arms)
    return {"n_jets_total": n_jets, "row_alignment_sha256": sha, "eps_s_default": 0.5,
            "arm_checkpoints": {a: "cd" * 32 for a in arms}, "min_per_class_test": 1000,
            "mlp_threads": 4,
            "tasks": {t: {"eps_s": [0.5], "n": 30000, "n_signal": 15000, "names": ["a", "b"],
                          "collapsed_at": ["R16_Q1"],
                          "arms": {a: {k: _entry(value(t, k, c, seed), k)
                                       for k in S.PROBES} for a, c in arms.items()},
                          "contrasts": {}} for t in TASKS}}


def write_mass(root, value, seeds=(1, 2, 3, 4, 5), **kw):
    for s in seeds:
        d = root / f"s{s}"
        d.mkdir(parents=True)
        (d / "probe_results.json").write_text(json.dumps(mass_doc(value, s, **kw)))
    return root


def mass_cells_of(root, drop=()):
    return S.index_cells(S.load_ladder(root, parse=S.parse_mass_arm)["rows"], drop)


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


# ---------------------------------------------------------------- C1: all three clauses

def test_c1_equivalence_covers_the_three_pairs_inside_the_predicted_equal_set(planted):
    """C1's third clause is "188 ~ 162 ~ 43": three pairs, none of them with 17."""
    assert S.C1["predicted_equal"] == (188, 162, 43)
    e = planted[0]["confirmatory"]["C1"]["clause3_equivalence"]
    assert [(r["fine"], r["coarse"]) for r in e["pairs"]] == [(188, 162), (188, 43), (162, 43)]
    assert e["levels"] == [188, 162, 43] and e["n_pairs_total"] == 3
    assert e["target_bound"] == pytest.approx(math.log(1.1))
    assert e["task"] == "bvc_resonant" and e["probe"] == "linear"
    # the intersection-union p is the worst pair's, and no pair involves 17
    assert e["p"] == max(r["p"] for r in e["pairs"])
    assert all(17 not in (r["fine"], r["coarse"]) for r in e["pairs"])


def test_predicted_equal_set_must_be_given_fine_to_coarse(tmp_path):
    cells = cells_of(write_ladder(tmp_path, ladder_values(step=1.0)))
    with pytest.raises(SystemExit, match="fine -> coarse"):
        S.equivalence_set(cells, "bvc_resonant", "linear", (43, 162, 188), [1, 2, 3, 4, 5])


def test_three_identical_fine_levels_give_a_fully_confirmed_c1(tmp_path):
    root = write_ladder(tmp_path, identical_fine_levels(step=1.0))
    res, out = run(root, tmp_path / "out")
    c1 = res["confirmatory"]["C1"]
    assert [c["verdict"] for c in c1["clauses"]] == ["confirmed"] * 3
    assert c1["composite_verdict"] == "confirmed in clauses 1-3"
    assert c1["overall"] == "confirmed" and c1["n_clauses_confirmed"] == 3
    e = c1["clause3_equivalence"]
    assert e["run"] and e["all_equivalent"] and e["n_equivalent"] == 3
    assert e["not_equivalent"] == [] and e["p"] < 0.05
    assert e["verdict"].startswith("equivalent within ±0.0953")
    assert "CONFIRMED IN CLAUSES 1-3" in out and "no effect" not in out.lower()


def test_one_unequal_pair_gives_the_partial_verdict(tmp_path):
    """The step is in the predicted place and the widest fine pair drifts past the bound.

    The shape is the one the four-granularity probes actually show: the two
    adjacent fine pairs sit inside ±ln(1.1) and the 188-versus-43 pair, whose
    mean is the sum of both, does not. Clauses 1 and 2 are untouched by it.
    """
    root = write_ladder(tmp_path, ladder_values(step=1.0, noise=0.01,
                                                offsets={162: 0.03, 43: 0.10}))
    res, out = run(root, tmp_path / "out")
    c1 = res["confirmatory"]["C1"]
    assert c1["argmax_is_predicted_step"] is True          # the step is still 43 -> 17
    assert [c["verdict"] for c in c1["clauses"]] == ["confirmed", "confirmed", "inconclusive"]
    assert c1["composite_verdict"] == "confirmed in clauses 1-2, inconclusive in clause 3"
    assert c1["overall"] == "partially confirmed" and c1["n_clauses_confirmed"] == 2
    e = c1["clause3_equivalence"]
    assert not e["all_equivalent"] and e["not_equivalent"] == [[188, 43]]
    assert e["n_equivalent"] == 2 and e["widest_pair"] == [188, 43]
    assert e["p"] == max(r["p"] for r in e["pairs"]) > 0.05
    assert e["verdict"].startswith("inconclusive at ±10%; equivalent within ±")
    assert "no effect" not in e["verdict"] and "rejected" not in e["verdict"]
    assert "INCONCLUSIVE IN CLAUSE 3" in out


def test_clause_verdict_vocabulary_and_composite_spans():
    with pytest.raises(SystemExit, match="clause verdicts"):
        S.clause(1, "t", "test", "rejected")
    def mk(*verdicts):
        return [S.clause(i + 1, "t", "test", v) for i, v in enumerate(verdicts)]
    assert S.composite_verdict(mk("confirmed", "confirmed", "inconclusive")) == \
        "confirmed in clauses 1-2, inconclusive in clause 3"
    assert S.composite_verdict(mk("confirmed", "inconclusive", "confirmed")) == \
        "confirmed in clause 1, inconclusive in clause 2, confirmed in clause 3"
    assert S.overall_verdict(mk("confirmed", "confirmed", "confirmed")) == "confirmed"
    assert S.overall_verdict(mk("confirmed", "not confirmed")) == "partially confirmed"
    assert S.overall_verdict(mk("not confirmed", "inconclusive")) == "not confirmed"


def test_c1_clauses_are_not_run_when_the_trend_test_is_not(tmp_path):
    root = write_ladder(tmp_path, ladder_values(), seeds=(3,))
    res, _ = run(root, tmp_path / "out")
    c1 = res["confirmatory"]["C1"]
    assert c1["run"] is False
    assert [c["verdict"] for c in c1["clauses"]] == ["not run"] * 3
    assert c1["overall"] == "not confirmed"


def test_a_zero_variance_pair_is_not_reported_as_equivalence(tmp_path):
    """Two cells censored at one floor give identical differences: not estimable."""
    root = write_ladder(tmp_path, ladder_values(),
                        censor={("bvc_resonant", 188), ("bvc_resonant", 162)})
    res, _ = run(root, tmp_path / "out")
    e = res["confirmatory"]["C1"]["clause3_equivalence"]
    assert [(r["fine"], r["coarse"]) for r in e["pairs"]] == [(188, 43), (162, 43)]
    assert e["n_pairs_estimable"] == 2 and e["n_pairs_total"] == 3
    assert e["all_equivalent"] is False          # a pair that cannot be tested is not a pass


# ------------------------------------------- the keys other tools read must not move

def test_make_tables_still_reads_every_key_it_used_to(planted, tmp_path):
    """experiments/FIGS/make_tables.py is owned elsewhere; nothing it reads may move."""
    MT = _mod("make_tables", "experiments/FIGS/make_tables.py")
    res, _ = planted
    src = tmp_path / "seed_level_results.json"
    src.write_text(json.dumps(res))
    em = MT.Emitter(tmp_path)
    MT.emit_design(em, res, src)
    MT.emit_levels(em, res, src)
    MT.emit_mde(em, res, src)
    MT.emit_tests(em, res, src)
    MT.emit_pairwise(em, res, src, res["levels_fine_to_coarse"][1])
    names = {n for n, _, _ in em.macros}
    for must in ("TrendStatCone", "TrendPCone", "TrendPMinCone", "TrendBlocksCone",
                 "TrendStepCone", "TrendIsoPCone", "TrendHolmCone", "ProbeNJets", "ProbeNSeeds"):
        assert must in names, must
    assert MT.table_tests(res).startswith("\\begin{table*}")
    assert MT.table_probe_ladder(res, "linear").startswith("\\begin{table*}")
    # and the C1 block still carries the trend keys verbatim, plus the new ones
    c1 = res["confirmatory"]["C1"]
    for k in ("task", "probe", "run", "family", "method", "n_blocks", "n_arrangements", "stat",
              "p", "p_min", "end_step_p_bound", "argmax_step", "predicted_step",
              "argmax_is_predicted_step", "contrasts_localisation_only", "isotonic",
              "alternative", "seed_sd_per_level", "blocks_used"):
        assert k in c1, k
    for k in ("prediction", "clauses", "clause3_equivalence", "composite_verdict", "overall",
              "n_clauses", "n_clauses_confirmed"):
        assert k in c1, k


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


# ---------------------------------------------------------------- S9: label recovery

OWN_RUNG = {188: "L188", 162: "L162", 43: "R42_Q1", 17: "R16_Q1"}
# Groups per rung: the config key counts RESONANT groups only, so R42_Q1 is 43
# classes. Only `chance` is computed from this; no test reads the value.
N_GROUPS = {"L188": 188, "L162": 162, "R63_Q1": 64, "R42_Q1": 43, "R29_Q1": 30,
            "R16_Q1": 17, "R3_VIS": 4, "R1_Q1": 2}


ALT = (1.0, -1.0, 1.0, -1.0, 0.0)      # a per-seed wobble that sums to zero


def recovery_values(gap=0.02, noise=0.001, persist=False, shift=0):
    """accuracy(rung, probe, level, seed): S9 planted exactly as it is written.

    A model pays `gap` per rung by which the target rung is FINER than its own
    vocabulary, so for the pair (finer F, coarser C) the advantage of F over C is
    a positive multiple of `gap` at every rung finer than C's own and exactly
    zero at C's own rung and every coarser one. The crossover is C's own rung by
    construction.

    The per-model wobble is DETERMINISTIC and sums to zero over the five seed
    indices, so a rung with no planted advantage has paired differences of mean
    exactly zero and non-zero spread: "not distinguishable" every time. Gaussian
    noise would instead put a true null through a 5% test eight rungs x six
    pairs x two probes deep, and roughly one cell in forty would land on a false
    positive and move a crossover -- a flaky test, not a robust one.

    `shift` moves the crossover `shift` rungs finer than C's own vocabulary:
    clause 1 still holds (the models still tie at and below their own rungs) and
    clause 2 fails. `persist=True` gives each model a flat handicap at every
    rung, so the advantage never reaches zero and both clauses fail.
    """
    base = np.linspace(0.30, 0.85, len(S.RUNGS))          # coarse rungs are easier
    block = np.array([0.0, 0.01, -0.02, 0.005, -0.008, 0.013])   # seed effect, model-free

    def value(rung, kind, level, seed):
        i, j = S.RUNGS.index(rung), S.LEVELS.index(level)
        own = S.RUNGS.index(OWN_RUNG[level])
        penalty = gap * (own if persist else max(0, own - shift - i))
        wobble = noise * ALT[(seed - 1) % len(ALT)] * (1 + i + j + S.PROBES.index(kind))
        return float(base[i] + block[seed] + wobble - penalty)
    return value


def recovery_cell(value, rung, level, seed):
    return {"linear": value(rung, "linear", level, seed),
            "n_fit": 126000, "n_train_available": 140000,
            "mlp": value(rung, "mlp", level, seed),
            "mlp_spread": 0.01, "mlp_n_iter": [19, 19, 17], "mlp_converged": True,
            "mlp_below_linear": False, "n_groups": N_GROUPS[rung],
            "chance": 1.0 / N_GROUPS[rung], "chance_margin": 0.0044, "chance_sigma": 5.0,
            "is_own_rung": OWN_RUNG[level] == rung,
            "is_finer_than_own": S.RUNGS.index(rung) < S.RUNGS.index(OWN_RUNG[level]),
            "not_recovered": False}


def recovery_doc(value, seed, omit=(), sha=SHA, n_test=60_000, extra_arms=(), skip=()):
    arms = {f"{PREFIX[lv]}-s{seed}" + ("b" if (lv, seed) == (162, 1) else ""): lv
            for lv in S.LEVELS if (lv, seed) not in omit}
    arms.update(extra_arms)
    return {"row_alignment_sha256": sha, "n_used": 200_000, "n_train": 140_000,
            "n_test": n_test, "chance_sigma": 5.0,
            "arms": {a: {"own_rung": OWN_RUNG[lv],
                         "rungs": {r: ({"skipped": "one group"} if (a, r) in skip
                                       else recovery_cell(value, r, lv, seed))
                                   for r in S.RUNGS}}
                     for a, lv in arms.items()}}


def write_recovery(root, value, seeds=(1, 2, 3, 4, 5), **kw):
    for s in seeds:
        d = root / f"s{s}"
        d.mkdir(parents=True)
        (d / "label_recovery.json").write_text(json.dumps(recovery_doc(value, s, **kw)))
    return root


def s9_of(root, seeds=(1, 2, 3, 4, 5), drop=()):
    return S.s9_analysis(S.load_recovery(root), [s for s in seeds if s not in drop], drop)


def test_recovery_fixture_mimics_the_committed_schema():
    """The fixture is checked against the one real label-recovery file in the repo."""
    real = json.loads((REPO / "experiments/FIGS/data/label_recovery_v3.json").read_text())
    fake = recovery_doc(recovery_values(), 3)
    assert set(fake) == set(real)
    ra = next(iter(real["arms"].values()))
    fa = next(iter(fake["arms"].values()))
    assert set(fa) == set(ra) and set(fa["rungs"]) == set(ra["rungs"]) == set(S.RUNGS)
    assert set(fa["rungs"]["L188"]) == set(ra["rungs"]["L188"])
    assert S.load_recovery([REPO / "experiments/FIGS/data/label_recovery_v3.json"])["rows"]


def test_recovery_loader_is_a_tidy_table_at_full_precision(tmp_path):
    value = recovery_values()
    data = S.load_recovery(write_recovery(tmp_path, value))
    rows = data["rows"]
    assert len(rows) == len(S.RUNGS) * 2 * 4 * 5
    assert len({(r["rung"], r["probe"], r["level"], r["seed"]) for r in rows}) == len(rows)
    for r in rows:
        assert r["accuracy"] == value(r["rung"], r["probe"], r["level"], r["seed"])
    assert data["own_rung"] == OWN_RUNG and data["n_test"] == 60_000
    s1b = [r for r in rows if r["arm"] == "l162-s1b"]
    assert len(s1b) == len(S.RUNGS) * 2 and {(r["level"], r["seed"]) for r in s1b} == {(162, 1)}


@pytest.mark.parametrize("field, kw", [("row_alignment_sha256", {"sha": "ef" * 32}),
                                       ("n_test", {"n_test": 59_999})])
def test_recovery_mismatched_alignment_is_refused(tmp_path, field, kw):
    value = recovery_values()
    write_recovery(tmp_path, value, seeds=(1, 2, 3, 4))
    write_recovery(tmp_path, value, seeds=(5,), **kw)
    with pytest.raises(SystemExit, match=field):
        S.load_recovery(tmp_path)


def test_recovery_loader_refuses_duplicates_bad_arms_and_unknown_rungs(tmp_path):
    v = recovery_values()
    write_recovery(tmp_path / "dup", v, seeds=(1,), extra_arms={"l162-s1": 162})
    with pytest.raises(SystemExit, match="duplicated cell"):
        S.load_recovery(tmp_path / "dup")
    write_recovery(tmp_path / "bad", v, seeds=(1,), extra_arms={"rand-d1": 17})
    with pytest.raises(SystemExit, match="does not parse"):
        S.load_recovery(tmp_path / "bad")
    write_recovery(tmp_path / "rung", v, seeds=(1,))
    f = tmp_path / "rung" / "s1" / "label_recovery.json"
    d = json.loads(f.read_text())
    d["arms"]["l188-s1"]["rungs"]["R99_Q1"] = d["arms"]["l188-s1"]["rungs"]["L188"]
    f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="is not one of"):
        S.load_recovery(tmp_path / "rung")


def test_recovery_disagreeing_own_rung_is_refused(tmp_path):
    v = recovery_values()
    write_recovery(tmp_path, v, seeds=(1, 2))
    d = json.loads((tmp_path / "s2" / "label_recovery.json").read_text())
    d["arms"]["r16q1-s2"]["own_rung"] = "R29_Q1"
    (tmp_path / "s2" / "label_recovery.json").write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="own vocabulary"):
        S.load_recovery(tmp_path)


def test_s9_advantage_runs_finer_minus_coarser_and_is_paired_by_seed_index(tmp_path):
    value = recovery_values()
    data = S.load_recovery(write_recovery(tmp_path, value))
    rcells = S.index_recovery(data["rows"])
    d = [value("L188", "linear", 188, s) - value("L188", "linear", 17, s) for s in range(1, 6)]
    row = S.recovery_row(rcells, "L188", "linear", 188, 17, [1, 2, 3, 4, 5])
    ref = stats.ttest_rel([value("L188", "linear", 188, s) for s in range(1, 6)],
                          [value("L188", "linear", 17, s) for s in range(1, 6)])
    assert row["n_pairs"] == 5 and row["df"] == 4 and row["seeds"] == [1, 2, 3, 4, 5]
    assert row["mean_diff"] > 0                                    # the finer model wins here
    assert row["mean_diff"] == pytest.approx(float(np.mean(d)), rel=1e-12)
    assert row["t"] == pytest.approx(ref.statistic, rel=1e-9)
    assert row["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert row["sign_flip"] == {"p": 2 / 2 ** 5, "floor": 2 / 2 ** 5, "n_arrangements": 32}
    assert row["advantage"] == "finer better" and "is_bound" not in row


def test_s9_crossover_is_the_coarser_models_own_rung_when_planted_that_way(tmp_path):
    s9 = s9_of(write_recovery(tmp_path, recovery_values()))
    assert set(s9["pairs"]) == {"188_vs_162", "188_vs_43", "188_vs_17", "162_vs_43",
                                "162_vs_17", "43_vs_17"}
    for key, per_probe in s9["pairs"].items():
        for kind in S.PROBES:
            c = per_probe[kind]["crossover"]
            assert c["crossover_rung"] == OWN_RUNG[per_probe[kind]["coarse"]], (key, kind)
            assert c["crossover_at_coarser_own_rung"] is True
            assert c["coarser_own_rung_in_bracket"] is True
            assert c["crossover_bracket"][0] == c["crossover_rung"]
            # every rung finer than the crossover carries a demonstrable advantage
            finer = S.RUNGS[:S.RUNGS.index(c["crossover_rung"])]
            assert all(c["per_rung_advantage"][r] == "finer better" for r in finer)
            assert all(c["per_rung_advantage"][r] == "not distinguishable"
                       for r in S.RUNGS[S.RUNGS.index(c["crossover_rung"]):])
    assert [c["verdict"] for c in s9["clauses"]] == ["confirmed", "confirmed"]
    assert s9["composite_verdict"] == "confirmed in clauses 1-2"
    assert s9["overall"] == "confirmed"
    assert all(w[k]["holds"] for w in s9["wins"].values() for k in S.PROBES)
    assert s9["endpoint"]["lower_is_better"] is False
    assert s9["endpoint"]["difference"].startswith("finer model")


def test_s9_fails_both_clauses_when_the_advantage_never_decays(tmp_path):
    """A flat handicap: the coarser model is behind even at its own vocabulary."""
    s9 = s9_of(write_recovery(tmp_path, recovery_values(persist=True)))
    for per_probe in s9["pairs"].values():
        c = per_probe["linear"]["crossover"]
        assert c["crossover_rung"] is None and c["crossover_bracket"] == [None, None]
        assert c["crossover_at_coarser_own_rung"] is False
        assert c["coarser_own_rung_in_bracket"] is False
        assert c["last_rung_with_advantage"] == S.RUNGS[-1]
    assert [c["verdict"] for c in s9["clauses"]] == ["not confirmed", "not confirmed"]
    assert "0 of 6 model pairs" in s9["clauses"][1]["detail"]
    assert s9["composite_verdict"] == "not confirmed in clauses 1-2"
    assert s9["overall"] == "not confirmed"
    # the 17-class model is beaten at its own rung, which is clause 1's failure
    w = s9["wins"]["17"]["linear"]
    assert w["own_rung"] == "R16_Q1" and not w["holds"] and w["n_beaten"] == w["n_estimable"]
    assert {b["rung"] for b in w["beaten_by"]} == {"R16_Q1", "R3_VIS", "R1_Q1"}


def test_s9_gives_the_partial_verdict_when_the_crossover_is_at_the_wrong_rung(tmp_path):
    """The advantage dies one rung EARLY: the models still tie at and below own."""
    s9 = s9_of(write_recovery(tmp_path, recovery_values(shift=1)))
    for per_probe in s9["pairs"].values():
        b = per_probe["linear"]
        c = b["crossover"]
        assert c["crossover_rung"] == S.RUNGS[S.RUNGS.index(OWN_RUNG[b["coarse"]]) - 1]
        assert c["crossover_at_coarser_own_rung"] is False
        assert c["coarser_own_rung_in_bracket"] is True      # still inside the bracket
    assert [c["verdict"] for c in s9["clauses"]] == ["confirmed", "not confirmed"]
    assert s9["composite_verdict"] == "confirmed in clause 1, not confirmed in clause 2"
    assert s9["overall"] == "partially confirmed"
    assert "0 of 6 model pairs cross over exactly" in s9["clauses"][1]["detail"]
    assert "6 of 6 have that rung inside the crossover bracket" in s9["clauses"][1]["detail"]


def test_s9_holm_is_within_each_pairs_own_ladder_of_rungs(tmp_path):
    s9 = s9_of(write_recovery(tmp_path, recovery_values()))
    rows = s9["pairs"]["188_vs_17"]["linear"]["rungs"]
    assert [r["rung"] for r in rows] == list(S.RUNGS)
    est = [r for r in rows if r["estimable"]]
    assert len(est) == len(S.RUNGS)
    assert [r["holm_reject"] for r in est] == \
        list(map(bool, S.holm([r["p"] for r in est], S.ALPHA)))


def test_s9_missing_cells_are_reported_not_silently_dropped(tmp_path):
    root = write_recovery(tmp_path, recovery_values(), omit={(43, 2)})
    s9 = s9_of(root)
    assert set(s9["missing_cells"]) == {f"{r}/{k}" for r in S.RUNGS for k in S.PROBES}
    assert all(v == [[43, 2]] for v in s9["missing_cells"].values())
    for per_probe in s9["pairs"].values():
        n = [r["n_pairs"] for r in per_probe["linear"]["rungs"]]
        assert set(n) == ({4} if 43 in (per_probe["linear"]["fine"],
                                        per_probe["linear"]["coarse"]) else {5})


def test_s9_skipped_rung_is_recorded(tmp_path):
    root = write_recovery(tmp_path, recovery_values(), skip={("r16q1-s3", "R1_Q1")})
    s9 = s9_of(root)
    assert s9["skipped_cells"] and s9["skipped_cells"][0]["rung"] == "R1_Q1"
    assert s9["skipped_cells"][0]["reason"] == "one group"
    assert s9["missing_cells"]["R1_Q1/linear"] == [[17, 3]]


def test_s9_cli_writes_its_own_file_and_leaves_the_ladder_cli_alone(tmp_path):
    root = write_recovery(tmp_path / "rec", recovery_values())
    out = tmp_path / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert S.main(["--label-recovery", str(root), "--out", str(out)]) == 0
    res = json.loads((out / "s9_label_recovery.json").read_text())
    text = buf.getvalue()
    assert not (out / "seed_level_results.json").exists()
    assert res["secondary"]["S9"]["overall"] == "confirmed"
    assert res["provenance"]["script_sha256"] == S._sha(REPO / "experiments/STATS/seed_level.py")
    assert len(res["provenance"]["inputs"]) == 5
    assert "S9: LABEL RECOVERY ACROSS THE CONTRACTION TREE" in text
    assert "crossover at R16_Q1" in text and "crossover there: YES" in text
    assert "higher is better" in text.lower() and "no effect" not in text.lower()
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        S.main(["--label-recovery", str(root), "--out", str(out)])
    # and the ladder still runs on its own, with no --label-recovery
    ladder = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    res2, _ = run(ladder, tmp_path / "out2")
    assert res2["confirmatory"]["C1"]["run"] is True
    with pytest.raises(SystemExit):
        S.main(["--out", str(tmp_path / "out3")])          # neither input given


def test_s9_and_the_ladder_can_run_in_one_call(tmp_path):
    ladder = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    rec = write_recovery(tmp_path / "rec", recovery_values())
    out = tmp_path / "out"
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main([str(ladder), "--label-recovery", str(rec), "--out", str(out)]) == 0
    assert (out / "seed_level_results.json").exists()
    assert (out / "s9_label_recovery.json").exists()


def test_s9_drop_pairs_is_honoured(tmp_path):
    root = write_recovery(tmp_path, recovery_values())
    s9 = s9_of(root, drop=(2,))
    assert s9["seeds_used"] == [1, 3, 4, 5]
    rows = s9["pairs"]["188_vs_17"]["linear"]["rungs"]
    assert all(r["n_pairs"] == 4 and r["df"] == 3 and 2 not in r["seeds"] for r in rows)


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


def _add_points(root, points):
    """Give every probe entry the extra working points `points` = {eps: (rej,
    bound, n_bkg_pass)}, leaving 0.50 as the default the flat fields mirror."""
    for f in sorted(root.glob("s*/probe_results.json")):
        d = json.loads(f.read_text())
        for t in d["tasks"].values():
            t["eps_s"] = [0.5] + sorted(points)
            for arm in t["arms"].values():
                for kind in S.PROBES:
                    ra = arm[kind]["rejection_at"]
                    for eps, (rej, bound, npass) in points.items():
                        ra[f"{eps:.2f}"] = {"rejection": rej, "eps_b": 1.0 / rej,
                                            "rejection_is_bound": bound,
                                            "n_bkg_pass": npass, "rel_stat_err": 0.1}
        f.write_text(json.dumps(d))


def test_every_working_point_is_summarised_and_90_percent_is_the_headline(tmp_path):
    """docs/PRESPEC_2026-09.md fixed 90 % signal efficiency as the headline
    working point for background rejection, blind, because 50 % leaves no
    background jets at the finer vocabularies. Reading only the probe's default
    point would report the censored number as the headline."""
    root = write_ladder(tmp_path / "in", ladder_values(step=1.0))
    _add_points(root, {0.70: (5000.0, True, 2), 0.90: (300.0, False, 40)})
    res, _ = run(root, tmp_path / "o")
    pts = res["levels"]["bvc_resonant"]["linear"][0]["rejection_points"]
    assert set(pts) == {"0.50", "0.70", "0.90"}, "every recorded point must be summarised"
    assert [p["is_headline"] for p in pts.values()] == [False, False, True]
    assert S.HEADLINE_EPS_S == "0.90"
    assert pts["0.90"]["median"] == 300.0 and pts["0.90"]["n_bound"] == 0
    assert pts["0.90"]["mean_n_bkg_pass"] == 40.0


def test_a_censored_working_point_carries_its_flag_and_its_surviving_count(tmp_path):
    """A point where any seed hit the cap is a statement about the size of the
    test sample, not about the models, so the count of bound seeds and the mean
    number of surviving background jets travel with the number."""
    root = write_ladder(tmp_path / "in", ladder_values(step=1.0))
    _add_points(root, {0.70: (11876.0, True, 0), 0.90: (300.0, False, 40)})
    pts = run(root, tmp_path / "o")[0]["levels"]["bvc_resonant"]["linear"][0]["rejection_points"]
    assert pts["0.70"]["n_bound"] == pts["0.70"]["n_seeds"] == 5
    assert pts["0.70"]["mean_n_bkg_pass"] == 0.0
    assert pts["0.90"]["n_bound"] == 0


def test_the_flat_rejection_fields_still_mirror_the_default_point(tmp_path):
    """The table generator and every earlier analysis read the flat fields. They
    must keep meaning 'the probe's first working point' even now that the
    headline is a different one, or old and new outputs stop being comparable."""
    root = write_ladder(tmp_path / "in", ladder_values(step=1.0))
    _add_points(root, {0.90: (300.0, False, 40)})
    lv = run(root, tmp_path / "o")[0]["levels"]["bvc_resonant"]["linear"][0]
    assert lv["rejection_eps_s"] == 0.5
    assert lv["rejection_median"] == 50.0, "the 0.50 fixture value, not the 0.90 one"
    assert lv["rejection_points"]["0.50"]["median"] == 50.0
    assert lv["headline_eps_s"] == "0.90"


def test_the_headline_point_is_named_in_the_printed_report(tmp_path):
    root = write_ladder(tmp_path / "in", ladder_values(step=1.0))
    _add_points(root, {0.70: (11876.0, True, 0), 0.90: (300.0, False, 40)})
    _, out = run(root, tmp_path / "o")
    assert "<- HEADLINE" in out
    assert "eps_s=0.90" in out and "eps_s=0.70" in out
    assert "CENSORED in 5/5" in out, "a capped point must say so in the report"


# ------------------------------------------------- C5: the mass-output 2x2

def test_the_mass_arm_parser_refuses_everything_the_ladder_parser_accepts_and_back(tmp_path):
    """The two parsers must stay separate. A mass arm reaching the ladder loader
    would claim a level that already has an arm, and a ladder arm reaching the
    mass loader would claim a corner of a 2x2 it is not part of."""
    assert S.parse_mass_arm("l162mass-s3") == ("162+mass", 3)
    assert S.parse_mass_arm("r16q1mass-s1") == ("17+mass", 1)
    assert S.parse_mass_arm("l162-s1b") == ("162", 1), "the alias applies here too"
    assert S.parse_mass_arm("r16q1-s5") == ("17", 5)
    for bad in ("l188-s1", "r42q1-s2", "rand-d1", "l162_mass-s1", "L162MASS-s1",
                "l162mass-s0", "l162mass", "l162mass-s1-x"):
        with pytest.raises(SystemExit, match="does not parse"):
            S.parse_mass_arm(bad)
    # and the ladder parser still refuses the mass names
    for bad in ("l162mass-s1", "r16q1mass-s1"):
        with pytest.raises(SystemExit, match="does not parse"):
            S.parse_arm(bad)


def test_the_mass_loader_refuses_two_arms_claiming_one_corner(tmp_path):
    """TWO run directories answer to the 162-class model at seed 1: mtx-l162-s1,
    trained at 1e-3 and excluded everywhere, and mtx-l162-s1b, the 5e-4 repair
    that counts. They alias to the same corner, so a job that cached both would
    put two different models in one cell. It must refuse, not average them."""
    root = tmp_path / "in"
    (root / "s1").mkdir(parents=True)
    doc = mass_doc(mass_values(), 1)          # already holds l162-s1b
    for t in doc["tasks"].values():
        t["arms"]["l162-s1"] = t["arms"]["l162-s1b"]
    (root / "s1" / "probe_results.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="duplicated cell"):
        S.load_ladder(root, parse=S.parse_mass_arm)


def test_the_did_is_the_interaction_and_its_sign_is_the_written_expectation(tmp_path):
    """Planted: the mass output helps at 17 (-0.5) and not at 162 (0.0). The
    written expectation is exactly that, and in log(1-AUC) -- lower is better --
    it must come out as a POSITIVE difference-in-differences."""
    root = write_mass(tmp_path / "in", mass_values(gain_162=0.0, gain_17=-0.5, noise=0.01))
    cells = mass_cells_of(root)
    did = S.pair_contrast(S.mass_did(cells, "bvc_resonant", "linear", (1, 2, 3, 4, 5)))
    assert did["estimable"] and did["n_pairs"] == 5
    assert did["mean_diff"] == pytest.approx(0.5, abs=0.05), "= gain_162 - gain_17"
    assert did["p"] < 0.01
    # and the per-level gains recover the planted effects separately
    g162 = S.pair_contrast(S.mass_gain(cells, "bvc_resonant", "linear", 162, (1, 2, 3, 4, 5)))
    g17 = S.pair_contrast(S.mass_gain(cells, "bvc_resonant", "linear", 17, (1, 2, 3, 4, 5)))
    assert g162["mean_diff"] == pytest.approx(0.0, abs=0.05)
    assert g17["mean_diff"] == pytest.approx(-0.5, abs=0.05)


def test_a_seed_missing_one_corner_is_dropped_whole_from_the_did(tmp_path):
    """A difference-in-differences needs all four corners. Contributing half of
    one would silently change the estimand."""
    root = write_mass(tmp_path / "in", mass_values(gain_17=-0.5), omit=(("162+mass", 3),))
    cells = mass_cells_of(root)
    did = S.mass_did(cells, "bvc_resonant", "linear", (1, 2, 3, 4, 5))
    assert did["seeds"] == [1, 2, 4, 5], "seed 3 lost a corner and leaves entirely"
    # the one-sided gain at 17 still has all five, which is the point of reporting both
    assert S.mass_gain(cells, "bvc_resonant", "linear", 17, (1, 2, 3, 4, 5))["seeds"] == \
        [1, 2, 3, 4, 5]


def test_c5_joins_the_confirmatory_family_only_when_the_2x2_is_supplied(tmp_path):
    """Without --mass, C5 is pending exactly as C2 and C3 are. With it, C5 carries
    a p and the family size is still five -- Holm was always over all five."""
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    res_no = run(lad, tmp_path / "o1")[0]
    fam_no = res_no["confirmatory"]["holm_family"]
    assert [h["test"] for h in fam_no] == ["C1", "C2", "C3", "C4", "C5"]
    assert [h["status"] for h in fam_no] == ["available"] + ["pending"] * 4
    assert "C5" not in res_no["confirmatory"]

    mass = write_mass(tmp_path / "mass", mass_values(gain_17=-0.5, noise=0.01))
    res, out = run(lad, tmp_path / "o2", "--mass", str(mass))
    fam = res["confirmatory"]["holm_family"]
    assert [h["test"] for h in fam] == ["C1", "C2", "C3", "C4", "C5"]
    assert [h["status"] for h in fam] == ["available", "pending", "pending", "pending",
                                          "available"]
    assert all(h["family_size"] == 5 for h in fam)
    c5 = res["confirmatory"]["C5"]
    assert c5["p"] == next(h for h in fam if h["test"] == "C5")["p_raw"]
    assert c5["task"] == "bvc_resonant"
    assert "mass output x granularity" in out


def test_c5_is_confirmatory_on_one_task_and_the_rest_are_labelled_exploratory(tmp_path):
    """The pre-registration fixed C5 on the b-versus-c probe. The 2x2 measures six
    tasks; the other five are reported but must never be promoted into the family."""
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    mass = write_mass(tmp_path / "mass", mass_values(gain_17=-0.5))
    res, out = run(lad, tmp_path / "o", "--mass", str(mass))
    c5 = res["confirmatory"]["C5"]
    assert c5["confirmatory"]["task"] == "bvc_resonant"
    assert {b["task"] for b in c5["exploratory"]} == set(TASKS) - {"bvc_resonant"}
    assert len(c5["exploratory"]) == 5
    # the family carries exactly one C5 entry, whatever the other tasks did
    fam = res["confirmatory"]["holm_family"]
    assert len([h for h in fam if h["test"].startswith("C5")]) == 1
    assert "EXPLORATORY" in out and "no multiplicity family" in out


def test_c5_refuses_a_2x2_scored_on_different_jets_from_the_ladder(tmp_path):
    """C1 and C5 share a multiplicity family, so they have to be the same
    measurement. A different row alignment means different test jets."""
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    mass = write_mass(tmp_path / "mass", mass_values(), sha="ef" * 32)
    with pytest.raises(SystemExit, match="different set of test jets"):
        run(lad, tmp_path / "o", "--mass", str(mass))


def test_the_mass_flag_needs_the_ladder_inputs(tmp_path):
    mass = write_mass(tmp_path / "mass", mass_values())
    with pytest.raises(SystemExit):
        S.main(["--out", str(tmp_path / "o"), "--mass", str(mass)])


def test_the_mass_2x2_never_reaches_the_ladder_analysis(tmp_path):
    """The whole reason the corners are strings: if a mass arm were handed to the
    ladder loader it would claim level 162 or 17 and collide with a real arm."""
    mass = write_mass(tmp_path / "mass", mass_values(), seeds=(1,))
    with pytest.raises(SystemExit, match="does not parse"):
        S.load_ladder(mass)


def test_a_c5_that_ran_but_is_not_estimable_is_not_reported_as_unmeasured(tmp_path):
    """Holm keeps C5 pending because there is no p to use, and that arithmetic is
    right. But "pending" is also the word for a test nobody has run, and a C5 that
    ran on one seed and could not be estimated is a different fact about the study.
    The analysis has to be able to tell them apart."""
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    mass = write_mass(tmp_path / "mass", mass_values(gain_17=-0.5), seeds=(1,))
    res, out = run(lad, tmp_path / "o", "--mass", str(mass))
    c5 = res["confirmatory"]["C5"]
    assert c5["measured"] is True
    assert c5["estimable"] is False
    assert "fewer than 2" in c5["not_estimable_reason"]
    assert c5["p"] is None
    # Holm is still conservative about it, which is correct
    assert next(h for h in res["confirmatory"]["holm_family"]
                if h["test"] == "C5")["status"] == "pending"
    assert "MEASURED, NOT ESTIMABLE" in out


def test_a_fully_estimable_c5_says_so(tmp_path):
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    mass = write_mass(tmp_path / "mass", mass_values(gain_17=-0.5, noise=0.01))
    res, out = run(lad, tmp_path / "o", "--mass", str(mass))
    c5 = res["confirmatory"]["C5"]
    assert c5["measured"] is True and c5["estimable"] is True
    assert c5["not_estimable_reason"] is None
    assert "MEASURED, NOT ESTIMABLE" not in out


def test_the_exploratory_lines_carry_the_nonlinear_probe_too(tmp_path):
    """D6 does not make an exception for a line that carries no verdict."""
    lad = write_ladder(tmp_path / "lad", ladder_values(step=1.0))
    mass = write_mass(tmp_path / "mass", mass_values(gain_17=-0.5))
    _, out = run(lad, tmp_path / "o", "--mass", str(mass))
    block = out.split("EXPLORATORY")[1].split("Holm over")[0]
    assert block.count("mlp") == 5, "one nonlinear line per exploratory task"
