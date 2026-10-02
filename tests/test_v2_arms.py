"""The v2 arm definitions: scripts/build_v2_arms.py and its outputs.

What must hold for each v2 arm to be the contrast it is registered as:
  - it is the base config with another `labels:` block and nothing else (I1),
    the `weights:` block byte-identical (I2), the label map materialized and
    reproduced by the expression (I4), jet_label never re-derived (I6);
  - the random partitions match the 17-class stream shares exactly, now
    permute two-prong decays too, and meet the 2026-10-01 rule: each probe
    pair merged in 2 or 3 of 5, no two pairs with equal or complementary merge
    columns, realised group shares within 5% of the 17-class ones;
  - F0 is flavour-blind everywhere, F1 differs from it by exactly one b/c cut,
    F1r by the same move with a random cut that separates no two classes
    differing only by b <-> c;
  - the axis doses and the levels where the pair and axis rules put each
    probe pair's loss are recomputed from the class names and the maps;
  - the lambda-matched arm's lambda is the one the v1 logs give;
  - leave-one-family-out removes exactly one 17-class group and keeps the
    parent's reweighting.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


v2 = _load("build_v2_arms")
brc = _load("build_rand_control")

BASE = ROOT / "configs" / "data" / "JetClassII_base.yaml"
GRID = json.loads((ROOT / "configs" / "arms" / "v2_grid.json").read_text())
ARMS = {a["name"]: a for a in GRID["arms"]}
ROWS = v2.read_map()
NAMES = v2.names_of(ROWS)
UNITS = brc.exact_share_units()
R16 = v2.column(ROWS, "R16_Q1")
SECTIONS = ["selection", "new_variables", "preprocess", "inputs", "labels",
            "observers", "weights"]


def sections(text):
    hits = [(m.group(1), m.start()) for m in re.finditer(r"^([a-z_]+):", text, re.M)
            if m.group(1) in SECTIONS]
    return {n: text[s:(hits[i + 1][1] if i + 1 < len(hits) else len(text))]
            for i, (n, s) in enumerate(hits)}


def truth(text):
    expr = re.search(r"truth_label:\s*(.*)", text).group(1)
    got = eval(expr, {"__builtins__": {}}, {"jet_label": np.arange(188)})
    return {n: int(g) for n, g in enumerate(np.asarray(got))}


def partition(m):
    g = collections.defaultdict(set)
    for n, k in m.items():
        g[k].add(n)
    return {frozenset(s) for s in g.values()}


def csv_map(path, col):
    with path.open() as f:
        return {int(r["jet_label"]): int(r[col]) for r in csv.DictReader(f)}


def share_profile(m):
    """Stream share of each resonant group, keyed by group id."""
    out = collections.Counter()
    for n in range(v2.QCD_LO):
        out[m[n]] += UNITS[n]
    return out


RAND_V2 = ROOT / "configs" / "labelmaps" / "rand_label_map.v2.csv"
SEL = json.loads((ROOT / "configs" / "labelmaps" / "rand_v2_selection.json").read_text())
REALISED_COUNTS = v2.realised_counts()
FLAV = ROOT / "configs" / "labelmaps" / "flavour_pair_map.v2.csv"
RAND_ARMS = [f"RAND2_p{d}" for d in range(1, 6)]

# where each new config's map is materialized
EXPECTED_MAP = {
    "R63_Q1": v2.column(ROWS, "R63_Q1"),
    "R29_Q1": v2.column(ROWS, "R29_Q1"),
    "R16_Q1_MASS_LM": R16,
    "FLAV_F0": csv_map(FLAV, "FLAV_F0"),
    "FLAV_F1": csv_map(FLAV, "FLAV_F1"),
    "FLAV_F1R": csv_map(FLAV, "FLAV_F1R"),
    **{a: csv_map(RAND_V2, a) for a in RAND_ARMS},
}
NEW_CONFIGS = sorted([ROOT / "configs" / "arms" / "R63_Q1.yaml",
                      ROOT / "configs" / "arms" / "R29_Q1.yaml"]
                     + list((ROOT / "configs" / "arms" / "v2").glob("*.yaml")))


# ----------------------------------------------------------------- registry
def test_registry_holds_the_requested_grid():
    want = {
        **{a: (1, 5) for a in ["L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS",
                                "R16_Q1_MASS"]},
        **{a: (1, 2) for a in RAND_ARMS},
        "FLAV_F0": (1, 5), "FLAV_F1": (1, 5), "FLAV_F1R": (1, 5),
        "R16_Q1_MASS_LM": (2, 5), "MPM": (2, 3),
        "R63_Q1": (3, 5), "R29_Q1": (3, 5),
        **{f"{a}_LOFO4P": (3, 3) for a in ["L188", "L162", "R42_Q1", "R16_Q1", "MPM"]},
    }
    got = {a["name"]: (a["tier"], a["runs"]) for a in GRID["arms"]}
    assert got == want
    assert len(GRID["arms"]) == len(ARMS), "arm names must be unique"


def test_registry_configs_exist_and_state_the_right_width():
    for a in GRID["arms"]:
        text = (ROOT / a["config"]).read_text()
        if a["num_classes"] is None:
            assert a["objective"] == "mpm"
            continue
        assert len(set(truth(text).values())) == a["num_classes"], a["name"]
        assert f"-o num_classes {a['num_classes']}\n" in text, a["name"]


def test_mass_lambdas_in_the_registry():
    lam = json.loads((ROOT / "configs" / "arms" / "v2" / "mass_lambda.v2.json").read_text())
    assert ARMS["L162_MASS"]["mass_lambda"] == ARMS["R16_Q1_MASS"]["mass_lambda"] == 5.0
    assert ARMS["R16_Q1_MASS_LM"]["mass_lambda"] == lam["lambda_m"]
    text = (ROOT / ARMS["R16_Q1_MASS_LM"]["config"]).read_text()
    assert f"--mass-lambda {lam['lambda_m']}\n" in text
    no_mass = [a["name"] for a in GRID["arms"] if a["mass_lambda"] is None
               and "mass_target:" in (ROOT / a["config"]).read_text()]
    assert not no_mass, f"mass configs registered without a lambda: {no_mass}"


def test_partition_seeds_are_recorded():
    assert [ARMS[a]["partition_seed"] for a in RAND_ARMS] == SEL["accepted_seeds"]
    assert (ARMS["FLAV_F0"]["partition_seed"] == ARMS["FLAV_F1"]["partition_seed"]
            == ARMS["FLAV_F1R"]["partition_seed"] == v2.FLAV_SEED)
    assert json.loads(v2.FLAV_JSON.read_text())["F1R"]["seed"] == v2.F1R_SEED


# ------------------------------------------------------ config invariants
@pytest.mark.parametrize("path", NEW_CONFIGS, ids=lambda p: p.name)
def test_new_config_is_the_base_with_another_labels_block(path):
    """I1 and I2: every section but `labels` is the base's, byte for byte."""
    got, base = sections(path.read_text()), sections(BASE.read_text())
    assert set(got) == set(base)
    assert [s for s in SECTIONS if s != "labels" and got[s] != base[s]] == []
    sha = lambda t: hashlib.sha256(t[re.search(r"^weights:", t, re.M).start():]  # noqa: E731
                                   .encode()).hexdigest()
    assert sha(path.read_text()) == sha(BASE.read_text())


@pytest.mark.parametrize("path", NEW_CONFIGS, ids=lambda p: p.name)
def test_new_config_reproduces_its_materialized_map(path):
    """I4 and I6: the expression reads jet_label only and equals the CSV."""
    text = path.read_text()
    expr = re.search(r"truth_label:\s*(.*)", text).group(1)
    assert set(re.findall(r"[A-Za-z_]\w*", expr)) == {"jet_label"}
    assert "aux_genpart" not in "\n".join(
        l for l in text.splitlines() if not l.lstrip().startswith("#"))
    want = EXPECTED_MAP[path.stem]
    assert truth(text) == want
    assert set(want.values()) == set(range(len(set(want.values()))))
    assert not re.search(r"^\s*(num_classes|network_config|fc_params)\s*:", text, re.M)


def test_every_registered_config_shares_one_weights_block():
    shas = {a["config"] for a in GRID["arms"]}
    digests = {hashlib.sha256(t[re.search(r"^weights:", t, re.M).start():].encode())
               .hexdigest() for t in ((ROOT / c).read_text() for c in shas)}
    assert len(digests) == 1


def test_committed_outputs_are_the_builders_own(tmp_path):
    """Nothing here may be a hand edit: a check-only rebuild must pass and the
    configs it would write must equal the committed ones."""
    assert v2.main(["--check-only"]) == 0
    base = BASE.read_text()
    f0, f1, f1r, _ = v2.build_flavour_pair(ROWS, UNITS, REALISED_COUNTS)
    lam = json.loads(v2.MASS_JSON.read_text())["lambda_m"]
    for path, text in v2.build_configs(base, ROWS, f0, f1, f1r, lam).items():
        assert path.read_text() == text, path.name
    assert (f0, f1, f1r) == tuple(EXPECTED_MAP[a] for a in ("FLAV_F0", "FLAV_F1", "FLAV_F1R"))
    assert v2.registry(ROWS, lam, v2.lofo_extra_selection(ROWS)) == GRID


def test_a_rebuild_reproduces_every_committed_output_byte_for_byte():
    """Every file the builder writes, built in memory, equals the committed bytes
    (the label-map csv keeps the csv module's \\r\\n)."""
    outputs, failed = v2.build_outputs()
    assert failed == 0
    assert set(outputs) == set(NEW_CONFIGS) | {v2.MASS_JSON, v2.FLAV_JSON, v2.FLAV_CSV,
                                               v2.PAIRS, v2.AXIS_JSON, v2.GRID}
    for path, text in outputs.items():
        assert path.read_bytes() == text.encode(), path.name


def _committed_selection():
    with v2.RAND_V2.open(newline="") as f:
        return json.loads(v2.RAND_SEL.read_text()), f.read()


@pytest.mark.parametrize("select", [False, True])
def test_a_failed_check_writes_nothing(monkeypatch, select):
    """A failing rebuild used to leave a rewritten flavour map (and, with
    --select-rand, a new selection record and label map) while it reported
    'nothing written'. Now nothing is written unless every check passes, and
    then everything is written together."""
    writes = []
    real_open = pathlib.Path.open

    def guarded_open(p, mode="r", *a, **k):
        if set(mode) & set("wax+"):
            writes.append(p)
            return open(os.devnull, mode, *a, **k)
        return real_open(p, mode, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", lambda p, *a, **k: writes.append(p))
    monkeypatch.setattr(pathlib.Path, "write_bytes", lambda p, *a, **k: writes.append(p))
    monkeypatch.setattr(pathlib.Path, "open", guarded_open)
    monkeypatch.setattr(v2, "select_rand", lambda workers: _committed_selection())
    argv = ["--select-rand", "2"] if select else []
    check = v2.check_config
    monkeypatch.setattr(v2, "check_config", lambda *a: ["forced failure"])
    assert v2.main(argv) == 1
    assert writes == []

    monkeypatch.setattr(v2, "check_config", check)
    assert v2.main(argv + ["--check-only"]) == 0
    assert writes == []
    assert v2.main(argv) == 0
    expected = set(NEW_CONFIGS) | {v2.MASS_JSON, v2.FLAV_JSON, v2.FLAV_CSV, v2.PAIRS,
                                   v2.AXIS_JSON, v2.GRID}
    assert set(writes) == expected | ({v2.RAND_SEL, v2.RAND_V2} if select else set())


def test_select_rand_draws_into_a_temporary_file_and_replays_the_record(monkeypatch):
    """select_rand writes nothing in the repository: the map is drawn into a
    temporary directory and returned as text. Fed the recorded pool, it returns
    the committed selection record. The pool draws (13-30 s each in pure Python)
    and the map writer are stood in for; both are covered by the
    reproducibility tests below. A seed that is not a recorded rule-3 draw
    stands in as one that fails rule 3."""
    import multiprocessing
    cand = {int(k): (tuple(x), True, SEL["max_rel_dev"][k], SEL["partition_sha256"][k])
            for k, x in SEL["merge_vectors"].items()}

    class Pool:
        def __init__(self, n, initializer=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, block, chunksize=1):
            return [(s, *cand.get(s, ((0,) * 7, False, 1.0, ""))) for s in block]

    drawn_to = []

    def map_writer(argv):
        out = pathlib.Path(argv[argv.index("--out") + 1])
        assert argv[argv.index("--seeds") + 1:argv.index("--out")] == \
            [str(s) for s in SEL["accepted_seeds"]]
        drawn_to.append(out)
        out.write_bytes(v2.RAND_V2.read_bytes())
        return 0

    before = {p: p.read_bytes() for p in (v2.RAND_SEL, v2.RAND_V2)}
    monkeypatch.setattr(multiprocessing, "Pool", Pool)
    monkeypatch.setattr(v2.brc, "main", map_writer)
    monkeypatch.setattr(v2, "_fast_available", lambda: True)
    record, text = v2.select_rand(2)
    assert {p: p.read_bytes() for p in before} == before
    assert drawn_to and not drawn_to[0].is_relative_to(ROOT) and not drawn_to[0].exists()
    assert text.encode() == before[v2.RAND_V2]
    assert json.loads(json.dumps(record)) == SEL


# -------------------------------------------------- random partitions v2
@pytest.mark.parametrize("arm", RAND_ARMS)
def test_random_partition_matches_the_17_class_shares_exactly(arm):
    m = EXPECTED_MAP[arm]
    assert len(set(m.values())) == 17
    assert sorted(share_profile(m).values()) == sorted(share_profile(R16).values())
    qcd = {n for n in m if n >= v2.QCD_LO}
    assert {n for n in m if m[n] in {m[q] for q in qcd}} == qcd, "QCD is one class"


@pytest.mark.parametrize("arm", RAND_ARMS)
def test_random_partition_permutes_two_prong_decays(arm):
    """v1 kept two-prong decays in their own stratum, which made its two-prong
    groups equal the 17-class ones (audit must-fix 2)."""
    m = EXPECTED_MAP[arm]
    two = {m[n] for n in range(15)}
    rest = {m[n] for n in range(15, v2.QCD_LO)}
    assert two & rest, f"{arm}: no group mixes two-prong with three/four-prong decays"


def test_random_partitions_are_distinct_controls():
    parts = [partition(EXPECTED_MAP[a]) for a in RAND_ARMS]
    v1 = [partition(csv_map(ROOT / "configs" / "labelmaps" / "rand_label_map.v1.csv",
                            f"RAND_d{d}")) for d in (1, 2, 3)]
    assert len({frozenset(p) for p in parts}) == 5
    for p in parts:
        assert p != partition(R16) and p not in v1


def test_random_partition_is_reproducible_from_its_seed(tmp_path):
    """Resonant-pool draws are identified by the seed alone, so the last
    accepted seed regenerates as column 1 exactly as it was drawn in the pool."""
    out = tmp_path / "p5.csv"
    brc.main(["--pool", "resonant", "--prefix", "RAND2_p",
              "--seeds", str(SEL["accepted_seeds"][-1]), "--out", str(out)])
    assert partition(csv_map(out, "RAND2_p1")) == partition(EXPECTED_MAP["RAND2_p5"])


# -------------------------------------------- balance rule for the five
def test_balance_rule_holds_on_the_committed_partitions():
    """Rule 1: each of the seven probe pairs is merged in 2 or 3 of the 5
    partitions. Rule 2: no two pairs have equal or complementary merge columns."""
    vec = [v2.merge_vector(EXPECTED_MAP[a], NAMES) for a in RAND_ARMS]
    cols = list(zip(*vec))
    assert [sum(c) for c in cols] == list(SEL["accepted_merged_count"].values())
    assert all(2 <= sum(c) <= 3 for c in cols)
    for i, j in [(i, j) for i in range(7) for j in range(i + 1, 7)]:
        assert cols[i] != cols[j] and cols[i] != tuple(1 - x for x in cols[j]), (i, j)
    assert {k: list(c) for k, c in zip(v2.BALANCE_PAIRS, cols)} == SEL["accepted_columns"]
    assert [list(x) for x in vec] == [SEL["merge_vectors"][str(s)]
                                      for s in SEL["accepted_seeds"]]
    assert SEL["pairs"] == {k: list(p) for k, p in v2.BALANCE_PAIRS.items()}
    assert SEL["merged_in"] == [2, 3]
    assert not set(SEL["accepted_seeds"]) & set(v2.SUPERSEDED_SEEDS)


@pytest.mark.parametrize("arm", RAND_ARMS)
def test_rule_3_realised_shares_on_the_committed_partitions(arm):
    """Every group's realised share is within 5% of the realised share of the
    17-class group it was built to match: group g against group g, which is the
    group carrying the same nominal share."""
    m = EXPECTED_MAP[arm]
    counts = REALISED["native_counts"]
    assert share_profile(m) == share_profile(R16), "group g has 17-class group g's share"
    c, t = collections.Counter(), collections.Counter()
    for n in m:
        c[m[n]] += counts[n]
        t[R16[n]] += counts[n]
    assert all(20 * abs(c[g] - t[g]) <= t[g] for g in t), \
        {g: round(c[g] / t[g], 4) for g in t}
    seed = str(SEL["accepted_seeds"][RAND_ARMS.index(arm)])
    assert [round(c[g] / t[g], 6) for g in sorted(t)] == SEL["accepted_realised_ratio"][seed]
    assert max(abs(c[g] / t[g] - 1) for g in t) == pytest.approx(SEL["max_rel_dev"][seed],
                                                                 abs=1e-6)


def test_selection_replays_from_the_record():
    """Uniform over the 5-subsets of the rule-3 draws that meet rules 1 and 2:
    the recorded draws and SELECT_SEED give back the accepted seeds after the
    recorded number of samples, and the pool was extended only while no
    5-subset met all three rules."""
    vectors = {int(k): tuple(x) for k, x in SEL["merge_vectors"].items()}
    assert sorted(vectors) == SEL["rule3_seeds"]
    assert v2.select(vectors) == (SEL["accepted_seeds"], SEL["samples_drawn"])
    assert [list(s) for s in v2.valid_subsets(vectors)] == SEL["valid_subsets"]
    assert SEL["accepted_seeds"] in SEL["valid_subsets"]
    pool = SEL["pool"]
    blocks = v2.POOL_BLOCKS
    used = next(i for i in range(len(blocks)) if blocks[i][-1] == pool["last_seed"])
    assert blocks[0][0] == pool["first_seed"] and pool["draws"] == 100 * (used + 1)
    for i in range(used):
        assert not v2.feasible({s: v for s, v in vectors.items() if s <= blocks[i][-1]})
    assert all(0 <= d <= 0.05 for d in SEL["max_rel_dev"].values())


def test_rule_helpers():
    """rule_ok is rules 1 and 2: column sums in 2-3, no equal or complementary
    columns."""
    rows5 = [(1, 1, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, 0), (1, 1, 0, 0, 0, 0, 0),
             (0, 0, 1, 0, 1, 0, 0), (1, 0, 1, 0, 0, 1, 1)]
    cols = list(zip(*rows5))
    assert all(2 <= sum(c) <= 3 for c in cols)
    assert v2.rule_ok(rows5)
    one, zero = (1,) * 7, (0,) * 7
    assert not v2.rule_ok([one, one, zero, zero, zero]), "every column equal"
    swap = [r[:6] + (1 - r[0],) for r in rows5]       # last column = complement of the first
    assert all(2 <= sum(c) <= 3 for c in zip(*swap)) and not v2.rule_ok(swap)
    assert v2.column_key((1, 0, 0, 1, 1)) == v2.column_key((0, 1, 1, 0, 0))
    pool = dict(enumerate(rows5, start=1))
    assert v2.valid_subsets(pool) == [(1, 2, 3, 4, 5)] and v2.feasible(pool)
    assert not v2.feasible({**{i: one for i in range(1, 4)}, **{i: zero for i in range(4, 7)}})


REALISED = json.loads(v2.REALISED.read_text())
LOADER_DIR = ROOT / "experiments" / "FIGS" / "data" / "v2_loader"


def test_realised_shares_are_the_dry_runs_counts():
    """The derived input: 20 epochs x 10.24M jets of LOADER_JOB, per native
    class, with the source file's digest."""
    r = REALISED
    assert r["source"]["job"] == v2.LOADER_JOB and r["source"]["file"] == v2.LOADER_FILE
    assert len(r["source"]["sha256"]) == 64
    assert r["loader"]["epochs"] == 20 and r["loader"]["samples_per_epoch"] == 10_240_000
    assert sum(r["native_counts"]) == r["total_jets"] == 20 * 10_240_000
    assert r["class_name"] == [NAMES[n] for n in range(len(NAMES))]
    assert r["share"] == [c / r["total_jets"] for c in r["native_counts"]]
    assert SEL["realised_shares"]["sha256"] == hashlib.sha256(
        v2.REALISED.read_bytes()).hexdigest()


def test_realised_shares_come_from_the_committed_dry_run():
    """Once the raw dry-run record is committed, it is the file the derived
    input was made from, and re-deriving gives the committed bytes."""
    raw = sorted(LOADER_DIR.glob("**/dryrun_s176*.json")) if LOADER_DIR.exists() else []
    if not raw:
        pytest.skip(f"{LOADER_DIR.relative_to(ROOT)} does not hold the dry run yet")
    hits = [p for p in raw
            if hashlib.sha256(p.read_bytes()).hexdigest() == REALISED["source"]["sha256"]]
    assert hits, f"no committed dry run has sha256 {REALISED['source']['sha256']}"
    assert json.dumps(v2.realised_from(hits[0]), indent=1) + "\n" == v2.REALISED.read_text()


def test_resonant_pool_is_share_matched_only():
    with pytest.raises(SystemExit):
        brc.main(["--pool", "resonant", "--match", "count", "--out", "/dev/null"])


# ------------------------------------------------------------ F0 and F1
ORB = v2.orbits(NAMES)


def test_orbits_are_quark_flavour_exchange_classes():
    same = [("label_X_bb", "label_X_cc"), ("label_X_bb", "label_X_sq"),
            ("label_X_YY_bbqq", "label_X_YY_ccqq"), ("label_X_YY_bcev", "label_X_YY_qqev")]
    diff = [("label_X_bb", "label_X_gg"), ("label_X_ee", "label_X_mm"),
            ("label_X_YY_bbqq", "label_X_YY_cqtauhv"), ("label_X_YY_bbg", "label_X_YY_ggb")]
    k = v2.orbit_key
    assert all(k(a) == k(b) for a, b in same)
    assert all(k(a) != k(b) for a, b in diff)
    assert len(ORB) == 43 and sum(map(len, ORB.values())) == 161


def test_f0_is_flavour_blind_and_share_exact():
    f0 = EXPECTED_MAP["FLAV_F0"]
    for o, mem in ORB.items():
        assert len({f0[n] for n in mem}) == 1, f"F0 cuts orbit {o}"
    assert share_profile(f0) == share_profile(R16), "group g carries 17-class group g's share"
    assert partition(f0) != partition(R16)


def test_f1_is_f0_with_one_b_vs_c_cut():
    f0, f1 = EXPECTED_MAP["FLAV_F0"], EXPECTED_MAP["FLAV_F1"]
    cut = [o for o, mem in ORB.items() if len({f1[n] for n in mem}) > 1]
    assert cut == [v2.SPLIT_ORBIT]
    with (ROOT / "hierarchy" / "01_class_master.csv").open() as f:
        has_b = {int(r["jet_label"]): r["has_b"] == "1" for r in csv.DictReader(f)}
    mem = ORB[v2.SPLIT_ORBIT]
    halves = {f1[n] for n in mem if has_b[n]}, {f1[n] for n in mem if not has_b[n]}
    assert all(len(h) == 1 for h in halves) and halves[0] != halves[1]
    idx = {s: n for n, s in NAMES.items()}
    assert f1[idx["label_X_YY_bbqq"]] != f1[idx["label_X_YY_ccqq"]]
    assert f0[idx["label_X_YY_bbqq"]] == f0[idx["label_X_YY_ccqq"]]
    assert share_profile(f1) == share_profile(R16)
    rec = json.loads(v2.FLAV_JSON.read_text())
    moved = {idx[s] for s in rec["split_classes_moved"]} | {
        n for o in rec["orbits_moved_B_to_A"] for n in ORB[o]}
    assert {n for n in f0 if f0[n] != f1[n]} == moved
    assert rec["max_abs_share_mismatch_units"] == {"F0": 0, "F1": 0, "F1R": 0}


def _orbit_of():
    return {n: v2.orbit_key(NAMES[n]) if n < v2.QCD_LO else "QCD" for n in NAMES}


def test_f1_takes_the_move_back_with_the_fewest_cross_orbit_changes():
    """Every share-exact move back is recorded with the native pairs it
    changes; F1 is the one changing the fewest across orbits (479; the
    alphabetical tie-break used to take 504). It keeps F0's semileptonic e/mu
    boundary, which the 504 option lost."""
    f0, f1 = EXPECTED_MAP["FLAV_F0"], EXPECTED_MAP["FLAV_F1"]
    rec = json.loads(v2.FLAV_JSON.read_text())
    idx = {s: n for n, s in NAMES.items()}
    cut = [idx[c] for c in rec["split_classes_moved"]]
    for o in rec["f1_options"]:
        m = v2.moved(f0, cut, o["group_B"], rec["group_A"], o["orbits_moved_B_to_A"], ORB)
        assert v2.pairs_changed(f0, m, _orbit_of()) == o["pairs_changed_from_F0"]
        assert sum(UNITS[n] for t in o["orbits_moved_B_to_A"] for n in ORB[t]) == \
            rec["moved_share_units"]
    cross = sorted(o["pairs_changed_from_F0"]["cross_orbit"] for o in rec["f1_options"])
    assert cross == [479, 495, 504]
    chosen = min(rec["f1_options"], key=lambda o: o["pairs_changed_from_F0"]["cross_orbit"])
    assert chosen["orbits_moved_B_to_A"] == rec["orbits_moved_B_to_A"]
    assert chosen["group_B"] == rec["group_B"]
    assert rec["pairs_changed_from_F0"]["F1"] == {"native_pairs": 600, "cross_orbit": 479,
                                                  "same_orbit": 121}
    assert rec["orbits_moved_B_to_A"] == ["X_YY_QQmm", "X_YY_QQtauhtauh", "X_mm"]
    ev, mv = ORB["X_YY_QQev"][0], ORB["X_YY_QQmv"][0]
    assert f0[ev] != f0[mv] and f1[ev] != f1[mv], "the semileptonic e/mu boundary stays"


def _b_is_c(n):
    """A class name with b and c both read as one heavy flavour, for finding
    b <-> c pairs a second way: two classes of one prefix with equal keys
    differ only by b <-> c. (Fine for b and c, single letters in every name;
    the light axes would also merge s with q this way.)"""
    pre, body = re.match(r"label_(X_YY|X)_(.*)", NAMES[n]).groups()
    return pre, "".join(sorted(body.replace("b", "H").replace("c", "H")))


def _q4_b_c_pairs():
    q4 = sorted(ORB[v2.SPLIT_ORBIT])
    return [(x, y) for i, x in enumerate(q4) for y in q4[i + 1:] if _b_is_c(x) == _b_is_c(y)]


# The F1r cut of commit f0f0b80, drawn without the b <-> c condition.
OLD_F1R_CUT = ["bbbb", "bbcc", "bbss", "cccc", "ccss", "ssqq", "qqqq", "qqqc", "qqqs",
               "qqbc", "qqbs"]


def test_f1r_is_f1_with_a_random_b_unaligned_cut():
    """F1r: the same orbit, the same nominal share, the same move back as F1,
    but the cut is a seeded random 11 of the 22 four-prong hadronic classes
    with 5 or 6 b-containing ones, X->YY->bbqq stays with X->YY->ccqq, no two
    classes that differ only by b <-> c are separated, and every group's
    realised share is within 1% of F1's (the 22 classes share one nominal
    share, but their realised shares are 3x apart)."""
    import random
    f0, f1, f1r = (EXPECTED_MAP[a] for a in ("FLAV_F0", "FLAV_F1", "FLAV_F1R"))
    rec = json.loads(v2.FLAV_JSON.read_text())
    r = rec["F1R"]
    idx = {s: n for n, s in NAMES.items()}
    q4 = sorted(ORB[v2.SPLIT_ORBIT])
    cut = sorted(idx[c] for c in r["classes_moved"])
    twins = _q4_b_c_pairs()
    # replay the sampler
    back = {n for o in rec["orbits_moved_B_to_A"] for n in ORB[o]}

    def realised_dev(c):
        m = v2.moved(f0, c, rec["group_B"], rec["group_A"], rec["orbits_moved_B_to_A"], ORB)
        got, ref = v2.group_sums(m, REALISED_COUNTS), v2.group_sums(f1, REALISED_COUNTS)
        return max(abs(got[g] / ref[g] - 1) for g in ref)

    rng = random.Random(v2.F1R_SEED)
    for k in range(r["samples_drawn"]):
        pick = sorted(rng.sample(q4, 11))
        ok = (sum(v2.has_b(NAMES[n]) for n in pick) in (5, 6)
              and (idx["label_X_YY_bbqq"] in pick) == (idx["label_X_YY_ccqq"] in pick)
              and all((x in pick) == (y in pick) for x, y in twins)
              and realised_dev(pick) <= 0.01)
        assert ok == (k == r["samples_drawn"] - 1)
    assert pick == cut and r["b_classes_moved"] in (5, 6)
    assert r["samples_drawn"] == 11276
    assert v2.F1R_TOL == (1, 100) and r["realised_tolerance_vs_F1"] == [1, 100]
    dev = r["largest_realised_deviation"]
    assert dev["F1R_vs_F1"] == round(realised_dev(cut), 6) <= 0.01
    assert dev["F1_vs_17_class"] == round(v2.largest_dev(f1, R16, REALISED_COUNTS), 6) < 0.01
    assert dev["F0_vs_17_class"] == round(v2.largest_dev(f0, R16, REALISED_COUNTS), 6) < 0.01
    assert dev["F1R_vs_17_class"] == round(v2.largest_dev(f1r, R16, REALISED_COUNTS), 6) < 0.02
    assert r["realised_shares"]["sha256"] == hashlib.sha256(v2.REALISED.read_bytes()).hexdigest()
    # F1r = F0 + the cut to B + F1's move back to A
    assert f1r == v2.moved(f0, cut, rec["group_B"], rec["group_A"],
                           rec["orbits_moved_B_to_A"], ORB)
    assert {n for n in f0 if f0[n] != f1r[n]} == set(cut) | back
    assert {n for n in f1 if f1[n] != f1r[n]} <= set(q4)
    cut_orbits = [o for o, mem in ORB.items() if len({f1r[n] for n in mem}) > 1]
    assert cut_orbits == [v2.SPLIT_ORBIT]
    # shares exact: every class of the orbit carries the same share
    assert len({UNITS[n] for n in q4}) == 1 == len(r["orbit_share_units"])
    assert share_profile(f1r) == share_profile(R16)
    assert f1r[idx["label_X_YY_bbqq"]] == f1r[idx["label_X_YY_ccqq"]]
    bpairs = [(x, y) for x in q4 for y in q4
              if v2.has_b(NAMES[x]) and not v2.has_b(NAMES[y])]
    assert r["orbit_b_nonb_pairs_split"] == {
        "F1": len(bpairs), "F1R": sum(f1r[x] != f1r[y] for x, y in bpairs), "of": len(bpairs)}
    assert 0.4 < r["orbit_b_nonb_pairs_split"]["F1R"] / len(bpairs) < 0.6


def test_f1r_separates_no_two_classes_that_differ_only_by_b_c():
    """PRESPEC A14: the F1r cut of commit f0f0b80 split 4 of the four-prong
    orbit's 11 b <-> c pairs (F1 splits 8), so it kept part of the boundary it
    is there to lack. The cut now splits none, and of the 705,432 ways to cut
    11 of the 22 classes, 67 meet every condition of the sampler; all 67 keep
    F0's merged/split status on the seven balance pairs."""
    f0, f1, f1r = (EXPECTED_MAP[a] for a in ("FLAV_F0", "FLAV_F1", "FLAV_F1R"))
    rec = json.loads(v2.FLAV_JSON.read_text())
    idx = {s: n for n, s in NAMES.items()}
    twins = _q4_b_c_pairs()
    assert len(twins) == 11
    assert rec["b_c_pairs_in_split_orbit"] == [[NAMES[x], NAMES[y]] for x, y in twins]
    split = lambda m: sum(m[x] != m[y] for x, y in twins)  # noqa: E731
    assert (split(f0), split(f1), split(f1r)) == (0, 8, 0)
    old = {idx[f"label_X_YY_{c}"] for c in OLD_F1R_CUT}
    assert sum((x in old) != (y in old) for x, y in twins) == 4

    q4 = sorted(ORB[v2.SPLIT_ORBIT])
    move = lambda c: v2.moved(f0, c, rec["group_B"], rec["group_A"],  # noqa: E731
                              rec["orbits_moved_B_to_A"], ORB)
    ref = v2.group_sums(f1, REALISED_COUNTS)
    status = lambda m: [m[idx[a]] == m[idx[b]] for a, b in v2.BALANCE_PAIRS.values()]  # noqa: E731
    ok = []
    for c in itertools.combinations(q4, 11):
        c = set(c)
        if (sum(v2.has_b(NAMES[n]) for n in c) in (5, 6)
                and all((x in c) == (y in c) for x, y in twins)):
            got = v2.group_sums(move(c), REALISED_COUNTS)
            if all(100 * abs(got[g] - ref[g]) <= ref[g] for g in ref):
                ok.append(sorted(c))
    assert len(ok) == 67 and sorted(idx[c] for c in rec["F1R"]["classes_moved"]) in ok
    assert all(status(move(c)) == status(f0) for c in ok)

    q = rec["quark_axis_pairs"]
    assert q["in_split_orbit"] == {"b<->c": 11, "b<->light": 37, "c<->light": 40}
    assert q["split"] == {"FLAV_F0": {"b<->c": 0, "b<->light": 0, "c<->light": 0},
                          "FLAV_F1": {"b<->c": 8, "b<->light": 27, "c<->light": 0},
                          "FLAV_F1R": {"b<->c": 0, "b<->light": 20, "c<->light": 20}}
    assert q["b<->c_dose"]["FLAV_F0"] == q["b<->c_dose"]["FLAV_F1R"] == {
        "realised": 0.0, "nominal": 0.0}
    assert round(q["b<->c_dose"]["FLAV_F1"]["realised"], 3) == 0.194


def test_a_flavour_pair_off_f0_on_a_balance_pair_fails_the_build(monkeypatch):
    """F1r's merged/split status on the seven balance pairs must be F0's; the
    build fails if it is not."""
    real = v2.build_flavour_pair

    def off(*a):
        f0, f1, f1r, rec = real(*a)
        rec["balance_pair_status"]["FLAV_F1R"]["visible"] = "merged"
        return f0, f1, f1r, rec

    monkeypatch.setattr(v2, "build_flavour_pair", off)
    assert v2.build_outputs()[1] == 1


def test_why_the_cut_is_in_the_four_prong_orbit():
    """Every orbit's share is a multiple of 297, so a share-exact swap needs a
    subset of the cut orbit that is too. The two-prong X->QQ orbit has none;
    the four-prong hadronic orbit needs 11 of its 22 classes, and exactly 11
    contain a b quark."""
    oshare = {o: sum(UNITS[n] for n in mem) for o, mem in ORB.items()}
    assert all(s % 297 == 0 for s in oshare.values())
    qq = ORB["X_QQ"]
    assert not [j for j in range(1, len(qq)) if (j * UNITS[qq[0]]) % 297 == 0]
    q4 = ORB[v2.SPLIT_ORBIT]
    assert [j for j in range(1, len(q4)) if (j * UNITS[q4[0]]) % 297 == 0] == [11]
    assert sum("b" in NAMES[n][len("label_X_YY_"):] for n in q4) == 11


# ----------------------------------------------------------- probe pairs
def test_probe_pair_table_is_current():
    pairs = json.loads(v2.PAIRS.read_text())
    tasks = v2.probe_tasks()
    for arm, st in pairs["status"].items():
        m = v2.column(ROWS, arm) if arm in v2.TREE else EXPECTED_MAP[arm]
        assert st == v2.pair_status(m, tasks, NAMES), arm
    assert set(pairs["status"]) == (set(v2.VOCABS) | set(RAND_ARMS)
                                    | {"FLAV_F0", "FLAV_F1", "FLAV_F1R"})
    assert pairs["flavour_cut_seed"] == {"FLAV_F1R": v2.F1R_SEED}


def test_each_balance_pair_has_a_probe_of_its_own():
    """PRESPEC A10 (2026-10-01): the seven pairs are seven readouts. Each maps
    to a task whose only sub-pair is that pair; X->bc vs X->bq and X->bc vs
    X->cs are no longer read through bc_vs_rest."""
    pairs = json.loads(v2.PAIRS.read_text())
    tasks = v2.probe_tasks()
    short = lambda s: s.replace("label_", "")  # noqa: E731
    assert set(pairs["balance_pairs"]) == set(v2.BALANCE_PAIRS)
    used = []
    for k, (a, b) in v2.BALANCE_PAIRS.items():
        t = pairs["balance_pairs"][k]["task"]
        assert pairs["balance_pairs"][k]["classes"] == [a, b]
        assert [p[0] for p in v2.sub_pairs(tasks[t], NAMES)] == [f"{short(a)}|{short(b)}"]
        used.append(t)
    assert len(set(used)) == 7 and "bc_vs_rest" not in used
    assert pairs["tree_first_merge"]["bc_vs_bq"]["X_bc|X_bq"] == {"level": "R42_Q1",
                                                                 "num_classes": 43}
    assert pairs["tree_first_merge"]["bc_vs_cs"]["X_bc|X_cs"] == {"level": "R29_Q1",
                                                                 "num_classes": 30}


def test_first_merge_level_agrees_with_probe_collapsed_at():
    """probe.py derives, per task, every tree level where it is unmeasurable;
    the first of them is where its first sub-pair merges."""
    tasks = v2.probe_tasks()
    first = v2.first_merge_level(ROWS, tasks, NAMES)
    for t, spec in tasks.items():
        levels = [x["level"] for x in first[t].values() if x["level"]]
        assert min(levels, key=v2.TREE.index) == spec["collapsed_at"][0], t
    assert first["bvc_resonant"]["X_bb|X_cc"] == {"level": "R29_Q1", "num_classes": 30}
    assert first["ee_vs_mm"]["X_ee|X_mm"] == {"level": "R63_Q1", "num_classes": 64}
    assert first["bc_vs_rest"]["X_bc|QCD"] == {"level": None, "num_classes": None}


def test_the_decisive_pair_differs_on_the_four_prong_pair_only():
    """F1 differs from F0 on the four-prong b vs c pair only; F1r agrees with
    F0 on every probe pair, so that pair is the manipulation check."""
    st = json.loads(v2.PAIRS.read_text())["status"]
    f0, f1, f1r = st["FLAV_F0"], st["FLAV_F1"], st["FLAV_F1R"]
    assert "X_YY_bbqq|X_YY_ccqq" in f0["bvc_4prong"]["merged"]
    assert "X_YY_bbqq|X_YY_ccqq" in f1["bvc_4prong"]["split"]
    assert "X_bb|X_cc" in f0["bvc_resonant"]["merged"] and "X_bb|X_cc" in f1["bvc_resonant"]["merged"]
    assert [t for t in f0 if f0[t] != f1[t]] == ["bvc_4prong"]
    assert f1r == f0
    rec = json.loads(v2.FLAV_JSON.read_text())["balance_pair_status"]
    assert [k for k in v2.BALANCE_PAIRS if rec["FLAV_F0"][k] != rec["FLAV_F1"][k]] == ["bbqq/ccqq"]
    assert rec["FLAV_F1R"] == rec["FLAV_F0"]


# ------------------------------------------------------- axes and doses
AXIS = json.loads(v2.AXIS_JSON.read_text())
IDX = {s: n for n, s in NAMES.items()}
GRID_VOCABS = {**{lvl: v2.column(ROWS, lvl) for lvl in
                  ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]},
               **{a: EXPECTED_MAP[a] for a in RAND_ARMS + ["FLAV_F0", "FLAV_F1", "FLAV_F1R"]}}


def test_class_names_read_as_decay_products():
    """Every resonant name is a prefix and a run of decay products, and no two
    classes of one prefix have the same products."""
    from collections import Counter
    assert v2.products("label_X_YY_bctauev") == ("X_YY", Counter(["b", "c", "taue", "v"]))
    assert v2.products("label_X_tauhtaue") == ("X", Counter(["tauh", "taue"]))
    assert v2.products("label_X_YY_qqmv") == ("X_YY", Counter(["q", "q", "m", "v"]))
    with pytest.raises(SystemExit):
        v2.products("label_X_YY_bbxy")
    keys = {(p, tuple(sorted(c.elements())))
            for p, c in (v2.products(NAMES[n]) for n in range(v2.QCD_LO))}
    assert len(keys) == v2.QCD_LO == 161


def _replacement_pairs(axis):
    """The pairs on `axis` found the other way round from the builder: every
    way of replacing one or more products on one side of the axis with
    products on the other, looked up among the class names."""
    slots = v2.AXES[axis]
    key = {}
    for n in range(v2.QCD_LO):
        pre, c = v2.products(NAMES[n])
        key[(pre, tuple(sorted(c.elements())))] = n
    out = set()
    for (pre, toks), n in key.items():
        options = [[t] + [o for one, other in slots if t in one for o in other] for t in toks]
        for combo in itertools.product(*options):
            m = key.get((pre, tuple(sorted(combo))))
            if m is not None and m != n:
                out.add((min(n, m), max(n, m)))
    return sorted(out)


@pytest.mark.parametrize("axis", list(v2.AXES))
def test_axis_pairs_are_found_the_same_two_ways(axis):
    pairs = v2.axis_pairs(NAMES, axis)
    assert pairs == _replacement_pairs(axis)
    assert AXIS["axes"][axis]["native_pairs"] == len(pairs)
    assert AXIS["axes"][axis]["slots"] == [[list(a), list(b)] for a, b in v2.AXES[axis]]


def test_b_c_pairs_are_the_flavour_exchange_pairs():
    """The b <-> c pairs hold the decision's examples, every pair that a full
    exchange of b with c in the name maps one onto the other, and exactly the
    pairs whose names agree once b and c are read as one flavour. Quark axes
    never leave a flavour orbit, the strict axes are parts of the light ones,
    and e <-> mu exchanges e with m and tau_e with tau_mu, not e with tau_mu."""
    bc = set(v2.axis_pairs(NAMES, "b<->c"))
    two = lambda a, b: tuple(sorted((IDX[f"label_{a}"], IDX[f"label_{b}"])))  # noqa: E731
    for a, b in [("bbbb", "cccc"), ("bbss", "ccss"), ("bbqq", "ccqq"), ("qqqb", "qqqc"),
                 ("qqbs", "qqcs"), ("bbcq", "ccbq"), ("bbbb", "bbcc"), ("bbqq", "qqbc")]:
        assert two(f"X_YY_{a}", f"X_YY_{b}") in bc
    swap = str.maketrans("bc", "cb")
    full = set()
    for n in range(v2.QCD_LO):
        pre, body = re.match(r"label_(X_YY|X)_(.*)", NAMES[n]).groups()
        m = IDX.get(f"label_{pre}_{body.translate(swap)}")
        if m is not None and m != n:
            full.add(tuple(sorted((n, m))))
    assert len(full) == 35 and full < bc and len(bc) == 50
    assert bc == {(i, j) for i in range(v2.QCD_LO) for j in range(i + 1, v2.QCD_LO)
                  if _b_is_c(i) == _b_is_c(j)}
    for ax in ("b<->c", "c<->light", "b<->light", "c<->s", "b<->s"):
        assert all(v2.orbit_key(NAMES[i]) == v2.orbit_key(NAMES[j])
                   for i, j in v2.axis_pairs(NAMES, ax)), ax
    for light, strict in v2.STRICT.items():
        assert set(v2.axis_pairs(NAMES, strict)) < set(v2.axis_pairs(NAMES, light))
    em = set(v2.axis_pairs(NAMES, "e<->mu"))
    for a, b in [("X_ee", "X_mm"), ("X_tauhtaue", "X_tauhtaum"), ("X_YY_bcev", "X_YY_bcmv"),
                 ("X_YY_bctauev", "X_YY_bctaumv"), ("X_YY_bbee", "X_YY_bbmm")]:
        assert two(a, b) in em
    assert two("X_YY_bcev", "X_YY_bctaumv") not in em


def _frac(m, pairs, w, out=()):
    from fractions import Fraction
    tot = sum(w[i] * w[j] for i, j in pairs if not {i, j} & set(out))
    split = sum(w[i] * w[j] for i, j in pairs if not {i, j} & set(out) and m[i] != m[j])
    return float(Fraction(split, tot)) if tot else None


def test_dose_leaves_out_every_pair_touching_the_left_out_classes():
    m = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}
    pairs, w = [(0, 1), (0, 2), (3, 4)], [1, 2, 3, 4, 5]
    assert v2.dose(m, pairs, w) == 22 / 25
    assert v2.dose(m, pairs, w, {0}) == 1.0
    assert v2.dose(m, pairs, w, {0, 3}) is None


def test_axis_doses_are_recomputed_from_the_maps():
    """Every dose, overall and with each probe pair's classes left out, on
    realised and nominal weights, recomputed exactly; each axis is left out
    for the probe pairs on it (or on its light parent)."""
    weights = {"realised": REALISED_COUNTS, "nominal": UNITS}
    assert list(AXIS["doses"]) == list(GRID_VOCABS)
    assert AXIS["weights"]["realised"] == {
        "file": "configs/labelmaps/realised_native_shares.v2.json",
        "sha256": hashlib.sha256(v2.REALISED.read_bytes()).hexdigest()}
    for ax in v2.AXES:
        pairs = _replacement_pairs(ax)
        own = [k for k, p in AXIS["probe_pairs"].items() if ax in (p["axis"], p["strict_axis"])]
        assert own, ax
        for v, m in GRID_VOCABS.items():
            d = AXIS["doses"][v][ax]
            assert list(d["excluding"]) == own
            for w, x in weights.items():
                assert d["all"][w] == _frac(m, pairs, x), (v, ax, w)
                for k in own:
                    p = AXIS["probe_pairs"][k]
                    out = {IDX[s] for s in p["signal"] + p["background"]}
                    assert d["excluding"][k][w] == _frac(m, pairs, x, out), (v, ax, k, w)
                    assert (d["excluding"][k][w] == 0) == (d["excluding"][k]["realised"] == 0)


def test_what_the_doses_say_about_the_tree_and_the_flavour_pair():
    """The two finest levels keep every axis whole, the 17-class vocabulary
    keeps none, F0 keeps no quark axis, F1r no b <-> c, and the random
    partitions keep part of every axis."""
    D = AXIS["doses"]
    for ax in v2.AXES:
        assert D["L188"][ax]["all"]["realised"] == D["L162"][ax]["all"]["realised"] == 1.0
        assert D["R16_Q1"][ax]["all"]["realised"] == 0.0
        assert all(0 < D[a][ax]["all"]["realised"] < 1 for a in RAND_ARMS)
    for ax in ("b<->c", "c<->light", "b<->light", "c<->s", "b<->s"):
        assert D["FLAV_F0"][ax]["all"]["realised"] == 0.0
    bc = {a: D[a]["b<->c"]["all"]["realised"] for a in ("FLAV_F1", "FLAV_F1R")}
    assert bc["FLAV_F1R"] == 0.0 < bc["FLAV_F1"]


def test_pair_rule_and_axis_rule_levels():
    """Pair rule: the first level at which a probe pair's classes share a class
    (probe_pairs.v2.json's tree_first_merge, within the levels the grid
    trains). Axis rule: the first level at which the dose of its axis, its
    classes left out, is 0. They agree on the two b vs c pairs (30 classes)
    and on e vs mu (64); X->bc vs X->bq and X->bc vs X->cs are where they part:
    the pair rule puts the loss at 43 and 30 classes, the axis rule at 17."""
    tfm = json.loads(v2.PAIRS.read_text())["tree_first_merge"]
    P = AXIS["probe_pairs"]
    assert AXIS["levels"] == ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]
    assert AXIS["bc_vs_rest_sub_pairs"] == {"X_bc|X_bq": "bc/bq", "X_bc|X_cs": "bc/cs",
                                            "X_bc|X_YY_qqb": "X_bc|X_YY_qqb",
                                            "X_bc|QCD": "X_bc|QCD"}
    assert list(P) == list(v2.BALANCE_PAIRS) + ["X_bc|X_YY_qqb", "X_bc|QCD"]
    for k, p in P.items():
        label = next(l for l, key in AXIS["bc_vs_rest_sub_pairs"].items() if key == k) \
            if p["task"] == "bc_vs_rest" else next(iter(tfm[p["task"]]))
        lvl = tfm[p["task"]][label]["level"]
        assert p["pair_rule_level"] == (lvl if lvl in AXIS["levels"] else None), k
        for key, ax in (("axis_rule_level", p["axis"]),
                        ("strict_axis_rule_level", p["strict_axis"])):
            assert p[key] == (None if ax is None else next(
                (l for l in AXIS["levels"]
                 if AXIS["doses"][l][ax]["excluding"][k]["realised"] == 0), None)), (k, key)
    got = {k: (p["axis"], p["pair_rule_level"], p["axis_rule_level"]) for k, p in P.items()}
    assert got == {"bb/cc": ("b<->c", "R29_Q1", "R29_Q1"),
                   "bbqq/ccqq": ("b<->c", "R29_Q1", "R29_Q1"),
                   "visible": (None, None, None), "bb/bbbb": (None, None, None),
                   "ee/mm": ("e<->mu", "R63_Q1", "R63_Q1"),
                   "bc/bq": ("c<->light", "R42_Q1", "R16_Q1"),
                   "bc/cs": ("b<->light", "R29_Q1", "R16_Q1"),
                   "X_bc|X_YY_qqb": (None, None, None), "X_bc|QCD": (None, None, None)}
    assert {k: (p["strict_axis"], p["strict_axis_rule_level"]) for k, p in P.items()
            if p["strict_axis"]} == {"bc/bq": ("c<->s", "R16_Q1"), "bc/cs": ("b<->s", "R16_Q1")}


def test_cells_where_a_random_partition_merges_a_pair_but_keeps_its_axis():
    """A random partition merges a probe pair that the 17-class vocabulary also
    merges, yet splits some other pair on that pair's axis: the axis account
    predicts it beats the 17-class model there, the pair account that it ties.
    Statuses from probe_pairs.v2.json; 13 cells."""
    st = json.loads(v2.PAIRS.read_text())["status"]
    want = []
    for k, p in AXIS["probe_pairs"].items():
        if not p["axis"]:
            continue
        label = "|".join(s.replace("label_", "") for s in p["signal"] + p["background"])
        for arm in RAND_ARMS:
            d = AXIS["doses"][arm][p["axis"]]["excluding"][k]
            if (label in st[arm][p["task"]]["merged"]
                    and label in st["R16_Q1"][p["task"]]["merged"] and d["realised"] > 0):
                want.append({"partition": arm, "pair": k, "axis": p["axis"], "dose": d,
                             "strict_dose": (AXIS["doses"][arm][p["strict_axis"]]["excluding"][k]
                                             if p["strict_axis"] else None)})
    assert AXIS["cells"] == want
    assert [(c["pair"], c["partition"][-2:]) for c in want] == [
        ("bb/cc", "p1"), ("bb/cc", "p3"), ("bb/cc", "p5"), ("bbqq/ccqq", "p1"),
        ("bbqq/ccqq", "p3"), ("ee/mm", "p1"), ("ee/mm", "p2"), ("ee/mm", "p4"),
        ("bc/bq", "p1"), ("bc/bq", "p2"), ("bc/bq", "p5"), ("bc/cs", "p1"), ("bc/cs", "p5")]


# ------------------------------------------- the pool scan's accelerator
def _pure_draw(seed):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        a = brc.share_draw(ROWS, UNITS, v2.TARGET, seed, 0, brc.POOLS["resonant"])
    return hashlib.sha256(json.dumps([a[n] for n in sorted(a)]).encode()).hexdigest()


def test_the_accepted_partitions_are_the_ones_the_pool_scan_drew():
    """The five maps are written by the unmodified pure-Python draw
    (build_rand_control.main); the record's digests come from the pool scan."""
    for d, s in enumerate(SEL["accepted_seeds"], start=1):
        m = EXPECTED_MAP[f"RAND2_p{d}"]
        got = hashlib.sha256(json.dumps([m[n] for n in sorted(m)]).encode()).hexdigest()
        assert got == SEL["partition_sha256"][str(s)], s


def _fast_draw(seed):
    pytest.importorskip("numba")
    fc = _load("fast_compositions")
    pure = brc.compositions
    try:
        brc.compositions = fc.compositions
        return _pure_draw(seed)
    finally:
        brc.compositions = pure


# Seeds the numba pool scan was compared on against the pure-Python draw when
# it was written (2026-10-01): 226 consecutive seeds from 100, the 27 rule-3
# draws of the pool and 70 others, 323 in all, every one identical. By default
# one rule-3 draw is redrawn both ways (about 30 s); V2_POOL_CROSSCHECK=1 redraws
# all 27 rule-3 draws and the 70 (pure Python, about 30-60 min single-threaded).
CROSSCHECK_OTHERS = [  # random.Random(7).sample(the pool's other seeds, 70)
    408, 484, 498, 509, 578, 587, 592, 610, 618, 676, 697, 808, 847, 875, 902, 949,
    1069, 1119, 1195, 1286, 1340, 1586, 1645, 1794, 1866, 1919, 1936, 2082, 2485,
    2641, 2687, 2766, 3111, 3166, 3351, 3367, 3543, 3551, 3595, 3620, 3670, 3830,
    3932, 4187, 4278, 4477, 4511, 4551, 4610, 4637, 4683, 4712, 4746, 4755, 4799,
    4802, 4850, 4887, 4897, 4898, 4919, 5193, 5262, 5289, 5356, 5456, 5697, 5710,
    5958, 6492]


@pytest.mark.parametrize("seed", SEL["rule3_seeds"] + CROSSCHECK_OTHERS
                         if os.environ.get("V2_POOL_CROSSCHECK") else SEL["rule3_seeds"][:1])
def test_the_numba_pool_scan_draws_what_the_pure_python_draw_does(seed):
    fast = _fast_draw(seed)
    assert fast == _pure_draw(seed)
    if str(seed) in SEL["partition_sha256"]:
        assert fast == SEL["partition_sha256"][str(seed)]


# -------------------------------------------------------------- lambda
LOG = """\
[..] INFO: Epoch #0 training
[..] INFO: Epoch #0 training
[..] INFO: Train AvgLoss: 2.00000, AvgLossReg: 0.10000, AvgLossTot: 2.50000, AvgAcc: 0.4 (lambda=5)
[..] INFO: Epoch #1 training
[..] INFO: Train AvgLoss: 1.00000, AvgLossReg: 0.10000, AvgLossTot: 1.50000, AvgAcc: 0.5 (lambda=5)
"""


def test_log_parser_skips_resumed_partial_epochs():
    ep = v2.parse_train_log(LOG)
    assert sorted(ep) == [0, 1]
    assert [v2.x_of(ep[e]) for e in (0, 1)] == [0.25, 0.5]


def test_log_parser_refuses_an_epoch_with_two_averages():
    with pytest.raises(ValueError):
        v2.parse_train_log(LOG + LOG.splitlines()[-1] + "\n")


def test_lambda_is_recomputed_from_the_committed_inputs():
    d = json.loads(v2.MASS_JSON.read_text())
    runs = d["runs"]
    assert sorted(runs) == sorted(v2.MASS_RUNS["L162"] + v2.MASS_RUNS["R16_Q1"])
    assert all(len(r["epochs"]) == 80 and len(r["train_log_sha256"]) == 64
               for r in runs.values())
    again = v2.mass_lambda(runs)
    assert again == {k: d[k] for k in again}
    pooled = d["variants"]["mean_over_epochs_0_79"]
    x162 = np.mean(pooled["x_162_per_run"])
    x17 = np.mean(pooled["x_17_per_run"])
    # at lambda_m the 17-class model's L_reg/L_cls equals the 162-class one's
    assert abs(d["lambda_m"] / 5 * x17 - x162) / x162 < 0.005
    assert d["lambda_m"] == 1.74


# ---------------------------------------------------------------- LOFO
FAMILY = v2.lofo_natives(ROWS)


def test_lofo_family_is_one_17_class_group_and_three_reweighting_categories():
    idx = {s: n for n, s in NAMES.items()}
    assert {R16[n] for n in FAMILY} == {R16[idx["label_X_YY_bbbb"]]}
    assert FAMILY == sorted(n for n in R16 if R16[n] == R16[idx["label_X_YY_bbbb"]])
    with (ROOT / "hierarchy" / "01_class_master.csv").open() as f:
        cat = {int(r["jet_label"]): r["reweight_group_name"] for r in csv.DictReader(f)}
    cats = {cat[n] for n in FAMILY}
    assert cats == {"label_X_YY_QQQQ", "label_X_YY_QQgg", "label_X_YY_gggg"}
    assert {n for n in cat if cat[n] in cats} == set(FAMILY), "whole categories only"


def test_lofo_selection_removes_exactly_the_family():
    expr = v2.lofo_extra_selection(ROWS)
    keep = eval(expr, {"__builtins__": {}}, {"jet_label": np.arange(188)})
    assert set(np.flatnonzero(~keep)) == set(FAMILY)
    assert set(re.findall(r"[A-Za-z_]\w*", expr)) == {"jet_label"}


def test_lofo_registry_uses_the_parent_config_and_width():
    """Each leave-one-family-out arm is its parent's config, width and
    objective with the family removed; the self-supervised one (PRESPEC A14)
    is the label-free reference for the unseen-family test."""
    for v in v2.VOCABS + ["MPM"]:
        a = ARMS[f"{v}_LOFO4P"]
        assert a["parent"] == v
        assert all(a[k] == ARMS[v][k] for k in ("config", "num_classes", "objective",
                                                "mass_lambda"))
        assert a["extra_selection"] == v2.lofo_extra_selection(ROWS)
    assert ARMS["MPM_LOFO4P"]["objective"] == "mpm"
    assert [a["name"] for a in GRID["arms"] if a["extra_selection"]] == \
        [f"{v}_LOFO4P" for v in v2.VOCABS + ["MPM"]]


def test_lofo_sample_file_check_from_the_pod():
    """Recorded by piping `build_v2_arms.py --lofo-pod-script` into a pod with
    weaver 0.4.17: weaver's own DataConfig.load(extra_selection=...) and
    _apply_selection on one training file."""
    rec = json.loads(v2.LOFO_CHECK.read_text())
    assert rec["weaver_core"] == "0.4.17"
    assert rec["extra_selection"] == v2.lofo_extra_selection(ROWS)
    assert rec["n_family_in_file"] > 0
    for v in v2.VOCABS:
        r = rec["arms"][v]
        local = hashlib.sha256((ROOT / "configs" / "arms" / f"{v}.yaml").read_bytes()).hexdigest()
        assert r["config_sha256_pod"] == r["config_sha256_local"] == local
        assert r["n_family_pass_parent"] > 0 and r["n_family_pass_lofo"] == 0
        assert r["n_pass_lofo"] == r["n_pass_parent"] - r["n_family_pass_parent"]


def test_why_lofo_is_not_a_selection_edit():
    """An exclusion inside `selection:` empties three reweighting categories
    before weaver builds its histograms, and make_weights then fails."""
    pytest.importorskip("weaver")
    import types
    import awkward as ak
    from weaver.utils.data.preprocess import WeightMaker
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, 2000)
    t = ak.Array({"pt": rng.uniform(200, 2500, 2000), "m": rng.uniform(20, 500, 2000),
                  "c0": (lab == 0).astype(int), "c1": (lab == 1).astype(int),
                  "c2": np.zeros(2000, int)})
    cfg = types.SimpleNamespace(
        reweight_branches=("pt", "m"),
        reweight_bins=([200, 600, 1200, 2500], [20, 100, 250, 500]),
        reweight_discard_under_overflow=False, reweight_basewgt=False,
        reweight_method="flat", reweight_threshold=10)
    wm = WeightMaker.__new__(WeightMaker)
    wm._data_config = cfg
    cfg.reweight_classes, cfg.class_weights = ["c0", "c1"], [1.0, 1.0]
    wm.make_weights(t)
    cfg.reweight_classes, cfg.class_weights = ["c0", "c1", "c2"], [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        wm.make_weights(t)


def test_extra_selection_is_applied_after_the_reweighting():
    pytest.importorskip("weaver")
    import inspect
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset
    src = inspect.getsource(SimpleIterDataset.__init__)
    assert src.index("WeightMaker(") < src.index("extra_selection=extra_selection)")
    cfg = DataConfig.load(str(ROOT / "configs" / "arms" / "R16_Q1.yaml"),
                          load_observers=False, extra_selection="(jet_label < 3)")
    assert cfg.selection.endswith("& ((jet_label < 3))")
