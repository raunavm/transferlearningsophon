"""The v2 arm definitions: scripts/build_v2_arms.py and its outputs.

What must hold for each v2 arm to be the contrast it is registered as:
  - it is the base config with another `labels:` block and nothing else (I1),
    the `weights:` block byte-identical (I2), the label map materialized and
    reproduced by the expression (I4), jet_label never re-derived (I6);
  - the random partitions match the 17-class stream shares exactly and now
    permute two-prong decays too;
  - F0 is flavour-blind everywhere, F1 differs from it by exactly one b/c cut;
  - the lambda-matched arm's lambda is the one the v1 logs give;
  - leave-one-family-out removes exactly one 17-class group and keeps the
    parent's reweighting.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
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
FLAV = ROOT / "configs" / "labelmaps" / "flavour_pair_map.v2.csv"
RAND_ARMS = [f"RAND2_p{d}" for d in range(1, 6)]

# where each new config's map is materialized
EXPECTED_MAP = {
    "R63_Q1": v2.column(ROWS, "R63_Q1"),
    "R29_Q1": v2.column(ROWS, "R29_Q1"),
    "R16_Q1_MASS_LM": R16,
    "FLAV_F0": csv_map(FLAV, "FLAV_F0"),
    "FLAV_F1": csv_map(FLAV, "FLAV_F1"),
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
        "FLAV_F0": (2, 2), "FLAV_F1": (2, 2), "R16_Q1_MASS_LM": (2, 5), "MPM": (2, 3),
        "R63_Q1": (3, 5), "R29_Q1": (3, 5),
        **{f"{a}_LOFO4P": (3, 3) for a in ["L188", "L162", "R42_Q1", "R16_Q1"]},
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
    assert ARMS["FLAV_F0"]["partition_seed"] == ARMS["FLAV_F1"]["partition_seed"]


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
    f0, f1, _ = v2.build_flavour_pair(ROWS, UNITS)
    lam = json.loads(v2.MASS_JSON.read_text())["lambda_m"]
    for path, text in v2.build_configs(base, ROWS, f0, f1, lam).items():
        assert path.read_text() == text, path.name
    assert (f0, f1) == (EXPECTED_MAP["FLAV_F0"], EXPECTED_MAP["FLAV_F1"])
    assert v2.registry(ROWS, lam, v2.lofo_extra_selection(ROWS)) == GRID


def test_a_rebuild_reproduces_every_committed_output_byte_for_byte():
    """Every file the builder writes, built in memory, equals the committed bytes
    (the label-map csv keeps the csv module's \\r\\n)."""
    outputs, failed = v2.build_outputs()
    assert failed == 0
    assert set(outputs) == set(NEW_CONFIGS) | {v2.MASS_JSON, v2.FLAV_JSON, v2.FLAV_CSV,
                                               v2.PAIRS, v2.GRID}
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
    expected = set(NEW_CONFIGS) | {v2.MASS_JSON, v2.FLAV_JSON, v2.FLAV_CSV, v2.PAIRS, v2.GRID}
    assert set(writes) == expected | ({v2.RAND_SEL, v2.RAND_V2} if select else set())


def test_select_rand_draws_into_a_temporary_file_and_replays_the_record(monkeypatch):
    """select_rand writes nothing in the repository: the map is drawn into a
    temporary directory and returned as text. Fed the recorded pool, it returns
    the committed selection record. The pool draws (about 30 s each) and the map
    writer are stood in for; both are covered by the reproducibility tests above."""
    import multiprocessing
    vectors = {int(k): tuple(x) for k, x in SEL["merge_vectors"].items()}

    class Pool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, block):
            return [(s, vectors[s]) for s in block]

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
    """Each of the seven probe pairs is merged in 2 or 3 of the 5 partitions."""
    vec = [v2.merge_vector(EXPECTED_MAP[a], NAMES) for a in RAND_ARMS]
    assert [sum(c) for c in zip(*vec)] == list(SEL["accepted_merged_count"].values())
    assert all(2 <= sum(c) <= 3 for c in zip(*vec))
    assert [list(x) for x in vec] == [SEL["merge_vectors"][str(s)]
                                      for s in SEL["accepted_seeds"]]
    assert SEL["pairs"] == {k: list(p) for k, p in v2.BALANCE_PAIRS.items()}
    assert SEL["merged_in"] == [2, 3]


def test_selection_replays_from_the_record():
    """Uniform over the rule-meeting 5-subsets of the pool: the recorded pool
    and SELECT_SEED give back the accepted seeds after the recorded number of
    samples, and the pool was extended only while the rule was infeasible."""
    vectors = {int(k): tuple(x) for k, x in SEL["merge_vectors"].items()}
    assert v2.select(vectors) == (SEL["accepted_seeds"], SEL["samples_drawn"])
    blocks = [list(b) for b in v2.POOL_BLOCKS]
    used = next(i for i in range(len(blocks))
                if sorted(vectors) == sorted(sum(blocks[:i + 1], [])))
    for i in range(used):
        assert not v2.feasible({s: vectors[s] for s in sum(blocks[:i + 1], [])})
    assert v2.feasible(vectors)


def test_rule_helpers():
    one, zero = (1,) * 7, (0,) * 7
    assert v2.rule_ok([one, one, zero, zero, zero]) and v2.rule_ok([one] * 3 + [zero] * 2)
    assert not v2.rule_ok([one] * 4 + [zero]) and not v2.rule_ok([one] + [zero] * 4)
    assert v2.feasible({1: one, 2: one, 3: zero, 4: zero, 5: zero})
    assert not v2.feasible({1: one, 2: zero, 3: zero, 4: zero, 5: zero, 6: zero})


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
    assert rec["max_abs_share_mismatch_units"] == {"F0": 0, "F1": 0}


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
    assert set(pairs["status"]) == set(v2.VOCABS) | set(RAND_ARMS) | {"FLAV_F0", "FLAV_F1"}


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
    st = json.loads(v2.PAIRS.read_text())["status"]
    f0, f1 = st["FLAV_F0"], st["FLAV_F1"]
    assert "X_YY_bbqq|X_YY_ccqq" in f0["bvc_4prong"]["merged"]
    assert "X_YY_bbqq|X_YY_ccqq" in f1["bvc_4prong"]["split"]
    assert "X_bb|X_cc" in f0["bvc_resonant"]["merged"] and "X_bb|X_cc" in f1["bvc_resonant"]["merged"]


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
    for v in v2.VOCABS:
        a = ARMS[f"{v}_LOFO4P"]
        assert a["config"] == ARMS[v]["config"] and a["num_classes"] == ARMS[v]["num_classes"]
        assert a["extra_selection"] == v2.lofo_extra_selection(ROWS)
    assert [a["name"] for a in GRID["arms"] if a["extra_selection"]] == \
        [f"{v}_LOFO4P" for v in v2.VOCABS]


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
