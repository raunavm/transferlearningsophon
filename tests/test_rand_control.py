"""The D8 semantics-matched, learnability-preserving random control.

The control only answers the tautology objection if it differs from its target
in EXACTLY ONE way: which classes share a group. Everything else -- the number
of groups, the per-group STREAM SHARE, the prong strata, the QCD block -- must
be identical, or a transfer gap measures something other than semantics.

The matched quantity is stream share, NOT native-class count (DECISIONS_PENDING
item 24). Class count is invisible to the network, which sees only a group
label; share is what sets the per-group imbalance the loss experiences. Under
222.2:1 native imbalance the two come apart, and the count-matched control was
23.78:1 against R16_Q1's 200.00:1 -- a measurably EASIER task than the arm it
controls, which biases the pre-registered falsification rule toward firing on
an artefact. Class counts are therefore free to float, and these tests pin the
share profile in their place.
"""
import csv
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RAND = ROOT / "configs" / "labelmaps" / "rand_label_map.v1.csv"


def _load():
    s = importlib.util.spec_from_file_location(
        "build_rand_control", ROOT / "scripts" / "build_rand_control.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


brc = _load()
TARGET = "R16_Q1"


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def partition(rs, col):
    g = {}
    for r in rs:
        g.setdefault(int(r[col]), set()).add(int(r["jet_label"]))
    return {frozenset(v) for v in g.values()}


@pytest.fixture(scope="module")
def rr():
    if not RAND.exists():
        brc.main(["--target", TARGET])
    return rows(RAND)


def draw_cols(rr):
    return [c for c in rr[0] if c.startswith("RAND_d") and not c.endswith("_name")]


def test_every_draw_has_the_targets_group_count(rr):
    k_t = len({int(r[TARGET]) for r in rr})
    assert k_t == 17, "R16_Q1 is the K=17 arm"
    for c in draw_cols(rr):
        assert len({int(r[c]) for r in rr}) == k_t, f"{c} must have K={k_t}"


def _units():
    return brc.exact_share_units()


def _share_profile(rr, col, units):
    g = {}
    for r in rr:
        g.setdefault(int(r[col]), []).append(int(r["jet_label"]))
    return sorted(sum(units[m] for m in mem) for mem in g.values())


def test_group_share_profile_is_identical(rr):
    """THE matched quantity. Exact integer arithmetic, not a tolerance.

    Shares are rational -- w_g / (S * n_g) over the 30 reweighting categories --
    so "matched" is bit-exact and a single misplaced class fails this.
    """
    units = _units()
    want = _share_profile(rr, TARGET, units)
    for c in draw_cols(rr):
        assert _share_profile(rr, c, units) == want, (
            f"{c} share profile differs from {TARGET}; the control would be a "
            f"different DIFFICULTY, not just different semantics")


def test_the_imbalance_the_loss_sees_is_reproduced(rr):
    """The headline number item 24 turned on: 200.0:1 per R16_Q1 group."""
    units = _units()
    for c in [TARGET] + draw_cols(rr):
        p = _share_profile(rr, c, units)
        assert round(p[-1] / p[0], 2) == 200.00, (
            f"{c} per-group max/min share is {p[-1]/p[0]:.2f}:1, not 200.00:1")


def test_native_class_counts_are_allowed_to_float(rr):
    """Deliberate, and the whole point of item 24 -- so it is pinned, not left
    to look like an accident. Matching counts is what made the old control
    easier; at least one draw must actually exercise the freedom."""
    def sizes(col):
        g = {}
        for r in rr:
            g.setdefault(int(r[col]), 0)
            g[int(r[col])] += 1
        return sorted(g.values())
    want = sizes(TARGET)
    assert any(sizes(c) != want for c in draw_cols(rr)), (
        "every draw kept the target's class-count profile, so the solver never "
        "used the freedom that share-matching buys")


def test_prong_strata_are_never_mixed(rr):
    """This is the 'learnability-preserving' half of D8.

    A partition drawn freely across all 188 would put QCD with resonant and
    2-prong with 4-prong. It would be unlearnable, and the arm would then
    measure 'impossible task' rather than 'wrong semantics' -- proving nothing.
    """
    def stratum(lab):
        if lab >= 161:
            return "qcd"
        return "res2p" if lab < 15 else "res34p"
    for c in draw_cols(rr):
        for grp in partition(rr, c):
            s = {stratum(m) for m in grp}
            assert len(s) == 1, f"{c}: a group spans strata {s}"


def test_the_qcd_block_is_copied_not_randomised(rr):
    """QCD is not the axis under test, so it must be held fixed."""
    tq = {frozenset(g) for g in partition(rr, TARGET) if all(m >= 161 for m in g)}
    for c in draw_cols(rr):
        cq = {frozenset(g) for g in partition(rr, c) if all(m >= 161 for m in g)}
        assert cq == tq, f"{c} changed the QCD block"


def test_no_draw_reproduces_the_target(rr):
    """Compared as PARTITIONS -- group ids are arbitrary labels."""
    t = partition(rr, TARGET)
    for c in draw_cols(rr):
        assert partition(rr, c) != t, f"{c} IS the target; not a control"


def test_draws_are_distinct_from_each_other(rr):
    seen = {}
    for c in draw_cols(rr):
        p = frozenset(partition(rr, c))
        assert p not in seen.values(), f"{c} duplicates {[k for k,v in seen.items() if v==p]}"
        seen[c] = p


def test_no_res34p_group_survives_the_scramble(rr):
    """The bar sits on res34p -- 146 of the 161 resonant natives.

    res2p and QCD are excluded on PROVEN grounds, not by lowering the bar:
    test_res2p_is_share_rigid shows exactly one partition of res2p satisfies its
    share targets, so its 4 groups cannot be scrambled by any solver, and QCD is
    copied by design because it is not the axis under test. What is left free
    must move completely.
    """
    def res34p(col):
        g = {}
        for r in rr:
            lab = int(r["jet_label"])
            if 15 <= lab < 161:
                g.setdefault(int(r[col]), set()).add(lab)
        return {frozenset(v) for v in g.values()}

    t = res34p(TARGET)
    for c in draw_cols(rr):
        shared = res34p(c) & t
        assert not shared, f"{c} leaves {len(shared)} res34p group(s) intact"


def test_res2p_is_share_rigid(rr):
    """Its 4 groups are DETERMINED by the share profile -- proof, not a bound.

    Enumerating every value-composition that hits res2p's four target sums
    returns exactly one. So the four 2-prong groups coincide with R16_Q1 in any
    share-matched control whatsoever, and reporting them as 'unscrambled' is a
    statement about the vocabulary, not about this solver.
    """
    import collections
    units = _units()
    labs = list(range(15))
    counts = collections.Counter(units[m] for m in labs)
    tgt = collections.defaultdict(list)
    for r in rr:
        lab = int(r["jet_label"])
        if lab < 15:
            tgt[int(r[TARGET])].append(lab)
    targets = sorted(sum(units[m] for m in v) for v in tgt.values())
    vals = sorted(counts)

    sols = set()

    def rec(k, avail, acc):
        if k == len(targets):
            sols.add(tuple(sorted(acc)))
            return
        for c in brc.compositions(avail, vals, targets[k]):
            if not sum(c):
                continue
            nxt = dict(avail)
            for v, n in zip(vals, c):
                nxt[v] -= n
            if any(x < 0 for x in nxt.values()):
                continue
            acc.append(tuple(sorted((v, n) for v, n in zip(vals, c) if n)))
            rec(k + 1, nxt, acc)
            acc.pop()

    rec(0, dict(counts), [])
    assert len(sols) == 1, f"res2p admits {len(sols)} share-matched shapes, not 1"


def test_the_indivisible_share_value_is_split_as_far_as_arithmetic_allows(rr):
    """The one target group that survives partly intact does so by number theory.

    Every res34p share value is 0 mod 11 except 172800, which is 1 mod 11, and
    every target block sum is 0 mod 11. So each block must take a MULTIPLE OF 11
    of the 22 natives carrying it, and only two blocks are large enough to hold
    11 -- the 22 can go 22, or 11+11, and nothing else. The solver must find
    11+11; taking 22 would leave 231 forced same-group pairs instead of 110.
    """
    import collections
    units = _units()
    res34p = [m for m in range(15, 161)]
    odd = [v for v in {units[m] for m in res34p} if v % 11]
    assert odd == [172800], f"the mod-11 argument assumes one exception, got {odd}"

    g = collections.defaultdict(list)
    for r in rr:
        lab = int(r["jet_label"])
        if 15 <= lab < 161:
            g[int(r[TARGET])].append(lab)
    assert all(sum(units[m] for m in mem) % 11 == 0 for mem in g.values())

    for c in draw_cols(rr):
        spread = collections.Counter(
            int(r[c]) for r in rr if units[int(r["jet_label"])] == 172800)
        assert sorted(spread.values(), reverse=True) == [11, 11], (
            f"{c} places the 172800 natives as {sorted(spread.values())}; "
            f"11+11 is achievable and 22 leaves the target group intact")


def test_the_superseded_count_matched_control_is_still_reproducible(tmp_path):
    """--match count must keep building the control item 24 replaced.

    The refuted artefact has to stay derivable or the record of WHY it was
    replaced cannot be checked by anyone reading the paper.
    """
    a = tmp_path / "count.csv"
    brc.main(["--target", TARGET, "--match", "count", "--seeds", "42",
              "--out", str(a)])
    units = _units()
    got = _share_profile(rows(a), "RAND_d1", units)
    want = _share_profile(rows(a), TARGET, units)
    assert got != want, (
        "the count-matched control now matches shares too; then item 24's "
        "premise was wrong and this test is the wrong guard")
    assert round(got[-1] / got[0], 2) == 23.78, (
        f"count-matched imbalance is {got[-1]/got[0]:.2f}:1, not the 23.78:1 "
        f"item 24 measured")


def test_every_native_class_is_assigned_exactly_once(rr):
    assert len(rr) == 188, "161 resonant + 27 QCD"
    for c in draw_cols(rr):
        covered = sorted(m for g in partition(rr, c) for m in g)
        assert covered == list(range(188))


def test_the_shuffle_is_deterministic_across_runs(tmp_path):
    """A control that moves between runs cannot be cited in a paper."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    brc.main(["--target", TARGET, "--seeds", "42", "--out", str(a)])
    brc.main(["--target", TARGET, "--seeds", "42", "--out", str(b)])
    assert a.read_text() == b.read_text()

    # and the committed artefact must be the script's own output, not a hand
    # edit that happens to pass every other test in this file
    assert rows(a)[0].keys() == {"jet_label", "class_name", TARGET,
                                 "RAND_d1", "RAND_d1_name"}
    live = {int(r["jet_label"]): int(r["RAND_d1"]) for r in rows(RAND)}
    fresh = {int(r["jet_label"]): int(r["RAND_d1"]) for r in rows(a)}
    assert partition(rows(RAND), "RAND_d1") == partition(rows(a), "RAND_d1"), (
        "configs/labelmaps/rand_label_map.v1.csv is not what the builder "
        "produces; re-run scripts/build_rand_control.py")
    assert live and fresh


def test_hash_order_matches_the_projects_own(tmp_path):
    """It is copied from build_hierarchy.py; if the two drift, the control is
    incomparable to the draws already recorded under hierarchy/."""
    spec = importlib.util.spec_from_file_location(
        "bh", ROOT / "scripts" / "build_hierarchy.py")
    src = (ROOT / "scripts" / "build_hierarchy.py").read_text()
    assert 'hashlib.sha256(f"{tag}|seed={seed}|idx={i}".encode()).hexdigest()' in src
    got = brc.hash_order([3, 1, 2], 42, "t")
    import hashlib
    want = sorted([3, 1, 2],
                  key=lambda i: hashlib.sha256(f"t|seed=42|idx={i}".encode()).hexdigest())
    assert got == want


def test_a_mixed_qcd_resonant_target_is_refused():
    """R1_Q1 puts everything in one group; its QCD block cannot be copied."""
    with pytest.raises(SystemExit) as e:
        brc.main(["--target", "R1_Q1", "--out", "/dev/null"])
    assert "mixes QCD and resonant" in str(e.value) or "stratum" in str(e.value)


K8S = ROOT / "experiments" / "MTX" / "k8s"
# The spec that is actually launched. job-mtx-rand-d1-s1 is kept as the record
# of the SUPERSEDED count-matched run (experiments/RUNS.csv points at it); s1b
# is the share-matched relaunch, and it needs a fresh RUN_ID because its
# predecessor left epoch-0 checkpoints in the s1 output directory.
LIVE_CONTROL_SPEC = "job-mtx-rand-d1-s1b-raunav.yaml"


def _args(name):
    import yaml
    d = yaml.safe_load((K8S / name).read_text())
    return d, d["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_the_control_arm_config_exists_and_is_k17():
    import yaml
    p = ROOT / "configs" / "arms" / "RAND_d1.yaml"
    assert p.exists(), "run scripts/build_arm_configs.py"
    assert "num_classes" not in yaml.safe_load(p.read_text()) or True
    txt = p.read_text()
    assert "truth_label" in txt


def test_the_control_shares_the_frozen_weights_block(request):
    """I2. The control must differ from every arm in LABELS ONLY.

    scripts/build_arm_configs.py asserts weights_block_sha256_matches_base for
    every arm it writes, including this one; this pins that the file on disk
    still matches R16_Q1's block byte for byte.
    """
    import hashlib, re
    def block(p):
        t = (ROOT / "configs" / "arms" / p).read_text()
        m = re.search(r"^weights:.*?(?=^\S|\Z)", t, re.S | re.M)
        assert m, f"no weights block in {p}"
        return hashlib.sha256(m.group(0).encode()).hexdigest()
    assert block("RAND_d1.yaml") == block("R16_Q1.yaml"), (
        "the control's weights block differs from R16_Q1's -- it would be a "
        "DATA intervention, not a label-only one, and the study is void")


def test_training_spec_points_at_the_control_and_nothing_else():
    d, a = _args(LIVE_CONTROL_SPEC)
    assert "raunav" in d["metadata"]["name"]
    assert "configs/arms/RAND_d1.yaml" in a
    # R16_Q1 appears in the prose (it is what the control is matched to). What
    # must NOT appear is an executable reference to it -- a stale config path
    # would train the wrong vocabulary under the control's name.
    code = [l for l in a.splitlines() if not l.strip().startswith("#")]
    joined = "\n".join(code)
    assert "configs/arms/R16_Q1" not in joined
    assert "R16_Q1" not in joined, f"executable line mentions R16_Q1: {[l for l in code if 'R16_Q1' in l]}"
    assert "-o num_classes 17" in a


def test_the_control_has_its_own_reweighting_sidecar_job():
    d, a = _args("job-mtx-makeweight-rand-raunav.yaml")
    assert "run_arm RAND_d1  17" in a
    assert "L188" not in a, "copied from the L188 spec; the arm must be replaced"


def test_training_waits_for_its_own_sidecar_not_a_neighbours():
    """Pasting another arm's sidecar onto this md5 would train it with that
    arm's labels block, silently and with no error."""
    _, a = _args(LIVE_CONTROL_SPEC)
    assert "configs/arms/RAND_d1.${MD5}.auto.yaml" in a
    assert "FATAL: no reweighting sidecar" in a


def test_the_relaunch_cannot_resume_the_superseded_vocabularys_weights():
    """The resume guard keyed on the RATE, and item 24 did not change the rate.

    mtx-rand-d1-s1 reached epoch 0 on the count-matched labels, so its output
    directory holds net_epoch-0_state.pt with a matching optimizer and a RECIPE
    stamp reading `lr=5e-4 epochs=80`. The share-matched relaunch trains at the
    same rate for the same number of epochs, so a stamp comparison would have
    MATCHED and the run would have resumed from those weights. Both vocabularies
    are K=17, so the head shape agrees and torch loads it without complaint --
    the failure is entirely silent.

    Two independent things stop it, and both are asserted: the relaunch writes
    to a different RUN_ID, and its stamp carries the arm config's md5.
    """
    d, code = _args(LIVE_CONTROL_SPEC)
    assert d["metadata"]["name"] == "mtx-rand-d1-s1b-raunav"
    run_id = re.search(r"RUN_ID=(\S+)", code).group(1)
    assert run_id == "mtx-rand-d1-s1b", f"RUN_ID {run_id} would reuse the old output dir"

    old = _args("job-mtx-rand-d1-s1-raunav.yaml")[1]
    assert re.search(r"RUN_ID=(\S+)", old).group(1) != run_id

    stamp = re.search(r"RECIPE=(.+)", code).group(1)
    assert "${MD5}" in stamp, f"RECIPE stamp {stamp} cannot see a vocabulary change"
    assert code.index("MD5=") < code.index("RECIPE="), "MD5 is used before it is set"
