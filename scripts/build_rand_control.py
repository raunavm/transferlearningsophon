#!/usr/bin/env python3
"""The D8 semantics-matched, learnability-preserving random control.

WHAT IT IS AND WHY IT IS THE NOVELTY LEG. The obvious objection to this whole
study is a tautology: "you merged label_X_bb and label_X_cc into one group, then
showed a probe separates b from c less well -- what did you expect?". The answer
is a vocabulary with the SAME number of groups and the SAME group-size profile
whose membership is RANDOM. If the random arm transfers as well as R16_Q1, then
what matters is how many groups there are and the thesis is wrong. If it
transfers worse, then WHICH distinctions are preserved is what matters, which is
the claim. Nothing else in the run matrix can settle that.

CONSTRUCTION -- the project's own, from scripts/build_hierarchy.py:814-847, not
a new one. Members are permuted WITHIN prong strata (res2p = native 0-14,
res34p = 15-160) and cut into blocks whose sizes are copied from the target
rung's resonant groups. So:

  preserved   K, the per-stratum group-size multiset, the 2-prong / 3-4-prong
              split, and the QCD block -- which is what "learnability-preserving"
              means. A partition drawn freely across all 188 would mix QCD with
              resonant and 2-prong with 4-prong, and would be unlearnable; then a
              transfer gap would measure "impossible task", not "wrong semantics",
              and the control would prove nothing.
  scrambled   which specific classes share a group. That is the semantics, and it
              is the only thing that differs from the target rung.

THE EXISTING ARTIFACTS ARE STALE. hierarchy/02_rung_rand42_d{1,2,3}_groups.csv
were built against the RETIRED R15 rung (scripts/build_lr_sweep.py:40) and have
15 resonant groups, so they match no arm in the current run matrix. They are not
reused.

TARGET RUNG IS A FLAG BECAUSE THE DOCUMENTS DISAGREE. docs/PRD_PLAN.md:40 says
K=17 (matched to R16_Q1); docs/RUN_MATRIX.md:27 says RAND42 at K=43 (matched to
R42_Q1). Default is R16_Q1: the coarse end is where the claimed effect is largest
and therefore where the tautology objection bites hardest. See DECISIONS_PENDING.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
OUT = REPO / "configs" / "labelmaps" / "rand_label_map.v1.csv"

# Native label blocks. 161 resonant + 27 QCD = 188 (docs/GROUND_TRUTH.md).
STRATA = {"res2p": range(0, 15), "res34p": range(15, 161)}
QCD_LO = 161
SEEDS = (42, 43, 44)


def hash_order(items, seed, tag):
    """Deterministic permutation, stable across Python versions and platforms.

    Copied from scripts/build_hierarchy.py:399 so the two agree exactly; a
    different shuffle here would make the control incomparable to the draws
    already recorded in hierarchy/.
    """
    def key(i):
        return hashlib.sha256(f"{tag}|seed={seed}|idx={i}".encode()).hexdigest()
    return sorted(items, key=key)


def read_rows():
    with MAP.open() as f:
        return list(csv.DictReader(f))


def target_profile(rows, rung):
    """(per-stratum group-size multiset, QCD group ids) of the target rung."""
    if rung not in rows[0]:
        raise SystemExit(f"FATAL: rung {rung} is not a column of {MAP}")
    members = {}
    for r in rows:
        members.setdefault(int(r[rung]), []).append(int(r["jet_label"]))
    sizes = {s: [] for s in STRATA}
    qcd_groups = set()
    for gid, mem in sorted(members.items()):
        if all(m >= QCD_LO for m in mem):
            qcd_groups.add(gid)
            continue
        if any(m >= QCD_LO for m in mem):
            raise SystemExit(
                f"FATAL: {rung} group {gid} mixes QCD and resonant natives; the "
                f"control's QCD block cannot be copied unambiguously")
        for s, rng in STRATA.items():
            if min(mem) in rng:
                sizes[s].append(len(mem))
                break
        else:
            raise SystemExit(f"FATAL: {rung} group {gid} falls in no stratum")
    return sizes, qcd_groups


def draw(rows, sizes, qcd_groups, rung, seed, d):
    """One random partition with the target's shape and scrambled membership."""
    assign = {}
    gid = 0
    for s, rng in STRATA.items():
        perm = hash_order(list(rng), seed, f"rand|draw{d}|{s}")
        pos = 0
        for size in sizes[s]:
            for m in perm[pos:pos + size]:
                assign[m] = gid
            pos += size
            gid += 1
        if pos != len(list(rng)):
            raise SystemExit(f"FATAL: draw {d} stratum {s}: consumed {pos} of "
                             f"{len(list(rng))}")
    # QCD keeps the target's own block, unchanged: it is not the axis under test
    qcd_map = {}
    for r in rows:
        lab = int(r["jet_label"])
        if lab >= QCD_LO:
            qcd_map.setdefault(int(r[rung]), len(qcd_map) + gid)
            assign[lab] = qcd_map[int(r[rung])]
    return assign



# ---------------------------------------------------------------------------
# SHARE MATCHING (--match share, the default).  See DECISIONS_PENDING item 24.
#
# WHY THE COUNT-MATCHED CONTROL ABOVE IS NOT SUFFICIENT.  `target_profile`
# copies len(mem) -- how many NATIVE CLASSES each group holds.  But the network
# never sees a native class; it sees a group label.  Class count is therefore
# invisible to the loss, while STREAM SHARE -- what fraction of the training
# jets carry the group label -- is exactly what sets the per-group imbalance the
# loss experiences.  Native classes are 222.2:1 imbalanced in share, so matching
# counts does not match jets:
#
#     vocabulary   K    per-group max/min share    exp(H)
#     R16_Q1       17          200.00 : 1           8.43
#     RAND_d1      17           23.78 : 1          12.70   <- count-matched
#
# A control that is measurably EASIER than the arm it controls biases the
# pre-registered falsification rule ("if RAND transfers as well as R16_Q1 the
# thesis is wrong") toward firing on an artefact.  So the matched quantity is
# the group's stream share, and the class count is allowed to float.
#
# WHAT THE LITERATURE ACTUALLY DOES, and why it does not settle this directly.
# The closest precedent is Huh, Agrawal and Efros, "What makes ImageNet good for
# transfer learning?" (arXiv:1608.08614).  Their random-vs-semantic control is
# the random / minimal split of section 5.6, and it matches CLASS COUNT:
# "Split A has 522 classes and split B has 478 classes (N.B.: for consistency,
# random splits A and B also had the same number of classes)" [V, sec 5.6].
# Their coarse label sets likewise vary "while keeping the total number of
# images constant" [V, sec 5.1].  So they hold both quantities fixed at once --
# but only because ImageNet is class-BALANCED (section 3: N images for every
# class), so matching class count IS matching image count there.  The two are
# the same number and the paper never has to choose.
#
# JetClass-II forces the choice: natives are 222.2:1 imbalanced in stream share
# (docs/GROUND_TRUTH.md), so the two quantities come apart, and only one of them
# is the one the loss sees.  Matching share is the reading of Huh et al. that
# survives the loss of balance.
#
# WHAT REMAINS UNSCRAMBLED IS FORCED, NOT A WEAK SOLVER.  Every res34p share
# value is divisible by 11 except 172800, which is 1 mod 11, and every target
# block sum is 0 mod 11.  So each block must take a MULTIPLE OF 11 of the 22
# classes carrying that value, and only two blocks are large enough to hold 11:
# the 22 can split 22, or 11+11, and nothing else.  The solver finds 11+11, so
# 110 surviving pairs in that one target group are unavoidable.  res2p is
# rigid outright -- exactly one value-composition satisfies its four share
# targets, so its 4 groups are share-determined and cannot be scrambled at all.
# Both facts are pinned by tests rather than asserted here.
#
# EXACTNESS.  Shares are rational: share_i = w_g / (S * n_g) with w_g the
# reweighting-category weight, n_g its class count and S = 6927/4000 the sum of
# the 30 weights.  They are converted to integers over a common denominator, so
# "matched" here means bit-exact, not approximately equal.
# ---------------------------------------------------------------------------

MASTER = REPO / "hierarchy" / "01_class_master.csv"
TRIALS = 32            # deterministic restarts; best-scrambling one is kept
COMP_CAP = 3000        # compositions enumerated per block
BRANCH_CAP = 80        # compositions tried per block before backtracking
NODE_BUDGET = 2_000_000  # search nodes per compositions() call


def exact_share_units():
    """Per-native stream share as EXACT integers on a common denominator."""
    from fractions import Fraction
    from math import gcd

    with MASTER.open() as f:
        master = list(csv.DictReader(f))
    cats = {}
    for r in master:
        cats.setdefault(r["reweight_group_name"],
                        (Fraction(r["reweight_group_weight"]),
                         int(r["reweight_group_n_classes"])))
    if len(cats) != 30:
        raise SystemExit(f"FATAL: {len(cats)} reweighting categories, expected 30")
    S = sum(w for w, _ in cats.values())
    share = {}
    for r in master:
        w, n = cats[r["reweight_group_name"]]
        share[int(r["jet_label"])] = w / (S * n)
    if sum(share.values()) != 1:
        raise SystemExit("FATAL: exact shares do not sum to 1")
    den = 1
    for v in share.values():
        den = den * v.denominator // gcd(den, v.denominator)
    units = {k: int(v * den) for k, v in share.items()}
    g = 0
    for u in units.values():
        g = gcd(g, u)
    return {k: v // g for k, v in units.items()}


def hshuf(items, tag):
    """hash_order with a tag that already carries the seed."""
    return sorted(items,
                  key=lambda i: hashlib.sha256(f"{tag}|{i!r}".encode()).hexdigest())


def hperm(items, tag):
    """Deterministic permutation in O(n), for lists too big to sort by digest.

    hshuf costs one sha256 per element; the composition lists below reach
    thousands of entries per block and are rebuilt for every restart, which put
    ~10^7 digests on the critical path. An affine map i -> (a*i + b) mod n with
    a coprime to n is a permutation, needs ONE digest for the whole list, and is
    just as reproducible. It is used only to pick a spread of candidates, never
    where the project's hash_order convention is load-bearing.
    """
    n = len(items)
    if n < 2:
        return list(items)
    h = int(hashlib.sha256(tag.encode()).hexdigest(), 16)
    a, b = 1 + (h % (n - 1)), (h >> 64) % n
    while math.gcd(a, n) != 1:
        a += 1
    return [items[(a * i + b) % n] for i in range(n)]


def compositions(avail, values, target, cap=COMP_CAP):
    """Every count-vector over `values` drawn from `avail` summing to target.

    `cap` and NODE_BUDGET bound the SEARCH, never the answer. Truncating can
    only make the solver miss a partition, never accept a wrong one: main()
    re-derives the finished draw's share profile and refuses it unless it equals
    the target's exactly, and share_draw raises if no restart succeeds. Without
    the node bound a degenerate instance -- one block asked to absorb a whole
    stratum -- explores a combinatorially large tree to find its single trivial
    solution, which reads as a hang rather than a failure.

    `values` is supplied in a per-block hash-shuffled order by the caller.  With
    a fixed ascending order the first feasible solution always takes the largest
    available multiple of whichever value comes first, so the one big block
    swallows every copy of the dominant share value and the target group holding
    them survives almost intact -- 110 of 262 residual pairs came from exactly
    that.  Reordering the values removes the bias for one hash call per block.
    """
    out = []
    budget = [NODE_BUDGET]

    def rec(i, rem, cur):
        budget[0] -= 1
        if budget[0] < 0:          # search bound, NOT a correctness bound
            return
        if rem == 0:
            out.append(tuple(cur))
            return
        if i == len(values) or len(out) >= cap:
            return
        v = values[i]
        hi = min(avail[v], rem // v)
        for c in range(hi, -1, -1):
            cur.append(c)
            rec(i + 1, rem - c * v, cur)
            cur.pop()
            if len(out) >= cap:
                return

    rec(0, target, [])
    return out


def solve_shape(counts, targets, tag):
    """Assign value-counts to blocks so each block's share sum is EXACT.

    counts  {share_value: how many classes in this stratum carry it}
    targets [block share sums, in the same integer units]
    Returns {block index: {share_value: count}} or None.
    """
    values = sorted(counts)
    avail = dict(counts)
    order = sorted(range(len(targets)), key=lambda b: targets[b])
    sol = {}

    def rec(k):
        if k == len(order):
            return True
        b = order[k]
        vals = hshuf(values, f"{tag}|order{b}")
        comps = [c for c in compositions(avail, vals, targets[b]) if sum(c)]
        for c in hperm(comps, f"{tag}|block{b}")[:BRANCH_CAP]:
            for v, n in zip(vals, c):
                avail[v] -= n
            sol[b] = {v: n for v, n in zip(vals, c) if n}
            if rec(k + 1):
                return True
            del sol[b]
            for v, n in zip(vals, c):
                avail[v] += n
        return False

    return dict(sol) if rec(0) else None


def _agreement(m):
    """Unordered class pairs that are together in BOTH partitions."""
    return sum(x * (x - 1) // 2 for row in m.values() for x in row.values())


def place_classes(shape, by_value, tgt_of, val_of, tag):
    """Turn a value-count shape into a concrete class->block map.

    Classes carrying the same share value are interchangeable for the sum
    constraint, so this is free to pick the placement that shares the FEWEST
    same-group pairs with the target partition -- i.e. maximum scrambling.
    """
    blocks = sorted(shape)
    m = {b: collections.Counter() for b in blocks}     # block -> target group
    out = {}
    need = {b: dict(shape[b]) for b in blocks}
    for v in hshuf(sorted(by_value), f"{tag}|values"):
        for lab in hshuf(by_value[v], f"{tag}|v{v}"):
            t = tgt_of[lab]
            cand = [b for b in blocks if need[b].get(v, 0) > 0]
            b = min(cand, key=lambda b: (m[b][t],
                                         hashlib.sha256(
                                             f"{tag}|{lab}|{b}".encode()).hexdigest()))
            out[lab] = b
            m[b][t] += 1
            need[b][v] -= 1

    # Local search: swapping two SAME-VALUE classes between blocks leaves every
    # block sum exact, so the whole neighbourhood is feasible by construction
    # and no repair step is needed.  Only four cells of `m` move, so the
    # objective delta is computed in closed form rather than by rescoring:
    # C(x) - C(x-1) = x-1 and C(x+1) - C(x) = x.
    improved = True
    while improved:
        improved = False
        for v, group in by_value.items():
            for a_i, i in enumerate(group):
                for j in group[a_i + 1:]:
                    b1, b2 = out[i], out[j]
                    t1, t2 = tgt_of[i], tgt_of[j]
                    if b1 == b2 or t1 == t2:
                        continue
                    delta = ((m[b1][t2] - (m[b1][t1] - 1))
                             + (m[b2][t1] - (m[b2][t2] - 1)))
                    if delta < 0:
                        m[b1][t1] -= 1; m[b1][t2] += 1
                        m[b2][t2] -= 1; m[b2][t1] += 1
                        out[i], out[j] = b2, b1
                        improved = True
    return out, _agreement(m)


def share_draw(rows, units, rung, seed, d):
    """One share-matched partition with maximally scrambled membership."""
    tgt_of = {int(r["jet_label"]): int(r[rung]) for r in rows}

    assign = {}
    gid = 0
    stats = {}
    for s, rng in STRATA.items():
        labs = [m for m in rng]
        counts = collections.Counter(units[m] for m in labs)
        groups = collections.defaultdict(list)
        for m in labs:
            groups[tgt_of[m]].append(m)
        targets = [sum(units[m] for m in mem) for _, mem in sorted(groups.items())]
        by_value = collections.defaultdict(list)
        for m in labs:
            by_value[units[m]].append(m)

        if len(targets) == 1:
            # Nothing to partition: the single block IS the stratum. Searching
            # for it is a large tree with one trivial answer at the bottom.
            for m in labs:
                assign[m] = gid
            gid += 1
            stats[s] = (sum(len(v) * (len(v) - 1) // 2 for v in groups.values()),
                        sum(len(v) * (len(v) - 1) // 2 for v in groups.values()), 1)
            continue

        best = None
        for t in range(TRIALS):
            shape = solve_shape(counts, targets, f"rand|seed={seed}|d{d}|{s}|t{t}")
            if shape is None:
                continue
            place, agree = place_classes(shape, by_value, tgt_of, units,
                                         f"rand|seed={seed}|d{d}|{s}|t{t}")
            if best is None or agree < best[0]:
                best = (agree, place, shape)
        if best is None:
            raise SystemExit(f"FATAL: no share-matched partition of {s} exists")

        agree, place, shape = best
        base = {b: gid + i for i, b in enumerate(sorted(shape))}
        for m, b in place.items():
            assign[m] = base[b]
        gid += len(shape)
        tot = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())
        stats[s] = (agree, tot, len(shape))

    # QCD keeps the target's own block: it is not the axis under test.
    qcd_map = {}
    for r in rows:
        lab = int(r["jet_label"])
        if lab >= QCD_LO:
            qcd_map.setdefault(tgt_of[lab], len(qcd_map) + gid)
            assign[lab] = qcd_map[tgt_of[lab]]

    for s, (agree, tot, nb) in stats.items():
        frac = agree / tot if tot else 0.0
        print(f"    {s}: {nb} groups, share-exact, "
              f"{agree}/{tot} target pairs survive ({frac:.1%})")
    return assign

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="R16_Q1",
                    help="rung whose SHAPE the control copies (default R16_Q1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--match", choices=("share", "count"), default="share",
                    help="quantity held fixed while membership is scrambled. "
                         "share (default) = per-group STREAM SHARE, what the "
                         "loss actually sees; count = per-group NATIVE-CLASS "
                         "count, which the network never observes -- kept only "
                         "so the superseded control stays reproducible "
                         "(DECISIONS_PENDING item 24)")
    a = ap.parse_args(argv)

    rows = read_rows()
    sizes, qcd_groups = target_profile(rows, a.target)
    k_target = len({int(r[a.target]) for r in rows})
    n_res = sum(sum(v) for v in sizes.values())
    if n_res != 161:
        raise SystemExit(f"FATAL: size template covers {n_res} resonant natives, not 161")

    # A target group that spans strata cannot be controlled: the control
    # preserves strata by construction, so it could not reproduce that group's
    # share under any membership. Count mode caught this only indirectly, as a
    # "consumed N of M" arithmetic failure inside draw(); share mode would
    # instead have solved a 1-block partition of the whole stratum, which is an
    # unbounded search for a partition that says nothing. Refuse it up front.
    def stratum(lab):
        if lab >= QCD_LO:
            return "qcd"
        for st, rng in STRATA.items():
            if lab in rng:
                return st
        raise SystemExit(f"FATAL: native {lab} falls in no stratum")

    spans = {}
    for r in rows:
        spans.setdefault(int(r[a.target]), set()).add(stratum(int(r["jet_label"])))
    bad = {g: s for g, s in spans.items() if len(s) > 1}
    if bad:
        raise SystemExit(
            f"FATAL: {a.target} group(s) {sorted(bad)} span strata "
            f"{[sorted(v) for v in bad.values()]}; a stratum-preserving control "
            f"cannot reproduce them")

    units = exact_share_units() if a.match == "share" else None

    tgt = {int(r["jet_label"]): int(r[a.target]) for r in rows}
    draws = {}
    for d, seed in enumerate(a.seeds, start=1):
        if a.match == "share":
            print(f"  draw {d} (seed {seed}):")
            assign = share_draw(rows, units, a.target, seed, d)
        else:
            assign = draw(rows, sizes, qcd_groups, a.target, seed, d)
        k = len(set(assign.values()))
        if k != k_target:
            raise SystemExit(f"FATAL: draw {d} has {k} groups, target {a.target} has {k_target}")
        if len(assign) != len(rows):
            raise SystemExit(f"FATAL: draw {d} assigns {len(assign)} of {len(rows)} natives")
        # A draw that reproduces the target is not a control. Compare as
        # PARTITIONS (group ids are arbitrary), not as label vectors.
        same = {frozenset(m for m, g in assign.items() if g == gg)
                for gg in set(assign.values())}
        tsame = {frozenset(m for m, g in tgt.items() if g == gg)
                 for gg in set(tgt.values())}
        if same == tsame:
            raise SystemExit(f"FATAL: draw {d} (seed {a.seeds[d-1]}) reproduces "
                             f"{a.target} exactly; it is not a control")
        draws[d] = assign
        shared = len(same & tsame)
        print(f"    K={k}, {shared} of {len(tsame)} groups identical to {a.target}")
        if a.match == "share":
            got = sorted(sum(units[m] for m in g) for g in same)
            want = sorted(sum(units[m] for m in g) for g in tsame)
            if got != want:
                raise SystemExit(f"FATAL: draw {d} share profile != {a.target}")

    out = pathlib.Path(a.out)
    # `<arm>_name` columns are emitted because scripts/build_arm_configs.py
    # requires them: the arm YAML's comment table names each GROUP, and using a
    # member class's name there once made R16_Q1 group 0 read "label_X_bb".
    # A random group has no physical name, so it is named by its index and
    # stratum -- which is honest, and says so.
    cols = ["jet_label", "class_name", a.target]
    for d in draws:
        cols += [f"RAND_d{d}", f"RAND_d{d}_name"]
    strat_of = {}
    for st, rng in STRATA.items():
        for m in rng:
            strat_of[m] = st
    names = {}
    for d, assign in draws.items():
        seen = {}
        for lab in sorted(assign):
            g = assign[lab]
            if g not in seen:
                st = strat_of.get(lab, "qcd")
                seen[g] = f"RAND{d}_{st}_{g:02d}"
        names[d] = seen
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            lab = int(r["jet_label"])
            row = [lab, r["class_name"], r[a.target]]
            for d in draws:
                row += [draws[d][lab], names[d][draws[d][lab]]]
            w.writerow(row)
    try:
        shown = out.relative_to(REPO)
    except ValueError:          # --out outside the repo, e.g. a test tmpdir
        shown = out
    print(f"\nwrote {shown}  (target {a.target}, K={k_target}, "
          f"{len(draws)} draws, seeds {a.seeds})")
    if a.match == "share":
        print("  matched quantity: per-group STREAM SHARE (exact); native-class "
              "counts float")
    else:
        print(f"  per-stratum size template: "
              + "; ".join(f"{s}={sorted(v, reverse=True)}" for s, v in sizes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
