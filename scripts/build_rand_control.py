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
import csv
import hashlib
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="R16_Q1",
                    help="rung whose SHAPE the control copies (default R16_Q1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    rows = read_rows()
    sizes, qcd_groups = target_profile(rows, a.target)
    k_target = len({int(r[a.target]) for r in rows})
    n_res = sum(sum(v) for v in sizes.values())
    if n_res != 161:
        raise SystemExit(f"FATAL: size template covers {n_res} resonant natives, not 161")

    tgt = {int(r["jet_label"]): int(r[a.target]) for r in rows}
    draws = {}
    for d, seed in enumerate(a.seeds, start=1):
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
        print(f"  draw {d} (seed {a.seeds[d-1]}): K={k}, "
              f"{shared} of {len(tsame)} groups identical to {a.target}")

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
    print(f"  per-stratum size template: "
          + "; ".join(f"{s}={sorted(v, reverse=True)}" for s, v in sizes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
