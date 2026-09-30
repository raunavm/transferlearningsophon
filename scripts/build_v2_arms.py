#!/usr/bin/env python3
"""Arm definitions for the v2 rerun grid (audit 2026-09-29: B6, B8, must-fix 10
and 11, Strand E objections 3 and 4). Configs and label maps only; no training
code.

WHAT IT WRITES
  configs/arms/R63_Q1.yaml, R29_Q1.yaml   the 64- and 30-class tree levels, built
                                          by build_arm_configs.build_one exactly
                                          like the four existing arms
  configs/arms/v2/RAND2_p{1..5}.yaml      five share-matched random partitions of
                                          all 161 resonant classes (map written by
                                          build_rand_control.py --pool resonant),
                                          drawn under a balance rule recorded in
                                          configs/labelmaps/rand_v2_selection.json
  configs/labelmaps/flavour_pair_map.v2.csv, flavour_pair.v2.json
  configs/arms/v2/FLAV_F0.yaml, FLAV_F1.yaml   the decisive pair, see below
  configs/arms/v2/R16_Q1_MASS_LM.yaml     17 classes + mass output at the
                                          loss-share-matched lambda
  configs/arms/v2/mass_lambda.v2.json     that lambda and every input to it
  configs/labelmaps/probe_pairs.v2.json   which probe pairs each vocabulary
                                          merges or splits
  configs/arms/v2_grid.json               the registry of every v2 arm

Every config is the base config with a different `labels:` block and nothing
else; the `weights:` block is copied byte for byte (I2) and checked here.

THE DECISIVE PAIR (F0, F1). A flavour ORBIT is the set of resonant classes that
are the same decay up to exchanging b, c and light (s, u/d) quarks: X->bb, X->cc,
..., X->sq are one orbit; X->YY->bbqq and X->YY->ccqq are in another. There are 43.
  F0  16 resonant groups + QCD, each group a union of whole orbits (so no group
      boundary depends on quark flavour), with every group's stream share EXACTLY
      equal to the matching 17-class group's, and membership otherwise scrambled
      against the 17-class vocabulary.
  F1  F0 with one orbit cut in two: the 11 b-containing four-prong hadronic
      classes (X->YY->bbqq, bbcc, bbbb, ...) move to another group B, and whole
      orbits of exactly equal share move from B back. One boundary now separates
      b from c (bbqq from ccqq); every other boundary stays flavour-blind, and the
      shares stay exact.
  Why that orbit. Every orbit's share is a multiple of 297 (= 27 x 11), so an
  exact swap needs a subset of equal divisibility. The two-prong orbit X->QQ
  admits no proper subset (its classes carry 3^1 only), and a four-prong subset
  must hold 11 or 22 of its 22 classes; the b-containing ones are exactly 11.

THE LAMBDA-MATCHED MASS ARM. x = lambda * L_reg / L_cls, from each v1 mass run's
per-epoch training averages in train.log (hybrid_mass.py logs AvgLoss = L_cls
and AvgLossReg = L_reg). lambda_m = 5 * x_162 / x_17 gives the 17-class model
the regression-to-classification loss ratio the 162-class model had at lambda
= 5. Matching x is the same as matching the share of the total, lambda L_reg /
(L_cls + lambda L_reg), because the share is x / (1 + x). It is a first-order
match: L_reg and L_cls are taken at their lambda = 5 values.

LEAVE-ONE-FAMILY-OUT is NOT a new config file, on purpose. Writing the exclusion
into `selection:` breaks weaver: its reweighting pass (WeightMaker, run by the
make_weight job on the config) applies `selection:` first, the three reweighting
categories of the family are then empty, and make_weights stops at
np.min(<empty>) (weaver 0.4.17 preprocess.py; pinned by a test). weaver's own
`--extra-selection` is applied AFTER the reweighting histograms are made
(dataset.py:361), so the LOFO arm is the parent config, the parent's reweighting
sidecar, and `--extra-selection` = extra_selection in the registry. Every
surviving jet then has exactly the parent's sampling weight.

Run:
  python3 scripts/build_v2_arms.py [--check-only] [--pairs-out PATH]
  python3 scripts/build_v2_arms.py --mass-logs DIR     # re-parse the log dumps
  python3 scripts/build_v2_arms.py --select-rand 4     # redraw pool, reselect
  python3 scripts/build_v2_arms.py --lofo-pod-script   # print the pod check
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import pathlib
import random
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_arm_configs as bac  # noqa: E402
import build_rand_control as brc  # noqa: E402

BASE = REPO / "configs" / "data" / "JetClassII_base.yaml"
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
RAND_V2 = REPO / "configs" / "labelmaps" / "rand_label_map.v2.csv"
FLAV_CSV = REPO / "configs" / "labelmaps" / "flavour_pair_map.v2.csv"
FLAV_JSON = REPO / "configs" / "labelmaps" / "flavour_pair.v2.json"
PAIRS = REPO / "configs" / "labelmaps" / "probe_pairs.v2.json"
ARMS = REPO / "configs" / "arms"
V2 = ARMS / "v2"
MASS_JSON = V2 / "mass_lambda.v2.json"
LOFO_CHECK = V2 / "lofo_sample_check.json"
GRID = ARMS / "v2_grid.json"

TREE = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
VOCABS = ["L188", "L162", "R42_Q1", "R16_Q1"]
TARGET = "R16_Q1"
QCD_LO = brc.QCD_LO

# The v2 random partitions, drawn at random under a BALANCE RULE fixed on
# 2026-09-29 before any pool draw was made. The first five (seeds 45-49, never
# run) merged the visible-content pair in none of five, so a prediction "a
# partition that splits a pair beats one that merges it" could not be tested on
# the pair C4 was about.
#   rule       each BALANCE_PAIRS pair is merged in 2 or 3 of the 5 partitions
#              and split in the rest; shares exact; QCD one class
#   pool       share-matched draws (build_rand_control.py --pool resonant) for
#              seeds 100-199; extended by the next block of 100, in order, only
#              if no 5-subset of the pool meets the rule
#   selection  random.Random(SELECT_SEED).sample(pool, 5) repeated until a
#              sample meets the rule: uniform over the 5-subsets that do
# Recorded in configs/labelmaps/rand_v2_selection.json: the rule, every seed
# tried with its merge pattern, the number of samples, the accepted seeds.
RAND_V2_PREFIX = "RAND2_p"
RAND_SEL = REPO / "configs" / "labelmaps" / "rand_v2_selection.json"
BALANCE_PAIRS = {
    "bb/cc": ("label_X_bb", "label_X_cc"),
    "bbqq/ccqq": ("label_X_YY_bbqq", "label_X_YY_ccqq"),
    "visible": ("label_X_YY_bbqq", "label_X_YY_cqtauhv"),
    "bb/bbbb": ("label_X_bb", "label_X_YY_bbbb"),
    "ee/mm": ("label_X_ee", "label_X_mm"),
    "bc/bq": ("label_X_bc", "label_X_bq"),
    "bc/cs": ("label_X_bc", "label_X_cs"),
}
MERGED_IN = (2, 3)
N_PARTITIONS = 5
POOL_BLOCKS = [range(100 + 100 * i, 200 + 100 * i) for i in range(4)]
SELECT_SEED = 20260929
MAX_SAMPLES = 10_000_000

FLAV_SEED = 1
FLAV_TRIALS = 64
SPLIT_ORBIT = "X_YY_QQQQ"

# The v1 mass runs the lambda is matched on (under /data/results/mtx).
MASS_RUNS = {"L162": [f"mtx-l162mass-s{i}" for i in range(1, 6)],
             "R16_Q1": [f"mtx-r16q1mass-s{i}" for i in range(1, 6)]}
LAMBDA_V1 = 5.0
EPOCHS = range(0, 80)

# The 17-class group left out, named by one member.
LOFO_MEMBER = "label_X_YY_bbbb"
LOFO_SAMPLE_FILE = "/jc2/jet_data/Res34P_0000.parquet"   # a training-split file

# probe.py names of the six probe tasks (audit B8, Strand E 3)
PROBE_TASKS = {
    "bvc_resonant": "b vs c, two-prong (X->bb vs X->cc)",
    "bvc_4prong": "b vs c, four-prong (X->YY->bbqq vs X->YY->ccqq)",
    "visible_content": "visible decay content (X->YY->bbqq vs X->YY->cq tau_h nu)",
    "retained_topology": "two- vs four-prong (X->bb vs X->YY->bbbb)",
    "ee_vs_mm": "e vs mu (X->ee vs X->mumu)",
    "bc_vs_rest": "|Vcb| (X->bc vs X->bq, X->cs, X->YY->qqb, QCD)",
}


# ------------------------------------------------------------------ inputs
def read_map() -> list[dict]:
    with MAP.open() as f:
        return list(csv.DictReader(f))


def column(rows, col) -> dict[int, int]:
    return {int(r["jet_label"]): int(r[col]) for r in rows}


def names_of(rows) -> dict[int, str]:
    return {int(r["jet_label"]): r["class_name"] for r in rows}


def probe_tasks() -> dict:
    s = importlib.util.spec_from_file_location(
        "probe", REPO / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return {t: m.TASKS[t] for t in PROBE_TASKS}


# ----------------------------------------------------------- probe pairs
def sub_pairs(task: dict, names: dict[int, str]) -> list[tuple[str, list, list]]:
    """(label, signal natives, background natives): one per background class,
    with the 27 QCD classes kept together as one background."""
    short = lambda n: names[n].replace("label_", "")  # noqa: E731
    sig = task["signal"]
    sname = "+".join(short(n) for n in sig)
    qcd = [n for n in task["background"] if n >= QCD_LO]
    out = [(f"{sname}|{short(n)}", sig, [n])
           for n in task["background"] if n < QCD_LO]
    if qcd:
        out.append((f"{sname}|QCD", sig, qcd))
    return out


def merged(mapping: dict[int, int], a: list[int], b: list[int]) -> bool:
    return bool({mapping[n] for n in a} & {mapping[n] for n in b})


def pair_status(mapping, tasks, names) -> dict:
    out = {}
    for t, spec in tasks.items():
        m, s = [], []
        for label, a, b in sub_pairs(spec, names):
            (m if merged(mapping, a, b) else s).append(label)
        out[t] = {"merged": m, "split": s}
    return out


def first_merge_level(rows, tasks, names) -> dict:
    """For each probe sub-pair, the first tree level (finest first) at which
    its classes share a group, with that level's class count."""
    k = {lvl: len(set(column(rows, lvl).values())) for lvl in TREE}
    out = {}
    for t, spec in tasks.items():
        out[t] = {}
        for label, a, b in sub_pairs(spec, names):
            hit = next((lvl for lvl in TREE if merged(column(rows, lvl), a, b)), None)
            out[t][label] = {"level": hit, "num_classes": k[hit] if hit else None}
    return out


# ------------------------------------------- random partitions: selection
def merge_vector(mapping: dict[int, int], names: dict[int, str]) -> tuple[int, ...]:
    idx = {s: n for n, s in names.items()}
    return tuple(int(mapping[idx[a]] == mapping[idx[b]])
                 for a, b in BALANCE_PAIRS.values())


def rule_ok(vectors) -> bool:
    lo, hi = MERGED_IN
    return all(lo <= sum(col) <= hi for col in zip(*vectors))


def feasible(vectors: dict[int, tuple]) -> bool:
    """Whether ANY 5 distinct pool seeds meet the rule: backtracking over the
    distinct merge patterns, so the rejection sampler is known to terminate."""
    lo, hi = MERGED_IN
    types = collections.Counter(vectors.values())
    keys = sorted(types)

    def rec(i, left, sums):
        if max(sums) > hi or min(sums) + left < lo:
            return False
        if left == 0:
            return True
        if i == len(keys):
            return False
        for m in range(min(left, types[keys[i]]), -1, -1):
            if rec(i + 1, left - m, [s + m * b for s, b in zip(sums, keys[i])]):
                return True
        return False

    return rec(0, N_PARTITIONS, [0] * len(BALANCE_PAIRS))


def select(vectors: dict[int, tuple]) -> tuple[list[int], int]:
    rng = random.Random(SELECT_SEED)
    pool = sorted(vectors)
    for n in range(1, MAX_SAMPLES + 1):
        pick = sorted(rng.sample(pool, N_PARTITIONS))
        if rule_ok([vectors[s] for s in pick]):
            return pick, n
    raise SystemExit(f"FATAL: no sample of {MAX_SAMPLES} met the rule")


def _pool_draw(seed: int):
    import contextlib
    import io
    rows = read_map()
    with contextlib.redirect_stdout(io.StringIO()):
        assign = brc.share_draw(rows, brc.exact_share_units(), TARGET, seed, 0,
                                brc.POOLS["resonant"])
    return seed, merge_vector(assign, names_of(rows))


def select_rand(workers: int) -> dict:
    """Draw the pool block by block until the rule is feasible, sample the five,
    write the selection record and regenerate rand_label_map.v2.csv."""
    import multiprocessing
    vectors = {}
    for block in POOL_BLOCKS:
        with multiprocessing.Pool(workers) as p:
            vectors.update(p.imap_unordered(_pool_draw, block))
        print(f"  pool seeds {min(vectors)}-{max(vectors)}: rule "
              f"{'feasible' if feasible(vectors) else 'infeasible'}")
        if feasible(vectors):
            break
    else:
        raise SystemExit("FATAL: the balance rule is infeasible on every pool block")
    accepted, n = select(vectors)
    pool = sorted(vectors)
    record = {
        "rule": (f"each of the {len(BALANCE_PAIRS)} pairs is merged in "
                 f"{MERGED_IN[0]} or {MERGED_IN[1]} of the {N_PARTITIONS} partitions "
                 "and split in the rest; shares exact; QCD one class"),
        "fixed": "2026-09-29, before any pool draw",
        "pairs": BALANCE_PAIRS, "merged_in": list(MERGED_IN),
        "draw": "build_rand_control.share_draw, --pool resonant, identified by seed",
        "selection": (f"random.Random({SELECT_SEED}).sample(sorted pool, "
                      f"{N_PARTITIONS}) until a sample meets the rule"),
        "select_seed": SELECT_SEED,
        "seeds_tried": pool,
        "merge_vectors": {str(s): list(vectors[s]) for s in pool},
        "pool_merge_rate": {k: sum(vectors[s][i] for s in pool) / len(pool)
                            for i, k in enumerate(BALANCE_PAIRS)},
        "samples_drawn": n,
        "accepted_seeds": accepted,
        "accepted_merged_count": {k: sum(vectors[s][i] for s in accepted)
                                  for i, k in enumerate(BALANCE_PAIRS)},
    }
    RAND_SEL.write_text(json.dumps(record, indent=1) + "\n")
    brc.main(["--pool", "resonant", "--prefix", RAND_V2_PREFIX, "--seeds",
              *map(str, accepted), "--out", str(RAND_V2)])
    return record


def rand_v2_seeds() -> list[int]:
    return json.loads(RAND_SEL.read_text())["accepted_seeds"]


# ------------------------------------------------------------- F0 / F1
def orbit_key(class_name: str) -> str:
    """Same decay up to b <-> c <-> light-quark exchange. Only quark letters
    are b, c, s, q in JetClass-II class names; tau, lepton and neutrino
    tokens contain none of them."""
    return re.sub("[bcsq]", "Q", class_name[len("label_"):])


def orbits(names: dict[int, str]) -> dict[str, list[int]]:
    out = collections.defaultdict(list)
    for n in sorted(names):
        if n < QCD_LO:
            out[orbit_key(names[n])].append(n)
    return dict(out)


def agreement(block_of: dict[int, int], tgt: dict[int, int]) -> int:
    """Unordered native pairs together in BOTH partitions."""
    c = collections.Counter((block_of[n], tgt[n]) for n in block_of)
    return sum(x * (x - 1) // 2 for x in c.values())


def place_orbits(shape, by_value, orb, tgt, tag):
    """Orbit -> block for a value-count shape, scrambled against `tgt`.

    Orbits of equal share are interchangeable for the sums, so swapping two of
    them between blocks keeps every block exact; local search on those swaps
    minimises the native pairs shared with the target partition."""
    need = {b: dict(shape[b]) for b in shape}
    place = {}
    for v in brc.hshuf(sorted(by_value), f"{tag}|values"):
        for o in brc.hshuf(by_value[v], f"{tag}|v{v}"):
            b = next(b for b in brc.hshuf(sorted(need), f"{tag}|{o}")
                     if need[b].get(v, 0) > 0)
            place[o] = b
            need[b][v] -= 1

    def score(p):
        return agreement({n: p[o] for o, mem in orb.items() for n in mem}, tgt)

    best = score(place)
    improved = True
    while improved:
        improved = False
        for group in by_value.values():
            for i, j in itertools.combinations(group, 2):
                if place[i] == place[j]:
                    continue
                place[i], place[j] = place[j], place[i]
                s = score(place)
                if s < best:
                    best, improved = s, True
                else:
                    place[i], place[j] = place[j], place[i]
    return place, best


def split_move(place, orb, oshare, units, names):
    """F1's move: the b-containing half S of SPLIT_ORBIT goes to a block B, and
    whole orbits T of B with share(T) == share(S) come back. Returns
    (S, B, T) with the fewest orbits in T, or None if no exact T exists."""
    s_nat = [n for n in orb[SPLIT_ORBIT] if "b" in names[n][len("label_X_YY_"):]]
    s_share = sum(units[n] for n in s_nat)
    a = place[SPLIT_ORBIT]
    for size in range(1, 5):
        hits = []
        for b in sorted(set(place.values()) - {a}):
            mine = sorted(o for o in place if place[o] == b)
            for t in itertools.combinations(mine, size):
                if sum(oshare[o] for o in t) == s_share:
                    hits.append((b, t))
        if hits:
            b, t = min(hits, key=lambda h: (h[1], h[0]))
            return s_nat, b, list(t)
    return None


def build_flavour_pair(rows, units):
    names = names_of(rows)
    tgt = column(rows, TARGET)
    orb = orbits(names)
    oshare = {o: sum(units[n] for n in mem) for o, mem in orb.items()}
    groups = sorted({tgt[n] for n in range(QCD_LO)})
    targets = [sum(units[n] for n in range(QCD_LO) if tgt[n] == g) for g in groups]
    counts = collections.Counter(oshare.values())
    by_value = collections.defaultdict(list)
    for o in sorted(orb):
        by_value[oshare[o]].append(o)

    best = None
    for t in range(FLAV_TRIALS):
        tag = f"flav|seed={FLAV_SEED}|t{t}"
        shape = brc.solve_shape(counts, targets, tag)
        if shape is None:
            continue
        place, agree = place_orbits(shape, by_value, orb, tgt, tag)
        move = split_move(place, orb, oshare, units, names)
        if move is not None and (best is None or agree < best[0]):
            best = (agree, t, place, move)
    if best is None:
        raise SystemExit("FATAL: no share-exact flavour-blind partition admits "
                         "the F1 split")
    agree, trial, place, (s_nat, b_blk, t_orbs) = best
    a_blk = place[SPLIT_ORBIT]
    qcd_gid = len(groups)
    f0 = {n: place[o] for o, mem in orb.items() for n in mem}
    f1 = dict(f0)
    for n in s_nat:
        f1[n] = b_blk
    for o in t_orbs:
        for n in orb[o]:
            f1[n] = a_blk
    for n in names:
        if n >= QCD_LO:
            f0[n] = f1[n] = qcd_gid

    def mismatch(m):
        got = [sum(units[n] for n in range(QCD_LO) if m[n] == g) for g in range(len(groups))]
        return max(abs(x - y) for x, y in zip(got, targets))

    record = {
        "construction": (
            "F0: each of the 16 resonant groups is a union of whole flavour orbits "
            "(same decay up to b/c/light-quark exchange), share-exact against the "
            "17-class groups (group g has 17-class group g's share), scrambled by "
            "local search against the 17-class partition; QCD is one class. F1: F0 "
            "with the b-containing classes of the four-prong hadronic orbit moved "
            "to group B and whole orbits of equal share moved from B to that "
            "orbit's group A."),
        "seed": FLAV_SEED, "trial": trial, "trials": FLAV_TRIALS,
        "n_orbits": len(orb),
        "orbit_of_class": {names[n]: orbit_key(names[n]) for n in range(QCD_LO)},
        "native_pairs_shared_with_17_class": agree,
        "native_pairs_in_17_class_groups": agreement(
            {n: tgt[n] for n in range(QCD_LO)}, tgt),
        "split_orbit": SPLIT_ORBIT,
        "split_classes_moved": [names[n] for n in s_nat],
        "group_A": a_blk, "group_B": b_blk,
        "orbits_moved_B_to_A": t_orbs,
        "share_units_denominator": sum(units.values()),
        "moved_share_units": sum(units[n] for n in s_nat),
        "max_abs_share_mismatch_units": {"F0": mismatch(f0), "F1": mismatch(f1)},
    }
    return f0, f1, record


def group_names(m: dict[int, int], tag: str) -> dict[int, str]:
    return {g: "QCD_ALL" if n >= QCD_LO else f"{tag}_{g:02d}" for n, g in m.items()}


def write_flavour_csv(rows, f0, f1):
    names = names_of(rows)
    with FLAV_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jet_label", "class_name", "orbit", "FLAV_F0", "FLAV_F0_name",
                    "FLAV_F1", "FLAV_F1_name"])
        for n in sorted(names):
            orb = orbit_key(names[n]) if n < QCD_LO else "QCD"
            row = [n, names[n], orb]
            for m, tag in ((f0, "F0"), (f1, "F1")):
                row += [m[n], group_names(m, tag)[m[n]]]
            w.writerow(row)


def read_two_col_map(path, col):
    with path.open() as f:
        r = list(csv.DictReader(f))
    return ({int(x["jet_label"]): int(x[col]) for x in r},
            {int(x[col]): x[f"{col}_name"] for x in r})


# ---------------------------------------------------------- mass lambda
LOSS_RE = re.compile(r"Train AvgLoss: ([\d.]+), AvgLossReg: ([\d.]+), "
                     r"AvgLossTot: ([\d.]+), AvgAcc: [\d.]+ \(lambda=([\d.]+)\)")
EPOCH_RE = re.compile(r"Epoch #(\d+) training")


def parse_train_log(text: str) -> dict[int, dict]:
    """Per-epoch training averages from a hybrid_mass.py train.log.

    Each 'Train AvgLoss' line belongs to the last 'Epoch #N training' before
    it. Resumes restart an epoch whose training never finished, so they leave a
    bare 'Epoch #N training' line and no average. An epoch with TWO averages
    would need a choice between them, so it is refused rather than guessed.
    """
    out, ep = {}, None
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1))
            continue
        m = LOSS_RE.search(line)
        if m:
            if ep is None or ep in out:
                raise ValueError(f"loss line without a unique epoch: {line!r}")
            cls, reg, tot, lam = map(float, m.groups())
            out[ep] = {"loss_cls": cls, "loss_reg": reg, "loss_tot": tot,
                       "lambda": lam}
            ep = None
    return out


def extract_mass_logs(dump_dir: pathlib.Path) -> dict:
    """Parse `sha256sum train.log; grep 'Epoch #|Train AvgLoss' train.log`
    dumps, one <run>.txt per run, into the committed input table."""
    runs = {}
    for arm, rs in MASS_RUNS.items():
        for run in rs:
            text = (dump_dir / f"{run}.txt").read_text()
            sha, path = text.splitlines()[0].split()
            ep = parse_train_log(text)
            if sorted(ep) != list(EPOCHS):
                raise SystemExit(f"FATAL: {run}: epochs {sorted(ep)[:3]}... not 0-79")
            runs[run] = {"arm": arm, "train_log": path, "train_log_sha256": sha,
                         "epochs": [ep[e] for e in EPOCHS]}
    return runs


def x_of(e: dict) -> float:
    return e["lambda"] * e["loss_reg"] / e["loss_cls"]


def mass_lambda(runs: dict) -> dict:
    """lambda_m = 5 x_162 / x_17, x = lambda L_reg / L_cls, several averagings."""
    for r in runs.values():
        if any(e["lambda"] != LAMBDA_V1 for e in r["epochs"]):
            raise SystemExit("FATAL: a v1 mass run was not trained at lambda = 5")

    def per_run(fn):
        return {arm: [fn(runs[r]["epochs"]) for r in rs] for arm, rs in MASS_RUNS.items()}

    variants = {
        "mean_over_epochs_0_79": per_run(lambda es: statistics.fmean(x_of(e) for e in es)),
        "epoch_0": per_run(lambda es: x_of(es[0])),
        "epoch_79": per_run(lambda es: x_of(es[79])),
        "ratio_of_epoch_sums": per_run(
            lambda es: LAMBDA_V1 * sum(e["loss_reg"] for e in es)
            / sum(e["loss_cls"] for e in es)),
    }
    out = {}
    for name, x in variants.items():
        fine, coarse = x["L162"], x["R16_Q1"]
        out[name] = {
            "x_162_per_run": fine, "x_17_per_run": coarse,
            "lambda_per_seed_pair": [LAMBDA_V1 * f / c for f, c in zip(fine, coarse)],
            "lambda_pooled": LAMBDA_V1 * statistics.fmean(fine) / statistics.fmean(coarse),
            "share_162_pooled": statistics.fmean(fine) / (1 + statistics.fmean(fine)),
            "share_17_pooled": statistics.fmean(coarse) / (1 + statistics.fmean(coarse)),
        }
    chosen = round(out["mean_over_epochs_0_79"]["lambda_pooled"], 2)
    return {"formula": "lambda_m = 5 * mean_s(x_162) / mean_s(x_17), "
                       "x = lambda * AvgLossReg / AvgLoss per epoch, "
                       "averaged over epochs 0-79 of each run",
            "lambda_m": chosen, "variants": out}


# --------------------------------------------------------- config text
def arm_text(base, path, mapping, names, source, mass=False, extra_label_lines=()):
    """build_arm_configs.build_one, with this generator's provenance.

    Only comment lines are rewritten; each rewrite must hit exactly once, so a
    change in build_one's wording stops the build instead of passing through."""
    arm = path.stem
    text = bac.build_one(base, arm, mapping, names, mass=mass)
    subs = [(f"# configs/arms/{arm}.yaml\n"
             "# GENERATED by scripts/build_arm_configs.py -- do not hand-edit.\n",
             f"# {path.relative_to(REPO)}\n"
             "# GENERATED by scripts/build_v2_arms.py (build_arm_configs.build_one)"
             " -- do not hand-edit.\n"),
            ("   ### GENERATED by scripts/build_arm_configs.py -- do not hand-edit.\n",
             "   ### GENERATED by scripts/build_v2_arms.py -- do not hand-edit.\n"),
            ("(invariant I4). Source: configs/labelmaps/rung_label_maps.v1.csv.\n",
             f"(invariant I4). Source: {source}.\n")]
    k = len(set(mapping.values()))
    anchor = f"   ### Launch this arm with:  -o num_classes {k}\n"
    subs.append((anchor, anchor + "".join(f"   ### {l}\n" for l in extra_label_lines)))
    for old, new in subs:
        if text.count(old) != 1:
            raise SystemExit(f"FATAL: {arm}: expected one {old.strip()!r}")
        text = text.replace(old, new)
    return text


def build_configs(base, rows, f0, f1, lam) -> dict[pathlib.Path, str]:
    tree_src = "configs/labelmaps/rung_label_maps.v1.csv"
    tree_names = lambda lvl: {int(r[lvl]): r[f"{lvl}_name"] for r in rows}  # noqa: E731
    out = {}
    for lvl in ("R63_Q1", "R29_Q1"):
        p = ARMS / f"{lvl}.yaml"
        out[p] = arm_text(base, p, column(rows, lvl), tree_names(lvl), tree_src)
    for d in range(1, N_PARTITIONS + 1):
        p = V2 / f"{RAND_V2_PREFIX}{d}.yaml"
        m, nm = read_two_col_map(RAND_V2, p.stem)
        out[p] = arm_text(base, p, m, nm, "configs/labelmaps/rand_label_map.v2.csv")
    for arm, m in (("FLAV_F0", f0), ("FLAV_F1", f1)):
        p = V2 / f"{arm}.yaml"
        out[p] = arm_text(base, p, m, group_names(m, arm[-2:]),
                          "configs/labelmaps/flavour_pair_map.v2.csv")
    p = V2 / "R16_Q1_MASS_LM.yaml"
    out[p] = arm_text(
        base, p, column(rows, TARGET), tree_names(TARGET), tree_src, mass=True,
        extra_label_lines=(f"and with:  --mass-lambda {lam}",
                           "(the 162-class mass model's loss share at lambda 5;",
                           " derivation in configs/arms/v2/mass_lambda.v2.json)"))
    return out


def check_config(text, expected: dict[int, int], base_sha) -> list[str]:
    fails = []
    expr = re.search(r"truth_label: (.*)", text).group(1)
    if bac.evaluate(expr) != expected:
        fails.append("expression does not reproduce the label map")
    if bac.weights_sha256(text) != base_sha:
        fails.append("weights block differs from the base")
    k = len(set(expected.values()))
    if set(expected.values()) != set(range(k)):
        fails.append("group ids not dense from zero")
    return fails


# ------------------------------------------------------------------ LOFO
def lofo_natives(rows) -> list[int]:
    names = names_of(rows)
    tgt = column(rows, TARGET)
    g = tgt[next(n for n, s in names.items() if s == LOFO_MEMBER)]
    return sorted(n for n in names if tgt[n] == g)


def lofo_extra_selection(rows) -> str:
    return f"~({bac.condition_for(lofo_natives(rows))})"


def lofo_pod_script(rows) -> str:
    """Self-contained check, run in a pod with weaver 0.4.17 and /jc2 mounted:
    weaver's own DataConfig.load(extra_selection=...) and _apply_selection on
    one training file, for each vocabulary's parent config."""
    shas = {a: hashlib.sha256((ARMS / f"{a}.yaml").read_bytes()).hexdigest()
            for a in VOCABS}
    return f'''import hashlib, json
import awkward as ak, numpy as np
import importlib.metadata as md
from weaver.utils.data.config import DataConfig
from weaver.utils.data.preprocess import _apply_selection
EXTRA = {lofo_extra_selection(rows)!r}
FAMILY = {lofo_natives(rows)!r}
FILE = {LOFO_SAMPLE_FILE!r}
SHAS = {shas!r}
t = ak.from_parquet(FILE, columns=["jet_label", "jet_pt", "jet_sdmass"])
fam = lambda x: int(np.isin(ak.to_numpy(x["jet_label"]), FAMILY).sum())
out = {{"file": FILE, "n_jets": len(t), "n_family_in_file": fam(t),
        "extra_selection": EXTRA, "weaver_core": md.version("weaver-core"), "arms": {{}}}}
for arm, sha in SHAS.items():
    path = f"/workspace/transferlearningsophon/configs/arms/{{arm}}.yaml"
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    parent = DataConfig.load(path, load_observers=False)
    lofo = DataConfig.load(path, load_observers=False, extra_selection=EXTRA)
    p, l = _apply_selection(t, parent.selection), _apply_selection(t, lofo.selection)
    out["arms"][arm] = {{"config_sha256_pod": got, "config_sha256_local": sha,
        "selection": lofo.selection, "n_pass_parent": len(p), "n_pass_lofo": len(l),
        "n_family_pass_parent": fam(p), "n_family_pass_lofo": fam(l)}}
print(json.dumps(out, indent=1))
'''


# -------------------------------------------------------------- registry
def registry(rows, lam, lofo_expr) -> dict:
    k = {lvl: len(set(column(rows, lvl).values())) for lvl in TREE}
    arms = []

    def add(name, config, num_classes, runs, tier, mass_lambda=None, **extra):
        arms.append({"name": name, "config": config, "num_classes": num_classes,
                     "mass_lambda": mass_lambda, "runs": runs, "tier": tier,
                     "extra_selection": extra.pop("extra_selection", None), **extra})

    for v in VOCABS:
        add(v, f"configs/arms/{v}.yaml", k[v], 5, 1, objective="classification")
    add("L162_MASS", "configs/arms/L162_MASS.yaml", k["L162"], 5, 1, LAMBDA_V1,
        objective="classification+mass")
    add("R16_Q1_MASS", "configs/arms/R16_Q1_MASS.yaml", k[TARGET], 5, 1, LAMBDA_V1,
        objective="classification+mass")
    for d, seed in enumerate(rand_v2_seeds(), start=1):
        add(f"{RAND_V2_PREFIX}{d}", f"configs/arms/v2/{RAND_V2_PREFIX}{d}.yaml",
            k[TARGET], 2, 1, objective="classification", partition_seed=seed)
    for arm in ("FLAV_F0", "FLAV_F1"):
        add(arm, f"configs/arms/v2/{arm}.yaml", k[TARGET], 2, 2,
            objective="classification", partition_seed=FLAV_SEED)
    add("R16_Q1_MASS_LM", "configs/arms/v2/R16_Q1_MASS_LM.yaml", k[TARGET], 5, 2, lam,
        objective="classification+mass")
    add("MPM", "configs/arms/L188.yaml", None, 3, 2, objective="mpm")
    for lvl in ("R63_Q1", "R29_Q1"):
        add(lvl, f"configs/arms/{lvl}.yaml", k[lvl], 5, 3, objective="classification")
    for v in VOCABS:
        add(f"{v}_LOFO4P", f"configs/arms/{v}.yaml", k[v], 3, 3,
            objective="classification", extra_selection=lofo_expr, parent=v)
    return {"generated_by": "scripts/build_v2_arms.py",
            "fields": {
                "config": "weaver --data-config; its make_weight sidecar is keyed "
                          "by this file's md5",
                "num_classes": "weaver -o num_classes; null for the self-supervised arm",
                "mass_lambda": "hybrid_mass.py --mass-lambda; null = no mass output",
                "extra_selection": "weaver --extra-selection; applied after the "
                                   "reweighting histograms, so the parent config "
                                   "and its sidecar are used unchanged",
                "runs": "number of pretraining seeds", "tier": "1 first"},
            "arms": arms}


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--mass-logs", type=pathlib.Path,
                    help="directory of <run>.txt log dumps; rewrites mass_lambda.v2.json")
    ap.add_argument("--lofo-pod-script", action="store_true")
    ap.add_argument("--pairs-out", type=pathlib.Path,
                    help="also write the probe-pair table here")
    ap.add_argument("--select-rand", type=int, metavar="WORKERS",
                    help="redraw the pool and select the five random partitions "
                         "(rewrites rand_v2_selection.json and rand_label_map.v2.csv)")
    a = ap.parse_args(argv)
    if a.select_rand:
        select_rand(a.select_rand)

    rows = read_map()
    if a.lofo_pod_script:
        print(lofo_pod_script(rows))
        return 0
    units = brc.exact_share_units()
    base = BASE.read_text()
    base_sha = bac.weights_sha256(base)

    if a.mass_logs:
        runs = extract_mass_logs(a.mass_logs)
    else:
        runs = json.loads(MASS_JSON.read_text())["runs"]
    lam = mass_lambda(runs)

    f0, f1, frec = build_flavour_pair(rows, units)
    if not a.check_only:
        write_flavour_csv(rows, f0, f1)
    maps = {"FLAV_F0": f0, "FLAV_F1": f1}
    for d in range(1, N_PARTITIONS + 1):
        arm = f"{RAND_V2_PREFIX}{d}"
        maps[arm] = read_two_col_map(RAND_V2, arm)[0]
    for lvl in ("R63_Q1", "R29_Q1"):
        maps[lvl] = column(rows, lvl)
    sel = json.loads(RAND_SEL.read_text())
    got = [list(merge_vector(maps[f"{RAND_V2_PREFIX}{d}"], names_of(rows)))
           for d in range(1, N_PARTITIONS + 1)]
    want = [sel["merge_vectors"][str(s)] for s in sel["accepted_seeds"]]
    if got != want or not rule_ok(got):
        print("  [FAIL] rand_label_map.v2.csv is not the recorded selection")
        failed = 1
    else:
        failed = 0
    maps["R16_Q1_MASS_LM"] = column(rows, TARGET)

    configs = build_configs(base, rows, f0, f1, lam["lambda_m"])
    for path, text in configs.items():
        fails = check_config(text, maps[path.stem], base_sha)
        print(f"  [{'FAIL' if fails else 'PASS'}] {path.relative_to(REPO)} "
              f"K={len(set(maps[path.stem].values()))} {'; '.join(fails)}")
        failed += bool(fails)

    tasks = probe_tasks()
    names = names_of(rows)
    status = {v: pair_status(column(rows, v), tasks, names) for v in VOCABS}
    for arm in list(maps):
        if arm.startswith(("RAND2", "FLAV")):
            status[arm] = pair_status(maps[arm], tasks, names)
    pairs = {"generated_by": "scripts/build_v2_arms.py",
             "probe_tasks": {t: {"description": PROBE_TASKS[t],
                                 "sub_pairs": [p[0] for p in sub_pairs(tasks[t], names)]}
                             for t in tasks},
             "partition_seeds": {**{f"{RAND_V2_PREFIX}{d}": s for d, s in
                                    enumerate(rand_v2_seeds(), start=1)},
                                 "FLAV_F0": FLAV_SEED, "FLAV_F1": FLAV_SEED},
             "status": status,
             "tree_first_merge": first_merge_level(rows, tasks, names)}

    lofo_expr = lofo_extra_selection(rows)
    got = bac.evaluate(f"1 * ({lofo_expr})")
    fam = set(lofo_natives(rows))
    if {n for n, v in got.items() if v == 0} != fam:
        print("  [FAIL] LOFO extra_selection does not remove exactly the family")
        failed += 1
    grid = registry(rows, lam["lambda_m"], lofo_expr)

    print(f"  lambda_m = {lam['lambda_m']}  "
          f"(pooled, unrounded {lam['variants']['mean_over_epochs_0_79']['lambda_pooled']:.4f})")
    print(f"  F0/F1: trial {frec['trial']}, {frec['native_pairs_shared_with_17_class']} "
          f"native pairs shared with the 17-class groups; share mismatch "
          f"{frec['max_abs_share_mismatch_units']}")
    if failed:
        print(f"\nBUILD FAILED - {failed} check(s), nothing written")
        return 1
    if a.check_only:
        return 0

    V2.mkdir(parents=True, exist_ok=True)
    for path, text in configs.items():
        path.write_text(text)
    MASS_JSON.write_text(json.dumps({**lam, "runs": runs}, indent=1) + "\n")
    FLAV_JSON.write_text(json.dumps(frec, indent=1) + "\n")
    PAIRS.write_text(json.dumps(pairs, indent=1) + "\n")
    if a.pairs_out:
        a.pairs_out.write_text(json.dumps(pairs, indent=1) + "\n")
    GRID.write_text(json.dumps(grid, indent=1) + "\n")
    print(f"wrote {len(configs)} configs, {GRID.relative_to(REPO)}, "
          f"{PAIRS.relative_to(REPO)}, {FLAV_CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
