#!/usr/bin/env python3
"""Build and validate the locked contraction ladder, eight rungs:

    L188 > L162 > R63_Q1 > R42_Q1 > R29_Q1 > R16_Q1 > R3_VIS > R1_Q1

Derives everything from `hierarchy/01_class_master.csv` using ONLY the stated
physical rules. Nothing is transcribed from prose; every group is computed and
then checked against the independently published group-size vectors.

The tree is the frozen label HIERARCHY. Which rungs become pretraining arms is
a separate, compute-gated decision that lives in docs/RUN_MATRIX.md -- an
8-rung tree does not imply 8 pretraining runs.

R3_VIS counts VISIBLE objects, so a neutrino does not add a prong. This follows
the JetClass-II authors' own description (arXiv:2405.12972 sec. 2): "This
results primarily in 4-prong signatures but can also be 3 prongs if an object
leaks out of the jet cone or if one of the objects is a neutrino."

Outputs
    configs/labelmaps/contraction_tree.v1.yaml     nested tree, full expansion
    configs/labelmaps/stream_share_audit.v1.csv    four-number audit, all rungs
    configs/labelmaps/rung_label_maps.v1.csv       materialized map, ALL rungs

Run:  python3 scripts/build_contraction_tree.py [--check-only]

Every assertion is a CI test (see tests/test_contraction_tree.py). Exit code is
non-zero if any assertion fails, so this doubles as the CI entry point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from collections import OrderedDict
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASTER = ROOT / "hierarchy" / "01_class_master.csv"
OUT_YAML = ROOT / "configs" / "labelmaps" / "contraction_tree.v1.yaml"
OUT_CSV = ROOT / "configs" / "labelmaps" / "stream_share_audit.v1.csv"
OUT_MAPS = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"

CLASS_WEIGHTS_SUM = Fraction("1.73175")

# ---------------------------------------------------------------- expected ---
# Independently published in the locked plan (report section 6.A). These are
# ASSERTIONS, not inputs: the tree is derived from the CSV and must reproduce
# them. If the derivation and these disagree, the build fails loudly.
EXPECTED_R16_RESONANT = 16
EXPECTED_R42_RESONANT = 42
EXPECTED_NATIVE = 188
EXPECTED_R16_SIZES = [10, 1, 2, 2, 29, 5, 10, 10, 10, 27, 5, 10, 10, 12, 12, 6]

# Group counts per rung, INCLUDING the single QCD_ALL group at every rung below
# L188. Independently stated; the derivation must reproduce them.
EXPECTED_RUNG_GROUPS = {
    "L188": 188, "L162": 162, "R63_Q1": 64, "R42_Q1": 43,
    "R29_Q1": 30, "R16_Q1": 17, "R3_VIS": 4, "R1_Q1": 2,
}

# The contraction order is FROZEN. Each rung must strictly coarsen the one
# before it; `check_chain_nesting` enforces that for every consecutive pair.
LADDER = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]

N_RESONANT = 161                 # native 0-160; 161-187 are QCD

# Canonical topology -> (published R16 id, published name). The ORDER here is
# the frozen group-id order: prong class ascending, then hadronic ->
# semileptonic -> leptonic within a prong class. It is semantic, not derivable
# from native label indices, so it is stated explicitly and frozen.
R16_ORDER: "OrderedDict[str, str]" = OrderedDict([
    ("PP",         "2P_HAD_2PARTON"),
    ("tauhtauh",   "2P_HAD_TAUH_TAUH"),
    ("tauhtaul",   "2P_SEMILEP_TAUH_TAUL"),
    ("ll",         "2P_LEP_LL"),
    ("PPP",        "3P_HAD_3PARTON"),
    ("Ptauhtauh",  "3P_HAD_1PARTON_TAUH_TAUH"),
    ("PPl",        "3P_SEMILEP_2PARTON_L"),
    ("Pll",        "3P_SEMILEP_1PARTON_LL"),
    ("Ptauhtaul",  "3P_SEMILEP_1PARTON_TAUH_TAUL"),
    ("PPPP",       "4P_HAD_4PARTON"),
    ("PPtauhtauh", "4P_HAD_2PARTON_TAUH_TAUH"),
    ("PPll",       "4P_SEMILEP_2PARTON_LL"),
    ("PPtauhtaul", "4P_SEMILEP_2PARTON_TAUH_TAUL"),
    ("PPlv",       "4P_SEMILEP_NU_2PARTON_L"),
    ("PPtaulv",    "4P_SEMILEP_NU_2PARTON_TAUL"),
    ("PPtauhv",    "4P_HADVIS_NU_2PARTON_TAUH"),
])

TIER_ORDER = ["B", "C", "LG"]

# Longest-first so 'tauhtauh' is not consumed as 'tauh'+'tauh'.
_TOKENS = ("tauhtauh", "tauhtaul", "tauhv", "taulv", "tauh", "taul",
           "ll", "lv", "l", "v")


# I2's weights-block digest, stamped into the artifact at sign-off rather than
# left as PENDING. Recomputed and asserted by
# tests/test_arm_configs.py::test_weights_block_is_byte_identical_and_matches_base;
# if that test goes red this constant is stale and the artifact is lying.
WEIGHTS_BLOCK_SHA256 = "51a31ba239e3cfe553efbafbd88b2e02536e86cbd442efc544d12d328711d20b"

def canonical_topology(code: str) -> str:
    """Map a native topology code to its R16 canonical form.

    R16 partitions by [topology, visible decay mode] with flavour collapsed
    everywhere, so a quark parton (Q) and a gluon parton (g) are the same
    object. Everything else is preserved verbatim.
    """
    out, i = [], 0
    while i < len(code):
        for tok in _TOKENS:
            if code.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                break
        else:
            out.append("P" if code[i] in "Qg" else code[i])
            i += 1
    return "".join(out)


def flavour_tier(row: dict) -> str:
    """Three-tier heavy-flavour hierarchy, priority B > C > LG."""
    if int(row["has_b"]) >= 1:
        return "B"
    if int(row["has_c"]) >= 1:
        return "C"
    return "LG"


def hf_counts(row: dict) -> tuple[int, int]:
    """(n_b, n_c) parton COUNTS, read from the `tokens` column.

    Must come from tokens, never from has_b/has_c: those columns are booleans,
    so label_X_bb has has_b == 1, not 2. Using them would silently collapse
    R63's entire reason for existing -- multiplicity -- back into presence.
    """
    toks = [t for t in row["tokens"].split("+") if t]
    return toks.count("b"), toks.count("c")


def r29_name(r42_name: str) -> str:
    """B and C both become HF; LG untouched; unsplit names pass through."""
    if "|" not in r42_name:
        return r42_name
    base, tier = r42_name.rsplit("|", 1)
    if tier not in ("B", "C", "LG"):
        raise ValueError(f"unrecognised flavour tier {tier!r} in {r42_name!r}")
    return f"{base}|{'HF' if tier in ('B', 'C') else 'LG'}"


def build_rungs(built: dict) -> "OrderedDict[str, dict[int, tuple[int, str]]]":
    """native jet_label -> (group_id, group_name), for every rung on the ladder.

    Group-id ordering rules, all deterministic and all semantic rather than
    derived from native label indices (which is why they are stated here and
    frozen, exactly as the R16 order is):

        R63_Q1  R42_Q1 id order, then first native-label appearance within an
                R42 group -- so R42's ordering is inherited, not reinvented.
                Verified to reproduce the reviewer's R63_Q1.csv ids exactly;
                sorting the cells by (n_b, n_c) instead does NOT, in either
                direction, so this rule is not interchangeable with it.
        R29_Q1  first appearance in R42_Q1 id order.
        R3_VIS  visible-object count ascending, then QCD_ALL last.
        R1_Q1   RESONANT_ALL, then QCD_ALL.
    """
    tree, rows = built["tree"], built["rows"]
    by_label = {int(r["jet_label"]): r for r in rows}
    n2r42 = {n: c["r42_id"] for g in tree for c in g["children"] for n in c["natives"]}
    r42_name = {c["r42_id"]: c["r42_name"] for g in tree for c in g["children"]}
    n2r16 = {n: g["r16_id"] for g in tree for c in g["children"] for n in c["natives"]}
    r16_name = {g["r16_id"]: g["r16_name"] for g in tree}
    labels = sorted(by_label)

    out: "OrderedDict[str, dict[int, tuple[int, str]]]" = OrderedDict()

    out["L188"] = {n: (n, by_label[n]["class_name"]) for n in labels}
    out["L162"] = {n: (n, by_label[n]["class_name"]) if n < N_RESONANT
                   else (N_RESONANT, "QCD_ALL") for n in labels}

    # R63: R42's topology x flavour-tier cell, refined by (n_b, n_c) counts.
    # The (n_b, n_c) suffix is applied UNIFORMLY, including to groups that do
    # not actually split -- a colourless group is |nb0_nc0. Uniform naming keeps
    # the rung self-describing; conditional suffixing produces names like
    # `...|C|nb0_nc1` that state the tier and the counts redundantly.
    r63_cell: dict[int, tuple[int, int, int]] = {}
    for n in labels:
        r63_cell[n] = (n2r42[n], *hf_counts(by_label[n])) if n < N_RESONANT \
            else (n2r42[n], -1, -1)
    first_seen: dict[tuple[int, int, int], int] = {}
    for n in labels:
        first_seen.setdefault(r63_cell[n], n)
    order63: "OrderedDict[tuple[int, int, int], int]" = OrderedDict()
    for cell in sorted(set(r63_cell.values()), key=lambda c: (c[0], first_seen[c])):
        order63[cell] = len(order63)
    out["R63_Q1"] = {}
    for n in labels:
        cell = r63_cell[n]
        nm = "QCD_ALL" if n >= N_RESONANT else \
            f"{r16_name[n2r16[n]]}|nb{cell[1]}_nc{cell[2]}"
        out["R63_Q1"][n] = (order63[cell], nm)

    out["R42_Q1"] = {n: (n2r42[n], r42_name[n2r42[n]]) for n in labels}

    order29: "OrderedDict[str, int]" = OrderedDict()
    for n in sorted(labels, key=lambda n: n2r42[n]):
        nm = r29_name(r42_name[n2r42[n]])
        order29.setdefault(nm, len(order29))
    out["R29_Q1"] = {n: (order29[r29_name(r42_name[n2r42[n]])],
                         r29_name(r42_name[n2r42[n]])) for n in labels}

    out["R16_Q1"] = {n: (n2r16[n], r16_name[n2r16[n]]) for n in labels}

    # R3_VIS: prong count on VISIBLE objects. A neutrino does not add a prong.
    vis = {n: int(by_label[n]["n_visible_objects"]) for n in labels if n < N_RESONANT}
    order3 = {v: i for i, v in enumerate(sorted(set(vis.values())))}
    out["R3_VIS"] = {n: (order3[vis[n]], f"{vis[n]}P_VIS") if n < N_RESONANT
                     else (len(order3), "QCD_ALL") for n in labels}

    out["R1_Q1"] = {n: (0, "RESONANT_ALL") if n < N_RESONANT else (1, "QCD_ALL")
                    for n in labels}

    return out


def check_chain_nesting(rungs) -> list[tuple[str, bool, str]]:
    """Every consecutive pair on the ladder must STRICTLY coarsen.

    Coarsen: every group of the finer rung sits inside exactly one group of the
    coarser rung. Strict: the coarser rung has fewer groups. A chain that
    coarsens but not strictly has a redundant rung; one that does not coarsen is
    not a ladder at all and every downstream contrast is uninterpretable.
    """
    checks = []
    names = list(rungs)
    for fine, coarse in zip(names, names[1:]):
        f, c = rungs[fine], rungs[coarse]
        blocks: dict[int, set[int]] = {}
        for n in f:
            blocks.setdefault(f[n][0], set()).add(c[n][0])
        bad = {g: sorted(v) for g, v in blocks.items() if len(v) > 1}
        n_f = len({v[0] for v in f.values()})
        n_c = len({v[0] for v in c.values()})
        checks.append((f"nests_{fine}_into_{coarse}", not bad and n_c < n_f,
                       f"{n_f} -> {n_c} groups" + (f" SPLIT={bad}" if bad else "")))
    return checks


def load_master() -> list[dict]:
    if not MASTER.exists():
        sys.exit(f"FATAL: {MASTER} not found")
    rows = list(csv.DictReader(MASTER.open()))
    if len(rows) != EXPECTED_NATIVE:
        sys.exit(f"FATAL: expected {EXPECTED_NATIVE} native labels, got {len(rows)}")
    return rows


def build(rows: list[dict]) -> dict:
    resonant = sorted((r for r in rows if r["block"] != "qcd"),
                      key=lambda r: int(r["jet_label"]))
    qcd = sorted((r for r in rows if r["block"] == "qcd"),
                 key=lambda r: int(r["jet_label"]))

    by_canon: dict[str, list[dict]] = {}
    for r in resonant:
        by_canon.setdefault(canonical_topology(r["topology_code"]), []).append(r)

    unknown = set(by_canon) - set(R16_ORDER)
    if unknown:
        sys.exit(f"FATAL: canonical topologies not in frozen order map: {sorted(unknown)}")
    missing = set(R16_ORDER) - set(by_canon)
    if missing:
        sys.exit(f"FATAL: frozen order map has topologies absent from data: {sorted(missing)}")

    tree, r42_id = [], 0
    for r16_id, (canon, name) in enumerate(R16_ORDER.items()):
        members = by_canon[canon]
        tiers = [t for t in TIER_ORDER if any(flavour_tier(m) == t for m in members)]
        children = []
        for t in tiers:
            natives = [m for m in members if flavour_tier(m) == t]
            children.append({
                "r42_id": r42_id,
                "r42_name": f"{name}|{t}" if len(tiers) > 1 else name,
                "flavour_tier": t,
                "rule": {"B": "has_b >= 1",
                         "C": "has_b == 0 and has_c >= 1",
                         "LG": "has_b == 0 and has_c == 0"}[t],
                "natives": {int(m["jet_label"]): m["class_name"] for m in natives},
            })
            r42_id += 1
        tree.append({
            "r16_id": r16_id,
            "r16_name": name,
            "canonical_topology": canon,
            "n_native": len(members),
            "colourless": len(tiers) == 1,
            "reweight_groups": sorted({m["reweight_group_name"] for m in members}),
            "children": children,
        })

    tree.append({
        "r16_id": len(R16_ORDER),
        "r16_name": "QCD_ALL",
        "canonical_topology": "QCD",
        "n_native": len(qcd),
        "colourless": True,
        "reweight_groups": sorted({m["reweight_group_name"] for m in qcd}),
        "children": [{
            "r42_id": r42_id,
            "r42_name": "QCD_ALL",
            "flavour_tier": "n/a",
            "rule": "jet_label in 161..187",
            "natives": {int(m["jet_label"]): m["class_name"] for m in qcd},
        }],
    })
    return {"tree": tree, "rows": rows}


# ------------------------------------------------------------------ checks ---
def run_checks(built: dict) -> list[tuple[str, bool, str]]:
    tree, rows = built["tree"], built["rows"]
    checks: list[tuple[str, bool, str]] = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1 - every native label exactly once
    seen: list[int] = []
    for g in tree:
        for c in g["children"]:
            seen.extend(c["natives"])
    chk("every_native_label_exactly_once",
        sorted(seen) == list(range(EXPECTED_NATIVE)),
        f"n={len(seen)} unique={len(set(seen))} expected={EXPECTED_NATIVE}")

    # 2 - no empty groups
    empty = [g["r16_name"] for g in tree if not g["children"]] + \
            [c["r42_name"] for g in tree for c in g["children"] if not c["natives"]]
    chk("no_empty_groups", not empty, f"empty={empty}")

    # 3 - consecutive ids from zero
    r16_ids = [g["r16_id"] for g in tree]
    r42_ids = [c["r42_id"] for g in tree for c in g["children"]]
    chk("consecutive_ids_from_zero",
        r16_ids == list(range(len(tree))) and r42_ids == list(range(len(r42_ids))),
        f"n_r16={len(r16_ids)} n_r42={len(r42_ids)}")

    # 4 - QCD always QCD_ALL below L188
    qcd_g = tree[-1]
    chk("qcd_always_QCD_ALL_below_L188",
        qcd_g["r16_name"] == "QCD_ALL" and len(qcd_g["children"]) == 1
        and qcd_g["n_native"] == 27,
        f"n_qcd={qcd_g['n_native']} n_children={len(qcd_g['children'])}")

    # 5 - compose(native->R42, R42->R16) == compose(native->R16)
    n2r42 = {n: c["r42_id"] for g in tree for c in g["children"] for n in c["natives"]}
    r42_2_r16 = {c["r42_id"]: g["r16_id"] for g in tree for c in g["children"]}
    n2r16 = {n: g["r16_id"] for g in tree for c in g["children"] for n in c["natives"]}
    chk("compose_native_R42_R16_equals_native_R16",
        all(r42_2_r16[n2r42[n]] == n2r16[n] for n in n2r16),
        f"checked {len(n2r16)} labels")

    # 6 - R42 strictly refines R16
    refines = all(len({n2r16[n] for n in c["natives"]}) == 1
                  for g in tree for c in g["children"])
    strict = len(r42_ids) > len(r16_ids)
    chk("R42_strictly_refines_R16", refines and strict,
        f"refines={refines} strict={strict} ({len(r42_ids)} > {len(r16_ids)})")

    # counts against the independently published vectors
    n_r16_res = len(tree) - 1
    n_r42_res = len(r42_ids) - 1
    chk("r16_resonant_group_count", n_r16_res == EXPECTED_R16_RESONANT,
        f"{n_r16_res} vs expected {EXPECTED_R16_RESONANT}")
    chk("r42_resonant_group_count", n_r42_res == EXPECTED_R42_RESONANT,
        f"{n_r42_res} vs expected {EXPECTED_R42_RESONANT}")
    sizes = [g["n_native"] for g in tree[:-1]]
    chk("r16_group_size_vector_matches_published", sizes == EXPECTED_R16_SIZES,
        f"derived={sizes}")

    # 7 - R16 groups are unions of WHOLE reweighting groups (exactness property)
    # Index by jet_label, NOT by row position. Row order happens to match label
    # order in the current CSV, but relying on that turns a clean assertion
    # failure into an IndexError the moment a label is missing or reordered.
    by_label: dict[int, dict] = {int(r["jet_label"]): r for r in rows}
    rw_of: dict[str, set[int]] = {}
    for r in rows:
        rw_of.setdefault(r["reweight_group_name"], set()).add(int(r["jet_label"]))
    straddle = []
    for g in tree:
        members = {n for c in g["children"] for n in c["natives"]}
        for rw in g["reweight_groups"]:
            if not rw_of[rw] <= members:
                straddle.append((g["r16_name"], rw))
    chk("r16_groups_are_unions_of_whole_reweight_groups", not straddle,
        f"straddling={straddle}")

    # R42 asymmetry: how many R42 groups cut inside a reweighting group
    cut = 0
    for g in tree:
        for c in g["children"]:
            members = set(c["natives"])
            for rw in {by_label[n]["reweight_group_name"] for n in members
                       if n in by_label}:
                if not rw_of[rw] <= members:
                    cut += 1
                    break
    chk("r42_cut_count_recorded", True, f"{cut} of {len(r42_ids)} R42 groups cut inside a reweight group")

    # ---------------------------------------------------- the full 8-rung ladder
    rungs = build_rungs(built)

    chk("ladder_rungs_match_frozen_order", list(rungs) == LADDER,
        f"derived={list(rungs)}")

    for tag, mapping in rungs.items():
        ids = {v[0] for v in mapping.values()}
        n_ok = len(mapping) == EXPECTED_NATIVE
        contiguous = ids == set(range(len(ids)))
        count_ok = len(ids) == EXPECTED_RUNG_GROUPS[tag]
        # one name per id AND one id per name, in both directions
        fwd, rev, bij = {}, {}, True
        for gid, nm in mapping.values():
            if fwd.setdefault(gid, nm) != nm or rev.setdefault(nm, gid) != gid:
                bij = False
        chk(f"rung_{tag}_wellformed", n_ok and contiguous and count_ok and bij,
            f"{len(ids)} groups (expected {EXPECTED_RUNG_GROUPS[tag]}), "
            f"contiguous={contiguous} bijective={bij}")

    checks.extend(check_chain_nesting(rungs))

    # R3_VIS is the whole point of the visible-object rule: it must actually
    # move the neutrino classes. If it does not, the rule was not applied.
    moved = sum(1 for n, r in build_rungs(built)["R3_VIS"].items()
                if n < N_RESONANT
                and int(by_label[n]["n_objects"]) != int(by_label[n]["n_visible_objects"]))
    chk("r3_vis_moves_the_neutrino_classes", moved == 30,
        f"{moved} natives have n_visible != n_objects (expected 30)")

    return checks


def effective_shares(built: dict) -> dict[int, Fraction]:
    """Exact effective stream share per R16 group.

    Valid only because every R16 group is a union of whole reweighting groups
    (check 7). Exact rational arithmetic on class_weights - no uniformity
    assumption anywhere.
    """
    tree, rows = built["tree"], built["rows"]
    weight_of = {}
    for r in rows:
        weight_of[r["reweight_group_name"]] = Fraction(str(r["reweight_group_weight"]))
    out = {}
    for g in tree:
        out[g["r16_id"]] = sum(weight_of[rw] for rw in g["reweight_groups"]) / CLASS_WEIGHTS_SUM
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="run assertions, write nothing")
    args = ap.parse_args()

    rows = load_master()
    built = build(rows)
    checks = run_checks(built)

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failed += not ok

    shares = effective_shares(built)
    total = sum(shares.values())
    ok_total = total == 1
    print(f"  [{'PASS' if ok_total else 'FAIL'}] effective_shares_sum_to_one"
          f"{'':<{max(0, width - 26)}}  sum={float(total):.10f} (exact rational)")
    failed += not ok_total

    print(f"\n{len(checks) + 1 - failed}/{len(checks) + 1} checks passed")
    if failed:
        print("BUILD FAILED - tree not written")
        return 1
    if args.check_only:
        print("--check-only: nothing written")
        return 0

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(built, shares)
    write_audit_csv(built, shares)
    write_rung_label_maps(built)
    print(f"\nwrote {OUT_YAML.relative_to(ROOT)}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MAPS.relative_to(ROOT)}")
    return 0


def write_rung_label_maps(built: dict) -> None:
    """Materialize the label array for EVERY rung on the ladder (invariant I4).

    L188 and L162 are trivial by construction, but "trivial" is not the same as
    "materialized": I4 requires every arm's label array to exist as a frozen
    artifact, never regenerated at training time. This file is the single place
    a trainer reads an arm's map from, and it supersedes the hand-maintained
    per-rung CSVs.
    """
    rows = built["rows"]
    name_of = {int(r["jet_label"]): r["class_name"] for r in rows}
    rungs = build_rungs(built)

    header = ["jet_label", "class_name"]
    for tag in LADDER:
        header += [tag, f"{tag}_name"]
    with OUT_MAPS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for n in range(EXPECTED_NATIVE):
            row = [n, name_of[n]]
            for tag in LADDER:
                gid, gname = rungs[tag][n]
                row += [gid, gname]
            w.writerow(row)


def _sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "UNAVAILABLE"


def write_yaml(built: dict, shares: dict[int, Fraction]) -> None:
    tree = built["tree"]
    L: list[str] = []
    a = L.append
    a("# configs/labelmaps/contraction_tree.v1.yaml")
    a("# GENERATED by scripts/build_contraction_tree.py - do not hand-edit.")
    a("# Regenerate and diff; every group here is derived from")
    a("# hierarchy/01_class_master.csv by the rules stated in `rules:` below.")
    a('schema_version: "contraction-tree/1.0.0"')
    a("family: sophon-label-granularity")
    a("frozen: true")
    a('frozen_on: "2026-08-27"        # PI sign-off, DECISIONS_PENDING items 1 + 4')
    a("")
    a("provenance:")
    a("  native_labels:")
    a("    repo: jet-universe/sophon")
    a("    commit: 9dd6dd6")
    a("    file: data/JetClassII/JetClassII_full.yaml")
    a("    n_labels: 188                    # 161 resonant (0-160) + 27 QCD (161-187)")
    a("    derived_from: hierarchy/01_class_master.csv")
    a(f"    class_master_sha256: {_sha256_of(MASTER)}")
    a("  weights_block:                     # invariant I2 - NEVER edited")
    a("    class_weights_sum: 1.73175")
    a("    sha256: " + WEIGHTS_BLOCK_SHA256)
    a("                                     # verified 2026-08-27 byte-identical across")
    a("                                     # all four configs/arms/*.yaml AND the base")
    a("")
    # Every rule value is a LITERAL BLOCK SCALAR. These are free prose and
    # contain ': ', trailing colons and quotes; as plain scalars they make the
    # artifact unparseable, which is a real defect in a document whose whole
    # job is to be the frozen, machine-readable reference.
    a("rules:                               # the ONLY rules used to build the tree")
    a("  R16_Q1: |")
    a("    partition by [topology, visible_mode]; flavour collapsed everywhere.")
    a("    A quark parton (Q) and a gluon parton (g) are the same object.")
    a("  R42_Q1: |")
    a("    R16 further split by three-tier flavour, priority B then C then LG:")
    a("      B  = has_b >= 1")
    a("      C  = has_b == 0 and has_c >= 1")
    a("      LG = has_b == 0 and has_c == 0")
    a("    Colourless groups (no flavour variation) are not split.")
    a("  R63_Q1: |")
    a("    R42 refined by heavy-flavour parton COUNTS (n_b, n_c), read from the")
    a("    `tokens` column. NOT from has_b/has_c, which are BOOLEANS: label_X_bb")
    a("    has has_b == 1, not 2. Deriving R63 from them would silently collapse")
    a("    multiplicity back into presence.")
    a("  R29_Q1: |")
    a("    R42 with the B and C tiers merged into HF; LG untouched.")
    a("  R3_VIS: |")
    a("    Prong count on VISIBLE objects, so a neutrino does not add a prong.")
    a("    Follows the JetClass-II authors, arXiv:2405.12972 sec. 2 -- '...can")
    a("    also be 3 prongs if an object leaks out of the jet cone or if one of")
    a("    the objects is a neutrino.'")
    a("    R16 group NAMES retain generated-object prong prefixes; the three")
    a("    _NU_ groups are exactly where the two conventions differ.")
    a("  R1_Q1: |")
    a("    native 0-160 -> RESONANT_ALL; 161-187 -> QCD_ALL.")
    a("  rung_numeral_convention: |")
    a("    R-RUNG NUMERALS COUNT RESONANT GROUPS ONLY; QCD_ALL is always one further")
    a("    group on top. So the head width is the numeral PLUS ONE:")
    a("      R63_Q1 -> 64    R42_Q1 -> 43    R29_Q1 -> 30    R16_Q1 -> 17")
    a("      R3_VIS ->  4    R1_Q1  ->  2")
    a("    L-rung numerals count TOTAL groups: L188 -> 188, L162 -> 162.")
    a("    configs/arms/*.yaml already set num_classes correctly (43, 17, ...), but a")
    a("    42-way head on R42_Q1 is a silent misindex rather than a crash, so the")
    a("    convention is stated here rather than left to be inferred from the name.")
    a("  group_id_order: |")
    a("    Primary keys, SEMANTIC and not derivable from native label indices:")
    a("    prong class ascending, then hadronic -> semileptonic -> leptonic")
    a("    within a prong class.")
    a("    TIE-BREAK, which is what actually fixes the ids wherever the primary")
    a("    keys are equal: FIRST NATIVE LABEL APPEARANCE within the parent group.")
    a("    This is load-bearing, not incidental -- ordering R63's (n_b, n_c)")
    a("    sub-cells by ascending counts instead reassigns the ids of 47 of 188")
    a("    labels while leaving the partition identical. Stated here because")
    a("    this artifact, not the generator, is the sign-off document.")
    a("")
    a(f"contraction_order: [{', '.join(LADDER)}]")
    a("")
    a("# The tree is the frozen label HIERARCHY. Which rungs become pretraining")
    a("# arms is a separate compute-gated decision -- see docs/RUN_MATRIX.md.")
    a("# An 8-rung tree does NOT imply 8 pretraining runs.")
    a("rungs:")
    a("  L188: {resonant: identity, qcd: identity, n: 188, role: qcd-granularity-rung,")
    a("         uniquely_retained_axis: QCD_b_vs_c}")
    a("  L162: {parent: L188, op: merge_qcd_all, resonant: identity, n: 162,")
    a("         role: fine-anchor, note: 'pure loss change, ZERO sampling change'}")
    a(f"  R63_Q1: {{parent: L162, n: {EXPECTED_RUNG_GROUPS['R63_Q1']},")
    a("         partition_by: [topology, visible_mode, n_b, n_c],")
    a("         uniquely_retained_axis: heavy_flavour_multiplicity,")
    a("         status: 'hierarchy-only pending the J9 multiplicity pre-test'}")
    a(f"  R42_Q1: {{parent: R63_Q1, n: {sum(len(g['children']) for g in tree)},")
    a("         partition_by: [topology, visible_mode, flavour_tier],")
    a("         uniquely_retained_axis: resonant_b_vs_c}")
    a(f"  R29_Q1: {{parent: R42_Q1, n: {EXPECTED_RUNG_GROUPS['R29_Q1']},")
    a("         partition_by: [topology, visible_mode, hf_vs_lg]}")
    a(f"  R16_Q1: {{parent: R29_Q1, n: {len(tree)},")
    a("         partition_by: [topology, visible_mode]}")
    a(f"  R3_VIS: {{parent: R16_Q1, n: {EXPECTED_RUNG_GROUPS['R3_VIS']},")
    a("         partition_by: [n_visible_objects]}")
    a(f"  R1_Q1: {{parent: R3_VIS, n: {EXPECTED_RUNG_GROUPS['R1_Q1']},")
    a("         partition_by: [resonant_vs_qcd], role: floor-anchor}")
    a("")
    a("tree:")
    for g in tree:
        share = shares[g["r16_id"]]
        a(f"  - r16_id: {g['r16_id']}")
        a(f"    r16_name: {g['r16_name']}")
        a(f"    canonical_topology: {g['canonical_topology']}")
        a(f"    n_native: {g['n_native']}")
        # NOT "colourless": QCD natives span B, C and LG. The accurate
        # statement is that this group is not split by flavour tier.
        a(f"    split_by_flavour_tier: {str(not g['colourless']).lower()}")
        a(f"    effective_share_exact: {float(share):.8f}   # = {share.numerator}/{share.denominator}")
        a(f"    reweight_groups: [{', '.join(g['reweight_groups'])}]")
        a("    children:")
        for c in g["children"]:
            a(f"      - r42_id: {c['r42_id']}")
            a(f"        r42_name: {c['r42_name']}")
            a(f"        flavour_tier: {c['flavour_tier']}")
            a(f"        rule: \"{c['rule']}\"")
            a("        natives:")
            for lab in sorted(c["natives"]):
                a(f"          {lab}: {c['natives'][lab]}")
    a("")
    a("tests:                               # all enforced by tests/test_contraction_tree.py")
    for t in ["every_native_label_exactly_once", "no_empty_groups",
              "consecutive_ids_from_zero", "qcd_always_QCD_ALL_below_L188",
              "compose_native_R42_R16_equals_native_R16", "R42_strictly_refines_R16",
              "weights_block_sha256_identical_across_arms",
              "first_1e6_streamed_jet_ids_hash_identical_across_arms",
              "materialized_not_seed_regenerated"]:
        a(f"  - {t}")
    OUT_YAML.write_text("\n".join(L) + "\n")


def write_audit_csv(built: dict, shares: dict[int, Fraction]) -> None:
    """Four-number audit skeleton.

    raw_stored / selected / unique_sampled are MEASURED quantities and are left
    empty here; they are filled by the G-0 HTTP-range audit. effective_share is
    exact algebra and is emitted now. Emitting empty cells rather than nominal
    guesses is deliberate: a nominal number in a measured column is how the
    repository's retired upper bounds (up to 389.8%) got quoted as fact.
    """
    rows = built["rows"]
    rungs = build_rungs(built)
    by_label = {int(r["jet_label"]): r for r in rows}
    rw_members: dict[str, set[int]] = {}
    weight_of: dict[str, Fraction] = {}
    for r in rows:
        rw = r["reweight_group_name"]
        rw_members.setdefault(rw, set()).add(int(r["jet_label"]))
        weight_of[rw] = Fraction(str(r["reweight_group_weight"]))

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "group_id", "group_name", "n_native",
                    "raw_stored_train", "selected_train", "unique_sampled_train",
                    "repeat_factor", "effective_share_exact", "share_provenance",
                    "test_selected"])
        for tag in LADDER:
            groups: dict[int, tuple[str, set[int]]] = {}
            for n, (gid, gname) in rungs[tag].items():
                groups.setdefault(gid, (gname, set()))[1].add(n)
            for gid in sorted(groups):
                gname, members = groups[gid]
                rws = {by_label[n]["reweight_group_name"] for n in members}
                exact = all(rw_members[rw] <= members for rw in rws)
                share = (sum(weight_of[rw] for rw in rws) / CLASS_WEIGHTS_SUM
                         if exact else None)
                w.writerow([tag, gid, gname, len(members), "", "", "", "",
                            f"{float(share):.8f}" if exact else "",
                            "EXACT_ALGEBRA_ON_CLASS_WEIGHTS" if exact
                            else "MEASURED_REQUIRED_cuts_inside_reweight_group", ""])


if __name__ == "__main__":
    sys.exit(main())
