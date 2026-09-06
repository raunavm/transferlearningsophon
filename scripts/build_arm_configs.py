#!/usr/bin/env python3
"""Emit configs/arms/<ARM>.yaml -- one weaver data config per vocabulary arm.

WHY THIS IS GENERATED AND NOT HAND-WRITTEN
------------------------------------------
Invariant I2 requires the `weights:` block to be BYTE-IDENTICAL across every
arm, because JetClass-II reweighting keys on NATIVE label categories: that is
the only reason regrouping the classifier vocabulary leaves the training stream
bit-identical. Three hand-maintained copies of a 40-line block drift. A
generator that copies the base file's bytes verbatim cannot.

Each arm therefore differs from the base in EXACTLY ONE place: the `labels:`
block. Everything else -- selection, new_variables, preprocess, inputs,
observers, weights -- is passed through unmodified, byte for byte.

`num_classes` is NOT in these files. It reaches the model through weaver's
`-o num_classes K` CLI option, which keeps the data configs differing in the
label map alone (invariant I1).

Source of truth for the maps: configs/labelmaps/rung_label_maps.v1.csv, built
and validated by scripts/build_contraction_tree.py.

Run:  python3 scripts/build_arm_configs.py [--check-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "data" / "JetClassII_base.yaml"
MAPS = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
OUT_DIR = ROOT / "configs" / "arms"

# The arms to emit. NOT every rung on the ladder -- the ladder is the frozen
# hierarchy, the run matrix is a separate compute-gated decision
# (docs/RUN_MATRIX.md).
#
# The HEADLINE CONTRAST is L162 vs R16_Q1, and its fine end is L162, NOT L188:
# L188 differs from L162 only in QCD granularity, which is a separate axis, and
# using it as the fine end would fold the QCD-vocabulary question into the
# headline contrast.
#
#   L188    QCD-granularity ablation; the only arm that keeps QCD b-vs-c.
#           Emitted since 2026-08-22. It is the IDENTITY rung -- resonant and
#           qcd both `identity` in the contraction tree -- so unlike every
#           coarser arm its label map does not depend on the R42_Q1 / R16_Q1
#           group membership that is still unsigned.
#   L162    fine anchor for the headline contrast.
#   R42_Q1  the only interior point. Never cut.
#   R16_Q1  coarse end; flavour erased by construction.
#
# ORDER IS LOAD-BEARING FOR NOTHING, but the FILE BYTES ARE: each arm's
# reweighting sidecar on the PVC is named for its config's md5, so a config
# that changes silently orphans its sidecar and the run guard fails at launch.
# Adding an arm must leave the others byte-identical; tests/test_arm_configs.py
# and the md5 check in the build log are what confirm it did.
ARMS = ["L188", "L162", "R42_Q1", "R16_Q1"]

# The mass-auxiliary twins (the 2x2 at the ends of the ladder, DECISIONS_PENDING
# item 14 addendum 2). <ARM>_MASS is <ARM> plus two labels the hybrid loop reads
# (experiments/MTX/hybrid_mass.py): the log mass-ratio target and its validity
# mask. Same truth_label, same K, same weights: block -- a mass arm and its twin
# differ in the loss term alone. genjet_sdmass is referenced from the labels,
# never listed under observers: tests/test_plumbing.py binds on that.
MASS_ARMS = ["L162", "R16_Q1"]
MASS_LABEL_LINES = [
    "",
    "      ### Mass-auxiliary target (D2 as amended by DECISIONS_PENDING item 3):",
    "      ### log(genjet_sdmass / jet_sdmass), the groomed log mass-ratio, zero",
    "      ### where the jet is unmatched. genjet_sdmass is a hard 0.0f when",
    "      ### unmatched (docs/GROUND_TRUTH.md), so the ratio is guarded and the",
    "      ### loss is masked on mass_valid. Trained by hybrid_mass.py; the",
    "      ### classification head is the SAME K as the twin arm (the arch adds",
    "      ### the mass node itself).",
    "      mass_target: np.log(np.maximum(genjet_sdmass, 1e-6) / jet_sdmass) * (genjet_sdmass > 0)",
    "      mass_valid: (genjet_sdmass > 0)",
]

N_NATIVE = 188


def load_maps() -> dict[str, dict[int, int]]:
    rows = list(csv.DictReader(MAPS.open()))
    if len(rows) != N_NATIVE:
        sys.exit(f"FATAL: {MAPS} has {len(rows)} rows, expected {N_NATIVE}")
    out: dict[str, dict[int, int]] = {}
    for arm in ARMS:
        if arm not in rows[0]:
            sys.exit(f"FATAL: {MAPS} has no column {arm!r}")
        out[arm] = {int(r["jet_label"]): int(r[arm]) for r in rows}
    return out


def condition_for(natives: list[int]) -> str:
    """Compact boolean over jet_label selecting exactly `natives`.

    Contiguous runs collapse to a half-open range, singletons stay equalities.
    Correctness does not rest on this collapsing being right -- the emitted
    expression is evaluated over jet_label = 0..187 and compared against the
    label map, both here and in tests/test_arm_configs.py.
    """
    runs: list[tuple[int, int]] = []
    for n in sorted(natives):
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    parts = [f"(jet_label == {a})" if a == b
             else f"((jet_label >= {a}) & (jet_label < {b + 1}))"
             for a, b in runs]
    return " | ".join(parts)


def truth_label_expr(mapping: dict[int, int]) -> str:
    """sum over groups of  gid * [jet_label in group].

    Group 0 is omitted: it contributes 0 * (...) = 0. Exactly one indicator is
    true for any jet, so the sum is that jet's group id.
    """
    groups: dict[int, list[int]] = {}
    for native, gid in mapping.items():
        groups.setdefault(gid, []).append(native)
    terms = [f"{gid} * ({condition_for(groups[gid])})"
             for gid in sorted(groups) if gid != 0]
    return " + ".join(terms)


def evaluate(expr: str) -> dict[int, int]:
    """Evaluate the emitted expression over jet_label = 0..187.

    This is the check that matters. However the expression is written, it must
    reproduce the label map exactly.
    """
    import numpy as np
    jet_label = np.arange(N_NATIVE)
    got = eval(expr, {"__builtins__": {}}, {"jet_label": jet_label, "np": np})
    return {int(n): int(v) for n, v in enumerate(np.asarray(got))}


def labels_block(arm: str, mapping: dict[int, int], names: dict[int, str],
                 mass: bool = False) -> str:
    n_groups = len(set(mapping.values()))
    lines = [
        "labels:",
        "   ### GENERATED by scripts/build_arm_configs.py -- do not hand-edit.",
        f"   ### Arm {arm}: {N_NATIVE} native labels -> {n_groups} groups.",
        "   ###",
        "   ### The map is MATERIALIZED here, never regenerated from a seed",
        "   ### (invariant I4). Source: configs/labelmaps/rung_label_maps.v1.csv.",
        "   ###",
        f"   ### Launch this arm with:  -o num_classes {n_groups}",
        "   type: custom",
        "   value:",
        f"      truth_label: {truth_label_expr(mapping)}",
        *(MASS_LABEL_LINES if mass else []),
        "",
        "   ### group_id -> group_name, for readers:",
    ]
    sizes: dict[int, int] = {}
    for gid in mapping.values():
        sizes[gid] = sizes.get(gid, 0) + 1
    for gid in sorted(sizes):
        lines.append(f"   ###   {gid:3d}  {names[gid]}  ({sizes[gid]} native)")
    return "\n".join(lines) + "\n\n"


def build_one(base_text: str, arm: str, mapping: dict[int, int],
              names: dict[int, str], mass: bool = False) -> str:
    m_lab = re.search(r"^labels:", base_text, re.M)
    m_obs = re.search(r"^observers:", base_text, re.M)
    if not m_lab or not m_obs or m_obs.start() < m_lab.start():
        sys.exit("FATAL: base config does not have labels: followed by observers:")
    header = (
        f"# configs/arms/{arm}.yaml\n"
        f"# GENERATED by scripts/build_arm_configs.py -- do not hand-edit.\n"
        f"#\n"
        f"# Differs from configs/data/JetClassII_base.yaml in the `labels:` block\n"
        f"# and NOTHING ELSE. The `weights:` block is copied byte for byte, which\n"
        f"# is what makes the training stream bit-identical across arms (I2).\n"
        f"# NEVER edit the weights: block. Regenerate this file instead.\n"
        f"#\n"
    )
    return (header + base_text[:m_lab.start()]
            + labels_block(arm, mapping, names, mass=mass)
            + base_text[m_obs.start():])


def weights_sha256(text: str) -> str:
    m = re.search(r"^weights:", text, re.M)
    if not m:
        sys.exit("FATAL: no weights: block")
    return hashlib.sha256(text[m.start():].encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not BASE.exists():
        sys.exit(f"FATAL: {BASE} not found")
    base_text = BASE.read_text()
    maps = load_maps()

    # The reader-facing comment table must name the RUNG's groups, not the
    # native class that happens to sort first inside each group. Using
    # `class_name` made R16_Q1 group 0 read "label_X_bb" when the group is
    # actually 2P_HAD_2PARTON -- a name that describes one of its 20-odd
    # members rather than the group. The machine-read truth_label expression
    # was never affected; the comment is the human record of what the arm
    # MEANS, and it was misleading.
    rows = list(csv.DictReader(MAPS.open()))
    group_names = {
        arm: {int(r[arm]): r[f"{arm}_name"] for r in rows} for arm in ARMS
    }

    base_sha = weights_sha256(base_text)
    built, failed = {}, 0
    # (output name, map arm, mass?) -- the MASS twins reuse their twin's map.
    todo = [(arm, arm, False) for arm in ARMS] + \
           [(f"{arm}_MASS", arm, True) for arm in MASS_ARMS]
    for arm, src, mass in todo:
        text = build_one(base_text, arm, maps[src], group_names[src], mass=mass)
        built[arm] = text

        expr = re.search(r"truth_label: (.*)", text).group(1)
        got = evaluate(expr)
        ok_map = got == maps[src]
        ok_wts = weights_sha256(text) == base_sha
        n_groups = len(set(maps[src].values()))
        ok_dense = set(got.values()) == set(range(n_groups))

        for label, ok in (("expression_reproduces_label_map", ok_map),
                          ("weights_block_sha256_matches_base", ok_wts),
                          ("group_ids_dense_from_zero", ok_dense)):
            print(f"  [{'PASS' if ok else 'FAIL'}] {arm:8s} {label}")
            failed += not ok
        print(f"         {arm:8s} num_classes = {n_groups}")

    # cross-arm: the whole point of I2
    shas = {a: weights_sha256(t) for a, t in built.items()}
    ok = len(set(shas.values())) == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] weights_block_identical_across_all_arms  "
          f"{list(shas.values())[0][:16]}")
    failed += not ok

    if failed:
        print(f"\nBUILD FAILED - {failed} check(s) failed, nothing written")
        return 1
    if args.check_only:
        print("\n--check-only: nothing written")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm, text in built.items():
        p = OUT_DIR / f"{arm}.yaml"
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
