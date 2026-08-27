#!/usr/bin/env python3
"""Emit review/groupings/ -- one CSV per rung, for PI sign-off.

Six rungs on the approved ladder:

    L188 -> L162 -> R42_Q1 -> R29_Q1 -> R16_Q1 -> R1_Q1

L188, L162, R42_Q1 and R16_Q1 come straight from
configs/labelmaps/rung_label_maps.v1.csv (built and validated by
build_contraction_tree.py). Two rungs are derived here:

    R29_Q1  merges the B and C flavour tiers of R42_Q1 into HF, keeping LG.
            Group ids are assigned by first appearance in R42_Q1 id order, so
            the semantic ordering (prong ascending, then hadronic ->
            semileptonic -> leptonic) is inherited rather than reinvented.
    R1_Q1   RESONANT_ALL (native 0-160) vs QCD_ALL (native 161-187).

Output contract, fixed by the reviewer and not to be varied:
  * exactly three columns: sophon_signal_name, group_id, group_name
  * exactly 188 rows per file, one per native label
  * group_id zero-padded to three digits in EVERY file, including R1_Q1
  * sorted by group_id ascending, then by native label index within the group
  * no extra columns -- supporting numbers live in SUMMARY.md

group_id is quoted in the output so Excel/Sheets cannot silently strip the
zero padding and turn 000 into 0.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

SRC = Path("configs/labelmaps/rung_label_maps.v1.csv")
OUT = Path("review/groupings")

N_NATIVE = 188
N_RESONANT = 161          # native 0-160
EXPECTED_GROUPS = {"L188": 188, "L162": 162, "R42_Q1": 43,
                   "R29_Q1": 30, "R16_Q1": 17, "R1_Q1": 2}


def load() -> list[dict]:
    rows = list(csv.DictReader(SRC.open()))
    rows.sort(key=lambda r: int(r["jet_label"]))
    assert len(rows) == N_NATIVE, f"expected {N_NATIVE} rows, got {len(rows)}"
    assert [int(r["jet_label"]) for r in rows] == list(range(N_NATIVE))
    return rows


def r29_name(r42_name: str) -> str:
    """B and C both become HF; LG is untouched; QCD_ALL passes through."""
    if "|" not in r42_name:
        return r42_name
    base, tier = r42_name.rsplit("|", 1)
    if tier not in ("B", "C", "LG"):
        raise ValueError(f"unrecognised flavour tier {tier!r} in {r42_name!r}")
    return f"{base}|{'HF' if tier in ('B', 'C') else 'LG'}"


def build(rows: list[dict]) -> dict[str, list[tuple[str, int, str]]]:
    """rung -> list of (sophon_signal_name, group_id, group_name), native order."""
    out: dict[str, list[tuple[str, int, str]]] = {}

    out["L188"] = [(r["class_name"], int(r["L188"]), r["class_name"]) for r in rows]

    out["L162"] = [(r["class_name"], int(r["L162"]),
                    r["class_name"] if int(r["jet_label"]) < N_RESONANT else "QCD_ALL")
                   for r in rows]

    out["R42_Q1"] = [(r["class_name"], int(r["R42_Q1"]), r["R42_Q1_name"]) for r in rows]
    out["R16_Q1"] = [(r["class_name"], int(r["R16_Q1"]), r["R16_Q1_name"]) for r in rows]

    # R29: assign ids by first appearance in R42 id order (inherits the ordering)
    order: "OrderedDict[str, int]" = OrderedDict()
    for r in sorted(rows, key=lambda r: int(r["R42_Q1"])):
        nm = r29_name(r["R42_Q1_name"])
        if nm not in order:
            order[nm] = len(order)
    out["R29_Q1"] = [(r["class_name"], order[r29_name(r["R42_Q1_name"])],
                      r29_name(r["R42_Q1_name"])) for r in rows]

    out["R1_Q1"] = [(r["class_name"],
                     0 if int(r["jet_label"]) < N_RESONANT else 1,
                     "RESONANT_ALL" if int(r["jet_label"]) < N_RESONANT else "QCD_ALL")
                    for r in rows]
    return out


def check(rung: str, recs: list[tuple[str, int, str]]) -> None:
    assert len(recs) == N_NATIVE, f"{rung}: {len(recs)} rows, expected {N_NATIVE}"
    ids = {g for _, g, _ in recs}
    assert len(ids) == EXPECTED_GROUPS[rung], \
        f"{rung}: {len(ids)} groups, expected {EXPECTED_GROUPS[rung]}"
    assert ids == set(range(len(ids))), f"{rung}: group ids are not contiguous from 0"
    # one name per id and one id per name
    fwd, rev = {}, {}
    for _, g, nm in recs:
        if fwd.setdefault(g, nm) != nm:
            raise AssertionError(f"{rung}: id {g} maps to both {fwd[g]!r} and {nm!r}")
        if rev.setdefault(nm, g) != g:
            raise AssertionError(f"{rung}: name {nm!r} maps to both {rev[nm]} and {g}")


def main() -> None:
    rows = load()
    built = build(rows)
    native_index = {r["class_name"]: int(r["jet_label"]) for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)

    for rung in ("L188", "L162", "R42_Q1", "R29_Q1", "R16_Q1", "R1_Q1"):
        recs = built[rung]
        check(rung, recs)
        recs = sorted(recs, key=lambda t: (t[1], native_index[t[0]]))
        path = OUT / f"{rung}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
            w.writerow(["sophon_signal_name", "group_id", "group_name"])
            for name, gid, gname in recs:
                w.writerow([name, f"{gid:03d}", gname])
        print(f"{path}  {len(recs)} rows  {len(set(g for _, g, _ in recs))} groups")


if __name__ == "__main__":
    main()
