#!/usr/bin/env python3
"""The D8 control's cost table, re-derived from the committed maps.

WHY THIS EXISTS. DECISIONS_PENDING item 24 carries a table of K, per-group
stream-share ratio, exp(H) and ARI-against-R16_Q1 for the count-matched and
share-matched controls, and the PI signed option C on the ARI column -- it is
the item's own statement of what C costs. NOTHING IN THE REPOSITORY COMPUTED IT.
`grep -l adjusted_rand` over every .py returns nothing, so the column was formed
by hand, which docs/REVIEW.md forbids: "any claim 'X equals N' must be
reproducible by a committed script".

It was also wrong. Re-derivation (2026-09-12) against the committed map:

    quantity              published   re-derived
    count-matched ARI         0.254       0.2536     reproduces
    share-matched ARI d1      0.347       0.4030     does NOT
    share-matched ARI d2      0.349       0.3815     does NOT
    share-matched ARI d3      0.393       0.3976     close

THE COUNT-MATCHED VALUE REPRODUCING IS WHAT MAKES THIS DECISIVE. It fixes the
convention -- standard ARI over all 188 natives -- so the share-matched
disagreement is a real error and not a different definition. The item states the
cost of option C as the rise 0.254 -> 0.347, i.e. +0.093. The true rise is
0.2536 -> 0.4030, **+0.1493, or 1.61x what was recorded**, in the direction that
makes the control LESS scrambled than the decision said.

C IS STILL THE RIGHT CHOICE and this does not reopen it. The count-matched
control had a per-group share ratio of 23.78:1 against the target's 200:1, so it
was a measurably easier task -- and a control that differs from its target in
task difficulty as well as in membership cannot support the falsification rule,
whatever its ARI. Matching the quantity the loss actually sees is what I1
requires. What changes is the honesty of the cost statement, and the reading it
supports: at ARI 0.40 the control agrees with R16_Q1 considerably more than the
item implied, and ALL of its disagreement lives in res34p -- which is exactly
why the two D8 probe axes added on 2026-09-12 had to be placed there.

Run:  python3 scripts/rand_control_stats.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import pathlib
import subprocess
import sys
import tempfile
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "configs" / "labelmaps" / "rand_label_map.v1.csv"


def adjusted_rand(a: list[int], b: list[int]) -> float:
    """ARI from the contingency table. No sklearn: the cluster image does not
    carry it, and a metric that cannot be recomputed where the data lives is
    how a number ends up hand-formed in the first place."""
    if len(a) != len(b):
        raise SystemExit(f"FATAL: length mismatch {len(a)} vs {len(b)}")
    tab: dict[tuple[int, int], int] = {}
    for x, y in zip(a, b):
        tab[(x, y)] = tab.get((x, y), 0) + 1
    ra: dict[int, int] = {}
    rb: dict[int, int] = {}
    for (x, y), n in tab.items():
        ra[x] = ra.get(x, 0) + n
        rb[y] = rb.get(y, 0) + n
    n = len(a)
    s_ij = sum(comb(v, 2) for v in tab.values())
    s_a = sum(comb(v, 2) for v in ra.values())
    s_b = sum(comb(v, 2) for v in rb.values())
    expected = s_a * s_b / comb(n, 2)
    maximum = (s_a + s_b) / 2.0
    if maximum == expected:
        raise SystemExit("FATAL: degenerate ARI denominator")
    return (s_ij - expected) / (maximum - expected)


def _brc():
    spec = importlib.util.spec_from_file_location(
        "build_rand_control", ROOT / "scripts" / "build_rand_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def group_share_stats(rows: list[dict], col: str) -> tuple[float, float]:
    """(max/min stream-share ratio, exp(Shannon entropy)) over the groups.

    Shares come from build_rand_control.exact_share_units(), the same exact
    integer arithmetic the solver matches on, so "matched" here means bit-exact
    rather than within a tolerance.
    """
    units = _brc().exact_share_units()
    per_group: dict[int, int] = {}
    for r in rows:
        g = int(r[col])
        per_group[g] = per_group.get(g, 0) + units[int(r["jet_label"])]
    tot = sum(per_group.values())
    shares = [v / tot for v in per_group.values()]
    h = -sum(s * math.log(s) for s in shares if s > 0)
    return max(shares) / min(shares), math.exp(h)


def regenerate(match: str) -> list[dict]:
    """Rebuild a control from the generator so the table cannot drift from it."""
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "m.csv"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_rand_control.py"),
             "--match", match, "--out", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"FATAL: generator failed for --match {match}:\n"
                             f"{r.stderr[-2000:]}")
        with out.open() as f:
            return list(csv.DictReader(f))


def table() -> list[dict]:
    share = regenerate("share")
    count = regenerate("count")

    with COMMITTED.open() as f:
        committed = list(csv.DictReader(f))
    if [r["RAND_d1"] for r in share] != [r["RAND_d1"] for r in committed]:
        raise SystemExit("FATAL: the share-matched regeneration does not match "
                         "the committed map. The table below would describe a "
                         "control that is not the one being trained.")

    out = []
    target = [int(r["R16_Q1"]) for r in committed]
    ratio, eh = group_share_stats(committed, "R16_Q1")
    out.append({"vocabulary": "R16_Q1 (target)", "match": "-",
                "k": len({*target}), "share_ratio": ratio, "exp_H": eh,
                "ari_vs_r16q1": 1.0})
    for match, rows in (("count", count), ("share", share)):
        for col in ("RAND_d1", "RAND_d2", "RAND_d3"):
            v = [int(r[col]) for r in rows]
            ratio, eh = group_share_stats(rows, col)
            out.append({"vocabulary": col, "match": match, "k": len({*v}),
                        "share_ratio": ratio, "exp_H": eh,
                        "ari_vs_r16q1": adjusted_rand(
                            [int(r["R16_Q1"]) for r in rows], v)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows = table()
    print(f"{'vocabulary':22s} {'match':6s} {'K':>3s} {'share max/min':>14s} "
          f"{'exp(H)':>8s} {'ARI vs R16_Q1':>14s}")
    for r in rows:
        print(f"{r['vocabulary']:22s} {r['match']:6s} {r['k']:3d} "
              f"{r['share_ratio']:13.2f}: {r['exp_H']:8.2f} "
              f"{r['ari_vs_r16q1']:14.4f}")

    c = next(r for r in rows if r["match"] == "count" and r["vocabulary"] == "RAND_d1")
    s = next(r for r in rows if r["match"] == "share" and r["vocabulary"] == "RAND_d1")
    rise = s["ari_vs_r16q1"] - c["ari_vs_r16q1"]
    print(f"\nTHE COST OF OPTION C, which is what item 24 was signed on:")
    print(f"  ARI rises {c['ari_vs_r16q1']:.4f} -> {s['ari_vs_r16q1']:.4f}, "
          f"a rise of {rise:+.4f}.")
    print(f"  DECISIONS_PENDING item 24 records this rise as +0.093 "
          f"({rise / 0.093:.2f}x understated).")
    print(f"  The share-matched control is LESS scrambled than the decision "
          f"said. C still stands -- the count-matched control's share ratio of "
          f"{c['share_ratio']:.2f}:1 against the target's "
          f"{rows[0]['share_ratio']:.2f}:1 made it an easier task, which is "
          f"fatal to the falsification rule regardless of ARI.")

    if a.out:
        import json
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
