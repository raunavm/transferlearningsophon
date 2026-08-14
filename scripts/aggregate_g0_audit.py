#!/usr/bin/env python3
"""Aggregate the G-0 per-native-label audit into the four-number group audit.

The cluster job (`experiments/G0/k8s/job-g0-audit-raunav.yaml`) measures per
NATIVE jet_label because the R42/R16 label map is not on GitHub. This script
does the group aggregation locally, where the map lives.

Inputs
    <g0_audit.json>                             from the cluster job
    configs/labelmaps/rung_label_maps.v1.csv    the locked label map

Outputs
    configs/labelmaps/stream_share_audit.v1.csv   measured columns filled in
    docs/G0_FINDINGS.md                           the gate readout

Run:  python3 scripts/aggregate_g0_audit.py path/to/g0_audit.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAPS = ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
AUDIT = ROOT / "configs" / "labelmaps" / "stream_share_audit.v1.csv"
FINDINGS = ROOT / "docs" / "G0_FINDINGS.md"

# Published shard-measured selection efficiencies, to be checked against the
# full-audit numbers. These were sampled-and-scaled; ours are complete.
PUBLISHED_EFF = {"Res2P": 85.8, "Res34P": 90.8, "QCD": 51.8}


def load_maps() -> list[dict]:
    if not MAPS.exists():
        sys.exit(f"FATAL: {MAPS} not found - run scripts/build_contraction_tree.py first")
    return list(csv.DictReader(MAPS.open()))


def block_of(jet_label: int, n_native_2p: int = 15) -> str:
    """res2p = 0..14, res34p = 15..160, qcd = 161..187."""
    if jet_label >= 161:
        return "QCD"
    return "Res2P" if jet_label < n_native_2p else "Res34P"


def main(path: str) -> int:
    data = json.loads(pathlib.Path(path).read_text())
    per_label = {int(k): v for k, v in data["per_label"].items()}
    maps = load_maps()

    # ---- block-level selection efficiency, measured vs published -----------
    blk_raw, blk_sel = defaultdict(int), defaultdict(int)
    for lab, v in per_label.items():
        b = block_of(lab)
        blk_raw[b] += v["raw"]
        blk_sel[b] += v["selected"]

    print("=== selection efficiency: FULL audit vs published sampled-and-scaled ===")
    eff_rows = []
    for b in ("Res2P", "Res34P", "QCD"):
        if not blk_raw[b]:
            continue
        eff = blk_sel[b] / blk_raw[b] * 100
        pub = PUBLISHED_EFF[b]
        eff_rows.append((b, blk_raw[b], blk_sel[b], eff, pub, eff - pub))
        print(f"  {b:<8} raw={blk_raw[b]:>12,}  sel={blk_sel[b]:>12,}  "
              f"eff={eff:6.2f}%   published={pub:5.1f}%   delta={eff-pub:+.2f}pp")

    tot_raw = sum(blk_raw.values())
    tot_sel = sum(blk_sel.values())
    print(f"  {'TOTAL':<8} raw={tot_raw:>12,}  sel={tot_sel:>12,}  "
          f"eff={tot_sel/tot_raw*100:6.2f}%")

    # ---- aggregate to R42 / R16 -------------------------------------------
    agg: dict[tuple[str, int], dict] = {}
    names: dict[tuple[str, int], str] = {}
    for row in maps:
        lab = int(row["jet_label"])
        v = per_label.get(lab)
        if v is None:
            continue
        for rung, gid_key, name_key in (("R16_Q1", "R16_Q1", "R16_Q1_name"),
                                        ("R42_Q1", "R42_Q1", "R42_Q1_name")):
            gid = int(row[gid_key])
            k = (rung, gid)
            names[k] = row[name_key]
            a = agg.setdefault(k, {"raw": 0, "selected": 0, "matched": 0, "n_native": 0})
            a["raw"] += v["raw"]
            a["selected"] += v["selected"]
            a["matched"] += v["selected_and_matched"]
            a["n_native"] += 1

    # ---- rewrite the audit CSV, filling measured columns -------------------
    existing = {(r["rung"], int(r["group_id"])): r for r in csv.DictReader(AUDIT.open())}
    with AUDIT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "group_id", "group_name", "n_native",
                    "raw_stored_train", "selected_train", "unique_sampled_train",
                    "repeat_factor", "effective_share_exact", "share_provenance",
                    "test_selected", "match_fraction"])
        for rung in ("R16_Q1", "R42_Q1"):
            for (r, gid) in sorted((k for k in agg if k[0] == rung), key=lambda x: x[1]):
                a = agg[(r, gid)]
                old = existing.get((r, gid), {})
                mf = a["matched"] / a["selected"] if a["selected"] else ""
                w.writerow([r, gid, names[(r, gid)], a["n_native"],
                            a["raw"], a["selected"], "", "",
                            old.get("effective_share_exact", ""),
                            old.get("share_provenance", ""),
                            "", f"{mf:.6f}" if mf != "" else ""])
    print(f"\nwrote {AUDIT.relative_to(ROOT)} with measured columns filled")

    # ---- gate readouts ----------------------------------------------------
    print("\n=== D2 gate: log(genjet_sdmass/jet_sdmass) per class ===")
    lrs = [(lab, v["log_ratio_mean"], v["log_ratio_std"], v["log_ratio_n"])
           for lab, v in sorted(per_label.items()) if v["log_ratio_mean"] is not None]
    if lrs:
        means = [x[1] for x in lrs]
        print(f"  classes with a usable profile: {len(lrs)}/188")
        print(f"  mean log-ratio  min={min(means):+.4f}  max={max(means):+.4f}")
        worst = sorted(lrs, key=lambda x: x[1])[:3]
        print("  most negative (mass most over-measured vs generator):")
        for lab, m, s, n in worst:
            print(f"    label {lab:3d}  mean={m:+.4f}  std={s:.4f}  n={n:,}")

    print("\n=== D2 gate: generator-match fraction ===")
    mfs = [(lab, v["match_fraction"]) for lab, v in sorted(per_label.items())
           if v["match_fraction"] is not None]
    if mfs:
        vals = [m for _, m in mfs]
        print(f"  overall (selected): {sum(v['selected_and_matched'] for v in per_label.values())/max(tot_sel,1)*100:.3f}%")
        print(f"  per-class min={min(vals)*100:.2f}%  max={max(vals)*100:.2f}%")
        low = sorted(mfs, key=lambda x: x[1])[:5]
        print("  lowest-match classes (mass head loses these):")
        for lab, m in low:
            print(f"    label {lab:3d}  {m*100:6.2f}%")

    FINDINGS.write_text(_findings_md(eff_rows, tot_raw, tot_sel, lrs, mfs))
    print(f"\nwrote {FINDINGS.relative_to(ROOT)}")
    return 0


def _findings_md(eff_rows, tot_raw, tot_sel, lrs, mfs) -> str:
    L = ["# G-0 findings", "",
         "Full audit over all 2,010 parquet files. Unlike the previous per-class",
         "numbers, these are **not** sampled-and-scaled — every file was read.", "",
         "## Selection efficiency", "",
         "| block | raw | selected | measured | published | delta |",
         "|---|---|---|---|---|---|"]
    for b, raw, sel, eff, pub, d in eff_rows:
        L.append(f"| {b} | {raw:,} | {sel:,} | **{eff:.2f}%** | {pub:.1f}% | {d:+.2f} pp |")
    L += [f"| **total** | **{tot_raw:,}** | **{tot_sel:,}** | "
          f"**{tot_sel/tot_raw*100:.2f}%** | — | — |", ""]
    if lrs:
        means = [x[1] for x in lrs]
        L += ["## D2 — mass-target profile", "",
              f"- classes with a usable `log(genjet_sdmass/jet_sdmass)` profile: "
              f"**{len(lrs)}/188**",
              f"- mean log-ratio spans **{min(means):+.4f}** to **{max(means):+.4f}**", ""]
    if mfs:
        vals = [m for _, m in mfs]
        L += ["## D2 — generator-match fraction", "",
              f"- per-class match fraction spans **{min(vals)*100:.2f}%** to "
              f"**{max(vals)*100:.2f}%**",
              "- the mask (`genjet_sdmass > 0`) is **not class-neutral**; the "
              "surviving fraction must be reported per class", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
