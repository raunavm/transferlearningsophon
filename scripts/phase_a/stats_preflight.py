#!/usr/bin/env python3
"""Phase A §6-A.5 (a)+(c): Kish table + weight-agreement report (G0 artifacts).

One columnar pass over the JetClass-II release — (jet_pt, jet_sdmass, jet_label)
only — accumulates a per-partition count cube C[label(188), pt_bin, msd_bin]
after the frozen training selection. Everything derives exactly from the cube
(all jets in a bin share a weight, so Σw and Σw² are exact bin sums):

  (a) kish_table: event-level N_eff = (Σw)²/Σw² per (partition × class-cell),
      weights = frozen global label-blind 1/ρ̂(pT, mSD) (configs/weights/
      global_pt_msd.yaml). No per-event UIDs exist in the release (ntupler
      defines none), so each jet is its own resampling unit — recorded in the
      output; the G corpus carries true UIDs.
  (c) weight-agreement: global weights vs the Sophon 30-group construction
      (class_weights[g]/ρ̂_g from the frozen E2 base config) — jet-weighted
      distribution of log(w_global/w_30); 95th pct of |log ratio| < 0.05 ⇒
      "equivalent", else contingency C1 arms (§17.2).

Also verifies the frozen eval-split manifest against the PVC (existence, jet
counts, sorted-file-list sha256) — a G0-blocking check.

Run (in-cluster):
  python3 scripts/phase_a/stats_preflight.py --jc2-dir /jc2/jet_data \
      --out /data/results/phase_a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

KISH_FPR4_FLOOR = 4e6            # §16.2: FPR 1e-4 reportable iff N_eff >= 4e6
EQUIV_THRESHOLD = 0.05           # §6-A.5(c): 95th pct |log ratio| < 0.05


def load_frozen():
    import yaml
    wcfg = yaml.safe_load((REPO / "configs/weights/global_pt_msd.yaml").read_text())
    manifest = json.loads((REPO / "manifests/release_eval_split.json").read_text())
    base = yaml.safe_load((REPO / "experiments/E2/data/JetClassII_full_base.yaml").read_text())
    maps = json.loads((REPO / "experiments/E2/labels/label_maps.json").read_text())
    return wcfg, manifest, base, maps


def partition_of(fname, ranges):
    m = re.match(r"(Res2P|Res34P|QCD)_(\d{4})\.parquet$", fname)
    fam, idx = m.group(1), int(m.group(2))
    for part in ("train", "val", "test"):
        lo, hi = ranges[fam][part]
        if lo <= idx <= hi:
            return part
    raise ValueError(f"{fname}: index outside all partitions")


def read_file(path, pt_edges, msd_edges):
    """Return (counts[188, npt, nmsd] int64, n_total, n_selected) for one file."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["jet_pt", "jet_sdmass", "jet_label"])
    pt = t["jet_pt"].to_numpy(zero_copy_only=False)
    msd = t["jet_sdmass"].to_numpy(zero_copy_only=False)
    lab = t["jet_label"].to_numpy(zero_copy_only=False).astype(np.int64)
    sel = (pt > 200) & (pt < 2500) & (msd > 20) & (msd < 500)  # frozen selection
    pt, msd, lab = pt[sel], msd[sel], lab[sel]
    ip = np.digitize(pt, pt_edges) - 1
    im = np.digitize(msd, msd_edges) - 1
    npt, nmsd = len(pt_edges) - 1, len(msd_edges) - 1
    ok = (ip >= 0) & (ip < npt) & (im >= 0) & (im < nmsd)
    flat = (lab[ok] * npt + ip[ok]) * nmsd + im[ok]
    counts = np.bincount(flat, minlength=188 * npt * nmsd).reshape(188, npt, nmsd)
    return counts, len(sel), int(sel.sum())


def kish(counts_cells, w):
    """counts_cells: (188, npt, nmsd) selected-jet counts for a cell's labels.
    All jets in a (pt,msd) bin share w[bin] ⇒ exact Σw, Σw² from bin sums."""
    n_by_bin = counts_cells.sum(axis=0)          # (npt, nmsd)
    sw = float((n_by_bin * w).sum())
    sw2 = float((n_by_bin * w ** 2).sum())
    n = int(n_by_bin.sum())
    neff = (sw ** 2 / sw2) if sw2 > 0 else 0.0
    return n, sw, neff


def wpct(values, weights, qs):
    """Weighted percentiles (values sorted ascending)."""
    o = np.argsort(values)
    v, cw = values[o], np.cumsum(weights[o])
    return [float(np.interp(q / 100 * cw[-1], cw, v)) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jc2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    jd, out = Path(args.jc2_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wcfg, manifest, base, maps = load_frozen()
    pt_edges = np.asarray(wcfg["bin_edges"]["jet_pt"], float)
    msd_edges = np.asarray(wcfg["bin_edges"]["jet_sdmass"], float)
    ranges = {f: {p: tuple(v[p]) for p in ("train", "val", "test")}
              for f, v in manifest["family_index_ranges"].items()}
    g30 = np.asarray(maps["g30"]["map"], np.int64)
    g10 = np.asarray(maps["g10sem"]["map"], np.int64)
    cw30 = np.asarray(base["weights"]["class_weights"], float)
    assert len(cw30) == 30 and g30.max() == 29

    # ---- enumerate + verify the frozen file lists ----
    all_files, missing = {}, []
    for fam, spec in manifest["family_index_ranges"].items():
        for i in range(spec["total"]):
            f = f"{fam}_{i:04d}.parquet"
            all_files[f] = partition_of(f, ranges)
            if not (jd / f).exists():
                missing.append(f)
    realized = {p: sorted(f for f, pp in all_files.items() if pp == p and f not in missing)
                for p in ("train", "val", "test")}
    list_hash = {p: hashlib.sha256("\n".join(v).encode()).hexdigest() for p, v in realized.items()}
    hash_match = {p: list_hash[p] == manifest["file_list_sha256"][p] for p in list_hash}
    print(f"files: {len(all_files)}  missing: {len(missing)}  hash_match: {hash_match}")
    if missing[:5]:
        print("  e.g. missing:", missing[:5])

    # ---- accumulate the count cube per partition ----
    cubes = {p: np.zeros((188, len(pt_edges) - 1, len(msd_edges) - 1), np.int64)
             for p in ("train", "val", "test")}
    tot = {p: [0, 0] for p in cubes}  # [n_total, n_selected]

    def work(fname):
        c, n, ns = read_file(str(jd / fname), pt_edges, msd_edges)
        return fname, c, n, ns

    todo = [f for f in all_files if f not in missing]
    with ThreadPoolExecutor(args.threads) as ex:
        for k, (fname, c, n, ns) in enumerate(ex.map(work, todo)):
            p = all_files[fname]
            cubes[p] += c
            tot[p][0] += n
            tot[p][1] += ns
            if (k + 1) % 200 == 0:
                print(f"  [{k + 1}/{len(todo)}] …")

    np.savez_compressed(out / "count_cube.npz",
                        pt_edges=pt_edges, msd_edges=msd_edges,
                        **{f"cube_{p}": cubes[p] for p in cubes})

    # ---- global weights (§9.4): 1/rho on the TRAIN partition, mean-1, clipped ----
    hist_tot = cubes["train"].sum(axis=0).astype(float)
    n_train = hist_tot.sum()
    w = np.zeros_like(hist_tot)
    nz = hist_tot > 0
    w[nz] = n_train / hist_tot[nz]              # ∝ 1/ρ̂, mean 1 over train jets... up to bin count
    w /= (hist_tot * w).sum() / n_train          # exact mean-1 normalization
    lo, hi = float(wcfg["clip"]["w_min"]), float(wcfg["clip"]["w_max"])
    clipped_frac = float(hist_tot[(w < lo) | (w > hi)].sum() / n_train)
    w = np.clip(w, lo, hi) * nz
    np.savez_compressed(out / "weights_global.npz", w=w, pt_edges=pt_edges,
                        msd_edges=msd_edges, clip=[lo, hi], clipped_jet_frac=clipped_frac)

    # ---- (a) Kish table ----
    rows = []
    cells = [("all", np.arange(188)), ("signal_agg", np.where(g10 <= 8)[0]),
             ("QCD", np.where(g10 == 9)[0])]
    cells += [(f"g10c{c}", np.where(g10 == c)[0]) for c in range(10)]
    cells += [(f"k188_{k}", np.array([k])) for k in range(188)]
    for part in ("train", "val", "test"):
        for cell, labs in cells:
            n, sw, neff = kish(cubes[part][labs], w)
            rows.append({"partition": part, "cell": cell, "mu": 50,
                         "n_selected": n, "sum_w": sw, "n_eff": neff,
                         "fpr_1e-4_ok": bool(neff >= KISH_FPR4_FLOOR)})
    import pandas as pd
    kt = pd.DataFrame(rows)
    kt.to_parquet(out / "kish_table.parquet", index=False)
    kt.to_csv(out / "kish_table.csv", index=False)

    # ---- (c) weight-agreement: global vs Sophon 30-group ----
    # per-group flat construction on the same grid: w30(g,bin) ∝ cw30[g]/ρ̂_g
    cube30 = np.zeros((30,) + hist_tot.shape)
    for k in range(188):
        cube30[g30[k]] += cubes["train"][k]
    w30 = np.zeros_like(cube30)
    for g in range(30):
        gnz = cube30[g] > 0
        w30[g][gnz] = cw30[g] / cube30[g][gnz]
    w30 /= (cube30 * w30).sum() / n_train        # mean-1 over train jets
    # jet-weighted log-ratio distribution over (group, bin) strata
    ratios, jn = [], []
    for g in range(30):
        gnz = (cube30[g] > 0) & (w > 0)
        ratios.append(np.log(w[gnz] / w30[g][gnz]))
        jn.append(cube30[g][gnz])
    ratios, jn = np.concatenate(ratios), np.concatenate(jn)
    p5, p25, p50, p75, p95 = wpct(ratios, jn, [5, 25, 50, 75, 95])
    abs95 = wpct(np.abs(ratios), jn, [95])[0]
    verdict = "equivalent" if abs95 < EQUIV_THRESHOLD else "NOT equivalent — contingency C1 ARMED"
    wa = {"log_ratio_pct": {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
          "abs_log_ratio_p95": abs95, "threshold": EQUIV_THRESHOLD, "verdict": verdict,
          "construction": "w_global=1/rho(pt,msd) vs w_30=cw[g]/rho_g(pt,msd), both mean-1 on train",
          "clipped_jet_frac_global": clipped_frac}
    (out / "weight_agreement.json").write_text(json.dumps(wa, indent=2))

    # ---- report ----
    qcd_test = kt[(kt.partition == "test") & (kt.cell == "QCD")].iloc[0]
    qcd_val = kt[(kt.partition == "val") & (kt.cell == "QCD")].iloc[0]
    rep = [
        "# Phase A statistics preflight — A.5(a)+(c)  (G0 evidence)", "",
        f"- files verified: {len(todo)}/{len(all_files)} present; missing={len(missing)}; "
        f"file-list hashes match manifest: {hash_match}",
        f"- selected jets: " + ", ".join(f"{p}: {tot[p][1]:,}/{tot[p][0]:,}" for p in tot),
        f"- global weights: grid {len(pt_edges)-1}x{len(msd_edges)-1}, clip [{lo},{hi}], "
        f"clipped jet fraction {clipped_frac:.3%}",
        "", "## Kish gate (§16.2): FPR 1e-4 needs N_eff >= 4e6", "",
        f"- eval(test) QCD cell: N_eff = {qcd_test.n_eff:,.0f}  -> FPR 1e-4 "
        f"{'REPORTABLE' if qcd_test['fpr_1e-4_ok'] else 'NOT reportable (use 1e-3)'}",
        f"- val QCD cell:        N_eff = {qcd_val.n_eff:,.0f}",
        "- full table: kish_table.parquet / .csv (all 188 classes + aggregates x partitions)",
        "- NOTE: release has no event UIDs -> jet-level Kish (no multi-jet cluster "
        "correction possible); G corpus will carry true UIDs.",
        "", "## Weight agreement (§6-A.5c)", "",
        f"- 95th pct |log(w_global/w_30)| = {abs95:.4f}  (threshold {EQUIV_THRESHOLD})",
        f"- VERDICT: **{verdict}**",
        f"- log-ratio percentiles: 5%={p5:+.4f} 25%={p25:+.4f} 50%={p50:+.4f} "
        f"75%={p75:+.4f} 95%={p95:+.4f}", "",
    ]
    (out / "phase_a_stats_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
