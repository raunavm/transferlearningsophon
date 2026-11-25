#!/usr/bin/env python3
"""Values for the writer and the figures agent (PI 2026-11-25, ledger 33 and the last cold readers), from committed data:
  gg_vs_qq     the X->gg vs X->qq linear-probe AUC per network and per arm (bootstrap/l3_bs3/probe_jc2bal_pairs_gg_vs_qq.json,
               bs3-go8), with each random grouping's status for the pair from the label maps (merged / apart), for figA2
  rand_bb_cc   X->bb vs X->cc linear-probe AUC of the random groupings that merge the pair and of those that keep it
               apart (fits_v5 jc2bal_pairs), the numbers the text quotes (Table D4)
  xbb_output   the X->bb output-score signal efficiency at 1% QCD efficiency, 188 and 162 classes (xbx/xbb, the
               zero_shot_xbb readout Fig. 5 draws), in percent (Table D6)
Writes experiments/FIGS/data/v2/stats/ledger33_exports.json.
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/FIGS"))
import mlst_results as M  # noqa: E402

D = ROOT / "experiments/FIGS/data/v2"
OUT = D / "stats/ledger33_exports.json"
RAND = {"RAND2_p1": "mtx-rand2p1-s1", "RAND2_p2": "mtx-rand2p2-s2", "RAND2_p3": "mtx-rand2p3-s1",
        "RAND2_p4": "mtx-rand2p4-s2", "RAND2_p5": "mtx-rand2p5-s2"}


def summary(v: list) -> dict:
    return {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else None, "runs": v}


def main() -> int:
    groups = M._groups(ROOT)
    status = lambda arm, a, b: "merged" if groups[arm][a] == groups[arm][b] else "apart"  # noqa: E731
    gg = json.loads((D / "bootstrap/l3_bs3/probe_jc2bal_pairs_gg_vs_qq.json").read_text())
    auc = {m: 1 - r["auc"]["one_minus_auc"] for m, r in gg["models"].items()}
    arms = {"L188": "l188", "L162": "l162", "R42_Q1": "r42q1", "R16_Q1": "r16q1"}
    out = {"gg_vs_qq": {"source": "bootstrap/l3_bs3/probe_jc2bal_pairs_gg_vs_qq.json", "signal": "label_X_gg",
                        "background": "label_X_qq", "test_jets": gg["class_counts"],
                        "per_model": auc,
                        "arms": {a: summary([v for m, v in auc.items() if m.startswith(f"mtx-{k}-")]) for a, k in arms.items()},
                        "random_groupings": {g: {"model": m, "auc": auc[m], "status": status(g, "label_X_gg", "label_X_qq")}
                                             for g, m in RAND.items()},
                        "status_of_label_sets": {a: status(a, "label_X_gg", "label_X_qq") for a in arms}}}
    for s in ("merged", "apart"):
        v = [r["auc"] for r in out["gg_vs_qq"]["random_groupings"].values() if r["status"] == s]
        out["gg_vs_qq"][f"random_{s}"] = summary(v) if v else None
    bb = {g: json.loads((D / f"linprobe/fits_v5/{m}.json").read_text())["cells"]["best70"]["jc2bal_pairs"]["bb_vs_cc"]["auc"]
          for g, m in RAND.items()}
    st = M.merge_status(ROOT)
    out["rand_bb_cc"] = {"source": "linprobe/fits_v5/<model>.json cells.best70.jc2bal_pairs.bb_vs_cc.auc",
                         **{s: {**summary([v for g, v in bb.items() if st[g]["bb_vs_cc"] == s]),
                                "groupings": [g for g in bb if st[g]["bb_vs_cc"] == s]} for s in ("merged", "apart")}}
    out["xbb_output"] = {"source": "xbx/xbb/<model>.json signals.xbb.zero_shot_xbb.eps_s_at_star (percent)",
                         **{a: summary([100 * json.loads((D / f"xbx/xbb/mtx-{k}-s{s}.json").read_text())["signals"]["xbb"]
                                        ["zero_shot_xbb"]["eps_s_at_star"] for s in (1, 2, 3)])
                            for a, k in (("L188", "l188"), ("L162", "l162"))}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_model",)} for k, v in out.items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
