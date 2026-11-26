#!/usr/bin/env python3
"""Appendix table (PI 2026-11-25, ledger 33, 2(a)): where the 17-class model's loss on JetClass sits. Per-class recall
of the ten JetClass classes for the 188- and the 17-class model, frozen (the linear probe at 10^6 training jets,
perclass_recall/probe_jc1.json) and fine-tuned (the one rule's cells at 10^6, perclass_recall/ft_jc1_N1000000.json),
mean +- SD over the three runs, and the 17 - 188 difference over its sigma (SC-1: run SDs and the per-class
paired test-jet bootstrap, B = 200, experiments/EVAL/perclass_recall.py). On JetClass's 200,000 test jets per class the
accuracy is the plain mean of the ten recalls. Writes paper/journal/v2/tables/perclass_jc1.tex and prints the classes
that differ beyond 3 sigma.

    python3 experiments/FIGS/perclass_jc1_table.py
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "FIGS"))
import make_tables as MT  # noqa: E402

DATA = ROOT / "experiments/FIGS/data/v2/perclass_recall"
OUT = ROOT / "paper/journal/v2/tables/perclass_jc1.tex"
FILES = (("frozen features, linear probe", "probe_jc1.json"), ("fine-tuned", "ft_jc1_N1000000.json"))
NAMES = {"label_QCD": "QCD", "label_Hbb": r"$H\to b\bar b$", "label_Hcc": r"$H\to c\bar c$", "label_Hgg": r"$H\to gg$",
         "label_H4q": r"$H\to 4q$", "label_Hqql": r"$H\to \ell\nu qq'$", "label_Zqq": r"$Z\to q\bar q$",
         "label_Wqq": r"$W\to qq'$", "label_Tbqq": r"$t\to bqq'$", "label_Tbl": r"$t\to b\ell\nu$"}
DIFF = "R16_Q1-L188"


def block(f: str) -> tuple[list, dict]:
    d = json.loads((DATA / f).read_text())
    if d["classes"] != list(NAMES) or d.get("balanced") is False:
        raise SystemExit(f"FATAL: {f}: classes {d['classes']} or an unequal test set")
    e = d["diffs"][DIFF]
    side = lambda ms, k: [d["models"][m]["recall"][k] for m in ms]  # noqa: E731
    rows = []
    for k, c in enumerate(d["classes"]):
        a, b = side(e["a"], k), side(e["b"], k)
        rows.append((c, statistics.fmean(b), statistics.stdev(b), statistics.fmean(a), statistics.stdev(a),
                     e["delta"][k], e["sigma"][k]))
    return rows, e


def main() -> int:
    blocks = [(h, *block(f)) for h, f in FILES]
    body = []
    for k, c in enumerate(NAMES):
        cells = []
        for _h, rows, _e in blocks:
            _, mb, sb, ma, sa, dl, sg = rows[k]
            cells += [MT.fmt_pm(mb, sb), MT.fmt_pm(ma, sa), f"{dl / sg:+.1f}".replace("-", "$-$")]
        body.append(NAMES[c] + " & " + " & ".join(cells) + r" \\")
    tex = "\n".join([
        "% written by experiments/FIGS/perclass_jc1_table.py (ledger 33, 2(a)); values from experiments/FIGS/data/v2/perclass_recall",
        r"\begin{table}[t]\centering",
        r"\caption{Recall of each JetClass class for the 188- and the 17-class models, with frozen features (linear probe) "
        r"and fine-tuned, both on $10^6$ training jets: mean $\pm$ standard deviation over three runs, and the difference "
        r"$17-188$ over its $\sigma$ (run spreads and the paired test-jet bootstrap).}\label{tab:perclass_jc1}",
        r"\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{frozen features} & \multicolumn{3}{c}{fine-tuned} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Class & 188 classes & 17 classes & $\Delta/\sigma$ & 188 classes & 17 classes & $\Delta/\sigma$ \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    for h, rows, e in blocks:
        lo = [NAMES[rows[k][0]] for k in e["lower"]]
        hi = [NAMES[rows[k][0]] for k in e["higher"]]
        print(f"{h}: 17 - 188 lower beyond 3 sigma in {lo}; higher in {hi}")
        for c, mb, sb, ma, sa, dl, sg in rows:
            print(f"  {c:11} 188 {mb:.4f}+-{sb:.4f}  17 {ma:.4f}+-{sa:.4f}  d={dl:+.4f} s={sg:.4f} z={dl / sg:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
