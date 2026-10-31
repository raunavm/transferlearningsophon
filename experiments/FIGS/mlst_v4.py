"""The paper's results from the v4 inputs (2026-10-31): class-balanced JetClass-II probes, the
fixed-budget fine-tuning, the benchmark tables with literature rows, and the design numbers of the
methods those inputs follow. Called by mlst_results.build; every number goes through its Emitter.

Inputs, under experiments/FIGS/data/v2/ (pulled by pull_mlst_data.py):
  linprobe/fits_v4/<model>.json      experiments/EVAL/probe_bal.py (balanced JetClass-II, pairs,
                                     JetClass/top/quark-gluon on the v4 grid, variants, hidden layer)
  linprobe/fits_v4_mlp/<model>.json  the nonlinear probes of the same script
  jc2bal/census.json, manifest.json  experiments/FT/stage_jc2_balanced.py
  budget/lrsweep_metrics.json        experiments/FT/budget_metrics.py over /data/results/ft_v2/lrsweep
  budget/budget_metrics.json         the same over /data/results/ft_v2/budget
A missing input leaves its macros as red markers and is listed as missing.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
FITS_V4, MLP_V4 = "linprobe/fits_v4", "linprobe/fits_v4_mlp"


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d: pathlib.Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


def remap(doc: dict) -> dict:
    """A v4 fit file in fits_v3's layout, so the existing tables and comparisons read it: the
    balanced JetClass-II task under jc2_188, its pairs under jc2_188_pairs at the largest size."""
    out = {**doc, "cells": {}}
    for ck, c in doc["cells"].items():
        cc = {k: v for k, v in c.items() if "." not in k and k not in ("jc2bal", "jc2bal_pairs")}
        if "jc2bal" in c:
            cc["jc2_188"] = c["jc2bal"]
        if "jc2bal_pairs" in c:
            n = max(c.get("jc2bal", {"1000000": 0}), key=int)
            cc["jc2_188_pairs"] = {n: c["jc2bal_pairs"]}
        out["cells"][ck] = cc
    return out


def load_fits(data: pathlib.Path) -> tuple[dict, dict]:
    """({model: fits in v3 layout}, {model: the raw v4 file}) of the v4 probes, or ({}, {})."""
    raw = _load(data / FITS_V4)
    out = {}
    for m, d in raw.items():
        r = remap(d)
        if m.startswith("init-"):
            # the untrained trunk and its BatchNorm twin are separate rows (mlst_results._ck)
            out[m] = {**r, "cells": {"init": r["cells"]["init"]}}
            if "init_bn" in r["cells"]:
                out["initbn" + m.removeprefix("init")] = {**r, "cells": {"init_bn": r["cells"]["init_bn"]}}
        else:
            out[m] = r
    return out, raw


def _int(fmt_int, x) -> str:
    return fmt_int(int(x))


def _pow(x: int) -> str:
    """10^5 -> \\ensuremath{10^{5}}; 2*10^5 -> \\ensuremath{2\\times10^{5}}."""
    e = int(math.floor(math.log10(x)))
    m = x // 10 ** e
    return rf"\ensuremath{{10^{{{e}}}}}" if m == 1 else rf"\ensuremath{{{m}\times10^{{{e}}}}}"


def design_macros(em, root: pathlib.Path, data: pathlib.Path, missing: list, fmt_int) -> None:
    """The numbers of the v4 methods, from the scripts that set them and the records they wrote."""
    bfb = _module("scripts/build_ft_budget_jobs.py", "build_ft_budget_jobs")
    src = root / "scripts/build_ft_budget_jobs.py"
    em.macro("FtBudgetSteps", _int(fmt_int, bfb.EPOCHS * bfb.STEPS), src, "EPOCHS * STEPS",
             "the largest number of optimizer steps below the full training set")
    em.macro("FtBudgetEval", _int(fmt_int, bfb.STEPS), src, "STEPS", "steps between validation passes")
    em.macro("FtBudgetPatience", str(bfb.PATIENCE), src, "PATIENCE",
             "validation passes without improvement before training stops")
    em.macro("FtHeadMult", f"{bfb.HEAD_MULT:g}", src, "HEAD_MULT", "the head's learning-rate multiplier")
    em.macro("FtHeadCap", _sci(repr(bfb.HEAD_CAP)), src, "HEAD_CAP", "the head's largest learning rate")
    em.macro("FtBudgetBatch", str(bfb.BATCH), src, "BATCH")
    import re
    wd = set(re.findall(r"weight_decay[ =]([0-9.]+)", src.read_text()))
    if len(wd) != 1:
        raise SystemExit(f"FATAL: {src} sets the weight decay to {sorted(wd)}")
    em.macro("FtBudgetWeightDecay", wd.pop(), src, "--optimizer-option weight_decay (every cell)")
    grid = lambda xs: ", ".join(_sci(x) for x in xs[:-1]) + " and " + _sci(xs[-1])
    em.macro("FtSweepPre", grid(bfb.SWEEP_PRE), src, "SWEEP_PRE", "backbone rates of the sweep")
    em.macro("FtSweepScratch", grid(bfb.SWEEP_SCRATCH), src, "SWEEP_SCRATCH", "from-scratch rates of the sweep")
    em.macro("FtSweepN", _int(fmt_int, bfb.SWEEP_N), src, "SWEEP_N", "training jets of the sweep")
    em.macro("FtValJets", _int(fmt_int, bfb.VAL_N), src, "VAL_N", "the fixed validation set")
    em.macro("FtValJetsBal", _int(fmt_int, bfb.VAL_SIZE["jc2b"]), src, "VAL_SIZE[jc2b]",
             "the balanced JetClass-II validation set")
    pb = _module("experiments/EVAL/probe_bal.py", "probe_bal")
    em.macro("LpVfourCMax", _sci(repr(max(pb.C_GRID_V4))), root / "experiments/EVAL/probe_bal.py", "C_GRID_V4",
             "largest inverse regularisation of the v4 probes")
    sjb = _module("experiments/FT/stage_jc2_balanced.py", "stage_jc2_balanced")
    src = root / "experiments/FT/stage_jc2_balanced.py"
    em.macro("JcBalTestPerClass", _int(fmt_int, sjb.TEST_PER_CLASS), src, "TEST_PER_CLASS")
    em.macro("JcBalValPerClass", str(sjb.VAL_PER_CLASS), src, "VAL_PER_CLASS")
    em.macro("JcBalPairTrainPerSide", _pow(sjb.PAIR_TRAIN_PER_SIDE), src, "PAIR_TRAIN_PER_SIDE")
    em.macro("JcBalPairTestPerSide", _pow(sjb.PAIR_TEST_PER_SIDE), src, "PAIR_TEST_PER_SIDE")
    census = data / "jc2bal" / "census.json"
    if census.exists():
        c = json.loads(census.read_text())
        test = np.asarray(c["counts"]["ft_test"])
        held = np.asarray(c["counts"]["ft_heldout"])
        short = np.flatnonzero(test < sjb.TEST_PER_CLASS)
        em.macro("JcBalNShort", str(len(short)), census, "counts.ft_test",
                 f"classes with fewer than {sjb.TEST_PER_CLASS} selected test jets")
        em.macro("JcBalRarestTest", _int(fmt_int, test.min()), census, "counts.ft_test (minimum)",
                 c["class_names"][int(test.argmin())])
        em.macro("JcBalShortTrainESix", str(int((held < max(sjb.SIZES) // 188).sum())), census,
                 "counts.ft_heldout", "classes with fewer held-out jets than the 1e6 quota")
        man = data / "jc2bal" / "manifest.json"
        if man.exists():
            em.macro("JcBalNTest", _int(fmt_int, json.loads(man.read_text())["outputs"]["test.parquet"]["rows"]),
                     man, "outputs[test.parquet].rows", "jets of the balanced test set")
        else:
            missing.append("balanced JetClass-II manifest -- experiments/FIGS/data/v2/jc2bal/manifest.json")
            em.macro("JcBalNTest", r"\pending{manifest}", src, "manifest not pulled yet")
    else:
        missing.append("balanced JetClass-II census -- experiments/FIGS/data/v2/jc2bal/census.json")
        for k in ("JcBalNShort", "JcBalRarestTest", "JcBalShortTrainESix", "JcBalNTest"):
            em.macro(k, r"\pending{census}", src, "census not pulled yet")
    sweep = data / "budget" / "lrsweep_metrics.json"
    if sweep.exists():
        pre, scr = choose_rates(json.loads(sweep.read_text()))
        em.macro("FtLrPre", _sci(pre), sweep, "argmax of the summed validation metric, pretrained")
        em.macro("FtLrScratch", _sci(scr), sweep, "argmax of the summed validation metric, from scratch")
    else:
        missing.append("learning-rate sweep -- experiments/FIGS/data/v2/budget/lrsweep_metrics.json")
        for k in ("FtLrPre", "FtLrScratch"):
            em.macro(k, r"\pending{sweep}", src, "sweep not read out yet")


def _sci(lr: str) -> str:
    m, e = f"{float(lr):.0e}".split("e")
    return rf"\ensuremath{{10^{{{int(e)}}}}}" if m == "1" else rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def choose_rates(doc: dict) -> tuple[str, str]:
    """The backbone rate of the pretrained runs and the rate from scratch: each the one whose
    best validation metric, summed over the swept tasks (top and quark/gluon at 1e4), is largest.
    The test scores play no part in the choice."""
    score: dict = {}
    for key, m in doc["cells"].items():
        leaf = key.split("/")[1]                         # l188-s1_lr3e-4 or scratch_lr1e-3
        init, lr = leaf.rsplit("_lr", 1)
        kind = "scratch" if init == "scratch" else "pre"
        score.setdefault(kind, {}).setdefault(lr, []).append(m["early_stop"]["best_metric"])
    pick = {}
    for kind, by_lr in score.items():
        full = {lr: v for lr, v in by_lr.items() if len(v) == 2}          # both tasks read out
        if len(full) != len(by_lr):
            raise SystemExit(f"FATAL: the sweep's {kind} cells are incomplete: {by_lr}")
        pick[kind] = max(full, key=lambda lr: sum(full[lr]))
    return pick["pre"], pick["scratch"]


def gpu_bound_macro(em, fits: dict, runs: dict, index) -> None:
    """MlstGpuTypeBound: the relative standard deviation of 1 - macro AUC on JetClass-II over the
    three 188-class runs, which trained on two GPU types: the bound on the GPU type's effect."""
    vals = [1 - fits[m]["cells"]["best70"]["jc2_188"][max(fits[m]["cells"]["best70"]["jc2_188"], key=int)][
        "macro_auc_ovr"] for m in runs["L188"] if m in fits]
    if len(vals) < 3:
        em.macro("MlstGpuTypeBound", r"\pending{188-class probes}", index, "fewer than three runs")
        return
    rel = 100 * float(np.std(vals, ddof=1) / np.mean(vals))
    em.macro("MlstGpuTypeBound", f"{rel:.1f}\\%", index,
             "relative SD of 1 - macro AUC (balanced JetClass-II, largest set) over the 188-class runs")


# ------------------------------------------------------------------ benchmark tables
LIT = REPO / "experiments/FIGS/data/literature_benchmarks.json"
BENCH_COLS = (("acc", "accuracy", 3), ("auc", "auc", 4), ("r50", "r50", None), ("r30", "r30", None))
METHOD_TEX = {"MIParT-L-f.t.": "MIParT-L-f.t.", "L-GATr-f.t.": "L-GATr-f.t."}


def _num(s) -> float | None:
    """The leading number of a printed cell ('1602 \\pm 81', '0.9858', '{,}' separators), or None."""
    if s is None:
        return None
    import re
    m = re.match(r"\s*(-?[\d{},.]+)", str(s).replace("{,}", "").replace(",", ""))
    return float(m.group(1)) if m else None


def _rej_place(task: str) -> int:
    return 0 if task == "top" else 1


def _value_cell(vals: list, col: str, task: str, fmt_pm, two) -> str:
    """mean +- SD (three or more), both values (two) or the value alone (one), at the precision of
    the field's tables: accuracy three decimals, AUC four, rejection integers (top) or one decimal."""
    v = [x for x in vals if x is not None]
    if not v:
        return "--"
    if len(v) >= 3:
        return fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    if len(v) == 2:
        return two(v[0], v[1], ", ")
    place = {"accuracy": 3, "auc": 4}.get(col, _rej_place(task))
    return f"{v[0]:,.{place}f}".replace(",", "{,}")


def _bold_best(rows: list, ncol: int) -> list:
    """Bold the largest value of each column (higher is better for every benchmark metric)."""
    best = []
    for j in range(ncol):
        vals = [_num(r[1][j]) for r in rows]
        best.append(max((x for x in vals if x is not None), default=None))
    out = []
    for name, cells in rows:
        new = [rf"\textbf{{{c}}}" if best[j] is not None and _num(c) == best[j] else c
               for j, c in enumerate(cells)]
        out.append((name, new))
    return out


def table_bench(task: str, per: dict, runs: dict, rows: tuple, label: dict, scratch: str, fmt_pm, two,
                kin: dict | None = None) -> str:
    """The published benchmark rows (literature_benchmarks.json) and ours, on the full training set:
    accuracy, AUC, R50 and R30. task 'top' or 'qg' (the exp particle-ID scenario)."""
    lit = json.loads(LIT.read_text())["top" if task == "top" else "qg_exp"]
    lit_rows = []
    for r in lit:
        cite = rf"~\cite{{{r['cite']}}}"
        lit_rows.append((METHOD_TEX.get(r["method"], r["method"]) + cite,
                         [("--" if r[k] is None else str(r[k]).replace(" ± ", r"\,$\pm$\,").replace("±", r"\,$\pm$\,"))
                          for k, _, _ in BENCH_COLS]))
    by = per[task]
    nmax = max(n for init, bn in by.items() for n in bn)
    ours = []
    ref = by.get(scratch, {}).get(nmax, [])
    ours.append(("from scratch", [_value_cell([r[c] for r in ref], c, task, fmt_pm, two) for _, c, _ in BENCH_COLS]))
    for arm in rows:
        recs = [by.get(m[len("mtx-"):], {}).get(nmax) for m in runs[arm]]
        if not any(recs):
            continue
        cells = [_value_cell([r[0][c] if r else None for r in recs], c, task, fmt_pm, two) for _, c, _ in BENCH_COLS]
        ours.append((label[arm], cells))
    for name, rec in (kin or {}).items():
        ours.append((name, [_value_cell([rec[c]], c, task, fmt_pm, two) for _, c, _ in BENCH_COLS]))
    allrows = _bold_best(lit_rows + ours, len(BENCH_COLS))
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & accuracy & AUC & $R_{50}$ & $R_{30}$ \\", r"\midrule"]
    lines += [f"{n} & " + " & ".join(c) + r" \\" for n, c in allrows[:len(lit_rows)]]
    lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{this work (Sophon network, fine-tuned)}} \\"]
    lines += [f"{n} & " + " & ".join(c) + r" \\" for n, c in allrows[len(lit_rows):]]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


# ParT's Table 1 columns: (our label, heading); the working point is linear_probe_v2.JC1_EPS_S's
JC1_COLS = (("label_Hbb", r"$H\to b\bar b$"), ("label_Hcc", r"$H\to c\bar c$"), ("label_Hgg", r"$H\to gg$"),
            ("label_H4q", r"$H\to 4q$"), ("label_Hqql", r"$H\to \ell\nu qq'$"), ("label_Tbqq", r"$t\to bqq'$"),
            ("label_Tbl", r"$t\to b\ell\nu$"), ("label_Wqq", r"$W\to qq'$"), ("label_Zqq", r"$Z\to q\bar q$"))
JC1_LIT_N = {"PFN": "100 M", "P-CNN": "100 M", "ParticleNet": "100 M", "ParT": "100 M"}
# a rejection is marked when fewer background jets than this pass in any run (Poisson > 22 %)
JC1_MIN_BKG = 20
LPV_EPS = _module("experiments/EVAL/linear_probe_v2.py", "linear_probe_v2").JC1_EPS_S


def _jc1_cells(cells: dict, init: str, n: str, seeds) -> list:
    """The jc1 blocks of one configuration's cells at size n (ft_curves.py cells)."""
    return [cells[init][n][s]["jc1"] for s in seeds if s in cells.get(init, {}).get(n, {})]


def table_jetclass(pre: dict, ref: dict, runs: dict, rows: tuple, label: dict, scratch: str, nmax: int,
                   fmt_pm, two) -> str:
    """JetClass in the form of ParT's Table 1: accuracy, macro AUC (one-vs-one, ParT's convention,
    which equals the one-vs-rest macro AUC on JetClass's equal class sizes, checked here), and each
    signal class's rejection of QCD at ParT's working point, marked where fewer than JC1_MIN_BKG QCD
    jets pass in any run. ParT's
    rows (100 M jets; Table 3's at 2 M) from literature_benchmarks.json, ours fine-tuned on nmax
    jets (curves files of experiments/FT/ft_curves.py cells: pre the pretrained runs' tree, ref
    the from-scratch reference's), mean +- SD over the runs."""
    lit = json.loads(LIT.read_text())["jetclass"]
    n = f"N{nmax}"
    head = (r"\begin{tabular}{ll" + "c" * (2 + len(JC1_COLS)) + "}", r"\toprule",
            r" & & & & \multicolumn{" + str(len(JC1_COLS)) + r"}{c}{rejection of QCD} \\",
            r"\cmidrule(lr){5-" + str(4 + len(JC1_COLS)) + "}",
            r" & jets & accuracy & AUC & " + " & ".join(h for _, h in JC1_COLS) + r" \\",
            r" & & & & " + " & ".join(r"$R_{%g}$" % (100 * LPV_EPS.get(c, 0.5)) for c, _ in JC1_COLS) + r" \\",
            r"\midrule")
    lines = list(head)
    for r in lit:
        name = r["method"].split(" (")[0]
        lines.append(rf"{name}~\cite{{{r['cite']}}} & {r.get('n_train', JC1_LIT_N.get(r['method'], ''))} & "
                     f"{r['acc']} & {r['auc_ovo']} & " + " & ".join(_sep_int(x) for x in r["rej"]) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{" + str(4 + len(JC1_COLS)) + r"}{l}{\textit{this work (Sophon network, fine-tuned)}} \\"]

    def row(name, blocks):
        def cell(vals, place):
            v = [x for x in vals if x is not None]
            if len(v) >= 3:
                return fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
            if len(v) == 2:
                return two(v[0], v[1], ", ")
            return f"{v[0]:,.{place}f}".replace(",", "{,}") if v else "--"
        for b in blocks:
            if abs(b["macro_auc_ovo"] - b["macro_auc_ovr"]) > 1e-9:
                raise SystemExit("FATAL: JetClass one-vs-one and one-vs-rest macro AUCs differ; the test "
                                 "set's classes are not of equal size and the table must show both")
        cols = [cell([b["accuracy"] for b in blocks], 3), cell([b["macro_auc_ovo"] for b in blocks], 4)]
        for c, _ in JC1_COLS:
            wp = [b["per_class"][c]["rejection"] for b in blocks]
            mark = r"$^\dagger$" if min(w["n_bkg_pass"] for w in wp) < JC1_MIN_BKG else ""
            cols.append(cell([w["rejection"] for w in wp], 0) + mark)
        return f"{name} & {nmax / 1e6:g} M & " + " & ".join(cols) + r" \\"
    seeds = sorted({s for nn in ref.get(scratch, {}).values() for s in nn})
    blocks = _jc1_cells(ref, scratch, n, seeds)
    if blocks:
        lines.append(row("from scratch", blocks))
    for arm in rows:
        blocks = [b for m in runs[arm] for b in _jc1_cells(pre, m[len("mtx-"):], n, ["s1"])]
        if blocks:
            lines.append(row(label[arm], blocks))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def _sep_int(x) -> str:
    """A printed integer with LaTeX's thousands separator ('10638' -> '10{,}638'), as the other tables."""
    s = str(x)
    return f"{int(s):,}".replace(",", "{,}") if s.isdigit() else s
