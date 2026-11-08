"""The paper's results from the v4 inputs (2026-10-31): class-balanced JetClass-II probes, the
fixed-budget fine-tuning, the benchmark tables with literature rows, and the design numbers of the
methods those inputs follow. Called by mlst_results.build; every number goes through its Emitter.

Inputs, under experiments/FIGS/data/v2/ (pulled by pull_mlst_data.py):
  linprobe/fits_v4/<model>.json      experiments/EVAL/probe_bal.py (balanced JetClass-II, pairs,
                                     JetClass/top/quark-gluon on the v4 grid, variants, hidden layer)
  linprobe/fits_v4_mlp/<model>.json  the nonlinear probes of the same script
  linprobe/fits_v4_sib/<model>.json  the X->ee vs X->mumu probe (probe_bal.py --sibling-pairs), merged
                                     into the v4 cells by load_fits
  curves_v2/<family>/<model>.json    ROC curves on a fixed signal-efficiency grid (experiments/FT/roc_points.py)
  jc2bal/census.json, manifest.json  experiments/FT/stage_jc2_balanced.py
  budget/lrsweep_metrics.json        experiments/FT/budget_metrics.py over /data/results/ft_v2/lrsweep_v2
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
SIB_V4 = "linprobe/fits_v4_sib"
SIB2_V4 = "linprobe/fits_v4_sib2"     # the same probe scored on every X->ee / X->mumu test jet; replaces SIB_V4
ROC_DIR = "curves_v2"
ROC_FAMILIES = ("jc1_ft", "jc1_probe", "pairs", "pairs_sib2", "top_ft", "qg_ft", "jc2_ovr")   # pairs_sib2: every X->ee / X->mumu test jet
# the long-budget JetClass cells at 1e6 (scripts/build_gallery_data_job.py --mode long-jc1): <family>_long replaces
# <family> whole once it holds every paper run and these from-scratch seeds (build_ft_budget_jobs.SCRATCH_SEEDS),
# renamed to the old reference's scratch-v2-s<k>; one recipe per figure
JC1_LONG, JC1_LONG_SCRATCH = ("jc1_ft", "jc1_ft_ovr"), ("s1", "s2", "s3")


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
    """({model: fits in v3 layout}, {model: the raw v4 file}) of the v4 probes, or ({}, {}). The
    X->ee vs X->mumu probe of fits_v4_sib joins the raw cells as one more class-balanced pair
    (cells[ckpt]["jc2bal_pairs"]["ee_vs_mm"]); fits_v4_sib2, scored on every test jet of the pair,
    replaces it where present; a model without a v4 file is not merged."""
    raw = _load(data / FITS_V4)
    for sibs in (_load(data / SIB_V4), _load(data / SIB2_V4)):
        for m, sib in sibs.items():
            if m not in raw:
                continue
            for ck, c in sib["cells"].items():
                raw[m]["cells"].setdefault(ck, {}).setdefault("jc2bal_pairs", {})["ee_vs_mm"] = c["jc2bal_pairs"]["ee_vs_mm"]
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
    em.macro("FtLongSteps", _int(fmt_int, bfb.LONG_EPOCHS * bfb.STEPS), src, "LONG_EPOCHS * STEPS",
             "the largest number of optimizer steps of the long budget at LONG_N jets (JetClass, JetClass-II)")
    counts = data / "jc2_train" / "counts.json"
    if counts.exists():
        em.macro("DesignNTrainSelected", _int(fmt_int, json.loads(counts.read_text())["n_selected"]), counts,
                 "n_selected", "selected jets of the pretrain_train files (experiments/FT/count_jc2_train.py)")
    em.macro("FtBudgetEval", _int(fmt_int, bfb.STEPS), src, "STEPS",
             "steps between validation passes at and above SMALL_N training jets")
    em.macro("FtBudgetEvalSmall", _int(fmt_int, bfb.SMALL_STEPS), src, "SMALL_STEPS",
             "steps between validation passes below SMALL_N training jets")
    em.macro("FtBudgetEvalSmallBelow", _pow(bfb.SMALL_N), src, "SMALL_N",
             "training jets below which the validation runs every SMALL_STEPS steps")
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
        launched = launched_scratch_rates(root)
        want = {"top": scr["leg_top"], "qg": scr["leg_qg"], "jc1": scr["leg_top"], "jc2b": scr["leg_top"]}
        if launched != want:
            raise SystemExit(f"FATAL: the launched from-scratch rates {launched} are not the sweep's {want}")
        em.macro("FtLrPre", _sci(pre), sweep, "argmax of the summed validation metric, pretrained")
        em.macro("FtLrScratchTop", _sci(scr["leg_top"]), sweep, "argmax of the validation metric on top, from scratch",
                 "agrees with LR of the launched job-ft-budget-bench, -jc1 and -jc2b from-scratch cells")
        em.macro("FtLrScratchQg", _sci(scr["leg_qg"]), sweep, "argmax of the validation metric on quark/gluon, from scratch",
                 "agrees with LR of the launched job-ft-budget-bench from-scratch cells")
    else:
        missing.append("learning-rate sweep -- experiments/FIGS/data/v2/budget/lrsweep_metrics.json")
        for k in ("FtLrPre", "FtLrScratchTop", "FtLrScratchQg"):
            em.macro(k, r"\pending{sweep}", src, "sweep not read out yet")


def _sci(lr: str) -> str:
    m, e = f"{float(lr):.0e}".split("e")
    return rf"\ensuremath{{10^{{{int(e)}}}}}" if m == "1" else rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def choose_rates(doc: dict) -> tuple[str, dict]:
    """The backbone rate of the pretrained runs, the one whose best validation metric summed over
    the swept tasks (top and quark/gluon at 1e4) is largest, and the rate from scratch of each task,
    {sweep tree: the rate with its largest best validation metric}, as launched (RUNS.csv: each task
    takes its own validation-best rate). The test scores play no part in the choice."""
    score: dict = {"pre": {}, "scratch": {}}
    for key, m in doc["cells"].items():
        tree, leaf = key.split("/")[:2]                  # leg_top, l188-s1_lr3e-4 or scratch_lr1e-3
        init, lr = leaf.rsplit("_lr", 1)
        kind = "scratch" if init == "scratch" else "pre"
        score[kind].setdefault(lr, {})[tree] = m["early_stop"]["best_metric"]
    for kind, by_lr in score.items():
        if any(len(v) != 2 for v in by_lr.values()):      # both tasks read out
            raise SystemExit(f"FATAL: the sweep's {kind} cells are incomplete: {by_lr}")
    pre, scr = score["pre"], score["scratch"]
    return max(pre, key=lambda lr: sum(pre[lr].values())), {t: max(scr, key=lambda lr: scr[lr][t]) for t in
                                                            sorted({t for v in scr.values() for t in v})}


def launched_scratch_rates(root: pathlib.Path) -> dict:
    """{task: learning rate} of the from-scratch cells (INIT empty) of the launched fixed-budget
    jobs; FATAL when one task ran at two rates."""
    import re
    out: dict = {}
    for kind in ("bench", "jc1", "jc2b"):
        text = (root / f"experiments/FT/k8s/budget/job-ft-budget-{kind}-raunav.yaml").read_text()
        for task, lr in re.findall(r"TASK=(\w+); INIT=; [^\n]*?LR=([^;]+);", text):
            if out.setdefault(task, lr) != lr:
                raise SystemExit(f"FATAL: job-ft-budget-{kind}: from-scratch {task} ran at {lr} and {out[task]}")
    return out


def _majority_gpu(comp: dict, m: str):
    c = comp.get(m)
    if not c:
        return None
    from collections import Counter
    return Counter(e["device"] for e in c["epochs"]).most_common(1)[0][0]


def gpu_bound_macro(em, fits: dict, runs: dict, comp: dict, index) -> None:
    """MlstGpuTypeBound: how far the label-set ratios of the run index whose 188-class run trained on a
    GPU type no other 188-class run used lie from those of the other indices. For each of the 162-, 43-
    and 17-class label sets, the ratio of 1 - AUC to the 188-class run of the same index (AUC averaged
    over the JetClass-II classes, largest training set of the fits the paper's ratios use); the bound is
    the largest |ratio of that index / mean ratio of the others - 1|, in per cent. A ratio between
    pretraining configurations on JetClass-II, the only form 1 - AUC takes in the paper (PI 2026-11-02);
    it includes the seed-to-seed spread, so it bounds the GPU effect from above. An observation on one
    run, not a test. MlstGpuOtherSeedsDiff: the largest relative difference between the label-set ratios
    of the other indices, the scale of the variation between seeds to read the bound against. Their
    188-class runs share a GPU type, but the coarser models in the same ratios do not, so it is not a
    same-GPU baseline."""
    def oma(m):
        c = fits.get(m, {}).get("cells", {}).get("best70", {}).get("jc2_188", {})
        return 1 - c[max(c, key=int)]["macro_auc_ovr"] if c else None
    idx = lambda m: m.rsplit("-s", 1)[1]
    gpu = {m: _majority_gpu(comp, m) for m in runs["L188"]}
    odd = [m for m, g in gpu.items() if g is not None and list(gpu.values()).count(g) == 1]
    if len(odd) != 1 or any(oma(m) is None for m in runs["L188"]):
        for k in ("MlstGpuTypeBound", "MlstGpuOtherSeedsDiff"):
            em.macro(k, r"\pending{188-class probes}", index, "no single 188-class run on its own GPU type, or a fit missing")
        return
    o = odd[0]
    dev, pair = {}, {}
    for arm in ("L162", "R42_Q1", "R16_Q1"):
        by = {idx(m): m for m in runs.get(arm, [])}
        r = {idx(m): oma(by[idx(m)]) / oma(m) for m in runs["L188"] if idx(m) in by and oma(by[idx(m)]) is not None}
        others = [v for k, v in r.items() if k != idx(o)]
        if idx(o) in r and len(others) >= 2:
            dev[arm] = abs(r[idx(o)] / float(np.mean(others)) - 1)
            pair[arm] = (max(others) - min(others)) / float(np.mean(others))
    if not dev:
        for k in ("MlstGpuTypeBound", "MlstGpuOtherSeedsDiff"):
            em.macro(k, r"\pending{label-set probes}", index, "no label set has every ratio")
        return
    worst = max(dev, key=dev.get)
    em.macro("MlstGpuTypeBound", f"{100 * dev[worst]:.1f}\\%", index,
             f"largest |label-set ratio of 1 - AUC of index {idx(o)} / mean of the other indices - 1|, JetClass-II "
             "largest set, per cent", f"{o} trained on {gpu[o]}; largest for {worst}; label sets "
             + ", ".join(f"{a} {100 * v:.2f}%" for a, v in dev.items()))
    em.macro("MlstGpuOtherSeedsDiff", f"{100 * max(pair.values()):.1f}\\%", index,
             f"largest (max - min) / mean of a label-set ratio of 1 - AUC over the indices other than {idx(o)}, "
             "JetClass-II largest set, per cent", "label sets " + ", ".join(f"{a} {100 * v:.2f}%" for a, v in pair.items()))


# ------------------------------------------------------------------ benchmark tables
LIT = REPO / "experiments/FIGS/data/literature_benchmarks.json"
BENCH_COLS = (("acc", "accuracy", 3), ("auc", "auc", 4), ("r50", "r50", None), ("r30", "r30", None))
METHOD_TEX = {"MIParT-L-f.t.": "MIParT-L-f.t.", "L-GATr-f.t.": "L-GATr-f.t."}
# the heading over our rows: how the network was used downstream (PI, 2026-11-07)
MODE_FT = "this work"      # the heading over our rows; each row names its network and how it was used
SCRATCH_LABEL = "no pretraining"
PARAMS_HEAD = r"\# params"


def our_params(k_out: int, n_inputs: int = 17) -> int:
    """Trainable parameters of our network with k_out outputs and n_inputs particle features: docs/GROUND_TRUTH.md's
    2,304,688 at the 188-class head on 17 inputs, less 513 per output (the last layer is 512 -> K) and 132 per input
    feature (the input normalization and first embedding layer) fewer; checked against the instantiated network
    (experiments/E1/ParT_sophon_arch_10c.py) by tests/test_mlst_results.py."""
    return 2_304_688 - 513 * (188 - k_out) - 132 * (17 - n_inputs)


def _params_cell(n: int) -> str:
    return f"{n / 1e6:.2f}\\,M"


# A cell of a generated table carries its runs' values as a token until mlst_results.uniform_columns prints the
# whole column at one precision: kind p (mean, SD), t (two runs), o (one value).
TOK_OPEN, TOK_CLOSE = "\x02", "\x03"


def tok(kind: str, *vals) -> str:
    return TOK_OPEN + kind + " " + " ".join(repr(float(v)) for v in vals) + TOK_CLOSE


def untok(cell: str):
    """(kind, values) of the first token in cell, or None."""
    import re
    m = re.search(TOK_OPEN + r"(\w) ([^" + TOK_CLOSE + "]*)" + TOK_CLOSE, str(cell))
    return None if m is None else (m.group(1), [float(x) for x in m.group(2).split()])


def tokens(fmt_pm) -> bool:
    """fmt_pm is mlst_results' table formatter, which writes tokens."""
    return getattr(fmt_pm, "tokens", False)


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
    the field's tables: accuracy and AUC four decimals, rejection integers (top) or one decimal."""
    v = [x for x in vals if x is not None]
    if not v:
        return "--"
    if len(v) >= 3:
        return fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    if len(v) == 2:
        return two(v[0], v[1], ", ")
    if tokens(fmt_pm):
        return tok("o", v[0])
    place = {"accuracy": 4, "auc": 4}.get(col, _rej_place(task))
    return f"{v[0]:,.{place}f}".replace(",", "{,}")


def _num_sd(s):
    """(value, SD) of a printed cell: 'm \\pm sd' -> (m, sd); 'a, b' (two runs) -> (their mean, their SD |a-b|/sqrt 2)."""
    import re
    u = untok(s)
    if u:
        k, v = u
        return (v[0], v[1]) if k == "p" else ((v[0] + v[1]) / 2, abs(v[0] - v[1]) / math.sqrt(2)) if k == "t" else (v[0], None)
    t = str(s).replace("{,}", "").replace(r"\textbf{", "").replace(r"$\geq$", "")
    m = re.match(r"\s*(?:\$\\geq\$)?(-?[\d.]+)\\,\$\\pm\$\\,(?:\$<\$)?([\d.]+)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    two = re.match(r"\s*(-?[\d.]+), (-?[\d.]+)", t)
    if two:
        a, b = float(two.group(1)), float(two.group(2))
        return (a + b) / 2, abs(a - b) / math.sqrt(2)
    return _num(t), None


def _bold_best(rows: list, ncol: int) -> list:
    """Bold the best value of each column (higher is better for every benchmark metric) and every value
    within one standard deviation of it, the larger of its own and the best one's (C1.14); a
    configuration with two runs competes with their mean and SD (PI review of the results book, 2026-11-07)."""
    best = []
    for j in range(ncol):
        vals = [_num_sd(r[1][j]) for r in rows]
        best.append(max((x for x in vals if x[0] is not None), default=(None, None), key=lambda x: x[0]))
    out = []
    for name, cells in rows:
        new = []
        for j, c in enumerate(cells):
            v, sd = _num_sd(c)
            b, bsd = best[j]
            hit = b is not None and v is not None and (v == b or v + max(sd or 0.0, bsd or 0.0) >= b)
            new.append(rf"\textbf{{{c}}}" if hit and c != "--" else c)
        out.append((name, new))
    return out


# the published rows each benchmark table shows (FIGURE_PLAN T3, T4)
LIT_ROWS = {"top": ("linear EFP", "ParticleNet", "ParT", "ParT-f.t.", "MIParT-L-f.t.", "L-GATr-f.t.", "OmniLearned-l"),
            "qg": ("constituent multiplicity", "linear EFP", "PFN-Ex", "ParticleNet", "ParT", "ParticleNet-f.t.",
                   "ParT-f.t.", "OmniLearned-m")}


def _lit_cell(x) -> str:
    """A published value as printed, with the table's thousands separator and plus-minus."""
    if x is None:
        return "--"
    parts = str(x).replace("±", " ± ").split(" ± ")
    parts = [_sep_int(p.strip()) if p.strip().isdigit() else p.strip() for p in parts]
    return r"\,$\pm$\,".join(parts)


def table_bench(task: str, per: dict, runs: dict, rows: tuple, label: dict, scratch: str, fmt_pm, two,
                kin: tuple | None = None, herwig: bool = False) -> str:
    """T3 (task 'top') and T4 ('qg', ParT's exp inputs; herwig=True adds the Herwig columns): the
    published rows of LIT_ROWS, then ours on the full training set: from scratch, the label sets of
    rows fine-tuned, and (kin = (row label, init)) the kinematic-only model. Columns: accuracy, AUC,
    R50, R30 (the caption says which inputs are set to zero); bold as _bold_best. A first column gives the
    trainable parameters: the published as the model's paper gives them, ours our_params (two outputs; the
    kinematic-only model on 7 inputs); it takes no part in the bold."""
    lit_all = {r["method"]: r for r in json.loads(LIT.read_text())["top" if task == "top" else "qg_exp"]}
    tasks = [task] + (["qg_herwig"] if herwig else [])
    nc = len(BENCH_COLS) * len(tasks)
    lit_rows = []
    for name in LIT_ROWS[task]:
        if name not in lit_all:
            continue
        r = lit_all[name]
        cells = [_lit_cell(r[k]) for k, _, _ in BENCH_COLS] + ["--"] * len(BENCH_COLS) * (len(tasks) - 1)
        lit_rows.append((METHOD_TEX.get(name, name) + rf"~\cite{{{r['cite']}}}", cells, r.get("params") or "--"))
    ours = []

    def recs_of(t, init):
        by = per.get(t, {})
        if not by:
            return []
        nmax = max(n for i, bn in by.items() for n in bn)
        return by.get(init, {}).get(nmax, [])
    ours.append((SCRATCH_LABEL, [_value_cell([r[c] for r in recs_of(t, scratch)], c, t, fmt_pm, two)
                                 for t in tasks for _, c, _ in BENCH_COLS], _params_cell(our_params(2))))
    for arm in rows:
        cells = []
        for t in tasks:
            recs = [(recs_of(t, m[len("mtx-"):]) or [None])[0] for m in runs[arm]]
            cells += [_value_cell([r[c] if r else None for r in recs], c, t, fmt_pm, two) for _, c, _ in BENCH_COLS]
        ours.append((label[arm], cells, _params_cell(our_params(2))))
    kin_row = None
    if kin:
        recs = recs_of(task, kin[1])
        kin_row = (kin[0], [_value_cell([recs[0][c]], c, task, fmt_pm, two) if recs else r"\pending{run}"
                            for _, c, _ in BENCH_COLS] + ["--"] * len(BENCH_COLS) * (len(tasks) - 1),
                   _params_cell(our_params(2, 7)))
    every = lit_rows + ours + ([kin_row] if kin_row else [])
    allrows = [(n, [p] + c) for (n, c), (_, _, p) in zip(_bold_best([(n, c) for n, c, _ in every], nc), every)]
    head = [r"\begin{tabular}{lc" + "c" * nc + "}", r"\toprule"]
    if herwig:
        head += [r" & & \multicolumn{4}{c}{Pythia} & \multicolumn{4}{c}{Herwig} \\",
                 r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}"]
    head += [rf" & {PARAMS_HEAD} & " + " & ".join(["accuracy & AUC & $R_{50}$ & $R_{30}$"] * len(tasks)) + r" \\",
             r"\midrule"]
    lines = head + [f"{n} & " + " & ".join(c) + r" \\" for n, c in allrows[:len(lit_rows)]]
    lines += [r"\midrule", r"\multicolumn{" + str(2 + nc) + r"}{l}{\textit{" + MODE_FT + r"}} \\"]
    lines += [f"{n} & " + " & ".join(c) + r" \\" for n, c in allrows[len(lit_rows):]]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


# ParT's Table 1 columns: (our label, heading); the working point is linear_probe_v2.JC1_EPS_S's
JC1_COLS = (("label_Hbb", r"$H\to b\bar b$"), ("label_Hcc", r"$H\to c\bar c$"), ("label_Hgg", r"$H\to gg$"),
            ("label_H4q", r"$H\to 4q$"), ("label_Hqql", r"$H\to \ell\nu qq'$"), ("label_Tbqq", r"$t\to bqq'$"),
            ("label_Tbl", r"$t\to b\ell\nu$"), ("label_Wqq", r"$W\to qq'$"), ("label_Zqq", r"$Z\to q\bar q$"))
JC1_LIT_N = {"PFN": "100 M", "P-CNN": "100 M", "ParticleNet": "100 M", "ParT": "100 M"}
# a rejection is marked when fewer background jets than this pass in any run (Poisson > 22 %)
JC1_MIN_BKG = 20
_LPV = _module("experiments/EVAL/linear_probe_v2.py", "linear_probe_v2")
LPV_EPS, LPV_CLASSES = _LPV.JC1_EPS_S, _LPV.JC1_CLASSES


# ------------------------------------------------------------------ ROC curves (experiments/FT/roc_points.py)
def paper_models() -> list:
    return [l.strip() for l in (REPO / "configs/arms/v2_paper_runs.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def load_family(data: pathlib.Path, fam: str) -> dict:
    """{model: doc} of curves_v2/<fam>; for a JC1_LONG family the long-budget docs (scratch-s<k> as
    scratch-v2-s<k>) when they hold every paper run and every JC1_LONG_SCRATCH seed, else the old ones."""
    docs = _load(data / ROC_DIR / fam)
    if fam not in JC1_LONG:
        return docs
    long = {f"scratch-v2-{m.removeprefix('scratch-')}" if m.startswith("scratch-") else m: d
            for m, d in _load(data / ROC_DIR / f"{fam}_long").items()}
    need = set(paper_models()) | {f"scratch-v2-{s}" for s in JC1_LONG_SCRATCH}
    return long if need <= set(long) else docs


def load_roc(data: pathlib.Path) -> dict:
    """{family: {model: doc}} of curves_v2/<family>/<model>.json (load_family); a family with no files has no key.
    FATAL when the docs of a family differ in eps_grid, when a doc's min_bkg_pass is not JC1_MIN_BKG,
    or when two models' curves of one key were scored on different jets (label_sha256)."""
    out = {}
    for fam in ROC_FAMILIES:
        docs = load_family(data, fam)
        if not docs:
            continue
        grids = {tuple(d["eps_grid"]) for d in docs.values()}
        if len(grids) != 1:
            raise SystemExit(f"FATAL: the {fam} curves use {len(grids)} different signal-efficiency grids")
        labels: dict = {}
        for m, d in docs.items():
            if d["min_bkg_pass"] != JC1_MIN_BKG:
                raise SystemExit(f"FATAL: {fam}/{m}: min_bkg_pass {d['min_bkg_pass']}, the paper's is {JC1_MIN_BKG}")
            for key, c in d["curves"].items():
                if labels.setdefault(key, (m, c["label_sha256"]))[1] != c["label_sha256"]:
                    raise SystemExit(f"FATAL: {fam} curve {key}: {m} and {labels[key][0]} were scored on different jets")
        out[fam] = docs
    return out


def roc_series(doc: dict, key: str) -> tuple | None:
    """(eps grid, rejection with NaN below the cut, i.e. where fewer than JC1_MIN_BKG background jets
    pass; all NaN when cut is null), or None when doc has no curve key."""
    c = doc["curves"].get(key)
    if c is None:
        return None
    eps = np.asarray(doc["eps_grid"], float)
    rej = np.array([np.nan if v is None else v for v in c["rejection"]], float)
    rej[eps < c["cut"]["eps_s"] if c["cut"] else slice(None)] = np.nan
    return eps, rej


def roc_limit(doc: dict, key: str) -> float:
    """n_background / JC1_MIN_BKG: the rejection at which every curve of that test set stops."""
    return doc["curves"][key]["n_background"] / JC1_MIN_BKG


def _jc1_cells(cells: dict, init: str, n: str, seeds) -> list:
    """The jc1 blocks of one configuration's cells at size n (ft_curves.py cells)."""
    return [cells[init][n][s]["jc1"] for s in seeds if s in cells.get(init, {}).get(n, {})]


def _jc_cell(vals, place, fmt_pm, two):
    v = [x for x in vals if x is not None]
    if tokens(fmt_pm) and v:          # runs that agree exactly keep their "+- 0" (sd 0)
        return (tok("p", float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) >= 3
                else tok("t", *v) if len(v) == 2 else tok("o", v[0]))
    if len(v) >= 3:
        sd = float(np.std(v, ddof=1))
        if sd == 0:     # runs that agree exactly: the value at the column's precision, and say so
            return f"{v[0]:,.{place}f}".replace(",", "{,}") + r"\,$\pm$\,0"
        return fmt_pm(float(np.mean(v)), sd)
    if len(v) == 2:
        return two(v[0], v[1], ", ")
    return f"{v[0]:,.{place}f}".replace(",", "{,}") if v else "--"


def table_jetclass(pre: dict, ref: dict, runs: dict, rows: tuple, label: dict, scratch: str, nmax: int,
                   fmt_pm, two, literature: bool = True) -> str:
    """JetClass in the form of ParT's Table 1 (T2): accuracy, AUC averaged over class pairs (one-vs-one,
    ParT's convention, which equals the one-vs-rest average on JetClass's equal class sizes, checked
    here), and each signal class's rejection of QCD at ParT's working point, marked where fewer than
    JC1_MIN_BKG QCD jets pass in any run. Published rows: those trained on 2 M jets (literature_benchmarks.json
    entries with n_train; MIParT and L-GATr when the file has them), the training size beside each
    row; ours fine-tuned on nmax jets (ft_curves.py cells: pre the pretrained runs' tree, ref the
    from-scratch reference's), mean +- SD over the runs. Bold as _bold_best (C1.14), every row."""
    n = f"N{nmax}"
    head = (r"\begin{tabular}{ll" + "c" * (3 + len(JC1_COLS)) + "}", r"\toprule",
            r" & & & & & \multicolumn{" + str(len(JC1_COLS)) + r"}{c}{rejection of QCD} \\",
            r"\cmidrule(lr){6-" + str(5 + len(JC1_COLS)) + "}",
            rf" & jets & {PARAMS_HEAD} & accuracy & AUC & " + " & ".join(h for _, h in JC1_COLS) + r" \\",
            r" & & & & & " + " & ".join(r"$R_{%g}$" % (100 * LPV_EPS.get(c, 0.5)) for c, _ in JC1_COLS) + r" \\",
            r"\midrule")
    lines = list(head)
    ncol = 2 + len(JC1_COLS)
    lit = []                                   # (row prefix, value cells, marks after the values)
    if literature:
        for r in json.loads(LIT.read_text())["jetclass"]:
            if not r.get("n_train"):
                continue           # the 100 M rows are context in the caption (FIGURE_PLAN T2)
            name = r["method"].split(" (")[0]
            lit.append((rf"{name}~\cite{{{r['cite']}}} & {r['n_train']} & {r.get('params') or '--'}",
                        [r["acc"], r["auc_ovo"]] + [_sep_int(x) for x in r["rej"]], [""] * ncol))

    def row(name, blocks):
        for b in blocks:
            if abs(b["macro_auc_ovo"] - b["macro_auc_ovr"]) > 1e-9:
                raise SystemExit("FATAL: JetClass one-vs-one and one-vs-rest macro AUCs differ; the test "
                                 "set's classes are not of equal size and the table must show both")
        cols = [_jc_cell([b["accuracy"] for b in blocks], 3, fmt_pm, two),
                _jc_cell([b["macro_auc_ovo"] for b in blocks], 4, fmt_pm, two)]
        marks = ["", ""]
        for c, _ in JC1_COLS:
            wp = [b["per_class"][c]["rejection"] for b in blocks]
            marks.append(r"$^\dagger$" if min(w["n_bkg_pass"] for w in wp) < JC1_MIN_BKG else "")
            cols.append(_jc_cell([w["rejection"] for w in wp], 0, fmt_pm, two))
        return f"{name} & {nmax / 1e6:g} M & {_params_cell(our_params(len(JC1_COLS) + 1))}", cols, marks
    ours = []
    seeds = sorted({s for nn in ref.get(scratch, {}).values() for s in nn})
    blocks = _jc1_cells(ref, scratch, n, seeds)
    if blocks:
        ours.append(row(SCRATCH_LABEL, blocks))
    for arm in rows:
        blocks = [b for m in runs[arm] for b in _jc1_cells(pre, m[len("mtx-"):], n, ["s1"])]
        if blocks:
            ours.append(row(label[arm], blocks))
    # C1.14: the best value of each column and every value within one SD of it in bold, chosen among the
    # values without a dagger (they carry no claim, the caption), which are never bold
    bold = _bold_best([(p, ["--" if m else c for c, m in zip(cells, mk)]) for p, cells, mk in lit + ours], ncol)
    bold = [(p, [c0 if m else c for c, c0, m in zip(cells, orig, mk)])
            for (p, cells), (_, orig, mk) in zip(bold, lit + ours)]
    for i, ((prefix, cells), (_, _, marks)) in enumerate(zip(bold, lit + ours)):
        if literature and i == len(lit):
            lines += [r"\midrule", r"\multicolumn{" + str(3 + ncol) + r"}{l}{\textit{" + MODE_FT + r"}} \\"]
        lines.append(prefix + " & " + " & ".join(c + mk for c, mk in zip(cells, marks)) + r" \\")
    if literature and not ours:
        lines += [r"\midrule", r"\multicolumn{" + str(3 + ncol) + r"}{l}{\textit{" + MODE_FT + r"}} \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def table_jetclass_nbkg(pre: dict, ref: dict, runs: dict, rows: tuple, label: dict, scratch: str, nmax: int) -> str:
    """A1: the QCD jets passing each signal class's working point, the fewest over a configuration's runs."""
    n = f"N{nmax}"
    lines = [r"\begin{tabular}{l" + "c" * len(JC1_COLS) + "}", r"\toprule",
             r" & " + " & ".join(h for _, h in JC1_COLS) + r" \\", r"\midrule"]

    def row(name, blocks):
        return name + " & " + " & ".join(_sep_int(min(b["per_class"][c]["rejection"]["n_bkg_pass"] for b in blocks))
                                         for c, _ in JC1_COLS) + r" \\"
    seeds = sorted({s for nn in ref.get(scratch, {}).values() for s in nn})
    blocks = _jc1_cells(ref, scratch, n, seeds)
    if blocks:
        lines.append(row(SCRATCH_LABEL, blocks))
    for arm in rows:
        blocks = [b for m in runs[arm] for b in _jc1_cells(pre, m[len("mtx-"):], n, ["s1"])]
        if blocks:
            lines.append(row(label[arm], blocks))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def _sep_int(x) -> str:
    """A printed integer with LaTeX's thousands separator ('10638' -> '10{,}638'), as the other tables."""
    s = str(x)
    return f"{int(s):,}".replace(",", "{,}") if s.isdigit() else s
