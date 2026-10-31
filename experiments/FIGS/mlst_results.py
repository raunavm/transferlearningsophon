"""The MLST paper's results tables, read by experiments/FIGS/make_tables.py (build()).

Every value is the mean +- standard deviation over the pretraining runs of a configuration
that the paper reports (configs/arms/v2_paper_runs.txt), PDG-rounded (make_tables.fmt_pm). A
configuration with runs still missing prints the runs it has and a red marker "n of N runs".

Configurations with two runs print both values: a standard deviation from two runs has one
degree of freedom (plan 2026-10-29, statistics). A lone run prints three significant figures.

Inputs, under experiments/FIGS/data/v2/ (experiments/FIGS/pull_mlst_data.py):
  linprobe/fits_v3/<model>.json      linear probes, JetClass-II at its 188 classes
                                     (experiments/EVAL/linear_probe_v2.py, 2026-10-29 refit)
  linprobe/fits_v3_mlp/<model>.json  the nonlinear probes beside the linear nulls
  head/<model>_mass.json             mass node, MLP mass regression, the uncorrected mass
  head/<model>_sculpting.json        QCD mass sculpting (experiments/EVAL/mass_sculpting.py)
  fillcontrol/<mask>/<model>.json    JetClass-II probes with the downstream data's missing
                                     features zeroed
  compute/<run>.json                 per epoch: the GPU and the seconds of training and validation
  ft_mlst/<tag>_<file>.json          fine-tuning read-outs (best70: every run, with the V100
                                     from-scratch references); best70_leg188_leg1_metrics:
                                     JetClass-II at 188 classes; topcheck_bench_metrics
  index.json                         {path: sha256} of every file above, as pulled: checked against
                                     them, and the source the macros counted over many files cite

Outputs:
  tables/mlst_probes.tex     linear probes at the largest training set, every task
  tables/mlst_probes_n.tex   the same at every training-set size
  tables/mlst_pairs.tex      binary probes between JetClass-II classes, merged or kept apart
  tables/mlst_mlp.tex        nonlinear probes on every task a conclusion rests on
  tables/mlst_fill.tex       JetClass-II probes with the downstream data's missing features zeroed
  tables/mlst_mass.tex       mass resolution: mass node, frozen features, uncorrected mass
  tables/mlst_sculpt.tex     QCD mass sculpting
  tables/mlst_ft.tex         full fine-tuning against the training-set size, every dataset
  tables/mlst_ft_rej.tex     background rejection after fine-tuning on the full training set
  tables/compute.tex         GPU types and hours of every run
and the macros the text quotes (Mlst*, Compute*).
"""
from __future__ import annotations

import collections
import importlib.util
import csv
import json
import math
import pathlib
import re

import numpy as np

_V4_SPEC = importlib.util.spec_from_file_location("mlst_v4", pathlib.Path(__file__).resolve().parent / "mlst_v4.py")
V4 = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(V4)

MAIN = ("L188", "L162", "R42_Q1", "R16_Q1")
MASS = ("L162_MASS", "R16_Q1_MASS")
RAND = tuple(f"RAND2_p{k}" for k in range(1, 6))
FLAV = ("FLAV_F0", "FLAV_F1", "FLAV_F1R")
INIT = "INIT"
INIT_MODELS = ("init-s1", "init-s2", "init-s3")
# the randomly initialized networks with their BatchNorm statistics recomputed on the training jets
# of the best70 twins (experiments/MTX/bn_twins_v2.py --init): fits_v3_initbn, checkpoint init_bn
INIT_BN = "INIT_BN"
INIT_BN_MODELS = ("initbn-s1", "initbn-s2", "initbn-s3")


def _ck(model: str) -> str:
    """The checkpoint a model is reported at."""
    return "init" if model.startswith("init-") else "init_bn" if model.startswith("initbn-") else PRIMARY
LABEL = {"L188": r"\Kfull{} classes", "L162": r"\Kres{} classes", "R42_Q1": r"\Kmid{} classes",
         "R16_Q1": r"\Kcoarse{} classes", "L162_MASS": r"\Kres{} classes, mass regression",
         "R16_Q1_MASS": r"\Kcoarse{} classes, mass regression", "FLAV_F0": "random partition, no flavor boundary",
         "FLAV_F1": r"random partition, $b$-quark boundary",
         "FLAV_F1R": "random partition, random flavor boundary", INIT: "randomly initialized network",
         INIT_BN: "randomly initialized network, BatchNorm recomputed",
         **{f"RAND2_p{k}": f"random partition {k}" for k in range(1, 6)}}
PRIMARY = "best70"
FITS, MLP_FITS = "linprobe/fits_v3", "linprobe/fits_v3_mlp"
# 2026-10-30: the nonlinear probe of every run on every task a conclusion rests on (supersedes
# MLP_FITS model by model), and the BatchNorm-recomputed random networks, linear and nonlinear
MLP_FITS_V4, FITS_INITBN, MLP_FITS_INITBN = ("linprobe/fits_v3_mlp2", "linprobe/fits_v3_initbn",
                                             "linprobe/fits_v3_mlp2_initbn")
# (dataset key, metric, column heading)
TASKS = (("jc2_188", "macro_auc_ovr", "JetClass-II"), ("jc2_188", "macro_auc_resonant", "resonant"),
         ("jc2_188", "macro_auc_qcd", "QCD"), ("jc1", "macro_auc_ovr", "JetClass"),
         ("top", "auc", "top"), ("qg", "auc", "q/g"), ("qg_herwig", "auc", "q/g, Herwig"))
# the training-size table: these tasks and the JetClass-II macro over the classes present in training
TASKS_N = (TASKS[0], ("jc2_188", "macro_auc_trained", "JetClass-II, classes present in training"),
           TASKS[2], TASKS[3], TASKS[4], TASKS[5])
# (key, heading, first side, second side): a side is native class names, or "qcd_with_b" /
# "qcd_with_c_no_b" (QCD classes by quark content, from the label map)
PAIRS = (("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", ("label_X_bc",), ("label_X_bq", "label_X_cs")),
         ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", ("label_X_bb",), ("label_X_cc",)),
         ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $ccqq$", ("label_X_YY_bbqq",), ("label_X_YY_ccqq",)),
         ("qcd_bb_vs_cc", r"QCD $bb$ vs $cc$", ("label_QCD_bb",), ("label_QCD_cc",)),
         ("qcd_b_vs_light", r"QCD with $b$ vs light", "qcd_with_b", ("label_QCD_light",)),
         ("qcd_c_vs_light", r"QCD with $c$, no $b$, vs light", "qcd_with_c_no_b", ("label_QCD_light",)))
PAIR_DS, PAIR_N = "jc2_188_pairs", "1000000"
# (task, heading, metric): the nonlinear probe's tasks, the multi-class ones from cell set v4
MLP_TASKS = (("jc2_188", "JetClass-II", "macro_auc_ovr"), ("jc2_188_res", "resonant", "macro_auc_resonant"),
             ("jc2_188_qcd", "QCD", "macro_auc_qcd"), ("jc1", "JetClass", "macro_auc_ovr"),
             ("top", "top", "auc"), ("qg", "q/g", "auc"), ("qg_herwig", "q/g, Herwig", "auc"),
             ("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", "auc"), ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", "auc"),
             ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $ccqq$", "auc"), ("qcd_bb_vs_cc", r"QCD $bb$ vs $cc$", "auc"),
             ("qcd_b_vs_light", r"QCD with $b$ vs light", "auc"),
             ("qcd_c_vs_light", r"QCD with $c$, no $b$, vs light", "auc"))
MASKS = ("TopReference", "EnergyFlowQG")
# Fine-tuning: (task, read-out file, data set inside it, metric, column heading)
FT_TAGS = ("best70",)
FT_TASKS = (("jc2", "leg188_leg1_metrics", None, "macro_auc_ovr", "JetClass-II"),
            ("jc2_qcd", "leg188_leg1_metrics", None, "macro_auc_qcd", "JetClass-II, QCD classes"),
            ("jc1", "leg2_metrics", None, "macro_auc_ovr", "JetClass"),
            ("top", "bench_metrics", "top", "auc", "top"),
            ("qg", "bench_metrics", "qg", "auc", "q/g"),
            ("qg_herwig", "bench_metrics_herwig", "qg", "auc", "q/g, Herwig"))
FT_REJ = (("top", "r50"), ("top", "r30"), ("qg", "r50"), ("qg", "r30"), ("qg_herwig", "r50"), ("qg_herwig", "r30"))
SCRATCH = "scratch-v2"


def _paper(root: pathlib.Path) -> dict:
    """{arm: [run directories]} of configs/arms/v2_paper_runs.txt, plus INIT: the untrained trunks."""
    slug = {a["name"].lower().replace("_", ""): a["name"]
            for a in json.loads((root / "configs/arms/v2_grid.json").read_text())["arms"]}
    out: dict = {}
    for ln in (root / "configs/arms/v2_paper_runs.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            arm = slug[re.fullmatch(r"mtx-([a-z0-9]+)-s\d+", ln.strip()).group(1)]
            out.setdefault(arm, []).append(ln.strip())
    out[INIT] = list(INIT_MODELS)
    out[INIT_BN] = list(INIT_BN_MODELS)
    return out


def _fit_cells(x):
    """Every selected fit (a dict with C and converged) under a checkpoint's cells."""
    if isinstance(x, dict):
        if "C" in x and "converged" in x:
            yield x
            return
        for k, v in x.items():
            if k != "search":
                yield from _fit_cells(v)


def _load(d: pathlib.Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


# a probe that did not converge (linear_probe_v2.py: its gradient did not fall below
# CONVERGED_GRAD within MAX_ITER) carries this mark, and the build counts them (MlstProbeUnconverged)
UNCONVERGED = r"$^{*}$"


class Unconverged(float):
    pass


def _sep(s: str) -> str:
    """Thousands separated as fmt_pm does."""
    head, dot, tail = s.partition(".")
    if "e" in s or len(head.lstrip("-")) <= 3:
        return s
    return f"{int(head):,}".replace(",", "{,}") + dot + tail


def _lone(x: float) -> str:
    """A single run: three significant figures (make_tables.fmt_one's rule), trailing zeros kept;
    from 100 on an integer (1,800, not 1.80e+03)."""
    if abs(x) >= 100:
        return _sep(str(int(round(x, 2 - int(math.floor(math.log10(abs(x))))))))
    return _sep(f"{x:#.3g}".rstrip("."))


def _two(a: float, b: float, sep: str) -> str:
    """Two runs: both values, to the decimal that shows their difference to two digits, but never
    beyond four significant figures (2026-10-31). A standard deviation from two runs has one
    degree of freedom, so it is not printed."""
    d = abs(a - b)
    if d == 0:
        return _lone(a) + sep + _lone(b)
    cap = 3 - math.floor(math.log10(max(abs(a), abs(b))))
    dec = max(0, min(1 - math.floor(math.log10(d)), cap))
    return sep.join(_sep(f"{x:.{dec}f}") for x in (a, b))


def _cell(values: list, n_expected: int, fmt_pm, scale: float = 1.0, sep: str = ", ") -> str:
    """mean +- SD over three or more runs present; two runs print both values; a lone run
    prints alone; missing runs add a red marker."""
    flag = any(isinstance(x, Unconverged) for x in values)
    v = [x * scale for x in values if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) > 2:
        body = fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    elif len(v) == 2:
        body = _two(v[0], v[1], sep)
    else:
        body = _lone(v[0])
    if flag:
        body += UNCONVERGED
    if len(v) < n_expected:
        body += rf" \pending{{{len(v)} of {n_expected}}}"
    return body


def _row(arm: str, runs: dict) -> str:
    """The row label; a random partition, pretrained with a single seed, also names that seed,
    since its comparisons with other partitions are not paired by seed."""
    if arm in RAND and len(runs[arm]) == 1:
        return f"{LABEL[arm]}, seed {runs[arm][0].rsplit('-s', 1)[1]}"
    return LABEL[arm]


UNCONVERGED_SEEN: set = set()


def _probe(fits: dict, model: str, ds: str, n: str, metric: str):
    ck = _ck(model)
    c = fits.get(model, {}).get("cells", {}).get(ck, {}).get(ds, {}).get(n)
    if c is None:
        return None
    if not c["converged"]:
        UNCONVERGED_SEEN.add((model, ds, n))
        return Unconverged(c[metric])
    return c[metric]


def _largest(fits: dict, ds: str) -> str:
    sizes = {n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}
    return max(sizes, key=int)


def table_probes(fits, runs, fmt_pm, rows) -> str:
    head = " & ".join(h for _, _, h in TASKS)
    lines = [r"\begin{tabular}{l" + "c" * len(TASKS) + "}", r"\toprule",
             r"pretraining & " + head + r" \\",
             r" & " + " & ".join(f"{int(_largest(fits, ds)):,}".replace(",", "{,}") + " jets"
                                for ds, _, _ in TASKS) + r" \\", r"\midrule"]
    for arm in rows:
        cells = [_cell([_oma(_probe(fits, m, ds, _largest(fits, ds), met)) for m in runs[arm]],
                       len(runs[arm]), fmt_pm, 1e3) for ds, met, _ in TASKS]
        lines.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def bn_macros(em, fits, runs, index) -> None:
    """MlstBnMaxDev: the largest |ratio - 1|, in per cent, of (1 - AUC) with the BatchNorm statistics
    recomputed (best70_bn) over the checkpoint as trained (best70), over every pretrained run of the
    paper and every task of Table probes at its largest training set."""
    dev = []
    for arm, ms in runs.items():
        if arm in (INIT, INIT_BN):
            continue
        for m in ms:
            cells = fits.get(m, {}).get("cells", {})
            for ds, met, _ in TASKS:
                n = _largest(fits, ds)
                a, b = (cells.get(ck, {}).get(ds, {}).get(n) for ck in (PRIMARY, "best70_bn"))
                if a is None or b is None:
                    raise SystemExit(f"FATAL: {m} {ds} N={n}: no fit at {PRIMARY} or best70_bn")
                dev.append(abs((1 - b[met]) / (1 - a[met]) - 1))
    em.macro("MlstBnMaxDev", f"{100 * max(dev):.1f}\\%", index,
             "largest |(1 - AUC) at best70_bn / (1 - AUC) at best70 - 1| over the paper's pretrained runs and "
             "the tasks of Table probes, per cent")


def table_probes_n(fits, runs, fmt_pm, rows) -> str:
    out = []
    for ds, met, title in TASKS_N:
        sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
        out += [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
                rf"{title} & " + " & ".join(f"{int(n):,}".replace(",", "{,}") for n in sizes) + r" \\",
                r"\midrule"]
        for arm in rows:
            out.append(LABEL[arm] + " & " + " & ".join(
                _cell([_oma(_probe(fits, m, ds, n, met)) for m in runs[arm]], len(runs[arm]), fmt_pm, 1e3)
                for n in sizes) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


def load_ft(data: pathlib.Path) -> tuple:
    """(tag, {task: {init: {N: [record per fine-tuning seed]}}}, tasks not yet read out) of the
    first read-out tag with any of its files present, or (None, {}, all tasks). A task whose file
    is missing has no cells, and its table prints a red marker."""
    d = data / "ft_mlst"
    for tag in FT_TAGS:
        if any((d / f"{tag}_{f}.json").exists() for _, f, *_ in FT_TASKS):
            break
    else:
        return None, {}, [k for k, *_ in FT_TASKS]
    per, absent = {}, []
    for key, f, ds, _, _ in FT_TASKS:
        if not (d / f"{tag}_{f}.json").exists():
            per[key] = {}
            absent.append(key)
            continue
        doc = json.loads((d / f"{tag}_{f}.json").read_text())
        cells = doc["cells"]
        cells = cells if ds is None else cells[ds]
        per[key] = {init: {int(n[1:]): [rec for _, rec in sorted(seeds.items())]
                           for n, seeds in by_n.items() if re.fullmatch(r"N\d+", n)}
                    for init, by_n in cells.items()}
    return tag, merge_budget(per, data / "budget" / "budget_metrics.json"), absent


# 2026-10-31 (plan 3.1, 3.4): below the full set every fine-tuning is the fixed-budget recipe
# (experiments/FT/budget_metrics.py), which supersedes the earlier cells there; JetClass-II is the
# class-balanced task at every size. (budget tree, sub-record or None) per task; the full sets.
BUDGET_TREE = {"top": ("leg_top", None), "qg": ("leg_qg", None), "qg_herwig": ("leg_qg", "herwig"),
               "jc1": ("leg2", None), "jc2": ("leg_jc2b", None), "jc2_qcd": ("leg_jc2b", None)}
FULL_N = {"top": 1_200_000, "qg": 1_600_000, "qg_herwig": 1_600_000, "jc1": 1_000_000}
BUDGET_SCRATCH = "scratch"          # build_ft_budget_jobs.cell's directory for a from-scratch cell


def merge_budget(per: dict, budget: pathlib.Path) -> dict:
    """per with the superseded cells removed (below FULL_N; every natural-mix JetClass-II cell) and
    the fixed-budget cells added, the from-scratch ones under SCRATCH."""
    out = {key: {init: {n: v for n, v in by_n.items() if key in FULL_N and n >= FULL_N[key]}
                 for init, by_n in cells.items()} for key, cells in per.items()}
    if not budget.exists():
        return out
    cells = json.loads(budget.read_text())["cells"]
    for key, (tree, sub) in BUDGET_TREE.items():
        for k in sorted(c for c in cells if c.startswith(tree + "/")):
            _, init, n, seed = k.split("/")
            rec = cells[k][sub] if sub else cells[k]
            init = SCRATCH if init == BUDGET_SCRATCH else init
            out.setdefault(key, {}).setdefault(init, {}).setdefault(int(n[1:]), []).append(rec)
    return out


def _ft(per: dict, key: str, model: str, n: int, metric: str):
    """One pretraining run's fine-tuned value: its one fine-tuning seed."""
    recs = per[key].get(model[len("mtx-"):], {}).get(n)
    if recs is None:
        return None
    if len(recs) != 1:
        raise SystemExit(f"FATAL: {model} {key} N={n}: {len(recs)} fine-tuning seeds, expected one")
    return recs[0][metric]


def _sizes(per: dict, key: str) -> list:
    return sorted({n for init, by_n in per[key].items() if init != SCRATCH for n in by_n})


def _ft_rows(per, key, runs, rows) -> tuple:
    """The rows of a task: the hierarchy and mass-regression runs always (a missing run prints a
    marker); a random partition only where it was fine-tuned (the benchmarks, not JetClass-II)."""
    return tuple(a for a in rows if a in MAIN + MASS
                 or any(per[key].get(m[len("mtx-"):]) for m in runs[a]))


def table_ft(per, runs, fmt_pm, rows) -> str:
    out = []
    for key, _, _, met, title in FT_TASKS:
        sizes = _sizes(per, key)
        if not sizes:
            out += [r"\begin{tabular}{lc}", r"\toprule", rf"{title} & \pending{{not yet read out}} \\",
                    r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
            continue
        out += [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
                rf"{title} & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in sizes) + r" \\",
                r"\midrule"]
        for arm in _ft_rows(per, key, runs, rows):
            out.append(_row(arm, runs) + " & " + " & ".join(
                _cell([_oma(_ft(per, key, m, n, met)) for m in runs[arm]], len(runs[arm]), fmt_pm, 1e3)
                for n in sizes) + r" \\")
        ref = per[key].get(SCRATCH, {})
        out.append("from scratch & " + " & ".join(
            _cell([1 - r[met] for r in ref[n]], len(ref[n]), fmt_pm, 1e3) if n in ref else "--"
            for n in sizes) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


def table_ft_rej(per, runs, fmt_pm, rows) -> tuple:
    """Background rejection 1/eps_B at 50 % and 30 % signal efficiency, full training set;
    and the fewest background jets passing behind any value."""
    heads = {"top": "top", "qg": "q/g", "qg_herwig": "q/g, Herwig"}
    fewest = []

    def rej(rec, k):
        if rec[f"{k}_is_bound"]:
            raise SystemExit(f"FATAL: a rejection {k} is only a bound: {rec.get('run', rec)}")
        fewest.append(rec[f"{k}_n_bkg_pass"])
        return rec[k]

    lines = [r"\begin{tabular}{l" + "c" * len(FT_REJ) + "}", r"\toprule",
             "pretraining & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{heads[t]}}}" for t in heads) + r" \\",
             " & " + " & ".join(rf"$R_{{{k[1:]}}}$" for _, k in FT_REJ) + r" \\", r"\midrule"]
    for arm in _ft_rows(per, FT_REJ[0][0], runs, rows) + (SCRATCH,):
        cells = []
        for key, k in FT_REJ:
            n = _sizes(per, key)[-1]
            if arm == SCRATCH:
                ref = per[key].get(SCRATCH, {}).get(n, [])
                cells.append(_cell([rej(r, k) for r in ref], len(ref), fmt_pm) if ref else "--")
            else:
                vals = []
                for m in runs[arm]:
                    recs = per[key].get(m[len("mtx-"):], {}).get(n)
                    vals.append(None if recs is None else rej(recs[0], k))
                cells.append(_cell(vals, len(runs[arm]), fmt_pm))
        lines.append(("from scratch" if arm == SCRATCH else _row(arm, runs)) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), min(fewest)


def _power(v: str) -> str:
    """A learning rate as the paper prints it: 1e-4 -> 10^{-4}, 5e-4 -> 5 x 10^{-4}."""
    x = float(v)
    e = math.floor(math.log10(x))
    m = round(x / 10 ** e, 6)
    return rf"\ensuremath{{10^{{{e}}}}}" if m == 1 else rf"\ensuremath{{{m:g}\times10^{{{e}}}}}"


def _one(cells: list, key: str, what: str):
    vals = {c[key] for c in cells}
    if len(vals) != 1:
        raise SystemExit(f"FATAL: {what}: {key} takes {sorted(map(str, vals))}, expected one value")
    return vals.pop()


# weaver 0.4.17's default --lr-scheduler flat+decay with an lr_mult group (train.py optim): the
# 1x groups keep their rate, the multiplied ones (the new head) fall by 0.01 over the last
# max(1, int(0.3 * epochs)) epochs. The JetClass fine-tuning sets no scheduler, so it runs this.
FLAT_DECAY = {"share": "30\\%", "to": "1\\%"}
LEGS_SPEC = "experiments/FT/k8s/job-ft-v2-legs-best70-l188-s3-raunav.yaml"
BENCH_SPEC = "experiments/FT/k8s/job-ft-v2-bench-best70-l188-s1-top-raunav.yaml"


def ft_recipe(em, data: pathlib.Path, paper_inits: set, fmt_int, missing: list, words=str) -> None:
    """The fine-tuning recipe and GPUs the paper states, from every DONE cell's own manifest
    (ft_mlst/manifests.json): one value across the cells that share it, or the build stops."""
    src = data / "ft_mlst/manifests.json"
    if not src.exists():
        missing.append("fine-tuning manifests -- experiments/FIGS/data/v2/ft_mlst/manifests.json")
        return
    m = json.loads(src.read_text())
    pre = [c for k, c in m.items() if k.startswith("best70/") and c["init"] in paper_inits]
    ref = [c for k, c in m.items() if k.startswith("ref_v100/")]
    legs = [c for c in pre if c["leg"] in ("1", "2")]
    bench = [c for c in pre if c["leg"] in ("top", "qg")]
    for c in pre:
        if c["ckpt_rule"] != PRIMARY or c["ft_seed"] != "1":
            raise SystemExit(f"FATAL: a paper cell is not {PRIMARY} seed 1: {c}")
    em.macro("MlstFtLr", _power(_one(pre, "lr", "pretrained fine-tuning")), src, "cells[best70].lr")
    em.macro("MlstFtHeadMult", _one(pre, "head_lr_mult", "pretrained fine-tuning"), src, "cells[best70].head_lr_mult")
    em.macro("MlstFtWeightDecay", _one(pre, "weight_decay", "pretrained fine-tuning"), src, "cells[best70].weight_decay")
    em.macro("MlstFtBatch", _one(pre, "batch_size", "pretrained fine-tuning"), src, "cells[best70].batch_size")
    if bench:
        em.macro("MlstFtBenchEpochs", _one(bench, "epochs", "top and q/g"), src, "cells[best70, top/qg].epochs")
        if _one(bench, "lr_schedule", "top and q/g") != "constant":
            raise SystemExit("FATAL: the benchmark fine-tuning is not at a constant rate")
    for n, w in (("1000", "EThree"), ("10000", "EFour"), ("100000", "EFive"), ("1000000", "ESix")):
        cells = [c for c in legs if c["n_train"] == n]
        if cells:
            em.macro(f"MlstFtEpochs{w}", _one(cells, "epochs", f"JetClass legs at N={n}"), src,
                     f"cells[best70, leg 1/2, n_train {n}].epochs")
    spec = data.parents[3] / LEGS_SPEC
    if "--lr-scheduler" in spec.read_text():
        raise SystemExit(f"FATAL: {LEGS_SPEC} sets a scheduler; the flat+decay macros assume weaver's default")
    em.macro("MlstFtHeadDecayShare", FLAT_DECAY["share"], spec, "no --lr-scheduler: weaver 0.4.17 flat+decay, "
             "the head's last epochs")
    em.macro("MlstFtHeadDecayTo", FLAT_DECAY["to"], spec, "no --lr-scheduler: weaver 0.4.17 flat+decay, final factor")
    drop = set(re.findall(r"fc_params '\[\(\d+,\s*([0-9.]+)\)\]'", spec.read_text()))
    if len(drop) != 1:
        raise SystemExit(f"FATAL: {LEGS_SPEC}: the head's dropout takes {sorted(drop)}")
    em.macro("MlstFtHeadDropout", drop.pop(), spec, "fc_params dropout of the fine-tuning head")
    em.macro("MlstFtMinEpochJets", fmt_int(min(int(c["samples_per_epoch"]) for c in pre)), src,
             "cells[best70].samples_per_epoch (smallest)")
    jc2 = [c for c in legs if c["leg"] == "1"]
    if jc2:
        if _one(jc2, "val_rule", "JetClass-II fine-tuning") != "fixed-sample-whole-pass":
            raise SystemExit("FATAL: the JetClass-II fine-tuning does not validate on one fixed sample")
        em.macro("MlstFtValJcii", fmt_int(int(_one(jc2, "val_jets", "JetClass-II fine-tuning"))), src,
                 "cells[best70, leg 1].val_jets")
    # JetClass draws its validation jets anew each epoch at one size; top and q/g at two (val_for)
    val = set(re.findall(r"--samples-per-epoch-val (\d+)", spec.read_text()))
    bench_spec = data.parents[3] / BENCH_SPEC
    vf = re.search(r'val_for \(\) \{ if \[ "\$1" -le 10000 \]; then echo (\d+); else echo (\d+); fi; \}',
                   bench_spec.read_text())
    if len(val) != 1 or not vf or vf.group(1) not in val:
        raise SystemExit(f"FATAL: the validation sizes in {LEGS_SPEC} and {BENCH_SPEC} are not the ones the paper states")
    em.macro("MlstFtValSmall", fmt_int(int(vf.group(1))), spec, "JetClass --samples-per-epoch-val; top and q/g at N <= 1e4")
    em.macro("MlstFtValLarge", fmt_int(int(vf.group(2))), bench_spec, "top and q/g --samples-per-epoch-val at N > 1e4")
    gpus = sorted({c["gpu_device_name"] for c in pre})
    em.macro("MlstFtGpuTypes", " and ".join(gpu_name(g) for g in gpus), src, "cells[best70].gpu_device_name")
    em.macro("MlstFtCellsDone", fmt_int(len(pre)), src, "DONE paper-run fine-tuning cells (count)")
    if ref:
        em.macro("MlstFtRefLr", _power(_one(ref, "lr", "from-scratch references")), src, "cells[ref_v100].lr")
        em.macro("MlstFtRefSeedsWord", words(len({c["ft_seed"] for c in ref})), src, "cells[ref_v100].ft_seed (count)")
        em.macro("MlstFtRefGpuTypes", " and ".join(gpu_name(g) for g in sorted({c["gpu_device_name"] for c in ref})),
                 src, "cells[ref_v100].gpu_device_name")
        if _one(ref, "recipe", "from-scratch references") != "part-from-scratch":
            raise SystemExit("FATAL: a reference cell is not at ParT's from-scratch recipe")
        if any(c["weight_decay"] != "0" for c in ref if c["leg"] in ("1", "2")):
            raise SystemExit("FATAL: a JetClass or JetClass-II reference has weight decay; the paper says none")
        bref = [c for c in ref if c["leg"] in ("top", "qg")]
        em.macro("MlstFtRefWeightDecayBench", _one(bref, "weight_decay", "top and q/g references") if bref
                 else r"\pending{V100 references}", src, "cells[ref_v100, top/qg].weight_decay")
    else:
        missing.append("from-scratch references on V100 -- ref_v100 cells")
        for k in ("MlstFtRefLr", "MlstFtRefGpuTypes", "MlstFtRefSeedsWord", "MlstFtRefWeightDecayBench"):
            em.macro(k, r"\pending{V100 references}", src, "no ref_v100 cell yet")


def merge_status(root: pathlib.Path) -> dict:
    """{arm: {pair: "merged" | "partly" | "apart"}} from the label maps: merged when every class
    of the pair's first side shares a group with every class of its second side."""
    maps = root / "configs/labelmaps"
    group: dict = collections.defaultdict(dict)
    for f, cols in (("rung_label_maps.v1.csv", MAIN), ("rand_label_map.v2.csv", RAND),
                    ("flavour_pair_map.v2.csv", FLAV)):
        for r in csv.DictReader((maps / f).open()):
            for c in cols:
                group[c][r["class_name"]] = r[c]
    group["L162_MASS"], group["R16_Q1_MASS"] = group["L162"], group["R16_Q1"]
    names = list(group["L188"])
    out = {}
    for arm, g in group.items():
        out[arm] = {}
        for key, _, a, b in PAIRS:
            same = [g[x] == g[y] for x in _side(names, a) for y in _side(names, b)]
            out[arm][key] = "merged" if all(same) else "partly" if any(same) else "apart"
    return out


def _side(names: list, side) -> tuple:
    """A pair's side as native class names: QCD classes by quark content, or the names given."""
    qcd = [n for n in names if n.startswith("label_QCD_") and n != "label_QCD_light"]
    flav = lambda n: n.removeprefix("label_QCD_")
    if side == "qcd_with_b":
        return tuple(n for n in qcd if "b" in flav(n))
    if side == "qcd_with_c_no_b":
        return tuple(n for n in qcd if "c" in flav(n) and "b" not in flav(n))
    return side


def table_pairs(fits, runs, fmt_pm, status, rows) -> str:
    """1 - AUC (x 10^3) of the binary probes; a dagger marks a pair the vocabulary merges."""
    mark = {"merged": r"$^{\dagger}$", "partly": r"$^{\ddagger}$", "apart": ""}
    lines = [r"\begin{tabular}{l" + "c" * len(PAIRS) + "}", r"\toprule",
             r"pretraining & " + " & ".join(h for _, h, _, _ in PAIRS) + r" \\", r"\midrule"]
    for arm in rows:
        cells = []
        for key, _, _, _ in PAIRS:
            vals = []
            for m in runs[arm]:
                c = fits.get(m, {}).get("cells", {}).get(PRIMARY, {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
                vals.append(None if c is None else 1 - c["auc"])
            cells.append(_cell(vals, len(runs[arm]), fmt_pm, 1e3) + mark[status.get(arm, {}).get(key, "apart")])
        lines.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def table_mass(mass, runs, fmt_pm, rows) -> str:
    """sigma_eff of the MLP mass regression per configuration, and that of the target itself:
    the value a constant prediction gives, against which any information in the features is
    measured. The ridge regression is not shown: for every model pretrained without the mass
    regression its residuals are wider than the target's (the MSE fit widens the 68 % core)."""
    lines = [r"\begin{tabular}{lc}", r"\toprule",
             r"pretraining & $\sigma_{\mathrm{eff}}$ \\", r"\midrule"]
    targets = set()
    for arm in rows:
        vals = []
        for m in runs[arm]:
            d = mass.get(m)
            ck = _ck(m)
            vals.append(None if d is None else d["arms"][ck]["mlp"]["sigma_eff"])
            if d is not None:
                targets.add(round(d["arms"][ck]["target"]["sigma_eff"], 12))
        lines.append(LABEL[arm] + " & " + _cell(vals, len(runs[arm]), fmt_pm) + r" \\")
    if len(targets) != 1:
        raise SystemExit(f"FATAL: the mass target's sigma_eff differs between runs: {sorted(targets)}")
    lines += [r"\midrule", "constant prediction & " + _lone(targets.pop()) + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


# What the text compares, as (configuration, reference): the ratio of their 1 - AUC, or of their
# sigma_eff, taken run by run between runs of the same index -- they share the training stream
# and the initial trunk -- as mean +- SD over the indices both have (Sec. Uncertainties).
CMP_PROBES = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L188", INIT), ("L162", INIT),
              ("R42_Q1", INIT), ("R16_Q1", INIT), ("L162_MASS", INIT), ("R16_Q1_MASS", INIT),
              ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"),
              ("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"), ("FLAV_F0", "R16_Q1"),
              ("L188", INIT_BN), ("L162", INIT_BN), ("R42_Q1", INIT_BN), ("R16_Q1", INIT_BN),
              ("L162_MASS", INIT_BN), ("R16_Q1_MASS", INIT_BN), (INIT_BN, INIT),
              # flavor removal on top of the QCD merge: the 43- and 17-class levels over the 162-class one
              ("R42_Q1", "L162"), ("R16_Q1", "L162"))
CMP_PAIRS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("FLAV_F1", "FLAV_F0"),
             ("FLAV_F1R", "FLAV_F1"), ("FLAV_F1R", "FLAV_F0"), ("FLAV_F0", "R16_Q1"))
CMP_MLP = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L188", INIT), ("L162", INIT),
           ("R42_Q1", INIT), ("R16_Q1", INIT), ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"),
           ("FLAV_F0", "L188"), ("FLAV_F1", "FLAV_F0"), ("FLAV_F1R", "FLAV_F1"), ("FLAV_F1R", "FLAV_F0"),
           ("FLAV_F0", "R16_Q1"), ("L188", INIT_BN), ("L162", INIT_BN), ("R42_Q1", INIT_BN), ("R16_Q1", INIT_BN),
           (INIT_BN, INIT))
CMP_MASS = (("R16_Q1", "L188"), ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
WORD = {"L188": "Full", "L162": "Res", "R42_Q1": "Mid", "R16_Q1": "Coarse", "L162_MASS": "ResMass",
        "R16_Q1_MASS": "CoarseMass", INIT: "Init", INIT_BN: "InitBn", "FLAV_F0": "Fzero", "FLAV_F1": "Fone",
        "FLAV_F1R": "Foner", SCRATCH: "Scratch"}
TASK_WORD = {("jc2_188", "macro_auc_ovr"): "JcTwo", ("jc2_188", "macro_auc_resonant"): "JcTwoRes",
             ("jc2_188", "macro_auc_qcd"): "JcTwoQcd", ("jc2_188", "macro_auc_trained"): "JcTwoTrained",
             ("jc1", "macro_auc_ovr"): "JcOne", ("top", "auc"): "Top", ("qg", "auc"): "Qg",
             ("qg_herwig", "auc"): "QgHerwig"}
PAIR_WORD = {"bc_vs_bq_cs": "Bc", "bb_vs_cc": "Bb", "bbqq_vs_ccqq": "Bbqq", "qcd_bb_vs_cc": "QcdBb",
             "qcd_b_vs_light": "QcdB", "qcd_c_vs_light": "QcdC", "top": "Top", "qg": "Qg",
             "qg_herwig": "QgHerwig", "jc2_188": "JcTwo", "jc2_188_res": "JcTwoRes", "jc2_188_qcd": "JcTwoQcd",
             "jc1": "JcOne"}
FT_WORD = {"jc2": "JcTwo", "jc2_qcd": "JcTwoQcd", "jc1": "JcOne", "top": "Top", "qg": "Qg",
           "qg_herwig": "QgHerwig"}
SIZE_WORD = {1000: "EThree", 10000: "EFour", 100000: "EFive", 1000000: "ESix"}
TEXT_SEP = " and "      # two runs in a sentence: "1.165 and 1.166"


def _oma(x):
    """1 - AUC, keeping the mark of a probe that did not converge."""
    return None if x is None else Unconverged(1 - x) if isinstance(x, Unconverged) else 1 - x


def _ratio(runs, arm, ref, value, fmt_pm, sep=TEXT_SEP) -> str:
    """value(arm) / value(ref) run by run over the run indices both configurations have."""
    a = {m.rsplit("-s", 1)[1]: value(m) for m in runs[arm]}
    b = {m.rsplit("-s", 1)[1]: value(m) for m in runs[ref]}
    both = sorted(set(a) & set(b))
    r = []
    for k in both:
        if a[k] is None or b[k] is None:
            r.append(None)
        else:
            x = a[k] / b[k]
            r.append(Unconverged(x) if isinstance(a[k], Unconverged) or isinstance(b[k], Unconverged) else x)
    return _cell(r, len(both), fmt_pm, sep=sep)


def _rand_over(runs, ref: str, value) -> list:
    """Each random partition (one run) over the `ref` run of the same run index."""
    base = {m.rsplit("-s", 1)[1]: value(m) for m in runs[ref]}
    out = []
    for arm in RAND:
        for m in runs[arm]:
            x, y = value(m), base.get(m.rsplit("-s", 1)[1])
            out.append(None if x is None or y is None else x / y)
    return out


def comparisons(em, fits, runs, status, index, fmt_pm) -> None:
    """The probe macros the results text quotes: MlstCmp<task or pair><arm>Over<ref>, run by run;
    MlstCmp<task or pair>RandOverFull, each random partition (one run) over the 188-class run of
    its run index, mean +- SD over the partitions; MlstPairRand{Merged,Apart}<pair>, the binary
    probes' (1 - AUC) x 10^3 over the random partitions that merge the pair and over those that
    keep it apart (one run each; the resonant pairs only, since every partition merges QCD)."""
    for ds, met, _ in TASKS + (TASKS_N[1],):
        n = _largest(fits, ds)
        w = TASK_WORD[(ds, met)]
        for arm, ref in CMP_PROBES:
            em.macro(f"MlstCmp{w}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: _oma(_probe(fits, m, ds, n, met)), fmt_pm), index,
                     f"linear probe {ds} N={n}: (1 - {met}) of {arm} over {ref}, run by run")
        full = {m.rsplit("-s", 1)[1]: _oma(_probe(fits, m, ds, n, met)) for m in runs["L188"]}
        r = []
        for arm in RAND:
            for m in runs[arm]:
                x, y = _oma(_probe(fits, m, ds, n, met)), full.get(m.rsplit("-s", 1)[1])
                r.append(None if x is None or y is None else x / y)
        em.macro(f"MlstCmp{w}RandOverFull", _cell(r, len(r), fmt_pm, sep=TEXT_SEP), index,
                 f"linear probe {ds} N={n}: (1 - {met}) of each random partition over the 188-class "
                 "run of the same index, mean +- SD over the partitions")
        for ref in ("R42_Q1", "R16_Q1"):
            r = _rand_over(runs, ref, lambda m: _oma(_probe(fits, m, ds, n, met)))
            em.macro(f"MlstCmp{w}RandOver{WORD[ref]}", _cell(r, len(r), fmt_pm, sep=TEXT_SEP), index,
                     f"linear probe {ds} N={n}: (1 - {met}) of each random partition over the {ref} run of "
                     "the same index, mean +- SD over the partitions")

    def pair(m, key):
        c = fits.get(m, {}).get("cells", {}).get(PRIMARY, {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
        return None if c is None else 1 - c["auc"]
    for key, _, _, _ in PAIRS:
        for arm, ref in CMP_PAIRS:
            em.macro(f"MlstCmp{PAIR_WORD[key]}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: pair(m, key), fmt_pm), index,
                     f"binary probe {key}: (1 - AUC) of {arm} over {ref}, run by run")
        if key.startswith("qcd_"):
            continue
        full = {m.rsplit("-s", 1)[1]: pair(m, key) for m in runs["L188"]}
        r = []
        for arm in RAND:
            for m in runs[arm]:
                x, y = pair(m, key), full.get(m.rsplit("-s", 1)[1])
                r.append(None if x is None or y is None else x / y)
        em.macro(f"MlstCmp{PAIR_WORD[key]}RandOverFull", _cell(r, len(r), fmt_pm, sep=TEXT_SEP), index,
                 f"binary probe {key}: (1 - AUC) of each random partition over the 188-class run "
                 "of the same index, mean +- SD over the partitions")
        for side in ("merged", "apart"):
            ms = [m for arm in RAND if status[arm][key] == side for m in runs[arm]]
            em.macro(f"MlstPairRand{side.title()}{PAIR_WORD[key]}",
                     _cell([pair(m, key) for m in ms], len(ms), fmt_pm, 1e3, sep=TEXT_SEP),
                     index, f"binary probe {key}: (1 - AUC) x 10^3 over the {len(ms)} random partitions that "
                     + {"merged": "merge it", "apart": "keep it apart"}[side])
            em.macro(f"MlstPairRand{side.title()}{PAIR_WORD[key]}N", str(len(ms)), index,
                     f"random partitions that {'merge' if side == 'merged' else 'keep apart'} {key} (count)")


# ------------------------------------------------------------------ nonlinear probes
def _mlp(mlp: dict, model: str, task: str):
    """1 - AUC of the nonlinear probe; jc2_188_res / jc2_188_qcd are parts of the jc2_188 cell."""
    met = {t: m for t, _, m in MLP_TASKS}[task]
    c = mlp.get(model, {}).get("cells", {}).get(_ck(model), {}).get(task.removesuffix("_res").removesuffix("_qcd"))
    return None if c is None else 1 - c[met]


def table_mlp(mlp, runs, fmt_pm, rows) -> str:
    """(1 - AUC) x 10^3 of the nonlinear probe (one hidden layer, three seeds averaged) on the
    binary tasks at their largest training set."""
    out = []
    for part in (MLP_TASKS[:7], MLP_TASKS[7:]):
        out += [r"\begin{tabular}{l" + "c" * len(part) + "}", r"\toprule",
                r"pretraining & " + " & ".join(h for _, h, _ in part) + r" \\", r"\midrule"]
        for arm in rows:
            cells = [_cell([_mlp(mlp, m, t) for m in runs[arm]], len(runs[arm]), fmt_pm, 1e3) for t, _, _ in part]
            out.append(LABEL[arm] + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


def mlp_macros(em, mlp, runs, index, fmt_pm) -> None:
    """As comparisons(), for the nonlinear probe; MlstMlpCmp<task>RandOverFull over the partitions."""
    unconverged = [(m, t) for m, d in mlp.items() for ck in d["cells"].values() for t, c in ck.items()
                   if not c["mlp"]["all_converged"]]
    em.macro("MlstMlpUnconverged", str(len(unconverged)), index, "nonlinear probes that did not reach "
             "their plateau within the epoch cap (count)")
    for t, _, _ in MLP_TASKS:
        for arm, ref in CMP_MLP:
            em.macro(f"MlstMlpCmp{PAIR_WORD[t]}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: _mlp(mlp, m, t), fmt_pm), index,
                     f"nonlinear probe {t}: (1 - AUC) of {arm} over {ref}, run by run")
        full = {m.rsplit("-s", 1)[1]: _mlp(mlp, m, t) for m in runs["L188"]}
        r = []
        for arm in RAND:
            for m in runs[arm]:
                x, y = _mlp(mlp, m, t), full.get(m.rsplit("-s", 1)[1])
                r.append(None if x is None or y is None else x / y)
        em.macro(f"MlstMlpCmp{PAIR_WORD[t]}RandOverFull", _cell(r, len(r), fmt_pm, sep=TEXT_SEP), index,
                 f"nonlinear probe {t}: (1 - AUC) of each random partition over the 188-class run of its index")


# ------------------------------------------------------------------ masked inputs
def _fill(fill: dict, mask: str, model: str, met: str = "macro_auc_ovr"):
    ck = _ck(model)
    c = fill.get(mask, {}).get(model, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(PAIR_N)
    return None if c is None else 1 - c[met]


def table_fill(fits, fill, runs, fmt_pm, rows) -> str:
    """JetClass-II (188 classes) at 10^6 training jets: the probe on the features as computed,
    and with the features the top-tagging and the quark/gluon data lack set to zero."""
    heads = ("all features", "top-tagging features", "quark/gluon features")
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"pretraining & " + " & ".join(heads) + r" \\",
             r"\midrule"]
    for arm in rows:
        base = [_oma(_probe(fits, m, "jc2_188", PAIR_N, "macro_auc_ovr")) for m in runs[arm]]
        masked = [[_fill(fill, k, m) for m in runs[arm]] for k in MASKS]
        lines.append(LABEL[arm] + " & " + " & ".join(_cell(v, len(runs[arm]), fmt_pm, 1e3)
                                                     for v in [base] + masked) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def fill_macros(em, fits, fill, runs, index, fmt_pm) -> None:
    """MlstFill<mask><arm>: (1 - AUC) masked over unmasked, run by run; MlstFill<mask><arm>Over<ref>:
    masked over masked between label sets, run by run."""
    for k, kw in zip(MASKS, ("Top", "Qg")):
        for arm in MAIN + (INIT,):
            r = [None if (x := _fill(fill, k, m)) is None or (y := _oma(_probe(fits, m, "jc2_188", PAIR_N,
                                                                              "macro_auc_ovr"))) is None
                 else x / y for m in runs[arm]]
            em.macro(f"MlstFill{kw}{WORD[arm]}", _cell(r, len(runs[arm]), fmt_pm, sep=TEXT_SEP), index,
                     f"JetClass-II 188 at 1e6: (1 - AUC) with the {k} features zeroed over without, {arm}")
        for arm, ref in (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L188", INIT)):
            em.macro(f"MlstFill{kw}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: _fill(fill, k, m), fmt_pm), index,
                     f"JetClass-II 188 at 1e6, {k} features zeroed: (1 - AUC) of {arm} over {ref}")


# ------------------------------------------------------------------ mass
def _mass(mass: dict, model: str, what: str, key: str = "sigma_eff", group: str | None = None):
    """what: 'mlp' (frozen features, median-centred target), 'head' (the mass node, raw target),
    'uncorrected' (the soft-drop mass), 'target' (the centred target's own spread)."""
    d = mass.get(model)
    if d is None:
        return None
    ck = _ck(model)
    if what == "uncorrected":
        r = d["uncorrected"]
    elif what == "head":
        r = d.get("heads", {}).get(ck)
        if r is None:
            return None
    elif what == "mlp":
        r = d["arms"][ck]["mlp"]
        if group:
            return d["arms"][ck]["binned"]["mlp"]["group"][group].get(key)
    else:
        r = d["arms"][ck]["target"]
    if group:
        return r["binned"]["group"][group].get(key)
    return r[key]


def table_mass(mass, runs, fmt_pm, rows) -> str:
    """sigma_eff of ln(m_reg / m_gen) on all, resonant and QCD test jets, and the median response
    m_reg / m_gen: the uncorrected soft-drop mass, the mass node of the models pretrained with
    the mass regression, and the MLP on frozen features (against the class-median-centred target,
    so its response is centred by construction)."""
    cols = ((None, "sigma_eff"), ("resonant", "sigma_eff"), ("qcd", "sigma_eff"), (None, "median_response"))
    head = (r"pretraining & $\sigma_{\mathrm{eff}}$ & resonant & QCD & "
            r"$m_{\mathrm{reg}}/m_{\mathrm{SD}}^{\mathrm{gen}}$ \\")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", head, r"\midrule"]
    unc = {round(_mass(mass, m, "uncorrected"), 12) for m in mass}
    if len(unc) != 1:
        raise SystemExit(f"FATAL: the uncorrected mass's sigma_eff differs between files: {sorted(unc)}")
    any_m = next(iter(mass))
    lines.append("uncorrected soft-drop mass & " + " & ".join(
        _lone(_mass(mass, any_m, "uncorrected", k, g)) for g, k in cols) + r" \\")
    lines.append(r"\midrule \multicolumn{5}{l}{mass node} \\")
    for arm in [a for a in rows if a in MASS]:
        lines.append(LABEL[arm] + " & " + " & ".join(
            _cell([_mass(mass, m, "head", k, g) for m in runs[arm]], len(runs[arm]), fmt_pm) for g, k in cols)
            + r" \\")
    lines.append(r"\midrule \multicolumn{5}{l}{MLP on frozen features, with the true class's median added back} \\")
    tgt = {round(_mass(mass, m, "target"), 12) for m in mass}
    if len(tgt) != 1:
        raise SystemExit(f"FATAL: the class-median target's sigma_eff differs between files: {sorted(tgt)}")
    # the reference of this block: the class median alone, which the MLP rows are read against
    lines.append(r"class median alone & " + _lone(tgt.pop()) + r" & -- & -- & -- \\")
    for arm in rows:
        vals = [_cell([_mass(mass, m, "mlp", k, g) for m in runs[arm]], len(runs[arm]), fmt_pm)
                for g, k in cols[:3]]
        lines.append(LABEL[arm] + " & " + " & ".join(vals) + r" & -- \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def mass_macros(em, mass, runs, index, fmt_pm) -> None:
    any_m = next(iter(mass))
    em.macro("MlstMassUncorrected", _lone(_mass(mass, any_m, "uncorrected")), index,
             "sigma_eff of the uncorrected soft-drop mass, ln(m_SD / m_gen), test jets")
    for g in ("resonant", "qcd"):
        em.macro(f"MlstMassUncorrected{g.title()}", _lone(_mass(mass, any_m, "uncorrected", group=g)), index,
                 f"sigma_eff of the uncorrected soft-drop mass, {g} jets")
    em.macro("MlstMassUncorrectedResponse", _lone(_mass(mass, any_m, "uncorrected", "median_response")),
             index, "median m_SD / m_gen of the uncorrected soft-drop mass")
    unc = _mass(mass, any_m, "uncorrected")
    for arm in MASS:
        em.macro(f"MlstMassHead{WORD[arm]}OverUncorr",
                 _cell([None if (h := _mass(mass, m, "head")) is None else h / unc for m in runs[arm]],
                       len(runs[arm]), fmt_pm, sep=TEXT_SEP), index,
                 f"sigma_eff of the mass node of {arm} over the uncorrected soft-drop mass's")
        em.macro(f"MlstMassHead{WORD[arm]}Response",
                 _cell([_mass(mass, m, "head", "median_response") for m in runs[arm]], len(runs[arm]),
                       fmt_pm, sep=TEXT_SEP), index, f"median m_reg / m_gen of the mass node of {arm}")
        for g in ("resonant", "qcd"):
            em.macro(f"MlstMassHead{WORD[arm]}{g.title()}",
                     _cell([_mass(mass, m, "head", group=g) for m in runs[arm]], len(runs[arm]), fmt_pm,
                           sep=TEXT_SEP), index, f"sigma_eff of the mass node of {arm}, {g} jets")
    for arm, ref in CMP_MASS:
        em.macro(f"MlstCmpMlp{WORD[arm]}Over{WORD[ref]}",
                 _ratio(runs, arm, ref, lambda m: _mass(mass, m, "mlp"), fmt_pm), index,
                 f"mass resolution, MLP on frozen features: sigma_eff of {arm} over {ref}, run by run")
    for arm in (INIT, "L188", "R16_Q1"):
        em.macro(f"MlstMassMlp{WORD[arm]}OverTarget",
                 _cell([None if m not in mass else _mass(mass, m, "mlp") / _mass(mass, m, "target")
                        for m in runs[arm]], len(runs[arm]), fmt_pm, sep=TEXT_SEP), index,
                 f"MLP sigma_eff of {arm} over the spread of the class-median-centred target")


def loss_share_macros(em, comp, runs, index, fmt_pm) -> None:
    """The realised share of the mass-regression term in the training loss, lambda * L_reg /
    (L_cls + lambda * L_reg), mean over the epochs of each run (compute/<run>.json)."""
    for arm in MASS:
        sh = []
        for m in runs[arm]:
            ep = comp.get(m, {}).get("epochs", [])
            v = [e["loss_share_reg"] for e in ep if e.get("loss_share_reg") is not None]
            sh.append(100 * float(np.mean(v)) if v else None)
        em.macro(f"MlstMassLossShare{WORD[arm]}", _cell(sh, len(runs[arm]), fmt_pm, sep=TEXT_SEP), index,
                 f"share of the mass-regression term in the training loss of {arm}, mean over epochs, percent")


# ------------------------------------------------------------------ sculpting
SCULPT_EFFS = ("0.1", "0.01")


def _sculpt(sculpt: dict, model: str, eff: str):
    d = sculpt.get(model)
    return None if d is None else d["working_points"][eff]["jsd"]


def _sculpt_floor(sculpt: dict, eff: str) -> float:
    """The divergence a cut that does not sculpt still shows: the expected Jensen-Shannon divergence
    (base 2) between a histogram of n jets drawn at random from the QCD mass distribution and that
    distribution, (k - 1) / (8 n ln 2) to leading order (chi^2 / (8 n ln 2), E[chi^2] = k - 1), with k
    the occupied bins and n the jets passing the cut. One value across every run, or the build stops."""
    vals = {round((int(np.count_nonzero(d["hist_before"])) - 1)
                  / (8 * d["working_points"][eff]["n_pass"] * math.log(2)), 12) for d in sculpt.values()}
    if len(vals) != 1:
        raise SystemExit(f"FATAL: sculpting floor at {eff} takes {sorted(vals)} across the runs")
    return vals.pop()


def table_sculpt(sculpt, runs, fmt_pm, rows) -> str:
    """Jensen-Shannon divergence (x 10^3) between the soft-drop mass distributions of all QCD test
    jets and of those passing a cut on 1 - p_QCD at the stated QCD efficiency, and below the runs the
    divergence that a random selection of as many QCD jets shows (_sculpt_floor)."""
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"pretraining & " + " & ".join(rf"$\epsilon_{{\mathrm{{QCD}}}} = {float(e):g}$" for e in SCULPT_EFFS)
             + r" \\", r"\midrule"]
    for arm in rows:
        lines.append(LABEL[arm] + " & " + " & ".join(
            _cell([_sculpt(sculpt, m, e) for m in runs[arm]], len(runs[arm]), fmt_pm, 1e3) for e in SCULPT_EFFS)
            + r" \\")
    if sculpt:
        lines += [r"\midrule", "random selection of as many QCD jets & "
                  + " & ".join(_lone(1e3 * _sculpt_floor(sculpt, e)) for e in SCULPT_EFFS) + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def size_macros(em, fits: dict, fmt_int, index) -> None:
    """Test, validation and largest training sizes of the probe tasks, from the fits (one value
    across every model, or the build stops)."""
    def one(ds, key, n=None):
        vals = set()
        for m, d in fits.items():
            cells = d["cells"].get(_ck(m), {}).get(ds, {})
            if cells:
                vals.add(cells[n or max(cells, key=int)][key])
        if len(vals) != 1:
            raise SystemExit(f"FATAL: {ds} {key} takes {sorted(vals)} across the fits")
        return vals.pop()
    for ds, w in (("jc1", "Jci"), ("top", "Top"), ("qg", "Qg")):
        em.macro(f"LpNTest{w}", fmt_int(one(ds, "n_jets")), index, f"{ds} test jets the probes score")
        em.macro(f"LpNTrainMax{w}", fmt_int(one(ds, "n_train")), index, f"{ds} largest training subset")
    em.macro("LpNTestAucJci", fmt_int(one("jc1", "n_jets_auc")), index, "JetClass test jets of the macro AUC")
    nval = {c["n_val"] for f in fits.values() for c in f["cells"].get(_ck("mtx-"), {}).get(PAIR_DS, {})
            .get(PAIR_N, {}).values()}
    if nval:
        em.macro("LpPairNValMin", fmt_int(min(nval)), index, "validation jets of the pair probe with the fewest")
        em.macro("LpPairNValMax", fmt_int(max(nval)), index, "validation jets of the pair probe with the most")
    em.macro("LpNValJci", fmt_int(one("jc1", "n_val")), index, "JetClass probe validation jets")


def composition_macros(em, data: pathlib.Path, root: pathlib.Path, missing: list) -> None:
    """The class mix of the JetClass-II training subset at 10^6 and of the test jets: QCD and
    two-prong shares (per cent), and the smallest class counts."""
    src = data / "jc2_test/counts.json"
    man = root / "experiments/FT/data/jc2_v2_manifest.json"
    if not src.exists():
        missing.append("JetClass-II test composition -- experiments/FIGS/data/v2/jc2_test/counts.json")
        return
    names = [r["class_name"] for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())]
    fam = ["qcd" if n.startswith("label_QCD") else "multi" if n.startswith("label_X_YY") else "two" for n in names]
    test = json.loads(src.read_text())
    train = json.loads(man.read_text())["class_coverage"][f"train_N{PAIR_N}_s1.parquet"]["counts"]

    def share(c, f):
        return 100 * sum(x for x, g in zip(c, fam) if g == f) / sum(c)
    for w, c, s in (("Test", test["counts"], src), ("Train", train, man)):
        em.macro(f"LpJcii{w}QcdPercent", f"{share(c, 'qcd'):.1f}\\%", s, "QCD share of the jets, per cent")
        em.macro(f"LpJcii{w}TwoProngPercent", f"{share(c, 'two'):.1f}\\%", s, "two-prong share of the jets, per cent")
    em.macro("LpJciiTestMinAucJets", str(min(test["counts_auc"])), src, "fewest jets of a class among the macro-AUC rows")
    em.macro("LpJciiTrainMinJets", str(min(train)), man, "fewest jets of a class in the 10^6 training subset")


def method_macros(em, root: pathlib.Path, words=str) -> None:
    """Settings the methods text states, read from the scripts that use them."""
    sc = root / "experiments/EVAL/mass_sculpting.py"
    m = re.search(r"^WORKING_POINTS = \(([^)]*)\)", sc.read_text(), re.M)
    effs = sorted((float(x) for x in m.group(1).split(",") if x.strip()), reverse=True)
    if [str(e) for e in effs] != list(SCULPT_EFFS):
        raise SystemExit(f"FATAL: {sc} WORKING_POINTS {effs} are not the table's {SCULPT_EFFS}")
    for name, e in zip(("SculptEffHigh", "SculptEffLow"), effs):
        em.macro(name, f"{100 * e:g}\\%", sc, "WORKING_POINTS", "QCD efficiency of a sculpting cut")
    pr = root / "experiments/EVAL/probe.py"
    src = pr.read_text()
    hidden = set(re.findall(r"torch\.nn\.Linear\(tr\.shape\[1\], (\d+)\)", src))
    seeds = re.search(r"^MLP_SEEDS = \(([^)]*)\)", src, re.M)
    if len(hidden) != 1 or not seeds:
        raise SystemExit(f"FATAL: {pr}: the MLP probe's hidden width or seeds not found")
    em.macro("LpMlpHidden", hidden.pop(), pr, "fit_mlp hidden layer", "units of the MLP probe's hidden layer")
    em.macro("LpMlpSeedsWord", words(len([s for s in seeds.group(1).split(",") if s.strip()])), pr, "MLP_SEEDS",
             "seeds of the MLP probe whose scores are averaged")
    mr = root / "experiments/EVAL/mass_resolution.py"
    m = re.search(r"^MIN_TRAIN_PER_CLASS = (\d+)", mr.read_text(), re.M)
    em.macro("MassMinTrainPerClass", m.group(1), mr, "MIN_TRAIN_PER_CLASS",
             "fewest training jets of a class the mass regression keeps")


def sculpt_macros(em, sculpt, runs, index, fmt_pm) -> None:
    for e, ew in zip(SCULPT_EFFS, ("Ten", "One")):
        n_pass = {d["working_points"][e]["n_pass"] for d in sculpt.values()}
        if len(n_pass) != 1:
            raise SystemExit(f"FATAL: {sorted(n_pass)} QCD jets pass at {e} across the runs")
        em.macro(f"MlstSculptNPass{ew}", f"{n_pass.pop():,}".replace(",", "{,}"), index,
                 f"QCD test jets passing the cut at QCD efficiency {e}, every run")
        em.macro(f"MlstSculptFloor{ew}", _lone(1e3 * _sculpt_floor(sculpt, e)), index,
                 f"JSD x 10^3 expected for a random selection of as many QCD jets as pass the cut at QCD "
                 f"efficiency {e}: (k - 1) / (8 n ln 2), k occupied mass bins, n jets passing")
        for arm in MAIN + MASS:
            em.macro(f"MlstSculpt{WORD[arm]}{ew}", _cell([_sculpt(sculpt, m, e) for m in runs[arm]],
                                                        len(runs[arm]), fmt_pm, 1e3, sep=TEXT_SEP), index,
                     f"JSD x 10^3 of the QCD m_SD distribution at QCD efficiency {e}, {arm}")
        for arm, ref in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"), ("R16_Q1", "L188"), ("L162", "L188")):
            em.macro(f"MlstSculpt{WORD[arm]}Over{WORD[ref]}{ew}",
                     _ratio(runs, arm, ref, lambda m: _sculpt(sculpt, m, e), fmt_pm), index,
                     f"JSD at QCD efficiency {e}: {arm} over {ref}, run by run")


# ------------------------------------------------------------------ fine-tuning comparisons
def _over_scratch(v: list, n_expected: int, ref: list, fmt_pm) -> str:
    """Each run's (1 - AUC) over the mean of the from-scratch seeds. With three or more runs: the
    mean ratio, and as its uncertainty the SD over the runs and the relative standard error of the
    from-scratch mean added in quadrature (the reference's own seeds and training subsets vary
    too); fewer runs print as _cell does."""
    sc = float(np.mean(ref))
    r = [None if x is None else x / sc for x in v]
    have = [x for x in r if x is not None]
    if len(have) < 3 or len(ref) < 2:
        return _cell(r, n_expected, fmt_pm, sep=TEXT_SEP)
    m = float(np.mean(have))
    rel = float(np.std(ref, ddof=1)) / math.sqrt(len(ref)) / sc
    body = fmt_pm(m, float(math.hypot(np.std(have, ddof=1), m * rel)))
    return body + (rf" \pending{{{len(have)} of {n_expected}}}" if len(have) < n_expected else "")


def ft_macros(em, per, runs, index, fmt_pm) -> None:
    """MlstFtCmp<task><arm>OverScratch<size>: each run's (1 - AUC) over the mean of the
    from-scratch seeds, mean +- (SD over the runs (+) standard error of the from-scratch mean);
    MlstFtCmp<task><arm>OverFull<size>: run by run."""
    for key, _, _, met, _ in FT_TASKS:
        sizes = _sizes(per, key)
        for n in sizes:
            sw = SIZE_WORD.get(n, "Max") if n != sizes[-1] or n in SIZE_WORD else "Max"
            ref = per[key].get(SCRATCH, {}).get(n)
            for arm in MAIN + MASS:
                v = [_oma(_ft(per, key, m, n, met)) for m in runs[arm]]
                em.macro(f"MlstFtCmp{FT_WORD[key]}{WORD[arm]}OverScratch{sw}",
                         _over_scratch(v, len(runs[arm]), [1 - r[met] for r in ref], fmt_pm) if ref
                         else r"\pending{scratch}", index,
                         f"fine-tuning {key} N={n}: (1 - {met}) of each {arm} run over the from-scratch mean, "
                         "SD over the runs and standard error of that mean in quadrature")
            for arm, refa in (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"),
                              ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")):
                em.macro(f"MlstFtCmp{FT_WORD[key]}{WORD[arm]}OverFull{sw}" if refa == "L188" else
                         f"MlstFtCmp{FT_WORD[key]}{WORD[arm]}Over{WORD[refa]}{sw}",
                         _ratio(runs, arm, refa, lambda m: _oma(_ft(per, key, m, n, met)), fmt_pm), index,
                         f"fine-tuning {key} N={n}: (1 - {met}) of {arm} over {refa}, run by run")
            full = {m.rsplit("-s", 1)[1]: _oma(_ft(per, key, m, n, met)) for m in runs["L188"]}
            r = [None if (x := _oma(_ft(per, key, m, n, met))) is None or full.get(m.rsplit("-s", 1)[1]) is None
                 else x / full[m.rsplit("-s", 1)[1]] for a in RAND for m in runs[a]]
            if any(x is not None for x in r):
                em.macro(f"MlstFtCmp{FT_WORD[key]}RandOverFull{sw}", _cell(r, len(r), fmt_pm, sep=TEXT_SEP), index,
                         f"fine-tuning {key} N={n}: (1 - {met}) of each random partition over the 188-class run "
                         "of its run index, mean +- SD over the partitions")
                rc = _rand_over(runs, "R16_Q1", lambda m: _oma(_ft(per, key, m, n, met)))
                em.macro(f"MlstFtCmp{FT_WORD[key]}RandOverCoarse{sw}", _cell(rc, len(rc), fmt_pm, sep=TEXT_SEP),
                         index, f"fine-tuning {key} N={n}: (1 - {met}) of each random partition over the R16_Q1 run "
                         "of its run index, mean +- SD over the partitions")
    # the values ParT reports (Tables 5 and 6), on the full training set: the AUC and R_30 of the
    # 188-class models and of the network trained from scratch, for the comparison in the text
    for key in ("top", "qg"):
        if not _sizes(per, key):
            continue
        n = _sizes(per, key)[-1]
        ref = per[key].get(SCRATCH, {}).get(n, [])
        for what, ww in (("auc", "Auc"), ("r30", "Rthirty")):
            em.macro(f"MlstFt{FT_WORD[key]}{ww}Full",
                     _cell([_ft(per, key, m, n, what) for m in runs["L188"]], len(runs["L188"]), fmt_pm,
                           sep=TEXT_SEP), index, f"fine-tuning {key} N={n}: {what} of the 188-class runs")
            em.macro(f"MlstFt{FT_WORD[key]}{ww}Scratch",
                     _cell([r[what] for r in ref], len(ref), fmt_pm, sep=TEXT_SEP) if ref else r"\pending{scratch}",
                     index, f"fine-tuning {key} N={n}: {what} of the network trained from scratch, every seed")


def topcheck_macros(em, data: pathlib.Path, per, index, fmt_pm, missing: list) -> None:
    """The recipe check at 10^4 top-tagging jets: (1 - AUC) x 10^3 after 100 epochs, the
    188-class runs and the reference from scratch, beside the 20 epochs of the main read-out."""
    f = data / "ft_mlst" / "topcheck_bench_metrics.json"
    if not f.exists():
        missing.append("top-tagging recipe check -- ft_mlst/topcheck_bench_metrics.json")
        for k in ("MlstTopCheckFullLong", "MlstTopCheckScratchLong", "MlstTopCheckFullShort",
                  "MlstTopCheckScratchShort"):
            em.macro(k, r"\pending{recipe check}", index, "the recipe check is not read out yet")
        return
    cells = json.loads(f.read_text())["cells"]["top"]
    full = [1 - c["N10000"]["s1"]["auc"] for i, c in cells.items() if i.startswith("l188-")]
    sc = [1 - r["auc"] for s, r in sorted(cells.get(SCRATCH, {}).get("N10000", {}).items())]
    em.macro("MlstTopCheckFullLong", _cell(full, 3, fmt_pm, 1e3, sep=TEXT_SEP), f,
             "top tagging at 1e4 jets, 100 epochs: (1 - AUC) x 10^3 of the 188-class runs")
    em.macro("MlstTopCheckScratchLong", _cell(sc, 3, fmt_pm, 1e3, sep=TEXT_SEP), f,
             "top tagging at 1e4 jets, 100 epochs: (1 - AUC) x 10^3 from scratch, three seeds")
    short = per.get("top", {})
    em.macro("MlstTopCheckFullShort", _cell([None if (r := short.get(f"l188-s{k}", {}).get(10000)) is None
                                             else 1 - r[0]["auc"] for k in (1, 2, 3)], 3, fmt_pm, 1e3,
                                            sep=TEXT_SEP), index,
             "top tagging at 1e4 jets, the main read-out's epochs: (1 - AUC) x 10^3 of the 188-class runs")
    em.macro("MlstTopCheckScratchShort", _cell([1 - r["auc"] for r in short.get(SCRATCH, {}).get(10000, [])], 3,
                                               fmt_pm, 1e3, sep=TEXT_SEP), index,
             "top tagging at 1e4 jets, the main read-out's epochs: (1 - AUC) x 10^3 from scratch")


def gpu_name(device: str) -> str:
    return device.replace("NVIDIA ", "").replace("GeForce ", "").replace("Tesla ", "").replace("-SXM2-32GB", "")


def table_compute(comp, runs, fmt_int) -> tuple[str, dict]:
    """Per run: hours on each GPU type (completed epochs, training and validation passes)."""
    lines = [r"\begin{tabular}{llr}", r"\toprule", r"run & GPU types (hours) & GPU-hours \\", r"\midrule"]
    hours = {}
    for arm, ms in runs.items():
        if arm in (INIT, INIT_BN):
            continue
        per = collections.Counter()
        for m in ms:
            if m not in comp:
                lines.append(rf"{LABEL[arm]}, seed {m[-1]} & \pending{{training}} & \\")
                continue
            h = collections.Counter()
            for e in comp[m]["epochs"]:
                h[gpu_name(e["device"])] += (e["train_seconds"] + e["val_seconds"]) / 3600
            per.update(h)
            types = ", ".join(f"{g} ({h[g]:.0f})" for g in sorted(h, key=lambda g: -h[g]))
            lines.append(rf"{LABEL[arm]}, seed {m[-1]} & {types} & {fmt_int(round(sum(h.values())))} \\")
        if per:
            hours[arm] = per
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), hours


# ------------------------------------------------------------------ test-sample uncertainty
# (heading, bootstrap family, task, metric, model-name prefix, baseline of run index k): the
# bootstrap files of scripts/build_curve_jobs.py, bootstrap/<family>/<task>_N<N>.json
BOOT_COLS = (("JetClass-II", "probe", "jc2_188", "macro", "mtx-", "init-s{k}"),
             ("JetClass", "probe", "jc1", "macro", "mtx-", "init-s{k}"),
             ("top", "probe", "top", "auc", "mtx-", "init-s{k}"),
             ("q/g", "probe", "qg", "auc", "mtx-", "init-s{k}"),
             ("JetClass-II", "jc2", "jc2_188", "macro", "", "scratch-s1-{k}"),
             ("JetClass", "ft", "jc1", "macro", "", "scratch-s1-{k}"),
             ("top", "ft", "top", "auc", "", "scratch-s1-{k}"),
             ("q/g", "ft", "qg", "auc", "", "scratch-s1-{k}"))
BOOT_ROWS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L188", None))


def _boot_largest(boot: dict, family: str, task: str):
    """The family's bootstrap document at the task's largest training set, or None."""
    docs = {int(k.rsplit("_N", 1)[1]): d for k, d in boot.get(family, {}).items() if k.rsplit("_N", 1)[0] == task}
    return docs[max(docs)] if docs else None


def boot_spreads(boot: dict, runs: dict, col: tuple, arm: str, ref) -> tuple:
    """(SD over runs of ln ratio, mean over runs of its test-sample SD) of (1 - AUC) of arm over
    ref (None: the column's baseline) at the largest training set; None where a run is missing."""
    _, fam, task, met, pre, base = col
    doc = _boot_largest(boot, fam, task)
    ln, test = [], []
    for m in runs[arm]:
        k = m.rsplit("-s", 1)[1]
        r = m[len("mtx-"):]
        b = base.format(k=k) if ref is None else f"{pre}{runs[ref][0][len('mtx-'):].rsplit('-s', 1)[0]}-s{k}"
        c = None if doc is None else doc["pairs"].get(f"{pre}{r}/{b}", {}).get(met)
        if c is None or c["ratio"] is None or c["ln_ratio_sd"] is None:
            return None, None
        ln.append(math.log(c["ratio"]))
        test.append(c["ln_ratio_sd"])
    return (float(np.std(ln, ddof=1)) if len(ln) > 1 else None), float(np.mean(test))


def _pct(x) -> str:
    """A relative SD in per cent to two significant figures."""
    if x is None:
        return "--"
    v = 100 * x
    return f"{v:.0f}" if v >= 10 else f"{v:#.2g}"


def table_boot(boot: dict, runs: dict) -> tuple[str, dict]:
    """Per comparison and task, at the largest training set: the SD of ln((1 - AUC) ratio) over
    the pretraining runs and from the test sample, in per cent; and the largest test-sample SD of
    a label-set comparison, for linear probes and for fine-tuning."""
    heads = [h for h, *_ in BOOT_COLS]
    lines = [r"\begin{tabular}{l" + "c" * len(BOOT_COLS) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{linear probe} & \multicolumn{4}{c}{fine-tuning} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "ratio & " + " & ".join(heads) + r" \\", r"\midrule"]
    worst = {"probe": [], "ft": []}
    for arm, ref in BOOT_ROWS:
        cells = []
        for col in BOOT_COLS:
            between, test = boot_spreads(boot, runs, col, arm, ref)
            cells.append(f"{_pct(between)} / {_pct(test)}")
            if ref is not None and test is not None:
                worst["probe" if col[1] == "probe" else "ft"].append(test)
        name = LABEL[arm] + " / " + (LABEL[ref] if ref else "baseline")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), worst


def boot_macros(em, boot: dict, worst: dict, index) -> None:
    b = {d["b"] for fam in boot.values() for d in fam.values()}
    if len(b) != 1:
        raise SystemExit(f"FATAL: the bootstraps use different numbers of resamples: {sorted(b)}")
    em.macro("MlstBootResamples", str(b.pop()), index, "bootstrap resamples of the test jets")
    for k, w in (("probe", "Probe"), ("ft", "Ft")):
        em.macro(f"MlstBootTestMax{w}", _pct(max(worst[k])) if worst[k] else r"\pending{bootstrap}", index,
                 f"largest test-sample SD of ln(1 - AUC ratio) of a label set over the 188 classes, {k}, per cent")


SPREAD = ("Mean $\\pm$ standard deviation over \\MlstNRunsWord{} pretraining runs; for "
          "configurations with \\RandFlavRuns{} runs both values are given, and a random partition has a "
          "single run.")
# tables without random partitions or flavor-boundary partitions
SPREAD3 = "Mean $\\pm$ standard deviation over \\MlstNRunsWord{} pretraining runs."
SPREAD_MASS = (SPREAD3[:-1] + "; the models with the mass regression have \\MlstMassNRunsWord{} runs, and both "
               "values are given.")
CAPTIONS = {
    "probes": ("tab:probes", "Linear probes on the frozen class token at the largest training set of each "
               "task: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better. JetClass-II: the macro AUC over its "
               "\\Kfull{} classes, and over its resonant and QCD classes separately; JetClass: the macro AUC; "
               "top and quark/gluon (q/g) tagging: the AUC, q/g also on jets generated with Herwig. " + SPREAD +
               " $^{*}$: a probe that did not converge."),
    "probes_n": ("tab:probesn", "Linear probes against the training-set size: "
                 "$(1-\\mathrm{AUC})\\times10^{3}$, lower is better; the JetClass-II macro AUC is also "
                 "given over the classes present in the training subset and over its QCD classes; column headers "
                 "are numbers of training jets. " + SPREAD3 +
                 " $^{*}$: a probe that did not converge."),
    "pairs": ("tab:pairs", "Binary linear probes between JetClass-II classes on the largest training "
              "subset: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better. $^{\\dagger}$: the label set "
              "merges the two sides into one class; $^{\\ddagger}$: it merges the first side with part "
              "of the second. " + SPREAD + " The $b$-quark boundary and the random flavor boundary lie "
              "only in the four-quark decays $X\\to YY\\to QQQQ$; every label set but the \\Kfull-class one "
              "merges all QCD classes."),
    "mlp": ("tab:mlp", "Nonlinear probes (a multilayer perceptron with a single hidden layer, trained until its "
            "validation AUC stops improving, with \\LpMlpSeedsWord{} seeds whose scores are averaged) at the largest "
            "training set of each task: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better, with the macro AUC on "
            "JetClass-II and JetClass. Upper part: the tasks of Table~\\ref{tab:probes}. Lower part: the binary "
            "probes between pairs of JetClass-II classes of Table~\\ref{tab:pairs}, on the same jets; which label "
            "sets merge each pair is marked there. The three \\Kfull-class runs agree on $X\\to bc$ to a few "
            "$10^{-6}$ in AUC although their scores differ; the value was re-derived from their features. "
            + SPREAD),
    "fill": ("tab:fill", "Linear probes on the \\Kfull-class JetClass-II task with \\FtSizeESix{} training jets: "
             "$(1-\\mathrm{AUC})\\times10^{3}$ on the features as computed, and with the input features "
             "that the top-tagging and the quark/gluon data lack set to zero, as in those data. Mean $\\pm$ "
             "standard deviation over the \\MlstNRunsWord{} pretraining runs of each level of the hierarchy and "
             "the \\MlstNRunsWord{} randomly initialized networks."),
    "mass": ("tab:mass", "Mass resolution $\\sigma_{\\mathrm{eff}}$ of $\\ln(m_{\\mathrm{reg}}/"
             "m_{\\mathrm{SD}}^{\\mathrm{gen}})$ on the JetClass-II test jets with a generator-level match, "
             "on all, resonant and QCD jets, and the median $m_{\\mathrm{reg}}/m_{\\mathrm{SD}}^{\\mathrm{gen}}$; "
             "lower $\\sigma_{\\mathrm{eff}}$ is better. The mass node of the models pretrained with the mass "
             "regression is scored on the target itself. The multilayer perceptron (MLP) on frozen features, "
             "averaged over __MLP_SEEDS__ seeds, regresses the target minus the median of the jet's true class "
             "and is scored with that median added back, so its rows are read against the class median alone, "
             "not against the uncorrected mass or the mass node. " + SPREAD_MASS),
    "sculpt": ("tab:sculpt", "Mass sculpting: Jensen--Shannon divergence $\\times10^{3}$ between the "
               "$m_{\\mathrm{SD}}$ distributions (__SCULPT_BINS__) of all QCD test jets and of those passing a cut on "
               "$1-p_{\\mathrm{QCD}}$ at the QCD efficiency given; lower is less sculpting. The last row is the "
               "divergence expected for a random selection of as many QCD jets, which a cut that does not "
               "sculpt the mass still shows because of the finite number of jets passing it. " + SPREAD_MASS),
    "ft": ("tab:ft", "Full fine-tuning against the training-set size: $(1-\\mathrm{AUC})\\times10^{3}$, "
           "lower is better, with the macro AUC (one-versus-rest) on the class-balanced JetClass-II test set, "
           "over all and over its QCD classes, and on JetClass, and the AUC on top and quark/gluon (q/g) "
           "tagging, q/g also on jets generated with Herwig. " + SPREAD + " Each pretrained run is fine-tuned "
           "once. From scratch: mean $\\pm$ standard deviation over \\MlstFtRefSeedsWord{} training seeds on the "
           "same jets."),
    "ft_rej": ("tab:ftrej", "Background rejection $R_{50}$ and $R_{30}$ ($1/\\epsilon_B$ at 50\\% and "
               "30\\% signal efficiency) after full fine-tuning on the whole training set. " + SPREAD +
               " From scratch: over its seeds. Every value rests on at least __FEWEST__ background jets "
               "passing the cut."),
    "bench_top": ("tab:benchtop", "Top tagging on the full training set, \\LitPartTopNTrainMillion{} million jets: accuracy, "
                  "AUC and background rejection $1/\\epsilon_B$ at 50\\% and 30\\% signal efficiency. Published "
                  "rows from the tables cited; f.t.\\ marks a model pretrained on JetClass and fine-tuned, and the "
                  "kinematic-input rows were pretrained on the kinematic features only, as ParT-f.t.\\ was. "
                  "Uncertainties: ours, the standard deviation over pretraining runs, one fine-tuning each (three "
                  "runs; two runs print both values; one run, the value alone); ParT, the spread over "
                  "\\LitPartFtHeadReinits{} fine-tunings of one pretrained model with different head initializations, "
                  "quoted for the run of median accuracy; OmniLearn, OmniLearned and L-GATr, the standard deviation of "
                  "five trainings. The best value of each column is in bold."),
    "bench_qg": ("tab:benchqg", "Quark/gluon tagging on the full training set, \\LitQgTrainMillion{} million jets, with "
                 "the particle type as five binary flags and the charge (ParT's exp scenario): accuracy, AUC and "
                 "$1/\\epsilon_B$ at 50\\% and 30\\% quark efficiency. Published rows from the tables cited, all "
                 "in the exp scenario; PFN-Ex's uncertainty is half the interquartile range of ten trainings, and the "
                 "others are as in Table~\\ref{tab:benchtop}, with nine trainings for OmniLearn and OmniLearned. The "
                 "best value of each column is in bold."),
    "boot": ("tab:boot", "Relative uncertainty of the ratios of $(1-\\mathrm{AUC})$ at the largest training "
             "set of each task, in per cent: the standard deviation of the logarithm of the ratio over the "
             "pretraining runs, then that from the finite test sample (a bootstrap of the test jets with "
             "\\MlstBootResamples{} resamples, averaged over the runs). The baseline is the randomly "
             "initialized network of the same seed for the linear probes and the network trained from "
             "scratch on the same subset for fine-tuning. The macro AUC is used for JetClass-II and JetClass."),
    "jetclass": ("tab:jetclass", "JetClass in the form of ParT's Table~1: accuracy, macro AUC one-vs-one "
                 "(ParT's convention) and one-vs-rest, and for each signal class the rejection of QCD jets at "
                 "50\\% signal efficiency (__WPHQQL__\\% for $H\\to\\ell\\nu qq'$, __WPTBL__\\% for $t\\to b\\ell\\nu$), on "
                 "the discriminant $p_c/(p_c+p_{\\mathrm{QCD}})$. The published rows were trained on the number of "
                 "jets given and scored on a \\LitPartJcTestMillion-million-jet test set; ours are fine-tuned on one million and "
                 "scored on two million (the AUCs on every fourth), mean $\\pm$ standard deviation over the "
                 "pretraining runs (the network trained from scratch: three seeds)."),
    "xbx": ("tab:xbx", "Resonance tagging in the soft-drop mass window, \\XbxWindowLo--\\XbxWindowHi~GeV: signal "
            "efficiency (\\%) at the QCD efficiency given, mean $\\pm$ standard deviation over the pretraining "
            "runs. Transfer: Sophon's classifier on the frozen features, its seeds averaged per run (the "
            "randomly initialized network: the linear probe). Direct: Sophon's discriminant from the output "
            "scores, where $X\\to b\\bar b$ is a class of its own."),
    "compute": ("tab:compute", "GPU types and GPU-hours of every pretraining run, summed over the "
                "training and validation passes of its completed epochs as each epoch recorded them."),
}


# (caption key, table file, wide, caption fill) of the tables build() writes as placeholders
PLACEHOLDERS = (("mlp", "mlst_mlp", False, None), ("fill", "mlst_fill", False, None),
                ("mass", "mlst_mass", False, {"__MLP_SEEDS__": r"\pending{n}"}),
                ("sculpt", "mlst_sculpt", False, {"__SCULPT_BINS__": r"\pending{bins}"}), ("ft", "mlst_ft", True, None),
                ("ft_rej", "mlst_ft_rej", True, {"__FEWEST__": r"\pending{n}"}), ("boot", "mlst_boot", True, None),
                ("xbx", "mlst_xbx", False, None), ("jetclass", "mlst_jetclass", True,
                 {"__WPHQQL__": r"\pending{n}", "__WPTBL__": r"\pending{n}"}))
# the resonance-tagging section's sample macros (experiments/XBS/xbx_report.py macros)
XBX_MACROS = ("XbxMx", "XbxWindowLo", "XbxWindowHi") + tuple(
    f"{s}{k}" for s in ("Xbb", "Xbs") for k in ("NTrainPerClass", "Repetition", "NEvalSignal", "NEvalQcd"))


def _float(key: str, body: str, wide: bool = False, fill: dict | None = None) -> str:
    env = "table*" if wide else "table"
    label, cap = CAPTIONS[key]
    for k, v in (fill or {}).items():
        cap = cap.replace(k, v)
    assert "__" not in cap, cap
    boxes = [r"\begin{adjustbox}{max width=\linewidth}" + "\n" + b.strip() + "\n" + r"\end{adjustbox}"
             for b in body.split("%SPLIT%")]
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", r"\small", rf"\caption{{{cap}}}",
                      rf"\label{{{label}}}", "\n\\par\\medskip\n".join(boxes), rf"\end{{{env}}}", ""])


def check_index(data: pathlib.Path) -> pathlib.Path:
    """index.json against the files beside it: every file listed, with the bytes pulled."""
    import hashlib
    index = data / "index.json"
    if not index.exists():
        raise SystemExit(f"FATAL: no {index}; pull with experiments/FIGS/pull_mlst_data.py")
    want = json.loads(index.read_text())
    have = {str(p.relative_to(data)) for p in data.rglob("*.json") if p != index}
    if have != set(want):
        raise SystemExit(f"FATAL: {index} lists {len(want)} files, {data} holds {len(have)}: "
                         f"{sorted(have ^ set(want))[:5]}")
    bad = [k for k, h in want.items() if hashlib.sha256((data / k).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} files differ from {index}: {bad[:5]}")
    return index


def mlp_seeds(mass: dict) -> str:
    n = {d["arms"][ck]["mlp"]["fit"]["n_seeds"] for d in mass.values() for ck in d["arms"]}
    if len(n) != 1:
        raise SystemExit(f"FATAL: the mass MLPs average different numbers of seeds: {sorted(n)}")
    return str(n.pop())


def xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int) -> None:
    """X->bb and X->bs from the pulled fits (data/v2/xbx/<signal>/<model>.json) through
    experiments/XBS/xbx_report.py, the code the dry run exercised; red markers until they exist."""
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("xbx_report", root / "experiments/XBS/xbx_report.py")
    XR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(XR)
    xfits = XR.load(data / "xbx") if (data / "xbx").is_dir() else {}
    MT = SimpleNamespace(fmt_pm=fmt_pm, fmt_int=fmt_int)
    have = {n for sigs in xfits.values() for n in sigs}
    if xfits:
        XR.macros(em, MT, xfits, index)
        out["tables/mlst_xbx.tex"] = _float("xbx", XR.table(
            SimpleNamespace(LABEL=LABEL, _cell=_cell, RAND=RAND, INIT_MODELS=INIT_MODELS), MT, runs, xfits))
    else:
        missing.append("X->bb and X->bs fits -- experiments/FIGS/data/v2/xbx/")
    for name in XBX_MACROS:
        if name not in em.provenance:
            sig = {"Xbb": "xbb", "Xbs": "xbs"}.get(name[:3])
            why = "sample not in" if sig and xfits and sig not in have else "samples not in"
            em.macro(name, r"\pending{" + why + "}", root / "experiments/XBS/xbx_report.py", "pending")


def build(em, root: pathlib.Path, missing: list, fmt_pm, fmt_int, words) -> dict:
    data = root / "experiments/FIGS/data/v2"
    fits, comp = _load(data / FITS), _load(data / "compute")
    fits.update({"initbn" + m.removeprefix("init"): d for m, d in _load(data / FITS_INITBN).items()})
    # the v4 probes (class-balanced JetClass-II, v4 grid; mlst_v4.py) replace fits_v3 once pulled
    fits_v4, raw_v4 = V4.load_fits(data)
    if fits_v4:
        fits = fits_v4
    else:
        missing.append("v4 linear probes -- experiments/FIGS/data/v2/linprobe/fits_v4/")
    if not fits:
        missing.append(f"linear probes -- experiments/FIGS/data/v2/{FITS}/")
        return {}
    V4.design_macros(em, root, data, missing, fmt_int)
    index = check_index(data)
    runs = _paper(root)
    n_paper = sum(len(v) for k, v in runs.items() if k not in (INIT, INIT_BN))
    done = [m for k, v in runs.items() if k not in (INIT, INIT_BN) for m in v if m in fits]
    em.macro("MlstProbeRunsDone", str(len(done)), index, "runs with a linear-probe fit (count)",
             "of the paper's runs")
    em.macro("MlstPaperRuns", str(n_paper), root / "configs/arms/v2_paper_runs.txt", "runs (count)")
    n_main = len(runs["L188"])
    em.macro("MlstSdRelUnc", f"{100 / math.sqrt(2 * (n_main - 1)):.0f}\\%", root / "configs/arms/v2_paper_runs.txt",
             "relative standard error of a standard deviation from the runs of a hierarchy level, "
             "1/sqrt(2(n-1)) for a normal sample")
    if len(done) < n_paper:
        missing.append(f"linear probes: {len(done)} of {n_paper} paper runs")
    status = merge_status(root)
    UNCONVERGED_SEEN.clear()
    out = {"tables/mlst_probes.tex": _float("probes", table_probes(fits, runs, fmt_pm, MAIN + MASS + RAND + FLAV + (INIT, INIT_BN)), True),
           "tables/mlst_probes_n.tex": _float("probes_n", table_probes_n(fits, runs, fmt_pm, MAIN + (INIT, INIT_BN)), True),
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, runs, fmt_pm, status, MAIN + RAND + FLAV), True)}
    comparisons(em, fits, runs, status, index, fmt_pm)
    bn_macros(em, fits, runs, index)
    V4.gpu_bound_macro(em, fits, runs, index)
    trained = {(m, ds, n) for (m, ds, n) in UNCONVERGED_SEEN if not m.startswith(("init-", "initbn-"))}
    em.macro("MlstProbeUnconverged", str(len(trained)), index,
             "linear probes of pretrained models that did not converge (count)")
    em.macro("MlstProbeUnconvergedInit", str(len(UNCONVERGED_SEEN) - len(trained)), index,
             "linear probes of the randomly initialized networks that did not converge (count)")
    cells = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {}))]
    edge = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {}))
            if c["C"] == max(d["c_grid"])]
    em.macro("MlstProbeFits", str(len(cells)), index, "selected linear-probe fits of the reported checkpoints (count)")
    em.macro("MlstProbeEdgeC", str(len(edge)), index,
             "of these, fits whose selected C is the largest of the grid (count)")
    mlp = {**_load(data / MLP_FITS), **_load(data / MLP_FITS_V4)}
    mlp.update({"initbn" + m.removeprefix("init"): d for m, d in _load(data / MLP_FITS_INITBN).items()})
    if mlp:
        out["tables/mlst_mlp.tex"] = _float("mlp", table_mlp(mlp, runs, fmt_pm, MAIN + MASS + RAND + FLAV + (INIT, INIT_BN)), True)
        mlp_macros(em, mlp, runs, index, fmt_pm)
    else:
        missing.append(f"nonlinear probes -- experiments/FIGS/data/v2/{MLP_FITS}/")
    fill = {k: _load(data / "fillcontrol" / k) for k in MASKS}
    if all(fill.values()):
        out["tables/mlst_fill.tex"] = _float("fill", table_fill(fits, fill, runs, fmt_pm, MAIN + (INIT,)))
        fill_macros(em, fits, fill, runs, index, fmt_pm)
    else:
        missing.append("masked-input probes -- experiments/FIGS/data/v2/fillcontrol/")
    loss_share_macros(em, comp, runs, index, fmt_pm)
    head = _load(data / "head")
    mass = {k.removesuffix("_mass"): v for k, v in head.items() if k.endswith("_mass")}
    sculpt = {k.removesuffix("_sculpting"): v for k, v in head.items() if k.endswith("_sculpting")}
    if mass:
        out["tables/mlst_mass.tex"] = _float("mass", table_mass(mass, runs, fmt_pm, MAIN + MASS + (INIT,)),
                                             fill={"__MLP_SEEDS__": words(int(mlp_seeds(mass)))})
        mass_macros(em, mass, runs, index, fmt_pm)
    else:
        missing.append("mass regression -- experiments/FIGS/data/v2/head/*_mass.json")
    if sculpt:
        edges = {tuple(d["bins_jet_sdmass"]) for d in sculpt.values()}
        if len(edges) != 1:
            raise SystemExit(f"FATAL: the sculpting files use {len(edges)} different mass binnings")
        e = edges.pop()
        width = {round(b - a, 9) for a, b in zip(e, e[1:])}
        bins = (f"{len(e) - 1} bins of {width.pop():g}~GeV from {e[0]:g} to {e[-1]:g}~GeV" if len(width) == 1
                else f"{len(e) - 1} bins from {e[0]:g} to {e[-1]:g}~GeV")
        out["tables/mlst_sculpt.tex"] = _float("sculpt", table_sculpt(sculpt, runs, fmt_pm, MAIN + MASS),
                                               fill={"__SCULPT_BINS__": bins})
        sculpt_macros(em, sculpt, runs, index, fmt_pm)
    else:
        missing.append("mass sculpting -- experiments/FIGS/data/v2/head/*_sculpting.json")
    # a directory with a dot in its name is a job's unfinished staging directory
    boot = {d.name: _load(d) for d in sorted((data / "bootstrap").glob("*")) if d.is_dir() and "." not in d.name}
    if boot:
        body, worst = table_boot(boot, runs)
        out["tables/mlst_boot.tex"] = _float("boot", body, True)
        boot_macros(em, boot, worst, index)
    else:
        missing.append("test-sample bootstrap -- experiments/FIGS/data/v2/bootstrap/")
        for k in ("MlstBootResamples", "MlstBootTestMaxProbe", "MlstBootTestMaxFt"):
            em.macro(k, r"\pending{bootstrap}", index, "no bootstrap pulled yet")
    composition_macros(em, data, root, missing)
    size_macros(em, fits, fmt_int, index)
    method_macros(em, root, words)
    ft_recipe(em, data, {m[len("mtx-"):] for k, v in runs.items() if k not in (INIT, INIT_BN) for m in v}, fmt_int, missing, words)
    tag, per, absent = load_ft(data)
    if absent:
        missing.append("fine-tuning read-outs -- experiments/FIGS/data/v2/ft_mlst/: " + ", ".join(absent))
    if tag is not None:
        out["tables/mlst_ft.tex"] = _float("ft", table_ft(per, runs, fmt_pm, MAIN + MASS + RAND + FLAV), True)
        body, fewest = table_ft_rej(per, runs, fmt_pm, MAIN + MASS + RAND + FLAV)
        out["tables/mlst_ft_rej.tex"] = _float("ft_rej", body, True, fill={"__FEWEST__": fmt_int(fewest)})
        ft_macros(em, per, runs, index, fmt_pm)
        topcheck_macros(em, data, per, index, fmt_pm, missing)
        bench_rows = MAIN + MASS + FLAV
        out["tables/mlst_bench_top.tex"] = _float("bench_top", V4.table_bench(
            "top", per, runs, bench_rows, LABEL, SCRATCH, fmt_pm, _two), False)
        out["tables/mlst_bench_qg.tex"] = _float("bench_qg", V4.table_bench(
            "qg", per, runs, bench_rows, LABEL, SCRATCH, fmt_pm, _two), False)
    xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int)
    full = [data / "curves" / f"full_{t}.json" for t in ("best70_leg2", "ref_v100_jc1_leg2")]
    if all(f.exists() for f in full):
        pre, ref = (json.loads(f.read_text())["cells"] for f in full)
        pooled = {**runs, "RAND": [m for a in RAND for m in runs[a]]}
        out["tables/mlst_jetclass.tex"] = _float("jetclass", V4.table_jetclass(
            pre, ref, pooled, MAIN + MASS + ("RAND",) + FLAV, {**LABEL, "RAND": "random partitions (all five)"},
            SCRATCH, 1_000_000, fmt_pm, _two), True,
            fill={"__WPHQQL__": f"{100 * V4.LPV_EPS['label_Hqql']:g}", "__WPTBL__": f"{100 * V4.LPV_EPS['label_Tbl']:g}"})
    else:
        missing.append("JetClass per-class curves -- experiments/FIGS/data/v2/curves/full_*.json")
    if comp:
        body, hours = table_compute(comp, runs, fmt_int)
        out["tables/compute.tex"] = _float("compute", body)
        # Whether a label set is confounded with a GPU type: each label set's share of its
        # GPU-hours on the type the four label sets used most, lowest and highest.
        total = collections.Counter()
        for arm in MAIN:
            total.update(hours.get(arm, {}))
        top = max(total, key=total.get)
        shares = [hours[a][top] / sum(hours[a].values()) for a in MAIN if a in hours]
        em.macro("ComputeMainGpu", top, index, "GPU type with the most hours over the four label sets")
        em.macro("ComputeMainGpuShareMin", f"{100 * min(shares):.0f}\\%", index,
                 "lowest share of a label set's GPU-hours on that type")
        em.macro("ComputeMainGpuShareMax", f"{100 * max(shares):.0f}\\%", index,
                 "highest share of a label set's GPU-hours on that type")
        em.macro("ComputeRunsDone", str(len(comp)), index, "runs with a compute record (count)")
        types = sorted({gpu_name(e["device"]) for c in comp.values() for e in c["epochs"]})
        em.macro("ComputeGpuTypes", ", ".join(types[:-1]) + " and " + types[-1] if len(types) > 1 else types[0],
                 index, "GPU types any finished run trained on")
        if len(comp) < n_paper:
            missing.append(f"compute records: {len(comp)} of {n_paper} paper runs")
    # a table whose data is not pulled yet is a marked placeholder, never the file an older
    # generator left behind
    for key, name, wide, fill in PLACEHOLDERS:
        out.setdefault(f"tables/{name}.tex", _float(key, r"\pending{no result yet}", wide, fill))
    return out
