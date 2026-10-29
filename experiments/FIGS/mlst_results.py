"""The MLST paper's results tables, read by experiments/FIGS/make_tables.py (build()).

Every value is the mean +- standard deviation over the pretraining runs of a configuration
that the paper reports (configs/arms/v2_paper_runs.txt), PDG-rounded (make_tables.fmt_pm). A
configuration with runs still missing prints the runs it has and a red marker "n of N runs".

Inputs, under experiments/FIGS/data/v2/ (experiments/FIGS/pull_mlst_data.py):
  linprobe/fits3000/<model>.json   linear probes (experiments/EVAL/linear_probe_v2.py)
  linprobe/mass/<model>.json       jet-mass regression (experiments/EVAL/mass_resolution.py)
  compute/<run>.json               per epoch: the GPU and the seconds of training and validation
  ft_mlst/<tag>_<file>.json        fine-tuning read-outs: best70 (every run) or, until it exists,
                                   best70_done18 (the runs finished before the split)
  index.json                       {path: sha256} of every file above, as pulled: checked against
                                   them, and the source the macros counted over many files cite

Outputs:
  tables/mlst_probes.tex     linear probes at the largest training set, every task
  tables/mlst_probes_n.tex   the same at every training-set size, JetClass-II and top tagging
  tables/mlst_pairs.tex      binary probes between JetClass-II classes, merged or kept apart
  tables/mlst_mass.tex       mass resolution from frozen features, with and without the mass output
  tables/mlst_ft.tex         full fine-tuning against the training-set size, every dataset
  tables/mlst_ft_rej.tex     background rejection after fine-tuning on the full training set
  tables/compute.tex         GPU types and hours of every run
and the macros the text quotes (Mlst*, Compute*).
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import re

import numpy as np

MAIN = ("L188", "L162", "R42_Q1", "R16_Q1")
MASS = ("L162_MASS", "R16_Q1_MASS")
RAND = tuple(f"RAND2_p{k}" for k in range(1, 6))
FLAV = ("FLAV_F0", "FLAV_F1", "FLAV_F1R")
INIT = "INIT"
INIT_MODELS = ("init-s1", "init-s2", "init-s3")
LABEL = {"L188": r"\Kfull{} classes", "L162": r"\Kres{} classes", "R42_Q1": r"\Kmid{} classes",
         "R16_Q1": r"\Kcoarse{} classes", "L162_MASS": r"\Kres{} classes, mass regression",
         "R16_Q1_MASS": r"\Kcoarse{} classes, mass regression", "FLAV_F0": "random partition, no flavor boundary",
         "FLAV_F1": r"random partition, $b$-quark boundary",
         "FLAV_F1R": "random partition, random flavor boundary", INIT: "randomly initialized network",
         **{f"RAND2_p{k}": f"random partition {k}" for k in range(1, 6)}}
PRIMARY = "best70"
# (dataset key, metric, column heading)
TASKS = (("jc2", "macro_auc_ovr", "JetClass-II"), ("jc1", "macro_auc_ovr", "JetClass"),
         ("top", "auc", "top"), ("qg", "auc", "q/g"), ("qg_herwig", "auc", "q/g, Herwig"))
PAIRS = (("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", ("label_X_bc",), ("label_X_bq", "label_X_cs")),
         ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", ("label_X_bb",), ("label_X_cc",)),
         ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $ccqq$", ("label_X_YY_bbqq",), ("label_X_YY_ccqq",)))
PAIR_N = "1000000"
# Fine-tuning: (task, read-out file, data set inside it, metric, column heading)
FT_TAGS = ("best70", "best70_done18")
FT_TASKS = (("jc2", "leg1_metrics", None, "macro_auc_ovr", "JetClass-II"),
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
    return out


def _load(d: pathlib.Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


# a probe of the untrained network that stopped at the iteration limit (linear_probe_v2.py
# MAX_ITER) carries this mark; a trained run's probe that did not converge stops the build
UNCONVERGED = r"$^{*}$"


class Unconverged(float):
    pass


def _lone(x: float) -> str:
    """Five significant figures with their trailing zeros, thousands separated as fmt_pm does."""
    s = f"{x:#.5g}".rstrip(".")
    head, dot, tail = s.partition(".")
    if "e" in s or len(head.lstrip("-")) <= 3:
        return s
    return f"{int(head):,}".replace(",", "{,}") + dot + tail


def _cell(values: list, n_expected: int, fmt_pm, scale: float = 1.0) -> str:
    """mean +- SD over the runs present; a lone run prints alone; missing runs add a red marker."""
    flag = any(isinstance(x, Unconverged) for x in values)
    v = [x * scale for x in values if x is not None]
    if not v:
        return r"\pending{no run yet}"
    # a lone run keeps its trailing zeros (0.96580, not 0.9658)
    body = fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else _lone(v[0])
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


def _probe(fits: dict, model: str, ds: str, n: str, metric: str):
    ck = "init" if model.startswith("init-") else PRIMARY
    c = fits.get(model, {}).get("cells", {}).get(ck, {}).get(ds, {}).get(n)
    if c is None:
        return None
    if not c["converged"]:
        if ck != "init":
            raise SystemExit(f"FATAL: {model} {ds} N={n}: the probe did not converge")
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


def table_probes_n(fits, runs, fmt_pm, rows) -> str:
    out = []
    for ds, met, title in (TASKS[0], TASKS[2]):
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
    """(tag, {task: {init: {N: [record per fine-tuning seed]}}}) of the first read-out tag
    whose files are all present, or (None, {})."""
    d = data / "ft_mlst"
    for tag in FT_TAGS:
        files = {f for _, f, *_ in FT_TASKS}
        if all((d / f"{tag}_{f}.json").exists() for f in files):
            break
    else:
        return None, {}
    per = {}
    for key, f, ds, _, _ in FT_TASKS:
        cells = json.loads((d / f"{tag}_{f}.json").read_text())["cells"]
        cells = cells if ds is None else cells[ds]
        per[key] = {init: {int(n[1:]): [rec for _, rec in sorted(seeds.items())]
                           for n, seeds in by_n.items() if re.fullmatch(r"N\d+", n)}
                    for init, by_n in cells.items()}
    return tag, per


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


def table_ft(per, runs, fmt_pm, rows) -> str:
    out = []
    for key, _, _, met, title in FT_TASKS:
        sizes = _sizes(per, key)
        out += [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
                rf"{title} & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in sizes) + r" \\",
                r"\midrule"]
        for arm in rows:
            out.append(LABEL[arm] + " & " + " & ".join(
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
    for arm in rows + (SCRATCH,):
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
        lines.append(("from scratch" if arm == SCRATCH else LABEL[arm]) + " & " + " & ".join(cells) + r" \\")
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
    else:
        missing.append("from-scratch references on V100 -- ref_v100 cells")
        for k in ("MlstFtRefLr", "MlstFtRefGpuTypes", "MlstFtRefSeedsWord"):
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
    out = {}
    for arm, g in group.items():
        out[arm] = {}
        for key, _, a, b in PAIRS:
            same = [g[x] == g[y] for x in a for y in b]
            out[arm][key] = "merged" if all(same) else "partly" if any(same) else "apart"
    return out


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
                c = fits.get(m, {}).get("cells", {}).get(PRIMARY, {}).get("jc2_pairs", {}).get(PAIR_N, {}).get(key)
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
            ck = "init" if m.startswith("init-") else PRIMARY
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
CMP_PROBES = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L188", INIT), ("R16_Q1", INIT),
              ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
CMP_PAIRS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("FLAV_F1", "FLAV_F0"),
             ("FLAV_F1R", "FLAV_F1"), ("FLAV_F1R", "FLAV_F0"))
CMP_MASS = (("R16_Q1", "L188"), ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))
WORD = {"L188": "Full", "L162": "Res", "R42_Q1": "Mid", "R16_Q1": "Coarse", "L162_MASS": "ResMass",
        "R16_Q1_MASS": "CoarseMass", INIT: "Init", "FLAV_F0": "Fzero", "FLAV_F1": "Fone", "FLAV_F1R": "Foner"}
TASK_WORD = {"jc2": "JcTwo", "jc1": "JcOne", "top": "Top", "qg": "Qg", "qg_herwig": "QgHerwig"}
PAIR_WORD = {"bc_vs_bq_cs": "Bc", "bb_vs_cc": "Bb", "bbqq_vs_ccqq": "Bbqq"}


def _oma(x):
    """1 - AUC, keeping the mark of a probe that stopped at the iteration limit."""
    return None if x is None else Unconverged(1 - x) if isinstance(x, Unconverged) else 1 - x


def _ratio(runs, arm, ref, value, fmt_pm) -> str:
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
    return _cell(r, len(both), fmt_pm)


def comparisons(em, fits, mass, runs, status, index, fmt_pm) -> None:
    """The macros the results text quotes: MlstCmp<task or pair or mass probe><arm>Over<ref>, and
    MlstPairRand{Merged,Apart}<pair>, the binary probes' (1 - AUC) x 10^3 over the random
    partitions that merge the pair and over those that keep it apart (one run each)."""
    for ds, met, _ in TASKS:
        n = _largest(fits, ds)
        for arm, ref in CMP_PROBES:
            em.macro(f"MlstCmp{TASK_WORD[ds]}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: _oma(_probe(fits, m, ds, n, met)), fmt_pm), index,
                     f"linear probe {ds} N={n}: (1 - {met}) of {arm} over {ref}, run by run")

    def pair(m, key):
        c = fits.get(m, {}).get("cells", {}).get(PRIMARY, {}).get("jc2_pairs", {}).get(PAIR_N, {}).get(key)
        return None if c is None else 1 - c["auc"]
    for key, _, _, _ in PAIRS:
        for arm, ref in CMP_PAIRS:
            em.macro(f"MlstCmp{PAIR_WORD[key]}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: pair(m, key), fmt_pm), index,
                     f"binary probe {key}: (1 - AUC) of {arm} over {ref}, run by run")
        for side in ("merged", "apart"):
            ms = [m for arm in RAND if status[arm][key] == side for m in runs[arm]]
            em.macro(f"MlstPairRand{side.title()}{PAIR_WORD[key]}", _cell([pair(m, key) for m in ms], len(ms), fmt_pm, 1e3),
                     index, f"binary probe {key}: (1 - AUC) x 10^3 over the random partitions that "
                     + {"merged": "merge it", "apart": "keep it apart"}[side])
    for probe in ("ridge", "mlp"):
        for arm, ref in CMP_MASS:
            em.macro(f"MlstCmp{probe.title()}{WORD[arm]}Over{WORD[ref]}",
                     _ratio(runs, arm, ref, lambda m: None if m not in mass else
                            mass[m]["arms"][PRIMARY][probe]["sigma_eff"], fmt_pm), index,
                     f"mass resolution, {probe}: sigma_eff of {arm} over {ref}, run by run")


def gpu_name(device: str) -> str:
    return device.replace("NVIDIA ", "").replace("GeForce ", "").replace("Tesla ", "").replace("-SXM2-32GB", "")


def table_compute(comp, runs, fmt_int) -> tuple[str, dict]:
    """Per run: hours on each GPU type (completed epochs, training and validation passes)."""
    lines = [r"\begin{tabular}{llr}", r"\toprule", r"run & GPU types (hours) & GPU-hours \\", r"\midrule"]
    hours = {}
    for arm, ms in runs.items():
        if arm == INIT:
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


CAPTIONS = {
    "probes": ("tab:probes", "Linear probes on the frozen class token at the largest training set of each "
               "task: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better, with the macro AUC "
               "(one-versus-rest) on JetClass-II and JetClass and the AUC on top and quark/gluon "
               "tagging (q/g), and q/g also on jets generated with Herwig. Mean $\\pm$ standard deviation "
               "over the pretraining runs. $^{*}$: a probe of the randomly initialized network stopped "
               "at the iteration limit."),
    "probes_n": ("tab:probesn", "Linear probes against the training-set size: "
                 "$(1-\\mathrm{AUC})\\times10^{3}$, lower is better, with the macro AUC on "
                 "JetClass-II and the AUC on top tagging. Mean $\\pm$ standard deviation over the "
                 "pretraining runs. $^{*}$: a probe of the randomly initialized network stopped at the "
                 "iteration limit."),
    "pairs": ("tab:pairs", "Binary linear probes between JetClass-II classes on the largest training "
              "subset: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better. $^{\\dagger}$: the label set "
              "merges the two sides into one class; $^{\\ddagger}$: it merges the first side with part "
              "of the second. Mean $\\pm$ standard deviation over the pretraining runs; random partitions 1 to 5 have a "
              "single pretraining run each. The $b$-quark boundary and the random flavor boundary lie "
              "only in the four-quark decays $X\\to YY\\to QQQQ$, and the random partition with a random flavor boundary keeps "
              "apart no decays that differ only by exchanging $b$ and $c$ quarks."),
    "mass": ("tab:mass", "Resolution $\\sigma_{\\mathrm{eff}}$ of the mass-correction target $y$ of "
             "Eq.~(\\ref{eq:masstarget}) regressed from the frozen features by a small multilayer "
             "perceptron (MLP); lower is better. The MLP has one hidden layer and stops early on a "
             "validation split; it is trained with __MLP_SEEDS__ seeds whose predictions are averaged. "
             "Mean $\\pm$ standard deviation over the pretraining runs. The last row is the "
             "$\\sigma_{\\mathrm{eff}}$ of the target itself, which a constant prediction gives."),
    "ft": ("tab:ft", "Full fine-tuning against the training-set size: $(1-\\mathrm{AUC})\\times10^{3}$, "
           "lower is better, with the macro AUC (one-versus-rest) on JetClass-II and JetClass and the AUC "
           "on top and quark/gluon (q/g) tagging, and q/g also on jets generated with Herwig. "
           "Mean $\\pm$ standard deviation over the pretraining runs, one fine-tuning each; from "
           "scratch, over its fine-tuning seeds."),
    "ft_rej": ("tab:ftrej", "Background rejection $R_{50}$ and $R_{30}$ ($1/\\epsilon_B$ at 50\\% and "
               "30\\% signal efficiency) after full fine-tuning on the whole training set. Mean $\\pm$ "
               "standard deviation over the pretraining runs; from scratch, over its fine-tuning seeds. "
               "Every value rests on at least __FEWEST__ background jets passing the cut."),
    "compute": ("tab:compute", "GPU types and GPU-hours of every pretraining run, summed over the "
                "training and validation passes of its completed epochs as each epoch recorded them."),
}


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


def build(em, root: pathlib.Path, missing: list, fmt_pm, fmt_int, words) -> dict:
    data = root / "experiments/FIGS/data/v2"
    fits, mass, comp = _load(data / "linprobe/fits3000"), _load(data / "linprobe/mass"), _load(data / "compute")
    if not fits:
        missing.append("linear probes -- experiments/FIGS/data/v2/linprobe/fits3000/")
        return {}
    index = check_index(data)
    runs = _paper(root)
    n_paper = sum(len(v) for k, v in runs.items() if k != INIT)
    done = [m for k, v in runs.items() if k != INIT for m in v if m in fits]
    em.macro("MlstProbeRunsDone", str(len(done)), index, "runs with a linear-probe fit (count)",
             "of the paper's runs")
    em.macro("MlstPaperRuns", str(n_paper), root / "configs/arms/v2_paper_runs.txt", "runs (count)")
    if len(done) < n_paper:
        missing.append(f"linear probes: {len(done)} of {n_paper} paper runs")
    out = {"tables/mlst_probes.tex": _float("probes", table_probes(fits, runs, fmt_pm, MAIN + MASS + RAND + FLAV + (INIT,)), True),
           "tables/mlst_probes_n.tex": _float("probes_n", table_probes_n(fits, runs, fmt_pm, MAIN + (INIT,)), True),
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, runs, fmt_pm, merge_status(root),
                                                                MAIN + RAND + FLAV)),
           "tables/mlst_mass.tex": _float("mass", table_mass(mass, runs, fmt_pm, MAIN + MASS + (INIT,)),
                                          fill={"__MLP_SEEDS__": words(int(mlp_seeds(mass)))})}
    comparisons(em, fits, mass, runs, merge_status(root), index, fmt_pm)
    ft_recipe(em, data, {m[len("mtx-"):] for k, v in runs.items() if k != INIT for m in v}, fmt_int, missing, words)
    tag, per = load_ft(data)
    if tag is None:
        missing.append("fine-tuning read-outs -- experiments/FIGS/data/v2/ft_mlst/")
    else:
        if tag != FT_TAGS[0]:
            missing.append(f"fine-tuning: the read-out of the runs finished so far ({tag}), not every run")
        out["tables/mlst_ft.tex"] = _float("ft", table_ft(per, runs, fmt_pm, MAIN + MASS), True)
        body, fewest = table_ft_rej(per, runs, fmt_pm, MAIN + MASS)
        out["tables/mlst_ft_rej.tex"] = _float("ft_rej", body, True, fill={"__FEWEST__": fmt_int(fewest)})
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
    return out
