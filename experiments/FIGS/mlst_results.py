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
         "R16_Q1": r"\Kcoarse{} classes", "L162_MASS": r"\Kres{} classes, mass output",
         "R16_Q1_MASS": r"\Kcoarse{} classes, mass output", "FLAV_F0": "F0", "FLAV_F1": "F1",
         "FLAV_F1R": "F1r", INIT: "untrained network",
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


def _cell(values: list, n_expected: int, fmt_pm, scale: float = 1.0) -> str:
    """mean +- SD over the runs present; a lone run prints alone; missing runs add a red marker."""
    flag = any(isinstance(x, Unconverged) for x in values)
    v = [x * scale for x in values if x is not None]
    if not v:
        return r"\pending{no run yet}"
    # a lone run keeps its trailing zeros (0.96580, not 0.9658)
    body = fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else f"{v[0]:#.5g}"
    if flag:
        body += UNCONVERGED
    if len(v) < n_expected:
        body += rf" \pending{{{len(v)} of {n_expected}}}"
    return body


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
        cells = [_cell([_probe(fits, m, ds, _largest(fits, ds), met) for m in runs[arm]],
                       len(runs[arm]), fmt_pm) for ds, met, _ in TASKS]
        lines.append(LABEL[arm] + " & " + " & ".join(cells) + r" \\")
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
                _cell([_probe(fits, m, ds, n, met) for m in runs[arm]], len(runs[arm]), fmt_pm)
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
                _cell([_ft(per, key, m, n, met) for m in runs[arm]], len(runs[arm]), fmt_pm)
                for n in sizes) + r" \\")
        ref = per[key].get(SCRATCH, {})
        out.append("from scratch & " + " & ".join(
            _cell([r[met] for r in ref[n]], len(ref[n]), fmt_pm) if n in ref else "--" for n in sizes) + r" \\")
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
        lines.append(LABEL[arm] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def table_mass(mass, runs, fmt_pm, rows) -> str:
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"pretraining & ridge & MLP \\", r"\midrule"]
    for arm in rows:
        cells = []
        for probe in ("ridge", "mlp"):
            vals = []
            for m in runs[arm]:
                d = mass.get(m)
                ck = "init" if m.startswith("init-") else PRIMARY
                vals.append(None if d is None else d["arms"][ck][probe]["sigma_eff"])
            cells.append(_cell(vals, len(runs[arm]), fmt_pm))
        lines.append(LABEL[arm] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


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
                lines.append(rf"{LABEL[arm]}, run {m[-1]} & \pending{{training}} & \\")
                continue
            h = collections.Counter()
            for e in comp[m]["epochs"]:
                h[gpu_name(e["device"])] += (e["train_seconds"] + e["val_seconds"]) / 3600
            per.update(h)
            types = ", ".join(f"{g} ({h[g]:.0f})" for g in sorted(h, key=lambda g: -h[g]))
            lines.append(rf"{LABEL[arm]}, run {m[-1]} & {types} & {fmt_int(round(sum(h.values())))} \\")
        if per:
            hours[arm] = per
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), hours


CAPTIONS = {
    "probes": ("tab:probes", "Linear probes on the frozen class token at the largest training set of each "
               "task: macro AUC (one-versus-rest) on JetClass-II and JetClass, AUC on top and "
               "quark/gluon tagging, the latter also on Herwig jets. Mean $\\pm$ standard deviation "
               "over the pretraining runs. $^{*}$: a probe of the untrained network stopped at the "
               "iteration limit."),
    "probes_n": ("tab:probesn", "Linear probes against the training-set size: macro AUC on JetClass-II "
                 "and AUC on top tagging. Mean $\\pm$ standard deviation over the pretraining runs."),
    "pairs": ("tab:pairs", "Binary linear probes between JetClass-II classes on the largest training "
              "subset: $(1-\\mathrm{AUC})\\times10^{3}$, lower is better. $^{\\dagger}$: the vocabulary "
              "merges the two sides into one class; $^{\\ddagger}$: it merges the first side with part "
              "of the second. Mean $\\pm$ standard deviation over the pretraining runs; each random "
              "partition has one run."),
    "mass": ("tab:mass", "Resolution $\\sigma_{\\mathrm{eff}}$ of the jet mass regressed from frozen "
             "features with a ridge and a small nonlinear (MLP) probe; lower is better. The MLP has one "
             "hidden layer, stops early on a validation split and is averaged over __MLP_SEEDS__ seeds. "
             "Mean $\\pm$ standard deviation over the pretraining runs."),
    "ft": ("tab:ft", "Full fine-tuning against the training-set size: macro AUC (one-versus-rest) on "
           "JetClass-II and JetClass, AUC on top and quark/gluon tagging, the latter also on Herwig jets. "
           "Mean $\\pm$ standard deviation over the pretraining runs, one fine-tuning each; from "
           "scratch, over its fine-tuning seeds."),
    "ft_rej": ("tab:ftrej", "Background rejection $R_{50}$ and $R_{30}$ ($1/\\varepsilon_B$ at 50\\% and "
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
    out = {"tables/mlst_probes.tex": _float("probes", table_probes(fits, runs, fmt_pm, MAIN + MASS + (INIT,)), True),
           "tables/mlst_probes_n.tex": _float("probes_n", table_probes_n(fits, runs, fmt_pm, MAIN + (INIT,)), True),
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, runs, fmt_pm, merge_status(root),
                                                                MAIN + RAND + FLAV)),
           "tables/mlst_mass.tex": _float("mass", table_mass(mass, runs, fmt_pm, MAIN + MASS + (INIT,)),
                                          fill={"__MLP_SEEDS__": words(int(mlp_seeds(mass)))})}
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
