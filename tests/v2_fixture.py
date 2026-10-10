"""A synthetic second grid laid out as experiments/FIGS/data/v2/ holds it (the layout in
experiments/FIGS/make_tables.py), for tests/test_seed_level_v2.py, tests/test_make_tables_v2.py
and tests/test_figures_v2.py. Not a test file. Every value is a closed-form function of
(arm, run, tag, readout, task, probe), so a test can compute the mean it expects:

    ln(1 - AUC) = BASE[task] + STEP * rank(arm) + 0.01 run + SHIFT[tag] + POOLED * pooled
                  (+ MERGE where a random partition merges the task's probe pair)

with rank 0 for the finest vocabulary. Runs 1-3 train on an RTX 3090 in the job specs, as
the grid's do (a test moves one to an L40 for I7); the untrained trunk is init-s1..3, tag
init. The flavour pair is planted so that every clause of A14's P2 holds: F0 sits at the
17-class model, F1 0.3 below it in ln(1 - AUC) (more than half the 43-to-17 gap of 0.4),
F1r with F0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = 3
# the grid as launched: its leave-one-family-out arms were cut on 2026-10-09, and this synthetic
# design still covers the code paths that read them
_GRID = {a["name"]: a for a in json.loads((REPO / "configs/arms/v2_grid.launched.json").read_text())["arms"]}
LOFO_SELECTION = _GRID["L188_LOFO4P"]["extra_selection"]
# name, classes, mass weight, runs, tier, objective, section of experiments/FIGS/data/v2/
ARMS = [("L188", 188, None, RUNS, 1, "classification", "probe_ladder"),
        ("L162", 162, None, RUNS, 1, "classification", "probe_ladder"),
        ("R42_Q1", 43, None, RUNS, 1, "classification", "probe_ladder"),
        ("R16_Q1", 17, None, RUNS, 1, "classification", "probe_ladder"),
        ("L162_MASS", 162, 5.0, RUNS, 1, "classification+mass", "probe_ladder"),
        ("R16_Q1_MASS", 17, 5.0, RUNS, 1, "classification+mass", "probe_ladder"),
        *[(f"RAND2_p{i}", 17, None, 2, 1, "classification", "random_partitions") for i in range(1, 6)],
        *[(f, 17, None, RUNS, 1, "classification", "random_partitions") for f in ("FLAV_F0", "FLAV_F1", "FLAV_F1R")],
        ("R16_Q1_MASS_LM", 17, 1.74, RUNS, 2, "classification+mass", "mass_lambda_matched"),
        ("MPM", None, None, 2, 2, "mpm", "self_supervised"),
        ("R63_Q1", 64, None, RUNS, 3, "classification", "levels_64_30"),
        *[(f"{p}_LOFO4P", c, None, RUNS, 3, "classification", "leave_one_family_out")
          for p, c in (("L188", 188), ("L162", 162), ("R42_Q1", 43), ("R16_Q1", 17))],
        ("MPM_LOFO4P", None, None, 2, 3, "mpm", "leave_one_family_out")]
# the contrasts as launched, beside the grid as launched (the cut models lost theirs on 2026-10-09)
_real = json.loads((REPO / "configs/analysis/contrasts.v2.launched.json").read_text())
LABELS = {n: _real["labels"][n] for n, *_ in ARMS} | {"INIT": "untrained trunk"}
RANK = {"L188": 0, "L162": 1, "R63_Q1": 2, "R42_Q1": 3, "R16_Q1": 4, "L162_MASS": 1.5,
        "R16_Q1_MASS": 5.0, "R16_Q1_MASS_LM": 4.75, "MPM": 5, "INIT": 8,
        "FLAV_F0": 4, "FLAV_F1": 3.25, "FLAV_F1R": 4, "MPM_LOFO4P": 5,
        **{f"RAND2_p{i}": 4 for i in range(1, 6)},
        **{f"{p}_LOFO4P": r for p, r in (("L188", 0), ("L162", 1), ("R42_Q1", 3), ("R16_Q1", 4))}}
TAGS = ("best70", "wavg", "bestval", "best70_bn", "bestval_bn")
SHIFT = {"best70": 0.0, "wavg": 0.03, "bestval": 0.0, "best70_bn": -0.02, "bestval_bn": -0.02,
         "init": 0.0}
POOLED = 0.1
STEP = 0.4
MERGE = 0.3
BASE = {"bvc_resonant": -6.0, "bvc_4prong": -5.0, "bc_vs_rest": -4.0}
EPS = {"bvc_resonant": [0.5, 0.7, 0.9], "bvc_4prong": [0.5, 0.7, 0.9], "bc_vs_rest": [0.6, 0.4]}
N_BKG, N_SIG = 1000, 200
ALIGN, MALIGN, RALIGN = "a" * 64, "m" * 64, "r" * 64
GPU = {k: "NVIDIA-GeForce-RTX-3090" for k in range(1, RUNS + 1)}
SIGNALS = ("label_X_bb", "label_X_YY_bbbb", "label_X_YY_qqqq")
FAMILY = ("label_X_YY_bbbb", "label_X_YY_qqqq")    # the left-out family's members among SIGNALS
LOFO_EFFECT = 0.4                                     # ln sigma_min, family unseen


def slug(arm: str) -> str:
    return arm.lower().replace("_", "")


def run_dir(arm: str, k: int) -> str:
    return f"init-s{k}" if arm == "INIT" else f"mtx-{slug(arm)}-s{k}"


def _pe():
    s = importlib.util.spec_from_file_location("paired_errors_fx", REPO / "experiments/STATS/paired_errors.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.CONTRASTS["v2"] = REPO / "configs/analysis/contrasts.v2.launched.json"   # the design as launched
    return m


_MERGED = None


def merged(arm: str, task: str) -> bool:
    """Whether random partition `arm` merges `task`'s probe pair (the real merge table)."""
    global _MERGED
    if _MERGED is None:
        pe = _pe()
        _MERGED = {d["task"]: set(d["merged"]) for d in
                   pe.partition_design(pe.load_spec(pe.CONTRASTS["v2"]), [f"RAND2_p{i}" for i in range(1, 6)])}
    return arm in _MERGED.get(task, set())


def log1m(arm, k, tag, readout, task, probe="linear"):
    return (BASE[task] + STEP * RANK[arm] + 0.01 * k + SHIFT[tag] + (MERGE if merged(arm, task) else 0.0)
            + (POOLED if readout == "pooled" else 0.0) + (0.05 if probe == "mlp" else 0.0))


def rejection(arm, k, eps):
    return 50.0 + 10.0 * RANK[arm] + k + 10 * float(eps)


def models(tier3: bool = True):
    """(arm, run, tag, readout, section) of every frozen readout the fixture writes."""
    out = []
    for name, _, _, runs, tier, obj, sec in ARMS:
        if tier == 3 and not tier3:
            continue
        for k in range(1, runs + 1):
            for tag in TAGS:
                for ro in (("pooled",) if obj == "mpm" else ("features", "pooled")):
                    out.append((name, k, tag, ro, sec))
    out += [("INIT", k, "init", ro, "probe_ladder") for k in range(1, RUNS + 1)
            for ro in ("features", "pooled")]
    return out


def _write(p: pathlib.Path, doc) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return p


def write_specs(root: pathlib.Path, gpu: dict = GPU) -> None:
    for name, _, _, runs, _, _, _ in ARMS:
        for k in range(1, runs + 1):
            spec = (f"# synthetic\nspec:\n  affinity:\n    nodeAffinity:\n"
                    f"            - key: nvidia.com/gpu.product\n              operator: In\n"
                    f"              values: [{gpu[k]}]\n")
            p = root / "experiments/MTX/k8s/v2/grid" / f"job-mtx2-{slug(name)}-s{k}-raunav.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(spec)


def write_design(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    arms = []
    for n, c, lam, r, t, o, _ in ARMS:
        a = {"name": n, "config": f"configs/arms/{n}.yaml", "num_classes": c, "mass_lambda": lam,
             "runs": r, "tier": t, "extra_selection": None, "objective": o}
        if n.endswith("_LOFO4P"):
            a.update(extra_selection=LOFO_SELECTION, parent=n.removesuffix("_LOFO4P"))
        arms.append(a)
    g = _write(root / "configs/arms/v2_grid.json", {"arms": arms})
    c = _write(root / "configs/analysis/contrasts.v2.json",
               {"version": "v2", "labels": LABELS, "checkpoints": _real["checkpoints"],
                "readouts": _real["readouts"], "twins": _real["twins"],
                "references": {"INIT": {**_real["references"]["INIT"], "runs": RUNS}},
                "contrasts": []})
    write_specs(root)
    return g, c


def v2(root: pathlib.Path, sec: str) -> pathlib.Path:
    return root / "experiments/FIGS/data/v2" / sec


def _probe_doc(arm, k, tag, ro):
    name = f"{run_dir(arm, k).removeprefix('mtx-')}@{tag}"
    tasks = {}
    for task in BASE:
        cells = {}
        for probe in ("linear", "mlp"):
            lm = log1m(arm, k, tag, ro, task, probe)
            at = {f"{e:.2f}": {"rejection": rejection(arm, k, e), "eps_b": 1 / rejection(arm, k, e),
                               "rejection_is_bound": False, "n_bkg_pass": int(N_BKG / rejection(arm, k, e)),
                               "rel_stat_err": 0.1} for e in EPS[task]}
            first = at[f"{EPS[task][0]:.2f}"]
            cells[probe] = {"auc": 1 - math.exp(lm), "log1m_auc": lm, "log1m_auc_censored": False,
                            "rejection": first["rejection"], "rejection_is_bound": False,
                            "rejection_eps_s": EPS[task][0], "rejection_at": at}
        tasks[task] = {"eps_s": EPS[task], "n": 3000, "n_signal": 1000, "n_signal_test": N_SIG,
                       "n_background_test": N_BKG, "names": [f"label_{task}_a", f"label_{task}_b"],
                       "collapsed_at": ["R16_Q1"], "arms": {name: cells}}
    return {"n_jets_total": 5000, "row_alignment_sha256": ALIGN, "eps_s_default": 0.5,
            "arm_checkpoints": {name: hashlib.sha256(f"{arm}{k}{tag}".encode()).hexdigest()},
            "readout": ro, "tasks": tasks}


def _mass_doc(arm, k, tag, ro):
    name = f"{run_dir(arm, k).removeprefix('mtx-')}@{tag}"
    entry = {"target": {"sigma_eff": 0.25},
             "provenance": {"checkpoint_sha256": hashlib.sha256(f"{arm}{k}{tag}".encode()).hexdigest()}}
    for probe, off in (("ridge", 0.0), ("mlp", -0.02)):
        s = 0.15 + 0.01 * RANK[arm] + 0.001 * k + off + SHIFT[tag] / 10
        entry[probe] = {"sigma_eff": s, "sigma68_central": s * 1.1, "sd": s * 1.5, "median": 0.0,
                        "fractional": s - 0.25, "tail_fraction": 0.05, "n": 200, "val_r2": 0.6 + 0.01 * k}
    return {"row_alignment_sha256": MALIGN, "resolution_statistic": "sigma_eff", "centering": "class",
            "tail_at": 0.3, "readout": ro, "n_jets_total": 1200, "n_jets_valid": 1100,
            "n_classes_used": 150, "centering_detail": {"n_jets_used": 1000, "split": [800, 100, 200]},
            "arms": {name: entry}}


RECOVERY_RUNGS = ("L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1")
SIZES = [100, 1000]


def recovery_acc(arm, k, rung, n):
    return 0.5 + 0.02 * RECOVERY_RUNGS.index(rung) - 0.03 * RANK[arm] + 0.001 * k + (0.01 if n == 1000 else 0)


def _recovery_doc(arm, k, tag, ro):
    name = f"{run_dir(arm, k).removeprefix('mtx-')}@{tag}"
    rungs = {}
    for rung in RECOVERY_RUNGS:
        cell = {"n_groups": 10, "chance": 0.1,
                "curve": [{"n_train": n, "balanced_accuracy": recovery_acc(arm, k, rung, n), "converged": True}
                          for n in SIZES]}
        if rung == "L188":
            cell["mlp"] = {"n_train": SIZES[-1], "balanced_accuracy": recovery_acc(arm, k, rung, SIZES[-1]) + 0.02,
                           "converged": True}
        rungs[rung] = cell
    return {"row_alignment_sha256": RALIGN, "readout": ro, "n_test": 500, "n_pool": SIZES[-1],
            "sizes": SIZES, "mlp": {"hidden": 64}, "arms": {name: {"own_rung": arm, "rungs": rungs}}}


def write_frozen(root: pathlib.Path, link_bestval: bool = True, tier3: bool = True) -> list:
    """Every frozen readout of models(); run 1's bestval is a byte-identical copy of its
    best70 file, named for best70, as the readout jobs leave a linked tag."""
    written = []
    for arm, k, tag, ro, sec in models(tier3):
        rd = run_dir(arm, k)
        src_tag = "best70" if (link_bestval and k == 1 and tag == "bestval") else tag
        src_tag = "best70_bn" if (link_bestval and k == 1 and tag == "bestval_bn") else src_tag
        written.append(_write(v2(root, sec) / "probe" / rd / tag / ro / "probe_results.json",
                              _probe_doc(arm, k, src_tag, ro)))
        _write(v2(root, sec) / "mass_resolution" / rd / tag / ro / "mass_resolution.json",
               _mass_doc(arm, k, src_tag, ro))
        if tag == "best70" and ro == "features" and arm in RECOVERY_RUNGS:
            _write(v2(root, "label_recovery") / "label_recovery_curve" / rd / tag / ro
                   / "label_recovery_curve.json", _recovery_doc(arm, k, tag, ro))
    return written


def ft_init(arm: str, k: int) -> str:
    return f"{slug(arm)}-v2-s{k}" if arm.startswith("MPM") else f"{slug(arm)}-s{k}"


SCRATCH_RANK = 9          # the from-scratch reference, worse than every pretrained model
FT_SIZES = ("N1000", "N10000", "N1000000")


def ft_value(arm, k, n, leg, rank=None):
    r = RANK[arm] if rank is None else rank
    return 0.1 * (1 + 0.1 * r + 0.01 * k) / (1 + math.log10(int(n[1:]))) * (1 if leg == 1 else 0.5)


def _ft_cell(v, classes):
    return {"macro_auc_ovr": 1 - v, "accuracy": 0.6 - v, "n_classes_present": classes, "n_jets": 5000,
            "n_jets_auc": 1250}


def write_ft(root: pathlib.Path, scope: str = "_t12", scratch_rank: float = SCRATCH_RANK) -> list:
    """The leg-1 and leg-2 read-outs of rule best70: every model at fine-tuning seed 1, the
    from-scratch reference beside them at fine-tuning seeds 1-3, as the v2 read-outs hold it."""
    out = []
    for leg, classes in ((1, 162), (2, 10)):
        cells = {}
        for name, _, _, runs, tier, _, _ in ARMS:
            if scope == "_t12" and tier == 3:
                continue
            for k in range(1, runs + 1):
                cells[ft_init(name, k)] = {n: {"s1": _ft_cell(ft_value(name, k, n, leg), classes)} for n in FT_SIZES}
        cells["scratch-v2"] = {n: {f"s{s}": _ft_cell(ft_value("INIT", s, n, leg, scratch_rank), classes)
                                   for s in (1, 2, 3)} for n in FT_SIZES}
        out.append(_write(v2(root, "finetune") / f"best70{scope}_leg{leg}_metrics.json",
                          {"row_alignment_sha256": ALIGN, "cells": cells}))
    _write(v2(root, "finetune_references") / "scratch_leg1_metrics.json",
           {"cells": {"scratch-v2": json.loads(out[0].read_text())["cells"]["scratch-v2"]}})
    return out


def bench_cell(arm, k, rule, leg):
    r = 400.0 / (1 + 0.1 * RANK.get(arm, SCRATCH_RANK) + 0.01 * k) * (1.02 if rule == "wavg" else 1.0)
    return {"accuracy": 0.9 - 0.001 * RANK.get(arm, SCRATCH_RANK) - 0.0003 * k,
            "auc": 0.98 - 0.001 * RANK.get(arm, SCRATCH_RANK) - 0.0002 * k,
            "log1m_auc": math.log(0.02 + 0.001 * RANK.get(arm, SCRATCH_RANK)), "log1m_auc_censored": False,
            "r50": r, "r50_is_bound": False, "r50_n_bkg_pass": int(200000 / r), "r30": 3 * r,
            "r30_is_bound": False, "r30_n_bkg_pass": int(200000 / (3 * r)), "n_jets": 200000}


def write_bench(root: pathlib.Path) -> None:
    """Both rules' benchmark read-outs at the freeze: best validation at two sizes (with the
    from-scratch reference at fine-tuning seeds 1-3) and the last epoch at the full set."""
    arms = [(n, r) for n, _, _, r, tier, _, _ in ARMS if tier < 3]
    for rule in ("best70", "wavg"):
        for kind, sets, sizes, scratch in (("bench_metrics", ("top", "qg"), ("N1000", "N100000"), True),
                                           ("bench_metrics_herwig", ("qg",), ("N1000", "N100000"), True),
                                           ("bench_metrics_last", ("top", "qg"), ("N100000",), False),
                                           ("bench_metrics_herwig_last", ("qg",), ("N100000",), False)):
            cells = {s: {ft_init(n, k): {N: {"s1": bench_cell(n, k, rule, s)} for N in sizes}
                         for n, r in arms for k in range(1, r + 1)} for s in sets}
            if scratch:
                for s in sets:
                    cells[s]["scratch-v2"] = {N: {f"s{f}": bench_cell("scratch", f, rule, s) for f in (1, 2, 3)}
                                              for N in sizes}
            _write(v2(root, "benchmarks") / f"{rule}_t12_{kind}.json", {"cells": cells})


def write_pretraining(root: pathlib.Path) -> None:
    for name, _, _, runs, _, _, _ in ARMS:
        for k in range(1, runs + 1):
            d = v2(root, "pretraining") / run_dir(name, k)
            _write(d / "best_window_epoch.json", {"epoch": 70 + k + (2 if name == "R16_Q1" else 0)})
            _write(d / "best_epoch.json", {"epoch": 40 + k if name == "L162" else 70 + k})


def _replicates(points: dict, b: int, seed: int) -> tuple[dict, dict]:
    """{key: ln point} -> replicate vectors with test noise shared within a (task, probe)."""
    rng = np.random.default_rng(seed)
    shared, vec, meta = {}, {}, {}
    for key, point in points.items():
        group = "|".join(key.split("|")[:3])
        sh = shared.setdefault(group, rng.normal(0, 0.01, b))
        vec[key] = np.exp(np.r_[point, point + sh + rng.normal(0, 0.003, b)])
        meta[key] = {"jets": group, "n": 1000}
    return vec, meta


def write_paired(root: pathlib.Path, wavg_shift: dict | None = None, b: int = 40) -> pathlib.Path:
    """ratios.json of the probes family, from paired_errors.ratios under the real contrasts
    file (its model names are the real grid's runs 1-3), on replicates whose point is the
    fixture's ln(1 - AUC) on two-prong and four-prong b vs c, at every checkpoint but the
    global best's twin, through both readouts; the models of tiers 1-2."""
    pe = _pe()
    shift = wavg_shift or {}
    points = {}
    for task in ("bvc_resonant", "bvc_4prong"):
        for kind in ("linear", "mlp"):
            for arm, k, tag, ro, _ in models(tier3=False):
                if arm in ("INIT", "MPM") or tag == "bestval_bn":
                    continue
                model = f"{ft_init(arm, k)}@{tag}" + ("" if ro == "features" else ":pooled")
                points[f"probe|{task}|{kind}|{model}|1-auc"] = (
                    log1m(arm, k, tag, ro, task, kind) + (shift.get(arm, 0.0) if tag == "wavg" else 0.0))
    res = pe.ratios(*_replicates(points, b, 7))
    res.pop("audit_b3")
    return _write(v2(root, "paired_errors") / "probes" / "ratios.json", {"provenance": {"synthetic": True}, **res})


def write_paired_ft(root: pathlib.Path, b: int = 40) -> pathlib.Path:
    """ratios.json of the fine-tuning family: 1 - macro AUC at 10^3 jets on both legs, rule
    best70 and wavg, every model of tiers 1-2 and the self-supervised runs."""
    pe = _pe()
    points = {}
    for leg in (1, 2):
        for rule in ("best70", "wavg"):
            for n, _, _, runs, tier, _, _ in ARMS:
                if tier == 3:
                    continue
                for k in range(1, runs + 1):
                    points[f"ft|leg{leg}|N1000|{ft_init(n, k)}@{rule}|1-macro_auc"] = math.log(
                        ft_value(n, k, "N1000", leg))
    res = pe.ratios(*_replicates(points, b, 11))
    res.pop("audit_b3")
    return _write(v2(root, "paired_errors") / "finetune" / "ratios.json", {"provenance": {"synthetic": True}, **res})


def write_loss_share(root: pathlib.Path) -> pathlib.Path:
    """loss_share.json as paired_errors.py a11-shares writes it, three mass-output arms."""
    keys = {"L162_MASS": ("162+mass", 0.15), "R16_Q1_MASS": ("17+mass", 0.33), "R16_Q1_MASS_LM": ("17+mass_matched", 0.15)}
    return _write(v2(root, "mass_lambda_matched") / "loss_share.json",
                  {"shares": {k: [s + 0.001 * r for r in (1, 2, 3)] for k, s in keys.values()},
                   "grad_shares": {k: [s / 2 + 0.001 * r for r in (1, 2, 3)] for k, s in keys.values()},
                   "grad_cosine": {k: [0.1 * r for r in (1, 2, 3)] for k, _ in keys.values()}})


# ------------------------------------------------------------------ anomaly
def jitter(arm, k) -> float:
    """A run-to-run scatter that differs between arms, so paired spreads are not zero."""
    return 0.004 * ((7 * k + 3 * int(4 * RANK[arm])) % 5)


def ln_sigma_min(arm, k, tag, ro, sig, fam="knn"):
    lofo = arm.endswith("_LOFO4P") and sig in FAMILY
    return (0.05 * RANK[arm] + 0.01 * k + jitter(arm, k) + SHIFT[tag] + (0.05 if ro == "pooled" else 0.0)
            + (LOFO_EFFECT if lofo else 0.0) + (0.02 if fam == "mahalanobis" else 0.0)
            + (0.03 if fam == "class_sum_matched" else 0.0))


def _rung(arm):
    base = arm.removesuffix("_MASS_LM").removesuffix("_MASS").removesuffix("_LOFO4P")
    return base if base in RECOVERY_RUNGS else "none"


def write_anomaly(root: pathlib.Path, tier3: bool = True, sets=("t12",)) -> None:
    """The merged anomaly results (knn, Mahalanobis) per checkpoint and readout, the output
    layers' heads, and anomaly_summary.py --grid over each, as experiments/FIGS/data/v2/anomaly/
    <set>/ holds them: the vocabulary ladder, its family-out arms, the self-supervised arms."""
    s = importlib.util.spec_from_file_location("anomaly_summary_fx", REPO / "experiments/EVAL/anomaly_summary.py")
    AS = importlib.util.module_from_spec(s)
    s.loader.exec_module(AS)
    for st in sets:
        arms = [(n, r, o) for n, c, lam, r, tier, o, _ in ARMS
                if (tier < 3 or (tier3 and st != "t12")) and lam is None and (n in RECOVERY_RUNGS or n.endswith("LOFO4P") or o == "mpm")]
        d = v2(root, "anomaly") / st
        heads = {"models": {}}
        for n, r, o in arms:
            if o == "mpm":
                continue
            for k in range(1, r + 1):
                heads["models"][f"{slug(n)}-s{k}"] = {"rung": _rung(n), "checkpoints": {
                    t: {"head": {"top1_accuracy": 0.6 + 0.001 * k, "mean_p_qcd_resonant": 0.1 + 0.001 * k},
                        "anomaly": {g: {N: {"class_sum_matched": {"sigma_min": math.exp(ln_sigma_min(n, k, t, "features", g,
                                                                                                         "class_sum_matched")),
                                                                  "max_sic": 3.0}} for N in ("2000", "4000")}
                                    for g in SIGNALS}} for t in TAGS}}
        hp = _write(d / "anomaly_heads.json", heads)
        for tag in ("best70", "wavg", "bestval", "best70_bn"):
            for ro in ("features", "pooled"):
                doc = {"row_alignment_sha256": "q" * 64, "sigma_t": 5.0, "stat_cut": 0.2, "min_bkg_pass": 25,
                       "trainings": 10, "n_bkg": 100000, "n_template": 100000, "arms": {}}
                for n, r, o in arms:
                    if o == "mpm" and ro != "pooled":
                        continue
                    for k in range(1, r + 1):
                        doc["arms"][f"{slug(n)}-s{k}"] = {"rung": _rung(n), "cache": {"readout": ro}, "signals": {
                            g: {N: {fam: {"sigma_min": math.exp(ln_sigma_min(n, k, tag, ro, g, fam)), "max_sic": 3.0 - 0.01 * k}
                                    for fam in ("knn", "mahalanobis")} | {"rng_seeds": [1]} for N in ("2000", "4000")}
                            for g in SIGNALS}}
                mp = _write(d / "merged" / tag / ro / "anomaly_results.json", doc)
                assert AS.main(["--grid", str(root / "configs/arms/v2_grid.json"), "--anomaly", str(mp),
                                "--heads", str(hp), "--out", str(d / "summary" / tag / ro)]) == 0


def write_paired_anomaly(root: pathlib.Path, tier3: bool = True) -> pathlib.Path:
    """The output ratio's paired ratios (paired_errors.py over anomaly_heads' sigma_min, B = 0)."""
    pe = _pe()
    vec, meta = {}, {}
    for n, _, lam, r, tier, o, _ in ARMS:
        if lam is not None or o == "mpm" or (tier == 3 and not tier3) or not (n in RECOVERY_RUNGS or n.endswith("LOFO4P")):
            continue
        for k in range(1, r + 1):
            for tag in ("best70", "wavg", "bestval", "best70_bn"):
                for g in SIGNALS:
                    key = f"anomaly|{g}|class_sum_matched|{ft_init(n, k)}@{tag}|sigma_min@2000"
                    vec[key] = np.array([math.exp(ln_sigma_min(n, k, tag, "features", g, "class_sum_matched"))])
                    meta[key] = {"jets": "q", "n": 100000, "max_sic": 3.0}
    res = pe.ratios(vec, meta)
    res.pop("audit_b3")
    return _write(v2(root, "paired_errors") / "anomaly" / "ratios.json", {"provenance": {"synthetic": True}, **res})


# ------------------------------------------------------------------ real data
def write_real_data(root: pathlib.Path, wavg_scale: float = 1.05) -> pathlib.Path:
    """The second grid's real-data fit, made from the first grid's committed fit_v6: every
    model renamed <run>-best70 (l162-s1b as run 1), a copy at -wavg with every yield scaled
    by `wavg_scale`, and its analysis written by refit_from_bins.per_arm_v2 on the fixture's
    grid; the injection summary is the first grid's. Values at best70 are then the first
    grid's, which the test checks."""
    s = importlib.util.spec_from_file_location("refit_fx", REPO / "experiments/AOJ/refit_from_bins.py")
    RB = importlib.util.module_from_spec(s)
    s.loader.exec_module(RB)
    v1 = json.loads((REPO / "experiments/FIGS/data/aoj_full_v1/fit_v6/results.json").read_text())
    names = {m: m.replace("-s1b", "-s1") + "-best70" for m in v1["models"] if m != "sophon-public"}

    def ren(x):
        if isinstance(x, dict):
            return {names.get(k, k): ren(v) for k, v in x.items()}
        if isinstance(x, list):
            return [ren(v) for v in x]
        return names.get(x, x) if isinstance(x, str) else x
    res = ren(v1)
    for m in list(names.values()):
        w = json.loads(json.dumps(res["models"][m]))
        w["top"]["signal_yield"] *= wavg_scale
        res["models"][m.replace("-best70", "-wavg")] = w
    d = v2(root, "real_data") / "t12"
    rp = _write(d / "fit_v6" / "results.json", res)
    grid = root / "configs/arms/v2_grid.json"
    _write(d / "analysis_v6" / "aoj_top.json",
           {"provenance": {"input": "/scratch/fit_v6/results.json",
                           "input_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
            "per_checkpoint": RB.per_arm_v2(res, grid)})
    inj = d / "injection" / "summary.json"
    inj.parent.mkdir(parents=True, exist_ok=True)
    inj.write_bytes((REPO / "experiments/FIGS/data/aoj_injection_v1/summary.json").read_bytes())
    return d


def write_real_data_t3(root: pathlib.Path, held_sha: str | None = None) -> pathlib.Path:
    """Tier 3's real-data fit, as fit_v6.py --v2 --shape-from writes it: the public
    checkpoint and the 64-class runs (the first grid's 188-class fits renamed, yields x0.9),
    the pooled shape held from t12's fit (held_from, held_from_sha256; `held_sha` replaces
    the hash, for the refusal test)."""
    s = importlib.util.spec_from_file_location("refit_fx3", REPO / "experiments/AOJ/refit_from_bins.py")
    RB = importlib.util.module_from_spec(s)
    s.loader.exec_module(RB)
    t12 = v2(root, "real_data") / "t12" / "fit_v6" / "results.json"
    r12 = json.loads(t12.read_text())
    models = {"sophon-public": r12["models"]["sophon-public"]}
    for tag in ("best70", "wavg"):
        for k in range(1, 6):
            m = json.loads(json.dumps(r12["models"][f"l188-s{k}-{tag}"]))
            m["top"]["signal_yield"] *= 0.9
            models[f"r63q1-s{k}-{tag}"] = m
    pool = [n for n in models if n.endswith("-best70")]
    res = {**r12, "models": models,
           "pooled_shape": {**r12["pooled_shape"], "pool": pool,
                            "held_from": "/data/results/aoj/full_v2/t12/fit_v6/results.json",
                            "held_from_sha256": held_sha or hashlib.sha256(t12.read_bytes()).hexdigest()},
           "shape_variations": {"top": {**r12["shape_variations"]["top"], "pool": pool}}}
    d = v2(root, "real_data") / "t3"
    rp = _write(d / "fit_v6" / "results.json", res)
    _write(d / "analysis_v6" / "aoj_top.json",
           {"provenance": {"input": "/scratch/fit_v6/results.json",
                           "input_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
            "per_checkpoint": RB.per_arm_v2(res, root / "configs/arms/v2_grid.json")})
    return d


def seed_level_v2(root: pathlib.Path, out: str = "analysis") -> pathlib.Path:
    """experiments/STATS/seed_level.py --v2 over every frozen section present."""
    s = importlib.util.spec_from_file_location("seed_level_fx", REPO / "experiments/STATS/seed_level.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    secs = [str(v2(root, sec)) for sec in ("probe_ladder", "levels_64_30", "leave_one_family_out",
                                           "random_partitions", "self_supervised", "mass_lambda_matched",
                                           "label_recovery") if v2(root, sec).is_dir()]
    dest = v2(root, "probe_ladder") / out
    assert m.main(["--v2", *secs, "--grid", str(root / "configs/arms/v2_grid.json"),
                   "--contrasts", str(root / "configs/analysis/contrasts.v2.json"), "--out", str(dest)]) == 0
    return dest
