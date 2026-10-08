"""A synthetic second grid laid out as experiments/FIGS/data/v2/ holds it (the layout in
experiments/FIGS/make_tables.py), for tests/test_seed_level_v2.py and
tests/test_make_tables_v2.py. Not a test file. Every value is a closed-form function of
(arm, run, tag, readout, task, probe), so a test can compute the mean it expects:

    ln(1 - AUC) = BASE[task] + STEP * rank(arm) + 0.01 run + SHIFT[tag] + POOLED * pooled

with rank 0 for the finest vocabulary. Runs 1-2 train on an RTX 3090 and run 3 on an
L40 in the job specs (I7 test); the untrained trunk is init-s1..3, tag init.
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
# name, classes, mass weight, runs, tier, objective, section of experiments/FIGS/data/v2/
ARMS = [("L188", 188, None, RUNS, 1, "classification", "probe_ladder"),
        ("L162", 162, None, RUNS, 1, "classification", "probe_ladder"),
        ("R42_Q1", 43, None, RUNS, 1, "classification", "probe_ladder"),
        ("R16_Q1", 17, None, RUNS, 1, "classification", "probe_ladder"),
        ("L162_MASS", 162, 5.0, RUNS, 1, "classification+mass", "probe_ladder"),
        ("R16_Q1_MASS", 17, 5.0, RUNS, 1, "classification+mass", "probe_ladder"),
        ("MPM", None, None, 2, 2, "mpm", "self_supervised"),
        ("R63_Q1", 64, None, RUNS, 3, "classification", "levels_64_30")]
LABELS = {"L188": "188", "L162": "162", "R63_Q1": "64", "R42_Q1": "43", "R16_Q1": "17",
          "L162_MASS": "162+mass", "R16_Q1_MASS": "17+mass", "MPM": "self-supervised",
          "INIT": "untrained trunk"}
RANK = {"L188": 0, "L162": 1, "R63_Q1": 2, "R42_Q1": 3, "R16_Q1": 4, "L162_MASS": 1.5,
        "R16_Q1_MASS": 4.5, "MPM": 5, "INIT": 8}
TAGS = ("best70", "wavg", "bestval", "best70_bn", "bestval_bn")
SHIFT = {"best70": 0.0, "wavg": 0.03, "bestval": 0.0, "best70_bn": -0.02, "bestval_bn": -0.02,
         "init": 0.0}
POOLED = 0.1
STEP = 0.4
BASE = {"bvc_resonant": -6.0, "bc_vs_rest": -4.0}
EPS = {"bvc_resonant": [0.5, 0.7, 0.9], "bc_vs_rest": [0.6, 0.4]}
N_BKG, N_SIG = 1000, 200
ALIGN, MALIGN, RALIGN = "a" * 64, "m" * 64, "r" * 64
GPU = {1: "NVIDIA-GeForce-RTX-3090", 2: "NVIDIA-GeForce-RTX-3090", 3: "NVIDIA-L40"}


def slug(arm: str) -> str:
    return arm.lower().replace("_", "")


def run_dir(arm: str, k: int) -> str:
    return f"init-s{k}" if arm == "INIT" else f"mtx-{slug(arm)}-s{k}"


def log1m(arm, k, tag, readout, task, probe="linear"):
    return (BASE[task] + STEP * RANK[arm] + 0.01 * k + SHIFT[tag]
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


def write_design(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    grid = {"arms": [{"name": n, "config": f"configs/arms/{n}.yaml", "num_classes": c,
                      "mass_lambda": lam, "runs": r, "tier": t, "extra_selection": None,
                      "objective": o} for n, c, lam, r, t, o, _ in ARMS]}
    g = _write(root / "configs/arms/v2_grid.json", grid)
    real = json.loads((REPO / "configs/analysis/contrasts.v2.json").read_text())
    c = _write(root / "configs/analysis/contrasts.v2.json",
               {"version": "v2", "labels": LABELS, "checkpoints": real["checkpoints"],
                "readouts": real["readouts"], "twins": real["twins"],
                "references": {"INIT": {**real["references"]["INIT"], "runs": RUNS}},
                "contrasts": []})
    for name, _, _, runs, _, _, _ in ARMS:
        for k in range(1, runs + 1):
            spec = (f"# synthetic\nspec:\n  affinity:\n    nodeAffinity:\n"
                    f"            - key: nvidia.com/gpu.product\n              operator: In\n"
                    f"              values: [{GPU[k]}]\n")
            p = root / "experiments/MTX/k8s/v2/grid" / f"job-mtx2-{slug(name)}-s{k}-raunav.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(spec)
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
        if tag == "best70" and ro == "features" and arm not in ("MPM",) and arm in RECOVERY_RUNGS:
            _write(v2(root, "label_recovery") / "label_recovery_curve" / rd / tag / ro
                   / "label_recovery_curve.json", _recovery_doc(arm, k, tag, ro))
    return written


def ft_init(arm: str, k: int) -> str:
    return f"{slug(arm)}-v2-s{k}" if arm == "MPM" else f"{slug(arm)}-s{k}"


def ft_value(arm, k, n, leg):
    return 0.1 * (1 + 0.1 * RANK[arm] + 0.01 * k) / (1 + math.log10(int(n[1:]))) * (1 if leg == 1 else 0.5)


def write_ft(root: pathlib.Path, scope: str = "_t12") -> list:
    out = []
    for leg, classes in ((1, 162), (2, 10)):
        cells = {}
        for name, _, _, runs, tier, _, _ in ARMS:
            if scope == "_t12" and tier == 3:
                continue
            for k in range(1, runs + 1):
                cells[ft_init(name, k)] = {n: {"s1": {"macro_auc_ovr": 1 - ft_value(name, k, n, leg),
                                                      "accuracy": 0.6 - ft_value(name, k, n, leg),
                                                      "n_classes_present": classes, "n_jets": 5000,
                                                      "n_jets_auc": 1250}}
                                           for n in ("N1000", "N10000")}
        out.append(_write(v2(root, "finetune") / f"best70{scope}_leg{leg}_metrics.json",
                          {"row_alignment_sha256": ALIGN, "cells": cells}))
    return out


def write_pretraining(root: pathlib.Path) -> None:
    for name, _, _, runs, _, _, _ in ARMS:
        for k in range(1, runs + 1):
            d = v2(root, "pretraining") / run_dir(name, k)
            _write(d / "best_window_epoch.json", {"epoch": 70 + k + (2 if name == "R16_Q1" else 0)})
            _write(d / "best_epoch.json", {"epoch": 40 + k if name == "L162" else 70 + k})


def _pe():
    s = importlib.util.spec_from_file_location("paired_errors_fx", REPO / "experiments/STATS/paired_errors.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def write_paired(root: pathlib.Path, wavg_shift: dict | None = None, b: int = 40) -> pathlib.Path:
    """ratios.json of the probes family, from paired_errors.ratios on replicates whose point
    is the fixture's ln(1 - AUC) on bvc_resonant (test noise shared, plus each model's own),
    under the real contrasts file: its model names are the real grid's runs 1-3."""
    pe = _pe()
    shift = wavg_shift or {}
    rng = np.random.default_rng(7)
    vec, meta = {}, {}
    for kind in ("linear", "mlp"):
        shared = rng.normal(0, 0.01, b)
        for arm, k, tag, ro, _ in models():
            if arm in ("INIT", "MPM", "R63_Q1") or tag == "bestval_bn":
                continue
            model = f"{ft_init(arm, k)}@{tag}" + ("" if ro == "features" else ":pooled")
            point = log1m(arm, k, tag, ro, "bvc_resonant", kind) + (shift.get(arm, 0.0) if tag == "wavg" else 0.0)
            key = f"probe|bvc_resonant|{kind}|{model}|1-auc"
            vec[key] = np.exp(np.r_[point, point + shared + rng.normal(0, 0.003, b)])
            meta[key] = {"jets": "j", "n": 1000}
    res = pe.ratios(vec, meta)
    res.pop("audit_b3")
    return _write(v2(root, "paired_errors") / "probes" / "ratios.json",
                  {"provenance": {"synthetic": True}, **res})


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
