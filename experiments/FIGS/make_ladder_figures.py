#!/usr/bin/env python3
"""The two figures that put pretraining vocabulary size on an axis.

F1  GRANULARITY ON THE X-AXIS. log(1 - AUC) of the frozen probes against the
    number of classes the model was pretrained on, one line per probe task,
    linear probe solid and nonlinear dashed, per-seed points behind the mean.
F2  THE PAIRING, MADE VISIBLE. The same result as five paired differences per
    task against the reference vocabulary, with the interval the pre-specified
    analysis computed -- because a reader cannot tell from F1 whether the gap
    is bigger than the seed-to-seed scatter, and the pairing is what makes it
    so.

EVERYTHING IS READ, NOTHING IS TYPED. The class counts on the x-axis, the tasks,
the seeds, the intervals and the rung at which each distinction is merged all
come out of files: `seed_level_results.json` for the measurements and the
intervals, `rung_label_maps.v1.csv` for the merge points. There is no number
literal in this file, and tests/test_ladder_figures.py greps for one.

A SATURATED CELL IS NOT A POINT. Where a probe reached AUC = 1 at the resolution
of the sample, log(1 - AUC) is a floor, not a measurement -- the electron/muon
task does this at the two finest vocabularies. Those cells are drawn as
downward carets at the floor, outside the mean line, so nobody reads a bound as
a value or a flat segment as an equality.

Usage:
    python3 experiments/FIGS/make_ladder_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style                                                       # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "experiments/FIGS/data/probe_ladder_v1/analysis/seed_level_results.json"
RUNG_MAP = REPO / "configs/labelmaps/rung_label_maps.v1.csv"
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]

TASK_LABELS = {
    "bvc_resonant": "b vs c, two-prong resonance",
    "bvc_qcd": "b vs c, QCD",
    "retained_topology": "two-prong vs four-prong",
    "ee_vs_mm": "electron vs muon pair",
    "bvc_4prong": "b vs c, four-prong",
    "visible_content": "visible decay content",
}


# ------------------------------------------------------------------ reading

def load(analysis: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(analysis).read_text())


def merge_levels(rung_map: pathlib.Path, names_by_task: dict, sizes: dict) -> dict:
    """The vocabulary size at which each task's two classes stop being distinct.

    Computed from the label map: the finest rung whose group column gives both
    of the task's classes the same value. This is what the annotation on F1
    claims, so it is derived here rather than copied from the probe file's
    `collapsed_at` -- which is then used as a cross-check.
    """
    with pathlib.Path(rung_map).open() as f:
        rows = {r["class_name"]: r for r in csv.DictReader(f)}
    out = {}
    for task, names in names_by_task.items():
        if any(n not in rows for n in names):
            continue
        merged = [r for r in RUNGS if len({rows[n][r] for n in names}) == 1]
        out[task] = {"rung": merged[0] if merged else None,
                     "level": sizes[merged[0]] if merged else None}
    return out


def vocabulary_sizes(rung_map: pathlib.Path) -> dict:
    """Classes per rung, counted from the map. The rung NAMES undercount by one."""
    with pathlib.Path(rung_map).open() as f:
        rows = list(csv.DictReader(f))
    return {r: len({row[r] for row in rows}) for r in RUNGS}


def task_names(analysis_dir: pathlib.Path) -> dict:
    """Which two classes each probe task separates, from any one ladder file.

    The class pair is a property of the task definition, identical in every
    seed's file; the loader refuses if two files disagree, because then the
    tasks are not the same task.
    """
    names, collapsed = {}, {}
    for p in sorted(pathlib.Path(analysis_dir).glob("s*.json")):
        for task, spec in json.loads(p.read_text())["tasks"].items():
            if names.setdefault(task, spec["names"]) != spec["names"]:
                raise SystemExit(f"FATAL: {p.name} defines task {task} over different classes "
                                 f"than an earlier file; these are not the same task")
            collapsed.setdefault(task, spec["collapsed_at"])
    return names, collapsed


def cells(A: dict, task: str, probe: str, level: int) -> list[dict]:
    rows = [r for r in A["table"] if r["task"] == task and r["probe"] == probe
            and r["level"] == level and not r.get("dropped_pair")]
    return sorted(rows, key=lambda r: r["seed"])


def plotted_tasks(A: dict) -> list:
    """The tasks the pre-specified analysis contrasts; the control tasks are tabulated only."""
    keys = set(A["pairwise_exploratory"])
    return [t for t in TASK_LABELS if t in keys] + sorted(keys - set(TASK_LABELS))


# ------------------------------------------------------------------ figure 1

def series(A: dict, task: str, probe: str, levels: list) -> dict:
    """One task/probe line: the measured levels, and the levels that only bound it."""
    xs, ys, sds, pts, bounds = [], [], [], [], []
    for lv in levels:
        rs = cells(A, task, probe, lv)
        if not rs:
            continue
        y = [r["log1m_auc"] for r in rs]
        if all(r["censored"] for r in rs):
            bounds.append(lv)           # a floor, not a value: drawn off the scale
            continue
        xs.append(lv)
        ys.append(float(np.mean(y)))
        sds.append(float(np.std(y, ddof=1)) if len(y) > 1 else 0.0)
        pts += [(lv, v) for v in y]
    return {"x": xs, "y": np.array(ys), "sd": np.array(sds), "points": pts, "bounds": bounds}


def fig1_granularity(A: dict, merges: dict, outdir=None):
    """log(1 - AUC) against vocabulary size, one line per task, two probe kinds.

    The y limits come from the measured cells only. A saturated cell would sit
    at the sample's floor, which is far below everything else and would flatten
    the whole figure into a line; it is drawn instead as a caret on the bottom
    axis, which is what "this is a bound, off the scale" looks like.
    """
    levels = A["levels_fine_to_coarse"]
    tasks = plotted_tasks(A)
    colour = style.colour_cycle(tasks)
    marker = style.marker_cycle(tasks)
    drawn = {(t, p): series(A, t, p, levels) for t in tasks for p in style.PROBE_LINESTYLES}

    fig, ax = plt.subplots(figsize=(style.FIG_W_TWO_COLUMN, style.FIG_W_TWO_COLUMN * 0.52))
    for task in tasks:
        merged_at = merges.get(task, {}).get("level")
        for probe, ls in style.PROBE_LINESTYLES.items():
            s = drawn[(task, probe)]
            if not s["x"]:
                continue
            label = None
            if probe == "linear":
                label = TASK_LABELS.get(task, task)
                if merged_at:
                    label += f" — merged at {merged_at}"
            ax.plot(s["x"], s["y"], linestyle=ls, color=colour[task], marker=marker[task],
                    fillstyle=style.PROBE_FILLSTYLES[probe], label=label, zorder=3)
            ax.fill_between(s["x"], s["y"] - s["sd"], s["y"] + s["sd"], color=colour[task],
                            alpha=0.15, lw=0, zorder=1)
            ax.plot([p[0] for p in s["points"]], [p[1] for p in s["points"]], linestyle="none",
                    marker=marker[task], color=colour[task], alpha=0.35,
                    markersize=plt.rcParams["lines.markersize"] * 0.6,
                    fillstyle=style.PROBE_FILLSTYLES[probe], zorder=2)
        # The vocabulary size at which the two classes stop being separate nodes.
        # It need not be one of the four measured sizes -- that is the point.
        if merged_at and min(levels) <= merged_at <= max(levels):
            ax.axvline(merged_at, color=colour[task], ls=":", lw=0.8, alpha=0.8, zorder=0)

    lo = min(p[1] for s in drawn.values() for p in s["points"])
    hi = max(p[1] for s in drawn.values() for p in s["points"])
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad * 2, hi + pad)
    for task in tasks:
        b = sorted({lv for p in style.PROBE_LINESTYLES for lv in drawn[(task, p)]["bounds"]})
        if b:
            ax.plot(b, [lo - pad * 1.5] * len(b), linestyle="none", marker="v",
                    color=colour[task], markerfacecolor="none",
                    markersize=plt.rcParams["lines.markersize"] * 1.5, zorder=4)
            ax.annotate("AUC saturated: off scale, a bound", xy=(b[0], lo - pad * 1.5),
                        xytext=(10, 3), textcoords="offset points", fontsize=6,
                        color=colour[task])

    ax.set_xscale("log")
    ax.set_xticks(levels, [str(lv) for lv in levels])
    ax.set_xlim(max(levels) * 1.15, min(levels) / 1.15)     # fine -> coarse, left to right
    ax.minorticks_off()
    ax.set_xlabel("classes in the pretraining label set  (fine $\\rightarrow$ coarse)")
    ax.set_ylabel("$\\log(1-\\mathrm{AUC})$   (lower is better)")
    ax.set_title("What a frozen probe can still read off, against pretraining vocabulary size\n"
                 "solid: linear probe.   dashed: nonlinear (MLP) probe.   "
                 "dotted vertical: where that distinction is merged away")
    # Below the axes: every corner inside them holds either a line or the
    # off-scale strip, and a legend over data is a legend that hides data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, borderaxespad=0)
    fig.tight_layout()
    return style.save(fig, "probe_ladder_granularity", outdir)


# ------------------------------------------------------------------ figure 2

def paired_rows(A: dict, task: str, probe: str, reference: int) -> list[dict]:
    """Per-seed differences against the reference vocabulary, with the stored interval.

    The analysis file stores each pair once, as coarser minus finer. Half of the
    pairs against the reference have the reference as the COARSER member, so
    their stored difference and interval are negated here to make every row read
    "this vocabulary minus the reference". The per-seed differences are
    recomputed from the same table rows and checked against the stored mean,
    which is what catches a sign flip rather than a comment claiming there
    isn't one.
    """
    out = []
    for r in A["pairwise_exploratory"][task][probe]:
        if not r["estimable"] or reference not in (r["fine"], r["coarse"]):
            continue
        other = r["coarse"] if r["fine"] == reference else r["fine"]
        flip = -1.0 if r["coarse"] == reference else 1.0
        ref = {c["seed"]: c["log1m_auc"] for c in cells(A, task, probe, reference)}
        d = [(c["log1m_auc"] - ref[c["seed"]], c["seed"])
             for c in cells(A, task, probe, other) if c["seed"] in ref]
        if abs(float(np.mean([x for x, _ in d])) - flip * r["mean_diff"]) > 1e-9:
            raise SystemExit(f"FATAL: per-seed differences for {task}/{probe}/{other} do not "
                             f"reproduce pairwise_exploratory's mean; the sign convention or "
                             f"the pairing is wrong")
        ci = sorted(flip * x for x in r["ci95"])
        out.append({"level": other, "diffs": [x for x, _ in d], "seeds": [s for _, s in d],
                    "mean": flip * r["mean_diff"], "ci95": ci, "is_bound": r["is_bound"],
                    "holm": r["holm_reject"]})
    return sorted(out, key=lambda r: -r["level"])


def fig2_paired(A: dict, reference: int, probe: str = "linear", outdir=None):
    """One panel per task: five paired differences, plus the interval over seeds."""
    tasks = plotted_tasks(A)
    ncol = int(np.ceil(len(tasks) / 2))
    # No shared x axis: the panels do not all have the same estimable level pairs
    # (a pair whose two cells both saturated has no spread and is dropped), so a
    # shared tick set would label one panel's points with another panel's levels.
    fig, axes = plt.subplots(2, ncol, figsize=(style.FIG_W_TWO_COLUMN,
                                               style.FIG_W_TWO_COLUMN * 0.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, task in zip(axes, tasks):
        rows = paired_rows(A, task, probe, reference)
        xs = np.arange(len(rows))
        ax.axhline(0, color="k", lw=0.8, zorder=1)
        seeds = sorted({s for r in rows for s in r["seeds"]})
        for s in seeds:                       # one polyline per seed: the pairing
            ys = [r["diffs"][r["seeds"].index(s)] if s in r["seeds"] else np.nan for r in rows]
            ax.plot(xs, ys, color="0.6", lw=0.6, marker="o", markersize=2.2, zorder=2)
        for x, r in zip(xs, rows):
            c = style.LEVEL_COLOURS.get(r["level"], "k")
            ax.errorbar([x], [r["mean"]],
                        yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]],
                        fmt=style.LEVEL_MARKERS.get(r["level"], "o"), color=c,
                        linestyle=style.PROBE_LINESTYLES[probe], capsize=3, zorder=3)
            if r["is_bound"]:
                ax.annotate("bound", xy=(x, r["mean"]), xytext=(6, 0),
                            textcoords="offset points", fontsize=6, color=c, va="center")
        ax.set_xticks(xs, [f"{r['level']}" for r in rows])
        ax.set_xlim(xs.min() - 0.4, xs.max() + 0.4)
        title = TASK_LABELS.get(task, task)
        if rows and all(r["is_bound"] for r in rows):
            title += "\n(reference saturated: every difference is a bound)"
        ax.set_title(title)
    for ax in axes[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[len(tasks) - ncol:len(tasks)]:
        ax.set_xlabel(f"pretraining label set, against the {reference}-class model")
    for ax in axes[:len(tasks):ncol]:
        ax.set_ylabel(f"$\\Delta\\log(1-\\mathrm{{AUC}})$ vs {reference}-class")
    n_seeds = len(A["seeds_used"])
    fig.suptitle(f"{n_seeds} paired differences per task, {style.PROBE_LABELS[probe]}: each grey "
                 f"line is one pretraining seed\npositive = worse than the {reference}-class "
                 f"model; bars are the 95% paired interval over seeds, from the pre-specified "
                 f"analysis", fontsize=plt.rcParams["axes.titlesize"])
    fig.tight_layout()
    return style.save(fig, f"probe_ladder_paired_{probe}", outdir)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", type=pathlib.Path, default=ANALYSIS)
    ap.add_argument("--rung-map", type=pathlib.Path, default=RUNG_MAP)
    ap.add_argument("--outdir", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)

    style.use_style()
    A = load(a.analysis)
    sizes = vocabulary_sizes(a.rung_map)
    names, collapsed = task_names(pathlib.Path(a.analysis).parent.parent)
    merges = merge_levels(a.rung_map, names, sizes)
    for task, m in merges.items():
        # The probe file recorded the same merge from its own reading of the tree.
        # Two answers to one question means one of them is wrong.
        first = next((r for r in RUNGS if r in collapsed.get(task, [])), None)
        if first != m["rung"]:
            raise SystemExit(f"FATAL: the label map merges {task} at {m['rung']} but the probe "
                             f"file recorded {first}; one of them read the tree wrong")

    written = fig1_granularity(A, merges, a.outdir)
    reference = A["levels_fine_to_coarse"][1]
    for probe in style.PROBE_LINESTYLES:
        written += fig2_paired(A, reference, probe, a.outdir)
    for p in written:
        print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
