#!/usr/bin/env python3
"""Fine-tuning below the full training set with a fixed step budget (plan item 3.1, 2026-10-31).

The launched cells at 1e3 and 1e4 jets ran 20 epochs of 19 steps (380 optimizer steps) on top
and quark/gluon tagging and 50 x 19 = 950 on JetClass, far short of convergence at a backbone
learning rate of 1e-4; every pretraining study we compare with trains small-data fine-tuning to
convergence. Below the full set every run now gets the same budget, pretrained or from scratch:

  * at most 20,000 steps at batch 512: 40 weaver epochs of 500 steps (--samples-per-epoch 256,000)
    at 1e5 jets and above, 400 epochs of 50 steps (25,600) below. The first sweep (500-step epochs,
    2026-11-01) had its best validation at the first check in 11 of 12 cells: 500 steps are 25
    passes over 1e4 jets, so the check came after the best and could not resolve it (PI: check
    every 50 steps below 1e5);
  * a validation pass after each epoch over the whole fixed validation file (200,000 jets), so
    every evaluation sees the same jets;
  * the best-validation checkpoint (weaver's rule), with training stopped once 10 consecutive
    evaluations bring no improvement (experiments/FT/early_stop.py);
  * the training subset s1 for every run, so a from-scratch seed varies only the training seed.

Pretrained runs: the checkpoint's best70 epoch (ft_v2.py resolve), the head re-initialised, the
backbone at --lr-pre with the head at 50x capped at 5e-3, constant rate, weight decay 0.01.
From scratch: --lr-scratch with weaver's flat+decay, weight decay 0.01 (ParT's top recipe). One
fine-tuning per pretraining run (seed 1); three training seeds from scratch.

Kinds of job (each one Indexed Job, any 24 GB+ GPU (build_mtx_launch.V2_FT_GPU), one cell per index):
  lrsweep   top and quark/gluon at 1e4 jets: 188-s1 at backbone {1e-4, 3e-4, 1e-3} and from
            scratch (seed 1) at {3e-4, 1e-3, 3e-3}; picked on the validation metric
  bench     top and quark/gluon at 1e3, 1e4, 1e5 for every paper run and from scratch
  jc1       JetClass at 1e4 and 1e5 (10^3 dropped, PI 2026-10-31)
  long      10^6 jets with four times the budget (80,000 steps, 160 epochs of 500; PI 2026-11-05, reversing
            the figure plan's decision 18: the from-scratch runs were still improving at the limit). Under
            /data/results/ft_v2/budget_long/. JetClass-II: from scratch and the three pretrained runs that
            reached the limit; the others stopped early under the same rule and stand. JetClass: every run,
            since all had the old fixed 10 epochs with no stopping rule, and one recipe per figure. From
            scratch keeps flat+decay, whose decay starts at 70%, so patience counts from there
  fillcheck top at the full set from 188-s1, zero fill and mean fill (experiments/FT/fill_config.py),
            ParT's recipe: does the fill move fine-tuning? (PI 2026-11-01: stays zero-fill if R30
            moves by less than the seed spread)

Output: /data/results/ft_v2/budget/<tree>/<init>/N<N>/s<S>/ (the sweep under .../lrsweep/),
with early_stop.json, ft_manifest.json, net_best_epoch_state.pt, net_last_epoch_state.pt and the
test outputs the read-outs take: features/logits.npy (top, quark/gluon, plus Herwig) or pred.root
(JetClass). Per-epoch states stay on the pod's local disk and are never written to /data.

    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.23 lrsweep
    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.24 --lr-pre 3e-4 --lr-scratch 1e-3 bench jc1
    python3 scripts/build_ft_budget_jobs.py --tag mtx-s2.60 --lr-pre 1e-3 --lr-scratch 1e-3 long-jc2b long-jc1
"""
import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import storage_line  # noqa: E402

OUT_DIR = ROOT / "experiments" / "FT" / "k8s" / "budget"
BUDGET_ROOT = "/data/results/ft_v2/budget"
FILLCHECK_ROOT = "/data/results/ft_v2/fillcheck"
LONG_ROOT = "/data/results/ft_v2/budget_long"
# from scratch with its whole flat+decay schedule (2026-11-07): the long cells' patience counted from the decay's
# start stopped them 10 of 48 decay epochs in, so their best checkpoints are all from the flat phase
LONG_FULL_ROOT = "/data/results/ft_v2/budget_long_fulldecay"
BUDGET_FULL_ROOT = "/data/results/ft_v2/budget_fulldecay"    # the same below 1e6: patience from epoch 0 stopped
# every from-scratch cell there before its decay began (2026-11-07)
# pretrained below 1e5 with the stopping rule from scratch now uses (PI 2026-11-08): every one of the 153 pretrained
# cells at 1e3 and 1e4 stopped by patience, so the 188- and 17-class runs are rerun to their last epoch, best validation
PRE_FULL_ROOT, PRE_FULL_RUNS = "/data/results/ft_v2/budget_prefull", ("l188", "r16q1")
PRE_FULL_SIZES = (("top", 1000), ("qg", 1000), ("top", 10000), ("qg", 10000), ("jc1", 10000), ("jc2b", 10000))
LONG_EPOCHS, LONG_N, LONG_JC2B = 160, 1_000_000, ("r16q1-s1", "r16q1mass-s1", "r16q1mass-s2")
SWEEP_ROOT = "/data/results/ft_v2/lrsweep_v2"     # the 500-step sweep stays in .../lrsweep (RUNS.csv)
MTX_V2 = "/data/results/mtx_v2"
PAPER_RUNS = ROOT / "configs" / "arms" / "v2_paper_runs.txt"
EPOCHS, STEPS, BATCH, PATIENCE, VAL_N = 40, 500, 512, 10, 200_000
SMALL_N, SMALL_STEPS = 100_000, 50       # below SMALL_N a check every 50 steps, same 20,000-step cap
HEAD_MULT, HEAD_CAP = 50.0, 5e-3
MEANFILL_TOP = "/workspace/meanfill/TopReference_meanfill.yaml"
SWEEP_PRE, SWEEP_SCRATCH, SWEEP_N = ("1e-4", "3e-4", "1e-3"), ("3e-4", "1e-3", "3e-3"), 10_000
SCRATCH_SEEDS = (1, 2, 3)
HERWIG_TEST = ["/data/finetune/qg_herwig/qg_herwig_chunk0.parquet",
               "/data/finetune/qg_herwig/qg_herwig_chunk1.parquet"]
JC1_CLASSES = ("HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q", "TTBar", "TTBarLep", "WToQQ", "ZToQQ",
               "ZJetsToNuNu")
# task -> (subset directory, data config, classes, network config, tree, sizes, test, memory)
TASKS = {
    "top": ("/data/finetune/top_sub", "configs/finetune/TopReference.yaml", 2,
            "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_000, 10_000, 100_000),
            "/data/finetune/top/top_test.parquet", "48Gi"),
    "topkin": ("/data/finetune/top_sub", "configs/finetune/TopReference_kin.yaml", 2,
               "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_000, 10_000, 100_000),
               "/data/finetune/top/top_test.parquet", "48Gi"),
    # top with the ten absent slots at the model's input-BatchNorm mean: the config is written in the
    # pod from the resolved checkpoint (experiments/FT/fill_config.py)
    "topmean": ("/data/finetune/top_sub", MEANFILL_TOP, 2,
                "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_top", (1_200_000,),
                "/data/finetune/top/top_test.parquet", "48Gi"),
    "qg": ("/data/finetune/qg_v2_sub", "configs/finetune/EnergyFlowQG.yaml", 2,
           "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_qg", (1_000, 10_000, 100_000),
           "/data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet", "48Gi"),
    "jc1": ("/data/finetune/jc1", "configs/finetune/JetClassI_sophon_noweight.yaml", 10,
            "experiments/E1/ParT_sophon_arch_10c.py", "leg2", (10_000, 100_000), "JC1_TEST", "64Gi"),
    # the class-balanced JetClass-II sets (stage_jc2_balanced.py): trained on the 188 native classes,
    # scored on every jet of the balanced test set through the native-label config
    "jc2b": ("/data/finetune/jc2_bal", "configs/finetune/JetClassII_L188_noweight.yaml", 188,
             "experiments/MTX/ParT_sophon_arch_mtx.py", "leg_jc2b", (10_000, 100_000, 1_000_000),
             "/data/finetune/jc2_bal/test.parquet", "64Gi"),
}
VAL_SIZE = {"jc2b": 20_480}        # the balanced validation file (188 x 109 at most); 200,000 elsewhere
TEST_CFG = {"jc2b": "configs/data/JetClassII_base.yaml"}
FULL = {"topkin": 1_200_000, "top": 1_200_000, "topmean": 1_200_000}   # the full training set: ParT's
# recipe, 20 epochs, no early stop (top's full set only in the fill check; bench stops at 1e5)
KIN_RUNS = ("l188kin-s1",)          # the 17-class kinematic-only run was stopped for good (2026-11-01)
FULL_MEM = "88Gi"
# One rule for every fine-tuning cell below the full top and quark/gluon sets (PI 2026-11-15): a fixed number of steps
# set only by the training size, validation on the same 200,000 jets every FIXED_CHECK steps (held in memory, one pass
# so every check scores every jet), the best validation checkpoint over the whole schedule,
# no patience. From scratch differs only in its learning rates and schedule (ParT's flat+decay, head x1).
# The image's weaver (0.4.17) holds data in memory only for training and validation together (--in-memory) and validates
# in the training batch: so the validation set is 199,680 jets (390 x 512, the largest multiple of the batch up to
# 200,000), val_fixed.parquet in every subset directory (stage_val_fixed.py), and each check scores all of it.
FIXED_ROOT = "/data/results/ft_v2/fixed"
# 1e4 and 1e3 set from the sweep's and the smoke's validation curves (PI 2026-11-16): at 1e4 every best check by 1,200
# steps (at 1e-2 the third 50-step check, so checks every 25); at 1e3 the smoke's best by 3,700 steps, the length kept
# at 20,000 (two top cells, quark/gluon untested)
FIXED_STEPS = {1_000: 20_000, 10_000: 3_000, 100_000: 20_000, 1_000_000: 80_000}
FIXED_VAL, FIXED_NO_PATIENCE = 390 * 512, 10 ** 6
# the check interval, like the steps, set only by the training size (PI 2026-11-15): every 50 steps below 1e6, every 500
# at 1e6, where every best checkpoint observed sits at 3,500 steps or later
FIXED_CHECK = {1_000: 50, 10_000: 25, 100_000: 50, 1_000_000: 500}
FIXED_VAL_FILE = {}               # every task: <subset dir>/val_fixed.parquet
FIXED_SIZES = {"top": (1_000, 10_000, 100_000), "topkin": (1_000, 10_000, 100_000), "qg": (1_000, 10_000, 100_000),
               "jc1": (10_000, 100_000, 1_000_000), "jc2b": (10_000, 100_000, 1_000_000)}
# the rates of the one-rule sweep (fixed_sweep, extended at the edge; 2026-11-15, best validation accuracy at 1e4):
# pretrained 1e-2 (top 0.9249, q/g 0.8260; 3e-3 and 3e-2 below on both), from scratch top 3e-3 and q/g 1e-3;
# from scratch 1e-3 (ParT's recipe) on JetClass and JetClass-II, which the sweep does not cover
FIXED_LR_PRE, FIXED_LR_SCRATCH = "1e-2", {"top": "3e-3", "qg": "1e-3"}
FIXED_JC2B_SMALL_RUNS = ("l188", "l162", "r42q1", "r16q1", "l162mass", "r16q1mass")   # JetClass-II below 1e6


def head_mult(lr: str) -> str:
    """The head's multiplier: 50x the backbone rate, capped at 5e-3 for the head."""
    return repr(min(HEAD_MULT, HEAD_CAP / float(lr)))


def paper_runs() -> list:
    return [l.strip()[len("mtx-"):] for l in PAPER_RUNS.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def cell(task: str, init: str, n: int, s: int, lr: str, root: str, sub: str = "") -> dict:
    """One cell. init '' is from scratch; sub names the cell's directory under its init. Below the
    full set: 20,000 steps in epochs of 500 steps (50 below 1e5 jets), early stop after 10 epochs
    without improvement; at the full set (FULL): ParT's 20 epochs of the whole set, kept to the end."""
    tree = TASKS[task][4]
    name = init or "scratch"
    leaf = f"{name}{sub}"
    full = FULL.get(task) == n
    steps = SMALL_STEPS if n < SMALL_N else STEPS
    return {"task": task, "init": init, "run_dir": f"{MTX_V2}/mtx-{init}" if init else "", "n": n, "s": s,
            "lr": lr, "mult": head_mult(lr) if init else "", "out": f"{root}/{tree}/{leaf}/N{n}/s{s}",
            "epochs": 20 if full else EPOCHS * STEPS // steps, "spe": n if full else steps * BATCH,
            "patience": 1000 if full else PATIENCE, "val": VAL_SIZE.get(task, VAL_N)}


def fixed_cell(task: str, init: str, n: int, s: int, lr: str, root: str = FIXED_ROOT) -> dict:
    """A cell of the one rule (FIXED_*): FIXED_STEPS[n] steps in checks of FIXED_CHECK[n] steps, no patience."""
    c = cell(task, init, n, s, lr, root)
    return {**c, "epochs": FIXED_STEPS[n] // FIXED_CHECK[n], "spe": FIXED_CHECK[n] * BATCH, "patience": FIXED_NO_PATIENCE,
            "val": FIXED_VAL, "valf": FIXED_VAL_FILE.get(task, ""), "valmem": 1}


def fixed_cells(sizes: set) -> list:
    """Every cell of the one rule at the given training sizes: each paper run (the kinematic-only run on top), and the
    three from-scratch trainings; JetClass-II below 1e6 for the label sets and mass networks only, as before."""
    out = []
    for task, task_sizes in FIXED_SIZES.items():
        for n in (m for m in task_sizes if m in sizes):
            runs = KIN_RUNS if task == "topkin" else paper_runs()
            if task == "jc2b" and n < LONG_N:
                runs = [r for r in runs if r.rsplit("-s", 1)[0] in FIXED_JC2B_SMALL_RUNS]
            out += [fixed_cell(task, r, n, 1, FIXED_LR_PRE) for r in runs]
            if task != "topkin":
                out += [fixed_cell(task, "", n, s, FIXED_LR_SCRATCH.get(task, "1e-3")) for s in SCRATCH_SEEDS]
    return out


FIXED_SWEEP_ROOT = "/data/results/ft_v2/fixed_sweep"
# the rate sweep redone under the one rule, one rate past each edge of the old sweep (PI 2026-11-15: both old choices,
# pretrained 1e-3 and quark/gluon from scratch 3e-3, sat at its edge); the same cells: top and quark/gluon at 1e4,
# l188-s1 and from-scratch seed 1. Chosen on the best validation accuracy; a choice at an edge extends the grid again.
FIXED_SWEEP_PRE, FIXED_SWEEP_SCRATCH, FIXED_SWEEP_N = ("1e-4", "3e-4", "1e-3", "3e-3"), ("3e-4", "1e-3", "3e-3", "1e-2"), 10_000
# 2026-11-15: top and q/g pretrained peaked at 3e-3, the grid edge, so the pretrained grid is extended (both rates at once)
FIXED_SWEEP_PRE_EXT = ("1e-2", "3e-2")


def fixed_sweep_cells(pre=FIXED_SWEEP_PRE, scratch=FIXED_SWEEP_SCRATCH, init="l188-s1", n=FIXED_SWEEP_N,
                      tasks=("top", "qg")) -> list:
    out = []
    for task in tasks:
        out += [fixed_cell(task, init, n, 1, lr, FIXED_SWEEP_ROOT) for lr in pre]
        out += [fixed_cell(task, "", n, 1, lr, FIXED_SWEEP_ROOT) for lr in scratch]
    for c in out:     # one directory per rate, as the old sweep (cell's sub)
        c["out"] = c["out"].replace(f"/N{n}/", f"_lr{c['lr']}/N{n}/")
    return out


# PI 2026-11-15: the pretrained rate checked at the other end of the label-set range (17 classes, seed 1), and the
# 10^3 length set from a smoke of 188 and 17 classes on top at the chosen rate on the long schedule
FIXED_SWEEP_17 = ("3e-3", "1e-2", "3e-2")


# PI 2026-11-20 (D2): JetClass-II at 1e6 under the one rule had its best check first in all three 188-class runs and a
# training accuracy that never rose above 0.39, below the linear probe's 0.4348; its rate came from the top and
# quark/gluon sweep at 1e4. The rate is swept on the row itself, 188 and 17 classes, seed 1 (1e-2 exists in FIXED_ROOT)
FIXED_SWEEP_JC2B = ("1e-4", "3e-4", "1e-3", "3e-3")


# PI 2026-11-20 ("sweep every row, so the methods sentence is true for all of them"): the rate is chosen per task and
# size, searched from the current rate in steps of about 3 until the best has a lower-scoring rate on each side.
# Pretrained (188 and 17 classes, seed 1): the rows that pass both checks run 3e-3 and 3e-2 beside the existing 1e-2
# (top and quark/gluon at 1e4 already have them in FIXED_SWEEP_ROOT); the failing rows are fixed-sweep-jc2b and -rows.
# From scratch (seed 1), every row, one step either side of its current rate (top 3e-3, elsewhere 1e-3); top and
# quark/gluon at 1e4 have them, quark/gluon at 1e3 is in -rows.
FIXED_PASSING_ROWS = (("top", 1_000), ("top", 100_000), ("qg", 100_000), ("jc1", 10_000), ("jc1", 100_000),
                      ("jc1", 1_000_000))
FIXED_SCRATCH_AROUND = {"top": ("1e-3", "1e-2"), "qg": ("3e-4", "3e-3"), "jc1": ("3e-4", "3e-3"), "jc2b": ("3e-4", "3e-3")}
FIXED_SCRATCH_ROWS = (("top", 1_000), ("top", 100_000), ("qg", 100_000), ("jc1", 10_000), ("jc1", 100_000),
                      ("jc1", 1_000_000), ("jc2b", 10_000), ("jc2b", 100_000), ("jc2b", 1_000_000))


# THE SINGLE SOURCE OF EACH ROW'S RATE (PI 2026-11-21): the rate every searched row settled on, pretrained and from
# scratch, in the order the rows settled (fixed_rate_cells' job indices follow it, so a new row is appended, never
# inserted). The builder (fixed_rate_cells), the read-out (paper_fixed_cells, experiments/FT/budget_metrics.py
# --expect-fixed) and Table A15 (experiments/FT/rate_search.py) read it; tests/test_rate_search.py fails when a row is
# read at any other rate or mixes rates. A row at its original rate (FIXED_LR_PRE, FIXED_LR_SCRATCH) keeps its
# FIXED_ROOT cells; a row at another rate is rerun whole into FIXED_RATE_ROOT. A row absent here has not settled.
FIXED_RATE_ROOT = "/data/results/ft_v2/fixed_rate"
FIXED_ROW_RATE = {("jc2b", 10_000): "1e-3",       # 2026-11-21: mean 0.3665 at 1e-3 (3e-3 0.3654, 1e-2 0.3594)
                  ("qg", 1_000): "3e-4",          # 2026-11-21: mean 0.7903 at 3e-4 (1e-4 0.7895, 1e-3 0.7874)
                  ("top", 100_000): "1e-2"}       # 2026-11-21: 0.9332 at 1e-2 (3e-3 0.9327, 3e-2 diverged); no rerun
# from scratch, seeds SCRATCH_SEEDS, where its search moved it: qg 1e3 best 1e-4 0.7826 (3e-5 0.7817, 3e-4 0.7705)
FIXED_ROW_RATE_SCRATCH = {("qg", 1_000): "1e-4"}
# PI 2026-11-21: the search goes no lower than 1e-5 (below it fine-tuning approaches training only the new output layer
# on frozen features); a row whose best is still its lowest rate there takes 1e-5
FIXED_SEARCH_FLOOR = "1e-5"
# the searched rows: every one-rule (task, size); the networks whose validation accuracy is averaged, pretrained (seed 1
# of 188 and of 17 classes) and from scratch (seed 1)
FIXED_SEARCH_ROWS = tuple((t, n) for t, sizes in FIXED_SIZES.items() if t != "topkin" for n in sizes)
FIXED_SEARCH_INITS = {"pre": ("l188-s1", "r16q1-s1"), "scratch": ("",)}


def original_rate(task: str, kind: str) -> str:
    return FIXED_LR_PRE if kind == "pre" else FIXED_LR_SCRATCH.get(task, "1e-3")


def row_rate(task: str, n: int, kind: str):
    """The recorded rate of a row (FIXED_ROW_RATE, FIXED_ROW_RATE_SCRATCH), or None while it has not settled. The
    kinematic-only network fine-tuned on top is a pretrained network of top's row."""
    return (FIXED_ROW_RATE if kind == "pre" else FIXED_ROW_RATE_SCRATCH).get(("top" if task == "topkin" else task, n))


def paper_fixed_cells() -> list:
    """Every one-rule cell the paper reads, at its row's recorded rate: the FIXED_ROOT cell where the row keeps its
    original rate (or has not settled: the preview), the FIXED_RATE_ROOT rerun where it moved."""
    out = []
    for c in fixed_cells(set(FIXED_STEPS)):
        kind = "pre" if c["init"] else "scratch"
        lr = row_rate(c["task"], c["n"], kind)
        out.append(c if lr in (None, c["lr"]) else fixed_cell(c["task"], c["init"], c["n"], c["s"], lr, FIXED_RATE_ROOT))
    return out


# searches whose best rate landed on an edge, extended outward (PI 2026-11-20: "extend outward only when the best rate
# is at an edge"): (task, n, init or "" for from scratch, rates)
FIXED_EDGE_EXT = (("qg", 1_000, "", ("3e-5", "1e-5")),)     # 2026-11-21: from scratch best at 1e-4, the lower edge


def fixed_sweep_edge_cells() -> list:
    out = []
    for task, n, init, rates in FIXED_EDGE_EXT:
        if init:
            out += [c for r in ("l188-s1", "r16q1-s1") for c in fixed_sweep_cells(rates, (), r, n, (task,))]
        else:
            out += fixed_sweep_cells((), rates, "", n, (task,))
    return out


# 2026-11-21, second round (init "pre": 188 and 17 classes seed 1; "": from scratch seed 1), by memory and pool:
# A100 first (JetClass-II 1e6), then 48Gi, then 24Gi. Edges: JetClass-II 1e6 and 1e5 pretrained best at 1e-4 (lower);
# JetClass 1e6 and JetClass-II 1e4 from scratch best at 3e-3 (upper). Ahead of the row's last cell, 1e-3 where 3e-3
# already beats 1e-2 and only 3e-2 is out (top 1e3, JetClass 1e6 pretrained), to save a round
FIXED_EDGE_EXT2 = (("jc2b", 1_000_000, "pre", ("3e-5", "1e-5")),
                   ("jc2b", 100_000, "pre", ("3e-5", "1e-5")), ("jc1", 1_000_000, "", ("1e-2", "3e-2")),
                   ("jc2b", 10_000, "", ("1e-2", "3e-2")), ("jc1", 1_000_000, "pre", ("1e-3",)),
                   ("top", 1_000, "pre", ("1e-3",)))


def fixed_sweep_edge2_cells() -> list:
    out = []
    for task, n, init, rates in FIXED_EDGE_EXT2:
        if init:
            out += [c for r in ("l188-s1", "r16q1-s1") for c in fixed_sweep_cells(rates, (), r, n, (task,))]
        else:
            out += fixed_sweep_cells((), rates, "", n, (task,))
    return out


def fixed_rate_cells() -> list:
    """Every pretrained cell of each row in FIXED_ROW_RATE whose rate is not the original, at that row's rate, into
    FIXED_RATE_ROOT; then the from-scratch seeds of each such row of FIXED_ROW_RATE_SCRATCH. Job indices follow the
    order the rows settled (0-15 JetClass-II 1e4, 16-42 quark/gluon 1e3 pretrained, 43-45 its from-scratch seeds)."""
    pre = [fixed_cell(task, c["init"], n, c["s"], lr, FIXED_RATE_ROOT) for (task, n), lr in FIXED_ROW_RATE.items()
           if lr != FIXED_LR_PRE for c in fixed_cells({n}) if c["task"] in (task, f"{task}kin") and c["init"]]
    return pre + [fixed_cell(task, "", n, s, lr, FIXED_RATE_ROOT) for (task, n), lr in FIXED_ROW_RATE_SCRATCH.items()
                  if lr != original_rate(task, "scratch") for s in SCRATCH_SEEDS]


# PI 2026-11-21: the top and quark/gluon 1e4 neighbours of the old sweep ran 20,000 steps checked every 50 (the 1e4 rule
# became 3,000 in checks of 25 on 2026-11-16), so they are rerun on the 1e4 schedule, seed 1, into their own tree; the
# old ones (FIXED_SWEEP_ROOT .../N10000, and the 40,000-step top 1e3 smoke cells) carry a SUPERSEDED file and are not read
FIXED_SWEEP_1E4_ROOT = "/data/results/ft_v2/fixed_sweep_1e4"


def fixed_sweep_1e4_cells() -> list:
    out = []
    for task in ("top", "qg"):
        for r in FIXED_SEARCH_INITS["pre"]:
            out += fixed_sweep_cells(("3e-3", "3e-2"), (), r, 10_000, (task,))
        out += fixed_sweep_cells((), FIXED_SCRATCH_AROUND[task], "", 10_000, (task,))
    for c in out:
        c["out"] = c["out"].replace(FIXED_SWEEP_ROOT + "/", FIXED_SWEEP_1E4_ROOT + "/")
    return out


def fixed_smoke_diverge_cells() -> list:
    """The smoke of the one divergence rule (2026-11-21): top 1e3 from 188 classes at a backbone rate of 10, which must
    stop at its first non-finite training loss, write DIVERGED and fail its index without a retry; and a normal top 1e4
    cell cut to 5 checks, the unchanged path. Into ft_v2/fixed_smoke; not results."""
    root = "/data/results/ft_v2/fixed_smoke"
    bad = fixed_cell("top", "l188-s1", 1_000, 1, "10", root)
    ok = {**fixed_cell("top", "l188-s1", 10_000, 1, FIXED_LR_PRE, root), "epochs": 5}
    return [bad, ok]


def fixed_search_plan() -> dict:
    """{(task, n, kind): {rate: [cell directory of each network averaged]}} of every searched row: the FIXED_ROOT cell
    at the row's original rate, and the cells of every search kind. The superseded cells are not in it."""
    plan = {}
    for c in fixed_cells(set(FIXED_STEPS)):
        kind = "pre" if c["init"] else "scratch"
        if c["task"] != "topkin" and c["s"] == 1 and c["init"] in FIXED_SEARCH_INITS[kind]:
            plan.setdefault((c["task"], c["n"], kind), {}).setdefault(c["lr"], []).append(c["out"])
    for c in (fixed_sweep_rows_cells() + fixed_sweep_jc2b_cells() + fixed_sweep_all_cells() + fixed_sweep_edge_cells()
              + fixed_sweep_edge2_cells() + fixed_sweep_1e4_cells()):
        kind = "pre" if c["init"] else "scratch"
        cells = plan.setdefault((c["task"], c["n"], kind), {}).setdefault(c["lr"], [])
        if c["out"] not in cells:
            cells.append(c["out"])
    return {row: dict(sorted(rates.items(), key=lambda x: float(x[0]))) for row, rates in sorted(plan.items())}


def fixed_sweep_all_cells() -> list:
    """The pretrained cells of the passing rows, then the from-scratch cells of every row not yet searched."""
    out = []
    for task, n in FIXED_PASSING_ROWS:
        for r in ("l188-s1", "r16q1-s1"):
            out += fixed_sweep_cells(("3e-3", "3e-2"), (), r, n, (task,))
    for task, n in FIXED_SCRATCH_ROWS:
        out += fixed_sweep_cells((), FIXED_SCRATCH_AROUND[task], "", n, (task,))
    return out


def fixed_head_jc2b_cells() -> list:
    """PI 2026-11-20: the head's rate tied to the backbone's (multiplier 1), 188 classes at 1e-3, JetClass-II 1e6; the
    sweep never varies the head (50x the backbone capped at 5e-3 is 5e-3 at every rate of the grid)."""
    c = fixed_sweep_cells(("1e-3",), (), "l188-s1", 1_000_000, ("jc2b",))[0]
    return [{**c, "mult": "1.0", "out": c["out"].replace("_lr1e-3/", "_lr1e-3_head1x/")}]


def fixed_sweep_jc2b_cells() -> list:
    return [c for r in ("l188-s1", "r16q1-s1")
            for c in fixed_sweep_cells(FIXED_SWEEP_JC2B, (), r, 1_000_000, ("jc2b",))]


# PI 2026-11-20 (D2, "those rows get the same sweep"): every other row where a fine-tuned network ends below its own
# linear probe (JetClass-II at 1e4 and 1e5) or has its best check first (quark/gluon at 1e3, 18 of 29 cells, two of
# the three from-scratch seeds among them, so from scratch is swept there too around its 1e-3)
FIXED_SWEEP_ROWS = (("jc2b", 10_000), ("jc2b", 100_000), ("qg", 1_000))
FIXED_SWEEP_SCRATCH_QG1E3 = ("1e-4", "3e-4", "3e-3")


def fixed_sweep_rows_cells() -> list:
    out = []
    for task, n in FIXED_SWEEP_ROWS:
        scratch = FIXED_SWEEP_SCRATCH_QG1E3 if (task, n) == ("qg", 1_000) else ()
        out += fixed_sweep_cells(FIXED_SWEEP_JC2B, scratch, "l188-s1", n, (task,))
        out += fixed_sweep_cells(FIXED_SWEEP_JC2B, (), "r16q1-s1", n, (task,))
    return out


def fixed_smoke_1e3_cells() -> list:
    return [c for r in ("l188-s1", "r16q1-s1") for c in fixed_sweep_cells((FIXED_LR_PRE,), (), r, 1_000, ("top",))]


def fixed_bench_cells() -> list:
    """Validation-cost benchmark (PI 2026-11-15): top 1e4 from 188-s1 for 10 checks, validation in memory and from the
    file loader, into ft_v2/fixed_bench; not results."""
    root = "/data/results/ft_v2/fixed_bench"
    mem = {**fixed_cell("top", "l188-s1", 10_000, 1, FIXED_LR_PRE, root), "epochs": 10}
    disk = {**mem, "valmem": 0, "out": mem["out"].replace("/l188-s1/", "/l188-s1_disk/")}
    big = {**fixed_cell("jc2b", "l188-s1", 1_000_000, 1, FIXED_LR_PRE, root), "epochs": 2}    # memory of a 1e6 cell
    return [mem, disk, big]


def lrsweep_cells() -> list:
    out = []
    for task in ("top", "qg"):
        out += [cell(task, "l188-s1", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_PRE]
        out += [cell(task, "", SWEEP_N, 1, lr, SWEEP_ROOT, f"_lr{lr}") for lr in SWEEP_SCRATCH]
    return out


def bench_cells(lr_pre: str, lr_scratch: str, tasks=("top", "qg"), runs=None, lr_scratch_task=None) -> list:
    """lr_scratch_task: {task: rate} where a task's own sweep chose another from-scratch rate."""
    runs = paper_runs() if runs is None else runs
    out = []
    for task in tasks:
        sizes = TASKS[task][5]
        lr_s = (lr_scratch_task or {}).get(task, lr_scratch)
        out += [cell(task, r, n, 1, lr_pre, BUDGET_ROOT) for n in sizes for r in runs]
        out += [cell(task, "", n, s, lr_s, BUDGET_ROOT) for n in sizes for s in SCRATCH_SEEDS]
    return out


def jc2b_cells(lr_pre: str, lr_scratch: str) -> list:
    """The hierarchy and the mass-output runs (the Methods' JetClass-II scope) and from scratch."""
    runs = [r for r in paper_runs() if r.split("-")[0] in ("l188", "l162", "r42q1", "r16q1", "l162mass", "r16q1mass")]
    return bench_cells(lr_pre, lr_scratch, ("jc2b",), runs)


def long_cells(task: str, lr_pre: str, lr_scratch: str) -> list:
    inits = LONG_JC2B if task == "jc2b" else paper_runs()
    out = [{**cell(task, "", LONG_N, s, lr_scratch, LONG_ROOT), "patience_from": LONG_EPOCHS - max(1, int(LONG_EPOCHS * 0.3))}
           for s in SCRATCH_SEEDS]      # weaver's flat+decay: the last 30% of the epochs decay
    out += [cell(task, r, LONG_N, 1, lr_pre, LONG_ROOT) for r in inits]
    return [{**c, "epochs": LONG_EPOCHS} for c in out]


def long_scratch_full_cells(task: str, lr_scratch: str) -> list:
    """From scratch at 1e6 with the long budget run to its last epoch, ParT's recipe in full: no early stop, the
    best validation epoch over all 160 (patience counted from the end)."""
    return [{**cell(task, "", LONG_N, s, lr_scratch, LONG_FULL_ROOT), "epochs": LONG_EPOCHS, "patience_from": LONG_EPOCHS}
            for s in SCRATCH_SEEDS]


def scratch_full_cells(lr_scratch: str, lr_scratch_qg: str) -> list:
    """From scratch below 1e6 (top and quark/gluon 1e3-1e5, JetClass and JetClass-II 1e4-1e5), each task's swept
    rate, the whole flat+decay schedule of its budget (400 epochs of 50 steps or 40 of 500), the best validation
    epoch over all."""
    out = []
    for task in ("top", "qg", "jc1", "jc2b"):
        lr = lr_scratch_qg if task == "qg" else lr_scratch
        sizes = [n for n in TASKS[task][5] if n < LONG_N]
        out += [{**(c := cell(task, "", n, s, lr, BUDGET_FULL_ROOT)), "patience_from": c["epochs"]}
                for n in sizes for s in SCRATCH_SEEDS]
    return out


def pre_full_cells(lr_pre: str) -> list:
    """The 188- and 17-class pretrained cells (three pretraining runs each) at every task and size below 1e5, their
    own recipe run to the last epoch with the best validation epoch over all, as from scratch: 36 cells, top at 1e3
    first (the size where the full schedule moved from scratch)."""
    return [{**(c := cell(task, f"{r}-s{k}", n, 1, lr_pre, PRE_FULL_ROOT)), "patience_from": c["epochs"]}
            for task, n in PRE_FULL_SIZES for r in PRE_FULL_RUNS for k in (1, 2, 3)]


# the pretrained cells of each (task, size) of PRE_FULL_SIZES as launched: the bench (top, quark/gluon), jc1, jc2b and
# kinematic-only kinds; topkin is top tagging with the kinematic-only run
PRE_FULL_TASK = {"top": ("top", "topkin"), "qg": ("qg",), "jc1": ("jc1",), "jc2b": ("jc2b",)}


def pre_full_rerun_cells(lr_pre: str, sizes) -> list:
    """The equal-stopping rerun (PI 2026-11-08, operationalised 2026-11-09): at each (task, size) where the 188- or the
    17-class cells moved, every other pretrained cell there (label sets, random partitions, flavor boundaries, mass
    models, kinematic-only), each the launched cell with only its tree and patience_from changed, as pre_full_cells."""
    sizes = [tuple(x) for x in sizes]
    if not sizes or not set(sizes) <= set(PRE_FULL_SIZES):
        raise SystemExit(f"FATAL: rerun sizes {sizes}: each one of {PRE_FULL_SIZES}")
    launched = (bench_cells(lr_pre, lr_pre, ("top", "qg", "jc1")) + jc2b_cells(lr_pre, lr_pre) + kin_cells(lr_pre))
    done = {(c["task"], c["init"], c["n"]) for c in pre_full_cells(lr_pre)}
    out = []
    for task, n in sizes:
        for o in launched:
            if o["init"] and o["task"] in PRE_FULL_TASK[task] and o["n"] == n and (o["task"], o["init"], n) not in done:
                c = cell(o["task"], o["init"], n, o["s"], o["lr"], PRE_FULL_ROOT)
                out.append({**c, "patience_from": c["epochs"]})
    return out


def readout_spec(name: str, cells: list, tag: str, line: int, out: str) -> str:
    """A CPU read-out (experiments/FT/budget_metrics.py) of PRE_FULL_ROOT, as job-budget-readout-e: it refuses to start
    unless every cell has DONE, refuses a read-out short of a cell, and never overwrites one already written."""
    trees = sorted({TASKS[c["task"]][4] for c in cells})
    dirs = " ".join(c["out"] for c in cells)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_ft_budget_jobs.py -- read-out of the {len(cells)} equal-stopping cells (PI 2026-11-08/09). CPU only.
  name: {name}
  namespace: cms-ml
spec:
  backoffLimit: 1
  activeDeadlineSeconds: 21600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          for T in 1 2 3 4 5; do git clone --depth 1 --branch "{tag}" https://github.com/raunavm/transferlearningsophon.git /workspace/repo && break; sleep 30; done
          cd /workspace/repo
          USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
          [ "${{USED}}" -le {line} ] || {{ echo "FATAL: /data is ${{USED}}% full"; exit 42; }}
          export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
          for c in {dirs}; do
            [ -f "${{c}}/DONE" ] || {{ echo "FATAL: ${{c}} not DONE"; exit 42; }}
          done
          OUT={out}
          if [ -e ${{OUT}} ]; then echo "skip ${{OUT}}: exists"; exit 0; fi
          python3 experiments/FT/budget_metrics.py --root {PRE_FULL_ROOT} --trees {" ".join(trees)} --out ${{OUT}}.new
          python3 -c "import json,sys;d=json.load(open(sys.argv[1]));n=len(d['cells']);sys.exit(0 if n=={len(cells)} and not d['missing'] else 'FATAL: %d cells, missing %s'%(n,d['missing']))" ${{OUT}}.new
          mv ${{OUT}}.new ${{OUT}}
          echo "READOUT COMPLETE"
        resources:
          requests: {{ memory: "24Gi", cpu: "4", ephemeral-storage: "5Gi" }}
          limits:   {{ memory: "24Gi", cpu: "4", ephemeral-storage: "5Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
"""


PRE_FULL_METRICS = f"{PRE_FULL_ROOT}/budget_prefull_metrics.json"


def long_rest_cells(lr_pre: str) -> list:
    """PI 2026-11-14 (compared networks trained identically): every pretrained run of the paper not yet fine-tuned on
    JetClass-II at 1e6 with the long budget (the label sets, both 162 + mass runs, 17 classes s2 and s3), so all 27 use
    160 epochs with the best validation epoch; their fixed-budget cells under budget/ are superseded, not deleted."""
    done = set(LONG_JC2B) | {r for r in paper_runs() if r.startswith(("rand", "flav"))}
    return [{**cell("jc2b", r, LONG_N, 1, lr_pre, LONG_ROOT), "epochs": LONG_EPOCHS} for r in paper_runs() if r not in done]


def long_ctrl_cells(lr_pre: str) -> list:
    """The five random partitions and the three controlled ones on JetClass-II at 1e6 with the long
    budget, so the JetClass-II fine-tuning figures carry the controls (PI, 2026-11-05)."""
    runs = [r for r in paper_runs() if r.startswith(("rand", "flav"))]
    return [{**cell("jc2b", r, LONG_N, 1, lr_pre, LONG_ROOT), "epochs": LONG_EPOCHS} for r in runs]


def kin_cells(lr_pre: str) -> list:
    """The kinematic-only runs on top tagging: below the full set at the swept rate, and the full set
    with ParT's fine-tuning recipe (1e-4, head x50)."""
    out = [cell("topkin", r, n, 1, lr_pre, BUDGET_ROOT) for n in TASKS["topkin"][5] for r in KIN_RUNS]
    out += [cell("topkin", r, FULL["topkin"], 1, "1e-4", BUDGET_ROOT) for r in KIN_RUNS]
    return out


def fillcheck_cells() -> list:
    """Top at the full set from 188-s1 with ParT's recipe (1e-4, head x50), zero and mean fill."""
    return [cell(task, "l188-s1", 1_200_000, 1, "1e-4", FILLCHECK_ROOT, sub)
            for task, sub in (("top", "_zero"), ("topmean", "_mean"))]


def _launch():
    spec = importlib.util.spec_from_file_location("build_mtx_launch", ROOT / "scripts" / "build_mtx_launch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CELL_SCRIPT = r"""
HALT=42
IDX=${JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}
case ${IDX} in
__CASES__  *) echo "FATAL: no cell for index ${IDX}"; exit ${HALT};;
esac
[ -f ${CELL}/DONE ] && { echo "skip ${CELL} (DONE)"; exit 0; }
USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)
[ "${USED}" -le __LINE__ ] || { echo "FATAL: /data at ${USED}%"; exit ${HALT}; }
case ${TASK} in
__TASKS__  *) echo "FATAL: task ${TASK}"; exit ${HALT};;
esac
python3 experiments/FT/gpu_probe.py || exit 43
RUN=/workspace/run; rm -rf ${RUN}; mkdir -p ${RUN}
if [ "${VALMEM}" = 1 ]; then
  # the same VAL jets at every check: in memory, one whole pass of VAL / batch steps; weaver's per-epoch files thinned
  VALF=${VALF:-${SUB}/val_fixed.parquet}
  TRAINARGS=(--in-memory --steps-per-epoch $((SPE / __BATCH__))); VALARGS=(--steps-per-epoch-val $((VAL / __BATCH__)))
  PRUNE=(--prune ${RUN}/net); VALRULE=fixed-set-in-memory
else
  VALF=${VALF:-${SUB}/val.parquet}
  TRAINARGS=(--samples-per-epoch ${SPE}); VALARGS=(--samples-per-epoch-val ${VAL}); PRUNE=(); VALRULE=fixed-file
fi
[ ! -f ${CELL}/DIVERGED ] || { echo "${CELL} diverged: not retried"; exit ${HALT}; }
[ ! -f ${CELL}/DONE ] || { echo "${CELL} is done"; exit 0; }
[ -d ${CELL} ] && mv ${CELL} ${CELL}.partial.$(date -u +%s)
mkdir -p ${CELL}
echo "$(date -u +%FT%TZ) pod=${POD_NAME} node=${NODE_NAME} ref=${REPO_REF}" >> ${CELL}/ATTEMPTS
if [ -n "${RUN_DIR}" ]; then
  python3 experiments/FT/ft_v2.py resolve --run-dir ${RUN_DIR} --rule best70 --link /workspace/ckpt/${INIT}.pt || exit ${HALT}
  cp /workspace/ckpt/${INIT}.pt.json ${CELL}/init_checkpoint.json
  if [ "${TASK}" = topmean ]; then
    python3 experiments/FT/fill_config.py --checkpoint /workspace/ckpt/${INIT}.pt \
      --src configs/finetune/TopReference.yaml --out ${CFG} || exit ${HALT}
    cp ${CFG} ${CELL}/
  fi
  LOAD=(--load-model-weights /workspace/ckpt/${INIT}.pt --exclude-model-weights "mod\.fc\..*")
  MULTARG=(--optimizer-option lr_mult "(r'mod\.fc\..*', ${MULT})")
  SCHED=(--lr-scheduler none); RECIPE="head_lr_mult=${MULT} lr_schedule=constant ckpt_rule=best70"
else
  LOAD=(); MULTARG=(); SCHED=(); RECIPE="head_lr_mult=1 lr_schedule=flat+decay recipe=part-from-scratch"
fi
python3 experiments/FT/smoke_checks.py manifest --out ${CELL}/ft_manifest.json task=${TASK} \
  init=${INIT:-scratch} run_dir=${RUN_DIR} checkpoint=${RUN_DIR:+/workspace/ckpt/${INIT}.pt} n_train=${N} ft_seed=${S} lr=${LR} ${RECIPE} weight_decay=0.01 \
  subset=${SUB}/train_N${N}_s1.parquet data_config=${CFG} network=${NET} num_classes=${K} batch_size=__BATCH__ \
  samples_per_epoch=${SPE} steps_per_epoch=$((SPE / __BATCH__)) max_epochs=${EP} \
  samples_per_epoch_val=${VAL} early_stop_patience=${PAT} val_rule=${VALRULE} val=${VALF} budget=v3.1
python3 experiments/FT/ft_weaver.py --seed ${S} --lean-val-metrics \
  --data-train ${SUB}/train_N${N}_s1.parquet --data-val ${VALF} --data-config ${CFG} \
  --network-config ${NET} -o num_classes ${K} -o fc_params '[(512,0.1)]' \
  --use-amp --batch-size __BATCH__ --num-workers 2 --fetch-by-files --fetch-step 1 --optimizer ranger \
  ${SCHED[@]+"${SCHED[@]}"} --optimizer-option weight_decay 0.01 ${MULTARG[@]+"${MULTARG[@]}"} \
  --start-lr ${LR} "${TRAINARGS[@]}" "${VALARGS[@]}" \
  --num-epochs ${EP} ${LOAD[@]+"${LOAD[@]}"} --model-prefix ${RUN}/net --log ${RUN}/train.log \
  > ${RUN}/stdout.log 2>&1 &
PID=$!
ES=0
python3 experiments/FT/early_stop.py watch --log ${RUN}/train.log --pid ${PID} --patience ${PAT} --patience-from ${PATFROM} \
  --max-epochs ${EP} ${PRUNE[@]+"${PRUNE[@]}"} --out ${CELL}/early_stop.json || ES=$?
wait ${PID} || true
cp ${RUN}/train.log ${RUN}/stdout.log ${CELL}/
# PI 2026-11-21, one divergence rule: a non-finite training loss is the rate's outcome, not a fault: recorded, never retried
if [ "${ES}" -eq 3 ] || grep -Eq "AvgLoss: (nan|inf|-inf)" ${RUN}/train.log; then
  cp ${CELL}/early_stop.json ${CELL}/DIVERGED 2>/dev/null || echo '{"diverged": true}' > ${CELL}/DIVERGED
  echo "FATAL: ${CELL} diverged (non-finite training loss)"; exit ${HALT}
fi
[ "${ES}" -eq 0 ] || { echo "FATAL: ${CELL}: training did not finish (early_stop rc=${ES})"; tail -30 ${RUN}/stdout.log; exit 1; }
grep -q "('steps_per_epoch', $((SPE / __BATCH__)))" ${RUN}/train.log || { echo "FATAL: weaver did not run $((SPE / __BATCH__)) steps per epoch"; exit 1; }
if [ -n "${RUN_DIR}" ]; then
  python3 experiments/FT/smoke_checks.py load-log --log ${RUN}/stdout.log
  grep -q "Parameters with lr multiplied by" ${RUN}/stdout.log || { echo "FATAL: weaver did not apply the head lr multiplier"; exit 1; }
fi
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_epoch'])" ${CELL}/early_stop.json)
LAST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['last_epoch'])" ${CELL}/early_stop.json)
cmp -s ${RUN}/net_best_epoch_state.pt ${RUN}/net_epoch-${BEST}_state.pt || { echo "FATAL: weaver's best state is not epoch ${BEST}"; exit ${HALT}; }
if [ "${VALMEM}" = 1 ]; then    # every check scored all VAL jets, never fewer
  NV=$(grep -c "Processed ${VAL} entries" ${RUN}/train.log); [ "${NV}" -ge "${EP}" ] || { echo "FATAL: ${NV} of ${EP} checks scored ${VAL} jets"; exit ${HALT}; }
fi
cp ${RUN}/net_best_epoch_state.pt ${CELL}/net_best_epoch_state.pt
cp ${RUN}/net_epoch-${LAST}_state.pt ${CELL}/net_last_epoch_state.pt
case ${TASK} in
  jc1)
    weaver --predict --data-test ${TEST} --data-config ${CFG} --network-config ${NET} -o num_classes ${K} \
      -o fc_params '[(512,0.1)]' --predict-gpus 0 --batch-size 512 --num-workers 2 --fetch-by-files --fetch-step 1 \
      --model-prefix ${CELL}/net_best_epoch_state.pt --predict-output ${CELL}/pred.root 2>&1 | tee ${CELL}/predict.log | tail -3
    [ -f ${CELL}/pred.root ] || { echo "FATAL: no pred.root in ${CELL}"; exit 1; }
    python3 experiments/FT/slim_outputs.py pred --file ${CELL}/pred.root;;
  *)
    python3 experiments/EVAL/extract_features.py --checkpoint ${CELL}/net_best_epoch_state.pt --num-classes ${K} \
      --arm FTB_${TASK}_${INIT:-scratch}_N${N}_s${S} --data-config ${TCFG} --data-test ${TEST} \
      --observers jet_pt jet_energy --out ${CELL}/features --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
    if [ "${TASK}" = "qg" ]; then
      python3 experiments/EVAL/extract_features.py --checkpoint ${CELL}/net_best_epoch_state.pt --num-classes ${K} \
        --arm FTB_${TASK}_${INIT:-scratch}_N${N}_s${S}_herwig --data-config ${CFG} --data-test __HERWIG__ \
        --observers jet_pt jet_energy --out ${CELL}/features_herwig --batch-size 512 --num-workers 1 --fetch-step 1 --save-logits
      mv ${CELL}/features_herwig/logits.npy ${CELL}/features/logits_herwig.npy
      mv ${CELL}/features_herwig/label188.npy ${CELL}/features/label_herwig.npy
    fi
    # the read-outs take logits, labels and observers only (the fine-tuned embeddings re-extract from the checkpoint):
    # features.npy (0.2-0.45 GB a cell) is checked and dropped, as in build_ft_jobs, so the rows fit under the line
    python3 -c "import numpy as np, sys; d = sys.argv[1]; F = np.load(d + '/features.npy', mmap_mode='r'); G = np.load(d + '/logits.npy', mmap_mode='r'); L = np.load(d + '/label188.npy', mmap_mode='r'); ok = F.shape[0] == G.shape[0] == L.shape[0] and np.isfinite(F).all() and np.isfinite(G).all(); sys.exit(0 if ok else 'FAIL features: ' + str((F.shape, G.shape, L.shape)))" ${CELL}/features || exit ${HALT}
    rm -f ${CELL}/features/features.npy ${CELL}/features_herwig/features.npy;;
esac
cat /sys/fs/cgroup/memory.peak > ${CELL}/memory_peak_bytes 2>/dev/null || true    # the pod's peak, to size requests
touch ${CELL}/DONE
echo "FT BUDGET CELL ${IDX} DONE: ${CELL}"
"""


def task_cases() -> str:
    out = []
    for t, (sub, cfg, k, net, _tree, _sizes, test, _mem) in TASKS.items():
        if test == "JC1_TEST":
            test_block = ("TEST=\"\"; for C in " + " ".join(JC1_CLASSES) + "; do for f in $(ls /data/JetClass/Pythia/"
                          "test_20M/${C}_*.root | sort | head -2); do TEST=\"${TEST} ${f}\"; done; done; "
                          "[ \"$(echo ${TEST} | wc -w)\" -eq 20 ] || { echo \"FATAL: JetClass test files\"; exit ${HALT}; }")
        else:
            test_block = f'TEST="{test}"'
        out.append(f"  {t}) SUB={sub}; CFG={cfg}; TCFG={TEST_CFG.get(t, cfg)}; K={k}; NET={net}; {test_block};;\n")
    return "".join(out)


def cell_cases(cells: list) -> str:
    return "".join(f"  {i}) TASK={c['task']}; INIT={c['init']}; RUN_DIR={c['run_dir']}; N={c['n']}; S={c['s']}; "
                   f"LR={c['lr']}; MULT={c['mult']}; EP={c['epochs']}; SPE={c['spe']}; PAT={c['patience']}; "
                   f"VAL={c['val']}; PATFROM={c.get('patience_from', 0)}; VALF={c.get('valf', '')}; VALMEM={c.get('valmem', 0)}; "
                   f"CELL={c['out']};;\n" for i, c in enumerate(cells))


GPU_POOLS = ("gpu", "a100")     # nvidia.com/<pool>: NRP's A100s are their own resource (cms-ml quota 26, shared)


def spec(name: str, cells: list, tag: str, parallelism: int, memory: str, cpu: int = 4,
         line: int = storage_line.GENERAL, pool: str = "gpu", product: str = "") -> str:
    if pool not in GPU_POOLS:
        raise SystemExit(f"FATAL: GPU pool {pool}: one of {GPU_POOLS}")
    launch = _launch()
    script = (CELL_SCRIPT.replace("__LINE__", str(storage_line.check(line))).replace("__CASES__", cell_cases(cells)).replace("__TASKS__", task_cases())
              .replace("__BATCH__", str(BATCH)).replace("__HERWIG__", " ".join(HERWIG_TEST)))
    # a node's DNS can fail for minutes (cph-dgx-node4/5/8, 2026-11-03): six cells used up their three
    # attempts at the clone, in seconds, before reaching the training script
    script = ("set -euo pipefail\n"
              'retry () { for T in 1 2 3 4 5; do "$@" && return 0; echo "attempt ${T} failed: $1"; sleep 60; done; return 1; }\n'
              'retry git clone -q --depth 1 --branch "${REPO_REF}" https://github.com/raunavm/transferlearningsophon.git '
              "/workspace/transferlearningsophon || exit 1\ncd /workspace/transferlearningsophon\n"
              "retry pip install --no-cache-dir -q pyarrow==25.0.1 || exit 1\n" + script.lstrip("\n"))
    body = "\n".join(("          " + ln) if ln else "" for ln in script.splitlines())
    exclude = ", ".join(f'"{n}"' for n in launch.V2_BAD_NODES + launch.V2_LATER_BAD_NODES)
    if len(name) > 63:
        raise SystemExit(f"{name}: too long")
    # the A100 resource itself picks the product; the general pool lists the products we accept
    products = "" if pool == "a100" else (
        "              - key: nvidia.com/gpu.product\n                operator: In\n"
        f"                values: [{', '.join(repr(g).replace(chr(39), chr(34)) for g in ([product] if product else launch.V2_FT_GPU))}]\n")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  # GENERATED by scripts/build_ft_budget_jobs.py -- fine-tuning with the fixed step budget (plan 3.1).
  name: {name}
  namespace: cms-ml
spec:
  completionMode: Indexed
  completions: {len(cells)}
  parallelism: {parallelism}
  backoffLimitPerIndex: 3
  podFailurePolicy:
    rules:
    - action: FailIndex
      onExitCodes: {{ containerName: main, operator: In, values: [42] }}
    - action: Ignore
      onPodConditions:
      - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: {launch.V2_IMAGE}
        command: ["/bin/bash", "-c"]
        env:
        - name: REPO_REF
          value: "{tag}"
        - name: NODE_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: spec.nodeName }} }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        args:
        - |
{body}
        resources:
          requests: {{ memory: "{memory}", cpu: "{cpu}", nvidia.com/{pool}: "1", ephemeral-storage: "20Gi" }}
          limits:   {{ memory: "{memory}", cpu: "{cpu}", nvidia.com/{pool}: "1", ephemeral-storage: "20Gi" }}
        volumeMounts:
        - {{ name: data, mountPath: /data }}
        - {{ name: dshm, mountPath: /dev/shm }}
      tolerations:
      - {{ key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-west"]
{products}              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{exclude}]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: transfer-learning-vol
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: "8Gi" }}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr-pre", default=None)
    ap.add_argument("--lr-scratch", default=None)
    ap.add_argument("--lr-scratch-qg", default=None, help="quark/gluon's own from-scratch rate (its sweep)")
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=4, help="CPUs per pod (3: fits CPU-starved nodes with a free GPU, PI 2026-11-06)")
    ap.add_argument("--name-suffix", default="", help="a second job of the same cells, e.g. cpu3")
    ap.add_argument("--mem", default="", help="memory per pod, overriding the rule below (64Gi: the long JetClass "
                    "cells peaked at 53 GiB over 22 pods, and 88Gi kept them off nodes with a free GPU, 2026-11-07)")
    ap.add_argument("--gpu-product", default="", help="one nvidia.com/gpu.product for every pod (the benchmark)")
    ap.add_argument("--gpu-pool", default="gpu", choices=GPU_POOLS,
                    help="a100: request nvidia.com/a100 (NRP's separate A100 pool) instead of nvidia.com/gpu")
    ap.add_argument("--part", default=None, help="with --indices: the name of one part of a kind split over GPU pools")
    ap.add_argument("--indices", default=None,
                    help="comma-separated indices of the full kind's job: a -retry job of those cells only")
    ap.add_argument("--rerun", action="append", default=[], metavar="TASK:N",
                    help="pre-full-rerun / readout-pre-full: a (task, size) of PRE_FULL_SIZES that moved, e.g. top:1000")
    storage_line.add_argument(ap)
    ap.add_argument("kinds", nargs="+", choices=("lrsweep", "bench", "jc1", "jc2b", "kin", "fillcheck", "long-jc2b", "long-jc1", "long-jc2b-ctrl", "long-jc2b-rest",
                                                 "long-scratch-full", "scratch-full", "pre-full", "pre-full-rerun",
                                                 "readout-pre-full", "fixed-1e5", "fixed-1e6", "fixed-rest", "fixed-bench", "fixed-sweep",
                                                 "fixed-sweep-ext", "fixed-sweep-jc2b", "fixed-sweep-rows", "fixed-head-jc2b", "fixed-sweep-all", "fixed-rate", "fixed-sweep-edge", "fixed-sweep-edge2",
                                                 "fixed-sweep-1e4", "fixed-smoke-diverge", "fixed-sweep-17", "fixed-smoke-1e3",
                                                 "fixed-1e4", "fixed-1e3"))
    a = ap.parse_args(argv)
    if a.part and not a.indices:
        ap.error("--part needs --indices")
    if a.indices and len(a.kinds) != 1:
        ap.error("--indices needs exactly one kind")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reruns = [(t, int(float(n))) for t, n in (r.split(":") for r in a.rerun)]
    for kind in a.kinds:
        if kind == "readout-pre-full":
            if not a.lr_pre:
                ap.error("readout-pre-full needs --lr-pre")
            cells = pre_full_cells(a.lr_pre) + (pre_full_rerun_cells(a.lr_pre, reruns) if reruns else [])
            name = "budget-readout-prefull-raunav"
            p = OUT_DIR / f"job-{name}.yaml"
            p.write_text(readout_spec(name, cells, a.tag, storage_line.check(a.storage_line), PRE_FULL_METRICS))
            print(p, len(cells), "cells")
            continue
        if kind == "lrsweep":
            cells, mem = lrsweep_cells(), "48Gi"
        elif kind == "fillcheck":
            cells, mem = fillcheck_cells(), "48Gi"
        elif kind.startswith("fixed-"):     # the one rule's rates are fixed (FIXED_LR_*), not taken from the sweep flags
            cells = {"fixed-1e5": lambda: fixed_cells({100_000}), "fixed-1e6": lambda: fixed_cells({1_000_000}),
                     "fixed-rest": lambda: fixed_cells({1_000, 10_000}), "fixed-1e4": lambda: fixed_cells({10_000}),
                     "fixed-1e3": lambda: fixed_cells({1_000}), "fixed-bench": fixed_bench_cells,
                     "fixed-sweep": fixed_sweep_cells,
                     "fixed-sweep-ext": lambda: fixed_sweep_cells(FIXED_SWEEP_PRE_EXT, ()),
                     "fixed-sweep-17": lambda: fixed_sweep_cells(FIXED_SWEEP_17, (), "r16q1-s1"),
                     "fixed-sweep-jc2b": fixed_sweep_jc2b_cells,
                     "fixed-sweep-rows": fixed_sweep_rows_cells,
                     "fixed-head-jc2b": fixed_head_jc2b_cells,
                     "fixed-sweep-all": fixed_sweep_all_cells,
                     "fixed-rate": fixed_rate_cells,
                     "fixed-sweep-edge": fixed_sweep_edge_cells,
                     "fixed-sweep-edge2": fixed_sweep_edge2_cells,
                     "fixed-sweep-1e4": fixed_sweep_1e4_cells,
                     "fixed-smoke-diverge": fixed_smoke_diverge_cells,
                     "fixed-smoke-1e3": fixed_smoke_1e3_cells}[kind]()
            # sized to measured use (NRP blocks accounts whose pods use little of what they request, 2026-11-10/15):
            # small cells ~16 GiB before the in-memory checks; 1e6 set from the smoke cell (fixed-bench index 2)
            mem = "24Gi"
        else:
            if not (a.lr_pre and a.lr_scratch):
                ap.error(f"{kind} needs --lr-pre and --lr-scratch (from the sweep)")
            per_task = {"qg": a.lr_scratch_qg} if a.lr_scratch_qg else None
            cells = {"bench": lambda: bench_cells(a.lr_pre, a.lr_scratch, ("top", "qg"), lr_scratch_task=per_task),
                     "jc1": lambda: bench_cells(a.lr_pre, a.lr_scratch, ("jc1",)),
                     "jc2b": lambda: jc2b_cells(a.lr_pre, a.lr_scratch),
                     "kin": lambda: kin_cells(a.lr_pre),
                     "long-jc2b": lambda: long_cells("jc2b", a.lr_pre, a.lr_scratch),
                     "long-jc1": lambda: long_cells("jc1", a.lr_pre, a.lr_scratch),
                     "long-jc2b-ctrl": lambda: long_ctrl_cells(a.lr_pre),
                     "long-jc2b-rest": lambda: long_rest_cells(a.lr_pre),
                     "pre-full": lambda: pre_full_cells(a.lr_pre),
                     "pre-full-rerun": lambda: pre_full_rerun_cells(a.lr_pre, reruns),
                     "scratch-full": lambda: scratch_full_cells(a.lr_scratch, a.lr_scratch_qg or a.lr_scratch),
                     "long-scratch-full": lambda: (long_scratch_full_cells("jc2b", a.lr_scratch)
                                                   + long_scratch_full_cells("jc1", a.lr_scratch))}[kind]()
            mem = "48Gi" if kind in ("bench", "kin", "pre-full-rerun") else "64Gi"
        if any(c["n"] >= 1_000_000 for c in cells):
            # weaver holds a whole training file in memory: 48Gi OOM-killed the top fill check at
            # 1.2M jets four times (2026-11-02); the old full-size top fine-tunings ran at 88Gi
            mem = FULL_MEM
        mem = a.mem or mem
        name = f"ft-budget-{kind}-raunav"
        if a.indices:
            cells = [cells[int(i)] for i in a.indices.split(",")]
            # --part: one part of a kind split over GPU pools (disjoint indices, never also run by another job)
            name = f"ft-budget-{kind}-{a.part}-raunav" if a.part else f"ft-budget-{kind}-retry-raunav"
        if a.name_suffix:
            name = name.replace("-raunav", f"-{a.name_suffix}-raunav")
        p = OUT_DIR / f"job-{name}.yaml"
        p.write_text(spec(name, cells, a.tag, a.parallelism, mem, a.cpu, a.storage_line, a.gpu_pool, a.gpu_product))
        print(p, len(cells), "cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
