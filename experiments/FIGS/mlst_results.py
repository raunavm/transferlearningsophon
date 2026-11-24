"""The MLST paper's results tables and macros, read by experiments/FIGS/make_tables.py (build()).

Every value is the mean +- standard deviation over the pretraining runs of a configuration that the
paper reports (configs/arms/v2_paper_runs.txt), PDG-rounded (make_tables.fmt_pm). A configuration
with runs still missing prints the runs it has and a red marker "n of N runs". Configurations with
two runs print mean +- SD in tables too (PI 2026-11-12) and both values in a sentence; a lone
run prints three significant figures. Never more than four significant figures in a table.

Rules of the rewrite (PI decisions for Phase B, 2026-11-02, docs/PAPER_OUTLINE.md):
  * 1 - AUC appears only as a ratio between pretraining configurations on JetClass-II; top tagging,
    quark/gluon and JetClass use accuracy, AUC and background rejection (R50, R30, ParT's per-class
    rejections), never 1 - AUC;
  * the mass resolution is ATLAS's (half the 15.9-84.1 % inter-quantile range over the median, with
    the median response; head/atlas_res/); sigma_eff is reported nowhere;
  * mass sculpting is the Jensen-Shannon distance (base 2, the square root of the divergence);
  * the linear probes on top and quark/gluon use one substitution for the missing inputs per task,
    chosen on validation and applied to every model including the randomly initialized network;
  * every ratio the text quotes has a ratio word <name>Pr (two significant figures, the direction
    in words): the ratio of the two sides' means as the tables print them (review of render dfe5942,
    LEDGER CS-6 and CS-8), stated only where its tab:tests row passes 3 sigma (PR_ROWS, ratio_words;
    PI 2026-11-23, LEDGER SP-12 step 3).

Inputs, under experiments/FIGS/data/v2/ (experiments/FIGS/pull_mlst_data.py):
  linprobe/fits_v3/, fits_v3_initbn/      linear probes, natural-composition JetClass-II (and the
                                          BatchNorm check, which only these fits have)
  linprobe/fits_v4/, fits_v4_mlp/          class-balanced JetClass-II probes (replace fits_v3 once pulled)
  linprobe/fits_v4_fill/                   the three substitutions of the missing inputs
  linprobe/fits_v4_sib/                    the X->ee vs X->mumu probe (joins the v4 pairs, mlst_v4.load_fits)
  curves_v2/<family>/<model>.json          ROC curves (mlst_v4.load_roc): F4/F6J cut points, test-set sizes
  linprobe/fits_v3_mlp2/, ..._initbn/      nonlinear probes (top, quark/gluon, JetClass; JetClass-II at
                                           natural composition, the only composition of the mass models
                                           and the numbered random partitions)
  head/atlas_res/<model>.json              ATLAS mass resolution and median response
  head/curves_v2/<model>_sculpting_v2.json, head/<model>_sculpting.json   QCD mass sculpting
  compute/<run>.json                       per epoch: GPU and seconds
  ft_mlst/*.json, budget/*.json            fine-tuning read-outs; curves/full_*.json  JetClass per class (jc1_full)
  bootstrap/<family>/*.json                test-sample bootstrap; partrepro/<variant>.json  ParT on top
  xbx/<signal>/<model>.json, xbx_scratch/<signal>/s<k>.json   X->bb and X->bs
  jc2bal/census.json, jc2_test/counts.json class counts; index.json  {path: sha256}

Outputs (paper/journal/tables/): main text mlst_summary, mlst_published (the comparison with published models,
PI 2026-11-12; mlst_jetclass, mlst_bench_top, mlst_bench_qg and mlst_bench_qg_herwig are still written for the draft
sections that input them and are placed nowhere), appendix mlst_labelsets (the other label sets fine-tuned),
T5 mlst_mass (T1 usecase_survival is make_tables.table_usecase); appendix mlst_probes, mlst_probes_n,
mlst_per_run, mlst_jetclass_all, mlst_jetclass_probe, mlst_jetclass_nbkg, mlst_ft, mlst_ft_rej (A1),
mlst_lrsweep (A4), mlst_mlp (A5), mlst_bn (A6), mlst_boot (A7), mlst_inputs (A8), mlst_partrepro (A9),
mlst_pairs (A11), mlst_per_class (A12), mlst_balance (A13), mlst_sculpt (A14), mlst_xbx and
mlst_xbx_rej (A15), mlst_fill (A16), mlst_mass_detail (A17), mlst_rand_prong (A18), compute.
"""
from __future__ import annotations

import calendar
import collections
import csv
import importlib.util
import functools
import io
import itertools
import json
import math
import pathlib
import re
import statistics
import time

import numpy as np

_V4_SPEC = importlib.util.spec_from_file_location("mlst_v4", pathlib.Path(__file__).resolve().parent / "mlst_v4.py")
V4 = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(V4)

MAIN = ("L188", "L162", "R42_Q1", "R16_Q1")
MASS = ("L162_MASS", "R16_Q1_MASS")
RAND = tuple(f"RAND2_p{k}" for k in range(1, 6))
FLAV = ("FLAV_F0", "FLAV_F1", "FLAV_F1R")
# the one row order of every table and figure family (PI 2026-11-13, L5): the label sets, each mass-regression network
# directly under its own label set (SETS), then the 17-class partitions (flavor boundaries, then the random
# partitions); a table adds the random network (probes) or from scratch (fine-tuning), the released Sophon checkpoint last
SETS = ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS")
ALL_ROWS = SETS + FLAV + RAND
INIT = "INIT"
INIT_MODELS = ("init-s1", "init-s2", "init-s3")
# the randomly initialized networks with their BatchNorm statistics recomputed on the training jets
# of the best70 twins (experiments/MTX/bn_twins_v2.py --init): checkpoint init_bn
INIT_BN = "INIT_BN"
INIT_BN_MODELS = ("initbn-s1", "initbn-s2", "initbn-s3")
# the 188-class model trained on the four-momenta only (plan 3.7): one run, not in the grid file
KIN_MODEL = "mtx-l188kin-s1"
KIN_INIT = KIN_MODEL[len("mtx-"):]
# the five random partitions as one row, mean +- SD over their runs (one each), as tab:jetclassall pools them
RAND_POOLED = "RAND"
PRIMARY = "best70"
# the released Sophon checkpoint (scripts/build_jc2bal_jobs.py sophon), probed as our runs are: a reference row of the
# frozen-probe tables below a rule, outside the controlled comparison (PI 2026-11-08)
SOPHON, SOPHON_MODEL, SOPHON_CK = "SOPHON", "sophon-released", "released"
FITS_SOPHON = "linprobe/sophon"


def _ck(model: str) -> str:
    """The checkpoint a model is reported at."""
    return ("init" if model.startswith("init-") else "init_bn" if model.startswith("initbn-")
            else SOPHON_CK if model == SOPHON_MODEL else PRIMARY)


# Row labels, as the figure legends: the network pretrained on JetClass-II by its classes, or the control by the shortest
# name the paper uses. A row never names the downstream mode (PI 2026-11-12): a table of one mode says it once, in an
# italic heading over its rows (HEADING, MODES, _headings). Where the mode changes across columns the column headings
# name it.
LABEL = {"L188": r"\Kfull{} classes", "L162": r"\Kres{} classes", "R42_Q1": r"\Kmid{} classes",
         "R16_Q1": r"\Kcoarse{} classes", "L162_MASS": r"\Kres{} classes + mass", "R16_Q1_MASS": r"\Kcoarse{} classes + mass",
         "FLAV_F0": "no flavor boundary", "FLAV_F1": r"$b$-quark boundary", "FLAV_F1R": "random flavor boundary",
         INIT: "random network", INIT_BN: "random network, BatchNorm recomputed",
         "KIN": r"\Kfull{} classes, kinematic inputs only", RAND_POOLED: "random partitions",
         SOPHON: "released Sophon checkpoint",
         **{f"RAND2_p{k}": f"random partition {k}" for k in range(1, 6)}}
SCRATCH_LABEL = V4.SCRATCH_LABEL      # "from scratch": a fine-tuning table's reference, never mode-tagged
HEADING = {"ft": "fine-tuned", "lp": "linear probe on the frozen features", "direct": "pretraining head used directly",
           "mlp": "multilayer-perceptron probe on the frozen features", "head": "mass output of the pretraining head"}
PARTITIONS = "17-class partitions"      # the subheading over the indented partition rows


def _full_row(ncol: int, text: str, italic: bool = False) -> str:
    """A heading across a tabular of ncol columns, flush with the row labels; the 17-class partitions subheading in
    italic in every table, as in the figures (PI 2026-11-14)."""
    italic = italic or text == PARTITIONS
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{" + (rf"\textit{{{text}}}" if italic else text) + r"} \\"


FITS, MLP_FITS = "linprobe/fits_v3", "linprobe/fits_v3_mlp"
MLP_FITS_V4, FITS_INITBN, MLP_FITS_INITBN = ("linprobe/fits_v3_mlp2", "linprobe/fits_v3_initbn",
                                             "linprobe/fits_v3_mlp2_initbn")
FITS_FILL, MLP_BAL = "linprobe/fits_v4_fill", "linprobe/fits_v4_mlp"
SOPHON_FILL_V5 = "linprobe/sophon_v5_fill"     # the released checkpoint's substitution probes (probe_released.py --fill-root)
# (dataset key, metric, column heading)
TASKS = (("jc2_188", "macro_auc_ovr", "JetClass-II"), ("jc2_188", "macro_auc_resonant", "resonant"),
         ("jc2_188", "macro_auc_qcd", "QCD"), ("jc1", "macro_auc_ovr", "JetClass"),
         ("top", "auc", "top"), ("qg", "auc", "quark/gluon"), ("qg_herwig", "auc", "quark/gluon, Herwig"))
TASKS_N = (TASKS[0], ("jc2_188", "macro_auc_trained", "JetClass-II, classes present in training"),
           TASKS[2], TASKS[3], TASKS[4], TASKS[5])
# the JetClass-II tasks, the only ones a ratio of 1 - AUC is taken on (PI, 2026-11-02)
JC2_TASKS = (TASKS[0], TASKS_N[1], TASKS[1], TASKS[2])
PAIRS = (("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", ("label_X_bc",), ("label_X_bq", "label_X_cs")),
         ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", ("label_X_bb",), ("label_X_cc",)),
         ("bbqq_vs_ccqq", r"$X\to YY\to b\bar{b}qq$ vs $c\bar{c}qq$", ("label_X_YY_bbqq",), ("label_X_YY_ccqq",)),
         ("ee_vs_mm", r"$X\to ee$ vs $\mu\mu$", ("label_X_ee",), ("label_X_mm",)),
         ("qcd_bb_vs_cc", r"QCD $b\bar{b}$ vs $c\bar{c}$", ("label_QCD_bb",), ("label_QCD_cc",)),
         ("qcd_b_vs_light", r"QCD with $b$ vs light", "qcd_with_b", ("label_QCD_light",)),
         ("qcd_c_vs_light", r"QCD with $c$, no $b$, vs light", "qcd_with_c_no_b", ("label_QCD_light",)))
# the class-pair figures' panel order (gallery_roc.PAIR_ORDER, PI 2026-11-13): the pair tables' columns
PAIR_ORDER = ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq", "qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light", "ee_vs_mm")
PAIRS_FIG = tuple(sorted(PAIRS, key=lambda p: PAIR_ORDER.index(p[0])))
# tab:pairs' columns: the X->ee vs mumu column is out (PI 2026-11-19, table_pairs' EE_NOTE check)
PAIRS_TABLE = tuple(p for p in PAIRS_FIG if p[0] != "ee_vs_mm")
PAIR_DS, PAIR_N = "jc2_188_pairs", "1000000"
MLP_TASKS = (("jc2_188", "JetClass-II", "macro_auc_ovr"), ("jc2_188_res", "resonant", "macro_auc_resonant"),
             ("jc2_188_qcd", "QCD", "macro_auc_qcd"), ("jc1", "JetClass", "macro_auc_ovr"),
             ("top", "top", "auc"), ("qg", "quark/gluon", "auc"), ("qg_herwig", "quark/gluon, Herwig", "auc"),
             ("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", "auc"), ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", "auc"),
             ("bbqq_vs_ccqq", r"$X\to YY\to b\bar{b}qq$ vs $c\bar{c}qq$", "auc"), ("qcd_bb_vs_cc", r"QCD $b\bar{b}$ vs $c\bar{c}$", "auc"),
             ("qcd_b_vs_light", r"QCD with $b$ vs light", "auc"),
             ("qcd_c_vs_light", r"QCD with $c$, no $b$, vs light", "auc"))
# the pairs with a nonlinear probe (no multilayer perceptron on X->ee vs X->mumu)
MLP_PAIRS = tuple(p for p in PAIRS if p[0] in {t for t, *_ in MLP_TASKS})
MLP_JC2 = ("jc2_188", "jc2_188_res", "jc2_188_qcd") + tuple(k for k, *_ in MLP_PAIRS)
# the class pairs of F4, in panel order
ROC_PAIRS = ("bb_vs_cc", "bbqq_vs_ccqq", "bc_vs_bq_cs", "ee_vs_mm")
# fewest class-balanced test jets of a class that is not hatched in F3 (MlstJcBalNThin) and may be
# selected for A3R; and how many classes A3R shows as most hurt and as least changed
THIN_TEST = 1000
NHURT, NSTABLE = 8, 4
# jets below which a bin of the mass resolution is null (experiments/EVAL/atlas_resolution.py, read from
# its source: importing it pulls the mass-regression code)
MASS_MIN_JETS = int(re.search(r"^MIN_JETS = (\d+)$", (pathlib.Path(__file__).resolve().parents[2]
                                                       / "experiments/EVAL/atlas_resolution.py").read_text(),
                              re.M).group(1))
# F9a's response axes cover the bins with this share of a series' largest bin (read from the figure's
# source, which imports matplotlib)
F9A_SHOWN_FROM = float(re.search(r"^SHOWN_FROM = ([0-9.]+)", (pathlib.Path(__file__).resolve().parent
                                                               / "figstyle/gallery_physics.py").read_text(),
                                 re.M).group(1))
# the zero-fill control's two masks (fillcontrol/<mask>/); the paper no longer reads them, the mock-up
# figures (mockup_figures.figure_f5) still do
MASKS = ("TopReference", "EnergyFlowQG")
# Fine-tuning: (task, read-out file, data set inside it, metric, column heading)
FT_TAGS = ("best70",)
FT_TASKS = (("jc2", "leg188_leg1_metrics", None, "macro_auc_ovr", "JetClass-II"),
            ("jc2_qcd", "leg188_leg1_metrics", None, "macro_auc_qcd", "JetClass-II"),
            ("jc1", "leg2_metrics", None, "macro_auc_ovr", "JetClass"),
            ("top", "bench_metrics", "top", "auc", "top"),
            ("qg", "bench_metrics", "qg", "auc", "quark/gluon"),
            ("qg_herwig", "bench_metrics_herwig", "qg", "auc", "quark/gluon, Herwig"))
# the metric each task's fine-tuning is reported in (PI 2026-11-02: no 1 - AUC outside JetClass-II)
FT_METRIC = {"jc2": ("macro_auc_ovr", "AUC"), "jc2_qcd": ("macro_auc_qcd", "AUC over the QCD classes"),
             "jc1": ("accuracy", "accuracy"), "top": ("r30", "$R_{30}$"), "qg": ("r30", "$R_{30}$"),
             "qg_herwig": ("r30", "$R_{30}$")}
FT_REJ = (("top", "r50"), ("top", "r30"), ("qg", "r50"), ("qg", "r30"), ("qg_herwig", "r50"), ("qg_herwig", "r30"))
BENCH4 = (("accuracy", "accuracy"), ("auc", "AUC"), ("r50", "$R_{50}$"), ("r30", "$R_{30}$"))
SCRATCH = "scratch-v2"

WORD = {"L188": "Full", "L162": "Res", "R42_Q1": "Mid", "R16_Q1": "Coarse", "L162_MASS": "ResMass",
        "R16_Q1_MASS": "CoarseMass", INIT: "Init", INIT_BN: "InitBn", "FLAV_F0": "Fzero", "FLAV_F1": "Fone",
        "FLAV_F1R": "Foner", SCRATCH: "Scratch", "KIN": "Kin"}
TASK_WORD = {("jc2_188", "macro_auc_ovr"): "JcTwo", ("jc2_188", "macro_auc_resonant"): "JcTwoRes",
             ("jc2_188", "macro_auc_qcd"): "JcTwoQcd", ("jc2_188", "macro_auc_trained"): "JcTwoTrained"}
PAIR_WORD = {"bc_vs_bq_cs": "Bc", "bb_vs_cc": "Bb", "bbqq_vs_ccqq": "Bbqq", "ee_vs_mm": "Eemm", "qcd_bb_vs_cc": "QcdBb",
             "qcd_b_vs_light": "QcdB", "qcd_c_vs_light": "QcdC", "top": "Top", "qg": "Qg",
             "qg_herwig": "QgHerwig", "jc2_188": "JcTwo", "jc2_188_res": "JcTwoRes", "jc2_188_qcd": "JcTwoQcd",
             "jc1": "JcOne"}
FT_WORD = {"jc2": "JcTwo", "jc2_qcd": "JcTwoQcd", "jc1": "JcOne", "top": "Top", "qg": "Qg",
           "qg_herwig": "QgHerwig"}
SIZE_WORD = {1000: "EThree", 10000: "EFour", 100000: "EFive", 1000000: "ESix"}
TEXT_SEP = " and "      # two runs in a sentence: "1.165 and 1.166"
# the full training sets of the fine-tuning tasks (ParT's recipe there, the budget recipe below)
FULL_N = {"top": 1_200_000, "qg": 1_600_000, "qg_herwig": 1_600_000, "jc1": 1_000_000}


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


def _pooled(runs: dict) -> dict:
    """runs with the five random partitions also as one configuration, RAND_POOLED."""
    return {**runs, RAND_POOLED: [m for a in RAND for m in runs.get(a, [])]}


def _paper_rows(rows) -> tuple:
    """rows with the five random partitions as one row, RAND_POOLED, where the first of them stood (after the
    flavor-boundary partitions, as in Figs. 2 and 3); PI 2026-11-12: in every table of the paper, the supplement's
    *_full tables keep each partition."""
    out = []
    for r in rows:
        if r not in RAND:
            out.append(r)
        elif RAND_POOLED not in out:
            out.append(RAND_POOLED)
    return tuple(out)


def _pretrained(runs: dict) -> list:
    return [m for k, v in runs.items() if k not in (INIT, INIT_BN) for m in v]


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


# a probe that did not converge carries this mark, and the build counts them (MlstProbeUnconverged)
UNCONVERGED = r"$^{*}$"


class Unconverged(float):
    pass


Bound = V4.Bound      # a rejection where no background jet passed: a lower limit (mlst_v4.rejection, one convention)


def _sep(s: str) -> str:
    """Thousands separated as fmt_pm does."""
    head, dot, tail = s.partition(".")
    if "e" in s or len(head.lstrip("-")) <= 3:
        return s
    return f"{int(head):,}".replace(",", "{,}") + dot + tail


def _lone(x: float) -> str:
    """A single run: three significant figures, trailing zeros kept; from 100 on an integer."""
    if abs(x) >= 100:
        return _sep(str(int(round(x, 2 - int(math.floor(math.log10(abs(x))))))))
    return _sep(f"{x:#.3g}".rstrip("."))


def _two(a: float, b: float, sep: str) -> str:
    """Two runs: both values. In a table at one precision per magnitude, so that a column reads alike:
    below 1 four decimals, below 10 two, below 100 one, from 100 integers of at most four significant
    figures. In a sentence (sep TEXT_SEP), to the decimal that shows their difference to two digits,
    never beyond four significant figures."""
    big = max(abs(a), abs(b))
    if sep != TEXT_SEP:
        if big >= 100:
            return sep.join(_tab1(x) for x in (a, b))
        dec = 4 if big < 1 else 2 if big < 10 else 1
        return sep.join(_sep(f"{x:.{dec}f}") for x in (a, b))
    d = abs(a - b)
    if d == 0:
        return _lone(a) + sep + _lone(b)
    cap = 3 - math.floor(math.log10(big))
    dec = max(0, min(1 - math.floor(math.log10(d)), cap))
    return sep.join(_sep(f"{x:.{dec}f}") for x in (a, b))


def _tab1(x: float) -> str:
    """A single value in a table: from 100 an integer of at most four significant figures (1,803;
    14,290); else three significant figures (an accuracy or AUC asks for four decimals: _cell frac)."""
    if abs(x) >= 100:
        return _sep(str(int(round(x, min(0, 3 - int(math.floor(math.log10(abs(x)))))))))
    return _lone(x)


def _cell(values: list, n_expected: int, fmt_pm, scale: float = 1.0, sep: str = ", ", frac: bool = False,
          run: bool = True) -> str:
    """mean +- SD over two or more runs present (PI 2026-11-12; in a sentence two runs print both values); a lone run
    prints alone, in a table with the single-run mark (run=False: a number that is not a run's, unmarked); missing
    runs add a red marker; a lower limit among them a geq sign."""
    flag = any(isinstance(x, Unconverged) for x in values)
    bound = any(isinstance(x, Bound) for x in values)
    v = [x * scale for x in values if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if V4.tokens(fmt_pm) and sep != TEXT_SEP:       # a table cell: its column's precision is set later
        body = (V4.tok("p", float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1
                else V4.tok("s" if run else "o", v[0]))
    elif len(v) > 2 or (len(v) == 2 and sep != TEXT_SEP):
        body = fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    elif len(v) == 2:
        body = _two(v[0], v[1], sep)
    else:       # in a table a lone run below 1 (accuracy, AUC) has four decimals like the means beside it
        body = f"{v[0]:.4f}" if frac else _tab1(v[0]) if sep != TEXT_SEP else _lone(v[0])
    if bound:
        body = r"$\geq$" + body
    if flag:
        body += UNCONVERGED
    if len(v) < n_expected:
        body += rf" \pending{{{len(v)} of {n_expected}}}"
    return body


class _TableFmt:
    """fmt_pm for the table builders: a cell's values as a token (mlst_v4.tok) that uniform_columns prints at
    one precision per column (PI, 2026-11-07); two(a, b, sep) the same for two runs. Runs that agree are an SD of
    zero, which _render marks (AGREE)."""
    tokens = True

    def __call__(self, mean, sd) -> str:
        return V4.tok("p", mean, sd or 0.0)

    @staticmethod
    def two(a, b, sep=", ") -> str:
        return V4.tok("p", (a + b) / 2, abs(a - b) / math.sqrt(2))


TAB = _TableFmt()


def _pct_cell(values: list, n_expected: int, fmt_pm) -> str:
    """_cell of values in per cent, the per-cent sign before any marker."""
    b = _cell(values, n_expected, fmt_pm)
    if b.startswith(r"\pending"):
        return b
    head, sp, tail = b.partition(r" \pending")
    mark = SINGLE if head.endswith(SINGLE) else ""         # the single-run mark after the per-cent sign
    return head.removesuffix(mark) + r"\%" + mark + (sp + tail if sp else "")


def _row(arm: str, runs: dict) -> str:
    """The row label; a random partition, pretrained with a single seed, also names that seed (the writing guide's
    glossary: "seed", never "run k", PI 2026-11-09)."""
    if arm in RAND and len(runs[arm]) == 1:
        return f"{LABEL[arm]}, seed {runs[arm][0].rsplit('-s', 1)[1]}"
    return LABEL[arm]


def _idx(m: str) -> str:
    return m.rsplit("-s", 1)[1]


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


def _rej(rec, k: str):
    """A rejection from a record: the value, a Bound when no background jet passed, None if absent."""
    if rec is None or rec.get(k) is None:
        return None
    return V4.rejection(rec[k], rec.get(f"{k}_is_bound"), rec.get(f"{k}_n_bkg_pass"))


def _probe_rec(fits: dict, model: str, ds: str, n: str | None = None):
    """A probe cell (all its metrics) at size n, default the largest that model has."""
    cells = fits.get(model, {}).get("cells", {}).get(_ck(model), {}).get(ds, {})
    if not cells:
        return None
    return cells.get(n or max(cells, key=int))


def _largest(fits: dict, ds: str) -> str:
    sizes = {n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}
    return max(sizes, key=int)


def _oma(x):
    """1 - AUC, keeping the mark of a probe that did not converge."""
    return None if x is None else Unconverged(1 - x) if isinstance(x, Unconverged) else 1 - x


# ------------------------------------------------------------------ ratios and their ratio words
# noun -> (more, less, form): "pct" gives "17\% more errors" between 0.5 and 2, "times" always
# "0.58 times the resolution"
PROSE_NOUN = {"errors": ("more errors", "fewer errors", "pct"),
              "rejection": ("more rejection", "less rejection", "pct"),
              "resolution": (None, None, "times"), "distance": (None, None, "times")}


def _sig2(x: float) -> str:
    """Two significant figures, trailing zeros kept: 17.1 -> 17, 0.596 -> 0.60, 127 -> 130."""
    if x == 0:
        return "0"
    v = round(x, 1 - math.floor(math.log10(abs(x))))
    d = max(1 - math.floor(math.log10(abs(v))), 0)
    return _sep(f"{v:.{d}f}")


def _phrase(r: float, noun: str) -> str:
    more, less, form = PROSE_NOUN[noun]
    if form == "times" or round(r, 2) >= 2.0 or r < 0.5:
        return f"{_sig2(r)} times the {noun}"
    if r > 1:
        return f"{_sig2(100 * (r - 1))}\\% {more}"
    if r < 1:
        return f"{_sig2(100 * (1 - r))}\\% {less}"
    return f"the same {noun}"


def _bounded(r: float, noun: str, nearer: bool) -> str:
    """One of two runs in words: the value nearer 1 as the weakest effect ('at least ...', or for a
    reduction written as a factor 'at most ...'), the farther one as the strongest ('up to ...')."""
    ph = _phrase(r, noun)
    if r < 1 and " times the " in ph:
        return ("at most " if nearer else "down to ") + ph
    return ("at least " if nearer else "up to ") + ph


def prose(vals: list, noun: str, farther: bool = False) -> str:
    """A ratio in words, its direction and value: three or more runs, the mean; two runs, the value
    nearer 1 (farther=True: the other); one run, its value. Whether the text states it is the 3-sigma
    test's (ratio_words; PI 2026-11-23, LEDGER SP-12 step 3: the 2-sigma spread rule is gone)."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) == 2:
        a, b = sorted(v, key=lambda x: abs(math.log(x)) if x > 0 else math.inf)
        return _bounded(b if farther else a, noun, not farther)
    return _phrase(float(np.mean(v)), noun)


def _pct_prose(vals: list) -> str:
    """A per-cent value in prose: two significant figures of the mean; two runs, both."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) == 2:
        return f"{_sig2(min(v))}--{_sig2(max(v))}\\%"
    return f"{_sig2(float(np.mean(v)))}\\%"


def _held(em) -> dict:
    """The ratio words an emitter has held for ratio_words(): {<name>Pr or <name>PrMax: its record}."""
    return vars(em).setdefault("ratio_words", {})


def _hold(em, name: str, src, path: str, vals=None, noun=None, farther=False, body=None) -> None:
    """A ratio word is never emitted where its ratio is: held, its values or a fixed body (an undefined
    ratio, a missing reference), until ratio_words() reads its tab:tests row."""
    _held(em)[name] = {"vals": vals, "noun": noun, "farther": farther, "body": body, "src": src, "path": path}


def _held_word(h: dict) -> str:
    """A held ratio word as it reads when its row passes: the fixed body, or the ratio in words (prose)."""
    return h["body"] if h["body"] is not None else prose(h["vals"], h["noun"], h["farther"])


def _emit_ratio(em, name: str, vals: list, n_expected: int, fmt_pm, src, path: str, noun: str,
                two_run: bool = False, body: str | None = None) -> None:
    """A ratio macro and its ratio word, held for ratio_words(); for a two-run configuration also the
    word of the run farther from 1 (<name>PrMax)."""
    em.macro(name, body if body is not None else _cell(vals, n_expected, fmt_pm, sep=TEXT_SEP), src, path)
    _hold(em, name + "Pr", src, path, vals, noun)
    if two_run:
        _hold(em, name + "PrMax", src, path, vals, noun, farther=True)


def _undefined(em, name: str, src, path: str, two_run: bool = False) -> None:
    """A ratio of 1 - AUC whose reference separates the test jets completely (1 - AUC = 0): a dash,
    and ratio words that say so."""
    em.macro(name, "--", src, path, "the reference's 1 - AUC is 0")
    for t in ("Pr", "PrMax") if two_run else ("Pr",):
        _hold(em, name + t, src, path, body="undefined, since the reference separates the test jets completely")


def _emit_pct(em, name: str, vals: list, n_expected: int, fmt_pm, src, path: str) -> None:
    em.macro(name, _pct_cell(vals, n_expected, fmt_pm), src, path)


def _ratio_vals(runs, arm, ref, value) -> tuple[list, int]:
    """value(arm) / value(ref) run by run over the run indices both configurations have."""
    a = {_idx(m): value(m) for m in runs[arm]}
    b = {_idx(m): value(m) for m in runs[ref]}
    both = sorted(set(a) & set(b))
    r = []
    for k in both:
        if a[k] is None or b[k] is None or b[k] == 0:
            r.append(None)
            continue
        x = a[k] / b[k]
        if isinstance(a[k], Unconverged) or isinstance(b[k], Unconverged):
            x = Unconverged(x)
        r.append(x)
    return r, len(both)


def _ratio(runs, arm, ref, value, fmt_pm, sep=TEXT_SEP) -> str:
    r, n = _ratio_vals(runs, arm, ref, value)
    return _cell(r, n, fmt_pm, sep=sep)


def _rand_over(runs, ref: str, value) -> list:
    """Each random partition (one run) over the `ref` run of the same run index."""
    base = {_idx(m): value(m) for m in runs[ref]}
    out = []
    for arm in RAND:
        for m in runs[arm]:
            x, y = value(m), base.get(_idx(m))
            out.append(None if x is None or y is None or y == 0 else x / y)
    return out


def _two_run(runs, *arms) -> bool:
    return any(len(runs.get(a, ())) == 2 for a in arms)


def _over_scratch(v: list, n_expected: int, ref: list, fmt_pm) -> str:
    """Each run's value over the mean of the from-scratch seeds. With three or more runs: the mean
    ratio, and as its uncertainty the SD over the runs and the relative standard error of the
    from-scratch mean added in quadrature; fewer runs print as _cell does."""
    sc = float(np.mean(ref))
    r = [None if x is None else x / sc for x in v]
    have = [x for x in r if x is not None]
    if len(have) < 3 or len(ref) < 2:
        return _cell(r, n_expected, fmt_pm, sep=TEXT_SEP)
    m = float(np.mean(have))
    rel = float(np.std(ref, ddof=1)) / math.sqrt(len(ref)) / sc
    body = fmt_pm(m, float(math.hypot(np.std(have, ddof=1), m * rel)))
    return body + (rf" \pending{{{len(have)} of {n_expected}}}" if len(have) < n_expected else "")


def _emit_over_ref(em, name, v, n_expected, ref, fmt_pm, src, path, noun, two_run=False) -> None:
    """A ratio to the mean of a reference (from scratch, a published value): table form with the
    reference's spread, and the ratio words from the per-run ratios."""
    if not ref:
        em.macro(name, r"\pending{reference}", src, path)
        for t in ("Pr", "PrMax") if two_run else ("Pr",):
            _hold(em, name + t, src, path, body=r"\pending{reference}")
        return
    sc = float(np.mean(ref))
    _emit_ratio(em, name, [None if x is None else x / sc for x in v], n_expected, fmt_pm, src, path, noun,
                two_run, body=_over_scratch(v, n_expected, ref, fmt_pm))


# ------------------------------------------------------------------ linear probes (A1, A11, A13)
def merge_status(root: pathlib.Path) -> dict:
    """{arm: {pair: "merged" | "partly" | "apart"}} from the label maps: merged when every class
    of the pair's first side shares a group with every class of its second side."""
    group = _groups(root)
    names = list(group["L188"])
    out = {}
    for arm, g in group.items():
        out[arm] = {}
        for key, _, a, b in PAIRS:
            same = [g[x] == g[y] for x in _side(names, a) for y in _side(names, b)]
            out[arm][key] = "merged" if all(same) else "partly" if any(same) else "apart"
    return out


def _groups(root: pathlib.Path) -> dict:
    """{configuration: {native class name: its group}} of every label map."""
    maps = root / "configs/labelmaps"
    group: dict = collections.defaultdict(dict)
    for f, cols in (("rung_label_maps.v1.csv", MAIN), ("rand_label_map.v2.csv", RAND),
                    ("flavour_pair_map.v2.csv", FLAV)):
        for r in csv.DictReader((maps / f).open()):
            for c in cols:
                group[c][r["class_name"]] = r[c]
    group["L162_MASS"], group["R16_Q1_MASS"] = group["L162"], group["R16_Q1"]
    return dict(group)


def _side(names: list, side) -> tuple:
    """A pair's side as native class names: QCD classes by quark content, or the names given."""
    qcd = [n for n in names if n.startswith("label_QCD_") and n != "label_QCD_light"]
    flav = lambda n: n.removeprefix("label_QCD_")
    if side == "qcd_with_b":
        return tuple(n for n in qcd if "b" in flav(n))
    if side == "qcd_with_c_no_b":
        return tuple(n for n in qcd if "c" in flav(n) and "b" not in flav(n))
    return side


def _ratio_to_full(fits, runs, arm, value) -> tuple[list, int]:
    """A configuration's value over the 188-class run of the same index (a random partition: its one
    run over the 188-class run of its seed); the randomly initialized networks, the released Sophon checkpoint
    and the network fine-tuned from scratch have no such ratio."""
    if arm in (INIT, INIT_BN, SOPHON, SCRATCH):
        return [], 0
    if arm == "L188":
        return [1.0 if value(m) is not None else None for m in runs[arm]], len(runs[arm])
    if arm == RAND_POOLED:      # the five random partitions as one row: each over the 188-class run of its seed
        r = _rand_over(runs, "L188", value)
        return r, len(r)
    return _ratio_vals(runs, arm, "L188", value)


def full_count_classes(data: pathlib.Path, per_class: int) -> set | None:
    """Native classes with at least per_class selected test jets (census.json), or None."""
    f = data / "jc2bal" / "census.json"
    if not f.exists():
        return None
    return {i for i, c in enumerate(json.loads(f.read_text())["counts"]["ft_test"]) if c >= per_class}


def _auc_over(rec, classes: set) -> float | None:
    if rec is None or classes is None:
        return None
    v = [a for c, a in rec["per_class_auc"].items() if int(c) in classes]
    return float(np.mean(v)) if v else None


def table_probes(fits, runs, fmt_pm, rows, full_count=None, fill=None, chosen=None, kin=False, rho=True) -> str:
    """A1, linear probes at the largest training set: JetClass-II accuracy and AUC averaged over all
    classes, the classes at the full test count, the resonant and the QCD classes, with the ratio of
    1 - AUC to the 188-class run as a labelled column; JetClass accuracy and AUC; top, quark/gluon
    and Herwig: accuracy, AUC, R50 and R30 under the substitution chosen on validation (chosen, from
    chosen_substitutions; zero, marked in the heading, while it is pending). kin=True adds the
    kinematic-only model to the lower part, top only (nothing missing, one run). SOPHON in rows: the released
    checkpoint last in both parts, below a rule (its missing inputs zero, _bench_rec). rho=False (the paper,
    PI 2026-11-12): no ratio column."""
    fill, chosen = fill or {}, chosen or {}
    n2 = _largest(fits, "jc2_188")
    k2 = 6 if rho else 5
    out = [r"\begin{tabular}{l" + "c" * (k2 + 2) + "}", r"\toprule",
           rf" & \multicolumn{{{k2}}}{{c}}{{JetClass-II}} & \multicolumn{{2}}{{c}}{{JetClass}} \\",
           rf"\cmidrule(lr){{2-{k2 + 1}}}\cmidrule(lr){{{k2 + 2}-{k2 + 3}}}",
           r" & accuracy & AUC & " + (r"$\rho$ & " if rho else "") + r"AUC, full count & AUC, resonant & AUC, QCD & "
           r"accuracy & AUC \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        r2 = [_probe_rec(fits, m, "jc2_188", n2) for m in ms]
        r1 = [_probe_rec(fits, m, "jc1") for m in ms]
        ratio, nr = _ratio_to_full(fits, runs, arm, lambda m: _oma(_probe(fits, m, "jc2_188", n2, "macro_auc_ovr")))
        cells = [_cell([None if r is None else r["accuracy"] for r in r2], n, fmt_pm, frac=True),
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_ovr") for m in ms], n, fmt_pm, frac=True),
                 *([_cell(ratio, nr, fmt_pm) if nr else "--"] if rho else []),   # beside the AUC it is the ratio of
                 _cell([_auc_over(r, full_count) for r in r2], n, fmt_pm, frac=True) if full_count else "--",
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_resonant") for m in ms], n, fmt_pm, frac=True),
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_qcd") for m in ms], n, fmt_pm, frac=True),
                 _cell([None if r is None else r["accuracy"] for r in r1], n, fmt_pm, frac=True),
                 _cell([None if r is None else r["macro_auc_ovr"] for r in r1], n, fmt_pm, frac=True)]
        out += [r"\midrule"] * (arm == SOPHON) + [_row(arm, runs) + " & " + " & ".join(cells) + r" \\"]
    out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%", r"\begin{tabular}{l" + "c" * 12 + "}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{h}}}" for h in ("top", "quark/gluon", "quark/gluon, Herwig"))
            + r" \\",
            " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{_subst_head(chosen, ds)}}}" for ds in ("top", "qg", "qg_herwig"))
            + r" \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
            " & " + " & ".join(h for _, h in BENCH4 * 3) + r" \\", r"\midrule"]
    def lower(arm):
        ms, n = runs[arm], len(runs[arm])
        cells = []
        for ds in ("top", "qg", "qg_herwig"):
            recs = [_bench_rec(fits, fill, chosen, m, ds) for m in ms]
            for k, _ in BENCH4:
                cells.append(_cell([_rej(r, k) if k.startswith("r") else (None if r is None else r[k]) for r in recs],
                                   n, fmt_pm, frac=not k.startswith("r")))
        return _row(arm, runs) + " & " + " & ".join(cells) + r" \\"
    out += [lower(arm) for arm in rows if arm not in (INIT, INIT_BN, SOPHON)]
    out += [lower(arm) for arm in rows if arm in (INIT, INIT_BN)]
    rec = _probe_rec(fits, KIN_MODEL, "top") if kin else None
    if rec is not None:         # after the references of L5, outside the network list (plan "tables")
        out.append(LABEL["KIN"] + " & " + " & ".join([_cell([_rej(rec, k) if k.startswith("r") else rec[k]], 1, fmt_pm, frac=not k.startswith("r"))
                                                  for k, _ in BENCH4] + ["--"] * 2 * len(BENCH4)) + r" \\")
    if SOPHON in rows:
        out += [r"\midrule", lower(SOPHON)]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


PROBES_N_FULL = (("jc2_188", "accuracy", "JetClass-II, accuracy"), ("jc2_188", "macro_auc_ovr", "JetClass-II, AUC"),
                 ("jc1", "accuracy", "JetClass, accuracy"), ("jc1", "macro_auc_ovr", "JetClass, AUC"),
                 ("top", "r30", r"top, $R_{30}$"), ("top", "auc", "top, AUC"),
                 ("qg", "r30", r"quark/gluon, $R_{30}$"), ("qg", "auc", "quark/gluon, AUC"),
                 ("qg_herwig", "r30", r"quark/gluon, Herwig, $R_{30}$"), ("qg_herwig", "auc", "quark/gluon, Herwig, AUC"))


def table_probes_n(fits, runs, fmt_pm, rows, fill=None, chosen=None, kin=False, label="tab:probesnfull") -> str:
    """tab:probesnfull (supplement), linear probes against the training-set size, every size, in the metrics of the
    training-size figures and their AUC versions (PROBES_N_FULL; top and quark/gluon under the substitution chosen on
    validation, as table_probes); kin=True adds the kinematic-only model last wherever it has fits (top). One
    longtable, a block per task and metric with its own heading of sizes, so that every configuration fits and the
    table breaks across pages."""
    fill, chosen = fill or {}, chosen or {}
    blocks = []
    for ds, met, title in PROBES_N_FULL:
        bench = ds in SUBST_KEY
        sizes = (_bench_sizes(fits, fill, chosen, ds) if bench else
                 sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int))
        if not sizes:
            continue
        if bench:
            title += f" ({_subst_head(chosen, ds)})"
        lines = [["training jets"] + [f"{int(n):,}".replace(",", "{,}") for n in sizes]]
        for arm in rows:
            vals = []
            for n in sizes:
                recs = [_bench_rec(fits, fill, chosen, m, ds, n) if bench else _probe_rec(fits, m, ds, n)
                        for m in runs[arm]]
                vals.append(_cell([_rej(r, met) if met == "r30" else (None if r is None else r[met]) for r in recs],
                                  len(runs[arm]), fmt_pm, frac=met != "r30"))
            lines.append([_row(arm, runs)] + vals)
        recs = [_probe_rec(fits, KIN_MODEL, ds, n) for n in sizes] if kin else []
        if any(recs):
            lines.append([LABEL["KIN"]] + ["--" if r is None else _cell([_rej(r, met) if met == "r30" else r[met]], 1, fmt_pm, frac=met != "r30")
                                      for r in recs])
        blocks.append((title, lines))
    return _task_longtable(blocks, label)


# a longtable across the whole line, its caption as wide and at the size of every other table's caption (a longtable
# sets its caption at the body's size, here \\small)
LT_SETUP = (r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\LTcapwidth}{\linewidth}"
            r"\makeatletter\def\LT@makecaption#1#2#3{\LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth{"
            r"\normalsize #1{#2. }#3\par\vskip\belowcaptionskip}\hss}}}\makeatother")


# the rule between two task blocks of a longtable (_task_longtable)
BLOCK_SEP = r"\midrule[\heavyrulewidth]"


def _head_rule(ncols: int) -> str:
    """A rule under a block's heading as booktabs draws a midrule in a longtable (a row of leaders across
    the columns), with no page break on either side: booktabs' midrule leaves a legal break at the glue below
    its rule, so a heading could end a page without its rows."""
    return (r"\noalign{\vskip\aboverulesep\nobreak}\multispan{" + str(ncols)
            + r"}{\leaders\hrule height\lightrulewidth\hfill}\cr\noalign{\nobreak\vskip\belowrulesep\nobreak}")




BLOCK_WHOLE = 20    # rows of a task block kept on one turned page (it holds about 26 lines at the body size)


def _task_longtable(blocks: list, label: str) -> str:
    """One longtable of task blocks [(title, [heading cells, row cells, ...])]: each block opens with
    its title across the table and its own heading row; a row may have fewer cells than the widest
    block (the rest stay empty). Set in landscape across the whole line at the body size, the row labels on
    one line and the columns spaced by the line's spare width (PI 2026-11-11: no wrapped label, no numbers run
    together, nothing below normal type); continuation pages say so. The caption is the placeholder __CAPTION__
    (_longtable)."""
    ncol = max((len(c) - 1 for _, rows in blocks for c in rows), default=1)
    row = lambda cells: " & ".join(cells) + r" \\"
    body = []
    for k, (title, (head, *rest)) in enumerate(blocks):
        # \\* keeps a block's title and heading on the page of its first row
        rows = [row(c) for c in rest]
        if len(rows) <= BLOCK_WHOLE:    # a block that fits a turned page never breaks: every page shows its heading
            rows[:-1] = [r + "*" for r in rows[:-1]]
        elif len(rows) >= 2:
            rows[-2] += "*"         # a block's last row never sits alone on a page
        body += ([BLOCK_SEP] if k else []) + [r"\multicolumn{" + str(ncol + 1) + "}{@{}l}{" + title + r"} \\*",
                                              row(head) + "*", _head_rule(ncol + 1)] + rows
    return "\n".join([r"\begin{landscape}", "{", LT_SETUP,
                      r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * ncol + "@{}}",
                      rf"\caption{{__CAPTION__}}\label{{{label}}}\\", r"\toprule", r"\endfirsthead",
                      r"\multicolumn{" + str(ncol + 1) + r"}{@{}l}{\textit{Table~\ref{" + label + r"} (continued)}} \\", r"\toprule", r"\endhead",
                      r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", "}", r"\end{landscape}", ""])


def _pair(fits, m, key):
    c = fits.get(m, {}).get("cells", {}).get(_ck(m), {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
    return None if c is None else c["auc"]


def separated(fits, runs, key, get=None) -> set:
    """Run indices whose 188-class probe (or get(model, pair)'s AUC) separates the pair's test jets (almost)
    completely, 1 - AUC below SEP_FLOOR: a ratio of 1 - AUC to that run would divide by a handful of misordered pairs
    (X->ee vs mumu gave ratios of 10^6 +- 10^7), so none is printed."""
    get = get or (lambda m, k: _pair(fits, m, k))
    return {_idx(m) for m in runs["L188"] if (a := get(m, key)) is not None and a > 1 - SEP_FLOOR}


POOL_MARK = r"$^{\parallel}$"     # the pooled random partitions: some of the five merge the pair, some keep it apart
NEAR_ONE = r"$>0.9999$"            # an AUC that rounds to 1.0000 (PI 2026-11-12): no digits beyond, no spread
SEP_FLOOR = 1e-4    # below this 1 - AUC (AUC 1.0000 at four decimals) no ratio to the run is printed
# the fine-tuned networks on the class pairs (PI 2026-11-08): experiments/FT/roc_points.py ft --family pairs_ft, one doc
# per model (scratch-s<k> from scratch) with "curves": {pair: {rejection, n_bkg_pass, n_background, ...}} on the doc's
# eps_grid; the JetClass-II fine-tuning at 1e6 scored with g_a / (g_a + g_b) on the pair's jets among the first 2,000,000
# TEST2M jets, natural composition (the jets of the head sculpting, scripts/build_head_jobs.py --ft-head, and for the one
# rule's cells --ft-head-fixed: mlst_v4.load_family reads its rerun pairs_ft_fixed once pulled)
PAIRS_FT = "curves_v2/pairs_ft"
PAIRS_LABEL_CM = 4.6        # the pair table's row labels wrap at this width and its blocks may be shrunk by
                            # SHRINK_OK, so that two pairs fit a block
FT_SCRATCH_MODELS = ("scratch-s1", "scratch-s2", "scratch-s3")
# tab:pairs and tab:pairsfull (PI 2026-11-18: no 1 - AUC anywhere; PI 2026-11-19: at 90% signal efficiency, where
# every pair but X->ee vs mumu is measured after fine-tuning; at 50% four of seven were lower limits for every network):
# the rejection 1/eps_B at PAIR_EPS of each run, read off the pair curves of both modes (the point eps_S = PAIR_EPS of
# the curve's grid): the linear probes' (load_pairs_lp: curves_v2 pairs, the v5 refit's pairs_v5 once pulled) and the
# fine-tuned networks' (load_pairs_ft). The names say R50 for history. Lower-bound
# rule: a run in which fewer than PAIR_MIN_PASS background jets pass at that efficiency (where every curve of the
# class-pair figures stops) has no measured R50; it gives the lower limit n_background / PAIR_MIN_PASS. A cell with such
# a run prints PAIR_GT and the smallest, over its runs, of the measured values and these limits (every run's R50 lies
# above it), with no spread; its column then has no bold (_bold).
PAIR_EPS = 0.9
PAIR_MIN_PASS = V4.JC1_MIN_BKG
PAIR_GT = r"$>$"


def pair_r50(doc: dict, key: str):
    """(R50, background jets passing at PAIR_EPS, background jets) of a pair curve doc's curve key, or None."""
    c = doc.get("curves", {}).get(key)
    if c is None:
        return None
    grid = np.asarray(doc["eps_grid"], float)
    i = int(np.argmin(np.abs(grid - PAIR_EPS)))
    if abs(grid[i] - PAIR_EPS) > 1e-9:
        raise SystemExit(f"FATAL: the pair curves' grid has no point at signal efficiency {PAIR_EPS}")
    return c["rejection"][i], int(c["n_bkg_pass"][i]), int(c["n_background"])


def _pair_r50(fits, m, key):
    """A linear probe's pair_r50 triple from its fit record, or None."""
    c = fits.get(m, {}).get("cells", {}).get(_ck(m), {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
    return None if c is None else (c["r50"], int(c["r50_n_bkg_pass"]), int(c["n_test_background"]))


def pair_cell(vals: list, n_expected: int, fmt_pm) -> str:
    """One cell of the pair tables from the runs' pair_r50 triples (None where a run has none): the lower-bound rule
    above where a run has fewer than PAIR_MIN_PASS background jets passing, else _cell's mean +- SD of R50."""
    have = [v for v in vals if v is not None]
    if not any(k < PAIR_MIN_PASS for _r, k, _n in have):
        return _cell([r for r, _k, _n in have], n_expected, fmt_pm)
    low = min(n / PAIR_MIN_PASS if k < PAIR_MIN_PASS else r for r, k, n in have)
    return (PAIR_GT + _tab1(low) + (SINGLE if len(have) == 1 else "")
            + (rf" \pending{{{len(have)} of {n_expected}}}" if len(have) < n_expected else ""))


def load_pairs_lp(data: pathlib.Path) -> dict:
    """{model: {pair: pair_r50 triple}} of the linear probes' pair curves (mlst_v4.load_family "pairs": the v5 refit's
    pairs_v5, every pair, once pulled)."""
    return {m: {k: pair_r50(d, k) for k in d["curves"]} for m, d in V4.load_family(data, "pairs").items()}


def load_pairs_ft(data: pathlib.Path) -> dict:
    """{model: {pair: pair_r50 triple}} of the fine-tuned pair curves (mlst_v4.load_family: the one rule's rerun
    once pulled), {} until pulled."""
    return {m: {k: pair_r50(d, k) for k in d["curves"]} for m, d in V4.load_family(data, "pairs_ft").items()}


def table_pairs(fits, runs, fmt_pm, status, rows, ft=None, lp=None) -> str:
    """A11, pairs of JetClass-II classes: R50 of each pair (the lower-bound rule above), under one heading over the
    pairs, in the class-pair figures' panel order; a dagger marks a pair the label set merges. The linear probes
    (binary probes, fits) first; ft (load_pairs_ft; {} prints the group pending) adds the fine-tuned networks as a
    second row group: the mode in each group's heading and not in the row labels, a row fine-tuned from scratch (no
    probe) and none for the random networks (no fine-tuning). The pooled random partitions (RAND_POOLED) are marked
    where some but not all of the five merge the pair (POOL_MARK)."""
    mark = {"merged": r"$^{\dagger}$", "partly": r"$^{\ddagger}$", "apart": "", "some": POOL_MARK}
    if RAND_POOLED in rows:
        def pooled(key):
            st = [status.get(a, {}).get(key, "apart") for a in RAND]
            return "merged" if all(x == "merged" for x in st) else "apart" if all(x == "apart" for x in st) else "some"
        status = {**status, RAND_POOLED: {key: pooled(key) for key, *_ in PAIRS}}
    # the probes' values off their pair curves at PAIR_EPS (lp, load_pairs_lp); the fit files store only R50 and R30
    modes = [("lp", tuple(rows), (lambda m, key: lp.get(m, {}).get(key)) if lp is not None
              else (lambda m, key: _pair_r50(fits, m, key)))]
    if ft is not None:          # until pulled ({}), one pending row in place of the group's rows
        modes.append(("ft", tuple(r for r in rows if r not in (INIT, INIT_BN)) + (SCRATCH,) if ft else (),
                      lambda m, key: ft.get(m, {}).get(key)))
        runs = {**runs, SCRATCH: list(FT_SCRATCH_MODELS)}
    # PI 2026-11-19 (B, out): the X->ee vs mumu column leaves the table, one caption sentence (EE_NOTE) in its place,
    # checked here: every pretrained network whose label set keeps the pair apart separates it completely
    for mode, arms, get in modes:
        for arm in arms:
            if arm in (INIT, INIT_BN, SCRATCH) or status.get(arm, {}).get("ee_vs_mm", "apart") != "apart":
                continue
            c = pair_cell([get(m, "ee_vs_mm") for m in runs[arm]], len(runs[arm]), fmt_pm)
            if c.strip() not in ("--", "") and not c.startswith(PAIR_GT) and r"\pending" not in c:
                raise SystemExit(f"FATAL: tab:pairs EE_NOTE: {arm} ({mode}) keeps X->ee vs mumu apart but separates it "
                                 f"incompletely ({c}); the column must come back")
    cols = PAIRS_TABLE
    n = len(cols)
    out = [r"\begin{tabular}{l" + "c" * n + "}", r"\toprule",
           rf" & \multicolumn{{{n}}}{{c}}{{$R_{{{round(100 * PAIR_EPS)}}}$}} \\", rf"\cmidrule(lr){{2-{1 + n}}}",
           " & " + " & ".join(h for _, h, *_ in cols) + r" \\", r"\midrule"]
    for k, (mode, arms, get) in enumerate(modes):
        if len(modes) > 1:
            out += ([r"\midrule"] if k else []) + [_full_row(1 + n, HEADING[mode], italic=True)]
        if not arms:
            out.append(_full_row(1 + n, r"\pending{no run yet}"))
        for arm in arms:
            # a merge is the label set's, so its mark stands in both modes
            cells = [pair_cell([get(m, key) for m in runs[arm]], len(runs[arm]), fmt_pm)
                     + mark[status.get(arm, {}).get(key, "apart")] for key, *_ in cols]
            out.append((SCRATCH_LABEL if arm == SCRATCH else _row(arm, runs)) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


# PI 2026-11-23 (LEDGER DL-R #4, CS-6): the main text quotes ratios of the linear probes' errors on these pairs (the
# ratio words of PR_ROWS on _BB, _BC, _BBQQ), so their AUCs are printed, the sides the ratios are formed from
# fourth review (B6): X->ee vs mumu joins them, the text quoting its AUCs at 43 and 188 classes and the random network
PAIR_AUC_COLS = tuple(p for p in PAIRS_FIG if p[0] in ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq", "ee_vs_mm"))


def table_pair_auc(fits, runs, fmt_pm, status, rows) -> str:
    """tab:pairauc: the AUC of the binary linear probes on the pairs whose error ratios the main text quotes, mean +- SD
    over the runs (_pair, the values the ratio words and tab:tests read), in the class-pair figures' panel order, the
    merge marks of tab:pairs."""
    mark = {"merged": r"$^{\dagger}$", "partly": r"$^{\ddagger}$", "apart": "", "some": POOL_MARK}
    if RAND_POOLED in rows:
        def pooled(key):
            st = [status.get(a, {}).get(key, "apart") for a in RAND]
            return "merged" if all(x == "merged" for x in st) else "apart" if all(x == "apart" for x in st) else "some"
        status = {**status, RAND_POOLED: {key: pooled(key) for key, *_ in PAIRS}}
    n = len(PAIR_AUC_COLS)
    # each pair's name on two lines (signal, then "vs" the background), so that the table stays upright (FG-25)
    halves = [h.split(" vs ", 1) for _, h, *_ in PAIR_AUC_COLS]
    out = [r"\begin{tabular}{l" + "c" * n + "}", r"\toprule", rf" & \multicolumn{{{n}}}{{c}}{{AUC}} \\",
           rf"\cmidrule(lr){{2-{1 + n}}}", " & " + " & ".join(a for a, _ in halves) + r" \\",
           " & " + " & ".join("vs " + b for _, b in halves) + r" \\", r"\midrule"]
    def auc(arm, key):
        # an AUC below one that rounds to one at four decimals prints as NEAR_ONE (LEDGER TN-2), as the text quotes it
        # (fourth review, B6: the X->ee vs mumu column)
        v = [x for m in runs[arm] if (x := _pair(fits, m, key)) is not None]
        if v and 0 < float(np.mean(v)) < 1 and round(float(np.mean(v)), 4) >= 1 and len(v) == len(runs[arm]):
            return NEAR_ONE
        return _cell([_pair(fits, m, key) for m in runs[arm]], len(runs[arm]), fmt_pm)

    for arm in rows:
        cells = [auc(arm, key) + mark[status.get(arm, {}).get(key, "apart")] for key, *_ in PAIR_AUC_COLS]
        lab = r"\quad " + INIT_BN_SUB if arm == INIT_BN else _row(arm, runs)      # under the random network, as Table 2
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


# What the text compares on JetClass-II, as (configuration, reference): the ratio of their 1 - AUC,
# run by run between runs of the same index -- they share the training stream and the initial
# trunk -- as mean +- SD over the indices both have. Only between pretraining configurations.
CMP_PROBES = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L162"),
              ("R16_Q1_MASS", "R16_Q1"), ("L162_MASS", "L188"), ("R16_Q1_MASS", "L188"),
              ("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"), ("FLAV_F0", "R16_Q1"),
              ("FLAV_F1", "R16_Q1"), ("FLAV_F1R", "R16_Q1"), ("FLAV_F1", "FLAV_F0"), ("FLAV_F1R", "FLAV_F0"),
              ("FLAV_F1R", "FLAV_F1"), ("R42_Q1", "L162"), ("R16_Q1", "L162"), ("R16_Q1", "R42_Q1"))
CMP_PAIRS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("FLAV_F1", "FLAV_F0"),
             ("FLAV_F1R", "FLAV_F1"), ("FLAV_F1R", "FLAV_F0"), ("FLAV_F0", "R16_Q1"), ("FLAV_F1", "R16_Q1"),
             ("FLAV_F1R", "R16_Q1"), ("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"))


def comparisons(em, fits, runs, status, index, fmt_pm) -> None:
    """The probe ratios the results text quotes, on JetClass-II only: MlstCmp<task or pair><arm>Over<ref>,
    run by run; MlstCmp<...>RandOver{Full,Mid,Coarse}, each random partition over the run of its
    seed, mean +- SD over the partitions; MlstPairRandMergedOverApart<pair>, the mean 1 - AUC of the
    partitions that merge the pair over that of those that keep it apart; each with its ratio word."""
    for ds, met, _ in JC2_TASKS:
        n = _largest(fits, ds)
        w = TASK_WORD[(ds, met)]
        f = lambda m, ds=ds, n=n, met=met: _oma(_probe(fits, m, ds, n, met))
        for arm, ref in CMP_PROBES:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstCmp{w}{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                        f"linear probe {ds} N={n}: (1 - {met}) of {arm} over {ref}, run by run", "errors",
                        _two_run(runs, arm, ref))
        for ref in ("L188", "R42_Q1", "R16_Q1"):
            r = _rand_over(runs, ref, f)
            _emit_ratio(em, f"MlstCmp{w}RandOver{WORD[ref]}", r, len(r), fmt_pm, index,
                        f"linear probe {ds} N={n}: (1 - {met}) of each random partition over the {ref} run "
                        "of its seed, mean +- SD over the partitions", "errors")
    for key, _, _, _ in PAIRS:
        f = lambda m, key=key: _oma(_pair(fits, m, key))
        # as tab:pairs: no ratio to a run whose 1 - AUC is below SEP_FLOOR (separated)
        zero = lambda ref, arms: ({_idx(m) for m in runs[ref] if f(m) is not None and f(m) < SEP_FLOOR}
                                  & {_idx(m) for a in arms for m in runs[a]})
        for arm, ref in CMP_PAIRS:
            name, path = (f"MlstCmp{PAIR_WORD[key]}{WORD[arm]}Over{WORD[ref]}",
                          f"binary probe {key}: (1 - AUC) of {arm} over {ref}, run by run")
            if zero(ref, (arm,)):
                _undefined(em, name, index, path, _two_run(runs, arm, ref))
                continue
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, name, r, k, fmt_pm, index, path, "errors", _two_run(runs, arm, ref))
        if key.startswith("qcd_"):
            continue            # every partition merges the QCD classes
        name, path = (f"MlstCmp{PAIR_WORD[key]}RandOverFull",
                      f"binary probe {key}: (1 - AUC) of each random partition over the 188-class run of its seed")
        if zero("L188", RAND):
            _undefined(em, name, index, path)
        else:
            r = _rand_over(runs, "L188", f)
            _emit_ratio(em, name, r, len(r), fmt_pm, index, path, "errors")
        side = {s: [f(m) for arm in RAND if status[arm][key] == s for m in runs[arm]] for s in ("merged", "apart")}
        for s, v in side.items():
            em.macro(f"MlstPairRand{s.title()}{PAIR_WORD[key]}N", str(len(v)), index,
                     f"random partitions that {'merge' if s == 'merged' else 'keep apart'} {key} (count)")
        mv, av = ([x for x in side[s] if x is not None] for s in ("merged", "apart"))
        if av and float(np.mean(av)) == 0:
            _undefined(em, f"MlstPairRandMergedOverApart{PAIR_WORD[key]}", index,
                       f"binary probe {key}: the partitions that keep it apart separate the test jets completely")
            continue
        val = [float(np.mean(mv)) / float(np.mean(av))] if mv and av else []
        _emit_ratio(em, f"MlstPairRandMergedOverApart{PAIR_WORD[key]}", val, 1, fmt_pm, index,
                    f"binary probe {key}: mean (1 - AUC) of the partitions that merge it over that of those "
                    "that keep it apart (one run each)", "errors",
                    body=_lone(val[0]) if val else r"\pending{partitions}")


def probe_abs_macros(em, fits, runs, index, fmt_pm) -> None:
    """Absolute JetClass-II values the captions anchor the ratios with: accuracy and AUC averaged over
    the classes of each configuration and of the randomly initialized networks; and the JetClass
    accuracy of the randomly initialized network with BatchNorm recomputed (F2(b), A10(b))."""
    n = _largest(fits, "jc2_188")
    for arm in MAIN + MASS + FLAV + (INIT, INIT_BN):
        ms = runs[arm]
        em.macro(f"MlstLpJcTwoAuc{WORD[arm]}", _cell([_probe(fits, m, "jc2_188", n, "macro_auc_ovr") for m in ms],
                                                     len(ms), fmt_pm, sep=TEXT_SEP), index,
                 f"linear probe JetClass-II N={n}: AUC averaged over the classes, {arm}")
        em.macro(f"MlstLpJcTwoAcc{WORD[arm]}", _cell([None if (r := _probe_rec(fits, m, "jc2_188", n)) is None
                                                      else r["accuracy"] for m in ms], len(ms), fmt_pm, sep=TEXT_SEP),
                 index, f"linear probe JetClass-II N={n}: accuracy, {arm}")
    v = [_probe(fits, m, "jc2_188", n, "macro_auc_ovr") for a in RAND for m in runs[a]]
    em.macro("MlstLpJcTwoAucRand", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
             f"linear probe JetClass-II N={n}: AUC averaged over the classes, the random partitions")
    n1, ms = _largest(fits, "jc1"), runs[INIT_BN]
    em.macro("MlstLpJcOneAccInitBn", _cell([None if (r := _probe_rec(fits, m, "jc1", n1)) is None else r["accuracy"]
                                            for m in ms], len(ms), fmt_pm, sep=TEXT_SEP),
             index, f"linear probe JetClass N={n1}: accuracy, {INIT_BN}")


def bn_macros(em, fits, runs, index) -> None:
    """MlstBnMaxDev: the largest |r_bn / r - 1|, in per cent, of a JetClass-II label-set ratio of
    1 - AUC (each pretrained configuration over the 188-class run of its index) with the BatchNorm
    statistics recomputed (best70_bn) against the checkpoint as trained (best70). Natural-composition
    probes only: the class-balanced fits score best70 alone. MlstBnMaxDevPair: the same on the class
    pairs that have a best70_bn fit (X->ee vs X->mumu has none: fits_v3 does not hold it and fits_v4_sib
    scores best70 alone), for the levels of the hierarchy, the mass models and the controlled partitions;
    a random partition has one run, and a regularization chosen on few validation jets can move with the statistics."""
    dev = []

    def val(m, ck, ds, n, met):
        c = fits.get(m, {}).get("cells", {}).get(ck, {}).get(ds, {}).get(n)
        if c is None:
            raise SystemExit(f"FATAL: {m} {ds} N={n}: no fit at {ck}")
        return 1 - c[met]
    for ds, met, _ in JC2_TASKS:
        n = _largest(fits, ds)
        for arm, ms in runs.items():
            if arm in (INIT, INIT_BN, "L188") or "L188" not in runs:
                continue
            for m in ms:
                ref = next((x for x in runs["L188"] if _idx(x) == _idx(m)), None)
                if ref is None:
                    continue
                r = val(m, PRIMARY, ds, n, met) / val(ref, PRIMARY, ds, n, met)
                rb = val(m, "best70_bn", ds, n, met) / val(ref, "best70_bn", ds, n, met)
                dev.append(abs(rb / r - 1))
    em.macro("MlstBnMaxDev", f"{100 * max(dev):.1f}\\%" if dev else r"\pending{probes}", index,
             "largest |r(best70_bn) / r(best70) - 1| of a JetClass-II ratio of 1 - AUC to the 188-class run, per cent")

    def oma(m, ck, key):
        c = fits.get(m, {}).get("cells", {}).get(ck, {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
        return None if c is None else 1 - c["auc"]
    pair, checked = [], set()
    for key, *_ in PAIRS:
        for arm in MAIN[1:] + MASS + FLAV:
            for m in runs.get(arm, ()):
                ref = next((x for x in runs.get("L188", ()) if _idx(x) == _idx(m)), None)
                v = [oma(x, ck, key) for ck in (PRIMARY, "best70_bn") for x in (m, ref)]
                if all(v):              # a fit missing, or one separates the test jets completely
                    pair.append(abs((v[2] / v[3]) / (v[0] / v[1]) - 1))
                    checked.add(key)
    em.macro("MlstBnMaxDevPair", f"{100 * max(pair):.1f}\\%" if pair else r"\pending{probes}", index,
             "largest |r(best70_bn) / r(best70) - 1| of a ratio of 1 - AUC on a class pair to the 188-class run, "
             "levels of the hierarchy, mass models and controlled partitions, per cent",
             "pairs with a check: " + ", ".join(k for k, *_ in PAIRS if k in checked)
             + "; without: " + (", ".join(k for k, *_ in PAIRS if k not in checked) or "none"))


def table_bn(fits3, runs, fmt_pm, rows) -> str:
    """A6: JetClass-II (natural composition, largest training set) as trained and with the BatchNorm
    statistics recomputed: for each, the AUC and beside it the ratio of 1 - AUC to the 188-class run."""
    n = _largest(fits3, "jc2_188")
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{BatchNorm as trained} & \multicolumn{2}{c}{BatchNorm recomputed} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r" & AUC & $\rho$ & AUC & $\rho$ \\", r"\midrule"]

    def auc(m, ck):
        c = fits3.get(m, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(n)
        return None if c is None else c["macro_auc_ovr"]
    for arm in rows:
        ms = runs[arm]
        cells = []
        for ck in (PRIMARY, "best70_bn"):
            r, k = _ratio_to_full(fits3, runs, arm, lambda m: _oma(auc(m, ck)))
            cells += [_cell([auc(m, ck) for m in ms], len(ms), fmt_pm, frac=True), _cell(r, k, fmt_pm) if k else "--"]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    a = [_cell([auc(m, "init") for m in INIT_MODELS], 3, fmt_pm, frac=True), "--",
         _cell([auc(m, "init_bn") for m in INIT_BN_MODELS], 3, fmt_pm, frac=True), "--"]
    out.append(LABEL[INIT] + " & " + " & ".join(a) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def table_balance(fits3, fits4, runs, fmt_pm, rows) -> str:
    """A13: JetClass-II at 10^6 training jets, natural composition (fits_v3) beside class-balanced
    (fits_v4): AUC averaged over the classes and the ratio of 1 - AUC to the 188-class run."""
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{mixed test sample} & \multicolumn{2}{c}{class-balanced} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r" & AUC & $\rho$ & AUC & "
           r"$\rho$ \\", r"\midrule"]
    for arm in rows:
        cells = []
        for f in (fits3, fits4):
            if not f:
                cells += [r"\pending{fits}"] * 2
                continue
            v = lambda m, f=f: _probe(f, m, "jc2_188", PAIR_N, "macro_auc_ovr")
            cells.append(_cell([v(m) for m in runs[arm]], len(runs[arm]), fmt_pm, frac=True))
            r, k = _ratio_to_full(f, runs, arm, lambda m, v=v: _oma(v(m)))
            cells.append(_cell(r, k, fmt_pm) if k else "--")
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ nonlinear probes (A5)
def load_mlp_natural(data: pathlib.Path) -> dict:
    """The nonlinear probes as fitted on natural-composition JetClass-II, with every task."""
    return {**_load(data / MLP_FITS), **_load(data / MLP_FITS_V4)}


def load_mlp(data: pathlib.Path) -> dict:
    """The nonlinear probes: top, quark/gluon, Herwig and JetClass from fits_v3_mlp2 (their tasks do
    not depend on the JetClass-II composition), from fits_v4_mlp only where fits_v3_mlp2 has no such
    cell (the kinematic-only model); JetClass-II and its pairs from fits_v4_mlp once
    pulled (class-balanced, as the linear probes), else from fits_v3_mlp2. JetClass-II values of the
    two versions are never mixed; the models fits_v4_mlp lacks keep theirs in load_mlp_natural."""
    mlp = load_mlp_natural(data)
    mlp.update({"initbn" + m.removeprefix("init"): d for m, d in _load(data / MLP_FITS_INITBN).items()})
    # fits_v5_mlp (PI 2026-11-14): every network, the mass-output runs and random partitions among them, on one CPU
    # model, in place of fits_v4_mlp whole (mlst_v4.pick)
    bal = _load(data / V4.pick(data, MLP_BAL))
    if not bal:
        return mlp
    jc2 = set(MLP_JC2)
    for d in mlp.values():
        for ck in d["cells"].values():
            for k in [k for k in ck if k in jc2]:
                del ck[k]
    for m, d in bal.items():
        for ck, cells in d["cells"].items():
            model = ("initbn" + m.removeprefix("init")) if ck == "init_bn" else m
            v = {}
            if "jc2bal" in cells:
                v["jc2_188"] = cells["jc2bal"]
            v.update({k.removeprefix("pair_"): c for k, c in cells.items() if k.startswith("pair_")})
            have = mlp.setdefault(model, {"cells": {}})["cells"].setdefault(ck, {})
            # a benchmark task fits_v3_mlp2 lacks (the kinematic-only model, fitted on top alone)
            v.update({k: c for k, c in cells.items() if k in ("jc1", "top", "qg", "qg_herwig") and k not in have})
            have.update(v)
    return mlp


# the statistics agent's refit of the MLP probes (bs3, 2026-11-23; LEDGER DL-9, CS-5b): one CPU model per family, the
# values its tab:tests rows were bootstrapped from (bootstrap/l3_bs3); pulled to data/v2/bs3_mlp/<family>/<model>.json
BS3_MLP = "bs3_mlp"


def load_bs3_mlp(data: pathlib.Path) -> dict:
    """{family: {file stem: doc}} of the bs3 MLP refit (bench: top and quark/gluon with the linear probes'
    substitution; bench0: with zeros; jc2bal: class-balanced JetClass-II; coarse: the 17 merged groups; kin: the
    kinematic-inputs network on top); {} where not pulled."""
    root = data / BS3_MLP
    return {f.name: {g.stem: json.loads(g.read_text()) for g in sorted(f.glob("*.json"))}
            for f in sorted(root.iterdir()) if f.is_dir()} if root.is_dir() else {}


def _bs3_model(stem: str, ck: str) -> str:
    """A refit file's network as mlst_results names its runs: init-s1.init_bn -> initbn-s1."""
    m = stem.split(".", 1)[0]
    return ("initbn" + m.removeprefix("init")) if ck == "init_bn" else m


def overlay_bs3_mlp(mlp: dict, bs3: dict) -> dict:
    """mlp with the bs3 refit in place of the earlier fits wherever the refit has the cell (CS-5b: Table 10 reads the
    values its tested gaps were computed from): JetClass-II (jc2bal), top and quark/gluon with zeros (bench0), the
    kinematic-inputs network on top (kin)."""
    for fam, rename in (("jc2bal", {"jc2bal": "jc2_188"}), ("bench0", {}), ("kin", {})):
        for stem, doc in bs3.get(fam, {}).items():
            for ck, cells in doc.get("cells", {}).items():
                model = _bs3_model(stem, ck)
                have = mlp.setdefault(model, {"cells": {}})["cells"].setdefault(ck, {})
                have.update({rename.get(k, k): c for k, c in cells.items() if isinstance(c, dict)})
    return mlp


def _bs3_val(bs3: dict, fam: str, model: str, ds: str, key: str):
    """One refit run's value (R30 as _rej reads it), or None."""
    for stem, doc in bs3.get(fam, {}).items():
        for ck, cells in doc.get("cells", {}).items():
            if _bs3_model(stem, ck) == model and ds in cells:
                c = cells[ds]
                return _rej(c, key) if key.startswith("r") else c.get(key)
    return None


def _mlp_val(mlp: dict, model: str, task: str, key: str | None = None):
    met = key or {t: m for t, _, m in MLP_TASKS}[task]
    c = mlp.get(model, {}).get("cells", {}).get(_ck(model), {}).get(task.removesuffix("_res").removesuffix("_qcd"))
    if c is None:
        return None
    return _rej(c, met) if met.startswith("r") else c.get(met)


def _mlp(mlp: dict, model: str, task: str):
    """1 - AUC of the nonlinear probe; jc2_188_res / jc2_188_qcd are parts of the jc2_188 cell."""
    v = _mlp_val(mlp, model, task)
    return None if v is None else 1 - v


def table_mlp(mlp, runs, fmt_pm, rows, raw_v4=None, nat=None, kin=False, rho=True) -> str:
    """A5: the nonlinear probe (one hidden layer, three seeds averaged) at the largest training set:
    JetClass-II AUC (all, resonant, QCD) with the ratio of 1 - AUC to the 188-class run; JetClass AUC;
    top, quark/gluon and Herwig AUC and R30; the class pairs' AUC and ratio; for the configurations with
    no class-balanced fit (their cells above a dash), JetClass-II and the pairs' ratios on natural
    composition (nat, load_mlp_natural) beside the levels of the hierarchy on the same test set, so that no
    ratio crosses the two compositions; and the linear probe on the head's 512-unit hidden layer beside
    the one on the class token. kin=True adds the kinematic-only model to the first block (top only). rho=False: the
    paper's copy, with no rho column and no linear-probe block (PI 2026-11-23, LEDGER CA-15: both in the data release's
    copy, tab:mlpfull)."""
    nat_only = [arm for arm in rows if nat and all(_mlp_val(mlp, m, "jc2_188") is None for m in runs[arm])
                and any(_mlp_val(nat, m, "jc2_188") is not None for m in runs[arm])]
    j = 4 if rho else 3             # the JetClass-II columns
    out = [r"\begin{tabular}{l" + "c" * (j + 7) + "}", r"\toprule",
           rf" & \multicolumn{{{j}}}{{c}}{{JetClass-II}} & JetClass & \multicolumn{{2}}{{c}}{{top}} & "
           r"\multicolumn{2}{c}{quark/gluon, Pythia} & \multicolumn{2}{c}{quark/gluon, Herwig} \\",
           rf"\cmidrule(lr){{2-{j + 1}}}" + "".join(rf"\cmidrule(lr){{{j + k}-{j + k + 1}}}" for k in (3, 5, 7)),
           r" & AUC & " + (r"$\rho$ & " if rho else "") + r"AUC, resonant & AUC, QCD & AUC & AUC & "
           r"$R_{30}$ & AUC & $R_{30}$ & AUC & $R_{30}$ \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        if arm in nat_only:
            cells = ["--"] * j
        else:
            r, k = _ratio_to_full(None, runs, arm, lambda m: _mlp(mlp, m, "jc2_188"))
            cells = [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm, frac=True) for t in ("jc2_188", "jc2_188_res", "jc2_188_qcd")]
            if rho:
                cells.insert(1, _cell(r, k, fmt_pm) if k else "--")       # beside the AUC it is the ratio of
        cells.append(_cell([_mlp_val(mlp, m, "jc1") for m in ms], n, fmt_pm, frac=True))
        for t in ("top", "qg", "qg_herwig"):
            cells += [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm, frac=True),
                      _cell([_mlp_val(mlp, m, t, "r30") for m in ms], n, fmt_pm)]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    if kin and _mlp_val(mlp, KIN_MODEL, "top") is not None:     # top only, nothing missing, one run
        out.append(LABEL["KIN"] + " & " + " & ".join(["--"] * (j + 1) + [_cell([_mlp_val(mlp, KIN_MODEL, "top")], 1, fmt_pm, frac=True),
                                                            _cell([_mlp_val(mlp, KIN_MODEL, "top", "r30")], 1, fmt_pm)]
                                                + ["--"] * 4) + r" \\")
    out += (_hidden_rows(raw_v4, runs, fmt_pm) if rho else []) + [r"\bottomrule", r"\end{tabular}"]
    for part in (MLP_PAIRS[:3], MLP_PAIRS[3:]):
        w = 2 if rho else 1         # AUC, and rho beside it in the release's copy
        out += ["%SPLIT%", r"\begin{tabular}{l" + "c" * w * len(part) + "}", r"\toprule",
                " & " + " & ".join(rf"\multicolumn{{{w}}}{{c}}{{{h}}}" for _, h, _, _ in part) + r" \\",
                "".join(rf"\cmidrule(lr){{{2 + w * i}-{1 + w * (i + 1)}}}" for i in range(len(part))),
                " & " + " & ".join([r"AUC & $\rho$" if rho else "AUC"] * len(part)) + r" \\",
                r"\midrule"]
        for arm in (a for a in rows if a not in nat_only):
            cells = []
            for key, _, _, _ in part:
                cells.append(_cell([_mlp_val(mlp, m, key) for m in runs[arm]], len(runs[arm]), fmt_pm, frac=True))
                if not rho:
                    continue
                r, k = _ratio_to_full(None, runs, arm, lambda m, key=key: _mlp(mlp, m, key))
                cells.append(_cell(r, k, fmt_pm) if k else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    if nat_only and not rho:        # the natural-composition block is ratios, which the paper's copy does not print
        raise SystemExit(f"FATAL: tab:mlp: {nat_only} have no class-balanced MLP probe")
    if nat_only:
        out += ["%SPLIT%", r"\begin{tabular}{l" + "c" * (2 + len(MLP_PAIRS)) + "}", r"\toprule",
                r" & \multicolumn{2}{c}{JetClass-II} & \multicolumn{" + str(len(MLP_PAIRS))
                + r"}{c}{$\rho$ on the class pairs} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-" + str(3 + len(MLP_PAIRS)) + "}",
                r"mixed test sample & AUC & $\rho$ & " + " & ".join(h for _, h, _, _ in MLP_PAIRS) + r" \\", r"\midrule"]
        for arm in (a for a in rows if a in MAIN or a in nat_only):      # the common row order (L5)
            cells = [_cell([_mlp_val(nat, m, "jc2_188") for m in runs[arm]], len(runs[arm]), fmt_pm, frac=True)]
            for t in ("jc2_188", *(k for k, *_ in MLP_PAIRS)):
                r, k = _ratio_to_full(None, runs, arm, lambda m, t=t: _mlp(nat, m, t))
                cells.append(_cell(r, k, fmt_pm) if k else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out + [""])


# PI SP-14 (R2, R4): the paper's Table 10 keeps only the networks and columns that a sentence or a tab:tests row reads;
# every other network and column is in the data release's copy (tab:mlpfull, table_mlp). Every sentence that cites
# tab:mlp states a difference with its multilayer-perceptron tab:tests row (results_jc, results_mass, appendix D), so
# the columns are the tasks and metrics of TESTS' multilayer-perceptron blocks that Table 10 holds and the networks
# their sides (mlp_paper_cells); the accuracy, the 17 merged groups and the jet mass are tab:tests', tab:coarse's and
# tab:mass's. In the order and with the headings of table_mlp: (task, metric or None for AUC, group, column)
MLP_PAPER_ORDER = (("jc2_188", None, "JetClass-II", "AUC"), ("jc2_188_res", None, "JetClass-II", "AUC, resonant"),
                   ("jc2_188_qcd", None, "JetClass-II", "AUC, QCD"), ("jc1", None, "JetClass", "AUC"),
                   *((t, m, g, h) for t, g in (("top", "top"), ("qg", "quark/gluon, Pythia"),
                                               ("qg_herwig", "quark/gluon, Herwig"))
                     for m, h in ((None, "AUC"), ("r30", "$R_{30}$"))),
                   *((k, None, h, "AUC") for k, h, *_ in MLP_PAIRS))
_MLP_HEAD_OF = {"JetClass-II, AUC": ("jc2_188", None), "JetClass-II, AUC over the resonant classes": ("jc2_188_res", None),
                "JetClass-II, AUC over the QCD classes": ("jc2_188_qcd", None), "JetClass, AUC": ("jc1", None),
                **{f"{h}, {mh}": (t, m) for t, h in (("top", "top tagging"), ("qg", "quark/gluon tagging (Pythia)"),
                                                     ("qg_herwig", "quark/gluon tagging (Herwig)"))
                   for m, mh in ((None, "AUC"), ("r30", "$R_{30}$"))},
                **{f"{h}, AUC": (k, None) for k, h, *_ in MLP_PAIRS}}


def mlp_paper_cells(tests=None) -> tuple:
    """(Table 10's columns in the paper, (task, metric) in MLP_PAPER_ORDER; its networks): those of TESTS'
    multilayer-perceptron blocks whose task and metric Table 10 holds, and their two sides (TESTS_ARM)."""
    cols, arms = set(), set()
    for head, rows in TESTS if tests is None else tests:
        if ", " + HEADING["mlp"] not in head:
            continue
        col = _MLP_HEAD_OF.get(head.split(", " + HEADING["mlp"])[0])
        if col is None:
            continue
        cols.add(col)
        arms |= {TESTS_ARM.get(x, x) for a, b, _ in rows for x in (a, b)}
    return [(t, m) for t, m, *_ in MLP_PAPER_ORDER if (t, m) in cols], arms


def table_mlp_paper(mlp, runs, fmt_pm, rows) -> str:
    """tab:mlp as the paper prints it (mlp_paper_cells): the multilayer-perceptron probe, three seeds averaged, at the
    largest training set, on the family the build reads; rows in the order of rows."""
    cols, arms = mlp_paper_cells()
    if not cols:
        raise SystemExit("FATAL (SP-14): tab:mlp: no tab:tests row of the multilayer-perceptron probe reads it")
    if arms - set(rows):
        raise SystemExit(f"FATAL (SP-14): tab:mlp: tab:tests reads networks the table has no row for: {arms - set(rows)}")
    spec = [next(o for o in MLP_PAPER_ORDER if o[:2] == c) for c in cols]
    groups = [(g, sum(1 for _ in it)) for g, it in itertools.groupby(o[2] for o in spec)]
    head, rules, at = [], [], 2
    for g, n in groups:
        head.append(rf"\multicolumn{{{n}}}{{c}}{{{g}}}" if n > 1 else g)
        rules.append(rf"\cmidrule(lr){{{at}-{at + n - 1}}}")
        at += n
    out = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", " & " + " & ".join(head) + r" \\", "".join(rules),
           " & " + " & ".join(o[3] for o in spec) + r" \\", r"\midrule"]
    for arm in (a for a in rows if a in arms):
        ms, n = runs[arm], len(runs[arm])
        cells = [_cell([_mlp_val(mlp, m, t, met) for m in ms], n, fmt_pm, frac=met is None) for t, met in cols]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def _hidden_rows(raw_v4, runs, fmt_pm) -> list:
    """The linear probe on the pretraining head's 512-unit hidden layer beside the one on the class token (188 classes),
    as rows under a rule at the foot of the nonlinear-probe table's first block, in its JetClass-II AUC and quark/gluon
    columns (a dash elsewhere): a block of its own was two rows alone on a page (PI 2026-11-11), under their mode's
    italic heading (D3; page review round 3)."""
    hid = [m for m in runs["L188"] if "jc2bal.hidden" in (raw_v4 or {}).get(m, {}).get("cells", {}).get(PRIMARY, {})]
    if not hid:
        return []

    def cell(m, key, met):
        c = raw_v4[m]["cells"][PRIMARY].get(key, {})
        c = c[max(c, key=int)] if c else None
        return None if c is None else (_rej(c, met) if met.startswith("r") else c[met])
    out = [r"\midrule", _full_row(12, HEADING["lp"], italic=True)]
    for name, sfx in ((LABEL["L188"] + ", class token", ""), (LABEL["L188"] + ", hidden layer of the pretraining head", ".hidden")):
        jc2, auc, r30 = (_cell([cell(m, k + sfx, met) for m in hid], len(runs["L188"]), fmt_pm, frac=met != "r30")
                         for k, met in (("jc2bal", "macro_auc_ovr"), ("qg", "auc"), ("qg", "r30")))
        out.append(name + " & " + " & ".join([jc2] + ["--"] * 6 + [auc, r30] + ["--"] * 2) + r" \\")
    return out


CMP_MLP = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"),
           ("FLAV_F0", "L188"), ("FLAV_F1", "FLAV_F0"), ("FLAV_F1R", "FLAV_F1"), ("FLAV_F1R", "FLAV_F0"),
           ("FLAV_F0", "R16_Q1"))


def mlp_macros(em, mlp, runs, index, fmt_pm) -> None:
    """As comparisons(), for the nonlinear probe on JetClass-II and its pairs (1 - AUC ratios between
    pretraining configurations); on top and quark/gluon the R30 of each label set over the randomly
    initialized network with BatchNorm recomputed (the D6 check beside the linear probe)."""
    unconverged = [(m, t) for m, d in mlp.items() for ck in d["cells"].values() for t, c in ck.items()
                   if isinstance(c, dict) and not c.get("mlp", {}).get("all_converged", True)]
    em.macro("MlstMlpUnconverged", str(len(unconverged)), index, "nonlinear probes that did not reach "
             "their plateau within the epoch cap (count)")
    for t in MLP_JC2:
        f = lambda m, t=t: _mlp(mlp, m, t)
        for arm, ref in CMP_MLP:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstMlpCmp{PAIR_WORD[t]}{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                        f"nonlinear probe {t}: (1 - AUC) of {arm} over {ref}, run by run", "errors", _two_run(runs, arm, ref))
        r = _rand_over(runs, "L188", f)
        _emit_ratio(em, f"MlstMlpCmp{PAIR_WORD[t]}RandOverFull", r, len(r), fmt_pm, index,
                    f"nonlinear probe {t}: (1 - AUC) of each random partition over the 188-class run of its seed", "errors")
    for t in ("top", "qg", "qg_herwig"):
        f = lambda m, t=t: _mlp_val(mlp, m, t, "r30")
        for arm in MAIN:
            r, k = _ratio_vals(runs, arm, INIT_BN, f)
            _emit_ratio(em, f"MlstMlp{PAIR_WORD[t]}Rthirty{WORD[arm]}OverInitBn", r, k, fmt_pm, index,
                        f"nonlinear probe {t}, missing inputs zero: R30 of {arm} over the BatchNorm-recomputed random "
                        "network of the same index", "rejection")


def mlp_natural_macros(em, nat, runs, index, fmt_pm) -> None:
    """MlstMlpNatJcTwo<mass>Over<same labels>: the nonlinear probe's 1 - AUC on natural-composition
    JetClass-II with over without the auxiliary mass regression, run by run; these models have no
    class-balanced nonlinear probe."""
    f = lambda m: _mlp(nat, m, "jc2_188")
    for arm, ref in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")):
        r, k = _ratio_vals(runs, arm, ref, f)
        _emit_ratio(em, f"MlstMlpNatJcTwo{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                    f"nonlinear probe, natural-composition JetClass-II (fits_v3_mlp2): (1 - AUC) of {arm} over {ref}, run by run",
                    "errors", _two_run(runs, arm, ref))


# ------------------------------------------------------------------ missing inputs (A16, F5)
# the three substitutions of the missing inputs (experiments/EVAL/input_variants.py): the cell key of
# each per task in fits_v4_fill (probe_bal.py --fill), and the JetClass-II variant of the same slots
SUBST = ("zero", "mean", "draws")
SUBST_WORD = {"zero": "zero", "mean": "the training mean", "draws": "random draws"}
SUBST_KEY = {"top": {"zero": "top", "mean": "top.ntop", "draws": "top.itop4"},
             "qg": {"zero": "qg", "mean": "qg.nqg", "draws": "qg.iqg4"},
             "qg_herwig": {"zero": "qg_herwig", "mean": "qg_herwig.nqg", "draws": "qg_herwig.iqg4"}}
# JetClass-II with the top (or quark/gluon) data's missing slots substituted: zero from fits_v4
# (jc2bal.mtop, .mqg), mean and draws from fits_v4_fill (jc2bal.ntop, .itop4 / .nqg, .iqg4)
JC2_SUBST = {"top": {"zero": "jc2bal.mtop", "mean": "jc2bal.ntop", "draws": "jc2bal.itop4"},
             "qg": {"zero": "jc2bal.mqg", "mean": "jc2bal.nqg", "draws": "jc2bal.iqg4"}}


def _raw_cells(raw: dict, model: str) -> dict:
    """A v4-layout file's cells at the checkpoint a model is reported at (initbn-sK reads init-sK's
    init_bn cells)."""
    base = "init-" + model.removeprefix("initbn-") if model.startswith("initbn-") else model
    return raw.get(base, {}).get("cells", {}).get(_ck(model), {})


def _top_of(cells: dict, key: str):
    c = cells.get(key) or {}
    return c[max(c, key=int)] if c else None


def choose_substitution(fill: dict, models: list, task: str) -> tuple:
    """(substitution, {substitution: summed validation cross-entropy}) of a task: the substitution
    whose probes, at the largest training set and each at its selected C, have the lowest validation
    cross-entropy summed over every model (PI 2026-11-02: one per task, applied to every model,
    the randomly initialized network included). (None, {}) until every model has all three."""
    tot = {}
    for s in SUBST:
        v = []
        for m in models:
            c = _top_of(_raw_cells(fill, m), SUBST_KEY[task][s])
            if c is None:
                return None, {}
            v.append(c["search"][str(c["C"])]["val_cross_entropy"])
        tot[s] = float(np.sum(v))
    return min(tot, key=tot.get), tot


def chosen_substitutions(fill: dict, runs: dict) -> tuple[dict, dict]:
    """({task: substitution or None}, {task: summed validation cross-entropies}) for top and
    quark/gluon over every pretrained run and BatchNorm-recomputed random network; the Herwig jets
    take the quark/gluon choice (the same probe, scored on other jets)."""
    models = _pretrained(runs) + list(INIT_BN_MODELS)
    chosen, tots = {}, {}
    for task in ("top", "qg"):
        chosen[task], tots[task] = choose_substitution(fill, models, task) if fill else (None, {})
    chosen["qg_herwig"] = chosen["qg"]
    return chosen, tots


def with_sophon_fill(data: pathlib.Path, fill: dict) -> dict:
    """fill with the released Sophon checkpoint's substitution probes merged in (v5: its top and quark/gluon probes
    with the substitutions the pretrained rows use, the same-fill refit of 18 Nov), as the tables and figures read
    them; unchanged before v5."""
    if V4.pick(data, FITS_SOPHON) == FITS_SOPHON:
        return fill
    sfill = _load(data / SOPHON_FILL_V5).get(SOPHON_MODEL)
    if sfill is None:
        raise SystemExit(f"FATAL: {V4.V5[FITS_SOPHON]} without {SOPHON_FILL_V5}: pull them together")
    return {**fill, SOPHON_MODEL: sfill}


def _bench_rec(fits: dict, fill: dict, chosen: dict, m: str, ds: str, n: str | None = None):
    """A top or quark/gluon probe cell at size n (default the largest) under the substitution chosen
    on validation (fits_v4_fill), or with the missing inputs zero (fits) while the choice is pending.
    The kinematic-only model has nothing missing and is read from fits; the released Sophon checkpoint only with the
    substitution probes of its same-fill refit (with_sophon_fill), never zero-filled (PI 2026-11-22): None without."""
    s = chosen.get(ds)
    if m == SOPHON_MODEL and m not in fill:
        return None
    if s is None or m == KIN_MODEL:
        return _probe_rec(fits, m, ds, n)
    c = _raw_cells(fill, m).get(SUBST_KEY[ds][s]) or {}
    return c.get(n or max(c, key=int)) if c else None


def _bench_sizes(fits: dict, fill: dict, chosen: dict, ds: str) -> list:
    s = chosen.get(ds)
    if s is None:
        return sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
    return sorted({n for f in fill.values() for ck in f["cells"].values() for n in ck.get(SUBST_KEY[ds][s], {})},
                  key=int)


def _subst_head(chosen: dict, ds: str) -> str:
    """The substitution a top or quark/gluon column group uses, for its heading."""
    s = chosen.get(ds)
    return f"missing inputs set to {SUBST_WORD[s]}" if s else r"missing inputs set to zero \pending{choice on validation}"


def _acc_drop(cells: dict, variant: str, base: str = "jc2bal"):
    """Relative drop in accuracy, per cent, of a JetClass-II probe with the slots substituted
    against the same model's probe with every input, at the largest training set."""
    a, b = _top_of(cells, base), _top_of(cells, variant)
    return None if a is None or b is None else 100 * (a["accuracy"] - b["accuracy"]) / a["accuracy"]


def _jc2_drop(raw_v4, fill, m, task, s):
    if s == "zero":
        return _acc_drop(_raw_cells(raw_v4, m), JC2_SUBST[task][s])
    return _acc_drop(_raw_cells(fill, m), JC2_SUBST[task][s])


def _fill_r30(fill, m, task, s):
    return _rej(_top_of(_raw_cells(fill, m), SUBST_KEY[task][s]), "r30")


def table_fill(raw_v4, fill, fits, runs, fmt_pm, rows, drop: bool = True, chosen: dict | None = None) -> str:
    """A16: every configuration with the missing inputs substituted by zero, the training mean or
    random draws. JetClass-II (largest training set): the relative drop in accuracy with the slots the
    top and the quark/gluon data lack substituted. Top and quark/gluon (largest training set): R30 of
    the linear probe under each substitution; the kinematic-only model has nothing missing. No bold (PI
    2026-11-12). drop=False (the paper): the top and quark/gluon block only, without the JetClass-II drops. chosen
    ({task: substitution}, the paper): that substitution's column is left out, as Table 1 prints it (a number has one
    table); the kinematic-only row stays (PI 2026-11-23, LEDGER SP-9: tab:fill_main folded in)."""
    subs = {t: tuple(s for s in SUBST if s != (chosen or {}).get(t)) for t in ("top", "qg")}
    k1, k2 = len(subs["top"]), len(subs["qg"])
    out = [] if not drop else [r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
           r" & \multicolumn{3}{c}{JetClass-II, top slots} & \multicolumn{3}{c}{JetClass-II, quark/gluon slots} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", " & " + " & ".join(SUBST * 2) + r" \\", r"\midrule"]
    for arm in rows if drop else ():
        ms = runs[arm]
        cells = []
        for t in ("top", "qg"):
            cells += [_pct_cell([_jc2_drop(raw_v4, fill, m, t, s) for m in ms], len(ms), fmt_pm) for s in SUBST]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    out += ([r"\bottomrule", r"\end{tabular}", "%SPLIT%"] if drop else []) + [r"\begin{tabular}{l" + "c" * (k1 + k2) + "}", r"\toprule",
            rf" & \multicolumn{{{k1}}}{{c}}{{top, $R_{{30}}$}} & \multicolumn{{{k2}}}{{c}}{{quark/gluon (Pythia), $R_{{30}}$}} \\",
            rf"\cmidrule(lr){{2-{1 + k1}}}\cmidrule(lr){{{2 + k1}-{1 + k1 + k2}}}",
            " & " + " & ".join(subs["top"] + subs["qg"]) + r" \\", r"\midrule"]
    kin = _rej(_probe_rec(fits, KIN_MODEL, "top"), "r30")
    # PI 2026-11-23 (LEDGER SP-9): tab:fill takes in tab:fill_main's rows, the kinematic-only model's among them
    kin_row = (LABEL["KIN"] + rf" & \multicolumn{{{k1}}}{{c}}{{" + (_lone(kin) + SINGLE if kin is not None else r"\pending{run}")
               + r" (nothing missing)} & " + " & ".join(["--"] * k2) + r" \\")
    for arm in rows:
        ms = runs[arm]
        cells = []
        for t in ("top", "qg"):
            cells += [_cell([_fill_r30(fill, m, t, s) for m in ms], len(ms), fmt_pm) for s in subs[t]]
        # PI 2026-11-14: "random network" is the network as initialized everywhere; the BatchNorm-recomputed one keeps
        # its own name (LABEL[INIT_BN]), here too
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    if kin_row:         # after the references of L5, outside the network list (plan "tables")
        out.append(kin_row)
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def fill_macros(em, raw_v4, fill, fits, runs, index, fmt_pm, missing: list) -> None:
    """The substitution chosen per task (LpSubstitutionTop, LpSubstitutionQg); the linear probes' R30
    on top and quark/gluon over the BatchNorm-recomputed random network (under the chosen
    substitution, and with zero: ...Zero), between label sets, and between substitutions; the
    relative drop in JetClass-II accuracy with the missing slots substituted (F5)."""
    chosen, tots = chosen_substitutions(fill, runs)
    for task, w in (("top", "Top"), ("qg", "Qg")):
        s = chosen[task]
        if s is None:
            missing.append(f"substitution fits for {task} -- experiments/FIGS/data/v2/{FITS_FILL}/ (every model)")
        em.macro(f"LpSubstitution{w}", SUBST_WORD[s] if s else r"\pending{fill fits}", index,
                 f"{FITS_FILL}: lowest summed validation cross-entropy over every model, largest training set",
                 "; ".join(f"{k} {v:.6g}" for k, v in tots[task].items()))
    for task in ("top", "qg", "qg_herwig"):
        tw = PAIR_WORD[task]
        s = chosen[task]
        sel = lambda m, task=task, s=s: _fill_r30(fill, m, task, s) if s else None
        zero = lambda m, task=task: _rej(_probe_rec(fits, m, task), "r30")
        for arm in MAIN + MASS:
            for f, sfx in ((sel, ""), (zero, "Zero")):
                r, k = _ratio_vals(runs, arm, INIT_BN, f)
                _emit_ratio(em, f"MlstLp{tw}Rthirty{WORD[arm]}OverInitBn{sfx}", r, k, fmt_pm, index,
                            f"linear probe {task}, largest set, {'chosen substitution' if not sfx else 'zero'}: R30 of {arm} "
                            "over the BatchNorm-recomputed random network of the same index", "rejection",
                            _two_run(runs, arm))
        for arm in ("L162", "R42_Q1", "R16_Q1"):
            for f, sfx in ((sel, ""), (zero, "Zero")):
                r, k = _ratio_vals(runs, arm, "L188", f)
                _emit_ratio(em, f"MlstLp{tw}Rthirty{WORD[arm]}OverFull{sfx}", r, k, fmt_pm, index,
                            f"linear probe {task}: R30 of {arm} over the 188-class run", "rejection")
        for arm in ("L188", INIT_BN):
            for s2 in ("mean", "draws"):
                r = [None if (a := _fill_r30(fill, m, task, s2)) is None or (b := _fill_r30(fill, m, task, "zero")) is None
                     else a / b for m in runs[arm]]
                _emit_ratio(em, f"MlstLp{tw}Rthirty{s2.title().replace('Draws', 'Draw')}OverZero{WORD[arm]}", r,
                            len(r), fmt_pm, index, f"linear probe {task}: R30 with {s2} over R30 with zero, {arm}",
                            "rejection")
        if task == "top":
            kin = _rej(_probe_rec(fits, KIN_MODEL, "top"), "r30")
            full = [x for x in (zero(m) for m in runs["L188"]) if x is not None]
            val = [kin / float(np.mean(full))] if kin is not None and full else []
            _emit_ratio(em, "MlstLpTopRthirtyKinOverFull", val, 1, fmt_pm, index,
                        "linear probe top: R30 of the kinematic-only model over the mean of the 188-class runs (zero)",
                        "rejection", body=_lone(val[0]) if val else r"\pending{kinematic-only probe}")
            em.macro("MlstLpTopRthirtyKin", _lone(kin) if kin is not None else r"\pending{kinematic-only probe}",
                     index, "linear probe top, largest set: R30 of the kinematic-only model (one run)")
        for arm in ("L188", INIT_BN):
            for s2 in SUBST:
                ms = runs[arm]
                em.macro(f"MlstLp{tw}Rthirty{WORD[arm]}{s2.title()}", _cell([_fill_r30(fill, m, task, s2) for m in ms],
                                                                            len(ms), fmt_pm, sep=TEXT_SEP), index,
                         f"linear probe {task}, largest set: R30 with {s2}, {arm}")
    for task, w in (("top", "Top"), ("qg", "Qg")):
        for arm in MAIN + MASS + FLAV + (INIT_BN,):
            for s in SUBST:
                ms = runs[arm]
                _emit_pct(em, f"MlstLpJcTwo{w}{s.title().replace('Draws', 'Draw')}AccDrop{WORD[arm]}",
                          [_jc2_drop(raw_v4, fill, m, task, s) for m in ms], len(ms), fmt_pm, index,
                          f"linear probe JetClass-II, largest set: relative drop in accuracy (%) with the {task} "
                          f"data's missing slots set to {s}, {arm}")


# ------------------------------------------------------------------ jet mass (T5, A17, F8a)
# ATLAS PUB-2024-015's resolution, half the 15.9-84.1 % inter-quantile range of the response over its
# median, with the median response (head/atlas_res/<model>.json, BRIEF interface 2026-11-02).
# (group in the file, macro suffix, heading): all jets unsuffixed -- the numbers the text and T5 quote
# (FIGURE_PLAN T5 "overall") -- then resonant and QCD jets (A17).
MASS_GROUPS = (("all", "", "all"), ("resonant", "Resonant", "resonant"), ("qcd", "Qcd", "QCD"))
MASS_ROWS = SETS + (INIT,)


def _ares(atlas: dict, model: str, est: str, group: str, key: str = "resolution", axis: str = "inclusive"):
    try:
        return atlas[model]["estimators"][est][group][axis][key]
    except (KeyError, TypeError):
        return None


GROOMED_TOL = 1e-9   # relative: quantiles and bootstraps rebuilt on another node differ in the last digits


def _groomed(atlas: dict, group: str, key: str, axis: str = "inclusive"):
    """The groomed mass's value: a property of the test jets, the same in every file up to rounding
    (GROOMED_TOL), or the build stops. The value of the first file in name order."""
    vals = [v for m in sorted(atlas) if (v := _ares(atlas, m, "groomed", group, key, axis)) is not None]
    if not vals:
        return None
    ref = np.array(vals[0], dtype=float)
    for v in vals[1:]:
        a = np.array(v, dtype=float)
        if a.shape != ref.shape or not np.allclose(a, ref, rtol=GROOMED_TOL, atol=0, equal_nan=True):
            raise SystemExit(f"FATAL: the groomed mass's {group} {key} ({axis}) differs between atlas_res files")
    return vals[0]


def table_mass(atlas, runs, fmt_pm) -> str:
    """T5: ATLAS's resolution and the median response over all test jets of the groomed (soft-drop) mass, of the
    mass output of the pretraining head of the two networks pretrained with a mass regression (the quantity of the
    binned mass figure, estimator "regressed"), and of an MLP regression on the frozen features of every network
    (FIGURE_PLAN T5: the overall value; resonant and QCD jets separately in A17). The two blocks carry their own
    headings (PI 2026-11-12)."""
    g = MASS_GROUPS[0][0]
    out = [r"\begin{tabular}{lcc}", r"\toprule", r" & resolution & median response \\"]
    for est, arms in (("regressed", MASS), ("mlp", MASS_ROWS)):
        out += [r"\midrule", _full_row(3, HEADING["head" if est == "regressed" else "mlp"], italic=True)]
        for arm in arms:
            ms = runs[arm]
            out.append(LABEL[arm] + " & " + " & ".join(_cell([_ares(atlas, m, est, g, k) for m in ms], len(ms), fmt_pm)
                                                       for k in ("resolution", "median")) + r" \\")
    out += [r"\midrule", "soft-drop mass & " + " & ".join(_groomed_cell(atlas, g, k, fmt_pm)      # a reference, last (L5)
                                                      for k in ("resolution", "median")) + r" \\"]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def _groomed_cell(atlas, g: str, k: str, fmt_pm, axis: str = "inclusive") -> str:
    """The soft-drop mass's value with its SD over the test-jet bootstrap (atlas_resolution: <key>_sd), PI 2026-11-21:
    every comparison with it is a claim, so it carries its own uncertainty."""
    v, sd = _groomed(atlas, g, k, axis), _groomed(atlas, g, f"{k}_sd", axis)
    if v is None:
        return "--"
    if sd is not None and V4.tokens(fmt_pm):
        return V4.tok("p", float(v), float(sd))
    return _cell([v], 1, fmt_pm, run=False)


def _binned_one(atlas, axis: str) -> list | None:
    """The bin edges of an axis, one list across every file and estimator, or the build stops."""
    e = {json.dumps(g[axis]["edges"]) for d in atlas.values() for est in d["estimators"].values()
         for g in est.values() if axis in g}
    if len(e) > 1:
        raise SystemExit(f"FATAL: the atlas_res files bin {axis} differently")
    return json.loads(e.pop()) if e else None


RIDGE = "ridge regression on the frozen features"
# the binned blocks of tab:massdetail (PI 2026-11-13, plan "tables"): what the binned mass figures draw, F9b and A17-res,
# A17-qcd (the mass output, by jet group) and A17-probe, A17-ridge (the probes, all jets); (estimator, jet group, rows)
MASS_DETAIL_BINNED = (("regressed", "all", MASS), ("regressed", "resonant", MASS), ("regressed", "qcd", MASS),
                      ("mlp", "all", MASS_ROWS), ("ridge", "all", MASS_ROWS))
# the 17-class partitions' probes (PI 2026-11-14, step 7; atlas_res_v3 from refit5-atlas-part), indented under PARTITIONS
# between the label sets and the random network (L5), the random partitions pooled, wherever the files hold them
MASS_PARTS = FLAV + (RAND_POOLED,)


def _mass_arms(atlas, runs, est: str, arms: tuple) -> list:
    """[(arm, indented)] of a probe block (est mlp or ridge): arms, with the partitions before the random network where
    the files hold their fits; the mass output's blocks (regressed) unchanged."""
    if est == "regressed" or not any(_ares(atlas, m, est, "all") is not None for a in MASS_PARTS
                                     for m in _pooled(runs).get(a, [])):
        return [(a, False) for a in arms]
    head = [a for a in arms if a != INIT]
    return [(a, False) for a in head] + [(PARTITIONS, True)] + [(a, True) for a in MASS_PARTS] \
        + [(a, False) for a in arms if a == INIT]


def mass_has_parts(atlas, runs) -> bool:
    return any(_ares(atlas, m, "mlp", "all") is not None for a in MASS_PARTS for m in _pooled(runs).get(a, []))


def table_mass_detail(atlas, runs, fmt_pm) -> str:
    """tab:massdetail (supplement): the values of the binned mass figures and the inclusive values Table 5 (tab:mass)
    does not hold. Upper part: over resonant and QCD jets, the soft-drop mass, the mass output of the pretraining head
    and the multilayer-perceptron probe; the ridge regression over all three jet groups. Lower parts: resolution and
    median response in the bins of jet pT and generator mass that the figures draw (atlas: atlas_res_fine, ten pT
    bins), per MASS_DETAIL_BINNED block, the soft-drop mass once over each jet group."""
    sec = lambda t, k: r"\multicolumn{" + str(k) + r"}{@{}l}{\textit{" + t + r"}} \\"   # noqa: E731
    group = {g: h for g, _, h in MASS_GROUPS}
    runs = _pooled(runs)
    # the upper part one tabular (page review 2026-11-13: the ridge rows as a second tabular were left alone on a
    # page): over all, resonant and QCD jets, a cell over all jets empty where Table 5 holds it (all but the ridge)
    groups, ng = ("all", "resonant", "qcd"), 3
    out = [r"\begin{tabular}{l" + "c" * 2 * ng + "}", r"\toprule",
           rf" & \multicolumn{{{ng}}}{{c}}{{resolution}} & \multicolumn{{{ng}}}{{c}}{{median response}} \\",
           rf"\cmidrule(lr){{2-{1 + ng}}}\cmidrule(lr){{{2 + ng}-{1 + 2 * ng}}}",
           "jets & " + " & ".join(group[g] for _ in range(2) for g in groups) + r" \\", r"\midrule"]
    # the soft-drop row with its SD (2026-11-21) a token like the network rows, so it is in the first part's
    # uniform_columns: printed at that part's precision per column, never left as a token
    sd_row = "soft-drop mass & " + " & ".join("" if g == "all" else _groomed_cell(atlas, g, k, fmt_pm)
                                              for k in ("resolution", "median") for g in groups) + r" \\"
    # each of the two former tabulars at its own precision per column, as they printed apart
    for i, ests in enumerate(((("regressed", HEADING["head"], MASS), ("mlp", HEADING["mlp"], MASS_ROWS)),
                              (("ridge", RIDGE, MASS_ROWS),))):
        part = [] if i else [sd_row]
        for est, name, arms in ests:
            # in \itshape, not \textit: _long starts a page at a second mode heading, and these are not modes
            part.append(r"\multicolumn{" + str(1 + 2 * ng) + r"}{@{}l}{{\itshape " + name + r"}} \\")
            for arm, ind in _mass_arms(atlas, runs, est, arms):
                if arm == PARTITIONS:
                    part.append(_full_row(1 + 2 * ng, PARTITIONS))
                    continue
                ms = runs.get(arm, [])
                part.append((r"\quad " if ind else "") + f"{LABEL[arm]} & " + " & ".join(
                    "" if g == "all" and est != "ridge" else _cell([_ares(atlas, m, est, g, k) for m in ms], len(ms), fmt_pm)
                    for k in ("resolution", "median") for g in groups) + r" \\")
        out += uniform_columns("\n".join(part)).split("\n")
    out += [r"\bottomrule", r"\end{tabular}"]
    for axis, title in (("vs_pt", r"jet $p_{\mathrm{T}}$ [GeV]"), ("vs_mgen", r"generator-level soft-drop mass [GeV]")):
        edges = _binned_one(atlas, axis)
        if not edges:
            continue
        bins = [f"{a:g}--{b:g}" for a, b in zip(edges, edges[1:])]
        for key, kname in (("resolution", "resolution"), ("median", "median response")):
            head = [r"\begin{tabular}{l" + "c" * len(bins) + "}", r"\toprule",
                    rf"\textit{{{kname}}} & \multicolumn{{{len(bins)}}}{{c}}{{{title}}} \\",
                    rf"\cmidrule(lr){{2-{1 + len(bins)}}}",
                    " & " + " & ".join(bins) + r" \\", r"\midrule"]
            out += ["%SPLIT%", *head]
            for k_est, (est, g, arms) in enumerate(MASS_DETAIL_BINNED):
                name = {"regressed": HEADING["head"], "mlp": HEADING["mlp"], "ridge": RIDGE}[est]
                if est == "ridge" and k_est:    # the ridge rows a block of their own, so no page holds a stub of them
                    out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%", *head]
                if est == "regressed":          # the soft-drop mass once a jet group, above the network outputs
                    gv = _groomed(atlas, g, key, axis) or []
                    gs = _groomed(atlas, g, f"{key}_sd", axis) or [None] * len(gv)
                    out.append(f"soft-drop mass, {group[g]} jets & " + " & ".join(
                        "--" if x is None else V4.tok("p", float(x), float(e)) if e is not None and V4.tokens(fmt_pm)
                        else f"{x:.4f}" for x, e in zip(gv, gs)) + r" \\")
                # italic, as every table's section headings; _long breaks a page only at a mode heading (HEADING)
                out.append(sec(f"{name}, {group[g]} jets", 1 + len(bins)))
                for arm, ind in _mass_arms(atlas, runs, est, arms):
                    if arm == PARTITIONS:
                        out.append(_full_row(1 + len(bins), PARTITIONS))
                        continue
                    ms = runs.get(arm, [])
                    per = [_ares(atlas, m, est, g, key, axis) for m in ms]
                    out.append((r"\quad " if ind else "") + f"{LABEL[arm]} & " + " & ".join(
                        _cell([None if p is None else p[i] for p in per], len(ms), fmt_pm) for i in range(len(bins))) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out + [""])


CMP_MASS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L162"),
            ("R16_Q1_MASS", "R16_Q1"))
# the frozen-feature MLP resolution of each mass network against its own label set, all jets, in the form of decision
# L3 (sigma^2 = each side's SD^2 over its runs + the paired test-jet bootstrap SD^2; experiments/EVAL/mass_sd_boot.py)
MASS_SD_BOOT = "head/mass_sd_boot_v3.json"
# tab:tests reads the v4 copy: v3's differences unchanged, with every network against the soft-drop mass and the
# differences the tested-differences table added (PI 2026-11-23, LEDGER SP-13; scripts/build_boot_tests_jobs.py)
# v5 (review of 5c40d69, B3): v4's differences, with every pretrained network against the random network and every bin
MASS_SD_BOOT_TESTS = "head/mass_sd_boot_v5.json"
MASS_BOOT_DIFFS = (("ResMassOverRes", "162 classes + mass - 162 classes"),
                   ("CoarseMassOverCoarse", "17 classes + mass - 17 classes"))


def mass_boot_macros(em, data: pathlib.Path) -> None:
    """MlstMassMlp<diff>Z: |delta| / sigma of the frozen-feature MLP resolution over all jets, each mass network minus
    its own label set (MASS_SD_BOOT). Section 4.6 states each as a narrower resolution, so the build stops if sigma is
    not the L3 sum or the difference is not narrower by more than 3 sigma (PI 2026-11-22: a rule is checked where the
    data is read). REQUIRED prints them pending until the file is pulled."""
    f = data / MASS_SD_BOOT
    if not f.exists():
        return
    diffs = json.loads(f.read_text())["diffs"]
    for w, key in MASS_BOOT_DIFFS:
        x = diffs[key]["mlp"]["all|inclusive"]["resolution"]
        l3 = math.sqrt(x["sd_a"] ** 2 + x["sd_b"] ** 2 + x["boot_sd"] ** 2)
        if not math.isclose(x["sigma"], l3, rel_tol=1e-9) or not (x["pass"] and x["better"] and x["delta"] < -3 * l3):
            raise SystemExit(f"FATAL: {f.name} {key}: the MLP resolution is not narrower by more than 3 sigma (L3), "
                             "which section 4.6 states")
        em.macro(f"MlstMassMlp{w}Z", f"{abs(x['z']):.1f}", f, f"diffs[{key}].mlp[all|inclusive].resolution.z",
                 "|delta| / sigma, sigma^2 = sd_a^2 + sd_b^2 + boot_sd^2 (decision L3)")


def mass_macros(em, atlas, runs, index, fmt_pm) -> None:
    """The mass numbers of the text, unsuffixed for all test jets (the outline's numbers), ...Resonant
    for resonant and ...Qcd for QCD jets: the groomed mass (MlstMassUncorrected...), the regression
    output of the mass models (MlstMassHead<arm>...: its resolution, median response, its resolution
    over the groomed mass's and over the frozen-feature MLP of the same label set without the
    regression, run by run), and the frozen-feature MLP (MlstMassMlp<arm>..., MlstCmpMlp<arm>Over<ref>...)."""
    for g, gs, _ in MASS_GROUPS:
        unc = _groomed(atlas, g, "resolution")
        em.macro(f"MlstMassUncorrected{gs}", _lone(unc) if unc is not None else r"\pending{atlas_res}", index,
                 f"head/atlas_res: groomed mass, {g} jets, ATLAS resolution")
        med = _groomed(atlas, g, "median")
        em.macro(f"MlstMassUncorrectedResponse{gs}", _lone(med) if med is not None else r"\pending{atlas_res}", index,
                 f"head/atlas_res: groomed mass, {g} jets, median response")
        for arm in MASS:
            ms = runs[arm]
            w = WORD[arm]
            reg = lambda m: _ares(atlas, m, "regressed", g)
            em.macro(f"MlstMassHead{w}{gs}", _cell([reg(m) for m in ms], len(ms), fmt_pm, sep=TEXT_SEP), index,
                     f"regression output of {arm}, {g} jets, ATLAS resolution")
            em.macro(f"MlstMassHead{w}Response{gs}", _cell([_ares(atlas, m, "regressed", g, "median") for m in ms],
                                                           len(ms), fmt_pm, sep=TEXT_SEP), index,
                     f"regression output of {arm}, {g} jets, median response")
            r = [None if (x := reg(m)) is None or not unc else x / unc for m in ms]
            _emit_ratio(em, f"MlstMassHead{w}OverUncorr{gs}", r, len(ms), fmt_pm, index,
                        f"resolution of the regression output of {arm} over the groomed mass's, {g} jets", "resolution", True)
            base = arm.removesuffix("_MASS")
            r, k = _ratio_vals(runs, arm, base, lambda m: reg(m) if m in runs[arm] else _ares(atlas, m, "mlp", g))
            _emit_ratio(em, f"MlstMassHead{w}OverMlp{WORD[base]}{gs}", r, k, fmt_pm, index,
                        f"resolution of the regression output of {arm} over the frozen-feature MLP of {base}, run by "
                        f"run, {g} jets", "resolution", True)
        for arm in MASS_ROWS:
            ms = runs[arm]
            em.macro(f"MlstMassMlp{WORD[arm]}{gs}", _cell([_ares(atlas, m, "mlp", g) for m in ms], len(ms), fmt_pm,
                                                         sep=TEXT_SEP), index,
                     f"frozen-feature MLP of {arm}, {g} jets, ATLAS resolution")
        f = lambda m, g=g: _ares(atlas, m, "mlp", g)
        for arm, ref in CMP_MASS:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstCmpMlp{WORD[arm]}Over{WORD[ref]}{gs}", r, k, fmt_pm, index,
                        f"frozen-feature MLP resolution of {arm} over {ref}, run by run, {g} jets", "resolution",
                        _two_run(runs, arm, ref))
    # F9a(b)'s x axis (figstyle/gallery_physics._response) covers the bins where the groomed mass or a mass
    # model's mean reaches SHOWN_FROM of its own largest bin: the share of QCD jets' groomed-mass response
    # inside that range (jets outside the histogram's edges count in the total, as in the figure).
    h = [atlas[m]["estimators"]["groomed"]["qcd"]["hist"] for m in sorted(atlas)
         if "hist" in atlas[m]["estimators"].get("groomed", {}).get("qcd", {})]
    if h:
        frac = lambda x: np.asarray(x["counts"], float) / (sum(x["counts"]) + x["underflow"] + x["overflow"])
        gr = frac(h[0])
        shown = [gr] + [np.mean(hs, axis=0) for arm in MASS
                        if (hs := [frac(atlas[m]["estimators"]["regressed"]["qcd"]["hist"]) for m in runs[arm]
                                   if m in atlas and "hist" in atlas[m]["estimators"].get("regressed", {}).get("qcd", {})])]
        on = np.flatnonzero(np.any([s >= F9A_SHOWN_FROM * s.max() for s in shown], axis=0))
        em.macro("MlstMassRespAxesShareQcd", _pct_prose([100 * gr[on[0]:on[-1] + 1].sum()]), index,
                 "head/atlas_res: share of QCD jets whose groomed-mass response lies inside F9a(b)'s x axis")


def loss_share_macros(em, comp, runs, index, fmt_pm, root: pathlib.Path | None = None) -> None:
    """The weight of the mass-regression term (MlstMassLambda, configs/arms/v2_grid.json, one value
    for both mass models or the build stops) and its realised share of the training loss, lambda *
    L_reg / (L_cls + lambda * L_reg), mean over the epochs of each run (compute/<run>.json)."""
    if root is not None:
        grid = root / "configs/arms/v2_grid.json"
        lam = {a.get("mass_lambda") for a in json.loads(grid.read_text())["arms"] if a["name"] in MASS}
        if len(lam) != 1 or None in lam:
            raise SystemExit(f"FATAL: the mass models' mass_lambda in {grid} is {sorted(map(str, lam))}, expected one value")
        em.macro("MlstMassLambda", f"{lam.pop():g}", grid, "arms[L162_MASS, R16_Q1_MASS].mass_lambda",
                 "weight of the mass-regression term in the pretraining loss, both mass models")
    for arm in MASS:
        sh = []
        for m in runs[arm]:
            ep = comp.get(m, {}).get("epochs", [])
            v = [e["loss_share_reg"] for e in ep if e.get("loss_share_reg") is not None]
            sh.append(100 * float(np.mean(v)) if v else None)
        em.macro(f"MlstMassLossShare{WORD[arm]}", _cell(sh, len(runs[arm]), fmt_pm, sep=TEXT_SEP), index,
                 f"share of the mass-regression term in the training loss of {arm}, mean over epochs, percent")


def _mass(mass: dict, model: str, what: str, key: str = "sigma_eff", group: str | None = None):
    """The older head/<model>_mass.json reader (deck figures only; the paper reads atlas_res)."""
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


# ------------------------------------------------------------------ sculpting (A14, F8b)
SCULPT_EFFS = ("0.1", "0.01")


def _sculpt(sculpt: dict, model: str, eff: str):
    """Jensen-Shannon distance (base 2) at a working point: the square root of the stored divergence."""
    d = sculpt.get(model)
    return None if d is None else math.sqrt(d["working_points"][eff]["jsd"])


def _floor_distance(k: int, n: int, n_ref: int | None = None) -> float:
    """The distance a random selection of n of the QCD jets shows: 8 n ln2 JSD ~ chi^2 with k - 1
    degrees of freedom to leading order, so E[sqrt(JSD)] = sqrt(2 / (8 n ln 2)) Gamma(k/2) / Gamma((k-1)/2).
    The selected jets are drawn without replacement from the n_ref jets of the reference histogram
    (all QCD test jets), which shrinks every bin's variance by 1 - n/n_ref (finite-population
    correction) and the distance by its square root: about 5 % at 10 % efficiency."""
    fpc = math.sqrt(1 - n / n_ref) if n_ref else 1.0
    return fpc * math.sqrt(2.0 / (8 * n * math.log(2))) * math.exp(math.lgamma(k / 2) - math.lgamma((k - 1) / 2))


def _floor_sd(k: int, n: int, n_ref: int | None = None) -> float:
    """The SD of that distance over random selections: the same chi distribution with k - 1 degrees of
    freedom, SD = scale sqrt(k - 1 - (E/scale)^2), scale = sqrt(1 / (8 n ln 2)), finite-population corrected."""
    scale = (math.sqrt(1 - n / n_ref) if n_ref else 1.0) * math.sqrt(1.0 / (8 * n * math.log(2)))
    return scale * math.sqrt(max(k - 1 - (_floor_distance(k, n, n_ref) / scale) ** 2, 0.0))


def _sculpt_floor(sculpt: dict, eff: str) -> float:
    """The divergence a cut that does not sculpt still shows: (k - 1) / (8 n ln 2) to leading order."""
    vals = {round((int(np.count_nonzero(d["hist_before"])) - 1)
                  / (8 * d["working_points"][eff]["n_pass"] * math.log(2)), 12) for d in sculpt.values()}
    if len(vals) != 1:
        raise SystemExit(f"FATAL: sculpting floor at {eff} takes {sorted(vals)} across the runs")
    return vals.pop()


def _sculpt_floor_distance(sculpt: dict, eff: str) -> float:
    vals = {round(_floor_distance(int(np.count_nonzero(d["hist_before"])), d["working_points"][eff]["n_pass"],
                                  d.get("n_qcd")), 12) for d in sculpt.values()}
    if len(vals) != 1:
        raise SystemExit(f"FATAL: sculpting floor at {eff} takes {sorted(vals)} across the runs")
    return vals.pop()


def _floor_cell(sculpt: dict, eff: str, fmt_pm) -> str:
    """The random-selection distance at a QCD-efficiency working point with its SD over random selections."""
    d = next(iter(sculpt.values()))
    k, n = int(np.count_nonzero(d["hist_before"])), d["working_points"][eff]["n_pass"]
    f, sd = _sculpt_floor_distance(sculpt, eff), _floor_sd(k, n, d.get("n_qcd"))
    return V4.tok("p", f, sd) if V4.tokens(fmt_pm) else fmt_pm(f, sd)


def _floor50_cell(sculpt: dict, ms: list, fmt_pm) -> str:
    """At 50% signal efficiency: the random-selection distance for the QCD jets each run's cut passes, the mean over
    the row's runs, with the SD over random selections at their mean count."""
    ds = [sculpt[m] for m in ms if m in sculpt and "sig_eff_50" in sculpt[m]]
    if not ds:
        return r"\pending{no run yet}"
    k = {int(np.count_nonzero(d["hist_before"])) for d in ds}
    if len(k) != 1:
        raise SystemExit(f"FATAL: the soft-drop histograms of {ms} have different filled bins")
    k = k.pop()
    fl = [_floor_distance(k, d["sig_eff_50"]["n_pass"], d.get("n_qcd")) for d in ds]
    n = float(np.mean([d["sig_eff_50"]["n_pass"] for d in ds]))
    sd = _floor_sd(k, int(round(n)), ds[0].get("n_qcd"))
    return V4.tok("p", float(np.mean(fl)), sd) if V4.tokens(fmt_pm) else fmt_pm(float(np.mean(fl)), sd)


SCULPT_FLOOR_WPS = ("0.1", "0.01", "eps_s=0.5")       # tab:sculpt's working points: eps_B 10%, 1%; eps_S 50%


def sculpt_above_floor(sculpt: dict, ms: list) -> dict:
    """PI 2026-11-21: how far a network's Jensen-Shannon distance (the mean over its runs ms) lies above the distance a
    random selection of as many QCD jets shows, at each working point of tab:sculpt: {wp: {distance, sd_runs, floor,
    sd_floor, z, pass}}, sigma^2 = the run SD^2 + the floor's SD^2 over random selections (_floor_sd), pass when the
    distance exceeds the floor by more than 3 sigma. At 50% signal efficiency each run's cut passes its own number of QCD
    jets, so the floor is the mean over runs (_floor50_cell). No caption or sentence claims sculpting at a working point
    that does not pass (tests/test_mlst_results.py)."""
    ds = [sculpt[m] for m in ms if m in sculpt]
    k = {int(np.count_nonzero(d["hist_before"])) for d in ds}
    if len(k) != 1:
        raise SystemExit(f"FATAL: the soft-drop histograms of {ms} have different filled bins")
    k, out = k.pop(), {}
    for wp in SCULPT_FLOOR_WPS:
        if wp.startswith("eps_s") and not all("sig_eff_50" in d for d in ds):
            continue
        at = [d["sig_eff_50"] if wp.startswith("eps_s") else d["working_points"][wp] for d in ds]
        dist = [math.sqrt(a["jsd"]) for a in at]
        floor = float(np.mean([_floor_distance(k, a["n_pass"], d.get("n_qcd")) for a, d in zip(at, ds)]))
        sd_floor = _floor_sd(k, int(round(np.mean([a["n_pass"] for a in at]))), ds[0].get("n_qcd"))
        sd_runs = float(np.std(dist, ddof=1)) if len(dist) > 1 else 0.0
        mu, sigma = float(np.mean(dist)), math.hypot(sd_runs, sd_floor)
        out[wp] = {"distance": mu, "sd_runs": sd_runs, "floor": floor, "sd_floor": sd_floor, "z": (mu - floor) / sigma,
                   "pass": mu - floor > 3 * sigma}
    return out


def table_sculpt_main(sculpt, runs, fmt_pm, rows) -> str:
    """tab:sculptmain (PI 2026-11-23, LEDGER CS-2b): the sculpting the main text states, compactly, each network used
    directly and its Jensen-Shannon distance at each working point of tab:tests' sculpting rows, beside the statistical
    floor, the distance a random selection of as many QCD jets shows (at 50% signal efficiency each row's own, as
    tab:sculpt; table_sculpt's cells, so the two print alike)."""
    v3 = all("sig_eff_50" in d for d in sculpt.values())
    lines = [r"\begin{tabular}{l" + "c" * len(SCULPT_EFFS) + ("cc" if v3 else "") + "}", r"\toprule",
             " & " + " & ".join(rf"$\epsB = {100 * float(e):g}\%$" for e in SCULPT_EFFS)
             + (r" & \multicolumn{2}{c}{$\varepsilon_S = 50\%$}" if v3 else "") + r" \\",
             (rf"\cmidrule(lr){{{2 + len(SCULPT_EFFS)}-{3 + len(SCULPT_EFFS)}}}" if v3 else ""),
             " & " + " & ".join(["distance"] * len(SCULPT_EFFS)) + (" & distance & floor" if v3 else "")
             + r" \\", r"\midrule"]       # fourth review (C7): one word, "floor", for the random-selection distance
    lines = [ln for ln in lines if ln]
    at50 = lambda m, k: None if (d := sculpt.get(m)) is None else d["sig_eff_50"][k]       # noqa: E731
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        cells = [_cell([_sculpt(sculpt, m, e) for m in ms], n, fmt_pm) for e in SCULPT_EFFS]
        if v3:
            cells += [_cell([None if (x := at50(m, "jsd")) is None else math.sqrt(x) for m in ms], n, fmt_pm),
                      _floor50_cell(sculpt, ms, fmt_pm)]
        lines.append(LABEL.get(arm, SCRATCH_LABEL) + " & " + " & ".join(cells) + r" \\")
    if sculpt:
        lines += [r"\midrule", "floor & " + " & ".join(_floor_cell(sculpt, e, fmt_pm) for e in SCULPT_EFFS)
                  + (" &  & " if v3 else "") + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def table_sculpt(sculpt, runs, fmt_pm, rows, label: dict | None = None) -> str:
    """A14 with M2 (PI 2026-11-07): at each QCD efficiency, the Jensen-Shannon distance (x 10^2) between the
    soft-drop mass distributions of all QCD test jets and of those passing the cut on 1 - p_QCD, and the share of
    resonant test jets passing the same cut; at the cut keeping half of the resonant jets (DisCo's working point),
    the QCD rejection R50 and the distance there. Below the runs the distance a random selection of as many QCD
    jets shows. The 50% columns need the files of mass_sculpting.py with sig_eff_50. PI 2026-11-21 (audit): the
    random-selection distance carries its SD over random selections (_floor_sd), and at 50% signal efficiency, where
    each network's cut passes its own number of QCD jets, each row has its own (the mean over its runs)."""
    v3 = all("sig_eff_50" in d for d in sculpt.values())
    lines = [r"\begin{tabular}{l" + "cc" * len(SCULPT_EFFS) + ("cccc" if v3 else "") + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\epsB = {100 * float(e):g}\%$}}" for e in SCULPT_EFFS)
             + (r" & \multicolumn{4}{c}{$\varepsilon_S = 50\%$}" if v3 else "") + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * q, 3 + 2 * q + (q == len(SCULPT_EFFS)))
                     for q in range(len(SCULPT_EFFS) + v3)),
             " & " + " & ".join([r"distance & $\varepsilon_S$ (\%)"] * len(SCULPT_EFFS))
             + (r" & $R_{50}$ & distance & random selection & $1/D_{\mathrm{JS}}$" if v3 else "") + r" \\", r"\midrule"]
    eff_s = lambda m, e: None if (d := sculpt.get(m)) is None else d["working_points"][e]["eff_sig"]
    at50 = lambda m, k: None if (d := sculpt.get(m)) is None else d["sig_eff_50"][k]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        cells = []
        for e in SCULPT_EFFS:
            # PI 2026-11-14: the distance unscaled, as the figures and tab:masseffect print it; eps_S in percent, as
            # tab:xbx
            cells += [_cell([_sculpt(sculpt, m, e) for m in ms], n, fmt_pm),
                      _cell([eff_s(m, e) if v3 else None for m in ms], n, fmt_pm, 1e2) if v3 else "--"]
        if v3:
            cells += [_cell([at50(m, "rejection") for m in ms], n, fmt_pm),
                      _cell([None if (x := at50(m, "jsd")) is None else math.sqrt(x) for m in ms], n, fmt_pm),
                      _floor50_cell(sculpt, ms, fmt_pm),
                      # F10c's vertical axis: the mean over runs of the inverse divergence
                      _cell([None if (x := at50(m, "jsd")) is None else 1 / x for m in ms], n, fmt_pm)]
        lines.append((label or LABEL).get(arm, SCRATCH_LABEL) + " & " + " & ".join(cells) + r" \\")
    if sculpt:
        lines += [r"\midrule", "random selection & "
                  # the distance with its SD over random selections (audit 2026-11-21); at 50% each row's own column
                  + " & ".join(f"{_floor_cell(sculpt, e, fmt_pm)} & --" for e in SCULPT_EFFS)
                  + (" & -- & -- & -- & --" if v3 else "") + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


# the fine-tuned networks' sculpting (experiments/EVAL/sculpting_ft.py), the one rule's JetClass-II cells at 1e6 only
# (scripts/build_gallery_data_job.py --mode sculpt-ft-fixed, pulled whole after its DONE): the older trees' files in
# head/sculpting_ft are never read
SCULPT_FT = "head/sculpting_ft_fixed"
SCULPT_BOOT_WPS = (("eps_b=0.1", r"$\epsB = 10\%$"), ("eps_b=0.01", r"$\epsB = 1\%$"),     # tab:sculpt's order
                   ("eps_s=0.5", r"$\epsS = 50\%$"))


def _sculpt_distance(sculpt: dict, m: str, wp: str) -> float:
    """The Jensen-Shannon distance of run m at a working point of sculpting_boot.json ("eps_b=0.01", "eps_s=0.5")."""
    d = sculpt[m]
    return math.sqrt(d["working_points"][wp.split("=")[1]]["jsd"] if wp.startswith("eps_b") else d["sig_eff_50"]["jsd"])


SCULPT_BOOT_L3 = "bootstrap/l3/sculpting_boot_l3.json"     # decision L3 (2026-11-13): the sides unpaired, every run
# the rerun's copy (scripts/build_boot_l3_jobs.py --rerun, pulled to l3_rerun/): the same heads with the five random
# partitions pooled against 188 classes (sculpting_boot.POOLED); read in place of the first once pulled
SCULPT_BOOT_L3_RERUN = "l3_rerun/sculpting_boot_l3.json"


def table_sculpt_boot(doc, runs, sculpt=None) -> str:
    """The sculpting differences of experiments/EVAL/sculpting_boot.py under decision L3 (bootstrap/l3/
    sculpting_boot_l3.json, "l3" block of each difference): delta = mean over the first network's runs - mean over the
    second's, the SD over the runs of each side, the bootstrap SD of delta over test jets, and delta / sigma with
    sigma^2 the sum of the three squares (|delta| > 3 sigma resolves a difference), one row group per working point,
    distances unscaled at four decimals (PI 2026-11-14: one unit with tab:sculpt and the figures); no bold (PI 2026-11-12). With sculpt (the per-run files) every number but the
    bootstrap SD is recomputed from them first, and a mismatch is fatal. Rows in the tables' order (ALL_ROWS) of the
    network, then of its reference."""
    num = lambda v: f"{v:.4f}".replace("-", "$-$")      # noqa: E731  unscaled distances (PI 2026-11-14), four decimals
    arm = {m.removeprefix("mtx-").rsplit("-s", 1)[0]: a for a, ms in runs.items() for m in ms}
    arm["rand2p"] = RAND_POOLED         # sculpting_boot.POOLED: the five random partitions as one side
    rows = ALL_ROWS + (RAND_POOLED,)
    diffs = doc["differences"]
    for name, d in diffs.items():
        for wp, _ in SCULPT_BOOT_WPS:
            x = d["l3"][wp]["distance"]
            if not math.isclose(x["sigma"], math.sqrt(x["sd_config"] ** 2 + x["sd_reference"] ** 2 + x["boot_sd"] ** 2),
                                rel_tol=1e-9) or x["pass"] != (abs(x["delta"]) > 3 * x["sigma"]):
                raise SystemExit(f"FATAL: {SCULPT_BOOT_L3} {name} {wp}: sigma is not the quadrature sum of the two run "
                                 "SDs and the bootstrap SD, or pass is not |delta| > 3 sigma")
            if sculpt:
                a = [_sculpt_distance(sculpt, m, wp) for m in x["runs_config"]]
                b = [_sculpt_distance(sculpt, m, wp) for m in x["runs_reference"]]
                if any(abs(u - v) > 1e-12 for u, v in ((np.mean(a) - np.mean(b), x["delta"]),
                                                       (np.std(a, ddof=1), x["sd_config"]),
                                                       (np.std(b, ddof=1), x["sd_reference"]))):
                    raise SystemExit(f"FATAL: {SCULPT_BOOT_L3} {name} {wp}: delta or a run SD differs from the "
                                     "sculpting files")
    order = lambda x: (rows.index(arm[x[1]["configuration"]]), rows.index(arm[x[1]["reference"]]))   # noqa: E731
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r" & $\Delta$ & run SD, first & run SD, second & bootstrap SD & $\Delta/\sigma$ \\", r"\midrule"]
    for k, (wp, head) in enumerate(SCULPT_BOOT_WPS):
        block = [_full_row(6, head, italic=True)]
        for name, d in sorted(diffs.items(), key=order):
            part = arm[d["configuration"]] in FLAV + RAND + (RAND_POOLED,)
            if part and PARTITIONS not in block[-1] and not block[-1].startswith(r"\quad "):
                block.append(_full_row(6, PARTITIONS))
            x = d["l3"][wp]["distance"]
            block.append((r"\quad " if part else "") + f"{LABEL[arm[d['configuration']]]} $-$ {LABEL[arm[d['reference']]]} & "
                         + " & ".join(num(x[q]) for q in ("delta", "sd_config", "sd_reference", "boot_sd"))
                         + f" & {x['z']:.2f}".replace("-", "$-$") + " \\\\")
        # a working point's rows never break apart (page review round 3: a lone row on the next page lost its heading);
        # between two of them a space, where the page may break (a rule there doubled the page's foot rule)
        out += ([r"\noalign{\penalty-100}\addlinespace"] if k else []) \
            + [ln.rstrip() + "*" for ln in block[:-1]] + block[-1:]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


QCDHF_BKG = (("bb", r"$b\bar b$"), ("b", "$b$"), ("qcd_all", "all QCD"))
QCDHF_WP = "r50"           # at 30% (the text's R30) too few single-b jets pass; charm and light QCD pass hardly at all


def _qcdhf(hf, m, ck, b) -> dict | None:
    return hf.get(m, {}).get("cells", {}).get(ck, {}).get("background", {}).get(b, {}).get("natural", {}).get(QCDHF_WP)


def table_qcdhf(hf, runs, fmt_pm, rows) -> str:
    """Should-add 11 (PI 2026-11-07): rejection R50 of the QCD heavy-flavor subclasses by Sophon's X->bb
    discriminant from the class-balanced linear probe at 1e6 (experiments/EVAL/qcd_hf_rejection.py), each subclass
    weighted to its natural test composition; a dagger where fewer than JC1_MIN_BKG jets pass in a run."""
    out = [r"\begin{tabular}{l" + "c" * len(QCDHF_BKG) + "}", r"\toprule",
           r" & \multicolumn{%d}{c}{QCD background, $R_{50}$} \\" % len(QCDHF_BKG),
           r"\cmidrule(lr){2-%d}" % (len(QCDHF_BKG) + 1), " & " + " & ".join(t for _, t in QCDHF_BKG) + r" \\",
           r"\midrule"]
    for arm in rows:
        ms, ck = ((INIT_MODELS, "init") if arm == INIT else (INIT_MODELS, "init_bn") if arm == INIT_BN
                  else (runs[arm], PRIMARY))
        cells = []
        for b, _ in QCDHF_BKG:
            cs = [_qcdhf(hf, m, ck, b) for m in ms]
            few = any(c is not None and c["n_bkg_pass"] < V4.JC1_MIN_BKG for c in cs)
            cells.append(_cell([None if c is None else c["rejection"] for c in cs], len(ms), fmt_pm)
                         + (r"$^\dagger$" if few else ""))
        out.append((LABEL[arm] if arm in (INIT, INIT_BN) else _row(arm, runs)) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


COARSE_TARGETS = (("R42_Q1", r"43 groups"), ("R16_Q1", r"17 groups"), ("FLAV_F0", r"17 groups with no flavor boundary"))
# review of render dfe5942 (B14; LEDGER R2): the paper's copy holds the 17 groups only, the one target a sentence states
COARSE_PAPER = (("R16_Q1", r"17 groups"),)


def table_coarse(co, runs, fmt_pm, rows, targets=COARSE_TARGETS, bs3: dict | None = None) -> str:
    """Should-add 5 (PI 2026-11-07): linear probes of every network on the coarse targets of class-balanced
    JetClass-II at 1e6 (probe_bal.py --coarse): the 43- and 17-group levels of the hierarchy and the 17-class
    partition with no flavor boundary (targets: the paper's COARSE_PAPER); accuracy and the macro AUC over the
    target's groups."""
    def val(m, ck, t, k):
        c = co.get(m, {}).get("cells", {}).get(ck, {}).get("jc2bal_coarse", {}).get(t, {}).get("1000000")
        return None if c is None else c[k]
    n = len(targets)
    out = [r"\begin{tabular}{l" + "cc" * n + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % w for _, w in targets) + r" \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(n)),
           " & " + " & ".join(["accuracy & AUC"] * n) + r" \\", r"\midrule"]
    for arm in rows:
        ms, ck = ((INIT_MODELS, "init") if arm == INIT else (INIT_MODELS, "init_bn") if arm == INIT_BN
                  else (runs[arm], SOPHON_CK) if arm == SOPHON else (runs[arm], PRIMARY))
        cells = [_cell([val(m, ck, t, k) for m in ms], len(ms), fmt_pm, frac=True)
                 for t, _ in targets for k in ("accuracy", "macro_auc_ovr")]
        out += [r"\midrule"] * (arm == SOPHON) + [
            (LABEL[arm] if arm in (INIT, INIT_BN) else _row(arm, runs)) + " & " + " & ".join(cells) + r" \\"]
    # LEDGER DL-9 (bs3): the MLP probe beside the linear-probe null on the 17 groups, the four label sets, in a block of
    # its own under its mode's heading (the values its tab:tests rows were bootstrapped from)
    if bs3 and bs3.get("coarse") and [t for t, _ in targets] == ["R16_Q1"]:
        out.append(_full_row(1 + 2 * n, HEADING["mlp"], italic=True))
        for arm in MAIN:
            ms = runs[arm]
            cells = [_cell([_bs3_val(bs3, "coarse", m, "jc2bal_coarse", k) for m in ms], len(ms), fmt_pm, frac=True)
                     for k in ("accuracy", "macro_auc_ovr")]
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def qcdhf_fill(hf) -> dict:
    """The most charm and light QCD jets passing the R50 cut in any run of a pretrained network, for the caption."""
    n = {b: max((c["n_bkg_pass"] for d in hf.values() for ck, cell in d.get("cells", {}).items() if ck == PRIMARY
                 if (c := cell.get("background", {}).get(b, {}).get("natural", {}).get(QCDHF_WP))), default=0)
         for b in ("c", "light")}
    return {"__QCDHF_C__": str(n["c"]), "__QCDHF_LIGHT__": str(n["light"])}


def sculpt_fill(sculpt, runs) -> dict:
    """The sentences of tab:sculpt's caption (the 17-class partitions in the paper's copy too, PI 2026-11-22)."""
    v3 = all("sig_eff_50" in d for d in sculpt.values())
    flav = any(m in sculpt for a in FLAV for m in runs[a])
    return {"__SCULPT_FLAV__": ", and with the three 17-class partitions with no, a $b$-quark and a random flavor "
                               "boundary" if flav else "",
            "__SCULPT_SIG__": (r" The signal is every resonant test jet, the \LitJetClassTwoNResonantClasses{} "
                               r"resonant classes taken together. Beside each distance, $\epsS$ is the fraction "
                               r"of these jets that pass the same cut. The last two columns use the cut that keeps "
                               r"half of them ($\epsS = 50\%$) and give the QCD rejection there, "
                               r"$R_{50} = 1/\epsB$, and the distance at that cut."
                               if v3 else ""),
            "__SCULPT_CUT__": (r", or half the resonant jets ($\epsS = 50\%$, $R_{50} = 1/\epsB$), $\epsS$ the "
                               r"resonant share passing (\%), $1/D_{\mathrm{JS}}$ the mean inverse divergence "
                               r"(distance squared)"
                               if v3 else ""),
            "__SCULPT_RUNS__": _runs("set", *(("flav",) if flav else ()), "mass", "rand")}


def _curve(sculpt_v2: dict, m: str) -> dict:
    """{efficiency: (distance, floor distance)} of a run's sculpting curve (curves_v2)."""
    d = sculpt_v2.get(m)
    if d is None:
        return {}
    k = int(np.count_nonzero(d["hist_before"]))
    return {round(p["eff"], 9): (p["js_distance"], _floor_distance(k, p["n_pass"], d.get("n_qcd"))) for p in d["curve"]}


def _window_around(effs: list, passing: list, around: float) -> tuple | None:
    """(lowest, highest) efficiency of the unbroken stretch of grid points in passing that holds the
    grid point nearest to around on a logarithmic axis; None when that point does not pass."""
    i = min(range(len(effs)), key=lambda k: abs(math.log(effs[k] / around)))
    if effs[i] not in passing:
        return None
    lo = hi = i
    while lo > 0 and effs[lo - 1] in passing:
        lo -= 1
    while hi < len(effs) - 1 and effs[hi + 1] in passing:
        hi += 1
    return effs[lo], effs[hi]


def sculpt_macros(em, sculpt, sculpt_v2, runs, index, fmt_pm) -> None:
    """Distances at the two working points (MlstSculpt<arm><Ten|One>), the random-selection floor,
    run-by-run ratios between configurations and to the floor, and on the stored efficiency grid the
    unbroken window around the tighter working point where the 17-class model sculpts more than the
    188-class one (the paired ratio's mean minus twice its SD above 1) and, at the lowest stored
    efficiency, the largest excess over the floor in units of the floor's SD."""
    for e, ew in zip(SCULPT_EFFS, ("Ten", "One")):
        n_pass = {d["working_points"][e]["n_pass"] for d in sculpt.values()}
        if len(n_pass) != 1:
            raise SystemExit(f"FATAL: {sorted(n_pass)} QCD jets pass at {e} across the runs")
        em.macro(f"MlstSculptNPass{ew}", f"{n_pass.pop():,}".replace(",", "{,}"), index,
                 f"QCD test jets passing the cut at QCD efficiency {e}, every run")
        floor = _sculpt_floor_distance(sculpt, e)
        em.macro(f"MlstSculptFloor{ew}", _lone(floor), index,
                 f"Jensen-Shannon distance expected for a random selection of as many QCD jets as pass at {e}")
        for arm in MAIN + MASS:
            ms = runs[arm]
            em.macro(f"MlstSculpt{WORD[arm]}{ew}", _cell([_sculpt(sculpt, m, e) for m in ms], len(ms), fmt_pm,
                                                        sep=TEXT_SEP), index,
                     f"Jensen-Shannon distance of the QCD m_SD distribution at QCD efficiency {e}, {arm}")
            r = [None if (x := _sculpt(sculpt, m, e)) is None else x / floor for m in ms]
            _emit_ratio(em, f"MlstSculpt{WORD[arm]}OverFloor{ew}", r, len(ms), fmt_pm, index,
                        f"distance at {e} over the random-selection floor, {arm}", "distance", _two_run(runs, arm))
        f = lambda m, e=e: _sculpt(sculpt, m, e)
        for arm, ref in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"), ("R16_Q1", "L188"), ("L162", "L188"),
                         ("R42_Q1", "L188"), ("R16_Q1", "R42_Q1")):
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstSculpt{WORD[arm]}Over{WORD[ref]}{ew}", r, k, fmt_pm, index,
                        f"distance at QCD efficiency {e}: {arm} over {ref}, run by run", "distance", _two_run(runs, arm, ref))
    curves = {m: _curve(sculpt_v2, m) for arm in MAIN + MASS for m in runs[arm]}
    effs = sorted(set.intersection(*(set(c) for c in curves.values()))) if all(curves.values()) else []
    if not effs:
        for k in ("MlstSculptDiffEffLo", "MlstSculptDiffEffHi", "MlstSculptLowestEff",
                  "MlstSculptLowestNPass", "MlstSculptLowestMaxFloorSd", "MlstSculptNConfigs"):
            em.macro(k, r"\pending{sculpting curves}", index, "head/curves_v2 not complete")
        return
    diff = []
    for e in effs:
        r, _ = _ratio_vals(runs, "R16_Q1", "L188", lambda m: curves[m][e][0])
        v = [x for x in r if x is not None]
        if len(v) >= 3 and np.mean(v) - 2 * np.std(v, ddof=1) > 1:
            diff.append(e)
    pc = lambda e: f"{_sig2(100 * e)}\\%"
    window = _window_around(effs, diff, float(SCULPT_EFFS[-1]))
    em.macro("MlstSculptDiffEffLo", pc(window[0]) if window else "none", index,
             "lowest stored QCD efficiency of the unbroken stretch around the tighter working point at which the "
             "17/188 distance ratio exceeds 1 by twice its SD", "passing grid points: " + ", ".join(f"{e:g}" for e in diff))
    em.macro("MlstSculptDiffEffHi", pc(window[1]) if window else "none", index, "highest efficiency of that stretch")
    # At the lowest efficiency so few jets pass that a random selection shows a comparable distance: how
    # far above that floor each configuration's mean lies, in units of the floor's own SD over random
    # selections (not the seed SD, which measures the runs' agreement, not closeness to the floor).
    lo = effs[0]
    d0 = sculpt_v2[runs[MAIN[0]][0]]
    k0 = int(np.count_nonzero(d0["hist_before"]))
    p0 = next(p for p in d0["curve"] if round(p["eff"], 9) == lo)
    sd0 = _floor_sd(k0, p0["n_pass"], d0.get("n_qcd"))
    excess = [(np.mean([curves[m][lo][0] for m in runs[arm]]) - curves[runs[arm][0]][lo][1]) / sd0
              for arm in MAIN + MASS]
    em.macro("MlstSculptLowestEff", pc(lo), index, "lowest stored QCD efficiency of the sculpting curves")
    em.macro("MlstSculptLowestNPass", f"{p0['n_pass']:,}".replace(",", "{,}"), index,
             "QCD test jets passing at the lowest stored efficiency")
    em.macro("MlstSculptLowestMaxFloorSd", f"{max(abs(x) for x in excess):.1f}", index,
             "largest distance of a configuration's mean from the random-selection floor at the lowest stored "
             "efficiency, in SDs of the floor over random selections")
    em.macro("MlstSculptNConfigs", str(len(MAIN + MASS)), index, "configurations with a sculpting curve (count)")


# ------------------------------------------------------------------ fine-tuning read-outs
def ft_source(data: pathlib.Path) -> str:
    """Which fine-tuning read-out load_ft takes below the full sets (2026-11-21: the preview's Fig. 8 read the one-rule
    cells and its Table A6 the older trees, from two builds; the tables and the figures each record this, and the book
    refuses two different ones)."""
    fixed = data / "budget" / FIXED
    if fixed.exists():
        return FIXED
    if ft_preview() and (data / "budget" / FIXED_PARTIAL).exists():
        return FIXED_PARTIAL + " (preview)"
    return "budget, budget_long, fulldecay, prefull (the trees before the one rule)"


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
    fixed = data / "budget" / FIXED
    if ft_preview() and not fixed.exists():     # a preview reads the one-rule cells finished so far (FIXED_PARTIAL)
        fixed = data / "budget" / FIXED_PARTIAL
        if not fixed.exists():
            raise SystemExit(f"FATAL: {PREVIEW_ENV}=1 but neither {FIXED} nor {FIXED_PARTIAL} in {data / 'budget'}")
    if fixed.exists():      # the one rule (PI 2026-11-15) replaces every older tree below the full sets
        return tag, merge_kin(merge_fixed(per, fixed, preview=fixed.name == FIXED_PARTIAL), data / "budget" / KIN_FULL), absent
    return tag, merge_kin(merge_budget(per, data / "budget" / "budget_metrics.json",
                                       data / "budget" / "budget_long_metrics.json",
                                       [data / "budget" / f for f in SCRATCH_FULL], data / "budget" / PRE_FULL),
                          data / "budget" / KIN_FULL), absent


# 2026-10-31 (plan 3.1, 3.4): below the full set every fine-tuning is the fixed-budget recipe
# (experiments/FT/budget_metrics.py), which supersedes the earlier cells there; JetClass-II is the
# class-balanced task at every size. (budget tree, sub-record or None) per task; the full sets.
BUDGET_TREE = {"top": ("leg_top", None), "qg": ("leg_qg", None), "qg_herwig": ("leg_qg", "herwig"),
               "jc1": ("leg2", None), "jc2": ("leg_jc2b", None), "jc2_qcd": ("leg_jc2b", None)}
BUDGET_SCRATCH = "scratch"          # build_ft_budget_jobs.cell's directory for a from-scratch cell
# From scratch with its whole flat+decay schedule (PI 2026-11-07; build_ft_budget_jobs scratch_full_cells and
# long_scratch_full_cells, read out by job-budget-readout-e): {read-out: its cells}. Each replaces every
# from-scratch cell of its (task, size), the fixed-budget and the long-budget alike; once both are pulled no
# from-scratch cell stopped by the patience rule is read.
SCRATCH_FULL = {"budget_fulldecay_metrics.json": 30, "budget_long_fulldecay_metrics.json": 6}
# Pretrained cells run to their last epoch (PI 2026-11-08, operationalised 2026-11-09; build_ft_budget_jobs pre_full_cells
# and pre_full_rerun_cells): the 188- and 17-class runs at every task and size below 1e5, and, where either moved by more
# than its old seed SD, every other pretrained cell there. Per (task, size), used complete or not at all: every pretrained
# model the fixed-budget read-out has there -> they replace the old cells; only the equal-stopping test's six -> the old
# cells stand (nothing moved); anything else is FATAL.
PRE_FULL = "budget_prefull_metrics.json"
PRE_FULL_TEST = tuple(f"{a}-s{k}" for a in ("l188", "r16q1") for k in (1, 2, 3))


# The one fine-tuning rule (PI 2026-11-15; scripts/build_ft_budget_jobs.py FIXED_*): every cell below the full top and
# quark/gluon sets, pretrained, kinematic-only and from scratch, from one read-out of ft_v2/fixed. Used complete or not at
# all: once it is pulled no cell of the older trees (budget, budget_long, budget_prefull, *_fulldecay) is read.
# 341 = top 3 x 31 (27 paper runs, the kinematic-only run, 3 from scratch) + quark/gluon 3 x 30 + JetClass 3 x 30
# + JetClass-II 2 x 19 (below 1e6 the label sets and mass networks only) + 30 at 1e6; the cells of the launched
# job-ft-budget-fixed-{1e3,1e4,1e5,1e6}*-raunav specs (tests/test_ft_fixed_readout.py counts them).
FIXED, FIXED_N = "fixed_metrics.json", 341
# A preview (PREVIEW_ENV=1, PI 2026-11-18) reads job-budget-readout-fixed-preview's partial read-out when the complete one
# is absent: every finished cell, a missing one absent (its table cell pending), a from-scratch size with a seed missing
# left out whole. Never the paper's build: without PREVIEW_ENV a partial read-out is not read at all.
FIXED_PARTIAL, PREVIEW_ENV = "fixed_metrics_partial.json", "MLST_FT_PREVIEW"
# The full sets that keep ParT's recipe under the one rule: top and quark/gluon. JetClass's largest set, 1e6 jets, is a
# one-rule cell (build_ft_budget_jobs.FIXED_SIZES), so the old JetClass cells at FULL_N are not read.
FIXED_PART_FULL = ("top", "qg", "qg_herwig")
# the kinematic-only network fine-tuned on the full top set with ParT's recipe (ft_v2/best70_kin, one cell)
KIN_FULL = "best70_kin_metrics.json"


def ft_preview() -> bool:
    import os
    return os.environ.get(PREVIEW_ENV) == "1"


def merge_fixed(per: dict, path: pathlib.Path, preview: bool = False) -> dict:
    """per's full top and quark/gluon cells (FIXED_PART_FULL) and every cell of the one-rule read-out; FATAL unless it is
    complete, or with preview (FIXED_PARTIAL) unless it names every cell, read or missing."""
    doc = json.loads(path.read_text())
    miss = doc.get("missing", [])
    if (miss and not preview) or len(doc["cells"]) + (len(miss) if preview else 0) != FIXED_N:
        raise SystemExit(f"FATAL: {path}: {len(doc['cells'])} cells (expected {FIXED_N}), missing {len(miss)}: {miss[:5]}; "
                         "the one-rule read-out is used complete or not at all" + (" (a preview names every cell)" if preview else ""))
    out = {key: {init: {n: v for n, v in by_n.items() if key in FIXED_PART_FULL and n >= FULL_N[key]}
                 for init, by_n in cells.items()} for key, cells in per.items()}
    for (key, init, n), recs in _budget_cells(path).items():
        if key in FIXED_PART_FULL and n >= FULL_N[key]:
            raise SystemExit(f"FATAL: {path}: a full-set cell {key} {init} N={n}; the full sets keep ParT's recipe")
        out.setdefault(key, {}).setdefault(init, {})[n] = recs
    for k in miss:      # preview: a from-scratch size short of a seed would read as complete, so it is left out
        tree, init, n, _ = k.split("/")
        if init == BUDGET_SCRATCH:
            for key, (t, _) in BUDGET_TREE.items():
                if t == tree:
                    out.get(key, {}).get(SCRATCH, {}).pop(int(n[1:]), None)
    # the edge rule (PI 2026-11-15), applied where the cells are read (PI 2026-11-20: it lived only in
    # test_comparisons_identical, which never saw the cells that broke it): no best check first or in the last 10% of
    # the schedule. The paper build stops; a preview leaves the whole row out, since the row is rerun at a swept rate
    # one rate per row (PI 2026-11-21): every cell read at its row's recorded rate (build_ft_budget_jobs.FIXED_ROW_RATE,
    # FIXED_ROW_RATE_SCRATCH; the original rate while a row has not settled). The paper build stops on any other; a
    # preview leaves the row out, as it waits for its rerun
    wrong = rate_mismatches(_budget_cells(path))
    if wrong and not preview:
        raise SystemExit(f"FATAL: {path}: {len(wrong)} cells not at their row's recorded rate: {wrong[:10]}")
    if not preview and any(row_unsettled(key, n) for key, _, n in _budget_cells(path)):
        raise SystemExit(f"FATAL: {path}: a row whose rate search has not settled (build_ft_budget_jobs.FIXED_ROW_RATE)")
    # RR-7 (PI 2026-11-23): a row whose rate search has not settled has no recorded rate, so a preview prints none of it
    # RR-7 (figure agent's check, 2026-11-23): a dropped row takes with it every key that re-scores the same runs (the
    # Herwig cells are the quark/gluon runs scored on Herwig jets, so rate_mismatches and edge_failures skip them)
    same_runs = lambda pairs: {(k, n) for t, n in pairs for k in BUDGET_TREE if BUDGET_TREE[k][0] == BUDGET_TREE[t][0]}  # noqa: E731
    unsettled = {(key, n) for key, _, n in _budget_cells(path) if row_unsettled(key, n)}
    for key, n in same_runs({(key, n) for key, _, n, _ in wrong} | unsettled):
        for by_n in out.get(key, {}).values():
            by_n.pop(n, None)
    edge = edge_failures(_budget_cells(path))
    if edge and not preview:
        raise SystemExit(f"FATAL: {path}: {len(edge)} cells with the best check at an edge of the schedule: {edge[:10]}")
    for key, n in same_runs({(key, n) for key, _, n, _ in edge}):
        for by_n in out.get(key, {}).values():
            by_n.pop(n, None)
    return out


# The mass items read only the truth-free refit (PI 2026-11-16/21): experiments/EVAL/atlas_resolution.py --centre
# global, one training median for every jet, never the jet's class; atlas_res (class-centred, the old
# mass_resolution.py path) is read by nothing. Absent: the items are pending, never filled from the old files.
MASS_RES, MASS_RES_FINE = "head/atlas_res_v3", "head/atlas_res_fine_v3"
BUDGET_TASK = {"top": "top", "qg": "qg", "qg_herwig": "qg", "jc1": "jc1", "jc2": "jc2b", "jc2_qcd": "jc2b"}


@functools.lru_cache(maxsize=1)
def _rate_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_ft_budget_jobs", pathlib.Path(__file__).resolve().parents[2]
                                                  / "scripts/build_ft_budget_jobs.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    return b


def expected_rate(key: str, init: str, n: int) -> str:
    """The rate a one-rule cell must have: its row's recorded rate, or the original while the row has not settled."""
    b, kind = _rate_builder(), "scratch" if init == SCRATCH else "pre"
    task = "topkin" if init == KIN_INIT else BUDGET_TASK[key]
    return b.row_rate(task, n, kind) or b.original_rate(BUDGET_TASK[key], kind)


def row_unsettled(key: str, n: int) -> bool:
    b = _rate_builder()
    return any(b.row_rate(BUDGET_TASK[key], n, kind) is None for kind in ("pre", "scratch"))


def rate_mismatches(cells: dict) -> list:
    """[(task, init, N, 'rate x, recorded y')] of the cells not at their row's recorded rate."""
    bad = []
    for (key, init, n), recs in sorted(cells.items()):
        if BUDGET_TREE[key][1]:         # a re-scoring of another key's cells (Herwig): its recipe is there
            continue
        want = expected_rate(key, init, n)
        for rec in recs:
            got = (rec.get("recipe") or {}).get("lr")
            if got != want:
                bad.append((key, init, n, f"rate {got}, recorded {want}"))
    return bad


def edge_failures(cells: dict) -> list:
    """[(task, init, N, best check of max)] of the cells whose best validation check is the first (0) or within the
    last 10% of the schedule, the one rule's edge rule."""
    bad = []
    for (key, init, n), recs in sorted(cells.items()):
        if BUDGET_TREE[key][1]:         # a re-scoring of another key's cells (Herwig): its training record is there
            continue
        for rec in recs:
            e = rec.get("early_stop") or {}
            best, ep = e.get("best_epoch"), e.get("max_epochs")
            if best is None or ep is None or best < 1 or best >= 0.9 * ep:
                bad.append((key, init, n, f"{best} of {ep}"))
    return bad


def merge_kin(out: dict, path: pathlib.Path) -> dict:
    """out with the kinematic-only network's full top cell (KIN_FULL) where it is pulled; FATAL unless it is that one cell."""
    if not path.exists():
        return out
    cells = _budget_cells(path)
    if list(cells) != [("top", KIN_INIT, FULL_N["top"])] or len(cells[("top", KIN_INIT, FULL_N["top"])]) != 1:
        raise SystemExit(f"FATAL: {path}: {sorted(cells)}; expected the one full top cell of {KIN_INIT}")
    out.setdefault("top", {}).setdefault(KIN_INIT, {})[FULL_N["top"]] = cells[("top", KIN_INIT, FULL_N["top"])]
    return out


def _budget_cells(path: pathlib.Path) -> dict:
    """{(task, init, N): [record per seed]} of a budget_metrics.py read-out, from-scratch under SCRATCH."""
    out = {}
    if not path.exists():
        return out
    cells = json.loads(path.read_text())["cells"]
    for key, (tree, sub) in BUDGET_TREE.items():
        for k in sorted(c for c in cells if c.startswith(tree + "/")):
            _, init, n, _seed = k.split("/")
            rec = cells[k][sub] if sub else cells[k]
            out.setdefault((key, SCRATCH if init == BUDGET_SCRATCH else init, int(n[1:])), []).append(rec)
    return out


def merge_budget(per: dict, budget: pathlib.Path, long: pathlib.Path | None = None, full=(),
                 prefull: pathlib.Path | None = None) -> dict:
    """per with the superseded cells removed (below FULL_N; every natural-mix JetClass-II cell) and
    the fixed-budget cells added, the from-scratch ones under SCRATCH. The long-budget cells at 1e6
    (budget_long, PI 2026-11-05) then replace every cell of the same task, model and size, the
    earlier fine-tuning and the fixed-budget alike: one recipe per (task, model, size). A tree of the
    long budget with any cell still missing is not used at all, so a figure never mixes the two recipes.
    Last, each full-schedule from-scratch read-out present (SCRATCH_FULL) replaces the from-scratch cells
    of its (task, size); one with a missing cell, a cell count other than SCRATCH_FULL's or a pretrained
    cell is FATAL, and with all of them present an old from-scratch budget cell left over is FATAL."""
    out = {key: {init: {n: v for n, v in by_n.items() if key in FULL_N and n >= FULL_N[key]}
                 for init, by_n in cells.items()} for key, cells in per.items()}
    for src in (budget, long):
        if src is None:
            continue
        unfinished = set()
        if src is long and src.exists():
            unfinished = {c.split("/")[0] for c in json.loads(src.read_text()).get("missing", [])}
        for (key, init, n), recs in _budget_cells(src).items():
            if BUDGET_TREE[key][0] in unfinished:
                continue
            out.setdefault(key, {}).setdefault(init, {})[n] = recs
    old = {(key, n) for src in (budget, long) if src is not None
           for key, init, n in _budget_cells(src) if init == SCRATCH}
    new = set()
    for src in full:
        if not src.exists():
            continue
        doc = json.loads(src.read_text())
        pre = sorted(k for k in doc["cells"] if k.split("/")[1] != BUDGET_SCRATCH)
        if doc.get("missing") or len(doc["cells"]) != SCRATCH_FULL[src.name] or pre:
            raise SystemExit(f"FATAL: {src}: {len(doc['cells'])} cells (expected {SCRATCH_FULL[src.name]}), missing "
                             f"{doc.get('missing')}, pretrained {pre}; a from-scratch read-out is used complete or not at all")
        for (key, _, n), recs in _budget_cells(src).items():
            out.setdefault(key, {}).setdefault(SCRATCH, {})[n] = recs
            new.add((key, n))
    if full and all(src.exists() for src in full) and old - new:
        raise SystemExit(f"FATAL: old from-scratch budget cells still read at {sorted(old - new)}")
    return merge_prefull(out, prefull)


def merge_prefull(out: dict, prefull: pathlib.Path | None) -> dict:
    """out with the pretrained cells of every (task, size) the equal-stopping read-out holds completely replaced by its
    cells (PRE_FULL's rule); a (task, size) with only PRE_FULL_TEST keeps its old cells; a partial set is FATAL."""
    if prefull is None or not prefull.exists():
        return out
    doc = json.loads(prefull.read_text())
    cells = _budget_cells(prefull)
    if doc.get("missing") or any(init == SCRATCH for _, init, _ in cells):
        raise SystemExit(f"FATAL: {prefull}: missing {doc.get('missing')} or a from-scratch cell; used complete or not at all")
    for key, n in sorted({(k, n) for k, _, n in cells}):
        have = {init for k, init, m in cells if (k, m) == (key, n)}
        want = {init for init, by_n in out.get(key, {}).items() if init != SCRATCH and n in by_n}
        if have == want:
            for init in want:
                out[key][init][n] = cells[(key, init, n)]
        elif have != set(PRE_FULL_TEST):
            raise SystemExit(f"FATAL: {prefull}: {key} N={n} holds {sorted(have)}, neither all {len(want)} pretrained "
                             f"models nor only the equal-stopping test's {PRE_FULL_TEST}")
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
    name = device.replace("NVIDIA ", "").replace("GeForce ", "").replace("Tesla ", "").replace("-SXM2-32GB", "")
    return A100_NAMES.get(name, name)


A100_NAMES = {"A100-PCIE-40GB": "A100 40GB PCIe", "A100-SXM4-80GB": "A100 80GB SXM4", "A100-SXM4-40GB": "A100 40GB SXM4"}


def _whole(name: str) -> str:
    """A GPU's name kept on one line of a wrapping list."""
    return r"\mbox{" + name + "}"


# records kept beside the pulled files that no pull writes: the image provenance (2026-11-14) and the Prometheus snapshots of
# the fine-tuning pods' running hours (taken from the laptop, 2026-11-22 on; ft_hours)
LOCAL_RECORDS = ("ft_mlst/image_provenance.json", "budget/ft_pod_hours_prom_*.json")


def _local_record(rel: pathlib.Path) -> bool:
    return any(rel.match(g) for g in LOCAL_RECORDS)


def check_index(data: pathlib.Path) -> pathlib.Path:
    """index.json against the files beside it: every file listed, with the bytes pulled."""
    import hashlib
    index = data / "index.json"
    if not index.exists():
        raise SystemExit(f"FATAL: no {index}; pull with experiments/FIGS/pull_mlst_data.py")
    want = json.loads(index.read_text())
    have = {str(p.relative_to(data)) for p in data.rglob("*.json") if p != index and not _local_record(p.relative_to(data))}
    if have != set(want):
        raise SystemExit(f"FATAL: {index} lists {len(want)} files, {data} holds {len(have)}: "
                         f"{sorted(have ^ set(want))[:5]}")
    bad = [k for k, h in want.items() if hashlib.sha256((data / k).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} files differ from {index}: {bad[:5]}")
    return index



# ------------------------------------------------------------------ fine-tuning tables and macros (A1, T3, T4)
def _ftv(per: dict, key: str, model: str, n: int, met: str):
    """One pretraining run's fine-tuned value; a rejection as _rej gives it."""
    recs = per[key].get(model[len("mtx-"):], {}).get(n)
    if recs is None:
        return None
    if len(recs) != 1:
        raise SystemExit(f"FATAL: {model} {key} N={n}: {len(recs)} fine-tuning seeds, expected one")
    return _rej(recs[0], met) if met.startswith("r") else recs[0][met]


def _scratch(per: dict, key: str, n: int, met: str) -> list:
    return [_rej(r, met) if met.startswith("r") else r[met] for r in per[key].get(SCRATCH, {}).get(n, [])]


def _sw(key: str, n: int) -> str:
    """Macro suffix of a training size: none at the full set of a benchmark, 10^k as a word below."""
    if key in FULL_N and n >= FULL_N[key]:
        return ""
    return SIZE_WORD.get(n, "Max")


FT_RATIO_N = 1_000_000


def _ft_ratio_cell(per, runs, arm, fmt_pm) -> str:
    """Fine-tuned JetClass-II at FT_RATIO_N jets: 1 - AUC over that of the 188-class run of the same
    index, run by run; the 188-class row prints 1."""
    r, n = _ratio_to_full(None, runs, arm, lambda m: _oma(_ftv(per, "jc2", m, FT_RATIO_N, "macro_auc_ovr")))
    if arm != "L188":
        return _cell(r, n, fmt_pm) if n else "--"
    have = sum(x is not None for x in r)
    return ((_cell([1.0], 1, fmt_pm, run=False) + AGREE + (rf" \pending{{{have} of {n}}}" if have < n else "")) if have
            else r"\pending{no run yet}")       # the runs agree (1 by definition), as tab:mlp and tab:bn print it


FT_FULL_AUC = (("jc1", "macro_auc_ovr", "AUC", "JetClass"), ("top", "auc", "AUC", "top"),
               ("qg", "auc", "AUC", "quark/gluon"), ("qg_herwig", "auc", "AUC", "quark/gluon, Herwig"))


def table_ft(per, runs, fmt_pm, rows, label="tab:ftfull") -> str:
    """A1, fine-tuning against the training-set size, in each task's metric (FT_METRIC): the AUC
    averaged over the classes on class-balanced JetClass-II (and over its QCD classes) and its accuracy, accuracy on
    JetClass, R30 on top, quark/gluon and Herwig; the network trained from scratch, then the
    kinematic-only model in a task where it has cells (top). The
    JetClass-II block adds one column: the ratio of 1 - AUC at FT_RATIO_N jets to the 188-class run.
    One longtable (_task_longtable), a block per task."""
    blocks = []
    tasks = [(key, *FT_METRIC[key], title) for key, _, _, _, title in FT_TASKS]
    tasks.insert(0, ("jc2", "accuracy", "accuracy", "JetClass-II"))      # class-balanced: accuracy first, as everywhere
    for auc in FT_FULL_AUC:     # the AUC of the training-size figures' AUC versions (supplement), beside each quantity
        at = max(i for i, t in enumerate(tasks) if t[0] == auc[0])
        tasks.insert(at + 1, auc)
    for key, met, mname, title in tasks:
        sizes = _sizes(per, key)
        if not sizes:
            blocks.append((title, [["training jets", r"\pending{not yet read out}"]]))
            continue
        ratio = key == "jc2" and met == "macro_auc_ovr"   # the only fine-tuned ratio of 1 - AUC (ROC_INTERFACE 3.3)
        lines = [["training jets"] + [f"{n:,}".replace(",", "{,}") for n in sizes]
                 + ([r"$\rho$"] if ratio else [])]
        for arm in _ft_rows(per, key, runs, rows):
            # the design fine-tunes the random partitions on JetClass-II at FT_RATIO_N jets only: a dash there;
            # any other cell without a run keeps its red marker
            vals = [[_ftv(per, key, m, n, met) for m in runs[arm]] for n in sizes]
            cells = ["--" if not any(x is not None for x in v) and arm in RAND + FLAV and key in ("jc2", "jc2_qcd")
                     and n != FT_RATIO_N else _cell(v, len(runs[arm]), fmt_pm, frac=met != "r30") for n, v in zip(sizes, vals)]
            if ratio:
                cells.append(_ft_ratio_cell(per, runs, arm, fmt_pm))
            lines.append([_row(arm, runs)] + cells)
        if per[key].get(KIN_INIT):        # the kinematic-only model, top only (one run, nothing missing)
            lines.append([LABEL["KIN"]] + [("--" if (v := _ftv(per, key, KIN_MODEL, n, met)) is None else _cell([v], 1, fmt_pm, frac=met != "r30"))
                                      for n in sizes] + ["--"] * ratio)
        lines.append([SCRATCH_LABEL] + [(_cell(v, len(v), fmt_pm, frac=met != "r30") if (v := _scratch(per, key, n, met)) else "--")
                                         for n in sizes] + ["--"] * ratio)
        blocks.append((f"{title}, {mname}", lines))
    return _task_longtable(blocks, label)


def table_ft_rej(per, runs, fmt_pm, rows) -> tuple:
    """A1, full-set fine-tuning on the benchmarks, every configuration: accuracy, AUC, R50 and R30 on
    top, quark/gluon and Herwig; and the fewest background jets passing behind any rejection."""
    fewest = []

    def val(rec, k):
        if k.startswith("r"):
            if rec[f"{k}_is_bound"]:
                raise SystemExit(f"FATAL: a rejection {k} is only a bound: {rec.get('cell', rec.get('run', rec))}")
            fewest.append(rec[f"{k}_n_bkg_pass"])
        return rec[k]
    tasks = ("top", "qg", "qg_herwig")
    lines = [r"\begin{tabular}{l" + "c" * 4 * len(tasks) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{h}}}" for h in ("top", "quark/gluon", "quark/gluon, Herwig"))
             + r" \\", "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tasks))),
             " & " + " & ".join(h for _, h in BENCH4 * len(tasks)) + r" \\", r"\midrule"]
    arms = [a for a in _ft_rows(per, "top", runs, rows)]
    for arm in arms + ["KIN", SCRATCH]:
        cells = []
        for key in tasks:
            if not _sizes(per, key):
                cells += [r"\pending{read-out}"] * 4
                continue
            n = _sizes(per, key)[-1]
            if arm == SCRATCH:
                recs = per[key].get(SCRATCH, {}).get(n, [])
                cells += [_cell([val(r, k) for r in recs], len(recs), fmt_pm, frac=k in ("accuracy", "auc")) if recs else "--" for k, _ in BENCH4]
            elif arm == "KIN":
                recs = per[key].get(KIN_INIT, {}).get(n, []) if key == "top" else []
                cells += [_cell([val(r, k) for r in recs], 1, fmt_pm, frac=k in ("accuracy", "auc")) if recs else "--" for k, _ in BENCH4]
            else:
                recs = [(per[key].get(m[len("mtx-"):], {}).get(n) or [None])[0] for m in runs[arm]]
                cells += [_cell([None if r is None else val(r, k) for r in recs], len(runs[arm]), fmt_pm, frac=k in ("accuracy", "auc")) for k, _ in BENCH4]
        name = SCRATCH_LABEL if arm == SCRATCH else LABEL["KIN"] if arm == "KIN" else _row(arm, runs)
        if arm == "KIN" and not any(c != "--" for c in cells):
            continue
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), (min(fewest) if fewest else None)


FT_R30_CMP = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"))


def _name_over(w: str, arm: str, ref: str) -> str:
    return f"{w}{WORD[arm]}Over{'Full' if ref == 'L188' else WORD[ref]}"


def _equiv_jets(sizes: list, vals: list, target: float) -> str:
    """The number of from-scratch training jets whose metric matches target: linear interpolation in
    the metric between neighbouring sizes on a logarithmic size axis; two significant figures."""
    pts = [(math.log10(n), v) for n, v in zip(sizes, vals) if v is not None]
    if len(pts) < 2:
        return r"\pending{from scratch}"
    if target > pts[-1][1]:
        return r"$>$" + _sep(str(sizes[-1]))
    if target < pts[0][1]:
        return r"$<$" + _sep(str(sizes[0]))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(y0, y1) <= target <= max(y0, y1) and y1 != y0:
            return _sig2(10 ** (x0 + (target - y0) / (y1 - y0) * (x1 - x0)))
    return r"\pending{not monotonic}"


def ft_macros(em, per, runs, index, fmt_pm) -> None:
    """The fine-tuning numbers of the text. JetClass-II: ratios of 1 - AUC between configurations
    (MlstFtCmp...); accuracy and AUC. JetClass: accuracy. Top, quark/gluon, Herwig: R30 of each
    configuration, its ratio to the network trained from scratch (the from-scratch spread in
    quadrature), between configurations, to ParT-f.t., Herwig over Pythia, and the from-scratch training
    jets that fine-tuning on few jets matches. Full set unsuffixed for the benchmarks; 10^k as a word."""
    for key in ("jc2", "jc2_qcd"):
        met = next(t[3] for t in FT_TASKS if t[0] == key)
        for n in _sizes(per, key):
            sw = SIZE_WORD.get(n, "Max")
            f = lambda m, key=key, n=n, met=met: _oma(_ftv(per, key, m, n, met))
            for arm, ref in FT_R30_CMP:
                r, k = _ratio_vals(runs, arm, ref, f)
                _emit_ratio(em, f"MlstFtCmp{_name_over(FT_WORD[key], arm, ref)}{sw}", r, k, fmt_pm, index,
                            f"fine-tuning {key} N={n}: (1 - {met}) of {arm} over {ref}, run by run", "errors",
                            _two_run(runs, arm, ref))
    for key in ("jc2", "jc1"):
        for n in _sizes(per, key):
            sw = _sw(key, n) or SIZE_WORD.get(n, "Max")
            sc = _scratch(per, key, n, "accuracy")
            for arm in MAIN + MASS:
                v = [_ftv(per, key, m, n, "accuracy") for m in runs[arm]]
                em.macro(f"MlstFt{FT_WORD[key]}Acc{WORD[arm]}{sw}", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                         f"fine-tuning {key} N={n}: accuracy, {arm}")
            em.macro(f"MlstFt{FT_WORD[key]}AccScratch{sw}", _cell(sc, len(sc), fmt_pm, sep=TEXT_SEP) if sc
                     else r"\pending{scratch}", index, f"fine-tuning {key} N={n}: accuracy from scratch")
            v = [None if (a := _ftv(per, key, m, n, "accuracy")) is None or not sc else a - float(np.mean(sc))
                 for m in runs["L188"]]
            em.macro(f"MlstFt{FT_WORD[key]}AccFullMinusScratch{sw}", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                     f"fine-tuning {key} N={n}: accuracy of each 188-class run minus the from-scratch mean (absolute)")
        if key == "jc2" and _sizes(per, key):
            n = _sizes(per, key)[-1]
            for w, v in (("Full", [_ftv(per, key, m, n, "macro_auc_ovr") for m in runs["L188"]]),
                         ("Scratch", _scratch(per, key, n, "macro_auc_ovr"))):
                em.macro(f"MlstFtJcTwoAuc{w}", _cell(v, max(len(v), 1), fmt_pm, sep=TEXT_SEP), index,
                         f"fine-tuning class-balanced JetClass-II N={n}: AUC averaged over the classes, {w}")
    lit = json.loads(V4.LIT.read_text())
    part = {t: V4._num(next(r["r30"] for r in lit[s] if r["method"] == "ParT-f.t."))
            for t, s in (("top", "top"), ("qg", "qg_exp"))}
    for key in ("top", "qg", "qg_herwig"):
        tw = FT_WORD[key]
        sizes = _sizes(per, key)
        for n in sizes:
            sw = _sw(key, n)
            ref = _scratch(per, key, n, "r30")
            em.macro(f"MlstFt{tw}RthirtyScratch{sw}", _cell(ref, len(ref), fmt_pm, sep=TEXT_SEP) if ref
                     else r"\pending{scratch}", index, f"fine-tuning {key} N={n}: R30 from scratch, every seed")
            f = lambda m, key=key, n=n: _ftv(per, key, m, n, "r30")
            for arm in MAIN + MASS:
                v = [f(m) for m in runs[arm]]
                em.macro(f"MlstFt{tw}Rthirty{WORD[arm]}{sw}", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                         f"fine-tuning {key} N={n}: R30, {arm}")
                _emit_over_ref(em, f"MlstFt{tw}Rthirty{WORD[arm]}OverScratch{sw}", v, len(v), ref, fmt_pm, index,
                               f"fine-tuning {key} N={n}: R30 of each {arm} run over the from-scratch mean, SD over runs "
                               "and standard error of that mean in quadrature", "rejection", _two_run(runs, arm))
            for arm, rf in FT_R30_CMP:
                r, k = _ratio_vals(runs, arm, rf, f)
                _emit_ratio(em, f"MlstFt{_name_over(tw + 'Rthirty', arm, rf)}{sw}", r, k, fmt_pm, index,
                            f"fine-tuning {key} N={n}: R30 of {arm} over {rf}, run by run", "rejection",
                            _two_run(runs, arm, rf))
            for rf in ("L188", "R16_Q1"):
                r = _rand_over(runs, rf, f)
                if any(x is not None for x in r):
                    _emit_ratio(em, f"MlstFt{tw}RthirtyRandOver{WORD[rf]}{sw}", r, len(r), fmt_pm, index,
                                f"fine-tuning {key} N={n}: R30 of each random partition over the {rf} run of its seed",
                                "rejection")
        if not sizes:
            continue
        n = sizes[-1]
        full = [_ftv(per, key, m, n, "r30") for m in runs["L188"]]
        if key in part:
            _emit_over_ref(em, f"MlstFt{tw}RthirtyFullOverPart", full, len(full), [part[key]], fmt_pm, index,
                           f"fine-tuning {key} N={n}: R30 of each 188-class run over ParT-f.t.'s published value",
                           "rejection")
            ref = per[key].get(SCRATCH, {}).get(n, [])
            em.macro(f"MlstFt{tw}AucFull", _cell([_ftv(per, key, m, n, "auc") for m in runs["L188"]], 3, fmt_pm,
                                                 sep=TEXT_SEP), index, f"fine-tuning {key} N={n}: AUC of the 188-class runs")
            em.macro(f"MlstFt{tw}AucScratch", _cell([r["auc"] for r in ref], len(ref), fmt_pm, sep=TEXT_SEP)
                     if ref else r"\pending{scratch}", index, f"fine-tuning {key} N={n}: AUC from scratch")
        if key == "top":
            have = [x for x in full if x is not None]
            em.macro("MlstFtTopRthirtyFullRelSd", f"{100 * np.std(have, ddof=1) / np.mean(have):.0f}\\%" if len(have) > 2
                     else r"\pending{runs}", index, "fine-tuning top, full set: relative SD of R30 over the 188-class runs")
            kin = (per[key].get(KIN_INIT, {}).get(n) or [None])[0]
            kv = _rej(kin, "r30")
            em.macro("MlstFtTopRthirtyKin", _lone(kv) if kv is not None else r"\pending{kinematic-only run}", index,
                     "fine-tuning top, full set: R30 of the kinematic-only model (one run)")
            val = [kv / float(np.mean(have))] if kv is not None and have else []
            _emit_ratio(em, "MlstFtTopRthirtyKinOverFull", val, 1, fmt_pm, index,
                        "fine-tuning top, full set: R30 of the kinematic-only model over the mean of the 188-class runs",
                        "rejection", body=_lone(val[0]) if val else r"\pending{kinematic-only run}")
        if key in ("top", "qg"):
            sc_mean = [float(np.mean(v)) if (v := [x for x in _scratch(per, key, k, "r30") if x is not None]) else None
                       for k in sizes]
            for small in (1000, 10000):
                if small not in sizes:
                    continue
                v = [x for x in (_ftv(per, key, m, small, "r30") for m in runs["L188"]) if x is not None]
                em.macro(f"MlstFt{tw}DataEquivJets{SIZE_WORD[small]}" if small != 1000 else f"MlstFt{tw}DataEquivJets",
                         _equiv_jets(sizes, sc_mean, float(np.mean(v))) if v else r"\pending{runs}", index,
                         f"fine-tuning {key}: from-scratch training jets whose R30 equals the 188-class mean at {small} jets "
                         "(interpolated linearly in R30 against log10 N)")
    if _sizes(per, "qg") and _sizes(per, "qg_herwig"):
        n = _sizes(per, "qg")[-1]
        for arm in MAIN + MASS:
            r = [None if (h := _ftv(per, "qg_herwig", m, n, "r30")) is None or not (p := _ftv(per, "qg", m, n, "r30"))
                 else h / p for m in runs[arm]]
            _emit_ratio(em, f"MlstFtQgRthirtyHerwigOverPythia{WORD[arm]}", r, len(r), fmt_pm, index,
                        f"fine-tuning q/g N={n}: R30 on Herwig over R30 on Pythia, each run of {arm}", "rejection",
                        _two_run(runs, arm))
        h, p = _scratch(per, "qg_herwig", n, "r30"), _scratch(per, "qg", n, "r30")
        r = [a / b for a, b in zip(h, p) if a is not None and b] if len(h) == len(p) else []
        _emit_ratio(em, "MlstFtQgRthirtyHerwigOverPythiaScratch", r, max(len(p), 1), fmt_pm, index,
                    f"fine-tuning q/g N={n}: R30 on Herwig over R30 on Pythia, each from-scratch seed", "rejection")


def gap_macros(em, fits, per, runs, index, fmt_pm) -> None:
    """MlstLp<task>GapFrac<size>: the share of the from-scratch-to-fine-tuned accuracy gap that the
    linear probe on the frozen features recovers, (probe - scratch) / (fine-tuned - scratch), per
    188-class run, with the from-scratch mean (F7)."""
    for key, ds, w in (("jc1", "jc1", "JcOne"), ("jc2", "jc2_188", "JcTwo")):
        for n in (10000, 100000):
            sc = [x for x in _scratch(per, key, n, "accuracy") if x is not None] if per.get(key) else []
            v = []
            for m in runs["L188"]:
                p = _probe_rec(fits, m, ds, str(n))
                f = _ftv(per, key, m, n, "accuracy") if per.get(key) else None
                s = float(np.mean(sc)) if sc else None
                v.append(None if p is None or f is None or s is None or f == s else 100 * (p["accuracy"] - s) / (f - s))
            _emit_pct(em, f"MlstLp{w}GapFrac{SIZE_WORD[n]}", v, len(v), fmt_pm, index,
                      f"{ds} N={n}: (probe - scratch) / (fine-tuned - scratch) accuracy, 188-class runs, per cent")


FT_PAPER_MANIFESTS = ("best70/leg_top/", "best70/leg_qg/")     # ft_mlst/manifests.json: the paper's fine-tuning cells


def ft_gpu_types(data: pathlib.Path) -> tuple[dict, dict]:
    """({init: sorted GPU names}, {init: fine-tuning cells whose read-out records no device}) of the
    fine-tuning cells (ft_mlst/manifests.json and the budget cells). budget_metrics.py records the
    device only for the top and quark/gluon cells (bench_metrics.cell_metrics), so the JetClass and
    class-balanced JetClass-II cells count as unrecorded until it records theirs."""
    out, nodev = collections.defaultdict(set), collections.Counter()
    f = data / "ft_mlst" / "manifests.json"
    if f.exists():      # audit 3 (2026-11-14): only the cells the paper reads, top and quark/gluon at best70
        for k, c in json.loads(f.read_text()).items():
            if k.startswith(FT_PAPER_MANIFESTS):
                out[c["init"]].add(gpu_name(c["gpu_device_name"]))
    long = data / "budget" / "budget_long_metrics.json"     # JetClass (and JetClass-II where done) at 1e6, as read
    if long.exists():
        for k, c in json.loads(long.read_text())["cells"].items():
            if (g := c.get("gpu") or c.get("manifest", {}).get("gpu_device_name")):
                out[k.split("/")[1]].add(gpu_name(g))
    full = [p for f in SCRATCH_FULL if (p := data / "budget" / f).exists()]
    pre = data / "budget" / PRE_FULL
    pre_cells = json.loads(pre.read_text())["cells"] if pre.exists() else {}
    old = data / "budget" / "budget_metrics.json"
    old_cells = json.loads(old.read_text())["cells"] if old.exists() else {}
    where = lambda cells: {(k.split("/")[0], k.split("/")[2]): set() for k in cells}
    held, have = where(old_cells), where(pre_cells)
    for cells, by in ((old_cells, held), (pre_cells, have)):
        for k in cells:
            if k.split("/")[1] != BUDGET_SCRATCH:
                by[(k.split("/")[0], k.split("/")[2])].add(k.split("/")[1])
    replaced = {w for w, inits in have.items() if inits and inits == held.get(w)}    # merge_prefull's rule
    for b in [old] + full + ([pre] if pre_cells else []):
        if not b.exists():
            continue
        for k, c in json.loads(b.read_text())["cells"].items():
            if full and b not in full and b != pre and k.split("/")[1] == BUDGET_SCRATCH:
                continue        # replaced by the full-schedule from-scratch cells (merge_budget)
            if b == old and (k.split("/")[0], k.split("/")[2]) in replaced and k.split("/")[1] != BUDGET_SCRATCH:
                continue        # replaced by the pretrained cells run to their last epoch (merge_prefull)
            g = c.get("gpu") or c.get("manifest", {}).get("gpu_device_name")
            if g:
                out[k.split("/")[1]].add(gpu_name(g))
            else:
                nodev[k.split("/")[1]] += 1
    return {k: sorted(v) for k, v in out.items()}, dict(nodev)


# ------------------------------------------------------------------ JetClass per class (T2, A1)
JC1_WORD = {"label_Hbb": "Hbb", "label_Hcc": "Hcc", "label_Hgg": "Hgg", "label_H4q": "Hfourq", "label_Hqql": "Hqql",
            "label_Tbqq": "Tbqq", "label_Tbl": "Tbl", "label_Wqq": "Wqq", "label_Zqq": "Zqq"}
JC1_N = "N1000000"


def _jc1_block(pre: dict, m: str):
    return pre.get(m[len("mtx-"):], {}).get(JC1_N, {}).get("s1", {}).get("jc1")


def jc1_full(data: pathlib.Path, runs: dict):
    """(pre, ref, source) of ft_curves.py's JetClass cells at 1e6: the long-budget tree's
    (curves/full_budget_long_leg2.json, its from-scratch cells under SCRATCH) once it holds every paper run
    and every from-scratch seed of V4.JC1_LONG_SCRATCH, else the old pretrained and from-scratch trees
    (one recipe per table); None when neither is pulled. With the long tree, the full-schedule from-scratch
    cells (curves/full_budget_long_fulldecay_leg2.json) replace its from-scratch ones once pulled, FATAL when
    one of those seeds is missing."""
    f = data / "curves" / "full_budget_long_leg2.json"
    if f.exists():
        cells = json.loads(f.read_text())["cells"]
        pre, ref = {i: v for i, v in cells.items() if i != "scratch"}, {SCRATCH: cells.get("scratch", {})}
        if all(_jc1_block(pre, m) for m in _pretrained(runs)) and \
                all(ref[SCRATCH].get(JC1_N, {}).get(s, {}).get("jc1") for s in V4.JC1_LONG_SCRATCH):
            sf = data / "curves" / "full_budget_long_fulldecay_leg2.json"
            if sf.exists():
                ref = {SCRATCH: json.loads(sf.read_text())["cells"].get("scratch", {})}
                if not all(ref[SCRATCH].get(JC1_N, {}).get(s, {}).get("jc1") for s in V4.JC1_LONG_SCRATCH):
                    raise SystemExit(f"FATAL: {sf} lacks a from-scratch seed of {V4.JC1_LONG_SCRATCH}")
            return pre, ref, f
    full = [data / "curves" / f"full_{t}.json" for t in ("best70_leg2", "ref_v100_jc1_leg2")]
    if not all(f.exists() for f in full):
        return None
    pre, ref = (json.loads(f.read_text())["cells"] for f in full)
    return pre, ref, full[0]


def _jc1_rej(pre: dict, m: str, c: str):
    b = _jc1_block(pre, m)
    return None if b is None else b["per_class"][c]["rejection"]


def jetclass_macros(em, pre: dict, runs: dict, src, fmt_pm, words) -> None:
    """MlstFtJcOneRej<class><arm>Over<ref>: the fine-tuned rejection of a JetClass signal class at
    ParT's working point, run by run; ...RandOverFull over the partitions (and its range);
    MlstFtJcOneAccRandMinusCoarse. No count of classes that differ: the 2-sigma-and-counting rule is gone (PI
    2026-11-23, LEDGER SP-12 step 3); a difference is stated only from its tab:tests row."""
    em.macro("MlstFtJcOneNSignal", words(len(V4.JC1_COLS)), src, "JC1_COLS (count)", "JetClass signal classes")
    for c, _ in V4.JC1_COLS:
        cw = JC1_WORD[c]
        f = lambda m, c=c: None if (x := _jc1_rej(pre, m, c)) is None else x["rejection"]
        for arm, ref in FT_R30_CMP:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstFtJcOneRej{cw}{WORD[arm]}Over{'Full' if ref == 'L188' else WORD[ref]}", r, k, fmt_pm,
                        src, f"cells.<run>.N1000000.s1.jc1.per_class.{c}: rejection of {arm} over {ref}, run by run",
                        "rejection", _two_run(runs, arm, ref))
        r = _rand_over(runs, "L188", f)
        _emit_ratio(em, f"MlstFtJcOneRej{cw}RandOverFull", r, len(r), fmt_pm, src,
                    f"per_class.{c}: rejection of each random partition over the 188-class run of its seed", "rejection")
        v = [x for x in r if x is not None]
        em.macro(f"MlstFtJcOneRej{cw}RandOverFullRange", f"{_sig2(min(v))}--{_sig2(max(v))}" if v else r"\pending{runs}",
                 src, f"per_class.{c}: smallest and largest of the partitions' ratios")
    acc = lambda ms: [b["accuracy"] for m in ms if (b := _jc1_block(pre, m)) is not None]
    a, b = acc([m for x in RAND for m in runs[x]]), acc(runs["R16_Q1"])
    em.macro("MlstFtJcOneAccRandMinusCoarse", f"{np.mean(a) - np.mean(b):+.3f}" if a and b else r"\pending{runs}", src,
             "cells.*.N1000000.s1.jc1.accuracy", "mean over the random partitions minus the mean over the 17-class runs")


# (fine-tuning jets, macro suffix): 1e6 for the JetClass statement, 1e4 for the advice line (PI 2026-11-09)
JC1_BOOT_SIZES = ((1_000_000, ""), (10_000, "EFour"))
JC1_BOOT_ARM = {"Res": "l162-l188", "Mid": "r42q1-l188", "Coarse": "r16q1-l188"}


def jc1_boot_macros(em, data: pathlib.Path, fmt_int, missing: list) -> None:
    """The JetClass accuracy against the 188-class model at each size of JC1_BOOT_SIZES (experiments/FT/jc1_acc_boot.py,
    PI 2026-11-09): a difference stands when |delta| > 3 sigma, sigma^2 = run SD^2 + bootstrap SD^2 over the test jets.
    MlstFtJcOneAcc{Full,Coarse}Boot<sfx>: the mean accuracies; MlstFtJcOneErr<arm>OverFullBoot<sfx>: the share of more
    errors with sigma, or 'consistent' where it does not pass; ...Z<sfx>: |delta| / sigma; MlstFtJcOneBootReplicas,
    ...BootJets (from the 1e6 document)."""
    for n, sfx in JC1_BOOT_SIZES:
        f = data / "budget" / f"jc1_acc_boot_N{n}.json"
        if f.exists():
            _jc1_boot(em, f, sfx, fmt_int, emit_common=not sfx)
        else:
            missing.append(f"JetClass accuracy bootstrap -- experiments/FIGS/data/v2/budget/{f.name}")


def _jc1_boot(em, f: pathlib.Path, sfx: str, fmt_int, emit_common: bool) -> None:
    doc = json.loads(f.read_text())
    for name, d in doc["diffs"].items():
        for q in ("accuracy", "error_ratio"):
            x = d[q]
            if not math.isclose(x["sigma"], math.hypot(x["run_sd"], x["boot_sd"]), rel_tol=1e-9) \
                    or x["pass"] != (abs(x["delta"]) > 3 * x["sigma"]):
                raise SystemExit(f"FATAL: {f.name} {name} {q}: sigma is not hypot(run SD, bootstrap SD) "
                                 "or pass is not |delta| > 3 sigma")
    c = doc["diffs"][JC1_BOOT_ARM["Coarse"]]
    em.macro("MlstFtJcOneAccFullBoot" + sfx, f"{np.mean(c['accuracy_ref']):.3f}", f, "diffs.r16q1-l188.accuracy_ref (mean)")
    em.macro("MlstFtJcOneAccCoarseBoot" + sfx, f"{np.mean(c['accuracy_config']):.3f}", f,
             "diffs.r16q1-l188.accuracy_config (mean)")
    for arm, key in JC1_BOOT_ARM.items():
        e = doc["diffs"][key]["error_ratio"]
        sig = float(f"{100 * e['sigma']:.1g}")
        dec = max(0, -math.floor(math.log10(sig)))
        body = (rf"({100 * e['delta']:.{dec}f}\,\ensuremath{{\pm}}\,{sig:.{dec}f})\% more errors" if e["pass"] and e["delta"] > 0
                else rf"({100 * -e['delta']:.{dec}f}\,\ensuremath{{\pm}}\,{sig:.{dec}f})\% fewer errors" if e["pass"]
                else "consistent")
        em.macro(f"MlstFtJcOneErr{arm}OverFullBoot{sfx}", body, f, f"diffs.{key}.error_ratio",
                 "delta and sigma = hypot(run SD, bootstrap SD) in per cent; 'consistent' when |delta| <= 3 sigma")
        em.macro(f"MlstFtJcOneErr{arm}OverFullZ{sfx}", f"{abs(e['z']):.1f}", f, f"diffs.{key}.error_ratio.z", "|delta| / sigma")
    if emit_common:
        em.macro("MlstFtJcOneBootReplicas", fmt_int(doc["replicas"]), f, "replicas")
        em.macro("MlstFtJcOneBootJets", fmt_int(doc["n_test_jets"]), f, "n_test_jets")


def table_jetclass_probe(fits, runs, fmt_pm, rows) -> str:
    """A1, JetClass linear probes in the form of ParT's Table 1: accuracy, AUC averaged over the classes, each
    against the rest (the key tab:probes reads; the stored one-vs-one value comes from float16 probabilities and
    differs by at most 3e-7), and each signal class's rejection of QCD at ParT's working point."""
    idx = {c: i for i, c in enumerate(V4.LPV_CLASSES)}
    head = [r"\begin{tabular}{l" + "c" * (2 + len(V4.JC1_COLS)) + "}", r"\toprule",
            r" & & & \multicolumn{" + str(len(V4.JC1_COLS)) + r"}{c}{rejection of QCD} \\",
            r"\cmidrule(lr){4-" + str(3 + len(V4.JC1_COLS)) + "}",
            r" & accuracy & AUC & " + " & ".join(h for _, h in V4.JC1_COLS) + r" \\",
            r" & & & " + " & ".join(r"$R_{%g}$" % (100 * V4.LPV_EPS.get(c, 0.5)) for c, _ in V4.JC1_COLS) + r" \\",
            r"\midrule"]
    out = list(head)
    # L5 (round-1 fix 2026-11-13): the five random partitions as one row, mean +- SD over the five, under the
    # partitions' subheading, as in tab:jetclassall; each partition's own value is in tab:perrun
    runs = _pooled(runs)
    rows = tuple(a for a in rows if a not in RAND)
    rows = tuple(a for a in rows if a not in FLAV) if any(a in FLAV for a in rows) else rows
    at = rows.index(INIT) if INIT in rows else len(rows)
    rows = rows[:at] + ((PARTITIONS,) + FLAV + (RAND_POOLED,)) + rows[at:]
    for arm in rows:
        if arm == PARTITIONS:
            out.append(_full_row(3 + len(V4.JC1_COLS), PARTITIONS))
            continue
        recs = [_probe_rec(fits, m, "jc1") for m in runs[arm]]
        cells = [_cell([None if r is None else r["accuracy"] for r in recs], len(recs), fmt_pm, frac=True),
                 _cell([None if r is None else r["macro_auc_ovr"] for r in recs], len(recs), fmt_pm, frac=True)]
        for c, _ in V4.JC1_COLS:
            w = [None if r is None else r["rejection_vs_qcd"].get(str(idx[c])) for r in recs]
            mark = r"$^\dagger$" if any(x is not None and x["n_bkg_pass"] < V4.JC1_MIN_BKG for x in w) else ""
            nb = [None if r is None else V4.background_count(r["rejection_vs_qcd"].values()) for r in recs]
            cells.append(_cell([None if x is None else V4.rejection(x["rejection"], x.get("is_bound"), x.get("n_bkg_pass"), n)
                                for x, n in zip(w, nb)], len(recs), fmt_pm) + mark)
        name = (r"\quad " if arm in FLAV + (RAND_POOLED,) else "") + _row(arm, runs)
        out += [r"\midrule"] * (arm == SOPHON) + [name + " & " + " & ".join(cells) + r" \\"]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ per class on JetClass-II (A12, F3)
def _orbits(root: pathlib.Path) -> tuple[dict, dict]:
    """({class: its quark-flavor orbit}, {class: its lepton-flavor key}): two resonant classes differ
    only in quark flavor when they share an orbit of flavour_pair_map.v2.csv, and only in lepton
    flavor (e and mu, tau_e and tau_mu) when their names agree with those replaced by one lepton."""
    q, lep = {}, {}
    tok = re.compile(r"tauh|taue|taum|[bcsqgemvQ]")
    lmap = {"e": "L", "m": "L", "taue": "tauL", "taum": "tauL"}
    for r in csv.DictReader((root / "configs/labelmaps/flavour_pair_map.v2.csv").open()):
        n = r["class_name"]
        if n.startswith("label_QCD"):
            continue
        q[n] = r["orbit"]
        head, _, prods = n.removeprefix("label_").rpartition("_")
        lep[n] = head + "_" + "".join(lmap.get(t, t) for t in tok.findall(prods))
    return q, lep


def _class_rows(root: pathlib.Path) -> list:
    """The 188 classes in the order of F3: by 17-class group, decays that differ only in flavor adjacent."""
    rows = list(csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open()))
    q, _ = _orbits(root)
    return sorted(rows, key=lambda r: (int(r["R16_Q1"]), q.get(r["class_name"], r["class_name"]), int(r["jet_label"])))


def _per_class(fits, m: str) -> dict | None:
    r = _probe_rec(fits, m, "jc2_188", PAIR_N)
    return None if r is None else {int(c): a for c, a in r["per_class_auc"].items()}


def _nat_as_fits(nat: dict) -> dict:
    """The natural-composition nonlinear probes in the layout of the linear fits, for per_class_ratios."""
    return {m: {"cells": {ck: {"jc2_188": {PAIR_N: c["jc2_188"]}} for ck, c in d["cells"].items() if "jc2_188" in c}}
            for m, d in nat.items()}


def per_class_ratios(fits, runs, arm) -> dict:
    """{class: [ratio of 1 - AUC to the 188-class run of the same index, per run]}."""
    out = collections.defaultdict(list)
    full = {_idx(m): _per_class(fits, m) for m in runs["L188"]}
    for m in runs[arm]:
        a, b = _per_class(fits, m), full.get(_idx(m))
        if a is None or b is None:
            continue
        for c, x in a.items():
            if c in b and b[c] < 1:
                out[c].append((1 - x) / (1 - b[c]))
    return out


def per_class_auc_diff(fits, runs, arm) -> dict:
    """{native label: [100 * (AUC_c of the run - AUC_c of the 188-class run of the same index), per index]}:
    the per-class AUC difference of F3, in percentage points."""
    out = collections.defaultdict(list)
    full = {_idx(m): _per_class(fits, m) for m in runs["L188"]}
    for m in runs[arm]:
        a, b = _per_class(fits, m), full.get(_idx(m))
        if a is None or b is None:
            continue
        for c, x in a.items():
            if c in b:
                out[c].append(100 * (x - b[c]))
    return dict(out)


def jc2_roc_classes(fits, runs, root, census) -> dict:
    """The classes of A3R: {"hurt": [NHURT labels], "stable": [NSTABLE labels], "drop": {c: pp},
    "change": {c: pp}, "ratio_dev": {c: fraction}}. drop_c = mean over indices of 100 (AUC_c^188 -
    AUC_c^17); change_c = max over the 162-, 43- and 17-class label sets of |mean over indices of
    100 (AUC_c^K - AUC_c^188)|; ratio_dev_c = max over the same label sets of |mean over indices of
    (1 - AUC_c^K) / (1 - AUC_c^188) - 1|. Candidates: classes with at least THIN_TEST class-balanced
    test jets (census["counts"]["ft_test"]; every class when census is None). hurt = the NHURT largest
    drop_c; stable = the NSTABLE smallest ratio_dev_c among the rest, not change_c: X->ee and X->mumu
    change least in AUC, since theirs is near 1, while their rejection falls threefold (seen in the
    first A3R, 2026-11-03). Each list in F3's row order (_class_rows, resonant before QCD)."""
    mean = {a: {c: float(np.mean(v)) for c, v in per_class_auc_diff(fits, runs, a).items() if v}
            for a in ("L162", "R42_Q1", "R16_Q1")}
    drop = {c: -v for c, v in mean["R16_Q1"].items()}
    change = {c: max(abs(mean[a][c]) for a in mean) for c in drop if all(c in mean[a] for a in mean)}
    ratio = {a: {c: float(np.mean(v)) for c, v in per_class_ratios(fits, runs, a).items() if v} for a in mean}
    dev = {c: max(abs(ratio[a][c] - 1) for a in ratio) for c in change if all(c in ratio[a] for a in ratio)}
    counts = census["counts"]["ft_test"] if census else None
    order = {int(r["jet_label"]): i for i, r in enumerate(_class_rows(root))}
    cand = sorted((c for c in dev if counts is None or counts[c] >= THIN_TEST), key=order.get)
    hurt = sorted(cand, key=lambda c: -drop[c])[:NHURT]
    stable = sorted((c for c in cand if c not in hurt), key=lambda c: dev[c])[:NSTABLE]
    return {"hurt": sorted(hurt, key=order.get), "stable": sorted(stable, key=order.get), "drop": drop, "change": change,
            "ratio_dev": dev}


def _and(items: list) -> str:
    """'a', 'a and b', 'a, b and c'; semicolons between items that hold a comma themselves."""
    sep = "; " if any("," in x for x in items) else ", "
    return items[0] if len(items) == 1 else sep.join(items[:-1]) + " and " + items[-1]


def auc_diff_macros(em, fits, runs, root, census, index, native_tex) -> None:
    """The per-class AUC differences of F3 and the class selection of A3R, in percentage points:
    MlstPerClassAucDiff{W}SeedSdMedian (median over classes of the SD over indices), ...{W}Min and
    ...{W}MinClass (the most negative mean difference and its class); MlstRocJcTwo{Hurt,Stable}List,
    ...HurtMinDrop, ...HurtMaxDrop, ...StableMaxChange (jc2_roc_classes)."""
    names = {int(r["jet_label"]): r["class_name"] for r in _class_rows(root)}
    for arm in ("L162", "R42_Q1", "R16_Q1"):
        w = WORD[arm]
        d = per_class_auc_diff(fits, runs, arm)
        sds = [float(np.std(v, ddof=1)) for v in d.values() if len(v) > 1]
        if sds:
            em.macro(f"MlstPerClassAucDiff{w}SeedSdMedian", _sig2(float(np.median(sds))), index,
                     f"median over classes of the SD over indices of 100 (AUC_c of {arm} - AUC_c of 188), pp")
        mean = {c: float(np.mean(v)) for c, v in d.items() if v}
        if mean:
            low = min(mean, key=mean.get)
            em.macro(f"MlstPerClassAucDiff{w}Min", _sig2(mean[low]), index,
                     f"most negative mean over indices of 100 (AUC_c of {arm} - AUC_c of 188), pp", names[low])
            em.macro(f"MlstPerClassAucDiff{w}MinClass", native_tex(names[low]), index, f"its class, {arm}")
    sel = jc2_roc_classes(fits, runs, root, census)
    if not sel["hurt"]:
        return
    src = f"jc2_roc_classes on the class-balanced probes; candidates with at least {THIN_TEST} test jets"
    for k, word in (("hurt", "Hurt"), ("stable", "Stable")):
        em.macro(f"MlstRocJcTwo{word}List", _and([native_tex(names[c]) for c in sel[k]]), index, src,
                 ", ".join(names[c] for c in sel[k]))
    drops = [sel["drop"][c] for c in sel["hurt"]]
    em.macro("MlstRocJcTwoHurtMinDrop", _sig2(min(drops)), index, src, "smallest AUC loss at 17 classes of the hurt, pp")
    em.macro("MlstRocJcTwoHurtMaxDrop", _sig2(max(drops)), index, src, "largest AUC loss at 17 classes of the hurt, pp")
    if sel["stable"]:
        em.macro("MlstRocJcTwoStableMaxChange", _sig2(100 * max(sel["ratio_dev"][c] for c in sel["stable"])), index, src,
                 "largest change of 1 - AUC at any label set of the stable classes, per cent of the 188-class value")


# ------------------------------------------------------------------ ROC curves (F4, F6J) and the X->ee vs X->mumu probe
JC1_WORD = {"label_Hbb": "Hbb", "label_Hcc": "Hcc", "label_Hgg": "Hgg", "label_H4q": "Hfourq", "label_Hqql": "Hqql",
            "label_Tbqq": "Tbqq", "label_Tbl": "Tbl", "label_Wqq": "Wqq", "label_Zqq": "Zqq"}
LEVEL_TEX = {"L188": r"\Kfull", "L162": r"\Kres", "R42_Q1": r"\Kmid", "R16_Q1": r"\Kcoarse"}
FLAV_PHRASE = {"FLAV_F0": "no flavor boundary", "FLAV_F1": r"a $b$-quark boundary", "FLAV_F1R": "a random flavor boundary"}


def _roc_curves(fam: dict, ms: list, key: str) -> list:
    """The curve of key of each model; None where the model has no file or no such curve."""
    return [fam.get(m, {}).get("curves", {}).get(key) for m in ms]


# grid points at or above its cut a run's ROC curve needs to be drawn (one point draws no line);
# paper_figures reads it, so the figures and the 'not reached' of _roc_cut agree
ROC_MIN_POINTS = 2


def _roc_cut(curves: list, grid: list) -> str | None:
    """The signal efficiency above which every run's curve is drawn: the largest cut.eps_s, two
    decimals; 'not reached' when a run has fewer than ROC_MIN_POINTS grid points with JC1_MIN_BKG
    background jets, i.e. its cut is null or fewer grid signal efficiencies lie at or above it (the
    figure's rule: a configuration is drawn only where every run survives, so one such run removes its
    curve); None without curves."""
    have = [c for c in curves if c is not None]
    if not have:
        return None
    gone = any(c["cut"] is None or sum(e >= c["cut"]["eps_s"] for e in grid) < ROC_MIN_POINTS for c in have)
    body = "not reached" if gone else f"{max(c['cut']['eps_s'] for c in have):.2f}"
    return body + (rf" \pending{{{len(have)} of {len(curves)}}}" if len(have) < len(curves) else "")


def no_curve_phrase(arms: list) -> str:
    """The configurations of F4 that draw no curve, as a plural noun phrase for prose and captions."""
    lv = [LEVEL_TEX[a] for a in MAIN if a in arms]
    fl = [a for a in FLAV if a in arms]
    parts = []
    if lv:
        parts.append("the " + _and([x + "-" for x in lv[:-1]] + [lv[-1] + "-class model" + "s" * (len(lv) > 1)]))
    if len(fl) == len(FLAV):
        parts.append("the three random partitions with controlled flavor boundaries")
    elif fl:
        parts.append("the random partition" + "s" * (len(fl) > 1) + " with " + " and with ".join(FLAV_PHRASE[a] for a in fl))
    return " and ".join(parts)


def top_r50_macro(em, roc: dict, runs: dict, index, fmt_pm) -> None:
    """MlstFtTopRfiftyCoarseOverFull: the fine-tuned rejection on top at 50% signal efficiency, 17-class
    over 188-class, run by run, from the ROC curves; a run whose curve stops above that efficiency is missing."""
    top = roc.get("top_ft", {})
    if not top:
        return

    def r50(m):
        s = V4.roc_series(top[m], "top") if m in top else None
        if s is None or not np.any(np.isclose(s[0], 0.5)):
            return None
        x = float(s[1][np.isclose(s[0], 0.5)][0])
        return None if math.isnan(x) else x
    r, k = _ratio_vals(runs, "R16_Q1", "L188", r50)
    _emit_ratio(em, "MlstFtTopRfiftyCoarseOverFull", r, k, fmt_pm, index,
                "curves_v2/top_ft curve top: rejection at eps_S = 0.5 of R16_Q1 over L188, run by run", "rejection")


def roc_macros(em, roc: dict, runs: dict, index, missing: list, fmt_int) -> None:
    """From the ROC curves (mlst_v4.load_roc): MlstRocCut{pair}{W} and MlstRocCutJcOne{class}Full (_roc_cut
    over a configuration's runs, against the family's grid; load_roc requires one grid per
    family), MlstRocNBkg{pair} and MlstRocRejMax{pair} (the test set's background jets and that over
    JC1_MIN_BKG, where every curve stops), MlstRocNoCurveEemm (the configurations of F4 that draw no
    X->ee vs X->mumu curve, by F4's rule: _roc_cut 'not reached', i.e. some run has fewer than
    ROC_MIN_POINTS grid points with JC1_MIN_BKG background jets; a noun phrase). The all-runs-null rule of ROC_INTERFACE 3.2 was
    replaced by the figure's rule (2026-11-02, to be flagged to the PI)."""
    def family(key):
        """X->ee vs X->mumu from every test jet of the pair (pairs_sib2) once read out, else the balanced set."""
        fam = roc.get("pairs_sib2", {}) if key == "ee_vs_mm" else {}
        fam = fam or roc.get("pairs", {})
        return fam, (next(iter(fam.values()))["eps_grid"] if fam else None)

    for key in ROC_PAIRS:
        p = PAIR_WORD[key]
        pairs, grid = family(key)
        for arm in MAIN + FLAV:
            body = _roc_cut(_roc_curves(pairs, runs[arm], key), grid)
            if body is not None:
                em.macro(f"MlstRocCut{p}{WORD[arm]}", body, index,
                         f"curves_v2/pairs curve {key}: largest cut.eps_s over the {arm} runs "
                         f"('not reached' when a run's cut is null or leaves fewer than {ROC_MIN_POINTS} grid points)")
        c = next((d["curves"][key] for d in pairs.values() if key in d["curves"]), None)
        if c is not None:
            em.macro(f"MlstRocNBkg{p}", fmt_int(c["n_background"]), index, f"curves_v2/pairs curve {key}: n_background")
            em.macro(f"MlstRocRejMax{p}", fmt_int(c["n_background"] / V4.JC1_MIN_BKG), index,
                     f"curves_v2/pairs curve {key}: n_background / {V4.JC1_MIN_BKG}")
    pairs, grid = family("ee_vs_mm")
    cur = {a: _roc_curves(pairs, runs[a], "ee_vs_mm") for a in MAIN + FLAV}
    if pairs and all(c is not None for v in cur.values() for c in v):
        none = [a for a, v in cur.items() if _roc_cut(v, grid).startswith("not reached")]
        if none:
            em.macro("MlstRocNoCurveEemm", no_curve_phrase(none), index,
                     "curves_v2/pairs curve ee_vs_mm: configurations of F4 with a run whose cut is null or leaves "
                     f"fewer than {ROC_MIN_POINTS} grid points (F4 draws no curve for them)", ", ".join(none))
        else:
            em.macro("MlstRocNoCurveEemm", r"\pending{every configuration draws an ee/mumu curve: rewrite the "
                     r"sentences that use this macro}", index, "curves_v2/pairs curve ee_vs_mm: every configuration "
                     "of F4 has a run with a cut")
            missing.append("MlstRocNoCurveEemm -- every configuration of F4 draws an X->ee vs X->mumu curve: the "
                           "results text and the F4 caption that name those without one must be rewritten")
    jc = roc.get("jc1_ft", {})
    jc_grid = next(iter(jc.values()))["eps_grid"] if jc else None
    for c, _ in V4.JC1_COLS:
        body = _roc_cut(_roc_curves(jc, runs["L188"], c), jc_grid)
        if body is not None:
            em.macro(f"MlstRocCutJcOne{JC1_WORD[c]}Full", body, index,
                     f"curves_v2/jc1_ft curve {c}: largest cut.eps_s over the 188-class runs "
                     f"('not reached' when a run's cut is null or leaves fewer than {ROC_MIN_POINTS} grid points)")


# PI 2026-11-23 (LEDGER CA-15, CA-16): the class-pair figures' captions. The random partitions that keep each pair
# apart, counted from the label maps (PS-5), and the random network the figures do not draw, as Fig. 2 treats it (its
# range in one clause, from the files): its rejection at half the signal jets, the mean over its initializations, the
# lowest and the highest over each figure's pairs, at two significant figures as gallery_labelset._init_range
# review of render dfe5942 (C): F4-other split in two, the resonance pair (F4-other) and the three QCD pairs (F4-qcd),
# each caption with the random network's values of its own pairs
PAIR_FIGS = {"Mech": ("bb_vs_cc", "bc_vs_bq_cs"), "Other": ("bbqq_vs_ccqq",),
             "Qcd": ("qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light")}
INIT_EPS = 0.5


def pair_figure_macros(em, data: pathlib.Path, root: pathlib.Path, runs: dict, status: dict, words,
                       missing: list) -> None:
    maps = root / "configs/labelmaps/rand_label_map.v2.csv"
    n = {k: sum(status[a][k] == "apart" for a in RAND) for k, *_ in PAIRS}
    for k, *_ in PAIRS:
        em.macro(f"MlstPairRandApart{PAIR_WORD[k]}Word", words(n[k]), maps, f"{'/'.join(RAND)} groups of {k}",
                 "random partitions that keep the pair apart (count, from the label maps)")
        # review of render 5f4e1b8: F4-mech's counts in words, the partitions that merge the pair and the one that merges
        # it in part (merge_status "partly", drawn in neither curve)
        for s, what in (("merged", "merge the pair"), ("partly", "merge the pair in part")):
            em.macro(f"MlstPairRand{s.title()}{PAIR_WORD[k]}Word", words(sum(status[a][k] == s for a in RAND)), maps,
                     f"{'/'.join(RAND)} groups of {k}", f"random partitions that {what} (count, from the label maps)")
    mech = [n[k] for k in PAIR_FIGS["Mech"]]
    em.macro("MlstPairRandApartMechWord", words(mech[0]) if len(set(mech)) == 1 else " and ".join(map(words, mech)),
             maps, f"{'/'.join(RAND)} groups of {' and '.join(PAIR_FIGS['Mech'])}",
             "random partitions that keep each pair of figure F4-mech apart (count, in panel order if they differ)")
    fam = V4.ROC_RERUN["pairs"] if (data / V4.ROC_DIR / V4.ROC_RERUN["pairs"]).is_dir() else "pairs"
    docs = V4.load_family(data, "pairs")

    def rej(m, k):
        s = V4.roc_series(docs[m], k) if m in docs else None
        if s is None:
            return None
        i = int(np.argmin(np.abs(s[0] - INIT_EPS)))
        return float(s[1][i]) if abs(s[0][i] - INIT_EPS) < 1e-9 and np.isfinite(s[1][i]) else None
    fmt = lambda x: f"{x:#.2g}" if x < 9.95 else f"{x:.0f}"   # noqa: E731  (# keeps a trailing zero: 0.10, 4.0)
    for fig, keys in PAIR_FIGS.items():
        v = [[rej(m, k) for m in runs[INIT]] for k in keys]
        src = data / V4.ROC_DIR / fam / f"{runs[INIT][0]}.json"
        if any(x is None for xs in v for x in xs) or not src.exists():
            missing.append(f"the random network's pair curves ({fig}) -- experiments/FIGS/data/v2/{V4.ROC_DIR}/{fam}/")
            for end in ("Min", "Max"):
                em.macro(f"MlstPairInitRhalf{fig}{end}", r"\pending{random network}", maps, "no curve")
            continue
        means = [float(np.mean(xs)) for xs in v]
        what = (f"curves[{'|'.join(keys)}].rejection at eps_grid {INIT_EPS}, mean over {', '.join(runs[INIT])} "
                f"({fam}/<run>.json)")
        em.macro(f"MlstPairInitRhalf{fig}Min", fmt(min(means)), src, what, "the random network, lowest over the pairs")
        em.macro(f"MlstPairInitRhalf{fig}Max", fmt(max(means)), src, what, "the random network, highest over the pairs")


def eemm_macros(em, fits: dict, runs: dict, index, fmt_pm, fmt_int, status: dict | None = None) -> None:
    """The X->ee vs X->mumu linear probe (fits_v4_sib, merged by mlst_v4.load_fits): MlstLpEemmAuc{W},
    MlstLpEemmAucRandMerged over the random partitions that merge the pair (status: merge_status),
    and its jets, MlstPairEemm{NTrain,NVal,NTestPerSide} (one value when every model agrees, else the range)."""
    for arm in MAIN + FLAV + (INIT, INIT_BN):
        v = [_pair(fits, m, "ee_vs_mm") for m in runs[arm]]
        if any(x is not None for x in v):
            em.macro(f"MlstLpEemmAuc{WORD[arm]}", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                     f"linear probe ee_vs_mm (fits_v4_sib): AUC, {arm}")
    v = [_pair(fits, m, "ee_vs_mm") for arm in RAND if status and status[arm]["ee_vs_mm"] == "merged" for m in runs[arm]]
    if any(x is not None for x in v):
        em.macro("MlstLpEemmAucRandMerged", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                 "linear probe ee_vs_mm (fits_v4_sib): AUC, the random partitions that merge the pair")
    recs = [c for m, d in fits.items()
            if (c := d["cells"].get(_ck(m), {}).get(PAIR_DS, {}).get(PAIR_N, {}).get("ee_vs_mm"))]
    if not recs:
        return
    span = lambda xs: fmt_int(min(xs)) if min(xs) == max(xs) else f"{fmt_int(min(xs))}--{fmt_int(max(xs))}"
    em.macro("MlstPairEemmNTrain", span([c["n_train"] for c in recs]), index, "fits_v4_sib ee_vs_mm: n_train")
    em.macro("MlstPairEemmNVal", span([c["n_val"] for c in recs]), index, "fits_v4_sib ee_vs_mm: n_val")
    em.macro("MlstPairEemmNTestPerSide", span([c[k] for c in recs for k in ("n_test_signal", "n_test_background")]),
             index, "fits_v4_sib ee_vs_mm: n_test_signal and n_test_background")


JC1_AUC_COLS = (("label_QCD", "QCD"),) + V4.JC1_COLS      # ParT's class order, as the other JetClass tables


RECALL_DIR = "perclass_recall"      # experiments/EVAL/perclass_recall.py: probe_<task>.json, ft_<task>_N<n>.json


def _recall(doc: dict, m: str) -> dict | None:
    """{class: recall} of model m in a perclass_recall file (JetClass: class names; JetClass-II: jet_label), None if
    the file lacks it."""
    r = (doc or {}).get("models", {}).get(m)
    if r is None:
        return None
    return {(doc["classes"][k] if doc.get("classes") else k): v for k, v in enumerate(r["recall"])}


# PI 2026-11-21: Table A2 marks each class whose recall lies more than 3 sigma below or above 188 classes (the
# perclass_recall differences: run spreads and the paired test-jet bootstrap, decision L3)
RECALL_LOW, RECALL_HIGH = r"$^{\downarrow}$", r"$^{\uparrow}$"
RECALL_DIFF_KEY = {RAND_POOLED: "RAND"}


def recall_marks(doc: dict, arm) -> tuple:
    """(class indices lower, higher) beyond 3 sigma than 188 classes for arm, in the file's class order mapped to
    JC1_AUC_COLS; empty for 188 classes itself and for the references."""
    d = ((doc or {}).get("diffs") or {}).get(f"{RECALL_DIFF_KEY.get(arm, arm)}-L188") if arm else None
    if not d:
        return set(), set()
    order = [doc["classes"].index(c) for c, _ in JC1_AUC_COLS]
    return ({order.index(i) for i in d["lower"]}, {order.index(i) for i in d["higher"]})


def jc1_recall_macros(em, doc: dict, arms, src: pathlib.Path) -> None:
    """The sentence of Sec. 4.2 on tab:jc1_class_auc (PI 2026-11-23, LEDGER SP-9: its comparison passes 3 sigma), from the
    linear-probe file's differences against 188 classes (sigma^2 = each side's run SD^2 + the paired test-jet bootstrap
    SD^2, experiments/EVAL/perclass_recall.py): the network-class cells of arms below and above 188 classes by more than
    3 sigma, of how many, the 17-class network's cells below, and the largest drop with its network and class."""
    names = dict(JC1_AUC_COLS)
    low = high = n = 0
    worst = None
    for a in arms:
        d = (doc.get("diffs") or {}).get(f"{RECALL_DIFF_KEY.get(a, a)}-L188")
        if not d:
            continue
        n += len(JC1_AUC_COLS)
        low, high = low + len(d["lower"]), high + len(d["higher"])
        for i in d["lower"]:
            if worst is None or d["delta"][i] < worst[0]:
                worst = (d["delta"][i], a, doc["classes"][i])
    coarse = (doc.get("diffs") or {}).get("R16_Q1-L188")
    for name, body, what in (("MlstJcOneRecallLowCells", str(low), "network-class cells below 188 classes by > 3 sigma"),
                             ("MlstJcOneRecallHighCells", str(high), "network-class cells above 188 classes by > 3 sigma"),
                             ("MlstJcOneRecallCells", str(n), "network-class cells compared with 188 classes"),
                             ("MlstJcOneRecallCoarseLowClasses", str(len(coarse["lower"])) if coarse else None,
                              "JetClass classes the 17-class network recalls less often than 188 by > 3 sigma")):
        if body is not None:
            em.macro(name, body, src, "diffs.*-L188.lower/higher (count)", f"linear probes, {what}")
    if worst:
        em.macro("MlstJcOneRecallWorstPp", f"{abs(worst[0]) * 100:.1f}", src, f"diffs.{worst[1]}-L188.delta",
                 "largest recall drop against 188 classes, percentage points")
        em.macro("MlstJcOneRecallWorstNet", LABEL[worst[1]], src, f"diffs.{worst[1]}-L188", "its network")
        em.macro("MlstJcOneRecallWorstClass", names[worst[2]], src, "classes", "its JetClass class")


def check_recall_tests(doc: dict, src: pathlib.Path) -> None:
    """PI 2026-11-23 (LEDGER DL-17): the per-class arrows of tab:jc1_class_auc and the filled points of PC-jc2 use exactly
    the SC-1 test, |delta| > 3 sigma with sigma^2 = each side's run SD^2 + the paired test-jet bootstrap SD^2 of the
    class. Checked where the file is read (RR-7), for every difference it holds: the build stops if a sigma is not that
    sum or a stored list of lower or higher classes is not the classes beyond 3 sigma. A side of one run has no run SD,
    so no sigma and no mark (a lower-bound sigma never backs a difference, PI SC-2)."""
    for key, d in ((doc or {}).get("diffs") or {}).items():
        n = len(d["delta"])
        if d.get("sigma") is None:
            if d["lower"] or d["higher"]:
                raise SystemExit(f"FATAL: {src.name} {key}: classes marked without a sigma (a side of one run)")
            continue
        for i in range(n):
            if not math.isclose(d["sigma"][i], math.sqrt(d["sd_a"][i] ** 2 + d["sd_b"][i] ** 2 + d["boot_sd"][i] ** 2),
                                rel_tol=1e-6):
                raise SystemExit(f"FATAL: {src.name} {key} class {i}: sigma is not the quadrature sum of the two run "
                                 "SDs and the paired bootstrap SD (PI SC-1)")
        lower = {i for i in range(n) if d["delta"][i] < -3 * d["sigma"][i]}
        higher = {i for i in range(n) if d["delta"][i] > 3 * d["sigma"][i]}
        if lower != set(d["lower"]) or higher != set(d["higher"]):
            raise SystemExit(f"FATAL: {src.name} {key}: the stored lower and higher classes are not those beyond 3 sigma")


RECALL_TEST_COLS = ("file", "read_out", "task", "network", "reference", "class", "delta", "run_sd_network",
                    "run_sd_reference", "bootstrap_sd", "sigma", "delta_over_sigma", "mark")


def recall_tests_csv(docs: dict) -> str:
    """The data release's per-class tests (PI 2026-11-23, LEDGER DL-17: "per-class delta / sigma goes to the release"):
    one row per perclass_recall file, difference and class, the recall difference of the network minus its reference,
    each side's run SD, the paired test-jet bootstrap SD, sigma, delta / sigma and the mark it gives (lower, higher or
    none; none without a sigma), as tab:jc1_class_auc's arrows and PC-jc2's filled points read them."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(RECALL_TEST_COLS)
    for name, doc in sorted(docs.items()):
        mode, task = name.split("_", 2)[:2]
        names = doc.get("classes")
        for key, d in sorted((doc.get("diffs") or {}).items()):
            net, ref = key.split("-", 1)
            for i, delta in enumerate(d["delta"]):
                sig = d["sigma"][i] if d.get("sigma") is not None else None
                mark = "lower" if i in d["lower"] else "higher" if i in d["higher"] else "none"
                w.writerow([f"{RECALL_DIR}/{name}.json", "linear probe" if mode == "probe" else "fine-tuned", task,
                            _release_name(net), _release_name(ref), names[i] if names else i, repr(delta),
                            "" if d.get("sd_a") is None else repr(d["sd_a"][i]),
                            "" if d.get("sd_b") is None else repr(d["sd_b"][i]), repr(d["boot_sd"][i]),
                            "" if sig is None else repr(sig), "" if not sig else repr(delta / sig), mark])
    return out.getvalue()


def jc2_recall_macros(em, doc: dict, src: pathlib.Path, words) -> None:
    """The per-class sentence of Sec. 4.1 on PC-jc2 (review of 5c40d69, A6: the count printed from the build): the
    JetClass-II classes the 17-class linear probe recalls less (more) often than 188 classes by more than 3 sigma,
    sigma^2 = each side's run SD^2 + the paired test-jet bootstrap SD^2 (PI SC-1), the file checked by
    check_recall_tests where it is read."""
    d = doc["diffs"]["R16_Q1-L188"]
    lower, higher = set(d["lower"]), set(d["higher"])
    n = len(d["delta"])
    em.macro("MlstJcTwoRecallCoarseLowClasses", str(len(lower)), src, "diffs.R16_Q1-L188 (count, |delta| > 3 sigma)",
             f"JetClass-II classes of {n} the 17-class linear probe recalls less often than 188 classes")
    em.macro("MlstJcTwoRecallCoarseHighClassesWord", words(len(higher)), src,
             "diffs.R16_Q1-L188 (count, |delta| > 3 sigma)",
             "JetClass-II classes the 17-class linear probe recalls more often than 188 classes")
    # fourth review (R8): how many of the n per-class tests pass 3 sigma in one direction by chance, a Gaussian tail
    em.macro("MlstJcTwoRecallChanceOneSide", f"{n * math.erfc(3 / math.sqrt(2)) / 2:.2g}", src,
             "diffs.R16_Q1-L188 (count of classes x one-sided Gaussian tail beyond 3 sigma)",
             "per-class tests expected beyond 3 sigma in one direction by chance")


def table_jc1_class_recall(ft_doc: dict, probe_doc: dict, runs: dict, fmt_pm, arms=ALL_ROWS, init=INIT) -> str:
    """Table A2 (tab:jc1_class_auc; PI 2026-11-18: recall, not AUC): each JetClass class's recall, the fraction of its
    test jets the network labels as that class, at 10^6 training jets (perclass_recall probe_jc1, ft_jc1_N1000000; the
    points of the JetClass per-class figures): arms (the paper: the random partitions pooled, runs from _pooled) as
    linear probes with the random network init, and fine-tuned with the network from scratch; a fine-tuned file not
    yet pulled leaves its cells pending."""
    scratch = sorted(m for m in (ft_doc or {}).get("models", {}) if m.startswith("scratch"))

    def rows(doc: dict, extra: list) -> list:
        out = []
        for key, name, ms in [(a, _row(a, runs), runs[a]) for a in arms] + [(None, n, m) for n, m in extra]:
            per = [p for m in ms if (p := _recall(doc, m))]
            low, high = recall_marks(doc, key)
            out.append(name + " & " + " & ".join(
                _cell([p[c] for p in per], len(ms) or 3, fmt_pm, frac=True)
                + (RECALL_LOW if i in low else RECALL_HIGH if i in high else "")
                for i, (c, _) in enumerate(JC1_AUC_COLS)) + r" \\")
        return out

    head = [r"\begin{tabular}{l" + "c" * len(JC1_AUC_COLS) + "}", r"\toprule",
            " & " + " & ".join(t for _, t in JC1_AUC_COLS) + r" \\", r"\midrule"]
    return "\n".join(head + rows(probe_doc, [(_row(init, runs), runs[init])])       # L1: linear probe first
                     + [r"\bottomrule", r"\end{tabular}", "%SPLIT%"] + head
                     + rows(ft_doc, [(SCRATCH_LABEL, scratch)]) + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ the JetClass confusion percentages (supplement)
# experiments/EVAL/perf_vs_pt.py's confusion_jc1 (pretrained networks fine-tuned on 10^6 JetClass jets) and
# confusion_jc1_fulldecay (from scratch on its full-schedule cells, in place of the superseded scratch files there);
# rows and columns in the confusion figures' class order (style_v5.JC1_CLASSES), what they draw: the 188-class matrix,
# every other network as the mean over runs of (run - the run of its reference with the same pretraining seed)
CONF_CLASSES = ("label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
                "label_Tbqq", "label_Tbl")
CONF_TEX = dict(JC1_AUC_COLS)
CONF_DIFF = (("L162", "L188"), ("L162_MASS", "L162"), ("R42_Q1", "L188"), ("R16_Q1", "L188"),
             ("R16_Q1_MASS", "R16_Q1"), ("FLAV_F0", "L188"), ("FLAV_F1", "L188"), ("FLAV_F1R", "L188"),
             (RAND_POOLED, "L188"))      # L5, each mass network against its own label set as in PC-conf-mass


def _conf_matrix(doc: dict) -> np.ndarray:
    """100 x the row-normalised [true][predicted] in CONF_CLASSES order (gallery_perclass._matrix)."""
    k = [doc["classes"].index(c) for c in CONF_CLASSES]
    rows = doc["row_normalised"]
    return 100 * np.array([[np.nan] * len(k) if rows[i] is None else [rows[i][j] for j in k] for i in k], float)


def table_jc1_conf(conf: dict, scratch: dict, runs: dict, per_part: int = 5) -> str:
    """tab:jc1conf: per network a block of ten rows (true class) by the predicted classes: the 188-class matrix and
    from scratch as percentages (mean +- SD over runs), every other network as the change from its reference in
    percentage points, the mean +- SD over runs of each run minus its reference's run of the same pretraining seed
    (gallery_perclass._diff, which the confusion figures draw). The predicted classes in parts of per_part columns
    (%SPLIT%), each a turned page's width at the body size."""
    runs = _pooled(runs)
    seed = lambda m: m.rsplit("-s", 1)[1]       # noqa: E731
    blocks = [(f"{LABEL['L188']}, percentage of the true class", [_conf_matrix(conf[m]) for m in runs["L188"]])]
    for a, b in CONF_DIFF:
        ref = {seed(m): _conf_matrix(conf[m]) for m in runs[b] if m in conf}
        d = [_conf_matrix(conf[m]) - ref[seed(m)] for m in runs[a] if m in conf and seed(m) in ref]
        if len(d) < len(runs[a]):
            raise SystemExit(f"FATAL: confusion_jc1 holds {len(d)} of the {len(runs[a])} runs of {a} with a reference run")
        name = ("17-class partition with " + {"FLAV_F0": "no flavor boundary", "FLAV_F1": "a $b$-quark boundary",
                                              "FLAV_F1R": "a random flavor boundary"}[a] if a in FLAV
                else "17-class random partitions" if a == RAND_POOLED else LABEL[a])
        blocks.append((f"{name} minus {LABEL[b]}, percentage points", d))
    blocks.append((f"{SCRATCH_LABEL}, percentage of the true class", [_conf_matrix(scratch[m]) for m in sorted(scratch)]))
    stats = [(t, np.mean(d, axis=0), np.std(d, axis=0, ddof=1)) for t, d in blocks]
    # round-1 fix 2026-11-13: one network a page, its two halves of predicted classes stacked, so that one matrix is
    # read without turning pages; one decimal place for the whole matrix (_column_place over all its cells), so that no
    # column prints coarser than its neighbours
    # the first network's two halves on two pages, each under its own heading (page review 2026-11-13: under the
    # caption the 188-class matrix broke inside its second half, whose columns then had no names)
    parts = []
    nc = len(CONF_CLASSES)
    for b, (t, mean, sd) in enumerate(stats):
        ok = lambda i, j: not np.isnan(mean[i, j])      # noqa: E731
        place = _column_place([("p", [float(mean[i, j]), float(sd[i, j])]) for i in range(nc) for j in range(nc) if ok(i, j)])
        out = [r"\begin{tabular}{l" + "c" * per_part + "}", r"\toprule", _full_row(1 + per_part, r"{\itshape " + t + "}"),
               r"\midrule"]
        for k, lo in enumerate(range(0, nc, per_part)):
            cols = range(lo, min(lo + per_part, nc))
            if k and b == 0:
                parts.append("\n".join(out + [r"\bottomrule", r"\end{tabular}", ""]))
                out = [r"\begin{tabular}{l" + "c" * per_part + "}", r"\toprule",
                       _full_row(1 + per_part, r"{\itshape " + t + " (continued)}"), r"\midrule"]
            out += ([r"\midrule"] if k and b else []) + [
                rf" & \multicolumn{{{per_part}}}{{c}}{{predicted class}} \\", rf"\cmidrule(lr){{2-{1 + per_part}}}",
                "true class & " + " & ".join(CONF_TEX[CONF_CLASSES[j]] for j in cols) + r" \\", r"\midrule"]
            out += [CONF_TEX[c] + " & " + " & ".join(_render("p", [float(mean[i, j]), float(sd[i, j])], place) if ok(i, j)
                                                    else "--" for j in cols) + r" \\" for i, c in enumerate(CONF_CLASSES)]
        parts.append("\n".join(out + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(parts)


def per_class_csv(fits, runs, root) -> str:
    """The data release's per-class table of JetClass-II (class-balanced test set, largest training set): the
    linear-probe AUC of each class against the rest, mean over the runs of each configuration."""
    arms = MAIN + MASS + RAND + FLAV
    per = {a: [p for m in runs[a] if (p := _per_class(fits, m))] for a in arms}
    lines = ["jet_label,class_name," + ",".join(_release_name(a) for a in arms)]
    for r in _class_rows(root):
        c = int(r["jet_label"])
        vals = [float(np.mean([p[c] for p in per[a] if c in p])) if any(c in p for p in per[a]) else "" for a in arms]
        lines.append(f"{c},{r['class_name']}," + ",".join("" if v == "" else f"{v:.5f}" for v in vals))
    return "\n".join(lines) + "\n"




def per_class_recall(docs: dict, runs, arm: str, mode: str) -> dict:
    """{class: [recall of the class, one a run of arm]} on JetClass-II, as the per-class figures read them
    (gallery_perclass.SOURCES["jc2"]): docs {"lp": perclass_recall probe_jc2, "ft": ft_jc2_N1000000}, the linear probe
    on one million class-balanced jets or the network fine-tuned on as many."""
    out = collections.defaultdict(list)
    for m in runs[arm]:
        for c, v in (_recall(docs.get(mode), m) or {}).items():
            out[c].append(v)
    return out


def table_per_class(docs: dict, runs, fmt_pm, root, native_tex) -> str:
    """tab:perclass (supplement; PI 2026-11-18: recall, not AUC): per class of JetClass-II, in the order of the
    per-class figures' grouping (by the 17-class group), the recall of the class for the label sets and the 17-class
    partitions (the random partitions pooled, runs from _pooled), as a linear probe and fine-tuned (docs as
    per_class_recall; a file not pulled leaves its block pending): every point of the JetClass-II per-class figures.
    Two turned longtables under one number, the label sets and then the 17-class partitions (ten columns of five
    decimals do not fit one turned page), each with its column names on every page and the two modes as blocks; each
    block's columns at their own precision."""
    classes = _class_rows(root)
    # at the body size (D14); a mass network's name in two lines, so that six columns fit a turned page
    out = [r"\begin{landscape}", "{", r"\setlength{\tabcolsep}{3pt}", LT_SETUP]
    name = lambda a: (r"\begin{tabular}[b]{@{}c@{}}" + LABEL[a].replace(" + mass", r"\\+ mass") + r"\end{tabular}"   # noqa: E731
                      if a in MASS else LABEL[a])
    # round-1 fix 2026-11-13: one longtable per group and mode, each from a fresh page, its continuation heading naming
    # the mode, so that no page of rows leaves its mode to be guessed
    first = True
    for g, (group, cols) in enumerate(((None, SETS), (PARTITIONS, FLAV + (RAND_POOLED,)))):
        head = ([rf" & \multicolumn{{{len(cols)}}}{{c}}{{{group}}} \\", rf"\cmidrule(lr){{2-{1 + len(cols)}}}"] if group else []) + [
            "class & " + " & ".join(name(a) for a in cols) + r" \\"]
        for k, mode in enumerate(("lp", "ft")):
            more = (r"\multicolumn{" + str(1 + len(cols)) + r"}{@{}l}{\textit{Table~\ref{tab:perclass} (continued), "
                    + HEADING[mode] + r"}} \\")
            out += ([] if first else [r"\newpage", r"\addtocounter{table}{-1}"]) + [
                r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * len(cols) + "@{}}",
                r"\caption{__CAPTION__}\label{tab:perclass}\\" if first else more, r"\toprule", *head, r"\midrule",
                r"\endfirsthead", more, r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
            first = False
            vals = {a: per_class_recall(docs, runs, a, mode) for a in cols}
            rows, last = [], None
            for r in classes:
                c = int(r["jet_label"])
                if last is not None and r["R16_Q1"] != last:
                    rows.append(r"\midrule")
                last = r["R16_Q1"]
                rows.append(native_tex(r["class_name"]) + " & " + " & ".join(
                    _cell(vals[a].get(c, []), len(runs[a]), fmt_pm, frac=True) for a in cols) + r" \\")
            out += [rf"\multicolumn{{{1 + len(cols)}}}{{@{{}}l}}{{\textit{{{HEADING[mode]}}}}} \\*"] + uniform_columns(
                "\n".join(rows)).split("\n")
            out.append(r"\end{longtable}")
    return "\n".join(out + ["}", r"\end{landscape}", ""])


def per_class_macros(em, fits, runs, root, index, fmt_pm, native_tex, nat: dict | None = None) -> None:
    """Where the cost sits (F3): for each coarser level, the classes whose 1 - AUC at least doubles,
    the class with the largest ratio, and for the classes merged with a decay that differs only in
    lepton flavor and only in quark flavor, their number, median ratio and share of the summed
    ln ratio over the resonant classes (the per-class and the aggregate cost); in the random partitions,
    the median ratio of the classes merged with a flavor counterpart over that of those kept apart, with
    the linear probes and (MlstMlpNatPerClass...) the nonlinear probes on natural composition (nat)."""
    q, lep = _orbits(root)
    groups = _groups(root)
    names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    by_name = {v: k for k, v in names.items()}

    def merged_with(arm, key_of):
        g = groups[arm]
        out = set()
        for a in key_of:
            for b in key_of:
                if a != b and key_of[a] == key_of[b] and g[a] == g[b]:
                    out.add(by_name[a])
        return out
    for arm in ("L162", "R42_Q1", "R16_Q1"):
        w = WORD[arm]
        rat = per_class_ratios(fits, runs, arm)
        mean = {c: float(np.mean(v)) for c, v in rat.items() if v}
        em.macro(f"MlstPerClass{w}NDoubled", str(sum(x >= 2 for x in mean.values())) if mean else r"\pending{probes}",
                 index, f"JetClass-II classes whose mean ratio of 1 - AUC ({arm} over 188) is at least 2 (count)")
        sds = [float(np.std(v, ddof=1)) for v in rat.values() if len(v) > 1]
        em.macro(f"MlstPerClass{w}SeedSdMedian", _sig2(float(np.median(sds))) if sds else r"\pending{probes}", index,
                 f"median over classes of the SD over runs of the per-class ratio, {arm} over 188")
        if mean:
            top = max(mean, key=mean.get)
            em.macro(f"MlstPerClass{w}TopClass", native_tex(names[top]), index, f"class with the largest mean ratio, {arm}")
            _emit_ratio(em, f"MlstPerClass{w}TopRatio", rat[top], len(runs[arm]), fmt_pm, index,
                        f"per-class ratio of 1 - AUC of that class, {arm} over 188, run by run", "errors")
        res = {c: x for c, x in mean.items() if not names[c].startswith("label_QCD")}
        tot = sum(math.log(x) for x in res.values() if x > 0)
        both = set(res)
        for kind, key_of, kw in (("lepton", lep, "Lepton"), ("quark", q, "Quark")):
            mc = merged_with(arm, key_of) & set(res)
            both &= mc
            v = [res[c] for c in mc]
            em.macro(f"MlstPerClass{w}{kw}N", str(len(v)), index,
                     f"resonant classes sharing their {arm} group with a decay that differs only in {kind} flavor (count)")
            med = [float(np.median(v))] if v else []
            _emit_ratio(em, f"MlstPerClass{w}{kw}Median", med, 1, fmt_pm, index,
                        f"median over those classes of the mean per-class ratio, {arm} over 188", "errors",
                        body=_lone(med[0]) if med else "--")
            em.macro(f"MlstPerClass{w}{kw}Share", f"{100 * sum(math.log(res[c]) for c in mc) / tot:.0f}\\%" if v and tot
                     else "--", index, f"their share of the summed ln ratio over the resonant classes, {arm}")
        em.macro(f"MlstPerClass{w}QuarkLeptonN", str(len(both)), index,
                 f"resonant classes counted in both the quark- and the lepton-flavor set, {arm} (count)")
    for prefix, probe, what in (("MlstPerClass", fits, "linear probe"),
                                ("MlstMlpNatPerClass", _nat_as_fits(nat or {}), "nonlinear probe, natural composition")):
        for kind, key_of, kw in (("quark", q, "Flav"), ("lepton", lep, "Lepton")):
            vals = []
            for arm in RAND:
                rat = per_class_ratios(probe, runs, arm)
                mean = {c: float(np.mean(v)) for c, v in rat.items() if v}
                sib = {by_name[a] for a in key_of if sum(key_of[b] == key_of[a] for b in key_of) > 1}
                mc = merged_with(arm, key_of)
                mv = [mean[c] for c in sib & mc if c in mean]
                av = [mean[c] for c in sib - mc if c in mean]
                vals.append(float(np.median(mv)) / float(np.median(av)) if mv and av else None)
            _emit_ratio(em, f"{prefix}Rand{kw}MergedOverApart", vals, len(vals), fmt_pm, index,
                        f"random partitions, {what}: median per-class ratio of the classes merged with a decay that differs only in "
                        f"{kind} flavor over that of the classes kept apart from all of them, mean +- SD over the partitions",
                        "errors")


# ------------------------------------------------------------------ prong-count boundaries of the partitions (A18)
def table_rand_prong(pre, runs, root, fmt_pm) -> str:
    """A18: per random partition, whether it keeps each prong-count boundary (no group holding classes
    of both prong counts, from the 17-class level's group names), the groups that mix prong counts, and
    its fine-tuned JetClass rejection per signal class over the 188-class run of its seed."""
    rows = list(csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open()))
    prong = {r["class_name"]: int(r["R16_Q1_name"][0]) for r in rows if not r["R16_Q1_name"].startswith("QCD")}
    rmap = list(csv.DictReader((root / "configs/labelmaps/rand_label_map.v2.csv").open()))
    pairs = ((2, 3), (2, 4), (3, 4))
    body = []
    for arm in RAND:
        g = collections.defaultdict(set)
        for r in rmap:
            if r["class_name"] in prong:
                g[r[arm]].add(prong[r["class_name"]])
        kept = ["$\\bullet$" if not any(a in s and b in s for s in g.values()) else "--" for a, b in pairs]
        mixed = sum(len(s) > 1 for s in g.values())
        m = runs[arm][0]
        ref = next((x for x in runs["L188"] if _idx(x) == _idx(m)), None)
        cells = []
        for c, _ in V4.JC1_COLS:
            a, b = _jc1_rej(pre, m, c), _jc1_rej(pre, ref, c) if ref else None
            cells.append(_lone(a["rejection"] / b["rejection"]) + (r"$^\dagger$" if min(a["n_bkg_pass"], b["n_bkg_pass"])
                                                                  < V4.JC1_MIN_BKG else "") if a and b else "--")
        body.append((arm, kept, mixed, cells))
    # when no partition keeps any prong-count boundary the three columns are only dashes: the caption says it
    show = any("bullet" in k for _, kept, _, _ in body for k in kept)
    nk = len(pairs) if show else 0
    out = [r"\begin{tabular}{l" + "c" * (nk + 1 + len(V4.JC1_COLS)) + "}", r"\toprule",
           (r" & \multicolumn{3}{c}{prong-count boundary kept}" if show else "") + r" & & \multicolumn{"
           + str(len(V4.JC1_COLS)) + r"}{c}{rejection over \Kfull{} classes} \\",
           (r"\cmidrule(lr){2-4}" if show else "") + rf"\cmidrule(lr){{{nk + 3}-{nk + 2 + len(V4.JC1_COLS)}}}",
           r"partition & " + ("2/3 & 2/4 & 3/4 & " if show else "") + "mixed groups & "
           + " & ".join(h for _, h in V4.JC1_COLS) + r" \\", r"\midrule"]
    for arm, kept, mixed, cells in body:
        out.append(_row(arm, runs) + " & " + "".join(k + " & " for k in kept if show) + f"{mixed} & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ test-sample uncertainty (A7)
# JetClass-II columns: the bootstrap of the test jets (bootstrap/<family>/<task>_N<N>.json) of the same
# composition as the fits the paper's ratios read: the class-balanced probe_v4 family
# (scripts/build_curve_jobs.py boot_probe_v4_body) when fits_v4 is in use, the natural-composition probe
# family only while the ratios still come from fits_v3; never one for the other. Top and
# quark/gluon columns: R30's counting uncertainty, 1/sqrt(passing background jets), which the fits and
# read-outs store as r30_rel_stat (the bootstrap scores 1 - AUC only, experiments/STATS/paired_bootstrap.py).
BOOT_COLS = (("JetClass-II", "probe", {True: (("probe_v4", "jc2bal"),), False: (("probe", "jc2_188"),)}, "macro"),
             ("top", "probe", "top", "r30"), ("quark/gluon", "probe", "qg", "r30"),
             ("JetClass-II", "ft", (("jc2bal_ft", "jc2bal"),), "macro"),
             ("top", "ft", "top", "r30"), ("quark/gluon", "ft", "qg", "r30"))
BOOT_ROWS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))


def _boot_largest(boot: dict, family: str, task: str):
    """The family's bootstrap document at the task's largest training set, or None."""
    docs = {int(k.rsplit("_N", 1)[1]): d for k, d in boot.get(family, {}).items() if k.rsplit("_N", 1)[0] == task}
    return docs[max(docs)] if docs else None


def boot_spreads(boot: dict, runs: dict, cands, arm: str, ref: str, met: str = "macro") -> tuple:
    """(SD over runs of ln ratio, mean over runs of its test-sample SD) of (1 - AUC) of arm over ref at
    the largest training set, from the first bootstrap family present; (None, None) where a run is missing."""
    for fam, task in cands:
        doc = _boot_largest(boot, fam, task)
        if doc is None:
            continue
        ln, test = [], []
        for m in runs[arm]:
            r = next((x for x in runs[ref] if _idx(x) == _idx(m)), None)
            c = doc["pairs"].get(f"{m}/{r}", {}).get(met)
            if c is None or c["ratio"] is None or c["ln_ratio_sd"] is None:
                return None, None
            ln.append(math.log(c["ratio"]))
            test.append(c["ln_ratio_sd"])
        return (float(np.std(ln, ddof=1)) if len(ln) > 1 else None), float(np.mean(test))
    return None, None


def count_spreads(get, runs: dict, arm: str, ref: str) -> tuple:
    """(SD over runs of ln R30 ratio, mean counting SD of ln ratio) from get(model) -> record."""
    ln, test = [], []
    for m in runs[arm]:
        r = next((x for x in runs[ref] if _idx(x) == _idx(m)), None)
        a, b = get(m), get(r) if r else None
        if a is None or b is None or not a.get("r30") or not b.get("r30") or a.get("r30_rel_stat") is None \
                or b.get("r30_rel_stat") is None:
            return None, None
        ln.append(math.log(a["r30"] / b["r30"]))
        test.append(math.hypot(a["r30_rel_stat"], b["r30_rel_stat"]))
    return (float(np.std(ln, ddof=1)) if len(ln) > 1 else None), float(np.mean(test))


def _pctsd(x) -> str:
    if x is None:
        return "--"
    v = 100 * x
    return f"{v:.0f}" if v >= 10 else f"{v:#.2g}"


R30_HEAD, OMA_HEAD, BULLET = r"$R_{30}$", r"$1-\mathrm{AUC}$", r"$\bullet$"


def check_boot_fills(docs: dict, chosen: dict) -> None:
    """Each top / quark/gluon bootstrap document reads every probe at the substitution chosen on validation."""
    for name, d in docs.items():
        task = name.rsplit("_N", 1)[0]
        if task not in ("top", "qg", "qg_herwig") or chosen.get(task) is None:
            continue
        tag = SUBST_KEY[task][chosen[task]].partition(".")[2]
        bad = [m for m, r in d["models"].items() if f".{tag}/" not in r["input"]] if tag else []
        if bad or not tag:
            raise SystemExit(f"FATAL: bootstrap {name} does not read the {chosen[task]} substitution's probes "
                             f"({bad[:3] or 'zero fill'}); rerun scripts/build_refit_v5_jobs.py boot with FILL_TAG")


def table_boot(boot: dict, runs: dict, fits=None, per=None, balanced: bool = True, fill=None, chosen=None) -> tuple[str, dict]:
    """A7: per label-set comparison, at the largest training set: the SD of the logarithm of the ratio
    over the pretraining runs and from the finite test sample, in per cent, in the task's metric.
    balanced: the JetClass-II ratios come from the class-balanced fits (fits_v4), so the probe column
    reads the class-balanced bootstrap only."""
    lines = [r"\begin{tabular}{l" + "c" * len(BOOT_COLS) + "}", r"\toprule",
             r" & \multicolumn{3}{c}{linear probe on frozen features} & \multicolumn{3}{c}{Sophon's network, fine-tuned} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             "ratio & " + " & ".join(h + ", " + (R30_HEAD if met == "r30" else OMA_HEAD)
                                     for h, _, _, met in BOOT_COLS) + r" \\", r"\midrule"]
    worst = {"probe": [], "ft": [], "probe_r30": [], "ft_r30": []}
    for arm, ref in BOOT_ROWS:
        cells = []
        for _, kind, src, met in BOOT_COLS:
            if met == "r30":
                if kind == "probe":
                    # the probe the tables print: the substitution chosen on validation (_bench_rec), zero while pending
                    get = lambda m, src=src: (None if fits is None else _bench_rec(fits, fill or {}, chosen or {}, m, src))
                else:
                    get = lambda m, src=src: (None if not per or not _sizes(per, src) else
                                              (per[src].get(m[len("mtx-"):], {}).get(_sizes(per, src)[-1]) or [None])[0])
                between, test = count_spreads(get, runs, arm, ref)
                key = f"{kind}_r30"
            else:
                between, test = boot_spreads(boot, runs, src[balanced] if isinstance(src, dict) else src, arm, ref)
                key = kind
            cells.append(f"{_pctsd(between)} / {_pctsd(test)}")
            if test is not None:
                worst[key].append(test)
        lines.append(_label_parts(LABEL[arm]) + " / " + _label_parts(LABEL[ref]) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), worst


def boot_macros(em, boot: dict, worst: dict, index) -> None:
    b = {d["b"] for fam in boot.values() for d in fam.values()}
    if len(b) > 1:
        raise SystemExit(f"FATAL: the bootstraps use different numbers of resamples: {sorted(b)}")
    em.macro("MlstBootResamples", str(b.pop()) if b else r"\pending{bootstrap}", index, "bootstrap resamples of the test jets")
    for k, w, what in (("probe", "Probe", "ratio of 1 - AUC on JetClass-II, linear probe"),
                       ("ft", "Ft", "ratio of 1 - AUC on JetClass-II, fine-tuning"),
                       ("probe_r30", "RthirtyProbe", "R30 ratio on top and quark/gluon, linear probe (counting)"),
                       ("ft_r30", "RthirtyFt", "R30 ratio on top and quark/gluon, fine-tuning (counting)")):
        em.macro(f"MlstBootTestMax{w}", _pctsd(max(worst[k])) if worst.get(k) else r"\pending{bootstrap}", index,
                 f"largest test-sample SD of the ln {what}, label sets over the 188-class model, per cent")


# ------------------------------------------------------------------ ParT reproduction (A9), sweep (A4), inputs (A8)
# review of render dfe5942 (C; LEDGER CA-6): one name, ParT, defined in the introduction
PARTREPRO = (("part", "ParT's training script, unchanged"), ("head", "the same script with Sophon's output head"),
             ("rescale", r"the same script with Sophon's jet-$p_{\mathrm{T}}$ rescaling"))


def partrepro(em, data: pathlib.Path, per, fmt_pm, missing: list, runs: dict | None = None) -> str | None:
    """A9 and its macros: ParT's top-tagging script on our cluster, unchanged, with Sophon's output head, and with
    Sophon's rescaling of the four-momenta: each of the two a single change to ParT's script, not cumulative
    (scripts/build_partrepro_jobs.py VARIANTS), one run each. ParT's published row and our network trained from
    scratch are in the comparison table only (tab:published; PI 2026-11-12: one table per number)."""
    docs = {v: json.loads(p.read_text()) for v, _ in PARTREPRO if (p := data / "partrepro" / f"{v}.json").exists()}
    keys = ("accuracy", "auc", "r50", "r30")
    n = _sizes(per, "top")[-1] if per and _sizes(per, "top") else None
    recs = per["top"].get(SCRATCH, {}).get(n, []) if n else []
    place = {"accuracy": 4, "auc": 4, "r50": 0, "r30": 0}
    for k in keys if recs else ():      # at the place of the fine-tuned top-tagging columns (from scratch, the label sets)
        v = [r[k] for r in recs]
        items = [("p", [float(np.mean(v)), float(np.std(v, ddof=1))])]
        for a in (runs or {}).get("__bench__", ()):
            w = [x for m in runs[a] if (x := _ft(per, "top", m, n, k)) is not None]
            if len(w) > 2:
                items.append(("p", [float(np.mean(w)), float(np.std(w, ddof=1))]))
        place[k] = _column_place(items)
    for v, _ in PARTREPRO:
        d = docs.get(v)
        src = data / "partrepro" / f"{v}.json" if d else data / "index.json"
        # as the table prints it (2026-11-23 review A8, LEDGER TN-4: the text's 1,711 against the table's 1,710)
        em.macro(f"MlstPartreproRthirty{v.title()}", _at(d["r30"], place["r30"]) if d else r"\pending{reproduction}",
                 src, "r30", f"ParT reproduction on top, {v}: R30 (one run), at tab:partrepro's place")
    for v in ("head", "rescale"):
        a, b = docs.get(v), docs.get("part")
        val = [a["r30"] / b["r30"]] if a and b else []
        _emit_ratio(em, f"MlstPartreproRthirty{v.title()}OverPart", val, 1, fmt_pm,
                    data / "partrepro" / f"{v}.json" if a else data / "index.json", "r30 / part r30",
                    "rejection", body=_lone(val[0]) if val else r"\pending{reproduction}")
    em.macro("MlstPartreproSliceDropRatio", r"\pending{jet-pT slice test not run}", data / "index.json",
             "open item O10: R30 in narrow pT slices of partrepro part and rescale")
    if not docs:
        missing.append("ParT reproduction -- experiments/FIGS/data/v2/partrepro/{part,head,rescale}.json")
        return None
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r" & accuracy & AUC & $R_{50}$ & $R_{30}$ & background jets passing at $R_{30}$ \\", r"\midrule"]
    # PI 2026-11-23 (LEDGER DL-R #16): the rescaling row is kept only if a sentence states it, and none does, so the table
    # prints ParT's script unchanged and with Sophon's output head (the rescaling run's values stay in its macros and the
    # data release)
    for v, name in PARTREPRO:       # DL-16c (PI 2026-11-24): the rescaling row restored, with its null sentence; one run each: the caption says so once (round-1 fix 2026-11-13), no mark per cell
        d = docs.get(v)
        out.append(name + " & " + (" & ".join([_at(d[k], place[k]) for k in keys]
                                              + [_sep(str(d['r30_n_bkg_pass']))])
                                   if d else r"\multicolumn{5}{c}{\pending{run}}") + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# The rate search of every row (PI 2026-11-20/21; experiments/FT/rate_search.py applies the rules to the raw per-cell
# facts that pull_mlst_data.rate_search_scan writes to budget/rate_search.json). Table A15 lists every rate tried.
RATE_SEARCH = "rate_search.json"
RATE_TASKS = (("top", "top tagging"), ("qg", "quark/gluon"), ("jc1", "JetClass"), ("jc2b", "JetClass-II"))


def _rate_search_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rate_search", pathlib.Path(__file__).resolve().parents[2]
                                                  / "experiments/FT/rate_search.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rate_search_rows(scan: dict, preview: bool = False) -> list:
    """Every row of the search, evaluated; the paper build stops unless every row has settled at its recorded rate with
    no issue (a cell off its schedule, a superseded cell, a recorded rate the search does not select)."""
    rows = _rate_search_module().evaluate(scan)
    bad = [r for r in rows if r["issues"] or not r["status"].startswith("settled")]
    if bad and not preview:
        raise SystemExit("FATAL: the rate search is not settled: " + "; ".join(
            f"{r['task']} {r['n']} {r['kind']}: {r['status']} {r['issues'][:2]}" for r in bad[:6]))
    return rows


def table_lrsweep(rows: list) -> str:
    """A15: the best validation accuracy at every rate tried, per task, training size and mode (pretrained: the mean of
    seed 1 of 188 and of 17 classes; from scratch: seed 1); a star on the rate chosen; "diverged" where the training loss
    became non-finite. Validation accuracy only (PI 2026-11-21): the comparison with frozen features is in the probe and
    fine-tuned tables, on test jets."""
    rates = sorted({lr for r in rows for lr in r["values"]}, key=float)
    out = [rf"\begin{{tabular}}{{lll{'c' * len(rates)}}}", r"\toprule",
           r" & training jets & & " + " & ".join(_power(lr) for lr in rates) + r" \\", r"\midrule"]
    star = r"$^{\ast}$"
    for i, (task, name) in enumerate(RATE_TASKS):
        block = [r for r in rows if r["task"] == task]
        if i:
            out.append(r"\midrule")
        for j, r in enumerate(sorted(block, key=lambda r: (r["n"], r["kind"] != "pre"))):
            # review of render dfe5942 (B11; LEDGER FT-7): a rate that ties the chosen one at four decimals prints, with
            # it, the digits that separate them (quark/gluon 10^4 from scratch, 0.805594 against 0.805589)
            nums = {lr: v for lr, v in r["values"].items() if isinstance(v, float)}
            # fourth review (C9): in a row still searching, the leading value's ties print their digits too
            best = (nums.get(r["best"]) if r["status"].startswith("settled") else max(nums.values()) if nums else None)
            tied = {lr for lr, v in nums.items() if best is not None and f"{v:.4f}" == f"{best:.4f}"}
            dec = 4
            while len(tied) > 1 and dec < 7 and len({f"{nums[lr]:.{dec}f}" for lr in tied}) < len(tied):
                dec += 1
            cells = []
            for lr in rates:
                v = r["values"].get(lr, "--")
                if v == "--":
                    cells.append("--")
                elif v is None:
                    cells.append(r"\pending{run}")
                elif v == "diverged":
                    cells.append("diverged")
                else:
                    cells.append(f"{v:.{dec if lr in tied else 4}f}"
                                 + (star if r["status"].startswith("settled") and lr == r["best"] else ""))
            first = name if j == 0 or r["kind"] == "pre" else ""   # review of 5f4e1b8 (C7): a page that opens mid-task names it
            size = _power(str(r["n"])) if r["kind"] == "pre" else ""
            mode = "pretrained" if r["kind"] == "pre" else SCRATCH_LABEL
            # review of render dfe5942 (C): a size's pretrained row never ends a page apart from its from-scratch row
            end = r" \\*" if r["kind"] == "pre" else r" \\"
            out.append(f"{first} & {size} & {mode} & " + " & ".join(cells) + end)
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


FEATURE_TEX = {"part_pt_scale_log": r"$\log p_{\mathrm{T}}$, rescaled", "part_e_scale_log": r"$\log E$, rescaled",
               "part_logptrel": r"$\log(p_{\mathrm{T}}/p_{\mathrm{T}}^{\mathrm{jet}})$",
               "part_logerel": r"$\log(E/E^{\mathrm{jet}})$", "part_deltaR": r"$\Delta R$",
               "part_charge": "electric charge", "part_isChargedHadron": "charged hadron",
               "part_isNeutralHadron": "neutral hadron", "part_isPhoton": "photon", "part_isElectron": "electron",
               "part_isMuon": "muon", "part_d0": r"$d_0$", "part_d0err": r"$\sigma_{d_0}$", "part_dz": r"$d_z$",
               "part_dzerr": r"$\sigma_{d_z}$", "part_deta": r"$\Delta\eta$", "part_dphi": r"$\Delta\phi$"}


def _features(path: pathlib.Path) -> list:
    block = re.search(r"^   pf_features:\n(.*?)^   pf_vectors:", path.read_text(), re.M | re.S)
    if block is None:
        raise SystemExit(f"FATAL: {path}: no pf_features block")
    return re.findall(r"^\s+- \[(\w+),", block.group(1), re.M)


def table_inputs(root: pathlib.Path, offsets: dict | None) -> str:
    """A8: the per-particle inputs (ParT's Table 2 layout), which the top and the quark/gluon data
    provide, where a zero lies after the input normalization (in standard deviations of the
    training particles); what the mean and the random-draw substitutions put in a missing slot is the same for
    every slot of a kind, so the caption says it (PI 2026-11-12; experiments/EVAL/input_variants.py)."""
    top = _features(root / "configs/finetune/TopReference.yaml")
    qg = _features(root / "configs/finetune/EnergyFlowQG.yaml")
    out = [r"\begin{tabular}{lccc}", r"\toprule",
           r"input & top & quark/gluon & zero after normalization \\", r"\midrule"]
    for ft, fq in zip(top, qg):
        name = ft.replace("part_zero_", "part_")
        missing_t, missing_q = ft.startswith("part_zero_"), fq.startswith("part_zero_")
        z = "--"
        if missing_t or missing_q:
            z = f"${round(offsets[name], 2) + 0.0:.2f}$" if offsets and name in offsets else r"\pending{offsets}"
        mt, mq = ("--" if missing_t else BULLET), ("--" if missing_q else BULLET)
        out.append(f"{FEATURE_TEX.get(name, name)} & {mt} & {mq} & {z}" + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# PI 2026-11-19 (B, main text) and 2026-11-22 (Part 2): one compact main-text table of the label sets and controls,
# name, number of classes, what is merged (two or three words) and runs, in place of the definitions scattered
# through the captions. The run counts are the paper's runs (_paper); the words follow the paper's methods text.
# review of render dfe5942 (B6; LEDGER TN-3, L5): each mass row under its label set, and the references in the order of
# Table 1 and tab:published (random network, from scratch, released Sophon checkpoint, then the kinematic-inputs
# diagnostic); (B5; LEDGER CA-15) the random network with BatchNorm recomputed defined here, under the random network
SETS_TABLE = (("L188", r"\Kfull", "none (Sophon's classes)"), ("L162", r"\Kres", "QCD quark content"),
              ("L162_MASS", r"\Kres", r"as \Kres, with a mass output"),
              # review of render 5f4e1b8 (B12): what each merges in words, not "b/c counts, g/q, e/mu" and "one b split"
              ("R42_Q1", r"\Kmid", r"numbers of $b$ and $c$ quarks, $s$ against $u$ or $d$ quark, "
                                    r"gluon against $u$, $d$ or $s$ quark, electron against muon"),
              # fifth review (A1): the 43-class step merges s with u and d quarks too (rung_label_maps.v1.csv, group 2
              # gg, qq, sq, ss), and the 17-class step removes only whether a b or c quark is present
              ("R16_Q1", r"\Kcoarse", r"as \Kmid, and whether a $b$ or $c$ quark is present"),
              ("R16_Q1_MASS", r"\Kcoarse", r"as \Kcoarse, with a mass output"),
              ("FLAV_F0", r"\Kcoarse", "random groups of flavor families"),
              ("FLAV_F1", r"\Kcoarse", "as no flavor boundary, its four-quark decays with a $b$ quark moved"),
              ("FLAV_F1R", r"\Kcoarse", "as no flavor boundary, as many four-quark decays moved at random"),
              (RAND_POOLED, r"\Kcoarse", "random groups"))
SETS_CONTROLS = ((INIT, "--", "no pretraining (initial weights)", None),
                 (INIT_BN, "--", "as random, BatchNorm statistics recomputed", None),
                 (SCRATCH, "--", "trained on the task alone", None), (SOPHON, r"\Kfull", "none; Sophon's training", 1),
                 ("KIN", r"\Kfull", "none; kinematic inputs only", 1))


# PI 2026-11-23 (LEDGER FG-25): tab:sets was 121.7 pt wider than the text block; what each network merges and its runs
# wrap in columns of these widths (cm), so the row labels, which never wrap, keep their width
# fourth review (C1): at 3.5 cm the merges wrapped into up to five lines and the table ran into the page footer; the
# label column (about 5.4 cm) and the class column leave about 6.8 cm for these two
SETS_WRAP_CM = (4.4, 2.0)


def table_sets(runs: dict) -> str:
    """tab:sets: every label set and control of the paper, its classes, what it merges from the 188 classes, its runs
    (the random partitions: five groupings, one run each)."""
    rand = [a for a in runs if a.startswith("RAND2_p")]
    wrap = "".join(r">{\raggedright\arraybackslash}p{" + f"{w:g}cm}}" for w in SETS_WRAP_CM)
    out = [r"\begin{tabular}{ll" + wrap + "}", r"\toprule", r"network & classes & merged from 188 classes & runs \\",
           r"\midrule"]
    for arm, k, words in SETS_TABLE:
        n = f"{len(rand)} groupings, one run each" if arm == RAND_POOLED else str(len(runs[arm]))
        out.append(f"{LABEL[arm]} & {k} & {words} & {n}" + r" \\")
    out.append(r"\midrule")
    for arm, k, words, n in SETS_CONTROLS:
        n = n if n is not None else {INIT: len(runs[INIT]), INIT_BN: len(runs.get(INIT_BN, INIT_BN_MODELS)),
                                     SCRATCH: len(FT_SCRATCH_MODELS)}[arm]
        lab = SCRATCH_LABEL if arm == SCRATCH else r"\quad " + INIT_BN_SUB if arm == INIT_BN else LABEL[arm]
        out.append(f"{lab} & {k} & {words} & {n}" + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


FLAGS = ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon")


def input_bn_macros(em, offsets: dict, src: pathlib.Path) -> None:
    """How far a zeroed type flag alone lies from the pretraining particles, in standard deviations."""
    em.macro("MlstInputBnZeroSigmaMax", f"{max(abs(offsets[f]) for f in FLAGS):.2f}", src, "offsets",
             "largest |offset| over the five type flags")


# ------------------------------------------------------------------ X->bb and X->bs (F9, A15)
def _xr30(xfits: dict, m: str, sig: str, name: str, key: str = "r30"):
    r = xfits.get(m, {}).get(sig, {}).get(name)
    if not r:
        return None
    n = xfits.get(m, {}).get(sig, {}).get("n_qcd_in_window")      # a QCD efficiency of zero: >= this count
    if isinstance(r, list):
        v = [V4.rejection(x.get(key), n_pass=0 if x.get(key) is None else None, n_bkg=n) for x in r]
        return None if any(x is None for x in v) else (Bound if any(isinstance(x, Bound) for x in v) else float)(
            float(np.mean(v)))
    return V4.rejection(r.get(key), n_pass=0 if r.get(key) is None else None, n_bkg=n)


def xbx_scratch(data: pathlib.Path) -> dict:
    """{signal: [from-scratch endpoints per seed]} (pulled to xbx_scratch/<signal>/s<k>.json)."""
    d = data / "xbx_scratch"
    return {s.name: [json.loads(p.read_text()) for p in sorted(s.glob("s*.json"))] for s in sorted(d.glob("*"))
            if s.is_dir()} if d.is_dir() else {}


def xbx_ratio_macros(em, xfits, scr, runs, index, fmt_pm) -> None:
    """Rejection 1/eps_B at 30 % signal efficiency in the mass window: the transfer head of the 188-class
    runs over the from-scratch network (X->bs), between label sets, the closest pretrained output node
    (X->bq) and Sophon's direct discriminant (X->bb) over the transfer head."""
    head = lambda sig: (lambda m: _xr30(xfits, m, sig, "head"))
    for sig, w, pre in (("xbs", "Xbs", "XbsRthirty"), ("xbb", "Xbb", "XbbRthirtyTransfer")):
        for arm in ("L162", "R42_Q1", "R16_Q1"):
            r, k = _ratio_vals(runs, arm, "L188", head(sig))
            _emit_ratio(em, f"{pre}{WORD[arm]}OverFull", r, k, fmt_pm, index,
                        f"xbx {sig}: R30 of the transfer head, {arm} over 188, run by run", "rejection")
        full = [head(sig)(m) for m in runs["L188"]]
        sc = [x.get("r30") for x in scr.get(sig, []) if x.get("r30") is not None]
        _emit_over_ref(em, f"{w}RthirtyFullOverScratch", full, len(full), sc, fmt_pm, index,
                       f"xbx {sig}: R30 of the 188-class transfer head over the from-scratch mean", "rejection")
    for sig, name, mac in (("xbs", "zero_shot_xbq", "XbsRthirtyNodeOverTransferFull"),
                           ("xbb", "zero_shot_xbb", "XbbRthirtyDirectOverTransferFull")):
        r = [None if (a := _xr30(xfits, m, sig, name)) is None or not (b := head(sig)(m)) else a / b for m in runs["L188"]]
        _emit_ratio(em, mac, r, len(r), fmt_pm, index, f"xbx {sig}: R30 of {name} over the transfer head, 188-class runs",
                    "rejection")


XBX_ROWS = SETS + FLAV + (RAND_POOLED,) + (INIT,)     # the rows of both X->bb / X->bs tables (PI 2026-11-12)


# the pages of the X->bb / X->bs rejection table (PI 2026-11-13): per signal, the read-outs of tab:xbx (the classifier on
# the frozen features, the linear probe, the direct discriminant); the X->bs closest-class output joins the X->bs page
# (PI 2026-11-21: the old tab:xbxnode merged in)
XBX_REJ_PAGES = (("xbb", ("head", "linear", "zero_shot_xbb")), ("xbs", ("head", "linear", "zero_shot_xbq")))
XBX_READOUT = {"head": "classifier", "linear": "linear probe", "zero_shot_xbb": "direct", "zero_shot_xbq": "closest class"}
XBX_SIGNAL = {"xbb": r"$X\to b\bar b$ against QCD jets", "xbs": r"$X\to bs$ against QCD jets"}


def table_xbx_rej(xfits, scr, runs, fmt_pm, pages=XBX_REJ_PAGES) -> str:
    """Rejection at 50 % and 30 % signal efficiency in the mass window, in the rows of tab:xbx (XBX_ROWS, the five
    random partitions as one row), then from scratch in the classifier's columns (xbx_scratch/<signal>): one page
    (%SPLIT%) per (signal, read-outs) of pages. A dash where every run is fitted and none has the read-out; a row of
    dashes only is left out."""
    runs = _pooled(runs)
    out = []
    for sig, names in pages:
        nc = 1 + 2 * len(names)
        lines = [r"\begin{tabular}{l" + "cc" * len(names) + "}", r"\toprule",
                 rf" & \multicolumn{{{nc - 1}}}{{c}}{{{XBX_SIGNAL[sig]}}} \\", rf"\cmidrule(lr){{2-{nc}}}",
                 " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{XBX_READOUT[n]}}}" for n in names) + r" \\",
                 "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(names))),
                 " & " + " & ".join([r"$R_{50}$ & $R_{30}$"] * len(names)) + r" \\", r"\midrule"]
        for arm in XBX_ROWS:
            ms = runs[arm]
            cells = []
            for name in names:
                if all(sig in xfits.get(m, {}) and xfits[m][sig].get(name) is None for m in ms):
                    cells += ["--", "--"]           # every run is fitted and none has this read-out: it cannot be formed
                    continue
                cells += [_cell([_xr30(xfits, m, sig, name, k) for m in ms], len(ms), fmt_pm) for k in ("r50", "r30")]
            if any(c != "--" for c in cells):
                lines.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        s_ = scr.get(sig, [])
        if s_ and "head" in names:
            lines.append(SCRATCH_LABEL + " & " + " & ".join(
                c for n in names for c in ([_cell([x.get(k) for x in s_], len(s_), fmt_pm) for k in ("r50", "r30")]
                                           if n == "head" else ["--", "--"])) + r" \\")
        out.append("\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(out)


XBX_MACROS = ("XbxMx", "XbxWindowLo", "XbxWindowHi") + tuple(
    f"{s}{k}" for s in ("Xbb", "Xbs") for k in ("NTrainPerClass", "Repetition", "NEvalSignal", "NEvalQcd"))


def xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int, words=str) -> tuple:
    """X->bb and X->bs from the pulled fits through experiments/XBS/xbx_report.py, and the rejection
    ratios and table of A15; red markers until they exist. Returns (the xbx_report module, the fits)."""
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("xbx_report", root / "experiments/XBS/xbx_report.py")
    XR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(XR)
    xfits = V4.load_xbx(data, XR)
    scr = xbx_scratch(data)
    MT = SimpleNamespace(fmt_pm=fmt_pm, fmt_int=fmt_int)
    have = {n for sigs in xfits.values() for n in sigs}
    if xfits:
        XR.macros(em, MT, xfits, index)
        out["tables/mlst_xbx.tex"] = _float("xbx", XR.table(
            SimpleNamespace(LABEL=LABEL, _cell=_cell, RAND=RAND, INIT_MODELS=INIT_MODELS, SCRATCH_LABEL=SCRATCH_LABEL),
            SimpleNamespace(fmt_pm=TAB), runs, xfits, scr))
        out["tables/mlst_xbx_rej.tex"] = _float("xbx_rej", table_xbx_rej(xfits, scr, runs, TAB), True, pages=True,
                                                whole="turned", colsep_pt=6)
        xbx_ratio_macros(em, xfits, scr, runs, index, fmt_pm)
    else:
        missing.append("X->bb and X->bs fits -- experiments/FIGS/data/v2/xbx/")
    # the from-scratch training jets a class of each signal (user 2026-11-19: the captions state them), one value over
    # the seeds or the build stops
    for sig, w in (("xbb", "Bb"), ("xbs", "Bs")):
        ns = {d["training"]["n_train_per_class"] for d in scr.get(sig, [])}
        if len(ns) > 1:
            raise SystemExit(f"FATAL: xbx_scratch/{sig}: the seeds train on different jet counts {sorted(ns)}")
        if ns:
            em.macro(f"MlstXbxScratchTrain{w}", fmt_int(ns.pop()), sorted((data / "xbx_scratch" / sig).glob("s*.json"))[0],
                     "training.n_train_per_class", "training jets a class of the from-scratch network, every seed")
        else:
            em.macro(f"MlstXbxScratchTrain{w}", r"\pending{from-scratch jets}", root / "experiments/XBS/xbx_report.py",
                     "pending", "pending until xbx_scratch is pulled")
    if not scr:
        missing.append("X->bs from scratch -- experiments/FIGS/data/v2/xbx_scratch/")
    # the seeds of the transfer heads (per pretraining run) and of training from scratch (per signal) in words, one
    # count each or the build stops; REQUIRED prints them pending until the files are pulled
    heads = {len(s[g]["head"]) for m, s in xfits.items() if m.startswith("mtx-") for g in s if s[g].get("head")}
    for name, n, files in (("MlstXbxHeadSeedsWord", heads, sorted((data / "xbx").rglob("mtx-*.json"))),
                           ("MlstXbxScratchSeedsWord", {len(v) for v in scr.values()},
                            sorted((data / "xbx_scratch").rglob("s*.json")))):
        if len(n) > 1:
            raise SystemExit(f"FATAL: {name}: the seed counts differ, {sorted(n)}")
        if n and files:
            em.macro(name, words(n.pop()), files[0], "seeds (count, the same for every model or signal)")
    for name in XBX_MACROS:
        if name not in em.provenance:
            sig = {"Xbb": "xbb", "Xbs": "xbs"}.get(name[:3])
            why = "sample not in" if sig and xfits and sig not in have else "samples not in"
            em.macro(name, r"\pending{" + why + "}", root / "experiments/XBS/xbx_report.py", "pending")
    return XR, xfits


# ------------------------------------------------------------------ tab:summary and tab:summaryauc (PI 2026-11-13, L2)
# the panels of the headline family (gallery_labelset.HV_PANELS): (fine-tuning task, its full training set, probe data
# set, headline metric): JetClass-II at 10^6 class-balanced jets, JetClass at 10^6, top at 1.2e6, quark/gluon at
# 1.6e6 scored on Pythia and on Herwig jets. Table 1 (tab:summary) holds the headline metric of the main-text figure
# (F2) and the released Sophon checkpoint's of its full version (F2a); tab:summaryauc (appendix) the AUC beside each
# of them in the full versions (F2a, F2b).
SUMMARY_PANELS = (("jc2", 1_000_000, "jc2_188", "accuracy"), ("jc1", 1_000_000, "jc1", "accuracy"),
                  ("top", 1_200_000, "top", "r30"), ("qg", 1_600_000, "qg", "r30"),
                  ("qg_herwig", 1_600_000, "qg_herwig", "r30"))
SUMMARY_AUC = {"jc2": "macro_auc_ovr", "jc1": "macro_auc_ovr", "top": "auc", "qg": "auc", "qg_herwig": "auc"}
SUMMARY_GRID = tuple(SUMMARY_PANELS)
SUMMARY_AUC_GRID = tuple((key, n, ds, SUMMARY_AUC[key]) for key, n, ds, _ in SUMMARY_PANELS)
SUMMARY_HEAD = {"accuracy": "accuracy", "r30": "$R_{30}$", "macro_auc_ovr": "AUC", "auc": "AUC",
                "macro_auc_resonant": "AUC, resonant", "macro_auc_qcd": "AUC, QCD"}
# PI 2026-11-23 (LEDGER SP-9; merge 5): tab:summaryauc takes JetClass-II's AUC over its resonant and over its QCD classes
# beside the AUC over every class (the class-balanced half of tab:balance; linear_probe_v2.native_metrics, the
# fine-tuned cells' macro_auc_resonant and macro_auc_qcd)
SUMMARY_JC2_GROUPS = ("macro_auc_resonant", "macro_auc_qcd")
# PI 2026-11-23 (LEDGER SP-13, step 4): the top and quark/gluon AUC columns leave the paper's table, as no main-text
# sentence quotes an AUC value of them (dropped_auc_check); their means stay in the supplement's full copy,
# tab:summaryaucfull, every column, which the data release writes as it writes every generated table
SUMMARY_AUC_DROPPED = ("top", "qg", "qg_herwig")
SUMMARY_AUC_FULL = (SUMMARY_AUC_GRID[:1] + tuple(("jc2", 1_000_000, "jc2_188", met) for met in SUMMARY_JC2_GROUPS)
                    + SUMMARY_AUC_GRID[1:])
SUMMARY_AUC_TABLE = tuple(g for g in SUMMARY_AUC_FULL if g[0] not in SUMMARY_AUC_DROPPED)
# the macros that print a value of a dropped column: the AUC of a network fine-tuned or probed on top or quark/gluon
# tagging (ft_macros' MlstFt{Top,Qg,QgHerwig}Auc{Full,Scratch}; FT_WORD names the tasks)
DROPPED_AUC_MACRO = re.compile(r"Mlst(?:Ft|Lp)(?:" + "|".join(FT_WORD[k] for k in SUMMARY_AUC_DROPPED) + r")Auc[A-Z]")


def dropped_auc_check(journal: pathlib.Path) -> None:
    """The condition of SP-13 step 4, checked where the text is read: the build stops if main.tex or a section (comments
    left out) uses a macro that prints a value of a column tab:summaryauc dropped."""
    files = [journal / "main.tex", *sorted((journal / "sections").glob("*.tex"))]
    used = {n for f in files if f.exists() for n in re.findall(r"\\([A-Za-z]+)", re.sub(r"(?<!\\)%.*", "", f.read_text()))}
    bad = sorted(n for n in used if DROPPED_AUC_MACRO.match(n))
    if bad:
        raise SystemExit(f"FATAL: the text quotes {bad}, the top or quark/gluon AUC that tab:summaryauc no longer prints "
                         "(PI 2026-11-23, LEDGER SP-13 step 4: a quoted value needs its table)")


# merge 4: Table 1 takes tab:xbx's two transfer-head columns, the classifier on frozen features (xbx_report's "head"
# read-out), X->bb and X->bs signal efficiency in percent at the QCD efficiency of the fits (1%), every row of the
# linear-probe part with a head (the random network has only a linear probe) and from scratch in the fine-tuned part
SUMMARY_XBX = (("xbb", r"$X\to b\bar b$"), ("xbs", r"$X\to bs$"))
SUMMARY_CONTROLS = FLAV + (RAND_POOLED,)
# its top and quark/gluon probes are drawn only with the same substitutions as ours (with_sophon_fill, PI 2026-11-22);
# a zero-filled probe there is never shown
SUMMARY_NO_SOPHON = ("top", "qg", "qg_herwig")
# the L3 bootstrap of each headline cell (A3, PI 2026-11-14): (mode, task) -> file under bootstrap/l3 (a linear probe
# of top or quark/gluon: the one of the substitution chosen on validation, "{s}"); metric -> its key there
SUMMARY_L3 = {("lp", "jc2"): "probe_jc2bal_N1000000.json", ("lp", "jc1"): "probe_jc1_N1000000.json",
              ("lp", "top"): "probe_top_N1200000.{s}.json", ("lp", "qg"): "probe_qg_N1600000.{s}.json",
              ("lp", "qg_herwig"): "probe_qg_herwig_N1600000.{s}.json",
              ("ft", "jc2"): "ft_jc2bal_N1000000.json", ("ft", "jc1"): "ft_jc1_N1000000.json",
              ("ft", "top"): "ft_top_N1200000.json", ("ft", "qg"): "ft_qg_N1600000.json",
              ("ft", "qg_herwig"): "ft_qg_herwig_N1600000.json"}
SUMMARY_L3_METRIC = {"accuracy": "accuracy", "r30": "r30", "macro_auc_ovr": "macro", "auc": "auc",
                     # the AUC over JetClass-II's resonant and over its QCD classes (the .all copy; tab:tests, SP-13)
                     "macro_auc_resonant": "macro_resonant", "macro_auc_qcd": "macro_qcd"}


def summary_l3(data: pathlib.Path, mode: str, key: str, met: str, chosen: dict) -> tuple[dict, str]:
    """(the L3 bootstrap document, its metric key) of a headline cell, ({}, "") where none is stored."""
    name = SUMMARY_L3.get((mode, key))
    if name is None or SUMMARY_L3_METRIC.get(met) is None:
        return {}, ""
    if "{s}" in name:
        sub = chosen.get(key)
        if sub is None:
            return {}, ""
        name = name.format(s=SUBST_KEY[key][sub].split(".")[-1])
    return _l3_doc(data, name, mode == "lp"), SUMMARY_L3_METRIC[met]


def summary_probe(fits, fill, chosen, ds: str, met: str):
    """A run's linear-probe value as Fig. 2 and its full version draw it (gallery_labelset._probe_value): at the largest
    training set; top and quark/gluon (also scored on Herwig jets, ds "qg_herwig") under the substitution chosen on
    validation. None when the task has no fits."""
    if ds in ("top", "qg", "qg_herwig"):
        sizes = _bench_sizes(fits, fill, chosen, ds)
        return (lambda m: _rej(_bench_rec(fits, fill, chosen, m, ds, sizes[-1]), met)) if sizes else None
    if not any(ck.get(ds) for f in fits.values() for ck in f.get("cells", {}).values()):
        return None
    n = _largest(fits, ds)
    return lambda m: (lambda r: None if r is None else r.get(met))(_probe_rec(fits, m, ds, n))


def summary_ft(per, key: str, n: int, met: str):
    """A run's fine-tuned value as Fig. 2 and its full version draw it (gallery_labelset._ft_value), or None without
    fine-tuning at n. The JetClass-II class-group AUCs (SUMMARY_JC2_GROUPS) are None in a read-out without them."""
    if not per.get(key) or n not in _sizes(per, key):
        return None
    if met in SUMMARY_JC2_GROUPS:
        return lambda m: (lambda r: None if r is None else r.get(met))(_ft_rec(per, key, m, n))
    return lambda m: _ft(per, key, m, n, met)


def _ft_rec(per: dict, key: str, model: str, n: int):
    """One pretraining run's fine-tuned record at n (its one fine-tuning seed), or None."""
    recs = per[key].get(model[len("mtx-"):], {}).get(n)
    if recs is not None and len(recs) != 1:
        raise SystemExit(f"FATAL: {model} {key} N={n}: {len(recs)} fine-tuning seeds, expected one")
    return recs[0] if recs else None


# review of 5c40d69 (B15, LEDGER CA-12): the mode heading over Table 1's X->bb / X->bs columns in the linear-probe part
XBX_HEAD = "transfer head"
# Table 1's row of the random network with BatchNorm recomputed, indented under the random network's (A2)
INIT_BN_SUB = "BatchNorm recomputed"


SOPHON_XBX_PENDING = r"\pending{no run yet}"     # Table 2's Sophon transfer-head cells until the SH-13 fits


def summary_xbx(xbx: dict | None) -> dict:
    """The X->bb / X->bs columns of Table 1 (SUMMARY_XBX): {"cell": f(models, signal) -> the transfer head's signal
    efficiency in percent, mean +- SD over the runs, "scratch": f(signal) -> from scratch's, "star": the one QCD
    efficiency of every fit}; {} without the fits. xbx: {"XR": xbx_report, "fits": its fits, "scr": xbx_scratch}."""
    if not xbx or not xbx.get("fits"):
        return {}
    XR, xfits, scr = xbx["XR"], xbx["fits"], xbx.get("scr") or {}
    stars = {r["eps_b_star"] for sigs in xfits.values() for v in sigs.values() if (r := XR.readout(v, "head"))}
    stars |= {x["eps_b_star"] for v in scr.values() for x in v}
    if len(stars) != 1:
        raise SystemExit(f"FATAL: the X->bb / X->bs fits use {len(stars)} QCD efficiencies {sorted(stars)}; Table 1 "
                         "states one")

    def cell(ms, sig, fmt_pm):
        v = [r["eps_s_at_star"] for m in ms if (r := XR.readout(xfits.get(m, {}).get(sig, {}), "head"))]
        return _cell(v, len(ms), fmt_pm, 100.0) if v else "--"

    def scratch(sig, fmt_pm):
        v = [x["eps_s_at_star"] for x in scr.get(sig, [])]
        return _cell(v, len(v), fmt_pm, 100.0) if v else "--"
    return {"cell": cell, "scratch": scratch, "star": stars.pop()}


def f2_init_macros(em, fits, fill, chosen, runs: dict, src) -> None:
    """Figure F2's random network, not drawn (PI 2026-11-20), in one clause of its caption, computed from the files
    (PI 2026-11-23, LEDGER CA-16): its mean accuracy on JetClass-II and on JetClass and the lowest and highest of its
    mean R30 on top and quark/gluon tagging, as Table 1's random-network row reads them (summary_probe), at two
    significant figures (as the class-pair figures' clauses, pair_figure_macros)."""
    fmt = lambda x: f"{x:#.2g}" if x < 9.95 else f"{x:.0f}"   # noqa: E731  (# keeps a trailing zero: 0.10, 4.0)
    # PI 2026-11-23 (LEDGER SH-13): the random network with BatchNorm recomputed, not drawn either, gets the same clause
    for arm, word in ((INIT, "Init"), (INIT_BN, "InitBn")):
        mean = {}
        for key, _, ds, met in SUMMARY_PANELS:
            g = summary_probe(fits, fill, chosen, ds, met)
            v = [g(m) for m in runs.get(arm, [])] if g else []
            mean[key] = float(np.mean([float(x) for x in v])) if v and None not in v else None
        r30 = [mean[k] for k in ("top", "qg", "qg_herwig")]
        out = {f"MlstFTwo{word}AccJcTwo": mean["jc2"], f"MlstFTwo{word}AccJcOne": mean["jc1"],
               f"MlstFTwo{word}RthirtyMin": min(r30) if None not in r30 else None,
               f"MlstFTwo{word}RthirtyMax": max(r30) if None not in r30 else None}
        for name, x in out.items():
            em.macro(name, fmt(x) if x is not None else r"\pending{random network}", src,
                     f"linear probes of the random network ({arm}), Table 1's cells, mean over its initializations",
                     "figure F2's caption: the random network, not drawn")


def _size_values(fits, fill, chosen, models: list, ds: str, met: str) -> list:
    """[mean over models at each training-set size] of a linear probe, as the training-size figures read their points
    (paper_figures.probe_series): every size of the task, top and quark/gluon under the substitution chosen on
    validation, a size where a run's rejection is a lower limit (no background jet passed) not drawn."""
    if ds in SUBST_KEY:
        sizes = _bench_sizes(fits, fill, chosen, ds)
        recs = lambda n: [_rej(_bench_rec(fits, fill, chosen, m, ds, n), met) for m in models]     # noqa: E731
    else:
        sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
        recs = lambda n: [(lambda r: None if r is None else r.get(met))(_probe_rec(fits, m, ds, n))     # noqa: E731
                          for m in models]
    out = []
    for n in sizes:
        v = [x for x in recs(n) if x is not None]
        if v and not any(isinstance(x, Bound) for x in v):
            out.append(float(np.mean([float(x) for x in v])))
    return out


def a10_init_macros(em, fits, fill, chosen, runs: dict, src) -> None:
    """Figure A10's random network, not drawn (PI 2026-11-19 (B): its points would set the y range), in one clause of
    its caption (PI 2026-11-23, LEDGER SH-13): the lowest and highest over the training-set sizes of its mean
    linear-probe accuracy on JetClass-II and on JetClass, and of its mean R30 on top and quark/gluon tagging (Pythia
    and Herwig), the points as the figure reads them (_size_values), at two significant figures (f2_init_macros)."""
    fmt = lambda x: f"{x:#.2g}" if x < 9.95 else f"{x:.0f}"   # noqa: E731  (# keeps a trailing zero: 0.10, 4.0)
    vals = {key: _size_values(fits, fill, chosen, runs.get(INIT, []), ds, met) for key, _, ds, met in SUMMARY_PANELS}
    r30 = vals["top"] + vals["qg"] + vals["qg_herwig"]
    groups = {"AccJcTwo": vals["jc2"], "AccJcOne": vals["jc1"],
              "Rthirty": r30 if all(vals[k] for k in ("top", "qg", "qg_herwig")) else []}
    for word, v in groups.items():
        for end, f in (("Min", min), ("Max", max)):
            em.macro(f"MlstATenInit{word}{end}", fmt(f(v)) if v else r"\pending{random network}", src,
                     f"linear probes of the random network ({INIT}), mean over its initializations at each training-set "
                     f"size, the {end.lower()} over the sizes", "figure A10's caption: the random network, not drawn")


def _summary_head(grid, xb: dict) -> list:
    """The two header rows of tab:summary and tab:summaryauc: the task over its columns, one heading over a task's
    several columns (JetClass-II with its class groups, quark/gluon over Pythia and Herwig) with its rule, then the
    metric and, under quark/gluon, the generator of the jets scored (both trained on Pythia jets); with xb the two
    X->bb / X->bs columns under their working point."""
    tasks = {"jc2": "JetClass-II", "jc1": "JetClass", "top": "top tagging", "qg": "quark/gluon tagging"}
    gen = {"qg": ", Pythia", "qg_herwig": ", Herwig"}
    groups = []                      # [task, first column, last column], columns counted from 2
    for c, (k, *_) in enumerate(grid, 2):
        t = "qg" if k.startswith("qg") else k
        if groups and groups[-1][0] == t:
            groups[-1][2] = c
        else:
            groups.append([t, c, c])
    if xb:
        c = 2 + len(grid)
        groups.append([rf"$\varepsilon_S$ (\%) at $\varepsilon_B = {100 * xb['star']:g}\%$", c, c + len(SUMMARY_XBX) - 1])
    name = lambda t, a, b: tasks.get(t, t) if a == b else rf"\multicolumn{{{b - a + 1}}}{{c}}{{{tasks.get(t, t)}}}"   # noqa: E731
    return [" & " + " & ".join(name(t, a, b) for t, a, b in groups) + r" \\",
            "".join(rf"\cmidrule(lr){{{a}-{b}}}" for _, a, b in groups if b > a),
            " & " + " & ".join([SUMMARY_HEAD[met] + gen.get(k, "") for k, _, _, met in grid]
                               + ([h for _, h in SUMMARY_XBX] if xb else [])) + r" \\"]


def summary_scratch(per, key: str, n: int, met: str) -> list:
    """Each from-scratch seed's value on n training jets, as Fig. 2 and its full version draw them
    (gallery_labelset._scratch_value)."""
    return [r.get(met) for r in per.get(key, {}).get(SCRATCH, {}).get(n, []) if r.get(met) is not None]


def table_randparts(fits, fill, chosen, per, runs, fmt_pm, grid=SUMMARY_GRID) -> str:
    """tab:randparts (PI 2026-11-19, E: each of the five random partitions on its own row, in an appendix table): Table
    1's tasks and metrics, as linear probes and fine-tuned, each partition's one run, then their mean +- SD (the
    random-partitions row of the other tables). No bold: single runs of one control."""
    rand = [a for a in runs if a.startswith("RAND2_p")]
    pooled = _pooled(runs)[RAND_POOLED]
    tasks = {"jc2": "JetClass-II", "jc1": "JetClass", "top": "top tagging", "qg": r"\multicolumn{2}{c}{quark/gluon tagging}"}
    qg = [i for i, (k, *_) in enumerate(grid, 2) if k.startswith("qg")]
    gen = {"qg": ", Pythia", "qg_herwig": ", Herwig"}
    head = [r"\begin{tabular}{l" + "c" * len(grid) + "}", r"\toprule",
            " & " + " & ".join(tasks[k] for k, *_ in grid if k in tasks) + r" \\", rf"\cmidrule(lr){{{qg[0]}-{qg[-1]}}}",
            " & " + " & ".join(SUMMARY_HEAD[met] + gen.get(k, "") for k, _, _, met in grid) + r" \\", r"\midrule"]
    parts = []
    for mode in ("lp", "ft"):
        get = ([summary_probe(fits, fill, chosen, ds, met) for _, _, ds, met in grid] if mode == "lp"
               else [summary_ft(per, k, n, met) for k, n, _, met in grid])
        body = [_full_row(1 + len(grid), HEADING[mode], italic=True)]
        for name, ms in [(LABEL[a], runs[a]) for a in rand] + [(LABEL[RAND_POOLED], pooled)]:
            c = []
            for g, (_, _, _, met) in zip(get, grid):
                v = [g(m) for m in ms] if g else []
                c.append(_cell(v, len(ms), fmt_pm, frac=met != "r30") if any(x is not None for x in v) else "--")
            body.append(f"{name} & " + " & ".join(c) + r" \\")
        parts.append("\n".join(body))
    parts = uniform_columns("\n%MODEBREAK%\n".join(parts)).split("\n%MODEBREAK%\n")
    return "%SPLIT%".join("\n".join(head + q.split("\n") + [r"\bottomrule", r"\end{tabular}", ""]) for q in parts)


# review of render 5f4e1b8 (B17): what each mode of Table 2 puts in the missing top and quark/gluon inputs
SUMMARY_FILL_NOTE = {"lp": r", missing inputs \LpSubstitutionTop{} (top), \LpSubstitutionQg{} (quark/gluon)",
                     "ft": ", missing inputs set to zero"}


def table_summary(fits, fill, chosen, per, runs, fmt_pm, grid=SUMMARY_GRID, key="summary", data=None,
                  xbx: dict | None = None) -> str:
    """The values of the headline family on grid, one %SPLIT% part (a turned page) per mode: linear probes (the label
    sets, each mass network under its own, the 17-class partitions with the five random partitions as one row, the
    random network, and below a rule the released Sophon checkpoint on JetClass-II and JetClass), then fine-tuned (the
    same networks, from scratch last). fits: the probe fits with the released checkpoint's merged in, as the figures
    read them; runs: with SOPHON. A row with no value for a task: a dash. A partition row with no value on any task is
    left out. key: the table's BOLD entry (tab:summary, tab:summaryauc). xbx (Table 1, PI 2026-11-23, LEDGER SP-9):
    the X->bb / X->bs transfer-head columns after the grid's (summary_xbx), the pretrained networks' in the
    linear-probe part (the released Sophon checkpoint's pending until its fits are in), from scratch's in the fine-tuned
    part, a dash in every other cell."""
    runs = _pooled(runs)
    mode_now = ["lp"]          # the mode cells() is building, for the L3 bootstrap of its cells (A3)
    xb = summary_xbx(xbx)

    def cells(arm, get):
        ms = runs[arm]
        out = []
        for g, (k, _, ds, met) in zip(get, grid):
            v = [g(m) for m in ms] if g and not (arm == SOPHON and ds in SUMMARY_NO_SOPHON
                                                 and SOPHON_MODEL not in fill) else []
            # review of render dfe5942 (C): a fine-tuned value not in yet is the rerun's pending mark, never the dash
            c = (_cell(v, len(ms), fmt_pm, frac=met != "r30") if any(x is not None for x in v)
                 else "--" if mode_now[0] == "lp" else V4.RERUN_PENDING)
            if data is not None and V4.TOK_OPEN in c:
                c = _boot_cell(c, *summary_l3(data, mode_now[0], k, met, chosen or {}), ms)
            out.append(c)
        if xb:
            # PI 2026-11-23 (LEDGER SH-13, CA-16): the released Sophon checkpoint's transfer heads are queued, pending
            # until their fits are in; a dash is only what the caption's dash clause names
            out += [(lambda c: SOPHON_XBX_PENDING if arm == SOPHON and c == "--" else c)(xb["cell"](ms, sig, fmt_pm))
                    if mode_now[0] == "lp" else "--" for sig, _ in SUMMARY_XBX]
        return out

    def scratch():
        out = []
        for k, n, _, met in grid:
            v = summary_scratch(per, k, n, met)
            out.append(_cell(v, len(v), fmt_pm, frac=met != "r30") if v else V4.RERUN_PENDING)
        return out + ([xb["scratch"](sig, fmt_pm) for sig, _ in SUMMARY_XBX] if xb else [])

    row = lambda name, c: f"{name} & " + " & ".join(c) + r" \\"       # noqa: E731
    nc = 1 + len(grid) + (len(SUMMARY_XBX) if xb else 0)
    head = [r"\begin{tabular}{l" + "c" * (nc - 1) + "}", r"\toprule", *_summary_head(grid, xb), r"\midrule"]
    head = [h for h in head if h]           # no rule row when no task spans columns
    parts = []
    for mode in ("lp", "ft"):
        mode_now[0] = mode
        get = ([summary_probe(fits, fill, chosen, ds, met) for _, _, ds, met in grid] if mode == "lp"
               else [summary_ft(per, k, n, met) for k, n, _, met in grid])
        # review of render 5f4e1b8 (B17): Table 2's block headings say what each mode puts in the missing top and
        # quark/gluon inputs (in the headings, as the caption is at its 135 words)
        title = HEADING[mode] + (SUMMARY_FILL_NOTE[mode] if key == "summary" else "")
        head_row = _full_row(nc, title, italic=True)
        if xb:      # the X->bb / X->bs columns are read by a transfer head, not a linear probe (CA-12), and in the
            # fine-tuned block only the network trained from scratch fills them (review of render 5f4e1b8, C5)
            head_row = (rf"\multicolumn{{{1 + len(grid)}}}{{@{{}}l}}{{\textit{{{title}}}}} & "
                        rf"\multicolumn{{{len(SUMMARY_XBX)}}}{{c}}{{\textit{{{XBX_HEAD if mode == 'lp' else SCRATCH_LABEL}}}}} \\")
        body = [head_row] + [row(LABEL[a], cells(a, get)) for a in SETS]
        ctl = [(r"\quad " + LABEL[a], cells(a, get)) for a in SUMMARY_CONTROLS]
        ctl = [(n, c) for n, c in ctl if any(x != "--" for x in c)]
        if ctl:
            body += [_full_row(nc, PARTITIONS)] + [row(n, c) for n, c in ctl]
        if mode == "lp":
            body.append(row(LABEL[INIT], cells(INIT, get)))
            if key == "summary" and INIT_BN in runs:
                # the random network with BatchNorm recomputed (methods), the reference of the text's ratios on top
                # and quark/gluon tagging: a quoted ratio is computable from a table in the paper (LEDGER CS-6; review
                # of 5c40d69, A2: its values at the chosen substitution were printed nowhere); under the random
                # network's row, so that its label keeps the turned page's width (TN-6: no wrapped label)
                body.append(row(r"\quad " + INIT_BN_SUB, cells(INIT_BN, get)))
            c = cells(SOPHON, get) if SOPHON in runs else []
            if any(x != "--" for x in c[:len(grid)]):
                body += [r"\midrule", row(LABEL[SOPHON], c)]
        else:
            body.append(row(SCRATCH_LABEL, scratch()))
        parts.append(_bold(key, "\n".join(body)))
    # D1 (round-1 fix 2026-11-13): one precision per column across both modes, so a column's linear-probe and
    # fine-tuned cells print at the same place
    parts = uniform_columns("\n%MODEBREAK%\n".join(parts)).split("\n%MODEBREAK%\n")
    return "%SPLIT%".join("\n".join(head + p.split("\n") + [r"\bottomrule", r"\end{tabular}", ""]) for p in parts)


# ------------------------------------------------------------------ tab:tests, the tested differences (appendix)
# PI 2026-11-23 (LEDGER SP-9; merge 6): one row per difference or null the text states (the claim inventories of
# scratchpad/part0/claims_main.md and claims_appendix.md, less the claims SP-10 removes), tab:sculptboot and tab:boot in
# it. PI SC-1 (1): sigma^2 = each side's run SD^2 + the paired test-jet bootstrap SD^2; D7 superseded: the test runs on
# the metric as printed (an AUC difference is minus that of 1 - AUC, sigma unchanged).
# A block is (heading, rows); a row (first, second, source), first and second an arm of LABEL or a name of TESTS_NAMES;
# a source is where its paired bootstrap is stored:
#   ("l3", mode, task, metric, difference, every)  bootstrap/l3 (fine-tuned) or l3_v5 / l3 (linear probes, _l3_doc) of
#       Table 1's cell (SUMMARY_L3; top and quark/gluon probes under the substitution chosen on validation), every: the
#       .all copy that holds the partitions and each network against 188 classes;
#   ("file", path, difference, metric)  bootstrap/<path> (PI 2026-11-23, LEDGER SP-13: the rows bootstrapped for
#       tab:tests), diffs[difference][metric]; each side's runs and values checked against those the paper prints
#       (table_tests); two independent test samples (Herwig against Pythia jets) are resampled apart, the two bootstrap
#       variances adding (PI SC-1a), and the row is marked;
#   ("mass", difference, probe)  MASS_SD_BOOT_TESTS, all jets, inclusive; ("mass_group", network, probe) its network
#       against the soft-drop mass on the same jets, which has no run spread;
#   RERUN  a bootstrap of fine-tuning runs since rerun (SP-11): \pending{rerun bootstrap};
#   None  no paired bootstrap stored: the row prints \pending{bootstrap} until a later step computes it.
# The sculpting rows follow from the sculpting bootstrap (l3_rerun/ once pulled, else bootstrap/l3), its "l3" block.
TESTS_NAMES = {"SCRATCH": SCRATCH_LABEL, "SDMASS": "soft-drop mass", "PART_PUB": "ParT from scratch, published",
               # the review's rows (5c40d69, B3): the linear probes with zeros against every fine-tuned network, and
               # the regressed mass in every bin of F9b, each a row of the least |Delta / sigma| (_least)
               "L188_LP_ZERO": r"\Kfull{} classes, zero", "INIT_BN_LP_ZERO": "random network, BatchNorm recomputed, zero",
               "FT_EVERY": "fine-tuned networks", "L162_MASS_BINS": r"\Kres{} classes + mass, head",
               "R16_Q1_MASS_BINS": r"\Kcoarse{} classes + mass, head",
               "PART_FT_PUB": "ParT fine-tuned, published", "PART_OURS": "ParT's script on our cluster",
               "PART_RESCALE": "ParT's script with Sophon's rescaling",
               "FLOOR": "random selection of as many QCD jets",
               "L188_DRAWS": r"\Kfull{} classes, random draws", "L188_ZERO": r"\Kfull{} classes, zero",
               "INIT_BN_ZERO": "random network, BatchNorm recomputed, zero", "L188_HERWIG": r"\Kfull{} classes, Herwig over Pythia",
               "SCRATCH_HERWIG": "from scratch, Herwig over Pythia", "L188_17": r"\Kfull{} classes on the 17 groups",
               "L162_17": r"\Kres{} classes on the 17 groups",
               "STEP_17_43": r"step from \Kmid{} to \Kcoarse{} classes", "STEP_43_162": r"step from \Kres{} to \Kmid{} classes",
               "STEP_162_188": r"step from \Kfull{} to \Kres{} classes",
               "R42_Q1_17": r"\Kmid{} classes on the 17 groups", "R16_Q1_17": r"\Kcoarse{} classes on the 17 groups",
               "L162_MASS_HEAD": r"\Kres{} classes + mass, head", "R16_Q1_MASS_HEAD": r"\Kcoarse{} classes + mass, head",
               # PI 2026-11-23 (LEDGER SP-12, step 3): the rows of the ratio words that had none (PR_ROWS); review of
               # render dfe5942 (C): one name, the MLP probe
               "L162_MLP": r"\Kres{} classes, multilayer-perceptron probe",
               "L188_HERWIG_JETS": r"\Kfull{} classes on Herwig jets", "L188_PYTHIA_JETS": r"\Kfull{} classes on Pythia jets",
               "SCRATCH_HERWIG_JETS": "from scratch on Herwig jets", "SCRATCH_PYTHIA_JETS": "from scratch on Pythia jets"}
_LEVELS = (("L162", "L188", "162-188"), ("R42_Q1", "L188", "43-188"), ("R16_Q1", "L188", "17-188"))
_MASSES = (("L162_MASS", "L162", "162mass-162"), ("R16_Q1_MASS", "R16_Q1", "17mass-17"))
_MASSES_ALL = (("L162_MASS", "l162mass-188"), ("R16_Q1_MASS", "r16q1mass-188"))
_PARTS_ALL = (("FLAV_F0", "flavf0-188"), ("FLAV_F1", "flavf1-188"), ("FLAV_F1R", "flavf1r-188"), (RAND_POOLED, "rand-188"))
_SIZES = (1_000, 10_000, 100_000)       # the training-set sizes below a million of Figs. A8 and A10 (SIZE_TASKS)
RERUN = ("rerun",)
# the side of a row named in TESTS_NAMES: the configuration whose runs it holds (the step's sides hold three, STEP_RUNS)
TESTS_ARM = {"L188_DRAWS": "L188", "L188_ZERO": "L188", "INIT_BN_ZERO": INIT_BN, "L188_HERWIG": "L188",
             "L188_LP_ZERO": "L188", "INIT_BN_LP_ZERO": INIT_BN,
             "SCRATCH_HERWIG": "SCRATCH", "L188_17": "L188", "L162_17": "L162", "R42_Q1_17": "R42_Q1",
             "R16_Q1_17": "R16_Q1", "L162_MASS_HEAD": "L162_MASS", "R16_Q1_MASS_HEAD": "R16_Q1_MASS", "L162_MLP": "L162",
             "L188_HERWIG_JETS": "L188", "L188_PYTHIA_JETS": "L188", "SCRATCH_HERWIG_JETS": "SCRATCH",
             "SCRATCH_PYTHIA_JETS": "SCRATCH"}
# the sides of each step row: three, (a - b) - (b - c); four, (a - b) - (c - d) (review of render dfe5942, BS2: the
# 43 -> 17 and 162 -> 43 steps against 188 -> 162, bootstrap/l3_v5/probe_jc2bal_N1000000.review2.json)
STEP_RUNS = {"step-r16q1-r42q1-l162": {"a": "R16_Q1", "b": "R42_Q1", "c": "L162"},
             "step-r42q1-l162-188": {"a": "R42_Q1", "b": "L162", "c": "L188"},
             "step-r16q1-r42q1-l162-188": {"a": "R16_Q1", "b": "R42_Q1", "c": "L162", "d": "L188"}}
_JC2_ALL = "l3_v5/probe_jc2bal_N1000000.all.json"
_JC2_REVIEW2 = "l3_v5/probe_jc2bal_N1000000.review2.json"


def _bench_block(task: str, mode: str, own: bool = False) -> tuple:
    """Table 1's R30 nulls on top or quark/gluon tagging (M009, M047, M232, M261, M303, M327): every network against
    188 classes (PI 2026-11-23, LEDGER CA-15: the mass networks too) and, own, each mass network against its own label
    set (results_mass reads it on fine-tuned top tagging only; PI 2026-11-23, LEDGER SP-13)."""
    rows = [(a, b, ("l3", mode, task, "r30", d, False)) for a, b, d in _LEVELS]
    rows += [(a, "L188", ("l3", mode, task, "r30", d, True)) for a, d in _PARTS_ALL]
    rows += [(a, b, ("l3", mode, task, "r30", d, False)) for a, b, d in _MASSES] if own else []
    rows += [(a, "L188", ("l3", mode, task, "r30", d, True)) for a, d in _MASSES_ALL]
    return tuple(rows)


def _jc2_lp(met: str, rows) -> tuple:
    """JetClass-II linear-probe rows (first, second, difference) of the .all bootstrap of Table 1's cell."""
    return tuple((a, b, ("l3", "lp", "jc2", met, d, True)) for a, b, d in rows)


def _review2_rows(met: str) -> tuple:
    """The review's rows of the JetClass-II linear probes at 10^6 (review of render dfe5942, BS2): the merges against the
    first, 188 -> 162 ("the costliest single merge", in AUC; the largest accuracy step), and the b-quark and random
    flavor boundaries against 17 classes (results_jc: between 17 classes and the random partitions)."""
    return (("STEP_43_162", "STEP_162_188", ("file", _JC2_REVIEW2, "step-r42q1-l162-188", met)),
            ("STEP_17_43", "STEP_162_188", ("file", _JC2_REVIEW2, "step-r16q1-r42q1-l162-188", met)),
            ("FLAV_F1", "R16_Q1", ("file", _JC2_REVIEW2, "flavf1-r16q1", met)),
            ("FLAV_F1R", "R16_Q1", ("file", _JC2_REVIEW2, "flavf1r-r16q1", met)))


def _pair_rows(key: str, rows, copy: str = "") -> tuple:
    """A class pair's linear-probe AUC rows (first, second, difference) of its bootstrap file (copy: a later file of the
    same pair, ".flav188")."""
    return tuple((a, b, ("file", f"l3_v5/probe_jc2bal_pairs_{key}{copy}.json", d, "auc")) for a, b, d in rows)


# results_jc: the flavor-boundary partitions recover part of the loss of 17 classes, below 188 classes (review of
# 5c40d69, B3; bootstrap/l3_v5/probe_jc2bal_pairs_<pair>.flav188.json)
_FLAV188 = {"bb_vs_cc": (("FLAV_F1", "L188", "flavf1-188"),),
            "bc_vs_bq_cs": (("FLAV_F1", "L188", "flavf1-188"), ("FLAV_F1R", "L188", "flavf1r-188"))}
# a network as the mass bootstrap names it ("162 classes + mass")
_MASS_NAME = lambda a: (LABEL[a].replace(r"\Kfull{}", "188").replace(r"\Kres{}", "162")      # noqa: E731
                        .replace(r"\Kmid{}", "43").replace(r"\Kcoarse{}", "17"))
# tab:tests rows from the review's bootstrap of fine-tuned top tagging (bootstrap/l3/ft_top_N1200000.review.json)
_FT_REVIEW = "l3/ft_top_N1200000.review.json"
_REVIEW_KEY = {"L188": "l188", "L162": "l162", "L162_MASS": "l162mass", "R42_Q1": "r42q1", "R16_Q1": "r16q1",
               "R16_Q1_MASS": "r16q1mass", "FLAV_F0": "flavf0", "FLAV_F1": "flavf1", "FLAV_F1R": "flavf1r",
               RAND_POOLED: "rand"}
# F9b's bins (jet pT, ten; generator-level mass, five): the regressed mass narrower than the soft-drop mass in every
# bin, one row a network, the least |Delta / sigma| of its bins (review of 5c40d69, B3; head/mass_sd_boot_v5.json)
F9B_BINS = tuple(f"all|pt_fine:{b}" for b in ("200-300", "300-400", "400-550", "550-700", "700-850", "850-1000",
                                              "1000-1250", "1250-1500", "1500-2000", "2000-2500")) + \
    tuple(f"all|mgen:{b}" for b in ("20-50", "50-100", "100-150", "150-250", "250-500"))


_PAIR = {k: h for k, h, *_ in PAIRS}
_QCD_PAIR = (("L162", "L188", "l162-188"), ("R42_Q1", "L188", "r42q1-188"), ("R16_Q1", "L188", "r16q1-188"))
_FT_TOP, _FT_QG = (r"top tagging, $R_{30}$, " + HEADING["ft"], r"quark/gluon tagging (Pythia), $R_{30}$, " + HEADING["ft"])
# PI 2026-11-23 (LEDGER SP-13): a row stays only if (a) a main-text sentence or caption states it, (b) it backs a method
# check the main text relies on (the ParT reproduction, the fill choice, the coarse-task null) or (c) it is a 3-sigma-
# passing result about label granularity, or one waiting for its bootstrap that stays if it passes; the others left
# (their values stay in the data release)
# SC-2 (#1; bs3, 2026-11-23; the PI is asked to confirm the term): the transfer-head rows use an analytic test-sample
# term, as their heads' per-jet scores were not kept; XBX_ANALYTIC = False puts them back to pending (no term)
XBX_ANALYTIC = True
XBX_EPS_B = 0.01
ANALYTIC_MARK = r"$^{\dagger}$"
LIT_BENCH = pathlib.Path(__file__).resolve().parent / "data" / "literature_benchmarks.json"
# R5-17 (bs3): the 17-class level, which also merges electron and muon, beside the 43-class one
_EE_17 = (("R16_Q1", "L188", ("file", "l3_bs3/probe_jc2bal_pairs_ee_vs_mm.r16q1.json", "r16q1-188", "auc")),)
_XBX_SRC = lambda sig: ("xbx_analytic", sig) if XBX_ANALYTIC else None       # noqa: E731  SC-2 (#1), easy to revert
_MLP_NETS = (("L162", "l162-188"), ("R42_Q1", "r42q1-188"), ("R16_Q1", "r16q1-188"), ("L162_MASS", "l162mass-188"),
             ("R16_Q1_MASS", "r16q1mass-188"), ("FLAV_F0", "flavf0-188"), ("FLAV_F1", "flavf1-188"),
             ("FLAV_F1R", "flavf1r-188"), (RAND_POOLED, "rand-188"))
# CS-5b (bs3, bs3_cs5b_mlp0_gaps.tsv): the 24 zero-filled MLP gaps that pass 3 sigma, (task, metric heading, metric,
# file, rows)
_MLP0_PASS = (       # CS-5c (PI 2026-11-24): only the label-granularity gaps; those against a random network are in Table 10
    ("top tagging", "AUC", "auc", "top_N1200000",
     (("L162_MASS", "FLAV_F0", "l162mass-flavf0"), ("FLAV_F0", "FLAV_F1", "flavf0-flavf1"))),
    ("top tagging", "$R_{30}$", "r30", "top_N1200000",
     (("R16_Q1", "FLAV_F1", "r16q1-flavf1"), ("L162_MASS", "FLAV_F0", "l162mass-flavf0"),
      ("FLAV_F0", "FLAV_F1", "flavf0-flavf1"), ("FLAV_F1", "FLAV_F1R", "flavf1-flavf1r"))))
TESTS = (
    # Figs. 2 and 3, Table 1 (M005-M007, M040-M043, M149-M153, M157, M158, M163, M164, M166, M299, M300)
    # review of 5c40d69 (A7, B3): the accuracy falls most between 162 and 43 classes (17 - 43 and the step), the random
    # partitions lie above 43 classes and the flavor-boundary partitions between 17 classes and the random partitions,
    # in accuracy as the text states them (bootstraps of 141f13ad, 769c8ed7)
    # PI 2026-11-23 (LEDGER DL-3b): "no resolved loss in the classification of the pretraining classes" needs every
    # metric the PDF prints for the mass networks against their label sets, accuracy too (_MASSES, the stored file's rows)
    (r"JetClass-II, accuracy, " + HEADING["lp"], tuple((a, b, ("l3", "lp", "jc2", "accuracy", d, False)) for a, b, d in _LEVELS)
     + tuple((a, b, ("l3", "lp", "jc2", "accuracy", d, False)) for a, b, d in _MASSES)
     + _jc2_lp("accuracy", (("R42_Q1", "L162", "r42q1-l162"), ("R16_Q1", "R42_Q1", "r16q1-r42q1")))
     + (("STEP_17_43", "STEP_43_162", ("file", _JC2_ALL, "step-r16q1-r42q1-l162", "accuracy")),)
     + _review2_rows("accuracy")
     + _jc2_lp("accuracy", tuple((a, "L188", d) for a, d in _PARTS_ALL)
               + ((RAND_POOLED, "R42_Q1", "rand-r42q1"), ("FLAV_F0", "R16_Q1", "flavf0-r16q1"),
                  ("FLAV_F1", "FLAV_F0", "flavf1-flavf0"), ("FLAV_F1R", "FLAV_F0", "flavf1r-flavf0"),
                  (RAND_POOLED, "FLAV_F1", "rand-flavf1"), (RAND_POOLED, "FLAV_F1R", "rand-flavf1r")))),
    (r"JetClass-II, AUC, " + HEADING["lp"], tuple((a, b, ("l3", "lp", "jc2", "macro_auc_ovr", d, False)) for a, b, d in _LEVELS)
     + _jc2_lp("macro_auc_ovr", (("R42_Q1", "L162", "r42q1-l162"), ("R16_Q1", "R42_Q1", "r16q1-r42q1")))
     + (("STEP_17_43", "STEP_43_162", ("file", _JC2_ALL, "step-r16q1-r42q1-l162", "macro")),)
     + _review2_rows("macro")
     + tuple((a, b, ("l3", "lp", "jc2", "macro_auc_ovr", d, False)) for a, b, d in _MASSES)
     + _jc2_lp("macro_auc_ovr", ((RAND_POOLED, "R16_Q1", "rand-r16q1"), (RAND_POOLED, "R42_Q1", "rand-r42q1"),
                                 ("FLAV_F0", "R16_Q1", "flavf0-r16q1"), ("FLAV_F1", "FLAV_F0", "flavf1-flavf0"),
                                 ("FLAV_F1R", "FLAV_F0", "flavf1r-flavf0"), (RAND_POOLED, "FLAV_F1", "rand-flavf1"),
                                 (RAND_POOLED, "FLAV_F1R", "rand-flavf1r")))),
    (r"JetClass-II, AUC over the resonant classes, " + HEADING["lp"], _jc2_lp("macro_auc_resonant", (("L162", "L188", "l162-188"),))),
    (r"JetClass-II, AUC over the QCD classes, " + HEADING["lp"], _jc2_lp("macro_auc_qcd", (("L162", "L188", "l162-188"),))),
    (r"JetClass-II, accuracy, " + HEADING["ft"], tuple((a, b, ("l3", "ft", "jc2", "accuracy", d, False)) for a, b, d in _LEVELS)),
    # M008, M045, M219 (the abstract's 17-class cost), M221, M226, M305
    (r"JetClass, accuracy, " + HEADING["ft"], tuple((a, b, ("l3", "ft", "jc1", "accuracy", d, False)) for a, b, d in _LEVELS + _MASSES)
     + tuple((a, "R16_Q1", RERUN) for a, _ in _PARTS_ALL)),
    # Figs. A10 and A8 below a million jets: the probes' label-set rows pass 3 sigma (PI 2026-11-23, LEDGER SP-13 (c));
    # no JetClass-II network is fine-tuned on 10^3 jets
    *((rf"JetClass-II, accuracy, {HEADING['lp']}, " + _power(f"{n:g}") + " training jets",
       tuple((a, "L188", ("file", f"l3_v5/probe_jc2bal_N{n}.v4.json", d, "accuracy"))
             for a, d in (("R42_Q1", "r42q1-188"), ("R16_Q1", "r16q1-188")))) for n in _SIZES),
    *((rf"JetClass-II, accuracy, {HEADING['ft']}, " + _power(f"{n:g}") + " training jets", (("R16_Q1", "L188", RERUN),))
      for n in _SIZES[1:]),
    (r"top tagging, $R_{30}$, " + HEADING["lp"], _bench_block("top", "lp")),
    (_FT_TOP, _bench_block("top", "ft", own=True)
     + (("L188", "SCRATCH", ("file", "l3/ft_top_N1200000.scratch.json", "188-scratch", "r30")),
        # results_bench: the kinematic-inputs network reaches the rejection of 188 classes fine-tuned with zeros (one
        # run: its sigma lacks that side's run spread, a lower bound, so the null holds; review of 5c40d69, B3)
        ("KIN", "L188", ("file", _FT_REVIEW, "kin-188", "r30")))),
    # results_bench: 188 classes against from scratch on every other metric tab:published prints (review of render
    # dfe5942, A1 and BS2; LEDGER CS-5: no null on R30 alone while another printed metric differs)
    *((rf"top tagging, {h}, " + HEADING["ft"], (("L188", "SCRATCH", ("file", "l3/ft_top_N1200000.scratch.json",
                                                                       "188-scratch", k)),))
      for k, h in (("accuracy", "accuracy"), ("auc", "AUC"), ("r50", "$R_{50}$"))),
    # results_bench: the linear probes of 188 classes and of the random network with zeros lie below every fine-tuned
    # network, one row each, the least |Delta / sigma| of the ten networks (review of 5c40d69, B3)
    (_FT_TOP + " against the linear probe with zeros",
     tuple(("FT_EVERY", z, ("least", tuple((a, z, ("file", _FT_REVIEW, f"{_REVIEW_KEY[a]}-{k}lpzero", "r30"))
                                           for a in SETS + FLAV + (RAND_POOLED,))))
           for z, k in (("L188_LP_ZERO", "188"), ("INIT_BN_LP_ZERO", "initbn")))),
    (r"quark/gluon tagging (Pythia), $R_{30}$, " + HEADING["lp"], _bench_block("qg", "lp")),
    (_FT_QG, _bench_block("qg", "ft")),
    (r"quark/gluon tagging (Herwig), $R_{30}$, " + HEADING["lp"], _bench_block("qg_herwig", "lp")),
    (r"quark/gluon tagging (Herwig), $R_{30}$, " + HEADING["ft"], _bench_block("qg_herwig", "ft")
     + (("L188_HERWIG", "SCRATCH_HERWIG", ("file", "l3/ft_qg_N1600000.herwig_over_pythia.json", "188-scratch", "r30_ratio")),)),
    (r"quark/gluon tagging, $R_{30}$, " + HEADING["ft"] + ", Herwig against Pythia jets",
     tuple((a, b, ("file", "l3/ft_qg_N1600000.herwig_pythia_unpaired.json", d, "r30"))
           for a, b, d in (("L188_HERWIG_JETS", "L188_PYTHIA_JETS", "herwig-pythia-188"),
                           ("SCRATCH_HERWIG_JETS", "SCRATCH_PYTHIA_JETS", "herwig-pythia-scratch")))),
    # tab:fill (M246, M253, M328, M329)
    (r"top tagging, $R_{30}$, " + HEADING["lp"] + ", missing inputs",
     tuple((a, b, ("file", "l3_v5/probe_top_N1200000.fill.json", d, "r30"))
           for a, b, d in (("L188_DRAWS", "L188_ZERO", "188draws-188zero"), ("L188", INIT_BN, "188-initbn")))),
    # PI 2026-11-24 (PW-2, CS-18b): the zeros - random network null (90%) and draws - kinematic null left with their sentences
    (r"quark/gluon tagging (Pythia), $R_{30}$, " + HEADING["lp"] + ", missing inputs",
     (("L188", INIT_BN, ("file", "l3_v5/probe_qg_N1600000.fill.json", "188-initbn", "r30")),)),
    # tab:published, tab:partrepro (M236-M238, M243, M258, M260, M240; results_bench: from scratch against ParT's)
    # review of 5c40d69 (A1): no sentence compares the fine-tuned networks with ParT fine-tuned on top tagging (the
    # b-quark boundary fails it on run SDs alone, and no paired term exists for 188 classes); the values stay side by
    # side in tab:published, and the row left with its sentence (PI 2026-11-23, LEDGER SP-13 (a)); review of render
    # dfe5942 (A1): our from scratch against ParT's is no null (an AUC gap of 7 run SDs, no per-jet scores for ParT's),
    # its sentence and row removed
    (_FT_TOP + ", published models", (("PART_OURS", "PART_PUB", ("part_published",)),)),     # A2 (bs3; SC-2)
    # DL-16c (PI 2026-11-24): Sophon's rescaling against the unchanged script, one run each, the counting terms only
    (_FT_TOP + ", ParT's training script", (("PART_RESCALE", "PART_OURS", ("part_rescale",)),)),
    (_FT_QG + ", published models", (("L188", "PART_FT_PUB", None),)),
    # Figs. 5 and A3: the class pairs, AUC (M187, M190, M194, M200, M334, the placement's sentences)
    *((rf"{_PAIR[k]}, AUC, " + HEADING["lp"], _pair_rows(k, rows) + _pair_rows(k, _FLAV188.get(k, ()), ".flav188")
       + (_EE_17 if k == "ee_vs_mm" else ()))
      for k, rows in (
        # PI 2026-11-24 (TX-7, PW-2): the no-flavor-boundary - 17 row (CS-9b) removed with its null, which resolved
        # only differences above 380%; Fig. 5 draws that curve's run spread instead
        ("bb_vs_cc", (("R42_Q1", "L188", "r42q1-188"), ("R16_Q1", "L188", "r16q1-188"), ("FLAV_F1", "R16_Q1", "flavf1-r16q1"))),
        ("bc_vs_bq_cs", _QCD_PAIR + (("FLAV_F1", "R16_Q1", "flavf1-r16q1"), ("FLAV_F1R", "R16_Q1", "flavf1r-r16q1"),
                                     ("FLAV_F0", "R16_Q1", "flavf0-r16q1"), ("FLAV_F0", "L188", "flavf0-188"))),
        ("bbqq_vs_ccqq", (("R42_Q1", "L188", "r42q1-188"), ("R16_Q1", "L188", "r16q1-188"),
                          ("FLAV_F1R", "FLAV_F1", "flavf1r-flavf1"))),
        *((k, _QCD_PAIR) for k in ("qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light")),
        # PI SP-10 (b): the 3-sigma-passing lepton-flavor result of the 43-class labels
        ("ee_vs_mm", (("R42_Q1", "L188", "r42q1-188"), ("R42_Q1", INIT_BN, "r42q1-initbn"))))),
    # tab:mlp (M157, M302): the multilayer-perceptron probes beside the linear-probe nulls
    # DL-3b (bs3): the MLP refit's bootstrap for the mass networks against their label sets
    (r"JetClass-II, AUC, " + HEADING["mlp"], tuple((a, b, ("file", "l3_bs3/mlp_jc2bal_N1000000.mass.json", k, "macro"))
                                                   for (a, b, _), k in zip(_MASSES, ("l162mass-l162", "r16q1mass-r16q1")))),
    # DL-9 (bs3): the MLP beside the linear-probe nulls N1 (JetClass-II accuracy) and N3 (the 17 merged groups)
    (r"JetClass-II, accuracy, " + HEADING["mlp"],
     (("L162", "L188", ("file", "l3_bs3/mlp_jc2bal_N1000000.json", "l162-188", "accuracy")),)),
    (r"JetClass-II, accuracy on the 17 merged groups, " + HEADING["mlp"],
     tuple((a, "L188_17", ("file", "l3_bs3/mlp_jc2bal_R16_Q1_N1000000.json", d, "accuracy"))
           for a, d in (("L162_17", "l162-188"), ("R42_Q1_17", "r42q1-188"), ("R16_Q1_17", "r16q1-188")))),
    # DL-9 (bs3), N2: the MLP with the linear probes' substitutions, every network against 188 classes, as Fig. 2
    *((rf"{h}, $R_{{30}}$, " + HEADING["mlp"] + ", missing inputs as for the linear probe",
       tuple((a, "L188", ("file", f"l3_bs3/mlp_{f}.json", d, "r30")) for a, d in _MLP_NETS))
      for h, f in (("top tagging", "top_N1200000"), ("quark/gluon tagging (Pythia)", "qg_N1600000"),
                   ("quark/gluon tagging (Herwig)", "qg_herwig_N1600000"))),
    # CS-5b (bs3): the gaps of Table 10's zero-filled columns that pass 3 sigma with the full sigma, each stated
    *((rf"{h}, {met_h}, " + HEADING["mlp"] + ", missing inputs set to zero",
       tuple((a, b, ("file", f"l3_bs3/mlp0_{f}.json", d, met)) for a, b, d in rows))
      for h, met_h, met, f, rows in _MLP0_PASS),
    (r"JetClass-II, AUC over the resonant classes, " + HEADING["mlp"], (("L162", "L188", None),)),
    # review of 5c40d69 (B4, LEDGER CS-18): the multilayer-perceptron probe beside the linear-probe nulls where tab:mlp
    # holds it like for like, the partition with no flavor boundary against 17 classes on X->bc and the 188-class
    # network against the random network on top tagging with the missing inputs at zero (bootstrap pending)
    (rf"{_PAIR['bc_vs_bq_cs']}, AUC, " + HEADING["mlp"], (("L162", "L188", None), ("R16_Q1", "L188", None),
                                                          ("FLAV_F0", "R16_Q1", None))),
    # tab:coarse: the label sets scored on the 17 merged groups
    (r"JetClass-II, accuracy on the 17 merged groups, " + HEADING["lp"],
     tuple((a, "L188_17", ("file", "l3_v5/probe_jc2bal_R16_Q1_N1000000.json", d, "accuracy"))
           for a, d in (("L162_17", "l162-188"), ("R42_Q1_17", "r42q1-188"), ("R16_Q1_17", "r16q1-188")))),
    # tab:mass, Fig. A4 (M285, M286, M293, M295, M297)
    (r"jet mass, resolution, " + HEADING["mlp"], tuple((a, b, ("mass", f"{LABEL[a]} - {LABEL[b]}".replace(r"\Kres{}", "162")
                                                                .replace(r"\Kcoarse{}", "17"), "mlp")) for a, b, _ in _MASSES)
     + (("R16_Q1", "L188", ("mass", "17 classes - 188 classes", "mlp")),)
     # results_mass: every pretrained network of tab:mass carries more of the mass than the random network (review of
     # 5c40d69, B3); the 17-class partitions, which tab:mass does not print, are not stated
     + tuple((a, INIT, ("mass", f"{_MASS_NAME(a)} - random network", "mlp")) for a in SETS)),
    (r"jet mass, resolution, " + HEADING["head"], (
        ("L162_MASS_HEAD", "SDMASS", ("mass_group", "162 classes + mass", "regressed")),
        ("R16_Q1_MASS_HEAD", "SDMASS", ("mass_group", "17 classes + mass", "regressed")),
        ("R16_Q1_MASS_HEAD", "L162_MASS_HEAD", ("mass", "17 classes + mass - 162 classes + mass", "regressed")),
        ("L162_MASS_HEAD", "L162_MLP", ("mass", "162 classes + mass, regressed - 162 classes, mlp", "cross")))),
    # the bins in words, not by figure: the results book sets this table with no figure reference (make_results_book)
    (r"jet mass, resolution, " + HEADING["head"] + r", every bin of jet $p_{\mathrm{T}}$ and generator-level mass",
     tuple((f"{a}_BINS", "SDMASS", ("least", tuple((f"{a}_HEAD", "SDMASS", ("mass_group", g, "regressed", b))
                                                   for b in F9B_BINS)))
           for a, g in (("L162_MASS", "162 classes + mass"), ("R16_Q1_MASS", "17 classes + mass")))),
    # Fig. 6 and Table 1's X->bb / X->bs columns (M309-M311, the placement's XBX); the flavor-boundary partitions against
    # 188 classes wait for their bootstrap and stay if they pass (SP-13 (c)); against 17 classes they fail for certain
    # review of render dfe5942 (C): one name, the transfer head
    *((rf"{h}, signal efficiency (\%), " + XBX_HEAD,
       (("R16_Q1", "L188", _XBX_SRC(sig)),) + tuple((a, "L188", _XBX_SRC(sig)) for a in FLAV)
       + ((("L188", "SCRATCH", _XBX_SRC(sig)),) if sig == "xbs" else ())) for sig, h in SUMMARY_XBX),
)
# the sculpting nulls (M284, M306, M336; F10b): every difference of the bootstrap file at each working point, in its order,
# but those no sentence states (results_mass names no mass network and no random partition against 188 classes; PI
# 2026-11-23, LEDGER SP-13)
TESTS_SCULPT = r"mass sculpting, Jensen--Shannon distance, " + HEADING["direct"]
SCULPT_FLOOR_HEAD = TESTS_SCULPT + r", $\epsB = 10\%$, against the floor"
SCULPT_ARM = {"rand2p": RAND_POOLED}
SCULPT_UNSTATED = (("L162_MASS", "L188"), ("R16_Q1_MASS", "L188"), (RAND_POOLED, "L188"))
# PI 2026-11-23 (LEDGER SP-12, step 3): every ratio word the text uses, <name>Pr or <name>PrMax, and the tab:tests row
# (heading, first, second) that tests its comparison, with the null phrase its sentence reads when the row does not
# pass 3 sigma (ratio_words); the null takes the place of the ratio in its sentence's grammar
_NULL = "no difference beyond three standard deviations"       # in parentheses
_NO_MORE, _NO_FEWER = "no more errors beyond three standard deviations", "no fewer errors beyond three standard deviations"
# LEDGER TX-1 (PI 2026-11-23): a null is never worded as an equality ("the resolution of"): "... reaches a resolution
# within three standard deviations of that of ...", "... gives a linear probe errors within three standard deviations
# of those of ..."
_same = lambda noun: ("errors within three standard deviations of those" if noun == "errors"      # noqa: E731
                      else f"a {noun} within three standard deviations of that")
_JC2_LP, _JC2_MLP = "JetClass-II, AUC, " + HEADING["lp"], "JetClass-II, AUC, " + HEADING["mlp"]
_BB, _BC, _BBQQ = (rf"{_PAIR[k]}, AUC, " + HEADING["lp"] for k in ("bb_vs_cc", "bc_vs_bq_cs", "bbqq_vs_ccqq"))
_TOP_FILL, _QG_FILL = (rf"{t}, $R_{{30}}$, " + HEADING["lp"] + ", missing inputs"
                       for t in ("top tagging", "quark/gluon tagging (Pythia)"))
_MASS_MLP, _MASS_HEAD = "jet mass, resolution, " + HEADING["mlp"], "jet mass, resolution, " + HEADING["head"]
RATIO_PAIR_HEADS = {_BB: 1, _BC: 2, _BBQQ: 3}       # the ratio words whose sides tab:pairauc prints: their column
PR_ROWS = {
    # results_jc
    "MlstCmpJcTwoMidOverFullPr": (_JC2_LP, "R42_Q1", "L188", _NO_MORE),
    "MlstCmpJcTwoCoarseOverFullPr": (_JC2_LP, "R16_Q1", "L188", _NO_MORE),
    # MlstCmpJcTwoResResOverFull: its null now prints its PW-1 share (fourth review, B1/C5)
    "MlstMlpCmpJcTwoResResOverFullPr": ("JetClass-II, AUC over the resonant classes, " + HEADING["mlp"], "L162", "L188",
                                        _NULL),
    "MlstCmpJcTwoQcdResOverFullPr": ("JetClass-II, AUC over the QCD classes, " + HEADING["lp"], "L162", "L188", _NO_MORE),
    "MlstCmpJcTwoRandOverCoarsePr": (_JC2_LP, RAND_POOLED, "R16_Q1", _NO_FEWER),
    "MlstCmpJcTwoRandOverMidPr": (_JC2_LP, RAND_POOLED, "R42_Q1", _NO_FEWER),
    "MlstCmpBbCoarseOverFullPr": (_BB, "R16_Q1", "L188", _same("errors")),
    "MlstCmpBbMidOverFullPr": (_BB, "R42_Q1", "L188", "a difference within three standard deviations"),
    # MlstCmpBcResOverFull: its null now prints its PW-1 share (fourth review, B1/C5)
    "MlstMlpCmpBcResOverFullPr": (rf"{_PAIR['bc_vs_bq_cs']}, AUC, " + HEADING["mlp"], "L162", "L188", _NULL),
    "MlstCmpBcMidOverFullPr": (_BC, "R42_Q1", "L188", "a difference within three standard deviations"),
    "MlstCmpBcCoarseOverFullPr": (_BC, "R16_Q1", "L188", _NO_MORE),
    "MlstCmpBbFoneOverCoarsePr": (_BB, "FLAV_F1", "R16_Q1", _same("errors")),
    "MlstCmpBcFoneOverCoarsePr": (_BC, "FLAV_F1", "R16_Q1", _NO_FEWER),
    "MlstCmpBcFonerOverCoarsePr": (_BC, "FLAV_F1R", "R16_Q1", _NULL),
    "MlstCmpBbqqFonerOverFonePr": (_BBQQ, "FLAV_F1R", "FLAV_F1", _NO_MORE),
    # review of 5c40d69 (B4, LEDGER CS-18): the multilayer-perceptron probe beside two linear-probe nulls
    "MlstMlpCmpBcFzeroOverCoarsePr": (rf"{_PAIR['bc_vs_bq_cs']}, AUC, " + HEADING["mlp"], "FLAV_F0", "R16_Q1", _NULL),
    # MlstMlpTopRthirtyFullOverInitBn: its sentence left with the 90% null (PI 2026-11-24, PW-2)
    # results_bench
    "MlstLpTopRthirtyDrawOverZeroFullPr": (_TOP_FILL, "L188_DRAWS", "L188_ZERO", _same("rejection")),
    "MlstFtQgRthirtyFullOverPartPr": (r"quark/gluon tagging (Pythia), $R_{30}$, " + HEADING["ft"] + ", published models",
                                      "L188", "PART_FT_PUB", _NULL),
    # MlstFtQgRthirtyCoarseOverFull: its null now prints its PW-1 share (fourth review, B1/C5)
    "MlstFtQgRthirtyHerwigOverPythiaFullPr": (r"quark/gluon tagging, $R_{30}$, " + HEADING["ft"] + ", Herwig against Pythia jets",
                                              "L188_HERWIG_JETS", "L188_PYTHIA_JETS", _same("rejection")),
    "MlstFtQgRthirtyHerwigOverPythiaScratchPr": (r"quark/gluon tagging, $R_{30}$, " + HEADING["ft"]
                                                 + ", Herwig against Pythia jets", "SCRATCH_HERWIG_JETS",
                                                 "SCRATCH_PYTHIA_JETS", "a rejection on Herwig jets within three standard deviations of that on Pythia jets"),
    "MlstLpQgRthirtyFullOverInitBnPr": (_QG_FILL, "L188", INIT_BN, _NULL),
    # discussion
    # MlstLpTopRthirtyFullOverInitBnZero: its null now prints its PW-1 share (fourth review, B1/C5)
    "MlstLpTopRthirtyFullOverInitBnPr": (_TOP_FILL, "L188", INIT_BN, _same("rejection")),
    # results_mass
    "MlstMassHeadResMassOverUncorrPr": (_MASS_HEAD, "L162_MASS_HEAD", "SDMASS", _same("resolution")),
    "MlstMassHeadResMassOverMlpResPr": (_MASS_HEAD, "L162_MASS_HEAD", "L162_MLP", _same("resolution")),
    "MlstCmpMlpResMassOverResPr": (_MASS_MLP, "L162_MASS", "L162", _same("resolution")),
    "MlstCmpMlpCoarseMassOverCoarsePr": (_MASS_MLP, "R16_Q1_MASS", "R16_Q1", _same("resolution")),
    "MlstCmpMlpCoarseOverFullPr": (_MASS_MLP, "R16_Q1", "L188", _NULL),
    # MlstCmpJcTwoResMassOverResPrMax: its null now prints its PW-1 share (fourth review, B1/C5)
    # MlstMlpCmpJcTwoResMassOverRes: its null now prints its PW-1 share (bs3, DL-3b rows wired)
    # MlstMlpCmpJcTwoCoarseMassOverCoarse: its null now prints its PW-1 share (bs3, DL-3b rows wired)
}


def _tests_l3(data: pathlib.Path, mode: str, key: str, met: str, chosen: dict, every: bool) -> tuple:
    """(the L3 document of Table 1's cell, its metric key), the .all copy with every."""
    name = SUMMARY_L3.get((mode, key))
    if name is None or SUMMARY_L3_METRIC.get(met) is None:
        return {}, ""
    if "{s}" in name:
        if chosen.get(key) is None:
            return {}, ""
        name = name.format(s=SUBST_KEY[key][chosen[key]].split(".")[-1])
    return _l3_doc(data, name.replace(".json", ".all.json") if every else name, mode == "lp"), SUMMARY_L3_METRIC[met]


def _tests_side(e: dict, first: str) -> tuple:
    """(delta, run SD of the first side, of the second, bootstrap SD, sigma) of a stored L3 difference, whatever the
    file names them; sigma must be the quadrature sum of the three (PI SC-1 (1))."""
    sd_a, sd_b = e.get("sd_a", e.get("sd_config")), e.get("sd_b", e.get("sd_reference"))
    out = (e["delta"], sd_a, sd_b, e["boot_sd"], e["sigma"])
    if None in out or not math.isclose(e["sigma"], math.sqrt(sd_a ** 2 + sd_b ** 2 + e["boot_sd"] ** 2), rel_tol=1e-9):
        raise SystemExit(f"FATAL: tab:tests, {first}: sigma is not the quadrature sum of the two run SDs and the "
                         "paired bootstrap SD (PI SC-1 (1))")
    return out


# PI 2026-11-23 (LEDGER SP-13, step 2): the mark of a row whose two sides are scored on independent test samples, each
# resampled apart, the two bootstrap variances adding (PI SC-1a)
TESTS_GLUE_ROWS = 9         # a tab:tests block of at most this many lines (heading included) never breaks
INDEPENDENT = r"$^{\diamond}$"     # no other table carries it (SINGLE is the section sign)
# a row with a side of one run, whose sigma lacks that side's run spread (a lower bound; review of 5c40d69, B3)
SINGLE_SIDE = r"$^{\circ}$"
# a row over several networks or bins, the one of least |Delta / sigma| (review of 5c40d69, B3)
LEAST = r"$^{\ast}$"
TESTS_HEAD = r" & $\Delta$ & $\sigma$ & $\Delta/\sigma$ \\"


def table_tests(data: pathlib.Path, runs: dict, fits, fill, chosen, per, sculpt_doc=None, mass_doc=None,
                rows_out: dict | None = None, co: dict | None = None, sides_out: dict | None = None,
                mlp: dict | None = None, bs3: dict | None = None, sculpt: dict | None = None) -> tuple:
    r"""tab:tests (TESTS): (the table, the bootstrap replica counts read); rows_out, when given, takes each TESTS row's
    result, {(heading, first, second): None | "stale" | (delta, run SDs, bootstrap SD, sigma)}, for the ratio words
    (ratio_words). A row prints its stored difference Delta (an AUC row minus the stored 1 - AUC difference), sigma and
    Delta / sigma (PI 2026-11-23, LEDGER SP-13: each side's mean and SD are in the value tables), Delta and sigma at the
    decimal place of sigma (make_tables.pdg), INDEPENDENT on a row whose sides are scored on independent test samples. A
    row is \pending{bootstrap} without a stored bootstrap, and \pending{rerun bootstrap} where its file's runs are not the
    paper's or their values are not the ones Table 1 prints (a fine-tuning run since rerun: SP-11). A "file" row's runs
    and values (each within a quarter of its bootstrap SD of the value the paper prints) are checked, and the build
    stops on one that is not the paper's. co: the probes on the 17 merged groups (tab:coarse). sides_out, when given,
    takes each row's run values of its two sides as the paper prints them, {(heading, first, second): (first's, second's)},
    from which ratio_words forms a quoted ratio (review of render dfe5942, A2, A3: LEDGER CS-6, CS-8)."""
    runs = {**_pooled(runs), "SCRATCH": [f"{SCRATCH}-s{k}" for k in (1, 2, 3)], "KIN": [KIN_MODEL]}
    ds_of = {"jc2": "jc2_188", "jc1": "jc1", "top": "top", "qg": "qg", "qg_herwig": "qg_herwig"}
    n_of = {k: n for k, n, *_ in SUMMARY_PANELS}
    name = lambda x: TESTS_NAMES.get(x) or LABEL[x]        # noqa: E731
    arm = lambda x: TESTS_ARM.get(x, x)                    # noqa: E731
    # printed as tab:sculptboot printed them (no token, no recorded formatter): the data release holds the files
    num = lambda v, place: (lambda t: (t[1:] if t.startswith("-") and not any(c in "123456789" for c in t) else t)  # noqa: E731
                            .replace("-", "$-$"))(f"{round(v, place):,.{max(place, 0)}f}".replace(",", "{,}"))
    replicas, independent, single = set(), set(), set()
    analytic = set()    # SC-2 (#1): the rows whose test-sample term is analytic (xbx_row)
    curhead = [None]    # the heading of the row being read
    cur = {}            # the sides of the row being read (sides_out)

    def l3_row(first, second, src):
        _, mode, key, met, diff, every = src
        doc, mkey = _tests_l3(data, mode, key, met, chosen or {}, every)
        e = (doc.get("diffs") or {}).get(diff, {}).get(mkey)
        if not e:
            return None
        replicas.add(doc.get("b"))
        if sorted(doc["diffs"][diff]["a"]) != sorted(runs[first]) or sorted(doc["diffs"][diff]["b"]) != sorted(runs[second]):
            return "stale"
        get = (summary_probe(fits, fill, chosen, ds_of[key], met) if mode == "lp"
               else summary_ft(per, key, n_of[key], met))
        auc = e.get("quantity") == "one_minus_auc"
        for m in runs[first] + runs[second]:
            rec = doc["models"].get(m, {}).get(mkey, {})
            fv = (1 - rec["one_minus_auc"]) if auc and "one_minus_auc" in rec else rec.get("value")
            bv = get(m) if get else None
            if fv is None or bv is None or abs(float(bv) - fv) > 0.25 * (rec.get("sd") or 0):
                return "stale"
        cur["sides"] = tuple([float(get(m)) for m in runs[x]] for x in (first, second))
        d, a, b, bs, sig = _tests_side(e, first)
        return (-d if auc else d), a, b, bs, sig

    def printed(path: str, side: str, mkey: str = ""):
        """f(run) -> the value the paper prints for one run of a side of a "file" row: the class-pair AUC of the
        figures and ratio words, tab:coarse's accuracy, Table 1's AUC or accuracy (the step, mkey), tab:fill's R30
        under the side's substitution (the kinematic-inputs network's, which has nothing missing), or Table 1's
        fine-tuned R30 on the full top or quark/gluon set (Herwig over Pythia, run by run)."""
        task = "top" if "_top_" in path else "qg"
        # the statistics agent's refit (bs3; LEDGER DL-9, CS-5b): the MLP values its rows were bootstrapped from, as
        # Table 10 (zeros, JetClass-II; overlay_bs3_mlp), Table 14 (the 17 groups) and the refit with the linear probes'
        # substitution (top and quark/gluon) read them
        bs3_task = "qg_herwig" if "_qg_herwig_" in path else task
        if path.startswith("l3_bs3/mlp0_"):
            return lambda m: _mlp_val(mlp or {}, m, bs3_task, mkey if mkey in ("auc", "accuracy") else "r30")
        if path.startswith("l3_bs3/mlp_jc2bal_R16_Q1_"):
            return lambda m: _bs3_val(bs3 or {}, "coarse", m, "jc2bal_coarse", "accuracy")
        if path.startswith("l3_bs3/mlp_jc2bal_"):
            return lambda m: _bs3_val(bs3 or {}, "jc2bal", m, "jc2bal", "macro_auc_ovr" if mkey == "macro" else mkey)
        if path.startswith("l3_bs3/mlp_"):
            return lambda m: _bs3_val(bs3 or {}, "bench", m, bs3_task, mkey if mkey in ("auc", "accuracy") else "r30")
        if "_pairs_" in path:       # the pair of the file, also of a later copy (".flav188")
            return lambda m: _pair(fits, m, re.search(r"_pairs_([a-z_]+?)(?:\.\w+)?\.json$", path).group(1))
        if side == "KIN" and ".kin." in path:
            return lambda m: _rej(_probe_rec(fits, m, "top"), "r30")
        if side.endswith("_LP_ZERO"):
            return lambda m: _fill_r30(fill, m, task, "zero")
        if "_R16_Q1_" in path:
            return lambda m: ((co or {}).get(m, {}).get("cells", {}).get(PRIMARY, {}).get("jc2bal_coarse", {})
                              .get("R16_Q1", {}).get("1000000") or {}).get("accuracy")
        if path in (_JC2_ALL, _JC2_REVIEW2):
            return summary_probe(fits, fill, chosen, "jc2_188", "accuracy" if mkey == "accuracy" else "macro_auc_ovr")
        size = re.search(r"probe_jc2bal_N(\d+)\.v4\.json$", path)
        if size:                # the probes of Fig. A10 and tab:probesn below a million jets
            return lambda m: (_probe_rec(fits, m, "jc2_188", size.group(1)) or {}).get("accuracy")
        if ".fill." in path or ".kin." in path:
            s = "draws" if side == "L188_DRAWS" else "zero" if side.endswith("ZERO") else (chosen or {}).get(task)
            return (lambda m: _fill_r30(fill, m, task, s)) if s else None
        keys = (("qg_herwig",) if side.endswith("HERWIG_JETS") else ("qg",) if side.endswith("PYTHIA_JETS")
                else ("qg_herwig", "qg") if side.endswith("HERWIG") else (task,))
        met = mkey if mkey in ("accuracy", "auc", "r50") else "r30"     # tab:published's metrics on the full sets

        def one(key, m):
            if m.startswith(SCRATCH):
                v = summary_scratch(per, key, FULL_N[key], met)
                k = int(m.rsplit("-s", 1)[1]) - 1
                return v[k] if k < len(v) else None
            g = summary_ft(per, key, FULL_N[key], met)
            return g(m) if g else None
        return lambda m: (lambda v: None if None in v else float(v[0]) / float(v[1]) if len(v) == 2 else v[0])(
            [one(k, m) for k in keys])

    def file_row(head, first, second, src):
        _, path, diff, mkey = src
        f = data / "bootstrap" / path
        doc = json.loads(f.read_text()) if f.exists() else {}
        e = (doc.get("diffs") or {}).get(diff, {}).get(mkey)
        if not e:
            return None
        replicas.add(doc.get("b"))
        stop = lambda why: SystemExit(f"FATAL: tab:tests, {first} - {second} ({head}): bootstrap/{path} {diff} {why}")  # noqa: E731
        d = doc["diffs"][diff]
        step = bool(d.get("step"))
        sides = STEP_RUNS[diff] if step else {"a": arm(first), "b": arm(second)}
        auc = e.get("quantity") == "one_minus_auc"
        vals, shown = {}, {}
        for s, a in sides.items():
            base = [re.sub(r"(@second|\.zero|\.lpzero)$", "", m) for m in d[s]]
            if sorted(base) != sorted(runs[a]):
                raise stop(f"side {s}: runs {sorted(d[s])}, not the paper's {sorted(runs[a])}")
            get = printed(path, a if step else first if s == "a" else second, mkey)
            vals[s], shown[s] = [], []
            for m, b in zip(d[s], base):
                rec = doc["models"].get(m, {}).get(mkey) or doc["models"].get(b, {}).get(f"{mkey}_second", {})
                fv = (1 - rec["one_minus_auc"]) if auc else rec.get("value")
                pv = get(b) if get else None
                if pv is None:          # the paper prints no value there (its inputs absent): pending, as l3_row
                    return "stale"
                if fv is None or abs(float(pv) - fv) > 0.25 * (rec.get("sd") or 0):
                    raise stop(f"{m}: {fv} against the printed {pv}, more than a quarter of its bootstrap SD apart")
                vals[s].append(fv)
                shown[s].append(float(pv))
        if step and len(sides) == 3:    # (a - b) - (b - c): b's mean twice, its run variance four times (sd_a, sd_b)
            sdr = {s: statistics.stdev(v) for s, v in vals.items()}
            mean = {s: statistics.fmean(v) for s, v in vals.items()}
            want = ((mean["a"] - mean["b"]) - (mean["b"] - mean["c"]),
                    math.sqrt(sdr["a"] ** 2 + 2 * sdr["b"] ** 2), math.sqrt(2 * sdr["b"] ** 2 + sdr["c"] ** 2))
            if not all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(want, (-e["delta"] if auc else e["delta"],
                                                                                e["sd_a"], e["sd_b"]))):
                raise stop("is not (a - b) - (b - c) with sd_a^2 = SDa^2 + 2 SDb^2, sd_b^2 = 2 SDb^2 + SDc^2")
        elif step:                      # (a - b) - (c - d), no side shared
            sdr = {s: statistics.stdev(v) for s, v in vals.items()}
            mean = {s: statistics.fmean(v) for s, v in vals.items()}
            want = ((mean["a"] - mean["b"]) - (mean["c"] - mean["d"]),
                    math.hypot(sdr["a"], sdr["b"]), math.hypot(sdr["c"], sdr["d"]))
            if not all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(want, (-e["delta"] if auc else e["delta"],
                                                                                e["sd_a"], e["sd_b"]))):
                raise stop("is not (a - b) - (c - d) with sd_a^2 = SDa^2 + SDb^2, sd_b^2 = SDc^2 + SDd^2")
        else:
            cur["sides"] = (shown["a"], shown["b"])
        # a side of one run has no run spread: sigma without it is a lower bound, enough for a null (the kinematic-
        # inputs network; review of 5c40d69, B3), marked; never a difference
        ones = [s for s in ("a", "b") if e.get(f"sd_{s}") is None and e.get(f"n_{s}") == 1]
        if ones:
            e = {**e, **{f"sd_{s}": 0.0 for s in ones}}
            e["sigma"] = math.sqrt(e["sd_a"] ** 2 + e["sd_b"] ** 2 + e["boot_sd"] ** 2)
            if abs(e["delta"]) > 3 * e["sigma"]:
                raise stop("has a single-run side and passes 3 sigma on a lower-bound sigma: no difference is claimed")
            single.add((head, first, second))
        if e.get("resampling") == "unpaired":
            if not math.isclose(e["boot_sd"], math.hypot(e["boot_sd_a"], e["boot_sd_b"]), rel_tol=1e-9):
                raise stop("(two independent samples): the bootstrap SD is not that of the two samples in quadrature "
                           "(PI SC-1a)")
            independent.add((head, first, second))
        out = _tests_side(e, first)
        return ((-out[0],) + out[1:]) if auc else out

    def sculpt_rows():
        out = []
        if not sculpt_doc:
            return out
        replicas.add(sculpt_doc.get("replicas"))
        arm = {m.removeprefix("mtx-").rsplit("-s", 1)[0]: a for a, ms in runs.items() if a != RAND_POOLED for m in ms}
        arm.update(SCULPT_ARM)
        order = ALL_ROWS + (RAND_POOLED,)
        diffs = sorted(sculpt_doc["differences"].values(), key=lambda d: (order.index(arm[d["configuration"]]),
                                                                           order.index(arm[d["reference"]])))
        diffs = [d for d in diffs if (arm[d["configuration"]], arm[d["reference"]]) not in SCULPT_UNSTATED]
        for wp, head in SCULPT_BOOT_WPS:
            rows = []
            for d in diffs:
                a, b = arm[d["configuration"]], arm[d["reference"]]
                x = d["l3"][wp]["distance"]
                ok = sorted(x["runs_config"]) == sorted(runs[a]) and sorted(x["runs_reference"]) == sorted(runs[b])
                rows.append((a, b, _tests_side(x, a) if ok else "stale"))
            out.append((f"{TESTS_SCULPT}, {head}", rows))
        return out

    def mass_row(first, second, src):
        kind, dname, probe, *where = src
        if kind == "mass_group":       # against the soft-drop mass of the same jets, which has no run spread
            e = ((mass_doc or {}).get("groups", {}).get(dname, {}).get(probe, {}).get(where[0] if where else "all|inclusive"))
            if not e:
                return None
            replicas.add(mass_doc.get("boot"))
            if len(e["runs"]) != len(runs[arm(first)]) or not math.isclose(
                    e["delta"], float(np.mean(e["runs"])) - e["softdrop"], rel_tol=1e-9, abs_tol=1e-12):
                return "stale"
            cur["sides"] = ([float(x) for x in e["runs"]], [float(e["softdrop"])])
            return _tests_side({**e, "sd_b": 0.0}, first)
        e = ((mass_doc or {}).get("diffs", {}).get(dname, {}).get(probe, {}).get("all|inclusive", {}).get("resolution"))
        if not e:
            return None
        replicas.add(mass_doc.get("boot"))
        if (len(e["a"]), len(e["b"])) != (len(runs[arm(first)]), len(runs[arm(second)])) or not math.isclose(
                e["delta"], float(np.mean(e["a"]) - np.mean(e["b"])), rel_tol=1e-9, abs_tol=1e-12):
            return "stale"
        cur["sides"] = ([float(x) for x in e["a"]], [float(x) for x in e["b"]])
        return _tests_side(e, first)

    def xbx_side(signal, arm_):
        """(per-network signal efficiencies in percent, their test SDs) of a side of a transfer-head row: each network
        the mean over its head trainings, its test term the root of their mean binomial variance (signal jets, and
        the QCD jets through the cut, the slope from the stored ROC at 1% QCD efficiency; tests/test_xbx_claims_l3.py);
        from scratch one network a seed."""
        def one(rec):
            c = rec["curve"]
            eb, s = np.array(c["eps_b"]), np.array(c["eps_s"])
            i = int(np.argmin(abs(eb - XBX_EPS_B)))
            slope = (s[i + 1] - s[i - 1]) / (eb[i + 1] - eb[i - 1])
            e = rec["eps_s_at_star"]
            return e, e * (1 - e) / rec["n_s"] + slope ** 2 * XBX_EPS_B * (1 - XBX_EPS_B) / rec["n_b"]
        out = []
        for m in (runs[arm_] if arm_ != "SCRATCH" else [f"s{k}" for k in range(1, 4)]):
            f = (data / "xbx_scratch" / signal / f"{m}.json") if arm_ == "SCRATCH" else (data / "xbx" / signal / f"{m}.json")
            if not f.exists():
                return None
            doc = json.loads(f.read_text())
            heads = [doc] if arm_ == "SCRATCH" else doc["signals"][signal]["head"]
            ev = [one(h) for h in heads]
            out.append((100 * float(np.mean([e for e, _ in ev])), 100 * math.sqrt(float(np.mean([v for _, v in ev])))))
        return out

    def xbx_row(first, second, signal):
        """SC-2 (#1; bs3, 2026-11-23): a transfer-head row with the analytic test-sample term in place of the paired
        bootstrap, as the heads' per-jet scores were not kept; the two sides' terms added (marked ANALYTIC_MARK)."""
        a, b = xbx_side(signal, first), xbx_side(signal, second)
        if a is None or b is None:
            return None
        va, vb = [x for x, _ in a], [x for x, _ in b]
        sa = statistics.stdev(va) if len(va) > 1 else 0.0
        sb = statistics.stdev(vb) if len(vb) > 1 else 0.0
        boot = math.hypot(float(np.mean([s for _, s in a])), float(np.mean([s for _, s in b])))
        cur["sides"] = (va, vb)
        analytic.add((curhead[0], first, second))
        return _tests_side({"delta": statistics.fmean(va) - statistics.fmean(vb), "sd_a": sa, "sd_b": sb,
                            "boot_sd": boot, "sigma": math.sqrt(sa ** 2 + sb ** 2 + boot ** 2)}, first)

    def part_row(first, second):
        """A2 (bs3, 2026-11-23; SC-2: a substitute term may confirm a null): ParT's script on our cluster, one run,
        against ParT's published from-scratch R30, the published SD over its trainings and our run's counting term
        (R30 over the root of its passing background jets) added; our side has no run spread (a lower bound)."""
        ours = _load(data / "partrepro").get("part")
        lit = json.loads(LIT_BENCH.read_text()) if LIT_BENCH.exists() else {}
        pub = next((r for r in lit.get("top", []) if r.get("method") == "ParT" and r.get("kind") == "scratch"), None)
        if not ours or not pub or not pub.get("r30"):
            return None
        mean, sd = (float(x.replace(",", "")) for x in pub["r30"].split("±"))
        count = ours["r30"] / math.sqrt(ours["r30_n_bkg_pass"])
        e = {"delta": ours["r30"] - mean, "sd_a": 0.0, "sd_b": sd, "boot_sd": count,
             "sigma": math.sqrt(sd ** 2 + count ** 2)}
        if abs(e["delta"]) > 3 * e["sigma"]:
            raise SystemExit("FATAL: tab:tests, ParT's script - published: passes 3 sigma on a lower-bound sigma")
        single.add((curhead[0], first, second))
        cur["sides"] = ([ours["r30"]], [mean])
        return _tests_side(e, first)

    def rescale_row(first, second):
        """DL-16c: ParT's script with Sophon's rescaling against it unchanged, one run each; sigma the two runs'
        counting terms, R30 over the root of its passing background jets (a lower bound: no run spread)."""
        pr = _load(data / "partrepro")
        a, b = pr.get("rescale"), pr.get("part")
        if not a or not b:
            return None
        ta, tb = (r["r30"] / math.sqrt(r["r30_n_bkg_pass"]) for r in (a, b))
        e = {"delta": a["r30"] - b["r30"], "sd_a": 0.0, "sd_b": 0.0, "boot_sd": math.hypot(ta, tb),
             "sigma": math.hypot(ta, tb)}
        if abs(e["delta"]) > 3 * e["sigma"]:
            raise SystemExit("FATAL: tab:tests, the rescaling row passes 3 sigma on a lower-bound sigma")
        single.add((curhead[0], first, second))
        cur["sides"] = ([a["r30"]], [b["r30"]])
        return _tests_side(e, first)

    def row(head, first, second, src):
        curhead[0] = head
        if src is not None and src[0] == "xbx_analytic":
            return xbx_row(first, second, src[1])
        if src is not None and src[0] == "part_published":
            return part_row(first, second)
        if src is not None and src[0] == "part_rescale":
            return rescale_row(first, second)
        return (None if src is None else "stale" if src == RERUN else l3_row(first, second, src) if src[0] == "l3"
                else file_row(head, first, second, src) if src[0] == "file" else least(head, src[1]) if src[0] == "least"
                else mass_row(first, second, src))

    def least(head, subs):
        """A row over several rows (every fine-tuned network, every bin): the one of least |Delta / sigma|, pending if
        any is (review of 5c40d69, B3: one row a statement, not one a network or bin)."""
        got = [(row(head, a, b, s), (head, a, b)) for a, b, s in subs]
        cur.pop("sides", None)          # a row over several has no two sides
        for bad in (None, "stale"):
            if any(v is bad for v, _ in got):
                return bad
        v, k = min(got, key=lambda g: abs(g[0][0]) / g[0][4])
        return v + ((len(subs), k in single),)

    blocks = []
    for head, rows in TESTS:
        got = []
        for first, second, src in rows:
            cur.clear()
            v = row(head, first, second, src)
            got.append((first, second, v, (head, first, second) in independent))
            if rows_out is not None:
                rows_out[(head, first, second)] = v[:5] if isinstance(v, tuple) else v
            if sides_out is not None and isinstance(v, tuple) and "sides" in cur:
                sides_out[(head, first, second)] = cur["sides"]
        blocks.append((head, got))
    blocks += [(h, [(a, b, v, False) for a, b, v in rs]) for h, rs in sculpt_rows()]
    # CS-2c (PI 2026-11-24): every head against the floor of a random selection of as many QCD jets at 10% QCD
    # efficiency (sculpt_above_floor: sigma^2 = run SD^2 + the floor's SD^2 over random selections)
    if sculpt:
        floor_rows = []
        for a in SETS + FLAV + (RAND_POOLED,):
            ms = [m for m in _pooled(runs)[a] if m in sculpt]
            if not ms:
                continue
            f = sculpt_above_floor(sculpt, ms)["0.1"]
            e = {"delta": f["distance"] - f["floor"], "sd_a": f["sd_runs"], "sd_b": 0.0, "boot_sd": f["sd_floor"],
                 "sigma": math.hypot(f["sd_runs"], f["sd_floor"])}
            v = _tests_side(e, a)
            floor_rows.append((a, "FLOOR", v, False))
            if rows_out is not None:
                rows_out[(SCULPT_FLOOR_HEAD, a, "FLOOR")] = v
        if floor_rows:
            blocks.append((SCULPT_FLOOR_HEAD, floor_rows))
    # PI SP-14 (the 25-page row rule of SP-13 re-applied to every row added since 28932fe8): each meets (a), (b) or
    # (c), the DL-9 MLP rows with the linear probes' substitutions (MLP_N2) by the Fig. 2 caption, so every row stays
    # (appendix layout report, task 2)

    def side(head, x):
        """A side's name under its heading: what the heading already says is not repeated (SP-14: each row names only
        its two sides)."""
        t = name(x)
        if "17 merged groups" in head:
            t = t.replace(" on the 17 groups", "")
        if HEADING["head"] in head:
            t = t.replace(", head", "")
        return t
    return _tests_tabular(blocks, side, single, analytic, num), {r for r in replicas if r}


def _tests_tabular(blocks: list, side, single: set, analytic: set, num) -> str:
    """tab:tests' tabular: each block under its italic heading (task, metric, read-out), a row its two sides."""
    out = [r"\begin{tabular}{lccc}", r"\toprule", TESTS_HEAD, r"\midrule"]
    for k, (head, rows) in enumerate(blocks):
        block = [_full_row(4, head, italic=True)]
        for first, second, v, ind in rows:
            lab = f"{side(head, first)} $-$ {side(head, second)}" + (INDEPENDENT if ind else "")
            if isinstance(v, tuple) and len(v) > 5:       # a row over several (least), and its mark
                lab += LEAST + (SINGLE_SIDE if v[5][1] else "")
            elif (head, first, second) in single:
                lab += SINGLE_SIDE
            if (head, first, second) in analytic:
                lab += ANALYTIC_MARK
            if v is None or v == "stale":
                block.append(lab + r" & \multicolumn{3}{c}{\pending{" + ("bootstrap" if v is None else "rerun bootstrap")
                             + r"}} \\")
                continue
            d, sig = v[0], v[4]
            place = _pdg_place(sig) + TESTS_EXTRA_DIGIT.get((head, first, second), 0)
            block.append(lab + " & " + num(d, place) + " & " + num(sig, place) + " & "
                         + f"{d / sig:.2f}".replace("-", "$-$") + r" \\")
        # a block's rows never break apart from its heading; between two blocks a space, where the page may break
        # LEDGER DL-14 (typographic): a block longer than TESTS_GLUE_ROWS lines may break after its first three rows
        # and before its last two, so that the caption and a long first block never leave a page empty (render e2305ce:
        # the 22 accuracy rows and the caption, unbreakable, pushed the table off a blank page)
        glue = [i < 3 or i >= len(block) - 3 or len(block) <= TESTS_GLUE_ROWS for i in range(len(block) - 1)]
        out += (([r"\noalign{\penalty-100}\addlinespace"] if k else [])
                + [ln.rstrip() + ("*" if g else "") for ln, g in zip(block[:-1], glue)] + block[-1:])
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


RATIO_WORD = re.compile(r"\\([A-Za-z]+Pr(?:Max)?)(?![A-Za-z])")


def text_ratio_words(journal: pathlib.Path) -> set:
    """The ratio words (Pr and PrMax macros) that main.tex and its sections use, comments left out."""
    files = [journal / "main.tex", *sorted((journal / "sections").glob("*.tex"))]
    return {n for f in files if f.exists() for n in RATIO_WORD.findall(re.sub(r"(?<!\\)%.*", "", f.read_text()))}


def _printed_mean(v: list, extra: int = 0) -> float:
    """A side's mean as a table prints it (_printed_place): at its SD's PDG place and extra places more where its table
    has them (EXTRA_DIGIT); a single value (the soft-drop mass, one run) or runs that agree, as it is."""
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m if _zero_sd(m, sd) else round(m, _pdg_place(sd) + extra)


# PI 2026-11-23 (LEDGER PW-1): wherever the results state a null, the smallest difference its test resolves, 3 sigma of
# its tab:tests row as a share of the reference's value (of its 1 - AUC for an AUC row, the errors the text counts);
# a statement over several rows gives the range. {macro: (headings, (first networks or None for every one), second)}
_PW_NETS = SETS[1:] + FLAV + (RAND_POOLED,)
_PW_R30 = {k: tuple(rf"{t}, $R_{{30}}$, " + HEADING[k] for t in ("top tagging", "quark/gluon tagging (Pythia)",
                                                                  "quark/gluon tagging (Herwig)")) for k in ("lp", "ft")}
PW_ROWS = {
    "MlstPwRthirtyLp": (_PW_R30["lp"], _PW_NETS, "L188"),
    "MlstPwRthirtyFt": (_PW_R30["ft"], _PW_NETS, "L188"),
    "MlstPwRthirtyFtTop": ((r"top tagging, $R_{30}$, " + HEADING["ft"],), _PW_NETS, "L188"),
    "MlstPwRthirtyFtQg": ((r"quark/gluon tagging (Pythia), $R_{30}$, " + HEADING["ft"],), _PW_NETS, "L188"),
    # PI 2026-11-24 (LEDGER SP-15): Fig. 3's Herwig null, now stated in section 4.4 with the figure gone
    "MlstPwRthirtyFtHerwig": ((r"quark/gluon tagging (Herwig), $R_{30}$, " + HEADING["ft"],), _PW_NETS, "L188"),
    "MlstPwMassClsRes": (("JetClass-II, AUC, " + HEADING["lp"],), ("L162_MASS",), "L162"),
    "MlstPwMassClsCoarse": (("JetClass-II, AUC, " + HEADING["lp"],), ("R16_Q1_MASS",), "R16_Q1"),
    # fourth review (B1; PI's "wherever the results state a null"): every other null of sections 4.1-4.6
    "MlstPwAccResLp": (("JetClass-II, accuracy, " + HEADING["lp"],), ("L162",), "L188"),
    "MlstPwCoarseTask": (("JetClass-II, accuracy on the 17 merged groups, " + HEADING["lp"],),
                         ("L162_17", "R42_Q1_17", "R16_Q1_17"), "L188_17"),
    "MlstPwResonantRes": (("JetClass-II, AUC over the resonant classes, " + HEADING["lp"],), ("L162",), "L188"),
    "MlstPwBcRes": ((rf"{_PAIR['bc_vs_bq_cs']}, AUC, " + HEADING["lp"],), ("L162",), "L188"),
    "MlstPwBcFlavZero": ((rf"{_PAIR['bc_vs_bq_cs']}, AUC, " + HEADING["lp"],), ("FLAV_F0",), "R16_Q1"),
    "MlstPwTopScratch": ((_FT_TOP,), ("L188",), "SCRATCH"),
    "MlstPwTopKin": ((_FT_TOP,), ("KIN",), "L188"),
    "MlstPwTopMassRes": ((_FT_TOP,), ("L162_MASS",), "L162"),
    "MlstPwTopMassCoarse": ((_FT_TOP,), ("R16_Q1_MASS",), "R16_Q1"),
    "MlstPwHerwigRatio": ((r"quark/gluon tagging (Herwig), $R_{30}$, " + HEADING["ft"],), ("L188_HERWIG",),
                          "SCRATCH_HERWIG"),
    "MlstPwHeadRes": (("jet mass, resolution, " + HEADING["head"],), ("R16_Q1_MASS_HEAD",), "L162_MASS_HEAD"),
    # LEDGER DL-9 (bs3): the MLP beside the linear-probe nulls; DL-3b: the MLP mass rows; R5-17: X->ee at 17 classes
    "MlstPwAccResMlp": (("JetClass-II, accuracy, " + HEADING["mlp"],), ("L162",), "L188"),
    "MlstPwCoarseTaskMlp": (("JetClass-II, accuracy on the 17 merged groups, " + HEADING["mlp"],), ("L162_17", "R42_Q1_17"),
                            "L188_17"),
    "MlstPwMlpMassRes": (("JetClass-II, AUC, " + HEADING["mlp"],), ("L162_MASS",), "L162"),
    "MlstPwMlpMassCoarse": (("JetClass-II, AUC, " + HEADING["mlp"],), ("R16_Q1_MASS",), "R16_Q1"),
    # the 188-class AUC rounds to one, so a share of its errors means nothing: 3 sigma in AUC ("abs")
    "MlstPwEeCoarse": ((rf"{_PAIR['ee_vs_mm']}, AUC, " + HEADING["lp"],), ("R16_Q1",), "L188", "abs"),
    "MlstPwPartRepro": ((_FT_TOP + ", published models",), ("PART_OURS",), "PART_PUB"),     # A2 (bs3), a lower bound
    "MlstPwPartRescale": ((_FT_TOP + ", ParT's training script",), ("PART_RESCALE",), "PART_OURS"),   # DL-16c, a floor
}
# LEDGER DL-9 (bs3): the MLP results the text states beside the linear-probe nulls, checked where tab:tests is read: on
# the 17 merged groups the 17-class model falls below 188 classes; with the linear probes' substitutions only the
# 17-class mass network differs from 188 classes in R30, above it on top tagging (the Fig. 2 caption)
MLP_STATED = {("JetClass-II, accuracy on the 17 merged groups, " + HEADING["mlp"], "R16_Q1_17", "L188_17"): -1,
              (r"top tagging, $R_{30}$, " + HEADING["mlp"] + ", missing inputs as for the linear probe", "R16_Q1_MASS",
               "L188"): 1}
MLP_N2 = tuple(rf"{h}, $R_{{30}}$, " + HEADING["mlp"] + ", missing inputs as for the linear probe"
               for h in ("top tagging", "quark/gluon tagging (Pythia)", "quark/gluon tagging (Herwig)"))


def mlp_claim_check(rows: dict) -> None:
    """FATAL unless the MLP rows the text states read as it says (MLP_STATED: the sign of a pass), and no other N2 row
    (MLP_N2) passes; a row still pending is let through (its sentence waits with it)."""
    for key, sign in MLP_STATED.items():
        v = rows.get(key)
        if isinstance(v, tuple) and not (sign * v[0] > 3 * v[4]):
            raise SystemExit(f"FATAL (DL-9): {key[1]} - {key[2]} under '{key[0]}' no longer passes 3 sigma as the text states")
    extra = [k for k, v in rows.items() if k[0] in MLP_N2 and k not in MLP_STATED and isinstance(v, tuple)
             and abs(v[0]) > 3 * v[4]]
    if extra:
        raise SystemExit(f"FATAL (DL-9): MLP rows the Fig. 2 caption states as nulls pass 3 sigma: {extra}")


# fifth review (A6; LEDGER TX-3): tab:tests rows printed with one more digit of sigma, so that a PW-1 share quoted from
# them is reproducible from the printed values (17 + mass - 17, MLP AUC: sigma 0.00113 printed 0.0011 gave 12%, not 13%)
TESTS_EXTRA_DIGIT = {("JetClass-II, AUC, " + HEADING["mlp"], "R16_Q1_MASS", "R16_Q1"): 1}
# fifth review (B4; LEDGER CS-6): PW_ROWS macros whose reference value no table prints get <name>Ref, printed in the text
# (the 188-class MLP accuracy on JetClass-II: Table 10 prints the AUC only)
PW_REF = {"MlstPwAccResMlp"}

# fifth review (B1): the introduction and discussion restate some nulls over several PW_ROWS at once; their range
PW_UNION = {"MlstPwMassClsAll": ("MlstPwMassClsRes", "MlstPwMassClsCoarse", "MlstPwMlpMassRes", "MlstPwMlpMassCoarse")}


def pw_macros(em, rows: dict, sides: dict, src) -> None:
    """PW_ROWS' macros, from table_tests' rows_out and sides_out: 3 sigma over the reference side's mean (over 1 - mean
    for an AUC row), in per cent at two significant figures, a range over several rows; pending while a row waits.
    PW_UNION's macros give the range over their PW_ROWS' rows, pending if any of them waits."""
    got = {}
    for name, (heads, firsts, second, *opt) in PW_ROWS.items():
        shares, wait = [], False
        for (h, a, b), v in rows.items():
            if h not in heads or b != second or a not in firsts:
                continue
            if not isinstance(v, tuple) or (h, a, b) not in sides:
                wait = True
                continue
            if abs(v[0]) > 3 * v[4]:        # a share is given only for a null (fourth review, B1)
                raise SystemExit(f"FATAL (PW-1): \\{name}: its row {a} - {b} under '{h}' passes 3 sigma, so the text "
                                 "that states a null there is wrong")
            ref = float(np.mean(sides[(h, a, b)][1]))
            ref = 1 - ref if "AUC" in h else ref
            shares.append(3 * v[4] if "abs" in opt else 100 * 3 * v[4] / ref)
            # fifth review (A6; LEDGER TX-3, CS-6): the share from the printed sigma and reference, at the quoted digits;
            # a row whose two differ gets one more digit of sigma in tab:tests (TESTS_EXTRA_DIGIT)
            sig = round(v[4], _pdg_place(v[4]) + TESTS_EXTRA_DIGIT.get((h, a, b), 0))
            pref = _printed_mean(sides[(h, a, b)][1])
            pref = 1 - pref if "AUC" in h else pref
            psh = 3 * sig if "abs" in opt else 100 * 3 * sig / pref
            print(f"PW-1 {'ok  ' if _sig2(psh) == _sig2(shares[-1]) else 'DIFF'} {name}: {a} - {b} under '{h}': "
                  f"full precision {_sig2(shares[-1])}, printed values {_sig2(psh)}")
            if name in PW_REF:          # fifth review (B4; LEDGER CS-6): the reference value no table prints
                vals = sides[(h, a, b)][1]
                place = _pdg_place(float(np.std(vals, ddof=1)))
                em.macro(name + "Ref", f"{_printed_mean(vals):.{place}f}", src,
                         f"PW-1: the reference value of \\{name}, the mean of its tab:tests row's second side")
        if wait or not shares:
            em.macro(name, r"\pending{bootstrap}", src, "PW-1: a row waits for its bootstrap")
            if name in PW_REF:
                em.macro(name + "Ref", r"\pending{bootstrap}", src, "PW-1: a row waits for its bootstrap")
            got[name] = None
            continue
        got[name] = shares
        lo, hi = min(shares), max(shares)
        pct = "" if "abs" in opt else "\\%"
        body = f"{_sig2(lo)}{pct}" if _sig2(lo) == _sig2(hi) else f"{_sig2(lo)}--{_sig2(hi)}{pct}"
        em.macro(name, body, src, f"PW-1: 3 sigma over the reference value, {len(shares)} tab:tests rows, per cent"
                 if pct else "PW-1: 3 sigma of the tab:tests row, in the metric's units (the reference rounds to one)")
    for name, parts in PW_UNION.items():
        if any(got.get(n) is None for n in parts):
            em.macro(name, r"\pending{bootstrap}", src, "PW-1: a row waits for its bootstrap")
            continue
        shares = [x for n in parts for x in got[n]]
        lo, hi = min(shares), max(shares)
        body = f"{_sig2(lo)}\\%" if _sig2(lo) == _sig2(hi) else f"{_sig2(lo)}--{_sig2(hi)}\\%"
        em.macro(name, body, src, f"PW-1: the range of {', '.join(parts)}, {len(shares)} tab:tests rows, per cent")


# fifth review (B1): the discussion's sculpting null is about the mass networks against their own label sets only
SCULPT_MASS_PAIRS = {("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")}


def sculpt_pw_macro(em, sculpt: dict, sculpt_doc, runs: dict, src) -> None:
    """MlstPwSculpt (fourth review, B1; LEDGER PW-1): the sculpting nulls of section 4.6, the range over tab:tests'
    sculpting rows (every stated pair, every working point) of 3 sigma over the reference network's mean distance
    (tab:sculptmain's cells); FATAL if a row the text states as a null passes; pending without the inputs."""
    if not sculpt_doc or not sculpt:
        for name in ("MlstPwSculpt", "MlstPwSculptMass"):
            em.macro(name, r"\pending{bootstrap}", src, "PW-1: the sculpting bootstrap or distances are absent")
        return
    arm = {m.removeprefix("mtx-").rsplit("-s", 1)[0]: a for a, ms in runs.items() if a != RAND_POOLED for m in ms}
    arm.update(SCULPT_ARM)
    pooled = _pooled(runs)
    shares, mass = [], []
    for d in sculpt_doc["differences"].values():
        a, b = arm[d["configuration"]], arm[d["reference"]]
        if (a, b) in SCULPT_UNSTATED:
            continue
        for wp, _ in SCULPT_BOOT_WPS:
            x = d["l3"][wp]["distance"]
            if abs(x["delta"]) > 3 * x["sigma"]:
                raise SystemExit(f"FATAL (PW-1): the sculpting row {a} - {b} at {wp} passes 3 sigma, so the text's "
                                 "sculpting null is wrong")
            if wp.startswith("eps_b="):
                ref = [_sculpt(sculpt, m, wp.removeprefix("eps_b=")) for m in pooled[b]]
            else:
                ref = [None if (s := sculpt.get(m)) is None else math.sqrt(s["sig_eff_50"]["jsd"]) for m in pooled[b]]
            if None in ref or not ref:
                for name in ("MlstPwSculpt", "MlstPwSculptMass"):
                    em.macro(name, r"\pending{distances}", src, "PW-1: a reference distance is absent")
                return
            shares.append(100 * 3 * x["sigma"] / float(np.mean(ref)))
            if (a, b) in SCULPT_MASS_PAIRS:
                mass.append(shares[-1])
    for name, sh, what in (("MlstPwSculpt", shares, "sculpting rows"),
                           ("MlstPwSculptMass", mass, "sculpting rows of a mass network against its label set")):
        if not sh:
            em.macro(name, r"\pending{bootstrap}", src, f"PW-1: no {what} in the sculpting bootstrap")
            continue
        lo, hi = min(sh), max(sh)
        em.macro(name, f"{_sig2(lo)}\\%" if _sig2(lo) == _sig2(hi) else f"{_sig2(lo)}--{_sig2(hi)}\\%", src,
                 f"PW-1: 3 sigma over the reference distance, {len(sh)} {what} of tab:tests, per cent")


RATIO_CHECK_COLS = ("macro", "heading", "first", "second", "noun", "ratio_full_precision", "ratio_printed_means",
                    "word_full_precision", "word_printed", "printed_mean_first", "printed_mean_second",
                    "full_mean_first", "full_mean_second", "agree")


def ratio_check_csv(em) -> str:
    """The data release's ratio_check.csv (PI 2026-11-23, LEDGER TX-3): every quoted ratio, from the full-precision
    means and from the printed means, and whether the two read the same at the quoted digits."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(RATIO_CHECK_COLS)
    for r in vars(em).get("ratio_check", []):
        w.writerow([r[0], r[1], _release_name(r[2]), _release_name(r[3]), r[4], repr(r[5]), repr(r[6]), r[7], r[8],
                    repr(r[9]), repr(r[10]), repr(r[11]), repr(r[12]),
                    "yes" if r[7] == r[8] else "no"])
    return out.getvalue()


def ratio_words(em, rows: dict, used: set, src, sides: dict | None = None) -> None:
    """Every ratio word, in one place (PI 2026-11-23, LEDGER SP-12 step 3): each of PR_ROWS reads its tab:tests row
    (rows: table_tests' rows_out). |Delta| > 3 sigma (PI SC-1 (1)): the ratio's direction and value in words (_phrase);
    not: the null phrase of its sentence; no stored bootstrap: the row's marker. The ratio is that of the two sides'
    means as the tables print them (sides: table_tests' sides_out; _printed_mean), of 1 - AUC for errors, so a reader
    computes it from the printed values (review of render dfe5942, A2, A3: LEDGER CS-6, "any ratio the text quotes must
    be computable from a table in the paper", and CS-8, "mean with mean"), never a mean or a bound of per-seed ratios.
    A ratio not in PR_ROWS has no word. The build stops on a ratio word emitted anywhere else, on one the text uses
    (used) without a row, on a row TESTS does not hold, on a passing row without its printed sides, and on a word whose
    direction is not its row's."""
    stray = sorted(n for n in em.provenance if n.endswith(("Pr", "PrMax")))
    if stray:
        raise SystemExit(f"FATAL: ratio words emitted outside ratio_words: {stray[:10]}")
    lost = sorted(used - set(PR_ROWS))
    if lost:
        raise SystemExit(f"FATAL: the text uses ratio words with no tab:tests row (PR_ROWS): {lost}")
    have = {(h, a, b) for h, rs in TESTS for a, b, _ in rs}
    held = _held(em)
    checks = vars(em).setdefault("ratio_check", [])
    for name, (head, first, second, null) in PR_ROWS.items():
        if (head, first, second) not in have:
            raise SystemExit(f"FATAL: \\{name}: tab:tests has no row {first} - {second} under '{head}'")
        h, v = held.get(name), rows.get((head, first, second))
        if h is None:
            word, why = r"\pending{no run yet}", "no ratio computed"
        elif h["body"] is not None:
            word, why = h["body"], "no ratio to state"
        elif v is None or v == "stale":
            word, why = r"\pending{" + ("bootstrap" if v is None else "rerun bootstrap") + "}", "no stored bootstrap"
        elif abs(v[0]) > 3 * v[4]:
            pair = (sides or {}).get((head, first, second))
            if not pair:
                raise SystemExit(f"FATAL: \\{name}: tab:tests row {first} - {second} passes with no printed values of "
                                 "its sides to form the ratio from (LEDGER CS-6)")
            # the pair AUCs print in tab:pairauc (LEDGER DL-R #4), at its extra digit (EXTRA_DIGIT, LEDGER TX-3)
            extra = EXTRA_DIGIT["pair_auc"].get(RATIO_PAIR_HEADS.get(head), 0)
            a, b = (_printed_mean(s, extra) for s in pair)
            if (1 - b if h["noun"] == "errors" else b) == 0:
                raise SystemExit(f"FATAL: \\{name}: its reference prints no error (or a zero), so no ratio")
            q = (1 - a) / (1 - b) if h["noun"] == "errors" else a / b
            up = -1 if h["noun"] == "errors" else 1         # more errors: the AUC or accuracy tested falls
            if (q - 1) * v[0] * up <= 0:
                raise SystemExit(f"FATAL: \\{name} reads {q:.4g}, against the sign of its tab:tests difference {v[0]:+.4g}")
            word, why = _phrase(q, h["noun"]), f"passes 3 sigma (|delta|/sigma = {abs(v[0]) / v[4]:.1f})"
            # PI 2026-11-23 (LEDGER TX-3): the ratio from the full-precision means beside the printed one, at the quoted
            # digits (ratio_check.csv in the data release; the build prints the ones that differ)
            fa, fb = (float(np.mean(s)) for s in pair)
            qf = (1 - fa) / (1 - fb) if h["noun"] == "errors" else fa / fb
            checks.append((name, head, first, second, h["noun"], qf, q, _phrase(qf, h["noun"]), word, a, b, fa, fb))
        else:
            word, why = null, f"does not pass 3 sigma (|delta|/sigma = {abs(v[0]) / v[4]:.1f})"
        em.macro(name, word, h["src"] if h else src, h["path"] if h else "not computed",
                 f"ratio word, tab:tests row {first} - {second} ({head}): {why}")


# ------------------------------------------------------------------ published models (main text), the other label sets (appendix)
# PI 2026-11-12: one main-text table beside the published models, with our network from scratch and fine-tuned from
# 188 classes only; every other label set's fine-tuned row in one appendix table. Apart from tab:summary a number
# has one table: the 188-class and from-scratch rows of JetClass, top and quark/gluon (Pythia) are in tab:published
# and not in tab:labelsets.
THIS_WORK = " (this work)"
EFN_N_TRAIN = 1_000_000     # the EFN paper (arXiv:1810.05165, App. A): one million training jets for its quark/gluon models
TASK_BLOCKS = (("jc1", "JetClass"), ("top", "top tagging"), ("qg", "quark/gluon tagging"),
               ("qg_herwig", "quark/gluon tagging, scored on Herwig jets"))
# tab:published is wider than the upright text block and taller than a turned page: two turned pages, cut between tasks
PUBLISHED_PAGES = (("jc1", "top"), ("qg",))


def _jets(n: float) -> str:
    return rf"{n / 1e6:g}\,M"


def _task_pages(key: str, head: list, pages: tuple, blocks: dict, heading, italic: bool = False) -> str:
    """One tabular per page (%SPLIT% between them, which _float sets one after the other), the tasks of a page
    under heading(task) and between rules. Each task's columns are at one precision of their own (a top-tagging
    rejection and a quark/gluon one share a column and not a size)."""
    ncol = len(_depth0_split(head[2], "&"))
    out = []
    for tasks in pages:
        body = [ln for k, t in enumerate(tasks) for ln in ([r"\midrule"] if k else []) + [_full_row(ncol, heading(t), italic)]
                + _ours_apart(key, _bold(key, "\n".join(blocks[t])))]
        out.append("\n".join(head + body + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(out)


def _ours_apart(key: str, block: str) -> list:
    """The lines of a bolded block at one precision per column (uniform_columns); in tab:published our rows (the
    "(this work)" rows and the kinematic-inputs network) at a precision of their own and never coarser than units,
    as Table 1 prints them (page review 2026-11-13: a column shared with a published +-130 had rounded our top-tagging
    R30 to tens, 1,410 +- 20 against Table 1's 1,412 +- 20). The published rows print as before."""
    whole = uniform_columns(block).split("\n")
    if key != "published":
        return whole
    lines = block.split("\n")
    ours = [i for i, ln in enumerate(lines) if THIS_WORK in ln or ln.startswith(LABEL["KIN"] + " &")]
    if not ours:
        return whole
    mine = uniform_columns("\n".join(lines[i] for i in ours), floor=0).split("\n")
    for i, ln in zip(ours, mine):
        whole[i] = ln
    return whole


def jc1_values(per: dict, runs: dict):
    """vals(arm, metric) for mlst_v4.jetclass_rows: each run's fine-tuned JetClass accuracy or AUC on one million
    training jets as Table 1, tab:summaryauc and the figures read them (summary_ft, summary_scratch; arm None: from
    scratch), so that every table prints one value for one quantity (PI 2026-11-13, L2)."""
    def vals(arm, met):
        if arm is None:
            return summary_scratch(per, "jc1", FULL_N["jc1"], met) or None
        g = summary_ft(per, "jc1", FULL_N["jc1"], met)
        v = [g(m) for m in runs.get(arm, [])] if g else []
        return v if v and all(x is not None for x in v) else None
    return vals


def table_published(per: dict, runs: dict, pre: dict, ref: dict, fmt_pm) -> str:
    """tab:published: per task (TASK_BLOCKS without Herwig) the published models, then our network trained from
    scratch and fine-tuned from the 188-class network; on top tagging also the 188-class network with kinematic
    inputs, fine-tuned (one run). Columns: parameters, training jets (the task's training set; a dash for an
    observable, which is not trained), accuracy, AUC, R50, R30; JetClass has no R50 and R30 here (its per-class
    rejections are tab:jetclassrej). The published rows are those of the task tables (mlst_v4.LIT_ROWS, the 2 M
    JetClass rows)."""
    ft = f"fine-tuned from {LABEL['L188']}{THIS_WORK}"
    blocks = {}
    lit, ours = V4.jetclass_rows(pre, ref, runs, ("L188",), LABEL, SCRATCH, FULL_N["jc1"], fmt_pm, fmt_pm.two,
                                 vals=jc1_values(per, runs))
    rows = []
    same = repeated_cells(per, runs)      # L17: the accuracy and R50 of these rows as tab:labelsets prints them
    for prefix, cells, _ in lit + sorted(ours, key=lambda r: not r[0].startswith(SCRATCH_LABEL)):
        name, jets, params = V4._depth0(prefix)
        name = {LABEL["L188"]: ft, SCRATCH_LABEL: SCRATCH_LABEL + THIS_WORK}.get(name, name)
        rows.append(f"{name} & {params} & " + jets.replace(" M", r"\,M") + f" & {cells[0]} & {cells[1]} & -- & -- \\\\")
    blocks["jc1"] = rows
    for task in ("top", "qg"):
        lit, ours = V4.bench_rows(task, per, runs, ("L188",), LABEL, SCRATCH, fmt_pm, fmt_pm.two,
                                  kin=(LABEL["KIN"], KIN_INIT) if task == "top" else None, published=True)
        # audit 3 (PI 2026-11-14): the EFN paper's rows (linear EFP, PFN-Ex) trained on one million jets (its App. A)
        n_of = lambda name: EFN_N_TRAIN if task == "qg" and any(k in name for k in ("EFP", "PFN")) else FULL_N[task]  # noqa: E731
        rows = [f"{name} & {params} & " + ("--" if kind == "observable" else _jets(n_of(name))) + " & "
                + " & ".join(cells) + r" \\" for name, cells, params, kind in lit]
        by = {name: (cells, params) for name, cells, params in ours}
        for name, arm in ((SCRATCH_LABEL, None), (LABEL["L188"], "L188")):
            if name in by:
                by[name] = ([same.get((arm, task, met), c) if met in ("accuracy", "r50") else c
                             for c, (_, met, _) in zip(by[name][0], V4.BENCH_COLS)], by[name][1])
        for name, shown in ((SCRATCH_LABEL, SCRATCH_LABEL + THIS_WORK), (LABEL["L188"], ft), (LABEL["KIN"], LABEL["KIN"])):
            if name in by:
                rows.append(f"{shown} & {by[name][1]} & {_jets(FULL_N[task])} & " + " & ".join(by[name][0]) + r" \\")
        blocks[task] = rows
    head = [r"\begin{tabular}{lcccccc}", r"\toprule",
            rf" & {V4.PARAMS_HEAD} & training jets & accuracy & AUC & $R_{{50}}$ & $R_{{30}}$ \\", r"\midrule"]
    return _task_pages("published", head, PUBLISHED_PAGES, blocks, dict(TASK_BLOCKS).get)


# tab:probes and tab:labelsets (PI 2026-11-13, plan "tables"): what Table 1 and tab:summaryauc lack of the top and
# quark/gluon networks, the accuracy and R50 (their R30 and AUC are there), one column group a task (top; quark/gluon
# scored on Pythia and on Herwig jets), the family's rows in the common order (L5): linear probes (tab:probes) and
# fine-tuned networks (tab:labelsets), one turned page each
BENCH_GROUPS = (("top", "top tagging"), ("qg", "quark/gluon, Pythia jets"), ("qg_herwig", "quark/gluon, Herwig jets"))
BENCH2 = (("accuracy", "accuracy"), ("r50", "$R_{50}$"))


def _bench_groups(key: str, mode: str, rows: list, refs: list) -> str:
    """The tabular of tab:probes or tab:labelsets: under the italic heading of mode, rows [(arm, cells)] in the order
    given (a partition indented under PARTITIONS), then refs [(label, cells) or a rule]; bold per column (_bold),
    each column at its own precision."""
    nc = 1 + len(BENCH_GROUPS) * len(BENCH2)
    k = len(BENCH2)
    head = [r"\begin{tabular}{l" + "c" * (nc - 1) + "}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{t}}}" for _, t in BENCH_GROUPS) + r" \\",
            "".join(rf"\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(BENCH_GROUPS))),
            " & " + " & ".join(h for _ in BENCH_GROUPS for _, h in BENCH2) + r" \\", r"\midrule"]
    body = [_full_row(nc, HEADING[mode], italic=True)]
    for arm, cells in rows:
        part = arm in FLAV + RAND + (RAND_POOLED,)
        if part and PARTITIONS not in body[-1] and not body[-1].startswith(r"\quad "):
            body.append(_full_row(nc, PARTITIONS))
        body.append((r"\quad " if part else "") + LABEL[arm] + " & " + " & ".join(cells) + r" \\")
    body += [r if r == r"\midrule" else f"{r[0]} & " + " & ".join(r[1]) + r" \\" for r in refs]
    return "\n".join(head + uniform_columns(_bold(key, "\n".join(body))).split("\n")
                     + [r"\bottomrule", r"\end{tabular}", ""])


def table_probes_bench(fits, runs, fmt_pm, rows, fill=None, chosen=None, kin=False) -> str:
    """tab:probes: linear probes at each task's largest training set (_bench_rec, under the substitution chosen on
    validation), the accuracy and R50 on top and quark/gluon (Pythia, Herwig); rows the networks (runs pooled or not),
    kin=True the kinematic-only model after them (top only), then INIT, and SOPHON below a rule, where in rows."""
    fill, chosen = fill or {}, chosen or {}

    def cells(ms):
        out = []
        for t, _ in BENCH_GROUPS:
            recs = [_bench_rec(fits, fill, chosen, m, t) for m in ms]
            out += [_cell([None if r is None else r["accuracy"] for r in recs], len(ms), fmt_pm, frac=True),
                    _cell([_rej(r, "r50") for r in recs], len(ms), fmt_pm)]
        return out
    nets = [(a, cells(runs[a])) for a in rows if a not in (INIT, INIT_BN, SOPHON)]
    # the references of L5 (random network), then the kinematic-inputs network outside the network list (plan "tables")
    refs = [(LABEL[a], cells(runs[a])) for a in rows if a in (INIT, INIT_BN)]
    if kin and _probe_rec(fits, KIN_MODEL, "top") is not None:
        refs.append((LABEL["KIN"], cells([KIN_MODEL])[:len(BENCH2)] + ["--"] * len(BENCH2) * (len(BENCH_GROUPS) - 1)))
    if SOPHON in rows:
        refs += [r"\midrule", (LABEL[SOPHON], cells(runs[SOPHON]))]
    return _bench_groups("probes", "lp", nets, refs)


LABELSETS_ROWS = SETS + FLAV + (RAND_POOLED,)


def bench_ft_cells(per: dict, runs: dict, fmt_pm, rows=LABELSETS_ROWS) -> tuple[list, list]:
    """The one source of the fine-tuned top and quark/gluon cells that tab:labelsets and tab:published share (PI
    2026-11-13, L17): the networks on the full training sets, read as Table 1 reads them (summary_ft,
    summary_scratch), the accuracy and R50 per BENCH_GROUPS task; ([(arm, cells)] of rows with a value, the five
    random partitions pooled, and from scratch's cells), each cell a token (fmt_pm) or "--"."""
    pooled = _pooled(runs)

    def cells(arm):
        out = []
        for t, _ in BENCH_GROUPS:
            for met, _ in BENCH2:
                if arm is None:
                    v = summary_scratch(per, t, FULL_N[t], met)
                    out.append(_cell(v, len(v), fmt_pm, frac=met == "accuracy") if v else "--")
                    continue
                g = summary_ft(per, t, FULL_N[t], met)
                v = [g(m) for m in pooled[arm]] if g else []
                out.append(_cell(v, len(pooled[arm]), fmt_pm, frac=met == "accuracy") if any(x is not None for x in v)
                           else "--")
        return out
    nets = [(a, c) for a, c in ((a, cells(a)) for a in rows) if any(x != "--" for x in c)]
    return nets, cells(None)


def repeated_cells(per: dict, runs: dict) -> dict:
    """{(arm, task, metric): the cell as tab:labelsets prints it, bold left out} for the rows tab:published repeats
    (L17): the network fine-tuned from 188 classes ("L188") and from scratch (None), on top and quark/gluon
    (Pythia). From bench_ft_cells, each column at the one precision uniform_columns gives tab:labelsets' column."""
    nets, scratch = bench_ft_cells(per, runs, TAB)
    lines = uniform_columns("\n".join(" & ".join(["x"] + c) + r" \\" for c in [c for _, c in nets] + [scratch])).split("\n")
    printed = dict(zip([a for a, _ in nets] + [None], lines))
    cols = [(t, met) for t, _ in BENCH_GROUPS for met, _ in BENCH2]
    return {(arm, t, met): _depth0_split(printed[arm].removesuffix(r"\\").strip(), "&")[1 + k].strip()
            for arm in ("L188", None) if arm in printed for k, (t, met) in enumerate(cols) if t in ("top", "qg")}


def table_labelsets(per: dict, runs: dict, fmt_pm, rows=None, key: str = "labelsets") -> str:
    """tab:labelsets: bench_ft_cells' networks (rows, default LABELSETS_ROWS: SETS, the 17-class partitions, the five
    random partitions pooled), then from scratch; key: the BOLD entry (labelsets, labelsets_full)."""
    nets, scratch = bench_ft_cells(per, runs, fmt_pm, rows or LABELSETS_ROWS)
    return _bench_groups(key, "ft", nets, [(SCRATCH_LABEL, scratch)])


# tab:probesn and tab:ft (PI 2026-11-13, plan "tables"): the points of the training-size figures (gallery_labelset
# SIZE5: A10 linear probes, A8 fine-tuned, and their partition slices) below the largest training set, whose values are
# Table 1's; one block a task in the figure's metric, one column a training-set size, the family's rows (L5) with the
# random partitions pooled, then the mode's reference. Values as paper_figures.probe_series, ft_series and
# scratch_series read them. The supplement's copies (tab:probesnfull, tab:ftfull) keep every size, every random
# partition and the AUC.
SIZE_TASKS = (("jc2", "jc2_188", "accuracy", "JetClass-II"), ("jc1", "jc1", "accuracy", "JetClass"),
              ("top", "top", "r30", "top tagging"), ("qg", "qg", "r30", "quark/gluon tagging, Pythia jets"),
              ("qg_herwig", "qg_herwig", "r30", "quark/gluon tagging, Herwig jets"))
SIZE_PAGES = (("jc2", "jc1"), ("top", "qg", "qg_herwig"))
SIZE_ROWS = SETS + FLAV + (RAND_POOLED,)
SIZE_METRIC = {"accuracy": "accuracy", "r30": "$R_{30}$"}


def size_probe(fits, fill, chosen, runs, ds: str, met: str, row: str) -> dict:
    """{training jets: [the linear probe of each run of row]}, as paper_figures.probe_series."""
    if ds in ("top", "qg", "qg_herwig"):
        sizes = _bench_sizes(fits, fill, chosen, ds)
        get = lambda m, n: (lambda r: _rej(r, met) if met.startswith("r") else None if r is None else r.get(met))(   # noqa: E731
            _bench_rec(fits, fill, chosen, m, ds, n))
    else:
        sizes = sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int)
        get = lambda m, n: (lambda r: None if r is None else r.get(met))(_probe_rec(fits, m, ds, n))    # noqa: E731
    out = {}
    for n in sizes:
        v = [x for m in runs[row] if (x := get(m, n)) is not None]
        if v:
            out[int(n)] = v
    return out


def size_ft(per, runs, key: str, met: str, row: str | None) -> dict:
    """{training jets: [the fine-tuned value of each run of row]}, as paper_figures.ft_series; row None: from scratch,
    as scratch_series."""
    if not per.get(key):
        return {}
    if row is None:
        return {n: [r[met] for r in recs if r.get(met) is not None]
                for n, recs in sorted(per[key].get(SCRATCH, {}).items()) if recs}
    out = {}
    for n in _sizes(per, key):
        v = [x for m in runs[row] if (x := _ft(per, key, m, n, met)) is not None]
        if v:
            out[n] = v
    return out


def table_by_size(key: str, mode: str, series, ref: tuple, largest: dict, rows=SIZE_ROWS) -> str:
    """tab:probesn (mode "lp") or tab:ft ("ft"): one turned or upright page per SIZE_PAGES entry, the tasks of a page
    under one heading row of sizes (refused if they differ), each task a block under its italic heading: rows (the
    partitions indented), then ref = (label, row id). series(task key, data set, metric, row id) gives {n: values};
    largest[task]: Table 1's training set, whose column is left out. A row with no value at the sizes shown is left
    out. Bold per block and column, each column of a block at its own precision."""
    pages = []
    for tasks in SIZE_PAGES:
        blocks, data = [], []
        for t in tasks:
            _, ds, met, title = next(x for x in SIZE_TASKS if x[0] == t)
            rs = {a: series(t, ds, met, a) for a in rows + (ref[1],)}
            data.append((met, title, rs, sorted({n for v in rs.values() for n in v if n != largest[t]})))
        # the paper build refuses tasks with different sizes; the preview book (ft_preview: the partial one-rule
        # read-out) shows every size any task has, "--" where a task has none yet
        if len({tuple(d[3]) for d in data}) > 1 and not ft_preview():
            raise SystemExit(f"FATAL: {key}: the tasks {tasks} have different training sizes")
        shown = sorted({n for d in data for n in d[3]})
        for met, title, rs, _ in data:
            sizes = shown
            nc = 1 + len(sizes)
            body = [_full_row(nc, f"{HEADING[mode]}, {title}, {SIZE_METRIC[met]}", italic=True)]
            for a in rows + (ref[1],):
                if not any(n in rs[a] for n in sizes):
                    continue
                part = a in FLAV + RAND + (RAND_POOLED,)
                if part and PARTITIONS not in body[-1] and not body[-1].startswith(r"\quad "):
                    body.append(_full_row(nc, PARTITIONS))
                name = ref[0] if a == ref[1] else (r"\quad " if part else "") + LABEL[a]
                body.append(name + " & " + " & ".join(_cell(rs[a][n], len(rs[a][n]), TAB, frac=met != "r30")
                                                      if n in rs[a] else "--" for n in sizes) + r" \\")
            blocks.append(uniform_columns(_bold(key, "\n".join(body))))
        nc = 1 + len(shown)
        head = [r"\begin{tabular}{l" + "c" * len(shown) + "}", r"\toprule",
                rf" & \multicolumn{{{len(shown)}}}{{c}}{{training jets}} \\", rf"\cmidrule(lr){{2-{nc}}}",
                " & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in shown) + r" \\", r"\midrule"]
        pages.append("\n".join(head + "\n\\midrule\n".join(blocks).split("\n") + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(pages)


# ------------------------------------------------------------------ tab:masseffect (PI 2026-11-12/13, brief B)
# The effect of the mass-regression head, as values (not differences): one row per task, metric and downstream mode,
# one column per network (162 classes, 162 classes + mass, 17 classes, 17 classes + mass), each mean +- SD over its
# runs. Every value is read through the accessor of the table or figure that holds it elsewhere (atlas_res, the
# headline value functions, the multilayer-perceptron probes, the X->bb / X->bs read-outs, the sculpting files).
MASS_EFFECT = ("L162", "L162_MASS", "R16_Q1", "R16_Q1_MASS")
# the pages of the table (a turned page each), by block
MASS_EFFECT_PAGES = (("mass", "jc2"), ("jc1", "top", "qg"), ("qg_herwig", "xbb", "xbs", "sculpt"))
MASS_EFFECT_BLOCKS = {"mass": "jet mass on JetClass-II, all jets", "jc2": "JetClass-II", "jc1": "JetClass",
                      "top": "top tagging", "qg": "quark/gluon tagging, Pythia jets",
                      "qg_herwig": "quark/gluon tagging, Herwig jets",
                      "xbb": r"$X\to b\bar b$ against QCD at 1\% QCD efficiency",
                      "xbs": r"$X\to bs$ against QCD at 1\% QCD efficiency",
                      "sculpt": "Jensen--Shannon distance of the QCD jet mass"}


def masseffect_rows(atlas, fits, fill, chosen, per, mlp, XR, xfits, sculpt, sculpt_ft, runs) -> list:
    """[(block, row label, value of a run or None, direction)], direction "lower", "unity" (best closest to 1) or
    "higher" for the bold rule. The direct X->bb discriminant only for a network with an X->bb class of its own
    (xbx_report.direct_ok)."""
    from types import SimpleNamespace
    no_direct = {m for a in MASS_EFFECT if not XR.direct_ok(SimpleNamespace(RAND=RAND), a) for m in runs[a]}
    W = {"lp": "linear probe", "mlp": "multilayer-perceptron probe", "ridge": "ridge probe", "ft": "fine-tuned",
         "head": "mass output of the pretraining head"}
    rows = []
    for est in ("regressed", "mlp", "ridge"):
        for key, word, d in (("resolution", "resolution", "lower"), ("median", "median response", "unity")):
            rows.append(("mass", f"{word}, {W['head' if est == 'regressed' else est]}",
                         lambda m, est=est, key=key: _ares(atlas, m, est, "all", key), d))
    for blk, key, ds, met, word in (("jc2", "jc2", "jc2_188", "accuracy", "accuracy"), ("jc2", "jc2", "jc2_188", "macro_auc_ovr", "AUC"),
                                    ("jc1", "jc1", "jc1", "accuracy", "accuracy"), ("jc1", "jc1", "jc1", "macro_auc_ovr", "AUC"),
                                    ("top", "top", "top", "r30", "$R_{30}$"), ("top", "top", "top", "auc", "AUC"),
                                    ("qg", "qg", "qg", "r30", "$R_{30}$"), ("qg", "qg", "qg", "auc", "AUC"),
                                    ("qg_herwig", "qg_herwig", "qg_herwig", "r30", "$R_{30}$"),
                                    ("qg_herwig", "qg_herwig", "qg_herwig", "auc", "AUC")):
        rows.append((blk, f"{word}, {W['lp']}", summary_probe(fits, fill, chosen, ds, met), "higher"))
    # the multilayer-perceptron probes the table of nonlinear probes holds for these networks: JetClass, and top with
    # the missing inputs at zero (MLST_TASKS)
    for blk, task, met, word in (("jc1", "jc1", "accuracy", "accuracy"), ("jc1", "jc1", "macro_auc_ovr", "AUC"),
                                 ("top", "top", "r30", "$R_{30}$")):
        rows.append((blk, f"{word}, {W['mlp']}", lambda m, task=task, met=met: _mlp_val(mlp, m, task, met), "higher"))
    for blk, n in (("jc2", 1_000_000), ("jc1", 1_000_000), ("top", FULL_N["top"]), ("qg", FULL_N["qg"]),
                   ("qg_herwig", FULL_N["qg_herwig"])):
        for met, word in ((("accuracy", "accuracy") if blk.startswith("jc") else ("r30", "$R_{30}$")),
                          ("macro_auc_ovr" if blk.startswith("jc") else "auc", "AUC")):
            rows.append((blk, f"{word}, {W['ft']}", summary_ft(per, blk, n, met), "higher"))
    order = {b: i for i, b in enumerate(MASS_EFFECT_BLOCKS)}
    rows.sort(key=lambda r: order[r[0]])        # stable: within a block linear probe, MLP probe, fine-tuned
    for sig in ("xbb", "xbs"):
        for name, word in (("head", "classifier"), ("linear", "linear probe"),       # Table 4's words
                           ("zero_shot_xbb", "direct")):
            if name == "zero_shot_xbb" and sig != "xbb":
                continue

            def eff(m, sig=sig, name=name):
                if name == "zero_shot_xbb" and m in no_direct:
                    return None
                r = XR.readout(xfits.get(m, {}).get(sig, {}), name)
                return None if r is None else 100 * r["eps_s_at_star"]
            rows.append((sig, f"signal efficiency (\\%), {word}", eff, "higher"))
    for src, word in ((sculpt, "network output used directly"), (sculpt_ft, W["ft"])):
        for wp, at in (("0.1", "10\\% QCD efficiency"), ("0.01", "1\\% QCD efficiency"), ("sig50", "50\\% signal efficiency")):
            rows.append(("sculpt", f"at {at}, {word}",
                         lambda m, src=src, wp=wp: None if (d := src.get(m)) is None else math.sqrt(
                             d["sig_eff_50"]["jsd"] if wp == "sig50" else d["working_points"][wp]["jsd"]), "lower"))
    return rows


def table_masseffect(rows, runs, fmt_pm) -> str:
    """tab:masseffect: the values of rows (masseffect_rows) for MASS_EFFECT, one %SPLIT% part (a turned page) per
    MASS_EFFECT_PAGES entry, the blocks under their headings between rules. Each row is printed at one precision
    (the rule of a column, applied to the row, D1) and bold by the rule of a column applied to the row (D19): the
    best value and every value within three standard deviations of it, best the lowest, the closest to 1 or the
    highest as the row's direction says."""
    nets = MASS_EFFECT
    # the rows as columns of a transposed table, so that uniform_columns and _bold treat each row as a column
    spec = {**BOLD["masseffect"], "lower": tuple(i for i, r in enumerate(rows, 1) if r[3] == "lower"),
            "unity": tuple(i for i, r in enumerate(rows, 1) if r[3] == "unity")}
    cells = [[_cell([get(m) for m in runs[a]], len(runs[a]), fmt_pm, frac=True) if get and any(
        get(m) is not None for m in runs[a]) else "--" for _, _, get, _ in rows] for a in nets]
    t = [r"\begin{tabular}{l" + "c" * len(rows) + "}", r"\toprule", " & " + " & ".join(["x"] * len(rows)) + r" \\", r"\midrule"]
    t += [LABEL[a] + " & " + " & ".join(c) + r" \\" for a, c in zip(nets, cells)]
    done = uniform_columns(_bold("masseffect", "\n".join(t + [r"\bottomrule", r"\end{tabular}"]), spec))
    printed = [_depth0_split(ln.rstrip().removesuffix(r"\\"), "&")[1:] for ln in done.split("\n")[4:4 + len(nets)]]
    head = [r"\begin{tabular}{lcccc}", r"\toprule", " & " + " & ".join(LABEL[a] for a in nets) + r" \\",
            " & " + " & ".join(f"{len(runs[a])} runs" for a in nets) + r" \\", r"\midrule"]
    parts = []
    for page in MASS_EFFECT_PAGES:
        body = []
        for k, blk in enumerate(page):
            body += ([r"\midrule"] if k else []) + [_full_row(1 + len(nets), MASS_EFFECT_BLOCKS[blk])]
            body += [f"{label} & " + " & ".join(p[i].strip() for p in printed) + r" \\"
                     for i, (b, label, _, _) in enumerate(rows) if b == blk]
        parts.append("\n".join(head + body + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(parts)


# ------------------------------------------------------------------ jet_pt_bins.csv (PI 2026-11-13, the jet-pT family)
# Every point of the jet-pT figures (figstyle/gallery_physics: PT, PT-part, PT-bbcc), from experiments/EVAL/perf_vs_pt.py's
# files: per bin of jet pT, the accuracy on class-balanced JetClass-II and on JetClass and the area under the ROC curve of
# X->bb against X->cc on JetClass-II, as a linear probe (jc2, jc1_probe) and fine-tuned (jc2_ft, jc1), for the label
# sets, the mass networks and the 17-class partitions (the random partitions pooled), mean +- SD over the runs. L18 (PI
# 2026-11-13): set at 8 pt the table (tab:ptbins) still ran to six turned pages, so it is a file of the data release.
PT_DIRS = ("jc2", "jc2_ft", "jc1_probe", "jc1", "jc1_fulldecay")
# the references of each block's mode (L5): the random network as a linear probe, from scratch fine-tuned, from the
# folder that holds them (JetClass from scratch on its full-schedule cells, as tab:jc1conf); none for the JetClass-II
# probes, which were not read out per bin for the random network
PT_REFS = {"jc1_probe": (INIT, "jc1_probe", "init-s"), "jc2_ft": (None, "jc2_ft", "scratch-s"),
           "jc1": (None, "jc1_fulldecay", "scratch-s")}
# (test set, quantity, perf_vs_pt folder per mode, value of a bin)
PT_BLOCKS = (("JetClass-II", "accuracy", (("lp", "jc2"), ("ft", "jc2_ft")), lambda b: b["accuracy"]),
             ("JetClass", "accuracy", (("lp", "jc1_probe"), ("ft", "jc1")), lambda b: b["accuracy"]),
             ("JetClass-II", "AUC of X->bb against X->cc", (("lp", "jc2"), ("ft", "jc2_ft")),
              lambda b: None if b["bb_vs_cc"] is None else b["bb_vs_cc"]["auc"]))
PT_MODE = {"lp": "linear probe", "ft": "fine-tuned"}
PT_CSV = "data_release/jet_pt_bins.csv"


def _release_name(name: str) -> str:
    """A network as the data release's CSVs name it (fourth review, R12): the paper's row label in plain text
    ("188 classes", "random partition 1", "no flavor boundary"), never the build's internal name; a name of the tested
    differences (TESTS_NAMES) as tab:tests prints it."""
    if name in LABEL:
        return _plain(name)
    if name in TESTS_NAMES:
        t = TESTS_NAMES[name]
        for k, n in (("Kfull", 188), ("Kres", 162), ("Kmid", 43), ("Kcoarse", 17)):
            t = t.replace(rf"\{k}{{}}", str(n))
        return t.replace("$b$", "b")
    return name


def _plain(arm) -> str:
    """A row name of the tables (L5) as plain text: "188 classes + mass", "b-quark boundary", "from scratch"."""
    if arm is None:
        return SCRATCH_LABEL
    t = LABEL[arm]
    for k, n in (("Kfull", 188), ("Kres", 162), ("Kmid", 43), ("Kcoarse", 17)):
        t = t.replace(rf"\{k}{{}}", str(n))
    return t.replace("$b$", "b")


def pt_bins_csv(perf: dict, runs: dict) -> str:
    """The data release's jet_pt_bins.csv: one row per network, mode, test set, quantity and bin of jet pT, with the
    mean, the standard deviation (ddof 1; empty for one run) and the number of runs; the networks in the L5 order
    (SETS, the flavor-boundary partitions, the pooled random partitions), then the mode's reference (PT_REFS). Values
    to six decimals."""
    lines = ["network,mode,test_set,quantity,pt_low_GeV,pt_high_GeV,mean,sd,n_runs"]
    for test_set, quantity, dirs, value in PT_BLOCKS:
        for mode, sub in dirs:
            docs = perf[sub]
            edges = {tuple(d["pt_edges"]) for d in docs.values()}
            if len(edges) != 1:
                raise SystemExit(f"FATAL: perf_vs_pt/{sub} holds {len(edges)} different pT binnings")
            e = edges.pop()
            rows = [(_plain(a), [docs[m] for m in runs[a] if m in docs]) for a in SETS + FLAV + (RAND_POOLED,)]
            if sub in PT_REFS:
                arm, folder, prefix = PT_REFS[sub]
                ref = [d for m, d in sorted(perf.get(folder, {}).items()) if m.startswith(prefix)]
                if ref and tuple(ref[0]["pt_edges"]) == tuple(e):
                    rows.append((_plain(arm), ref))
            for name, have in rows:
                for i in range(len(e) - 1):
                    v = [x for d in have if (x := value(d["bins"][i])) is not None]
                    if not v:
                        continue
                    sd = f"{np.std(v, ddof=1):.6f}" if len(v) > 1 else ""
                    lines.append(f"{name},{PT_MODE[mode]},{test_set},{quantity},{e[i]},{e[i + 1]},"
                                 f"{np.mean(v):.6f},{sd},{len(v)}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ compute (appendix, Data availability)
def _val_acc(comp: dict, m: str):
    """The validation accuracy of run m at its last pretraining epoch, the last point of the pretraining-curve
    figures (VAL, VAL-part), or None."""
    e = (comp.get(m) or {}).get("epochs") or []
    return e[-1].get("val_acc") if e else None


# PI 2026-11-22: the compute table counts every fine-tuning GPU-hour, the 15 Nov sweeps included (budget/ft_attempts.json,
# every attempt of every cell under ft_v2). An attempt's hours: its pod's running time in the Prometheus snapshots
# (budget/ft_pod_hours_prom_*.json, the largest over them) where a snapshot holds the pod's whole run, else its end less
# its start. A completed attempt's own times agree with Prometheus (-20 h on 2,142 h, 483 attempts, 22 Nov); an
# interrupted one writes nothing at its end, so before the snapshots its hours run to the next attempt's start, an upper
# bound (over 177 interrupted attempts that bound was 3.9 times their running time). An attempt with no start record
# (its ATTEMPTS lost) starts when the attempt before it was set aside, and is counted without hours when it is the first.
FT_PROM_EDGE_S = 900          # a pod first running within 15 min of a snapshot's earliest sample may have started before it
FT_SEARCH_TREES = ("lrsweep", "fixed_sweep")      # the rate searches: lrsweep, lrsweep_v2, fixed_sweep, fixed_sweep_1e4


def ft_hours(data: pathlib.Path) -> dict | None:
    """{"pre"/"scratch": Counter(GPU name: hours), "search": hours of the rate searches, "bound": hours that are upper
    bounds, "n_bound": their attempts, "n_unknown": attempts counted without hours, "n_attempts", "n_cells"} of every
    fine-tuning attempt, or None before budget/ft_attempts.json is pulled."""
    f = data / "budget" / "ft_attempts.json"
    if not f.exists():
        return None
    stamp = lambda t: calendar.timegm(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ"))     # noqa: E731
    prom, earliest = {}, None
    for snap in sorted((data / "budget").glob("ft_pod_hours_prom_*.json")):
        pods = json.loads(snap.read_text())["pods"]
        first = lambda v: calendar.timegm(time.strptime(v["first_running_utc"], "%Y-%m-%dT%H:%MZ"))      # noqa: E731
        edge = min(first(v) for v in pods.values() if v["first_running_utc"]) + FT_PROM_EDGE_S
        earliest = min(earliest or edge, edge - FT_PROM_EDGE_S)
        for pod, v in pods.items():
            if v["first_running_utc"] and first(v) > edge:
                prom[pod] = max(prom.get(pod, 0.0), v["running_hours"])
    out = {"pre": collections.Counter(), "scratch": collections.Counter(), "search": 0.0, "bound": 0.0, "n_bound": 0,
           "n_unknown": 0, "n_attempts": 0}
    cells = json.loads(f.read_text())["cells"]
    for cell, atts in cells.items():
        init = next((a["init"] for a in atts if a["init"] is not None), None)
        kind = "scratch" if (init or "").startswith("scratch") or init == "" or (init is None and "scratch" in cell) \
            else "pre"
        prev = None
        for a in atts:
            out["n_attempts"] += 1
            start, prev = a["start"] or prev, (a["end"] if a["end_kind"] == "next" else None)
            if a["pod"] in prom:
                h = prom[a["pod"]]
            elif start:
                h = (stamp(a["end"]) - stamp(start)) / 3600
                if a["end_kind"] == "next":
                    out["bound"] += h
                    out["n_bound"] += 1
            else:
                out["n_unknown"] += 1
                continue
            out[kind][gpu_name(a["gpu"]) if a["gpu"] else "GPU not recorded"] += h
            if cell.split("/")[0].startswith(FT_SEARCH_TREES):
                out["search"] += h
    out["n_cells"] = len(cells)
    # the day of the snapshots' earliest sample: every pod first running after it (FT_PROM_EDGE_S) is measured whole
    out["prom_from"] = time.strftime("%d %B %Y", time.gmtime(earliest)).lstrip("0") if earliest else None
    return out


FT_HOURS_HEAD = "fine-tuning, every attempt of every run"
FT_HOURS_ROWS = (("pre", "of the pretrained networks"), ("scratch", "from scratch"))


def table_compute(comp, runs, fmt_int, extra=(), ft_gpus=None, ft_nodev=None, ft_h=None,
                  label: str = "tab:compute") -> tuple[str, dict]:
    """Per pretraining run: its validation accuracy at the last epoch, hours on each GPU type (completed epochs,
    training and validation passes, as each epoch recorded its device), their sum, and the GPU types its fine-tuning
    cells ran on (their manifests record the device, not the hours), with a marker counting its cells whose read-out
    records no device (ft_nodev). extra: (label, model) rows beside the grid. A two-column table after it, on the next
    page, gives each network's validation accuracy at the last epoch as the pretraining-curve figures draw it, the
    mean +- SD over its runs (the five random partitions pooled), in the common row order (L5)."""
    # a longtable in landscape at the body size whose two lists of GPU types wrap inside their columns and which
    # breaks across pages; the caption is the placeholder __CAPTION__
    acc = r"\begin{tabular}[b]{@{}c@{}}validation\\accuracy\end{tabular}"
    head = rf"network & {acc} & GPU types (hours) & GPU-hours & fine-tuning GPU types \\"
    wrap = lambda cm: r">{\raggedright\arraybackslash}p{" + cm + "cm}"      # noqa: E731
    lines = [r"\begin{landscape}", "{", LT_SETUP,
             r"\begin{longtable}{@{\extracolsep{\fill}}lc" + wrap("5.4") + "r" + wrap("6.9") + "@{}}",
             rf"\caption{{__CAPTION__}}\label{{{label}}}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{Table~\ref{{{label}}} (continued)}}}} \\", r"\toprule", head,
             r"\midrule", r"\endhead"]
    # the closing rule follows the rows (below), no last foot: one taller than the empty foot lost the repeated
    # heading on a page broken in the last chunk (longtable's end-of-table branch, as _long)
    hours = {}
    arms = [a for a in ALL_ROWS if a in runs] + [a for a in runs if a not in ALL_ROWS + (INIT, INIT_BN)]   # L5 order
    # the partitions under their subheading, indented, as in the last block (round-1 fix 2026-11-13)
    items = [(arm, (r"\quad " if arm in FLAV + RAND else "") + f"{LABEL[arm]}, seed {_idx(m)}", m)
             for arm in arms for m in runs[arm]]
    items += [(None, lab, m) for lab, m in extra]
    for arm, name, m in items:
        if arm in FLAV + RAND and _full_row(5, PARTITIONS) not in lines:
            lines.append(_full_row(5, PARTITIONS))
        ftg = ", ".join(map(_whole, (ft_gpus or {}).get(m[len("mtx-"):], []))) or "--"
        if (k := (ft_nodev or {}).get(m[len("mtx-"):])):
            ftg += rf" \pending{{{k} cells without a device}}"
        if m not in comp:
            lines.append(rf"{name} & -- & \pending{{training}} & & {ftg} \\")
            continue
        h = collections.Counter()
        for e in comp[m]["epochs"]:
            h[gpu_name(e["device"])] += (e["train_seconds"] + e["val_seconds"]) / 3600
        if arm is not None:
            hours.setdefault(arm, collections.Counter()).update(h)
        types = ", ".join(_whole(f"{g} ({h[g]:.0f})") for g in sorted(h, key=lambda g: -h[g]))
        a = _val_acc(comp, m)
        lines.append(rf"{name} & {'--' if a is None else V4.tok('o', a)} & {types} & {fmt_int(round(sum(h.values())))} & {ftg} \\")
    if ft_h:                  # PI 2026-11-22: fine-tuning's GPU-hours, every attempt of every cell, under their heading
        lines.append(_full_row(5, FT_HOURS_HEAD))
        for k, name in FT_HOURS_ROWS:
            h = ft_h[k]
            types = ", ".join(_whole(f"{g} ({h[g]:.0f})") for g in sorted(h, key=lambda g: -h[g]))
            lines.append(rf"\quad {name} & -- & {types or '--'} & {fmt_int(round(sum(h.values())))} & -- \\")
    # the partition rows and their subheading never break apart, nor from the row after them (D14: a page opened on
    # "b-quark boundary, seed 2", and the kinematic-inputs row then stood alone on the last page)
    group = [i for i, ln in enumerate(lines) if ln.startswith(r"\quad ") or ln == _full_row(5, PARTITIONS)]
    for i in group[:-1] + ([group[-1]] if group and group[-1] < len(lines) - 1 else []):
        lines[i] = lines[i].rstrip() + "*"
    # the last point of the pretraining-curve figures: each network's mean over its runs, a two-column table of its own
    # on the next page (page review round 3: under the five-column heading three of its columns were empty)
    pooled = _pooled(runs)
    summary = [r"\begin{tabular}{@{}lc@{}}", r"\toprule",
               r"network & \begin{tabular}[b]{@{}c@{}}validation accuracy at the last pretraining epoch,\\mean $\pm$ "
               r"standard deviation over the runs\end{tabular} \\", r"\midrule"]
    for arm in [a for a in SETS + FLAV if a in runs] + [RAND_POOLED]:
        v = [x for m in pooled.get(arm, []) if (x := _val_acc(comp, m)) is not None]
        if not v:
            continue
        cell = V4.tok("p", float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else V4.tok("s", v[0])
        if arm in FLAV + (RAND_POOLED,) and _full_row(2, PARTITIONS) not in summary:
            summary.append(_full_row(2, PARTITIONS))
        summary.append((r"\quad " if arm in FLAV + (RAND_POOLED,) else "") + rf"{LABEL[arm]} & {cell} \\")
    summary += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(uniform_columns("\n".join(lines)).split("\n") + [
        CLOSE_RULE, r"\end{longtable}", r"\newpage", rf"\noindent\textit{{Table~\ref{{{label}}} (continued)}}\par\smallskip",
        r"\noindent" + uniform_columns("\n".join(summary)), "}", r"\end{landscape}", ""]), hours


def _stage_models(runs, extra=()) -> list:
    """The pretraining runs of tab:compute's first row: every network of table_compute's rows, extra included."""
    arms = [a for a in ALL_ROWS if a in runs] + [a for a in runs if a not in ALL_ROWS + (INIT, INIT_BN)]
    return [m for a in arms for m in runs[a]] + [m for _, m in extra]


def _pretraining_hours(comp, models) -> collections.Counter:
    """GPU-hours per GPU type of the models' completed epochs, training and validation passes, as each epoch recorded
    its device."""
    h = collections.Counter()
    for m in models:
        for e in comp.get(m, {}).get("epochs", []):
            h[gpu_name(e["device"])] += (e["train_seconds"] + e["val_seconds"]) / 3600
    return h


def _printed_hours(hh) -> tuple[str, int]:
    """A stage's GPU types with their hours as printed, most first, and its total, the sum of those printed hours, so
    that the table's numbers add up to its totals and to the text's (review of render dfe5942, A2: 3,906 + 630 against
    4,537; LEDGER CS-6)."""
    parts = [(g, round(hh[g])) for g in sorted(hh, key=lambda g: -hh[g])]
    return ", ".join(_whole(f"{g} ({h})") for g, h in parts), sum(h for _, h in parts)


def table_compute_stages(comp, runs, fmt_int, extra=(), ft_h=None) -> str:
    """tab:compute (PI 2026-11-23, LEDGER SP-9): one row per stage, its GPU types with their hours and its GPU-hours:
    pretraining, every run of every network of table_compute's rows (extra included), as each epoch recorded its device
    (training and validation passes of the completed epochs), then fine-tuning of the pretrained networks and from
    scratch, every attempt of every cell (ft_hours). A stage's GPU-hours are the sum of its printed hours
    (_printed_hours). Each run's row is in table_compute's copy, tab:computefull."""
    models = _stage_models(runs, extra)
    h = _pretraining_hours(comp, models)
    have = sum(m in comp for m in models)
    pre = f"pretraining, {have} runs" + (rf" \pending{{{len(models) - have} runs without a record}}" if have < len(models) else "")
    rows = [(pre, h)] + [(f"fine-tuning {name}", ft_h[k]) for k, name in FT_HOURS_ROWS if ft_h]
    lines = [r"\begin{tabular}{l>{\raggedright\arraybackslash}p{5.2cm}r}", r"\toprule",
             r"stage & GPU types (hours) & GPU-hours \\", r"\midrule"]
    for name, hh in rows:
        types, total = _printed_hours(hh)
        lines.append(rf"{name} & {types or '--'} & {fmt_int(total)} \\")
    if ft_h and ft_h.get("search") is not None:   # fourth review (B6): the text quotes the rate searches' share
        lines.append(rf"\quad of which choosing the learning rates &  & {fmt_int(round(ft_h['search']))} \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def compute_macros(em, comp, runs, hours, index, fmt_int, extra=()) -> None:
    total = collections.Counter()
    for arm in MAIN:
        total.update(hours.get(arm, {}))
    top = max(total, key=total.get) if total else None
    shares = [hours[a][top] / sum(hours[a].values()) for a in MAIN if a in hours] if top else []
    paper = round(sum(h for arm in hours.values() for h in arm.values()))
    em.macro("ComputeTotalGpuHours", fmt_int(paper), index, "GPU-hours of every reported pretraining run, training and "
             "validation passes")
    # tab:compute's pretraining row as printed, the kinematic-only model included (extra; _printed_hours)
    allh = _printed_hours(_pretraining_hours(comp, _stage_models(runs, extra)))[1]
    em.macro("MlstComputeTotalGpuHours", fmt_int(allh), index,
             "GPU-hours of every pretraining run the paper reports, the kinematic-only model included, the sum of "
             "tab:compute's printed hours")
    em.macro("ComputeMainGpu", top or r"\pending{compute}", index, "GPU type with the most hours over the four label sets")
    em.macro("ComputeMainGpuShareMin", f"{100 * min(shares):.0f}\\%" if shares else r"\pending{compute}", index,
             "lowest share of a label set's GPU-hours on that type")
    em.macro("ComputeMainGpuShareMax", f"{100 * max(shares):.0f}\\%" if shares else r"\pending{compute}", index,
             "highest share of a label set's GPU-hours on that type")
    paper_runs = set(_pretrained(runs))
    em.macro("ComputeRunsDone", str(len(paper_runs & set(comp))), index, "paper runs with a compute record (count)")
    types = sorted({gpu_name(e["device"]) for m, c in comp.items() if m in paper_runs or m == KIN_MODEL for e in c["epochs"]})
    t = ", ".join(types[:-1]) + " and " + types[-1] if len(types) > 1 else (types[0] if types else r"\pending{compute}")
    em.macro("ComputeGpuTypes", t, index, "GPU types any finished paper run trained on")
    em.macro("MlstComputeGpuTypes", t, index, "GPU types of the pretraining runs the paper reports")


def ft_hours_macros(em, fth, index, fmt_int) -> None:
    """The fine-tuning GPU-hours of tab:compute's last rows, for the text: the total, the rate searches' share, the
    hours that are upper bounds and their attempts, and the attempts counted without hours."""
    pend = r"\pending{fine-tuning GPU-hours}"
    tot = sum(_printed_hours(fth[k])[1] for k, _ in FT_HOURS_ROWS) if fth else None     # tab:compute's rows as printed
    for name, v, what in (("MlstFtGpuHours", tot, "GPU-hours of every fine-tuning attempt, the rate searches included"),
                          ("MlstFtGpuHoursSearch", fth and round(fth["search"]), "of them, the rate searches'"),
                          ("MlstFtGpuHoursBound", fth and round(fth["bound"]), "of them, upper bounds (attempts "
                           "interrupted before the Prometheus snapshots, to the next attempt's start)"),
                          ("MlstFtAttemptsBound", fth and fth["n_bound"], "those attempts (count)"),
                          ("MlstFtAttemptsUnknown", fth and fth["n_unknown"], "attempts with no start record, counted "
                           "without hours"),
                          ("MlstFtAttempts", fth and fth["n_attempts"], "fine-tuning attempts in all (count)"),
                          ("MlstFtCells", fth and fth["n_cells"], "fine-tuning cells in all (count)")):
        em.macro(name, pend if v is None else fmt_int(v), index, what)
    em.macro("MlstFtPromFrom", (fth or {}).get("prom_from") or pend, index,
             "the first day the monitoring snapshots hold every fine-tuning attempt's running time")


# ------------------------------------------------------------------ per-run values (A1)
def table_per_run(fits, per, runs, fmt_pm, fill=None, chosen=None) -> str:
    """tab:perrun: every run's own value at each task's largest training set, in the headline family's five panels
    (SUMMARY_GRID, read as tab:summary reads them: JetClass-II and JetClass accuracy, R30 on top, quark/gluon Pythia
    and Herwig), as a linear probe and fine-tuned. Rows in L5's order with the partitions under their subheading, the
    random network's initializations (probes only), from scratch (fine-tuned only) and the kinematic-only network
    (top only). Each row names its seed (the glossary: "seed", never "run k")."""
    fill, chosen = fill or {}, chosen or {}
    grid = SUMMARY_GRID
    gen = {"qg": "Pythia", "qg_herwig": "Herwig"}
    task = {"jc2": "JetClass-II", "jc1": "JetClass", "top": "top"}
    nk = len(grid)
    qg = [i for i, (k, *_) in enumerate(grid) if k.startswith("qg")]
    # four short header lines (task, generator, metric), so that the ten columns fit a turned page at the body size
    tasks = " & ".join([task[k] for k, *_ in grid if k in task] + [rf"\multicolumn{{{len(qg)}}}{{c}}{{quark/gluon}}"])
    gens = " & ".join(gen.get(k, "") for k, *_ in grid)
    mets = " & ".join(SUMMARY_HEAD[met] for *_, met in grid)
    rule = "".join(rf"\cmidrule(lr){{{2 + o + qg[0]}-{2 + o + qg[-1]}}}" for o in (0, nk))
    head = (rf" & \multicolumn{{{nk}}}{{c}}{{linear probe on the frozen features}} & \multicolumn{{{nk}}}{{c}}{{fine-tuned}} \\" "\n"
            rf"\cmidrule(lr){{2-{1 + nk}}}\cmidrule(lr){{{2 + nk}-{1 + 2 * nk}}}" "\n"
            f" & {tasks} & {tasks} \\\\\n{rule}\n & {gens} & {gens} \\\\\n & {mets} & {mets} \\\\")
    one = lambda x: "--" if x is None else f"{x:.4f}" if abs(x) < 1 else _tab1(x)   # accuracy, AUC: four decimals
    probe = [summary_probe(fits, fill, chosen, ds, met) for _, _, ds, met in grid]
    tuned = [summary_ft(per, k, n, met) for k, n, _, met in grid] if per else [None] * nk
    rows = []

    def row(name, m, lp=True, ft=True):
        cells = [one(g(m) if g and lp else None) for g in probe] + [one(g(m) if g and ft else None) for g in tuned]
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    for a in SETS:
        first = len(rows)
        for m in runs[a]:
            row(f"{LABEL[a]}, seed {_idx(m)}", m)
        rows[first:-1] = [r + "*" for r in rows[first:-1]]      # a network's runs stay on one page
    at = len(rows)
    rows.append(_full_row(1 + 2 * nk, PARTITIONS))
    for a in FLAV + RAND:
        for m in runs[a]:
            row(rf"\quad {LABEL[a]}, seed {_idx(m)}", m)
    rows[at:-1] = [r + "*" for r in rows[at:-1]]      # the partitions stay on one page under their subheading
    for m in runs.get(INIT, []):
        row(f"{LABEL[INIT]}, initialization {_idx(m)}", m, ft=False)
    if per:
        sc = [summary_scratch(per, k, n, met) for k, n, _, met in grid]
        for s_ in range(max(map(len, sc), default=0)):
            rows.append(f"{SCRATCH_LABEL}, seed {s_ + 1} & " + " & ".join(["--"] * nk) + " & "
                        + " & ".join(one(v[s_] if s_ < len(v) else None) for v in sc) + r" \\")
    kt = _bench_rec(fits, fill, chosen, KIN_MODEL, "top")
    kf = _ftv(per, "top", KIN_MODEL, _sizes(per, "top")[-1], "r30") if per and _sizes(per, "top") else None
    rows.append(f"{LABEL['KIN']} & " + " & ".join(one(_rej(kt, "r30")) if k == "top" else "--" for k, *_ in grid) + " & "
                + " & ".join(one(kf) if k == "top" else "--" for k, *_ in grid) + r" \\")
    # in landscape across the whole line at the body size, the row labels on one line (PI 2026-11-11)
    return "\n".join([r"\begin{landscape}", "{", LT_SETUP,
                      r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * 2 * nk + "@{}}",
                      r"\caption{__CAPTION__}\label{tab:perrun}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
                      rf"\multicolumn{{{1 + 2 * nk}}}{{@{{}}l}}{{\textit{{Table~\ref{{tab:perrun}} (continued)}}}} \\", r"\toprule",
                      head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", "}",
                      r"\end{landscape}", ""])


# ------------------------------------------------------------------ captions and assembly
SPREAD = ("Mean $\\pm$ SD over the \\MlstNRunsWord{} pretraining runs of each level of the class "
          "hierarchy; configurations with \\RandFlavRuns{} runs: over those. Every random partition has 17 "
          "classes; each numbered one (1--5) has a single run, whose pretraining seed its row "
          "names. The random network without pretraining: three initializations.")
SPREAD3 = "Mean $\\pm$ SD over the \\MlstNRunsWord{} pretraining runs."
# the paper's tables (PI 2026-11-12): the five random partitions as one row
# PI 2026-11-21, the one description of the random partitions, everywhere
RAND_WORDS = "the 161 resonant classes split at random into 16 groups, with QCD kept as the 17th"
POOLED = ("Random partitions (" + RAND_WORDS + "): mean $\\pm$ SD over the five partitions, one run each.")
SPREAD_POOLED = ("Mean $\\pm$ SD over the \\MlstNRunsWord{} pretraining runs of each level of the class hierarchy; "
                 "configurations with \\RandFlavRuns{} runs: over those. " + POOLED + " The random network "
                 "without pretraining: three initializations.")
SPREAD_MASS = (SPREAD3[:-1] + "; the models with the mass regression have \\MlstMassNRunsWord{} runs, and both "
               "values are given. The random network without pretraining: three initializations.")
RATIO_COL = ("The ratio column is $\\rho$ of equation~(\\ref{eq:ratio}), $(1-\\mathrm{AUC})$ over that of the "
             "\\Kfull-class run of the same seed (a random partition: of its seed).")
AUC_DEF = "AUC: area under the curve of signal efficiency against background efficiency."
KIN_NOTE = ("The \\Kfull-class model with kinematic inputs only, a diagnostic of the missing inputs, has one run and "
            "no input missing, and is evaluated on top tagging only.")
# the reference row of the frozen-probe tables; the differences are the verified ones (PI 2026-11-08): the stream and its
# loader (pretrain_v2.py: a fresh seeded stream every epoch), the initialization, the checkpoint rule (best_window_epoch)
SOPHON_NOTE = ("Below the rule, the released Sophon checkpoint~\\cite{sophon}, probed in the same way, lies outside the "
               "controlled comparison: it was pretrained on another stream of training jets, read by weaver's loader (our "
               "runs draw a fresh seeded stream every epoch), from its own initialization, and selected by its own "
               "checkpoint rule (ours: the best validation epoch of the last ten).")
# the top and quark/gluon claims of the tables that print each metric (PI 2026-11-21), from the paired test-jet bootstrap
# of every printed network and metric on the v5 probes (bootstrap/l3_v5/*.all.json, |delta| > 3 sigma of the runs and the
# test jets); gallery_labelset.TOP_QG_PROBE_CLAIMS holds them, and tests/test_mlst_results.py checks both
# PI 2026-11-22: tab:probes now holds the 17-class partitions, so its sentence covers them (bootstrap/l3_v5
# probe_top/qg .all files hold every partition's difference; TOP_QG_PROBE_CLAIMS: rand-188 r50 below on top)
TOPQG_PROBE_SET = (" Beyond three standard deviations of the runs and the test jets, \\Kcoarse{} classes + mass is more accurate "
                   "than \\Kfull{} classes on top tagging, with a higher $R_{50}$, and the random partitions have a lower "
                   "$R_{50}$ on top tagging; no other label set, mass network or 17-class partition differs from "
                   "\\Kfull{} classes in these metrics, nor a mass network from its own label set.")
TOPQG_FT_SET = (" Beyond three standard deviations of the runs and the test jets, no pretrained network fine-tuned here differs from "
                "\\Kfull{} classes in these metrics, nor a mass network from its own label set.")
TOPQG_PROBE_FULL = (" Beyond three standard deviations of the runs and the test jets, on top tagging \\Kcoarse{} classes + mass "
                    "has a higher accuracy, $R_{50}$ and AUC than \\Kfull{} classes and the random partitions, taken "
                    "together, a lower $R_{50}$ and AUC, and on Pythia quark/gluon tagging \\Kcoarse{} classes + mass has a lower "
                    "AUC; no other difference from \\Kfull{} classes, or of a mass network from its own label set, "
                    "passes in these metrics.")
SOPHON_SHORT = "Below the rule, the released Sophon checkpoint, outside the controlled comparison (table~\\ref{tab:probes})."


EE_NOTE = (" The pair $X\\to ee$ vs $\\mu\\mu$ is not listed: every pretrained network whose label set keeps the two "
           "classes apart separates them completely, as a linear probe and fine-tuned.")

CAPTIONS = {
    "probes": ("tab:probes", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls (rows), "
               "trained and scored on each task at its largest training set. "
               "Upper part: JetClass-II (class-balanced test set) accuracy and AUC averaged over its \\Kfull{} classes "
               "each against the rest, over the classes with the full \\JcBalTestPerClass{} test jets, over the resonant "
               "and over the QCD classes; JetClass accuracy and AUC. Lower part: top and quark/gluon "
               "tagging (also on Herwig jets) with the missing inputs substituted as chosen on validation for "
               "every model (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon; column headings): "
               "accuracy, AUC and the background rejections $R_{50}$ and $R_{30}$, $1/\\epsB$ at 50\\% and 30\\% signal "
               "efficiency. "
               + SPREAD_POOLED
               + " " + KIN_NOTE + " " + SOPHON_NOTE + " " + AUC_DEF + " $^{*}$: a probe that did not converge."),
    "probes_full": ("tab:probesfull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls (rows, every random partition), "
               "trained and scored on each task at its largest training set. "
               "Upper part: JetClass-II (class-balanced test set) accuracy and AUC averaged over its \\Kfull{} classes "
               "each against the rest, over the classes with the full \\JcBalTestPerClass{} test jets, over the resonant "
               "and over the QCD classes; JetClass accuracy and AUC. Lower part: top and quark/gluon "
               "tagging (also on Herwig jets) with the missing inputs substituted as chosen on validation for "
               "every model (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon; column headings): "
               "accuracy, AUC and the background rejections $R_{50}$ and $R_{30}$, $1/\\epsB$ at 50\\% and 30\\% signal "
               "efficiency. "
               + RATIO_COL + " " + SPREAD
               + TOPQG_PROBE_FULL + " " + KIN_NOTE + " " + SOPHON_NOTE + " " + AUC_DEF + " $^{*}$: a probe that did not converge."),
    "probes_n": ("tab:probesn", "The training-size points of the linear probes below each task's largest training set."),
    "ft": ("tab:ft", "The training-size points of the fine-tuned networks below each task's full training set."),
    "probes_n_full": ("tab:probesnfull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained on "
                 "each task with the number of jets in the column headers: class-balanced "
                 "JetClass-II AUC averaged over the classes, JetClass accuracy, and $R_{30}$ on top, quark/gluon and Herwig jets with "
                 "the missing inputs substituted as chosen on validation (\\LpSubstitutionTop{} for top, "
                 "\\LpSubstitutionQg{} for quark/gluon). " + SPREAD + " Here the \\RandVtwoNPartitions{} random "
                 "partitions are pooled into one row, mean $\\pm$ SD over their runs. " + KIN_NOTE
                 + " $^{*}$: a probe that did not converge."),
    # PI 2026-11-23 (LEDGER DL-R #4): the AUCs behind the main text's error ratios on these pairs (CS-6)
    "pair_auc": ("tab:pairauc", "The area under the receiver operating characteristic curve (AUC) of binary linear "
                 "probes on the frozen features separating pairs of JetClass-II classes."),
    "pairs": ("tab:pairs", "Pairs of JetClass-II classes separated by networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and by the controls: "
              "a binary linear probe on the frozen features, trained on up to \\JcBalPairTrainPerSide{} jets of each "
              "side from held-out JetClass-II files and scored on up to \\JcBalPairTestPerSide{} of each side from the "
              "JetClass-II test files, each side in its natural class mix; "
              "the network fine-tuned on $10^{6}$ class-balanced JetClass-II jets with the \\Kfull{} classes, "
              "scored on the pair's jets among \\LpNTestJcii{} JetClass-II test jets of the mixed test sample by the "
              "summed output scores of the first side over those of both sides. The two modes are scored on different "
              "test jets, so they are compared within a mode only. "
              "$R_{90}$, one over the fraction of background jets kept when the cut keeps 90\\% of the signal jets with the first side as signal. "
              "No pretraining: fine-tuned from scratch, three seeds, no probe. $^{\\dagger}$: the label set merges the two "
              "sides into one class; $^{\\ddagger}$: it merges the first side with part of the second; $^{\\parallel}$: some of the five random partitions merge the two sides. " + SPREAD_POOLED
              + " The $b$-quark boundary and the random flavor boundary lie only in the four-quark decays; every label "
              "set but the \\Kfull-class one merges all QCD classes. The random network without pretraining, as initialized and "
              "with its BatchNorm statistics recomputed, has no label set, and no fine-tuning." + EE_NOTE + " A mark applies to "
              "the configuration."),
    "pairs_full": ("tab:pairsfull", "Pairs of JetClass-II classes separated by networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and by the controls, every random partition: "
              "a binary linear probe on the frozen features, trained on up to \\JcBalPairTrainPerSide{} jets of each "
              "side from held-out JetClass-II files and scored on up to \\JcBalPairTestPerSide{} of each side from the "
              "JetClass-II test files, each side in its natural class mix; "
              "the network fine-tuned on $10^{6}$ class-balanced JetClass-II jets with the \\Kfull{} classes, "
              "scored on the pair's jets among \\LpNTestJcii{} JetClass-II test jets of the mixed test sample by the "
              "summed output scores of the first side over those of both sides. The two modes are scored on different "
              "test jets, so they are compared within a mode only. "
              "$R_{90}$, one over the fraction of background jets kept when the cut keeps 90\\% of the signal jets with the first side as signal. "
              "No pretraining: fine-tuned from scratch, three seeds, no probe. $^{\\dagger}$: the label set merges the two "
              "sides into one class; $^{\\ddagger}$: it merges the first side with part of the second. " + SPREAD
              + " The $b$-quark boundary and the random flavor boundary lie only in the four-quark decays; every label "
              "set but the \\Kfull-class one merges all QCD classes. The random network without pretraining, as initialized and "
              "with its BatchNorm statistics recomputed, has no label set and no fine-tuning. The $X\\to ee$ vs $\\mu\\mu$ probe is trained on the "
              "jets of the two classes in the class-balanced training set (\\MlstPairEemmNTrain{} jets) and scored on every "
              "JetClass-II test jet of the two classes (\\MlstPairEemmNTestPerSide{} per side). A mark applies to the "
              "configuration."),
    "mlp": ("tab:mlp", "Nonlinear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained and scored "
            "on each task (a multilayer perceptron with one hidden layer of \\LpMlpHidden{} units, "
            "\\LpMlpSeedsWord{} seeds averaged) at the largest training set: class-balanced JetClass-II AUC (all, resonant and QCD "
            "classes) with the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run of the same seed, JetClass AUC, and AUC and $R_{30}$ "
            "on top, quark/gluon and Herwig jets (missing inputs set to zero); then the class pairs of "
            "table~\\ref{tab:pairs}, class-balanced, except $X\\to ee$ vs $\\mu\\mu$, which has no nonlinear probe. "
            "The models with the mass regression and the numbered random partitions have no nonlinear probe on the "
            "class-balanced test set: they show a dash in its columns and are left out of the pair blocks; the next block gives "
            "them on the mixed test sample beside the levels of the hierarchy on that set, and no $\\rho$ "
            "crosses the two test sets. Last, the linear probe on the 512-unit hidden layer of the \\Kfull-class models' "
            "pretraining head beside the one on the class token, with the missing inputs also set to zero. " + SPREAD + " " + KIN_NOTE),
    "coarse": ("tab:coarse", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
               "and 17 classes and of the controls, trained on "
               "one million class-balanced JetClass-II jets and scored on its class-balanced test set, with the "
               "classes merged into coarser targets: the 43 and the 17 groups of the class hierarchy, and the 17 "
               "groups of the partition with no flavor boundary. Accuracy and the area under the curve averaged over "
               "the target's groups; the test set is balanced over the 188 JetClass-II classes, not over the groups. "
               + SPREAD_POOLED + " " + SOPHON_SHORT),
    "coarse_full": ("tab:coarsefull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
               "and 17 classes and of the controls (every random partition), trained on "
               "one million class-balanced JetClass-II jets and scored on its class-balanced test set, with the "
               "classes merged into coarser targets: the 43 and the 17 groups of the class hierarchy, and the 17 "
               "groups of the partition with no flavor boundary. Accuracy and the area under the curve averaged over "
               "the target's groups; the test set is balanced over the 188 JetClass-II classes, not over the groups. "
               + SPREAD + " " + SOPHON_SHORT),
    "qcdhf": ("tab:qcdhf", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
              "and 17 classes and of the controls, trained on "
              "one million class-balanced JetClass-II jets and scored on its class-balanced test set: background "
              "rejection $R_{50}$ of Sophon's $X\\to b\\bar b$ discriminant, the $X\\to b\\bar b$ probability over "
              "itself plus the summed QCD probabilities, at 50\\% $X\\to b\\bar b$ efficiency, for QCD jets with "
              "two or more $b$ quarks, with one $b$ quark, and for all QCD jets, each subclass weighted to its share of "
              "the natural test set. QCD jets with charm and no $b$ quark, and light QCD jets, are not shown: at most "
              "__QCDHF_C__ and __QCDHF_LIGHT__ of them pass in any run of a pretrained network. $^\\dagger$: fewer than \\MlstMinBkgPass{} "
              "jets pass in a run. " + SPREAD_POOLED),
    "qcdhf_full": ("tab:qcdhffull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 "
              "and 17 classes and of the controls (every random partition), trained on "
              "one million class-balanced JetClass-II jets and scored on its class-balanced test set: background "
              "rejection $R_{50}$ of Sophon's $X\\to b\\bar b$ discriminant, the $X\\to b\\bar b$ probability over "
              "itself plus the summed QCD probabilities, at 50\\% $X\\to b\\bar b$ efficiency, for QCD jets with "
              "two or more $b$ quarks, with one $b$ quark, and for all QCD jets, each subclass weighted to its share of "
              "the natural test set. QCD jets with charm and no $b$ quark, and light QCD jets, are not shown: at most "
              "__QCDHF_C__ and __QCDHF_LIGHT__ of them pass in any run of a pretrained network. $^\\dagger$: fewer than \\MlstMinBkgPass{} "
              "jets pass in a run. " + SPREAD),
    "bn": ("tab:bn", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained and scored on "
           "JetClass-II (natural composition, largest training set), with the BatchNorm "
           "statistics as accumulated in pretraining and recomputed on training jets: for each, the AUC averaged over the classes "
           "and beside it the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run of the same seed. The class-balanced probes of the "
           "pretrained networks use their checkpoints as trained, so this check uses the mixed test sample. " + SPREAD),
    "balance": ("tab:balance", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained on "
                "$10^{6}$ JetClass-II jets and scored on the mixed test sample "
                "and on the class-balanced test set: AUC averaged over the classes and the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to "
                "the \\Kfull-class run of the same seed. " + SPREAD),
    "jc1_class_auc": ("tab:jc1_class_auc", "Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and the "
                      "controls, trained on one million JetClass jets: the recall of each JetClass class, the fraction "
                      "of its test jets the network assigns to it. Upper part: linear probes on the frozen features, "
                      "beside the random network without pretraining. Lower part: fine-tuned, beside the network "
                      "trained from scratch. " + SPREAD + " From scratch: three training seeds."),
    "fill_main": ("tab:fill_main", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with the inputs a "
                  "data set lacks set to zero, to the training mean or to random draws of a real particle, trained at the "
                  "largest training set. Upper part: relative "
                  "drop in JetClass-II accuracy (percent) when the slots the top or the quark/gluon data lack are "
                  "substituted, against the same model with every input. Lower part: $R_{30}$ on top and quark/gluon "
                  "tagging. Mean and SD over three runs. The random network with BatchNorm recomputed, three "
                  "initializations, has its batch-normalization statistics re-estimated on \\DesignBnJets{} training "
                  "jets. The \\Kfull-class model with kinematic inputs only has nothing missing (one run). The controls "
                  "are in table~\\ref{tab:fill}."),
    "fill": ("tab:fill", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, "
             "trained at the largest training set: $R_{30}$ on top and quark/gluon jets with the inputs a data set "
             "lacks (\\FillTopNZero{} for top, \\FillQgNZero{} for quark/gluon) set to zero, to the training mean or "
             "to random draws of a real particle; the model with kinematic inputs only has nothing missing. The "
             "substitution chosen on validation is \\LpSubstitutionTop{} for top and \\LpSubstitutionQg{} for "
             "quark/gluon. The relative drop in JetClass-II accuracy under each substitution is in "
             "table~\\ref{tab:fillfull}. " + SPREAD_POOLED),
    "fill_full": ("tab:fillfull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, with the three "
             "substitutions of the missing inputs (zero, the training mean, random draws of a "
             "real particle), trained at the largest training set. Upper part: relative drop in "
             "JetClass-II accuracy (percent) when the slots the top (\\FillTopNZero{}) or the quark/gluon "
             "(\\FillQgNZero{}) data lack are substituted, against the same model with every input. Lower part: "
             "$R_{30}$ of the linear probe on top and quark/gluon jets under each substitution; the model with "
             "kinematic inputs only has nothing missing. The substitution chosen on validation is "
             "\\LpSubstitutionTop{} for top and \\LpSubstitutionQg{} for quark/gluon. " + SPREAD),
    "mass": ("tab:mass", "Jet mass from frozen features: a multilayer perceptron, trained on JetClass-II, regresses "
             "the mass from the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes (two of them also with a mass regression) and of the random "
             "network without pretraining; the soft-drop mass is the reference. Resolution: $\\sigma_r$ of "
             "equation~(\\ref{eq:resolution}), half the 15.9--84.1\\% quantile range of the response "
             "$r = m/m_{\\mathrm{gen}}$, a jet's mass over its generator-level soft-drop mass, divided by its median; "
             "and the median response, over all JetClass-II test jets with a generator match (resonant and QCD jets "
             "separately in table~\\ref{tab:massdetail}); lower resolution is better. The regression target is "
             "centered on one training median, with no class information, as for the soft-drop mass, whose SD is the "
             "bootstrap over test jets. A multilayer perceptron is used because the mass is not linear in the features "
             "(ridge regression in table~\\ref{tab:massdetail}). The random network without pretraining shows what a "
             "network that never trained exposes. The mass regression in pretraining has the weight "
             "$\\lambda = \\MlstMassLambda{}$ at both label sets, a share of the training loss of "
             "\\MlstMassLossShareResMass\\% (\\Kres{} classes) and \\MlstMassLossShareCoarseMass\\% (\\Kcoarse{} "
             "classes). " + SPREAD_MASS),
    "mass_detail": ("tab:massdetail", "Jet-mass regression on JetClass-II from networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, in detail, over all "
                    "test jets and over resonant and QCD jets. Upper part: resolution and median response of the "
                    "regression output of the models pretrained with the mass regression, and of ridge and multilayer "
                    "perceptron regressions on frozen features. Lower parts: resolution and median response of the "
                    "soft-drop mass and of the regression output against jet $p_{\\mathrm{T}}$ and against the "
                    "generator-level soft-drop mass; __BLANK__ The $p_{\\mathrm{T}}$ bins hold every jet, also those with a "
                    "generator-level mass below 20~GeV, whose response is very broad: hence a soft-drop resolution of QCD "
                    "jets above 1 at high $p_{\\mathrm{T}}$; the generator-mass bins span 20--500~GeV. " + SPREAD_MASS),
    "sculpt_main": ("tab:sculptmain", "Mass sculpting by the pretraining heads used directly, beside the random-selection floor."),
    "sculpt": ("tab:sculpt", "Mass sculpting by the output scores of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with 162 and 17 classes plus a mass regression__SCULPT_FLAV__, on the JetClass-II test jets: Jensen--Shannon distance (base 2) $\\times10^{2}$ between the "
               "$m_{\\mathrm{SD}}$ distributions (__SCULPT_BINS__) of all QCD test jets and of those passing a cut on "
               "$1-g_{\\mathrm{QCD}}$ at the QCD efficiency $\\epsB$ given; lower is less sculpting.__SCULPT_SIG__ The last row is the "
               "distance expected for a random selection of as many of the QCD test jets (drawn without replacement), with its SD "
               "over random selections; at 50\\% signal efficiency, where each network's cut passes its own number of QCD jets, "
               "that distance is given for each row. " + SPREAD_MASS),
    "sculpt_boot": ("tab:sculptboot", "Mass sculpting by the pretraining heads, differences between label sets: "
                    "$\\Delta$, the Jensen--Shannon distance of table~\\ref{tab:sculpt} ($\\times10^{2}$) of the "
                    "configuration minus that of the reference, paired by pretraining seed and averaged over the seeds "
                    "both have; the SD of the paired differences over the seeds (run SD); the SD "
                    "of $\\Delta$ over __REPLICAS__ bootstrap replicas of the test jets, the QCD and the "
                    "resonant jets each resampled with replacement, the same jets for every network (bootstrap SD); "
                    "and $\\Delta/\\sigma$, $\\sigma^{2}$ the sum of the two squares. At the QCD efficiencies $\\epsB$ of table~\\ref{tab:sculpt} and at the "
                    "cut keeping half of the resonant jets ($\\epsS = 50\\%$)."),
    "ft_full": ("tab:ftfull", "Fine-tuning of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, and the network trained from scratch, on each "
           "task with the number of training jets in the column headers, in each task's metric: on class-balanced "
           "JetClass-II the AUC averaged over the classes (all classes and the QCD classes) and the accuracy over its "
           "\\Kfull{} classes, accuracy on JetClass, and $R_{30}$ on top, quark/gluon and Herwig jets. " + SPREAD + " Each pretrained run is "
           "fine-tuned once; from scratch: \\MlstFtRefSeedsWord{} training seeds on the same jets as the pretrained "
           "models. The last column of "
           "the JetClass-II block is $\\rho$ of equation~(\\ref{eq:ratio}) at $10^{6}$ training jets, $1-\\mathrm{AUC}$ "
           "over that of the \\Kfull-class run of the same seed. A dash: not run at that size (the random partitions "
           "are fine-tuned on JetClass-II at $10^{6}$ jets only). " + KIN_NOTE),
    "ft_rej": ("tab:ftrej", "Fine-tuning of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, and the network trained from scratch, "
               "on the full training sets of top and quark/gluon (also scored on Herwig jets): accuracy, AUC and the background rejections $R_{50}$ and $R_{30}$, $1/\\epsB$ at "
               "50\\% and 30\\% signal efficiency. " + SPREAD + " From scratch: three training seeds. Every rejection rests on at least __FEWEST__ background jets passing the cut."),
    "bench_top": ("tab:benchtop", "Top tagging: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 "
                  "classes and fine-tuned, or trained from scratch, on the full top training set (\\LitPartTopNTrainMillion{} million "
                  "jets): accuracy, AUC and the background rejections $R_{50}$ and $R_{30}$, $1/\\epsB$ at 50\\% and 30\\% signal "
                  "efficiency, and the number of trainable parameters. "
                  "Published rows from the tables cited, the parameters from each model's paper (ParT and ParticleNet-f.t.: "
                  "ParT's JetClass models; linear EFP: a linear classifier on energy flow polynomials); f.t.\\ marks a "
                  "model pretrained on JetClass and fine-tuned. Ours: \\MlstNParamsFt{} trainable parameters, "
                  "\\MlstNParamsFtKin{} with kinematic inputs only. Our "
                  "pretrained models see the \\FillTopNZero{} inputs the top data lack set to zero; the model with "
                  "kinematic inputs only, a diagnostic of that mismatch, has none missing (one run). Uncertainties: "
                  "ours, the SD over the pretraining runs, one fine-tuning each, and over three seeds "
                  "from scratch; ParT, the spread over \\LitPartFtHeadReinits{} fine-tunings of one pretrained model "
                  "with different output-layer initializations, quoted for the run of median accuracy; L-GATr and "
                  "OmniLearned, the SD of five trainings; the other published uncertainties as given there."),
    "bench_qg": ("tab:benchqg", "Quark/gluon tagging: Sophon's network pretrained on JetClass-II with 188, 162, 43 "
                 "and 17 classes and fine-tuned, or trained from scratch, on the full training set (\\LitQgTrainMillion{} million "
                 "Pythia jets) with the particle type as five flags and the charge (ParT's exp inputs), "
                 "scored on Pythia jets (Herwig jets in table~\\ref{tab:benchqgherwig}): accuracy, AUC and $1/\\epsB$ at 50\\% and 30\\% quark "
                 "efficiency, and the number of trainable parameters. Our models see the \\FillQgNZero{} displacement "
                 "inputs set to zero. Published rows (exp inputs) from the tables cited, the parameters from each model's "
                 "paper as in table~\\ref{tab:benchtop}; constituent multiplicity and linear EFP (a linear classifier on "
                 "energy flow polynomials) use no particle type; PFN-Ex's uncertainty is half the interquartile range of ten trainings, "
                 "the others as in table~\\ref{tab:benchtop}, with nine trainings for OmniLearned."),
    "bench_qg_herwig": ("tab:benchqgherwig", "Quark/gluon tagging scored on Herwig jets: Sophon's network pretrained on "
                        "JetClass-II with 188, 162, 43 and 17 classes and fine-tuned, or trained from scratch, on the full "
                        "Pythia training set as in table~\\ref{tab:benchqg}: accuracy, AUC and $1/\\epsB$ at 50\\% and 30\\% "
                        "quark efficiency. Mean $\\pm$ SD over the pretraining runs, one fine-tuning each; three seeds "
                        "from scratch."),
    "boot": ("tab:boot", "Linear probes on the frozen features and fine-tuning of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes: uncertainty of the "
             "comparisons between their label sets at the largest training set, in percent: the "
             "standard deviation of the logarithm of the ratio over the pretraining runs, then that from the finite "
             "test sample. JetClass-II: the ratio of $1-\\mathrm{AUC}$, the test-sample part from a bootstrap of the "
             "__JC2COMP__ test jets (\\MlstBootResamples{} resamples within each class). Top and quark/gluon: the ratio of "
             "$R_{30}$, the test-sample part from the number of background jets passing the cut."),
    "jetclass": ("tab:jetclass", "JetClass: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 "
                 "classes and fine-tuned, or trained from scratch, on one million JetClass jets, in the form of ParT's Table~1: "
                 "the number of trainable parameters, accuracy and AUC averaged over all pairs of "
                 "classes (one against one, ParT's convention; on JetClass's equal class sizes it equals the "
                 "one-against-rest average, checked). Published rows: trained from scratch on 2 million jets and scored on "
                 "ParT's \\LitPartJcTestMillion-million-jet test set (ParT on 100 million jets reaches __PART100__); "
                 "ours scored on two million; parameters as in "
                 "table~\\ref{tab:benchtop}, ours \\MlstNParamsJc{}. "
                 + SPREAD3 + " From scratch: three seeds. The per-class rejections are in table~\\ref{tab:jetclassrej}."),
    "jetclass_rej": ("tab:jetclassrej", "JetClass, the networks pretrained with 188, 162, 43 and 17 classes and fine-tuned, or trained from scratch, on one million jets, whose accuracy and AUC are in tables~\\ref{tab:summary} and~\\ref{tab:summaryauc}: for each signal class the "
                     "rejection of QCD jets at 50\\% signal efficiency (__WPHQQL__\\% for $H\\to\\ell\\nu qq'$, __WPTBL__\\% "
                     "for $t\\to b\\ell\\nu$) on $D_S = g_S/(g_S+g_{\\mathrm{QCD}})$ of equation~(\\ref{eq:discr}), beside "
                     "the published rows trained on 2 million jets. " + SPREAD3 + " From scratch: three seeds. "
                     "$^\\dagger$: fewer than $\\nmin$ QCD jets pass in at least one run (equation~(\\ref{eq:cut})), so that "
                     "run's rejection has a counting uncertainty above __MINBKGPC__\\%."),
    "jetclass_all": ("tab:jetclassall", "JetClass fine-tuning as tables~\\ref{tab:published} and~\\ref{tab:jetclassrej} of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the "
                     "controls. "
                     + SPREAD + " The row of all five random partitions gives the mean $\\pm$ SD over "
                     "them. $^\\dagger$ as in table~\\ref{tab:jetclassrej}; a mark on a configuration with two runs: in at least one of them."),
    "jetclass_probe": ("tab:jetclassprobe", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, "
                       "trained and scored on JetClass, largest training set, in "
                       "the columns of table~\\ref{tab:jetclassall} (AUC as there: on JetClass's equal class sizes the one-against-one and one-against-rest averages coincide). "
                       + SPREAD + " " + SOPHON_SHORT + " $^\\dagger$ as there."),
    "jetclass_nbkg": ("tab:jetclassnbkg", "QCD jets passing each signal class's working point after fine-tuning networks pretrained on JetClass-II with 188, 162, 43 and 17 classes "
                      "and the controls on JetClass (50\\% signal efficiency; 99\\% for "
                      "$H\\to\\ell\\nu qq'$, 99.5\\% for $t\\to b\\ell\\nu$), the "
                      "fewest over the runs of each row: the counts behind the rejections of "
                      "tables~\\ref{tab:jetclassrej} and~\\ref{tab:jetclassall}."),
    "per_class": ("tab:perclass", "The recall of each JetClass-II class, the fraction of its test jets the network "
                  "assigns to it, as a linear probe on the frozen features and fine-tuned, for every network of the "
                  "per-class figures."),
    "per_run": ("tab:perrun", "Each run of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, and of the network trained from scratch, at "
                "the largest training set of each task: linear probe on the frozen features and fine-tuning. Linear probes on top and quark/gluon with the missing inputs substituted as chosen on "
                "validation (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon); fine-tuning with "
                "them set to zero. JetClass-II: class-balanced test set. The \\Kfull-class model with kinematic inputs only has "
                "one run, on top only; from scratch has no linear probe (dashes)."),
    "rand_prong": ("tab:randprong", "Prong-count boundaries of the five random 17-class partitions pretrained on "
                   "JetClass-II, and each partition's fine-tuned rejection of QCD per JetClass signal class divided by "
                   "that of the \\Kfull-class run of the same seed. The rejection is the inverse QCD efficiency at 50\\% signal "
                   "efficiency, 99\\% for $H\\to\\ell\\nu qq'$ and 99.5\\% for $t\\to b\\ell\\nu$, as in "
                   "table~\\ref{tab:jetclassrej}. __PRONG__ The column mixed groups counts the "
                   "groups that hold classes of more than one prong count. A dagger marks a ratio where fewer than "
                   "$\\nmin$ QCD jets pass in the partition's run or in that \\Kfull-class run."),
    "lrsweep": ("tab:lrsweep", "Learning-rate choice for training Sophon's network, pretrained on JetClass-II with "
                "\\Kfull{} or \\Kcoarse{} classes (seed 1) or trained from scratch, on $10^{4}$ top and quark/gluon jets: the "
                "best validation accuracy of each rate on top and quark/gluon and their sum. Pretrained: the rate of the pretrained "
                "layers, with the new output head at 50 times that rate, at most $5\\times10^{-3}$. $^{\\ast}$: the rate chosen, on "
                "the sum averaged over \\Kfull{} and \\Kcoarse{} classes for the pretrained weights and on each task's accuracy "
                "for the networks trained from scratch; the test jets play no part."),
    # the colon and semicolon replaced and "controls" named plainly (the caption rules; PI 2026-11-23, LEDGER FG-25 fix)
    "sets": ("tab:sets", "The label sets pretrained on JetClass-II and the networks they are compared with. A label set "
                         "assigns each of Sophon's \\Kfull{} classes to one of its own, and each row gives the network's "
                         "name, its number of classes, what it merges and its pretraining runs. The \\Kcoarse-class "
                         "partitions keep \\Kcoarse{} classes and the training-jet share of each \\Kcoarse-class group but "
                         "regroup the decays, and a flavor boundary keeps apart decays that differ only in quark flavor. A "
                         "mass network also regresses the jet mass. The random network, also with its "
                         "BatchNorm statistics recomputed on \\DesignBnJets{} training jets, and training from scratch are "
                         "references without pretraining."),
    "randparts": ("tab:randparts", "Each of the five random partitions, one pretraining run each, as a linear probe and "
                                   "fine-tuned, in the tasks and metrics of table~\\ref{tab:summary}; the last row of each "
                                   "block is their mean $\\pm$ standard deviation, the random-partitions row of the other "
                                   "tables. Each partition assigns the resonant classes to as many groups as the "
                                   "\\Kcoarse-class level has resonant groups, each holding the training-jet share of one "
                                   "\\Kcoarse-class group, and keeps every QCD jet in one more class; the assignments are "
                                   "drawn at random and selected to cut across the \\Kcoarse-class groups. "
                                   "Values are as in table~\\ref{tab:summary}; a dash marks a value not yet read out."),
    "inputs": ("tab:inputs", "The per-particle inputs: which the top and the quark/gluon data provide ($\\bullet$), "
               "and where a zero lies after the network's input normalization, in SDs of the training particles. The "
               "mean substitution puts the running mean of the input normalization in every missing slot; the "
               "random-draw substitution puts a species drawn from JetClass-II's frequencies in the type flags and a "
               "charge drawn given the species (top), and the displacement of a JetClass-II particle of the same "
               "species in the four displacement slots, averaged over four draws."),
    "partrepro": ("tab:partrepro", "ParT's top-tagging training script run from scratch on our cluster, unchanged and with "
                  "one change at a time, either Sophon's output head or Sophon's rescaling of every jet's four-momenta "
                  "to $p_{\\mathrm{T}}=500$~GeV. The last column counts the background jets that pass the cut "
                  "of $R_{30}$. ParT's published result and our network trained from scratch, which has both "
                  "changes and the missing inputs set to zero, are in table~\\ref{tab:published}."),
    "xbx": ("tab:xbx", "Resonance tagging in the soft-drop mass window, \\XbxWindowLo--\\XbxWindowHi~GeV: signal "
            "efficiency (\\%) at the QCD efficiency given, mean $\\pm$ SD over the pretraining runs. "
            "Transfer: a classifier on the frozen features, its seeds averaged per run (the randomly initialized "
            "network: the linear probe). Direct: Sophon's discriminant from the output scores, where $X\\to b\\bar b$ "
            "is a class of its own. " + POOLED + " From scratch: the network trained on each signal, three seeds."),
    "xbx_rej": ("tab:xbxrej", "Resonance tagging in the mass window: rejection $1/\\epsB$ of QCD jets at 50\\% and "
                "30\\% signal efficiency. Transfer: the classifier on frozen features; direct: Sophon's discriminant "
                "from the output scores; closest node: the pretrained $X\\to bq$ output against QCD; from scratch: "
                "the network trained on each signal, three seeds. " + SPREAD_POOLED),
    "published": ("tab:published", "Comparison with published models on JetClass, top tagging and quark/gluon tagging. "
                  "Our rows are the network trained from scratch and fine-tuned from the network pretrained with "
                  "\\Kfull{} classes, as the mean $\\pm$ SD over three runs, with the inputs that the top and "
                  "quark/gluon data lack set to zero. The row with kinematic inputs is a \\Kfull-class network "
                  "pretrained and fine-tuned on kinematic inputs only. Training jets counts the task's own training "
                  "set. Models named f.t., and OmniLearned, were pretrained on other data. AUC is the area under the "
                  "receiver operating characteristic curve, on JetClass averaged over the classes, each against the "
                  "rest, which with JetClass's equal class sizes equals the average over class pairs that ParT "
                  "reports. $R_{50}$ and $R_{30}$ are one over the fraction of background jets kept when the cut keeps 50\\% or 30\\% of the signal jets (30\\% is the working point of the top-tagging comparison), with quarks as the signal in quark/gluon tagging. The per-class rejections on JetClass "
                  "are in table~\\ref{tab:jetclassrej}."),
    "labelsets_full": ("tab:labelsetsfull", "Every row of the label-set table with the 17-class partitions."),
    "jc1_class_auc_full": ("tab:jc1_class_aucfull", "Every row of the JetClass per-class table with the 17-class partitions."),
    "sculpt_full": ("tab:sculptfull", "Every row of the sculpting table with the 17-class partitions."),
    "labelsets": ("tab:labelsets", "Fine-tuned results of the networks pretrained with \\Kres{}, \\Kmid{} and "
                  "\\Kcoarse{} classes and with the 17-class partitions, and of every network trained on Pythia "
                  "quark/gluon jets when it is scored on Herwig jets. The \\Kfull-class and from-scratch rows of the "
                  "other tasks are in table~\\ref{tab:published}, which defines the columns. Values are the mean "
                  "$\\pm$ SD over three runs, two for each flavor-boundary partition, and over the five random "
                  "partitions. The per-class rejections on JetClass are in table~\\ref{tab:jetclassrej}."),
    "summary": ("tab:summary", "The headline metric of every network of the headline figure, as a linear probe and "
                "fine-tuned."),
    "masseffect": ("tab:masseffect", "The networks pretrained with 162 and 17 classes, each without and with an added "
                   "mass output, on every task."),
    "jc1_conf": ("tab:jc1conf", "The JetClass confusion percentages of every fine-tuned network, as the confusion "
                 "figures draw them."),
    "summary_auc": ("tab:summaryauc", "The AUC beside each headline metric of table~\\ref{tab:summary}."),
    # PI 2026-11-23 (LEDGER SP-13, step 4): the supplement's copy with the top and quark/gluon columns
    "summary_auc_full": ("tab:summaryaucfull", "The AUC beside each headline metric of table~\\ref{tab:summary}, "
                         "on every task."),
    # PI 2026-11-23 (LEDGER SP-9): one row per stage in the appendix; each run's row in the data release (compute_full)
    # review of 5c40d69 (B7, LEDGER CA-10): sentences, no colon
    "compute": ("tab:compute", "GPU types and GPU-hours of each stage. Pretraining counts every run of every network, "
                "summed over the training and validation passes of its epochs as each epoch recorded its device, and "
                "fine-tuning counts the pretrained networks and training from scratch, every attempt of every fine-tuning run. "
                "Each run's hours, GPU types and last validation accuracy are in the data release."),
    "mlp_full": ("tab:mlpfull", "Every row of the multilayer-perceptron table, with each random partition on its own "
                 "row."),
    # PI 2026-11-23 (LEDGER SP-9, merge 6): every difference and null the text states, tab:sculptboot and tab:boot in it
    "tests": ("tab:tests", "The tested differences: for every difference or null the text states, the two networks, "
              "the metric and task, the difference, sigma and the difference over sigma."),
    "compute_full": ("tab:computefull", "GPU types and GPU-hours of every pretraining run, summed over the training and "
                "validation passes of its epochs as each epoch recorded its device, the GPU types its fine-tuning ran "
                "on, and its validation accuracy on its own pretraining classes after the last of its 80 epochs. The "
                "last part of the table gives that accuracy as the pretraining-curve figures draw it at the last epoch, the mean "
                "$\\pm$ standard deviation over \\MlstNRunsWord{} runs per label set, \\RandFlavRuns{} per "
                "flavor-boundary partition and \\MlstMassNRunsWord{} per mass network, and "
                "over the five random partitions with one run each. A flavor boundary is a class boundary between "
                "decays that differ only in quark flavor. The network pretrained with \\Kfull{} classes on the "
                "kinematic inputs only has one run and is left out of the last part."),
}
# (caption key, table file, wide, caption fill) of the tables written as placeholders until their data exist
PLACEHOLDERS = (("probes", "mlst_probes", True, None), ("probes_n", "mlst_probes_n", True, None),
                ("probes_n_full", "mlst_probes_n_full", True, None), ("ft_full", "mlst_ft_full", True, None),
                ("pairs", "mlst_pairs", True, None), ("pair_auc", "mlst_pair_auc", False, None), ("sculpt_main", "mlst_sculpt_main", False, None), ("mlp", "mlst_mlp", True, None), ("bn", "mlst_bn", False, None),
                ("balance", "mlst_balance", False, None), ("fill", "mlst_fill", True, None), ("fill_main", "mlst_fill_main", True, None),
                ("jc1_class_auc", "mlst_jc1_class_auc", True, None),
                ("mass", "mlst_mass", False, {"__PROBE_CENTRE__": ""}), ("mass_detail", "mlst_mass_detail", True, {"__BLANK__": "", "__MASS_RUNS__": ""}),
                ("sculpt", "mlst_sculpt", True, {"__SCULPT_BINS__": r"\pending{bins}", "__SCULPT_FLAV__": "", "__SCULPT_SIG__": "", "__SCULPT_CUT__": "", "__SCULPT_RUNS__": ""}), ("sculpt_boot", "mlst_sculpt_boot", False, {"__REPLICAS__": r"\pending{n}", "__SCULPT_BOOT_RAND__": ""}), ("ft", "mlst_ft", True, None),
                ("ft_rej", "mlst_ft_rej", True, {"__FEWEST__": r"\pending{n}"}), ("boot", "mlst_boot", True, {"__JC2COMP__": "class-balanced"}),
                ("bench_top", "mlst_bench_top", True, None), ("bench_qg", "mlst_bench_qg", True, None),
                ("xbx", "mlst_xbx", False, None), ("xbx_rej", "mlst_xbx_rej", True, None),
                ("rand_prong", "mlst_rand_prong", True, {"__PRONG__": ""}),
                ("lrsweep", "mlst_lrsweep", False, None), ("inputs", "mlst_inputs", True, None),
                ("partrepro", "mlst_partrepro", False, None), ("compute", "compute", False, None),
                ("compute_full", "compute_full", False, None), ("mlp_full", "mlst_mlp_full", True, None),
                ("tests", "mlst_tests", True, {"__REPLICAS__": r"\pending{n}"}),
                ("jetclass_all", "mlst_jetclass_all", True, None), ("jetclass_probe", "mlst_jetclass_probe", True, None),
                ("jetclass_nbkg", "mlst_jetclass_nbkg", True, None),
                ("coarse", "mlst_coarse", True, None), ("qcdhf", "mlst_qcdhf", False, {"__QCDHF_C__": r"\pending{n}", "__QCDHF_LIGHT__": r"\pending{n}"}),
                ("summary", "mlst_summary", False, None), ("summary_auc", "mlst_summary_auc", True, None),
                ("summary_auc_full", "mlst_summary_auc_full", True, None),
                ("sets", "mlst_sets", False, None), ("randparts", "mlst_randparts", True, None),
                ("masseffect", "mlst_masseffect", True, None), ("jc1_conf", "mlst_jc1_conf", True, None), ("bench_qg_herwig", "mlst_bench_qg_herwig", False, None),
                ("published", "mlst_published", True, None), ("labelsets", "mlst_labelsets", True, None),
                ("labelsets_full", "mlst_labelsets_full", True, None),
                ("jc1_class_auc_full", "mlst_jc1_class_auc_full", True, None),
                ("sculpt_full", "mlst_sculpt_full", True, {"__SCULPT_RUNS__": ""}),
                ("probes_full", "mlst_probes_full", True, None), ("pairs_full", "mlst_pairs_full", True, None),
                ("coarse_full", "mlst_coarse_full", True, None),
                ("qcdhf_full", "mlst_qcdhf_full", False, {"__QCDHF_C__": r"\pending{n}", "__QCDHF_LIGHT__": r"\pending{n}"}),
                ("fill_full", "mlst_fill_full", True, None),
                ("jetclass_rej", "mlst_jetclass_rej", True,
                 {k: r"\pending{n}" for k in ("__WPHQQL__", "__WPTBL__", "__MINBKGPC__")}),
                ("jetclass", "mlst_jetclass", True,
                 {k: r"\pending{n}" for k in ("__WPHQQL__", "__WPTBL__", "__MINBKG__", "__MINBKGPC__", "__PART100__")}))
LONGTABLES = (("per_class", "mlst_per_class"), ("per_run", "mlst_per_run"))

# The macros the new paper's outline names (docs/PAPER_OUTLINE.md, PI decisions 2026-11-02), with what each
# waits on: one still not emitted after build() becomes a red marker and is listed as missing, so a
# sentence written with it compiles and shows what it waits for. (name, waits on, twins)
REQUIRED = (
    *((f"MlstFTwoInit{k}", "the random network's linear probes", "") for k in ("AccJcTwo", "AccJcOne", "RthirtyMin",
                                                                             "RthirtyMax")),
    ("MlstJcTwoRecallCoarseLowClasses", "perclass_recall/probe_jc2.json", ""),
    ("MlstJcTwoRecallCoarseHighClassesWord", "perclass_recall/probe_jc2.json", ""),
    ("MlstFtJcOneErrCoarseOverFullBootEFour", "the JetClass bootstrap at 1e4 (after the equal-stopping read-out)", ""),
    ("MlstMassMlpResMassOverResZ", "the mass bootstrap (head/mass_sd_boot_v3.json)", ""),
    ("MlstMassMlpCoarseMassOverCoarseZ", "the mass bootstrap (head/mass_sd_boot_v3.json)", ""),
    ("MlstXbxHeadSeedsWord", "the X->bb and X->bs fits", ""), ("MlstXbxScratchSeedsWord", "xbx_scratch", ""),
    ("DesignNTrainSelected", "a committed count of the selected pretraining jets", ""),
    ("FtBudgetEvalSmall", "build_ft_budget_jobs.py", ""),
    ("LpSubstitutionTop", "fill fits", ""), ("LpSubstitutionQg", "fill fits", ""),
    ("MlstFtJcOneRejHfourqCoarseOverFull", "JetClass curves", "Pr"), ("MlstFtJcOneRejHfourqMidOverFull", "JetClass curves", "Pr"),
    ("MlstFtJcOneRejTbqqRandOverFull", "JetClass curves", "Pr"), ("MlstFtJcOneAccRandMinusCoarse", "JetClass curves", ""),
    ("MlstFtCmpJcTwoCoarseOverFullESix", "class-balanced JetClass-II fine-tuning", "Pr"),
    ("MlstPerClassRandFlavMergedOverApart", "probes", "Pr"), ("MlstPerClassCoarseNDoubled", "probes", ""),
    ("MlstPerClassCoarseQuarkLeptonN", "probes", ""), ("MlstBnMaxDevPair", "probes", ""),
    ("MlstMlpNatJcTwoResMassOverRes", "natural-composition nonlinear probes", "Pr"),
    ("MlstMlpNatJcTwoCoarseMassOverCoarse", "natural-composition nonlinear probes", "Pr"),
    ("MlstMlpNatPerClassRandFlavMergedOverApart", "natural-composition nonlinear probes", "Pr"),
    ("MlstMlpNatPerClassRandLeptonMergedOverApart", "natural-composition nonlinear probes", "Pr"),
    ("MlstLpBbQgZeroAccDropFull", "pair probes with the quark/gluon inputs zeroed (open item O10, not run)", ""),
    ("MlstLpBbQgZeroAccDropCoarse", "pair probes with the quark/gluon inputs zeroed (open item O10, not run)", ""),
    ("MlstPairRandMergedOverApartBb", "probes", "Pr"),
    ("MlstFtTopRthirtyFullOverScratch", "fine-tuning read-outs", "Pr"), ("MlstFtTopRthirtyScratch", "fine-tuning read-outs", ""),
    ("MlstFtTopRthirtyCoarseOverFull", "fine-tuning read-outs", "Pr"),
    ("MlstFtTopRfiftyCoarseOverFull", "ROC curves of the top fine-tuning (curves_v2/top_ft)", "Pr"),
    ("MlstFtTopRthirtyRes", "fine-tuning read-outs", ""), ("MlstFtTopRthirtyMid", "fine-tuning read-outs", ""),
    ("MlstFtTopRthirtyCoarse", "fine-tuning read-outs", ""),
    ("MlstFtTopRthirtyKinOverFull", "kinematic-only top fine-tuning", "Pr"),
    ("MlstFtTopRthirtyFullOverPart", "fine-tuning read-outs", "Pr"),
    ("MlstFtTopRthirtyResMassOverRes", "fine-tuning read-outs", "Pr PrMax"),
    ("MlstFtTopRthirtyFullOverScratchEThree", "fixed-budget fine-tuning (budget_metrics.json)", "Pr"),
    ("MlstFtTopDataEquivJets", "fixed-budget fine-tuning (budget_metrics.json)", ""),
    ("MlstFtQgRthirtyFullOverPart", "fine-tuning read-outs", "Pr"),
    ("MlstFtQgRthirtyCoarseOverFull", "fine-tuning read-outs", ""),      # no ratio word: PW-1 (fourth review)
    ("MlstFtQgRthirtyHerwigOverPythiaFull", "fine-tuning read-outs", "Pr"),
    ("MlstFtQgRthirtyHerwigOverPythiaScratch", "fine-tuning read-outs", "Pr"),
    ("MlstPartreproRthirtyRescaleOverPart", "partrepro", "Pr"), ("MlstPartreproSliceDropRatio", "not run", ""),
    ("MlstLpJcTwoTopZeroAccDropFull", "fits_v4", "Pr"), ("MlstLpJcTwoQgZeroAccDropFull", "fits_v4", "Pr"),
    ("MlstLpTopRthirtyDrawOverZeroFull", "fill fits", "Pr"), ("MlstLpTopRthirtyKinOverFull", "kinematic-only probe", "Pr"),
    ("MlstLpTopRthirtyFullOverInitBn", "fill fits", "Pr"), ("MlstLpQgRthirtyFullOverInitBn", "fill fits", "Pr"),
    ("MlstLpTopRthirtyFullOverInitBnZero", "probes", ""), ("MlstLpQgRthirtyFullOverInitBnZero", "probes", "Pr"),
    ("MlstLpJcOneGapFracEFour", "fixed-budget fine-tuning (budget_metrics.json)", "Pr"),
    ("MlstMassHeadResMassOverUncorr", "head/atlas_res_v3", "Pr PrMax"),
    ("MlstMassHeadResMassOverMlpRes", "head/atlas_res_v3", "Pr PrMax"),
    ("MlstCmpMlpResMassOverRes", "head/atlas_res_v3", "Pr PrMax"), ("MlstCmpMlpCoarseOverFull", "head/atlas_res_v3", "Pr"),
    ("MlstMassHeadResMassResponse", "head/atlas_res_v3", ""), ("MlstMassHeadCoarseMassResponse", "head/atlas_res_v3", ""),
    ("MlstMassUncorrectedResponse", "head/atlas_res_v3", ""),
    ("MlstMassRespAxesShareQcd", "head/atlas_res_v3", ""),
    ("MlstCmpJcTwoResMassOverRes", "probes", "Pr PrMax"),
    ("MlstSculptFullOverFloorOne", "sculpting", "Pr"), ("MlstSculptCoarseOverFullOne", "sculpting", "Pr"),
    ("MlstSculptMidOne", "sculpting", ""), ("MlstSculptCoarseMassOverCoarseOne", "sculpting", "Pr PrMax"),
    ("MlstSculptResMassOverResOne", "sculpting", "Pr PrMax"), ("MlstSculptResMassOverResTen", "sculpting", "Pr PrMax"),
    ("MlstSculptCoarseMassOverCoarseTen", "sculpting", "Pr PrMax"),
    ("MlstSculptDiffEffLo", "sculpting curves", ""), ("MlstSculptDiffEffHi", "sculpting curves", ""),
    ("MlstSculptLowestEff", "sculpting curves", ""), ("MlstSculptLowestNPass", "sculpting curves", ""), ("MlstSculptLowestMaxFloorSd", "sculpting curves", ""),
    ("XbsRthirtyFullOverScratch", "X->bs fits and from-scratch runs", "Pr"),
    ("XbsRthirtyCoarseOverFull", "X->bs fits", "Pr"), ("XbsRthirtyNodeOverTransferFull", "X->bs fits", "Pr"),
    ("XbbRthirtyDirectOverTransferFull", "X->bb fits", "Pr"), ("XbbRthirtyTransferCoarseOverFull", "X->bb fits", "Pr"),
    ("MlstComputeTotalGpuHours", "compute records", ""), ("MlstGpuTypeBound", "probes", ""),
    ("MlstGpuOtherSeedsDiff", "probes", ""),
    ("MlstInputBnZeroSigmaMax", "a committed JSON of the input-normalization offsets of mtx-l188-s1 (A8)", ""),
    ("MlstBootTestMaxProbe", "bootstrap", ""), ("MlstBootTestMaxFt", "bootstrap of the balanced JetClass-II fine-tuning", ""),
    # the ROC redesign (ROC_INTERFACE 3.2, 2026-11-02)
    ("MlstMinBkgPass", "mlst_v4.JC1_MIN_BKG", ""), ("MlstMassMinJets", "atlas_resolution.MIN_JETS", ""),
    *((f"MlstRocCut{p}Full", "ROC curves of the class pairs (curves_v2/pairs)", "") for p in ("Bb", "Bbqq", "Bc")),
    ("MlstRocNoCurveEemm", "ROC curves of the class pairs (curves_v2/pairs)", ""),
    *((f"MlstRoc{k}{p}", "ROC curves of the class pairs (curves_v2/pairs)", "")
      for k in ("NBkg", "RejMax") for p in ("Bb", "Eemm")),
    *((f"MlstRocCutJcOne{c}Full", "ROC curves of the JetClass fine-tuning (curves_v2/jc1_ft)", "") for c in ("Tbqq", "Tbl")),
    *((f"MlstRocJcTwo{k}", "class-balanced per-class probes (fits_v4)", "")
      for k in ("HurtList", "StableList", "HurtMinDrop", "HurtMaxDrop", "StableMaxChange")),
    *((f"MlstPerClassAucDiff{k}", "class-balanced per-class probes (fits_v4)", "")
      for k in ("CoarseSeedSdMedian", "MidMin", "MidMinClass")),
    *((f"MlstLpEemmAuc{w}", "the ee/mumu probe (fits_v4_sib)", "") for w in ("Full", "Mid", "Coarse", "InitBn", "RandMerged")),
    *((f"MlstPairEemm{k}", "the ee/mumu probe (fits_v4_sib)", "") for k in ("NTrain", "NVal", "NTestPerSide")),
)


# The captions as printed (PI 2026-11-11: about six lines at most, the details moved into the text). CAPTIONS keeps
# every table's full account, the source for the text when it is written; a table without an entry here prints its
# CAPTIONS text. Placeholders (__X__) are filled as in CAPTIONS.
SD_SHORT = "Mean $\\pm$ SD over the pretraining runs of each configuration."
SD_SHORT_POOLED = SD_SHORT + " " + POOLED
NETS = "networks pretrained on JetClass-II with 188, 162, 43 and 17 classes"
# the two rows of a random network (PI 2026-11-12: defined in each caption of a table that shows the recomputed one;
# experiments/MTX/bn_twins_v2.py --init, pretrain_v2.recompute_bn over BN_JETS jets)
BN_DEF = (" The batch-normalization (BatchNorm) statistics of the random network with BatchNorm recomputed are "
          "re-estimated on \\DesignBnJets{} training jets.")
ALL_NETS = ", of the two mass networks, of every 17-class partition"     # the supplement's full copies
# the captions of the paper's tables (PI 2026-11-12: what the table compares, then the definitions and run counts
# needed to read it, in sentences; every mark of the table explained, and no other)
PNETS = "the networks pretrained with 188, 162, 43 and 17 classes"
PART_DEF = " A flavor boundary keeps apart decays that differ only in quark flavor."
PUBLISHED_NOTE = "A dash marks a value not reported or not applicable."
MASSEFFECT_NOTE = "Quark/gluon tagging trains on Pythia jets."


def _placement():
    spec = importlib.util.spec_from_file_location("gallery_placement", pathlib.Path(__file__).parent / "figstyle"
                                                  / "gallery_placement.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


def _table_no(key: str) -> str:
    """"table S2": a supplement table named from a paper table's caption by its number of the day (the paper book holds
    no label of the supplement, so a \\ref would print ??)."""
    return "table " + _placement().number(key).removeprefix("Table ")


def _runs(*kinds) -> str:
    """The run-count sentence of a caption (PI 2026-11-13: one short sentence), from the kinds of row the table has:
    "set" (a label set, three pretraining runs; the random network's three initializations and the three seeds from
    scratch are three runs too), "mass" and "flav" (two each; build() checks that the two counts are equal), "rand"
    (five random partitions, one run each)."""
    two = " and ".join(w for k, w in (("mass", "mass network"), ("flav", "flavor-boundary partition"))
                           if k in kinds)
    out = " Values are the mean $\\pm$ standard deviation over \\MlstNRunsWord{} runs"
    if two:
        out += ", \\MlstMassNRunsWord{} per " + two
    if "rand" in kinds:
        # fourth review (B11; LEDGER CA-8): the spread is between five groupings of one run each, said in every caption
        out += (", and" if two else " and") + " five one-run random partitions"
    out += "."
    if "rand1" in kinds:        # a supplement copy with each random partition on its own row
        out += " Each random partition is a single run on its own row."
    return out


# the pair tables' lower limit (table_pairs' rule, PI 2026-11-18)
PAIR_GT_NOTE = (" $>$~marks a lower limit where fewer than \\MlstMinBkgPass{} background jets pass at 90\\% signal "
                "efficiency in a run, the number of background jets over \\MlstMinBkgPass{}, the smallest over the runs.")
# a mark that only some tables carry: its sentence is added when a cell of the printed table has it
KEY_MARKS = {
    # PI 2026-11-23 (LEDGER DL-17): the SC-1 test with the per-class bootstrap, in one clause (check_recall_tests)
    "jc1_class_auc": ((r"\downarrow", " $^{\\downarrow}$ ($^{\\uparrow}$)~marks a recall below (above) that of 188 "
                                       "classes by more than three standard deviations, both run spreads and the "
                                       "paired bootstrap over the class's test jets combined."),),
    "jc1_class_auc_full": ((r"\downarrow", " $^{\\downarrow}$ ($^{\\uparrow}$)~marks a recall more than three standard "
                                            "deviations, of the runs and the test jets, below (above) that of 188 "
                                            "classes."),),
    "probes": (("^{*}", " $^{*}$~marks a probe that did not converge."),),
    # PI 2026-11-23 (LEDGER SP-13, step 2; SC-1a): the rows whose two sides are scored on independent samples
    "tests": ((r"^{\diamond}", " $^{\\diamond}$~marks sides scored on independent test samples, each resampled on its "
                               "own."),
              # review of 5c40d69 (B3): a side of one run (the kinematic-inputs network), and a row over several
              (r"^{\circ}", " $^{\\circ}$~marks a side of one run, whose missing run spread makes $\\sigma$ a lower "
                            "bound."),
              (r"^{\ast}", " $^{\\ast}$~marks the least $|\\Delta|/\\sigma$ of several networks or bins."),
              # SC-2 (#1): the transfer heads, whose per-jet scores were not kept
              (r"^{\dagger}", " $^{\\dagger}$~marks the analytic test-sample term of section~\\ref{sec:uncertainties} in "
                               "place of the bootstrap.")),
    "qcdhf": ((r"\dagger", " $^\\dagger$~marks a value for which fewer than \\MlstMinBkgPass{} jets pass in a run."),),
    "jetclass_rej": ((r"\dagger", " $^\\dagger$~marks a value with fewer than \\MlstMinBkgPass{} QCD jets passing in a "
                                  "run."),),
    "lrsweep": ((r"^{\ast}", " $^{\\ast}$~marks the rate chosen, ``diverged'' a non-finite training loss, counted "
                             "lower than any rate that trained, and a dash a rate not tried."),),
    "pair_auc": ((r"\dagger", " $^{\\dagger}$~marks a pair the label set merges into one class."),
                 (r"\ddagger", " $^{\\ddagger}$~marks a pair the label set merges in part."),
                 (r"\parallel", " $^{\\parallel}$~marks a pair some random partitions merge.")),
    "pairs": ((r"\dagger", " $^{\\dagger}$~marks a pair the label set merges into one class."),
              (r"\ddagger", " $^{\\ddagger}$~marks a pair the label set merges in part."),
              (r"\parallel", " $^{\\parallel}$~marks a pair some random partitions merge."), (PAIR_GT, PAIR_GT_NOTE)),
    # the supplement's full copies (PI caption brief: marks explained in sentences, no colons)
    "pairs_full": ((r"\dagger", " $^{\\dagger}$~marks a pair that the label set merges into one class."),
                   (r"\ddagger", " $^{\\ddagger}$~marks a pair where the label set merges the first class with "
                                  "part of the second."), (PAIR_GT, PAIR_GT_NOTE)),
    "qcdhf_full": ((r"\dagger", " $^\\dagger$~marks a value for which fewer than \\MlstMinBkgPass{} jets pass in a run."),),
    "probes_full": (("^{*}", " $^{*}$~marks a probe that did not converge."),),
    "probes_n_full": (("^{*}", " $^{*}$~marks a probe that did not converge."),),
    "jetclass_all": ((r"\dagger", " $^\\dagger$~marks a value for which fewer than \\MlstMinBkgPass{} QCD jets pass in a "
                                  "run."),),
    "jetclass_probe": ((r"\dagger", " $^\\dagger$~marks a value for which fewer than \\MlstMinBkgPass{} QCD jets pass in "
                                    "a run."),),
}
# one definition and one wording of a multi-class AUC in every caption (PI 2026-11-12); mlst_v4.jetclass_rows stops the
# build if the JetClass average over class pairs, ParT's, differs from it
AUC_OVR = ("AUC is the area under the receiver operating characteristic curve, averaged over the classes, each against "
           "the rest.")
AUC_JC = (" With JetClass's equal class sizes this equals the average over class pairs that the Particle "
          "Transformer paper reports.")
R_SUB = (" $R$ with a subscript is the inverse QCD efficiency at that signal efficiency in percent, 50\\% except "
         "99\\% for $H\\to\\ell\\nu qq'$ and 99.5\\% for $t\\to b\\ell\\nu$.")
RHO_SHORT = "the ratio $\\rho$ of $1-\\mathrm{AUC}$ to the \\Kfull-class run of the same seed"
CAPTION_SHORT = {
    # PI 2026-11-23 (LEDGER CS-2b): the sculpting the main text states, each network beside its statistical floor
    "sculpt_main": ("Jensen--Shannon distance (base 2) between the soft-drop mass distributions of all QCD test jets and of "
                    "those passing a cut on one minus the summed QCD output of the pretraining head used directly. The cut "
                    "keeps a fraction $\\epsB$ of the QCD jets or half the resonant jets ($\\varepsilon_S = 50\\%$). "
                    "The floor is the distance a random selection of as many QCD jets shows, in the last row at each "
                    "QCD efficiency and, where each network's cut passes its own number of QCD jets "
                    "($\\varepsilon_S = 50\\%$), in its own column." + _runs("set", "flav", "mass", "rand")),
    # PI 2026-11-23 (LEDGER DL-R #4, CS-6): the AUCs the main text's error ratios on these pairs are formed from
    # fourth review (B6): the X->ee vs mumu column brings the near-one note; the test-jet count (appendix B.3) and the
    # five-decimal cap give way for it
    "pair_auc": ("The area under the receiver operating characteristic curve (AUC) of binary linear probes on the frozen "
                 "features separating pairs of JetClass-II classes, the second-named the background. The $X\\to bc$ "
                 "column carries one digit more than the standard deviation's rounding gives, so that the ratios "
                 "$\\rho$ of equation~(\\ref{eq:ratio}), which the text quotes, follow from it."
                 + _runs("set", "flav", "mass", "rand")),
    # PI 2026-11-23 (LEDGER SP-9, merge 6; SC-1 (1), (4)): what the table holds, no finding; at most 135 words.
    # PI 2026-11-23 (LEDGER SP-13, step 2): Delta, sigma and Delta / sigma, independent-sample rows marked (KEY_MARKS)
    # review of 5c40d69 (B3): two marks and the rows over several added, the run counts left to tab:sets (135 words)
    "tests": ("Every difference or null the text states, the mean over runs of the first network minus that of the "
              "second, on the metric, task and read-out of the heading above. $\\sigma$ adds in quadrature each "
              "network's standard deviation over its runs and that of $\\Delta$ over __REPLICAS__ bootstrap replicas "
              "of the test jets, drawn once for both. A step row compares two merging steps. Run counts are those of table~\\ref{tab:sets}, the five random partitions forming "
              "one side."),
    # PI 2026-11-23 (LEDGER CA-15, option a): the rho columns and the linear-probe block, which left tab:mlp, are here
    "mlp_full": ("Every row of table~\\ref{tab:mlp}, with each random partition on its own row, $\\rho$, the ratio of "
                 "$1-\\mathrm{AUC}$ to that of the \\Kfull-class run of the same seed, beside each JetClass-II and "
                 "class-pair AUC, and a last block of linear probes of the \\Kfull-class networks on the class token and "
                 "on the \\DesignHeadHidden{} outputs of the pretraining head's first layer and its activation applied "
                 "to it." + BN_DEF + PART_DEF + _runs("set", "flav", "mass", "rand1")),
    "probes": ("Linear probes of table~\\ref{tab:summary} on top and quark/gluon tagging, Pythia and Herwig, in accuracy "
              "and $R_{50}$, one over the fraction of background jets kept when the cut keeps 50\\% of the signal jets. The missing inputs "
              "(table~\\ref{tab:inputs}) are \\LpSubstitutionTop{} for top and \\LpSubstitutionQg{} for quark/gluon, "
              "chosen on validation, for the released Sophon checkpoint too." + TOPQG_PROBE_SET + _runs("set", "flav", "mass", "rand")),
    "probes_n": ("Linear probes of table~\\ref{tab:summary} at the smaller training sets in the header, in accuracy "
                "on class-balanced JetClass-II and JetClass and in $R_{30}$, one over the fraction of background jets kept when the cut keeps 30\\% of the signal jets, the working point of the top-tagging comparison, on top and quark/gluon tagging, Pythia and Herwig. Training jets are counted "
                "in total, and the missing inputs are set as in table~\\ref{tab:probes}." + _runs("set", "flav", "mass", "rand")),
    "probes_n_full": "Linear probes of table~\\ref{tab:probesn} at every training-set size, with each random "
                     "partition on its own row and the area under the receiver operating characteristic curve (AUC) "
                     "beside each quantity. On JetClass-II and JetClass the AUC is averaged over the classes, each "
                     "against the rest. The missing top inputs are set to random draws of a real particle and the "
                     "missing quark/gluon inputs to the training mean, the substitutions chosen on validation in "
                     "table~\\ref{tab:fill}. The network pretrained with \\Kfull{} classes on kinematic inputs only "
                     "is probed on top tagging only." + _runs("set", "flav", "mass", "rand1", "init"),
    "jc1_conf": "JetClass confusion matrices of " + PNETS + ", the two mass networks (\\Kres{} classes + mass, "
                "\\Kcoarse{} classes + mass) and the 17-class partitions, fine-tuned on one million JetClass jets, "
                "and of the same network trained from scratch on the same one million jets with its own, longer "
                "learning-rate schedule. Each row is a true class and gives the percentage of its test jets that the "
                "network assigns to each predicted class. The first and last blocks give these percentages. Every "
                "other block gives, in percentage points, the network named first in its heading minus the network "
                "named after ``minus'', each run paired with the run of the same pretraining seed and the "
                "differences averaged over the runs." + PART_DEF
                + _runs("set", "flav", "mass", "rand", "scratch"),
    "per_class": "The recall of each JetClass-II class, the fraction of its test jets the network assigns to it, "
                 "for " + PNETS + ", the two mass networks "
                 "(\\Kres{} classes + mass, \\Kcoarse{} classes + mass) and the 17-class partitions, as linear probes on the frozen features trained on one million "
                 "class-balanced JetClass-II jets and fine-tuned on as many. The label sets come first and the "
                 "17-class partitions second, each with a linear-probe block and a fine-tuned block, named in every "
                 "page's heading. The columns name each label set by its number of classes and each 17-class "
                 "partition by its boundary, and the classes are grouped by the 17 classes of the coarsest label set, "
                 "with decays that differ only in quark flavor next to each other." + PART_DEF
                 + _runs("set", "flav", "mass", "rand"),
    "probes_full": "Linear probes on the frozen features of " + PNETS + ALL_NETS + ", of the random networks and of the "
                   "released Sophon checkpoint, at each task's largest training set. The first column blocks give "
                   "JetClass-II, on its class-balanced test set, and JetClass, the later blocks top and quark/gluon "
                   "tagging, also scored on Herwig jets. The columns are accuracy, AUC, " + RHO_SHORT + ", and $R_{50}$ and "
                   "$R_{30}$, one over the fraction of background jets kept when the cut keeps 50\\% or 30\\% of the signal jets (30\\% is the working point of the top-tagging comparison). On JetClass-II "
                   "and JetClass the " + AUC_OVR + AUC_JC + " On JetClass-II the average is also given over the "
                   "classes with the full \\JcBalTestPerClass{} test jets (full count), over the resonant classes and "
                   "over the QCD classes. The linear probes on top and quark/gluon tagging set the missing inputs to "
                   "\\LpSubstitutionTop{} for top and to \\LpSubstitutionQg{} for quark/gluon, as chosen on "
                   "validation. The network pretrained with \\Kfull{} classes on kinematic inputs only has one run, "
                   "on top tagging only, and the released Sophon checkpoint, set apart by a rule, is outside the "
                   "controlled comparison." + BN_DEF + PART_DEF + _runs("set", "flav", "mass", "rand1"),
    "pairs": ("The rejection at 90\\% signal efficiency, $R_{90}$, separating pairs of JetClass-II classes, "
             "for linear probes and networks fine-tuned on a million class-balanced jets, the modes on different test "
             "jets. BatchNorm recomputed means batch-normalization statistics re-estimated on \\DesignBnJets{} training jets."
             + _runs("set", "flav", "mass", "rand")),
    "pairs_full": ("Every row of table~\\ref{tab:pairs}, with each random partition." + BN_DEF + PART_DEF
                   + _runs("set", "flav", "mass", "rand1")),
    # PI 2026-11-23 (LEDGER SP-9): the random partitions pooled, each in tab:mlpfull (the data release); (LEDGER CA-15) the
    # proposal of docs/CAPTION_CUTS_1123.md with option (a), no rho column and no linear-probe block; review of render
    # dfe5942 (B7): room for the bold sentence, the kinematic-inputs and BatchNorm-recomputed networks defined in tab:sets
    # (LEDGER CA-15) and named by their rows
    "mlp": ("Multilayer-perceptron probes on the frozen features of every network pretrained here and the random "
            "networks, on JetClass-II, JetClass, top-quark against QCD and quark against gluon jets, and pairs of "
            "JetClass-II classes differing only in quark flavor, which coarser label sets merge, the second-named the "
            "background. AUC is the area under the receiver operating characteristic curve, averaged over all, "
            "resonant or QCD classes, each against the rest, and $R_{30}$ is defined as in table~\\ref{tab:summary}. "
            "Missing top and quark/gluon inputs are set to zero." + _runs("set", "flav", "mass", "rand")),
    # PI 2026-11-23 (LEDGER CA-15): the proposal of docs/CAPTION_CUTS_1123.md, no PART_DEF (tab:sets defines it)
    "coarse": "Linear probes on the frozen features of " + PNETS + ", of the two mass networks, "
              "of the 17-class partitions and of the random network, trained on one million class-balanced "
              "JetClass-II jets whose 188 classes are merged into the 17 groups of the 17-class label set. Accuracy is "
              "the fraction of test jets assigned their true group. The last block gives multilayer-perceptron probes "
              "of the four label sets. "
              + AUC_OVR.replace("the classes", "the target's groups") + _runs("set", "flav", "mass", "rand", "init"),
    "coarse_full": "Linear probes of the networks of table~\\ref{tab:probesfull} on one million class-balanced "
                   "JetClass-II jets whose classes are merged into the 43 or 17 classes of the label sets or the 17 of "
                   "the partition with no flavor boundary. "
                   + AUC_OVR.replace("the classes", "the target's groups") + BN_DEF + PART_DEF
                   + _runs("set", "flav", "mass", "rand1"),
    "qcdhf": "Rejection of QCD jets by the $X\\to b\\bar b$ discriminant of linear probes on the frozen features of "
             + PNETS + ", of the two mass networks, of the 17-class partitions and of the "
             "random network, "
             "trained on one million class-balanced JetClass-II jets. The discriminant is the probe's $X\\to b\\bar b$ score divided by the sum of that score and the QCD scores. "
             "$R_{50}$ is the inverse QCD efficiency at 50\\% $X\\to b\\bar b$ efficiency, for QCD jets with two or "
             "more $b$ quarks, with one $b$ quark, and for all QCD jets." + PART_DEF
             + _runs("set", "flav", "mass", "rand", "init"),
    "qcdhf_full": "Rejection of QCD jets by the $X\\to b\\bar b$ discriminant of linear probes on the frozen features of "
                  + PNETS + ALL_NETS + " and of the random networks, trained on one million class-balanced "
                  "JetClass-II jets. The discriminant is the probe's $X\\to b\\bar b$ score divided by the sum of that "
                  "score and the QCD scores, and $R_{50}$ is the inverse QCD efficiency at 50\\% $X\\to b\\bar b$ "
                  "efficiency, for QCD jets with two or more $b$ quarks, with one $b$ quark, and for all QCD jets."
                  + BN_DEF + PART_DEF + _runs("set", "flav", "mass"),
    "bn": "Linear probes on the frozen features of " + PNETS + ALL_NETS + " and of the random network, on JetClass-II "
          "on its mixed test sample, with the batch-normalization (BatchNorm) statistics as accumulated in pretraining "
          "and as recomputed on \\DesignBnJets{} training jets. The columns give the AUC and " + RHO_SHORT + ". " + AUC_OVR + PART_DEF
          + _runs("set", "flav", "mass"),
    # PI 2026-11-23 (LEDGER SP-9): the linear-probe half only (its comparison passes 3 sigma,
    # scratchpad/part0/jc1_class_auc_check.md); SC-1 (4): the caption says what the table holds, the count is the text's;
    # review of render dfe5942 (B5; LEDGER CA-15): no flavor-boundary definition, tab:sets defines it
    "jc1_class_auc": ("The recall of each JetClass class, the fraction of its test jets the network assigns to it, "
                      "for linear probes trained on one million JetClass jets on the frozen features of every "
                      "pretrained network and of the random network." + _runs("set", "flav", "mass", "rand")),
    # the supplement's copy with the 17-class partitions (the network rule, PI 2026-11-14)
    "jc1_class_auc_full": ("Every row of table~\\ref{tab:jc1_class_auc} and the 17-class partitions, the recall of "
                           "each JetClass class, the fraction of its test jets the network assigns to it, "
                           "for linear probes and networks fine-tuned on one million JetClass jets." + PART_DEF
                           + _runs("set", "flav", "mass", "rand")),
    # PI 2026-11-23 (LEDGER CA-15): the proposal of docs/CAPTION_CUTS_1123.md; (CA-16) the released checkpoint's absent
    # row in the PI's words; PI 2026-11-22 (LEDGER CA-1): 141 words as printed, cut to 135 ("not repeated here" and
    # "draws of a real particle" out; the headings name the two fills shown); review of render dfe5942 (B8): no Herwig
    # column, and why, in the words of the checkpoint's clause (LEDGER CA-16)
    "fill": ("$R_{30}$, one over the fraction of background jets kept when the cut keeps 30\\% of signal jets, the "
             "top-tagging working point, of table~\\ref{tab:summary}'s linear probes on top-quark against QCD and quark "
             "against gluon jets, with missing inputs at zero, the training mean or real-particle draws. "
             "Table~\\ref{tab:summary} uses \\LpSubstitutionTop{} for top and \\LpSubstitutionQg{} for quark/gluon. "
             "BatchNorm recomputed means "
             "batch-normalization statistics re-estimated on \\DesignBnJets{} training jets."
             + _runs("set", "flav", "mass", "rand")),
    "fill_full": "Linear probes on the frozen features of every pretrained network and of the random network with "
                 "BatchNorm recomputed, with the missing inputs set to zero, the training mean or random draws of a "
                 "real particle. The columns give the relative drop in JetClass-II accuracy when the slots that the top "
                 "or the quark/gluon data lack are substituted, and $R_{30}$, one over the fraction of background jets kept when the cut keeps 30\\% of the signal jets, the working point of the top-tagging comparison, on top and quark/gluon tagging. The network on kinematic inputs only lacks "
                 "nothing and is scored on top tagging only. Herwig jets are not shown, as the substitution is chosen "
                 "on Pythia jets." + BN_DEF + PART_DEF + _runs("set", "flav", "mass", "rand1"),
    "mass": ("Jet-mass resolution and median response of $r = m_{\\mathrm{est}}/\\mgen$, the estimated mass over the "
            "generator-level soft-drop mass, on JetClass-II test jets with a generator-level match, for the soft-drop "
            "mass, the mass output of the pretraining head and a multilayer-perceptron probe on the frozen features. "
            "The resolution is half the 15.9--84.1\\% quantile range of $r$ over its median.__PROBE_CENTRE__ The best "
            "resolution is the smallest and the best "
            "median response the closest to one." + _runs("set", "mass")
            # review of 5c40d69 (C): the soft-drop mass has no runs; its spread is the test-jet bootstrap's
            + " The soft-drop mass has the standard deviation over bootstrap resamples of the test jets."),
    "boot": f"Linear probes on the frozen features and fine-tuning of {NETS}: uncertainty (percent) of the comparisons "
            "between their label sets at the largest training set, as the SD of the logarithm of "
            "the ratio over the pretraining runs / that from the finite test sample. JetClass-II: the ratio of "
            "$1-\\mathrm{AUC}$; top and quark/gluon: the ratio of $R_{30}$.",
    # review of 5c40d69 (B7, LEDGER CA-10): sentences, no colon; the rates from the builder's macros
    "lrsweep": "The best validation accuracy at each learning rate tried for fine-tuning and training from scratch, at "
               "every task and training-set size below the full top-tagging and quark/gluon sets. A pretrained row "
               "averages one run each of the networks pretrained with \\Kfull{} and \\Kcoarse{} classes, at that "
               "rate for the pretrained layers and \\FtHeadCap{} for the new head. A from-scratch row is one "
               "run, every weight at that rate. Rates spaced by factors of about three are added until the best has a "
               "lower-scoring rate on each side, none below \\FtSearchFloor. Every network of a task and size uses "
               "the rate chosen there.",
    "mass_detail": "Jet-mass resolution and median response on JetClass-II in detail, defined as in "
                   "table~\\ref{tab:mass}. In the upper part a cell over all jets is empty where "
                   "table~\\ref{tab:mass} holds it. The lower parts give, in bins of jet $p_{\\mathrm{T}}$ and "
                   "generator-level soft-drop mass, the mass output over all, resonant and QCD jets and the two regression "
                   "probes over all jets, and "
                   "__BLANK__ The soft-drop mass's $\\pm$ is its standard deviation over test-jet bootstrap "
                   "resamples.__MASS_RUNS__",
    "sculpt": ("Jensen--Shannon distance (base 2) between the soft-drop mass distributions of all QCD "
              "test jets and those passing a cut on one minus the summed QCD output, of networks used directly (first "
              "part) and fine-tuned (second part). The cut keeps a fraction $\\epsB$ of QCD "
              "jets, as random selection does__SCULPT_CUT__.__SCULPT_RUNS__"),
    "sculpt_full": ("Every row of table~\\ref{tab:sculpt} and the 17-class partitions, the Jensen--Shannon distance "
                    "between the soft-drop mass distributions of all QCD test jets and those passing the cuts of that "
                    "table, used directly (first part) and fine-tuned (second part)." + PART_DEF + "__SCULPT_RUNS__"),
    "sculpt_boot": ("Differences in mass sculpting between pairs of pretraining heads used directly (the first part of "
                   "table~\\ref{tab:sculpt}), first minus second, at the cuts of that table. Each mass network is compared with its own label set "
                   "and every other network with \\Kfull{} classes. $\\Delta$, positive where the first sculpts more, "
                   "is the difference of mean Jensen--Shannon distances. The run columns give each "
                   "network's standard deviation over its runs, the bootstrap column that of $\\Delta$ over __REPLICAS__ "
                   "replicas of the test jets, and $\\sigma$ adds the three in quadrature, a difference counting where "
                   "$|\\Delta|/\\sigma$ exceeds three. A label set has \\MlstNRunsWord{} pretraining runs, a mass "
                   "network and a flavor-boundary partition \\MlstMassNRunsWord{}__SCULPT_BOOT_RAND__."),
    "ft": ("The fine-tuned networks of table~\\ref{tab:summary} and from scratch at the smaller training sets, in accuracy "
          "on class-balanced JetClass-II and JetClass and $R_{30}$, one over the fraction of background jets kept when the cut keeps 30\\% of the signal jets, the working point of the top-tagging comparison, on top and quark/gluon tagging, Pythia and Herwig, with the missing top and quark/gluon inputs at "
          "zero. The 17-class partitions are compared at the largest training sets, in table~\\ref{tab:summary}." + _runs("set", "mass")),
    "ft_full": "Fine-tuning of table~\\ref{tab:ft} at every training-set size, with each random partition on its own "
               "row and the area under the receiver operating characteristic curve (AUC) beside each quantity. The "
               "JetClass-II blocks add the AUC averaged over the QCD classes and $\\rho$, $1-\\mathrm{AUC}$ at $10^{6}$ "
               "jets over that of the \\Kfull-class run of the same seed. On JetClass-II and JetClass the AUC is "
               "averaged over the classes, each against the rest. A dash marks a size at "
               "which the network was not fine-tuned. On JetClass-II the 17-class partitions are fine-tuned only on "
               "one million jets, and $\\rho$ is not defined for from scratch. JetClass-II and "
               "JetClass were not fine-tuned on 1,000 jets, and the particle inputs that the top and quark/gluon data "
               "lack are set to zero. The network pretrained with \\Kfull{} classes on "
               "kinematic inputs only has one run and is fine-tuned on top tagging only."
               + _runs("set", "flav", "mass", "rand1", "scratch"),
    "bench_top": "Top tagging: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 classes and fine-tuned, or "
                 "trained from scratch, on the full top training set (\\LitPartTopNTrainMillion{} million jets), "
                 "beside published models: accuracy, AUC, the background rejections $R_{50}$ and $R_{30}$ "
                 "($1/\\epsB$ at 50\\% and 30\\% signal efficiency) and the number of trainable parameters. Ours: "
                 "mean $\\pm$ SD over the pretraining runs, three seeds from scratch, one run with kinematic inputs "
                 "only; published uncertainties as given there.",
    "bench_qg": "Quark/gluon tagging: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 classes "
                "and fine-tuned, or trained from scratch, on the full training set (\\LitQgTrainMillion{} million Pythia jets; "
                "particle type as five flags and the charge) and scored on Pythia jets, beside "
                "published models: accuracy, AUC, $1/\\epsB$ at 50\\% and 30\\% quark efficiency and the number of "
                "trainable parameters. Uncertainties as in table~\\ref{tab:benchtop}; Herwig jets in "
                "table~\\ref{tab:ftrej}.",
    "jetclass": "JetClass: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 classes and fine-tuned, or "
                "trained from scratch, on one million JetClass jets, beside published models (2 million jets, from "
                "scratch): the number of trainable parameters, accuracy and AUC. " + AUC_OVR + AUC_JC + " Ours: mean "
                "$\\pm$ SD over three pretraining runs; from scratch: three seeds. Per-class rejections in "
                "table~\\ref{tab:jetclassall}.",
    "jetclass_rej": ("Rejection of QCD jets for each JetClass class, the inverse QCD efficiency at the signal "
                    "efficiency in the subscript, 50\\% except __WPHQQL__\\% for $H\\to\\ell\\nu qq'$ and "
                    "__WPTBL__\\% for $t\\to b\\ell\\nu$. Fine-tuned on one million JetClass jets, "
                    "beside from scratch and the published models of the Particle Transformer paper~\\cite{part} "
                    "(two million jets, no uncertainty). Linear probes are in table~\\ref{tab:jetclassprobe}."
                    + _runs("set", "flav", "mass", "rand")),
    "jetclass_all": "JetClass fine-tuning, in accuracy, AUC and the rejection of QCD jets for each signal class, of "
                    + PNETS + ", of the two mass networks and of the 17-class partitions, "
                    "beside training from scratch, each fine-tuned on one million JetClass jets and with "
                    + f"{V4.our_params(len(V4.JC1_COLS) + 1) / 1e6:.2f}" + " million parameters. " + AUC_OVR + AUC_JC + R_SUB + PART_DEF
                    + _runs("set", "flav", "mass", "rand"),
    "jetclass_probe": "Linear probes on the frozen features of " + PNETS + ALL_NETS + ", of the random networks and of "
                      "the released Sophon checkpoint, trained and scored on JetClass at the largest training set, in "
                      "the columns of table~\\ref{tab:jetclassall}. The released Sophon checkpoint, set apart by a rule, "
                      "is outside the controlled comparison. " + AUC_OVR + R_SUB + BN_DEF     # short: the table fits its pages
                      + _runs("set", "flav", "mass", "rand"),
    "xbx_rej": ("Rejection of QCD jets when tagging $X\\to b\\bar b$ and $X\\to bs$ jets with soft-drop mass "
               "\\XbxWindowLo--\\XbxWindowHi~GeV, for the networks and read-outs of table~\\ref{tab:xbx}, $R_{50}$ and "
               "$R_{30}$ being the inverse QCD efficiencies at 50\\% and 30\\% signal efficiency. A dash marks a "
               "read-out a network lacks. Closest class: the pretrained output of the class that contains $X\\to bq$ "
               "against the QCD outputs, nothing trained; the random network has none. From scratch, trained on each signal, stands in the classifier columns."
               + _runs("set", "flav", "mass", "rand")),
    "per_run": "The value of each run of " + PNETS + ", of the two networks with an added mass regression (+ mass), "
               "of the 17-class partitions, of the random network "
               "and of training from scratch, at each task's largest training set, as a linear probe on the frozen "
               "features and fine-tuned, with quark/gluon tagging also scored on Herwig jets. The linear probes on top and quark/gluon tagging set the missing inputs to "
               "\\LpSubstitutionTop{} for top and to \\LpSubstitutionQg{} for quark/gluon, as chosen on validation, "
               "and fine-tuning sets them to zero. JetClass-II is scored on its class-balanced test set. The network "
               "with kinematic inputs has one run, on top tagging only, and a dash marks a value that does not "
               "exist. $R_{30}$ is one over the fraction of background jets kept when the cut keeps 30\\% of the signal jets, the working point of the top-tagging comparison. Each row names its pretraining seed, the seed of "
               "training from scratch or the random network's initialization." + PART_DEF,
    "jetclass_nbkg": "The number of QCD test jets that pass each signal class's working point after fine-tuning on one "
                     "million JetClass jets, the fewest over the runs of each row, behind the rejections of "
                     "table~\\ref{tab:jetclassall}, for " + PNETS + ", the two networks with an "
                     "added mass output and the 17-class partitions, and for training from scratch. The working point is 50\\% "
                     "signal efficiency, 99\\% for $H\\to\\ell\\nu qq'$ and 99.5\\% for $t\\to b\\ell\\nu$." + PART_DEF,
    # PI 2026-11-23 (LEDGER SP-9, merge 4; SC-1 (4): a table caption says what the table holds, its findings in the text
    # and in tab:tests): the X->bb / X->bs columns of tab:xbx
    # review of 5c40d69 (B7, LEDGER CA-3): signal and background named, and the from-scratch training size per class
    # review of render dfe5942 (B8): the dashes defined, the words cut to keep 135
    # PI 2026-11-23 (LEDGER SH-13): the dash clause, without a colon; the 135 words cut from content, not conventions:
    # "the top-tagging working point" for the same reason in fewer words, the transfer head named by the block heading
    # (B17) and the QCD efficiency by the column heading; the released Sophon checkpoint's transfer heads are pending
    # PI 2026-11-24 (LEDGER SP-15): Fig. 3's Sophon clause moves here with its figure's values; R30 defined by
    # pointing to figure 2, the parallel figure (LEDGER CA-15), to stay within 135 words
    "summary": ("Accuracy on class-balanced JetClass-II and JetClass, $R_{30}$ (figure~\\ref{fig:F2}) "
                "for top-quark against QCD and quark against gluon jets, and the $X\\to b\\bar b$ and $X\\to bs$ "
                "signal efficiency against QCD jets in the mass window \\XbxWindowLo--\\XbxWindowHi~GeV. Training "
                "from scratch uses \\MlstXbxScratchTrainBb{} ($X\\to b\\bar b$) or "
                "\\MlstXbxScratchTrainBs{} ($X\\to bs$) jets a class. Dashes mark where the random networks have "
                "no pretrained head for a transfer head and resonance tagging fine-tunes no network. The released "
                "Sophon checkpoint enters as a frozen-feature reference (figure~\\ref{fig:F2})."
                + _runs("set", "flav", "mass", "rand")),
    "masseffect": ("The networks pretrained with 162 and 17 classes, without and with an added mass output, on every "
                  "task, metric and read-out (rows). Definitions, training sets and inputs are those of "
                  "tables~\\ref{tab:summary}, \\ref{tab:mass}, \\ref{tab:xbx} and~\\ref{tab:sculpt}, the fine-tuned "
                  "sculpting rows using the networks fine-tuned on JetClass-II. A dash marks a read-out a network "
                  "lacks, and repeated values are rounded per row." + _runs("set", "mass")),
    # PI 2026-11-23 (LEDGER SP-9, merge 5; SC-1 (4)); SP-13 step 4: JetClass-II and JetClass only, so the released
    # Sophon checkpoint's dashes (its top and quark/gluon cells) left with their columns
    "summary_auc": ("The area under the receiver operating characteristic curve (AUC) of the linear probes and "
                    "fine-tuned networks of table~\\ref{tab:summary} but the random network with BatchNorm recomputed, "
                    "on the same sets, averaged on JetClass-II and "
                    "JetClass over the classes, each against the rest, and on JetClass-II also over its resonant and "
                    "over its QCD classes alone." + _runs("set", "flav", "mass", "rand")),
    # PI 2026-11-23 (LEDGER SP-13, step 4): the supplement's copy, with the top and quark/gluon columns
    "summary_auc_full": ("Every column of table~\\ref{tab:summaryauc} and the AUC of the same networks on top tagging "
                         "and on quark/gluon tagging, scored on Pythia and on Herwig jets. The released Sophon "
                         "checkpoint's dashes are those of table~\\ref{tab:summary}." + _runs("set", "flav", "mass", "rand")),
    # review of render dfe5942 (B10; LEDGER CA-3): the tasks, signal and background named, in the first sentence
    "published": ("Published models beside ours on JetClass, top-quark against QCD jets and quark against gluon jets, ours "
                  "the mean $\\pm$ standard deviation over three runs. Models marked "
                 "fine-tuned and OmniLearned are pretrained on other data, training jets counting the downstream set. "
                 "AUC is the area under the receiver operating characteristic curve, on JetClass "
                 "averaged over the classes, each against the rest, equal here to ParT's average over class pairs. "
                 "$R_{50}$ and $R_{30}$ are one over the fraction of background jets kept when the cut keeps 50\\% or "
                 "30\\% of the signal jets (30\\%, the top-tagging working point)."),
    "xbx": ("Signal efficiency in percent of $X\\to b\\bar b$ and $X\\to bs$ jets against QCD jets at the QCD "
           "efficiency $\\varepsilon_B$ in the header, relative to the jets with soft-drop mass "
           "\\XbxWindowLo--\\XbxWindowHi~GeV. The pretrained "
           "networks, not fine-tuned, are read out on frozen features by a multilayer-perceptron classifier starting "
           "from their output head or by a linear probe, or directly by the $X\\to b\\bar b$ output against the QCD "
           "outputs. From scratch is the same network trained on each signal alone, on \\MlstXbxScratchTrainBb{} "
           "($X\\to b\\bar b$) or \\MlstXbxScratchTrainBs{} ($X\\to bs$) jets a class; a dash marks a read-out a "
           "network lacks."
           + _runs("set", "flav", "mass", "rand")),
    "inputs": ("The per-particle inputs, with a bullet ($\\bullet$) where the top or quark/gluon data provide an input "
              "and a dash where they lack it. The last column, for inputs one data set lacks, gives where zero lies "
              "after the 188-class network's input normalization, in standard deviations of the pretraining particles "
              "(0.00 for the charge and displacements, symmetric about zero). The mean substitution uses the running "
              "mean of this input normalization over the pretraining particles. The random-draw substitution gives a "
              "top-jet particle a type drawn from JetClass-II's frequencies and a charge given the type, and every "
              "particle the displacements of a JetClass-II particle of its type, averaged over four draws. Their "
              "results are in table~\\ref{tab:fill}. An input marked rescaled is taken after Sophon's rescaling of "
              "each jet to $p_{\\mathrm{T}}=\\DesignRescalePt$~GeV."),
    # PI 2026-11-23 (LEDGER CA-15): the proposal of docs/CAPTION_CUTS_1123.md, no finding, no typed number
    "partrepro": ("ParT's training script for top tagging, top-quark against QCD jets, run from "
                  "scratch unchanged and with one of two changes taken from Sophon, not combined, its output head, a "
                  "\\DesignHeadHidden{}-unit hidden layer before the output, and its rescaling of each jet to "
                  "$p_{\\mathrm{T}}=\\DesignRescalePt$~GeV. Accuracy is the fraction of test "
                  "jets classified correctly, AUC the area under the receiver operating characteristic curve, and "
                  "$R_{50}$ and $R_{30}$ one over the fraction of background jets kept when the cut keeps 50\\% or 30\\% "
                  "of signal jets, 30\\% being the top-tagging working point. Every value is from a single run, and the "
                  "relative Poisson uncertainty of $R_{30}$ is one over the square root of the last column, the "
                  "background jets that pass."),
    "labelsets": ("The fine-tuned networks of table~\\ref{tab:summary} and training from scratch on the full top and "
                 "quark/gluon training sets, in accuracy and $R_{50}$, one over the fraction of background jets kept when the cut keeps 50\\% of the signal jets, with top jets and quarks as signal, scored on Pythia and Herwig jets (top tagging "
                 "Pythia only). The rows for \\Kfull{} classes and from scratch repeat those of "
                 "table~\\ref{tab:published}." + TOPQG_FT_SET + _runs("set", "flav", "mass", "rand")),
    "labelsets_full": ("Every row of table~\\ref{tab:labelsets} and the 17-class partitions, fine-tuned on the full top "
                       "and quark/gluon training sets, in accuracy and $R_{50}$, one over the fraction of background jets kept when the cut keeps 50\\% of the signal jets, scored on Pythia and Herwig jets." + TOPQG_FT_SET + PART_DEF
                       + _runs("set", "flav", "mass", "rand")),

}


def printed_caption(key: str) -> str:
    return CAPTION_SHORT.get(key, CAPTIONS[key][1])


TEXT_CM = 15.5      # widest tabular kept whole (estimated): the 15 cm text block, shrunk by a few per cent at most
LABEL_CM = 6.3      # a wider row label wraps, at its commas, in a table that has to be split
SHRINK_OK = 1.0     # nothing is shrunk (PI 2026-11-11); was 1.08 with adjustbox
BODY_SCALE = 12 / 10.95     # the tables are set at the body size, 12 pt (PI 2026-11-12); the width model was calibrated at \\small
# LS-1 (the user, 2026-11-24: no page sideways) and PI DL-14/SP-14 (no text below the template's table size): a table
# too wide for the upright text block at the body size is set at the largest of these sizes at which it fits, down to
# the iopjournal template's table size (iopjournal.cls sets every tabular at \fontsize{8}{10}); (command, points, leading)
TABLE_SIZES = (("", 12.0, 14.5), (r"\small", 10.95, 13.6), (r"\footnotesize", 10.0, 12.0),
               (r"\fontsize{8}{10}\selectfont", 8.0, 10.0))
TEMPLATE_SIZE = TABLE_SIZES[-1][0]
_SIZE_NOW = [TABLE_SIZES[0]]    # the size _cell_cm and _height_cm estimate at (_float sets it)


def _depth0_split(s: str, sep: str) -> list[str]:
    """Split s on sep outside braces."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        c = s[i]
        depth += (c == "{") - (c == "}")
        if depth == 0 and s.startswith(sep, i) and (sep != "&" or s[i - 1:i] != "\\"):
            out.append(cur)
            cur, i = "", i + len(sep)
            continue
        cur += c
        i += 1
    return out + [cur]


def _cell_cm(t: str) -> float:
    """Estimated printed width of a cell at the body size of the 12 pt paper (calibrated at \\small on the
    generated tables, measured with LaTeX, and scaled by BODY_SCALE), its text scaled to the size being tried
    (_SIZE_NOW), its column padding not."""
    t = re.sub(r"\\multicolumn\{\d+\}\{(?:@\{\})?\w+\}", "", t)
    t = t.replace(BOLD_ON, "")      # a bold value (_bold) is as wide as a light one to 2% (BOLD_ON): one width model
    bold = 1.1 if r"\textbf" in t else 1.0
    t = re.sub(r"~?\\cite\{[^}]*\}", " [00]", t.replace(r"\mbox", ""))
    for a, b in ((r"$\pm$", "xx"), (r"{,}", ","), (r"\,", ""), (r"$<$", "x"), (r"^\dagger", ""), (r"^{\dagger}", ""),
                 (r"\to", "x"), (r"\bar", ""), (r"\Kfull{}", "188"), (r"\Kres{}", "162"), (r"\Kmid{}", "43"),
                 (r"\Kcoarse{}", "17"), (r"\Kfull", "188"), (r"\Kres", "162"), (r"\Kmid", "43"), (r"\Kcoarse", "17")):
        t = t.replace(a, b)
    t = re.sub(r"_\{([^}]*)\}", lambda m: "x" * max(1, round(0.7 * len(m.group(1)))), t)
    t = re.sub(r"\\[a-zA-Z]+", "x", t)
    t = re.sub(r"[{}$^_~]", "", t).strip()
    return ((0.177 * len(t) + 0.1 * sum(c.isupper() for c in t)) * bold * BODY_SCALE * _SIZE_NOW[0][1] / 12
            + 0.42)      # capitals are wider


def split_wide(tab: str, budget: float = TEXT_CM, label_cm: float = LABEL_CM) -> list[str]:
    """A tabular wider than the text, cut into tabulars stacked one above the other: the data columns
    are dealt into blocks that each fit, every block repeating the row-label column(s); a header cell
    spanning several columns is repeated in each block it reaches (a cut through such a group costs
    one and a half blocks, so groups stay whole where the width allows), full-width rows (section labels) span each block, cmidrules follow their columns, and
    a body row with only dashes in a block is left out of it. Columns l and c only; anything else is
    returned unchanged."""
    m = re.match(r"\s*\\begin\{tabular\}\{([lc]+)\}(.*)\\end\{tabular\}\s*$", tab, re.S)
    if not m:
        return [tab]
    spec, inner = m.group(1), m.group(2)
    if re.search(r"\\\\\[|\\cmidrule\{|\\addlinespace|\\hline", inner):
        raise SystemExit("FATAL: split_wide reads rows ending in a bare \\\\ and \\cmidrule(lr) rules only")
    n, nlab = len(spec), len(spec) - len(spec.lstrip("l"))
    nlab = max(1, min(nlab, n - 1))
    rule = re.compile(r"\s*(\\(?:toprule|midrule|bottomrule)|\\cmidrule\(\w*\)\{\d+-\d+\})")
    rows = []                                       # (rules, cells or None); a cell is (start, span, align, text)
    for seg in _depth0_split(inner, r"\\"):
        rules = []
        while (r := rule.match(seg)):
            rules.append(r.group(1))
            seg = seg[r.end():]
        cells, col = [], 0
        if seg.strip():
            for c in _depth0_split(seg, "&"):
                mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{((?:@\{\})?\w+)\}\{(.*)\}\s*$", c, re.S)
                span, al, txt = (int(mc.group(1)), mc.group(2), mc.group(3)) if mc else (1, None, c.strip())
                cells.append((col, span, al, txt))
                col += span
        rows.append((rules, cells or None))
    width = [0.0] * n
    for _, cells in rows:
        for s0, span, _, txt in cells or []:
            if span == 1:
                width[s0] = max(width[s0], _cell_cm(txt))
    spans = [(s0, s0 + span, _cell_cm(txt)) for _, cells in rows for s0, span, _, txt in cells or [] if 1 < span < n]
    lab = sum(width[:nlab])

    def wide(i, j):                                 # a block of columns i..j-1, widened by any header cell in it
        return lab + sum(width[i:j]) + sum(max(0.0, w - sum(width[max(a, i):min(b, j)]))
                                           for a, b, w in spans if a < j and b > i and b > nlab)
    if wide(nlab, n) <= budget:
        return [tab]
    atom = max((_cell_cm(a) for _, cells in rows for s0, span, _, txt in cells or [] if s0 == 0 and span == 1
                for a in (re.findall(r"\\mbox\{((?:[^{}]|\{[^{}]*\})*)\}", txt) or [txt])), default=0.0)
    # (a label without parts, _label_parts, is one atom: it never wraps)
    # a cut through a spanning header cell repeats the cell in both blocks; cuts between groups are preferred
    cross = [sum(s0 < j < s0 + span for _, cells in rows for s0, span, _, _ in cells or [] if span < n)
             for j in range(n + 1)]

    def blocks(lab_cm):
        nonlocal lab
        lab = lab_cm + sum(width[1:nlab])
        best = {nlab: (0, 0, 0.0, [])}              # end column -> (blocks, crossings, widest block, starts)
        for j in range(nlab + 1, n + 1):
            for i in range(nlab, j):
                w = wide(i, j)
                if i not in best or (w > budget and j - i > 1):
                    continue
                k, x, mx, starts = best[i]
                cand = (k + 1, x + (cross[i] if i > nlab else 0), max(mx, w), starts + [i])
                # a cut through a header group costs more than one more block: a pair's AUC and rho stay together
                if j not in best or (cand[0] + 1.5 * cand[1], cand[2]) < (best[j][0] + 1.5 * best[j][1], best[j][2]):
                    best[j] = cand
        return best[n][3] + [n]
    # the label column at the width of the longest label, at most label_cm, never below the widest part a label
    # cannot break (its network); the blocks are cut for that width
    label_w = max(atom, min(width[0], label_cm))
    if label_w + wide(nlab, n) - lab <= budget * SHRINK_OK:     # a few per cent over: one block, shrunk by adjustbox
        return [tab if label_w >= width[0] else re.sub(
            r"^(\s*\\begin\{tabular\}\{)l", lambda m: m.group(1) + r">{\raggedright\arraybackslash}p{"
            + f"{label_w - 0.42:.1f}" + "cm}", tab, count=1)]
    starts = blocks(label_w)
    wrap = label_w < width[0]
    label_cm = label_w - 0.42
    lspec = (r">{\raggedright\arraybackslash}p{" + f"{label_cm:.1f}" + "cm}" if wrap else "l") + "l" * (nlab - 1)
    out = []
    for a, b in zip(starts, starts[1:]):
        keep = lambda c: c < nlab or a <= c < b           # noqa: E731
        new = lambda c: c if c < nlab else c - a + nlab   # noqa: E731
        lines, body = [r"\begin{tabular}{" + lspec + spec[a:b] + "}"], False
        for rules, cells in rows:
            body |= any(r in (r"\midrule",) for r in rules)
            rr = []
            for r in rules:
                mc = re.match(r"\\cmidrule\((\w*)\)\{(\d+)-(\d+)\}", r)
                if not mc:
                    rr.append(r)
                    continue
                lo, hi = int(mc.group(2)) - 1, int(mc.group(3)) - 1
                cols = [c for c in range(lo, hi + 1) if keep(c)]
                if cols:
                    rr.append(rf"\cmidrule({mc.group(1)}){{{new(cols[0]) + 1}-{new(cols[-1]) + 1}}}")
            line = "".join(rr)
            if cells is not None:
                parts, shown = [], False
                for s0, span, al, txt in cells:
                    k = sum(keep(c) for c in range(s0, s0 + span))
                    if k:
                        parts.append(rf"\multicolumn{{{k}}}{{{al or 'c'}}}{{{txt}}}" if al or k > 1 else txt)
                        shown |= txt.strip() not in ("", "--") if s0 >= nlab or span > 1 or not body else False
                        shown |= not body and bool(txt.strip())
                if shown:
                    line += ("\n" if line else "") + " & ".join(parts) + r" \\"
            if line:
                lines.append(line)
        # rows left out of a block can leave two rules in a row (header rule, then a section's rule; a section's
        # rule, then the bottom rule)
        lines = [ln for i, ln in enumerate(lines) if not (ln.strip() == r"\midrule" and i + 1 < len(lines)
                                                          and lines[i + 1].lstrip().startswith((r"\midrule", r"\bottomrule")))]
        out.append("\n".join(lines + [r"\end{tabular}"]))
    return out


# how our rows were used downstream (PI 2026-11-12): one mode per table, or one per %SPLIT% part (a list), named once
# in an italic heading over the rows (HEADING, _headings), never on a row; a table whose columns change mode (boot,
# per_run, summary, the X->bb tables) or whose builder writes the headings (pairs, mass_detail) is not listed.
MODES = {"bn": "lp", "balance": "lp", "fill": "lp", "fill_main": "lp",
         "jetclass_probe": "lp", "ft_rej": "ft", "jetclass_all": "ft", "jetclass_nbkg": "ft", "rand_prong": "ft",
         "lrsweep": "ft", "probes_n_full": "lp", "ft_full": "ft", "mlp": "mlp", "mlp_full": "mlp",
         "jc1_class_auc": "lp", "pair_auc": "lp", "sculpt_main": "direct", "sculpt": ["direct", "ft"], "jc1_class_auc_full": ["lp", "ft"],
         "sculpt_full": ["direct", "ft"],
         "qcdhf": "lp", "coarse": "lp", "probes_full": "lp", "fill_full": "lp", "qcdhf_full": "lp", "coarse_full": "lp",
         "jetclass": "ft", "jetclass_rej": "ft", "bench_top": "ft", "bench_qg": "ft", "bench_qg_herwig": "ft",
         }


def _headings(tab: str, mode: str | None = None, star: bool = False) -> str:
    """The headings over the rows of a tabular (PI 2026-11-12). mode (a key of HEADING): the italic heading of the
    downstream mode before each run of rows that name a network (a LABEL or from scratch), unless the builder wrote
    one. The partition rows (flavor boundaries, random partitions) indented under the subheading PARTITIONS. A
    rule between two rows does not end a run (the released checkpoint below its rule); a row of anything else does
    (the soft-drop mass, a published model). star: the heading stays on the page of the row after it (a longtable)."""
    known = sorted(set(LABEL.values()) | {SCRATCH_LABEL, INIT_BN_SUB}, key=len, reverse=True)
    part = {LABEL[a] for a in FLAV + RAND + (RAND_POOLED,)}
    ours = tuple(rf"\textit{{{h}}}" for h in HEADING.values())
    out, prev = [], ""          # prev: "row" a network's row, "part" a partition's, "head"/"parthead" a heading
    for ln in tab.split("\n"):
        t = ln.strip()
        if not t.endswith((r"\\", r"\\*")) or "&" not in t:
            if t.startswith((r"\toprule", r"\begin", r"\end")):
                prev = ""
            elif t.startswith(r"\multicolumn"):
                prev = "parthead" if PARTITIONS in t else "head" if any(h in t for h in ours) else ""
            out.append(ln)
            continue
        if t.startswith(r"\multicolumn") and any(h in t for h in ours):     # a builder's heading split by columns
            prev = "head"
            out.append(ln)
            continue
        cells = _depth0_split(t, "&")
        first = cells[0].strip()
        name = first.removeprefix(r"\quad ").strip()
        k = next((k for k in known if re.fullmatch(re.escape(k) + r"(, seed \d+)?", name)), None)
        if k is None:
            prev = ""
            out.append(ln)
            continue
        ncol = sum(int(m.group(1)) if (m := re.match(r"\s*\\multicolumn\{(\d+)\}", c)) else 1 for c in cells)
        if mode and not prev:
            out.append(_full_row(ncol, HEADING[mode], italic=True) + "*" * star)
        if k in part:
            if prev not in ("part", "parthead"):
                out.append(_full_row(ncol, PARTITIONS) + "*" * star)
            if not first.startswith(r"\quad "):
                ln = ln.replace(first, r"\quad " + first, 1)
            prev = "part"
        else:
            prev = "row"
        out.append(ln)
    return "\n".join(out)


TOKEN = re.compile(re.escape(V4.TOK_OPEN) + r"(\w) ([^" + V4.TOK_CLOSE + r"]*)" + re.escape(V4.TOK_CLOSE))
PM = r"\,$\pm$\,"


def _pdg_place(sd: float) -> int:
    """The decimal place make_tables.pdg rounds sd to."""
    e = math.floor(math.log10(sd))
    top = round(sd / 10 ** (e - 2))
    if top >= 1000:
        top, e = 100, e + 1
    return 1 - e if top <= 354 else -e


def _at(x: float, place: int) -> str:
    t = f"{round(x, place):,.{max(place, 0)}f}".replace(",", "{,}")
    return t[1:] if t.startswith("-") and not any(c in "123456789" for c in t) else t


# A bold value is set in the bold series that is not extended: in Latin Modern (the book) 2% wider than the light
# value where \textbf's extended bold is 13% wider and pushes the widest tables off the page (measured with LaTeX);
# in a Times class this is the ordinary bold. The mark after a value stays light.
BOLD_ON = r"{\fontseries{b}\selectfont "


def _b(t: str) -> str:
    return BOLD_ON + t + "}"


SINGLE = r"$^{\S}$"       # a value from a single run (PI 2026-11-12: never bare)
AGREE = r"$^{\P}$"        # the SD over runs is zero or below the last digit shown: the value once, no "+- 0" and no "<"
MARK_NOTES = ((SINGLE, r" $^{\S}$~marks a value from a single run."),
              (AGREE, r" $^{\P}$~marks a run spread below the last digit shown."),
              # the one rejection convention (mlst_v4.rejection), stated once where a table has such a cell
              (r"$\geq$", r" $\geq$~marks a lower limit where no background jet passes, the number of background jets."))


def _mark_notes(cap: str, body: str) -> str:
    """The caption with a sentence for each of SINGLE and AGREE that a cell of the printed body carries."""
    return cap + "".join(note for mark, note in MARK_NOTES if mark in body)


# the finest place a column of values without a standard deviation is printed at (_column_place); a mean +- SD prints
# at its SD's own PDG place, however fine (fifth review B5, LEDGER SH-5: a cap at five decimals printed SDs of 6e-6
# as +- 0.00001)
MAX_DECIMALS = 5


def _sd_place(sd: float) -> int:
    """The decimal place of a standard deviation at one significant figure, at two if its first digit is 1
    (PI 2026-11-12). An SD of 0.95 to 0.99 rounds to 1.0 at the same place."""
    e = math.floor(math.log10(sd))
    return 1 - e if int(round(sd / 10 ** e, 9)) == 1 else -e


def _column_place(items: list) -> int:
    """One decimal place for a column (PI 2026-11-12): the place _sd_place gives the median of its standard
    deviations; one digit finer if a standard deviation would print as zero there; never past MAX_DECIMALS (a
    column of AUCs that all round to 1). With no standard deviation, the place _two gives values of that size
    (four decimals below 1, two below 10, one below 100), within four significant figures."""
    xs = [abs(v) for k, vals in items for v in (vals[:1] if k == "p" else vals)]
    big = max(xs, default=0.0)
    sds = [v[1] for k, v in items if k == "p" and not _zero_sd(v[0], v[1])]
    if not sds:
        place = 4 if big < 1 else 2 if big < 10 else 1 if big < 100 else 0
        return min(place, 3 - math.floor(math.log10(big))) if big > 0 else place
    place = _sd_place(statistics.median(sds))
    if any(not round(sd, place) for sd in sds):
        place += 1
    if any(0 < x < 100 for x in xs):    # a small value beside large ones (a random network's 12 beside 1,100)
        place = max(place, 0)           # keeps its own digits
    return min(place, MAX_DECIMALS)


def _zero_sd(mean: float, sd: float) -> bool:
    """An SD within float noise of zero is zero: identical runs whose mean is not exactly representable give an SD
    of ~1e-17 rather than 0."""
    return sd <= 1e-12 * max(1.0, abs(mean))


# PI 2026-11-23 (LEDGER TX-3): a table whose quoted ratio differs, from full-precision and from printed means, at the
# quoted digits gets one more digit: {table key: extra decimal places of its mean +- SD cells}. tab:pairauc: the 17-class
# AUC on X->bc vs bq, cs printed as 0.885 made "94% more errors" and "21% fewer errors" where the full means give 95% and
# 22% (render 26d1efd, build log "TX-3 DIFF")
# the digit goes to the column that needs it, X->bc vs bq, cs (column 2 of tab:pairauc), so the table stays upright
EXTRA_DIGIT = {"pair_auc": {2: 1}}
_EXTRA_NOW = [{}]       # {column: extra places} of the table being set (_float), read by _printed_place
_COL_NOW = [0]          # the column of the token being set (uniform_columns, _bold)


def _printed_place(kind: str, v: list, place: int) -> int:
    """The decimal place _render prints a token at: a mean +- SD at the SD's PDG place (SH-5, no cap), every
    other token at its column's place. A mean +- SD prints alike in every table, whatever the column's place
    (2026-11-23 review A8, LEDGER TN-4: Table 1's 1,800 +- 60 had printed as 1,799 +- 56 in tab:published)."""
    if kind.lower() == "p" and not _zero_sd(v[0], v[1]):
        return _pdg_place(v[1]) + _EXTRA_NOW[0].get(_COL_NOW[0], 0)
    return place


def _render(kind: str, v: list, place: int, floor: int | None = None) -> str:
    """A token: p mean +- SD rounded per cell as the PDG rounds (PI 2026-11-14: the SD to one or two significant
    figures, _pdg_place, at its own place), the mean to match, or the mean with AGREE where the SD is zero or below
    the last digit shown; s a single run's value with SINGLE at its column's place; o a number that is not a run's, at
    its column's place (floor: uniform_columns never sets that place coarser)."""
    bold, kind = kind.isupper(), kind.lower()       # upper case: a cell _bold chose; the mark stays light
    place = _printed_place(kind, v, place)
    if kind == "p" and not (_zero_sd(v[0], v[1]) or not round(v[1], place)):
        t, mark = _at(v[0], place) + PM + _at(v[1], place), ""
    elif kind == "p":
        t, mark = _at(v[0], place), AGREE
    else:
        t, mark = ", ".join(_at(x, place) for x in v), SINGLE if kind == "s" else ""
    return (_b(t) if bold else t) + mark


def uniform_columns(tab: str, floor: int | None = None) -> str:
    """The value tokens of a tabular (one cell per line, as the builders write them) printed at one precision
    per column (_column_place), the standard deviation included; floor: the coarsest place allowed (0: units)."""
    lines = tab.split("\n")
    where, items = [], collections.defaultdict(list)
    for i, ln in enumerate(lines):
        if V4.TOK_OPEN not in ln:
            continue
        col = 0
        for j, c in enumerate(_depth0_split(ln, "&")):
            m = TOKEN.search(c)
            if m:
                where.append((i, j, col))
                items[col].append((m.group(1).lower(), [float(x) for x in m.group(2).split()]))
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
            col += int(mc.group(1)) if mc else 1
    place = {c: _column_place(v) if floor is None else max(_column_place(v), floor) for c, v in items.items()}
    cells = {}
    for i, j, col in where:
        parts = cells.setdefault(i, _depth0_split(lines[i], "&"))
        _COL_NOW[0] = col
        parts[j] = TOKEN.sub(lambda m: _render(m.group(1), [float(x) for x in m.group(2).split()], place[col], floor),
                             parts[j])
        for mark in (SINGLE, AGREE):         # a per-cent sign after the token (_pct_cell) goes before its mark
            parts[j] = parts[j].replace(mark + r"\%", r"\%" + mark)
    for i, parts in cells.items():
        lines[i] = "&".join(parts)
    return "\n".join(lines)


# Bold (PI 2026-11-12): in the tables that compare networks, per block and column the best value and every value not
# resolved from it at three standard deviations. lower / unity: the columns (0 the row labels) where the smallest
# value, or the one closest to 1, is best (else the largest); skip: columns never bold (counts); published: the
# table has published rows, which take part by the value and the standard deviation they print (none: zero); whole:
# the tabular is one block whatever its headings; blocks: the caption says "their column under their italic heading" (of: other words); best: the
# caption's sentence before the bold note.
BOLD = {"summary": {"blocks": True}, "summary_auc": {"blocks": True}, "summary_auc_full": {"blocks": True}, "published": {"of": "their column and task", "published": True, "skip": (1, 2)},
        "labelsets": {}, "probes": {}, "pairs": {"blocks": True}, "probes_n": {"blocks": True}, "ft": {"blocks": True},
        "jc1_class_auc": {"blocks": True}, "jc1_class_auc_full": {"blocks": True}, "labelsets_full": {},
        "pair_auc": {"blocks": True}, "sculpt_main": {"lower": (1, 2, 3), "skip": (4,), "best": " Lower distances are better."},
        "mass": {"blocks": True, "lower": (1,), "unity": (2,)},
        "sculpt": {"lower": (1, 3, 6), "best": " Lower distances and higher $\\epsS$, $R_{50}$ and $1/D_{\\mathrm{JS}}$ "
                                               "are better."},
        "jetclass_rej": {"whole": True, "published": True, "skip": (1,)},
        "sculpt_full": {"lower": (1, 3, 6)},
        "xbx": {}, "xbx_rej": {}, "coarse": {}, "qcdhf": {}, "fill": {},
        "mlp": {"blocks": True},        # review of render dfe5942 (B7; LEDGER BD-1): its bold and the bold sentence
        "masseffect": {"of": "their row", "best": " The best resolution and distance are the lowest, the best median "
                                                  "response the closest to one (distances from one compared), and "
                                                  "every other value the highest."}}
# masseffect is built transposed: each of its rows is a column to _bold
# PI 2026-11-14 (A3, L3): sigma^2 = each row's SD^2 over runs + the bootstrap SD^2 over test jets of their difference,
# where the L3 bootstrap holds the two rows' runs (BOOT_REG); a published value with no stated uncertainty is never bold
# PI 2026-11-19 (A, bold), restated 2026-11-22: the caption sentence; tab:masseffect is transposed (its rows are the
# columns compared)
BOLD_NOTE = " Bold marks the best value in each column."
BOLD_NOTE_ROW = " Bold marks the best value in each row."


NEAR_ONE_NOTE = " A column whose best value is $>0.9999$ carries no bold."
# reference rows; so is a number that is no run's (token o). The kinematic-inputs network is a one-run diagnostic
# outside the network list (page review 2026-11-13: its single run, SD 0, made it bold among the references)
NEVER_BOLD = (LABEL[INIT], LABEL[INIT_BN], LABEL["KIN"], LABEL[SOPHON],
              SCRATCH_LABEL)      # PI 2026-11-14 (audit 3): from scratch is a reference too (tab:xbx's one weak training)
# the non-network references of the mass and sculpting tables (their labels may carry a jet group: "soft-drop mass,
# resonant jets"); the caption already names them as references (2026-11-22 checklist: they could be bolded)
REFERENCE_ROWS = ("soft-drop mass", "random selection", "floor")     # "floor": tab:sculptmain (fourth review, C7)
_NUM = r"\d+(?:\{,\}\d{3})*(?:\.\d+)?"
PLAIN = re.compile(rf"({_NUM})(?:{re.escape(PM)}({_NUM}))?")


# the L3 bootstrap of a printed cell (A3, PI 2026-11-14): token id -> {"models": its runs, "side": the RMS of their
# per-run bootstrap SDs (None if not stored), "diffs": [(runs a, runs b, bootstrap SD of mean a - mean b)]}; a "p"
# token carries the id as its third value (_boot_cell). 
BOOT_REG: dict = {}


def _l3_doc(data: pathlib.Path, name: str, probe: bool) -> dict:
    """bootstrap/l3_v5/name for a linear probe once the v5 refits are read, else bootstrap/l3/name; {} if absent."""
    dirs = (["bootstrap/l3_v5"] if probe and V4.pick(data, "linprobe/fits_v4") != "linprobe/fits_v4" else []) + ["bootstrap/l3"]
    for d in dirs:
        if (data / d / name).exists():
            return json.loads((data / d / name).read_text())
    return {}


def _boot_cell(body: str, doc: dict, metric: str, models: list) -> str:
    """body's "p" token with the id of its L3 bootstrap (BOOT_REG) as a third value, when doc holds the metric for
    some of the models; otherwise body unchanged."""
    m = TOKEN.search(body) if body else None
    per = [doc.get("models", {}).get(x, {}).get(metric, {}).get("sd") for x in models]
    per = [v for v in per if v is not None]
    if not m or m.group(1) != "p" or not per:
        return body
    diffs = [(set(d["a"]), set(d["b"]), d[metric]["boot_sd"]) for d in doc.get("diffs", {}).values()
             if isinstance(d, dict) and metric in d and d[metric].get("boot_sd") is not None]
    tag = float(len(BOOT_REG) + 1)
    BOOT_REG[tag] = {"models": set(models), "side": math.sqrt(sum(v * v for v in per) / len(per)), "diffs": diffs}
    return body[:m.end() - 1] + " " + repr(tag) + body[m.end() - 1:]


def _bold(key: str, tab: str, spec: dict | None = None) -> str:
    """The rows of a table (a tabular, or the rows of one block of it) with the cells to print bold marked: an
    upper-case token kind, which _render prints bold (a published cell is wrapped here). A block is the rows under one
    heading across the table (PARTITIONS is no such heading). PI 2026-11-19 (A, bold), restated 2026-11-22: per block
    and column the single best value, every value tied with it at the printed precision too (the column's place,
    uniform_columns); references (NEVER_BOLD, REFERENCE_ROWS) never; a published value with or without an uncertainty
    takes part like any other. No bold in a block's column with fewer than two values, or with a value printed as a
    bound (NEAR_ONE, PAIR_GT), whose best is not a printed value. spec: the settings in place of BOLD[key]
    (tab:masseffect, whose directions follow its rows)."""
    spec = spec or BOLD[key]
    lines = tab.split("\n")
    found, void = collections.defaultdict(list), set()      # (block, column) -> [(line, cell, printed value)]
    items = collections.defaultdict(list)                    # column -> every token in it, as uniform_columns sees them
    block, body = 0, r"\toprule" not in tab
    rows = []
    for i, ln in enumerate(lines):
        t = ln.strip()
        if not body:
            body = t.startswith(r"\midrule")
            continue
        parts = _depth0_split(ln, "&")
        if len(parts) == 1:
            block += t.startswith(r"\multicolumn") and PARTITIONS not in t and not spec.get("whole")
            continue
        label = parts[0].strip().removeprefix(r"\quad ").strip()
        ref = label in NEVER_BOLD or label == INIT_BN_SUB or label.startswith(REFERENCE_ROWS)
        col = 0
        for j, c in enumerate(parts):
            toks = TOKEN.findall(c)
            if len(toks) == 1:
                items[col].append((toks[0][0].lower(), [float(x) for x in toks[0][1].split()]))
            if ref or col in spec.get("skip", ()):
                pass
            elif NEAR_ONE in c or PAIR_GT in c:
                void.add((block, col))
            elif len(toks) == 1 and toks[0][0] in "ps":
                v = [float(x) for x in toks[0][1].split()]
                _COL_NOW[0] = col
                rows.append((block, col, i, j, v[0], _printed_place(toks[0][0], v, None)))
            elif spec.get("published") and (m := PLAIN.fullmatch(c.strip().removesuffix(r"\\").strip())):
                # a published value takes part as printed, with or without an uncertainty (PI 2026-11-19)
                rows.append((block, col, i, j, float(m.group(1).replace("{,}", "")), "published"))
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
            col += int(mc.group(1)) if mc else 1
    place = {c: _column_place(v) for c, v in items.items()}
    for block, col, i, j, x, at in rows:
        # each value at full precision with the place it prints at: a token at the place _render prints it, a
        # published value at the decimals it is printed with
        found[block, col].append((i, j, x, _published_place(lines[i], j) if at == "published"
                                  else place.get(col, 4) if at is None else at))
    chosen = collections.defaultdict(set)
    for (block, col), cells in found.items():
        if (block, col) in void or len(cells) < 2:
            continue
        score = ((lambda x: abs(x - 1)) if col in spec.get("unity", ()) else (lambda x: x) if col in spec.get("lower", ())
                 else (lambda x: -x))
        # BD-1 (fourth review, B8): the best is decided at full precision; a value ties with it, and is bold too, when
        # the two read the same at the printed place of the coarser of the two (37 +- 4 ties with 36.9 +- 0.9 only
        # if 36.9 is the best; 2026-11-23 review C had compared each value at its own place, so 37 beat 36.9)
        _, _, xb, pb = min(cells, key=lambda c: score(c[2]))
        for i, j, v, pv in cells:  # unity: the distances from one compared (page review 2026-11-13)
            q = min(pv, pb)
            if math.isclose(score(round(v, q)), score(round(xb, q)), rel_tol=0, abs_tol=1e-12):
                chosen[i].add(j)
    for i, js in chosen.items():
        parts = _depth0_split(lines[i], "&")
        for j in js:
            if TOKEN.search(parts[j]):
                parts[j] = TOKEN.sub(lambda m: V4.TOK_OPEN + m.group(1).upper() + " " + m.group(2) + V4.TOK_CLOSE, parts[j])
            else:
                parts[j] = PLAIN.sub(lambda m: _b(m.group(0)), parts[j], count=1)
        lines[i] = "&".join(parts)
    return "\n".join(lines)


def _published_place(line: str, j: int) -> int:
    """The decimals a published cell (cell j of a row) is printed with: 2,766 -> 0, 0.9877 -> 4."""
    m = PLAIN.search(_depth0_split(line, "&")[j])
    s = m.group(1).replace("{,}", "") if m else ""
    return len(s.split(".", 1)[1]) if "." in s else 0


def _label_parts(t: str) -> str:
    """A row label that wraps only at its ', ' ('random network, | BatchNorm recomputed')."""
    parts = t.split(", ")
    return " ".join(rf"\mbox{{{q}{',' if k < len(parts) - 1 else ''}}}" for k, q in enumerate(parts))


def _modes(key: str) -> list:
    m = MODES.get(key)
    return m if isinstance(m, list) else [m] if m else []


def _mode(key: str, part: int = 0) -> str | None:
    """The mode (a key of HEADING) of a table's part."""
    m = _modes(key)
    return m[min(part, len(m) - 1)] if m else None


def _labels(tab: str) -> str:
    """Every row label that is a LABEL (or from scratch), with a seed number or not, kept whole between its
    commas."""
    known = sorted(set(LABEL.values()) | {SCRATCH_LABEL}, key=len, reverse=True)
    out = []
    for ln in tab.split("\n"):
        first = _depth0_split(ln, "&")[0] if "&" in ln else ""
        k = next((k for k in known if re.fullmatch(re.escape(k) + r"(, seed \d+)?\s*", first)), None)
        if k:
            ln = _label_parts(first.strip()) + " " + ln[len(first):]
        out.append(ln)
    return "\n".join(out)


def _common_label(boxes: list, label_cm: float = LABEL_CM) -> list:
    """Blocks of one table with one first-column width: that of the widest label, at most label_cm (estimated)."""
    labs = [c for b in boxes for ln in b.split("\n") if "&" in ln and not ln.lstrip().startswith(("\\multicolumn", "\\begin"))
            for c in [_depth0_split(ln, "&")[0]] if c.strip()]
    atom = max((_cell_cm(a) for c in labs for a in re.findall(r"\\mbox\{((?:[^{}]|\{[^{}]*\})*)\}", c) or [c]),
               default=0.0)
    w = max(min(max(map(_cell_cm, labs), default=0.0), label_cm), atom) - 0.42
    spec = r">{\raggedright\arraybackslash}p{" + f"{w:.1f}" + "cm}"
    return [re.sub(r"\\begin\{tabular\}\{(>\{\\raggedright\\arraybackslash\}p\{[\d.]+cm\}|l)",
                   lambda m: r"\begin{tabular}{" + spec, b, count=1) for b in boxes]


CONTINUED = (r"\providecommand{\TableContinued}[2]{\label{#2}\ifnum\getpagerefnumber{#2}>\getpagerefnumber{#1}\relax"
             r"\makebox[\linewidth][l]{\textit{Table~\ref{#1} (continued)}}\par\smallskip\fi}")


# LS-1 (the user, 2026-11-24: "pages shouldnt be sideways"; PI SP-14): no table is set in landscape. A tabular estimated
# wider than this (the 15 cm text block less the estimate's error) is cut into blocks of columns stacked one above the
# other (split_wide), each repeating the row labels, at the tables' size: never shrunk
PORTRAIT_CM = 14.6
NO_WRAP = 99.0          # label_cm of a row-label column that never wraps


def _stretch(tab: str) -> str:
    """A tabular set across the whole line (tabular* with the space dealt between its columns), so that every table
    and every block of a table has one width and one left edge (PI 2026-11-11)."""
    return re.sub(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                  lambda m: r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + m.group(1) + "@{}}",
                  tab).replace(r"\end{tabular}", r"\end{tabular*}")      # every tabular of a block


TAIL_KEEP = 5        # rows at the end of a longtable block that never break apart
SHORT_BLOCK = 3      # a second mode block of at most this many rows does not start a page
# a longtable's closing rule, bound to the last row and with no glue below it, where the page could break and open a
# page holding only the repeated heading (booktabs' \bottomrule leaves \belowbottomsep below it; page review round 3)
CLOSE_RULE = r"\noalign{\nobreak\vskip\aboverulesep\hrule height\heavyrulewidth}"


def _long(blocks: list, cap: str, label: str, tail: list, whole: bool = False, sep: list = (), size: str = "") -> str:
    """Blocks of a table taller than a page, upright across the whole line (LS-1). A block that fits a page stays
    whole on one (no stub of a few rows under a repeated heading, no heading stranded at the foot of a page); a block
    taller than the page is a longtable whose rows break under its repeated heading. The caption above the first
    block, "Table N (continued)" above every later block and page; one table number. whole: every block is a page its
    builder chose and stays whole whatever the estimate. size: the rows' type size (the captions stay at the body
    size)."""
    cont = rf"\makebox[\linewidth][l]{{\textit{{Table~\ref{{{label}}} (continued)}}}}"
    out = ["{", LT_SETUP, *sep]
    for k, b in enumerate(blocks):
        fits = whole or _height_cm([b], "") + (0.45 * (len(cap) / 90 + 1) if k == 0 else 0.6) <= PAGE_FULL_CM
        if fits:
            if whole and k:         # a page its builder chose never shares one with the block before it
                out.append(r"\newpage")
            # no skip above a caption that heads the page (a longtable's has none either): ten points more for the
            # rows of a page its builder chose (tab:labelsets' first page was 6 pt over with the bold sentence)
            head = ([r"\setlength{\abovecaptionskip}{0pt}", rf"\captionof{{table}}{{{cap}}}", rf"\label{{{label}}}", r"\par\medskip"] if k == 0
                    else [cont, r"\par\smallskip"])
            out += [r"\begin{center}", r"\begin{minipage}{\linewidth}\centering",
                    *head, *([size] if size else []), _stretch(b), r"\end{minipage}", r"\end{center}"]
            continue
        m = re.match(r"\s*\\begin\{tabular\}\{([lc]+)\}(.*)\\end\{tabular\}\s*$", b, re.S)
        if not m or r"\midrule" not in m.group(2):
            raise SystemExit(f"FATAL: {label}: a block that is no plain tabular with a heading cannot be set as a longtable")
        spec, inner = m.group(1), m.group(2)
        head, rows = inner.split(r"\midrule", 1)
        head = head.replace(r"\toprule", "").strip()
        rows = rows.rsplit(r"\bottomrule", 1)[0].strip()
        # a second block heading (the fine-tuned rows after the probes) starts a page, so that its rows never continue on
        # a page without it (2026-11-12 review: Table A10's fine-tuned rows read as probes after the break)
        # only a mode heading (HEADING), not an italic working-point heading (tab:sculptboot, round-1 fix 2026-11-13)
        modes = tuple(re.escape(h) for h in HEADING.values())
        mode_head = re.compile(r"\\multicolumn\{\d+\}\{(?:@\{\})?l\}\{\\textit\{(?:" + "|".join(modes) + r")\}")
        heads = [i for i, ln in enumerate(rows.split("\n")) if mode_head.match(ln.strip())]
        if len(heads) > 1:
            lines = rows.split("\n")
            for n, i in reversed(list(enumerate(heads[1:], 1))):
                # a block of a few rows may share the page of the rows before it (on a page of its own it was two
                # rows alone: tab:mlp's linear probes, page review round 3); where it does not fit, the page breaks
                # below the rule above its heading, never inside it (the rows below its heading never break apart)
                stop = heads[n + 1] if n + 1 < len(heads) else len(lines)
                if sum(ln.rstrip().endswith("\\\\") for ln in lines[i + 1:stop]) > SHORT_BLOCK:
                    lines.insert(i, r"\newpage")
                elif lines[i - 1].strip() == r"\midrule":
                    lines[i - 1] = r"\midrule\noalign{\penalty-50}"
            rows = "\n".join(lines)
        # the last rows of a block stay together, so that no page holds a stub of one or two rows under a repeated
        # heading (round-1 fix 2026-11-13, tab:coarsefull, tab:pairsfull)
        lines = rows.split("\n")
        # the same for the rows before each forced page (a mode's last rows, tab:pairsfull's random network rows)
        cuts = [i for i, ln in enumerate(lines) if ln.strip() == r"\newpage"] + [len(lines)]
        start = 0
        for cut in cuts:
            ends = [i for i in range(start, cut) if lines[i].rstrip().endswith("\\\\")]
            last = ends[-TAIL_KEEP:]
            if any(mode_head.match(lines[i].strip()) for i in last):     # not across a mode heading
                last = last[max(k for k, i in enumerate(last) if mode_head.match(lines[i].strip())):]
            for i in last[:-1]:
                lines[i] = lines[i].rstrip() + "*"
            start = cut
        # the indented rows under the subheading "17-class partitions" never break apart, so that no page opens on
        # partition rows without it (page review round 3: tab:probesfull, tab:coarsefull, tab:pairsfull, tab:jetclassprobe)
        group = None
        for i, ln in enumerate(lines + [""]):
            if ln.lstrip().startswith(r"\multicolumn") and PARTITIONS in ln:
                group = []
            elif group is not None and ln.lstrip().startswith(r"\quad "):
                group.append(i)
            elif group is not None:
                for j in group[:-1]:
                    if lines[j].rstrip().endswith("\\\\"):
                        lines[j] = lines[j].rstrip() + "*"
                group = None
        # and a rule between rows never ends a page (tab:mlp's rows under the rule, tab:coarsefull's released
        # checkpoint): booktabs' \midrule may break at its own glue below the rule, so the same rule without a breakpoint
        rule = (r"\noalign{\nobreak\vskip\aboverulesep\hrule height\lightrulewidth\nobreak\vskip\belowrulesep"
                r"\nobreak}")
        lines = [rule if ln.strip() == r"\midrule" else ln for ln in lines]
        # nor is a sub-heading (an italic working point, "17-class partitions") ever the last line of a page
        # (page review 2026-11-13: tab:sculptboot's "eps_S = 50%" stood alone at the foot of a page)
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith(r"\multicolumn") and ln.rstrip().endswith("\\\\"):
                lines[i] = ln.rstrip() + "*"
        rows = "\n".join(lines)
        more = rf"\multicolumn{{{len(spec)}}}{{@{{}}l}}{{\textit{{Table~\ref{{{label}}} (continued)}}}} \\[\smallskipamount]"
        # a later block on a fresh page, so that its first heading is never left behind on the page before
        # (the first block's caption as far above its table as that of a block set whole: a medskip, added once)
        out += ([r"\newpage", r"\addtocounter{table}{-1}"] if k else [r"\addtolength{\belowcaptionskip}{\medskipamount}"]) + [      # a longtable steps the counter: one number a table
                *([size] if size else []),        # the caption's size is LT_SETUP's \normalsize
                r"\begin{longtable}{@{\extracolsep{\fill}}" + spec + "@{}}",
                (rf"\caption{{{cap}}}\label{{{label}}}\\" if k == 0 else more), r"\toprule", head, r"\midrule", r"\endfirsthead",
                more, r"\toprule", head, r"\midrule", r"\endhead",
                # the closing rule after the rows, not a last foot: a last foot taller than the (empty) foot made a
                # page broken in the table's last chunk open without the repeated heading (longtable's end-of-table
                # branch; tab:jetclassprobe, page review round 3)
                rows, CLOSE_RULE, r"\end{longtable}"]
    return "\n".join(out + [*tail, "}", ""])


def upright_check(out: dict, journal: pathlib.Path) -> None:
    """LS-1 (the user, 2026-11-24: "pages shouldnt be sideways"): the build stops on a table the paper inputs that is
    set in landscape."""
    used = {m for f in sorted((journal / "sections").glob("*.tex")) + [journal / "main.tex"] if f.exists()
            for m in re.findall(r"\\input\{(tables/[^}]+)\}", re.sub(r"(?<!\\)%.*", "", f.read_text()))}
    turned = sorted(k for k in out if k.removesuffix(".tex") in used and "landscape" in out[k])
    if turned:
        raise SystemExit(f"FATAL (LS-1): tables of the paper set in landscape: {turned}")


OPTIONAL_FILL = {"__PROBE_CENTRE__": ""}      # a sentence that depends on the data read, empty without it


APPENDIX_FLOATS = ("inputs", "compute", "coarse", "pair_auc", "sculpt_main", "partrepro")


def _float(key: str, body: str, wide: bool = False, fill: dict | None = None, notes: list | None = None,
           label_cm: float = NO_WRAP, budget: float = PORTRAIT_CM, portrait_label_cm: float | None = None,
           colsep_pt: float = 4.0, pages: bool = False, whole: str | None = None, size: str = "",
           pad: bool = False) -> str:
    """A generated table (PI 2026-11-11): at the full text width, upright (LS-1, the user 2026-11-24: no page is
    sideways); a tabular wider than the text block at the body size is set at the largest of TABLE_SIZES at which it
    fits, down to the template's table size (PI DL-14, SP-14), and only a tabular wider than the text block at that
    size is cut into blocks of columns stacked one above the other (split_wide), each repeating the row labels; a
    row label never wraps; nothing is scaled. label_cm and budget are kept for the callers that passed them and no
    longer narrow anything. portrait_label_cm: upright and whole whatever the estimate, the row labels wrapping at
    that width, 0 never (PI 2026-11-12: top, quark/gluon; one block). pages: the body's %SPLIT% parts, which were the
    turned pages of a table before LS-1 (tab:summary, tab:published, tab:labelsets), are blocks like any other. whole:
    the column padding at colsep_pt; "upright" also sets no part cut and keeps the body size or the size named,
    whatever the width estimate says (a width measured with LaTeX; the text block is 15.0 cm); "turned" (the
    landscape setting before LS-1) is now set like any other wide table. size: the type size its call names (one of
    TABLE_SIZES), never cut; the caption stays at the body size."""
    env = "table*" if wide else "table"
    # PI 2026-11-23 (LEDGER DL-14, typographic only): the appendix's upright tables may share a float page ([tbp]); the
    # main text's stay at the top of a text page ([t])
    spec = "[tbp]" if key in APPENDIX_FLOATS else "[t]"
    label, cap = CAPTIONS[key][0], printed_caption(key)
    for k, v in {**OPTIONAL_FILL, **(fill or {})}.items():
        cap = cap.replace(k, v)
    assert "__" not in cap, cap
    if r"\dagger" not in body:         # a mark the caption explains but no cell carries (\ddagger is another)
        cap = cap.replace(" Bold and $^\\dagger$ as there.", " Bold as there.").replace(" $^\\dagger$ as there.", "")
        cap = re.sub(r" \$\^\{?\\dagger\}?\$: [^.]*\.", "", cap)
    if r"\ddagger" not in body:
        cap = re.sub(r" \$\^\{\\ddagger\}\$: [^.]*\.", "", cap)
    if POOL_MARK not in body:
        cap = re.sub(r"; \$\^\{\\parallel\}\$: [^.]*\.", ".", cap)
    if "^{*}" not in body:
        cap = cap.replace(" $^{*}$: a probe that did not converge.", "")
    # bold needs the headings (its blocks) and the tokens (the unrounded values); a part its builder printed
    # (tab:summary, tab:published, tab:labelsets, tab:probes) has no token left and was marked there
    _EXTRA_NOW[0] = EXTRA_DIGIT.get(key, {})
    try:
        parts = [_labels(uniform_columns(_bold(key, _headings(b.strip(), _mode(key, k)))) if key in BOLD and V4.TOK_OPEN in b
                         else _headings(uniform_columns(b.strip()), _mode(key, k))) for k, b in enumerate(body.split("%SPLIT%"))]
    finally:
        _EXTRA_NOW[0] = {}
    parts = [re.sub(r"\\mbox\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", t) if "begin{tabular}" in t else t for t in parts]
    if key in BOLD and any(BOLD_ON in t for t in parts):
        cap += BOLD[key].get("best", "")
        cap += BOLD_NOTE_ROW if BOLD[key].get("of") == "their row" else BOLD_NOTE
        # fourth review (C8): "random network" only where its own row is printed; a table with only the network with
        # BatchNorm recomputed names that one
        init = any(re.search(r"(^|\n)\s*" + re.escape(LABEL[INIT]) + r"\s*&", t) for t in parts)
        refs = (["random network"] if init else ["random network with BatchNorm recomputed"]
                if any(LABEL[INIT_BN] in t for t in parts) else [])
        refs += [w for lab, w in ((LABEL["KIN"], r"\Kfull-class model with kinematic inputs only"),
                                  (LABEL[SOPHON], "released Sophon checkpoint"))
                 if any(lab in t for t in parts)]          # the reference rows the table has
        refs += [w for w in ("from scratch", "soft-drop mass", "random selection", "floor")    # PI 2026-11-14: from scratch too
                 if any(re.search(r"(^|\n)\s*" + w + r"\s*&", t) for t in parts)]
        if refs:
            one = "From scratch" if refs[0] == "from scratch" else f"The {refs[0]}"
            cap += (f" {one} is a reference, never bold." if len(refs) == 1 else
                    f" References ({', '.join(refs)}) are never bold.")
        if any(NEAR_ONE in t for t in parts):
            cap += NEAR_ONE_NOTE
    cap += "".join(note for mark, note in KEY_MARKS.get(key, ()) if mark in body)
    cap = _mark_notes(cap, "\n".join(parts))
    tail = [rf"\par\smallskip{{\noindent {n}}}" for n in notes or []]
    if portrait_label_cm is not None:
        lab = (r">{\raggedright\arraybackslash}p{" + f"{portrait_label_cm:.1f}" + "cm}") if portrait_label_cm else "l"
        boxes = [_stretch(re.sub(r"^(\s*\\begin\{tabular\}\{)l", lambda m: m.group(1) + lab, t, count=1)) for t in parts]
        # the column padding 4 pt (LaTeX's default 6 pt): the type stays at the tables' size, nothing is scaled
        return "\n".join([rf"\begin{{{env}}}{spec}", r"\centering", rf"\setlength{{\tabcolsep}}{{{colsep_pt:g}pt}}",
                          rf"\caption{{{cap}}}", rf"\label{{{label}}}", *([size] if size else []),
                          "\n\\par\\bigskip\n".join(boxes), *tail, rf"\end{{{env}}}", ""])
    # LS-1 (PI SP-14): upright, at the largest size of TABLE_SIZES at which no part is cut, down to the template's
    # table size; at that size a part still wider than the text block is cut into blocks of columns (split_wide).
    # whole="upright" (a width measured with LaTeX) or a size its call names: that size, never cut
    fixed = whole == "upright" or size
    ladder = [next((s for s in TABLE_SIZES if s[0] == size), (size, 12.0, 14.5))] if fixed else TABLE_SIZES
    try:
        for _SIZE_NOW[0] in ladder:
            cut = [t.strip() for b in parts for t in ([b] if fixed else split_wide(b, PORTRAIT_CM, NO_WRAP))]
            if len(cut) == len(parts):
                break
        size = _SIZE_NOW[0][0]
        return _set(cut, cap, label, tail, env, spec, size,
                    # pad: the column padding at colsep_pt also for a table that is cut (tab:pairsfull)
                    [rf"\setlength{{\tabcolsep}}{{{colsep_pt:g}pt}}"] if whole or pad else [])
    finally:
        _SIZE_NOW[0] = TABLE_SIZES[0]


def _set(cut: list, cap: str, label: str, tail: list, env: str, spec: str, size: str, sep: list) -> str:
    """_float's blocks set upright at size: a float when they fit a page, blocks breaking between pages when not, and
    longtables when a block is taller than a page."""
    if any(_height_cm([b], "") > PAGE_FULL_CM for b in cut):     # a block taller than a page: longtables
        return _long(cut, cap, label, tail, sep=sep, size=size)
    boxes = [_stretch(t) for t in cut]
    if _height_cm(boxes, cap) <= PAGE_CM:
        return "\n".join([rf"\begin{{{env}}}{spec}", r"\centering", *sep,
                          rf"\caption{{{cap}}}", rf"\label{{{label}}}", *([size] if size else []),
                          "\n\\par\\bigskip\n".join(boxes), *tail,
                          rf"\end{{{env}}}", ""])
    # taller than a page: not a float, so that it breaks between its blocks. The caption and the first block are one
    # unbreakable unit; a later block that starts a page says that it continues the table.
    out = [CONTINUED, r"\begin{center}", *sep,
           r"\begin{minipage}{\linewidth}\centering",
           rf"\captionof{{table}}{{{cap}}}", rf"\label{{{label}}}", r"\par\medskip", *([size] if size else []), boxes[0],
           r"\end{minipage}"]
    for k, b in enumerate(boxes[1:], 2):
        out += [r"\par\bigskip", r"\begin{minipage}{\linewidth}\centering", rf"\TableContinued{{{label}}}{{{label}:{k}}}",
                *([size] if size else []), b, r"\end{minipage}"]
    return "\n".join(out + [*tail, r"\end{center}", ""])


PAGE_FULL_CM = 23.0     # what a page holds of a table's blocks (estimated)


PAGE_CM = 17.0      # a table estimated taller than this (of the 24.7 cm text height) is set to break across pages


def _height_cm(boxes: list, cap: str) -> float:
    """Estimated height of a table at the size being tried (_SIZE_NOW): rows (a wrapped row label adds lines), rules
    and gaps between blocks, and the caption at the body size."""
    h = 0.45 * (len(cap) / 90 + 1)
    line_cm = 0.52 * _SIZE_NOW[0][2] / 14.5
    for b in boxes:
        # a wrapping row-label column, the first of the spec (tab:sets' wrapping columns are later ones, FG-25)
        wrap = re.search(r"(?:\\begin\{(?:tabular|longtable)\}\{|\\extracolsep\{\\fill\}\})>\{\\raggedright"
                         r"\\arraybackslash\}p\{([\d.]+)", b)
        lab = float(wrap.group(1)) + 0.42 if wrap else LABEL_CM
        for row in b.split(r"\\")[:-1]:
            first = _depth0_split(row.split("rule")[-1] if "rule" in row.split("&")[0] else row, "&")[0]
            h += line_cm * (math.ceil(_cell_cm(first) / lab) if wrap else 1)
        h += 1.0
    return h


def _longtable(key: str, body: str) -> str:
    """A longtable with its caption, each task block at one precision per column, the mode heading first."""
    body = BLOCK_SEP.join(uniform_columns(b) for b in body.split(BLOCK_SEP))
    body = _labels(_headings(body, _mode(key), star=True))
    cap = _mark_notes(printed_caption(key), body)
    if "^{*}" not in body.replace(r"\caption{__CAPTION__}", ""):      # a mark the caption explains but no cell carries
        cap = cap.replace(" $^{*}$: a probe that did not converge.", "")
    return body.replace("__CAPTION__", cap)


def _native_tex(root: pathlib.Path):
    spec = importlib.util.spec_from_file_location("appendix_tables", pathlib.Path(__file__).resolve().parent / "appendix_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.native_tex


def confusion_share_macros(em, data: pathlib.Path, words, missing: list) -> None:
    """PI 2026-11-21/23 (LEDGER CS-10, DL-8b): the pre-registered confusion-share test (experiments/EVAL/confusion_share.py,
    pulled to stats/confusion_share.json): its Spearman correlation, the bar set before it ran, the groupings and
    whether it passed; the build stops if the file's verdict is not its correlation against its bar."""
    f = data / "stats" / "confusion_share.json"
    if not f.exists():
        missing.append("the confusion-share test -- experiments/FIGS/data/v2/stats/confusion_share.json")
        for n in ("MlstConfShareRho", "MlstConfShareBar", "MlstConfShareNWord"):
            em.macro(n, r"\pending{confusion test}", f, "not pulled")
        return
    d = json.loads(f.read_text())
    if bool(d["pass"]) != (d["spearman"] >= d["pass_rho"]):
        raise SystemExit(f"FATAL: {f.name}: pass is {d['pass']} with Spearman {d['spearman']} against {d['pass_rho']}")
    em.macro("MlstConfShareRho", f"{d['spearman']:.2f}", f, "spearman", "confusion-share test: Spearman rank correlation")
    em.macro("MlstConfShareBar", f"{d['pass_rho']:.1f}", f, "pass_rho", "its bar, set before the test ran")
    em.macro("MlstConfShareNWord", words(len(d["groupings"])), f, "groupings (count)", "17-group label sets tested")


def ft_val_macros(em, data: pathlib.Path, missing: list) -> None:
    """The JetClass-II classes absent from the fine-tuning validation jets (PI 2026-11-15, LEDGER FT-2: "Keep,
    disclose"; review of render dfe5942, B11): those whose held-out jets the largest training set uses up. The
    validation sets, of the probes (val.parquet) and of fine-tuning (val_200k, cut to 199,680 jets by
    experiments/FT/stage_val_fixed.py, which keeps every class), are drawn from the same remainder
    (experiments/FT/stage_jc2_balanced.py), so a class absent from one is absent from the other."""
    man = data / "jc2bal" / "manifest.json"
    if not man.exists():
        missing.append("balanced JetClass-II manifest -- experiments/FIGS/data/v2/jc2bal/manifest.json")
        em.macro("FtValAbsentClasses", r"\pending{manifest}", man, "manifest not pulled yet")
        return
    cov = json.loads(man.read_text())["outputs"]["val.parquet"]["coverage"]
    em.macro("FtValAbsentClasses", str(len(cov["classes_absent"])), man,
             "outputs[val.parquet].coverage.classes_absent (count)",
             "JetClass-II classes with no held-out jet left for validation")


def required_pass(em, index, missing: list) -> None:
    # a twin is a ratio word: one without a tab:tests row is never emitted, one with a row is ratio_words' (SP-12)
    for name, waits, twins in REQUIRED:
        for n in [name] + [name + t for t in twins.split() if name + t in PR_ROWS]:
            if n not in em.provenance:
                em.macro(n, r"\pending{" + waits.replace("_", r"\_") + "}", index, "not computed yet",
                         f"waits on {waits}")
                missing.append(f"{n} -- waits on {waits}")


def build(em, root: pathlib.Path, missing: list, fmt_pm, fmt_int, words) -> dict:
    data = root / "experiments/FIGS/data/v2"
    # the v5 refits replace their families whole, every linear-probe family together (mlst_v4.check_v5)
    V4.check_v5(data)
    nat = V4.pick(data, FITS)
    fits3, comp = _load(data / nat), _load(data / "compute")
    # fits_v5_nat holds each untrained trunk's init_bn cells in its own file, fits_v3 in fits_v3_initbn
    fits3.update({"initbn" + m.removeprefix("init"): d for m, d in
                  (_load(data / FITS_INITBN) if nat == FITS else {k: v for k, v in fits3.items() if k.startswith("init-")}).items()})
    # the v4 probes (class-balanced JetClass-II, v4 grid; mlst_v4.py) replace fits_v3 once pulled
    fits_v4, raw_v4 = V4.load_fits(data)
    fits = fits_v4 or fits3
    if not fits_v4:
        missing.append("v4 linear probes -- experiments/FIGS/data/v2/linprobe/fits_v4/")
    if not fits:
        missing.append(f"linear probes -- experiments/FIGS/data/v2/{FITS}/")
        return {}
    V4.design_macros(em, root, data, missing, fmt_int)
    ft_val_macros(em, data, missing)
    confusion_share_macros(em, data, words, missing)
    for name, k, n_in, what in (("MlstNParamsFt", 2, 17, "two outputs (top, quark/gluon)"),
                                ("MlstNParamsFtKin", 2, 7, "two outputs, the 7 kinematic inputs"),
                                ("MlstNParamsJc", len(V4.JC1_COLS) + 1, 17, "the ten JetClass classes")):
        em.macro(name, fmt_int(V4.our_params(k, n_in)), root / "experiments/FIGS/mlst_v4.py", f"our_params({k}, {n_in})",
                 f"trainable parameters of our fine-tuned network, {what}")
    index = check_index(data)
    runs = _paper(root)
    # _runs() writes one count, \MlstMassNRunsWord, for the mass networks and the flavor-boundary partitions
    if len({len(runs[a]) for a in MASS + FLAV if a in runs}) > 1:
        raise SystemExit("FATAL: the mass networks and the flavor-boundary partitions have different run counts; "
                         "_runs() must name them apart")
    pre_runs = _pretrained(runs)
    done = [m for m in pre_runs if m in fits]
    em.macro("MlstProbeRunsDone", str(len(done)), index, "runs with a linear-probe fit (count)", "of the paper's runs")
    em.macro("MlstPaperRuns", str(len(pre_runs)), root / "configs/arms/v2_paper_runs.txt", "runs (count)")
    em.macro("MlstSdRelUnc", f"{100 / math.sqrt(2 * (len(runs['L188']) - 1)):.0f}\\%",
             root / "configs/arms/v2_paper_runs.txt",
             "relative standard error of a standard deviation from the runs of a hierarchy level, 1/sqrt(2(n-1))")
    if len(done) < len(pre_runs):
        missing.append(f"linear probes: {len(done)} of {len(pre_runs)} paper runs")
    status = merge_status(root)
    UNCONVERGED_SEEN.clear()
    native_tex = _native_tex(root)
    sjb = V4._module("experiments/FT/stage_jc2_balanced.py", "stage_jc2_balanced")
    full_count = full_count_classes(data, sjb.TEST_PER_CLASS) if fits_v4 else None
    rows_init = ALL_ROWS + (INIT, INIT_BN)
    # the substitution of the missing inputs, chosen once per task on validation (PI 2026-11-02): every
    # top and quark/gluon probe the tables show uses it; zero, marked, until the fill fits are pulled
    fill = _load(data / V4.pick(data, FITS_FILL))
    if not fill:
        missing.append(f"substitution probes -- experiments/FIGS/data/v2/{FITS_FILL}/")
    chosen, _ = chosen_substitutions(fill, runs)
    # the released Sophon checkpoint: a row of the frozen-probe tables only, never one of the runs; pending until pulled
    sophon = _load(data / V4.pick(data, FITS_SOPHON)).get(SOPHON_MODEL)
    fill = with_sophon_fill(data, fill)
    if not sophon:
        missing.append(f"the released Sophon checkpoint's probes -- experiments/FIGS/data/v2/{FITS_SOPHON}/")
    fits_s = {**fits, SOPHON_MODEL: V4.remap(sophon)} if sophon else fits
    runs_s = {**runs, SOPHON: [SOPHON_MODEL]}
    pairs_ft = load_pairs_ft(data)
    pairs_lp = load_pairs_lp(data)
    if not pairs_lp:
        missing.append("linear-probe class-pair curves -- experiments/FIGS/data/v2/curves_v2/pairs_v5/")
    if not pairs_ft:
        missing.append(f"fine-tuned class pairs -- experiments/FIGS/data/v2/{PAIRS_FT}/ (or its rerun {PAIRS_FT}_fixed/)")
    # the paper's tables pool the five random partitions into one row and print no ratio column; the supplement's
    # *_full copies keep both (PI 2026-11-12)
    pooled_s = _pooled(runs_s)
    # the network rule (PI 2026-11-14): the paper's tables outside the headline, class-pair and resonance families
    # show the label sets and mass networks; the 17-class partitions are in their supplement copies
    out = {"tables/mlst_probes.tex": _float("probes", table_probes_bench(fits_s, pooled_s, TAB, SETS + FLAV + (RAND_POOLED, INIT, SOPHON),
                                                                         fill, chosen, kin=True), True, pages=True, whole="turned", colsep_pt=6),
           "tables/mlst_probes_full.tex": _float("probes_full", table_probes(fits_s, runs_s, TAB, rows_init + (SOPHON,),
                                                                             full_count, fill, chosen, kin=True), True),
           # the linear-probe training-size figures (A10, A10-part): every point below Table 1's in the appendix,
           # every point and the AUC versions' in the supplement
           "tables/mlst_probes_n.tex": _float("probes_n", table_by_size(
               "probes_n", "lp", lambda t, ds, met, a: size_probe(fits, fill, chosen, _pooled(runs), ds, met, a),
               (LABEL[INIT], INIT), {t: max(size_probe(fits, fill, chosen, runs, ds, met, "L188"))
                                     for t, ds, met, _ in SIZE_TASKS}), True, whole="upright"),
           "tables/mlst_probes_n_full.tex": _longtable("probes_n_full", table_probes_n(
               fits, runs, TAB, ALL_ROWS + (INIT,), fill, chosen, kin=True)),
           "tables/mlst_pair_auc.tex": _float("pair_auc", table_pair_auc(fits, _pooled(runs), TAB, status,
                                                                      _paper_rows(rows_init)), whole="turned",
                                                colsep_pt=3),     # upright at 1 pt with three pairs (c1253a9); the
           # fourth pair (X->ee vs mumu, fourth review B6) is about 20 pt more than the upright block: turned
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, _pooled(runs), TAB, status,
                                                                _paper_rows(rows_init), pairs_ft, pairs_lp), True,
                                           label_cm=PAIRS_LABEL_CM, budget=TEXT_CM * SHRINK_OK),
           "tables/mlst_pairs_full.tex": _float("pairs_full", table_pairs(fits, runs, TAB, status,
                                                                          rows_init, pairs_ft, pairs_lp),
                                                True, label_cm=PAIRS_LABEL_CM, budget=TEXT_CM * SHRINK_OK, pad=True),
           "tables/mlst_balance.tex": _float("balance", table_balance(fits3, fits_v4, runs, TAB, ALL_ROWS + (INIT_BN,)))}
    em.macro("MlstPairsNWord", words(len(PAIRS_TABLE)), root / "experiments/FIGS/mlst_results.py",
             "PAIRS_TABLE (count)", "class pairs of tab:pairs")
    comparisons(em, fits, runs, status, index, fmt_pm)
    pair_figure_macros(em, data, root, runs, status, words, missing)
    eemm_macros(em, fits, runs, index, fmt_pm, fmt_int, status)
    probe_abs_macros(em, fits, runs, index, fmt_pm)
    if fits3:
        bn_macros(em, fits3, runs, index)
        out["tables/mlst_bn.tex"] = _float("bn", table_bn(fits3, runs, TAB, ALL_ROWS))
    co = _load(data / V4.pick(data, "linprobe/fits_v4_coarse"))
    if co:
        co_s = {**co, SOPHON_MODEL: sophon} if sophon else co
        # the random partitions pooled, kept in the supplement beside its full version (PI 2026-11-13: delete
        # nothing); cut between its targets as the full version is (the three side by side overran a turned page)
        out["tables/mlst_coarse.tex"] = _float("coarse", table_coarse(co_s, pooled_s, TAB,
                                                                      _paper_rows(ALL_ROWS + (INIT,)), COARSE_PAPER,
                                                                      bs3=load_bs3_mlp(data)), True)
        out["tables/mlst_coarse_full.tex"] = _float("coarse_full", table_coarse(co_s, runs_s, TAB,
                                                                                ALL_ROWS + (INIT, INIT_BN, SOPHON)), True)
    else:
        missing.append("probes on the coarse targets -- experiments/FIGS/data/v2/linprobe/fits_v4_coarse/")
    hf = _load(data / V4.pick(data, "linprobe/fits_v4_qcdhf"))
    if hf:
        # the random partitions pooled, kept beside the full version (PI 2026-11-13: delete nothing)
        out["tables/mlst_qcdhf.tex"] = _float("qcdhf", table_qcdhf(hf, _pooled(runs), TAB, _paper_rows(ALL_ROWS + (INIT,))),
                                              fill=qcdhf_fill(hf))
        out["tables/mlst_qcdhf_full.tex"] = _float("qcdhf_full", table_qcdhf(hf, runs, TAB, ALL_ROWS + (INIT, INIT_BN)),
                                                   fill=qcdhf_fill(hf))
    else:
        missing.append("QCD heavy-flavor subclasses -- experiments/FIGS/data/v2/linprobe/fits_v4_qcdhf/")
    V4.gpu_bound_macro(em, fits, runs, comp, index)
    mlp_nat = load_mlp_natural(data)
    per_class_macros(em, fits, runs, root, index, fmt_pm, native_tex, mlp_nat)
    out["data_release/jetclass2_per_class.csv"] = per_class_csv(fits, runs, root)
    # per-class recall (PI 2026-11-18): Table A2 and its supplement copy on JetClass, tab:perclass on JetClass-II; the
    # linear probes required, the fine-tuned blocks pending until their file is pulled
    rec = _load(data / RECALL_DIR)
    ft_rec = {t: rec.get(f"ft_{t}_N{FT_RATIO_N}", {}) for t in ("jc1", "jc2")}
    # PI 2026-11-23 (LEDGER DL-17): every file the per-class marks read, checked against the SC-1 test, and its per-class
    # delta / sigma in the data release (build_csv/perclass_recall_tests.csv)
    read = {k: d for k, d in rec.items() if k in ("probe_jc1", "probe_jc2") or k in (f"ft_{t}_N{FT_RATIO_N}" for t in ("jc1", "jc2"))}
    for k, d in read.items():
        check_recall_tests(d, data / RECALL_DIR / f"{k}.json")
    if read:
        out["data_release/perclass_recall_tests.csv"] = recall_tests_csv(read)
    if rec.get("probe_jc1"):
        # PI 2026-11-21: every network (the random partitions pooled), with the 3-sigma marks; PI 2026-11-23 (LEDGER
        # SP-9): the paper's table is the linear-probe half, the fine-tuned half is the supplement copy's
        rows_a2 = _paper_rows(ALL_ROWS)
        out["tables/mlst_jc1_class_auc.tex"] = _float("jc1_class_auc", table_jc1_class_recall(
            ft_rec["jc1"], rec["probe_jc1"], _pooled(runs), TAB, rows_a2).split("%SPLIT%")[0], True)
        jc1_recall_macros(em, rec["probe_jc1"], rows_a2, data / RECALL_DIR / "probe_jc1.json")
        out["tables/mlst_jc1_class_auc_full.tex"] = _float("jc1_class_auc_full", table_jc1_class_recall(
            ft_rec["jc1"], rec["probe_jc1"], _pooled(runs), TAB, _paper_rows(ALL_ROWS)), True)
    if rec.get("probe_jc2"):
        jc2_recall_macros(em, rec["probe_jc2"], data / RECALL_DIR / "probe_jc2.json", words)
        # every point of the JetClass-II per-class figures (PI 2026-11-13: restored, supplement)
        out["tables/mlst_per_class.tex"] = _longtable("per_class", table_per_class(
            {"lp": rec["probe_jc2"], "ft": ft_rec["jc2"]}, _pooled(runs), TAB, root, native_tex))
    missing += [f"per-class recall -- experiments/FIGS/data/v2/{RECALL_DIR}/{k}.json" for k in
                ("probe_jc1", "probe_jc2", f"ft_jc1_N{FT_RATIO_N}", f"ft_jc2_N{FT_RATIO_N}") if not rec.get(k)]

    conf, conf_s = _load(data / "confusion_jc1"), _load(data / "confusion_jc1_fulldecay")
    if conf and conf_s:
        # a negative change prints with a minus sign, not a hyphen
        out["tables/mlst_jc1_conf.tex"] = re.sub(r"& -(?=\d)", "& $-$", _float(
            "jc1_conf", table_jc1_conf(conf, conf_s, runs), True, whole="turned", colsep_pt=6, pages=True))
    else:
        missing.append("JetClass confusion matrices -- experiments/FIGS/data/v2/confusion_jc1/, confusion_jc1_fulldecay/")
    census = data / "jc2bal" / "census.json"
    census_doc = json.loads(census.read_text()) if census.exists() else None
    if fits_v4:
        # the per-class AUC differences of F3 and the classes of A3R: class-balanced probes only
        auc_diff_macros(em, fits, runs, root, census_doc, index, native_tex)
        if census_doc:
            test = census_doc["counts"]["ft_test"]
            em.macro("MlstJcBalNThin", str(sum(min(c, sjb.TEST_PER_CLASS) < THIN_TEST for c in test)), census,
                     "counts.ft_test", f"classes with fewer than {THIN_TEST} class-balanced test jets (F3 hatching)")
    # the ROC figures: the background-jet cut, and the curves' cut points and test-set sizes (curves_v2)
    em.macro("MlstMinBkgPass", str(V4.JC1_MIN_BKG), root / "experiments/FIGS/mlst_v4.py", "JC1_MIN_BKG",
             "fewest background jets behind a drawn ROC point or a quoted rejection")
    em.macro("MlstMassMinJets", str(MASS_MIN_JETS), root / "experiments/EVAL/atlas_resolution.py", "MIN_JETS",
             "jets below which a mass bin is null")
    roc = V4.load_roc(data)
    missing += [f"ROC curves -- experiments/FIGS/data/v2/{V4.ROC_DIR}/{f}/" for f in V4.ROC_FAMILIES if f not in roc]
    roc_macros(em, roc, runs, index, missing, fmt_int)
    top_r50_macro(em, roc, runs, index, fmt_pm)
    ours = {x for x in UNCONVERGED_SEEN if x[0] != SOPHON_MODEL}
    trained = {(m, ds, n) for (m, ds, n) in ours if not m.startswith(("init-", "initbn-"))}
    em.macro("MlstProbeUnconverged", str(len(trained)), index, "linear probes of pretrained models that did not converge (count)")
    em.macro("MlstProbeUnconvergedInit", str(len(ours) - len(trained)), index,
             "linear probes of the randomly initialized networks that did not converge (count)")
    cells = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {}))]
    edge = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {})) if c["C"] == max(d["c_grid"])]
    em.macro("MlstProbeFits", str(len(cells)), index, "selected linear-probe fits of the reported checkpoints (count)")
    em.macro("MlstProbeEdgeC", str(len(edge)), index, "of these, fits whose selected C is the largest of the grid (count)")
    mlp = load_mlp(data)
    bs3 = load_bs3_mlp(data)
    overlay_bs3_mlp(mlp, bs3)
    if mlp:
        # PI 2026-11-23 (LEDGER SP-9): the paper's table with the random partitions pooled, each one in its copy; (LEDGER
        # CA-15, option a) the rho columns and the class-token against hidden-layer block in the copy only
        out["tables/mlst_mlp.tex"] = _float("mlp", table_mlp_paper(mlp, _pooled(runs), TAB, _paper_rows(rows_init)), True)
        out["tables/mlst_mlp_full.tex"] = _float("mlp_full", table_mlp(mlp, runs, TAB, rows_init, raw_v4, mlp_nat,
                                                                       kin=True), True)
        mlp_macros(em, mlp, runs, index, fmt_pm)
        mlp_natural_macros(em, mlp_nat, runs, index, fmt_pm)
    else:
        missing.append(f"nonlinear probes -- experiments/FIGS/data/v2/{MLP_FITS_V4}/")
    out["tables/mlst_fill.tex"] = _float("fill", table_fill(raw_v4, fill, fits, _pooled(runs), TAB,
                                                            SETS + FLAV + (RAND_POOLED, INIT_BN), drop=False, chosen=chosen), True)
    out["tables/mlst_fill_full.tex"] = _float("fill_full", table_fill(raw_v4, fill, fits, runs, TAB, ALL_ROWS + (INIT_BN,)), True)
    # the random network's row: the text quotes its accuracy drops and its R30 under the chosen substitutions
    out["tables/mlst_fill_main.tex"] = _float("fill_main", table_fill(raw_v4, fill, fits, runs, TAB, MAIN + (INIT_BN,)), True)
    fill_macros(em, raw_v4, fill, fits, runs, index, fmt_pm, missing)
    loss_share_macros(em, comp, runs, index, fmt_pm, root)
    atlas = _load(data / MASS_RES)
    if atlas:
        # review of render dfe5942 (A7; LEDGER PF-4): what the probe uses, exactly; its target is generator-level. The
        # target written out, not referred to: the results book has no equations (its preview at b0ee1a8 stopped here)
        # fourth review (A5): the target is generator-level truth; what the centering removes is the jet's class
        centre = (" The probe regresses $\\theta = \\ln(\\mgen/\\msd)$, shifted by one training-jet median for every "
                  "jet, so no class label is used.")
        out["tables/mlst_mass.tex"] = _float("mass", table_mass(atlas, runs, TAB), fill={"__PROBE_CENTRE__": centre})
        # the binned values as the mass figures draw them (atlas_res_fine_v3: ten jet-pT bins, the same jets and
        # inclusive values as atlas_res_v3)
        fine = _load(data / MASS_RES_FINE) or atlas
        out["tables/mlst_mass_detail.tex"] = _float("mass_detail", md := table_mass_detail(fine, runs, TAB), True, fill={"__BLANK__": (
            r"bins with fewer than \MlstMassMinJets{} jets are blank." if " & --" in md.split("%SPLIT%", 1)[-1]
            else r"every bin holds at least \MlstMassMinJets{} jets."),
            "__MASS_RUNS__": (PART_DEF + _runs("set", "mass", "flav", "rand", "init")) if mass_has_parts(fine, runs)
            else _runs("set", "mass", "init")})
        mass_macros(em, atlas, runs, index, fmt_pm)
    else:
        missing.append(f"ATLAS mass resolution, truth-free refit -- experiments/FIGS/data/v2/{MASS_RES}/")
    mass_boot_macros(em, data)
    head = _load(data / "head")
    sculpt_v2 = {k.removesuffix("_sculpting_v2"): v for k, v in _load(data / "head" / "curves_v2").items()
                 if k.endswith("_sculpting_v2")}
    # mass_sculpting.py with eps_S, R50 and the flavor controls (2026-11-07), one script version for every model:
    # replaces both the working points and the curves once it holds every run the old files hold
    v3 = {k.removesuffix("_sculpting"): v for k, v in _load(data / "head" / "sculpting_v3").items()}
    old = {k.removesuffix("_sculpting") for k in head if k.endswith("_sculpting")}
    if v3 and old <= set(v3):
        head = {k: v for k, v in head.items() if not k.endswith("_sculpting")} | {f"{m}_sculpting": d for m, d in v3.items()}
        sculpt_v2 = dict(v3)
    elif v3:
        missing.append(f"sculpting_v3 lacks {sorted(old - set(v3))}")
    sculpt = {**{k.removesuffix("_sculpting"): v for k, v in head.items() if k.endswith("_sculpting")}, **sculpt_v2}
    if sculpt:
        edges = {tuple(d["bins_jet_sdmass"]) for d in sculpt.values()}
        if len(edges) != 1:
            raise SystemExit(f"FATAL: the sculpting files use {len(edges)} different mass binnings")
        e = edges.pop()
        width = {round(b - a, 9) for a, b in zip(e, e[1:])}
        bins = (f"{len(e) - 1} bins of {width.pop():g}~GeV from {e[0]:g} to {e[-1]:g}~GeV" if len(width) == 1
                else f"{len(e) - 1} bins from {e[0]:g} to {e[-1]:g}~GeV")
        # round-1 fix 2026-11-13: a second part for the networks fine-tuned on JetClass-II (head/sculpting_ft, the
        # fine-tuned sculpting figures' files), every network and from scratch, so that those curves have a home
        sft = _load(data / SCULPT_FT)
        sft = {k.removesuffix("_sculpting"): v for k, v in sft.items()}
        runs_ft = {**_pooled(runs), SCRATCH: sorted(m for m in sft if m.startswith("scratch-s"))}
        ft_part = ("%SPLIT%" + table_sculpt(sft, runs_ft, TAB, SETS + FLAV + (RAND_POOLED, SCRATCH))) if sft else ""
        rand = (RAND_POOLED,) if any(m in sculpt for m in _pooled(runs)[RAND_POOLED]) else ()
        # PI 2026-11-23 (LEDGER CS-2b): the main text states the sculpting null, so the appendix prints each network's
        # distance beside its statistical floor, compactly (the full table stays in the release)
        out["tables/mlst_sculpt_main.tex"] = _float("sculpt_main", table_sculpt_main(sculpt, _pooled(runs), TAB,
                                                                                    SETS + FLAV + rand), colsep_pt=3)
        out["tables/mlst_sculpt.tex"] = _float("sculpt", table_sculpt(sculpt, _pooled(runs), TAB, SETS + FLAV + rand) + ft_part,
                                               True, fill={"__SCULPT_BINS__": bins, **sculpt_fill(sculpt, runs)},
                                               whole="turned", colsep_pt=3)
        # the supplement's copy with the 17-class partitions (the network rule, PI 2026-11-14), the random partitions
        # pooled as they are drawn once their head outputs exist
        ft_full = ("%SPLIT%" + table_sculpt(sft, runs_ft, TAB, SETS + FLAV + (RAND_POOLED, SCRATCH))) if sft else ""
        out["tables/mlst_sculpt_full.tex"] = _float("sculpt_full", table_sculpt(
            sculpt, _pooled(runs), TAB, SETS + FLAV + ((RAND_POOLED,) if any(m in sculpt for m in _pooled(runs)[RAND_POOLED])
                                                       else ())) + ft_full, True,
            fill={"__SCULPT_RUNS__": sculpt_fill(sculpt, runs)["__SCULPT_RUNS__"]}, whole="turned", colsep_pt=3)
        sculpt_macros(em, sculpt, sculpt_v2, runs, index, fmt_pm)
    else:
        missing.append("mass sculpting -- experiments/FIGS/data/v2/head/*_sculpting.json")
    boot_file = data / (SCULPT_BOOT_L3_RERUN if (data / SCULPT_BOOT_L3_RERUN).exists() else SCULPT_BOOT_L3)
    if boot_file.exists():
        sb = json.loads(boot_file.read_text())
        out["tables/mlst_sculpt_boot.tex"] = _float("sculpt_boot", table_sculpt_boot(sb, runs, sculpt),
                                                    fill={"__REPLICAS__": fmt_int(sb["replicas"]), "__SCULPT_BOOT_RAND__": (
                                                        ", and the five random partitions, one run each, form one side"
                                                        if any(d["configuration"] == "rand2p" for d in sb["differences"].values())
                                                        else "")}, whole="turned", colsep_pt=6)
    else:
        missing.append(f"sculpting bootstrap -- experiments/FIGS/data/v2/{SCULPT_BOOT_L3_RERUN}")
        sb = None
    mass_boot = data / MASS_SD_BOOT_TESTS
    tests_in = (sb, json.loads(mass_boot.read_text()) if mass_boot.exists() else None)
    composition_macros(em, data, root, missing)
    size_macros(em, fits, fmt_int, index)
    method_macros(em, root, words)
    ft_recipe(em, data, {m[len("mtx-"):] for m in pre_runs}, fmt_int, missing, words)
    tag, per, absent = load_ft(data)
    # PI 2026-11-23 (LEDGER RR-7): every training-size point, in the tables, the figures and the release, reads the
    # fixed-schedule cell at its row's recorded rate, or the build stops: without the one rule's read-out the tables would
    # print the trees before it. Here in build, not in load_ft, which the gallery modules call at import.
    if ft_source(data).split()[0] not in (FIXED, FIXED_PARTIAL) and any(
            n < FULL_N.get(key, math.inf) for key, cells in per.items() for by_n in cells.values() for n in by_n):
        raise SystemExit(f"FATAL (RR-7): the fine-tuning read-out is {ft_source(data)}; a training-size point reads only "
                         "the one rule's fixed-schedule cells")
    out["tables/ft_source.txt"] = ft_source(data) + "\n"
    if absent:
        missing.append("fine-tuning read-outs -- experiments/FIGS/data/v2/ft_mlst/: " + ", ".join(absent))
    if not (data / "budget" / "budget_metrics.json").exists():
        missing.append("fixed-budget fine-tuning -- experiments/FIGS/data/v2/budget/budget_metrics.json")
    # bootstrap/l3 (decision L3, 2026-11-13) has its own layout ("diffs" with an "l3" block) and is not tab:boot's
    boot = {d.name: _load(d) for d in sorted((data / "bootstrap").glob("*")) if d.is_dir() and "." not in d.name
            and d.name != "l3" and d.name != "l3_v5"}
    if V4.pick(data, "bootstrap/probe_v4") != "bootstrap/probe_v4":
        # the bootstrap of the v5 probes stands in tab:boot's probe_v4 column, after its top and quark/gluon inputs are
        # checked to be the probes the tables print (the substitutions chosen on validation)
        boot["probe_v4"] = boot.pop("probe_v5")
        check_boot_fills(boot["probe_v4"], chosen)
    boot.pop("probe_v5", None)
    body, worst = table_boot(boot, runs, fits, per if tag else None, balanced=bool(fits_v4), fill=fill, chosen=chosen)
    if fits_v4 and "probe_v4" not in boot:
        missing.append("test-sample bootstrap of the class-balanced probes -- experiments/FIGS/data/v2/bootstrap/probe_v4/")
    out["tables/mlst_boot.tex"] = _float("boot", body, True, fill={"__JC2COMP__": "class-balanced" if fits_v4
                                                                   else "natural-composition"})
    boot_macros(em, boot, worst, index)
    # X->bb and X->bs before Table 1, which takes two of its columns (PI 2026-11-23, LEDGER SP-9)
    XR, xfits = xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int, words)
    xbx = {"XR": XR, "fits": xfits, "scr": xbx_scratch(data)}
    if tag is not None:
        # the fine-tuned training-size figures (A8, A8-part, F8): as the probes' above
        out["tables/mlst_ft.tex"] = _float("ft", table_by_size(
            "ft", "ft", lambda t, ds, met, a: size_ft(per, _pooled(runs), t, met, None if a == SCRATCH else a),
            (SCRATCH_LABEL, SCRATCH), {k: n for k, n, *_ in SUMMARY_PANELS}, rows=SETS), True, whole="upright")
        out["tables/mlst_ft_full.tex"] = _longtable("ft_full", table_ft(per, runs, TAB, ALL_ROWS))
        body, fewest = table_ft_rej(per, runs, TAB, ALL_ROWS)
        out["tables/mlst_ft_rej.tex"] = _float("ft_rej", body, True,
                                               fill={"__FEWEST__": fmt_int(fewest) if fewest else "--"})
        ft_macros(em, per, runs, index, fmt_pm)
        topcheck_macros(em, data, per, index, fmt_pm, missing)
        # quark/gluon on Pythia only, Herwig apart (PI 2026-11-12); top and quark/gluon are wider than the upright
        # text block at the columns' precision, so _float cuts them into blocks of columns, no row label wrapped
        out["tables/mlst_bench_top.tex"] = _float("bench_top", V4.table_bench(
            "top", per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two, kin=(LABEL["KIN"], KIN_INIT)),
            True)
        out["tables/mlst_bench_qg.tex"] = _float("bench_qg", V4.table_bench(
            "qg", per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two), True)
        out["tables/mlst_bench_qg_herwig.tex"] = _float("bench_qg_herwig", V4.table_bench_herwig(
            per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two))
        # PI 2026-11-23 (LEDGER SP-9): eight columns, the column padding narrowed to keep them on a turned page
        out["tables/mlst_summary.tex"] = _float("summary", table_summary(fits_s, fill, chosen, per, runs_s, TAB, data=data,
                                                                         xbx=xbx),
                                                True, pages=True, whole="turned", colsep_pt=3)
        f2_init_macros(em, fits_s, fill, chosen, runs_s, index)
        a10_init_macros(em, fits_s, fill, chosen, runs_s, index)
        out["tables/mlst_randparts.tex"] = _float("randparts", table_randparts(fits_s, fill, chosen, per, runs_s, TAB),
                                                  True, pages=True, whole="turned", colsep_pt=6)
        # PI 2026-11-23 (LEDGER SP-9): seven columns, the resonant and QCD AUC beside JetClass-II's; cut by the width
        # rule (split_wide) where a turned page does not hold them
        out["tables/mlst_summary_auc.tex"] = _float("summary_auc", table_summary(
            fits_s, fill, chosen, per, runs_s, TAB, SUMMARY_AUC_TABLE, "summary_auc", data=data), True, pages=True,
            colsep_pt=3)
        out["tables/mlst_summary_auc_full.tex"] = _float("summary_auc_full", table_summary(
            fits_s, fill, chosen, per, runs_s, TAB, SUMMARY_AUC_FULL, "summary_auc_full", data=data), True, pages=True,
            colsep_pt=3)
        out["tables/mlst_labelsets.tex"] = _float("labelsets", table_labelsets(per, runs, TAB), True, pages=True,
                                                  whole="turned", colsep_pt=6)
        out["tables/mlst_labelsets_full.tex"] = _float("labelsets_full", table_labelsets(per, runs, TAB, key="labelsets_full"),
                                                       True, pages=True, whole="turned", colsep_pt=6)
    # PI 2026-11-23 (LEDGER SP-9): every difference or null the text states, with its paired bootstrap where stored
    tested = {}         # each row's result, which the ratio words read (ratio_words, PI 2026-11-23, LEDGER SP-12 step 3)
    sided = {}          # and its two sides' printed run values, the ratio's (review of render dfe5942, A2, A3)
    body, reps = table_tests(data, runs, fits_s, fill, chosen, per if tag else {}, *tests_in, rows_out=tested, co=co,
                             sides_out=sided, mlp=mlp, bs3=bs3, sculpt=sculpt)
    reps = fmt_int(reps.pop()) if len(reps) == 1 else "the stated number of"
    # upright (LS-1) at the template's table size (PI SP-14), one column of rows: two columns do not fit the 15 cm
    # text block with no row label wrapped (TN-6), half a line holding about 27 characters of label at 8 pt and the
    # labels running to 69
    out["tables/mlst_tests.tex"] = _float("tests", body, True, whole="upright", colsep_pt=6, size=TEMPLATE_SIZE,
                                          fill={"__REPLICAS__": reps})
    gap_macros(em, fits, per if tag else {}, runs, index, fmt_pm)
    out["tables/mlst_per_run.tex"] = _longtable("per_run", table_per_run(fits, per if tag else {}, runs, TAB, fill,
                                                                         chosen))
    # every point of the jet-pT figures (PI 2026-11-13, supplement)
    perf = {sub: _load(data / "perf_vs_pt" / sub) for sub in PT_DIRS}
    if all(perf.values()):
        out[PT_CSV] = pt_bins_csv(perf, _pooled(runs))       # L18: a file of the data release, no typeset table
    else:
        missing.append("performance per jet-pT bin -- experiments/FIGS/data/v2/perf_vs_pt/"
                       + ", ".join(k for k, v in perf.items() if not v))
    if tag is not None and atlas and xfits and sculpt:
        rows = masseffect_rows(atlas, fits_s, fill, chosen, per, mlp, XR, xfits, sculpt, _load(data / SCULPT_FT), runs)
        out["tables/mlst_masseffect.tex"] = _float("masseffect", table_masseffect(rows, runs, TAB), True, pages=True,
                                                   whole="turned", colsep_pt=6, notes=[MASSEFFECT_NOTE])
    t = partrepro(em, data, per if tag else {}, fmt_pm, missing, runs={**runs, "__bench__": MAIN})
    if t:
        out["tables/mlst_partrepro.tex"] = _float("partrepro", t)
    sweep = data / "budget" / RATE_SEARCH
    if sweep.exists():
        rows = rate_search_rows(json.loads(sweep.read_text()), preview=ft_preview())
        out["tables/mlst_lrsweep.tex"] = _float("lrsweep", table_lrsweep(rows), wide=True)
    else:
        missing.append(f"the rate search of every row -- experiments/FIGS/data/v2/budget/{RATE_SEARCH}")
    out["tables/mlst_sets.tex"] = _float("sets", table_sets(runs))
    offsets = data / "head" / "input_bn_offsets.json"
    out["tables/mlst_inputs.tex"] = _float("inputs", table_inputs(root, json.loads(offsets.read_text())["offsets"]
                                                                  if offsets.exists() else None), True)
    if not offsets.exists():
        missing.append("input-normalization offsets of mtx-l188-s1 -- experiments/FIGS/data/v2/head/input_bn_offsets.json")
    else:
        input_bn_macros(em, json.loads(offsets.read_text())["offsets"], offsets)
    full = jc1_full(data, runs)
    if full:
        pre, ref, src = full
        pooled = {**runs, "RAND": [m for a in RAND for m in runs[a]]}
        lab = {**LABEL, "RAND": LABEL[RAND_POOLED]}
        lit = next(r for r in json.loads(V4.LIT.read_text())["jetclass"] if r["method"] == "ParT")
        jfill = {"__WPHQQL__": f"{100 * V4.LPV_EPS['label_Hqql']:g}", "__WPTBL__": f"{100 * V4.LPV_EPS['label_Tbl']:g}",
                 "__MINBKG__": str(V4.JC1_MIN_BKG), "__MINBKGPC__": f"{100 / math.sqrt(V4.JC1_MIN_BKG):.0f}",
                 "__PART100__": f"an accuracy of {lit['acc']} and an AUC of {lit['auc_ovo']}"}
        out["tables/mlst_jetclass.tex"] = _float("jetclass", V4.table_jetclass(
            pre, ref, pooled, MAIN, lab, SCRATCH, 1_000_000, TAB, TAB.two, part="summary", vals=jc1_values(per, pooled)),
            True, fill=jfill)
        out["tables/mlst_jetclass_rej.tex"] = _float("jetclass_rej", V4.table_jetclass(
            pre, ref, pooled, SETS + FLAV + ("RAND",), lab, SCRATCH, 1_000_000, TAB, TAB.two, part="rej"), True, fill=jfill,
            whole="turned", colsep_pt=2.5)
        out["tables/mlst_jetclass_all.tex"] = _float("jetclass_all", V4.table_jetclass(
            pre, ref, pooled, SETS + FLAV + ("RAND",), lab, SCRATCH, 1_000_000, TAB, TAB.two, literature=False,
            vals=jc1_values(per, pooled)), True)
        out["tables/mlst_jetclass_nbkg.tex"] = _float("jetclass_nbkg", V4.table_jetclass_nbkg(
            pre, ref, pooled, SETS + FLAV + ("RAND",), lab, SCRATCH, 1_000_000), True)
        if tag is not None:
            out["tables/mlst_published.tex"] = _float("published", table_published(per, runs, pre, ref, TAB), True, pages=True, whole="turned", colsep_pt=6,
                                                            notes=[PUBLISHED_NOTE])
        jetclass_macros(em, pre, runs, src, fmt_pm, words)
        jc1_boot_macros(em, data, fmt_int, missing)
        out["tables/mlst_rand_prong.tex"] = _float("rand_prong", rp := table_rand_prong(pre, runs, root, TAB), True, pad=True,
                                                   whole="turned", fill={"__PRONG__": (
            r"A bullet ($\bullet$) marks a prong-count boundary that a partition keeps, with no group holding classes "
            "of both prong counts." if r"\bullet" in rp else "No partition keeps a boundary between two, three and "
            "four prongs, because for each pair of these prong counts every partition has a group holding classes of "
            "both.")})
    else:
        missing.append("JetClass per-class curves -- experiments/FIGS/data/v2/curves/full_*.json")
    out["tables/mlst_jetclass_probe.tex"] = _float("jetclass_probe", table_jetclass_probe(fits_s, runs_s, TAB,
                                                                                          rows_init + (SOPHON,)), True)
    if comp:
        ftg, nodev = ft_gpu_types(data)
        if nodev:
            missing.append(f"GPU of {sum(nodev.values())} fixed-budget fine-tuning cells (JetClass, JetClass-II): "
                           "experiments/FT/budget_metrics.py records it for top and quark/gluon only")
        fth = ft_hours(data)
        if fth is None:
            missing.append("fine-tuning GPU-hours -- experiments/FIGS/data/v2/budget/ft_attempts.json")
        # PI 2026-11-23 (LEDGER SP-9): the appendix's table one row per stage, each run's row in the data release
        body, hours = table_compute(comp, runs, fmt_int, extra=((LABEL["KIN"], KIN_MODEL),), ft_gpus=ftg,
                                    ft_nodev=nodev, ft_h=fth, label="tab:computefull")
        out["tables/compute_full.tex"] = body.replace("__CAPTION__", printed_caption("compute_full"))
        out["tables/compute.tex"] = _float("compute", table_compute_stages(comp, runs, fmt_int,
                                                                           extra=((LABEL["KIN"], KIN_MODEL),), ft_h=fth))
        compute_macros(em, comp, runs, hours, index, fmt_int, extra=((LABEL["KIN"], KIN_MODEL),))
        ft_hours_macros(em, fth, index, fmt_int)
        if len(set(pre_runs) & set(comp)) < len(pre_runs):
            missing.append(f"compute records: {len(set(pre_runs) & set(comp))} of {len(pre_runs)} paper runs")
    ratio_words(em, tested, text_ratio_words(root / "paper/journal"), index, sided)
    out["data_release/ratio_check.csv"] = ratio_check_csv(em)
    pw_macros(em, tested, sided, index)
    mlp_claim_check(tested)
    sculpt_pw_macro(em, sculpt, tests_in[0], runs, index)
    bad = [k for k, v in tested.items() if k[0] == SCULPT_FLOOR_HEAD and isinstance(v, tuple) and not v[0] > 3 * v[4]]
    if bad:     # CS-2c: the text says every head sculpts at 10% QCD efficiency
        raise SystemExit(f"FATAL (CS-2c): heads not above the sculpting floor at 10% QCD efficiency: {bad}")
    for r in vars(em).get("ratio_check", []):       # LEDGER TX-3: printed in the build log, one line a quoted ratio
        print(f"TX-3 {'ok  ' if r[7] == r[8] else 'DIFF'} {r[0]}: full {r[5]:.6g} ({r[7]}), printed means {r[6]:.6g} "
              f"({r[8]}); means {r[11]!r}, {r[12]!r} printed {r[9]!r}, {r[10]!r}")
    diff = [r[0] for r in vars(em).get("ratio_check", []) if r[7] != r[8]]
    if diff:        # LEDGER TX-3, RR-7: checked where the ratios are formed; the fix is a digit more in their table
        raise SystemExit(f"FATAL: quoted ratios that differ from full-precision means at the quoted digits: {diff}; "
                         "give their table one more digit (EXTRA_DIGIT)")
    dropped_auc_check(root / "paper/journal")
    required_pass(em, index, missing)
    # a table whose data is not pulled yet is a marked placeholder, never the file an older generator left behind
    # built only where the table is absent (setdefault evaluated the placeholder for every table, and a caption with a
    # fill the data writes, tab:jc1classauc's __JC1_RECALL__, stopped the build, 2026-11-22)
    for key, name, wide, fill_ in PLACEHOLDERS:
        if f"tables/{name}.tex" not in out:
            out[f"tables/{name}.tex"] = _float(key, r"\pending{no result yet}", wide, fill_)
    for key, name in LONGTABLES:
        if f"tables/{name}.tex" not in out:
            out[f"tables/{name}.tex"] = _float(key, r"\pending{no result yet}", True)
    left = [k for k, v in out.items() if V4.TOK_OPEN in v] + [n for n, b, _ in em.macros if V4.TOK_OPEN in b]
    if left:
        raise SystemExit(f"FATAL: unprinted value tokens in {left}")
    upright_check(out, root / "paper/journal")
    return out
